"""Picture(s) to voxel model with the language model, in rounds.

1. Each reference is read into a small indexed grid (``images.index_picture``); its colours start the palette.
2. A starting model: ``inflate`` (the front puffed up), ``hull`` (front + side carved), ``extrude``, ``empty``, or the
   current content of a model (``into`` with ``start=current``: refine what is there).
3. The model gets the reference pictures as images, their grids as text, the palette, the starting model as a contact
   sheet, and the operation language; it answers JSON ``{"ops": [...], "palette_add": [...], "notes": "..."}``.
4. Review rounds: it sees the reference next to a contact sheet of the result (front, left, top, back, two isometric
   views), the front silhouette differences and any operation errors, and answers corrective operations (or
   ``{"done": true}``).

Every round is kept in the job log with its render, and the result is saved as a model (a new one, or a new version of
``into``).
"""

from __future__ import annotations

import logging
import threading
import time
import traceback
from dataclasses import dataclass, field
from typing import Any, Optional

import numpy as np

from . import builders
from .errors import SeuratError
from .hoard_link.ids import new_id
from .images import Indexed, index_picture, to_png
from .llm import ModelClient, parse_json
from .palette import merge
from .render import ortho, png_bytes, sheet
from .voxels import KEY_CHARS, MAX_SIDE, VoxelModel, apply_ops, norm_hex

log = logging.getLogger("seurat.generate")

ROLES = ("front", "side", "top", "back", "extra")
STARTS = ("auto", "inflate", "hull", "extrude", "empty", "current")

OPS_GUIDE = """OPERATIONS (JSON objects, applied in order; c is a palette index, c:0 erases with the same shape):
SHAPE (use these to build volumes; most models need only these):
- {"op":"sphere","center":[x,y,z],"radii":[rx,ry,rz],"c":i}       ellipsoid (heads, bodies, bellies); "r":n for a ball;
  centres may be .5
- {"op":"line","from":[x,y,z],"to":[x,y,z],"r":r0,"r_to":r1,"c":i}  CAPSULE from one point to another, radius r0 at
  "from" tapering to r1 at "to": necks, tails, arms, legs, horns, feather tips. Without r: a one-voxel line.
- {"op":"prism","plane":"top","points":[[u,v],...],"from":a,"to":b,"c":i}   a POLYGON drawn on a plane and given
  thickness: WINGS, fins, ears, tail fans, blades, crests. plane front: u = x, v = y, thickness along z from a to b;
  plane side: u = z, v = y, thickness along x; plane top: u = x, v = z, thickness along y (a wing spread out sideways
  is a polygon on the top plane, 1-3 voxels thick, raised to its height with from/to = y).
- {"op":"cylinder","axis":"y","center":[x,z],"from":y0,"to":y1,"r":n,"c":i}   center = the two other coordinates
- {"op":"lathe","center":[x,z],"from":y0,"radii":[r0,r1,...],"c":i}   solid of revolution, one radius per y going up
- {"op":"box","from":[x,y,z],"to":[x,y,z],"c":i,"hollow":false}   inclusive corners
- {"op":"inflate","plane":"front","origin":[x0,y0],"center":zc,"depth":d,"profile":"round","rows":[...]}  pixel rows
  (front view) puffed into a cushion up to d voxels thick around z = zc (round | soft | flat).
- {"op":"mirror","axis":"x","keep":"low"}   copy the low-x half onto the high-x half (left-right symmetry)
- {"op":"move","by":[dx,dy,dz]}   {"op":"clear"}   {"op":"recolor","from_c":a,"to_c":b}
COLOUR AND DETAIL:
- {"op":"paint","side":"front","origin":[x0,y0],"rows":[...],"depth":1}  recolour the visible surface seen from a side
  (front|back|left|right|top|bottom) pixel by pixel; "." leaves a pixel alone. left: first column = back; right: first
  column = front; back: first column = the model's right side; top: first row = back.
- {"op":"layer","plane":"front","origin":[x0,y0],"from":z0,"to":z1,"rows":[...]}  pixel rows copied into a slab
  (plane side: columns = z, at x; plane top: rows = z, at y). Rows: the FIRST row is the TOP.
  Characters: "." = leave as it is, "0" = erase, 1-9 a-z A-Z = palette indices 1-61.
- {"op":"voxels","list":[[x,y,z,c],...]}   single voxels (eyes, claws; use sparingly)"""

SYSTEM = """You are a voxel sculptor working inside Seurat's Hoard, a voxel studio. You turn reference pictures (sprites,
pixel art, drawings, photos) into 3D voxel models that read clearly from every side, in the style of good voxel art:
solid rounded volumes, clean silhouettes, believable anatomy, flat colours, no stray voxels.

COORDINATES: the grid is size [X, Y, Z]. x grows to the right, y grows UP (y=0 is the floor), z grows TOWARDS the
viewer (z = Z-1 is the front surface). The FRONT view looks at the model from +z; the "left" view from -x; the top view
from +y. Every coordinate must stay inside 0..X-1, 0..Y-1, 0..Z-1. The creature faces +z (towards the viewer).

""" + OPS_GUIDE + """

HOW TO WORK (sculpt first, colour comes later):
- {paint_rule}
- Think like a sculptor in 3D, not in flat 2D layers. Plan the anatomy first: where the body sits in x, y and z, how
  long and thick it is, where the head is, how the neck joins, where each wing starts (shoulder), how far it spreads
  and in which direction, where the tail goes (usually backwards, towards z = 0), where the legs touch the floor.
  Write down that plan as coordinates in your thinking before answering.
- Build with parts: an ellipsoid for the body and the head, capsules (line with r) for necks, tails, legs and arms,
  prisms for wings, fins, ears and tail fans, then carve with c:0 where needed. Symmetric creatures: build the left
  half (low x) and mirror it. Use the palette colour of each part (body colour on the body, wing colour on the wings).
- Game and overworld sprites are NOT exact projections: they are drawn from above at an angle and squashed. Use them for
  the parts, their proportions and colours; do not copy them as flat slabs. A side picture tells you how deep the
  creature is and where the neck, wings and tail go along z.
- The model must look good from EVERY side: from the side a bird is a body with a neck forward and a tail backwards,
  wings spreading sideways (x) and maybe upwards, never a flat board. From the top it must have depth (z) too.
- Keep the answer compact: at most about 150 operations. Never list hundreds of single voxels.

ANSWER with ONE JSON object and nothing else:
{"notes":"one or two sentences on what you built","palette_add":["#rrggbb", ...],"ops":[ ... ]}
palette_add (optional) appends colours: the first new one gets the next free index."""

PAINT_AUTO = ("Seurat paints the reference pictures onto your shape automatically after every round (front picture on "
              "the front, back picture on the back, side picture on both sides), so your job is the SHAPE: the volume, "
              "the silhouette from every side and the colour of each part. You only need paint for details the pictures "
              "cannot give (underside, top).")
PAINT_SELF = ("Colour comes after the shape: once the volume is right, use paint with the reference rows to put the face "
              "and the details on the curved surface, and darker shades on the back and the underside.")

REVIEW = """Review round {round} of {rounds}. The first picture(s) are the reference; the last picture is a contact sheet
of the CURRENT model (front, left, top, back flat views and two isometric views; labels on each tile).

Current model: size {size}, {count} voxels, bounds {bounds}.
{guard}{silhouette}
{errors}
{reference}
Look at every tile and ask: does it look like the creature from this side? Answer corrective operations applied ON TOP
of the current model (use c:0 to carve). Fix the biggest problems first: flat boards instead of volumes, missing or
wrong parts (neck, wings, tail, legs), wrong proportions, holes, floating pieces, the front silhouette. {paint_rule}
Never answer only {{"op":"clear"}}: if you rebuild, include the whole new model in the same answer.
If the model is already good, answer {{"done": true, "notes": "why", "ops": []}}.
Answer ONE JSON object: {{"notes": "...", "palette_add": [...], "ops": [...]}}"""


@dataclass
class Job:
    id: str
    params: dict[str, Any]
    status: str = "queued"                  # queued | running | done | failed | cancelled
    step: str = ""
    log: list[dict[str, Any]] = field(default_factory=list)
    rounds: list[dict[str, Any]] = field(default_factory=list)    # {round, notes, applied, errors, count, render}
    answers: list[str] = field(default_factory=list)              # raw model answers (renders dir, .txt)
    model_id: str = ""
    model_name: str = ""
    llm: str = ""
    error: str = ""
    warning: str = ""                       # the model failed part-way; what was built before is saved
    created: float = field(default_factory=time.time)
    finished: float = 0.0
    cancel: threading.Event = field(default_factory=threading.Event)

    def say(self, text: str) -> None:
        self.step = text
        self.log.append({"t": time.time(), "text": text})
        log.info("job %s: %s", self.id, text)

    def to_dict(self) -> dict[str, Any]:
        return {"id": self.id, "status": self.status, "step": self.step, "log": self.log[-80:], "rounds": self.rounds,
                "model_id": self.model_id, "model_name": self.model_name, "llm": self.llm, "error": self.error, "warning": self.warning, "answers": self.answers,
                "created": self.created, "finished": self.finished,
                "params": {k: v for k, v in self.params.items() if k != "images"}}


class Generator:
    def __init__(self, services: Any):
        self.svc = services
        self.jobs: dict[str, Job] = {}
        self._queue: list[str] = []
        self._cv = threading.Condition()
        self._threads: list[threading.Thread] = []
        self._stop = False

    # ------------------------------------------------------------------ queue
    def start(self) -> None:
        """Generations run side by side (``parallel_jobs`` workers): the model server answers several at once, and a
        round with long thinking takes many minutes."""
        if not self._threads:
            n = int(self.svc.settings["parallel_jobs"]) if hasattr(self.svc, "settings") else 2
            for i in range(max(1, n)):
                t = threading.Thread(target=self._worker, name=f"seurat-generate-{i + 1}", daemon=True)
                t.start()
                self._threads.append(t)

    def stop(self) -> None:
        with self._cv:
            self._stop = True
            self._cv.notify_all()
        for job in self.jobs.values():
            job.cancel.set()

    def submit(self, params: dict[str, Any]) -> Job:
        refs = params.get("references") or []
        if not refs:
            raise SeuratError("invalid", "Give at least one reference picture.", "Upload a PNG first (reference_add) and pass its id.")
        for r in refs:
            self.svc.store.reference(r["id"])
        if params.get("into"):
            self.svc.store.get(params["into"])
        if params.get("start") == "current" and not params.get("into"):
            raise SeuratError("invalid", "start=current needs into (the model to refine).")
        job = Job(id=new_id("g"), params=params)
        job.say("en cola")
        with self._cv:
            self.jobs[job.id] = job
            self._queue.append(job.id)
            self._cv.notify_all()
        self._prune()
        return job

    def get(self, job_id: str) -> Job:
        job = self.jobs.get(job_id)
        if not job:
            raise SeuratError("not_found", f"No generation job {job_id!r}.")
        return job

    def cancel(self, job_id: str) -> Job:
        job = self.get(job_id)
        job.cancel.set()
        if job.status == "queued":
            job.status = "cancelled"
            job.say("cancelado")
        return job

    def list(self) -> list[dict[str, Any]]:
        return [j.to_dict() for j in sorted(self.jobs.values(), key=lambda j: -j.created)]

    def busy(self) -> bool:
        return any(j.status in ("queued", "running") for j in self.jobs.values())

    def _prune(self, keep: int = 30) -> None:
        done = sorted((j for j in self.jobs.values() if j.status not in ("queued", "running")), key=lambda j: j.created)
        for j in done[:-keep]:
            self.jobs.pop(j.id, None)

    def _worker(self) -> None:
        while True:
            with self._cv:
                while not self._queue and not self._stop:
                    self._cv.wait()
                if self._stop:
                    return
                job_id = self._queue.pop(0)
            job = self.jobs.get(job_id)
            if not job or job.status == "cancelled":
                continue
            job.status = "running"
            try:
                self.run(job)
                job.status = "cancelled" if job.cancel.is_set() else "done"
            except SeuratError as exc:
                job.status, job.error = "failed", exc.message
                job.say("error: " + exc.message)
            except Exception as exc:  # noqa: BLE001
                job.status, job.error = "failed", f"{type(exc).__name__}: {exc}"
                job.say("error inesperado: " + job.error)
                log.error("generation %s failed:\n%s", job.id, traceback.format_exc())
            job.finished = time.time()

    # ------------------------------------------------------------------ one job
    def run(self, job: Job) -> VoxelModel:
        p = job.params
        settings = self.svc.settings
        max_side = int(p.get("max_side") or settings["max_side"])
        colors = int(p.get("colors") or settings["colors"])
        rounds = max(0, int(p.get("rounds") if p.get("rounds") is not None else settings["review_rounds"]))
        use_model = p.get("use_model", True)
        job.say("leyendo las referencias")
        refs: dict[str, list[tuple[dict[str, Any], Indexed, bytes]]] = {}
        for r in p["references"]:
            role = r.get("role") or "front"
            if role not in ROLES:
                raise SeuratError("invalid", f"Unknown role {role!r}; use {', '.join(ROLES)}.")
            meta = self.svc.store.reference(r["id"])
            raw = self.svc.store.reference_bytes(r["id"])
            ix = index_picture(raw, max_side=max_side, colors=colors)
            refs.setdefault(role, []).append((meta, ix, to_png(raw, 1024)))
        if "front" not in refs:
            first_role = next(iter(refs))
            refs["front"] = [refs[first_role].pop(0)]
            if not refs[first_role]:
                refs.pop(first_role)
        front = refs["front"][0][1]
        side = refs.get("side", [(None, None, None)])[0][1]
        top = refs.get("top", [(None, None, None)])[0][1]

        # ---- palette: the reference colours, a darker shade of each, a few greys
        palette: list[str] = []
        for role in ROLES:
            for _, ix, _ in refs.get(role, []):
                palette, _ = merge(palette, ix.palette, limit=40)
        shades = []
        for c in list(palette):
            r, g, b = (int(c[i: i + 2], 16) for i in (1, 3, 5))
            shades.append("#%02x%02x%02x" % (int(r * 0.68), int(g * 0.68), int(b * 0.72)))
        palette, _ = merge(palette, shades + ["#151317", "#3a3640", "#8a8590", "#e8e4ea"], limit=len(KEY_CHARS))

        # ---- starting model
        start = p.get("start") or "auto"
        fallback = ""
        if start not in STARTS:
            raise SeuratError("invalid", f"start must be one of {', '.join(STARTS)}.")
        depth = int(p.get("depth") or 0)
        name = (p.get("name") or "").strip() or (refs["front"][0][0].get("name") or "Modelo") + " (voxel)"
        into = p.get("into") or ""
        if start == "current":
            model = self.svc.store.get(into)
            model.palette, _ = merge(model.palette, palette, limit=255)
        else:
            if start == "auto":
                # The model sculpts from nothing: the instant builders make flat slabs from angled sprites.
                fallback = "hull" if side is not None else "inflate"
                start = "empty" if use_model else fallback
            if start == "empty":
                d = depth or (side.width if side is not None else max(4, round(max(front.width, front.height) * 0.8)))
                h = max(front.height, side.height if side is not None else 0)
                model = VoxelModel.empty("tmp", name, [front.width, h, min(d, MAX_SIDE)], palette)
            else:
                model = builders.build(start, front, side=side, top=top, depth=depth, name=name, base_palette=palette)
            if depth and model.size[2] < depth:
                model.resize([model.size[0], model.size[1], depth])
        job.say(f"modelo inicial: {start}, {model.size[0]}x{model.size[1]}x{model.size[2]}, {model.count} voxels")
        self._record(job, 0, "inicio: " + start, {"applied": 0, "errors": []}, model)

        llm_name = ""
        if use_model and not job.cancel.is_set():
            try:
                llm_name = self._model_rounds(job, model, refs, rounds, start, p)
            except SeuratError as exc:
                if model.count == 0 and fallback:
                    model = builders.build(fallback, front, side=side, top=top, depth=depth, name=name, base_palette=palette)
                    start = fallback
                    self._record(job, len(job.rounds), "sin modelo: " + fallback, {"applied": 0, "errors": []}, model)
                elif model.count == 0:
                    raise
                # Keep what there is (the starting model and any round that worked) instead of losing it.
                job.warning = exc.message
                job.say(f"el modelo falló ({exc.message[:200]}); se guarda lo que había hasta ahí"
                        + (f" (o, si no había nada, el modelo instantáneo «{fallback}»)" if fallback else ""))
        job.llm = llm_name
        if job.cancel.is_set():
            job.say("cancelado: se guarda lo que había")

        # ---- save
        if into:
            target = self.svc.store.get(into)
            target.grid, target.palette = model.grid, model.palette
            target.source = {"kind": "generated", "mode": start, "references": [r["id"] for r in p["references"]],
                             "prompt": p.get("prompt") or "", "llm": llm_name, "rounds": len(job.rounds) - 1, "job": job.id}
            saved = self.svc.store.save(target, version_note="before generation " + job.id)
        else:
            model.id = self.svc.store.new_id()
            model.name = name
            model.source = {"kind": "generated", "mode": start, "references": [r["id"] for r in p["references"]],
                            "prompt": p.get("prompt") or "", "llm": llm_name, "rounds": len(job.rounds) - 1, "job": job.id}
            saved = self.svc.store.save(model)
        job.model_id, job.model_name = saved.id, saved.name
        job.say(f"guardado: {saved.name} ({saved.count} voxels)")
        return saved

    def _save_answer(self, job: Job, rnd: int, text: str, suffix: str = "") -> None:
        name = f"{job.id}-r{rnd}{suffix}.txt"
        try:
            (self.svc.config.renders_dir / name).write_text(text or "", encoding="utf-8")
            job.answers.append(name)
        except OSError:
            pass

    def _record(self, job: Job, rnd: int, notes: str, result: dict[str, Any], model: VoxelModel) -> None:
        name = f"{job.id}-r{rnd}.png"
        path = self.svc.config.renders_dir / name
        path.parent.mkdir(parents=True, exist_ok=True)
        if model.count:
            path.write_bytes(png_bytes(sheet(model, 900)))
        job.rounds.append({"round": rnd, "notes": notes, "applied": result.get("applied", 0), "errors": result.get("errors", [])[:12],
                           "count": model.count, "size": model.size, "render": name if model.count else ""})

    # ------------------------------------------------------------------ the model's part
    def _model_rounds(self, job: Job, model: VoxelModel, refs: dict, rounds: int, start: str, p: dict[str, Any]) -> str:
        settings = self.svc.settings
        client: ModelClient = self.svc.llm
        effort = p.get("effort") or settings["effort"]
        max_tokens = int(settings["max_tokens"])
        thinking_tokens = int(settings["thinking_tokens"])
        timeout = float(settings["timeout_s"])
        auto_paint = bool(settings["auto_paint"] if p.get("auto_paint") is None else p["auto_paint"])
        paint_rule = PAINT_AUTO if auto_paint else PAINT_SELF
        system = SYSTEM.replace("{paint_rule}", paint_rule)
        ref_images = [img for role in ROLES for _, _, img in refs.get(role, [])][:4]
        text = self._first_prompt(model, refs, start, p)
        images = list(ref_images)
        if model.count:
            images.append(png_bytes(sheet(model, 900)))
        llm_name = ""
        total = 1 + rounds
        guard = ""
        for rnd in range(1, total + 1):
            if job.cancel.is_set():
                break
            label = f"ronda {rnd} de {total}" + (": pidiendo al modelo" if rnd == 1 else ": revisión")
            job.say(label)

            def progress(n_text: int, n_reason: int, label=label) -> None:
                job.step = f"{label} · " + (f"pensando ~{n_reason // 4} tokens" if n_reason and not n_text else
                                            f"recibidos ~{n_text // 4} tokens" + (f" (tras pensar ~{n_reason // 4})" if n_reason else ""))

            def ask(prompt: str) -> Any:
                return client.chat(system, prompt, images, max_tokens=max_tokens, effort=effort, timeout=timeout,
                                   on_progress=progress, cancel=job.cancel, thinking_tokens=thinking_tokens)

            answer = ask(text)
            llm_name = answer.model or llm_name
            self._save_answer(job, rnd, answer.text)
            job.say(f"respuesta de {llm_name or 'el modelo'} en {answer.seconds:.0f} s, ~{len(answer.text) // 4} tokens"
                    + (f" tras pensar ~{answer.reasoning_chars // 4}" if answer.reasoning_chars else "")
                    + {"length": " (cortada por el límite de tokens)", "loop": " (cortada: se repetía)",
                       "timeout": " (cortada por tiempo)"}.get(answer.finish, ""))
            if answer.finish == "cancelled":
                break
            try:
                data = parse_json(answer.text)
            except ValueError:
                job.say("la respuesta no traía JSON; se le pide de nuevo")
                answer = ask(text + "\n\nYour previous answer was not valid JSON. Answer ONLY the JSON object, compact.")
                self._save_answer(job, rnd, answer.text, suffix="b")
                try:
                    data = parse_json(answer.text)
                except ValueError:
                    raise SeuratError("failed", "The model did not answer valid JSON twice.", "Try again, or with fewer colours / a smaller size.") from None
            if data.get("salvaged"):
                job.say(f"respuesta incompleta: se aprovechan las {len(data.get('ops') or [])} operaciones completas")
            added = []
            for c in (data.get("palette_add") or [])[:40]:
                try:
                    hexc = norm_hex(c)
                except SeuratError:
                    continue
                if hexc not in model.palette and len(model.palette) < 255:
                    model.palette.append(hexc)
                    added.append(hexc)
            ops = data.get("ops") or []
            if not isinstance(ops, list):
                ops = []
            before_grid, before_count = model.grid.copy(), model.count
            result = apply_ops(model, ops, atomic=False) if ops else {"applied": 0, "errors": [], "count": model.count}
            notes = str(data.get("notes") or "")[:400]
            guard = ""
            reason = self._destructive(ops, before_count, model.count)
            if reason:
                model.grid = before_grid
                guard = (f"YOUR LAST ANSWER WAS REJECTED AND UNDONE: {reason}. The model below is the one from before it. "
                         "Answer a complete set of operations.\n")
                job.say(f"ronda {rnd}: respuesta descartada ({reason}); se mantiene el modelo anterior")
                result = {**result, "errors": [f"rejected: {reason}"] + list(result.get("errors") or [])}
            else:
                job.say(f"ronda {rnd}: {result['applied']} operaciones aplicadas, {len(result['errors'])} con error, {model.count} voxels"
                        + (f"; {len(added)} colores nuevos" if added else ""))
                if auto_paint and model.count:
                    painted = self._auto_paint(model, refs)
                    if painted:
                        job.say(f"ronda {rnd}: referencias pintadas sobre la forma ({painted} voxels)")
            self._record(job, rnd, notes, result, model)
            if (data.get("done") and not reason) or rnd == total:
                if data.get("done"):
                    job.say("el modelo da el resultado por bueno")
                break
            front_meta, front_ix, _ = refs["front"][0]
            reference = ("The FRONT reference rows (plane front, origin [0,0]; same palette indices):\n"
                         + "\n".join(self._map_rows(front_ix, model.palette)))
            text = REVIEW.format(round=rnd + 1, rounds=total, size=model.size, count=model.count, bounds=model.bounds(),
                                 guard=guard, silhouette=self._silhouette_report(model, refs), errors=self._errors_text(result),
                                 reference=reference, paint_rule=paint_rule)
            text += "\n\n" + OPS_GUIDE + "\n\nPalette (index: colour):\n" + self._palette_text(model)
            images = list(ref_images) + ([png_bytes(sheet(model, 900))] if model.count else [])
        return llm_name

    @staticmethod
    def _destructive(ops: list, before: int, after: int) -> str:
        """Why a round's answer should be undone ('' when it is fine): it emptied the model, only cleared it, or
        destroyed most of it."""
        kinds = [str(o.get("op")) for o in ops if isinstance(o, dict)]
        if after == 0 and (before > 0 or ops):
            return "it left the model empty"
        if kinds and all(k == "clear" for k in kinds):
            return "it only cleared the model"
        if before >= 200 and after < before * 0.25:
            return f"it destroyed most of the model ({before} -> {after} voxels)"
        return ""

    def _auto_paint(self, model: VoxelModel, refs: dict) -> int:
        """Paint the reference pictures onto the model's surface: side pictures on both sides, the back picture on
        the back, the front picture last on the front (it wins at the edges). Returns how many voxels changed."""
        sx, sy, sz = model.size
        before = model.grid.copy()
        ops: list[dict[str, Any]] = []
        for _, ix, _ in refs.get("side", [])[:1]:
            rows = self._map_rows(ix, model.palette)
            z0 = max(0, (sz - ix.width) // 2)
            ops.append({"op": "paint", "side": "left", "origin": [z0, 0], "rows": rows})
            ops.append({"op": "paint", "side": "right", "origin": [max(0, sz - z0 - ix.width), 0], "rows": [r[::-1] for r in rows]})
        for _, ix, _ in refs.get("back", [])[:1]:
            ops.append({"op": "paint", "side": "back", "origin": [max(0, (sx - ix.width) // 2), 0], "rows": self._map_rows(ix, model.palette)})
        for _, ix, _ in refs.get("front", [])[:1]:
            ops.append({"op": "paint", "side": "front", "origin": [0, 0], "rows": self._map_rows(ix, model.palette)})
        apply_ops(model, ops, atomic=False)
        return int((model.grid != before).sum())

    @staticmethod
    def _palette_text(model: VoxelModel) -> str:
        return ", ".join(f"{i + 1}{'(' + KEY_CHARS[i] + ')' if i < len(KEY_CHARS) else ''}:{c}" for i, c in enumerate(model.palette))

    def _first_prompt(self, model: VoxelModel, refs: dict, start: str, p: dict[str, Any]) -> str:
        sx, sy, sz = model.size
        lines = [f"Build a voxel model in a grid of size [{sx}, {sy}, {sz}] (x, y, z)."]
        if p.get("prompt"):
            lines.append("What the user wants: " + str(p["prompt"]).strip())
        lines.append("Palette (index(character): colour): " + self._palette_text(model))
        lines.append("Grid characters in the reference rows use the SAME indices as the palette.")
        for role in ROLES:
            for meta, ix, _ in refs.get(role, []):
                # the reference's own colours mapped to model palette indices
                mapped = self._map_rows(ix, model.palette)
                where = {"front": "plane front, origin [0,0] (fills x 0..%d, y 0..%d)" % (ix.width - 1, ix.height - 1),
                         "side": "the model seen from its left side (first column = back z=0, last column = front; the creature faces right)",
                         "top": "plane top (first row = back)", "back": "the model seen from behind (columns are mirrored)",
                         "extra": "extra view for context"}[role]
                lines.append(f"\nReference '{meta.get('name')}' as the {role.upper()} view, {ix.width}x{ix.height} pixels, {where}:")
                lines += mapped
        if model.count:
            lines.append(f"\nThe grid already holds a STARTING MODEL ({start}: {model.count} voxels, bounds {model.bounds()}); "
                         "its contact sheet is the last picture. Your operations are applied ON TOP of it: reshape it "
                         "(carve with c:0, add volume, repaint) or clear it first with {\"op\":\"clear\"} if it is a bad start.")
        else:
            lines.append("\nThe grid is empty: sculpt the whole model.")
        lines.append("\nThe pictures attached are the references (in the order above)" + (" and the starting model's contact sheet." if model.count else "."))
        return "\n".join(lines)

    @staticmethod
    def _map_rows(ix: Indexed, palette: list[str]) -> list[str]:
        index = {c.lower(): i + 1 for i, c in enumerate(palette)}
        lut = ["."] + [KEY_CHARS[index[c.lower()] - 1] if index.get(c.lower(), 99) <= len(KEY_CHARS) else "1" for c in ix.palette]
        return ["".join(lut[int(v)] for v in row) for row in ix.grid]

    @staticmethod
    def _errors_text(result: dict[str, Any]) -> str:
        errs = result.get("errors") or []
        if not errs:
            return "All your operations applied without errors."
        return "These operations FAILED and were skipped:\n" + "\n".join(f"- {e}" for e in errs[:15])

    @staticmethod
    def _silhouette_report(model: VoxelModel, refs: dict) -> str:
        """Where the front silhouette differs from the front reference (placed at origin [0,0])."""
        front = refs["front"][0][1]
        img = np.asarray(ortho(model, "front", 1, shade=False))[..., 3] > 0          # (sy, sx), row 0 = top
        sy, sx = img.shape
        ref = np.zeros((sy, sx), dtype=bool)
        h, w = front.height, front.width
        ref_mask = front.grid > 0
        oy = sy - h                                                                  # origin [0,0] = bottom-left
        ys = slice(max(0, oy), min(sy, oy + h))
        ref[ys, : min(w, sx)] = ref_mask[ys.start - oy: ys.stop - oy, : min(w, sx)]
        missing = ref & ~img
        extra = img & ~ref
        if not missing.any() and not extra.any():
            return "Front silhouette: identical to the front reference."
        rows = []
        for y in range(sy):
            if missing[y].any() or extra[y].any():
                line = "".join("+" if extra[y, x] else ("-" if missing[y, x] else ("#" if img[y, x] else ".")) for x in range(sx))
                rows.append(f"y={sy - 1 - y:3d} {line}")
        return (f"Front silhouette vs the reference: {int(missing.sum())} voxels missing ('-'), {int(extra.sum())} too many ('+'):\n"
                + "\n".join(rows[:64]))
