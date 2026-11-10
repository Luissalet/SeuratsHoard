"""Everything the app runs, built once per process: settings, the model store, the model client, the generator and
the agent token."""

from __future__ import annotations

import logging
import time
from typing import Any, Iterable, Optional

from . import __version__
from .config import Config, Settings
from .errors import SeuratError
from .generate import Generator
from .hoard_link.tokens import read_or_create_token
from .llm import ModelClient
from .palette import DEFAULT_PALETTE
from .store import Store
from .voxels import VoxelModel, apply_ops

log = logging.getLogger("seurat")


class Services:
    def __init__(self, config: Config, *, link_factory=None):
        self.config = config
        config.paths.ensure()
        config.renders_dir.mkdir(parents=True, exist_ok=True)
        self.settings = Settings(config.settings_path)
        self.token = read_or_create_token(config.token_path)
        self.store = Store(config.data_dir)
        self.llm = ModelClient(self.settings, link_factory=link_factory)
        self.generator = Generator(self)
        self.started = time.time()

    def start(self) -> None:
        if self.config.background:
            self.generator.start()

    def stop(self) -> None:
        self.generator.stop()

    # ------------------------------------------------------------------ models
    def create_model(self, name: str = "", size: Optional[Iterable[int]] = None, palette: Optional[list[str]] = None) -> VoxelModel:
        model = VoxelModel.empty(self.store.new_id(), (name or "Sin título").strip()[:120], size or self.settings["new_size"],
                                 palette or DEFAULT_PALETTE)
        return self.store.save(model)

    def apply(self, model_id: str, ops: list[Any], *, note: str = "") -> dict[str, Any]:
        model = self.store.get(model_id)
        result = apply_ops(model, ops, atomic=True)
        self.store.save(model, version_note=note or f"{len(ops)} operations")
        return {**result, "id": model.id, "bounds": model.bounds(), "size": model.size}

    # ------------------------------------------------------------------ pictures as sprite views
    def reference_sheet(self, ref_id: str, cols: int = 0, rows: int = 0):
        from . import sheets
        return sheets.read(self.store.reference_bytes(ref_id), int(cols or 0), int(rows or 0))

    def expand_references(self, entries: list[dict[str, Any]]) -> list[dict[str, Any]]:
        """Reference entries ({id, role, frames?, views?, cols?, rows?, yaw?, pitch?}) as the pictures they stand for:
        a sheet (role ``sheet``) becomes one entry per side it shows (its views, or the guessed ones), an entry with
        ``frames`` takes those frames of the picture (the first is the main one, the rest poses), an animation without
        frames gives its standing frame with the others as poses, and a plain picture stays itself. Each result has
        role, name, meta, png (the main picture), rgba (cut out, real pixel size), poses, yaw, pitch."""
        from PIL import Image
        import io
        from . import sheets
        from .images import cutout
        out: list[dict[str, Any]] = []
        cache: dict[tuple, Any] = {}
        for e in entries or []:
            rid = str(e.get("id") or "")
            role = str(e.get("role") or "front")
            meta = self.store.reference(rid)
            raw = self.store.reference_bytes(rid)
            name = meta.get("name") or role
            cols, rows = int(e.get("cols") or 0), int(e.get("rows") or 0)

            def sheet(rid=rid, raw=raw, cols=cols, rows=rows):
                key = (rid, cols, rows)
                if key not in cache:
                    cache[key] = sheets.read(raw, cols, rows)
                return cache[key]

            def frames_entry(sh, vrole, idx, yaw=None, pitch=None):
                main = sh.frame(idx[0])
                h, w = main.rgba.shape[:2]
                return {"id": rid, "role": vrole, "name": f"{name} #{idx[0]}", "meta": {**meta, "name": f"{name} #{idx[0]}", "width": w, "height": h},
                        "png": main.png(), "rgba": main.rgba, "poses": [sh.frame(i).rgba for i in idx[1:]],
                        "yaw": yaw, "pitch": pitch, "frames": list(idx)}

            if role == "sheet":
                sh = sheet()
                views = e.get("views") or sheets.guess_views(sh)
                for v in views:
                    vrole = str(v.get("role") or "none")
                    idx = [int(i) for i in (v.get("frames") or [])]
                    if vrole in ("front", "back", "side", "view") and idx:
                        out.append(frames_entry(sh, vrole, idx, v.get("yaw"), v.get("pitch")))
                continue
            if e.get("frames"):
                out.append(frames_entry(sheet(), role, [int(i) for i in e["frames"]], e.get("yaw"), e.get("pitch")))
                continue
            try:
                animated = int(getattr(Image.open(io.BytesIO(raw)), "n_frames", 1) or 1) > 1
            except Exception:  # noqa: BLE001
                animated = False
            if animated:
                sh = sheet()
                main = sheets.medoid(sh.frames)
                idx = [main.index] + [f.index for f in sh.frames if f is not main]
                out.append({**frames_entry(sh, role, idx, e.get("yaw"), e.get("pitch")), "name": name})
                continue
            out.append({"id": rid, "role": role, "name": name, "meta": meta, "png": raw, "rgba": cutout(raw), "poses": [],
                        "yaw": e.get("yaw"), "pitch": e.get("pitch"), "frames": []})
        return out

    def sprite_refs(self, entries: list[dict[str, Any]]) -> list:
        """The sprite views (``spritecam.SpriteRef``) of reference entries; see :meth:`expand_references`."""
        from . import spritecam
        return [spritecam.SpriteRef(e["role"], e["name"], e["rgba"],
                                    yaw=None if e.get("yaw") is None else float(e["yaw"]),
                                    pitch=None if e.get("pitch") is None else float(e["pitch"]), poses=list(e["poses"]))
                for e in self.expand_references(entries) if e["role"] in ("front", "back", "side", "view")]

    def run_program(self, code: str, *, model_id: str = "", name: str = "", size: Optional[Iterable[int]] = None,
                    sprites: Optional[list[dict[str, str]]] = None) -> dict[str, Any]:
        """Run a voxel program (PALETTE + build(g), see program.py) into a model: an existing one (its grid and palette
        are replaced, the previous content stays as a version) or a new one. ``sprites`` ([{id, role}] with role front,
        back or side) gives the program the sprite tools (SPRITE_PALETTE, SIDE, carve_side, paint_view...); a new model
        then gets a grid of one voxel per sprite pixel."""
        from . import spritecam
        from .program import ProgramError, run_program
        refs = self.sprite_refs(sprites or [])
        if model_id:
            model = self.store.get(model_id)
            dims = list(size) if size else model.size
        else:
            dims = list(size or (spritecam.grid_size(refs) if refs else self.settings["new_size"]))
            model = VoxelModel.empty(self.store.new_id(), (name or "Programa").strip()[:120], dims)
        data = spritecam.program_data(refs, [int(v) for v in dims], float(self.settings["sprite_pitch"]),
                                      float(self.settings["sprite_side_pitch"])) if refs else None
        try:
            res = run_program(code, [int(v) for v in dims], timeout=float(self.settings["program_timeout_s"]), sprites=data)
        except ProgramError as exc:
            raise SeuratError("program_failed", f"The program failed: {exc.message}", exc.trace[-800:] or None) from None
        model.grid, model.palette = res.grid, res.palette
        model.source = {**(model.source or {}), "program": code}
        if name and model_id:
            model.name = name.strip()[:120]
        saved = self.store.save(model, version_note="program")
        return {"id": saved.id, "name": saved.name, "count": saved.count, "size": saved.size, "colors": len(saved.palette),
                "bounds": saved.bounds()}

    def replace(self, model_id: str, data: dict[str, Any]) -> VoxelModel:
        """Full content from the editor (voxels, palette, size, name)."""
        current = self.store.get(model_id)
        merged = {**current.to_dict(voxels=False), **{k: v for k, v in data.items() if k in ("name", "size", "palette", "voxels", "notes")}}
        merged["id"] = model_id
        if "voxels" not in data:
            merged["voxels"] = current.voxel_list()
        model = VoxelModel.from_dict(merged)
        model.created, model.source = current.created, current.source
        return self.store.save(model, version_note=str(data.get("version_note") or "editor"))

    def duplicate(self, model_id: str, name: str = "") -> VoxelModel:
        src = self.store.get(model_id)
        copy = src.copy()
        copy.id = self.store.new_id()
        copy.name = (name or src.name + " (copia)")[:120]
        copy.created = time.time()
        copy.source = {"kind": "copy", "of": src.id}
        return self.store.save(copy)

    def generate(self, params: dict[str, Any]):
        job = self.generator.submit(params)
        if not self.config.background:
            self.generator.jobs[job.id].status = "running"
            try:
                self.generator.run(job)
                job.status = "done"
            except SeuratError as exc:
                job.status, job.error = "failed", exc.message
            job.finished = time.time()
        return job

    # ------------------------------------------------------------------ status
    def status(self) -> dict[str, Any]:
        return {"version": __version__, "models": len(self.store.list_models()), "references": len(self.store.references()),
                "generating": self.generator.busy(), "llm": self.llm.status(), "settings": self.settings.get()}

    def health(self) -> dict[str, Any]:
        return {"models": len(self.store.list_models()), "generating": self.generator.busy()}
