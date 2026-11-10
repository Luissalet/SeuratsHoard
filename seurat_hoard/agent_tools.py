"""Tools for assistants. One catalogue drives ``/api/agent/*`` (the MCP bridge), the web UI (``/api/ui/call``) and
``mcp_server.py``. ``model_look`` and ``reference_look`` return pictures (MCP images) so an assistant can see the models."""

from __future__ import annotations

import base64
import time
from pathlib import Path
from typing import Any, Literal, Optional

from pydantic import BaseModel, Field

from . import builders, lit, sheets, spritecam
from .errors import SeuratError
from .exporters import FORMATS, export, from_vox, import_model
from .hoard_link.agentkit import Empty, Tool, ann
from .hoard_link.agentkit import call_tool as _call_tool
from .hoard_link.agentkit import tool_catalog as _catalog
from .hoard_link.agentkit import uncapped  # noqa: F401 - re-exported for the UI routes
from .images import index_picture, to_png
from .render import VIEWS, png_bytes, render
from .rounds import list_snapshots, load_snapshot
from .services import Services
from .voxels import OPS, norm_hex

AGENT_INSTRUCTIONS = """Seurat's Hoard is a voxel studio: voxel models (a grid of palette colours, y up, z towards the viewer) that the user draws in a 3D editor, and pictures turned into voxel models.
Pictures to voxels: reference_add (a PNG/JPG path on this PC, or base64) gives a reference id. build_from_image makes an instant model from it (inflate = puffed-up front, extrude = slab, hull = front + side silhouettes carved). generate_start asks the language model (DeepSeek on the Sparks by default, reasoning at length: minutes per round) for a real voxel version: it sees the pictures, sculpts the shape with operations, Seurat paints the pictures onto it, and it reviews its own renders for review rounds; with no pictures, a prompt and a size build an object or a whole scene from the description. Poll generate_status until done, then model_look.
Editing: model_ops applies operations (voxel, voxels, box, line, sphere, cylinder, layer with pixel rows, recolor, mirror, move, clear; c is a palette index, c:0 erases); model_get shows the palette and bounds. Every change keeps the previous content as a version (model_versions / model_restore).
model_look returns a picture of a model (sheet = six views, or one view; lit views have real light: shadows, ambient occlusion, a ground; any yaw/pitch). model_export writes .vox (MagicaVoxel), .glb, .stl (printing, millimetres) or PNG renders / isometric sprites and returns the path; model_spin writes a turntable GIF or WebM.
Sprite sheets and animations: reference_frames reads every frame of a GIF or every cell of a sheet and guesses which frames show which side (walk sheets: rows are directions, columns are steps); pass the reference with role sheet (or role front/back/side with frames) to generate_start or model_sprite_match and each side becomes a camera view, its other frames poses. model_sprite_calibrate finds the camera angles from which a model's silhouette matches each sprite (and can save them as settings); generations do it by themselves after the runs.
Scenes: generate_start with method scene plans a big scene (up to 256 a side) as objects, builds each in its own program and places them. generate_round opens any round of a generation (its voxels are kept) and can save it as a model."""


def _d(first: str, detail: str = "", synonyms: str = "") -> str:
    parts = [first]
    if detail:
        parts.append(detail)
    if synonyms:
        parts.append("Sinónimos: " + synonyms)
    return "\n".join(parts)


def _img(data: bytes) -> dict[str, str]:
    return {"data": base64.b64encode(data).decode(), "mime": "image/png"}


# ---------------------------------------------------------------------------------------------------------- arguments
class ModelId(BaseModel):
    model_id: str = Field(..., min_length=1, max_length=64)


class CreateArgs(BaseModel):
    name: str = Field("", max_length=120)
    size: Optional[list[int]] = Field(None, min_length=3, max_length=3, description="[x, y, z] voxels, each 1..256; default from settings.")


class GetArgs(ModelId):
    voxels: bool = Field(False, description="Include the voxel list [[x,y,z,c],...] (can be long).")


class LookArgs(ModelId):
    view: Literal[VIEWS] = Field("sheet", description="sheet (six labelled views; sheet_lit with the two 3-D views lit) or one of front, back, left, right, top, bottom, iso, iso_front_left, iso_back_left, iso_back_right, a game overworld camera (looking down): overworld_front, overworld_back (45 degrees above), overworld_left, overworld_right (true profile), or a LIT view with real light (cast shadows, ambient occlusion, a ground): lit (front-right), lit_front_left, lit_back_left, lit_back_right, lit_front, lit_top.")
    size: Optional[int] = Field(None, ge=1, le=2048, description="Pixels per voxel (flat views), face size (isometric) or the long side in pixels (lit views, default 640).")
    yaw: Optional[float] = Field(None, ge=-360, le=360, description="Lit views: the camera's degrees around the model (0 = in front, 90 = from the right, 180 = behind).")
    pitch: Optional[float] = Field(None, ge=-89, le=89, description="Lit views: the camera's degrees above the horizon.")


class OpsArgs(ModelId):
    ops: list[dict[str, Any]] = Field(..., description="Operations, applied in order and all-or-nothing: " + ", ".join(sorted(OPS))
                                      + ". See the app instructions or generate's prompt for the fields.")
    note: str = Field("", max_length=200, description="What this change is (kept with the previous version).")


class ProgramArgs(BaseModel):
    code: str = Field(..., min_length=10, max_length=60000, description="Python: PALETTE = ['#rrggbb', ...] (index 1 = first) and def build(g) filling g[x, y, z] (numpy uint8, x right, y up, z towards the viewer). Ready: np, math, X, Y, Z, xs/ys/zs, ellipsoid, sphere, capsule, box, polygon, mirror_x, shade_by_height.")
    model_id: str = Field("", max_length=64, description="Replace this model's content (kept as a version); empty = a new model.")
    name: str = Field("", max_length=120)
    size: Optional[list[int]] = Field(None, min_length=3, max_length=3, description="[X, Y, Z]; default the model's size, one voxel per sprite pixel with sprites, or settings.new_size.")
    sprites: list["SpriteRole"] = Field(default_factory=list, max_length=6, description="Game sprites (reference ids with role front, back or side) for the sprite tools: SPRITE_PALETTE, SIDE (Z, Y) = the side sprite in grid coordinates, SIDE_MASK, extrude_side(g, x0, x1, c=None), carve_side(g), paint_side(g, depth=1), paint_view(g, role, depth=1.5).")


class RenameArgs(ModelId):
    name: str = Field(..., min_length=1, max_length=120)


class ResizeArgs(ModelId):
    size: list[int] = Field(..., min_length=3, max_length=3)
    anchor: Literal["bottom-center", "origin"] = "bottom-center"


class DeleteArgs(ModelId):
    confirm: bool = Field(False, description="Must be true. The model moves to data/trash.")


class RestoreArgs(ModelId):
    version: str = Field(..., min_length=1, max_length=64)


class PaletteArgs(ModelId):
    palette: Optional[list[str]] = Field(None, description="Whole palette (#rrggbb list; index 1 = first).")
    set: Optional[dict[str, str]] = Field(None, description="Change some colours: {\"3\": \"#ff0000\"}.")


class ExportArgs(ModelId):
    format: Literal["vox", "glb", "stl", "png", "sprites"] = "vox"
    view: Literal[VIEWS] = Field("iso", description="png only.")
    size: Optional[int] = Field(None, ge=1, le=64)
    folder: str = Field("", max_length=500, description="Where to write; empty = data/exports.")


class ImportArgs(BaseModel):
    path: str = Field(..., min_length=1, max_length=500, description="A MagicaVoxel .vox or Minecraft .schem (WorldEdit/Sponge) file on this PC.")
    name: str = Field("", max_length=120)


class RefAddArgs(BaseModel):
    path: str = Field("", max_length=500, description="A picture file on this PC (PNG, JPG, GIF, WebP, BMP).")
    data_base64: str = Field("", description="Or the picture itself, base64.")
    name: str = Field("", max_length=120)


class RefId(BaseModel):
    reference_id: str = Field(..., min_length=1, max_length=64)


class RefGridArgs(RefId):
    max_side: int = Field(32, ge=4, le=128)
    colors: int = Field(16, ge=2, le=48)


class RefDeleteArgs(RefId):
    confirm: bool = False


class BuildArgs(BaseModel):
    reference_id: str = Field(..., description="Front view.")
    mode: Literal["sprite", "inflate", "extrude", "hull"] = Field("sprite", description="sprite: the picture on front and back with a gentle rounded thickness and the 2D outline kept off the sides (sprite-to-voxel game mod look); inflate: puffed up; extrude: a slab; hull: front + side silhouettes.")
    side_id: str = Field("", description="hull: the side view (the model seen from its left).")
    top_id: str = Field("", description="hull: optional top view.")
    depth: int = Field(0, ge=0, le=256, description="inflate/extrude thickness; 0 = automatic.")
    max_side: int = Field(32, ge=4, le=128)
    colors: int = Field(16, ge=2, le=48)
    name: str = Field("", max_length=120)


class ViewSpec(BaseModel):
    role: Literal["front", "back", "side", "view", "none"]
    frames: list[int] = Field(default_factory=list, max_length=96, description="Frame numbers (reference_frames): the first is the main one, the rest poses.")
    yaw: Optional[float] = Field(None, ge=-360, le=360)
    pitch: Optional[float] = Field(None, ge=0, le=89)


class RefRole(BaseModel):
    id: str
    role: Literal["front", "side", "top", "back", "extra", "sheet", "view"] = Field("front", description="What the picture shows. sheet = a sprite sheet or an animation: all its sides are read from its frames (views, or guessed: rows of a walk sheet are directions, columns are steps). view = a side whose camera is unknown (found by calibration).")
    frames: list[int] = Field(default_factory=list, max_length=96, description="Use these frames of a sheet or animation (reference_frames numbers): the first is the main view, the rest poses of the same side. Empty: the whole picture (an animation: its standing frame, the others as poses).")
    views: list[ViewSpec] = Field(default_factory=list, max_length=12, description="role sheet: which frames show which side; empty = guessed.")
    cols: int = Field(0, ge=0, le=64, description="Sheets: columns of the grid (0 = found from the gaps).")
    rows: int = Field(0, ge=0, le=64, description="Sheets: rows of the grid (0 = found from the gaps).")
    yaw: Optional[float] = Field(None, ge=-360, le=360, description="The camera of this view when known (degrees around: 0 front, 90 right, 180 back).")
    pitch: Optional[float] = Field(None, ge=0, le=89, description="The camera's degrees above the horizon when known.")


class GenerateArgs(BaseModel):
    references: list[RefRole] = Field(default_factory=list, max_length=6, description="Pictures with their role; the first front is the main view. Empty: build from the prompt alone (a description of an object or a scene).")
    prompt: str = Field("", max_length=2000, description="What the user wants (style, what it is, what to stress).")
    name: str = Field("", max_length=120)
    into: str = Field("", max_length=64, description="Save as a new version of this model instead of a new model.")
    start: Literal["auto", "inflate", "hull", "extrude", "empty", "current"] = "auto"
    rounds: Optional[int] = Field(None, ge=0, le=6, description="Review rounds after the first answer; default from settings.")
    max_side: Optional[int] = Field(None, ge=4, le=128)
    colors: Optional[int] = Field(None, ge=2, le=48)
    depth: int = Field(0, ge=0, le=256)
    size: Optional[list[int]] = Field(None, min_length=3, max_length=3, description="Grid [x, y, z] when there are no pictures; default from settings (new_size; scenes: scene_size, up to 256 a side).")
    effort: Optional[Literal["off", "low", "medium", "high", "max"]] = None
    use_model: bool = Field(True, description="false = only the starting model (no language model).")
    auto_paint: Optional[bool] = Field(None, description="Paint the reference pictures onto the sculpted shape after every round; default from settings.")
    method: Optional[Literal["agent", "code", "ops", "scene", "parts"]] = Field(None, description="parts (the default when there is a side sprite): a character from its sprites part by part, the model decides what each part of the profile is (body, wings, legs, fan, the far wing only drawn for looks) and Seurat does the geometry; agent (default otherwise): the model runs, looks at and fixes its voxel program in a loop until it is done; code: fixed review rounds of programs; ops: JSON operations; scene: a big scene by parts (the model plans objects, builds each in its own program, Seurat places them; rounds > 0 adds one layout check).")
    steps: Optional[int] = Field(None, ge=2, le=60, description="agent: actions allowed (run, look, done); scene: actions per object; default from settings.")
    calibrate: Optional[bool] = Field(None, description="Sprite target: find each sprite's camera from the model's silhouette after the runs (default from settings, on).")
    minutes: Optional[int] = Field(None, ge=5, le=600, description="agent: time budget; default from settings.")
    continue_from: str = Field("", max_length=64, description="Carry on from this model; new versions go to it. A model built by parts carries on by parts from its plan and its own sprites, with the prompt as what to change (no references needed). agent: from the model's voxel program and brief (e.g. a step of an earlier generation, restored first).")
    target: Literal["auto", "sprite", "free"] = Field("auto", description="code/agent: sprite = the front/back/side pictures are game sprites to rebuild in 3D (judged from the game camera after every run); free = pictures are loose hints; auto = sprite when they are small pixel art.")
    sprite_pitch: Optional[int] = Field(None, ge=0, le=89, description="Sprite target: degrees above the horizon of the camera of the front and back sprites; default from settings (45).")
    sprite_side_pitch: Optional[int] = Field(None, ge=0, le=89, description="Sprite target: degrees above the horizon of the camera of the side sprites; default from settings (0: a true profile).")


class SpriteRole(BaseModel):
    id: str
    role: Literal["front", "back", "side", "view", "sheet"] = Field("front", description="front, back, side, view (camera unknown: found by calibration) or sheet (a sprite sheet or animation: its sides are read from its frames).")
    frames: list[int] = Field(default_factory=list, max_length=96, description="Frames of a sheet/animation to use (first = main view, rest = poses).")
    views: list[ViewSpec] = Field(default_factory=list, max_length=12, description="role sheet: which frames show which side; empty = guessed.")
    cols: int = Field(0, ge=0, le=64)
    rows: int = Field(0, ge=0, le=64)
    yaw: Optional[float] = Field(None, ge=-360, le=360)
    pitch: Optional[float] = Field(None, ge=0, le=89)


ProgramArgs.model_rebuild()


class SpriteMatchArgs(ModelId):
    references: list[SpriteRole] = Field(..., min_length=1, max_length=6, description="The game sprites (reference ids) with what they show: front (facing the camera), back (from behind), side, view, or sheet (all the sides of a sprite sheet / animation).")
    sprite_pitch: Optional[int] = Field(None, ge=0, le=89, description="Degrees above the horizon of the camera of the front and back sprites; default from settings (45).")
    sprite_side_pitch: Optional[int] = Field(None, ge=0, le=89, description="Degrees above the horizon of the camera of the side sprites; default from settings (0: a true profile).")


class CalibrateArgs(SpriteMatchArgs):
    wide: bool = Field(True, description="true: search every plausible camera (front/back 0-80 degrees above, 20 around; free views all around); false: only near the current one (as generations do).")
    apply: bool = Field(False, description="Save the front/back and side camera heights found as the settings sprite_pitch and sprite_side_pitch.")


class FramesArgs(BaseModel):
    reference_id: str = Field(..., min_length=1, max_length=64)
    cols: int = Field(0, ge=0, le=64, description="Columns of the sheet's grid; 0 = found from the gaps between the sprites.")
    rows: int = Field(0, ge=0, le=64, description="Rows of the sheet's grid; 0 = found from the gaps.")


class SpinArgs(ModelId):
    format: Literal["gif", "webm"] = Field("gif", description="gif (loops anywhere) or webm (VP9 video, smaller; needs ffmpeg).")
    frames: int = Field(36, ge=8, le=120, description="Frames in one turn.")
    seconds: float = Field(3.0, ge=0.5, le=20, description="Length of one turn.")
    pitch: float = Field(25.0, ge=-10, le=80, description="Camera degrees above the horizon.")
    size: int = Field(384, ge=96, le=1024, description="Width in pixels.")
    ground: bool = Field(True, description="A ground under the model that catches its shadow.")
    folder: str = Field("", max_length=500, description="Where to write; empty = data/exports.")


class RoundArgs(BaseModel):
    job_id: str = Field(..., min_length=1, max_length=64)
    round: int = Field(..., ge=0, le=100000)
    save: bool = Field(False, description="Save that round as a new model (to open and edit it); false = only look at it.")
    name: str = Field("", max_length=120)


class JobId(BaseModel):
    job_id: str = Field(..., min_length=1, max_length=64)


class SettingsArgs(BaseModel):
    max_side: Optional[int] = Field(None, ge=4, le=128)
    colors: Optional[int] = Field(None, ge=2, le=48)
    review_rounds: Optional[int] = Field(None, ge=0, le=6)
    effort: Optional[Literal["off", "low", "medium", "high", "max"]] = None
    max_tokens: Optional[int] = Field(None, ge=1000, le=128000)
    thinking_tokens: Optional[int] = Field(None, ge=0, le=900000)
    timeout_s: Optional[int] = Field(None, ge=30, le=7200)
    auto_paint: Optional[bool] = None
    paint_sides: Optional[bool] = Field(None, description="Also paint the side picture on both sides.")
    method: Optional[Literal["agent", "code", "ops"]] = None
    agent_steps: Optional[int] = Field(None, ge=2, le=60)
    agent_minutes: Optional[int] = Field(None, ge=5, le=600)
    agent_step_thinking: Optional[int] = Field(None, ge=0, le=200000)
    code_size: Optional[int] = Field(None, ge=8, le=128)
    program_timeout_s: Optional[int] = Field(None, ge=5, le=600)
    sprite_pitch: Optional[int] = Field(None, ge=0, le=89, description="Sprite target: camera height (degrees) of front/back sprites.")
    sprite_side_pitch: Optional[int] = Field(None, ge=0, le=89, description="Sprite target: camera height (degrees) of side sprites.")
    sprite_calibrate: Optional[bool] = Field(None, description="Sprite target: find each sprite's camera from the model's silhouette during generations.")
    review_render: Optional[Literal["lit", "flat"]] = Field(None, description="The 3-D views of review sheets and looks: lit (shadows, ambient occlusion) or flat isometric.")
    scene_steps: Optional[int] = Field(None, ge=2, le=30, description="Scenes by parts: agent actions per object.")
    scene_objects: Optional[int] = Field(None, ge=1, le=16, description="Scenes by parts: most objects in a plan.")
    scene_size: Optional[list[int]] = Field(None, min_length=3, max_length=3, description="Scenes by parts: default scene grid (16-256 a side).")
    parallel_jobs: Optional[int] = Field(None, ge=1, le=6, description="Generations at the same time (after a restart).")
    llm_url: Optional[str] = Field(None, max_length=300)
    llm_model: Optional[str] = Field(None, max_length=120)
    voxel_mm: Optional[float] = Field(None, gt=0, le=50)
    glb_voxel_size: Optional[float] = Field(None, gt=0, le=10)
    new_size: Optional[list[int]] = Field(None, min_length=3, max_length=3)


# ---------------------------------------------------------------------------------------------------------- handlers
def _summary(svc: Services, model_id: str, voxels: bool = False) -> dict[str, Any]:
    m = svc.store.get(model_id)
    out = m.to_dict(voxels=voxels)
    out["colors_used"] = m.colors_used()
    return out


def _look(svc: Services, a: LookArgs) -> dict[str, Any]:
    m = svc.store.get(a.model_id)
    if m.count == 0:
        raise SeuratError("invalid", "The model is empty.")
    data = png_bytes(render(m, a.view, size=a.size, background="dark" if not a.view.startswith("sheet") else "transparent",
                            yaw=a.yaw, pitch=a.pitch))
    return {"id": m.id, "name": m.name, "view": a.view, "count": m.count, "size": m.size, "_image": _img(data)}


def _sprite_refs(svc: Services, a: SpriteMatchArgs) -> tuple[list, float, float]:
    refs = svc.sprite_refs([r.model_dump() for r in a.references])
    if not refs:
        raise SeuratError("invalid", "None of those pictures gives a front, back, side or view sprite.",
                          "With role sheet, check reference_frames: its guessed views may all be 'none'; pass views.")
    fb = float(a.sprite_pitch if a.sprite_pitch is not None else svc.settings["sprite_pitch"])
    side = float(a.sprite_side_pitch if a.sprite_side_pitch is not None else svc.settings["sprite_side_pitch"])
    return refs, fb, side


def _sprite_match(svc: Services, a: SpriteMatchArgs) -> dict[str, Any]:
    m = svc.store.get(a.model_id)
    if m.count == 0:
        raise SeuratError("invalid", "The model is empty.")
    refs, fb, side = _sprite_refs(svc, a)
    matches = spritecam.match(m, refs, fb, side)
    return {"id": m.id, "name": m.name, "match": round(spritecam.mean_iou(matches), 3), "views": [v.to_dict() for v in matches],
            "summary": [v.summary() for v in matches], "_image": _img(png_bytes(spritecam.sheet(matches)))}


def _calibrate(svc: Services, a: CalibrateArgs) -> dict[str, Any]:
    m = svc.store.get(a.model_id)
    if m.count == 0:
        raise SeuratError("invalid", "The model is empty.")
    refs, fb, side = _sprite_refs(svc, a)
    before = spritecam.mean_iou(spritecam.match(m, refs, fb, side))
    cams = spritecam.calibrate(m, refs, fb, side, search=None if a.wide else spritecam.SEARCH_GENERATION)
    spritecam.apply_cameras(refs, cams, min_gain=-1.0)          # every sprite takes the best camera found
    matches = spritecam.match(m, refs, fb, side)
    suggested = spritecam.pitches(cams)
    applied: dict[str, Any] = {}
    if a.apply:
        applied = {k: v for k, v in suggested.items() if v is not None}
        if applied:
            svc.settings.update(applied)
    return {"id": m.id, "name": m.name, "cameras": cams, "match_before": round(before, 3),
            "match": round(spritecam.mean_iou(matches), 3), "suggested_settings": suggested, "applied": applied,
            "summary": [v.summary() for v in matches], "_image": _img(png_bytes(spritecam.sheet(matches)))}


def _frames(svc: Services, a: FramesArgs) -> dict[str, Any]:
    sh = svc.reference_sheet(a.reference_id, a.cols, a.rows)
    views = sheets.guess_views(sh)
    return {"reference_id": a.reference_id, **sh.to_dict(), "views": views,
            "_image": _img(sheets.png_bytes(sheets.contact(sh, views)))}


def spin_file(svc: Services, model_id: str, fmt: str, frames: int = 36, seconds: float = 3.0, pitch: float = 25.0,
              size: int = 384, ground: bool = True) -> tuple[bytes, str]:
    m = svc.store.get(model_id)
    data = lit.spin(m, fmt, frames=frames, pitch=pitch, size=size, seconds=seconds, ground=ground)
    return data, f"{safe_name(m.name)} giro.{fmt}"


def _spin(svc: Services, a: SpinArgs) -> dict[str, Any]:
    data, filename = spin_file(svc, a.model_id, a.format, a.frames, a.seconds, a.pitch, a.size, a.ground)
    path = _exports_dir(svc, a.folder) / filename
    path.write_bytes(data)
    return {"path": str(path), "bytes": len(data), "format": a.format, "frames": a.frames, "mime": lit.SPIN_FORMATS[a.format]}


def round_model(svc: Services, job_id: str, rnd: int, save: bool = False, name: str = "") -> tuple[Any, dict[str, Any]]:
    """A round of a generation as a model (and, with ``save``, stored as a new one)."""
    job = svc.generator.jobs.get(job_id)
    base = (job.model_name if job and job.model_name else (job.params.get("name") if job else "")) or "Generación"
    model = load_snapshot(svc.config.renders_dir, job_id, rnd, name or f"{base} (ronda {rnd})")
    out: dict[str, Any] = {"job_id": job_id, "round": rnd, "name": model.name, "size": model.size, "count": model.count,
                           "colors": len(model.palette), "rounds_with_voxels": list_snapshots(svc.config.renders_dir, job_id)}
    if save:
        model.id = svc.store.new_id()
        model.source = {"kind": "round", "job": job_id, "round": rnd}
        svc.store.save(model)
        out["id"] = model.id
    return model, out


def _round(svc: Services, a: RoundArgs) -> dict[str, Any]:
    model, out = round_model(svc, a.job_id, a.round, a.save, a.name)
    if model.count:
        out["_image"] = _img(png_bytes(render(model, "lit", size=640, background="dark")))
    return out


def _palette(svc: Services, a: PaletteArgs) -> dict[str, Any]:
    m = svc.store.get(a.model_id)
    if a.palette is not None:
        pal = [norm_hex(c) for c in a.palette][:255]
        if len(pal) < max(m.colors_used() or {0: 0}):
            raise SeuratError("invalid", f"The model uses index {max(m.colors_used())}; the palette needs at least that many colours.")
        m.palette = pal
    for k, v in (a.set or {}).items():
        i = int(k)
        if not 1 <= i <= 255:
            raise SeuratError("invalid", f"Index {i} outside 1..255.")
        while len(m.palette) < i:
            m.palette.append("#808080")
        m.palette[i - 1] = norm_hex(v)
    svc.store.save(m, version_note="palette")
    return {"id": m.id, "palette": m.palette}


def _exports_dir(svc: Services, folder: str) -> Path:
    path = Path(folder).expanduser() if folder else svc.config.data_dir / "exports"
    path.mkdir(parents=True, exist_ok=True)
    return path


def safe_name(name: str) -> str:
    keep = "".join(ch if ch.isalnum() or ch in " -_()" else "_" for ch in name).strip()
    return keep[:80] or "modelo"


def export_model(svc: Services, model_id: str, fmt: str, view: str = "iso", size: Optional[int] = None) -> tuple[bytes, str]:
    m = svc.store.get(model_id)
    s = svc.settings
    data = export(m, fmt, view=view, size=size, voxel_mm=s["voxel_mm"], voxel_size=s["glb_voxel_size"])
    ext = {"sprites": "png"}.get(fmt, fmt)
    suffix = {"sprites": " sprites", "png": f" {view}"}.get(fmt, "")
    return data, f"{safe_name(m.name)}{suffix}.{ext}"


def _export(svc: Services, a: ExportArgs) -> dict[str, Any]:
    data, filename = export_model(svc, a.model_id, a.format, a.view, a.size)
    path = _exports_dir(svc, a.folder) / filename
    path.write_bytes(data)
    return {"path": str(path), "bytes": len(data), "format": a.format, "mime": FORMATS[a.format]}


def _import(svc: Services, a: ImportArgs) -> dict[str, Any]:
    path = Path(a.path).expanduser()
    if not path.is_file():
        raise SeuratError("not_found", f"No file {a.path}.")
    m = import_model(path.read_bytes(), svc.store.new_id(), a.name or path.stem)
    m.source = {**(m.source or {}), "file": path.name}
    svc.store.save(m)
    return {"id": m.id, "name": m.name, "size": m.size, "count": m.count}


def _ref_add(svc: Services, a: RefAddArgs) -> dict[str, Any]:
    if a.path:
        path = Path(a.path).expanduser()
        if not path.is_file():
            raise SeuratError("not_found", f"No file {a.path}.")
        data, name = path.read_bytes(), a.name or path.stem
    elif a.data_base64:
        try:
            data = base64.b64decode(a.data_base64.split(",", 1)[-1], validate=False)
        except ValueError:
            raise SeuratError("invalid", "data_base64 is not base64.") from None
        name = a.name
    else:
        raise SeuratError("invalid", "Give path or data_base64.")
    return svc.store.add_reference(data, name)


def _ref_look(svc: Services, a: RefId) -> dict[str, Any]:
    meta = svc.store.reference(a.reference_id)
    return {**meta, "_image": _img(to_png(svc.store.reference_bytes(a.reference_id), 1024))}


def _ref_grid(svc: Services, a: RefGridArgs) -> dict[str, Any]:
    ix = index_picture(svc.store.reference_bytes(a.reference_id), max_side=a.max_side, colors=a.colors)
    return {"reference_id": a.reference_id, **ix.to_dict()}


def _build(svc: Services, a: BuildArgs) -> dict[str, Any]:
    front = index_picture(svc.store.reference_bytes(a.reference_id), max_side=a.max_side, colors=a.colors)
    side = index_picture(svc.store.reference_bytes(a.side_id), max_side=a.max_side, colors=a.colors) if a.side_id else None
    top = index_picture(svc.store.reference_bytes(a.top_id), max_side=a.max_side, colors=a.colors) if a.top_id else None
    name = a.name or svc.store.reference(a.reference_id)["name"] + f" ({a.mode})"
    model = builders.build(a.mode, front, side=side, top=top, depth=a.depth, model_id=svc.store.new_id(), name=name)
    model.source = {"kind": "built", "mode": a.mode, "references": [r for r in (a.reference_id, a.side_id, a.top_id) if r]}
    svc.store.save(model)
    return {"id": model.id, "name": model.name, "size": model.size, "count": model.count}


def _generate(svc: Services, a: GenerateArgs) -> dict[str, Any]:
    params = a.model_dump()
    params["references"] = [r.model_dump() for r in a.references]
    job = svc.generate(params)
    return {"job_id": job.id, "status": job.status, "hint": "Poll generate_status; it takes from a few seconds to several minutes."}


def _settings_set(svc: Services, a: SettingsArgs) -> dict[str, Any]:
    changes = a.model_dump(exclude_none=True)
    if not changes:
        raise SeuratError("invalid", "Nothing to change.")
    out = svc.settings.update(changes)
    svc.llm.status(refresh=True)
    return out


def _delete(svc: Services, a: DeleteArgs) -> dict[str, Any]:
    if not a.confirm:
        raise SeuratError("confirm_required", f"Model {a.model_id} would move to the trash.", "Repeat with confirm=true if the user asked for it.")
    return svc.store.delete(a.model_id)


def _ref_delete(svc: Services, a: RefDeleteArgs) -> dict[str, Any]:
    if not a.confirm:
        raise SeuratError("confirm_required", f"Reference {a.reference_id} would be deleted.", "Repeat with confirm=true.")
    return svc.store.delete_reference(a.reference_id)


def _resize(svc: Services, a: ResizeArgs) -> dict[str, Any]:
    m = svc.store.get(a.model_id)
    m.resize(a.size, a.anchor)
    svc.store.save(m, version_note=f"resize {a.size}")
    return {"id": m.id, "size": m.size, "count": m.count}


def _rename(svc: Services, a: RenameArgs) -> dict[str, Any]:
    m = svc.store.get(a.model_id)
    m.name = a.name.strip()
    svc.store.save(m, version_note="rename")
    return {"id": m.id, "name": m.name}


TOOLS: list[Tool] = [
    Tool("status", _d("What Seurat has: models, references, whether a generation runs and which language model would answer.",
                      synonyms="estado, modelo de lenguaje"), Empty, ann(read_only=True), lambda s, a: s.status()),
    Tool("model_list", _d("List voxel models, newest first (id, name, size, voxel count).", synonyms="modelos, listar"),
         Empty, ann(read_only=True), lambda s, a: {"models": s.store.list_models()}),
    Tool("model_create", _d("Create an empty voxel model.", synonyms="nuevo modelo"), CreateArgs, ann(draft_safe=True),
         lambda s, a: s.create_model(a.name, a.size).to_dict(voxels=False)),
    Tool("model_get", _d("A model's size, palette (index -> colour), colours used, bounds; voxels=true adds the voxel list."),
         GetArgs, ann(read_only=True), lambda s, a: _summary(s, a.model_id, a.voxels)),
    Tool("model_look", _d("See a model: returns a picture (sheet = six labelled views, or one view).", synonyms="ver modelo, render"),
         LookArgs, ann(read_only=True), _look, timeout_s=60, capped=False),
    Tool("model_sprite_match", _d("Compare a model with game sprites seen from the game camera: each sprite next to the model "
                                  "rendered from the same camera (front/back from well above, side nearly in profile) at the "
                                  "sprite's resolution, the silhouette diff and match scores (1 = identical).",
                                  "Use it to judge a voxel rebuild of overworld sprites; the agent generation does it after every run.",
                                  "comparar con sprites, coincidencia sprite"),
         SpriteMatchArgs, ann(read_only=True), _sprite_match, timeout_s=120, capped=False),
    Tool("model_program", _d("Build a model by running a voxel program (Python with numpy: PALETTE + build(g)) in a sandbox: "
                             "no files, no network, a time limit. The program is kept with the model.",
                             "Helpers: ellipsoid, sphere, capsule (tapered tube), box, polygon (flat shape with thickness), mirror_x, shade_by_height; numpy masks over xs, ys, zs.",
                             "programa voxel, código voxel, script"),
         ProgramArgs, ann(draft_safe=True), lambda s, a: s.run_program(a.code, model_id=a.model_id, name=a.name, size=a.size,
                                                                       sprites=[r.model_dump() for r in a.sprites]), timeout_s=120),
    Tool("model_ops", _d("Change a model with operations (all-or-nothing): voxel, voxels, box, line, sphere, cylinder, layer "
                         "(pixel rows on the front, side or top plane), recolor, mirror, move, clear. c = palette index, c:0 erases.",
                         "Axes: x right, y up, z towards the viewer. The previous content is kept as a version.",
                         "dibujar, editar, pintar voxels"),
         OpsArgs, ann(draft_safe=True), lambda s, a: s.apply(a.model_id, a.ops, note=a.note), timeout_s=60),
    Tool("model_rename", _d("Rename a model."), RenameArgs, ann(idempotent=True), _rename),
    Tool("model_resize", _d("Change a model's grid size (voxels stay anchored at the bottom centre)."), ResizeArgs, ann(), _resize),
    Tool("model_duplicate", _d("Copy a model."), ModelId, ann(draft_safe=True), lambda s, a: s.duplicate(a.model_id).to_dict(voxels=False)),
    Tool("model_palette", _d("Replace a model's palette or change some of its colours.", synonyms="paleta, colores"),
         PaletteArgs, ann(), _palette),
    Tool("model_versions", _d("Earlier versions of a model (kept on every change)."), ModelId, ann(read_only=True),
         lambda s, a: {"versions": s.store.versions(a.model_id)}),
    Tool("model_restore", _d("Bring back an earlier version (the current one becomes a version too)."), RestoreArgs, ann(),
         lambda s, a: s.store.restore(a.model_id, a.version).to_dict(voxels=False)),
    Tool("model_delete", _d("Move a model to the trash (confirm=true)."), DeleteArgs, ann(destructive=True), _delete),
    Tool("model_export", _d("Write a model to a file: vox (MagicaVoxel), glb (Blender, Godot, Unity, web), stl (3D printing, "
                            "millimetres), png (one view) or sprites (isometric from four sides). Returns the path.",
                            synonyms="exportar, imprimir, magicavoxel, sprite"),
         ExportArgs, ann(), _export, timeout_s=120),
    Tool("model_import_vox", _d("Import a MagicaVoxel .vox or a Minecraft .schem (WorldEdit/Sponge schematic: blocks become their "
                                "texture colours) from this PC as a new model.", synonyms="importar vox, importar schematic, minecraft"),
         ImportArgs, ann(draft_safe=True), _import),
    Tool("reference_add", _d("Add a reference picture (path on this PC or base64) to turn into voxels; returns its id.",
                             synonyms="subir imagen, sprite, referencia"),
         RefAddArgs, ann(draft_safe=True), _ref_add),
    Tool("reference_list", _d("List the reference pictures."), Empty, ann(read_only=True), lambda s, a: {"references": s.store.references()}),
    Tool("reference_look", _d("See a reference picture."), RefId, ann(read_only=True), _ref_look, capped=False),
    Tool("reference_grid", _d("A reference read as a small indexed grid (rows of characters, palette): what the builders and the model use."),
         RefGridArgs, ann(read_only=True), _ref_grid),
    Tool("reference_delete", _d("Delete a reference picture (confirm=true)."), RefDeleteArgs, ann(destructive=True), _ref_delete),
    Tool("build_from_image", _d("Instant voxel model from a picture without the language model: sprite (the sprite as a voxel diorama piece), inflate (puffed-up front), "
                                "extrude (slab) or hull (front + side silhouettes carved).", synonyms="convertir a voxel rápido"),
         BuildArgs, ann(draft_safe=True), _build, timeout_s=120),
    Tool("generate_start", _d("Ask the language model (DeepSeek on the Sparks by default) for a voxel version of one or more "
                              "pictures, or of a description alone (objects, characters, whole scenes). It sees the pictures, sculpts the shape with operations (ellipsoids, capsules, wing prisms), Seurat paints the "
                              "pictures onto it, and it reviews its own renders. With effort max a round takes several minutes. Returns a job id.",
                              "Roles: front (main), side (seen from the model's left), top, back, extra; sheet (a sprite sheet or "
                              "animation: every side it shows, its other frames as poses) and view (a side whose camera is found by "
                              "calibration). method scene: a big scene planned as objects, each built in its own program and placed.",
                              "generar voxel, deepseek, versión voxel de una imagen"),
         GenerateArgs, ann(draft_safe=True), _generate, timeout_s=60),
    Tool("generate_status", _d("Progress of a generation: log, rounds (with notes and errors) and, when done, the model id."),
         JobId, ann(read_only=True), lambda s, a: s.generator.get(a.job_id).to_dict()),
    Tool("generate_round", _d("Any round of a generation (its voxels are kept with its renders): returns a lit picture of it; "
                              "save=true stores it as a new model to open and edit.",
                              "Rounds are numbered as in generate_status (0 = the start or the scene plan).",
                              "ronda intermedia, ver ronda en 3D, recuperar ronda"),
         RoundArgs, ann(draft_safe=True), _round, timeout_s=120, capped=False),
    Tool("reference_frames", _d("Every frame of an animated picture (GIF, animated WebP/PNG) or every cell of a sprite sheet "
                                "(grid found from the gaps, or cols x rows), cut out, numbered, with a guess of which frames show "
                                "which side (front, back, side; the first of each is the standing frame, the rest poses). Returns a "
                                "numbered contact picture.",
                                "Then use the reference in generate_start / model_sprite_match with role sheet (all sides) or with a "
                                "role and frames.", "fotogramas, hoja de sprites, sprites de caminar, gif animado"),
         FramesArgs, ann(read_only=True), _frames, timeout_s=120, capped=False),
    Tool("model_sprite_calibrate", _d("Find the camera of each game sprite: the yaw and height from which the model's silhouette "
                                      "matches it best (a sweep of angles, then finer steps), instead of the fixed 45/0 degrees. "
                                      "Returns each camera, the match before and after and the comparison picture.",
                                      "apply=true saves the front/back and side heights found as settings; generations calibrate by "
                                      "themselves after their runs (setting sprite_calibrate).", "calibrar cámara, ángulo del sprite"),
         CalibrateArgs, ann(read_only=False), _calibrate, timeout_s=300, capped=False),
    Tool("model_spin", _d("Write a turntable animation of a model with real light (shadows, ambient occlusion, a ground): GIF or "
                          "WebM. Returns the path.", synonyms="animación de giro, gif giratorio, webm, turntable"),
         SpinArgs, ann(), _spin, timeout_s=300),
    Tool("generate_list", _d("Recent generations."), Empty, ann(read_only=True), lambda s, a: {"jobs": s.generator.list()}),
    Tool("generate_cancel", _d("Stop a generation (what it built so far is saved)."), JobId, ann(),
         lambda s, a: s.generator.cancel(a.job_id).to_dict()),
    Tool("settings_get", _d("Settings: reading size and colours, review rounds, reasoning effort and thinking tokens, auto paint, model URL, export scales."),
         Empty, ann(read_only=True), lambda s, a: s.settings.get()),
    Tool("settings_set", _d("Change settings."), SettingsArgs, ann(idempotent=True), _settings_set),
]

# ---------------------------------------------------------------------------------------------- characters by parts
class PartsRef(BaseModel):
    id: str
    role: Literal["side", "front", "back", "extra", "sheet"] = Field("side", description="side = the profile that is cut into parts (needed); front and back give widths and are compared with the model.")
    frames: list[int] = Field(default_factory=list, max_length=96)
    cols: int = Field(0, ge=0, le=64)
    rows: int = Field(0, ge=0, le=64)


class PartsReadArgs(BaseModel):
    references: list[PartsRef] = Field(..., min_length=1, max_length=6)
    facing: Literal["right", "left"] = Field("right", description="Where the head points in the side sprite.")


class PartsBuildArgs(BaseModel):
    references: list[PartsRef] = Field(..., min_length=1, max_length=6)
    plan: Optional[dict[str, Any]] = Field(None, description='{"facing": "right", "parts": [{"name", "type": solid | fin | fan | pair | wings | decal | skip, "where": [[col0, row0, col1, row1] or {"poly": [[col, row], ...]}], "colors": "letters", ...sizes}]}. Sizes by type: solid width; fin thickness; fan base, width; pair offset, width; wings root [col, row], span, out_up, out_back, thickness. Empty: the whole profile as one rounded solid.')
    name: str = Field("", max_length=120)
    into: str = Field("", max_length=64, description="Rebuild this model instead of making a new one (its voxels stay as a version).")


class PartsAdjustArgs(ModelId):
    set: dict[str, Any] = Field(default_factory=dict, description='Values to change, by part name: {"wings": {"span": 44, "out_up": 0.8}, "body": {"width": 14}}.')
    add: list[dict[str, Any]] = Field(default_factory=list, max_length=24, description="Whole parts to add (one with an existing name replaces it).")
    remove: list[str] = Field(default_factory=list, max_length=24)
    plan: Optional[dict[str, Any]] = Field(None, description="A whole new plan instead of changes.")


def _stack(pictures: list[bytes]) -> bytes:
    """Several PNGs one under another in one picture."""
    import io

    from PIL import Image
    imgs = [Image.open(io.BytesIO(b)).convert("RGBA") for b in pictures if b]
    out = Image.new("RGBA", (max(i.width for i in imgs), sum(i.height for i in imgs) + 6 * (len(imgs) - 1)), (12, 10, 14, 255))
    y = 0
    for i in imgs:
        out.paste(i, (0, y))
        y += i.height + 6
    buf = io.BytesIO()
    out.save(buf, "PNG")
    return buf.getvalue()


def _parts_out(res: dict[str, Any]) -> dict[str, Any]:
    pics = res.pop("pictures")
    return {**res, "_image": _img(_stack([pics["parts"], pics["match"], pics["lit"]])),
            "hint": "The picture: the side sprite and its pixels by part; each sprite against the model from its camera; the model lit. "
                    "Tune sizes with model_parts_adjust (\"far\": \"keep\" on a wings or pair part builds the far wing or limb Seurat left out)."}


def _parts_read(svc: Services, a: PartsReadArgs) -> dict[str, Any]:
    from . import partsgen
    res = partsgen.read_profile_info(svc, [r.model_dump() for r in a.references], a.facing)
    return {**{k: v for k, v in res.items() if k != "picture"}, "_image": _img(res["picture"])}


def _parts_build(svc: Services, a: PartsBuildArgs) -> dict[str, Any]:
    from . import partsgen
    return _parts_out(partsgen.build_model(svc, [r.model_dump() for r in a.references], a.plan, name=a.name, into=a.into))


def _parts_adjust(svc: Services, a: PartsAdjustArgs) -> dict[str, Any]:
    from . import partsgen
    return _parts_out(partsgen.adjust_model(svc, a.model_id, a.set or None, a.add or None, a.remove or None, a.plan))


TOOLS += [
    Tool("sprite_parts_read", _d("What a character's sprites say before building it by parts: the side sprite (the profile) as a "
                                 "pixel map with colour letters, the front and back maps, the widths they show row by row, and a "
                                 "picture of the sprites with rulers.",
                                 "Write the plan for model_from_parts from it: shapes are in the side sprite's columns and rows.",
                                 "leer sprites, perfil, mapa de píxeles"),
         PartsReadArgs, ann(read_only=True), _parts_read, timeout_s=60, capped=False),
    Tool("model_from_parts", _d("Build a character from its game sprites part by part, with no language model: the side sprite is "
                                "the exact profile and a plan says what each part is across the body (solid, fin, fan, pair, wings, "
                                "decal, skip for the far wing or leg a side sprite only draws for looks). Seurat does the geometry, "
                                "keeps the sprite's colours and compares the result with the front and back sprites.",
                                "Without a plan: the whole profile as one rounded solid. The plan stays with the model.",
                                "sprite a voxel por partes, personaje desde sprites, alas, plan de partes"),
         PartsBuildArgs, ann(draft_safe=True), _parts_build, timeout_s=180, capped=False),
    Tool("model_parts_adjust", _d("Change the plan of a model built by parts (a width, a wingspan, how the wings open, a part's "
                                  "shapes or type, parts added or removed) and rebuild it; the previous voxels stay as a version.",
                                  synonyms="ajustar partes, envergadura, ancho del cuerpo"),
         PartsAdjustArgs, ann(), _parts_adjust, timeout_s=180, capped=False),
]

TOOLS_BY_NAME = {t.name: t for t in TOOLS}


def tool_catalog() -> list[dict[str, Any]]:
    return _catalog(TOOLS)


def call_tool(svc: Services, name: str, arguments: Optional[dict[str, Any]], caller: str = "agent") -> Any:
    return _call_tool(TOOLS_BY_NAME, svc, name, arguments)
