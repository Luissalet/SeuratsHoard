"""Tools for assistants. One catalogue drives ``/api/agent/*`` (the MCP bridge), the web UI (``/api/ui/call``) and
``mcp_server.py``. ``model_look`` and ``reference_look`` return pictures (MCP images) so an assistant can see the models."""

from __future__ import annotations

import base64
import time
from pathlib import Path
from typing import Any, Literal, Optional

from pydantic import BaseModel, Field

from . import builders
from .errors import SeuratError
from .exporters import FORMATS, export, from_vox
from .hoard_link.agentkit import Empty, Tool, ann
from .hoard_link.agentkit import call_tool as _call_tool
from .hoard_link.agentkit import tool_catalog as _catalog
from .hoard_link.agentkit import uncapped  # noqa: F401 - re-exported for the UI routes
from .images import index_picture, to_png
from .render import VIEWS, png_bytes, render
from .services import Services
from .voxels import OPS, norm_hex

AGENT_INSTRUCTIONS = """Seurat's Hoard is a voxel studio: voxel models (a grid of palette colours, y up, z towards the viewer) that the user draws in a 3D editor, and pictures turned into voxel models.
Pictures to voxels: reference_add (a PNG/JPG path on this PC, or base64) gives a reference id. build_from_image makes an instant model from it (inflate = puffed-up front, extrude = slab, hull = front + side silhouettes carved). generate_start asks the language model (DeepSeek on the Sparks by default, reasoning at length: minutes per round) for a real voxel version: it sees the pictures, sculpts the shape with operations, Seurat paints the pictures onto it, and it reviews its own renders for review rounds; with no pictures, a prompt and a size build an object or a whole scene from the description. Poll generate_status until done, then model_look.
Editing: model_ops applies operations (voxel, voxels, box, line, sphere, cylinder, layer with pixel rows, recolor, mirror, move, clear; c is a palette index, c:0 erases); model_get shows the palette and bounds. Every change keeps the previous content as a version (model_versions / model_restore).
model_look returns a picture of a model (sheet = six views, or one view). model_export writes .vox (MagicaVoxel), .glb, .stl (printing, millimetres) or PNG renders / isometric sprites and returns the path."""


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
    view: Literal[VIEWS] = Field("sheet", description="sheet (six labelled views) or one of front, back, left, right, top, bottom, iso, iso_front_left, iso_back_left, iso_back_right.")
    size: Optional[int] = Field(None, ge=1, le=64, description="Pixels per voxel (flat views) or face size (isometric).")


class OpsArgs(ModelId):
    ops: list[dict[str, Any]] = Field(..., description="Operations, applied in order and all-or-nothing: " + ", ".join(sorted(OPS))
                                      + ". See the app instructions or generate's prompt for the fields.")
    note: str = Field("", max_length=200, description="What this change is (kept with the previous version).")


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
    path: str = Field(..., min_length=1, max_length=500, description="A MagicaVoxel .vox file on this PC.")
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
    mode: Literal["inflate", "extrude", "hull"] = "inflate"
    side_id: str = Field("", description="hull: the side view (the model seen from its left).")
    top_id: str = Field("", description="hull: optional top view.")
    depth: int = Field(0, ge=0, le=256, description="inflate/extrude thickness; 0 = automatic.")
    max_side: int = Field(32, ge=4, le=128)
    colors: int = Field(16, ge=2, le=48)
    name: str = Field("", max_length=120)


class RefRole(BaseModel):
    id: str
    role: Literal["front", "side", "top", "back", "extra"] = "front"


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
    size: Optional[list[int]] = Field(None, min_length=3, max_length=3, description="Grid [x, y, z] when there are no pictures; default from settings (new_size).")
    effort: Optional[Literal["off", "low", "medium", "high", "max"]] = None
    use_model: bool = Field(True, description="false = only the starting model (no language model).")
    auto_paint: Optional[bool] = Field(None, description="Paint the reference pictures onto the sculpted shape after every round; default from settings.")


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
    data = png_bytes(render(m, a.view, size=a.size, background="dark" if a.view != "sheet" else "transparent"))
    return {"id": m.id, "name": m.name, "view": a.view, "count": m.count, "size": m.size, "_image": _img(data)}


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
    m = from_vox(path.read_bytes(), svc.store.new_id(), a.name or path.stem)
    m.source = {"kind": "imported", "file": path.name}
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
    Tool("model_import_vox", _d("Import a MagicaVoxel .vox file from this PC as a new model.", synonyms="importar vox"),
         ImportArgs, ann(draft_safe=True), _import),
    Tool("reference_add", _d("Add a reference picture (path on this PC or base64) to turn into voxels; returns its id.",
                             synonyms="subir imagen, sprite, referencia"),
         RefAddArgs, ann(draft_safe=True), _ref_add),
    Tool("reference_list", _d("List the reference pictures."), Empty, ann(read_only=True), lambda s, a: {"references": s.store.references()}),
    Tool("reference_look", _d("See a reference picture."), RefId, ann(read_only=True), _ref_look, capped=False),
    Tool("reference_grid", _d("A reference read as a small indexed grid (rows of characters, palette): what the builders and the model use."),
         RefGridArgs, ann(read_only=True), _ref_grid),
    Tool("reference_delete", _d("Delete a reference picture (confirm=true)."), RefDeleteArgs, ann(destructive=True), _ref_delete),
    Tool("build_from_image", _d("Instant voxel model from a picture without the language model: inflate (puffed-up front), "
                                "extrude (slab) or hull (front + side silhouettes carved).", synonyms="convertir a voxel rápido"),
         BuildArgs, ann(draft_safe=True), _build, timeout_s=120),
    Tool("generate_start", _d("Ask the language model (DeepSeek on the Sparks by default) for a voxel version of one or more "
                              "pictures, or of a description alone (objects, characters, whole scenes). It sees the pictures, sculpts the shape with operations (ellipsoids, capsules, wing prisms), Seurat paints the "
                              "pictures onto it, and it reviews its own renders. With effort max a round takes several minutes. Returns a job id.",
                              "Roles: front (main), side (seen from the model's left), top, back, extra.",
                              "generar voxel, deepseek, versión voxel de una imagen"),
         GenerateArgs, ann(draft_safe=True), _generate, timeout_s=60),
    Tool("generate_status", _d("Progress of a generation: log, rounds (with notes and errors) and, when done, the model id."),
         JobId, ann(read_only=True), lambda s, a: s.generator.get(a.job_id).to_dict()),
    Tool("generate_list", _d("Recent generations."), Empty, ann(read_only=True), lambda s, a: {"jobs": s.generator.list()}),
    Tool("generate_cancel", _d("Stop a generation (what it built so far is saved)."), JobId, ann(),
         lambda s, a: s.generator.cancel(a.job_id).to_dict()),
    Tool("settings_get", _d("Settings: reading size and colours, review rounds, reasoning effort and thinking tokens, auto paint, model URL, export scales."),
         Empty, ann(read_only=True), lambda s, a: s.settings.get()),
    Tool("settings_set", _d("Change settings."), SettingsArgs, ann(idempotent=True), _settings_set),
]

TOOLS_BY_NAME = {t.name: t for t in TOOLS}


def tool_catalog() -> list[dict[str, Any]]:
    return _catalog(TOOLS)


def call_tool(svc: Services, name: str, arguments: Optional[dict[str, Any]], caller: str = "agent") -> Any:
    return _call_tool(TOOLS_BY_NAME, svc, name, arguments)
