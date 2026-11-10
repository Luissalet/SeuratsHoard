"""Routes for the bundled editor: models (full voxel data in and out), operations, renders, exports, reference
pictures, the instant builders, generations and settings. Whatever the buttons do is also an agent tool
(``POST /api/ui/call`` runs those handlers)."""

from __future__ import annotations

from typing import Any, Optional
from urllib.parse import quote

from fastapi import APIRouter, Request
from fastapi.responses import FileResponse, Response
from pydantic import BaseModel, Field

from .. import lit, sheets
from ..agent_tools import TOOLS_BY_NAME, export_model, round_model, spin_file
from ..errors import SeuratError
from ..exporters import FORMATS, from_vox, import_model
from ..images import index_picture, thumbnail_png
from ..render import png_bytes, render
from .deps import services, tool

router = APIRouter()


class CallBody(BaseModel):
    name: str = Field(..., min_length=1, max_length=100)
    arguments: Optional[dict[str, Any]] = None


class CreateBody(BaseModel):
    name: str = ""
    size: Optional[list[int]] = None


class ReplaceBody(BaseModel):
    name: Optional[str] = None
    size: Optional[list[int]] = None
    palette: Optional[list[str]] = None
    voxels: Optional[list[list[int]]] = None
    notes: Optional[str] = None
    version_note: str = ""


class OpsBody(BaseModel):
    ops: list[dict[str, Any]]
    note: str = ""


class RestoreBody(BaseModel):
    version: str


class RenameBody(BaseModel):
    name: str = Field(..., min_length=1, max_length=120)


def _file(data: bytes, filename: str, mime: str, download: bool) -> Response:
    disposition = "attachment" if download else "inline"
    return Response(data, media_type=mime, headers={"Content-Disposition": f"{disposition}; filename*=UTF-8''{quote(filename)}",
                                                     "Cache-Control": "no-cache", "X-Content-Type-Options": "nosniff"})


@router.get("/api/status")
def status(request: Request):
    return services(request).status()


# ---------------------------------------------------------------------------------------------------------- models
@router.get("/api/models")
def models(request: Request):
    return {"models": services(request).store.list_models()}


@router.post("/api/models")
def create(request: Request, body: CreateBody):
    return services(request).create_model(body.name, body.size).to_dict()


@router.get("/api/models/{model_id}")
def get_model(request: Request, model_id: str):
    return services(request).store.get(model_id).to_dict()


@router.put("/api/models/{model_id}")
def replace(request: Request, model_id: str, body: ReplaceBody):
    data = body.model_dump(exclude_none=True)
    return services(request).replace(model_id, data).to_dict(voxels=False)


@router.post("/api/models/{model_id}/ops")
def ops(request: Request, model_id: str, body: OpsBody):
    return services(request).apply(model_id, body.ops, note=body.note)


@router.post("/api/models/{model_id}/duplicate")
def duplicate(request: Request, model_id: str):
    return services(request).duplicate(model_id).to_dict(voxels=False)


@router.post("/api/models/{model_id}/rename")
def rename(request: Request, model_id: str, body: RenameBody):
    return tool(request, "model_rename", {"model_id": model_id, "name": body.name})


@router.delete("/api/models/{model_id}")
def delete(request: Request, model_id: str):
    return services(request).store.delete(model_id)


@router.get("/api/models/{model_id}/versions")
def versions(request: Request, model_id: str):
    return {"versions": services(request).store.versions(model_id)}


@router.post("/api/models/{model_id}/restore")
def restore(request: Request, model_id: str, body: RestoreBody):
    return services(request).store.restore(model_id, body.version).to_dict()


@router.get("/api/models/{model_id}/render.png")
def render_png(request: Request, model_id: str, view: str = "iso", size: Optional[int] = None, bg: str = "transparent",
               yaw: Optional[float] = None, pitch: Optional[float] = None):
    m = services(request).store.get(model_id)
    if m.count == 0:
        raise SeuratError("invalid", "The model is empty.")
    size = max(1, min(int(size), 2048)) if size else None
    return Response(png_bytes(render(m, view, size=size, background=bg, yaw=yaw, pitch=pitch)), media_type="image/png",
                    headers={"Cache-Control": "no-cache"})


@router.get("/api/models/{model_id}/spin/{fmt}")
def spin(request: Request, model_id: str, fmt: str, frames: int = 36, seconds: float = 3.0, pitch: float = 25.0, size: int = 384,
         ground: bool = True, download: bool = True):
    if fmt not in lit.SPIN_FORMATS:
        raise SeuratError("invalid", f"Unknown animation format {fmt!r}; use gif or webm.")
    data, filename = spin_file(services(request), model_id, fmt, max(8, min(frames, 120)), max(0.5, min(seconds, 20.0)),
                               max(-10.0, min(pitch, 80.0)), max(96, min(size, 1024)), ground)
    return _file(data, filename, lit.SPIN_FORMATS[fmt], download)


@router.post("/api/models/{model_id}/calibrate")
def calibrate(request: Request, model_id: str, body: dict[str, Any]):
    res = tool(request, "model_sprite_calibrate", {**body, "model_id": model_id})
    img = res.pop("_image", None)
    if img:
        res["image"] = f"data:{img['mime']};base64,{img['data']}"
    return res


@router.get("/api/models/{model_id}/export/{fmt}")
def export_file(request: Request, model_id: str, fmt: str, view: str = "iso", size: Optional[int] = None, download: bool = True):
    if fmt not in FORMATS:
        raise SeuratError("invalid", f"Unknown format {fmt!r}; use one of {', '.join(FORMATS)}.")
    data, filename = export_model(services(request), model_id, fmt, view, size)
    return _file(data, filename, FORMATS[fmt], download)


@router.post("/api/models/import")
@router.post("/api/models/import-vox")
async def import_vox(request: Request, name: str = "modelo importado"):
    svc = services(request)
    data = await request.body()
    m = import_model(data, svc.store.new_id(), name[:120])
    svc.store.save(m)
    return m.to_dict(voxels=False)


# ---------------------------------------------------------------------------------------------------------- references
@router.get("/api/references")
def references(request: Request):
    return {"references": services(request).store.references()}


@router.post("/api/references")
async def add_reference(request: Request, name: str = ""):
    data = await request.body()
    return services(request).store.add_reference(data, name)


@router.get("/references/{ref_id}")
def reference_file(request: Request, ref_id: str):
    store = services(request).store
    meta = store.reference(ref_id)
    return FileResponse(store.reference_path(ref_id), headers={"Cache-Control": "no-cache", "X-Content-Type-Options": "nosniff"},
                        filename=meta["file"], content_disposition_type="inline")


@router.get("/references/{ref_id}/thumb.png")
def reference_thumb(request: Request, ref_id: str, side: int = 256):
    return Response(thumbnail_png(services(request).store.reference_bytes(ref_id), max(32, min(side, 1024))), media_type="image/png")


@router.get("/api/references/{ref_id}/grid")
def reference_grid(request: Request, ref_id: str, max_side: int = 32, colors: int = 16):
    ix = index_picture(services(request).store.reference_bytes(ref_id), max_side=max(4, min(max_side, 128)), colors=max(2, min(colors, 48)))
    return {"reference_id": ref_id, **ix.to_dict()}


@router.get("/api/references/{ref_id}/indexed.png")
def reference_indexed(request: Request, ref_id: str, max_side: int = 32, colors: int = 16):
    ix = index_picture(services(request).store.reference_bytes(ref_id), max_side=max(4, min(max_side, 128)), colors=max(2, min(colors, 48)))
    return Response(ix.preview_png(), media_type="image/png", headers={"Cache-Control": "no-cache"})


@router.get("/api/references/{ref_id}/frames")
def reference_frames(request: Request, ref_id: str, cols: int = 0, rows: int = 0):
    res = tool(request, "reference_frames", {"reference_id": ref_id, "cols": max(0, min(cols, 64)), "rows": max(0, min(rows, 64))})
    res.pop("_image", None)
    return res


@router.get("/api/references/{ref_id}/frames.png")
def reference_frames_png(request: Request, ref_id: str, cols: int = 0, rows: int = 0, labels: bool = True):
    sh = services(request).reference_sheet(ref_id, max(0, min(cols, 64)), max(0, min(rows, 64)))
    img = sheets.contact(sh, sheets.guess_views(sh) if labels else None)
    return Response(sheets.png_bytes(img), media_type="image/png", headers={"Cache-Control": "no-cache"})


@router.post("/api/references/{ref_id}/rename")
def rename_reference(request: Request, ref_id: str, body: RenameBody):
    return services(request).store.rename_reference(ref_id, body.name)


@router.delete("/api/references/{ref_id}")
def delete_reference(request: Request, ref_id: str):
    return services(request).store.delete_reference(ref_id)


# ---------------------------------------------------------------------------------------------------------- builders and generation
@router.post("/api/build")
def build(request: Request, body: dict[str, Any]):
    return tool(request, "build_from_image", body)


@router.post("/api/generate")
def generate(request: Request, body: dict[str, Any]):
    return tool(request, "generate_start", body)


@router.get("/api/jobs")
def jobs(request: Request):
    return {"jobs": services(request).generator.list()}


@router.get("/api/jobs/{job_id}")
def job(request: Request, job_id: str):
    return services(request).generator.get(job_id).to_dict()


@router.post("/api/jobs/{job_id}/cancel")
def cancel(request: Request, job_id: str):
    return services(request).generator.cancel(job_id).to_dict()


@router.get("/api/jobs/{job_id}/rounds/{rnd}/model")
def round_voxels(request: Request, job_id: str, rnd: int):
    """A round of a generation with its voxels, for the editor (not stored as a model)."""
    model, out = round_model(services(request), job_id, rnd)
    return {**model.to_dict(), **out, "id": "", "preview": True}


@router.post("/api/jobs/{job_id}/rounds/{rnd}/save")
def round_save(request: Request, job_id: str, rnd: int, body: Optional[dict[str, Any]] = None):
    return tool(request, "generate_round", {"job_id": job_id, "round": rnd, "save": True, "name": str((body or {}).get("name") or "")[:120]})


@router.get("/renders/{name}")
def render_file(request: Request, name: str):
    svc = services(request)
    if "/" in name or "\\" in name or ".." in name or not name.endswith(".png"):
        raise SeuratError("not_found", "No such render.")
    path = svc.config.renders_dir / name
    if not path.is_file():
        raise SeuratError("not_found", "No such render.")
    return FileResponse(path, media_type="image/png")


# ---------------------------------------------------------------------------------------------------------- settings and the shared call
@router.get("/api/settings")
def settings(request: Request):
    return services(request).settings.get()


@router.post("/api/settings")
def settings_set(request: Request, body: dict[str, Any]):
    return tool(request, "settings_set", body)


@router.post("/api/ui/call")
def ui_call(request: Request, body: CallBody):
    if body.name not in TOOLS_BY_NAME:
        raise SeuratError("not_found", f"Unknown tool {body.name}.")
    result = tool(request, body.name, body.arguments or {})
    if isinstance(result, dict):
        result.pop("_image", None)
    return result
