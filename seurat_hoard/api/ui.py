"""Routes for the bundled editor: models (full voxel data in and out), operations, renders, exports, reference
pictures, the instant builders, generations and settings. Whatever the buttons do is also an agent tool
(``POST /api/ui/call`` runs those handlers)."""

from __future__ import annotations

from typing import Any, Optional
from urllib.parse import quote

from fastapi import APIRouter, Request
from fastapi.responses import FileResponse, Response
from pydantic import BaseModel, Field

from ..agent_tools import TOOLS_BY_NAME, export_model
from ..errors import SeuratError
from ..exporters import FORMATS, from_vox
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
def render_png(request: Request, model_id: str, view: str = "iso", size: Optional[int] = None, bg: str = "transparent"):
    m = services(request).store.get(model_id)
    if m.count == 0:
        raise SeuratError("invalid", "The model is empty.")
    return Response(png_bytes(render(m, view, size=size, background=bg)), media_type="image/png", headers={"Cache-Control": "no-cache"})


@router.get("/api/models/{model_id}/export/{fmt}")
def export_file(request: Request, model_id: str, fmt: str, view: str = "iso", size: Optional[int] = None, download: bool = True):
    if fmt not in FORMATS:
        raise SeuratError("invalid", f"Unknown format {fmt!r}; use one of {', '.join(FORMATS)}.")
    data, filename = export_model(services(request), model_id, fmt, view, size)
    return _file(data, filename, FORMATS[fmt], download)


@router.post("/api/models/import-vox")
async def import_vox(request: Request, name: str = "modelo importado"):
    svc = services(request)
    data = await request.body()
    m = from_vox(data, svc.store.new_id(), name[:120])
    m.source = {"kind": "imported"}
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
