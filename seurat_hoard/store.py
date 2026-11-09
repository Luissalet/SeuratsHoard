"""Where models and reference pictures live: ``<data>/models/<id>/model.json`` (plus the last versions) and
``<data>/references/<id>.<ext>`` with a JSON index. Writes are atomic; one lock per store."""

from __future__ import annotations

import json
import re
import threading
import time
from pathlib import Path
from typing import Any, Optional

from .errors import SeuratError
from .hoard_link.atomic import read_json, write_json_atomic
from .hoard_link.ids import new_id
from .images import load
from .voxels import VoxelModel

KEEP_VERSIONS = 40
EDITOR_RUN_S = 300           # editor autosaves closer than this share one version
ID_RE = re.compile(r"^[A-Za-z0-9_-]{1,64}$")
REF_EXT = {"PNG": "png", "JPEG": "jpg", "GIF": "gif", "WEBP": "webp", "BMP": "bmp"}


def _check_id(value: str, what: str) -> str:
    if not ID_RE.match(str(value or "")):
        raise SeuratError("not_found", f"No {what} {value!r}.")
    return value


class Store:
    def __init__(self, data_dir: Path):
        self.root = Path(data_dir)
        self.models_dir = self.root / "models"
        self.refs_dir = self.root / "references"
        self.models_dir.mkdir(parents=True, exist_ok=True)
        self.refs_dir.mkdir(parents=True, exist_ok=True)
        self._lock = threading.RLock()

    # ------------------------------------------------------------------------------------------------ models
    def _mdir(self, model_id: str) -> Path:
        return self.models_dir / _check_id(model_id, "model")

    def list_models(self) -> list[dict[str, Any]]:
        out = []
        for d in self.models_dir.iterdir():
            meta = read_json(d / "model.json", default=None) if d.is_dir() else None
            if not isinstance(meta, dict):
                continue
            vox = meta.get("voxels") or []
            out.append({"id": meta.get("id"), "name": meta.get("name"), "size": meta.get("size"), "count": len(vox),
                        "updated": meta.get("updated"), "created": meta.get("created"),
                        "versions": len(list((d / "versions").glob("*.json"))) if (d / "versions").exists() else 0,
                        "source": {k: v for k, v in (meta.get("source") or {}).items() if k in ("kind", "mode", "references")}})
        return sorted(out, key=lambda m: -(m.get("updated") or 0))

    def get(self, model_id: str) -> VoxelModel:
        data = read_json(self._mdir(model_id) / "model.json", default=None)
        if not isinstance(data, dict):
            raise SeuratError("not_found", f"No model {model_id!r}.", "model_list shows the ids.")
        return VoxelModel.from_dict(data)

    def exists(self, model_id: str) -> bool:
        try:
            return (self._mdir(model_id) / "model.json").exists()
        except SeuratError:
            return False

    def new_id(self) -> str:
        return new_id("m")

    def save(self, model: VoxelModel, *, version_note: str = "") -> VoxelModel:
        """Write ``model``; the previous content becomes a version (the last ``KEEP_VERSIONS`` are kept)."""
        with self._lock:
            d = self._mdir(model.id)
            d.mkdir(parents=True, exist_ok=True)
            current = d / "model.json"
            if current.exists():
                vdir = d / "versions"
                vdir.mkdir(exist_ok=True)
                prev = read_json(current, default=None)
                # The editor autosaves every few seconds: a run of its saves keeps one version (the state before the
                # run), so restore points from before a generation or a restore are not pushed out by autosaves.
                latest = max(vdir.glob("*.json"), default=None)
                if (version_note == "editor" and isinstance(prev, dict) and prev.get("version_note") == "editor"
                        and latest is not None and time.time() - latest.stat().st_mtime < EDITOR_RUN_S):
                    prev = None
                if isinstance(prev, dict):
                    prev["version_note"] = prev.get("version_note") or version_note or ""
                    write_json_atomic(vdir / f"{int(time.time() * 1000)}.json", prev)
                for old in sorted(vdir.glob("*.json"))[:-KEEP_VERSIONS]:
                    old.unlink(missing_ok=True)
            model.updated = time.time()
            data = model.to_dict()
            data["version_note"] = version_note
            write_json_atomic(current, data)
        return model

    def versions(self, model_id: str) -> list[dict[str, Any]]:
        vdir = self._mdir(model_id) / "versions"
        out = []
        for f in sorted(vdir.glob("*.json"), reverse=True) if vdir.exists() else []:
            meta = read_json(f, default={}) or {}
            out.append({"version": f.stem, "updated": meta.get("updated"), "count": len(meta.get("voxels") or []),
                        "note": meta.get("version_note") or ""})
        return out

    def restore(self, model_id: str, version: str) -> VoxelModel:
        f = self._mdir(model_id) / "versions" / f"{_check_id(version, 'version')}.json"
        data = read_json(f, default=None)
        if not isinstance(data, dict):
            raise SeuratError("not_found", f"No version {version!r} of {model_id!r}.")
        model = VoxelModel.from_dict(data)
        return self.save(model, version_note="before restoring " + version)

    def delete(self, model_id: str) -> dict[str, Any]:
        import shutil
        d = self._mdir(model_id)
        if not d.exists():
            raise SeuratError("not_found", f"No model {model_id!r}.")
        trash = self.root / "trash"
        trash.mkdir(exist_ok=True)
        target = trash / f"{model_id}-{int(time.time())}"
        shutil.move(str(d), str(target))
        return {"deleted": model_id, "kept_in": str(target)}

    # ------------------------------------------------------------------------------------------------ references
    def _index(self) -> dict[str, Any]:
        data = read_json(self.refs_dir / "index.json", default={}) or {}
        return data if isinstance(data, dict) else {}

    def add_reference(self, data: bytes, name: str = "") -> dict[str, Any]:
        img = load(data)
        from PIL import Image
        import io
        fmt = (Image.open(io.BytesIO(data)).format or "PNG").upper()
        ext = REF_EXT.get(fmt, "png")
        rid = new_id("r")
        with self._lock:
            (self.refs_dir / f"{rid}.{ext}").write_bytes(data)
            index = self._index()
            meta = {"id": rid, "name": (name or f"referencia {len(index) + 1}")[:120], "file": f"{rid}.{ext}",
                    "width": img.width, "height": img.height, "bytes": len(data), "added": time.time()}
            index[rid] = meta
            write_json_atomic(self.refs_dir / "index.json", index)
        return meta

    def references(self) -> list[dict[str, Any]]:
        return sorted(self._index().values(), key=lambda r: -(r.get("added") or 0))

    def reference(self, ref_id: str) -> dict[str, Any]:
        meta = self._index().get(_check_id(ref_id, "reference"))
        if not meta:
            raise SeuratError("not_found", f"No reference {ref_id!r}.", "reference_list shows the ids.")
        return meta

    def reference_bytes(self, ref_id: str) -> bytes:
        meta = self.reference(ref_id)
        return (self.refs_dir / meta["file"]).read_bytes()

    def reference_path(self, ref_id: str) -> Path:
        return self.refs_dir / self.reference(ref_id)["file"]

    def rename_reference(self, ref_id: str, name: str) -> dict[str, Any]:
        with self._lock:
            index = self._index()
            meta = index.get(_check_id(ref_id, "reference"))
            if not meta:
                raise SeuratError("not_found", f"No reference {ref_id!r}.")
            meta["name"] = name[:120]
            write_json_atomic(self.refs_dir / "index.json", index)
        return meta

    def delete_reference(self, ref_id: str) -> dict[str, Any]:
        with self._lock:
            index = self._index()
            meta = index.pop(_check_id(ref_id, "reference"), None)
            if not meta:
                raise SeuratError("not_found", f"No reference {ref_id!r}.")
            (self.refs_dir / meta["file"]).unlink(missing_ok=True)
            write_json_atomic(self.refs_dir / "index.json", index)
        return {"deleted": ref_id}


def dumps_compact(obj: Any) -> str:
    return json.dumps(obj, ensure_ascii=False, separators=(",", ":"))
