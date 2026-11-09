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
