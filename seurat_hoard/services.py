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

    def run_program(self, code: str, *, model_id: str = "", name: str = "", size: Optional[Iterable[int]] = None,
                    sprites: Optional[list[dict[str, str]]] = None) -> dict[str, Any]:
        """Run a voxel program (PALETTE + build(g), see program.py) into a model: an existing one (its grid and palette
        are replaced, the previous content stays as a version) or a new one. ``sprites`` ([{id, role}] with role front,
        back or side) gives the program the sprite tools (SPRITE_PALETTE, SIDE, carve_side, paint_view...); a new model
        then gets a grid of one voxel per sprite pixel."""
        from . import spritecam
        from .program import ProgramError, run_program
        refs = [spritecam.sprite_ref(r["role"], self.store.reference(r["id"]).get("name") or r["role"], self.store.reference_bytes(r["id"]))
                for r in (sprites or [])]
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
