"""The voxels of every round of a generation, kept next to its renders (``<data>/renders/<job>-r<round>.npz``) so any
intermediate round can be opened in the 3-D editor or saved as a model of its own, not only the result."""

from __future__ import annotations

import re
from pathlib import Path

import numpy as np

from .errors import SeuratError
from .voxels import VoxelModel

ID_RE = re.compile(r"^[A-Za-z0-9_-]{1,64}$")


def snapshot_name(job_id: str, rnd: int) -> str:
    return f"{job_id}-r{int(rnd)}.npz"


def save_snapshot(renders_dir: Path, job_id: str, rnd: int, model: VoxelModel) -> str:
    """Write the round's grid and palette; returns the file name ('' when it could not be written)."""
    name = snapshot_name(job_id, rnd)
    try:
        renders_dir.mkdir(parents=True, exist_ok=True)
        with open(renders_dir / name, "wb") as fh:
            np.savez_compressed(fh, grid=model.grid.astype(np.uint8), palette=np.array(model.palette, dtype="U7"))
    except OSError:
        return ""
    return name


def load_snapshot(renders_dir: Path, job_id: str, rnd: int, name: str = "") -> VoxelModel:
    """A round as a model (id empty: it is not stored as a model until it is saved)."""
    if not ID_RE.match(str(job_id or "")):
        raise SeuratError("not_found", f"No generation {job_id!r}.")
    path = renders_dir / snapshot_name(job_id, rnd)
    if not path.is_file():
        raise SeuratError("not_found", f"Round {rnd} of {job_id} has no saved voxels.",
                          "Only rounds that built something are kept (and only since this version of Seurat).")
    with np.load(path, allow_pickle=False) as data:
        grid = data["grid"].astype(np.uint8)
        palette = [str(c) for c in data["palette"].tolist()]
    model = VoxelModel.empty("", name or f"{job_id} ronda {rnd}", list(grid.shape), palette or ["#ffffff"])
    model.grid = grid
    return model


def list_snapshots(renders_dir: Path, job_id: str) -> list[int]:
    if not ID_RE.match(str(job_id or "")):
        return []
    out = []
    for f in renders_dir.glob(f"{job_id}-r*.npz"):
        tail = f.stem[len(job_id) + 2:]
        if tail.isdigit():
            out.append(int(tail))
    return sorted(out)
