"""Scenes by parts: a big scene (up to 256 voxels a side) planned as objects, each object built by its own voxel program
in its own grid (up to 128 a side), then placed - turned, moved, set down on what is under it - into one model.

The plan is JSON from the language model (``generate`` asks for it): the scene's size and a list of objects, each with a
description, its own grid size and one or more placements (the same bench twice, turned). ``compose`` does the placing
and merges the objects' palettes into one.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Optional

import numpy as np

from .errors import SeuratError
from .voxels import MAX_SIDE, norm_hex

MAX_PART = 128           # an object's own grid, a side (the ground may span the whole scene)
MAX_OBJECTS = 16
ROTATIONS = (0, 90, 180, 270)


@dataclass
class Placement:
    position: list[int]                 # [x, y, z] of the turned object's lowest corner in the scene
    rotation: int = 0                   # degrees around the vertical; 90 = the object's front (+z) then faces +x
    settle: bool = True                 # drop it onto whatever is under it (the ground, a table)

    def to_dict(self) -> dict[str, Any]:
        return {"position": list(self.position), "rotation": self.rotation, "settle": self.settle}


@dataclass
class Part:
    name: str
    description: str
    size: list[int]
    placements: list[Placement]
    ground: bool = False                # the terrain under everything: built first, placed first, never settled
    grid: Optional[np.ndarray] = None
    palette: list[str] = field(default_factory=list)
    program: str = ""
    notes: str = ""

    def to_dict(self, program: bool = True) -> dict[str, Any]:
        out = {"name": self.name, "description": self.description, "size": list(self.size), "ground": self.ground,
               "placements": [p.to_dict() for p in self.placements], "count": int(np.count_nonzero(self.grid)) if self.grid is not None else 0}
        if program and self.program:
            out["program"] = self.program
        return out


def _int3(v: Any, what: str) -> list[int]:
    if not isinstance(v, (list, tuple)) or len(v) != 3:
        raise SeuratError("invalid", f"{what} must be [x, y, z].")
    try:
        return [int(round(float(a))) for a in v]
    except (TypeError, ValueError):
        raise SeuratError("invalid", f"{what} must be numbers.") from None


def parse_plan(data: dict[str, Any], scene_size: list[int], max_objects: int = MAX_OBJECTS) -> list[Part]:
    """The language model's plan as parts, cleaned: sizes inside the limits, positions inside the scene, rotations
    snapped to quarter turns. A ground object comes first."""
    X, Y, Z = scene_size
    objs = data.get("objects") if isinstance(data, dict) else None
    if not isinstance(objs, list) or not objs:
        raise SeuratError("failed", "The plan has no objects.")
    parts: list[Part] = []
    for i, o in enumerate(objs[:max_objects]):
        if not isinstance(o, dict):
            continue
        name = str(o.get("name") or f"object {i + 1}").strip()[:60]
        ground = bool(o.get("ground"))
        size = _int3(o.get("size") or [16, 16, 16], f"{name}: size")
        lim = [X, min(Y, MAX_PART), Z] if ground else [MAX_PART] * 3
        size = [max(2, min(v, lim[k], scene_size[k])) for k, v in enumerate(size)]
        raw = o.get("placements")
        if not isinstance(raw, list) or not raw:
            raw = [{"position": o.get("position") or [0, 0, 0], "rotation": o.get("rotation") or 0}]
        places = []
        for pl in raw[:12]:
            if not isinstance(pl, dict):
                continue
            pos = _int3(pl.get("position") or [0, 0, 0], f"{name}: position")
            rot = int(round(float(pl.get("rotation") or 0) / 90.0)) * 90 % 360
            places.append(Placement(pos, rot, bool(pl.get("settle", not ground)) and not ground))
        if not places:
            continue
        parts.append(Part(name, str(o.get("description") or name).strip()[:1200], size, places, ground))
    if not parts:
        raise SeuratError("failed", "The plan has no usable objects.")
    parts.sort(key=lambda p: not p.ground)
    return parts


def turn(grid: np.ndarray, rotation: int) -> np.ndarray:
    """The grid turned around the vertical by ``rotation`` degrees (90: what faced +z faces +x)."""
    k = (int(rotation) // 90) % 4
    out = grid
    for _ in range(k):
        out = out.transpose(2, 1, 0)[:, :, ::-1]
    return np.ascontiguousarray(out)


def merge_palettes(palettes: list[list[str]], limit: int = 255) -> tuple[list[str], list[np.ndarray]]:
    """One palette for all the parts and, per part, a lookup from its indices to the merged ones (equal colours shared;
    past ``limit`` colours the nearest existing one is used)."""
    merged: list[str] = []
    index: dict[str, int] = {}
    luts = []
    for pal in palettes:
        lut = np.zeros(256, np.uint8)
        for i, c in enumerate(pal[:255]):
            try:
                h = norm_hex(c)
            except Exception:  # noqa: BLE001
                h = "#ff00ff"
            if h not in index:
                if len(merged) < limit:
                    merged.append(h)
                    index[h] = len(merged)
                else:
                    rgb = np.array([int(h[j:j + 2], 16) for j in (1, 3, 5)])
                    allc = np.array([[int(m[j:j + 2], 16) for j in (1, 3, 5)] for m in merged])
                    index[h] = int(((allc - rgb) ** 2).sum(axis=1).argmin()) + 1
            lut[i + 1] = index[h]
        luts.append(lut)
    return merged, luts


def _settle_y(scene: np.ndarray, obj: np.ndarray, x0: int, z0: int, y_plan: int) -> int:
    """The height at which the object rests on what is already in the scene under its feet (its lowest layer)."""
    occ = obj > 0
    if not occ.any():
        return y_plan
    ys = np.nonzero(occ.any(axis=(0, 2)))[0]
    low = int(ys.min())
    feet = occ[:, low: low + 2, :].any(axis=1)                       # (x, z) columns the object stands on
    X, Y, Z = scene.shape
    top = -1
    for (fx, fz) in np.argwhere(feet):
        sx, sz = x0 + int(fx), z0 + int(fz)
        if 0 <= sx < X and 0 <= sz < Z:
            col = np.nonzero(scene[sx, :, sz])[0]
            if len(col):
                top = max(top, int(col.max()))
    return top + 1 - low


def compose(scene_size: list[int], parts: list[Part]) -> tuple[np.ndarray, list[str], list[dict[str, Any]]]:
    """Place every part (turned, moved, settled) into one grid. Returns the grid, the merged palette and where each
    placement ended (clipped parts are reported)."""
    X, Y, Z = [max(1, min(MAX_SIDE, int(v))) for v in scene_size]
    built = sorted([p for p in parts if p.grid is not None and np.count_nonzero(p.grid)], key=lambda q: not q.ground)
    palette, luts = merge_palettes([p.palette for p in built])
    lut_of = {id(p): lut for p, lut in zip(built, luts)}
    scene = np.zeros((X, Y, Z), np.uint8)
    report = []
    for p in sorted(built, key=lambda q: not q.ground):
        lut = lut_of[id(p)]
        for pl in p.placements:
            obj = lut[turn(p.grid, pl.rotation)]
            x0, y0, z0 = pl.position
            if pl.settle:
                y0 = max(0, _settle_y(scene, obj, x0, z0, y0))
            sx, sy, sz = obj.shape
            a = [max(0, x0), max(0, y0), max(0, z0)]
            b = [min(X, x0 + sx), min(Y, y0 + sy), min(Z, z0 + sz)]
            entry = {"name": p.name, "position": [x0, y0, z0], "rotation": pl.rotation, "size": [sx, sy, sz]}
            if any(b[k] <= a[k] for k in range(3)):
                entry["outside"] = True
                report.append(entry)
                continue
            src = obj[a[0] - x0:b[0] - x0, a[1] - y0:b[1] - y0, a[2] - z0:b[2] - z0]
            dst = scene[a[0]:b[0], a[1]:b[1], a[2]:b[2]]
            put = src > 0
            dst[put] = src[put]
            lost = int(np.count_nonzero(obj)) - int(put.sum())
            if lost > 0:
                entry["clipped"] = lost
            report.append(entry)
    return scene, palette, report


def layout_boxes(scene_size: list[int], parts: list[Part]) -> tuple[np.ndarray, list[str]]:
    """The plan as coloured boxes (one colour per object, hollow so overlaps show): a preview before anything is built."""
    X, Y, Z = [max(1, min(MAX_SIDE, int(v))) for v in scene_size]
    colours = ["#8f8a7e", "#d9a441", "#4f8fd1", "#c8553d", "#6aa84f", "#9b6fd1", "#e07bb0", "#4fb8b0", "#c9c26b",
               "#7d5a3c", "#5d6d7e", "#e8e4ea", "#a3c4f3", "#f4a261", "#2a9d8f", "#264653"]
    g = np.zeros((X, Y, Z), np.uint8)
    for i, p in enumerate(parts):
        c = (i % len(colours)) + 1
        for pl in p.placements:
            sx, sy, sz = p.size if pl.rotation in (0, 180) else [p.size[2], p.size[1], p.size[0]]
            if p.ground:
                sy = min(sy, 2)
            x0, y0, z0 = pl.position
            if pl.settle:
                y0 = max(0, min(y0, Y - 1))
            a = [max(0, x0), max(0, y0), max(0, z0)]
            b = [min(X, x0 + sx), min(Y, y0 + sy), min(Z, z0 + sz)]
            if any(b[k] <= a[k] for k in range(3)):
                continue
            box = np.ones((b[0] - a[0], b[1] - a[1], b[2] - a[2]), bool)
            if min(box.shape) > 2 and not p.ground:
                box[1:-1, 1:-1, 1:-1] = False
            sub = g[a[0]:b[0], a[1]:b[1], a[2]:b[2]]
            sub[box] = c
    return g, colours
