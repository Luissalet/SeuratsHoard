"""The voxel model and the operations that build it.

A model is a grid ``size = [sx, sy, sz]`` of palette indices (0 = empty, 1..255 = ``palette[i - 1]``). Axes: x to the
right, y up, z towards the viewer, so the *front* view looks along -z and a sprite drawn on the front plane reads with
its first row at the top (largest y).

Every change, from the editor, an assistant or the model, is a list of operations (``apply_ops``). They are small JSON
objects so a language model can write them; each one is checked and an error names the operation and the problem.

========  =======================================================================================================
op        fields
========  =======================================================================================================
voxel     ``at [x,y,z]``, ``c``
voxels    ``list [[x,y,z,c], ...]``
box       ``from [x,y,z]``, ``to [x,y,z]`` (inclusive), ``c``, ``hollow`` (bool, optional)
line      ``from``, ``to``, ``c``; with ``r`` (and ``r_to``) a capsule (necks, tails, limbs)
prism     ``plane`` front|side|top, ``points`` [[u,v],...] (a polygon on that plane), ``from``/``to`` along the third axis,
          ``c``: flat shapes with thickness (wings, fins, ears, blades)
sphere    ``center [x,y,z]`` (may be .5), ``r`` or ``radii [rx,ry,rz]``, ``c``, ``hollow`` (optional)
cylinder  ``axis`` x|y|z, ``center`` [the two other coordinates], ``from``, ``to`` (along the axis), ``r``, ``c``
layer     ``plane`` front|side|top, ``rows`` [strings], ``at`` (or ``from``/``to`` for a slab), ``origin [u,v]``,
          ``key`` {char: c} (optional; default ``1-9a-zA-Z`` = 1..61, ``.`` or space = skip, ``0`` = erase)
inflate   ``rows`` (front-view pixel rows, as layer), ``origin [x,y]``, ``center`` (z of the middle), ``depth`` (thickness at
          the thickest point), ``profile`` round|flat|soft (round = cushion/ball, soft = in between), ``key``
lathe     ``center [x, z]``, ``from`` (y of the first radius), ``radii`` [r per y, upwards], ``c``, ``hollow`` (optional):
          a solid of revolution around a vertical axis (bottles, trees, towers, columns)
paint     ``side`` front|back|left|right|top|bottom, ``rows``, ``origin [u,v]``, ``depth`` (default 1), ``key``: recolour the
          surface seen from that side, pixel by pixel (the first voxel along each line of sight), so faces and patterns
          follow curved shapes. Columns/rows as the views: front x/y, back x mirrored, left z (back first), right z (front
          first), top x/z (back first), bottom x/z (front first).
recolor   ``from_c``, ``to_c`` (all voxels of one colour)
mirror    ``axis`` x|y|z, ``keep`` both|low|high (both = union, low/high = copy that half onto the other)
move      ``by [dx,dy,dz]`` (voxels pushed outside are dropped)
clear     (everything)
========  =======================================================================================================

``c`` is a palette index; ``c: 0`` erases with the same shape. ``layer`` planes: *front* draws columns as x and rows as
y (top row = highest y) at ``z = at``; *side* draws columns as z and rows as y at ``x = at`` (the model seen from its
left, from -x: the first column is the back, z = 0, the last the front); *top* draws columns as x and rows as z (seen
from above: first row = back, z = 0) at ``y = at``.
"""

from __future__ import annotations

import math
import re
import time
from dataclasses import dataclass, field
from typing import Any, Iterable, Optional

import numpy as np

from .errors import SeuratError

MAX_SIDE = 256
MAX_VOXELS = 1_000_000
MAX_OPS = 20_000
KEY_CHARS = "123456789abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ"   # 1..61
DEFAULT_KEY = {ch: i + 1 for i, ch in enumerate(KEY_CHARS)}
HEX_RE = re.compile(r"^#?([0-9a-fA-F]{6})$")
PLANES = ("front", "side", "top")


def norm_hex(value: Any) -> str:
    m = HEX_RE.match(str(value or "").strip())
    if not m:
        raise SeuratError("invalid", f"Colour {value!r} is not #rrggbb.")
    return "#" + m.group(1).lower()


def hex_rgb(value: str) -> tuple[int, int, int]:
    v = norm_hex(value)
    return int(v[1:3], 16), int(v[3:5], 16), int(v[5:7], 16)


@dataclass
class VoxelModel:
    id: str
    name: str
    grid: np.ndarray                                   # uint8 (sx, sy, sz)
    palette: list[str] = field(default_factory=list)   # palette[i - 1] = colour of index i
    notes: str = ""
    created: float = field(default_factory=time.time)
    updated: float = field(default_factory=time.time)
    source: dict[str, Any] = field(default_factory=dict)

    # ------------------------------------------------------------------ basics
    @classmethod
    def empty(cls, id: str, name: str, size: Iterable[int], palette: Optional[list[str]] = None) -> "VoxelModel":
        sx, sy, sz = check_size(size)
        from .palette import DEFAULT_PALETTE
        return cls(id=id, name=name, grid=np.zeros((sx, sy, sz), dtype=np.uint8), palette=list(palette or DEFAULT_PALETTE))

    @property
    def size(self) -> list[int]:
        return [int(n) for n in self.grid.shape]

    @property
    def count(self) -> int:
        return int(np.count_nonzero(self.grid))

    def bounds(self) -> Optional[list[list[int]]]:
        idx = np.argwhere(self.grid)
        if not len(idx):
            return None
        return [idx.min(axis=0).tolist(), idx.max(axis=0).tolist()]

    def colors_used(self) -> dict[int, int]:
        vals, counts = np.unique(self.grid[self.grid > 0], return_counts=True)
        return {int(v): int(c) for v, c in zip(vals, counts)}

    def voxel_list(self) -> list[list[int]]:
        idx = np.argwhere(self.grid)
        if not len(idx):
            return []
        cols = self.grid[idx[:, 0], idx[:, 1], idx[:, 2]]
        return np.concatenate([idx, cols[:, None]], axis=1).astype(int).tolist()

    def to_dict(self, voxels: bool = True) -> dict[str, Any]:
        out = {"id": self.id, "name": self.name, "size": self.size, "palette": list(self.palette), "count": self.count,
               "bounds": self.bounds(), "notes": self.notes, "created": self.created, "updated": self.updated,
               "source": self.source}
        if voxels:
            out["voxels"] = self.voxel_list()
        return out

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "VoxelModel":
        sx, sy, sz = check_size(data.get("size") or [32, 32, 32])
        grid = np.zeros((sx, sy, sz), dtype=np.uint8)
        vox = data.get("voxels") or []
        if vox:
            arr = np.asarray(vox, dtype=np.int64).reshape(-1, 4)
            ok = ((arr[:, 0] >= 0) & (arr[:, 0] < sx) & (arr[:, 1] >= 0) & (arr[:, 1] < sy) & (arr[:, 2] >= 0)
                  & (arr[:, 2] < sz) & (arr[:, 3] > 0) & (arr[:, 3] < 256))
            arr = arr[ok]
            grid[arr[:, 0], arr[:, 1], arr[:, 2]] = arr[:, 3].astype(np.uint8)
        palette = [norm_hex(c) for c in (data.get("palette") or [])][:255]
        if not palette:
            from .palette import DEFAULT_PALETTE
            palette = list(DEFAULT_PALETTE)
        return cls(id=str(data["id"]), name=str(data.get("name") or "Sin título"), grid=grid, palette=palette,
                   notes=str(data.get("notes") or ""), created=float(data.get("created") or time.time()),
                   updated=float(data.get("updated") or time.time()), source=dict(data.get("source") or {}))

    def copy(self) -> "VoxelModel":
        return VoxelModel(id=self.id, name=self.name, grid=self.grid.copy(), palette=list(self.palette), notes=self.notes,
                          created=self.created, updated=self.updated, source=dict(self.source))

    def resize(self, size: Iterable[int], anchor: str = "bottom-center") -> None:
        """New grid size; the voxels keep their place relative to the bottom centre (or the origin with ``anchor=origin``)."""
        sx, sy, sz = check_size(size)
        new = np.zeros((sx, sy, sz), dtype=np.uint8)
        ox, oy, oz = self.grid.shape
        if anchor == "origin":
            dx = dz = 0
        else:
            dx, dz = (sx - ox) // 2, (sz - oz) // 2
        src = self.grid
        xs = slice(max(0, -dx), min(ox, sx - dx))
        zs = slice(max(0, -dz), min(oz, sz - dz))
        ys = slice(0, min(oy, sy))
        part = src[xs, ys, zs]
        new[xs.start + dx: xs.start + dx + part.shape[0], 0: part.shape[1], zs.start + dz: zs.start + dz + part.shape[2]] = part
        self.grid = new

    def palette_rgb(self) -> np.ndarray:
        """(256, 3) uint8: row i = colour of index i (row 0 unused)."""
        out = np.zeros((256, 3), dtype=np.uint8)
        for i, c in enumerate(self.palette[:255]):
            out[i + 1] = hex_rgb(c)
        return out


def check_size(size: Iterable[int]) -> tuple[int, int, int]:
    try:
        sx, sy, sz = (int(v) for v in size)
    except (TypeError, ValueError):
        raise SeuratError("invalid", "size must be three whole numbers [x, y, z].") from None
    for v in (sx, sy, sz):
        if not 1 <= v <= MAX_SIDE:
            raise SeuratError("invalid", f"Each side must be between 1 and {MAX_SIDE} (got {[sx, sy, sz]}).")
    return sx, sy, sz


# ====================================================================================================== operations
class OpError(SeuratError):
    def __init__(self, index: int, op: Any, message: str):
        name = op.get("op") if isinstance(op, dict) else "?"
        super().__init__("invalid", f"Operation {index} ({name}): {message}", "Fix that operation and send the list again.",
                         op_index=index)


def _ints(value: Any, n: int, what: str) -> list[int]:
    if not isinstance(value, (list, tuple)) or len(value) != n:
        raise ValueError(f"{what} must be a list of {n} numbers")
    try:
        return [int(round(float(v))) for v in value]
    except (TypeError, ValueError):
        raise ValueError(f"{what} must contain numbers") from None


def _floats(value: Any, n: int, what: str) -> list[float]:
    if not isinstance(value, (list, tuple)) or len(value) != n:
        raise ValueError(f"{what} must be a list of {n} numbers")
    try:
        return [float(v) for v in value]
    except (TypeError, ValueError):
        raise ValueError(f"{what} must contain numbers") from None


def _color(op: dict[str, Any], model: VoxelModel, key: str = "c") -> int:
    if key not in op:
        raise ValueError(f"missing {key} (palette index, 0 = erase)")
    try:
        c = int(op[key])
    except (TypeError, ValueError):
        raise ValueError(f"{key} must be a palette index") from None
    if not 0 <= c <= 255:
        raise ValueError(f"{key}={c} is outside 0..255")
    if c > len(model.palette):
        raise ValueError(f"{key}={c} but the palette has {len(model.palette)} colours")
    return c


def _clip_box(model: VoxelModel, a: list[int], b: list[int]) -> Optional[tuple[slice, slice, slice]]:
    lo = [max(0, min(a[i], b[i])) for i in range(3)]
    hi = [min(model.grid.shape[i] - 1, max(a[i], b[i])) for i in range(3)]
    if any(lo[i] > hi[i] for i in range(3)):
        return None
    return slice(lo[0], hi[0] + 1), slice(lo[1], hi[1] + 1), slice(lo[2], hi[2] + 1)


def _put(model: VoxelModel, pts: np.ndarray, c: int) -> int:
    if not len(pts):
        return 0
    sx, sy, sz = model.grid.shape
    ok = (pts[:, 0] >= 0) & (pts[:, 0] < sx) & (pts[:, 1] >= 0) & (pts[:, 1] < sy) & (pts[:, 2] >= 0) & (pts[:, 2] < sz)
    pts = pts[ok]
    model.grid[pts[:, 0], pts[:, 1], pts[:, 2]] = c
    return int(len(pts))


def _box(model: VoxelModel, op: dict[str, Any]) -> int:
    a, b = _ints(op.get("from"), 3, "from"), _ints(op.get("to"), 3, "to")
    c = _color(op, model)
    box = _clip_box(model, a, b)
    if box is None:
        return 0
    if op.get("hollow"):
        sub = model.grid[box]
        inner = np.zeros(sub.shape, dtype=bool)
        if all(n > 2 for n in sub.shape):
            inner[1:-1, 1:-1, 1:-1] = True
        sub[~inner] = c
        return int((~inner).sum())
    model.grid[box] = c
    return int(np.prod([s.stop - s.start for s in box]))


def _line(model: VoxelModel, op: dict[str, Any]) -> int:
    """A line of voxels; with ``r`` (and optionally ``r_to`` at the far end) a capsule: necks, tails, limbs, horns."""
    a = np.array(_floats(op.get("from"), 3, "from"))
    b = np.array(_floats(op.get("to"), 3, "to"))
    c = _color(op, model)
    r0 = float(op.get("r") or 0)
    r1 = float(op.get("r_to", r0) or 0)
    if r0 <= 0 and r1 <= 0:
        n = int(np.abs(b - a).max()) + 1
        t = np.linspace(0.0, 1.0, n)[:, None]
        pts = np.rint(a + (b - a) * t).astype(np.int64)
        return _put(model, pts, c)
    rmax = max(r0, r1)
    lo = [int(math.floor(min(a[i], b[i]) - rmax)) for i in range(3)]
    hi = [int(math.ceil(max(a[i], b[i]) + rmax)) for i in range(3)]
    box = _clip_box(model, lo, hi)
    if box is None:
        return 0
    gx, gy, gz = np.meshgrid(*[np.arange(s.start, s.stop) for s in box], indexing="ij")
    p = np.stack([gx, gy, gz], axis=-1).astype(np.float64)
    ab = b - a
    denom = float(ab @ ab) or 1.0
    t = np.clip(((p - a) @ ab) / denom, 0.0, 1.0)
    closest = a + t[..., None] * ab
    dist = np.linalg.norm(p - closest, axis=-1)
    mask = dist <= (r0 + (r1 - r0) * t) + 0.35
    sub = model.grid[box]
    sub[mask] = c
    return int(mask.sum())


def _prism(model: VoxelModel, op: dict[str, Any]) -> int:
    """A polygon (``points`` [[u, v], ...] on the front, side or top plane, as layer) extruded from ``from`` to ``to``
    along the third axis: wings, fins, ears, tails, flat blades."""
    plane = str(op.get("plane") or "front")
    if plane not in PLANES:
        raise ValueError("plane must be front, side or top")
    pts = op.get("points")
    if not isinstance(pts, list) or len(pts) < 3:
        raise ValueError("points must be at least three [u, v] corners")
    poly = [tuple(_floats(p, 2, "points[]")) for p in pts]
    try:
        d0, d1 = sorted([int(round(float(op.get("from")))), int(round(float(op.get("to"))))])
    except (TypeError, ValueError):
        raise ValueError("needs from and to (along the third axis)") from None
    c = _color(op, model)
    sx, sy, sz = model.grid.shape
    nu, nv = {"front": (sx, sy), "side": (sz, sy), "top": (sx, sz)}[plane]
    uu, vv = np.meshgrid(np.arange(nu) + 0.0, np.arange(nv) + 0.0, indexing="ij")
    inside = np.zeros((nu, nv), dtype=bool)
    j = len(poly) - 1
    for i in range(len(poly)):          # even-odd rule on pixel centres
        (ui, vi), (uj, vj) = poly[i], poly[j]
        cond = ((vi > vv) != (vj > vv)) & (uu < (uj - ui) * (vv - vi) / ((vj - vi) or 1e-9) + ui)
        inside ^= cond
        j = i
    # the polygon's own edges count as inside (thin shapes stay connected)
    for i in range(len(poly)):
        (u0, v0), (u1, v1) = poly[i - 1], poly[i]
        n = int(max(abs(u1 - u0), abs(v1 - v0))) + 1
        for t in np.linspace(0, 1, n * 2):
            u, v = int(round(u0 + (u1 - u0) * t)), int(round(v0 + (v1 - v0) * t))
            if 0 <= u < nu and 0 <= v < nv:
                inside[u, v] = True
    total = 0
    us, vs = np.nonzero(inside)
    for d in range(d0, d1 + 1):
        dcol = np.full(len(us), d)
        if plane == "front":
            p = np.stack([us, vs, dcol], axis=1)
        elif plane == "side":
            p = np.stack([dcol, vs, us], axis=1)
        else:
            p = np.stack([us, dcol, vs], axis=1)
        total += _put(model, p.astype(np.int64), c)
    return total


def _sphere(model: VoxelModel, op: dict[str, Any]) -> int:
    center = _floats(op.get("center"), 3, "center")
    if "radii" in op:
        radii = _floats(op.get("radii"), 3, "radii")
    elif "r" in op:
        r = float(op["r"])
        radii = [r, r, r]
    else:
        raise ValueError("needs r or radii")
    if any(r <= 0 for r in radii):
        raise ValueError("radius must be > 0")
    c = _color(op, model)
    lo = [int(math.floor(center[i] - radii[i])) for i in range(3)]
    hi = [int(math.ceil(center[i] + radii[i])) for i in range(3)]
    box = _clip_box(model, lo, hi)
    if box is None:
        return 0
    gx, gy, gz = np.meshgrid(*[np.arange(s.start, s.stop) for s in box], indexing="ij")
    d = ((gx - center[0]) / radii[0]) ** 2 + ((gy - center[1]) / radii[1]) ** 2 + ((gz - center[2]) / radii[2]) ** 2
    mask = d <= 1.0 + 1e-9
    if op.get("hollow"):
        inner = (((gx - center[0]) / max(radii[0] - 1, 0.5)) ** 2 + ((gy - center[1]) / max(radii[1] - 1, 0.5)) ** 2
                 + ((gz - center[2]) / max(radii[2] - 1, 0.5)) ** 2) <= 1.0
        mask &= ~inner
    sub = model.grid[box]
    sub[mask] = c
    return int(mask.sum())


def _cylinder(model: VoxelModel, op: dict[str, Any]) -> int:
    axis = str(op.get("axis") or "y")
    if axis not in ("x", "y", "z"):
        raise ValueError("axis must be x, y or z")
    ca, cb = _floats(op.get("center"), 2, "center")
    try:
        f, t = int(round(float(op.get("from")))), int(round(float(op.get("to"))))
        r = float(op.get("r"))
    except (TypeError, ValueError):
        raise ValueError("needs from, to (along the axis) and r") from None
    if r <= 0:
        raise ValueError("r must be > 0")
    c = _color(op, model)
    ax = "xyz".index(axis)
    others = [i for i in range(3) if i != ax]
    lo, hi = [0, 0, 0], [0, 0, 0]
    lo[ax], hi[ax] = min(f, t), max(f, t)
    lo[others[0]], hi[others[0]] = int(math.floor(ca - r)), int(math.ceil(ca + r))
    lo[others[1]], hi[others[1]] = int(math.floor(cb - r)), int(math.ceil(cb + r))
    box = _clip_box(model, lo, hi)
    if box is None:
        return 0
    grids = np.meshgrid(*[np.arange(s.start, s.stop) for s in box], indexing="ij")
    mask = (grids[others[0]] - ca) ** 2 + (grids[others[1]] - cb) ** 2 <= r * r + 1e-9
    sub = model.grid[box]
    sub[mask] = c
    return int(mask.sum())


def _layer(model: VoxelModel, op: dict[str, Any]) -> int:
    plane = str(op.get("plane") or "front")
    if plane not in PLANES:
        raise ValueError("plane must be front, side or top")
    rows = op.get("rows")
    if not isinstance(rows, list) or not rows or not all(isinstance(r, str) for r in rows):
        raise ValueError("rows must be a non-empty list of strings")
    key = dict(DEFAULT_KEY)
    for ch, val in (op.get("key") or {}).items():
        if len(str(ch)) != 1:
            raise ValueError(f"key {ch!r} must be one character")
        key[str(ch)] = int(val)
    if "at" in op:
        depth = [int(round(float(op["at"])))] * 2
    elif "from" in op and "to" in op:
        depth = sorted([int(round(float(op["from"]))), int(round(float(op["to"])))])
    else:
        raise ValueError("needs at, or from and to")
    ou, ov = _ints(op.get("origin") or [0, 0], 2, "origin")
    n_rows = len(rows)
    pts: dict[int, list[tuple[int, int]]] = {}
    for r, line in enumerate(rows):
        for col, ch in enumerate(line):
            if ch in (".", " "):
                continue
            if ch == "0":
                c = 0
            elif ch in key:
                c = key[ch]
            else:
                raise ValueError(f"row {r} uses {ch!r}, which is not in the key")
            if not 0 <= c <= len(model.palette):
                raise ValueError(f"{ch!r} maps to {c}, outside the palette")
            u = ou + col
            v = (ov + (n_rows - 1 - r)) if plane in ("front", "side") else (ov + r)
            pts.setdefault(c, []).append((u, v))
    total = 0
    for c, uv in pts.items():
        uv_arr = np.asarray(uv, dtype=np.int64)
        for d in range(depth[0], depth[1] + 1):
            dcol = np.full((len(uv_arr), 1), d, dtype=np.int64)
            if plane == "front":      # u = x, v = y, depth = z
                p = np.concatenate([uv_arr[:, :1], uv_arr[:, 1:], dcol], axis=1)
            elif plane == "side":     # u = z, v = y, depth = x
                p = np.concatenate([dcol, uv_arr[:, 1:], uv_arr[:, :1]], axis=1)
            else:                     # top: u = x, v = z, depth = y
                p = np.concatenate([uv_arr[:, :1], dcol, uv_arr[:, 1:]], axis=1)
            total += _put(model, p, c)
    return total


def distance_inside(mask: np.ndarray) -> np.ndarray:
    """City-block distance of every True pixel of ``mask`` to the nearest False one (or the border)."""
    h, w = mask.shape
    d = np.where(mask, h + w, 0).astype(np.int64)
    padded = np.zeros((h + 2, w + 2), dtype=np.int64)
    padded[1:-1, 1:-1] = d
    for _ in range(max(h, w)):
        cur = padded[1:-1, 1:-1]
        best = np.minimum.reduce([cur, padded[:-2, 1:-1] + 1, padded[2:, 1:-1] + 1, padded[1:-1, :-2] + 1, padded[1:-1, 2:] + 1])
        best = np.where(mask, best, 0)
        if (best == cur).all():
            break
        padded[1:-1, 1:-1] = best
    return padded[1:-1, 1:-1]


def _rows_grid(op: dict[str, Any], model: VoxelModel) -> np.ndarray:
    """``rows`` + ``key`` as an (h, w) array of palette indices (-1 = skip), row 0 = top."""
    rows = op.get("rows")
    if not isinstance(rows, list) or not rows or not all(isinstance(r, str) for r in rows):
        raise ValueError("rows must be a non-empty list of strings")
    key = dict(DEFAULT_KEY)
    for ch, val in (op.get("key") or {}).items():
        if len(str(ch)) != 1:
            raise ValueError(f"key {ch!r} must be one character")
        key[str(ch)] = int(val)
    w = max(len(r) for r in rows)
    out = np.full((len(rows), w), -1, dtype=np.int64)
    for r, line in enumerate(rows):
        for col, ch in enumerate(line):
            if ch in (".", " "):
                continue
            c = 0 if ch == "0" else key.get(ch)
            if c is None:
                raise ValueError(f"row {r} uses {ch!r}, which is not in the key")
            if not 0 <= c <= len(model.palette):
                raise ValueError(f"{ch!r} maps to {c}, outside the palette")
            out[r, col] = c
    return out


def _inflate(model: VoxelModel, op: dict[str, Any]) -> int:
    grid = _rows_grid(op, model)
    ox, oy = _ints(op.get("origin") or [0, 0], 2, "origin")
    try:
        zc = float(op.get("center", model.grid.shape[2] / 2 - 0.5))
        depth = float(op.get("depth", model.grid.shape[2]))
    except (TypeError, ValueError):
        raise ValueError("center and depth must be numbers") from None
    if depth <= 0:
        raise ValueError("depth must be > 0")
    profile = str(op.get("profile") or "round")
    if profile not in ("round", "flat", "soft"):
        raise ValueError("profile must be round, flat or soft")
    mask = grid >= 0
    dist = distance_inside(mask).astype(np.float64)
    dmax = max(1.0, float(dist.max()))
    t = dist / dmax
    if profile == "round":
        f = np.sqrt(np.clip(1 - (1 - t) ** 2, 0, 1))
    elif profile == "soft":
        f = np.sqrt(t)
    else:
        f = np.ones_like(t)
    half = np.maximum(0.5, f * depth / 2.0)
    h = grid.shape[0]
    total = 0
    pts_by_c: dict[int, list[list[int]]] = {}
    for r in range(h):
        for col in range(grid.shape[1]):
            c = int(grid[r, col])
            if c < 0:
                continue
            x, y = ox + col, oy + (h - 1 - r)
            lo, hi = int(math.floor(zc - half[r, col] + 0.5)), int(math.ceil(zc + half[r, col] - 0.5))
            for z in range(lo, hi + 1):
                pts_by_c.setdefault(c, []).append([x, y, z])
    for c, pts in pts_by_c.items():
        total += _put(model, np.asarray(pts, dtype=np.int64), c)
    return total


def _lathe(model: VoxelModel, op: dict[str, Any]) -> int:
    cx, cz = _floats(op.get("center"), 2, "center")
    radii = op.get("radii")
    if not isinstance(radii, list) or not radii:
        raise ValueError("radii must be a list of radii, one per y going up")
    try:
        radii = [float(r) for r in radii]
        y0 = int(round(float(op.get("from", 0))))
    except (TypeError, ValueError):
        raise ValueError("radii and from must be numbers") from None
    c = _color(op, model)
    sx, sy, sz = model.grid.shape
    xs, zs = np.meshgrid(np.arange(sx), np.arange(sz), indexing="ij")
    d2 = (xs - cx) ** 2 + (zs - cz) ** 2
    total = 0
    for i, r in enumerate(radii):
        y = y0 + i
        if not 0 <= y < sy or r <= 0:
            continue
        mask = d2 <= r * r + 1e-9
        if op.get("hollow") and r > 1:
            mask &= d2 > (r - 1) ** 2
        model.grid[:, y, :][mask] = c
        total += int(mask.sum())
    return total


SIDES = ("front", "back", "left", "right", "top", "bottom")


def _paint(model: VoxelModel, op: dict[str, Any]) -> int:
    side = str(op.get("side") or "front")
    if side not in SIDES:
        raise ValueError("side must be front, back, left, right, top or bottom")
    grid = _rows_grid(op, model)
    ou, ov = _ints(op.get("origin") or [0, 0], 2, "origin")
    depth = max(1, int(op.get("depth") or 1))
    g = model.grid
    sx, sy, sz = g.shape
    h = grid.shape[0]
    total = 0
    for r in range(h):
        for col in range(grid.shape[1]):
            c = int(grid[r, col])
            if c < 0:
                continue
            u = ou + col
            if side in ("top", "bottom"):
                v = ov + r
            else:
                v = ov + (h - 1 - r)
            if side == "front":                    # u = x, v = y, look along -z
                x, y = u, v
                if not (0 <= x < sx and 0 <= y < sy):
                    continue
                line = np.flatnonzero(g[x, y, :])
                if not len(line):
                    continue
                zs = range(line.max(), max(-1, line.max() - depth), -1)
                cells = [(x, y, z) for z in zs]
            elif side == "back":                   # seen from behind: u counts from the right side of the model
                x, y = sx - 1 - u, v
                if not (0 <= x < sx and 0 <= y < sy):
                    continue
                line = np.flatnonzero(g[x, y, :])
                if not len(line):
                    continue
                cells = [(x, y, z) for z in range(line.min(), min(sz, line.min() + depth))]
            elif side in ("left", "right"):
                z = u if side == "left" else sz - 1 - u
                y = v
                if not (0 <= z < sz and 0 <= y < sy):
                    continue
                line = np.flatnonzero(g[:, y, z])
                if not len(line):
                    continue
                cells = ([(x, y, z) for x in range(line.min(), min(sx, line.min() + depth))] if side == "left"
                         else [(x, y, z) for x in range(line.max(), max(-1, line.max() - depth), -1)])
            else:                                  # top: u = x, v = z (back first); bottom: v = z (front first)
                x = u
                z = v if side == "top" else sz - 1 - v
                if not (0 <= x < sx and 0 <= z < sz):
                    continue
                line = np.flatnonzero(g[x, :, z])
                if not len(line):
                    continue
                cells = ([(x, y, z) for y in range(line.max(), max(-1, line.max() - depth), -1)] if side == "top"
                         else [(x, y, z) for y in range(line.min(), min(sy, line.min() + depth))])
            for cx, cy, cz in cells:
                if g[cx, cy, cz]:
                    g[cx, cy, cz] = c
                    total += 1
    return total


def _mirror(model: VoxelModel, op: dict[str, Any]) -> int:
    axis = str(op.get("axis") or "x")
    if axis not in ("x", "y", "z"):
        raise ValueError("axis must be x, y or z")
    keep = str(op.get("keep") or "both")
    ax = "xyz".index(axis)
    g = model.grid
    flipped = np.flip(g, axis=ax)
    n = g.shape[ax]
    if keep == "both":
        model.grid = np.where(g > 0, g, flipped)
    elif keep in ("low", "high"):
        half = n // 2
        idx = [slice(None)] * 3
        if keep == "low":
            idx[ax] = slice(n - half, n)
        else:
            idx[ax] = slice(0, half)
        g[tuple(idx)] = flipped[tuple(idx)]
    else:
        raise ValueError("keep must be both, low or high")
    return model.count


def _move(model: VoxelModel, op: dict[str, Any]) -> int:
    d = _ints(op.get("by"), 3, "by")
    g = model.grid
    out = np.zeros_like(g)
    src = [slice(max(0, -d[i]), min(g.shape[i], g.shape[i] - d[i])) for i in range(3)]
    dst = [slice(s.start + d[i], s.stop + d[i]) for i, s in enumerate(src)]
    if all(s.stop > s.start for s in src):
        out[tuple(dst)] = g[tuple(src)]
    model.grid = out
    return model.count


def _recolor(model: VoxelModel, op: dict[str, Any]) -> int:
    a, b = _color(op, model, "from_c"), _color(op, model, "to_c")
    mask = model.grid == a
    model.grid[mask] = b
    return int(mask.sum())


def _voxels(model: VoxelModel, op: dict[str, Any]) -> int:
    items = op.get("list")
    if not isinstance(items, list):
        raise ValueError("list must be [[x,y,z,c], ...]")
    total = 0
    by_c: dict[int, list[list[int]]] = {}
    for i, item in enumerate(items):
        x, y, z, c = _ints(item, 4, f"list[{i}]")
        if not 0 <= c <= len(model.palette):
            raise ValueError(f"list[{i}] uses colour {c}, outside the palette")
        by_c.setdefault(c, []).append([x, y, z])
    for c, pts in by_c.items():
        total += _put(model, np.asarray(pts, dtype=np.int64), c)
    return total


def _voxel(model: VoxelModel, op: dict[str, Any]) -> int:
    x, y, z = _ints(op.get("at"), 3, "at")
    return _put(model, np.asarray([[x, y, z]], dtype=np.int64), _color(op, model))


def _clear(model: VoxelModel, op: dict[str, Any]) -> int:
    model.grid[:] = 0
    return 0


OPS = {"voxel": _voxel, "voxels": _voxels, "box": _box, "line": _line, "sphere": _sphere, "ellipsoid": _sphere,
       "cylinder": _cylinder, "layer": _layer, "extrude": _layer, "inflate": _inflate, "lathe": _lathe, "paint": _paint, "prism": _prism, "recolor": _recolor, "mirror": _mirror, "move": _move,
       "clear": _clear}


def apply_ops(model: VoxelModel, ops: list[Any], *, atomic: bool = True) -> dict[str, Any]:
    """Apply ``ops`` in order. ``atomic`` (default): any invalid operation leaves the model untouched and raises an
    :class:`OpError` naming it. Otherwise invalid ones are skipped and reported in ``errors``."""
    if not isinstance(ops, list):
        raise SeuratError("invalid", "ops must be a list of operations.")
    if len(ops) > MAX_OPS:
        raise SeuratError("too_big", f"{len(ops)} operations; the limit is {MAX_OPS}.", "Use box, layer or sphere for big areas.")
    work = model.copy() if atomic else model
    errors: list[str] = []
    touched = 0
    for i, op in enumerate(ops):
        try:
            if not isinstance(op, dict) or op.get("op") not in OPS:
                raise ValueError(f"unknown op; use one of {', '.join(sorted(OPS))}")
            touched += OPS[op["op"]](work, op)
        except ValueError as exc:
            if atomic:
                raise OpError(i, op, str(exc)) from None
            errors.append(f"{i} ({op.get('op') if isinstance(op, dict) else '?'}): {exc}")
    if work.count > MAX_VOXELS:
        raise SeuratError("too_big", f"The model would have {work.count} voxels (limit {MAX_VOXELS}).")
    if atomic:
        model.grid = work.grid
    model.updated = time.time()
    return {"applied": len(ops) - len(errors), "touched": touched, "count": model.count, "errors": errors}
