"""Deterministic picture-to-voxel builders: instant results with no model, and the starting point the model refines.

* ``extrude``: the front picture as a slab ``depth`` voxels thick.
* ``inflate``: the front picture puffed up like a cushion: thick in the middle, thin at the edges (depth follows the
  distance to the outline), symmetric front and back.
* ``hull``: front + side (+ top) silhouettes carved into a block (the visual hull), coloured from each view.
"""

from __future__ import annotations

from typing import Optional

import numpy as np

from .errors import SeuratError
from .images import Indexed
from .palette import merge
from .voxels import MAX_SIDE, VoxelModel, distance_inside

MODES = ("sprite", "extrude", "inflate", "hull")


def _distance_inside(mask: np.ndarray) -> np.ndarray:
    return distance_inside(mask)


def _palette_map(ix: Indexed, palette: list[str]) -> tuple[list[str], np.ndarray]:
    pal, index = merge(palette, ix.palette)
    lut = np.zeros(len(ix.palette) + 1, dtype=np.uint8)
    for i, c in enumerate(ix.palette):
        lut[i + 1] = index[c.lower()]
    return pal, lut


def build(mode: str, front: Indexed, *, side: Optional[Indexed] = None, top: Optional[Indexed] = None, depth: int = 0,
          model_id: str = "tmp", name: str = "Sin título", base_palette: Optional[list[str]] = None) -> VoxelModel:
    if mode not in MODES:
        raise SeuratError("invalid", f"mode must be one of {', '.join(MODES)}.")
    h, w = front.height, front.width
    palette = list(base_palette or [])
    palette, lut = _palette_map(front, palette)
    fgrid = lut[front.grid]                     # (h, w) palette indices, 0 = empty
    mask = fgrid > 0
    if mode == "hull":
        if side is None:
            raise SeuratError("invalid", "hull needs a side view as well as the front.", "Add a side picture or use inflate.")
        return _hull(front, side, top, palette, model_id, name)
    if mode == "extrude":
        d = max(1, min(int(depth or max(2, min(h, w) // 6)), MAX_SIDE))
        model = VoxelModel.empty(model_id, name, [w, h, d], palette)
        for z in range(d):
            model.grid[:, :, z] = np.flipud(fgrid).T
        return model
    if mode == "sprite":
        return _sprite(front, fgrid, mask, palette, depth, model_id, name)
    # inflate
    dist = _distance_inside(mask)
    dmax = max(1, int(dist.max()))
    half_max = max(1, int(depth // 2) if depth else max(1, round(min(h, w) * 0.35)))
    t = dist / dmax
    half = np.where(mask, np.rint(half_max * np.sqrt(np.clip(1 - (1 - t) ** 2, 0, 1))), -1).astype(np.int64)
    d = 2 * half_max + 1
    model = VoxelModel.empty(model_id, name, [w, h, d], palette)
    zc = half_max
    flipped = np.flipud(fgrid)               # row 0 = bottom (y = 0)
    fhalf = np.flipud(half)
    for y in range(h):
        for x in range(w):
            k = fhalf[y, x]
            if k >= 0:
                model.grid[x, y, zc - k: zc + k + 1] = flipped[y, x]
    return model


def _fit(ix: Indexed, height: int) -> np.ndarray:
    """``ix.grid`` scaled (nearest) to ``height`` rows, keeping its aspect."""
    if ix.height == height:
        return ix.grid
    k = height / ix.height
    w = max(1, round(ix.width * k))
    ys = np.minimum((np.arange(height) / k).astype(int), ix.height - 1)
    xs = np.minimum((np.arange(w) / k).astype(int), ix.width - 1)
    return ix.grid[ys][:, xs]


def _hull(front: Indexed, side: Indexed, top: Optional[Indexed], palette: list[str], model_id: str, name: str) -> VoxelModel:
    h = front.height
    palette, flut = _palette_map(front, palette)
    palette, slut = _palette_map(side, palette)
    fgrid = np.flipud(flut[front.grid])                       # (h, w): [y, x]
    sgrid = np.flipud(slut[_fit(side, h)])                    # (h, d): [y, z] (side seen from +x: column 0 = back, z = 0)
    w, d = fgrid.shape[1], sgrid.shape[1]
    occ = (fgrid > 0).T[:, :, None] & (sgrid > 0)[None, :, :]                     # (w, h, d) via broadcasting [x, y, z]
    tgrid = None
    if top is not None:
        palette, tlut = _palette_map(top, palette)
        tg = tlut[top.grid]                                   # rows = z (back first), cols = x
        zs = np.minimum((np.arange(d) * top.height / d).astype(int), top.height - 1)
        xs = np.minimum((np.arange(w) * top.width / w).astype(int), top.width - 1)
        tgrid = tg[zs][:, xs]                                 # (d, w): [z, x]
        occ &= (tgrid > 0).T[:, None, :]
    model = VoxelModel.empty(model_id, name, [w, h, d], palette)
    colour = np.broadcast_to(fgrid.T[:, :, None], occ.shape).copy()
    # the outermost voxels along x take the side view's colour where it has one
    side_col = np.broadcast_to(sgrid[None, :, :], occ.shape)
    for xsel in (lambda o: o.argmax(axis=0), lambda o: o.shape[0] - 1 - o[::-1].argmax(axis=0)):
        xi = xsel(occ)                                        # (h, d)
        has = occ.any(axis=0)
        yy, zz = np.nonzero(has)
        xx = xi[yy, zz]
        colour[xx, yy, zz] = np.where(side_col[xx, yy, zz] > 0, side_col[xx, yy, zz], colour[xx, yy, zz])
    if tgrid is not None:
        yi = occ.shape[1] - 1 - occ[:, ::-1, :].argmax(axis=1)   # (w, d): highest y per column
        has = occ.any(axis=1)
        xx, zz = np.nonzero(has)
        yy = yi[xx, zz]
        tc = tgrid.T[xx, zz]
        colour[xx, yy, zz] = np.where(tc > 0, tc, colour[xx, yy, zz])
    # front-most voxels keep the front colour (it is the main view)
    zi = occ.shape[2] - 1 - occ[:, :, ::-1].argmax(axis=2)       # (w, h)
    has = occ.any(axis=2)
    xx, yy = np.nonzero(has)
    zz = zi[xx, yy]
    colour[xx, yy, zz] = fgrid.T[xx, yy]
    model.grid = np.where(occ, colour, 0).astype(np.uint8)
    return model


def _sprite(front: Indexed, fgrid: np.ndarray, mask: np.ndarray, palette: list[str], depth: int, model_id: str,
            name: str) -> VoxelModel:
    """The sprite as a voxel diorama piece (the look of sprite-to-voxel game mods): the picture on the front and the
    back, a gentle rounded thickness, and the dark 2D outline kept only on those two faces, so the sides take the colour
    of the part next to them instead of a black wall."""
    h, w = fgrid.shape
    rgb = np.array([[int(c[i:i + 2], 16) for i in (1, 3, 5)] for c in palette] or [[0, 0, 0]], dtype=np.float64)
    luma = np.zeros(len(palette) + 1)
    if len(palette):
        luma[1:] = rgb @ np.array([0.299, 0.587, 0.114])
    dark = mask & (luma[fgrid] < 55)
    # outline = dark pixels touching the empty background or other outline pixels on the silhouette
    pad = np.pad(mask, 1)
    edge = mask & ~(pad[:-2, 1:-1] & pad[2:, 1:-1] & pad[1:-1, :-2] & pad[1:-1, 2:])
    outline = dark & (edge | (np.pad(dark, 1)[:-2, 1:-1] | np.pad(dark, 1)[2:, 1:-1]))
    # the colour each outline pixel shows on the sides: its nearest non-outline neighbour, darkened
    inner = mask & ~outline
    side_c = fgrid.copy()
    ys, xs = np.nonzero(outline)
    iy, ix = np.nonzero(inner)
    shade: dict[int, int] = {}
    if len(iy):
        for y, x in zip(ys, xs):
            k = int(np.argmin((iy - y) ** 2 + (ix - x) ** 2))
            c = int(fgrid[iy[k], ix[k]])
            if c not in shade:
                r, g, b = (rgb[c - 1] * 0.7).astype(int)
                hexc = "#%02x%02x%02x" % (r, g, b)
                if hexc not in palette and len(palette) < 255:
                    palette.append(hexc)
                shade[c] = palette.index(hexc) + 1 if hexc in palette else c
            side_c[y, x] = shade[c]
    dist = _distance_inside(mask)
    dmax = max(1, float(dist.max()))
    half_max = max(1, int(depth // 2) if depth else max(2, round(min(h, w) * 0.06)))
    t = dist / dmax
    half = np.where(mask, np.rint(half_max * np.sqrt(np.clip(1 - (1 - t) ** 2, 0, 1))), -1).astype(np.int64)
    d = 2 * half_max + 1
    model = VoxelModel.empty(model_id, name, [w, h, d], palette)
    zc = half_max
    flipped, fside, fhalf = np.flipud(fgrid), np.flipud(side_c), np.flipud(half)
    for y in range(h):
        for x in range(w):
            k = fhalf[y, x]
            if k < 0:
                continue
            model.grid[x, y, zc - k: zc + k + 1] = fside[y, x]
            model.grid[x, y, zc + k] = flipped[y, x]          # the picture on the front face
            model.grid[x, y, zc - k] = flipped[y, x]          # and on the back
    return model
