"""Software renders of a voxel model (no GPU, no browser): orthographic views, isometric views and contact sheets.

They are what an assistant or the model sees of a model (``model_look``, the correction rounds of a generation) and the
"PNG / sprites" export: isometric sprites from four sides, flat views for pixel art.
"""

from __future__ import annotations

import io
from typing import Optional

import numpy as np
from PIL import Image, ImageDraw, ImageFont

from .errors import SeuratError
from .voxels import VoxelModel

ORTHO = ("front", "back", "left", "right", "top", "bottom")
ISO = ("iso", "iso_front_left", "iso_back_left", "iso_back_right")   # iso = seen from front-right
VIEWS = ORTHO + ISO + ("sheet",)
BG_DARK = (26, 22, 30, 255)


def _shade(rgb: np.ndarray, k) -> np.ndarray:
    return np.clip(rgb.astype(np.float64) * np.asarray(k)[..., None] if np.ndim(k) else rgb.astype(np.float64) * k, 0, 255).astype(np.uint8)


def _oriented(grid: np.ndarray, view: str) -> np.ndarray:
    """The grid turned so that ``view`` looks along -z with y up: front stays, back turns 180, left/right 90, top/bottom
    tip the model over."""
    if view == "front":
        return grid
    if view == "back":
        return np.rot90(grid, 2, axes=(0, 2))
    if view == "left":      # from -x: z becomes the horizontal axis (back on the left)
        return np.rot90(grid, -1, axes=(0, 2))
    if view == "right":     # from +x
        return np.rot90(grid, 1, axes=(0, 2))
    if view == "top":       # from +y, back of the model at the top of the picture
        return np.rot90(grid, 1, axes=(1, 2))
    if view == "bottom":
        return np.rot90(grid, -1, axes=(1, 2))
    raise SeuratError("invalid", f"Unknown view {view!r}; use one of {', '.join(VIEWS)}.")


def ortho(model: VoxelModel, view: str = "front", scale: int = 8, background=(0, 0, 0, 0), shade: bool = True) -> Image.Image:
    g = _oriented(model.grid, view)
    sx, sy, sz = g.shape
    occ = g > 0
    has = occ.any(axis=2)                                      # (x, y)
    zi = sz - 1 - occ[:, :, ::-1].argmax(axis=2)               # nearest voxel to the viewer
    col = np.take_along_axis(g, zi[:, :, None], axis=2)[:, :, 0]
    pal = model.palette_rgb()
    rgb = pal[col]                                             # (x, y, 3)
    if shade and sz > 1:
        k = 0.62 + 0.38 * (zi / (sz - 1))
        rgb = _shade(rgb, k)
    img = np.zeros((sy, sx, 4), dtype=np.uint8)
    img[..., :] = background
    flipped_has = np.flipud(has.T)
    img[flipped_has, :3] = np.flipud(rgb.transpose(1, 0, 2))[flipped_has]
    img[flipped_has, 3] = 255
    out = Image.fromarray(img, "RGBA")
    if scale > 1:
        out = out.resize((sx * scale, sy * scale), Image.NEAREST)
    return out


def iso(model: VoxelModel, view: str = "iso", size: int = 8, background=(0, 0, 0, 0), outline: bool = True) -> Image.Image:
    """Isometric picture; ``size`` is the width in pixels of one voxel's top face edge. ``iso`` looks from the
    front-right (+x, +y, +z); the others turn the model so the camera sits front-left, back-left or back-right."""
    k = {"iso": 0, "iso_front_left": 1, "iso_back_left": 2, "iso_back_right": 3}[view]
    g = np.rot90(model.grid, k, axes=(2, 0)) if k else model.grid
    sx, sy, sz = g.shape
    pal = model.palette_rgb()
    occ = g > 0
    pad = np.pad(occ, 1)
    top_vis = ~pad[1:-1, 2:, 1:-1] & occ
    xp_vis = ~pad[2:, 1:-1, 1:-1] & occ
    zp_vis = ~pad[1:-1, 1:-1, 2:] & occ
    any_vis = top_vis | xp_vis | zp_vis
    xs, ys, zs = np.nonzero(any_vis)
    if not len(xs):
        return Image.new("RGBA", (max(1, size * 4), max(1, size * 4)), background)
    # Ambient occlusion: each visible face darkens with the voxels around it in the layer it faces, so creases,
    # wing roots and the ground contact read as volume instead of flat colour.
    p = pad.astype(np.int8)
    def around(layer_shift, axes):
        tot = np.zeros(g.shape, np.int8)
        for d1 in (-1, 0, 1):
            for d2 in (-1, 0, 1):
                if d1 == 0 and d2 == 0:
                    continue
                sh = [1, 1, 1]
                sh[layer_shift[0]] += layer_shift[1]
                sh[axes[0]] += d1
                sh[axes[1]] += d2
                tot += p[sh[0]:sh[0] + sx, sh[1]:sh[1] + sy, sh[2]:sh[2] + sz]
        return tot
    ao_top, ao_x, ao_z = around((1, 1), (0, 2)), around((0, 1), (1, 2)), around((2, 1), (0, 1))
    a = size * 0.8660254           # cos 30 * edge
    b = size * 0.5                 # sin 30 * edge
    # screen of a corner (x, y, z): u = (x - z) * a ; v = (x + z) * b - y * size
    w_px = int((sx + sz) * a) + 4
    h_px = int((sx + sz) * b + sy * size) + 4
    ox = sz * a + 2
    oy = sy * size + 2
    img = Image.new("RGBA", (w_px, h_px), background)
    draw = ImageDraw.Draw(img)
    order = np.argsort(xs + ys + zs, kind="stable")
    line = (0, 0, 0, 60) if outline and size >= 4 else None

    def P(x, y, z):
        return (ox + (x - z) * a, oy + (x + z) * b - y * size)

    for i in order:
        x, y, z = int(xs[i]), int(ys[i]), int(zs[i])
        base = pal[g[x, y, z]].astype(np.float64)
        if top_vis[x, y, z]:
            c = tuple(int(v) for v in np.clip(base * (1.0 - 0.06 * ao_top[x, y, z]), 0, 255)) + (255,)
            draw.polygon([P(x, y + 1, z), P(x + 1, y + 1, z), P(x + 1, y + 1, z + 1), P(x, y + 1, z + 1)], fill=c, outline=line)
        if xp_vis[x, y, z]:
            c = tuple(int(v) for v in np.clip(base * 0.8 * (1.0 - 0.06 * ao_x[x, y, z]), 0, 255)) + (255,)
            draw.polygon([P(x + 1, y, z), P(x + 1, y + 1, z), P(x + 1, y + 1, z + 1), P(x + 1, y, z + 1)], fill=c, outline=line)
        if zp_vis[x, y, z]:
            c = tuple(int(v) for v in np.clip(base * 0.64 * (1.0 - 0.06 * ao_z[x, y, z]), 0, 255)) + (255,)
            draw.polygon([P(x, y, z + 1), P(x + 1, y, z + 1), P(x + 1, y + 1, z + 1), P(x, y + 1, z + 1)], fill=c, outline=line)
    return img


def _label(img: Image.Image, text: str) -> Image.Image:
    out = Image.new("RGBA", (img.width, img.height + 18), BG_DARK)
    out.paste(img, (0, 18), img)
    d = ImageDraw.Draw(out)
    try:
        font = ImageFont.load_default(size=13)
    except TypeError:
        font = ImageFont.load_default()
    d.text((4, 2), text, fill=(235, 225, 200, 255), font=font)
    return out


def sheet(model: VoxelModel, target: int = 1024) -> Image.Image:
    """Contact sheet: front, left, top, back as flat views and two isometric ones, labelled (what the model reviews)."""
    sx, sy, sz = model.size
    longest = max(sx, sy, sz)
    cell = max(160, target // 3)
    scale = max(1, cell // max(longest, 1))
    iso_size = max(2, int(cell / (longest * 1.9)))
    tiles = [("front", ortho(model, "front", scale, BG_DARK)), ("left", ortho(model, "left", scale, BG_DARK)),
             ("top", ortho(model, "top", scale, BG_DARK)), ("back", ortho(model, "back", scale, BG_DARK)),
             ("iso front-right", iso(model, "iso", iso_size, BG_DARK)), ("iso back-left", iso(model, "iso_back_left", iso_size, BG_DARK))]
    tiles = [_label(t, n) for n, t in tiles]
    cw = max(t.width for t in tiles) + 8
    ch = max(t.height for t in tiles) + 8
    out = Image.new("RGBA", (cw * 3, ch * 2), (12, 10, 14, 255))
    for i, t in enumerate(tiles):
        out.paste(t, ((i % 3) * cw + 4, (i // 3) * ch + 4), t)
    return out


def render(model: VoxelModel, view: str = "iso", *, size: Optional[int] = None, background: str = "transparent") -> Image.Image:
    bg = (0, 0, 0, 0) if background == "transparent" else BG_DARK
    if view == "sheet":
        return sheet(model, size or 1024)
    longest = max(model.size)
    if view in ORTHO:
        scale = size if size else max(1, 512 // max(longest, 1))
        return ortho(model, view, scale, bg)
    if view in ISO:
        return iso(model, view, size if size else max(2, int(512 / (longest * 1.9))), bg)
    raise SeuratError("invalid", f"Unknown view {view!r}; use one of {', '.join(VIEWS)}.")


def png_bytes(img: Image.Image) -> bytes:
    buf = io.BytesIO()
    img.save(buf, "PNG", optimize=False)
    return buf.getvalue()


def sprites(model: VoxelModel, size: Optional[int] = None) -> Image.Image:
    """Sprite strip: the model from four isometric sides (front-right, front-left, back-left, back-right)."""
    longest = max(model.size)
    s = size or max(2, int(256 / (longest * 1.9)))
    frames = [iso(model, v, s) for v in ISO]
    w = max(f.width for f in frames)
    h = max(f.height for f in frames)
    out = Image.new("RGBA", (w * 4, h), (0, 0, 0, 0))
    for i, f in enumerate(frames):
        out.paste(f, (i * w + (w - f.width) // 2, h - f.height), f)
    return out
