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
#: Handheld-game overworld cameras: the front and back sprites are seen from well above, the side ones nearly in profile.
OVERWORLD = {"overworld_front": (0.0, 45.0), "overworld_back": (180.0, 45.0), "overworld_left": (-90.0, 20.0),
             "overworld_right": (90.0, 20.0)}
VIEWS = ORTHO + ISO + tuple(OVERWORLD) + ("sheet",)
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


_NORMALS = np.array([[1, 0, 0], [-1, 0, 0], [0, 1, 0], [0, -1, 0], [0, 0, 1], [0, 0, -1]], dtype=np.float64)
# corners of each face of the unit cube at the origin, in drawing order
_FACES = {
    0: [(1, 0, 0), (1, 1, 0), (1, 1, 1), (1, 0, 1)], 1: [(0, 0, 0), (0, 0, 1), (0, 1, 1), (0, 1, 0)],
    2: [(0, 1, 0), (0, 1, 1), (1, 1, 1), (1, 1, 0)], 3: [(0, 0, 0), (1, 0, 0), (1, 0, 1), (0, 0, 1)],
    4: [(0, 0, 1), (1, 0, 1), (1, 1, 1), (0, 1, 1)], 5: [(0, 0, 0), (0, 1, 0), (1, 1, 0), (1, 0, 0)]}
_LIGHT = np.array([-0.35, 1.0, 0.55]) / np.linalg.norm([-0.35, 1.0, 0.55])


def camera_basis(yaw: float, pitch: float) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """(towards the camera, screen right, screen up) for a camera at ``yaw`` degrees around the model (0 = in front, on
    +z; 90 = on +x; 180 = behind; -90 = on -x) and ``pitch`` degrees above the horizon."""
    yw, pt = np.radians(yaw), np.radians(max(-89.0, min(89.0, pitch)))
    c = np.array([np.sin(yw) * np.cos(pt), np.sin(pt), np.cos(yw) * np.cos(pt)])
    r = np.cross([0.0, 1.0, 0.0], c)
    r /= np.linalg.norm(r)
    return c, r, np.cross(c, r)


def camera(model: VoxelModel, yaw: float = 0.0, pitch: float = 30.0, scale: float = 6.0, background=(0, 0, 0, 0),
           shade: bool = True) -> Image.Image:
    """Orthographic picture from any direction (see :func:`camera_basis`), ``scale`` pixels per voxel edge. Faces are
    lit from above-front-left; this is how a game camera looking down on the model sees it."""
    g = model.grid
    occ = g > 0
    if not occ.any():
        return Image.new("RGBA", (8, 8), background)
    c, r, u = camera_basis(yaw, pitch)
    pad = np.pad(occ, 1)
    sx, sy, sz = g.shape
    exposed = [~pad[2:, 1:-1, 1:-1], ~pad[:-2, 1:-1, 1:-1], ~pad[1:-1, 2:, 1:-1], ~pad[1:-1, :-2, 1:-1],
               ~pad[1:-1, 1:-1, 2:], ~pad[1:-1, 1:-1, :-2]]
    faces = [f for f in range(6) if _NORMALS[f] @ c > 1e-6]
    pal = model.palette_rgb().astype(np.float64)
    corner = np.array([[0, 0, 0], [sx, 0, 0], [0, sy, 0], [0, 0, sz], [sx, sy, 0], [sx, 0, sz], [0, sy, sz], [sx, sy, sz]], np.float64)
    us, vs = corner @ r, corner @ u
    u0, v1 = us.min(), vs.max()
    w_px, h_px = int(np.ceil((us.max() - u0) * scale)) + 2, int(np.ceil((v1 - vs.min()) * scale)) + 2
    img = Image.new("RGBA", (w_px, h_px), background)
    draw = ImageDraw.Draw(img)
    items = []
    for f in faces:
        xs, ys, zs = np.nonzero(occ & exposed[f])
        if not len(xs):
            continue
        k = (0.5 + 0.5 * max(0.0, float(_NORMALS[f] @ _LIGHT))) if shade else 1.0
        items.append((f, xs, ys, zs, k))
    if not items:
        return img
    allx = np.concatenate([it[1] for it in items]); ally = np.concatenate([it[2] for it in items])
    allz = np.concatenate([it[3] for it in items])
    fid = np.concatenate([np.full(len(it[1]), it[0]) for it in items])
    kk = np.concatenate([np.full(len(it[1]), it[4]) for it in items])
    depth = (allx + 0.5) * c[0] + (ally + 0.5) * c[1] + (allz + 0.5) * c[2]
    order = np.argsort(depth, kind="stable")
    quad = {f: np.array(_FACES[f], np.float64) for f in range(6)}
    for i in order:
        x, y, z, f = allx[i], ally[i], allz[i], fid[i]
        pts = quad[f] + (x, y, z)
        pu, pv = (pts @ r - u0) * scale + 1, (v1 - pts @ u) * scale + 1
        col = np.clip(pal[g[x, y, z]] * kk[i], 0, 255).astype(int)
        draw.polygon(list(zip(pu.tolist(), pv.tolist())), fill=(int(col[0]), int(col[1]), int(col[2]), 255))
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
    if view in OVERWORLD:
        yaw, pitch = OVERWORLD[view]
        return camera(model, yaw, pitch, size if size else max(2, 512 / (longest * 1.5)), bg)
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
