"""Sprite target: a model judged by how it looks from the game camera that drew its sprites.

Handheld-game overworld sprites are not flat views. The camera looks down on the world, so the front and back sprites
show the subject from well above (the vertical axis of the picture mixes height and depth: a bird's tail fan, seen
from behind, lands at the bottom because it is the nearest part) and the side sprites are true profiles, drawn with no
camera angle: the most reliable view, one sprite pixel per voxel in z and y. The voxel program gets the side sprite as an
array in grid coordinates with helpers to carve to it and paint from it (``program_data``). To compare
a voxel model with them Seurat renders the model from the same camera, brings it to the sprite's resolution and lays the
two silhouettes over each other: what is missing, what is extra, how well they match. That is the feedback the model
gets after every run, and what ``model_sprite_match`` answers.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Optional

import numpy as np
from PIL import Image, ImageDraw, ImageFont

from .images import cutout
from .render import BG_DARK, camera
from .voxels import VoxelModel

#: default camera heights (degrees above the horizon) of the overworld sprites
FRONT_BACK_PITCH = 45.0
SIDE_PITCH = 0.0
#: the side view counts double in the overall match: it is a true profile
WEIGHTS = {"side": 2.0}
ZOOM = 6


@dataclass
class SpriteRef:
    role: str                      # front | back | side
    name: str
    rgba: np.ndarray               # (h, w, 4), background removed, cropped, pixel-art enlargement undone


@dataclass
class ViewMatch:
    role: str
    name: str
    yaw: float
    pitch: float
    iou: float                     # silhouette overlap: 1 = identical
    missing: float                 # share of the sprite's pixels the model does not cover
    extra: float                   # model pixels outside the sprite, relative to the sprite's area
    colour: float                  # mean colour difference where both are present (0 = same, 1 = opposite)
    sprite_aspect: float           # width / height of the sprite's subject
    model_aspect: float            # width / height of the model seen from that camera
    fit: str = "pixel"             # pixel: one voxel per sprite pixel; area: scaled to the sprite
    size_ratio: tuple = (1.0, 1.0)  # model width and height from that camera / the sprite's (pixel fit: 1 = right size)
    panels: list[Image.Image] = field(default_factory=list)
    missing_mask: Optional[np.ndarray] = None     # (h, w) sprite pixels the model does not cover
    extra_mask: Optional[np.ndarray] = None       # (h, w) model pixels outside the sprite

    def summary(self) -> str:
        where = {"front": "in front", "back": "behind", "side": "at the side"}.get(self.role, self.role)
        shape = ""
        if self.fit == "pixel":
            wr, hr = self.size_ratio
            parts = []
            if abs(wr - 1) > 0.06:
                parts.append(f"{wr:.0%} of the sprite's width")
            if abs(hr - 1) > 0.06:
                parts.append(f"{hr:.0%} of its height")
            return (f"{self.role.upper()} sprite '{self.name}' (camera {where}, {self.pitch:.0f} degrees above, one voxel per pixel): "
                    f"silhouette match {self.iou:.2f}, missing {self.missing:.0%} of the sprite, extra {self.extra:.0%}, colour "
                    f"difference {self.colour:.2f}" + ("; from this camera the model is " + " and ".join(parts) if parts else "; size right") + ".")
        if self.model_aspect > self.sprite_aspect * 1.12:
            shape = "; from this camera the model is too WIDE for its height (or too short)"
        elif self.model_aspect < self.sprite_aspect / 1.12:
            shape = "; from this camera the model is too NARROW for its height (or too tall)"
        return (f"{self.role.upper()} sprite '{self.name}' (camera {where}, {self.pitch:.0f} degrees above): silhouette match "
                f"{self.iou:.2f}, missing {self.missing:.0%} of the sprite, extra {self.extra:.0%}, colour difference "
                f"{self.colour:.2f}; width/height sprite {self.sprite_aspect:.2f} vs model {self.model_aspect:.2f}{shape}.")

    def to_dict(self) -> dict[str, Any]:
        return {"role": self.role, "name": self.name, "yaw": self.yaw, "pitch": self.pitch, "iou": round(self.iou, 3),
                "missing": round(self.missing, 3), "extra": round(self.extra, 3), "colour": round(self.colour, 3),
                "sprite_aspect": round(self.sprite_aspect, 3), "model_aspect": round(self.model_aspect, 3), "fit": self.fit,
                "size_ratio": [round(float(v), 3) for v in self.size_ratio]}


def sprite_ref(role: str, name: str, data: bytes) -> SpriteRef:
    return SpriteRef(role, name, cutout(data))


def _bbox(mask: np.ndarray) -> Optional[tuple[int, int, int, int]]:
    ys, xs = np.nonzero(mask)
    if not len(xs):
        return None
    return int(xs.min()), int(ys.min()), int(xs.max()) + 1, int(ys.max()) + 1


def _fit(render: Image.Image, target_shape: tuple[int, int], target_box: tuple[int, int, int, int]) -> np.ndarray:
    """The render cropped to the model and scaled (uniformly) so its area matches the sprite subject's, centred on it,
    on a canvas the size of the sprite."""
    arr = np.asarray(render)
    box = _bbox(arr[..., 3] >= 128)
    h_t, w_t = target_shape
    out = np.zeros((h_t, w_t, 4), np.uint8)
    if box is None:
        return out
    crop = render.crop(box)
    tx0, ty0, tx1, ty1 = target_box
    f = float(np.sqrt(((tx1 - tx0) * (ty1 - ty0)) / max(1, crop.width * crop.height)))
    nw, nh = max(1, round(crop.width * f)), max(1, round(crop.height * f))
    small = np.asarray(crop.resize((nw, nh), Image.BOX))
    cx, cy = (tx0 + tx1) / 2, (ty0 + ty1) / 2
    ox, oy = int(round(cx - nw / 2)), int(round(cy - nh / 2))
    x0, y0 = max(0, ox), max(0, oy)
    x1, y1 = min(w_t, ox + nw), min(h_t, oy + nh)
    if x1 > x0 and y1 > y0:
        out[y0:y1, x0:x1] = small[y0 - oy:y1 - oy, x0 - ox:x1 - ox]
    return out


def _big(arr: np.ndarray, zoom: int = ZOOM) -> Image.Image:
    img = Image.fromarray(arr, "RGBA")
    bg = Image.new("RGBA", img.size, (58, 54, 64, 255))
    bg.alpha_composite(img)
    return bg.resize((img.width * zoom, img.height * zoom), Image.NEAREST)


_SAMPLES = np.array([[a, b, c] for a in (0.25, 0.75) for b in (0.25, 0.75) for c in (0.25, 0.75)])


def project(model: VoxelModel, yaw: float, pitch: float) -> np.ndarray:
    """The model seen from the camera at ONE PIXEL PER VOXEL (the sprite's own scale), flat colours: RGBA (h, w, 4).
    Each voxel is sampled at eight inner points, the nearest one wins each pixel; in a true profile (pitch 0, from the
    side) every voxel is exactly one pixel."""
    from .render import camera_basis
    occ = np.argwhere(model.grid > 0)
    if not len(occ):
        return np.zeros((1, 1, 4), np.uint8)
    c, r, u = camera_basis(yaw, pitch)
    pts = (occ[:, None, :] + _SAMPLES[None]).reshape(-1, 3)
    vox = np.repeat(np.arange(len(occ)), len(_SAMPLES))
    pu, pv, dep = pts @ r, -(pts @ u), pts @ c
    ix = np.floor(pu - np.floor(pu.min())).astype(np.int64)
    iy = np.floor(pv - np.floor(pv.min())).astype(np.int64)
    h, w = int(iy.max()) + 1, int(ix.max()) + 1
    key = iy * w + ix
    order = np.lexsort((dep, key))                 # per pixel, the nearest sample last
    last = np.r_[key[order][1:] != key[order][:-1], True]
    win = order[last]
    pal = model.palette_rgb()
    out = np.zeros((h * w, 4), np.uint8)
    g = model.grid
    cols = g[occ[vox[win], 0], occ[vox[win], 1], occ[vox[win], 2]]
    out[key[win], :3] = pal[cols]
    out[key[win], 3] = 255
    return out.reshape(h, w, 4)


def _place(m: np.ndarray, t_mask: np.ndarray, search: int = 4) -> np.ndarray:
    """``m`` (h, w, 4) placed on a canvas the size of the sprite: centres of the two boxes together, then the integer
    shift (within ``search`` pixels) that overlaps the sprite best."""
    h_t, w_t = t_mask.shape
    mm = m[..., 3] >= 128
    mb, tb = _bbox(mm), _bbox(t_mask)
    if mb is None or tb is None:
        return np.zeros((h_t, w_t, 4), np.uint8)
    ox0 = int(round((tb[0] + tb[2]) / 2 - (mb[0] + mb[2]) / 2))
    oy0 = int(round((tb[1] + tb[3]) / 2 - (mb[1] + mb[3]) / 2))

    def put(ox, oy):
        out = np.zeros((h_t, w_t, 4), np.uint8)
        y0, x0 = max(0, oy), max(0, ox)
        y1, x1 = min(h_t, oy + m.shape[0]), min(w_t, ox + m.shape[1])
        if y1 > y0 and x1 > x0:
            out[y0:y1, x0:x1] = m[y0 - oy:y1 - oy, x0 - ox:x1 - ox]
        return out

    best, best_s = None, -1.0
    for dy in range(-search, search + 1):
        for dx in range(-search, search + 1):
            cand = put(ox0 + dx, oy0 + dy)
            cm = cand[..., 3] >= 128
            score = (cm & t_mask).sum() / max(1, (cm | t_mask).sum()) - 1e-4 * (abs(dx) + abs(dy))
            if score > best_s:
                best, best_s = cand, score
    return best


def match_view(model: VoxelModel, ref: SpriteRef, yaw: float, pitch: float, fit: str = "auto") -> ViewMatch:
    """``fit``: pixel = the model at one voxel per sprite pixel (what a sprite rebuild should be; a model of the wrong
    size shows as red or blue bands); area = scaled to the sprite's size (any model); auto = pixel unless the model is
    far from the sprite's size."""
    t = ref.rgba
    t_mask = t[..., 3] >= 128
    t_box = _bbox(t_mask) or (0, 0, t.shape[1], t.shape[0])
    proj = project(model, yaw, pitch)
    p_box = _bbox(proj[..., 3] >= 128)
    if fit == "auto":
        ratio = ((p_box[2] - p_box[0]) * (p_box[3] - p_box[1])) / max(1, (t_box[2] - t_box[0]) * (t_box[3] - t_box[1])) if p_box else 1.0
        fit = "pixel" if 0.4 <= ratio <= 2.5 else "area"
    if fit == "pixel":
        m = _place(proj, t_mask)
        model_aspect = ((p_box[2] - p_box[0]) / max(1, p_box[3] - p_box[1])) if p_box else 0.0
    else:
        longest = max(model.size)
        scale = max(2.0, 4.0 * max(t.shape[:2]) / max(1, longest))     # render finer than the sprite, then reduce
        r = camera(model, yaw, pitch, scale, shade=False)              # flat colours, like the sprite
        m = _fit(r, t.shape[:2], t_box)
        r_box = _bbox(np.asarray(r)[..., 3] >= 128)
        model_aspect = ((r_box[2] - r_box[0]) / max(1, r_box[3] - r_box[1])) if r_box else 0.0
    m_mask = m[..., 3] >= 128
    inter, union = int((t_mask & m_mask).sum()), int((t_mask | m_mask).sum())
    area = max(1, int(t_mask.sum()))
    both = t_mask & m_mask
    paint = both & (t[..., :3].astype(np.int32).sum(axis=2) > DARK)    # the sprite's dark outline is not a colour
    colour = float(np.abs(t[..., :3].astype(np.int32) - m[..., :3].astype(np.int32))[paint].sum(axis=1).mean() / 765.0) if paint.any() else 1.0
    vm = ViewMatch(ref.role, ref.name, yaw, pitch, inter / max(1, union), float((t_mask & ~m_mask).sum()) / area,
                   float((m_mask & ~t_mask).sum()) / area, colour,
                   (t_box[2] - t_box[0]) / max(1, t_box[3] - t_box[1]), model_aspect, fit=fit,
                   size_ratio=(((p_box[2] - p_box[0]) / max(1, t_box[2] - t_box[0]),
                                (p_box[3] - p_box[1]) / max(1, t_box[3] - t_box[1])) if p_box else (0.0, 0.0)))
    diff = np.zeros_like(t)
    diff[both] = (150, 150, 150, 255)
    diff[t_mask & ~m_mask] = (235, 60, 50, 255)
    diff[m_mask & ~t_mask] = (60, 120, 240, 255)
    vm.panels = [_big(t), _big(m), _big(diff)]
    vm.missing_mask, vm.extra_mask = t_mask & ~m_mask, m_mask & ~t_mask
    return vm


def match(model: VoxelModel, refs: list[SpriteRef], front_back_pitch: float = FRONT_BACK_PITCH,
          side_pitch: float = SIDE_PITCH, fit: str = "auto") -> list[ViewMatch]:
    """Each sprite against the model from its camera. A side sprite is tried from both sides and the better one kept
    (sheets draw one side and mirror it)."""
    out = []
    for ref in refs:
        if ref.role == "front":
            out.append(match_view(model, ref, 0.0, front_back_pitch, fit))
        elif ref.role == "back":
            out.append(match_view(model, ref, 180.0, front_back_pitch, fit))
        elif ref.role == "side":
            a, b = match_view(model, ref, -90.0, side_pitch, fit), match_view(model, ref, 90.0, side_pitch, fit)
            out.append(a if a.iou >= b.iou else b)
    return out


def _font(size: int = 13):
    try:
        return ImageFont.load_default(size=size)
    except TypeError:
        return ImageFont.load_default()


def sheet(matches: list[ViewMatch]) -> Image.Image:
    """One row per sprite: the sprite, the model from the same camera at the sprite's resolution, and the silhouette
    difference (red: in the sprite but missing in the model; blue: in the model but not in the sprite)."""
    if not matches:
        return Image.new("RGBA", (8, 8), BG_DARK)
    font = _font(13)
    head = 20
    rows = []
    for vm in matches:
        w = sum(p.width for p in vm.panels) + 8 * (len(vm.panels) + 1)
        h = max(p.height for p in vm.panels) + head + 8
        row = Image.new("RGBA", (w, h), BG_DARK)
        d = ImageDraw.Draw(row)
        x = 8
        labels = [f"{vm.role} sprite", f"model, camera {vm.pitch:.0f} deg above", f"match {vm.iou:.2f} (red missing, blue extra)"]
        for p, text in zip(vm.panels, labels):
            row.paste(p, (x, head))
            d.text((x, 3), text, fill=(235, 225, 200, 255), font=font)
            x += p.width + 8
        rows.append(row)
    out = Image.new("RGBA", (max(r.width for r in rows), sum(r.height for r in rows)), (12, 10, 14, 255))
    y = 0
    for r in rows:
        out.paste(r, (0, y))
        y += r.height
    return out


def mean_iou(matches: list[ViewMatch]) -> float:
    """Weighted mean silhouette match (the side, a true profile, counts double)."""
    if not matches:
        return 0.0
    w = [WEIGHTS.get(m.role, 1.0) for m in matches]
    return float(np.average([m.iou for m in matches], weights=w))


def board(refs: list[SpriteRef], front_back_pitch: float = FRONT_BACK_PITCH, side_pitch: float = SIDE_PITCH,
          zoom: int = ZOOM) -> Image.Image:
    """The target sprites side by side at a readable size, each labelled with its camera."""
    font = _font(13)
    head = 20
    tiles = []
    for ref in refs:
        pitch = side_pitch if ref.role == "side" else front_back_pitch
        where = {"front": "facing the camera", "back": "from behind", "side": "from the side"}.get(ref.role, "")
        tiles.append((f"{ref.role.upper()} sprite ({where}, camera {pitch:.0f} deg above)", _big(ref.rgba, zoom)))
    w = sum(max(t.width, 8 * len(text)) for text, t in tiles) + 12 * (len(tiles) + 1)
    h = max(t.height for _, t in tiles) + head + 12
    out = Image.new("RGBA", (w, h), BG_DARK)
    d = ImageDraw.Draw(out)
    x = 12
    for text, t in tiles:
        out.paste(t, (x, head + 4))
        d.text((x, 3), text, fill=(235, 225, 200, 255), font=font)
        x += max(t.width, 8 * len(text)) + 12
    return out


def colours(refs: list[SpriteRef], limit: int = 24) -> list[str]:
    """The sprites' own colours, most used first, without the near-black outline."""
    from collections import Counter
    count: Counter = Counter()
    for ref in refs:
        px = ref.rgba[ref.rgba[..., 3] >= 128][:, :3].astype(np.int32)
        px = px[px.sum(axis=1) > 120]
        count.update(map(tuple, px.tolist()))
    return ["#%02x%02x%02x" % c for c, _ in count.most_common(limit)]


# ---------------------------------------------------------------------------------------------- data for the program
DARK = 120          # an RGB sum at or below this is the sprite's outline (or a dark separation line)


def _by_role(refs: list[SpriteRef], role: str) -> Optional[SpriteRef]:
    return next((r for r in refs if r.role == role), None)


def grid_size(refs: list[SpriteRef], margin: int = 4) -> list[int]:
    """A grid that holds the sprites at one voxel per pixel: X from the front/back width (their horizontal axis is
    exactly x), Y and Z from the side sprite (a true profile: columns are z, rows are y)."""
    side = _by_role(refs, "side")
    fb = [r for r in refs if r.role in ("front", "back")]
    longest = max(max(r.rgba.shape[:2]) for r in refs)
    x = max((r.rgba.shape[1] for r in fb), default=longest)
    y = side.rgba.shape[0] if side is not None else longest
    z = side.rgba.shape[1] if side is not None else longest
    up = lambda v: int(min(128, max(16, -(-(v + 2 * margin) // 4) * 4)))
    return [up(x), up(y), up(z)]


def _palette_index(rgba: np.ndarray, palette: list[str]) -> np.ndarray:
    """(h, w) indices into ``palette`` (1-based, 0 = empty); outline pixels take the colour of the nearest inner pixel."""
    pal = np.array([[int(c[i:i + 2], 16) for i in (1, 3, 5)] for c in palette], np.int32)
    opaque = rgba[..., 3] >= 128
    rgb = rgba[..., :3].astype(np.int32)
    inner = opaque & (rgb.sum(axis=2) > DARK)
    d = ((rgb[..., None, :] - pal[None, None]) ** 2).sum(axis=3)
    idx = d.argmin(axis=2) + 1
    out = np.where(inner, idx, 0)
    # outline: borrow the nearest inner colour (a few growing passes)
    todo = opaque & ~inner
    for _ in range(6):
        if not todo.any():
            break
        grown = out.copy()
        for dy, dx in ((1, 0), (-1, 0), (0, 1), (0, -1)):
            sh = np.roll(np.roll(out, dy, axis=0), dx, axis=1)
            take = todo & (grown == 0) & (sh > 0)
            grown[take] = sh[take]
        todo &= grown == 0
        out = grown
    out[todo] = 1
    return out


def program_data(refs: list[SpriteRef], size: list[int], front_back_pitch: float = FRONT_BACK_PITCH,
                 side_pitch: float = SIDE_PITCH) -> dict[str, Any]:
    """What the voxel program receives about the sprites: their palette, the side sprite placed in grid coordinates
    (columns = z, rows = y, one pixel per voxel, centred in z, standing on y = 2) and every sprite with its camera for
    ``paint_view``."""
    palette = colours(refs, limit=32)
    X, Y, Z = size
    data: dict[str, Any] = {"palette": palette, "views": {}}
    for ref in refs:
        idx = _palette_index(ref.rgba, palette)
        yaw = {"front": 0.0, "back": 180.0, "side": -90.0}.get(ref.role, 0.0)
        pitch = side_pitch if ref.role == "side" else front_back_pitch
        data["views"].setdefault(ref.role, {"idx": idx.tolist(), "yaw": yaw, "pitch": pitch, "name": ref.name})
    side = _by_role(refs, "side")
    if side is not None:
        idx = np.asarray(data["views"]["side"]["idx"])
        h, w = idx.shape
        side_zy = np.zeros((Z, Y), np.int64)
        z0, y0 = max(0, (Z - w) // 2), 2
        for r in range(h):
            y = y0 + (h - 1 - r)
            if not 0 <= y < Y:
                continue
            row = idx[r, : max(0, min(w, Z - z0))]
            side_zy[z0: z0 + len(row), y] = row
        data["side"] = side_zy.tolist()
        data["side_place"] = {"z0": z0, "y0": y0, "w": w, "h": h}
    return data


def char_map(ref: SpriteRef, palette: list[str]) -> str:
    """The sprite as text, one character per pixel ('.' empty, '#' outline, letters = palette colours), rows from the
    top, with column numbers every 10."""
    keys = "abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ"
    idx = _palette_index(ref.rgba, palette)
    rgba = ref.rgba
    dark = (rgba[..., 3] >= 128) & (rgba[..., :3].astype(np.int32).sum(axis=2) <= DARK)
    h, w = idx.shape
    lines = ["     " + "".join(str((c // 10) % 10) if c % 10 == 0 else " " for c in range(w))]
    for r in range(h):
        line = "".join("." if idx[r, c] == 0 else "#" if dark[r, c] else keys[(idx[r, c] - 1) % len(keys)] for c in range(w))
        lines.append(f"{r:4d} {line}")
    return "\n".join(lines)


def _components(mask: np.ndarray, min_px: int = 4) -> list[tuple[int, int, int, int, int]]:
    """Connected areas of ``mask`` (4-neighbour) as (pixels, col0, col1, row0, row1), biggest first."""
    seen = np.zeros_like(mask, bool)
    h, w = mask.shape
    out = []
    for r0, c0 in zip(*np.nonzero(mask)):
        if seen[r0, c0]:
            continue
        stack, px = [(r0, c0)], []
        seen[r0, c0] = True
        while stack:
            r, c = stack.pop()
            px.append((r, c))
            for dr, dc in ((1, 0), (-1, 0), (0, 1), (0, -1)):
                rr, cc = r + dr, c + dc
                if 0 <= rr < h and 0 <= cc < w and mask[rr, cc] and not seen[rr, cc]:
                    seen[rr, cc] = True
                    stack.append((rr, cc))
        if len(px) >= min_px:
            rs, cs = [p[0] for p in px], [p[1] for p in px]
            out.append((len(px), min(cs), max(cs), min(rs), max(rs)))
    return sorted(out, reverse=True)


def where(vm: ViewMatch, data: Optional[dict] = None, size: Optional[list[int]] = None, limit: int = 5) -> str:
    """The biggest missing and extra areas of a view in numbers the program can use: for the side sprite (a true
    profile placed as SIDE) in grid z and y; for front and back in grid x (their horizontal axis) and sprite rows."""
    if vm.missing_mask is None:
        return ""
    lines = []
    place = (data or {}).get("side_place")
    X = (size or [0])[0]
    h, w = vm.missing_mask.shape
    for label, mask in (("missing (red)", vm.missing_mask), ("extra (blue)", vm.extra_mask)):
        comps = _components(mask)[:limit]
        if not comps:
            continue
        parts = []
        for n, c0, c1, r0, r1 in comps:
            if vm.role == "side" and place:
                z0, z1 = place["z0"] + c0, place["z0"] + c1
                ytop = place["y0"] + place["h"] - 1
                parts.append(f"{n} px at z {z0}-{z1}, y {ytop - r1}-{ytop - r0}")
            elif vm.role in ("front", "back") and X:
                centre = (w - 1) / 2
                if vm.role == "front":
                    x0, x1 = X / 2 + (c0 - centre), X / 2 + (c1 - centre)
                else:                                    # seen from behind: x grows to the left
                    x0, x1 = X / 2 - (c1 - centre), X / 2 - (c0 - centre)
                parts.append(f"{n} px at x {max(0, int(round(x0)))}-{min(X - 1, int(round(x1)))}, sprite rows {r0}-{r1} (from the top)")
            else:
                parts.append(f"{n} px at sprite columns {c0}-{c1}, rows {r0}-{r1}")
        lines.append(f"  {label}: " + "; ".join(parts))
    return "\n".join(lines)
