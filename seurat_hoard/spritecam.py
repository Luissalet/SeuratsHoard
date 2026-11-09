"""Sprite target: a model judged by how it looks from the game camera that drew its sprites.

Handheld-game overworld sprites are not flat views. The camera looks down on the world, so the front and back sprites
show the subject from well above (the vertical axis of the picture mixes height and depth: a bird's tail fan, seen
from behind, lands at the bottom because it is the nearest part) and the side sprites are nearly in profile. To compare
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
SIDE_PITCH = 20.0
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
    panels: list[Image.Image] = field(default_factory=list)

    def summary(self) -> str:
        where = {"front": "in front", "back": "behind", "side": "at the side"}.get(self.role, self.role)
        shape = ""
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
                "sprite_aspect": round(self.sprite_aspect, 3), "model_aspect": round(self.model_aspect, 3)}


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


def match_view(model: VoxelModel, ref: SpriteRef, yaw: float, pitch: float) -> ViewMatch:
    t = ref.rgba
    t_mask = t[..., 3] >= 128
    t_box = _bbox(t_mask) or (0, 0, t.shape[1], t.shape[0])
    longest = max(model.size)
    scale = max(2.0, 4.0 * max(t.shape[:2]) / max(1, longest))     # render finer than the sprite, then reduce
    r = camera(model, yaw, pitch, scale)
    m = _fit(r, t.shape[:2], t_box)
    m_mask = m[..., 3] >= 128
    inter, union = int((t_mask & m_mask).sum()), int((t_mask | m_mask).sum())
    area = max(1, int(t_mask.sum()))
    both = t_mask & m_mask
    paint = both & (t[..., :3].astype(np.int32).sum(axis=2) > 120)     # the sprite's dark outline is not a colour
    colour = float(np.abs(t[..., :3].astype(np.int32) - m[..., :3].astype(np.int32))[paint].sum(axis=1).mean() / 765.0) if paint.any() else 1.0
    r_box = _bbox(np.asarray(r)[..., 3] >= 128)
    model_aspect = ((r_box[2] - r_box[0]) / max(1, r_box[3] - r_box[1])) if r_box else 0.0
    vm = ViewMatch(ref.role, ref.name, yaw, pitch, inter / max(1, union), float((t_mask & ~m_mask).sum()) / area,
                   float((m_mask & ~t_mask).sum()) / area, colour,
                   (t_box[2] - t_box[0]) / max(1, t_box[3] - t_box[1]), model_aspect)
    diff = np.zeros_like(t)
    diff[both] = (150, 150, 150, 255)
    diff[t_mask & ~m_mask] = (235, 60, 50, 255)
    diff[m_mask & ~t_mask] = (60, 120, 240, 255)
    vm.panels = [_big(t), _big(m), _big(diff)]
    return vm


def match(model: VoxelModel, refs: list[SpriteRef], front_back_pitch: float = FRONT_BACK_PITCH,
          side_pitch: float = SIDE_PITCH) -> list[ViewMatch]:
    """Each sprite against the model from its camera. A side sprite is tried from both sides and the better one kept
    (sheets draw one side and mirror it)."""
    out = []
    for ref in refs:
        if ref.role == "front":
            out.append(match_view(model, ref, 0.0, front_back_pitch))
        elif ref.role == "back":
            out.append(match_view(model, ref, 180.0, front_back_pitch))
        elif ref.role == "side":
            a, b = match_view(model, ref, -90.0, side_pitch), match_view(model, ref, 90.0, side_pitch)
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
    return float(np.mean([m.iou for m in matches])) if matches else 0.0


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
