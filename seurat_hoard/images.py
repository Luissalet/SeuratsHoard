"""Reference pictures: read a PNG / sprite, find its real pixels and turn it into a small indexed grid.

The grid is what both the deterministic builders (``builders.py``) and the language model read: rows of characters,
one per pixel, ``.`` for transparent, ``1-9a-zA-Z`` for the colours of a small palette. A sprite enlarged by an integer
factor (every pixel drawn as a 4x4 block, say) is brought back to its real resolution first.
"""

from __future__ import annotations

import io
import math
from collections import Counter
from dataclasses import dataclass, field
from typing import Optional

import numpy as np
from PIL import Image

from .errors import SeuratError
from .voxels import KEY_CHARS

MAX_BYTES = 25 * 1024 * 1024
MAX_PIXELS = 40_000_000


@dataclass
class Indexed:
    width: int
    height: int
    palette: list[str]                 # palette[i - 1] = colour of index i
    grid: np.ndarray                   # (h, w) int, 0 = transparent; row 0 = top of the picture
    scale: int = 1                     # pixel-art enlargement that was undone
    notes: list[str] = field(default_factory=list)

    def rows(self) -> list[str]:
        chars = "." + KEY_CHARS
        return ["".join(chars[int(v)] for v in row) for row in self.grid]

    def to_dict(self, rows: bool = True) -> dict:
        out = {"width": self.width, "height": self.height, "palette": self.palette, "scale": self.scale, "notes": self.notes,
               "opaque": int(np.count_nonzero(self.grid))}
        if rows:
            out["rows"] = self.rows()
        return out

    def preview_png(self, max_side: int = 512) -> bytes:
        rgba = np.zeros((self.height, self.width, 4), dtype=np.uint8)
        for i, c in enumerate(self.palette):
            m = self.grid == i + 1
            rgba[m] = (int(c[1:3], 16), int(c[3:5], 16), int(c[5:7], 16), 255)
        img = Image.fromarray(rgba, "RGBA")
        k = max(1, max_side // max(self.width, self.height))
        img = img.resize((self.width * k, self.height * k), Image.NEAREST)
        buf = io.BytesIO()
        img.save(buf, "PNG")
        return buf.getvalue()


def load(data: bytes) -> Image.Image:
    if not data:
        raise SeuratError("invalid", "The picture is empty.")
    if len(data) > MAX_BYTES:
        raise SeuratError("too_big", "The picture is larger than 25 MB.")
    try:
        img = Image.open(io.BytesIO(data))
        img.load()
    except Exception:  # noqa: BLE001 - Pillow raises many types for broken files
        raise SeuratError("invalid", "That file is not a picture Seurat can read (PNG, JPG, GIF, WebP, BMP).") from None
    if img.width * img.height > MAX_PIXELS:
        raise SeuratError("too_big", "The picture has too many pixels.")
    if getattr(img, "is_animated", False):
        img.seek(0)
    return img.convert("RGBA")


def _background_to_alpha(arr: np.ndarray, notes: list[str]) -> np.ndarray:
    """A picture without transparency whose four corners share one colour: that colour is the background."""
    if (arr[..., 3] < 250).any():
        return arr
    corners = [tuple(arr[0, 0, :3]), tuple(arr[0, -1, :3]), tuple(arr[-1, 0, :3]), tuple(arr[-1, -1, :3])]
    bg, n = Counter(corners).most_common(1)[0]
    if n < 3:
        return arr
    diff = np.abs(arr[..., :3].astype(np.int32) - np.array(bg, dtype=np.int32)).sum(axis=2)
    # flood from the borders only, so the same colour inside the figure stays
    mask = diff <= 24
    reach = np.zeros_like(mask)
    reach[0, :], reach[-1, :], reach[:, 0], reach[:, -1] = mask[0, :], mask[-1, :], mask[:, 0], mask[:, -1]
    for _ in range(max(arr.shape[:2])):
        grown = reach.copy()
        grown[1:, :] |= reach[:-1, :]
        grown[:-1, :] |= reach[1:, :]
        grown[:, 1:] |= reach[:, :-1]
        grown[:, :-1] |= reach[:, 1:]
        grown &= mask
        if (grown == reach).all():
            break
        reach = grown
    if reach.mean() > 0.97:
        return arr
    out = arr.copy()
    out[reach, 3] = 0
    notes.append("background colour %s made transparent" % ("#%02x%02x%02x" % bg))
    return out


def _crop(arr: np.ndarray) -> np.ndarray:
    alpha = arr[..., 3] >= 128
    if not alpha.any():
        raise SeuratError("invalid", "The picture is completely transparent.")
    ys, xs = np.where(alpha)
    return arr[ys.min(): ys.max() + 1, xs.min(): xs.max() + 1]


def _pixel_scale(arr: np.ndarray) -> int:
    """The integer factor a pixel-art picture was enlarged by (1 when it is not pixel art)."""
    h, w = arr.shape[:2]
    if min(h, w) < 8:
        return 1
    rgb = arr[..., :4].astype(np.int32)

    def runs(a: np.ndarray) -> list[int]:
        out = []
        for line in a[:: max(1, len(a) // 40)]:
            change = np.any(line[1:] != line[:-1], axis=1)
            edges = np.flatnonzero(change) + 1
            bounds = np.concatenate([[0], edges, [len(line)]])
            out += [int(b - a2) for a2, b in zip(bounds[:-1], bounds[1:])][1:-1]
        return out

    lengths = runs(rgb) + runs(rgb.transpose(1, 0, 2))
    if len(lengths) < 6:
        return 1
    g = 0
    for n in lengths:
        g = math.gcd(g, n)
        if g == 1:
            return 1
    return g if g >= 2 and h % g == 0 and w % g == 0 else 1


def index_picture(data: bytes, *, max_side: int = 64, colors: int = 16) -> Indexed:
    """``data`` (PNG/JPG/...) to an :class:`Indexed` grid at most ``max_side`` pixels wide or tall, ``colors`` colours."""
    notes: list[str] = []
    arr = np.asarray(load(data)).copy()
    arr = _background_to_alpha(arr, notes)
    arr = _crop(arr)
    scale = _pixel_scale(arr)
    if scale > 1:
        arr = arr[scale // 2:: scale, scale // 2:: scale]
        notes.append(f"pixel art enlarged x{scale}: read at its real size")
    h, w = arr.shape[:2]
    img = Image.fromarray(arr, "RGBA")
    if max(h, w) > max_side:
        k = max_side / max(h, w)
        nw, nh = max(1, round(w * k)), max(1, round(h * k))
        img = img.resize((nw, nh), Image.NEAREST if scale > 1 or _few_colors(arr) else Image.LANCZOS)
        notes.append(f"reduced from {w}x{h} to {nw}x{nh}")
    rgba = np.asarray(img).copy()
    opaque = rgba[..., 3] >= 128
    colors = max(2, min(int(colors), len(KEY_CHARS)))
    rgb_img = Image.fromarray(rgba[..., :3], "RGB")
    q = rgb_img.quantize(colors=colors, method=Image.Quantize.MEDIANCUT, dither=Image.Dither.NONE)
    idx = np.asarray(q).astype(np.int64)
    pal = q.getpalette()[: 3 * 256]
    counts = Counter(idx[opaque].tolist())
    order = [i for i, _ in counts.most_common()]
    remap = {old: new + 1 for new, old in enumerate(order)}
    grid = np.zeros(idx.shape, dtype=np.int64)
    for old, new in remap.items():
        grid[(idx == old) & opaque] = new
    palette = ["#%02x%02x%02x" % tuple(pal[3 * old: 3 * old + 3]) for old in order]
    return Indexed(width=grid.shape[1], height=grid.shape[0], palette=palette, grid=grid, scale=scale, notes=notes)


def _few_colors(arr: np.ndarray) -> bool:
    flat = arr.reshape(-1, 4)
    return len(np.unique(flat[flat[:, 3] >= 128][:, :3], axis=0)) <= 64


def thumbnail_png(data: bytes, side: int = 256) -> bytes:
    img = load(data)
    img.thumbnail((side, side), Image.NEAREST if max(img.size) <= side else Image.LANCZOS)
    buf = io.BytesIO()
    img.save(buf, "PNG")
    return buf.getvalue()


def to_png(data: bytes, max_side: Optional[int] = None) -> bytes:
    """Any readable picture as PNG bytes (optionally no bigger than ``max_side``), e.g. for the model."""
    img = load(data)
    if max_side and max(img.size) > max_side:
        img.thumbnail((max_side, max_side), Image.LANCZOS)
    buf = io.BytesIO()
    img.save(buf, "PNG")
    return buf.getvalue()
