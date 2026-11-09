"""Palettes: the default one for new models and the helpers that merge, trim and match colours."""

from __future__ import annotations

import colorsys
from typing import Iterable

import numpy as np


def _hex(r: float, g: float, b: float) -> str:
    return "#%02x%02x%02x" % tuple(int(round(max(0.0, min(1.0, v)) * 255)) for v in (r, g, b))


def _build_default() -> list[str]:
    out = [_hex(v, v, v) for v in (0.07, 0.18, 0.32, 0.47, 0.62, 0.76, 0.88, 1.0)]       # greys, black to white
    hues = [0.0, 0.04, 0.08, 0.12, 0.16, 0.25, 0.36, 0.47, 0.55, 0.62, 0.72, 0.83, 0.92]
    for h in hues:                                                                    # dark / base / light per hue
        out += [_hex(*colorsys.hsv_to_rgb(h, 0.75, 0.42)), _hex(*colorsys.hsv_to_rgb(h, 0.78, 0.82)),
                _hex(*colorsys.hsv_to_rgb(h, 0.38, 0.97))]
    out += ["#f2d3b3", "#d9a066", "#a86b45", "#5c3a24"]                               # skin and wood tones
    return out


DEFAULT_PALETTE: list[str] = _build_default()


def rgb_array(palette: Iterable[str]) -> np.ndarray:
    pal = list(palette)
    return np.array([[int(c[1:3], 16), int(c[3:5], 16), int(c[5:7], 16)] for c in pal], dtype=np.float64).reshape(-1, 3)


def nearest_index(palette: list[str], colors: np.ndarray) -> np.ndarray:
    """1-based palette index nearest (in a perceptual-ish weighted RGB) to each colour of ``colors`` (N, 3)."""
    pal = rgb_array(palette)
    w = np.array([0.30, 0.59, 0.11]) ** 0.5
    d = (((colors[:, None, :] - pal[None, :, :]) * w) ** 2).sum(axis=2)
    return d.argmin(axis=1) + 1


def merge(palette: list[str], extra: Iterable[str], limit: int = 255) -> tuple[list[str], dict[str, int]]:
    """``palette`` plus the colours of ``extra`` it lacks (up to ``limit``); returns it and {hex: index}."""
    out = list(palette)
    index = {c: i + 1 for i, c in enumerate(out)}
    for c in extra:
        c = c.lower()
        if c not in index and len(out) < limit:
            out.append(c)
            index[c] = len(out)
    return out, index
