"""A character rebuilt in 3D from its game sprites, part by part.

The side sprite of an overworld sheet is a true profile: it fixes the length (z) and the height (y) of everything, one
pixel per voxel. What it does not say is how each part is built across the body (x): a body is a rounded solid on the
middle plane, a crest is a thin fin, legs come in pairs, wings are two flat sheets that open outwards, a tail can be a
fan. So the profile is cut into parts and each part gets one of those structures:

* ``solid``  on the middle plane, puffed up round (as wide as it is thick in profile, times ``aspect``)
* ``fin``    on the middle plane, a slab of constant ``thickness`` (crests, dorsal fins, blades)
* ``fan``    on the middle plane, widening away from its root (tail fans, skirts, manes)
* ``pair``   two mirrored copies off the middle plane (legs, arms, ears, horns, side fins)
* ``wings``  two mirrored flat sheets whose distance to the middle plane grows upwards and/or backwards
* ``decal``  a marking only on the flanks (eyes, cheek spots): the part under it keeps its own colour inside
* ``skip``   not built: the far wing, arm or leg a side sprite lets peek out behind the near one for looks (the
  mirrored near one already is it)

The model that comes out has exactly the sprite's profile (every sprite pixel that is not skipped is a column of
voxels, nothing outside),
the sprite's colours on its flanks, and real volume. The numbers (widths, how far the wings open, where the legs sit)
are tuned against the front and back sprites, which are drawn from above with heights kept: an oblique camera
(``project``), not a rotation.

Who decides what is a wing? The plan (``{"parts": [...]}``): written by the language model, by a person, or guessed
(``auto_plan``). The geometry is never the model's job.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import Any, Optional

import numpy as np

TYPES = ("solid", "fin", "fan", "pair", "wings", "decal", "skip")
MEDIAN = ("solid", "fin", "fan")
LATERAL = ("pair", "wings", "decal", "skip")
MAX_SIDE = 128


class PlanError(ValueError):
    """The plan cannot be built as written; the message says what to change."""


# ---------------------------------------------------------------------------------------------- small 2-D tools
def _neigh4(mask: np.ndarray) -> np.ndarray:
    """True where a pixel has at least one 4-neighbour inside ``mask``."""
    p = np.pad(mask, 1)
    return p[:-2, 1:-1] | p[2:, 1:-1] | p[1:-1, :-2] | p[1:-1, 2:]


def edt_inside(mask: np.ndarray) -> np.ndarray:
    """Euclidean distance from the centre of every True pixel to the nearest False pixel's edge (pixels on the rim get
    0.5), exact, for small pictures (sprites)."""
    h, w = mask.shape
    out = np.zeros((h, w), np.float64)
    ys, xs = np.nonzero(mask)
    if not len(ys):
        return out
    pad = np.pad(mask, 1)                      # outside the picture counts as empty
    oy, ox = np.nonzero(~pad)
    oy, ox = oy - 1, ox - 1
    # only empty pixels next to the mask matter
    near = _neigh4(np.pad(mask, 1))[oy + 1, ox + 1] | np.pad(_neigh4(mask), 1)[oy + 1, ox + 1]
    p8 = np.pad(mask, 2)
    grown = np.zeros_like(p8)
    for dy in (-1, 0, 1):
        for dx in (-1, 0, 1):
            grown |= np.roll(np.roll(p8, dy, 0), dx, 1)
    ring = grown[1:-1, 1:-1] & ~pad
    oy, ox = np.nonzero(ring)
    oy, ox = oy - 1, ox - 1
    if not len(oy):
        out[mask] = max(h, w)
        return out
    best = np.full(len(ys), np.inf)
    for s in range(0, len(oy), 4096):
        d2 = (ys[:, None] - oy[None, s:s + 4096]) ** 2 + (xs[:, None] - ox[None, s:s + 4096]) ** 2
        best = np.minimum(best, d2.min(axis=1))
    out[ys, xs] = np.sqrt(best) - 0.5
    return out


def local_radius(dist: np.ndarray) -> np.ndarray:
    """For every pixel, the radius of the biggest inscribed disc that covers it (``dist`` from :func:`edt_inside`)."""
    h, w = dist.shape
    out = dist.copy()
    ys, xs = np.nonzero(dist > 0)
    order = np.argsort(-dist[ys, xs])
    yy, xx = np.mgrid[0:h, 0:w]
    done = np.zeros((h, w), bool)
    for i in order:
        y, x, r = int(ys[i]), int(xs[i]), float(dist[ys[i], xs[i]])
        if r <= 1.0:
            break
        if done[y, x] and out[y, x] >= r + 1e-9:
            # covered by a bigger disc already and not a centre worth stamping
            pass
        y0, y1, x0, x1 = max(0, int(y - r)), min(h, int(y + r) + 2), max(0, int(x - r)), min(w, int(x + r) + 2)
        sub = (yy[y0:y1, x0:x1] - y) ** 2 + (xx[y0:y1, x0:x1] - x) ** 2 <= (r + 0.5) ** 2
        sub &= dist[y0:y1, x0:x1] > 0
        view = out[y0:y1, x0:x1]
        view[sub] = np.maximum(view[sub], r)
        done[y0:y1, x0:x1] |= sub
    return out


def components(mask: np.ndarray, conn8: bool = False) -> tuple[np.ndarray, int]:
    """Connected areas of ``mask``: (labels (h, w) with 1..n, 0 outside; n)."""
    h, w = mask.shape
    lab = np.zeros((h, w), np.int32)
    n = 0
    steps = ((1, 0), (-1, 0), (0, 1), (0, -1)) + (((1, 1), (1, -1), (-1, 1), (-1, -1)) if conn8 else ())
    for r0, c0 in zip(*np.nonzero(mask)):
        if lab[r0, c0]:
            continue
        n += 1
        lab[r0, c0] = n
        stack = [(int(r0), int(c0))]
        while stack:
            r, c = stack.pop()
            for dr, dc in steps:
                rr, cc = r + dr, c + dc
                if 0 <= rr < h and 0 <= cc < w and mask[rr, cc] and not lab[rr, cc]:
                    lab[rr, cc] = n
                    stack.append((rr, cc))
    return lab, n


def grow_labels(lab: np.ndarray, into: np.ndarray, passes: int = 64) -> np.ndarray:
    """Labels spread into the unlabelled pixels of ``into`` (nearest label first, 4-neighbours, then diagonals)."""
    out = lab.copy()
    todo = into & (out == 0)
    for _ in range(passes):
        if not todo.any():
            break
        grown = out.copy()
        for dy, dx in ((1, 0), (-1, 0), (0, 1), (0, -1), (1, 1), (1, -1), (-1, 1), (-1, -1)):
            sh = np.zeros_like(out)
            ys = slice(max(0, dy), out.shape[0] + min(0, dy))
            xs = slice(max(0, dx), out.shape[1] + min(0, dx))
            ys2 = slice(max(0, -dy), out.shape[0] + min(0, -dy))
            xs2 = slice(max(0, -dx), out.shape[1] + min(0, -dx))
            sh[ys, xs] = out[ys2, xs2]
            take = todo & (grown == 0) & (sh > 0)
            grown[take] = sh[take]
        if (grown == out).all():
            break
        out = grown
        todo &= out == 0
    return out


def _hex(rgb) -> str:
    return "#%02x%02x%02x" % tuple(int(v) for v in rgb[:3])


def _rgb(c: str) -> tuple[int, int, int]:
    return int(c[1:3], 16), int(c[3:5], 16), int(c[5:7], 16)


def _luma(rgb) -> float:
    return 0.299 * float(rgb[0]) + 0.587 * float(rgb[1]) + 0.114 * float(rgb[2])


def polygon_mask(shape: tuple[int, int], pts: list) -> np.ndarray:
    """Pixels (row, col) whose centre is inside the polygon ``pts`` = [[col, row], ...] (edges included, roughly)."""
    h, w = shape
    rr, cc = np.mgrid[0:h, 0:w]
    inside = np.zeros((h, w), bool)
    n = len(pts)
    j = n - 1
    for i in range(n):
        ci, ri = float(pts[i][0]), float(pts[i][1])
        cj, rj = float(pts[j][0]), float(pts[j][1])
        cond = ((ri > rr) != (rj > rr)) & (cc < (cj - ci) * (rr - ri) / ((rj - ri) or 1e-9) + ci)
        inside ^= cond
        j = i
    # the vertices and edges themselves count
    for i in range(n):
        ci, ri = pts[i]
        cj, rj = pts[(i + 1) % n]
        steps = int(max(abs(cj - ci), abs(rj - ri))) * 2 + 1
        for t in np.linspace(0, 1, steps):
            r, c = int(round(ri + (rj - ri) * t)), int(round(ci + (cj - ci) * t))
            if 0 <= r < h and 0 <= c < w:
                inside[r, c] = True
    return inside


# ---------------------------------------------------------------------------------------------- the side sprite
KEYS = "abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ"
CELL_KEYS = "ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz0123456789"


@dataclass
class Profile:
    """The side sprite read for building: colours, ink (outline and detail lines) and cells (areas between lines)."""
    rgba: np.ndarray                    # (h, w, 4) facing right
    mask: np.ndarray                    # (h, w) bool
    palette: list[str]                  # the sprite's colours, most used first
    idx: np.ndarray                     # (h, w) palette index 1-based, 0 empty
    ink: np.ndarray                     # (h, w) outline / detail-line pixels
    rim: np.ndarray                     # (h, w) ink pixels on the outer edge of the figure
    cells: np.ndarray                   # (h, w) cell id per pixel (ink pixels belong to the nearest cell), 0 empty
    cell_core: np.ndarray               # (h, w) cell id for non-ink pixels only
    info: list[dict[str, Any]] = field(default_factory=list)   # one per cell: id, key, colour, area, box, centre
    flipped: bool = False
    letters: list[str] = field(default_factory=list)           # the colours that fill (not ink), in palette order:
                                                               # letter a is letters[0], as the pixel maps show them

    @property
    def shape(self) -> tuple[int, int]:
        return self.mask.shape


def ink_colours(rgba: np.ndarray) -> list[tuple[int, int, int]]:
    """The colours a sprite uses as ink (its outline and inner lines): dark, and found on the figure's edge far more
    than their share (pixel artists outline each part with a darker tone of its colour, or with black)."""
    mask = rgba[..., 3] >= 128
    if not mask.any():
        return []
    rgb = rgba[..., :3].astype(np.int32)
    edge = mask & ~(np.pad(mask, 1)[:-2, 1:-1] & np.pad(mask, 1)[2:, 1:-1] & np.pad(mask, 1)[1:-1, :-2] & np.pad(mask, 1)[1:-1, 2:])
    cols, inv = np.unique(rgb[mask].reshape(-1, 3), axis=0, return_inverse=True)
    total = np.bincount(inv, minlength=len(cols)).astype(np.float64)
    on_edge = np.bincount(inv[edge[mask]], minlength=len(cols)).astype(np.float64)
    lum = np.array([_luma(c) for c in cols])
    area_lum = float(np.average(lum, weights=total))
    bright = float(np.max(lum[total >= max(3, 0.02 * total.sum())])) if len(cols) else 255.0
    edge_share = float(edge.sum()) / float(mask.sum())
    out = []
    for c, n, e, l in zip(cols, total, on_edge, lum):
        if l <= 40:                                              # black or nearly: always ink
            out.append(tuple(int(v) for v in c))
            continue
        dark = l < 0.62 * bright and l < area_lum
        on_rim = (e / n) >= max(0.3, 1.6 * edge_share)
        if dark and on_rim:
            out.append(tuple(int(v) for v in c))
    return out


def read_profile(rgba: np.ndarray, facing: str = "right", min_cell: int = 3) -> Profile:
    """``rgba``: the side sprite (h, w, 4), background removed and cropped. ``facing``: where its head points."""
    rgba = np.asarray(rgba).copy()
    flipped = facing == "left"
    if flipped:
        rgba = rgba[:, ::-1].copy()
    mask = rgba[..., 3] >= 128
    if not mask.any():
        raise PlanError("the side sprite is empty")
    h, w = mask.shape
    rgb = rgba[..., :3].astype(np.int32)
    cols, inv, counts = np.unique(rgb[mask].reshape(-1, 3), axis=0, return_inverse=True, return_counts=True)
    order = np.argsort(-counts, kind="stable")
    rank = np.empty(len(cols), np.int64)
    rank[order] = np.arange(len(cols))
    palette = [_hex(cols[i]) for i in order]
    idx = np.zeros((h, w), np.int64)
    idx[mask] = rank[inv] + 1
    inks = {tuple(c) for c in ink_colours(rgba)}
    ink_idx = [i + 1 for i, c in enumerate(palette) if _rgb(c) in inks]
    ink = mask & np.isin(idx, ink_idx)
    pm = np.pad(mask, 1)
    outer = mask & ~(pm[:-2, 1:-1] & pm[2:, 1:-1] & pm[1:-1, :-2] & pm[1:-1, 2:])
    rim = ink & outer
    # a cell is a patch of one colour between lines: the smallest piece a part can be made of
    core = np.zeros((h, w), np.int32)
    n = 0
    for i in range(1, len(palette) + 1):
        if i in ink_idx:
            continue
        lab, k = components(mask & (idx == i))
        core[lab > 0] = lab[lab > 0] + n
        n += k
    cells = grow_labels(core, mask)
    if not (cells > 0).any():                                    # a sprite drawn only with ink: one cell
        cells = mask.astype(np.int32)
    # renumber by size (biggest first), so the cells worth naming get the first keys
    ids = [i for i in np.unique(cells) if i > 0]
    area = {i: int((core == i).sum()) for i in ids}
    firsts = sorted(ids, key=lambda i: (-area[i], int(np.argmax((cells == i).ravel()))))
    remap = np.zeros(int(cells.max()) + 1, np.int32)
    for new, old in enumerate(firsts, start=1):
        remap[old] = new
    cells = remap[cells]
    cell_core = np.where(mask & ~ink, cells, 0)
    info = []
    for i in range(1, len(firsts) + 1):
        m = cell_core == i
        if not m.any():
            m = cells == i
        rs, cs = np.nonzero(m)
        main = int(np.bincount(idx[m]).argmax())
        info.append({"id": i, "key": CELL_KEYS[i - 1] if i <= len(CELL_KEYS) and int(m.sum()) >= min_cell else "",
                     "colour": palette[main - 1], "colour_index": main,
                     "area": int(m.sum()), "box": [int(cs.min()), int(rs.min()), int(cs.max()), int(rs.max())],
                     "centre": [round(float(cs.mean()), 1), round(float(rs.mean()), 1)]})
    letters = [c for i, c in enumerate(palette) if (i + 1) not in ink_idx]
    return Profile(rgba, mask, palette, idx, ink, rim, cells, cell_core, info, flipped, letters)


def cell_map(p: Profile) -> str:
    """The cells as text: one character per pixel (the cell's key; '.' empty), with row numbers and a column ruler."""
    h, w = p.shape
    lines = ["     " + "".join(str((c // 10) % 10) if c % 10 == 0 else " " for c in range(w))]
    for r in range(h):
        lines.append(f"{r:4d} " + "".join("." if p.cells[r, c] == 0 else (p.info[p.cells[r, c] - 1]["key"] or "+") for c in range(w)))
    return "\n".join(lines)


def colour_map(p: Profile) -> str:
    """The sprite as text: '.' empty, '#' ink, letters = the colours that fill (a = the most used)."""
    return text_map(p.rgba, p.letters, {_rgb(c) for i, c in enumerate(p.palette) if c not in p.letters})


# ---------------------------------------------------------------------------------------------- the plan
PAINTS = ("auto", "none", "front", "back", "both")
FACE_IOU = 0.82       # a front or back sprite this close to the model's own silhouette is a picture of the same pose

DEFAULTS: dict[str, dict[str, Any]] = {
    "solid": {"aspect": 1.0, "width": None, "paint": "auto"},
    "fin": {"thickness": 2},
    "fan": {"base": None, "spread": 0.5, "width": None, "root": None, "towards": None, "rim": None, "paint": "auto"},
    "pair": {"offset": None, "aspect": 1.0, "width": None, "thickness": None, "lean": 0.0, "root": None, "side": "both", "far": "auto"},
    "wings": {"offset": None, "out_up": 0.5, "out_back": 1.0, "span": None, "thickness": 2, "root": None, "side": "both", "curl": 0.0, "far": "auto"},
    "decal": {"side": "both"},
    "skip": {},
}

def normalise_plan(plan: dict[str, Any]) -> dict[str, Any]:
    """Checks a plan and fills the defaults in. Raises :class:`PlanError` with what is wrong."""
    if not isinstance(plan, dict) or not isinstance(plan.get("parts"), list) or not plan["parts"]:
        raise PlanError('the plan needs "parts": a list with at least one part')
    out: dict[str, Any] = {"facing": "left" if str(plan.get("facing") or "right").lower().startswith("l") else "right",
                           "parts": []}
    for k in ("outline", "ink", "camera", "notes"):
        if k in plan:
            out[k] = plan[k]
    if plan.get("eye") is not None:
        try:
            out["eye"] = [float(plan["eye"][0]), float(plan["eye"][1])]
        except (TypeError, ValueError, IndexError):
            raise PlanError('"eye" must be [column, row] of the side sprite') from None
    names = set()
    for i, raw in enumerate(plan["parts"]):
        if not isinstance(raw, dict):
            raise PlanError(f"part {i + 1} is not an object")
        name = str(raw.get("name") or f"part{i + 1}").strip()[:40]
        if name in names:
            name = f"{name}_{i + 1}"
        names.add(name)
        kind = str(raw.get("type") or "solid").lower().strip()
        kind = {"body": "solid", "round": "solid", "blob": "solid", "slab": "fin", "crest": "fin", "wing": "wings", "sheet": "wings",
                "sheets": "wings", "legs": "pair", "pairs": "pair", "eye": "decal", "mark": "decal", "tail_fan": "fan", "far": "skip", "ghost": "skip",
                "omit": "skip", "none": "skip", "hidden": "skip"}.get(kind, kind)
        if kind not in TYPES:
            raise PlanError(f"part '{name}': unknown type {raw.get('type')!r} (use one of {', '.join(TYPES)})")
        part: dict[str, Any] = {"name": name, "type": kind}
        for key in ("where", "cells", "colors", "colours", "except"):
            if raw.get(key) is not None:
                part["colors" if key == "colours" else key] = raw[key]
        for key, default in DEFAULTS[kind].items():
            val = raw.get(key, default)
            if key in ("root", "towards") and val is not None:
                try:
                    val = [float(val[0]), float(val[1])]
                except (TypeError, ValueError, IndexError):
                    raise PlanError(f"part '{name}': {key} must be [column, row] of the side sprite") from None
            elif key == "paint":
                val = str(val or "auto").lower()
                if val not in PAINTS:
                    raise PlanError(f"part '{name}': paint must be one of {', '.join(PAINTS)}")
            elif key == "far":
                val = str(val or "auto").lower()
                if val not in ("auto", "keep"):
                    raise PlanError(f"part '{name}': far must be auto or keep")
            elif key == "side":
                val = str(val or "both").lower()
                if val not in ("both", "near", "far"):
                    raise PlanError(f"part '{name}': side must be both, near or far")
            elif val is not None:
                try:
                    val = float(val)
                except (TypeError, ValueError):
                    raise PlanError(f"part '{name}': {key} must be a number") from None
            part[key] = val
        if raw.get("fixed"):
            part["fixed"] = list(raw["fixed"]) if isinstance(raw["fixed"], (list, tuple)) else True
        out["parts"].append(part)
    return out


def _colour_set(p: Profile, spec: Any) -> Optional[np.ndarray]:
    """Pixels of the colours in ``spec``: palette letters ("bc"), "#rrggbb" values or a list of either; a colour that
    is not in the sprite takes the nearest one. None when ``spec`` is empty."""
    if not spec:
        return None
    items = list(spec) if isinstance(spec, str) and not spec.startswith("#") else (list(spec) if isinstance(spec, (list, tuple)) else [spec])
    want: set[int] = set()
    pal = np.array([_rgb(c) for c in p.palette], np.int32)
    for it in items:
        it = str(it).strip()
        if not it:
            continue
        if it.startswith("#") and len(it) == 7:
            try:
                rgb = np.array(_rgb(it.lower()), np.int32)
            except ValueError:
                continue
            want.add(int(((pal - rgb) ** 2).sum(axis=1).argmin()) + 1)
        elif len(it) == 1 and it in KEYS:
            k = KEYS.index(it)
            if k < len(p.letters):
                want.add(p.palette.index(p.letters[k]) + 1)
    if not want:
        return None
    return np.isin(p.idx, sorted(want))


def _where_mask(p: Profile, part: dict[str, Any], key: str = "where", precise: bool = False,
                geometry: bool = False) -> Optional[np.ndarray]:
    """The pixels a part asks for with its shapes, or None when it gives none. A shape is a rectangle
    [col0, row0, col1, row1], a polygon [[col, row], ...] or {"rect"|"poly": ..., "colors": ...} to take only some
    colours inside it. Columns and rows are those of the side sprite as given (before any flip). ``precise``: only the
    deliberate shapes (polygons, shapes with a colour filter), not the bare rectangles. ``geometry``: the shapes as
    drawn, whatever their colour filters."""
    h, w = p.shape
    shapes = part.get(key)
    if shapes is None:
        return None
    if isinstance(shapes, dict) or (isinstance(shapes, (list, tuple)) and shapes and isinstance(shapes[0], (int, float))):
        shapes = [shapes]
    out = np.zeros((h, w), bool)
    default_cf = _colour_set(p, part.get("colors")) if key == "where" else None
    for s in shapes:
        cf = default_cf
        if isinstance(s, dict):
            if s.get("colors") or s.get("colours"):
                cf = _colour_set(p, s.get("colors") or s.get("colours"))
            s = s.get("rect") or s.get("poly") or s.get("polygon") or []
        if not isinstance(s, (list, tuple)) or not s:
            continue
        m = np.zeros((h, w), bool)
        if isinstance(s[0], (int, float)):
            if len(s) != 4:
                raise PlanError(f"part '{part['name']}': a rectangle is [col0, row0, col1, row1]")
            c0, r0, c1, r1 = (int(round(float(v))) for v in s)
            c0, c1, r0, r1 = min(c0, c1), max(c0, c1), min(r0, r1), max(r0, r1)
            if p.flipped:
                c0, c1 = w - 1 - c1, w - 1 - c0
            m[max(0, r0): r1 + 1, max(0, c0): c1 + 1] = True
        else:
            if len(s) < 3:
                raise PlanError(f"part '{part['name']}': a polygon needs at least three [col, row] points")
            pts = [[(w - 1 - float(a)) if p.flipped else float(a), float(b)] for a, b in s]
            m = polygon_mask((h, w), pts)
        if precise and cf is None and isinstance(s[0], (int, float)):
            continue                 # a bare rectangle is a rough shape; polygons and colour filters are deliberate
        if cf is not None and not geometry:
            m &= cf | p.ink          # lines inside the shape come along with the colours they separate
        out |= m
    return out


def assign(p: Profile, plan: dict[str, Any]) -> np.ndarray:
    """(h, w) part number per pixel (1-based, 0 empty): every pixel of the sprite ends in exactly one part.

    A part takes the cells it names (``cells``: keys or ids) and the pixels inside its shapes (``where``), optionally
    only of some colours (``colors``). Shapes are rough: a cell that is one part's but for a sliver goes to it whole
    (colour changes and lines are the real borders); a cell that several shapes really share is split along them. Where
    shapes overlap, the one lying mostly inside the other wins; shapes that overlap only at their borders leave each
    cell with the part that has the rest of it, and in a cell both really share a polygon or a colour filter beats a bare
    rectangle and two rectangles meet halfway. A
    decal is the exception: it is exactly the pixels of its shapes. What nobody asks for joins the neighbouring part
    (or the first solid)."""
    h, w = p.shape
    parts = plan["parts"]
    label = np.zeros((h, w), np.int32)
    ncell = int(p.cells.max())
    want = np.zeros((len(parts), h, w), bool)
    exact = np.zeros((len(parts), h, w), bool)        # asked for with a deliberate shape, not a rough rectangle
    key_to_id = {c["key"]: c["id"] for c in p.info if c.get("key")}
    for k, part in enumerate(parts):
        spec = part.get("cells")
        if spec is not None:
            for it in (list(spec) if isinstance(spec, (str, list, tuple)) else [spec]):
                i = int(it) if isinstance(it, (int, np.integer)) or (isinstance(it, str) and it.isdigit()) else key_to_id.get(str(it), 0)
                if 1 <= i <= ncell:
                    want[k] |= p.cells == i
                    exact[k] |= p.cells == i
        m = _where_mask(p, part)
        if m is not None:
            want[k] |= m & p.mask
            fine = _where_mask(p, part, precise=True)
            if fine is not None:
                exact[k] |= fine & p.mask
        elif part.get("colors") and spec is None:
            cf = _colour_set(p, part["colors"])
            if cf is not None:
                want[k] |= cf & p.mask
        em = _where_mask(p, part, "except")
        if em is not None:
            want[k] &= ~em
    core = p.cell_core
    normal = [k for k, part in enumerate(parts) if part["type"] != "decal"]
    # Shapes are rough and overlap. A shape that lies inside another one is the more specific of the two and
    # wins there (a wing drawn over a body given as one big rectangle); shapes that only overlap at their borders
    # contest those pixels, and each patch of colour settles it.
    geo = {}
    for k in normal:
        g = _where_mask(p, parts[k], geometry=True)
        geo[k] = g if g is not None else want[k]
    area = {k: int(geo[k].sum()) for k in normal}
    inside = {(k, j) for k in normal for j in normal
              if k != j and area[k] and int((geo[k] & geo[j]).sum()) >= 0.85 * area[k]
              and (area[k] < area[j] or (area[k] == area[j] and k > j))}
    owner = np.zeros((h, w), np.int32)            # part + 1 where one part has the pixel, -1 where several contest it
    claims = want[normal].sum(axis=0) if normal else np.zeros((h, w), np.int32)
    for k in normal:
        owner[want[k] & (claims == 1)] = k + 1
    for r, c in zip(*np.nonzero(claims > 1)):
        who = [k for k in normal if want[k, r, c]]
        who = [k for k in who if not any((j, k) in inside for j in who)] or who
        owner[r, c] = who[0] + 1 if len(who) == 1 else -1
    for i in range(1, ncell + 1):
        cm = core == i
        n = int(cm.sum())
        if not n:
            continue
        piece = np.where(cm, owner, 0)
        if not piece.any():
            continue
        sure = np.bincount(piece[piece > 0], minlength=len(parts) + 1)[1:]
        big = max(6, 0.12 * n)
        for r, c in zip(*np.nonzero(piece == -1)):
            who = [k for k in normal if want[k, r, c]]
            who = [k for k in who if not any((j, k) in inside for j in who)] or who
            holders = [k for k in who if sure[k] >= big]
            fine = [k for k in who if exact[k, r, c]]
            # the part that has the rest of the patch keeps this bit too; in a patch they really share, a deliberate
            # shape (a polygon, a colour filter) beats a rough rectangle (the later of two deliberate ones wins), and
            # two rough rectangles meet halfway
            if len(holders) == 1:
                piece[r, c] = holders[0] + 1
            elif fine:
                piece[r, c] = fine[-1] + 1
        if (piece == -1).any():
            seeds = np.where(piece > 0, piece, 0)
            near = grow_labels(seeds, cm & (piece != 0)) if seeds.any() else seeds
            for r, c in zip(*np.nonzero(piece == -1)):
                who = [k for k in normal if want[k, r, c]]
                piece[r, c] = near[r, c] if near[r, c] - 1 in who else who[-1] + 1
        counts = np.bincount(piece[cm], minlength=len(parts) + 1)[1:]
        top = int(counts.argmax())
        others = np.delete(counts, top)
        if counts[top] >= 0.6 * n and (not len(others) or others.max() <= big):
            # a patch that is one part's but for a sliver goes to it whole
            label[cm] = top + 1
        else:
            # a patch that really spans several parts (a wing and a body of the same colour): split along the shapes,
            # what no shape takes goes to the nearest piece
            piece = grow_labels(piece, cm)
            label[cm] = piece[cm]
    # lines: inside one part's shape they are that part's; otherwise they join what is next to them
    only = p.ink & (label == 0) & (owner > 0)
    label[only] = owner[only]
    label = grow_labels(label, p.mask)
    if (p.mask & (label == 0)).any():
        first_solid = next((k for k, part in enumerate(parts) if part["type"] == "solid"), 0)
        label[p.mask & (label == 0)] = first_solid + 1
    for k, part in enumerate(parts):
        if part["type"] == "decal":
            label[want[k]] = k + 1
    return label


def _check_eye(p: Profile, plan: dict[str, Any], parts: list[dict[str, Any]], label: np.ndarray, notes: list[str]) -> None:
    """The eye the plan points at must be in something that can be a head, at the end the character faces. A wing
    read as the head is the costliest mistake a plan can make, and the eye settles it."""
    eye = plan.get("eye")
    if eye is None:
        return
    h, w = p.shape
    c, r = int(round(eye[0])), int(round(eye[1]))
    shown = f"[{c}, {r}]"
    if p.flipped:
        c = w - 1 - c
    rs, cs = np.nonzero(p.mask)
    i = int(np.argmin((rs - r) ** 2 + (cs - c) ** 2))
    if (rs[i] - r) ** 2 + (cs[i] - c) ** 2 > 9:
        raise PlanError(f'the eye at {shown} is not on the character: "eye" is [column, row] of the side sprite')
    part = parts[int(label[rs[i], cs[i]]) - 1]
    if part["type"] in ("wings", "fan", "fin", "skip"):
        raise PlanError(f"the eye at {shown} lies in '{part['name']}', a part of type {part['type']}: the shape with the eye is the "
                        f"HEAD (a solid), not a {part['type']}. Read the side sprite again from the head")
    # the profile is stored facing right: the eye belongs in the front half
    if cs[i] < 0.33 * (w - 1):
        raise PlanError(f"the eye at {shown} is in the back third of a character said to face {plan.get('facing', 'right')}: either it faces "
                        f"the other way or that is not the eye. The head is the shape with the eye, at the end it faces")
    part["eye"] = shown
    # an eye is drawn with something the rest of the part is not: another colour, or a dark spot that is not outline
    cls = families_of(p.rgba).classify(p.rgba)
    mine = (label == label[rs[i], cs[i]]) & ~p.ink
    main = int(np.bincount(cls[mine]).argmax()) if mine.any() else -1
    rr, cc = np.mgrid[0:h, 0:w]
    near = p.mask & ((rr - rs[i]) ** 2 + (cc - cs[i]) ** 2 <= 5)
    lab, _ = components(p.ink, conn8=True)
    outline = np.isin(lab, np.unique(lab[p.rim & p.ink])) & p.ink
    other = near & ~p.ink & (cls != main)
    pupil = near & p.ink & ~outline
    if not (other.any() or pupil.any()):
        notes.append(f"nothing around the eye at {shown} looks like an eye (only the colours of '{part['name']}' and outline): "
                     f"make sure that is the eye and that shape the head")


def _check_stand(p: Profile, parts: list[dict[str, Any]], label: np.ndarray, notes: list[str]) -> None:
    """What the figure stands on. A narrow foot at the very bottom that belongs to a part on the middle line becomes
    ONE central leg; legs are a pair."""
    h, w = p.shape
    rows = np.nonzero(p.mask.any(axis=1))[0]
    if not len(rows):
        return
    foot = p.mask & (np.arange(h)[:, None] >= rows.max() - 1)
    cols = np.nonzero(foot.any(axis=0))[0]
    if len(cols) > 0.35 * w:
        return                                   # it lies on the ground (a snake, a fish, a slime): nothing to say
    owners = np.bincount(label[foot], minlength=len(parts) + 1)[1:]
    k = int(owners.argmax())
    if parts[k]["type"] == "solid":              # a fan or a fin that hangs lowest is a tail, and fine
        notes.append(f"the character stands on '{parts[k]['name']}' (columns {cols.min()}-{cols.max()} at the bottom rows), a part on the "
                     f"middle line: that builds ONE leg in the middle. If those pixels are a leg or a foot, give them (the whole leg, "
                     f"up to the body) to a pair part")


def _check_skips(p: Profile, parts: list[dict[str, Any]], masks: list[np.ndarray], notes: list[str]) -> None:
    """A far copy has the colours of the near one. A skip part of colours that no wings or pair part has is more likely
    a marking of the body that would be lost: say so."""
    skips = [k for k, part in enumerate(parts) if part["type"] == "skip" and not part.get("auto") and part.get("pixels", 0) >= 6]
    if not skips:
        return
    cls = families_of(p.rgba).classify(p.rgba)
    twice: set = set()
    for k, part in enumerate(parts):
        if part["type"] in ("wings", "pair"):
            c = cls[masks[k] & ~p.ink]
            if len(c):
                count = np.bincount(c)
                twice |= {int(i) for i in np.nonzero(count >= 0.15 * len(c))[0]}
    for k in skips:
        c = cls[masks[k] & ~p.ink]
        if not len(c):
            continue
        if int(np.bincount(c).argmax()) not in twice:
            notes.append(f"'{parts[k]['name']}' (skip) has colours that no wings or pair part has: a far wing or limb looks like the near "
                         f"one. If it is a marking of the body (a belly, a stripe) it is being lost: remove that skip part")


def far_copies(p: Profile, label: np.ndarray, parts: list[dict[str, Any]]) -> list[tuple[int, np.ndarray]]:
    """Side sprites let the far wing, arm or leg peek out behind the near one, for looks. Inside a part that exists
    twice (wings, pair) it shows as a second area of the same colours cut off from the main one by a line. Returns
    [(part index, mask of what is taken for the far copy)] for the parts whose ``far`` is auto.

    The near one is the bigger drawing: the area that, with the areas of other colours it carries (a rim, a
    highlight), has most pixels. Another area of its colours with at least 0.3 of its size is the far copy, together
    with the small areas and lines that hang only from it."""
    out: list[tuple[int, np.ndarray]] = []
    cls = None
    for k, part in enumerate(parts):
        if part["type"] not in ("wings", "pair") or str(part.get("far") or "auto") != "auto":
            continue
        m = label == k + 1
        comp, n = components(m & ~p.ink)
        if n < 2:
            continue
        if cls is None:
            cls = families_of(p.rgba).classify(p.rgba)
        sizes = np.bincount(comp.ravel(), minlength=n + 1)
        sizes[0] = 0
        dom = np.zeros(n + 1, np.int64)
        for i in range(1, n + 1):
            c = cls[comp == i]
            c = c[c != Families.INK]
            dom[i] = int(np.bincount(c).argmax()) if len(c) else -1
        near = [None] + [_neigh4(_neigh4(comp == i)) | (comp == i) for i in range(1, n + 1)]
        touch = np.zeros((n + 1, n + 1), bool)
        for i in range(1, n + 1):
            for j in np.unique(comp[near[i]]):
                if j and j != i:
                    touch[i, j] = touch[j, i] = True
        big = [i for i in range(1, n + 1) if sizes[i] >= 6]
        if len(big) < 2:
            continue
        score = {i: sizes[i] + sum(sizes[j] for j in range(1, n + 1) if touch[i, j] and dom[j] != dom[i]) for i in big}
        main = max(big, key=lambda i: (score[i], sizes[i]))
        cands = [i for i in big if i != main and dom[i] == dom[main] and sizes[i] >= max(12, 0.3 * sizes[main])]
        if not cands:
            continue
        kept, far = {main}, set(cands)
        rest = [i for i in range(1, n + 1) if i not in kept and i not in far]
        changed = True
        while changed and rest:
            changed = False
            for i in list(rest):
                if any(touch[i, j] for j in kept):
                    kept.add(i)
                elif any(touch[i, j] for j in far):
                    far.add(i)
                else:
                    continue
                rest.remove(i)
                changed = True
        far_px = np.isin(comp, sorted(far))
        keep_px = (comp > 0) & ~far_px
        p8 = np.pad(keep_px, 1)
        by_kept = (p8[:-2, :-2] | p8[:-2, 1:-1] | p8[:-2, 2:] | p8[1:-1, :-2] | p8[1:-1, 2:] | p8[2:, :-2] | p8[2:, 1:-1] | p8[2:, 2:])
        # the line between the two is the near one's outline; lines that touch only the far copy go with it
        lines = m & p.ink & ~by_kept
        grown = grow_labels(np.where(far_px, 1, np.where(keep_px, 2, 0)).astype(np.int32), far_px | keep_px | lines)
        out.append((k, far_px | (lines & (grown == 1))))
    return out


# ---------------------------------------------------------------------------------------------- building
@dataclass
class Built:
    grid: np.ndarray                    # (X, Y, Z) uint8 palette indices
    palette: list[str]
    label: np.ndarray                   # (h, w) part number per sprite pixel
    place: dict[str, int]               # z0, y0, cx (x of the middle plane: voxels cx-1 | cx), w, h
    parts: list[dict[str, Any]]         # the plan's parts with what was computed (width, offset, root...)
    notes: list[str] = field(default_factory=list)
    pid: Optional[np.ndarray] = None    # (X, Y, Z) part number of each voxel (1-based)


def _inpaint(idx: np.ndarray, known: np.ndarray, region: np.ndarray) -> np.ndarray:
    """``idx`` with every pixel of ``region`` that is not ``known`` taking the value of the nearest known pixel."""
    lab = np.where(known & region, idx, 0).astype(np.int32)
    return grow_labels(lab, region)


def _ortho_fill(region: np.ndarray, allowed: np.ndarray) -> np.ndarray:
    """``region`` plus the ``allowed`` pixels it encloses: those with region pixels on both sides along their row AND
    along their column (the part of a body hidden behind a wing drawn over it)."""
    out = region.copy()
    for _ in range(3):
        left = np.maximum.accumulate(out, axis=1)
        right = np.maximum.accumulate(out[:, ::-1], axis=1)[:, ::-1]
        up = np.maximum.accumulate(out, axis=0)
        down = np.maximum.accumulate(out[::-1], axis=0)[::-1]
        add = allowed & ~out & left & right & up & down
        if not add.any():
            break
        out |= add
    return out


def _auto_root(part_mask: np.ndarray, core: np.ndarray) -> tuple[float, float]:
    """Where a limb joins the rest: the centre of its pixels that touch ``core`` (the solids), else its lowest point
    nearest the core's centre."""
    touch = part_mask & _neigh4(_neigh4(core & ~part_mask) | (core & ~part_mask))
    if touch.any():
        rs, cs = np.nonzero(touch)
        return float(cs.mean()), float(rs.mean())
    rs, cs = np.nonzero(part_mask)
    if core.any():
        cr, cc = np.nonzero(core)
        d = (rs - cr.mean()) ** 2 + (cs - cc.mean()) ** 2
        k = int(np.argmin(d))
        return float(cs[k]), float(rs[k])
    return float(cs.mean()), float(rs.max())


def build(p: Profile, plan: dict[str, Any], margin: int = 2, outline: Optional[str] = None,
          label: Optional[np.ndarray] = None) -> Built:
    """The voxel model of a plan. ``outline``: what becomes of the sprite's dark outline on the figure's edge: ``drop``
    (the default: it takes the colour of the part next to it, as voxel artists do), ``keep`` or ``shade`` (a darker
    tone of the part)."""
    plan = normalise_plan(plan)
    outline = outline or str(plan.get("outline") or "drop")
    h, w = p.shape
    parts = [dict(pt) for pt in plan["parts"]]
    notes: list[str] = []
    if label is None:
        label = assign(p, plan)
        _check_eye(p, plan, parts, label, notes)
        for k, mask in far_copies(p, label, parts):
            rs, cs = np.nonzero(mask)
            parts.append({"name": f"{parts[k]['name']} (far)", "type": "skip", "auto": parts[k]["name"]})
            label = label.copy()
            label[mask] = len(parts)
            notes.append(f"'{parts[k]['name']}': the area at columns {cs.min()}-{cs.max()}, rows {rs.min()}-{rs.max()} ({len(rs)} px) has the "
                         f"same colours as the rest of it behind a line of its own: taken for the FAR one drawn behind the near one, "
                         f"and not built (the mirrored near one is it).")
    masks = [label == k + 1 for k in range(len(parts))]
    for k, part in enumerate(parts):
        part["pixels"] = int(masks[k].sum())
        if not part["pixels"]:
            notes.append(f"part '{part['name']}' got no pixels: check its shapes, cells and colours")
    _check_skips(p, parts, masks, notes)
    _check_stand(p, parts, label, notes)
    lateral_px = np.zeros((h, w), bool)
    for k, part in enumerate(parts):
        if part["type"] in LATERAL:
            lateral_px |= masks[k]
    # ---- what each median part covers once the limbs drawn over it are taken away
    solid_ids = [k for k, part in enumerate(parts) if part["type"] == "solid" and part["pixels"]]
    median_ids = [k for k, part in enumerate(parts) if part["type"] in MEDIAN and part["pixels"]]
    full = [m.copy() for m in masks]
    if median_ids:
        med_union = np.zeros((h, w), bool)
        for k in median_ids:
            med_union |= masks[k]
        filled = _ortho_fill(med_union, lateral_px)
        hidden = filled & ~med_union
        if hidden.any():
            # each hidden pixel joins the nearest median part
            lab = np.zeros((h, w), np.int32)
            for k in median_ids:
                lab[masks[k]] = k + 1
            lab = grow_labels(lab, filled)
            for k in median_ids:
                full[k] = lab == k + 1
    # decals lie on whatever is under them: the nearest median part (or pair) takes their pixels too
    for k, part in enumerate(parts):
        if part["type"] == "decal" and part["pixels"]:
            lab = np.zeros((h, w), np.int32)
            for j, other in enumerate(parts):
                if other["type"] not in ("decal", "skip") and other["pixels"]:
                    lab[full[j]] = j + 1
            if not lab.any():
                continue
            lab = grow_labels(lab, lab.astype(bool) | masks[k])
            for j in np.unique(lab[masks[k]]):
                if j > 0:
                    full[j - 1] = full[j - 1] | (masks[k] & (lab == j))
            part["on"] = [parts[j - 1]["name"] for j in np.unique(lab[masks[k]]) if j > 0]
    solid_union = np.zeros((h, w), bool)
    for k in solid_ids:
        solid_union |= full[k]
    # ---- the solids as one rounded volume
    half = np.zeros((len(parts), h, w), np.float64)        # half-width (voxels) of each part at each pixel
    centre = np.zeros((len(parts), h, w), np.float64)      # distance of the part's middle to the middle plane
    if solid_ids:
        dist = edt_inside(solid_union)
        rad = local_radius(dist)
        base = np.sqrt(np.clip(dist * (2 * rad - dist), 0, None))
        asp = np.zeros((h, w), np.float64)
        for k in solid_ids:
            part = parts[k]
            a = float(part.get("aspect") or 1.0)
            if part.get("width"):
                peak = float(base[full[k]].max()) if full[k].any() else 1.0
                a = float(part["width"]) / 2.0 / max(0.5, peak)
            part["aspect_used"] = round(a, 3)
            asp[full[k]] = a
        # blend the aspect across the joins so a neck flows into a body
        wsum = solid_union.astype(np.float64)
        acc = asp.copy()
        for _ in range(2):
            pa, pw = np.pad(acc, 1), np.pad(wsum, 1)
            acc = sum(pa[1 + dy: 1 + dy + h, 1 + dx: 1 + dx + w] for dy in (-1, 0, 1) for dx in (-1, 0, 1))
            wsum2 = sum(pw[1 + dy: 1 + dy + h, 1 + dx: 1 + dx + w] for dy in (-1, 0, 1) for dx in (-1, 0, 1))
            acc = np.where(solid_union, acc / np.maximum(wsum2, 1e-9), 0)
            wsum = solid_union.astype(np.float64)
        for k in solid_ids:
            hk = np.where(full[k], acc * base, 0)
            if parts[k].get("width") and float(hk.max()) > 0.5:
                # the blend with thinner or fatter neighbours moved the part off the width asked for: bring it back
                hk = hk * (float(parts[k]["width"]) / 2.0) / float(hk.max())
            half[k] = np.where(full[k], np.maximum(1.0, hk), 0)
            parts[k]["width_built"] = int(2 * round(float(half[k].max())))
    body_half = half[solid_ids].max(axis=0) if solid_ids else np.zeros((h, w))

    def body_half_at(c: float, r: float, reach: int = 2) -> float:
        r0, r1 = max(0, int(round(r)) - reach), min(h, int(round(r)) + reach + 1)
        c0, c1 = max(0, int(round(c)) - reach), min(w, int(round(c)) + reach + 1)
        sub = body_half[r0:r1, c0:c1]
        return float(sub.max()) if sub.size and sub.max() > 0 else 0.0

    rr, cc = np.mgrid[0:h, 0:w].astype(np.float64)
    for k, part in enumerate(parts):
        if not part["pixels"]:
            continue
        kind, m = part["type"], full[k]
        if kind == "fin":
            t = max(2.0, float(part.get("thickness") or 2))
            half[k] = np.where(m, t / 2.0, 0)
        elif kind == "fan":
            root = part.get("root")
            if root is None:
                root = _auto_root(m, solid_union)
            elif p.flipped:
                root = [w - 1 - root[0], root[1]]
            tw = part.get("towards")
            if tw is not None and p.flipped:
                tw = [w - 1 - tw[0], tw[1]]
            rs, cs = np.nonzero(m)
            if tw is None:
                far = int(np.argmax((cs - root[0]) ** 2 + (rs - root[1]) ** 2))
                # towards the far end, steadied by the centre of the part
                tw = [0.5 * (cs[far] + cs.mean()), 0.5 * (rs[far] + rs.mean())]
            d = np.array([tw[0] - root[0], tw[1] - root[1]], np.float64)
            d /= max(1e-6, float(np.hypot(*d)))
            s = np.clip((cc - root[0]) * d[0] + (rr - root[1]) * d[1], 0, None)
            base_w = part.get("base")
            if base_w is None:
                base_w = max(2.0, 2.0 * body_half_at(root[0], root[1]) * 0.8)
            spread = float(part.get("spread") or 0.0)
            if part.get("width") and float(s[m].max()) > 1e-6:
                spread = max(0.0, (float(part["width"]) - float(base_w)) / 2.0) / float(s[m].max())
                part["spread_used"] = round(spread, 3)
            lin = float(base_w) / 2.0 + spread * s
            dd = edt_inside(m)
            # rounded towards its upper and lower edges, so that its end is a fan and not a crate
            rim = max(0.5, float(part.get("rim") or 0.6 * float(dd.max())))
            rho = np.sqrt(np.clip(1 - (1 - np.minimum(1.0, (dd + 0.5) / rim)) ** 2, 0, 1))
            hk = lin * rho
            if part.get("width") and float(hk[m].max()) > 0.5:
                hk = hk * (float(part["width"]) / 2.0) / float(hk[m].max())
            half[k] = np.where(m, np.maximum(1.0, hk), 0)
            part["root_used"], part["base_used"] = [round(float(root[0]), 1), round(float(root[1]), 1)], round(float(base_w), 1)
            part["width_built"] = int(2 * round(float(half[k].max())))
        elif kind == "pair":
            root = part.get("root")
            if root is None:
                root = _auto_root(m, solid_union)
            elif p.flipped:
                root = [w - 1 - root[0], root[1]]
            dd = edt_inside(m)
            if part.get("thickness"):
                hw = np.where(m, max(1.0, float(part["thickness"]) / 2.0), 0)
            else:
                rad = local_radius(dd)
                a = float(part.get("aspect") or 1.0)
                b = np.sqrt(np.clip(dd * (2 * rad - dd), 0, None))
                if part.get("width"):
                    a = float(part["width"]) / 2.0 / max(0.5, float(b[m].max()))
                hw = np.where(m, np.maximum(1.0, a * b), 0)
            off = part.get("offset")
            if off is None:
                # sit on the flank of the body where the limb joins, half sunk in
                off = max(float(hw.max()), body_half_at(root[0], root[1]) - 0.3 * float(hw.max()))
            lean = float(part.get("lean") or 0.0)
            dist_root = np.hypot(cc - root[0], rr - root[1])
            centre[k] = np.where(m, np.maximum(hw, float(off) + lean * dist_root), 0)
            half[k] = hw
            part["root_used"], part["offset_used"] = [round(float(root[0]), 1), round(float(root[1]), 1)], round(float(off), 1)
        elif kind == "wings":
            root = part.get("root")
            if root is None:
                root = _auto_root(m, solid_union)
            elif p.flipped:
                root = [w - 1 - root[0], root[1]]
            t = max(2.0, float(part.get("thickness") or 2))
            off = part.get("offset")
            if off is None:
                off = max(t / 2.0, body_half_at(root[0], root[1]) - t / 2.0)
            up, back = float(part.get("out_up") or 0.0), float(part.get("out_back") or 0.0)
            # a flat sheet hinged at the root: what lies before the hinge stays against the flank
            gain = np.maximum(0.0, up * (root[1] - rr) + back * (root[0] - cc))
            if part.get("span") and float(gain[m].max()) > 1e-6:
                # the wingspan is given: keep the direction the sheet opens in, scale how fast
                grow = max(0.0, (float(part["span"]) / 2.0 - float(off) - t / 2.0)) / float(gain[m].max())
                up, back, gain = up * grow, back * grow, gain * grow
                part["out_up_used"], part["out_back_used"] = round(up, 3), round(back, 3)
            c_line = float(off) + gain
            curl = float(part.get("curl") or 0.0)
            if curl:
                reach = np.hypot(cc - root[0], rr - root[1])
                c_line = c_line + curl * (reach / max(1.0, float(reach[m].max()))) ** 2 * float(reach[m].max())
            c_line = np.maximum(c_line, t / 2.0)
            # a tilted sheet steps sideways between neighbouring pixels: thicken it by that step so it has no holes
            step = max(abs(up), abs(back))
            hw = t / 2.0 + max(0.0, step - 1.0) / 2.0
            centre[k] = np.where(m, c_line, 0)
            half[k] = np.where(m, hw, 0)
            part["root_used"], part["offset_used"] = [round(float(root[0]), 1), round(float(root[1]), 1)], round(float(off), 1)
            part["span_built"] = int(round(2 * float((c_line[m] + hw).max())))
    # ---- colours
    palette = list(p.palette)
    base_idx = np.zeros((len(parts), h, w), np.int64)      # the part's own colour under lines, decals and what hides it
    decal_px = np.zeros((h, w), bool)
    for k, part in enumerate(parts):
        if part["type"] == "decal":
            decal_px |= masks[k]

    def shade_of(index: int, factor: float) -> int:
        r, g, b = _rgb(palette[index - 1])
        hexc = "#%02x%02x%02x" % (int(r * factor), int(g * factor), min(255, int(b * factor * 1.06)))
        if hexc not in palette:
            if len(palette) >= 255:
                return index
            palette.append(hexc)
        return palette.index(hexc) + 1

    for k, part in enumerate(parts):
        if not part["pixels"] or part["type"] in ("decal", "skip"):
            continue
        known = masks[k] & ~p.ink & ~decal_px
        if not known.any():
            known = masks[k] & ~decal_px
        if not known.any():
            known = masks[k]
        base_idx[k] = _inpaint(p.idx, known, full[k])
    # what the surface shows at each sprite pixel: the sprite itself, with its ink toned down for 3-D
    owner = np.zeros((h, w), np.int32)                     # the part (not a decal) each visible pixel lies on
    skipped = np.zeros((h, w), bool)
    for k, part in enumerate(parts):
        if part["type"] == "skip":
            skipped |= masks[k]
    for k, part in enumerate(parts):
        if part["pixels"] and part["type"] not in ("decal", "skip"):
            owner[masks[k]] = k + 1
    for k, part in enumerate(parts):
        if part["pixels"] and part["type"] == "decal":
            for j, other in enumerate(parts):
                if other["type"] not in ("decal", "skip") and other["pixels"]:
                    owner[masks[k] & full[j] & (owner == 0)] = j + 1
    owner = grow_labels(owner, p.mask & ~skipped)
    surface = p.idx.copy()
    ink_mode = str(plan.get("ink") or "soften")
    # a decal keeps its own dark pixels (a pupil), but the line that outlines it is ink like any other
    decal_edge = decal_px & p.ink & _neigh4(p.mask & ~decal_px)
    rs, cs = np.nonzero(p.ink & (~decal_px | decal_edge))
    for r, c in zip(rs, cs):
        k = owner[r, c] - 1
        if k < 0 or not base_idx[k][r, c]:
            continue
        b = int(base_idx[k][r, c])
        if p.rim[r, c]:
            surface[r, c] = int(p.idx[r, c]) if outline == "keep" else (shade_of(b, 0.72) if outline == "shade" else b)
        elif ink_mode == "soften":
            surface[r, c] = shade_of(b, 0.62)
        elif ink_mode == "drop":
            surface[r, c] = b
    # ---- the grid
    reach = 1.0
    for k, part in enumerate(parts):
        if part["pixels"]:
            reach = max(reach, float((centre[k] + half[k]).max()))
    hx = int(math.ceil(reach)) + margin
    X = min(MAX_SIDE, 2 * hx)
    X += X % 2
    Y = min(MAX_SIDE, h + 2 * margin)
    Z = min(MAX_SIDE, w + 2 * margin)
    cx, z0, y0 = X // 2, (Z - w) // 2, margin
    pid = np.zeros((X, Y, Z), np.uint8)                     # which part each voxel belongs to (1-based)
    yy = (y0 + (h - 1 - np.arange(h)))                      # grid y of each sprite row
    zz = z0 + np.arange(w)

    def put(lo: np.ndarray, hi: np.ndarray, m: np.ndarray, k: int):
        """Part ``k`` fills x in [lo, hi] (inclusive, per pixel) where ``m``."""
        rs, cs = np.nonzero(m)
        if not len(rs):
            return
        lo_i = np.clip(lo[rs, cs], 0, X - 1).astype(np.int64)
        hi_i = np.clip(hi[rs, cs], 0, X - 1).astype(np.int64)
        gy, gz = yy[rs], zz[cs]
        ok = (gy >= 0) & (gy < Y) & (gz >= 0) & (gz < Z) & (hi_i >= lo_i)
        lo_i, hi_i, gy, gz = lo_i[ok], hi_i[ok], gy[ok], gz[ok]
        if not len(lo_i):
            return
        for x in range(int(lo_i.min()), int(hi_i.max()) + 1):
            sel = (lo_i <= x) & (x <= hi_i)
            if sel.any():
                pid[x, gy[sel], gz[sel]] = k + 1

    order = ([k for k in median_ids if parts[k]["type"] == "solid"] + [k for k in median_ids if parts[k]["type"] != "solid"]
             + [k for k, part in enumerate(parts) if part["type"] == "pair" and part["pixels"]]
             + [k for k, part in enumerate(parts) if part["type"] == "wings" and part["pixels"]])
    for k in order:
        part, m = parts[k], full[k]
        if part["type"] in MEDIAN:
            n = np.maximum(1, np.rint(half[k])).astype(np.int64)
            put(cx - n, cx + n - 1, m, k)
        else:
            n = half[k]
            side = part.get("side") or "both"
            lo_near = np.rint(cx - centre[k] - n).astype(np.int64)
            hi_near = np.maximum(lo_near, np.rint(cx - centre[k] + n).astype(np.int64) - 1)
            hi_near = np.minimum(hi_near, cx - 1)
            lo_near = np.minimum(lo_near, hi_near)
            if side in ("both", "near"):
                put(lo_near, hi_near, m, k)
            if side in ("both", "far"):
                put(2 * cx - 1 - hi_near, 2 * cx - 1 - lo_near, m, k)
    # ---- colour: what is exposed shows the sprite, what is inside (or hidden behind another part in the sprite) shows
    # the part's own colour
    occ = pid > 0
    pad = np.pad(occ, 1)
    exposed = occ & ~(pad[2:, 1:-1, 1:-1] & pad[:-2, 1:-1, 1:-1] & pad[1:-1, 2:, 1:-1] & pad[1:-1, :-2, 1:-1]
                      & pad[1:-1, 1:-1, 2:] & pad[1:-1, 1:-1, :-2])
    vx, vy, vz = np.nonzero(occ)
    vr, vc = (h - 1) - (vy - y0), vz - z0
    vk = pid[vx, vy, vz].astype(np.int64) - 1
    inside = (vr >= 0) & (vr < h) & (vc >= 0) & (vc < w)
    vr, vc = np.clip(vr, 0, h - 1), np.clip(vc, 0, w - 1)
    col = base_idx[vk, vr, vc]
    shows = exposed[vx, vy, vz] & inside & (owner[vr, vc] == vk + 1) & ~decal_px[vr, vc]
    col = np.where(shows, surface[vr, vc], col)
    col = np.where(col > 0, col, p.idx[vr, vc])
    grid = np.zeros((X, Y, Z), np.uint8)
    grid[vx, vy, vz] = np.clip(col, 1, 255).astype(np.uint8)
    # ---- the flank is the sprite, exactly: the first voxel met from each side takes the sprite's pixel
    any_col = occ.any(axis=0)                               # (Y, Z)
    first = occ.argmax(axis=0)
    last = X - 1 - occ[::-1].argmax(axis=0)
    rs, cs = np.nonzero(p.mask & ~skipped)
    gy, gz = yy[rs], zz[cs]
    ok = any_col[gy, gz]
    grid[first[gy[ok], gz[ok]], gy[ok], gz[ok]] = surface[rs[ok], cs[ok]]
    sym = all((part.get("side") or "both") == "both" for part in parts if part["type"] in ("pair", "wings", "decal"))
    if sym:
        grid[last[gy[ok], gz[ok]], gy[ok], gz[ok]] = surface[rs[ok], cs[ok]]
    missing = int((~ok).sum())
    if missing:
        notes.append(f"{missing} sprite pixels got no voxel")
    for part in parts:
        part.pop("where", None)
    return Built(grid, palette, label, {"z0": z0, "y0": y0, "cx": cx, "w": w, "h": h}, parts, notes, pid)


# ---------------------------------------------------------------------------------------------- the game camera
def project(grid: np.ndarray, rgb: np.ndarray, role: str, depth: float = 0.6, with_index: bool = False):
    """The model as the overworld camera of a sprite draws it, one pixel per voxel: RGBA (h, w, 4).

    Sprites keep heights (a voxel up is a pixel up) and add depth: seen from the front, what is nearer (z) is drawn
    lower by ``depth`` pixels per voxel; from behind, what is further forward is drawn higher. ``depth`` 0 is a flat
    elevation, about 0.3-0.8 the usual top-down look. ``role``: front, back, side (the profile from -x) or side_far.
    ``rgb``: (256, 3) colour of each palette index. ``with_index``: also the voxel seen at each pixel, as (h, w, 3)
    grid coordinates (-1 where empty)."""
    occ = np.argwhere(grid > 0)
    if not len(occ):
        empty = np.zeros((1, 1, 4), np.uint8)
        return (empty, np.full((1, 1, 3), -1, np.int64)) if with_index else empty
    samples = np.array([[a, b, c] for a in (0.25, 0.75) for b in (0.25, 0.75) for c in (0.25, 0.75)])
    pts = (occ[:, None, :] + samples[None]).reshape(-1, 3)
    vox = np.repeat(np.arange(len(occ)), len(samples))
    x, y, z = pts[:, 0], pts[:, 1], pts[:, 2]
    if role == "front":
        u, v, dep = x, y - depth * z, z + depth * y
    elif role == "back":
        u, v, dep = -x, y + depth * z, -z + depth * y
    elif role == "side_far":
        u, v, dep = -z, y, x
    else:
        u, v, dep = z, y, -x
    ix = np.floor(u - np.floor(u.min())).astype(np.int64)
    iy = np.floor(np.ceil(v.max()) - v).astype(np.int64)
    hh, ww = int(iy.max()) + 1, int(ix.max()) + 1
    key = iy * ww + ix
    order = np.lexsort((dep, key))
    last = np.r_[key[order][1:] != key[order][:-1], True]
    win = order[last]
    out = np.zeros((hh * ww, 4), np.uint8)
    seen = occ[vox[win]]
    out[key[win], :3] = rgb[grid[seen[:, 0], seen[:, 1], seen[:, 2]]]
    out[key[win], 3] = 255
    if not with_index:
        return out.reshape(hh, ww, 4)
    index = np.full((hh * ww, 3), -1, np.int64)
    index[key[win]] = seen
    return out.reshape(hh, ww, 4), index.reshape(hh, ww, 3)


def palette_rgb(palette: list[str]) -> np.ndarray:
    out = np.zeros((256, 3), np.uint8)
    for i, c in enumerate(palette[:255]):
        out[i + 1] = _rgb(c)
    return out


def _hsv(rgb: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Hue (degrees), saturation and value (0..1) of (..., 3) colours."""
    c = rgb.astype(np.float64) / 255.0
    mx, mn = c.max(axis=-1), c.min(axis=-1)
    d = mx - mn
    r, g, b = c[..., 0], c[..., 1], c[..., 2]
    hue = np.zeros_like(mx)
    nz = d > 1e-9
    rm = nz & (mx == r)
    gm = nz & (mx == g) & ~rm
    bm = nz & ~rm & ~gm
    hue[rm] = (60 * ((g - b)[rm] / d[rm])) % 360
    hue[gm] = 60 * ((b - r)[gm] / d[gm]) + 120
    hue[bm] = 60 * ((r - g)[bm] / d[bm]) + 240
    sat = np.where(mx > 1e-9, d / np.maximum(mx, 1e-9), 0)
    return hue, sat, mx


class Families:
    """Colour families of a set of sprites: reds with reds, golds with golds, greys with greys. Two pictures of the same
    character agree at a pixel when the colours there are of the same family (shades do not matter); the dark ink of
    an outline agrees with anything."""

    INK, LIGHT, GREY = 0, 1, 2

    def __init__(self, colours: list[tuple[int, int, int]]):
        cols = np.array(sorted(set(tuple(int(v) for v in c) for c in colours)) or [(0, 0, 0)], np.int32)
        hue, sat, val = _hsv(cols)
        cls = np.full(len(cols), -1, np.int64)
        cls[val < 0.42] = self.INK
        grey = (cls < 0) & (sat < 0.2)
        cls[grey & (val >= 0.62)] = self.LIGHT
        cls[grey & (val < 0.62)] = self.GREY
        chroma = np.flatnonzero(cls < 0)
        nxt = 3
        if len(chroma):
            order = chroma[np.argsort(hue[chroma])]
            hs = hue[order]
            # clusters along the hue circle: a gap of more than 18 degrees starts a new family
            gaps = np.diff(np.r_[hs, hs[0] + 360])
            start = int(np.argmax(gaps)) + 1 if len(hs) > 1 else 0
            order = np.r_[order[start:], order[:start]]
            hs = hue[order]
            prev = None
            for i, h in zip(order, hs):
                if prev is not None and ((h - prev) % 360) > 18:
                    nxt += 1
                cls[i] = nxt
                prev = h
            nxt += 1
        self.cols, self.cls, self.n = cols, cls, nxt

    def classify(self, rgba: np.ndarray) -> np.ndarray:
        """(h, w) family of each opaque pixel (-1 where empty): the family of the nearest known colour."""
        h, w = rgba.shape[:2]
        out = np.full((h, w), -1, np.int64)
        m = rgba[..., 3] >= 128
        if not m.any():
            return out
        px = rgba[m][:, :3].astype(np.int32)
        uniq, inv = np.unique(px, axis=0, return_inverse=True)
        d = ((uniq[:, None, :] - self.cols[None]) ** 2).sum(axis=2)
        out[m] = self.cls[d.argmin(axis=1)][inv]
        return out

    def agree(self, a: np.ndarray, b: np.ndarray) -> np.ndarray:
        """How much two family maps agree where both have a pixel (1 same family, 0 different)."""
        same = (a == b).astype(np.float64)
        ink = (a == self.INK) | (b == self.INK)
        near = ((a == self.LIGHT) & (b == self.GREY)) | ((a == self.GREY) & (b == self.LIGHT))
        return np.where(ink, 1.0, np.where(near, 0.6, same))


def families_of(*pictures: np.ndarray) -> Families:
    cols: list[tuple[int, int, int]] = []
    for pic in pictures:
        if pic is None:
            continue
        m = pic[..., 3] >= 128
        cols += [tuple(int(v) for v in c) for c in np.unique(pic[m][:, :3], axis=0)]
    return Families(cols)


def compare(img: np.ndarray, sprite: np.ndarray, fam: Optional[Families] = None, search: int = 5) -> dict[str, Any]:
    """How a projected model (RGBA) matches a sprite (RGBA) at the best placement: ``iou`` (silhouette overlap),
    ``match`` (overlap counting only the pixels whose colours are of the same family: the wings must be where the
    sprite's wings are, not just anything) and where they differ (``missing``, ``extra`` relative to the sprite)."""
    t = sprite[..., 3] >= 128
    m = img[..., 3] >= 128
    th, tw = t.shape
    if not m.any() or not t.any():
        return {"iou": 0.0, "match": 0.0, "score": 0.0, "shift": [0, 0], "missing": 1.0, "extra": 0.0}
    fam = fam or families_of(sprite, img)
    tc, mc = fam.classify(sprite), fam.classify(img)
    tb = np.nonzero(t.any(axis=0))[0], np.nonzero(t.any(axis=1))[0]
    mb = np.nonzero(m.any(axis=0))[0], np.nonzero(m.any(axis=1))[0]
    ox0 = int(round((tb[0][0] + tb[0][-1]) / 2 - (mb[0][0] + mb[0][-1]) / 2))
    oy0 = int(round((tb[1][0] + tb[1][-1]) / 2 - (mb[1][0] + mb[1][-1]) / 2))
    t_area, m_area = int(t.sum()), int(m.sum())
    best = None
    for dy in range(-search, search + 1):
        for dx in range(-search, search + 1):
            ox, oy = ox0 + dx, oy0 + dy
            y0, x0 = max(0, oy), max(0, ox)
            y1, x1 = min(th, oy + m.shape[0]), min(tw, ox + m.shape[1])
            if y1 <= y0 or x1 <= x0:
                continue
            ts = t[y0:y1, x0:x1]
            ms = m[y0 - oy:y1 - oy, x0 - ox:x1 - ox]
            both = ts & ms
            inter = int(both.sum())
            union = max(1, t_area + m_area - inter)
            iou = inter / union
            if best is not None and iou < best["iou"] - 0.15:
                continue
            good = float(fam.agree(tc[y0:y1, x0:x1][both], mc[y0 - oy:y1 - oy, x0 - ox:x1 - ox][both]).sum()) if inter else 0.0
            match = good / union
            score = 0.75 * match + 0.25 * iou - 1e-4 * (abs(dx) + abs(dy))
            if best is None or score > best["score"]:
                best = {"iou": iou, "match": match, "score": score, "shift": [ox, oy],
                        "missing": (t_area - inter) / t_area, "extra": (m_area - inter) / t_area}
    return best or {"iou": 0.0, "match": 0.0, "score": 0.0, "shift": [0, 0], "missing": 1.0, "extra": 0.0}


# ---------------------------------------------------------------------------------------------- fitting the numbers
def evaluate(built: Built, sprites: dict[str, np.ndarray], depths: dict[str, float], weights: Optional[dict[str, float]] = None,
             rgb: Optional[np.ndarray] = None, fam: Optional[Families] = None) -> dict[str, Any]:
    """How the model matches the front and back sprites from their cameras: per view and the weighted mean score."""
    rgb = palette_rgb(built.palette) if rgb is None else rgb
    fam = fam or families_of(*[sprites.get(r) for r in ("side", "front", "back")])
    views, total, wsum = {}, 0.0, 0.0
    for role in ("front", "back"):
        if sprites.get(role) is None:
            continue
        img = project(built.grid, rgb, role, float(depths.get(role, 0.6)))
        res = compare(img, sprites[role], fam)
        res["depth"] = float(depths.get(role, 0.6))
        views[role] = res
        wt = float((weights or {}).get(role, 1.0))
        total += wt * res["score"]
        wsum += wt
    return {"score": total / wsum if wsum else 0.0, "views": views}


def tune_camera(built: Built, sprites: dict[str, np.ndarray], fam: Optional[Families] = None,
                start: Optional[dict[str, float]] = None) -> dict[str, float]:
    """The depth of the front and back cameras that makes the model look most like each sprite (0 = flat elevation;
    sheets differ, and so do the two sprites of one creature)."""
    rgb = palette_rgb(built.palette)
    fam = fam or families_of(*[sprites.get(r) for r in ("side", "front", "back")])
    out = {}
    for role in ("front", "back"):
        if sprites.get(role) is None:
            continue
        best = (-1.0, float((start or {}).get(role, 0.5)))
        for d in (0.0, 0.15, 0.3, 0.45, 0.6, 0.75, 0.9, 1.05):
            sc = compare(project(built.grid, rgb, role, d), sprites[role], fam)["score"]
            if sc > best[0] + 1e-4:
                best = (sc, d)
        out[role] = best[1]
    return out


def paint_caps(built: Built, sprites: dict[str, np.ndarray], depths: dict[str, float], fam: Optional[Families] = None) -> int:
    """Fans keep the side sprite on their flanks, but their end (a tail fan seen from behind) is drawn in another
    sprite: that sprite's picture of the part is laid over the voxels its camera sees and the side cameras do not.
    Only where the colours are of the part's own families. Returns how many voxels changed."""
    if built.pid is None:
        return 0
    fam = fam or families_of(*[sprites.get(r) for r in ("side", "front", "back")])
    grid, pid = built.grid, built.pid
    X = grid.shape[0]
    occ = grid > 0
    first = occ.argmax(axis=0)
    last = X - 1 - occ[::-1].argmax(axis=0)
    xs = np.arange(X)[:, None, None]
    side_seen = occ & ((xs == first[None]) | (xs == last[None]))
    rgb = palette_rgb(built.palette)
    pal_cls = fam.classify(np.concatenate([rgb[1:len(built.palette) + 1], np.full((len(built.palette), 1), 255, np.uint8)], axis=1)[None])[0]
    changed = 0
    for k, part in enumerate(built.parts):
        want = str(part.get("paint") or "none")
        if part["type"] != "fan" or want == "none" or not part.get("pixels"):
            continue
        mine = pid == k + 1
        used = np.unique(grid[mine])
        fams = {int(pal_cls[i - 1]) for i in used if i > 0} - {Families.INK}
        best = None
        for role in (("back", "front") if want == "auto" else (want,)):
            sprite = sprites.get(role)
            if sprite is None:
                continue
            img, index = project(grid, rgb, role, float(depths.get(role, 0.5)), with_index=True)
            res = compare(img, sprite, fam)
            ox, oy = res["shift"]
            seen = index[..., 0] >= 0
            ix, iy, iz = index[..., 0].clip(0), index[..., 1].clip(0), index[..., 2].clip(0)
            part_px = seen & (pid[ix, iy, iz] == k + 1)
            paintable = part_px & ~side_seen[ix, iy, iz]
            if paintable.sum() < 8:
                continue
            if best is None or paintable.sum() > best[0]:
                best = (int(paintable.sum()), role, sprite, index, part_px, paintable, (ox, oy))
        if best is None:
            continue
        _, role, sprite, index, part_px, paintable, (ox, oy) = best
        th, tw = sprite.shape[:2]
        scls = fam.classify(sprite)
        # the sprite's picture of this part: the areas of its colour families (with the lines among them) that the
        # model's part falls on
        wanted = np.isin(scls, sorted(fams))
        lab, n = components(wanted | ((scls == Families.INK) & _neigh4(wanted)), conn8=True)
        rs, cs = np.nonzero(part_px)
        pr, pc = rs + oy, cs + ox
        ok = (pr >= 0) & (pr < th) & (pc >= 0) & (pc < tw)
        hits = np.bincount(lab[pr[ok], pc[ok]], minlength=n + 1)
        hits[0] = 0
        if not hits.any():
            continue
        region = lab == int(hits.argmax())
        tr, tc = np.nonzero(region)
        t0r, t1r, t0c, t1c = tr.min(), tr.max(), tc.min(), tc.max()
        m0r, m1r, m0c, m1c = rs.min(), rs.max(), cs.min(), cs.max()
        qr, qc = np.nonzero(paintable)
        sr = np.rint(t0r + (qr - m0r) * (t1r - t0r) / max(1, m1r - m0r)).astype(np.int64).clip(0, th - 1)
        sc = np.rint(t0c + (qc - m0c) * (t1c - t0c) / max(1, m1c - m0c)).astype(np.int64).clip(0, tw - 1)
        good = region[sr, sc]
        vox = index[qr, qc]
        for (x, y, z), r, c, g in zip(vox, sr, sc, good):
            if not g:
                continue
            cls = int(scls[r, c])
            cur = int(grid[x, y, z])
            if cls == Families.INK:
                br, bg, bb = _rgb(built.palette[cur - 1])
                hexc = "#%02x%02x%02x" % (int(br * 0.62), int(bg * 0.62), min(255, int(bb * 0.66)))
            elif cls in fams:
                hexc = _hex(sprite[r, c])
            else:
                continue
            if hexc not in built.palette:
                if len(built.palette) >= 255:
                    continue
                built.palette.append(hexc)
            new = built.palette.index(hexc) + 1
            if new != cur:
                grid[x, y, z] = new
                changed += 1
        part["painted"] = role
    return changed + paint_faces(built, sprites, depths, fam, side_seen)


def surface_normals(occ: np.ndarray, reach: int = 2) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Which way the surface looks at each voxel (x, y, z parts, not normalised): away from where the neighbouring
    voxels are, over ``reach`` voxels around."""
    f = np.pad(occ.astype(np.float32), reach)
    X, Y, Z = occ.shape
    out = [np.zeros(occ.shape, np.float32) for _ in range(3)]
    for dx in range(-reach, reach + 1):
        for dy in range(-reach, reach + 1):
            for dz in range(-reach, reach + 1):
                if not (dx or dy or dz):
                    continue
                near = f[reach + dx:reach + dx + X, reach + dy:reach + dy + Y, reach + dz:reach + dz + Z]
                out[0] -= dx * near
                out[1] -= dy * near
                out[2] -= dz * near
    return out[0], out[1], out[2]


def paint_faces(built: Built, sprites: dict[str, np.ndarray], depths: dict[str, float], fam: Families,
                side_seen: Optional[np.ndarray] = None) -> int:
    """The side sprite colours a solid all around, which is right for its flanks and wrong for what faces the front
    camera (a face) or the back one (the hair, a back pattern). Where the front or back sprite shows the same pose as
    the model (their silhouettes overlap at least ``FACE_IOU``), or the part asks for it, the voxels that camera sees
    and that look more at it than sideways take the sprite's own pixel. Lines are toned down as on the flanks; a small dark spot
    inside (a pupil, a nostril) is kept. Returns how many voxels changed."""
    if built.pid is None:
        return 0
    grid, pid = built.grid, built.pid
    solids = {k + 1: str(part.get("paint") or "auto") for k, part in enumerate(built.parts) if part["type"] == "solid"}
    if not solids or all(v == "none" for v in solids.values()):
        return 0
    changed = 0
    nx, _, nz = surface_normals(grid > 0)
    for role in ("front", "back"):
        sprite = sprites.get(role)
        if sprite is None:
            continue
        # what looks more at this camera than at the side ones: the flanks stay the side sprite's
        facing = (nz if role == "front" else -nz) > np.abs(nx)
        img, index = project(grid, palette_rgb(built.palette), role, float(depths.get(role, 0.5)), with_index=True)
        res = compare(img, sprite, fam)
        same_pose = res["iou"] >= FACE_IOU
        wanted = {k for k, v in solids.items() if v in (role, "both") or (v == "auto" and same_pose)}
        if not wanted:
            continue
        ox, oy = res["shift"]
        th, tw = sprite.shape[:2]
        opaque = sprite[..., 3] >= 128
        inks = {tuple(c) for c in ink_colours(sprite)}
        ink = opaque & np.array([[tuple(int(v) for v in px[:3]) in inks for px in row] for row in sprite])
        # the outline and the lines that hang from it are drawing; ink that touches neither is a feature of its own
        rim = opaque & ~(np.pad(opaque, 1)[:-2, 1:-1] & np.pad(opaque, 1)[2:, 1:-1] & np.pad(opaque, 1)[1:-1, :-2] & np.pad(opaque, 1)[1:-1, 2:])
        lab, n = components(ink, conn8=True)
        lines = np.zeros(n + 1, bool)
        lines[np.unique(lab[rim & ink])] = True
        sizes = np.bincount(lab.ravel(), minlength=n + 1)
        lines |= sizes > 6
        lines[0] = False
        scls = fam.classify(sprite)
        pal_cls = fam.classify(np.concatenate([palette_rgb(built.palette)[1:len(built.palette) + 1],
                                               np.full((len(built.palette), 1), 255, np.uint8)], axis=1)[None])[0]
        fams: dict[int, set] = {}
        rs, cs = np.nonzero(index[..., 0] >= 0)
        done: dict[int, int] = {}
        for r, c in zip(rs, cs):
            x, y, z = (int(v) for v in index[r, c])
            k = int(pid[x, y, z])
            if k not in wanted or not facing[x, y, z]:
                continue
            sr, sc = r + oy, c + ox
            if not (0 <= sr < th and 0 <= sc < tw) or not opaque[sr, sc]:
                continue
            cur = int(grid[x, y, z])
            if ink[sr, sc] and lines[lab[sr, sc]]:
                if rim[sr, sc]:
                    continue                                   # the outline is not a colour of the character
                br, bg, bb = _rgb(built.palette[cur - 1])
                hexc = "#%02x%02x%02x" % (int(br * 0.62), int(bg * 0.62), min(255, int(bb * 0.66)))
            else:
                if not same_pose:
                    # asked for on a sprite in another pose: only colours this part has, so no wing lands on the chest
                    if k not in fams:
                        fams[k] = {int(pal_cls[i - 1]) for i in np.unique(grid[pid == k]) if i > 0} - {Families.INK}
                    if int(scls[sr, sc]) not in fams[k] and int(scls[sr, sc]) != Families.INK:
                        continue
                hexc = _hex(sprite[sr, sc])
            if hexc not in built.palette:
                if len(built.palette) >= 255:
                    continue
                built.palette.append(hexc)
            new = built.palette.index(hexc) + 1
            if new != cur:
                grid[x, y, z] = new
                changed += 1
                done[k] = done.get(k, 0) + 1
        for k, n_px in done.items():
            part = built.parts[k - 1]
            part["faces"] = (part.get("faces", "") + " " + role).strip()
    return changed


# ---------------------------------------------------------------------------------------------- reading the other sprites
def symmetry_axis(sprite: np.ndarray) -> float:
    """The column (may be .5) a front or back sprite is most symmetric about."""
    m = sprite[..., 3] >= 128
    w = m.shape[1]
    best, axis = -1.0, (w - 1) / 2
    for twice in range(max(0, w - 1 - 6), w - 1 + 7):         # 2 * axis around the middle
        a = twice / 2.0
        cols = np.arange(w)
        mir = np.rint(2 * a - cols).astype(int)
        ok = (mir >= 0) & (mir < w)
        inter = (m[:, cols[ok]] & m[:, mir[ok]]).sum()
        score = inter / max(1, m.sum())
        if score > best + 1e-9:
            best, axis = score, a
    return axis


def measure(sprite: np.ndarray, role: str) -> str:
    """What a front or back sprite says about widths, as text: its size, and for each row how wide the figure is in
    all and how wide the piece on its middle line is (a body or head between two wings)."""
    m = sprite[..., 3] >= 128
    h, w = m.shape
    axis = symmetry_axis(sprite)
    lines = [f"{role.upper()} sprite: {w} wide x {h} tall, symmetric about column {axis:g}. Row: total width / width of the "
             "piece on the middle line (rows where they differ have something to each side: wings, arms, legs)."]
    mid = int(round(axis))
    prev = None
    for r in range(h):
        cols = np.nonzero(m[r])[0]
        if not len(cols):
            continue
        total = int(cols.max() - cols.min() + 1)
        c = min(max(mid, 0), w - 1)
        if not m[r, c] and c + 1 < w and m[r, c + 1]:
            c += 1
        if not m[r, c] and c - 1 >= 0 and m[r, c - 1]:
            c -= 1
        if m[r, c]:
            lo = c
            while lo > 0 and m[r, lo - 1]:
                lo -= 1
            hi = c
            while hi < w - 1 and m[r, hi + 1]:
                hi += 1
            centre = hi - lo + 1
        else:
            centre = 0
        cur = (total, centre)
        if cur != prev:
            lines.append(f"  row {r}: {total} / {centre}")
        prev = cur
    return "\n".join(lines)


def auto_plan(p: Profile) -> dict[str, Any]:
    """A plan when nobody wrote one: the whole profile as one rounded solid. It gives a chunky figure with the exact
    profile; wings, legs and fans need someone (or the model) to say so."""
    h, w = p.shape
    return {"facing": "left" if p.flipped else "right",
            "parts": [{"name": "body", "type": "solid", "where": [[0, 0, w - 1, h - 1]], "aspect": 1.1}]}


# ---------------------------------------------------------------------------------------------- pictures of the work
def _zoomed(arr: np.ndarray, zoom: int):
    from PIL import Image
    img = Image.fromarray(arr, "RGBA")
    bg = Image.new("RGBA", img.size, (58, 54, 64, 255))
    bg.alpha_composite(img)
    return bg.resize((img.width * zoom, img.height * zoom), Image.NEAREST)


def _font(size: int = 13):
    from PIL import ImageFont
    try:
        return ImageFont.load_default(size=size)
    except TypeError:
        return ImageFont.load_default()


PART_COLOURS = [(230, 80, 60), (70, 160, 230), (240, 200, 60), (120, 200, 90), (200, 110, 220), (90, 220, 210), (240, 140, 60),
                (160, 160, 170), (250, 120, 160), (120, 120, 240), (200, 220, 120), (150, 100, 60), (60, 150, 110), (235, 235, 235)]


def parts_picture(p: Profile, label: np.ndarray, parts: list[dict[str, Any]], zoom: int = 12):
    """The side sprite next to the same sprite coloured by part, with the parts' names and a ruler of columns and
    rows: what the plan's shapes actually caught."""
    from PIL import Image, ImageDraw
    h, w = p.shape
    img = np.zeros((h, w, 4), np.uint8)
    for k in range(len(parts)):
        m = label == k + 1
        img[m, :3] = PART_COLOURS[k % len(PART_COLOURS)]
        img[m, 3] = 255
    img[p.ink & (img[..., 3] > 0), :3] = (img[p.ink & (img[..., 3] > 0), :3] * 0.62).astype(np.uint8)
    left, right = _zoomed(p.rgba, zoom), _zoomed(img, zoom)
    pad, ruler = 10, 22
    out = Image.new("RGBA", (ruler + left.width + pad + ruler + right.width + pad, ruler + left.height + pad + 18), (22, 19, 26, 255))
    d = ImageDraw.Draw(out)
    f, small = _font(13), _font(10)
    for ox, pic in ((ruler, left), (ruler + left.width + pad + ruler, right)):
        out.paste(pic, (ox, ruler))
        for c in range(0, w, 5):
            d.text((ox + c * zoom + 1, 8), str(c), fill=(200, 190, 170, 255), font=small)
            d.line([(ox + c * zoom, ruler - 4), (ox + c * zoom, ruler)], fill=(200, 190, 170, 255))
        for r in range(0, h, 5):
            d.text((ox - ruler + 2, ruler + r * zoom), str(r), fill=(200, 190, 170, 255), font=small)
    ox = ruler + left.width + pad + ruler
    for k, part in enumerate(parts):
        m = label == k + 1
        if not m.any():
            continue
        rs, cs = np.nonzero(m)
        i = int(np.argmin((rs - rs.mean()) ** 2 + (cs - cs.mean()) ** 2))
        text = part["name"]
        x, y = ox + int(cs[i] * zoom) - 3 * len(text), ruler + int(rs[i] * zoom) - 6
        d.rectangle([x - 2, y - 1, x + 7 * len(text) + 2, y + 13], fill=(0, 0, 0, 150))
        d.text((x, y), text, fill=(255, 255, 255, 255), font=f)
    d.text((ruler, ruler + left.height + 4), "side sprite (columns across, rows down)  |  the same pixels by part", fill=(235, 225, 200, 255), font=f)
    return out


def match_picture(built: Built, sprites: dict[str, np.ndarray], depths: dict[str, float], fam: Optional[Families] = None,
                  zoom: int = 6):
    """For the front and back sprites: the sprite, the model from that sprite's camera and where their silhouettes
    differ (red: the sprite has it, the model does not; blue: the model has it, the sprite does not)."""
    from PIL import Image, ImageDraw
    rgb = palette_rgb(built.palette)
    fam = fam or families_of(*[sprites.get(r) for r in ("side", "front", "back")])
    rows, results = [], {}
    f = _font(13)
    for role in ("front", "back"):
        t = sprites.get(role)
        if t is None:
            continue
        depth = float(depths.get(role, 0.5))
        img = project(built.grid, rgb, role, depth)
        res = compare(img, t, fam)
        res["depth"] = depth
        results[role] = res
        th, tw = t.shape[:2]
        ox, oy = res["shift"]
        # a canvas that holds both, so nothing of the model is cut off
        x0, y0 = min(0, ox), min(0, oy)
        x1, y1 = max(tw, ox + img.shape[1]), max(th, oy + img.shape[0])
        cw, ch = x1 - x0, y1 - y0
        ta = np.zeros((ch, cw, 4), np.uint8)
        ma = np.zeros((ch, cw, 4), np.uint8)
        ta[-y0:-y0 + th, -x0:-x0 + tw] = t
        ma[oy - y0:oy - y0 + img.shape[0], ox - x0:ox - x0 + img.shape[1]] = img
        tm, mm = ta[..., 3] >= 128, ma[..., 3] >= 128
        diff = np.zeros_like(ta)
        diff[tm & mm] = (150, 150, 150, 255)
        diff[tm & ~mm] = (235, 60, 50, 255)
        diff[mm & ~tm] = (60, 120, 240, 255)
        tiles = [(f"{role} sprite", _zoomed(ta, zoom)), ("model from that camera", _zoomed(ma, zoom)),
                 (f"silhouettes: overlap {res['iou']:.2f} (red missing, blue extra)", _zoomed(diff, zoom))]
        row = Image.new("RGBA", (sum(i.width for _, i in tiles) + 8 * 4, tiles[0][1].height + 28), (22, 19, 26, 255))
        d = ImageDraw.Draw(row)
        x = 8
        for text, pic in tiles:
            row.paste(pic, (x, 22))
            d.text((x, 4), text, fill=(235, 225, 200, 255), font=f)
            x += pic.width + 8
        rows.append(row)
    if not rows:
        return None, results
    out = Image.new("RGBA", (max(r.width for r in rows), sum(r.height for r in rows)), (12, 10, 14, 255))
    y = 0
    for r in rows:
        out.paste(r, (0, y))
        y += r.height
    return out, results


def text_map(rgba: np.ndarray, palette: list[str], ink: Optional[set] = None) -> str:
    """Any sprite as text with the letters of ``palette`` ('.' empty, '#' ink): one character per pixel, row numbers
    and a column ruler. A colour that is not in the palette takes the nearest one's letter."""
    m = rgba[..., 3] >= 128
    h, w = m.shape
    pal = np.array([_rgb(c) for c in palette], np.int32)
    inks = ink if ink is not None else {tuple(c) for c in ink_colours(rgba)}
    lines = ["     " + "".join(str((c // 10) % 10) if c % 10 == 0 else " " for c in range(w))]
    cache: dict[tuple, str] = {}
    for r in range(h):
        row = []
        for c in range(w):
            if not m[r, c]:
                row.append(".")
                continue
            key = tuple(int(v) for v in rgba[r, c, :3])
            if key not in cache:
                cache[key] = "#" if key in inks else KEYS[int(((pal - np.array(key)) ** 2).sum(axis=1).argmin()) % len(KEYS)]
            row.append(cache[key])
        lines.append(f"{r:4d} " + "".join(row))
    return "\n".join(lines)


def sprites_picture(sprites: dict[str, np.ndarray], zoom: int = 10):
    """The sprites enlarged side by side, each with its name and a ruler of columns and rows."""
    from PIL import Image, ImageDraw
    tiles = []
    f, small = _font(13), _font(10)
    ruler = 22
    titles = {"side": "SIDE sprite (a true profile)", "front": "FRONT sprite (camera above)", "back": "BACK sprite (camera above)"}
    for role in ("side", "front", "back"):
        t = sprites.get(role)
        if t is None:
            continue
        pic = _zoomed(t, zoom)
        h, w = t.shape[:2]
        tile = Image.new("RGBA", (ruler + pic.width + 6, ruler + pic.height + 22), (22, 19, 26, 255))
        d = ImageDraw.Draw(tile)
        tile.paste(pic, (ruler, ruler + 18))
        d.text((ruler, 2), titles[role], fill=(235, 225, 200, 255), font=f)
        for c in range(0, w, 5):
            d.text((ruler + c * zoom + 1, 26), str(c), fill=(200, 190, 170, 255), font=small)
        for r in range(0, h, 5):
            d.text((2, ruler + 18 + r * zoom), str(r), fill=(200, 190, 170, 255), font=small)
        tiles.append(tile)
    if not tiles:
        return Image.new("RGBA", (8, 8), (22, 19, 26, 255))
    out = Image.new("RGBA", (sum(t.width for t in tiles) + 8 * (len(tiles) + 1), max(t.height for t in tiles) + 16), (12, 10, 14, 255))
    x = 8
    for t in tiles:
        out.paste(t, (x, 8))
        x += t.width + 8
    return out


def cutout(data: bytes) -> np.ndarray:
    """A sprite file as RGBA with its flat background gone - also the pockets of background the figure encloses
    (between a wing and the body), which a flood from the border does not reach - cropped and at its real pixel size."""
    from collections import Counter
    from .images import _background_to_alpha, _crop, _pixel_scale, load
    raw = np.asarray(load(data)).copy()
    opaque = not (raw[..., 3] < 250).any()
    corners = [tuple(int(v) for v in raw[y, x, :3]) for y, x in ((0, 0), (0, -1), (-1, 0), (-1, -1))]
    bg, n = Counter(corners).most_common(1)[0]
    arr = _background_to_alpha(raw, [])
    if opaque and n >= 3 and (arr[..., 3] < 128).any():
        diff = np.abs(arr[..., :3].astype(np.int32) - np.array(bg, np.int32)).sum(axis=2)
        arr[diff <= 8, 3] = 0
    arr = _crop(arr)
    scale = _pixel_scale(arr)
    if scale > 1:
        arr = arr[scale // 2:: scale, scale // 2:: scale]
    out = arr.copy()
    out[out[..., 3] < 128] = 0
    out[out[..., 3] >= 128, 3] = 255
    return out


def apply_changes(plan: dict[str, Any], set_: Optional[dict] = None, add: Optional[list] = None, remove: Optional[list] = None) -> dict[str, Any]:
    """A plan with some of its parts changed (``set_``: {part name: {key: value}}), removed or added. Unknown part
    names raise :class:`PlanError`."""
    import copy
    out = copy.deepcopy(plan)
    parts = out.get("parts") or []
    names = [str(pt.get("name")) for pt in parts]
    for name in remove or []:
        if str(name) in names:
            i = names.index(str(name))
            parts.pop(i)
            names.pop(i)
    for name, changes in (set_ or {}).items():
        if name in ("facing", "outline", "ink") and not isinstance(changes, dict):
            out[name] = changes
            continue
        if str(name) not in names:
            raise PlanError(f"there is no part called {name!r} (parts: {', '.join(names)})")
        if not isinstance(changes, dict):
            raise PlanError(f"the changes of part {name!r} must be an object of values")
        part = parts[names.index(str(name))]
        for key, val in changes.items():
            if val is None:
                part.pop(key, None)
            else:
                part[key] = val
    for part in add or []:
        if isinstance(part, dict):
            if str(part.get("name")) in names:
                parts[names.index(str(part.get("name")))] = part
            else:
                parts.append(part)
                names.append(str(part.get("name")))
    out["parts"] = parts
    return out


def report(built: Built) -> str:
    """What was built, one line per part, with the numbers Seurat used."""
    lines = []
    for part in built.parts:
        bits = [f"{part['pixels']} px"]
        for key, text in (("width_built", "width"), ("span_built", "span"), ("offset_used", "offset"), ("root_used", "root"),
                          ("out_up_used", "out_up"), ("out_back_used", "out_back"), ("base_used", "base"), ("thickness", "thickness"),
                          ("on", "on"), ("painted", "end painted from the sprite"),
                          ("faces", "face painted from the sprite:")):
            if part.get(key) is not None:
                bits.append(f"{text} {part[key]}")
        if part["type"] == "wings" and part.get("out_up_used") is None:
            bits.append(f"out_up {part.get('out_up')}, out_back {part.get('out_back')}")
        lines.append(f"- {part['name']} ({part['type']}): " + ", ".join(bits))
    lines += [f"! {n}" for n in built.notes]
    return "\n".join(lines)
