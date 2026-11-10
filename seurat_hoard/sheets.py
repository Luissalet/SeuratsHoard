"""Animated sprites and sprite sheets: every frame of an animation (GIF, animated WebP or PNG) or every cell of a sheet
(walk cycles, four or eight directions), cut out like a single sprite, and a guess of which frames show which side.

A walk sheet usually has one row (or column) per direction (front, side, back...) and one cell per step of the walk; an
animation may walk in several directions one after another. Seurat uses the directions as extra cameras of the same
model and the steps as extra poses of each direction: the standing frame (the one most like the others) is the main view,
the rest count as poses (``spritecam``).
"""

from __future__ import annotations

import io
from dataclasses import dataclass, field
from typing import Any, Optional

import numpy as np
from PIL import Image, ImageDraw, ImageFont, ImageSequence

from .errors import SeuratError
from .images import _background_to_alpha, _pixel_scale, load

MAX_FRAMES = 96
ROLES = ("front", "back", "side", "view", "none")


@dataclass
class Frame:
    index: int
    row: int
    col: int
    box: tuple[int, int, int, int]      # where the cell is in the sheet (x0, y0, x1, y1), at the sheet's own pixels
    rgba: np.ndarray                    # the subject: background removed, cropped, pixel-art enlargement undone
    duration: int = 0                   # animations: milliseconds

    def to_dict(self) -> dict[str, Any]:
        h, w = self.rgba.shape[:2]
        return {"index": self.index, "row": self.row, "col": self.col, "box": list(self.box), "width": w, "height": h,
                "pixels": int((self.rgba[..., 3] >= 128).sum()), **({"duration": self.duration} if self.duration else {})}

    def png(self) -> bytes:
        buf = io.BytesIO()
        Image.fromarray(self.rgba, "RGBA").save(buf, "PNG")
        return buf.getvalue()


@dataclass
class Sheet:
    kind: str                           # animation | grid | single
    width: int
    height: int
    rows: int
    cols: int
    scale: int
    frames: list[Frame] = field(default_factory=list)
    notes: list[str] = field(default_factory=list)

    def to_dict(self) -> dict[str, Any]:
        return {"kind": self.kind, "width": self.width, "height": self.height, "rows": self.rows, "cols": self.cols,
                "scale": self.scale, "count": len(self.frames), "frames": [f.to_dict() for f in self.frames], "notes": self.notes}

    def frame(self, index: int) -> Frame:
        for f in self.frames:
            if f.index == index:
                return f
        raise SeuratError("invalid", f"There is no frame {index} (the picture has {len(self.frames)}: 0..{len(self.frames) - 1}).")


# ---------------------------------------------------------------------------------------------- reading
def _background(arr: np.ndarray) -> np.ndarray:
    """A flat background made transparent, also when the picture already has some transparent pixels (GIF frames often
    mix an opaque backdrop with a transparent colour): the corner colour, flooded from the borders through itself and
    through what is already transparent."""
    arr = _background_to_alpha(arr, [])
    opaque = arr[..., 3] >= 128
    if not opaque.any():
        return arr
    corners = [(0, 0), (0, -1), (-1, 0), (-1, -1)]
    cols = [tuple(int(v) for v in arr[y, x, :3]) for y, x in corners if opaque[y, x]]
    if len(cols) < 2:                    # a sprite on a transparent background: nothing to do
        return arr
    from collections import Counter
    bg, n = Counter(cols).most_common(1)[0]
    if n < 2 or n < len(cols) - 1:
        return arr
    match = opaque & (np.abs(arr[..., :3].astype(np.int32) - np.array(bg)).sum(axis=2) <= 24)
    passable = match | ~opaque
    reach = np.zeros_like(passable)
    reach[0, :], reach[-1, :], reach[:, 0], reach[:, -1] = passable[0, :], passable[-1, :], passable[:, 0], passable[:, -1]
    for _ in range(sum(arr.shape[:2])):
        grown = reach.copy()
        grown[1:, :] |= reach[:-1, :]
        grown[:-1, :] |= reach[1:, :]
        grown[:, 1:] |= reach[:, :-1]
        grown[:, :-1] |= reach[:, 1:]
        grown &= passable
        if (grown == reach).all():
            break
        reach = grown
    gone = reach & match
    if gone.mean() > 0.97:
        return arr
    out = arr.copy()
    out[gone, 3] = 0
    return out


def _clean(arr: np.ndarray, scale: int) -> Optional[np.ndarray]:
    """A cell or frame: cropped to its subject, at its real pixel size, alpha 0 or 255. None when it is empty."""
    alpha = arr[..., 3] >= 128
    if not alpha.any():
        return None
    ys, xs = np.nonzero(alpha)
    out = arr[ys.min(): ys.max() + 1, xs.min(): xs.max() + 1]
    if scale > 1:
        out = out[scale // 2:: scale, scale // 2:: scale]
    out = out.copy()
    out[out[..., 3] < 128] = 0
    out[out[..., 3] >= 128, 3] = 255
    return out if (out[..., 3] > 0).any() else None


def _bands(occ: np.ndarray) -> list[tuple[int, int]]:
    """Runs of True in a 1-D array, as (start, end) with end exclusive."""
    out, start = [], None
    for i, v in enumerate(occ.tolist() + [False]):
        if v and start is None:
            start = i
        elif not v and start is not None:
            out.append((start, i))
            start = None
    return out


def _spans(occ: np.ndarray, size: int, forced: int) -> list[tuple[int, int]]:
    """How an axis of a sheet is cut into cells: ``forced`` equal cells, or equal cells at the period of the subjects
    when they sit on a regular grid, or else one cell per run of subject pixels (split at the middle of the gaps)."""
    if forced > 0:
        return [(round(i * size / forced), round((i + 1) * size / forced)) for i in range(forced)]
    bands = _bands(occ)
    if len(bands) <= 1:
        return [(0, size)]
    centres = np.array([(a + b) / 2 for a, b in bands])
    step = float(np.median(np.diff(centres)))
    if step > 0:
        est = size / step
        k = int(round(est))
        cell = size / max(1, k)
        # a regular grid only when every run of subject pixels sits inside one cell, one run per cell (a margin around the
        # sheet or uneven gaps would otherwise cut the sprites)
        inside = all(int(a // cell) == int((b - 1) // cell) for a, b in bands)
        cells = {int(a // cell) for a, b in bands}
        if k >= len(bands) and abs(est - k) < 0.15 and inside and len(cells) == len(bands):
            return [(round(i * cell), round((i + 1) * cell)) for i in range(k)]
    cuts = [0] + [(bands[i][1] + bands[i + 1][0]) // 2 for i in range(len(bands) - 1)] + [size]
    return list(zip(cuts[:-1], cuts[1:]))


def _period(profile: np.ndarray, size: int, strict: float = 0.45) -> int:
    """How many equal cells an axis holds when its sprites touch (no empty line between them): the period of the
    subject's pixel count along the axis (autocorrelation), among the cell sizes that divide the axis. 1 if none."""
    p = profile.astype(np.float64) - profile.mean()
    den = float((p * p).sum())
    if den <= 0 or size < 12:
        return 1
    scores = {}
    for lag in range(6, size // 2 + 1):
        rest = size % lag
        if min(rest, lag - rest) > 2:                    # the cells must fill the axis (a pixel or two of slack)
            continue
        scores[lag] = float((p[:-lag] * p[lag:]).sum()) / den * size / (size - lag)
    if not scores:
        return 1
    top = max(scores.values())
    if top < strict:
        return 1
    best_lag = min(lag for lag, s in scores.items() if s >= top - 0.1)      # the smallest cell that repeats as well
    return int(round(size / best_lag))


def _repeat_spans(alpha: np.ndarray, axis: int, strict: float) -> Optional[list[tuple[int, int]]]:
    """Equal cells along an axis whose sprites touch, found from the repeat over the part the sprites occupy (a margin
    around the sheet does not count); the first and last cells reach the edges."""
    occ = alpha.any(axis=axis)
    idx = np.flatnonzero(occ)
    if not len(idx):
        return None
    a, b = int(idx[0]), int(idx[-1]) + 1
    k = _period(alpha.sum(axis=axis)[a:b], b - a, strict)
    if k <= 1:
        return None
    spans = [(a + round(i * (b - a) / k), a + round((i + 1) * (b - a) / k)) for i in range(k)]
    spans[0] = (0, spans[0][1])
    spans[-1] = (spans[-1][0], len(occ))
    return spans


def read(data: bytes, cols: int = 0, rows: int = 0) -> Sheet:
    """Every frame of an animation, or every cell of a sheet (``cols`` x ``rows`` when given, else found from the gaps
    between the sprites, or from their repeat when they touch)."""
    load(data)                                         # size and format checks
    img = Image.open(io.BytesIO(data))
    n = int(getattr(img, "n_frames", 1) or 1)
    notes: list[str] = []
    if n > 1:
        raw, durations = [], []
        for i, fr in enumerate(ImageSequence.Iterator(img)):
            if i >= MAX_FRAMES:
                notes.append(f"only the first {MAX_FRAMES} frames are read")
                break
            raw.append(_background(np.asarray(fr.convert("RGBA")).copy()))
            durations.append(int(fr.info.get("duration") or img.info.get("duration") or 0))
        scale = _pixel_scale(raw[0]) if raw else 1
        frames = []
        for i, (arr, d) in enumerate(zip(raw, durations)):
            sub = _clean(arr, scale)
            if sub is not None:
                frames.append(Frame(len(frames), 0, i, (0, 0, arr.shape[1], arr.shape[0]), sub, d))
        if not frames:
            raise SeuratError("invalid", "Every frame of the animation is empty.")
        notes.insert(0, f"animation of {len(frames)} frames")
        if scale > 1:
            notes.append(f"pixel art enlarged x{scale}: read at its real size")
        return Sheet("animation", img.width, img.height, 1, len(frames), scale, frames, notes)
    arr = _background_to_alpha(np.asarray(img.convert("RGBA")).copy(), notes)
    alpha = arr[..., 3] >= 128
    if not alpha.any():
        raise SeuratError("invalid", "The picture is completely transparent.")
    h, w = alpha.shape
    xs, ys = _spans(alpha.any(axis=0), w, cols), _spans(alpha.any(axis=1), h, rows)
    # sprites that touch across one axis (no empty line between the rows) still repeat with the cell size
    if not rows and len(ys) == 1 and (len(xs) > 1 or not cols):
        found = _repeat_spans(alpha, 1, 0.45 if len(xs) > 1 else 0.7)
        if found:
            ys = found
            notes.append(f"rows found from the repeat of the sprites ({len(ys)})")
    if not cols and len(xs) == 1 and (len(ys) > 1 or not rows):
        found = _repeat_spans(alpha, 0, 0.45 if len(ys) > 1 else 0.7)
        if found:
            xs = found
            notes.append(f"columns found from the repeat of the sprites ({len(xs)})")
    cells = []
    for r, (y0, y1) in enumerate(ys):
        for c, (x0, x1) in enumerate(xs):
            cells.append((r, c, (x0, y0, x1, y1), arr[y0:y1, x0:x1]))
    # one scale for the whole sheet: the biggest cell decides
    biggest = max(cells, key=lambda t: int((t[3][..., 3] >= 128).sum()))
    crop = _clean(biggest[3], 1)
    scale = _pixel_scale(crop) if crop is not None else 1
    frames = []
    for r, c, box, cell in cells:
        sub = _clean(cell, scale)
        if sub is not None:
            frames.append(Frame(len(frames), r, c, box, sub))
    if frames and len(frames) > 1:
        # bits that are not frames (a detached sparkle, a stray pixel): far fewer pixels than the typical frame
        typical = float(np.median([(f.rgba[..., 3] > 0).sum() for f in frames]))
        kept = [f for f in frames if (f.rgba[..., 3] > 0).sum() >= 0.15 * typical]
        if len(kept) < len(frames):
            notes.append(f"{len(frames) - len(kept)} small specks ignored")
        frames = [Frame(i, f.row, f.col, f.box, f.rgba, f.duration) for i, f in enumerate(kept)]
    kind = "grid" if len(frames) > 1 else "single"
    used_rows = sorted({f.row for f in frames})
    used_cols = sorted({f.col for f in frames})
    if kind == "grid":
        notes.insert(0, f"sheet of {len(used_rows)} rows x {len(used_cols)} columns, {len(frames)} frames")
    if scale > 1:
        notes.append(f"pixel art enlarged x{scale}: read at its real size")
    return Sheet(kind, w, h, len(used_rows), len(used_cols), scale, frames, notes)


# ---------------------------------------------------------------------------------------------- which frame is which side
def _mask(f: Frame) -> np.ndarray:
    return f.rgba[..., 3] >= 128


def _common(a: np.ndarray, b: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Two masks on one canvas: centred in x, standing on the same bottom row (feet together)."""
    h, w = max(a.shape[0], b.shape[0]), max(a.shape[1], b.shape[1])

    def put(m):
        out = np.zeros((h, w), bool)
        x0 = (w - m.shape[1]) // 2
        out[h - m.shape[0]:, x0:x0 + m.shape[1]] = m
        return out
    return put(a), put(b)


def similarity(a: np.ndarray, b: np.ndarray) -> float:
    a, b = _common(a, b)
    return float((a & b).sum()) / max(1, int((a | b).sum()))


def symmetry(f: Frame) -> float:
    """How much a frame looks like its mirror image, in outline and in colours (front and back views are symmetric; a
    side view is not, even when its outline nearly is: the face, the eye or the arm is on one side)."""
    m = _mask(f)
    shape = similarity(m, m[:, ::-1])
    rgb = f.rgba[..., :3].astype(np.int32)
    both = m & m[:, ::-1]
    colour = float((np.abs(rgb - rgb[:, ::-1]).sum(axis=2) < 60)[both].mean()) if both.any() else 0.0
    return 0.5 * shape + 0.5 * colour


def medoid(frames: list[Frame]) -> Frame:
    """The frame most like all the others: in a walk cycle, the standing pose."""
    if len(frames) <= 2:
        return frames[0]
    masks = [_mask(f) for f in frames]
    scores = [sum(similarity(m, o) for o in masks) for m in masks]
    return frames[int(np.argmax(scores))]


def _detail(f: Frame) -> float:
    """Detail inside the upper part (a face: eyes, mouth, skin) - the front usually has more than the back: colour
    changes between neighbouring pixels away from the outline, and colour variety."""
    top = f.rgba[: max(2, int(f.rgba.shape[0] * 0.6))]
    m = top[..., 3] >= 128
    if not m.any():
        return 0.0
    inner = m.copy()
    inner[1:, :] &= m[:-1, :]
    inner[:-1, :] &= m[1:, :]
    inner[:, 1:] &= m[:, :-1]
    inner[:, :-1] &= m[:, 1:]
    rgb = top[..., :3].astype(np.int32)
    dx = (np.abs(rgb[:, 1:] - rgb[:, :-1]).sum(axis=2) > 60) & inner[:, 1:] & inner[:, :-1]
    dy = (np.abs(rgb[1:] - rgb[:-1]).sum(axis=2) > 60) & inner[1:] & inner[:-1]
    edges = float(dx.sum() + dy.sum()) / max(1.0, float(np.sqrt(m.sum())))
    return edges + 0.1 * float(len(np.unique(rgb[m] // 16, axis=0)))


def guess_views(sheet: Sheet) -> list[dict[str, Any]]:
    """Groups of frames that show one side of the subject, with a role each: [{"role", "frames": [main, poses...]}].
    An animation is one side (its frames are poses) unless it turns: then each run of alike frames is a side. On a
    sheet, the frames are grouped by rows or by columns (whichever groups more alike frames: one direction per group, one
    step of the walk per frame); asymmetric groups are side views, the symmetric ones front and back (the one with more
    detail at the top, a face, is the front)."""
    frames = sheet.frames
    if not frames:
        return []
    if sheet.kind != "grid":
        runs = _runs(frames)
        if len(runs) < 2:
            main = medoid(frames)
            return [{"role": "front", "frames": [main.index] + [f.index for f in frames if f is not main]}]
        return _assign(runs)

    def groups_by(key):
        out: dict[int, list[Frame]] = {}
        for f in frames:
            out.setdefault(getattr(f, key), []).append(f)
        return list(out.values())

    def within(groups):
        vals = [similarity(_mask(a), _mask(b)) for g in groups for i, a in enumerate(g) for b in g[i + 1:]]
        return float(np.mean(vals)) if vals else 0.0

    by_row, by_col = groups_by("row"), groups_by("col")
    if len(by_row) == 1 and len(by_col) > 1:
        # one row: the steps of one direction, or one frame per direction
        groups = by_row if within(by_row) >= 0.72 else [[f] for f in frames]
    elif len(by_col) == 1:
        groups = by_col if within(by_col) >= 0.72 else [[f] for f in frames]
    else:
        groups = by_row if within(by_row) >= within(by_col) else by_col
    return _assign(groups)


def _runs(frames: list[Frame]) -> list[list[Frame]]:
    """An animation that walks in several directions one after another (front steps, then side steps...): the runs of
    alike frames, cut where a frame differs clearly from the one before. One run when it never turns."""
    n = len(frames)
    if n < 4:
        return [frames]
    cache: dict[tuple[int, int], float] = {}

    def sim(i: int, j: int) -> float:
        if (i, j) not in cache:
            cache[(i, j)] = look_alike(frames[i], frames[j])
        return cache[(i, j)]

    # directions in an animation come as runs of the same length (eight steps front, eight to the left...): the length
    # whose runs are most alike inside and most different from each other
    margins: dict[int, float] = {}
    for size in range(2, n // 2 + 1):
        if n % size or n // size > 8:
            continue
        runs = [list(range(a, a + size)) for a in range(0, n, size)]
        inside = [sim(i, j) for r in runs for x, i in enumerate(r) for j in r[x + 1:]]
        across = [sim(i, j) for a, ra in enumerate(runs) for rb in runs[a + 1:] for i in ra[::max(1, size // 4)] for j in rb[::max(1, size // 4)]]
        margins[size] = float(np.mean(inside)) - float(np.mean(across))
    if not margins or max(margins.values()) < 0.12:
        return [frames]
    top = max(margins.values())
    # shorter runs are always a little more alike inside: the longest run that is nearly as clear wins (a whole walk)
    size = max(s for s, m in margins.items() if m >= 0.8 * top)
    return [frames[a:a + size] for a in range(0, n, size)]


def look_alike(a: Frame, b: Frame) -> float:
    """How alike two frames are in outline and colours (feet together, centred)."""
    ma, mb = _common(_mask(a), _mask(b))
    shape = float((ma & mb).sum()) / max(1, int((ma | mb).sum()))
    h, w = ma.shape

    def put(f):
        out = np.zeros((h, w, 3), np.int32)
        fh, fw = f.rgba.shape[:2]
        x0 = (w - fw) // 2
        out[h - fh:, x0:x0 + fw] = f.rgba[..., :3]
        return out
    both = ma & mb
    colour = float((np.abs(put(a) - put(b)).sum(axis=2) < 80)[both].mean()) if both.any() else 0.0
    return 0.5 * shape + 0.5 * colour


def _assign(groups: list[list[Frame]]) -> list[dict[str, Any]]:
    """Roles for groups of frames, one side each: the clearly asymmetric ones (up to two) are sides, of the rest the one
    with more detail at the top (a face) is the front and the plainest the back."""
    infos = []
    for g in groups:
        main = medoid(g)
        infos.append({"frames": [main.index] + [f.index for f in g if f is not main], "sym": symmetry(main), "detail": _detail(main)})
    by_sym = sorted(range(len(infos)), key=lambda i: infos[i]["sym"])
    top = infos[by_sym[-1]]["sym"]
    if len(infos) >= 3:
        # with three or more sides, the least symmetric ones (up to two, leaving two) are the sides when they are clearly
        # less symmetric than the most symmetric one
        side_like = [i for i in by_sym[: min(2, len(infos) - 2) or 1] if infos[i]["sym"] < top - 0.03]
    else:
        side_like = [i for i in by_sym if infos[i]["sym"] < 0.86][:1]
    rest = sorted((i for i in range(len(infos)) if i not in side_like), key=lambda i: -infos[i]["detail"])
    roles: dict[int, str] = {i: "side" for i in side_like}
    if rest:
        roles[rest[0]] = "front"
    if len(rest) > 1:
        roles[rest[-1]] = "back"
    return [{"role": roles.get(i, "none"), "frames": info["frames"]} for i, info in enumerate(infos)]


# ---------------------------------------------------------------------------------------------- pictures
def _font(size: int = 12):
    try:
        return ImageFont.load_default(size=size)
    except TypeError:
        return ImageFont.load_default()


def contact(sheet: Sheet, views: Optional[list[dict[str, Any]]] = None, cell: int = 96) -> Image.Image:
    """Every frame enlarged on a checker, numbered, laid out as in the sheet (an animation: eight to a row); with
    ``views``, labelled with the role of its group (the main frame of a group is marked with *)."""
    role_of: dict[int, str] = {}
    for v in views or []:
        for k, idx in enumerate(v.get("frames") or []):
            role_of[int(idx)] = f"{v.get('role')}{'*' if k == 0 else ''}"
    if sheet.kind == "animation":
        place = {f.index: (f.index // 8, f.index % 8) for f in sheet.frames}
    else:
        place = {f.index: (f.row, f.col) for f in sheet.frames}
    rows = max(r for r, _ in place.values()) + 1
    cols = max(c for _, c in place.values()) + 1
    pad, head = 6, 16
    out = Image.new("RGBA", (cols * (cell + pad) + pad, rows * (cell + head + pad) + pad), (22, 19, 26, 255))
    d = ImageDraw.Draw(out)
    font = _font(12)
    for f in sheet.frames:
        r, c = place[f.index]
        x = pad + c * (cell + pad)
        y = pad + r * (cell + head + pad)
        tile = Image.new("RGBA", (cell, cell), (58, 54, 64, 255))
        td = ImageDraw.Draw(tile)
        for ty in range(0, cell, 12):
            for tx in range(0, cell, 12):
                if (tx // 12 + ty // 12) % 2:
                    td.rectangle([tx, ty, tx + 11, ty + 11], fill=(68, 64, 74, 255))
        img = Image.fromarray(f.rgba, "RGBA")
        k = max(1, min((cell - 8) // max(1, img.width), (cell - 8) // max(1, img.height)))
        big = img.resize((img.width * k, img.height * k), Image.NEAREST)
        tile.alpha_composite(big, ((cell - big.width) // 2, cell - 4 - big.height))
        out.paste(tile, (x, y + head))
        label = f"{f.index}" + (f" {role_of[f.index]}" if f.index in role_of else "")
        d.text((x + 2, y + 1), label, fill=(235, 225, 200, 255), font=font)
    return out


def png_bytes(img: Image.Image) -> bytes:
    buf = io.BytesIO()
    img.save(buf, "PNG")
    return buf.getvalue()
