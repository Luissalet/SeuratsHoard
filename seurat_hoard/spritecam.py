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
    role: str                      # front | back | side | view (any other side: its camera is found by calibration)
    name: str
    rgba: np.ndarray               # (h, w, 4), background removed, cropped, pixel-art enlargement undone
    yaw: Optional[float] = None    # the camera of this sprite when known (calibrated or given); None = the role's default
    pitch: Optional[float] = None
    poses: list = field(default_factory=list)      # other frames of the same side (walk steps): (h, w, 4) arrays

    def camera(self, front_back_pitch: float = 45.0, side_pitch: float = 0.0) -> tuple[Optional[float], float]:
        """(yaw, pitch) of this sprite's camera: its own when set, else the role's (side: yaw None = try both sides)."""
        pitch = self.pitch if self.pitch is not None else (side_pitch if self.role == "side" else front_back_pitch)
        if self.yaw is not None:
            return float(self.yaw), float(pitch)
        return {"front": 0.0, "back": 180.0}.get(self.role), float(pitch)


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
    pose_ious: list = field(default_factory=list)  # the other frames of this side (walk steps) against the same render

    @property
    def score(self) -> float:
        """The view's share of the overall match: the main frame, and with poses half main frame, half their mean."""
        if not self.pose_ious:
            return self.iou
        return 0.5 * self.iou + 0.5 * float(np.mean(self.pose_ious))

    def summary(self) -> str:
        text = self._summary()
        if self.pose_ious:
            text += (f" Its {len(self.pose_ious)} other frames (poses of the walk) match {min(self.pose_ious):.2f}-"
                     f"{max(self.pose_ious):.2f}: a static model in a neutral pose sits between them.")
        return text

    def _summary(self) -> str:
        where = {"front": "in front", "back": "behind", "side": "at the side"}.get(self.role, f"{self.yaw:.0f} degrees around")
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
                "size_ratio": [round(float(v), 3) for v in self.size_ratio],
                **({"poses": [round(float(v), 3) for v in self.pose_ious], "score": round(self.score, 3)} if self.pose_ious else {})}


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
    for pose in ref.poses or []:
        p_mask = pose[..., 3] >= 128
        if fit == "pixel":
            pm = _place(proj, p_mask)[..., 3] >= 128
        else:
            pm = _fit(r, pose.shape[:2], _bbox(p_mask) or (0, 0, pose.shape[1], pose.shape[0]))[..., 3] >= 128
        vm.pose_ious.append(float((p_mask & pm).sum()) / max(1, int((p_mask | pm).sum())))
    return vm


def match(model: VoxelModel, refs: list[SpriteRef], front_back_pitch: float = FRONT_BACK_PITCH,
          side_pitch: float = SIDE_PITCH, fit: str = "auto") -> list[ViewMatch]:
    """Each sprite against the model from its camera. A side sprite is tried from both sides and the better one kept
    (sheets draw one side and mirror it)."""
    out = []
    for ref in refs:
        yaw, pitch = ref.camera(front_back_pitch, side_pitch)
        if ref.role not in ("front", "back", "side", "view"):
            continue
        if yaw is not None:
            out.append(match_view(model, ref, yaw, pitch, fit))
        elif ref.role == "side":
            a, b = match_view(model, ref, -90.0, pitch, fit), match_view(model, ref, 90.0, pitch, fit)
            out.append(a if a.iou >= b.iou else b)
        else:                                       # a view whose camera is not known yet: the best yaw around
            best = calibrate_one(model, ref, front_back_pitch, side_pitch, fit=fit)
            out.append(match_view(model, ref, best["yaw"], best["pitch"], fit))
    return out


# ---------------------------------------------------------------------------------------------- camera calibration
class Silhouette:
    """A model's silhouette from any camera at one pixel per voxel, fast (no colours, no depth): what calibration
    compares with the sprites at dozens of camera angles."""

    def __init__(self, model: VoxelModel):
        occ = np.argwhere(model.grid > 0)
        self.empty = not len(occ)
        self.pts = (occ[:, None, :] + _SAMPLES[None]).reshape(-1, 3) if len(occ) else np.zeros((0, 3))
        self.model = model

    def mask(self, yaw: float, pitch: float) -> np.ndarray:
        from .render import camera_basis
        if self.empty:
            return np.zeros((1, 1), bool)
        _, r, u = camera_basis(yaw, pitch)
        pu, pv = self.pts @ r, -(self.pts @ u)
        ix = np.floor(pu - np.floor(pu.min())).astype(np.int64)
        iy = np.floor(pv - np.floor(pv.min())).astype(np.int64)
        h, w = int(iy.max()) + 1, int(ix.max()) + 1
        out = np.zeros(h * w, bool)
        out[iy * w + ix] = True
        return out.reshape(h, w)


def _mask_iou(m: np.ndarray, t_mask: np.ndarray, fit: str, search: int = 4) -> float:
    """Silhouette match of a model mask on a sprite mask: pixel fit = one voxel per pixel at the best shift; area fit =
    scaled to the sprite subject's area first."""
    tb = _bbox(t_mask)
    mb = _bbox(m)
    if tb is None or mb is None:
        return 0.0
    if fit == "area":
        crop = Image.fromarray((m[mb[1]:mb[3], mb[0]:mb[2]] * 255).astype(np.uint8), "L")
        f = float(np.sqrt(((tb[2] - tb[0]) * (tb[3] - tb[1])) / max(1, crop.width * crop.height)))
        crop = crop.resize((max(1, round(crop.width * f)), max(1, round(crop.height * f))), Image.BOX)
        m = np.asarray(crop) >= 128
        mb = _bbox(m)
        if mb is None:
            return 0.0
    h_t, w_t = t_mask.shape
    ox0 = int(round((tb[0] + tb[2]) / 2 - (mb[0] + mb[2]) / 2))
    oy0 = int(round((tb[1] + tb[3]) / 2 - (mb[1] + mb[3]) / 2))
    t_area = int(t_mask.sum())
    m_area = int(m.sum())
    best = 0.0
    for dy in range(-search, search + 1):
        for dx in range(-search, search + 1):
            ox, oy = ox0 + dx, oy0 + dy
            y0, x0 = max(0, oy), max(0, ox)
            y1, x1 = min(h_t, oy + m.shape[0]), min(w_t, ox + m.shape[1])
            inter = int((t_mask[y0:y1, x0:x1] & m[y0 - oy:y1 - oy, x0 - ox:x1 - ox]).sum()) if y1 > y0 and x1 > x0 else 0
            iou = inter / max(1, t_area + m_area - inter)
            if iou > best:
                best = iou
    return best


def _fit_mode(sil: Silhouette, t_mask: np.ndarray, yaw: float, pitch: float, fit: str) -> str:
    if fit != "auto":
        return fit
    pm, tb = _bbox(sil.mask(yaw, pitch)), _bbox(t_mask)
    if pm is None or tb is None:
        return "pixel"
    ratio = ((pm[2] - pm[0]) * (pm[3] - pm[1])) / max(1, (tb[2] - tb[0]) * (tb[3] - tb[1]))
    return "pixel" if 0.4 <= ratio <= 2.5 else "area"


#: how far calibration looks from each role's usual camera: (yaw centres, yaw half-range, pitch range)
SEARCH = {"front": ((0.0,), 20.0, (0.0, 80.0)), "back": ((180.0,), 20.0, (0.0, 80.0)),
          "side": ((-90.0, 90.0), 20.0, (0.0, 60.0)), "view": ((0.0,), 180.0, (0.0, 80.0))}
#: during a generation the camera may only move this far (the model is still being built: a wide search would bend the
#: camera to a wrong model)
SEARCH_GENERATION = {"front": ((0.0,), 10.0, (20.0, 70.0)), "back": ((180.0,), 10.0, (20.0, 70.0)),
                     "side": ((-90.0, 90.0), 10.0, (0.0, 30.0)), "view": ((0.0,), 180.0, (0.0, 70.0))}


def calibrate_one(model: VoxelModel, ref: SpriteRef, front_back_pitch: float = FRONT_BACK_PITCH, side_pitch: float = SIDE_PITCH,
                  fit: str = "auto", search: Optional[dict] = None, sil: Optional[Silhouette] = None) -> dict[str, Any]:
    """The camera (yaw, pitch) from which the model's silhouette matches this sprite best: a coarse sweep around the
    role's usual camera, then finer steps around the best. Returns the best and what the current camera gave."""
    sil = sil or Silhouette(model)
    t_mask = ref.rgba[..., 3] >= 128
    centres, half, (p_lo, p_hi) = (search or SEARCH).get(ref.role, SEARCH["view"])
    cur_yaw, cur_pitch = ref.camera(front_back_pitch, side_pitch)
    fit_m = _fit_mode(sil, t_mask, cur_yaw if cur_yaw is not None else centres[0], cur_pitch, fit)
    cache: dict[tuple[float, float], float] = {}

    def score(yaw: float, pitch: float) -> float:
        key = (round(yaw % 360.0, 2), round(pitch, 2))
        if key not in cache:
            cache[key] = _mask_iou(sil.mask(yaw, pitch), t_mask, fit_m)
        return cache[key]

    if cur_yaw is None:
        before = max((score(c, cur_pitch), c) for c in centres)
        before_iou, before_yaw = before
    else:
        before_iou, before_yaw = score(cur_yaw, cur_pitch), cur_yaw
    ystep = 15.0 if half >= 90 else 5.0
    pstep = 10.0 if half >= 90 else 5.0
    cands = []
    for c in centres:
        for dy in np.arange(-half, half + 1e-6, ystep):
            if half >= 180 and dy >= 180:
                continue
            for p in np.arange(p_lo, p_hi + 1e-6, pstep):
                cands.append((score(c + dy, p), c + dy, p))
    cands.sort(reverse=True)
    best_s, best_y, best_p = cands[0]
    for ys, ps in ((ystep / 2.5, pstep / 2.5), (1.0, 1.0)):
        local = []
        for dy in (-2 * ys, -ys, 0.0, ys, 2 * ys):
            for dp in (-2 * ps, -ps, 0.0, ps, 2 * ps):
                y, p = best_y + dy, min(p_hi, max(p_lo, best_p + dp))
                if half < 180 and min(abs(((y - c + 180) % 360) - 180) for c in centres) > half + 1e-6:
                    continue
                local.append((score(y, p), y, p))
        best_s, best_y, best_p = max(local + [(best_s, best_y, best_p)])
    yaw = ((best_y + 180.0) % 360.0) - 180.0
    return {"role": ref.role, "name": ref.name, "yaw": round(float(yaw), 1), "pitch": round(float(best_p), 1),
            "iou": round(float(best_s), 3), "before": {"yaw": None if cur_yaw is None else round(float(before_yaw), 1),
                                                       "pitch": round(float(cur_pitch), 1), "iou": round(float(before_iou), 3)},
            "gain": round(float(best_s - before_iou), 3), "fit": fit_m}


def calibrate(model: VoxelModel, refs: list[SpriteRef], front_back_pitch: float = FRONT_BACK_PITCH,
              side_pitch: float = SIDE_PITCH, fit: str = "auto", search: Optional[dict] = None) -> list[dict[str, Any]]:
    """:func:`calibrate_one` for every sprite (the model's silhouette points are computed once)."""
    if model.count == 0:
        return []
    sil = Silhouette(model)
    return [calibrate_one(model, r, front_back_pitch, side_pitch, fit, search, sil) for r in refs
            if r.role in ("front", "back", "side", "view")]


def apply_cameras(refs: list[SpriteRef], cams: list[dict[str, Any]], min_gain: float = 0.0) -> list[str]:
    """Give each sprite its calibrated camera when it improves the match by at least ``min_gain`` (a sprite without a
    known camera always takes it). Returns a line per change."""
    changes = []
    for ref, cam in zip([r for r in refs if r.role in ("front", "back", "side", "view")], cams):
        if ref.yaw is None and ref.role == "view" or cam["gain"] >= min_gain:
            moved = ref.yaw is None or abs(((cam["yaw"] - ref.yaw + 180) % 360) - 180) > 0.4 or ref.pitch is None \
                or abs(cam["pitch"] - ref.pitch) > 0.4
            ref.yaw, ref.pitch = cam["yaw"], cam["pitch"]
            if moved:
                changes.append(f"{ref.role} '{ref.name}': yaw {cam['yaw']:.0f}, {cam['pitch']:.0f} degrees above "
                               f"(match {cam['before']['iou']:.2f} -> {cam['iou']:.2f})")
    return changes


def pitches(cams: list[dict[str, Any]]) -> dict[str, Optional[int]]:
    """Settings that follow from a calibration: the front/back camera height (mean of the front and back sprites) and the
    side camera height."""
    fb = [c["pitch"] for c in cams if c["role"] in ("front", "back")]
    sd = [c["pitch"] for c in cams if c["role"] == "side"]
    return {"sprite_pitch": int(round(float(np.mean(fb)))) if fb else None,
            "sprite_side_pitch": int(round(float(np.mean(sd)))) if sd else None}


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
    measure = ImageDraw.Draw(Image.new("RGBA", (4, 4)))
    for vm in matches:
        labels = [f"{vm.role} sprite", f"model, camera {vm.pitch:.0f} deg above", f"match {vm.iou:.2f} (red missing, blue extra)"]
        # each column as wide as its picture or its label, so small sprites do not overlap their labels
        widths = [max(p.width, int(measure.textlength(t, font=font)) + 4) for p, t in zip(vm.panels, labels)]
        w = sum(widths) + 8 * (len(vm.panels) + 1)
        h = max(p.height for p in vm.panels) + head + 8
        row = Image.new("RGBA", (w, h), BG_DARK)
        d = ImageDraw.Draw(row)
        x = 8
        for p, text, cw in zip(vm.panels, labels, widths):
            row.paste(p, (x, head))
            d.text((x, 3), text, fill=(235, 225, 200, 255), font=font)
            x += cw + 8
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
    per_role: dict[str, int] = {}
    for m in matches:
        per_role[m.role] = per_role.get(m.role, 0) + 1
    # a role's weight is shared by its sprites (two side sprites of a walk sheet weigh as much as one)
    w = [WEIGHTS.get(m.role, 1.0) / per_role[m.role] for m in matches]
    return float(np.average([m.score for m in matches], weights=w))


def board(refs: list[SpriteRef], front_back_pitch: float = FRONT_BACK_PITCH, side_pitch: float = SIDE_PITCH,
          zoom: int = ZOOM) -> Image.Image:
    """The target sprites side by side at a readable size, each labelled with its camera."""
    font = _font(13)
    head = 20
    tiles = []
    for ref in refs:
        yaw, pitch = ref.camera(front_back_pitch, side_pitch)
        where = {"front": "facing the camera", "back": "from behind", "side": "from the side"}.get(
            ref.role, f"from {yaw:.0f} deg around" if yaw is not None else "camera to be found")
        main = _big(ref.rgba, zoom)
        if ref.poses:
            # the other frames of the walk under the main one, smaller
            small = [_big(p, max(1, zoom // 2)) for p in ref.poses[:8]]
            strip_w = sum(s.width for s in small) + 4 * (len(small) - 1)
            tile = Image.new("RGBA", (max(main.width, strip_w), main.height + 18 + max(s.height for s in small)), BG_DARK)
            tile.paste(main, (0, 0))
            ImageDraw.Draw(tile).text((0, main.height + 2), f"{len(ref.poses)} more frames (poses):", fill=(200, 190, 170, 255), font=_font(11))
            sx = 0
            for s in small:
                tile.paste(s, (sx, main.height + 18))
                sx += s.width + 4
            main = tile
        tiles.append((f"{ref.role.upper()} sprite ({where}, camera {pitch:.0f} deg above)", main))
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
    longest = max(max(max(a.shape[:2]) for a in [r.rgba] + list(r.poses)) for r in refs)
    x = max((a.shape[1] for r in fb for a in [r.rgba] + list(r.poses)), default=longest)
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
        yaw, pitch = ref.camera(front_back_pitch, side_pitch)
        if yaw is None:
            yaw = -90.0 if ref.role == "side" else 0.0
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
