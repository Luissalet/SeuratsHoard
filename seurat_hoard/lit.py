"""Lit renders of a voxel model without a GPU: a key light with cast shadows, ambient occlusion in the creases and a
ground that catches the shadow, like the editor's three.js view. They are what the review rounds and ``look`` show the
model, and the frames of the spin animations (GIF / WebM).

How: every visible voxel face gets its lighting at its four corners (direct light times a shadow ray marched through
the grid towards the light, plus sky light times the occlusion of the voxels around it). The light is fixed in the
world, so it is computed once per model and every camera (each frame of a spin) only projects: each face is sampled
as a small grid of points, coloured by interpolating its corners, and the nearest point wins each pixel.
"""

from __future__ import annotations

import io
import shutil
import subprocess
import tempfile
from pathlib import Path
from typing import Optional

import numpy as np
from PIL import Image

from .errors import SeuratError
from .render import BG_DARK, camera_basis
from .voxels import VoxelModel

LIGHT = np.array([-0.45, 1.0, 0.62]) / np.linalg.norm([-0.45, 1.0, 0.62])   # from above, front-left
GROUND = np.array([0.56, 0.54, 0.58])
AMBIENT, DIFFUSE = 0.46, 0.72
DENSITY = 1.25            # samples per pixel along each side of a face (above 1: no holes)
POINT_BUDGET = 4_000_000  # samples per picture before the 2x supersampling is dropped

#: (normal axis, sign) of the six faces and two tangent axes for each
_FACES = [(0, 1), (0, -1), (1, 1), (1, -1), (2, 1), (2, -1)]
#: lit views: (yaw, pitch) like the isometric ones (yaw 45 = from the front-right)
VIEWS = {"lit": (45.0, 30.0), "lit_front_left": (-45.0, 30.0), "lit_back_left": (-135.0, 30.0),
         "lit_back_right": (135.0, 30.0), "lit_front": (0.0, 15.0), "lit_top": (20.0, 70.0)}


def _tangents(axis: int) -> tuple[int, int]:
    return [(1, 2), (0, 2), (0, 1)][axis]


class LitScene:
    """A model with its lighting baked on every exposed face (and a ground plane under it)."""

    def __init__(self, model: VoxelModel, ground: bool = True, light: np.ndarray = LIGHT):
        self.model = model
        self.light = np.asarray(light, np.float64) / np.linalg.norm(light)
        g = model.grid
        self.occ = g > 0
        self.size = np.array(g.shape)
        pal = model.palette_rgb().astype(np.float64) / 255.0
        origins, e1s, e2s, normals, albedo, kinds = [], [], [], [], [], []
        pad = np.pad(self.occ, 1)
        sx, sy, sz = g.shape
        for axis, sign in _FACES:
            sl = [slice(1, -1)] * 3
            sl[axis] = slice(2, None) if sign > 0 else slice(None, -2)
            exposed = self.occ & ~pad[tuple(sl)]
            vox = np.argwhere(exposed)
            if not len(vox):
                continue
            t1, t2 = _tangents(axis)
            o = vox.astype(np.float64).copy()
            if sign > 0:
                o[:, axis] += 1.0
            n = np.zeros(3)
            n[axis] = sign
            e1, e2 = np.zeros(3), np.zeros(3)
            e1[t1], e2[t2] = 1.0, 1.0
            origins.append(o)
            e1s.append(np.repeat(e1[None], len(o), 0))
            e2s.append(np.repeat(e2[None], len(o), 0))
            normals.append(np.repeat(n[None], len(o), 0))
            albedo.append(pal[g[vox[:, 0], vox[:, 1], vox[:, 2]]])
            kinds.append(np.zeros(len(o), np.int8))
        # what is drawn is framed on the voxels there are, not on the whole grid
        filled = np.argwhere(self.occ)
        self.lo = filled.min(axis=0).astype(np.float64) if len(filled) else np.zeros(3)
        self.hi = filled.max(axis=0).astype(np.float64) + 1 if len(filled) else self.size.astype(np.float64)
        self.lo[1] = 0.0                                    # the model stands on the floor
        if ground:
            # a ground at y = 0 around the model (30% of its footprint of margin), fading out at the edges
            x0, z0, x1, z1 = int(self.lo[0]), int(self.lo[2]), int(self.hi[0]), int(self.hi[2])
            m = int(max(4, round(0.3 * max(x1 - x0, z1 - z0))))
            gx, gz = np.meshgrid(np.arange(x0 - m, x1 + m), np.arange(z0 - m, z1 + m), indexing="ij")
            gx, gz = gx.ravel(), gz.ravel()
            inside = (gx >= 0) & (gx < sx) & (gz >= 0) & (gz < sz)
            covered = np.zeros(len(gx), bool)
            covered[inside] = self.occ[gx[inside], 0, gz[inside]]
            gx, gz = gx[~covered], gz[~covered]
            o = np.stack([gx, np.zeros(len(gx)), gz], 1).astype(np.float64)
            origins.append(o)
            e1s.append(np.repeat(np.array([[1.0, 0, 0]]), len(o), 0))
            e2s.append(np.repeat(np.array([[0, 0, 1.0]]), len(o), 0))
            normals.append(np.repeat(np.array([[0, 1.0, 0]]), len(o), 0))
            albedo.append(np.repeat(GROUND[None], len(o), 0))
            kinds.append(np.ones(len(o), np.int8))
            self.margin = m
        else:
            self.margin = 0
        if not origins:
            raise SeuratError("invalid", "The model is empty.")
        self.o = np.concatenate(origins)
        self.e1 = np.concatenate(e1s)
        self.e2 = np.concatenate(e2s)
        self.n = np.concatenate(normals)
        self.albedo = np.concatenate(albedo)
        self.kind = np.concatenate(kinds)           # 0 = a voxel face, 1 = ground
        self._bake()

    # ------------------------------------------------------------------ lighting
    def _occ_at(self, p: np.ndarray) -> np.ndarray:
        """Occupancy at points (N, 3) (floor to the voxel; outside the grid is empty)."""
        i = np.floor(p).astype(np.int64)
        ok = np.all((i >= 0) & (i < self.size), axis=1)
        out = np.zeros(len(p), bool)
        if ok.any():
            j = i[ok]
            out[ok] = self.occ[j[:, 0], j[:, 1], j[:, 2]]
        return out

    def _shadow(self, p: np.ndarray) -> np.ndarray:
        """1 where the light reaches the point, 0 where a voxel is in the way (a ray marched through the grid)."""
        lit = np.ones(len(p))
        step = 0.7
        n_steps = int(np.ceil(np.linalg.norm(self.size) / step)) + 2
        alive = np.arange(len(p))
        pos = p.copy()
        for _ in range(n_steps):
            if not len(alive):
                break
            pos[alive] += self.light * step
            q = pos[alive]
            hit = self._occ_at(q)
            lit[alive[hit]] = 0.0
            inside = np.all((q >= -1) & (q <= self.size + 1), axis=1) & ~hit
            # once above the model nothing can block any more
            inside &= q[:, 1] <= self.size[1] + 0.5
            alive = alive[inside]
        return lit

    def _sky(self, p: np.ndarray, n: np.ndarray) -> np.ndarray:
        """Share of the sky a point sees (ambient occlusion): short rays in a hemisphere around its normal."""
        dirs = np.array([[0, 1, 0], [0.7, 0.7, 0], [-0.7, 0.7, 0], [0, 0.7, 0.7], [0, 0.7, -0.7],
                         [0.5, 0.5, 0.5], [-0.5, 0.5, 0.5], [0.5, 0.5, -0.5], [-0.5, 0.5, -0.5]], np.float64)
        dirs /= np.linalg.norm(dirs, axis=1)[:, None]
        seen = np.zeros(len(p))
        total = np.zeros(len(p))
        ax = np.argmax(np.abs(n), axis=1)
        sg = np.sign(n[np.arange(len(n)), ax])
        for d0 in dirs:
            # the hemisphere turned to each face's normal: its "up" (d0[1]) goes along the normal, the rest along the face
            d = np.zeros((len(p), 3))
            for a in range(3):
                sel = ax == a
                if not sel.any():
                    continue
                t1, t2 = _tangents(a)
                d[sel, a] = d0[1] * sg[sel]
                d[sel, t1] = d0[0]
                d[sel, t2] = d0[2]
            free = np.ones(len(p), bool)
            for k in (0.6, 1.4, 2.5, 4.0):
                free &= ~self._occ_at(p + d * k)
            w = d0[1] + 0.3
            seen += free * w
            total += w
        return seen / total

    def _bake(self) -> None:
        """Lighting at the four corners of every face (slightly inside the face, off its surface)."""
        corners = [(0.12, 0.12), (0.88, 0.12), (0.12, 0.88), (0.88, 0.88)]
        n_faces = len(self.o)
        light = np.zeros((n_faces, 4))
        heavy = n_faces > 160_000
        ndotl = np.clip(self.n @ self.light, 0.0, 1.0)
        for k, (a, b) in enumerate(corners):
            if heavy and k > 0:
                light[:, k] = light[:, 0]
                continue
            pa, pb = (0.5, 0.5) if heavy else (a, b)
            p = self.o + self.e1 * pa + self.e2 * pb + self.n * 0.04
            shadow = np.zeros(n_faces)
            sel = ndotl > 0
            for s in range(0, int(sel.sum()), 120_000):
                idx = np.flatnonzero(sel)[s:s + 120_000]
                shadow[idx] = self._shadow(p[idx])
            sky = np.ones(n_faces)
            for s in range(0, n_faces, 120_000):
                sky[s:s + 120_000] = self._sky(p[s:s + 120_000] + self.n[s:s + 120_000] * 0.02, self.n[s:s + 120_000])
            hemi = 0.82 + 0.18 * self.n[:, 1]
            light[:, k] = AMBIENT * hemi * (0.35 + 0.65 * sky) + DIFFUSE * ndotl * shadow
        self.light_c = light
        # ground fades out towards its edges
        alpha = np.ones((n_faces, 4))
        if self.margin:
            g = self.kind == 1
            for k, (a, b) in enumerate(corners):
                p = self.o[g] + self.e1[g] * a + self.e2[g] * b
                dx = np.maximum(0, np.maximum(self.lo[0] - p[:, 0], p[:, 0] - self.hi[0]))
                dz = np.maximum(0, np.maximum(self.lo[2] - p[:, 2], p[:, 2] - self.hi[2]))
                alpha[g, k] = np.clip(1.0 - np.hypot(dx, dz) / self.margin, 0, 1) ** 1.5
        self.alpha_c = alpha

    # ------------------------------------------------------------------ pictures
    def extent(self, yaw: float, pitch: float) -> tuple[float, float, float, float]:
        """Screen bounds (u0, u1, v0, v1) of the model (not the ground) from that camera, in voxel units."""
        _, r, u = camera_basis(yaw, pitch)
        (x0, y0, z0), (x1, y1, z1) = self.lo, self.hi            # the voxels there are, standing on the floor
        corner = np.array([[x, y, z] for x in (x0, x1) for y in (y0, y1) for z in (z0, z1)], np.float64)
        us, vs = corner @ r, corner @ u
        return float(us.min()), float(us.max()), float(vs.min()), float(vs.max())

    def render(self, yaw: float, pitch: float, scale: float, background=BG_DARK, *, canvas: Optional[tuple[int, int]] = None,
               centre: Optional[np.ndarray] = None, supersample: Optional[int] = None, ground: bool = True) -> Image.Image:
        """The model from a camera ``yaw`` degrees around and ``pitch`` above (see ``render.camera_basis``), ``scale``
        pixels per voxel. ``canvas`` and ``centre`` fix the frame (spins); otherwise it is cropped to the model."""
        c, r, u = camera_basis(yaw, pitch)
        vis = (self.n @ c) > 1e-6
        if not ground:
            vis &= self.kind == 0
        faces = np.flatnonzero(vis)
        if supersample is None:
            # smoother edges when the points stay within budget (each face is sampled k x k times)
            k2 = np.ceil(DENSITY * scale * 2)
            supersample = 2 if len(faces) * k2 * k2 <= POINT_BUDGET else 1
        ss = max(1, int(supersample))
        s = scale * ss
        if canvas is None:
            u0, u1, v0, v1 = self.extent(yaw, pitch)
            pad = 0.16 * max(u1 - u0, v1 - v0) + 1          # room for the ground and the cast shadow
            u0, u1, v0, v1 = u0 - pad, u1 + pad, v0 - pad, v1 + pad
        else:
            ctr = self.size / 2.0 if centre is None else centre
            cu, cv = float(ctr @ r), float(ctr @ u)
            hw, hh = canvas[0] / (2 * scale), canvas[1] / (2 * scale)
            u0, u1, v0, v1 = cu - hw, cu + hw, cv - hh, cv + hh
        W, H = int(np.ceil((u1 - u0) * s)), int(np.ceil((v1 - v0) * s))
        if canvas is not None:
            W, H = canvas[0] * ss, canvas[1] * ss
        rgb = np.zeros((W * H, 3), np.float32)
        alpha = np.zeros(W * H, np.float32)
        # painter's order: faces from the farthest to the nearest (unit squares on a grid: their centres order them),
        # so a later sample simply covers an earlier one
        centre_d = (self.o[faces] + 0.5 * self.e1[faces] + 0.5 * self.e2[faces]) @ c
        faces = faces[np.argsort(centre_d, kind="stable")]
        k = int(min(48, max(2, np.ceil(DENSITY * s))))
        t = ((np.arange(k) + 0.5) / k).astype(np.float32)
        tu, tv = np.meshgrid(t, t, indexing="ij")
        tu, tv = tu.ravel(), tv.ravel()                                  # (k*k,)
        w00, w10, w01, w11 = (1 - tu) * (1 - tv), tu * (1 - tv), (1 - tu) * tv, tu * tv
        edge = np.minimum(np.minimum(tu, 1 - tu), np.minimum(tv, 1 - tv)) < 0.06
        chunk = max(1, int(1_500_000 // (k * k)))
        r32, u32 = r.astype(np.float32), u.astype(np.float32)
        for start in range(0, len(faces), chunk):
            f = faces[start:start + chunk]
            o, e1, e2 = self.o[f].astype(np.float32), self.e1[f].astype(np.float32), self.e2[f].astype(np.float32)
            # screen position of each sample: the face origin's plus the tangents' (no 3-D points needed)
            ou, ov = o @ r32, o @ u32
            px = np.floor(((ou[:, None] + (e1 @ r32)[:, None] * tu[None] + (e2 @ r32)[:, None] * tv[None]) - u0) * s).astype(np.int64).ravel()
            py = np.floor((v1 - (ov[:, None] + (e1 @ u32)[:, None] * tu[None] + (e2 @ u32)[:, None] * tv[None])) * s).astype(np.int64).ravel()
            ok = (px >= 0) & (px < W) & (py >= 0) & (py < H)
            L = self.light_c[f].astype(np.float32)
            lt = (L[:, 0:1] * w00 + L[:, 1:2] * w10 + L[:, 2:3] * w01 + L[:, 3:4] * w11)        # (F, k*k)
            A = self.alpha_c[f].astype(np.float32)
            al = (A[:, 0:1] * w00 + A[:, 1:2] * w10 + A[:, 2:3] * w01 + A[:, 3:4] * w11)
            # a faint line between voxels so the blocks read, as in the editor
            lt = lt * np.where(edge[None, :] & (self.kind[f] == 0)[:, None], np.float32(0.86), np.float32(1.0))
            col = (self.albedo[f].astype(np.float32)[:, None, :] * lt[..., None]).reshape(-1, 3)
            al = al.reshape(-1)
            key = py * W + px
            key, col, al = key[ok], col[ok], al[ok]
            order = np.argsort(key, kind="stable")                      # keeps the far-to-near order within a pixel
            last = np.r_[key[order][1:] != key[order][:-1], True]
            win = order[last]
            rgb[key[win]] = col[win]
            alpha[key[win]] = al[win]
        bg = np.array(background[:3], np.float64) / 255.0
        bga = (background[3] if len(background) > 3 else 255) / 255.0
        if bga > 0:
            out_rgb = rgb * alpha[:, None] + bg[None] * (1 - alpha[:, None])
            out_a = np.clip(alpha + bga * (1 - alpha), 0, 1)
        else:                                                         # transparent: straight colour, the alpha apart
            out_rgb, out_a = rgb, alpha
        img = np.zeros((H, W, 4), np.uint8)
        img[..., :3] = np.clip(np.power(np.clip(out_rgb, 0, 1), 1 / 1.08) * 255, 0, 255).reshape(H, W, 3).astype(np.uint8)
        img[..., 3] = (out_a * 255).reshape(H, W).astype(np.uint8)
        out = Image.fromarray(img, "RGBA")
        if ss > 1:
            out = out.resize((max(1, W // ss), max(1, H // ss)), Image.LANCZOS)
        return out


def lit(model: VoxelModel, yaw: float = 45.0, pitch: float = 30.0, target: int = 512, background=BG_DARK,
        ground: bool = True) -> Image.Image:
    """A lit picture about ``target`` pixels on its long side."""
    scene = LitScene(model, ground=ground)
    u0, u1, v0, v1 = scene.extent(yaw, pitch)
    scale = target / max(1e-6, 1.34 * max(u1 - u0, v1 - v0))
    return scene.render(yaw, pitch, scale, background, ground=ground)


def view(model: VoxelModel, name: str, target: int = 512, background=BG_DARK) -> Image.Image:
    yaw, pitch = VIEWS[name]
    return lit(model, yaw, pitch, target, background)


# ---------------------------------------------------------------------------------------------- spin animations
SPIN_FORMATS = {"gif": "image/gif", "webm": "video/webm"}


def spin_frames(model: VoxelModel, frames: int = 36, pitch: float = 25.0, size: int = 384, background=BG_DARK,
                ground: bool = True, start: float = 45.0) -> list[Image.Image]:
    """The model turning once around (the camera circles it with the light fixed in the world), every frame the same
    size and centre so it does not jump."""
    if model.count == 0:
        raise SeuratError("invalid", "The model is empty.")
    scene = LitScene(model, ground=ground)
    sx, sy, sz = (float(v) for v in scene.hi - scene.lo)           # the voxels there are, not the whole grid
    diag = float(np.hypot(sx, sz))
    pr = np.radians(pitch)
    height = sy * np.cos(pr) + diag * np.sin(pr)
    scale = size / max(1e-6, 1.1 * max(diag, height))
    w = int(size) // 2 * 2
    h = int(np.ceil(height * scale * 1.1)) // 2 * 2 + 2
    centre = (scene.lo + scene.hi) / 2.0
    out = []
    for i in range(max(2, frames)):
        yaw = start + 360.0 * i / max(2, frames)
        out.append(scene.render(yaw, pitch, scale, background, canvas=(w, h), centre=centre, ground=ground))
    return out


def ffmpeg_path() -> Optional[str]:
    try:
        import imageio_ffmpeg  # type: ignore
        return imageio_ffmpeg.get_ffmpeg_exe()
    except Exception:  # noqa: BLE001 - optional
        return shutil.which("ffmpeg")


def spin(model: VoxelModel, fmt: str = "gif", frames: int = 36, pitch: float = 25.0, size: int = 384, seconds: float = 3.0,
         ground: bool = True) -> bytes:
    """A turntable animation: ``gif`` (loops everywhere) or ``webm`` (VP9, smaller and smoother; needs ffmpeg)."""
    if fmt not in SPIN_FORMATS:
        raise SeuratError("invalid", f"Unknown animation format {fmt!r}; use gif or webm.")
    imgs = spin_frames(model, frames, pitch, size, BG_DARK, ground)
    ms = max(20, int(round(1000 * seconds / len(imgs))))
    if fmt == "gif":
        pal = [im.convert("RGB").quantize(colors=255, method=Image.Quantize.MEDIANCUT, dither=Image.Dither.NONE) for im in imgs]
        buf = io.BytesIO()
        pal[0].save(buf, "GIF", save_all=True, append_images=pal[1:], duration=ms, loop=0, disposal=2, optimize=False)
        return buf.getvalue()
    exe = ffmpeg_path()
    if not exe:
        raise SeuratError("unavailable", "WebM needs ffmpeg, which is not on this PC.", "Install ffmpeg (winget install ffmpeg) or export a GIF.")
    w, h = imgs[0].size
    raw = b"".join(im.convert("RGB").tobytes() for im in imgs)
    fps = max(1.0, len(imgs) / max(0.1, seconds))
    with tempfile.TemporaryDirectory(prefix="seurat-spin-") as tmp:
        out = Path(tmp) / "spin.webm"
        cmd = [exe, "-hide_banner", "-loglevel", "error", "-y", "-f", "rawvideo", "-pix_fmt", "rgb24", "-s", f"{w}x{h}",
               "-r", f"{fps:.3f}", "-i", "-", "-c:v", "libvpx-vp9", "-b:v", "0", "-crf", "32", "-pix_fmt", "yuv420p",
               "-row-mt", "1", "-deadline", "good", str(out)]
        try:
            proc = subprocess.run(cmd, input=raw, capture_output=True, timeout=300)
        except subprocess.TimeoutExpired:
            raise SeuratError("failed", "ffmpeg took too long to write the WebM.") from None
        if proc.returncode != 0 or not out.exists():
            raise SeuratError("failed", "ffmpeg could not write the WebM: " + (proc.stderr or b"").decode("utf-8", "replace")[-300:])
        return out.read_bytes()
