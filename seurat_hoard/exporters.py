"""Files out of a voxel model: MagicaVoxel ``.vox``, ``.glb`` (glTF binary with vertex colours), binary ``.stl`` for
printing, and PNG (one view, or isometric sprites).

GLB merges same-coloured coplanar faces into rectangles (greedy meshing). STL keeps one square per face so every edge
is shared by exactly two triangles: a closed mesh the slicer accepts without repairs.
"""

from __future__ import annotations

import json
import struct
from typing import Iterator

import numpy as np

from .errors import SeuratError
from .render import png_bytes, render, sprites
from .voxels import VoxelModel

FORMATS = {"vox": "application/octet-stream", "glb": "model/gltf-binary", "stl": "model/stl", "png": "image/png",
           "sprites": "image/png"}

# The six faces: normal axis, direction, and the two in-plane axes (u, v) chosen so (u x v) points along the normal.
FACES = [(0, +1, 1, 2), (0, -1, 2, 1), (1, +1, 2, 0), (1, -1, 0, 2), (2, +1, 0, 1), (2, -1, 1, 0)]


def _exposed(grid: np.ndarray, axis: int, sign: int) -> np.ndarray:
    occ = grid > 0
    shifted = np.zeros_like(occ)
    src = [slice(None)] * 3
    dst = [slice(None)] * 3
    if sign > 0:
        src[axis], dst[axis] = slice(1, None), slice(None, -1)
    else:
        src[axis], dst[axis] = slice(None, -1), slice(1, None)
    shifted[tuple(dst)] = occ[tuple(src)]
    return occ & ~shifted


def unit_quads(model: VoxelModel) -> Iterator[tuple[np.ndarray, np.ndarray]]:
    """Every exposed face as 4 corners (counter-clockwise seen from outside) and its normal."""
    g = model.grid
    for axis, sign, ua, va in FACES:
        mask = _exposed(g, axis, sign)
        pts = np.argwhere(mask)
        if not len(pts):
            continue
        normal = np.zeros(3)
        normal[axis] = sign
        for p in pts:
            base = p.astype(np.float64).copy()
            if sign > 0:
                base[axis] += 1
            du = np.zeros(3)
            dv = np.zeros(3)
            du[ua] = 1
            dv[va] = 1
            yield np.array([base, base + du, base + du + dv, base + dv]), normal


def greedy_quads(model: VoxelModel) -> list[tuple[np.ndarray, np.ndarray, int]]:
    """Exposed faces merged into rectangles per plane and colour: (corners (4,3), normal (3,), colour index)."""
    g = model.grid
    out = []
    for axis, sign, ua, va in FACES:
        mask = _exposed(g, axis, sign)
        colour = np.where(mask, g, 0)
        n = g.shape[axis]
        for d in range(n):
            idx = [slice(None)] * 3
            idx[axis] = d
            plane = colour[tuple(idx)]          # 2D over the two remaining axes in increasing order
            rem = [i for i in range(3) if i != axis]
            # make plane[u, v] with u = ua, v = va
            if rem != [ua, va]:
                plane = plane.T
            plane = plane.copy()
            nu, nv = plane.shape
            for u in range(nu):
                v = 0
                while v < nv:
                    c = plane[u, v]
                    if c == 0:
                        v += 1
                        continue
                    v2 = v
                    while v2 + 1 < nv and plane[u, v2 + 1] == c:
                        v2 += 1
                    u2 = u
                    while u2 + 1 < nu and (plane[u2 + 1, v: v2 + 1] == c).all():
                        u2 += 1
                    plane[u: u2 + 1, v: v2 + 1] = 0
                    base = np.zeros(3)
                    base[axis] = d + (1 if sign > 0 else 0)
                    base[ua], base[va] = u, v
                    du = np.zeros(3)
                    dv = np.zeros(3)
                    du[ua] = u2 - u + 1
                    dv[va] = v2 - v + 1
                    normal = np.zeros(3)
                    normal[axis] = sign
                    out.append((np.array([base, base + du, base + du + dv, base + dv]), normal, int(c)))
                    v = v2 + 1
    return out


# ---------------------------------------------------------------------------------------------------------------- .vox
def to_vox(model: VoxelModel) -> bytes:
    """MagicaVoxel 150. MagicaVoxel is z-up: our (x, y, z) with y up and z towards the viewer becomes (x, sz-1-z, y)."""
    sx, sy, sz = model.size
    pts = model.voxel_list()
    if len(pts) > 0xFFFFFFFF:
        raise SeuratError("too_big", "Too many voxels for .vox.")
    xyzi = bytearray(struct.pack("<I", len(pts)))
    for x, y, z, c in pts:
        xyzi += struct.pack("<BBBB", x, sz - 1 - z, y, c)
    size = struct.pack("<III", sx, sz, sy)
    rgba = bytearray()
    pal = model.palette_rgb()
    for i in range(1, 256):          # .vox palette entry k (0-based) is colour index k + 1
        r, g, b = (int(v) for v in pal[i])
        rgba += struct.pack("<BBBB", r, g, b, 255 if i <= len(model.palette) else 0)
    rgba += b"\x00\x00\x00\x00"

    def chunk(cid: bytes, content: bytes) -> bytes:
        return cid + struct.pack("<II", len(content), 0) + content

    children = chunk(b"SIZE", size) + chunk(b"XYZI", bytes(xyzi)) + chunk(b"RGBA", bytes(rgba))
    main = b"MAIN" + struct.pack("<II", 0, len(children)) + children
    return b"VOX " + struct.pack("<I", 150) + main


def from_vox(data: bytes, model_id: str, name: str) -> VoxelModel:
    """Read the first model of a MagicaVoxel file (z-up back to y-up)."""
    if data[:4] != b"VOX ":
        raise SeuratError("invalid", "Not a MagicaVoxel .vox file.")
    pos = 8
    size = None
    voxels: list[tuple[int, int, int, int]] = []
    palette = None

    def walk(p: int, end: int):
        nonlocal size, voxels, palette
        while p + 12 <= end:
            cid = data[p: p + 4]
            n, m = struct.unpack("<II", data[p + 4: p + 12])
            body = data[p + 12: p + 12 + n]
            if cid == b"SIZE" and size is None:
                size = struct.unpack("<III", body[:12])
            elif cid == b"XYZI" and not voxels:
                count = struct.unpack("<I", body[:4])[0]
                arr = np.frombuffer(body[4: 4 + 4 * count], dtype=np.uint8).reshape(-1, 4)
                voxels = [tuple(int(v) for v in row) for row in arr]
            elif cid == b"RGBA":
                rgba = np.frombuffer(body[:1024], dtype=np.uint8).reshape(-1, 4)
                palette = ["#%02x%02x%02x" % tuple(int(v) for v in row[:3]) for row in rgba[:255]]
            if cid == b"MAIN":
                walk(p + 12 + n, p + 12 + n + m)
            p += 12 + n + m
        return p

    walk(pos, len(data))
    if size is None:
        raise SeuratError("invalid", "The .vox file has no model.")
    mx, my, mz = size                  # MagicaVoxel x, y (depth), z (up)
    model = VoxelModel.empty(model_id, name, [min(mx, 256), min(mz, 256), min(my, 256)], palette or None)
    if palette:
        used = max((c for *_, c in voxels), default=1)
        model.palette = palette[: max(used, 1)]
    for x, y, z, c in voxels:
        X, Y, Z = x, z, my - 1 - y
        if 0 <= X < model.grid.shape[0] and 0 <= Y < model.grid.shape[1] and 0 <= Z < model.grid.shape[2]:
            model.grid[X, Y, Z] = c
    return model


# ---------------------------------------------------------------------------------------------------------------- .glb
def to_glb(model: VoxelModel, voxel_size: float = 0.1) -> bytes:
    quads = greedy_quads(model)
    if not quads:
        raise SeuratError("invalid", "The model is empty: nothing to export.")
    sx, sy, sz = model.size
    centre = np.array([sx / 2.0, 0.0, sz / 2.0])
    pal = model.palette_rgb().astype(np.float32) / 255.0
    pos, nor, col, idx = [], [], [], []
    for corners, normal, c in quads:
        base = len(pos)
        for p in corners:
            pos.append((p - centre) * voxel_size)
            nor.append(normal)
            rgb = pal[c] ** 2.2                        # glTF vertex colours are linear
            col.append([*rgb, 1.0])
        idx += [base, base + 1, base + 2, base, base + 2, base + 3]
    pos_a = np.asarray(pos, dtype=np.float32)
    nor_a = np.asarray(nor, dtype=np.float32)
    col_a = np.asarray(col, dtype=np.float32)
    idx_a = np.asarray(idx, dtype=np.uint32)
    blobs = [idx_a.tobytes(), pos_a.tobytes(), nor_a.tobytes(), col_a.tobytes()]
    offsets, binary = [], b""
    for blob in blobs:
        offsets.append(len(binary))
        binary += blob + b"\x00" * ((4 - len(blob) % 4) % 4)
    gltf = {
        "asset": {"version": "2.0", "generator": "Seurat's Hoard"},
        "scene": 0, "scenes": [{"nodes": [0]}], "nodes": [{"mesh": 0, "name": model.name}],
        "meshes": [{"name": model.name, "primitives": [{"attributes": {"POSITION": 1, "NORMAL": 2, "COLOR_0": 3}, "indices": 0, "material": 0}]}],
        "materials": [{"name": "voxels", "pbrMetallicRoughness": {"baseColorFactor": [1, 1, 1, 1], "metallicFactor": 0.0, "roughnessFactor": 0.9}}],
        "buffers": [{"byteLength": len(binary)}],
        "bufferViews": [
            {"buffer": 0, "byteOffset": offsets[0], "byteLength": len(blobs[0]), "target": 34963},
            {"buffer": 0, "byteOffset": offsets[1], "byteLength": len(blobs[1]), "target": 34962},
            {"buffer": 0, "byteOffset": offsets[2], "byteLength": len(blobs[2]), "target": 34962},
            {"buffer": 0, "byteOffset": offsets[3], "byteLength": len(blobs[3]), "target": 34962},
        ],
        "accessors": [
            {"bufferView": 0, "componentType": 5125, "count": len(idx_a), "type": "SCALAR"},
            {"bufferView": 1, "componentType": 5126, "count": len(pos_a), "type": "VEC3",
             "min": pos_a.min(axis=0).tolist(), "max": pos_a.max(axis=0).tolist()},
            {"bufferView": 2, "componentType": 5126, "count": len(nor_a), "type": "VEC3"},
            {"bufferView": 3, "componentType": 5126, "count": len(col_a), "type": "VEC4"},
        ],
    }
    js = json.dumps(gltf, separators=(",", ":")).encode()
    js += b" " * ((4 - len(js) % 4) % 4)
    total = 12 + 8 + len(js) + 8 + len(binary)
    return (struct.pack("<III", 0x46546C67, 2, total) + struct.pack("<II", len(js), 0x4E4F534A) + js
            + struct.pack("<II", len(binary), 0x004E4942) + binary)


# ---------------------------------------------------------------------------------------------------------------- .stl
def to_stl(model: VoxelModel, voxel_mm: float = 2.0) -> bytes:
    """Binary STL in millimetres (``voxel_mm`` per voxel), z up as slicers expect, resting on z = 0."""
    quads = list(unit_quads(model))
    if not quads:
        raise SeuratError("invalid", "The model is empty: nothing to export.")
    sx, sy, sz = model.size
    tris = []
    for corners, normal in quads:
        for a, b, c in ((0, 1, 2), (0, 2, 3)):
            tris.append((normal, corners[a], corners[b], corners[c]))
    out = bytearray(b"Seurat's Hoard STL".ljust(80, b" "))
    out += struct.pack("<I", len(tris))

    def conv(p):      # y-up -> z-up, keeping it right-handed: (x, y, z) -> (x, -z, y), shifted to positive
        return (p[0] * voxel_mm, (sz - p[2]) * voxel_mm, p[1] * voxel_mm)

    for normal, a, b, c in tris:
        n = (normal[0], -normal[2], normal[1])
        out += struct.pack("<12fH", *n, *conv(a), *conv(b), *conv(c), 0)
    return bytes(out)


# ---------------------------------------------------------------------------------------------------------------- png
def to_png(model: VoxelModel, view: str = "iso", size: int | None = None) -> bytes:
    if model.count == 0:
        raise SeuratError("invalid", "The model is empty: nothing to render.")
    return png_bytes(render(model, view, size=size))


def to_sprites(model: VoxelModel, size: int | None = None) -> bytes:
    if model.count == 0:
        raise SeuratError("invalid", "The model is empty: nothing to render.")
    return png_bytes(sprites(model, size))


def export(model: VoxelModel, fmt: str, **opts) -> bytes:
    if fmt == "vox":
        return to_vox(model)
    if fmt == "glb":
        return to_glb(model, float(opts.get("voxel_size") or 0.1))
    if fmt == "stl":
        return to_stl(model, float(opts.get("voxel_mm") or 2.0))
    if fmt == "png":
        return to_png(model, str(opts.get("view") or "iso"), opts.get("size"))
    if fmt == "sprites":
        return to_sprites(model, opts.get("size"))
    raise SeuratError("invalid", f"Unknown format {fmt!r}; use one of {', '.join(FORMATS)}.")
