"""Runs one voxel program written by the language model, in its own process (``python -I sandbox_runner.py``).

stdin: JSON ``{"code": str, "size": [X, Y, Z], "out": path}``. The program defines ``PALETTE`` (list of "#rrggbb",
index 1 = first colour) and ``build(g)``, which fills ``g`` (numpy uint8, shape (X, Y, Z), x right, y up, z towards the
viewer) with palette indices. The grid and the palette are written to ``out`` (``.npz``); stdout gets one JSON line
with the result or the error. Only numpy, math, random, itertools and functools can be imported; no files, no network,
no processes (the parent also checks the source and kills the process after its time limit).
"""

from __future__ import annotations

import builtins
import json
import math
import sys
import traceback

import numpy as np

ALLOWED_IMPORTS = {"math", "numpy", "random", "itertools", "functools", "colorsys"}
SAFE_BUILTINS = {name: getattr(builtins, name) for name in (
    "abs", "all", "any", "bool", "dict", "divmod", "enumerate", "filter", "float", "frozenset", "int", "isinstance",
    "len", "list", "map", "max", "min", "pow", "print", "range", "reversed", "round", "set", "slice", "sorted", "str",
    "sum", "tuple", "zip", "ValueError", "Exception", "IndexError", "KeyError", "TypeError", "ZeroDivisionError",
    "ArithmeticError", "AssertionError", "RuntimeError", "StopIteration", "True", "False", "None", "hasattr", "callable",
    "iter", "next", "complex", "object", "type", "super", "property", "staticmethod", "classmethod", "repr", "hex", "ord",
    "chr", "format", "id", "hash", "issubclass") if hasattr(builtins, name)}


def _import(name, globals=None, locals=None, fromlist=(), level=0):
    if name.split(".")[0] not in ALLOWED_IMPORTS:
        raise ImportError(f"import of {name!r} is not allowed (only {', '.join(sorted(ALLOWED_IMPORTS))})")
    return builtins.__import__(name, globals, locals, fromlist, level)


def helpers(shape):
    X, Y, Z = shape
    xs, ys, zs = np.indices(shape, dtype=np.float32)
    gx, gz = np.indices((X, Z), dtype=np.float32)          # 2D ground coordinates (X, Z) for height maps

    def _region(lo, hi):
        """Slices of the grid between lo and hi (clipped) and open-grid coordinates for that box, so a part only touches
        the voxels near it instead of the whole grid (keeps big grids fast)."""
        sl, co = [], []
        for axis, (a, b, n) in enumerate(zip(lo, hi, (X, Y, Z))):
            i0 = max(0, int(math.floor(min(a, b))))
            i1 = min(n, int(math.ceil(max(a, b))) + 1)
            if i1 <= i0:
                return None
            sl.append(slice(i0, i1))
            shp = [1, 1, 1]; shp[axis] = i1 - i0
            co.append(np.arange(i0, i1, dtype=np.float32).reshape(shp))
        return tuple(sl), co

    def _set_in(g, reg, mask, c):
        if reg is None:
            return 0
        view = g[reg[0]]
        mask = np.broadcast_to(mask, view.shape)
        view[mask] = c
        return int(mask.sum())

    def column_fill(g, c, height, base=None):
        """Fill each ground column (x, z) from base (default 0) up to height[x, z] (2D arrays of shape (X, Z), or
        numbers): terrain, hills, plateaus, water levels."""
        h = np.broadcast_to(np.asarray(height, np.float32), (X, Z))[:, None, :]
        b = np.broadcast_to(np.asarray(0 if base is None else base, np.float32), (X, Z))[:, None, :]
        yy = np.arange(Y, dtype=np.float32)[None, :, None]
        return _set(g, (yy >= b) & (yy <= h), c)

    def _set(g, mask, c):
        mask = np.broadcast_to(mask, g.shape)
        g[mask] = c
        return int(mask.sum())

    def ellipsoid(g, c, center, radii):
        """Fill an ellipsoid (centre [x,y,z], radii [rx,ry,rz]) with colour index c (0 erases)."""
        cx, cy, cz = (float(v) for v in center)
        rx, ry, rz = (max(0.5, float(r)) for r in radii)
        reg = _region((cx - rx, cy - ry, cz - rz), (cx + rx, cy + ry, cz + rz))
        if reg is None:
            return 0
        x, y, z = reg[1]
        return _set_in(g, reg, ((x - cx) / rx) ** 2 + ((y - cy) / ry) ** 2 + ((z - cz) / rz) ** 2 <= 1.0, c)

    def sphere(g, c, center, r):
        return ellipsoid(g, c, center, (r, r, r))

    def capsule(g, c, a, b, r0, r1=None):
        """A tube from point a to point b whose radius goes from r0 to r1 (necks, tails, limbs, feathers, horns)."""
        r1 = r0 if r1 is None else r1
        a = np.asarray(a, np.float32); b = np.asarray(b, np.float32)
        pad = max(float(r0), float(r1)) + 1.0
        reg = _region(np.minimum(a, b) - pad, np.maximum(a, b) + pad)
        if reg is None:
            return 0
        x, y, z = reg[1]
        ab = b - a
        L2 = float(ab @ ab) or 1.0
        t = np.clip(((x - a[0]) * ab[0] + (y - a[1]) * ab[1] + (z - a[2]) * ab[2]) / L2, 0.0, 1.0)
        d2 = (x - (a[0] + t * ab[0])) ** 2 + (y - (a[1] + t * ab[1])) ** 2 + (z - (a[2] + t * ab[2])) ** 2
        r = r0 + (r1 - r0) * t
        return _set_in(g, reg, d2 <= (r + 0.35) ** 2, c)

    def box(g, c, lo, hi):
        """Fill the box between corners lo and hi (inclusive)."""
        reg = _region([float(v) for v in lo], [float(v) for v in hi])
        if reg is None:
            return 0
        lo_i = [min(a, b) for a, b in zip(lo, hi)]; hi_i = [max(a, b) for a, b in zip(lo, hi)]
        x, y, z = reg[1]
        m = (x >= lo_i[0]) & (x <= hi_i[0]) & (y >= lo_i[1]) & (y <= hi_i[1]) & (z >= lo_i[2]) & (z <= hi_i[2])
        return _set_in(g, reg, m, c)

    def polygon(g, c, points, plane="top", lo=0, hi=0):
        """A flat polygon with thickness: points [[u,v],...] on plane top (u=x, v=z, thickness along y from lo to hi),
        front (u=x, v=y, along z) or side (u=z, v=y, along x). Wings, fins, ears, blades, tail fans."""
        pts = [(float(u), float(v)) for u, v in points]
        if not pts:
            return 0
        us = [p[0] for p in pts]; vs = [p[1] for p in pts]
        w0, w1 = float(min(lo, hi)), float(max(lo, hi))
        if plane == "top":        # u=x, v=z, w=y
            reg = _region((min(us), w0, min(vs)), (max(us), w1, max(vs)))
            pick = lambda x, y, z: (x, z, y)
        elif plane == "front":    # u=x, v=y, w=z
            reg = _region((min(us), min(vs), w0), (max(us), max(vs), w1))
            pick = lambda x, y, z: (x, y, z)
        else:                     # side: u=z, v=y, w=x
            reg = _region((w0, min(vs), min(us)), (w1, max(vs), max(us)))
            pick = lambda x, y, z: (z, y, x)
        if reg is None:
            return 0
        U, V, W = pick(*reg[1])
        inside = np.zeros(np.broadcast_shapes(U.shape, V.shape), bool)
        j = len(pts) - 1
        for i in range(len(pts)):
            (ui, vi), (uj, vj) = pts[i], pts[j]
            inside ^= ((vi > V) != (vj > V)) & (U < (uj - ui) * (V - vi) / ((vj - vi) or 1e-9) + ui)
            j = i
        return _set_in(g, reg, inside & (W >= w0) & (W <= w1), c)

    def mirror_x(g):
        """Copy the left half (low x) onto the right half: symmetric creatures."""
        half = X // 2
        g[X - half:, :, :] = g[:half, :, :][::-1, :, :]
        return g

    def shade_by_height(g, c, c_dark, below_y):
        """Recolour colour c to c_dark below height below_y (undersides)."""
        yy = np.arange(Y, dtype=np.float32)[None, :, None]
        return _set(g, (g == c) & (yy < below_y), c_dark)

    return {"X": X, "Y": Y, "Z": Z, "xs": xs, "ys": ys, "zs": zs, "gx": gx, "gz": gz, "column_fill": column_fill,
            "ellipsoid": ellipsoid, "sphere": sphere,
            "capsule": capsule, "box": box, "polygon": polygon, "mirror_x": mirror_x, "shade_by_height": shade_by_height}


def main() -> None:
    req = json.loads(sys.stdin.read())
    shape = tuple(int(v) for v in req["size"])
    env = {"__builtins__": dict(SAFE_BUILTINS, __import__=_import), "np": np, "numpy": np, "math": math,
           "__name__": "voxel_program"}
    env.update(helpers(shape))
    try:
        exec(compile(req["code"], "<voxel program>", "exec"), env)
        if "build" not in env or not callable(env["build"]):
            raise ValueError("the program must define build(g)")
        palette = env.get("PALETTE") or []
        if not isinstance(palette, (list, tuple)) or not palette:
            raise ValueError("the program must define PALETTE, a list of '#rrggbb' colours")
        g = np.zeros(shape, dtype=np.uint8)
        out = env["build"](g)
        if isinstance(out, np.ndarray) and out.shape == shape:
            g = out
        g = np.asarray(g)
        if g.shape != shape:
            raise ValueError(f"the grid changed shape to {g.shape}; keep {shape}")
        g = np.clip(g, 0, 255).astype(np.uint8)
        np.savez_compressed(req["out"], grid=g, palette=np.array([str(c) for c in palette][:255]))
        print(json.dumps({"ok": True, "count": int(np.count_nonzero(g))}))
    except BaseException as exc:  # noqa: BLE001
        src = req["code"].splitlines()
        where = []
        for fr in traceback.extract_tb(exc.__traceback__):
            if fr.filename == "<voxel program>" and 0 < (fr.lineno or 0) <= len(src):
                where.append(f"line {fr.lineno}, in {fr.name}: {src[fr.lineno - 1].strip()}")
        print(json.dumps({"ok": False, "error": f"{type(exc).__name__}: {exc}", "trace": "\n".join(where[-6:])[-1500:]}))


if __name__ == "__main__":
    main()
