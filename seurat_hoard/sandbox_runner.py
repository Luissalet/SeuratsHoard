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

    def column_fill(g, c, height, base=None):
        """Fill each ground column (x, z) from base (default 0) up to height[x, z] (2D arrays of shape (X, Z), or
        numbers): terrain, hills, plateaus, water levels."""
        h = np.broadcast_to(np.asarray(height, np.float32), (X, Z))[:, None, :]
        b = np.broadcast_to(np.asarray(0 if base is None else base, np.float32), (X, Z))[:, None, :]
        return _set(g, (ys >= b) & (ys <= h), c)

    def _set(g, mask, c):
        g[mask] = c
        return int(mask.sum())

    def ellipsoid(g, c, center, radii):
        """Fill an ellipsoid (centre [x,y,z], radii [rx,ry,rz]) with colour index c (0 erases)."""
        cx, cy, cz = center
        rx, ry, rz = (max(0.5, float(r)) for r in radii)
        return _set(g, ((xs - cx) / rx) ** 2 + ((ys - cy) / ry) ** 2 + ((zs - cz) / rz) ** 2 <= 1.0, c)

    def sphere(g, c, center, r):
        return ellipsoid(g, c, center, (r, r, r))

    def capsule(g, c, a, b, r0, r1=None):
        """A tube from point a to point b whose radius goes from r0 to r1 (necks, tails, limbs, feathers, horns)."""
        r1 = r0 if r1 is None else r1
        a = np.asarray(a, np.float32); b = np.asarray(b, np.float32)
        ab = b - a
        L2 = float(ab @ ab) or 1.0
        t = np.clip(((xs - a[0]) * ab[0] + (ys - a[1]) * ab[1] + (zs - a[2]) * ab[2]) / L2, 0.0, 1.0)
        px, py, pz = a[0] + t * ab[0], a[1] + t * ab[1], a[2] + t * ab[2]
        d2 = (xs - px) ** 2 + (ys - py) ** 2 + (zs - pz) ** 2
        r = r0 + (r1 - r0) * t
        return _set(g, d2 <= (r + 0.35) ** 2, c)

    def box(g, c, lo, hi):
        """Fill the box between corners lo and hi (inclusive)."""
        (x0, y0, z0), (x1, y1, z1) = lo, hi
        m = (xs >= min(x0, x1)) & (xs <= max(x0, x1)) & (ys >= min(y0, y1)) & (ys <= max(y0, y1)) \
            & (zs >= min(z0, z1)) & (zs <= max(z0, z1))
        return _set(g, m, c)

    def polygon(g, c, points, plane="top", lo=0, hi=0):
        """A flat polygon with thickness: points [[u,v],...] on plane top (u=x, v=z, thickness along y from lo to hi),
        front (u=x, v=y, along z) or side (u=z, v=y, along x). Wings, fins, ears, blades, tail fans."""
        pts = [(float(u), float(v)) for u, v in points]
        if plane == "top":
            U, V, W = xs, zs, ys
        elif plane == "front":
            U, V, W = xs, ys, zs
        else:
            U, V, W = zs, ys, xs
        inside = np.zeros(shape, bool)
        j = len(pts) - 1
        for i in range(len(pts)):
            (ui, vi), (uj, vj) = pts[i], pts[j]
            cond = ((vi > V) != (vj > V)) & (U < (uj - ui) * (V - vi) / ((vj - vi) or 1e-9) + ui)
            inside ^= cond
            j = i
        return _set(g, inside & (W >= min(lo, hi)) & (W <= max(lo, hi)), c)

    def mirror_x(g):
        """Copy the left half (low x) onto the right half: symmetric creatures."""
        half = X // 2
        g[X - half:, :, :] = g[:half, :, :][::-1, :, :]
        return g

    def shade_by_height(g, c, c_dark, below_y):
        """Recolour colour c to c_dark below height below_y (undersides)."""
        return _set(g, (g == c) & (ys < below_y), c_dark)

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
