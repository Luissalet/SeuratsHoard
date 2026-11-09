"""Voxel programs: the language model writes Python that fills the grid (loops, curves, symmetry), Seurat runs it in
a separate process with a time limit (``sandbox_runner.py``) and gets the grid and the palette back.

Writing code is how models build good voxel art: a wing becomes a loop of tapered feathers, a body a few blended
ellipsoids, symmetry one call. The fixed operation list (``voxels.apply_ops``) stays for small edits.
"""

from __future__ import annotations

import json
import os
import re
import subprocess
import sys
import tempfile
from dataclasses import dataclass
from pathlib import Path
from typing import Optional

import numpy as np

from .voxels import norm_hex

RUNNER = Path(__file__).with_name("sandbox_runner.py")
MAX_CODE = 60_000
#: Source the program may not contain: files, processes, introspection, numpy's file functions.
FORBIDDEN = re.compile(r"\b(open|exec|eval|compile|globals|locals|vars|getattr|setattr|delattr|input|breakpoint|"
                       r"os|sys|subprocess|socket|shutil|pathlib|ctypes|importlib|builtins|save|savez|savetxt|load|"
                       r"loadtxt|genfromtxt|fromfile|tofile|memmap|DataSource|lib\.format)\b|__")


class ProgramError(Exception):
    def __init__(self, message: str, trace: str = ""):
        super().__init__(message)
        self.message, self.trace = message, trace


@dataclass
class ProgramResult:
    grid: np.ndarray
    palette: list[str]
    count: int


def check_source(code: str) -> None:
    if not code or not code.strip():
        raise ProgramError("empty program")
    if len(code) > MAX_CODE:
        raise ProgramError(f"program too long ({len(code)} characters, at most {MAX_CODE})")
    m = FORBIDDEN.search(code)
    if m:
        raise ProgramError(f"the program uses {m.group(0)!r}, which is not allowed (no files, processes or dunder names)")


def run_program(code: str, size: list[int], timeout: float = 60.0, sprites: Optional[dict] = None) -> ProgramResult:
    """``sprites``: :func:`spritecam.program_data` for a sprite target (SPRITE_PALETTE, SIDE, carve_side, paint_view...)."""
    check_source(code)
    with tempfile.TemporaryDirectory(prefix="seurat-prog-") as tmp:
        out = os.path.join(tmp, "out.npz")
        env = {k: v for k, v in os.environ.items() if k.upper() in ("SYSTEMROOT", "PATH", "TEMP", "TMP", "HOME", "USERPROFILE")}
        env["OMP_NUM_THREADS"] = "2"
        try:
            proc = subprocess.run([sys.executable, "-I", str(RUNNER)], input=json.dumps({"code": code, "size": size, "out": out, **({"sprites": sprites} if sprites else {})}),
                                  capture_output=True, text=True, timeout=timeout, cwd=tmp, env=env)
        except subprocess.TimeoutExpired:
            raise ProgramError(f"the program ran longer than {timeout:.0f} s and was stopped (vectorise loops with numpy)") from None
        line = (proc.stdout or "").strip().splitlines()[-1:] or [""]
        try:
            res = json.loads(line[0])
        except ValueError:
            raise ProgramError("the program crashed", (proc.stderr or proc.stdout or "")[-1500:]) from None
        if not res.get("ok"):
            raise ProgramError(res.get("error") or "error", res.get("trace") or "")
        with np.load(out, allow_pickle=False) as data:      # closed before the folder goes (Windows keeps it locked)
            grid = data["grid"].astype(np.uint8)
            raw_palette = data["palette"].tolist()
        palette: list[str] = []
        for c in raw_palette:
            try:
                palette.append(norm_hex(str(c)))
            except Exception:  # noqa: BLE001
                palette.append("#ff00ff")
    used = int(grid.max()) if grid.size else 0
    if used > len(palette):
        grid[grid > len(palette)] = 0
    return ProgramResult(grid, palette, int(np.count_nonzero(grid)))
