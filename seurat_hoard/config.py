"""Process configuration (environment) and the editable settings (``<data>/settings.json``)."""

from __future__ import annotations

import copy
import os
import threading
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from . import APP_ID, DEFAULT_PORT
from .hoard_link.appconfig import AppPaths, env_int, env_str
from .hoard_link.atomic import read_json, write_json_atomic
from .hoard_link.guard import parse_allowed_hosts

REPO_ROOT = Path(__file__).resolve().parent.parent

DEFAULT_SETTINGS: dict[str, Any] = {
    "max_side": 32,          # a reference is read at most this many pixels wide/tall (= voxels of the model)
    "colors": 16,            # colours kept from a reference
    "review_rounds": 3,      # correction rounds after the first answer (the model sees its own result)
    "effort": "max",         # how hard the model reasons: off, low, medium, high, max (above off, rounds take minutes)
    "max_tokens": 24000,     # longest answer per round, not counting the thinking
    "thinking_tokens": 200000,  # thinking per round before the answer is asked for (effort above off)
    "timeout_s": 7200,       # per model call
    "auto_paint": True,      # after the model shapes the volume, paint the reference pictures onto it
    "parallel_jobs": 1,      # generations running at the same time (read at start); one at a time thinks faster
    "llm_url": "",           # empty: the family's model (DeepSeek on the Sparks through Hoard Link); else an OpenAI-compatible URL
    "llm_model": "",
    "voxel_mm": 2.0,         # STL: millimetres per voxel
    "glb_voxel_size": 0.1,   # GLB: metres per voxel
    "new_size": [32, 32, 32],
}
EFFORTS = ("off", "low", "medium", "high", "max")


@dataclass
class Config:
    data_dir: Path = field(default_factory=lambda: REPO_ROOT / "data")
    port: int = DEFAULT_PORT
    allowed_hosts: tuple[str, ...] = ()
    data_dir_configured: bool = False
    background: bool = True          # the generation worker; tests switch it off and run jobs inline

    @property
    def paths(self) -> AppPaths:
        return AppPaths(APP_ID, REPO_ROOT, self.data_dir, self.data_dir_configured)

    @property
    def token_path(self) -> Path:
        return self.paths.token_path

    @property
    def settings_path(self) -> Path:
        return self.data_dir / "settings.json"

    @property
    def renders_dir(self) -> Path:
        return self.data_dir / "renders"

    @classmethod
    def from_env(cls) -> "Config":
        data_env = env_str("SEURAT_DATA_DIR")
        return cls(
            data_dir=Path(data_env).expanduser() if data_env else REPO_ROOT / "data",
            port=env_int("SEURAT_PORT", default=DEFAULT_PORT) or DEFAULT_PORT,
            allowed_hosts=tuple(parse_allowed_hosts(os.environ.get("SEURAT_ALLOWED_HOSTS", ""))),
            data_dir_configured=bool(data_env),
        )


class Settings:
    """Thread-safe JSON settings with defaults; ``update`` validates and writes atomically."""

    def __init__(self, path: Path):
        self.path = path
        self._lock = threading.RLock()
        self._data = self._load()

    def _load(self) -> dict[str, Any]:
        raw = read_json(self.path, default={}) or {}
        data = copy.deepcopy(DEFAULT_SETTINGS)
        if isinstance(raw, dict):
            data.update({k: v for k, v in raw.items() if k in DEFAULT_SETTINGS})
        try:
            return normalize(data)
        except ValueError:
            return copy.deepcopy(DEFAULT_SETTINGS)

    def get(self) -> dict[str, Any]:
        with self._lock:
            return copy.deepcopy(self._data)

    def __getitem__(self, key: str) -> Any:
        with self._lock:
            return copy.deepcopy(self._data[key])

    def update(self, changes: dict[str, Any]) -> dict[str, Any]:
        with self._lock:
            data = copy.deepcopy(self._data)
            for key, value in dict(changes).items():
                if key not in DEFAULT_SETTINGS:
                    raise ValueError(f"Unknown setting: {key}")
                if value is None:
                    continue
                data[key] = value
            data = normalize(data)
            self._data = data
            write_json_atomic(self.path, data)
        return self.get()


def _clamp(v: Any, lo: float, hi: float, kind=int):
    return kind(min(max(kind(v), lo), hi))


def normalize(data: dict[str, Any]) -> dict[str, Any]:
    data["max_side"] = _clamp(data["max_side"], 4, 128)
    data["colors"] = _clamp(data["colors"], 2, 48)
    data["review_rounds"] = _clamp(data["review_rounds"], 0, 6)
    if data["effort"] not in EFFORTS:
        raise ValueError("effort must be off, low, medium, high or max")
    data["max_tokens"] = _clamp(data["max_tokens"], 1000, 128000)
    data["timeout_s"] = _clamp(data["timeout_s"], 30, 7200)
    data["thinking_tokens"] = _clamp(data["thinking_tokens"], 0, 900000)
    data["auto_paint"] = bool(data["auto_paint"])
    data["parallel_jobs"] = _clamp(data["parallel_jobs"], 1, 6)
    data["llm_url"] = str(data.get("llm_url") or "").strip()
    data["llm_model"] = str(data.get("llm_model") or "").strip()
    data["voxel_mm"] = _clamp(data["voxel_mm"], 0.1, 50.0, float)
    data["glb_voxel_size"] = _clamp(data["glb_voxel_size"], 0.001, 10.0, float)
    size = data.get("new_size") or [32, 32, 32]
    if not isinstance(size, (list, tuple)) or len(size) != 3:
        raise ValueError("new_size must be [x, y, z]")
    data["new_size"] = [_clamp(v, 1, 256) for v in size]
    return data
