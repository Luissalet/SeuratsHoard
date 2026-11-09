"""`python -m seurat_hoard` - run the app with uvicorn on 127.0.0.1 (the shared Hoard Link launcher)."""

from __future__ import annotations

from . import DEFAULT_PORT, SERVICE
from .hoard_link.service import run_main


def main() -> int:
    return run_main(service=SERVICE, package="seurat_hoard", default_port=DEFAULT_PORT, app_factory="seurat_hoard.main:create_app",
                    data_dir_env="SEURAT_DATA_DIR", port_env="SEURAT_PORT", open_browser_default=False, title="Seurat's Hoard")


if __name__ == "__main__":
    raise SystemExit(main())
