"""Stdio MCP bridge for Seurat's Hoard.

Every tool call is proxied to the running app (`POST /api/agent/call`) with the Bearer token from
`<DATA_DIR>/mcp-token`; the tool list comes from `GET /api/agent/tools`. Pictures (`model_look`, `reference_look`)
arrive as MCP image content. When nothing answers, the bridge starts the app (`python -m seurat_hoard`) and waits for
it; SEURAT_BRIDGE_AUTOSTART=0 turns that off."""

from __future__ import annotations

import sys

from seurat_hoard.hoard_link.bridge import CatalogBridge


def main() -> int:
    CatalogBridge(app="seurat", service="seurats-hoard", package="seurat_hoard", default_port=5250, data_dir_env="SEURAT_DATA_DIR",
                  title="Seurat's Hoard", root=__file__, default_timeout=120.0, image_content=True).run_bridge()
    return 0


if __name__ == "__main__":
    sys.exit(main())
