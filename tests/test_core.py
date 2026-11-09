"""The voxel engine, the renders, the exports and the picture readers, without a server."""

from __future__ import annotations

import io
import json
import struct

import numpy as np
import pytest
from PIL import Image

from seurat_hoard import builders
from seurat_hoard.errors import SeuratError
from seurat_hoard.exporters import export, from_vox, to_glb, to_stl, to_vox
from seurat_hoard.images import index_picture
from seurat_hoard.llm import parse_json
from seurat_hoard.render import ortho, render, sheet
from seurat_hoard.voxels import VoxelModel, apply_ops

ART = """
................
.....111111.....
...1122222211...
..122222222221..
..122332223321..
.12223322233221.
.12222222222221.
.12222244222221.
.12222244222221.
..122222222221..
..112222222211..
...1155555511...
....15511551....
....111..111....
................
................""".split()
COLS = {".": (255, 255, 255), "1": (40, 30, 60), "2": (90, 200, 120), "3": (250, 250, 250), "4": (200, 60, 80), "5": (60, 140, 90)}


def sprite_png(scale: int = 4) -> bytes:
    arr = np.array([[COLS[c] for c in row] for row in ART], dtype=np.uint8)
    img = Image.fromarray(arr).resize((16 * scale, 16 * scale), Image.NEAREST)
    buf = io.BytesIO()
    img.save(buf, "PNG")
    return buf.getvalue()


def marker_model() -> VoxelModel:
    m = VoxelModel.empty("t", "t", [12, 10, 8])
    apply_ops(m, [{"op": "box", "from": [0, 0, 0], "to": [11, 0, 7], "c": 5},
                  {"op": "box", "from": [9, 1, 5], "to": [11, 9, 7], "c": 13},
                  {"op": "box", "from": [0, 1, 0], "to": [2, 3, 2], "c": 35},
                  {"op": "layer", "plane": "front", "at": 7, "origin": [3, 2], "rows": ["aaa", "a..", "aaa"]}])
    return m


def chars(img: Image.Image, model: VoxelModel) -> list[str]:
    a = np.asarray(img)
    names = {tuple(int(v) for v in model.palette_rgb()[i]): ch for i, ch in ((13, "O"), (10, "R"), (35, "B"))}
    return ["".join("." if p[3] == 0 else names.get(tuple(int(v) for v in p[:3]), "g") for p in row) for row in a]


def test_ops_shapes_and_errors():
    m = VoxelModel.empty("t", "t", [16, 16, 16])
    r = apply_ops(m, [{"op": "sphere", "center": [8, 8, 8], "r": 4, "c": 3}, {"op": "line", "from": [0, 0, 15], "to": [15, 0, 15], "c": 4},
                      {"op": "cylinder", "axis": "y", "center": [2, 2], "from": 0, "to": 5, "r": 1.5, "c": 6},
                      {"op": "voxels", "list": [[15, 15, 0, 7]]}])
    assert r["applied"] == 4 and m.grid[8, 8, 8] == 3 and m.grid[15, 0, 15] == 4 and m.grid[7, 0, 15] == 4 and m.grid[2, 3, 2] == 6 and m.grid[15, 15, 0] == 7
    before = m.grid.copy()
    with pytest.raises(SeuratError) as err:
        apply_ops(m, [{"op": "box", "from": [0, 0, 0], "to": [3, 3, 3], "c": 2}, {"op": "nope"}])
    assert "Operation 1" in err.value.message and (m.grid == before).all()          # all-or-nothing
    r = apply_ops(m, [{"op": "box", "from": [0, 0, 0], "to": [1, 1, 1], "c": 2}, {"op": "box", "from": [0, 0], "c": 1}], atomic=False)
    assert r["applied"] == 1 and len(r["errors"]) == 1 and m.grid[1, 1, 1] == 2
    apply_ops(m, [{"op": "box", "from": [0, 0, 0], "to": [1, 1, 1], "c": 0}])
    assert m.grid[0, 0, 0] == 0
    with pytest.raises(SeuratError):
        apply_ops(m, [{"op": "voxel", "at": [1, 1, 1], "c": 200}])                  # outside the palette


def test_layer_planes_and_mirror():
    m = VoxelModel.empty("t", "t", [4, 3, 5])
    apply_ops(m, [{"op": "layer", "plane": "front", "at": 4, "rows": ["1..", "...", "..2"]}])
    assert m.grid[0, 2, 4] == 1 and m.grid[2, 0, 4] == 2                          # first row is the top
    apply_ops(m, [{"op": "layer", "plane": "side", "at": 3, "rows": ["3...."]}])
    assert m.grid[3, 0, 0] == 3                                                   # side: first column = back (z = 0)
    apply_ops(m, [{"op": "layer", "plane": "top", "from": 1, "to": 2, "rows": ["4"]}])
    assert m.grid[0, 1, 0] == 4 and m.grid[0, 2, 0] == 4
    apply_ops(m, [{"op": "mirror", "axis": "x"}])
    assert m.grid[3, 2, 4] == 1                                                   # x 0 -> 3


def test_views_are_oriented():
    m = marker_model()
    front = chars(ortho(m, "front", 1, shade=False), m)
    assert front[5] == "...RRR...OOO" and front[6] == "BBBR.....OOO"           # tower right, letter in front of the blue
    left = chars(ortho(m, "left", 1, shade=False), m)
    assert left[6] == "BBB..OOR"                                                # back on the left, nearest voxel wins
    top = chars(ortho(m, "top", 1, shade=False), m)
    assert top[0].startswith("BBB") and top[-1] == "gggRRRgggOOO"               # back at the top of the picture
    assert sheet(m, 600).width > 300 and render(m, "iso").width > 50


def test_exports_round_trip_and_are_valid():
    m = marker_model()
    vox = to_vox(m)
    back = from_vox(vox, "b", "b")
    assert back.size == m.size and (back.grid == m.grid).all() and back.palette[12] == m.palette[12]
    glb = to_glb(m)
    magic, version, total = struct.unpack("<III", glb[:12])
    assert magic == 0x46546C67 and version == 2 and total == len(glb)
    jlen = struct.unpack("<I", glb[12:16])[0]
    gltf = json.loads(glb[20: 20 + jlen])
    assert gltf["meshes"][0]["primitives"][0]["attributes"]["COLOR_0"] == 3
    stl = to_stl(m, 2.0)
    n = struct.unpack("<I", stl[80:84])[0]
    assert len(stl) == 84 + 50 * n
    # closed mesh: every edge is used by exactly two triangles
    edges: dict = {}
    for i in range(n):
        vals = struct.unpack("<12f", stl[84 + 50 * i: 84 + 50 * i + 48])
        pts = [tuple(round(v, 4) for v in vals[3 + 3 * k: 6 + 3 * k]) for k in range(3)]
        for a, b in ((0, 1), (1, 2), (2, 0)):
            key = tuple(sorted((pts[a], pts[b])))
            edges[key] = edges.get(key, 0) + 1
    diagonal_ok = all(c in (2, 4) for c in edges.values())       # 4 where two voxels touch along an edge only
    assert diagonal_ok
    for fmt in ("png", "sprites"):
        assert export(m, fmt)[:8] == b"\x89PNG\r\n\x1a\n"
    with pytest.raises(SeuratError):
        to_stl(VoxelModel.empty("e", "e", [2, 2, 2]))


def test_pixel_art_is_read_at_its_real_size():
    ix = index_picture(sprite_png(4), max_side=64, colors=8)
    assert (ix.width, ix.height, ix.scale) == (14, 13, 4)                      # white background cropped, x4 undone
    assert len(ix.palette) == 5 and ix.rows()[0] == "....222222...."
    small = index_picture(sprite_png(4), max_side=8, colors=4)
    assert max(small.width, small.height) == 8 and len(small.palette) <= 4


def test_builders():
    ix = index_picture(sprite_png(), max_side=64)
    inflated = builders.build("inflate", ix, model_id="a")
    assert inflated.size[:2] == [14, 13] and inflated.size[2] >= 3
    front = chars(ortho(inflated, "front", 1, shade=False), inflated)
    assert sum(ch != "." for ch in "".join(front)) == int((ix.grid > 0).sum())  # front silhouette = the picture
    ext = builders.build("extrude", ix, depth=3)
    assert ext.size[2] == 3 and ext.count == 3 * int((ix.grid > 0).sum())
    hull = builders.build("hull", ix, side=ix)
    assert hull.count > 0 and hull.size == [14, 13, 14]
    with pytest.raises(SeuratError):
        builders.build("hull", ix)


def test_parse_json_from_model_answers():
    assert parse_json('Sure!\n```json\n{"ops": [{"op": "clear"}]}\n```') == {"ops": [{"op": "clear"}]}
    assert parse_json('<think>{"x":1}</think> text {"ops": [], "notes": "a } b"} trailing')["notes"] == "a } b"
    with pytest.raises(ValueError):
        parse_json("no json here")


def test_parse_json_salvages_a_cut_answer_and_prefers_the_ops_object():
    cut = '{"notes": "a", "palette_add": ["#112233"], "ops": [{"op": "clear"}, {"op": "box", "from": [0,0,0], "to": [1,1,1], "c": 1}, {"op": "box", "fr'
    got = parse_json(cut)
    assert got["salvaged"] and len(got["ops"]) == 2 and got["palette_add"] == ["#112233"] and got["notes"] == "a"
    assert parse_json('{"op": "box"} then {"ops": [], "done": true}')["done"] is True


def test_streaming_client_reads_sse_reports_progress_and_cuts_loops(monkeypatch, tmp_path):
    import threading
    from contextlib import contextmanager

    from seurat_hoard import llm
    from seurat_hoard.config import Settings

    settings = Settings(tmp_path / "s.json")
    settings.update({"llm_url": "http://spark:8888/v1", "llm_model": "deepseek"})
    sent = {}

    def lines(parts):
        for p in parts:
            yield "data: " + json.dumps({"model": "deepseek", "choices": [{"delta": p, "finish_reason": None}]})
        yield "data: [DONE]"

    @contextmanager
    def fake_stream(method, url, json=None, timeout=None):
        sent.update(url=url, body=json)
        parts = [{"reasoning_content": "hmm"}, {"content": '{"ops": ['}, {"content": '{"op": "clear"}]}'}]
        yield SimpleNamespace(status_code=200, iter_lines=lambda: lines(parts), read=lambda: b"")

    from types import SimpleNamespace
    monkeypatch.setattr(llm.httpx, "stream", fake_stream)
    seen = []
    client = llm.ModelClient(settings)
    ans = client.chat("sys", "user", [b"\x89PNG"], max_tokens=100, effort="off", timeout=30, on_progress=lambda a, b: seen.append((a, b)))
    assert ans.text == '{"ops": [{"op": "clear"}]}' and ans.reasoning_chars == 3 and seen
    assert sent["url"] == "http://spark:8888/v1/chat/completions" and sent["body"]["stream"] is True
    assert sent["body"]["chat_template_kwargs"]["enable_thinking"] is False
    assert sent["body"]["messages"][1]["content"][1]["type"] == "image_url"

    @contextmanager
    def looping(method, url, json=None, timeout=None):
        parts = [{"content": '{"ops": ['}] + [{"content": '{"op":"voxel","at":[1,1,1],"c":1},'}] * 2000
        yield SimpleNamespace(status_code=200, iter_lines=lambda: lines(parts), read=lambda: b"")

    monkeypatch.setattr(llm.httpx, "stream", looping)
    ans = client.chat("sys", "user", [], max_tokens=100, effort=None, timeout=30)
    assert ans.finish == "loop" and len(ans.text) < 40000
    stop = threading.Event()
    stop.set()
    assert client.chat("sys", "user", [], max_tokens=100, effort=None, timeout=30, cancel=stop).finish == "cancelled"


def test_inflate_and_lathe_ops():
    m = VoxelModel.empty("t", "t", [16, 16, 16])
    rows = ["1" * 16] * 16
    apply_ops(m, [{"op": "inflate", "origin": [0, 0], "center": 7.5, "depth": 16, "profile": "soft", "rows": rows}])
    thick = lambda g, x, y: int((g[x, y, :] > 0).sum())
    assert thick(m.grid, 7, 7) == 16 and thick(m.grid, 0, 7) < 8                 # thick in the middle, thin at the edge
    m = VoxelModel.empty("t", "t", [10, 6, 10])
    apply_ops(m, [{"op": "inflate", "origin": [0, 0], "center": 4.5, "depth": 8, "rows": ["..1111..", ".111111.", "11111111", ".111111."]}])
    assert m.grid[0, 0, :].sum() == 0 and m.grid[2, 3, 4] == 1                    # outside the rows stays empty
    m2 = VoxelModel.empty("t", "t", [9, 5, 9])
    apply_ops(m2, [{"op": "lathe", "center": [4, 4], "from": 0, "radii": [4, 3, 2, 1], "c": 2}])
    assert m2.grid[4, 0, 0] == 2 and m2.grid[4, 3, 2] == 0 and m2.grid[4, 3, 4] == 2 and m2.grid[:, 4, :].sum() == 0


def test_paint_follows_the_surface():
    m = VoxelModel.empty("t", "t", [5, 3, 6])
    apply_ops(m, [{"op": "box", "from": [0, 0, 0], "to": [4, 2, 2], "c": 1}, {"op": "box", "from": [2, 0, 3], "to": [2, 2, 5], "c": 1}])
    apply_ops(m, [{"op": "paint", "side": "front", "rows": ["22222", "22222", "22222"]}])
    assert m.grid[0, 0, 2] == 2 and m.grid[2, 0, 5] == 2 and m.grid[2, 0, 4] == 1     # the frontmost voxel of each column
    apply_ops(m, [{"op": "paint", "side": "back", "rows": ["3...."], "depth": 2}])
    assert m.grid[4, 0, 0] == 3 and m.grid[4, 0, 1] == 3 and m.grid[0, 0, 0] == 1      # back: first column = right side
    apply_ops(m, [{"op": "paint", "side": "top", "rows": ["4"], "origin": [2, 5]}])
    assert m.grid[2, 2, 5] == 4
    apply_ops(m, [{"op": "paint", "side": "left", "rows": ["5"], "origin": [0, 0]}])
    assert m.grid[0, 0, 0] == 5
    apply_ops(m, [{"op": "paint", "side": "right", "rows": ["6"], "origin": [0, 0]}])
    assert m.grid[2, 0, 5] == 6                                                         # right: first column = front


def test_capsule_and_prism():
    from seurat_hoard.voxels import VoxelModel, apply_ops
    m = VoxelModel.empty("t", "t", [30, 20, 30])
    res = apply_ops(m, [{"op": "line", "from": [5, 5, 5], "to": [25, 5, 5], "r": 2, "c": 1}])
    assert not res["errors"] and m.grid[15, 5, 5] == 1 and m.grid[15, 7, 5] == 1 and m.grid[15, 9, 5] == 0
    m2 = VoxelModel.empty("t", "t", [30, 20, 30])
    res = apply_ops(m2, [{"op": "prism", "plane": "top", "points": [[0, 0], [20, 0], [0, 20]], "from": 10, "to": 11, "c": 2}])
    assert not res["errors"]
    assert m2.grid[2, 10, 2] == 2 and m2.grid[2, 11, 2] == 2 and m2.grid[2, 12, 2] == 0 and m2.grid[19, 10, 19] == 0
    bad = apply_ops(VoxelModel.empty("t", "t", [8, 8, 8]), [{"op": "prism", "plane": "top", "points": [[0, 0]], "from": 1, "to": 1, "c": 1}], atomic=False)
    assert bad["errors"]


def test_destructive_guard_and_auto_paint():
    from seurat_hoard.generate import Generator
    from seurat_hoard.voxels import VoxelModel, apply_ops
    assert Generator._destructive([{"op": "clear"}], 500, 0)
    assert Generator._destructive([{"op": "clear"}, {"op": "box", "from": [0, 0, 0], "to": [1, 1, 1], "c": 1}], 500, 8)
    assert Generator._destructive([{"op": "box", "from": [0, 0, 0], "to": [1, 1, 1], "c": 0}], 1000, 100)
    assert not Generator._destructive([{"op": "box", "from": [0, 0, 0], "to": [1, 1, 1], "c": 1}], 0, 8)

    class Ix:
        def __init__(self, grid, palette):
            import numpy as np
            self.grid = np.array(grid)
            self.palette = palette
            self.height, self.width = self.grid.shape
    m = VoxelModel.empty("t", "t", [4, 4, 6], ["#ffffff", "#ff0000", "#0000ff"])
    apply_ops(m, [{"op": "box", "from": [0, 0, 1], "to": [3, 3, 4], "c": 1}])
    refs = {"front": [({}, Ix([[1] * 4] * 4, ["#ff0000"]), b"")], "side": [({}, Ix([[0, 0, 1, 1, 1, 1]] * 4, ["#0000ff"]), b"")]}
    changed = Generator._auto_paint(Generator.__new__(Generator), m, refs)
    assert changed > 0
    assert m.grid[1, 1, 4] == 2           # front surface red
    assert m.grid[0, 1, 2] == 3           # left side blue (z=2 is column 2)
    assert m.grid[1, 1, 2] == 1           # inside untouched
