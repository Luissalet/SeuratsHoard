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
    gen = Generator.__new__(Generator)
    gen.svc = type("Svc", (), {"settings": {"paint_sides": True}})()
    changed = gen._auto_paint(m, refs)
    assert changed > 0
    assert m.grid[1, 1, 4] == 2           # front surface red
    assert m.grid[0, 1, 2] == 3           # left side blue (z=2 is column 2)
    assert m.grid[1, 1, 2] == 1           # inside untouched
    gen.svc.settings["paint_sides"] = False
    m2 = VoxelModel.empty("t", "t", [4, 4, 6], ["#ffffff", "#ff0000", "#0000ff"])
    apply_ops(m2, [{"op": "box", "from": [0, 0, 1], "to": [3, 3, 4], "c": 1}])
    gen._auto_paint(m2, refs)
    assert m2.grid[0, 1, 2] == 1 and m2.grid[1, 1, 4] == 2     # sides keep the part colour by default


def test_thinking_budget_then_answer_from_the_reasoning(monkeypatch):
    import contextlib, json as _json
    import httpx
    from seurat_hoard import llm as L

    calls = []

    class Resp:
        status_code = 200

        def __init__(self, lines):
            self._lines = lines

        def iter_lines(self):
            yield from self._lines

        def read(self):
            return b""

    @contextlib.contextmanager
    def fake_stream(method, url, json=None, timeout=None):
        calls.append(json)
        if json["chat_template_kwargs"].get("enable_thinking"):
            lines = ["data: " + _json.dumps({"choices": [{"delta": {"reasoning_content": "plan step %d. " % i}}]}) for i in range(50)]
        else:
            chunk = {"model": "m", "choices": [{"delta": {"content": '{"ops": []}'}, "finish_reason": "stop"}]}
            lines = ["data: " + _json.dumps(chunk), "data: [DONE]"]
        yield Resp(lines)

    monkeypatch.setattr(httpx, "stream", fake_stream)

    class S(dict):
        pass
    client = L.ModelClient(S(llm_url="http://x/v1", llm_model="m"))
    seen = []
    ans = client.chat("sys", "user", [], max_tokens=1000, effort="max", timeout=60, thinking_tokens=20,
                      on_progress=lambda t, r: seen.append((t, r)))
    assert ans.followed_up and ans.text == '{"ops": []}' and ans.reasoning_tokens == 20
    assert len(calls) == 2
    follow = calls[1]
    assert follow["chat_template_kwargs"] == {"enable_thinking": False} and "plan step 0." in follow["messages"][-2]["content"]
    assert follow["max_tokens"] == 1000 and calls[0]["max_tokens"] == 1000 + 20 + 2000


def test_thinking_cut_by_the_server_is_continued(monkeypatch):
    import contextlib, json as _json
    import httpx
    from seurat_hoard import llm as L

    calls = []

    class Resp:
        status_code = 200

        def __init__(self, lines):
            self._lines = lines

        def iter_lines(self):
            yield from self._lines

        def read(self):
            return b""

    @contextlib.contextmanager
    def fake_stream(method, url, json=None, timeout=None):
        calls.append(json)
        n = len(calls)
        lines = ["data: " + _json.dumps({"choices": [{"delta": {"reasoning_content": f"part {n} thought. "}}],
                                          "usage": {"completion_tokens": 32768, "reasoning_tokens": 32768}})]
        if n == 1:                                   # the server's cap: thinking stops, no answer
            lines.append("data: " + _json.dumps({"choices": [{"delta": {}, "finish_reason": "length"}]}))
        else:
            lines.append("data: " + _json.dumps({"choices": [{"delta": {"content": '{"ops": [1]}'}, "finish_reason": "stop"}],
                                                  "usage": {"completion_tokens": 32780, "reasoning_tokens": 32768}}))
        lines.append("data: [DONE]")
        yield Resp(lines)

    monkeypatch.setattr(httpx, "stream", fake_stream)
    client = L.ModelClient({"llm_url": "http://x/v1", "llm_model": "m"})
    ans = client.chat("sys", "user", [], max_tokens=1000, effort="max", timeout=60, thinking_tokens=200000)
    assert ans.text == '{"ops": [1]}' and ans.thinking_calls == 2 and not ans.followed_up
    assert ans.reasoning_tokens == 65536 and "part 1 thought" in calls[1]["messages"][-2]["content"]
    assert calls[1]["chat_template_kwargs"].get("enable_thinking") is True


def test_an_answer_cut_by_the_server_is_written_again(monkeypatch):
    import contextlib, json as _json
    import httpx
    from seurat_hoard import llm as L
    calls = []

    class Resp:
        status_code = 200

        def __init__(self, lines):
            self._lines = lines

        def iter_lines(self):
            yield from self._lines

        def read(self):
            return b""

    @contextlib.contextmanager
    def fake_stream(method, url, json=None, timeout=None):
        calls.append(json)
        if json["chat_template_kwargs"].get("enable_thinking"):
            lines = ["data: " + _json.dumps({"choices": [{"delta": {"reasoning_content": "plan. "}}],
                                              "usage": {"completion_tokens": 30000, "reasoning_tokens": 30000}}),
                     "data: " + _json.dumps({"choices": [{"delta": {"content": '{"notes": "x", "code": "PALETTE = [1, 2'}, "finish_reason": "length"}],
                                              "usage": {"completion_tokens": 32768, "reasoning_tokens": 30000}})]
        else:
            lines = ["data: " + _json.dumps({"choices": [{"delta": {"content": '{"notes": "x", "code": "ok"}'}, "finish_reason": "stop"}]})]
        yield Resp(lines + ["data: [DONE]"])

    monkeypatch.setattr(httpx, "stream", fake_stream)
    ans = L.ModelClient({"llm_url": "http://x/v1", "llm_model": "m"}).chat("s", "u", [], max_tokens=1000, effort="max", timeout=60,
                                                                          thinking_tokens=200000)
    assert ans.followed_up and ans.text == '{"notes": "x", "code": "ok"}' and len(calls) == 2


def _schem_bytes(w, h, l, palette, ids, version=2):
    import gzip, struct

    def s(x):
        b = x.encode()
        return struct.pack(">H", len(b)) + b

    def tag(t, name, payload):
        return bytes([t]) + s(name) + payload

    def varints(vals):
        out = bytearray()
        for v in vals:
            while True:
                b = v & 0x7F
                v >>= 7
                out.append(b | (0x80 if v else 0))
                if not v:
                    break
        return bytes(out)

    pal = b"".join(tag(3, k, struct.pack(">i", v)) for k, v in palette.items()) + b"\0"
    data = varints(ids)
    blocks = tag(10, "Palette", pal) + tag(7, "BlockData" if version == 2 else "Data", struct.pack(">i", len(data)) + data)
    body = (tag(3, "Version", struct.pack(">i", version)) + tag(2, "Width", struct.pack(">h", w))
            + tag(2, "Height", struct.pack(">h", h)) + tag(2, "Length", struct.pack(">h", l)))
    if version == 2:
        body += blocks + b"\0"
        return gzip.compress(tag(10, "Schematic", body))
    body += tag(10, "Blocks", blocks + b"\0") + b"\0"
    return gzip.compress(tag(10, "", tag(10, "Schematic", body) + b"\0"))


def test_schem_import_versions_and_colours():
    from seurat_hoard.exporters import import_model
    from seurat_hoard.schem import block_color
    w, h, l = 3, 2, 2
    palette = {"minecraft:air": 0, "minecraft:red_wool": 1, "minecraft:oak_stairs[facing=east]": 2, "minecraft:mystery_block": 3}
    ids = [0] * (w * h * l)
    ids[0] = 1                                    # x=0, y=0, z=0
    ids[2 + 1 * w + 1 * w * l] = 2                # x=2, y=1, z=1
    ids[1] = 300                                  # a multi-byte varint index outside the palette: empty
    for version in (2, 3):
        m = import_model(_schem_bytes(w, h, l, palette, ids, version), "m", "t")
        assert m.size == [3, 2, 2] and m.count == 2
        assert m.palette[m.grid[0, 0, 0] - 1] == "#a12722" and m.palette[m.grid[2, 1, 1] - 1] == "#a2834f"
    assert block_color("minecraft:cave_air") is None and block_color("minecraft:blue_concrete_powder") == "#464ba6"
    import pytest as _p
    from seurat_hoard.errors import SeuratError
    with _p.raises(SeuratError) as e:
        import_model(b"<!doctype html><html>", "m", "t")
    assert "web page" in e.value.message


def test_sprite_builder_keeps_the_outline_off_the_sides():
    import numpy as np
    from seurat_hoard.builders import build
    from seurat_hoard.images import Indexed

    grid = np.zeros((12, 12), dtype=np.uint8)
    grid[1:11, 1:11] = 1                       # black outline
    grid[2:10, 2:10] = 2                       # red body
    ix = Indexed(width=12, height=12, palette=["#000000", "#cc2222"], grid=grid)
    m = build("sprite", ix, depth=6, base_palette=[])
    pal = m.palette
    front_z = m.grid[1, 5].nonzero()[0].max()
    assert pal[m.grid[1, 5, front_z] - 1] == "#000000"                   # outline on the front face
    mid = m.grid[1, 5].nonzero()[0]
    if len(mid) > 2:
        assert pal[m.grid[1, 5, mid[len(mid) // 2]] - 1] != "#000000"     # but not inside the side wall


def test_camera_render_and_sprite_match():
    from seurat_hoard import spritecam
    from seurat_hoard.render import camera, camera_basis

    c, r, u = camera_basis(0, 0)
    assert np.allclose(c, [0, 0, 1]) and np.allclose(r, [1, 0, 0]) and np.allclose(u, [0, 1, 0])
    c, r, u = camera_basis(-90, 30)
    assert c[0] < 0 and c[1] > 0 and r[2] > 0.99          # from the left the face (+z) is on the right
    m = VoxelModel.empty("t", "t", [16, 16, 16], ["#ff0000", "#00ff00"])
    m.grid[4:12, 0:8, 4:12] = 1
    m.grid[4:12, 8:10, 4:12] = 2
    top = np.asarray(camera(m, 0, 80, 4))
    side = np.asarray(camera(m, 0, 0, 4))
    green = lambda a: ((a[..., 1] > 100) & (a[..., 0] < 80) & (a[..., 3] > 0)).sum()
    assert green(top) > green(side)                        # from above the green lid fills the picture
    # a sprite made from the model's own camera view matches it best
    img = camera(m, 0, 45, 1)
    buf = io.BytesIO(); img.save(buf, "PNG")
    ref = spritecam.sprite_ref("front", "self", buf.getvalue())
    good = spritecam.match_view(m, ref, 0, 45)
    other = VoxelModel.empty("o", "o", [16, 16, 16], ["#ff0000"])
    other.grid[7:9, 0:15, 7:9] = 1
    bad = spritecam.match_view(other, ref, 0, 45)
    assert good.iou > 0.8 > bad.iou and "silhouette match" in good.summary()
    assert spritecam.sheet([good, bad]).width > 100 and spritecam.colours([ref])


def test_sprite_program_tools_make_the_profile_exact():
    from seurat_hoard import spritecam
    from seurat_hoard.program import run_program

    # a side sprite: a bird-ish profile with a dark outline, on a flat background
    side = np.zeros((20, 30, 4), np.uint8)
    side[...] = (200, 100, 220, 255)
    side[6:16, 4:22] = (20, 20, 20, 255)
    side[7:15, 5:21] = (220, 60, 40, 255)
    side[3:8, 20:27] = (20, 20, 20, 255)
    side[4:7, 21:26] = (240, 200, 40, 255)
    front = np.zeros((20, 26, 4), np.uint8)
    front[...] = (200, 100, 220, 255)
    front[5:15, 3:23] = (220, 60, 40, 255)
    pngs = {}
    for name, arr in (("side", side), ("front", front)):
        buf = io.BytesIO(); Image.fromarray(arr, "RGBA").save(buf, "PNG"); pngs[name] = buf.getvalue()
    refs = [spritecam.sprite_ref("side", "s", pngs["side"]), spritecam.sprite_ref("front", "f", pngs["front"])]
    size = spritecam.grid_size(refs)
    assert size[0] >= 20 and size[2] >= 23 and size[1] >= 14
    data = spritecam.program_data(refs, size, 45, 0)
    assert len(data["palette"]) >= 2 and np.asarray(data["side"]).shape == (size[2], size[1])
    code = ("PALETTE = SPRITE_PALETTE + ['#101010']\n"
            "def build(g):\n"
            "    box(g, len(PALETTE), (X//2-6, 0, 0), (X//2+6, Y-1, Z-1))\n"
            "    carve_side(g)\n"
            "    paint_view(g, 'front')\n"
            "    paint_side(g)\n")
    res = run_program(code, size, 60, sprites=data)
    m = VoxelModel.empty("t", "t", size, res.palette)
    m.grid = res.grid
    vm = spritecam.match_view(m, refs[0], -90, 0)
    assert vm.iou > 0.9 and vm.colour < 0.15, vm.summary()
    assert "z = " not in spritecam.char_map(refs[0], data["palette"]) and "#" in spritecam.char_map(refs[0], data["palette"])
    ext = run_program("PALETTE = SPRITE_PALETTE\ndef build(g):\n    extrude_side(g, 2, 4)\n", size, 60, sprites=data)
    assert ext.count == int(np.asarray(data["side"]).astype(bool).sum()) * 3
