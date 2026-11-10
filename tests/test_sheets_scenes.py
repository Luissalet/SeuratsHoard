"""Sprite sheets and animations, camera calibration, lit renders and spins, rounds in 3-D and scenes by parts."""

from __future__ import annotations

import io
import json
from types import SimpleNamespace

import numpy as np
import pytest
from PIL import Image

from seurat_hoard import lit, scene, sheets, spritecam
from seurat_hoard.voxels import VoxelModel, apply_ops

from .test_app import AgentLink, FakeLink


# ------------------------------------------------------------------------------------------------ a walk sheet
def figure(kind: str, step: int) -> np.ndarray:
    """A 20x28 character: front (face), back (hair), side (facing right, a nose and a backpack); legs move with step."""
    a = np.zeros((28, 20, 4), np.uint8)
    yy, xx = np.mgrid[0:28, 0:20]

    def put(mask, rgb):
        a[mask] = (*rgb, 255)
    cx = 10 if kind in ("front", "back") else 9
    put(((xx - cx + 0.5) / 6) ** 2 + ((yy - 16) / 6) ** 2 <= 1, (60, 90, 200))
    hx = cx - 0.5 + (2 if kind == "side" else 0)
    put(((xx - hx) / 5) ** 2 + ((yy - 7) / 5) ** 2 <= 1, (90, 50, 20) if kind == "back" else (240, 200, 160))
    if kind == "front":
        put((yy <= 3) & (((xx - hx) / 5) ** 2 + ((yy - 7) / 5) ** 2 <= 1), (90, 50, 20))
        for ex in (cx - 3, cx + 2):
            put((yy == 7) & (xx == ex), (250, 250, 250))
            put((yy == 8) & (xx == ex), (20, 20, 30))
        put((yy == 10) & (xx >= cx - 1) & (xx <= cx), (200, 40, 40))
    if kind == "side":
        put((yy >= 7) & (yy <= 8) & (xx >= cx + 6) & (xx <= cx + 8), (240, 200, 160))
        put((yy == 6) & (xx == cx + 4), (20, 20, 30))
        put((yy >= 12) & (yy <= 19) & (xx >= 0) & (xx <= 3), (120, 80, 40))
    legs = [(cx - 4, 0), (cx + 2, 0)] if kind != "side" else [(cx - 2, 0), (cx + 1, 0)]
    for k, (lx, _) in enumerate(legs):
        lift = 2 if (step == 1 and k == 0) or (step == 2 and k == 1) else 0
        dx = (2 if step == 1 else -2 if step == 2 else 0) * (1 if k == 0 else -1) if kind == "side" else 0
        put((yy >= 21) & (yy <= 27 - lift) & (xx >= lx + dx) & (xx <= lx + dx + 1), (40, 40, 60))
    return a


def subject(a: np.ndarray) -> tuple[int, int]:
    ys, xs = np.nonzero(a[..., 3])
    return int(ys.max() - ys.min() + 1), int(xs.max() - xs.min() + 1)


def walk_sheet(scale: int = 1) -> bytes:
    rows = [("front", False), ("side", True), ("side", False), ("back", False)]      # down, left, right, up
    sheet = np.zeros((4 * 32, 3 * 32, 4), np.uint8)
    for r, (kind, mirror) in enumerate(rows):
        for c in range(3):
            f = figure(kind, c)
            if mirror:
                f = f[:, ::-1]
            sheet[r * 32 + 2: r * 32 + 30, c * 32 + 6: c * 32 + 26] = f
    img = Image.fromarray(sheet, "RGBA")
    if scale > 1:
        img = img.resize((img.width * scale, img.height * scale), Image.NEAREST)
    buf = io.BytesIO()
    img.save(buf, "PNG")
    return buf.getvalue()


def walk_gif() -> bytes:
    frames = [Image.fromarray(figure("front", s), "RGBA").resize((40, 56), Image.NEAREST) for s in (0, 1, 0, 2)]
    pal = [f.convert("RGB").quantize(colors=16) for f in frames]
    buf = io.BytesIO()
    pal[0].save(buf, "GIF", save_all=True, append_images=pal[1:], duration=120, loop=0)
    return buf.getvalue()


def test_sheet_grid_frames_and_guessed_sides():
    sh = sheets.read(walk_sheet(scale=3))
    assert sh.kind == "grid" and (sh.rows, sh.cols) == (4, 3) and len(sh.frames) == 12
    assert sh.scale == 3 and sh.frames[0].rgba.shape[:2] == subject(figure("front", 0))   # pixel art read at its real size
    views = sheets.guess_views(sh)
    roles = [v["role"] for v in views]
    assert sorted(roles) == ["back", "front", "side", "side"], views
    front = next(v for v in views if v["role"] == "front")
    back = next(v for v in views if v["role"] == "back")
    assert sh.frame(front["frames"][0]).row == 0 and sh.frame(back["frames"][0]).row == 3
    assert front["frames"][0] == 0 and len(front["frames"]) == 3                 # the standing frame first, then the steps
    forced = sheets.read(walk_sheet(), cols=3, rows=4)
    assert len(forced.frames) == 12
    img = sheets.contact(sh, views)
    assert img.width > 3 * 96


def test_animation_frames_and_standing_frame():
    sh = sheets.read(walk_gif())
    assert sh.kind == "animation" and len(sh.frames) == 4 and sh.frames[0].duration == 120
    assert sh.scale == 2 and sh.frames[0].rgba.shape[:2] == subject(figure("front", 0))
    views = sheets.guess_views(sh)
    assert views == [{"role": "front", "frames": [0, 1, 2, 3]}] or views[0]["frames"][0] in (0, 2)


def crop(a: np.ndarray) -> np.ndarray:
    ys, xs = np.nonzero(a[..., 3])
    return a[ys.min(): ys.max() + 1, xs.min(): xs.max() + 1]


def test_touching_rows_margins_and_a_turning_animation():
    # rows with no empty line between them (the cells are as tall as the sprites), and a margin around the sheet
    figs = [[crop(figure(k, s)) for s in range(3)] for k in ("front", "side", "back")]
    ch = max(f.shape[0] for row in figs for f in row)
    sheet = np.zeros((3 * ch + 7, 3 * 32 + 9, 4), np.uint8)
    for r, row in enumerate(figs):
        for c, f in enumerate(row):
            y = 3 + r * ch + (ch - f.shape[0])
            sheet[y: y + f.shape[0], 5 + c * 32: 5 + c * 32 + f.shape[1]] = f
    buf = io.BytesIO()
    Image.fromarray(sheet, "RGBA").save(buf, "PNG")
    sh = sheets.read(buf.getvalue())
    assert (sh.rows, sh.cols, len(sh.frames)) == (3, 3, 9), sh.notes
    assert sorted(v["role"] for v in sheets.guess_views(sh)) == ["back", "front", "side"]
    # an animation that walks front, left, right and back one after another
    seq = [figure("front", s) for s in (0, 1, 0, 2)] + [figure("side", s)[:, ::-1] for s in (0, 1, 0, 2)] \
        + [figure("side", s) for s in (0, 1, 0, 2)] + [figure("back", s) for s in (0, 1, 0, 2)]
    frames = [Image.fromarray(a, "RGBA").resize((40, 56), Image.NEAREST) for a in seq]
    buf = io.BytesIO()
    frames[0].save(buf, "PNG", save_all=True, append_images=frames[1:], duration=100, loop=0)
    sh = sheets.read(buf.getvalue())
    assert sh.kind == "animation" and len(sh.frames) == 16
    views = sheets.guess_views(sh)
    assert [v["role"] for v in views] == ["front", "side", "side", "back"], views
    assert all(len(v["frames"]) == 4 for v in views)


def test_calibration_finds_the_camera_that_drew_the_sprite():
    m = VoxelModel.empty("t", "t", [20, 24, 20])
    apply_ops(m, [{"op": "box", "from": [3, 0, 3], "to": [16, 4, 16], "c": 3},
                  {"op": "box", "from": [8, 5, 13], "to": [11, 22, 15], "c": 10},
                  {"op": "box", "from": [4, 5, 4], "to": [7, 9, 7], "c": 20}])
    drawn = spritecam.project(m, 0.0, 30.0)
    ref = spritecam.SpriteRef("front", "drawn at 30", drawn)
    before = spritecam.match(m, [ref], 45.0, 0.0)[0].iou
    cam = spritecam.calibrate_one(m, ref, 45.0, 0.0)
    assert abs(cam["pitch"] - 30) <= 3 and abs(cam["yaw"]) <= 5 and cam["iou"] > 0.95 and cam["iou"] > before
    changes = spritecam.apply_cameras([ref], [cam])
    assert changes and ref.pitch == cam["pitch"]
    assert spritecam.match(m, [ref], 45.0, 0.0)[0].iou > before
    assert spritecam.pitches([cam])["sprite_pitch"] == round(cam["pitch"])
    # a view whose camera is unknown finds its yaw all around
    side = spritecam.SpriteRef("view", "from the right", spritecam.project(m, 90.0, 20.0))
    found = spritecam.calibrate_one(m, side, 45.0, 0.0)
    assert abs(((found["yaw"] - 90 + 180) % 360) - 180) <= 8 and found["iou"] > 0.9


def test_poses_count_in_the_score():
    m = VoxelModel.empty("t", "t", [16, 16, 16])
    apply_ops(m, [{"op": "sphere", "center": [8, 7, 8], "radii": [5, 6, 4], "c": 4}])
    main = spritecam.project(m, 0.0, 45.0)
    shifted = np.zeros_like(main)
    shifted[:, 2:] = main[:, :-2]
    shifted[: main.shape[0] // 2] = 0
    ref = spritecam.SpriteRef("front", "walk", main, poses=[main, shifted])
    vm = spritecam.match(m, [ref])[0]
    assert len(vm.pose_ious) == 2 and vm.pose_ious[0] > 0.95 and vm.score < vm.iou
    assert "other frames" in vm.summary() and vm.to_dict()["poses"]
    assert spritecam.board([ref]).height > spritecam.board([spritecam.SpriteRef("front", "x", main)]).height


# ------------------------------------------------------------------------------------------------ lit renders and spins
def lit_model() -> VoxelModel:
    m = VoxelModel.empty("t", "torre", [10, 14, 10])
    apply_ops(m, [{"op": "box", "from": [1, 0, 1], "to": [8, 2, 8], "c": 3}, {"op": "box", "from": [3, 3, 3], "to": [6, 12, 6], "c": 10},
                  {"op": "box", "from": [2, 12, 2], "to": [7, 13, 7], "c": 20}])
    return m


def test_lit_render_casts_shadows_on_the_ground():
    m = lit_model()
    sc = lit.LitScene(m)
    # the ground right behind the tower, away from the light, is in its shadow; far from it, it is lit
    shade = sc.light_c[sc.kind == 1].mean(axis=1)
    pts = sc.o[sc.kind == 1]
    behind = (np.abs(pts[:, 0] - 6) < 1) & (np.abs(pts[:, 2] - 0) < 1)     # opposite the light (from front-left, above)
    far = (pts[:, 0] < 0) & (pts[:, 2] > 9)                                    # front-left, towards the light
    assert behind.any() and far.any() and shade[behind].mean() < shade[far].mean()
    img = sc.render(45, 30, 8)
    a = np.asarray(img)
    assert img.width > 80 and (a[..., 3] == 255).all()
    from seurat_hoard.render import render
    assert render(m, "lit", size=200).width <= 210 and render(m, "lit_top", size=200, background="dark").height > 50
    assert render(m, "sheet_lit", size=600).width >= 500


def test_spin_gif_and_webm():
    m = lit_model()
    gif = lit.spin(m, "gif", frames=8, size=128, seconds=1)
    im = Image.open(io.BytesIO(gif))
    assert im.format == "GIF" and im.n_frames == 8
    if lit.ffmpeg_path():
        webm = lit.spin(m, "webm", frames=8, size=128, seconds=1)
        assert webm[:4] == b"\x1a\x45\xdf\xa3"                              # Matroska/WebM
    with pytest.raises(Exception):
        lit.spin(m, "avi")


# ------------------------------------------------------------------------------------------------ scenes by parts
def test_turn_and_compose_settle_merge():
    g = np.zeros((4, 2, 6), np.uint8)
    g[1, 0, 5] = 1                                                         # at the front (+z)
    t = scene.turn(g, 90)
    assert t.shape == (6, 2, 4) and t[5, 0, 2] == 1                       # now at +x
    assert (scene.turn(g, 360) == g).all() and scene.turn(g, 180)[2, 0, 0] == 1
    ground = scene.Part("suelo", "", [20, 2, 20], [scene.Placement([0, 0, 0], 0, False)], ground=True,
                        grid=np.ones((20, 2, 20), np.uint8), palette=["#55aa55"])
    tree = np.zeros((3, 6, 3), np.uint8)
    tree[1, :4, 1] = 1
    tree[:, 4:, :] = 2
    parts = [scene.Part("árbol", "", [3, 6, 3], [scene.Placement([5, 0, 5]), scene.Placement([12, 0, 12])], grid=tree,
                        palette=["#884400", "#55aa55"]),
             ground]
    grid, pal, report = scene.compose([20, 16, 20], parts)
    assert pal == ["#55aa55", "#884400"]                                  # equal colours shared, the ground placed first
    assert grid[6, 2, 6] == 2 and grid[6, 1, 6] == 1 and grid[6, 6, 6] == 1   # trunk set down on the ground (top y = 1)
    assert grid[13, 2, 13] == 2 and len(report) == 3
    out = scene.compose([20, 16, 20], [scene.Part("x", "", [3, 6, 3], [scene.Placement([30, 0, 30])], grid=tree, palette=["#000000"])])[2]
    assert out[0].get("outside")
    parts2 = scene.parse_plan({"objects": [{"name": "banco", "size": [200, 8, 4], "placements": [{"position": [1, 0, 2], "rotation": 95}]},
                                           {"name": "suelo", "ground": True, "size": [64, 4, 64]}]}, [64, 32, 64])
    assert parts2[0].ground and parts2[1].size[0] == 64 and parts2[1].placements[0].rotation == 90
    boxes, _ = scene.layout_boxes([64, 32, 64], parts2)
    assert boxes.any()


class SceneLink(FakeLink):
    """Plans a plaza (ground, a bench placed twice), builds each object (run, done) and checks the layout."""

    def chat(self, messages, images=None, **kw):
        self.calls.append({"messages": messages, "images": images or [], **kw})
        system = messages[0]["content"]
        last = messages[-1]["content"]
        last = last[0]["text"] if isinstance(last, list) else last
        if "every object is now built and placed" in last:
            text = json.dumps({"notes": "move the second bench", "placements": [{"name": "banco", "index": 1, "position": [30, 0, 8], "rotation": 270}]})
        elif system.startswith("You are a voxel scene planner"):
            text = json.dumps({"notes": "a small square", "objects": [
                {"name": "suelo", "ground": True, "description": "paving stones", "size": [48, 3, 48], "placements": [{"position": [0, 0, 0]}]},
                {"name": "banco", "description": "a wooden bench", "size": [10, 5, 4],
                 "placements": [{"position": [6, 0, 20], "rotation": 0}, {"position": [30, 0, 20], "rotation": 90}]}]})
        elif last.startswith("It ran"):
            text = json.dumps({"action": "done", "notes": "ok"})
        else:
            code = "PALETTE = ['#a0a0a0', '#8a5a2b']\ndef build(g):\n    box(g, 2 if Y < 4 else 1, (0, 0, 0), (X-1, min(Y-1, 2), Z-1))\n"
            text = json.dumps({"action": "run", "notes": "first", "code": code})
        return SimpleNamespace(text=text, model="deepseek-v4.1-flash")


def make_client(tmp_path, link):
    from fastapi.testclient import TestClient

    from seurat_hoard.config import Config
    from seurat_hoard.main import create_app
    from seurat_hoard.services import Services

    cfg = Config(data_dir=tmp_path / "data", port=5250, data_dir_configured=True, background=False)
    svc = Services(cfg, link_factory=lambda: link)
    return TestClient(create_app(cfg, services=svc), base_url="http://127.0.0.1:5250", client=("127.0.0.1", 50000)), svc


def test_scene_by_parts(tmp_path):
    link = SceneLink()
    client, svc = make_client(tmp_path, link)
    with client as c:
        job = c.post("/api/generate", json={"prompt": "una plaza con dos bancos", "method": "scene", "size": [48, 24, 48],
                                            "rounds": 1, "steps": 3}).json()
        st = c.get(f"/api/jobs/{job['job_id']}").json()
        assert st["status"] == "done", st
        assert [p["name"] for p in st["parts"]] == ["suelo", "banco"] and st["parts"][1]["count"] > 0
        assert st["rounds"][0]["notes"].startswith("plan") and any(r.get("part") == "banco" for r in st["rounds"])
        assert any("revisión: 1 colocaciones" in l["text"] for l in st["log"])
        model = c.get(f"/api/models/{st['model_id']}").json()
        assert model["size"] == [48, 24, 48] and model["source"]["kind"] == "scene" and len(model["source"]["parts"]) == 2
        assert "program" in model["source"]["parts"][1]
        ys = {v[1] for v in model["voxels"] if v[0] < 16 and v[2] >= 20}
        assert min(ys) == 0 and 3 in ys                                    # the bench stands on the 3-high ground
        # every round with voxels opens in the editor and can be saved
        rnd = st["rounds"][-1]["round"]
        r = c.get(f"/api/jobs/{st['id']}/rounds/{rnd}/model").json()
        assert r["preview"] and r["voxels"] and r["size"] == [48, 24, 48]


def test_rounds_open_in_3d_and_save(tmp_path):
    link = AgentLink()
    client, svc = make_client(tmp_path, link)
    with client as c:
        job = c.post("/api/generate", json={"prompt": "a bird", "size": [16, 12, 16], "method": "agent", "steps": 8}).json()
        st = c.get(f"/api/jobs/{job['job_id']}").json()
        snaps = [r for r in st["rounds"] if r.get("snapshot")]
        assert snaps, st["rounds"]
        rnd = snaps[0]["round"]
        r = c.get(f"/api/jobs/{st['id']}/rounds/{rnd}/model").json()
        assert r["id"] == "" and r["count"] > 50 and len(r["voxels"]) == r["count"]
        saved = c.post(f"/api/jobs/{st['id']}/rounds/{rnd}/save", json={"name": "ronda guardada"}).json()
        m = c.get(f"/api/models/{saved['id']}").json()
        assert m["name"] == "ronda guardada" and m["source"]["kind"] == "round" and m["count"] == r["count"]
        head = {"Authorization": "Bearer " + svc.token}
        res = c.post("/api/agent/call", json={"name": "generate_round", "arguments": {"job_id": st["id"], "round": rnd}}, headers=head).json()
        assert res["_image"]["mime"] == "image/png" and rnd in res["rounds_with_voxels"]
        assert c.get(f"/api/jobs/{st['id']}/rounds/999/model").status_code == 404
        assert c.get("/api/jobs/..%2F/rounds/1/model").status_code in (404, 400)


def test_sheet_in_tools_generation_calibration_and_spin(tmp_path):
    link = AgentLink()
    client, svc = make_client(tmp_path, link)
    with client as c:
        head = {"Authorization": "Bearer " + svc.token}
        ref = c.post("/api/references?name=walk", content=walk_sheet(scale=2), headers={"Content-Type": "image/png"}).json()
        fr = c.post("/api/agent/call", json={"name": "reference_frames", "arguments": {"reference_id": ref["id"]}}, headers=head).json()
        assert fr["count"] == 12 and fr["_image"]["mime"] == "image/png" and len(fr["views"]) == 4
        assert c.get(f"/api/references/{ref['id']}/frames").json()["rows"] == 4
        assert c.get(f"/api/references/{ref['id']}/frames.png").headers["content-type"] == "image/png"
        # a sheet as the generation's pictures: four sides, two more frames (poses) each
        job = c.post("/api/generate", json={"references": [{"id": ref["id"], "role": "sheet"}], "method": "agent", "steps": 8,
                                            "target": "sprite"}).json()
        st = c.get(f"/api/jobs/{job['job_id']}").json()
        assert st["status"] == "done", st
        assert sorted(s["role"] for s in st["sprites"]) == ["back", "front", "side", "side"] and all(s["poses"] == 2 for s in st["sprites"])
        assert any(l["text"].startswith("fotogramas") for l in st["log"])
        user0 = link.calls[0]["messages"][-1]["content"]
        user0 = user0[0]["text"] if isinstance(user0, list) else user0
        assert "more frames of the same side" in user0
        views = [r for r in st["rounds"] if r.get("views")]
        assert views and any(v.get("poses") for v in views[0]["views"])
        mid = st["model_id"]
        # frames chosen by hand, then the calibration tool and its REST route
        refs = [{"id": ref["id"], "role": "front", "frames": [0, 1, 2]}, {"id": ref["id"], "role": "side", "frames": [6, 7, 8]}]
        cal = c.post(f"/api/models/{mid}/calibrate", json={"references": refs}).json()
        assert len(cal["cameras"]) == 2 and cal["image"].startswith("data:image/png") and cal["match"] >= cal["match_before"] - 1e-6
        assert cal["suggested_settings"]["sprite_pitch"] is not None and not cal["applied"]
        cal2 = c.post("/api/agent/call", json={"name": "model_sprite_calibrate", "arguments": {
            "model_id": mid, "references": refs, "apply": True}}, headers=head).json()
        assert cal2["applied"] and svc.settings["sprite_pitch"] == cal2["applied"]["sprite_pitch"]
        # spins: REST download and the tool writing a file
        gif = c.get(f"/api/models/{mid}/spin/gif?frames=8&size=128")
        assert gif.status_code == 200 and gif.headers["content-type"] == "image/gif" and gif.content[:3] == b"GIF"
        res = c.post("/api/agent/call", json={"name": "model_spin", "arguments": {"model_id": mid, "frames": 8, "size": 128,
                                                                                   "folder": str(tmp_path / "out")}}, headers=head).json()
        assert res["path"].endswith(".gif") and (tmp_path / "out").exists()
        look = c.post("/api/agent/call", json={"name": "model_look", "arguments": {"model_id": mid, "view": "lit", "yaw": 200, "pitch": 40}},
                      headers=head).json()
        assert look["_image"]["mime"] == "image/png"
        assert c.get(f"/api/models/{mid}/render.png?view=lit_back_left&size=256").status_code == 200
        assert c.get(f"/api/models/{mid}/spin/avi").status_code == 400
