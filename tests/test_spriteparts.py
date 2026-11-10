"""Characters from sprites, part by part: the profile reader, the plan, the builder, the game camera and the generation
with a stand-in for the language model."""

from __future__ import annotations

import io
import json
from types import SimpleNamespace

import numpy as np
import pytest
from PIL import Image, ImageDraw

from seurat_hoard import spriteparts as sp

RED, DARK_RED, PALE_RED, GOLD, GREY, WHITE, INK = ((200, 60, 40), (140, 40, 25), (230, 110, 90), (235, 160, 20), (150, 160, 175),
                                                 (240, 240, 250), (0, 0, 0))


def bird_side() -> np.ndarray:
    """A small bird in profile, facing right: body, head, beak, a raised wing with the far wing peeking out behind it,
    a tail, a leg and an eye, with a black outline."""
    img = Image.new("RGBA", (42, 32), (0, 0, 0, 0))
    d = ImageDraw.Draw(img)
    d.polygon([(15, 2), (21, 1), (21, 12), (15, 13)], fill=PALE_RED + (255,), outline=INK + (255,))        # far wing
    d.polygon([(2, 17), (11, 15), (11, 23), (1, 25)], fill=GOLD + (255,), outline=INK + (255,))             # tail
    d.ellipse([10, 12, 27, 25], fill=RED + (255,), outline=INK + (255,))                                    # body
    d.ellipse([24, 5, 34, 15], fill=RED + (255,), outline=INK + (255,))                                     # head
    d.polygon([(34, 8), (40, 10), (34, 12)], fill=GOLD + (255,), outline=INK + (255,))                      # beak
    d.polygon([(8, 4), (15, 3), (18, 15), (11, 16)], fill=DARK_RED + (255,), outline=INK + (255,))          # near wing
    d.rectangle([17, 25, 20, 30], fill=GREY + (255,), outline=INK + (255,))                                 # leg
    d.point((30, 9), fill=WHITE + (255,))
    d.point((31, 9), fill=INK + (255,))
    return np.asarray(img).copy()


def bird_back() -> np.ndarray:
    """The same bird from behind: a body with the head above it and a wing to each side."""
    img = Image.new("RGBA", (44, 30), (0, 0, 0, 0))
    d = ImageDraw.Draw(img)
    d.polygon([(2, 2), (8, 1), (18, 16), (14, 18)], fill=DARK_RED + (255,), outline=INK + (255,))
    d.polygon([(41, 2), (35, 1), (25, 16), (29, 18)], fill=DARK_RED + (255,), outline=INK + (255,))
    d.ellipse([15, 12, 28, 26], fill=RED + (255,), outline=INK + (255,))
    d.ellipse([17, 5, 26, 14], fill=RED + (255,), outline=INK + (255,))
    return np.asarray(img).copy()


PLAN = {"facing": "right", "parts": [
    {"name": "body", "type": "solid", "where": [[10, 12, 27, 25]], "width": 12},
    {"name": "head", "type": "solid", "where": [[24, 5, 34, 15]]},
    {"name": "beak", "type": "solid", "where": [[34, 8, 41, 12]], "colors": ["#eba014"]},
    {"name": "tail", "type": "fan", "where": [[0, 15, 11, 25]], "colors": ["#eba014"], "base": 4, "width": 14},
    {"name": "far wing", "type": "skip", "where": [{"poly": [[15, 1], [21, 1], [21, 12], [18, 12], [16, 4]]}]},
    {"name": "wings", "type": "wings", "where": [{"poly": [[8, 4], [15, 3], [18, 15], [11, 16]]}], "root": [15, 14],
     "out_up": 1.0, "out_back": 0.3, "span": 36},
    {"name": "legs", "type": "pair", "where": [[17, 25, 20, 30]], "offset": 3, "width": 2},
    {"name": "eye", "type": "decal", "where": [[30, 9, 31, 9]]},
]}


def png_of(arr: np.ndarray) -> bytes:
    buf = io.BytesIO()
    Image.fromarray(arr, "RGBA").save(buf, "PNG")
    return buf.getvalue()


def test_profile_finds_the_ink_and_the_patches_of_colour():
    p = sp.read_profile(bird_side())
    assert p.shape == bird_side().shape[:2]
    assert "#000000" in p.palette and "#000000" not in p.letters           # ink has no letter
    assert p.ink.any() and p.rim.any() and (p.rim <= p.ink).all()
    # every pixel belongs to a cell, and the big patches are separate cells
    assert ((p.cells > 0) == p.mask).all()
    body, wing = p.cells[18, 20], p.cells[9, 13]
    assert body != wing and p.info[body - 1]["colour"] == "#c83c28" and p.info[wing - 1]["colour"] == "#8c2819"
    text = sp.colour_map(p)
    assert "#" in text and text.splitlines()[1].startswith("   0 ")
    flipped = sp.read_profile(bird_side(), facing="left")
    assert (flipped.mask == p.mask[:, ::-1]).all()


def test_the_model_has_the_sprite_s_profile_and_real_volume():
    side = bird_side()
    p = sp.read_profile(side)
    b = sp.build(p, PLAN)
    assert not b.notes
    g = b.grid
    assert (g[::-1] > 0).tolist() == (g > 0).tolist()                      # left-right symmetric
    names = [pt["name"] for pt in b.parts]
    skip = b.label == names.index("far wing") + 1
    assert skip.sum() > 10
    # seen from the side: every sprite pixel that is not the far wing is there, and nothing outside the sprite
    sil = (g > 0).any(axis=0)                                              # (Y, Z)
    h, w = p.shape
    got = np.zeros((h, w), bool)
    pl = b.place
    got[:, :] = sil[pl["y0"]: pl["y0"] + h, pl["z0"]: pl["z0"] + w][::-1]
    assert (got <= p.mask).all() and (got >= (p.mask & ~skip)).all()
    # the flank shows the sprite's colours (ink toned down), e.g. the dark red wing and the gold tail
    rgb = sp.palette_rgb(b.palette)
    img = sp.project(g, rgb, "side")
    assert img.shape[:2] == (h, w) or img[..., 3].sum() > 0
    wing = b.parts[names.index("wings")]
    assert 32 <= wing["span_built"] <= 36 and wing["root_used"] == [15.0, 14.0]
    assert b.parts[names.index("body")]["width_built"] == 12
    assert b.parts[names.index("tail")]["width_built"] in (12, 14)
    # two legs with air between them
    y = pl["y0"] + 1
    row = (b.pid[:, y, :] == names.index("legs") + 1).any(axis=1)
    xs = np.flatnonzero(row)
    assert len(xs) >= 4 and (np.diff(xs) > 1).any()
    # the eye is on both flanks and nowhere else
    eye_white = b.palette.index("#f0f0fa") + 1
    ex, ey, ez = np.nonzero(g == eye_white)
    assert len(ex) == 2 and ex[0] + ex[1] == g.shape[0] - 1
    # the outline is not black any more: the edge takes the colour of its part
    assert "#000000" in b.palette and (g == b.palette.index("#000000") + 1).sum() <= 2      # only the pupil


def test_shapes_are_rough_and_plans_are_checked():
    p = sp.read_profile(bird_side())
    rough = {"parts": [{"name": "all", "type": "solid", "where": [[0, 0, 60, 60]]},
                       {"name": "wing", "type": "wings", "where": [[9, 5, 16, 14]], "colors": ["#8c2819"]}]}
    label = sp.assign(p, sp.normalise_plan(rough))
    wing_cell = p.cells == p.cells[9, 13]
    assert (label[wing_cell & ~p.ink] == 2).all()                           # the whole patch, not just the rectangle
    assert (label[p.mask] > 0).all()
    with pytest.raises(sp.PlanError):
        sp.normalise_plan({"parts": [{"name": "x", "type": "tentacle"}]})
    with pytest.raises(sp.PlanError):
        sp.normalise_plan({"parts": []})
    with pytest.raises(sp.PlanError):
        sp.build(p, {"parts": [{"name": "x", "type": "solid", "where": [[1, 2, 3]]}]})
    changed = sp.apply_changes(PLAN, {"wings": {"span": 30}}, remove=["eye"], add=[{"name": "crest", "type": "fin", "where": [[26, 3, 30, 6]]}])
    names = [pt["name"] for pt in changed["parts"]]
    assert "eye" not in names and names[-1] == "crest" and changed["parts"][names.index("wings")]["span"] == 30
    assert PLAN["parts"][5]["span"] == 36                                   # the original is untouched
    with pytest.raises(sp.PlanError):
        sp.apply_changes(PLAN, {"tentacle": {"width": 2}})
    auto = sp.build(p, sp.auto_plan(p))
    assert (auto.grid > 0).sum() > p.mask.sum() * 3                         # one rounded solid with volume


def test_the_game_camera_compares_with_the_other_sprites():
    p = sp.read_profile(bird_side())
    b = sp.build(p, PLAN)
    back = bird_back()
    sprites = {"side": bird_side(), "back": back}
    fam = sp.families_of(*sprites.values())
    cams = sp.tune_camera(b, sprites, fam)
    assert set(cams) == {"back"} and 0 <= cams["back"] <= 1.05
    res = sp.evaluate(b, sprites, cams, fam=fam)["views"]["back"]
    assert 0.3 < res["iou"] <= 1 and res["match"] <= res["iou"] + 1e-9
    same = sp.compare(back, back, fam)
    assert same["iou"] == 1.0 and same["shift"] == [0, 0]
    assert abs(sp.symmetry_axis(back) - 21.5) <= 1
    text = sp.measure(back, "back")
    assert "symmetric about column" in text and "row" in text
    # a wider wingspan shows from behind
    wide = sp.build(p, sp.apply_changes(PLAN, {"wings": {"span": 44}}))
    rgb = sp.palette_rgb(wide.palette)
    assert sp.project(wide.grid, rgb, "back", 0.3).shape[1] > sp.project(b.grid, sp.palette_rgb(b.palette), "back", 0.3).shape[1]
    pic, results = sp.match_picture(b, sprites, cams, fam)
    assert pic.width > 200 and "back" in results
    assert sp.parts_picture(p, b.label, b.parts).width > 400


def test_a_fan_s_end_is_painted_from_the_back_sprite():
    side = bird_side()
    p = sp.read_profile(side)
    back = bird_back().copy()
    # give the back sprite a striped gold tail under the body
    d = Image.fromarray(back, "RGBA")
    dr = ImageDraw.Draw(d)
    for x in range(15, 29):
        dr.line([(x, 24), (x, 29)], fill=(GOLD if x % 2 else (160, 100, 10)) + (255,))
    back = np.asarray(d).copy()
    b = sp.build(p, PLAN)
    before = b.grid.copy()
    n = sp.paint_caps(b, {"side": side, "back": back}, {"back": 0.3})
    assert n > 0 and (b.grid != before).sum() == n
    assert ((b.grid > 0) == (before > 0)).all()                             # colours only
    tail = [pt for pt in b.parts if pt["name"] == "tail"][0]
    assert tail.get("painted") == "back"
    # the flank still is the side sprite
    X = b.grid.shape[0]
    occ = b.grid > 0
    first = occ.argmax(axis=0)
    ys, zs = np.nonzero(occ.any(axis=0))
    assert (b.grid[first[ys, zs], ys, zs] == before[first[ys, zs], ys, zs]).all()


def test_cutout_removes_the_background_the_figure_encloses():
    side = bird_side()
    h, w = side.shape[:2]
    flat = np.zeros((h + 8, w + 8, 4), np.uint8)
    flat[..., :3] = (200, 120, 232)
    flat[..., 3] = 255
    m = side[..., 3] > 0
    flat[4: 4 + h, 4: 4 + w][m] = side[m]
    # a pocket of background closed in by ink
    flat[4 + 18, 4 + 15: 4 + 18] = (200, 120, 232, 255)
    out = sp.cutout(png_of(flat))
    rs, cs = np.nonzero(m)
    assert out.shape[:2] == (rs.max() - rs.min() + 1, cs.max() - cs.min() + 1)
    assert not (out[..., 3] > 0)[18 - rs.min(), 15 - cs.min(): 18 - cs.min()].any()
    assert (np.abs(out[out[..., 3] > 0][:, :3].astype(int) - np.array([200, 120, 232])).sum(axis=1) > 8).all()


class PartsLink:
    """Answers like the model in a parts generation: a plan with a mistake, the plan, an adjustment, a look, done."""

    def __init__(self):
        self.calls = []
        self.sync = SimpleNamespace(chat=self.chat, resolve=self.resolve)

    def resolve(self, capability):
        return SimpleNamespace(state="resolved", url="http://spark:8888/v1/chat/completions", model="deepseek-v4.1-flash",
                               provider="test", api="link", reason="test")

    def chat(self, messages, images=None, **kw):
        self.calls.append({"messages": messages, "images": images or [], **kw})
        n = len(self.calls)
        answers = [{"action": "build", "notes": "wrong type", "plan": {"parts": [{"name": "b", "type": "tentacle"}]}},
                   {"action": "build", "notes": "body, wings, legs", "plan": PLAN},
                   {"action": "adjust", "notes": "wider wings", "set": {"wings": {"span": 40}, "nobody": {"width": 3}}},
                   {"action": "adjust", "notes": "wider wings", "set": {"wings": {"span": 40}}},
                   {"action": "look", "view": "back_left"},
                   {"action": "done", "notes": "reads as the bird"}]
        return SimpleNamespace(text=json.dumps(answers[min(n, len(answers)) - 1]), model="deepseek-v4.1-flash")


def test_the_far_wing_inside_a_wing_is_left_out_by_itself():
    """One rough shape over both wings of the profile: the far one (same colours, behind a line of its own) is not
    built unless the part says keep."""
    p = sp.read_profile(bird_side())
    parts = [dict(q) for q in PLAN["parts"] if q["name"] not in ("far wing", "wings")]
    wing = {"name": "wings", "type": "wings", "where": [[8, 1, 21, 16]], "root": [15, 14], "span": 36}
    built = sp.build(p, sp.normalise_plan({"facing": "right", "parts": parts[:1] + [wing] + parts[1:]}))
    far = [q for q in built.parts if q.get("auto") == "wings"]
    assert len(far) == 1 and far[0]["type"] == "skip" and far[0]["pixels"] >= 30
    near = next(q for q in built.parts if q["name"] == "wings")
    assert near["pixels"] >= 60 and any("FAR" in n for n in built.notes)
    rs, cs = np.nonzero(built.label == built.parts.index(far[0]) + 1)
    assert cs.min() >= 15 and rs.max() <= 13                                # the pale wing behind, not the near one
    kept = sp.build(p, sp.normalise_plan({"facing": "right", "parts": parts[:1] + [dict(wing, far="keep")] + parts[1:]}))
    assert not [q for q in kept.parts if q.get("auto")]
    with pytest.raises(sp.PlanError):
        sp.normalise_plan({"parts": [dict(wing, far="maybe")]})


def test_overlapping_shapes_follow_the_patches_of_colour():
    """Rectangles written by a model overlap. A patch stays with the part that has the rest of it, and a shape lying
    inside another one wins there whatever the order."""
    img = Image.new("RGBA", (20, 10), (0, 0, 0, 0))
    d = ImageDraw.Draw(img)
    d.rectangle([0, 0, 19, 9], fill=RED + (255,), outline=INK + (255,))
    d.rectangle([10, 1, 18, 8], fill=GOLD + (255,))
    p = sp.read_profile(np.asarray(img).copy())
    red = np.all(p.rgba[..., :3] == RED, axis=2)
    gold = np.all(p.rgba[..., :3] == GOLD, axis=2)
    borders = sp.normalise_plan({"parts": [{"name": "a", "type": "solid", "where": [[0, 0, 12, 9]]},
                                           {"name": "b", "type": "solid", "where": [[7, 0, 19, 9]]}]})
    label = sp.assign(p, borders)
    assert (label[red] == 1).all() and (label[gold] == 2).all()
    nested = sp.normalise_plan({"parts": [{"name": "spot", "type": "solid", "where": [[10, 1, 18, 8]]},
                                          {"name": "all", "type": "solid", "where": [[0, 0, 19, 9]]}]})
    label = sp.assign(p, nested)
    assert (label[gold] == 1).all() and (label[red] == 2).all()
    assert (label[p.mask] > 0).all()


def test_a_face_comes_from_the_front_sprite_when_it_shows_the_same_pose():
    """A round head: the side sprite colours its flanks, the front sprite what looks at the front camera; a front
    sprite in another pose leaves it alone unless the part asks."""
    side = Image.new("RGBA", (16, 16), (0, 0, 0, 0))
    ImageDraw.Draw(side).ellipse([0, 0, 15, 15], fill=RED + (255,), outline=INK + (255,))
    front = Image.new("RGBA", (16, 16), (0, 0, 0, 0))
    d = ImageDraw.Draw(front)
    d.ellipse([0, 0, 15, 15], fill=RED + (255,), outline=INK + (255,))
    d.rectangle([5, 6, 6, 7], fill=WHITE + (255,))
    d.rectangle([9, 6, 10, 7], fill=WHITE + (255,))
    sprites = {"side": np.asarray(side).copy(), "front": np.asarray(front).copy()}
    plan = {"parts": [{"name": "head", "type": "solid", "where": [[0, 0, 15, 15]], "width": 16}]}
    white = "#%02x%02x%02x" % WHITE

    def painted(plan, sprites):
        built = sp.build(sp.read_profile(sprites["side"]), sp.normalise_plan(plan))
        sp.paint_caps(built, sprites, {"front": 0.0})
        where = np.argwhere(built.grid == built.palette.index(white) + 1) if white in built.palette else np.zeros((0, 3), int)
        return built, where

    built, eyes = painted(plan, sprites)
    assert len(eyes) >= 6 and built.parts[0]["faces"] == "front"
    assert eyes[:, 2].min() > built.grid.shape[2] / 2                      # on the front half only
    side_view = sp.project(built.grid, sp.palette_rgb(built.palette), "side", 0.0)
    assert (side_view[..., 3] >= 128).sum() == (sprites["side"][..., 3] >= 128).sum()
    _, none = painted({"parts": [dict(plan["parts"][0], paint="none")]}, sprites)
    assert len(none) == 0
    other = np.zeros((16, 30, 4), np.uint8)                                # the front sprite of something else entirely
    other[2:14, 1:29] = WHITE + (255,)
    _, off = painted(plan, {"side": sprites["side"], "front": other})
    assert len(off) == 0
    with pytest.raises(sp.PlanError):
        sp.normalise_plan({"parts": [dict(plan["parts"][0], paint="everything")]})


def test_generation_by_parts_and_its_tools(tmp_path):
    from fastapi.testclient import TestClient

    from seurat_hoard.config import Config
    from seurat_hoard.main import create_app
    from seurat_hoard.services import Services

    cfg = Config(data_dir=tmp_path / "data", port=5250, data_dir_configured=True, background=False)
    link = PartsLink()
    svc = Services(cfg, link_factory=lambda: link)
    with TestClient(create_app(cfg, services=svc), base_url="http://127.0.0.1:5250", client=("127.0.0.1", 50000)) as c:
        ids = {}
        for role, arr in (("side", bird_side()), ("back", bird_back())):
            ids[role] = c.post(f"/api/references?name=bird {role}", content=png_of(arr), headers={"Content-Type": "image/png"}).json()["id"]
        refs = [{"id": i, "role": r} for r, i in ids.items()]
        # no method given: sprites with a side view are built by parts
        job = c.post("/api/generate", json={"references": refs, "prompt": "the bird of the sprites", "steps": 8}).json()
        st = c.get(f"/api/jobs/{job['job_id']}").json()
        assert st["status"] == "done", st
        assert len(link.calls) == 6
        system = link.calls[0]["messages"][0]["content"]
        assert "PART TYPES" in system and '"skip"' in system and "FAR wing" in system
        first = link.calls[0]["messages"][-1]["content"]
        first = first[0]["text"] if isinstance(first, list) else first
        assert "SIDE sprite" in first and "BACK sprite" in first and "symmetric about column" in first
        assert len(link.calls[0]["images"]) == 1                              # the sprites with rulers
        second = link.calls[1]["messages"][-1]["content"]
        second = second[0]["text"] if isinstance(second, list) else second
        assert "could NOT be built" in second and "tentacle" in second
        third = link.calls[2]["messages"][-1]["content"]
        third = third[0]["text"] if isinstance(third, list) else third
        assert "wings (wings)" in third and "silhouettes overlap" in third and "far wing (skip)" in third
        assert len(link.calls[2]["images"]) == 3                              # parts, sprite comparison, lit views
        fourth = link.calls[3]["messages"][-1]["content"]
        fourth = fourth[0]["text"] if isinstance(fourth, list) else fourth
        assert "NOT applied" in fourth and "nobody" in fourth
        assert len(link.calls[5]["images"]) == 1                              # the look
        good = [r for r in st["rounds"] if r.get("render")]
        assert len(good) == 2 and good[0]["parts_render"] and (cfg.data_dir / "renders" / good[0]["sprite_render"]).exists()
        model = c.get(f"/api/models/{st['model_id']}").json()
        src = model["source"]
        assert src["method"] == "parts" and src["plan"]["parts"][5]["span"] == 40 and "partial" not in src
        assert model["count"] > 800 and model["size"][0] >= 40
        head = {"Authorization": "Bearer " + svc.token}

        def call(name, arguments):
            return c.post("/api/agent/call", json={"name": name, "arguments": arguments}, headers=head).json()

        info = call("sprite_parts_read", {"references": refs})
        assert info["side"]["width"] == bird_side().shape[1] or info["side"]["width"] > 30
        assert "a" in info["palette"] and info["_image"]["mime"] == "image/png" and "widths" in info["others"]["back"]
        adj = call("model_parts_adjust", {"model_id": st["model_id"], "set": {"body": {"width": 16}}})
        assert "body (solid)" in adj["report"] and "width 16" in adj["report"] and adj["_image"]["mime"] == "image/png"
        assert len(c.get(f"/api/models/{st['model_id']}/versions").json()["versions"]) >= 1
        built = call("model_from_parts", {"references": refs, "name": "Bird by hand", "plan": PLAN})
        assert built["name"] == "Bird by hand" and built["count"] > 800 and built["views"]["back"]["iou"] > 0.3
        auto = call("model_from_parts", {"references": [{"id": ids["side"], "role": "side"}]})
        assert auto["count"] > 500 and auto["plan"]["parts"][0]["type"] == "solid"
        bad = c.post("/api/agent/call", json={"name": "model_from_parts", "arguments": {"references": [{"id": ids["back"], "role": "back"}]}},
                     headers=head)
        assert bad.status_code == 400 and "side" in json.dumps(bad.json())
        # asking for the old loop still gives it
        assert "parts" not in str(svc.settings.get().get("method"))
