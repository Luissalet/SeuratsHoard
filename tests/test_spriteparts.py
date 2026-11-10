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


PLAN = {"facing": "right", "eye": [30, 9], "parts": [
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
        self.checks = []
        self.answers = None
        self.sync = SimpleNamespace(chat=self.chat, resolve=self.resolve)

    def resolve(self, capability):
        return SimpleNamespace(state="resolved", url="http://spark:8888/v1/chat/completions", model="deepseek-v4.1-flash",
                               provider="test", api="link", reason="test")

    def chat(self, messages, images=None, **kw):
        if "You check how a game sprite was read" in str(messages[0]["content"]):
            # the second opinion on the first plan that builds: a call of its own, outside the conversation
            self.checks.append({"messages": messages, "images": images or []})
            wrong = [{"part": "legs", "really": "the tail feathers", "why": "test"}, {"part": "body", "really": "a wing", "sure": False}]
            return SimpleNamespace(text=json.dumps({"head": "head", "wrong": wrong if len(self.checks) == 1 else [], "missing": [],
                                                    "verdict": "ok"}), model="deepseek-v4.1-flash")
        self.calls.append({"messages": messages, "images": images or [], **kw})
        n = len(self.calls)
        answers = self.answers or [{"action": "build", "notes": "wrong type", "plan": {"parts": [{"name": "b", "type": "tentacle"}]}},
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


def test_the_eye_settles_which_shape_is_the_head():
    """A plan that calls a wing the head, or faces the wrong way, is refused with the reason."""
    p = sp.read_profile(bird_side())
    assert [q["name"] for q in sp.build(p, sp.normalise_plan(PLAN)).parts if q.get("eye") == "[30, 9]"] == ["eye"]
    with pytest.raises(sp.PlanError, match="HEAD"):
        sp.build(p, sp.normalise_plan(dict(PLAN, eye=[13, 9])))             # on the wing
    with pytest.raises(sp.PlanError, match="back third"):
        sp.build(p, sp.normalise_plan(dict(PLAN, eye=[12, 20])))            # on the body, at the tail end
    with pytest.raises(sp.PlanError, match="not on the character"):
        sp.build(p, sp.normalise_plan(dict(PLAN, eye=[40, 30])))
    with pytest.raises(sp.PlanError):
        sp.normalise_plan(dict(PLAN, eye="head"))
    left = sp.read_profile(bird_side()[:, ::-1], "left")
    flipped = dict(PLAN, facing="left", eye=[41 - 30, 9],
                   parts=[dict(q, where=[[41 - s[2], s[1], 41 - s[0], s[3]] if isinstance(s, list) else
                                         {"poly": [[41 - c, r] for c, r in s["poly"]]} for s in q["where"]],
                               **({"root": [41 - q["root"][0], q["root"][1]]} if q.get("root") else {})) for q in PLAN["parts"]])
    assert [q["name"] for q in sp.build(left, sp.normalise_plan(flipped)).parts if q.get("eye") == "[11, 9]"] == ["eye"]


def test_wings_open_the_way_the_other_sprites_say():
    """A wings part that does not say how it opens is tried from straight up to straight back and keeps the opening
    whose back view overlaps the back sprite best; one that says it is left alone."""
    p = sp.read_profile(bird_side())
    sprites = {"side": bird_side(), "back": bird_back()}
    free = sp.normalise_plan({"facing": "right", "eye": [30, 9],
                              "parts": [{k: v for k, v in q.items() if k not in ("out_up", "out_back")} for q in PLAN["parts"]]})
    tuned = sp.tune_wings(p, free, sprites)
    wings = next(q for q in tuned["parts"] if q["type"] == "wings")
    assert wings["opening"] in sp.WING_OPENINGS and wings["out_up"] is not None
    assert next(q for q in free["parts"] if q["type"] == "wings")["out_up"] is None          # the plan given is not touched
    scores = {}
    for deg, (up, back) in ((30, (0.866, 0.5)), (75, (0.259, 0.966)), (wings["opening"], (wings["out_up"], wings["out_back"]))):
        plan = sp.normalise_plan({**free, "parts": [dict(q, out_up=up, out_back=back) if q["type"] == "wings" else q for q in free["parts"]]})
        built = sp.build(p, plan)
        scores[deg] = sp.evaluate(built, sprites, sp.tune_camera(built, sprites))["views"]["back"]["iou"]
    assert scores[wings["opening"]] >= max(scores.values()) - 1e-6
    assert "opening found by Seurat" in sp.report(sp.build(p, tuned))
    fixed = sp.normalise_plan(PLAN)
    assert sp.tune_wings(p, fixed, sprites) is fixed
    assert sp.tune_wings(p, free, {"side": bird_side()}) is free                              # nothing to compare with


def test_nothing_floats_beside_the_body():
    """Legs set wider apart than the body is thick there would hang in the air: they are brought in until they touch
    it, without a bar and without a word; a part made of lines only is never black."""
    p = sp.read_profile(bird_side())
    wide = sp.normalise_plan({**PLAN, "parts": [dict(q, offset=9, width=2) if q["name"] == "legs" else q for q in PLAN["parts"]]})
    built = sp.build(p, wide)
    legs = next(q for q in built.parts if q["name"] == "legs")
    assert legs["settled"] >= 1 and "brought in to touch the body" in sp.report(built)
    solids = [k + 1 for k, q in enumerate(built.parts) if q["type"] == "solid"]
    assert not sp._floating(np.where(built.grid > 0, built.pid, 0), solids).any()
    assert not any("float" in n or "bar" in n for n in built.notes)
    k = built.parts.index(legs) + 1
    xs = np.nonzero((built.pid == k).any(axis=(1, 2)))[0]
    assert xs.max() - xs.min() + 1 < 2 * 9                                    # closer together than asked
    assert (built.pid == k).sum() == (built.pid[::-1] == k).sum() and ((built.pid == k) == (built.pid[::-1] == k)).all()
    assert "settled" not in next(q for q in sp.build(p, sp.normalise_plan(PLAN)).parts if q["name"] == "legs")
    lines = sp.normalise_plan({**PLAN, "parts": [q for q in PLAN["parts"] if q["name"] != "legs"]
                               + [{"name": "legs", "type": "pair", "where": [[17, 25, 17, 30]], "offset": 3, "width": 2}]})
    built = sp.build(p, lines)                                                 # only the leg's outline
    k = [q["name"] for q in built.parts].index("legs") + 1
    assert "#000000" not in {built.palette[i - 1] for i in np.unique(built.grid[built.pid == k])}


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
    # one patch that two parts really share: rough rectangles meet halfway, a polygon is taken at its word
    plain = Image.new("RGBA", (20, 10), (0, 0, 0, 0))
    ImageDraw.Draw(plain).rectangle([0, 0, 19, 9], fill=RED + (255,), outline=INK + (255,))
    q = sp.read_profile(np.asarray(plain).copy())
    label = sp.assign(q, borders)
    assert label[5, 8] == 1 and label[5, 11] == 2
    drawn = sp.normalise_plan({"parts": [{"name": "a", "type": "solid", "where": [[0, 0, 12, 9]]},
                                         {"name": "b", "type": "solid", "where": [{"poly": [[7, 0], [19, 0], [19, 9], [7, 9]]}]}]})
    label = sp.assign(q, drawn)
    assert label[5, 8] == 2 and label[5, 5] == 1


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


def test_a_head_is_painted_when_only_the_rest_of_the_figure_holds_another_pose():
    """A front sprite with the arms out is another pose as a whole, but over the rows of the head it is the model: the
    head takes its face from it, the body does not."""
    side = Image.new("RGBA", (16, 30), (0, 0, 0, 0))
    d = ImageDraw.Draw(side)
    d.rectangle([5, 14, 10, 29], fill=GREY + (255,), outline=INK + (255,))
    d.ellipse([0, 0, 15, 15], fill=RED + (255,), outline=INK + (255,))
    front = Image.new("RGBA", (30, 30), (0, 0, 0, 0))
    d = ImageDraw.Draw(front)
    d.rectangle([12, 14, 17, 29], fill=GREY + (255,), outline=INK + (255,))
    d.rectangle([0, 17, 29, 21], fill=GOLD + (255,), outline=INK + (255,))                 # the arms, held out
    d.ellipse([7, 0, 22, 15], fill=RED + (255,), outline=INK + (255,))
    d.rectangle([12, 6, 13, 7], fill=WHITE + (255,))
    d.rectangle([16, 6, 17, 7], fill=WHITE + (255,))
    sprites = {"side": np.asarray(side).copy(), "front": np.asarray(front).copy()}
    plan = sp.normalise_plan({"parts": [{"name": "head", "type": "solid", "where": [[0, 0, 15, 14]], "width": 16},
                                        {"name": "body", "type": "solid", "where": [[5, 15, 10, 29]], "width": 6}]})
    built = sp.build(sp.read_profile(sprites["side"]), plan)
    fam = sp.families_of(*sprites.values())
    whole = sp.compare(sp.project(built.grid, sp.palette_rgb(built.palette), "front", 0.0), sprites["front"], fam)
    assert whole["iou"] < sp.FACE_IOU
    sp.paint_caps(built, sprites, {"front": 0.0}, fam)
    head, body = built.parts
    assert head.get("faces") == "front" and not body.get("faces")
    white, gold = "#%02x%02x%02x" % WHITE, "#%02x%02x%02x" % GOLD
    assert (built.grid == built.palette.index(white) + 1).sum() >= 6 and gold not in built.palette


def test_a_sheet_opens_the_way_it_is_drawn_and_starts_on_the_body():
    """A flipper hangs below its root: the opening Seurat finds is measured from straight down for it, and the sheet
    starts where it meets the body, not out where the body is widest."""
    img = Image.new("RGBA", (40, 34), (0, 0, 0, 0))
    d = ImageDraw.Draw(img)
    d.ellipse([4, 2, 35, 23], fill=RED + (255,), outline=INK + (255,))
    d.polygon([(14, 22), (22, 22), (12, 33), (6, 33)], fill=GOLD + (255,), outline=INK + (255,))
    side = np.asarray(img).copy()
    back = Image.new("RGBA", (44, 34), (0, 0, 0, 0))
    d = ImageDraw.Draw(back)
    d.ellipse([12, 2, 31, 23], fill=RED + (255,), outline=INK + (255,))
    d.polygon([(14, 20), (2, 31), (6, 33), (18, 23)], fill=GOLD + (255,), outline=INK + (255,))
    d.polygon([(29, 20), (41, 31), (37, 33), (25, 23)], fill=GOLD + (255,), outline=INK + (255,))
    sprites = {"side": side, "back": np.asarray(back).copy()}
    p = sp.read_profile(side)
    plan = sp.normalise_plan({"facing": "right", "parts": [
        {"name": "body", "type": "solid", "where": [[4, 2, 35, 23]], "colors": ["#c83c28"], "width": 20},
        {"name": "flippers", "type": "wings", "where": [[5, 22, 23, 33]], "colors": ["#eba014"], "root": [18, 21], "span": 40}]})
    built = sp.build(p, sp.tune_wings(p, plan, sprites))
    flippers = built.parts[1]
    assert flippers["hangs"] and "degrees from straight down" in sp.report(built)
    assert not built.notes, built.notes                                       # no bar, nothing floating
    mine = np.argwhere(built.pid == 2)
    low = mine[mine[:, 1] == mine[:, 1].min()]                                # its lowest voxels are its widest apart
    cx = built.grid.shape[0] / 2
    assert np.abs(low[:, 0] - cx).max() > np.abs(mine[mine[:, 1] == mine[:, 1].max()][:, 0] - cx).max() + 3
    assert flippers["span_built"] >= 36


def test_a_sprite_facing_left_is_shown_and_reported_in_its_own_columns():
    """The profile is kept facing right inside; what the plan's author reads back (the parts picture, the roots, the
    notes) uses the columns of the sprite as it was given."""
    side = bird_side()[:, ::-1].copy()
    w = side.shape[1]

    def mirrored(shape):
        if isinstance(shape, dict):
            return {"poly": [[w - 1 - c, r] for c, r in shape["poly"]]}
        return [w - 1 - shape[2], shape[1], w - 1 - shape[0], shape[3]]

    parts = [dict(q, where=[mirrored(sh) for sh in q["where"]], **({"root": [w - 1 - q["root"][0], q["root"][1]]} if q.get("root") else {}))
             for q in PLAN["parts"] if q["type"] != "skip"]
    plan = sp.normalise_plan({"facing": "left", "eye": [w - 1 - 30, 9], "parts": parts})
    p = sp.read_profile(side, "left")
    built = sp.build(p, plan)
    wings = next(q for q in built.parts if q["name"] == "wings")
    assert wings["root_used"] == [w - 1 - 15, 14]
    far = next(n for n in built.notes if "FAR" in n)
    assert f"columns {w - 1 - 21}-" in far                                    # the far wing, where the sprite has it
    pic = np.asarray(sp.parts_picture(p, built.label, built.parts, zoom=4).convert("RGBA"))
    ruler = 22
    beak = pic[ruler + 10 * 4 + 2, ruler + (w - 1 - 37) * 4 + 2, :3]              # the beak is on the left, as in the sprite
    assert tuple(int(v) for v in beak) == GOLD


def doll() -> dict[str, np.ndarray]:
    """A doll in three drawings: a round head on a thin body with a leg, in profile; from the front a head that is a
    wide bell (narrow on top), two eyes, the body, the arms held out and two legs apart."""
    side = Image.new("RGBA", (16, 34), (0, 0, 0, 0))
    d = ImageDraw.Draw(side)
    d.rectangle([5, 14, 10, 27], fill=GREY + (255,), outline=INK + (255,))
    d.rectangle([6, 27, 9, 33], fill=GOLD + (255,), outline=INK + (255,))
    d.ellipse([0, 0, 15, 15], fill=RED + (255,), outline=INK + (255,))
    front = Image.new("RGBA", (30, 34), (0, 0, 0, 0))
    d = ImageDraw.Draw(front)
    d.rectangle([12, 14, 17, 27], fill=GREY + (255,), outline=INK + (255,))
    d.rectangle([0, 17, 29, 21], fill=PALE_RED + (255,), outline=INK + (255,))                 # the arms, held out
    d.rectangle([8, 27, 11, 33], fill=GOLD + (255,), outline=INK + (255,))
    d.rectangle([18, 27, 21, 33], fill=GOLD + (255,), outline=INK + (255,))
    d.polygon([(11, 0), (18, 0), (24, 15), (5, 15)], fill=RED + (255,), outline=INK + (255,))  # a bell
    d.rectangle([11, 9, 12, 10], fill=WHITE + (255,))
    d.rectangle([17, 9, 18, 10], fill=WHITE + (255,))
    d.line([(10, 13), (19, 13)], fill=INK + (255,))                                            # a mouth: a line of the drawing
    return {"side": np.asarray(side).copy(), "front": np.asarray(front).copy()}


DOLL = {"facing": "right", "parts": [
    {"name": "head", "type": "solid", "where": [[0, 0, 15, 14]], "front": [[5, 0, 24, 15]]},
    {"name": "body", "type": "solid", "where": [[5, 15, 10, 27]], "front": [[12, 15, 17, 27]]},
    {"name": "legs", "type": "pair", "where": [[6, 28, 9, 33]], "front": [[8, 27, 21, 33]]}]}


def test_each_part_is_built_and_painted_from_how_the_other_sprites_draw_it():
    """A part placed in the front sprite is as wide, at every height, as that sprite draws it, whatever the rest of the
    figure does there; legs stand as far apart as it sets them; and what the part shows to the front is that sprite's
    own drawing of it, without its lines."""
    sprites = doll()
    p = sp.read_profile(sprites["side"])
    built = sp.build(p, sp.normalise_plan(DOLL), sprites=sprites)
    head, body, legs = built.parts
    assert head["across"] == body["across"] == legs["across"] == "front" and "front" in built.views
    assert "across the body as the front sprite draws it" in sp.report(built)
    cx, y0, h = built.place["cx"], built.place["y0"], built.place["h"]

    def width_at(k: int, row: int) -> int:
        xs = np.nonzero((built.pid[:, y0 + (h - 1 - row), :] == k).any(axis=1))[0]
        return int(xs.max() - xs.min() + 1) if len(xs) else 0

    assert width_at(1, 2) <= 12 and width_at(1, 12) >= 17                         # a bell: narrow on top, wide below
    assert abs(width_at(1, 12) - 19) <= 2
    assert body["width_built"] <= 8                                                # the arms (30 across) did not widen the body
    k = 3
    xs = np.nonzero((built.pid == k).any(axis=(1, 2)))[0]
    assert xs.min() == cx - 7 and xs.max() == cx + 6                               # the legs where the front sprite has them
    assert not (built.pid[cx - 3: cx + 3] == k).any()                              # and apart: nothing between them
    sp.paint_caps(built, sprites, {"front": 0.0})
    white = built.palette.index("#%02x%02x%02x" % WHITE) + 1
    eyes = np.argwhere(built.grid == white)
    assert len(eyes) >= 4 and set(built.pid[tuple(eyes.T)]) == {1} and eyes[:, 2].min() > built.grid.shape[2] / 2
    front = sp.project(built.grid, sp.palette_rgb(built.palette), "front", 0.0)
    assert (0, 0, 0) not in {tuple(int(v) for v in px[:3]) for px in front[front[..., 3] > 0]}     # no line of the drawing
    assert "front" in head["faces"]
    # a width given by hand keeps the round form; a shape that catches nothing is said
    by_hand = sp.build(p, sp.normalise_plan({**DOLL, "parts": [dict(DOLL["parts"][0], width=10)] + DOLL["parts"][1:]}), sprites=sprites)
    assert by_hand.parts[0]["width_built"] == 10 and "across" not in by_hand.parts[0]
    lost = sp.build(p, sp.normalise_plan({**DOLL, "parts": [dict(DOLL["parts"][0], front=[[0, 0, 1, 1]])] + DOLL["parts"][1:]}), sprites=sprites)
    assert any("in the front sprite caught nothing" in n for n in lost.notes)
    # the pictures of the reading show every sprite
    assert sp.reading_picture(p, built).height > sp.parts_picture(p, built.label, built.parts).height
    assert sp.sides_picture(built, sprites, {"front": 0.0}) is not None


def test_wings_are_laid_out_from_their_drawing_from_above():
    """The back sprite is drawn from above: what it shows of the wings is their outline. Wings placed there are that
    outline laid out to each side from the root, their tips raised as high as the side sprite raises them; Seurat does
    not look for an opening for them and their pixels in the profile build nothing on the middle plane."""
    side, back = bird_side(), bird_back()
    sprites = {"side": side, "back": back}
    p = sp.read_profile(side)
    parts = [dict(q) for q in PLAN["parts"]]
    for q in parts:
        if q["type"] == "wings":
            for key in ("out_up", "out_back", "span"):
                q.pop(key, None)
            q["back"] = [[1, 0, 18, 18], [25, 0, 42, 18]]
    plan = sp.normalise_plan({"facing": "right", "eye": [30, 9], "parts": parts})
    assert sp.tune_wings(p, plan, sprites) is plan
    built = sp.build(p, plan, sprites=sprites)
    wings = next(q for q in built.parts if q["name"] == "wings")
    k = built.parts.index(wings) + 1
    assert wings["across"] == "back" and wings.get("opening") is None and wings["flat"] and wings["rise"] > 8
    assert "laid out as the back sprite draws it from above" in sp.report(built)
    view = built.views["back"]
    assert abs(wings["span_built"] - 2 * float(view.outer[k - 1].max())) <= 3
    cx = built.place["cx"]
    mine = np.argwhere(built.pid == k)
    top, bottom = mine[mine[:, 1] >= mine[:, 1].max() - 1], mine[mine[:, 1] <= mine[:, 1].min() + 1]
    assert np.abs(top[:, 0] - cx + 0.5).min() > np.abs(bottom[:, 0] - cx + 0.5).max()          # tips high and far out
    assert not (built.pid[cx - 1: cx + 1] == k).any()                                          # nothing of them on the middle plane
    assert (built.pid == k).sum() == (built.pid[::-1] == k).sum()                              # one to each side


def test_a_crest_drawn_wide_is_two_crossed_blades():
    """A fin stays thin along the body; when the front sprite draws it wide it gets a second thin blade across, as wide
    as that sprite says, instead of becoming a ball."""
    side = Image.new("RGBA", (16, 24), (0, 0, 0, 0))
    d = ImageDraw.Draw(side)
    d.rectangle([4, 0, 11, 8], fill=GOLD + (255,), outline=INK + (255,))
    d.ellipse([0, 8, 15, 23], fill=RED + (255,), outline=INK + (255,))
    front = Image.new("RGBA", (20, 24), (0, 0, 0, 0))
    d = ImageDraw.Draw(front)
    d.rectangle([2, 0, 17, 8], fill=GOLD + (255,), outline=INK + (255,))
    d.ellipse([2, 8, 17, 23], fill=RED + (255,), outline=INK + (255,))
    sprites = {"side": np.asarray(side).copy(), "front": np.asarray(front).copy()}
    plan = sp.normalise_plan({"parts": [{"name": "head", "type": "solid", "where": [[0, 9, 15, 23]], "front": [[2, 9, 17, 23]]},
                                        {"name": "crest", "type": "fin", "where": [[4, 0, 11, 7]], "front": [[2, 0, 17, 7]]}]})
    built = sp.build(sp.read_profile(sprites["side"]), plan, sprites=sprites)
    mine = built.pid == 2
    across = mine.sum(axis=0)                                # voxels across at each (height, place along the body)
    assert built.parts[1]["across"] == "front" and across.max() >= 14
    assert np.median(across[across > 0]) <= 2                # thin almost everywhere: a slab, not a block
    wide_at = np.nonzero((across >= 10).any(axis=0))[0]
    assert 1 <= len(wide_at) <= 3                            # the blade across is thin along the body too


def test_a_model_has_no_drawn_lines_but_keeps_small_dark_features():
    side = Image.new("RGBA", (20, 14), (0, 0, 0, 0))
    d = ImageDraw.Draw(side)
    d.ellipse([0, 0, 19, 13], fill=RED + (255,), outline=INK + (255,))
    d.line([(4, 7), (15, 7)], fill=INK + (255,))                                   # a long line of the drawing
    d.point((14, 4), fill=INK + (255,))                                            # a pupil
    p = sp.read_profile(np.asarray(side).copy())
    plan = {"parts": [{"name": "head", "type": "solid", "where": [[0, 0, 19, 13]]}]}
    flank = sp.project(sp.build(p, sp.normalise_plan(plan)).grid, sp.palette_rgb(sp.build(p, sp.normalise_plan(plan)).palette), "side", 0.0)
    colours = {tuple(int(v) for v in px[:3]) for px in flank[flank[..., 3] > 0]}
    assert colours == {RED, INK} and (np.all(flank[..., :3] == INK, axis=2) & (flank[..., 3] > 0)).sum() == 1
    soft = sp.build(p, sp.normalise_plan({**plan, "ink": "soften"}))
    assert len(soft.palette) > len(sp.build(p, sp.normalise_plan(plan)).palette)   # the old look: lines as darker tones


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
        assert "INDEPENDENT CHECK" in third and "'legs' is really the tail feathers" in third and "a wing" not in third
        # the first plan that builds is checked, and so is the plan that answers the faults found; no more
        assert len(link.checks) == 2 and len(link.checks[0]["images"]) == 2      # the sprites and the parts picture, nothing else
        check_text = link.checks[0]["messages"][-1]["content"]
        check_text = check_text[0]["text"] if isinstance(check_text, list) else check_text
        assert "- wings (wings): columns" in check_text and "faces right" in check_text and len(link.checks[0]["messages"]) == 2
        fifth = link.calls[4]["messages"][-1]["content"]
        fifth = fifth[0]["text"] if isinstance(fifth, list) else fifth
        assert "found the reading right" in fifth
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
        # carrying on a model built by parts: what the user asks now goes first, and the model can go back a step
        link.answers = [{"action": "adjust", "notes": "wider", "set": {"body": {"width": 18}}},
                        {"action": "revert", "notes": "it was better before", "step": 0},
                        {"action": "revert", "step": 7},
                        {"action": "done", "notes": "as it was"}]
        link.calls = []
        job2 = c.post("/api/generate", json={"continue_from": st["model_id"], "prompt": "make the body wider"}).json()
        st2 = c.get(f"/api/jobs/{job2['job_id']}").json()
        assert st2["status"] == "done" and st2["model_id"] == st["model_id"], st2
        texts = [m["messages"][-1]["content"] for m in link.calls]
        texts = [t[0]["text"] if isinstance(t, list) else t for t in texts]
        assert "ALREADY BUILT" in texts[0] and "make the body wider" in texts[0] and "SIDE sprite" in texts[0]
        assert len(link.calls[0]["images"]) == 4                              # the sprites, and the model as it is: parts, comparison, lit
        assert "width 18" in texts[1] and "revert to: 0" in texts[1]
        assert "Step 2." in texts[2] and "width 18" not in texts[2]
        assert "no built step 7" in texts[3]
        assert len(link.checks) == 2                                          # a model that carries on is not read again
        again = c.get(f"/api/models/{st['model_id']}").json()
        assert again["source"]["plan"]["parts"][0]["width"] == src["plan"]["parts"][0]["width"] and again["count"] == model["count"]
        assert len(c.get(f"/api/models/{st['model_id']}/versions").json()["versions"]) >= 1
        link.answers = None
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
