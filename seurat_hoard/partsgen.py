"""Characters from game sprites, part by part: the language model decides what each part of the side sprite is (body,
wings, legs, a tail fan, the far wing that is only drawn for looks...) and Seurat does the geometry (``spriteparts``).

The generation is a loop of actions like the agent method, but the model never writes voxels or code: it answers a
PLAN (parts with a type, rough shapes and a few sizes read off the front and back sprites), sees what its shapes caught,
how the model compares with the front and back sprites from their cameras and how it looks lit from several sides, and
adjusts. The same plan can be built without a model (``build_model``): written by hand, by an assistant over MCP, or the
automatic one-solid plan.
"""

from __future__ import annotations

import io
import json
import logging
import time
from typing import Any, Optional

import numpy as np
from PIL import Image

from . import spriteparts as sp
from .errors import SeuratError
from .llm import parse_json
from .voxels import MAX_SIDE, VoxelModel

log = logging.getLogger("seurat.parts")

VIEWS = {"front_right": (45.0, 25.0), "front_left": (-45.0, 25.0), "back_left": (-135.0, 25.0), "back_right": (135.0, 25.0),
         "front": (0.0, 10.0), "back": (180.0, 10.0), "side": (-90.0, 5.0), "top": (0.0, 80.0)}

SYSTEM = """You are a voxel character artist working inside Seurat's Hoard, a voxel studio. You rebuild a character from its
small game sprites as a 3D voxel model, the way voxel mods rebuild handheld-game overworld sprites: one voxel per sprite
pixel, the sprite's own design, colours and proportions.

HOW IT IS BUILT. You do not place voxels and you do not write code. The SIDE sprite is a true profile of the character
(drawn with no camera angle), so Seurat turns every pixel of it into a column of voxels across the body: seen from the
side, the model IS the side sprite. What a profile cannot say is how each part is built ACROSS the body (left to right).
That is your job: cut the side sprite into PARTS and say what each part is. Seurat does the geometry, keeps the sprite's
colours, tones its ink lines down and mirrors everything left-right.

READING THE SIDE SPRITE. Find the head before anything else; a wrong head ruins everything after it.
- The head is the shape that carries the EYE (and the beak, mouth or snout), at the end the character faces. A shape
  with no eye in it is not the head, however high it stands: it is a wing, a crest or the far wing.
- Colours agree between sprites. What the front sprite has at its top centre is the head: the shape of the side sprite
  with those colours and markings is the head. Do the same for the tail (bottom of the back sprite), the belly, the wings.
- The tail is at the end opposite to the head; wings and arms rise from the back or shoulders, between the two; legs
  hang under the body.
The plan says where the eye is: "eye": [column, row] of the side sprite. Seurat checks that it lies in a part that can
be a head, on the side the plan says the character faces.

PART TYPES
- "solid": on the middle line of the body, puffed up round: body, head, neck, snout, beak, a thick tail, a bushy crest.
  "width": how wide it is across at its widest, in voxels (= pixels of the front and back sprites). Without width it is
  as wide as it is thick in profile. Its flanks take the side sprite; when the front or back sprite shows the same
  pose as the model, Seurat paints what faces that camera (a face, the back of the hair) from it by itself.
  "paint": "none" keeps the side sprite's colours all around, "front", "back" or "both" forces the painting.
- "fin": a thin slab on the middle line: a crest or plume of feathers, a sail, a dorsal fin, a blade, a comb.
  "thickness": 2 to 6. A crest is a fin (or a solid with a small width), never a ball.
- "fan": on the middle line, narrow at its root and widening away from it, with a flat end: a tail fan, a skirt, a mane.
  "base": its width where it starts, "width": its width at the far end. Seurat paints its end from the back sprite.
- "pair": something the character has twice, one on each side, drawn once in the profile: legs, feet, arms, ears, horns,
  side fins. "offset": how far from the middle line the middle of each one is; "width": how thick each one is across.
- "wings": two flat sheets that start at the body and open outwards: wings, big ears, the halves of a cape.
  "root": [column, row] of the side sprite where the wing joins the body (the shoulder). "span": total width from tip to
  tip, read from the front or back sprite. "thickness": 2 or 3. How the sheet opens (straight up in a thin V, or back
  with its face to the front and back cameras, or between) is NOT yours to give: Seurat builds every opening and keeps
  the one whose front and back views match those sprites best. A flat sheet cannot do more than that, so do not spend
  actions on the wings' angle: get their pixels, their root and their span right.
- "decal": a marking that is only on the flanks: an eye, a cheek spot. It is exactly the pixels of its shapes.
- "skip": NOT BUILT. Side sprites often let the FAR wing, arm, leg or ear peek out behind the near one, shifted, only so
  the drawing reads well. That is a stylistic choice of the sprite, not anatomy: from the front and from behind both
  wings sit at the same height, and in 3D the mirrored near one already IS the far one. Give the far copy its own part
  of type skip (whatever of the body lies under it is completed). Never build both copies as one big wing. How to spot
  it: a second shape with the near wing's colours right beside it, usually behind and a little higher, that leaves the
  body and ends in a point with nothing on it. It is NOT a neck: a neck is what joins the head to the body.
  Seurat also looks for it inside every wings and pair part (a second area of the same colours behind a line of its
  own) and leaves it out by itself; the report says so. Do not bring it back to gain overlap with the front or back
  sprite: seen from there the far wing IS the mirrored near one.

WHERE A PART IS. "where": a list of shapes in SIDE-sprite pixels, numbered as in the pixel map (column from the left,
row from the top): a rectangle [col0, row0, col1, row1] (inclusive) or a polygon {"poly": [[col, row], ...]}. Add
"colors" (letters of the pixel map, e.g. "bc") to a part, or to one shape as {"rect": [...], "colors": "bc"}, to take
only those colours inside it: a gold tail next to a red wing is easy that way. Shapes can be rough: Seurat snaps them
to the sprite's patches of colour and its ink lines (a patch that is one part's but for a sliver goes to it whole), and
pixels nobody takes join the neighbouring part. Where shapes overlap at their borders each patch stays with the part
that has the rest of it; a shape drawn inside another one (a wing over a body given as one big rectangle) wins there;
in a patch two parts share, a polygon or a shape with "colors" is taken at its word and two bare rectangles meet halfway.

ACTIONS. Answer ONE JSON object each time:
{"action": "build", "notes": "what you decided", "plan": {"facing": "right", "eye": [30, 9], "parts": [{"name": "body", "type": "solid",
 "where": [[10, 12, 24, 30]], "width": 12}, ...]}}       ("facing": where the head points in the side sprite)
{"action": "adjust", "notes": "...", "set": {"wings": {"span": 44, "root": [18, 15]}, "body": {"width": 14}},
 "add": [{...a whole part...}], "remove": ["name"]}        changes the last plan and rebuilds; give "where" again to
                                                            reshape a part; a part added with an existing name replaces it;
                                                            "eye": [column, row] beside "set" moves the eye
{"action": "look", "view": "front_right" | "front_left" | "back_left" | "back_right" | "front" | "back" | "side" | "top"}
{"action": "revert", "notes": "...", "step": 2}            goes back to the plan of an earlier step that was better
{"action": "done", "notes": "why it is finished"}

AFTER EVERY BUILD you get three pictures:
1. The side sprite next to the same pixels coloured by part, with the parts' names. Check that every part caught exactly
   its pixels: a wing that swallowed part of the body, a tail cut short, an eye that missed. Fix shapes first.
2. For the front and the back sprite: the sprite, the model seen from that sprite's camera, and where the silhouettes
   differ (red: in the sprite, missing in the model; blue: in the model, not in the sprite).
3. The model lit from several sides.

HOW TO JUDGE IT. The profile is exact by construction. From the front and from behind the model must have the sprite's
structure and widths: the same things in the same places, as wide as the sprite draws them. Those two sprites are drawn
by hand, each in its own pose (wings a little higher or lower, a tail hanging differently) and from a camera above, so
do not chase a pixel-perfect overlap: chase the right structure and the right sizes, and a figure that reads as the
character from every side in the lit views. Usual faults: wings too short or too long (span), opening the wrong way
(out_up / out_back), not touching the body (root), a body too thin or too fat (width), legs fused in the middle
(offset), the far wing built as if it belonged to the near one or as a neck or a slab on the middle line (skip it:
between the wings the back sprite has only the head and what is on it), a crest built as a ball, a part that took its
neighbour's pixels, a tail fan built as a ball. The wings' silhouettes never match those two sprites exactly: they hold the wings in another
pose than the side sprite, and the opening is already the best a flat sheet gives. Leave it. Change a few things at a time with adjust and finish with done when it reads right."""

FIRST = """Rebuild this character in voxels from its sprites.
{prompt}
The FIRST picture shows the sprites enlarged with rulers (columns across, rows down){extra}.

COLOURS (letter in the pixel maps = colour; '#' = ink: outline and inner lines; '.' = empty):
{palette}

SIDE sprite, {sw}x{sh} pixels. This is the profile you cut into parts; columns and rows are those of your shapes:
{side_map}

{others}
WHAT THE OTHER SPRITES SAY ABOUT WIDTHS (use them for width, span, offset):
{measures}

You have up to {steps} actions and about {minutes} minutes. Think the anatomy through first: where the eye is and so
which shape is the head; which pixels are the body, each limb; which limb is the far copy drawn for looks; how wide each part is in the front and back sprites;
how the wings leave the body. Then answer a first complete plan with action build."""

CHECK = """You check how a game sprite was read before it is built in 3D. Someone cut the SIDE sprite of a character into
named parts. Your pictures: (1) the character's sprites enlarged with rulers (side, front, back); (2) the side sprite
and, beside it, the same pixels coloured by part with each part's name{extra}.
Go part by part and say whether its name fits what those pixels really are.
- Start with the head: it is the shape that carries the eye (and the beak, mouth or snout), at the end the character
  faces. Colours agree between sprites: what the front sprite has at its top centre is the head.
- Then the tail (the opposite end), the wings or arms (they rise from the back or the shoulders), the legs (what the
  character stands on) and the markings of the body (a belly patch is not a foot, a crest is not a wing).
- A second wing or limb with the near one's colours peeking out behind it is the FAR copy, drawn for looks. It must be
  left out (a part of type skip, or the grey part Seurat labels "(far)"): never a neck, a head or a piece of the near
  wing. And a part of type skip must really be such a far copy, not a marking of the body.
The list you get gives, for every part, the columns and rows it covers in the side sprite (read them on the rulers:
do not go by where a label happens to be drawn) and its main colours.
Answer ONE JSON object:
{{"head": "name of the part that really is the head", "wrong": [{{"part": "its name", "really": "what those pixels are",
"why": "in a few words", "sure": true}}], "missing": ["something the character has that no part covers"],
"verdict": "ok" or "redo"}}
"redo" only when the reading is wrong at its root (the head, where it faces, a wing taken for the body). "sure": false
when you only suspect it. A part may take a few pixels of its neighbour or be cut a little short: that is not a fault,
and neither is a missing part for something small. When the reading is right answer empty lists and "ok": do not
invent faults, and do not judge sizes or the 3D, only what each part is."""

CONTINUE = """

THIS CHARACTER IS ALREADY BUILT: you are continuing it, not starting over. Its report and plan are below, and the
pictures after the sprites show it (its pixels by part; each sprite against the model; the model lit).
{request}
Keep what is right. Answer with adjust (build only if the plan is wrong at its root), or done if nothing has to change.
Built: {count} voxels in a grid of {size}.
{report}
Front and back sprites against the model from their cameras: {match}.
The plan as built:
{plan}"""

AFTER = """Step {step}. Built: {count} voxels in a grid of {size}.
{report}
Front and back sprites against the model from their cameras: {match}.
Pictures: (1) the side sprite and its pixels by part, (2) each sprite, the model from its camera and the silhouette
difference, (3) the model lit from {views}.
Be a harsh critic: first the parts picture (did every part catch its pixels?), then the structure and widths against
the front and back sprites, then how it reads in 3D. List what is wrong, then answer the next action: adjust (a few
changes), build (a whole new plan), look, revert (if an earlier step was better than this one), or done if it reads
right from every side."""


def png(img: Image.Image) -> bytes:
    buf = io.BytesIO()
    img.save(buf, "PNG")
    return buf.getvalue()


def lit_sheet(model: VoxelModel, names: tuple[str, ...] = ("front_right", "back_left", "front", "back", "side", "top"),
              target: int = 300) -> Image.Image:
    """The model lit from several sides in one labelled picture (flat isometric views if the lit renderer is missing)."""
    from PIL import ImageDraw
    tiles = []
    try:
        from . import lit
        scene = lit.LitScene(model, ground=True)
        for name in names:
            yaw, pitch = VIEWS[name]
            u0, u1, v0, v1 = scene.extent(yaw, pitch)
            scale = target / max(1e-6, 1.12 * max(u1 - u0, v1 - v0))
            tiles.append((name.replace("_", " "), scene.render(yaw, pitch, scale, supersample=2 if scale < 12 else 1)))
    except Exception:  # noqa: BLE001 - the software renderer always works
        from .render import BG_DARK, iso, ortho
        s = max(2, int(target / (max(model.size) * 1.9)))
        tiles = [("front right", iso(model, "iso", s, BG_DARK)), ("back left", iso(model, "iso_back_left", s, BG_DARK)),
                 ("front", ortho(model, "front", max(1, target // max(model.size)), BG_DARK)),
                 ("back", ortho(model, "back", max(1, target // max(model.size)), BG_DARK))]
    font = sp._font(13)
    cols = 3 if len(tiles) > 4 else 2
    cw = max(t.width for _, t in tiles) + 8
    ch = max(t.height for _, t in tiles) + 26
    rows = -(-len(tiles) // cols)
    out = Image.new("RGBA", (cw * cols, ch * rows), (12, 10, 14, 255))
    d = ImageDraw.Draw(out)
    for i, (name, t) in enumerate(tiles):
        x, y = (i % cols) * cw + 4, (i // cols) * ch + 4
        d.text((x, y), name, fill=(235, 225, 200, 255), font=font)
        out.paste(t, (x, y + 18))
    return out


def lit_view(model: VoxelModel, name: str, target: int = 640) -> Image.Image:
    yaw, pitch = VIEWS.get(name, VIEWS["front_right"])
    try:
        from . import lit
        return lit.lit(model, yaw, pitch, target)
    except Exception:  # noqa: BLE001
        from .render import camera
        return camera(model, yaw, pitch, max(2.0, target / (max(model.size) * 1.5)), background=(26, 22, 30, 255))


# ---------------------------------------------------------------------------------------------- the sprites of a request
def read_sprites(svc: Any, entries: list[dict[str, Any]]) -> tuple[dict[str, np.ndarray], list[dict[str, Any]]]:
    """({role: RGBA sprite} for side/front/back, [the other pictures as {name, png}]) from reference entries
    ({id, role}). Sheets and animations give their standing frame of each side when the service can read them."""
    sprites: dict[str, np.ndarray] = {}
    others: list[dict[str, Any]] = []
    expanded = None
    if any((e.get("role") == "sheet" or e.get("frames")) for e in entries) and hasattr(svc, "expand_references"):
        expanded = svc.expand_references(entries)
    if expanded is not None:
        for e in expanded:
            if e["role"] in ("side", "front", "back") and e["role"] not in sprites:
                sprites[e["role"]] = np.asarray(e["rgba"]).copy()
            elif e["role"] not in ("side", "front", "back"):
                others.append({"name": e.get("name") or e["role"], "png": e["png"]})
        return sprites, others
    for e in entries:
        role = str(e.get("role") or "front")
        raw = svc.store.reference_bytes(e["id"])
        name = svc.store.reference(e["id"]).get("name") or role
        if role in ("side", "front", "back") and role not in sprites:
            sprites[role] = sp.cutout(raw)
        else:
            others.append({"name": name, "png": raw})
    return sprites, others


def construct(profile: sp.Profile, plan: dict[str, Any], sprites: dict[str, np.ndarray],
              camera: Optional[dict[str, float]] = None) -> tuple[sp.Built, dict[str, float], dict[str, Any]]:
    """Plan to voxels: build, find the cameras of the front and back sprites, paint the fans' ends. Returns the model
    as built, the cameras and how each view matches."""
    fam = sp.families_of(*[sprites.get(r) for r in ("side", "front", "back")])
    built = sp.build(profile, sp.tune_wings(profile, plan, sprites, fam))
    if max(built.grid.shape) > MAX_SIDE:
        raise sp.PlanError(f"the model would be {built.grid.shape} voxels: reduce spans and widths")
    depths = sp.tune_camera(built, sprites, fam, camera)
    sp.paint_caps(built, sprites, depths, fam)
    views = sp.evaluate(built, sprites, depths, fam=fam)["views"]
    return built, depths, views


def _profile_for(side: np.ndarray, plan: dict[str, Any]) -> sp.Profile:
    facing = "left" if str(plan.get("facing") or "right").lower().startswith("l") else "right"
    return sp.read_profile(side, facing)


def build_model(svc: Any, entries: list[dict[str, Any]], plan: Optional[dict[str, Any]] = None, *, name: str = "",
                into: str = "", prompt: str = "") -> dict[str, Any]:
    """A voxel model from sprites and a plan, with no language model (``plan`` None: the automatic one-solid plan).
    Saves it (a new model, or a new version of ``into``) and returns its summary with pictures of the result."""
    sprites, _ = read_sprites(svc, entries)
    if "side" not in sprites:
        raise SeuratError("invalid", "Building by parts needs the side sprite (role side): it is the profile.",
                          "Pass the reference of the side view with role side; front and back are optional.")
    try:
        if plan is None:
            plan = sp.auto_plan(sp.read_profile(sprites["side"]))
        plan = sp.normalise_plan(plan)
        profile = _profile_for(sprites["side"], plan)
        built, depths, views = construct(profile, plan, sprites)
    except sp.PlanError as exc:
        raise SeuratError("invalid", f"The plan cannot be built: {exc}") from None
    if into:
        model = svc.store.get(into)
    else:
        model = VoxelModel.empty(svc.store.new_id(), name or "Sprite por partes", list(built.grid.shape), built.palette)
    model.grid, model.palette = built.grid, list(built.palette)
    if name:
        model.name = name
    model.source = {"kind": "generated", "method": "parts", "plan": plan, "camera": depths, "prompt": prompt,
                    "references": [e["id"] for e in entries], "entries": [{"id": e["id"], "role": e.get("role") or "front"} for e in entries]}
    saved = svc.store.save(model, version_note="parts plan") if into else svc.store.save(model)
    match_img, _ = sp.match_picture(built, sprites, depths)
    return {"id": saved.id, "name": saved.name, "size": saved.size, "count": saved.count, "plan": plan, "camera": depths,
            "report": sp.report(built), "views": {r: {k: (round(v, 3) if isinstance(v, float) else v) for k, v in res.items()}
                                                   for r, res in views.items()},
            "pictures": {"parts": png(sp.parts_picture(profile, built.label, built.parts)),
                         "match": png(match_img) if match_img is not None else b"", "lit": png(lit_sheet(saved))}}


def adjust_model(svc: Any, model_id: str, set_: Optional[dict] = None, add: Optional[list] = None,
                 remove: Optional[list] = None, plan: Optional[dict] = None) -> dict[str, Any]:
    """Change the plan a model was built from (some values of some parts, parts added or removed, or a whole new plan)
    and rebuild it in place; the previous voxels stay as a version."""
    model = svc.store.get(model_id)
    src = model.source or {}
    if not src.get("plan") or not src.get("entries"):
        raise SeuratError("invalid", "This model was not built by parts: it has no plan to adjust.",
                          "Build it with model_from_parts or a generation with method parts first.")
    try:
        new_plan = plan if plan else sp.apply_changes(src["plan"], set_, add, remove)
    except sp.PlanError as exc:
        raise SeuratError("invalid", str(exc)) from None
    return build_model(svc, src["entries"], new_plan, into=model_id, prompt=src.get("prompt") or "")


def read_profile_info(svc: Any, entries: list[dict[str, Any]], facing: str = "right") -> dict[str, Any]:
    """What a plan is written from: the side sprite's palette letters, its pixel map, the maps of the other sprites,
    what they say about widths, and a picture of the sprites with rulers."""
    sprites, _ = read_sprites(svc, entries)
    if "side" not in sprites:
        raise SeuratError("invalid", "The side sprite (role side) is needed: it is the profile.")
    profile = sp.read_profile(sprites["side"], facing)
    palette, inks = _union_palette(profile, sprites)
    return {"palette": {sp.KEYS[i]: c for i, c in enumerate(palette[: len(sp.KEYS)])},
            "side": {"width": profile.shape[1], "height": profile.shape[0], "map": sp.text_map(sprites["side"], palette, inks)},
            "others": {r: {"width": sprites[r].shape[1], "height": sprites[r].shape[0], "map": sp.text_map(sprites[r], palette, inks),
                           "widths": sp.measure(sprites[r], r)} for r in ("front", "back") if r in sprites},
            "types": list(sp.TYPES), "picture": png(sp.sprites_picture(sprites))}


def _union_palette(profile: sp.Profile, sprites: dict[str, np.ndarray]) -> tuple[list[str], set]:
    palette = list(profile.palette)
    inks: set = set()
    for role in ("side", "front", "back"):
        t = sprites.get(role)
        if t is None:
            continue
        inks |= {tuple(c) for c in sp.ink_colours(t)}
        for c in np.unique(t[t[..., 3] >= 128][:, :3], axis=0):
            hexc = sp._hex(c)
            if hexc not in palette:
                palette.append(hexc)
    # ink has no letter: keep the letters for the colours that fill
    fills = [c for c in palette if sp._rgb(c) not in inks]
    return fills, inks


# ---------------------------------------------------------------------------------------------- the generation
def _colour_word(rgb: tuple[int, int, int]) -> str:
    import colorsys
    h, sat, v = colorsys.rgb_to_hsv(rgb[0] / 255, rgb[1] / 255, rgb[2] / 255)
    if v < 0.22:
        return "black"
    if sat < 0.16:
        return "white" if v > 0.82 else "grey"
    for limit, word in ((18, "red"), (42, "orange"), (68, "yellow"), (165, "green"), (200, "cyan"), (262, "blue"), (300, "purple"), (340, "pink")):
        if h * 360 < limit:
            return ("dark " if v < 0.5 else "") + word
    return ("dark " if v < 0.5 else "") + "red"


def second_opinion(client: Any, pictures: list[bytes], plan: dict[str, Any], built: sp.Built, profile: sp.Profile, *, prompt: str = "", artwork: bool = False,
                   effort: str = "max", thinking: int = 8000, timeout: float = 1800, cancel: Any = None) -> tuple[dict[str, Any], str]:
    """A fresh look at the reading of the sprite: the model, without the reasoning that made the plan, checks the parts
    picture name by name. Returns (its answer, what to tell the author of the plan; '' when it could not be read)."""
    parts = []
    for k, part in enumerate(built.parts):
        if not part.get("pixels") or built.label is None:
            continue
        mine = built.label == k + 1
        rs, cs = np.nonzero(mine)
        if not len(rs):
            continue
        words: dict[str, int] = {}
        for px in profile.rgba[mine & ~profile.ink][:, :3]:
            word = _colour_word((int(px[0]), int(px[1]), int(px[2])))
            words[word] = words.get(word, 0) + 1
        main = [w for w, n in sorted(words.items(), key=lambda kv: -kv[1]) if n >= 0.08 * max(1, sum(words.values()))][:3]
        if profile.flipped:
            cs = profile.shape[1] - 1 - cs
        parts.append(f"- {part['name']} ({part['type']}): columns {cs.min()}-{cs.max()}, rows {rs.min()}-{rs.max()}, {len(rs)} px"
                     + (", " + " and ".join(main) if main else ""))
    text = ((f"The character: {prompt.strip()}\n" if prompt.strip() else "")
            + f"The plan says it faces {plan.get('facing', 'right')}" + (f" and that its eye is at column {plan['eye'][0]:.0f}, row {plan['eye'][1]:.0f}"
                                                                         if plan.get("eye") else "") + ".\nThe parts:\n" + "\n".join(parts))
    try:
        answer = client.chat(CHECK.format(extra="; (3) artwork of the character, to know its anatomy" if artwork else ""), text, pictures,
                             max_tokens=3000, effort=effort, timeout=timeout, cancel=cancel, thinking_tokens=thinking)
        data = parse_json(answer.text)
    except Exception as exc:  # noqa: BLE001 - the check is a help, never a reason to fail
        log.warning("parts check not read: %s: %s", type(exc).__name__, str(exc)[:300])
        return {}, ""
    if not isinstance(data, dict):
        return {}, ""
    computed = {str(part["name"]) for part in built.parts if part.get("auto")}       # what Seurat left out itself is not the author's to fix
    wrong = [w for w in (data.get("wrong") or []) if isinstance(w, dict) and w.get("part") and w.get("sure", True) is not False
             and str(w["part"]) not in computed]
    data["wrong"] = wrong
    missing = [str(m) for m in (data.get("missing") or []) if m]
    if not wrong and not missing and str(data.get("verdict") or "ok").lower() != "redo":
        return data, "\nAn independent check of the parts picture found the reading right (head: %s)." % (data.get("head") or "?")
    lines = ["\nAN INDEPENDENT CHECK of the parts picture, by someone who has not seen your reasoning, says:"]
    if data.get("head"):
        lines.append(f"- the head is the part named '{data['head']}'")
    for w in wrong[:8]:
        lines.append(f"- '{w['part']}' is really {str(w.get('really') or '?')[:120]}" + (f" ({str(w['why'])[:160]})" if w.get("why") else ""))
    for m in missing[:6]:
        lines.append(f"- nothing covers: {m[:160]}")
    if str(data.get("verdict") or "").lower() == "redo":
        lines.append("- its verdict: the reading is wrong at its root, write the plan again")
    lines.append("Look at those pixels again before anything else. Fix what it is right about; if you are sure it is mistaken "
                 "somewhere, say why in your notes.")
    return data, "\n".join(lines)


def _base_model(svc: Any, p: dict[str, Any]) -> Optional[VoxelModel]:
    """The model a generation carries on from, when it was built by parts: ``continue_from``, or the model it writes
    into when the start is the current model."""
    mid = p.get("continue_from") or (p.get("into") if p.get("start") == "current" else "")
    if not mid:
        return None
    try:
        m = svc.store.get(str(mid))
    except Exception:  # noqa: BLE001
        return None
    src = m.source or {}
    return m if src.get("method") == "parts" and isinstance(src.get("plan"), dict) else None


def _drop_keep(plan: dict[str, Any], opening: bool = False) -> list[str]:
    """Takes out of a plan written by the model what is not its to decide: "far": "keep" and the opening of the wings
    (out_up, out_back, curl), which Seurat finds against the sprites, unless ``opening`` (the user asked for a change in
    words, and it may be about the wings). A person can give them in the panel or with the tools. Returns the names of
    the parts that had "far": "keep"."""
    names = []
    for part in plan.get("parts") or []:
        if not isinstance(part, dict):
            continue
        if str(part.get("far") or "").lower() == "keep":
            part.pop("far")
            names.append(str(part.get("name")))
        if not opening:
            for key in ("out_up", "out_back", "curl"):
                part.pop(key, None)
    return names


def wants(gen: Any, job: Any) -> bool:
    """Whether a generation job is one for this method: asked for (``method: parts``), or a sprite target with a side
    sprite and no method given while the setting ``sprite_method`` is parts (the default)."""
    p = job.params
    method = p.get("method")
    if method == "parts":
        return True
    if not method and p.get("use_model") is not False and _base_model(gen.svc, p) is not None:
        return True                              # a model built by parts carries on by parts
    if method or p.get("continue_from") or p.get("start") == "current" or p.get("use_model") is False:
        return False
    try:
        default = gen.svc.settings["sprite_method"]
    except Exception:  # noqa: BLE001 - a settings object without the key
        default = "parts"
    if default != "parts" or (p.get("target") or "auto") == "free":
        return False
    refs = p.get("references") or []
    side = [r for r in refs if (r.get("role") or "front") == "side"]
    sheets = [r for r in refs if r.get("role") == "sheet"]
    if not side and sheets and hasattr(gen.svc, "expand_references"):
        # a sprite sheet that shows the character from the side has its side sprite too
        try:
            return any(e["role"] == "side" for e in gen.svc.expand_references(sheets))
        except Exception:  # noqa: BLE001
            return False
    if not side:
        return False
    if p.get("target") == "sprite":
        return True
    try:
        from .images import is_sprite
        return is_sprite(gen.svc.store.reference_bytes(side[0]["id"]))
    except Exception:  # noqa: BLE001
        return False


def run(gen: Any, job: Any) -> VoxelModel:
    """One generation by parts (called by ``Generator.run``)."""
    svc, p = gen.svc, job.params
    settings = svc.settings

    def setting(key: str, default: Any) -> Any:
        try:
            return settings[key]
        except Exception:  # noqa: BLE001
            return default

    job.say("leyendo los sprites")
    entries = [{**r, "role": r.get("role") or "front"} for r in (p.get("references") or [])]
    start = _base_model(svc, p)
    if start is not None and start.source.get("entries"):
        # carrying on a model: the sprites it was made from, whatever is attached now (unless one of them is gone)
        own = [dict(e) for e in start.source["entries"]]
        try:
            for r in own:
                svc.store.reference(r["id"])
            entries = own
        except Exception:  # noqa: BLE001
            pass
    for r in entries:
        svc.store.reference(r["id"])
    sprites, others = read_sprites(svc, entries)
    if "side" not in sprites:
        raise SeuratError("invalid", "The parts method needs the side sprite (role side): it is the profile.",
                          "Add the side view with role side, or use method agent.")
    name = (p.get("name") or "").strip() or ((svc.store.reference(next(e["id"] for e in entries if e["role"] == "side")).get("name") or "Sprite")
                                             .replace(" side", "").replace("(sprite)", "").strip() + " (voxel)")
    into = p.get("into") or ""
    use_model = p.get("use_model", True)
    base = _base_model(svc, p)
    if base is not None and not into:
        into = base.id                          # carrying on a model: its new versions are its own
    profile = sp.read_profile(sprites["side"])
    palette, inks = _union_palette(profile, sprites)
    model = svc.store.get(into) if into else VoxelModel.empty(svc.store.new_id(), name, [8, 8, 8], list(profile.palette))
    state: dict[str, Any] = {"plan": None, "built": None, "camera": None, "views": {}, "profile": profile}

    def record(rnd: int, notes: str, errors: list[str]) -> None:
        built = state["built"]
        entry: dict[str, Any] = {"round": rnd, "notes": notes, "applied": 1 if built is not None and not errors else 0,
                                 "errors": errors[:12], "count": model.count, "size": model.size, "render": ""}
        if built is not None and not errors:
            rd = svc.config.renders_dir
            rd.mkdir(parents=True, exist_ok=True)
            (rd / f"{job.id}-r{rnd}.png").write_bytes(state["pics"][2])
            entry["render"] = f"{job.id}-r{rnd}.png"
            if state["pics"][1]:
                (rd / f"{job.id}-r{rnd}-sprites.png").write_bytes(state["pics"][1])
                entry["sprite_render"] = f"{job.id}-r{rnd}-sprites.png"
            (rd / f"{job.id}-r{rnd}-parts.png").write_bytes(state["pics"][0])
            entry["parts_render"] = f"{job.id}-r{rnd}-parts.png"
            ious = [v["iou"] for v in state["views"].values()]
            if ious:
                entry["match"] = round(float(np.mean(ious)), 3)
                entry["views"] = [{"role": r, "iou": round(v["iou"], 3), "missing": round(v["missing"], 3), "extra": round(v["extra"], 3),
                                   "depth": v.get("depth")} for r, v in state["views"].items()]
                job.match = entry["match"]
            try:
                from .rounds import save_snapshot
                entry["snapshot"] = save_snapshot(rd, job.id, rnd, model)
            except Exception:  # noqa: BLE001 - snapshots are optional
                pass
        job.rounds.append(entry)

    def apply(plan: dict[str, Any], rnd: int, notes: str) -> str:
        """Builds ``plan``; on success the model, the pictures and the round are updated. Returns the problem ('' = ok)."""
        try:
            plan = sp.normalise_plan(plan)
            prof = _profile_for(sprites["side"], plan)
            built, depths, views = construct(prof, plan, sprites, state["camera"])
        except sp.PlanError as exc:
            return str(exc)
        except Exception as exc:  # noqa: BLE001 - a plan must never take the generation down
            log.exception("parts build failed")
            return f"{type(exc).__name__}: {exc}"
        if int((built.grid > 0).sum()) == 0:
            return "the plan built nothing: no part caught any pixel"
        state.update(plan=plan, built=built, camera=depths, views=views, profile=prof)
        model.grid, model.palette = built.grid, list(built.palette)
        match_img, _ = sp.match_picture(built, sprites, depths)
        state["pics"] = [png(sp.parts_picture(prof, built.label, built.parts)), png(match_img) if match_img is not None else b"",
                         png(lit_sheet(model))]
        model.source = {"kind": "generated", "method": "parts", "plan": plan, "camera": depths, "job": job.id,
                        "prompt": p.get("prompt") or "", "references": [e["id"] for e in entries],
                        "entries": [{"id": e["id"], "role": e["role"]} for e in entries], "partial": True}
        if not into:
            svc.store.save(model, version_note=f"generation {job.id} step {rnd}")
            job.model_id, job.model_name = model.id, model.name
        record(rnd, notes, [])
        job.say(f"paso {rnd}: {model.count} voxels, {len(built.parts)} partes; "
                + ", ".join(f"{r} {v['iou']:.2f}" for r, v in views.items()))
        return ""

    llm_name = ""
    if base is not None and use_model:
        problem = apply(json.loads(json.dumps(base.source["plan"])), 0, "plan de partida")
        if problem:
            raise SeuratError("invalid", f"The model's plan cannot be built any more: {problem}",
                              "Generate it again from its sprites.")
    if not use_model:
        problem = apply(p.get("plan") or sp.auto_plan(profile), 1, "plan automático" if not p.get("plan") else "plan dado")
        if problem:
            raise SeuratError("invalid", f"The plan cannot be built: {problem}")
    else:
        llm_name = _loop(gen, job, model, sprites, others, profile, palette, inks, state, apply, setting)
    job.llm = llm_name
    if state["built"] is None:
        if job.cancel.is_set():
            job.say("cancelado antes de construir nada: no se guarda ningún modelo")
            return model
        raise SeuratError("failed", "The model never answered a plan that could be built.",
                          "Try again, or build with a hand-written plan (model_from_parts).")
    model.name = name if not into else model.name
    model.source = {**model.source, "llm": llm_name, "rounds": len(job.rounds)}
    model.source.pop("partial", None)
    saved = svc.store.save(model, version_note="before generation " + job.id) if into else svc.store.save(model)
    job.model_id, job.model_name = saved.id, saved.name
    job.say(f"guardado: {saved.name} ({saved.count} voxels)")
    return saved


def _loop(gen: Any, job: Any, model: VoxelModel, sprites: dict[str, np.ndarray], others: list[dict[str, Any]],
          profile: sp.Profile, palette: list[str], inks: set, state: dict[str, Any], apply, setting) -> str:
    svc, p = gen.svc, job.params
    client = svc.llm
    effort = p.get("effort") or setting("effort", "max")
    max_tokens = int(setting("max_tokens", 24000))
    step_thinking = int(setting("agent_step_thinking", 30000))
    timeout = float(setting("timeout_s", 7200))
    max_steps = int(p.get("steps") or setting("parts_steps", 8))
    deadline = time.time() + 60 * float(p.get("minutes") or setting("agent_minutes", 90))
    from .images import to_png
    extra_imgs = [to_png(o["png"], 1024) for o in others[:2]]
    board = png(sp.sprites_picture(sprites))
    letters = ", ".join(f"{sp.KEYS[i]}={c}" for i, c in enumerate(palette[: len(sp.KEYS)]))
    other_maps, measures = [], []
    for role in ("front", "back"):
        t = sprites.get(role)
        if t is None:
            continue
        other_maps.append(f"{role.upper()} sprite, {t.shape[1]}x{t.shape[0]} pixels (for widths and structure only; your shapes are "
                          f"never in these coordinates):\n{sp.text_map(t, palette, inks)}\n")
        measures.append(sp.measure(t, role))
    if not measures:
        measures.append("There is no front or back sprite: choose widths from what you know of the character.")
    h, w = profile.shape
    user_text = FIRST.format(
        prompt=("What the user wants: " + str(p["prompt"]).strip()) if p.get("prompt") else "",
        extra=("; the other picture(s) are artwork of the character, only to understand its anatomy" if extra_imgs else ""),
        palette=letters, sw=w, sh=h, side_map=sp.text_map(sprites["side"], palette, inks),
        others="\n".join(other_maps), measures="\n".join(measures), steps=max_steps,
        minutes=int((deadline - time.time()) / 60))
    user_images = [board] + extra_imgs
    plans: dict[int, dict[str, Any]] = {}
    free_opening = False

    def match_text() -> str:
        return ", ".join(f"{r}: silhouettes overlap {v['iou']:.2f}, {v['missing']:.0%} of the sprite missing, {v['extra']:.0%} extra"
                         for r, v in state["views"].items()) or "no front or back sprite to compare with"

    if state["built"] is not None:
        # carrying on from a model built by parts: its plan is step 0, and what the user asks now comes first
        plans[0] = state["plan"]
        asked = str(p.get("prompt") or "").strip()
        free_opening = bool(asked)
        user_text += CONTINUE.format(
            request=("What the user asks for now, do it first: " + asked + "\nIf it is about how the wings open you may, this time, set "
                     '"out_up" (raises the sheet in a V: the higher a pixel, the further out) and "out_back" (turns its face to the '
                     "front and back cameras: the further back a pixel, the further out) on the wings; 0.5 and 0.9 is a usual opening.")
            if asked else "Look for what is still wrong and fix it.",
            count=model.count, size=model.size, report=sp.report(state["built"]), match=match_text(),
            plan=json.dumps(_plan_text(state["plan"]), ensure_ascii=False))
        user_images = [board] + extra_imgs[:1] + [b for b in state["pics"] if b]
    history: list[dict[str, Any]] = []
    llm_name = ""
    for step in range(1, max_steps + 1):
        if job.cancel.is_set():
            break
        left = deadline - time.time()
        if left <= 0:
            job.say("se acabó el tiempo")
            break
        if (step == max_steps or left < 240) and state["built"] is not None:
            user_text += "\n\nThis is your LAST action: one final adjust or build if something important is still wrong, else done."
        label = f"paso {step} de {max_steps}"

        def progress(n_text: int, n_reason: int, label=label) -> None:
            job.step = f"{label} · " + (f"pensando: {n_reason} tokens" if n_reason and not n_text else
                                        f"escribiendo: {n_text} tokens" + (f" (tras pensar {n_reason})" if n_reason else ""))

        job.say(label + ": el modelo decide")
        answer = client.chat(SYSTEM, user_text, user_images, max_tokens=max_tokens, effort=effort, timeout=timeout,
                             on_progress=progress, cancel=job.cancel,
                             thinking_tokens=step_thinking * (2 if state["built"] is None else 1), history=history)
        llm_name = answer.model or llm_name
        if answer.finish == "cancelled":
            break
        gen._save_answer(job, step, answer.text, suffix="-parts")
        if answer.reasoning:
            gen._save_answer(job, step, answer.reasoning, suffix="-think")
        try:
            data = parse_json(answer.text)
        except ValueError:
            data = {}
        action = str(data.get("action") or ("build" if isinstance(data.get("plan"), dict) else "")).lower()
        notes = str(data.get("notes") or "")[:600]
        history.append({"role": "user", "content": gen._content(user_text, user_images)})
        history.append({"role": "assistant", "content": json.dumps({k: data.get(k) for k in ("action", "notes", "plan", "set", "add", "remove", "eye", "view", "step")
                                                                    if data.get(k) is not None}, ensure_ascii=False)})
        for msg in history[:-2]:                      # old pictures are dropped, the words stay
            if isinstance(msg.get("content"), list):
                msg["content"] = [part for part in msg["content"] if part.get("type") == "text"]
        job.say(f"{label}: {action or 'respuesta sin acción'}" + (f" — {notes[:160]}" if notes else "")
                + f" ({answer.seconds:.0f} s" + (f", pensó {answer.reasoning_tokens}" if answer.reasoning_tokens else "") + ")")
        if action == "done":
            if state["built"] is not None:
                job.say("el modelo da el resultado por terminado")
                break
            user_text, user_images = "There is no model yet: answer a first plan with action build.", [board]
            continue
        if action == "look":
            if state["built"] is None:
                user_text, user_images = "There is no model yet: answer a first plan with action build.", [board]
                continue
            view = str(data.get("view") or "front_right")
            view = view if view in VIEWS else "front_right"
            user_text, user_images = f"The picture is the model lit from {view.replace('_', ' ')}. Next action?", [png(lit_view(model, view))]
            continue
        if action == "adjust":
            if state["plan"] is None:
                user_text, user_images = "There is no plan to adjust yet: answer a complete plan with action build.", [board]
                continue
            try:
                plan = sp.apply_changes(state["plan"], data.get("set") if isinstance(data.get("set"), dict) else None,
                                        data.get("add") if isinstance(data.get("add"), list) else None,
                                        data.get("remove") if isinstance(data.get("remove"), list) else None)
            except sp.PlanError as exc:
                user_text, user_images = f"The adjust was NOT applied: {exc}. Nothing changed. Next action?", []
                continue
            if isinstance(data.get("eye"), list):
                plan = dict(plan, eye=data["eye"])
        elif action == "build" and isinstance(data.get("plan"), dict):
            plan = data["plan"]
        elif action == "revert":
            try:
                back = int(data.get("step"))
            except (TypeError, ValueError):
                back = -1
            if back not in plans:
                user_text, user_images = (f"There is no built step {data.get('step')!r} to go back to; the built steps are "
                                          f"{sorted(plans) or 'none'}. Next action?"), []
                continue
            plan = json.loads(json.dumps(plans[back]))
            notes = notes or f"vuelta al paso {back}"
        else:
            user_text, user_images = ("Your answer had no valid action. Answer ONE JSON object: build (with a complete plan), adjust "
                                      "(with set / add / remove), look (with a view), revert (with a step) or done."), []
            continue
        kept_far = _drop_keep(plan, opening=free_opening)
        problem = apply(plan, step, notes)
        if problem:
            job.say(f"{label}: el plan no se pudo construir ({problem[:200]})")
            job.rounds.append({"round": step, "notes": notes, "applied": 0, "errors": [problem[:300]], "count": model.count,
                               "size": model.size, "render": ""})
            user_text, user_images = (f"The plan could NOT be built: {problem}\nFix it and answer again"
                                      + (" (the model shown before is unchanged)." if state["built"] is not None else "."), [])
            continue
        built = state["built"]
        plans[step] = state["plan"]
        user_text = AFTER.format(step=step, count=model.count, size=model.size, report=sp.report(built), match=match_text(),
                                 views="front right, back left, front, back, side and top")
        if len(plans) > 1:
            user_text += "\nBuilt steps you can revert to: " + ", ".join(str(n) for n in sorted(plans) if n != step) + "."
        if (0 not in plans and setting("parts_check", True)
                and (state.get("checks", 0) == 0 or (state.get("checks") == 1 and state.get("check_faults")))):
            # the first plan that builds gets a second opinion on how it read the sprite, before sizes are tuned; if it
            # found faults, so does the plan that answers them
            state["checks"] = state.get("checks", 0) + 1
            job.say(f"{label}: comprobación independiente de la lectura del sprite")
            opinion, told = second_opinion(client, [board, state["pics"][0]] + extra_imgs[:1], state["plan"], built, state["profile"],
                                           prompt=str(p.get("prompt") or ""), artwork=bool(extra_imgs), effort=effort,
                                           thinking=max(4000, step_thinking // 4), timeout=timeout, cancel=job.cancel)
            if told:
                user_text += told
                faults = (len([w for w in (opinion.get("wrong") or []) if isinstance(w, dict) and w.get("sure", True) is not False])
                          + len(opinion.get("missing") or []) + (1 if str(opinion.get("verdict")).lower() == "redo" else 0))
                state["check_faults"] = faults
                job.say(f"{label}: la comprobación " + ("da la lectura por buena" if not faults
                                                         else f"señala {faults} cosa(s)" + (" y pide rehacer el plan" if str(opinion.get("verdict")).lower() == "redo" else "")))
                if job.rounds:
                    job.rounds[-1]["check"] = {k: opinion.get(k) for k in ("head", "wrong", "missing", "verdict")}
        if kept_far:
            user_text += ('\n"far": "keep" was ignored on ' + ", ".join(kept_far) + ": the far wing or limb a side sprite lets peek out "
                          "is drawn for looks and is not built here. If that area is really something else, give it its own part.")
        if not (state["plan"] or {}).get("eye"):
            user_text += ('\nThe plan has no "eye": add "eye": [column, row] (where the eye is in the side sprite) with your next build so '
                          "Seurat can check the head.")
        user_text += "\nThe plan as built:\n" + json.dumps(_plan_text(state["plan"]), ensure_ascii=False)
        user_images = [b for b in state["pics"] if b]
    return llm_name


def _plan_text(plan: dict[str, Any]) -> dict[str, Any]:
    """The plan without the values left to their defaults, to show it back compactly."""
    out = {"facing": plan.get("facing", "right"), "parts": []}
    if plan.get("eye"):
        out["eye"] = plan["eye"]
    for part in plan["parts"]:
        d = {k: v for k, v in part.items() if v is not None and not (k in sp.DEFAULTS.get(part["type"], {}) and v == sp.DEFAULTS[part["type"]][k])}
        out["parts"].append(d)
    return out
