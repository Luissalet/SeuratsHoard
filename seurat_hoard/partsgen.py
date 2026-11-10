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
  tip, read from the front or back sprite. "out_up" and "out_back": how the sheet leaves the body. From the side the
  sheet always shows its whole face. out_back turns that face to the front and back cameras too (the further back a
  pixel is in the profile, the further out it sits): without it the wings are a thin line seen from the front or from
  behind, which is almost never right. out_up raises them in a V (the higher a pixel, the further out). Leave both
  out to get out_up 0.5, out_back 1, good for most wings; give more out_up for a steep V, more out_back for wings
  held flatter. "thickness": 2 or 3.
- "decal": a marking that is only on the flanks: an eye, a cheek spot. It is exactly the pixels of its shapes.
- "skip": NOT BUILT. Side sprites often let the FAR wing, arm, leg or ear peek out behind the near one, shifted, only so
  the drawing reads well. That is a stylistic choice of the sprite, not anatomy: from the front and from behind both
  wings sit at the same height, and in 3D the mirrored near one already IS the far one. Give the far copy its own part
  of type skip (whatever of the body lies under it is completed). Never build both copies as one big wing. How to spot
  it: a second shape with the near wing's colours right beside it, usually behind and a little higher, that leaves the
  body and ends in a point with nothing on it. It is NOT a neck: a neck is what joins the head to the body.
  Seurat also looks for it inside every wings and pair part (a second area of the same colours behind a line of its
  own) and leaves it out by itself; the report says so, and "far": "keep" on that part builds it after all.

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
{"action": "adjust", "notes": "...", "set": {"wings": {"span": 44, "out_up": 0.8}, "body": {"width": 14}},
 "add": [{...a whole part...}], "remove": ["name"]}        changes the last plan and rebuilds; give "where" again to
                                                            reshape a part; a part added with an existing name replaces it;
                                                            "eye": [column, row] beside "set" moves the eye
{"action": "look", "view": "front_right" | "front_left" | "back_left" | "back_right" | "front" | "back" | "side" | "top"}
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
neighbour's pixels, a tail fan built as a ball. Do NOT tune out_up / out_back to shave red or blue pixels off the wings: those sprites hold
the wings in another pose than the side sprite, so wing silhouettes never match; change them only when the wings look
wrong in the lit views (a thin line from the front, lying flat, pointing forwards). Change a few things at a time with adjust and finish with done when it reads right."""

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

AFTER = """Built: {count} voxels in a grid of {size}.
{report}
Front and back sprites against the model from their cameras: {match}.
Pictures: (1) the side sprite and its pixels by part, (2) each sprite, the model from its camera and the silhouette
difference, (3) the model lit from {views}.
Be a harsh critic: first the parts picture (did every part catch its pixels?), then the structure and widths against
the front and back sprites, then how it reads in 3D. List what is wrong, then answer the next action: adjust (a few
changes), build (a whole new plan), look, or done if it reads right from every side."""


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
    built = sp.build(profile, plan)
    if max(built.grid.shape) > MAX_SIDE:
        raise sp.PlanError(f"the model would be {built.grid.shape} voxels: reduce spans and widths")
    fam = sp.families_of(*[sprites.get(r) for r in ("side", "front", "back")])
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
def wants(gen: Any, job: Any) -> bool:
    """Whether a generation job is one for this method: asked for (``method: parts``), or a sprite target with a side
    sprite and no method given while the setting ``sprite_method`` is parts (the default)."""
    p = job.params
    method = p.get("method")
    if method == "parts":
        return True
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
        history.append({"role": "assistant", "content": json.dumps({k: data.get(k) for k in ("action", "notes", "plan", "set", "add", "remove", "eye", "view")
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
        else:
            user_text, user_images = ("Your answer had no valid action. Answer ONE JSON object: build (with a complete plan), adjust "
                                      "(with set / add / remove), look (with a view) or done."), []
            continue
        problem = apply(plan, step, notes)
        if problem:
            job.say(f"{label}: el plan no se pudo construir ({problem[:200]})")
            job.rounds.append({"round": step, "notes": notes, "applied": 0, "errors": [problem[:300]], "count": model.count,
                               "size": model.size, "render": ""})
            user_text, user_images = (f"The plan could NOT be built: {problem}\nFix it and answer again"
                                      + (" (the model shown before is unchanged)." if state["built"] is not None else "."), [])
            continue
        built = state["built"]
        match = ", ".join(f"{r}: silhouettes overlap {v['iou']:.2f}, {v['missing']:.0%} of the sprite missing, {v['extra']:.0%} extra"
                          for r, v in state["views"].items()) or "no front or back sprite to compare with"
        user_text = AFTER.format(count=model.count, size=model.size, report=sp.report(built), match=match,
                                 views="front right, back left, front, back, side and top")
        if not (state["plan"] or {}).get("eye"):
            user_text += ('\nThe plan has no "eye": add "eye": [column, row] (where the eye is in the side sprite) with your next build so '
                          "Seurat can check the head.")
        user_text += "\nThe plan as built:\n" + json.dumps(_plan_text(state["plan"]), ensure_ascii=False)
        user_images = [b for b in state["pics"] if b]
    return llm_name


def _plan_text(plan: dict[str, Any]) -> dict[str, Any]:
    """The plan without the values left to their defaults, to show it back compactly."""
    out = {"facing": plan.get("facing", "right"), "parts": []}
    for part in plan["parts"]:
        d = {k: v for k, v in part.items() if v is not None and not (k in sp.DEFAULTS.get(part["type"], {}) and v == sp.DEFAULTS[part["type"]][k])}
        out["parts"].append(d)
    return out
