# Seurat's Hoard — what it can and cannot do

App id `seurat`, service `seurats-hoard`, port 5250, MCP stdio bridge `mcp_server.py` (36 tools; `model_look`,
`model_sprite_match`, `model_sprite_calibrate`, `reference_frames`, `generate_round` and `reference_look` return MCP
images). Connector preset: its own `faustus-plugin.json`.

## Can

- Keep voxel models (up to 256 per side, one million voxels, 255-colour palette) with a version on every change
  (editor autosaves in a run share one version), duplicate, rename, resize, trash.
- Edit with operations (`model_ops`, all-or-nothing): voxel, voxels, box, line (with `r`/`r_to` a tapered capsule),
  prism (a polygon on a plane with thickness: wings, fins, ears), sphere/ellipsoid, cylinder, layer, inflate, paint,
  lathe, recolor, mirror, move, clear.
- 3D editor in the browser: add/erase/paint/fill/eyedropper/box, X/Z symmetry, layer slicer, undo/redo (300), palette
  editing, iso/front/side/back/top views, drag-and-drop of pictures and .vox files.
- Read reference pictures (PNG, JPG, GIF, WebP, BMP): crop, flat background to transparent, pixel-art scale undone,
  reduced to a size and a number of colours (`reference_grid`).
- Instant models from pictures: sprite 3D (picture on front and back, gentle rounded thickness, dark outline kept off the
  sides), inflate, extrude, hull (front + side + optional top).
- Import Minecraft schematics (`.schem`, WorldEdit/Sponge v1-3): blocks become their texture colours (table + rules for
  the dyed families); a web page saved as .schem is recognised and explained.
- Agent generations (default method `agent`): after the design brief the model works in a tool loop with the conversation
  kept: actions `run` (complete program; answer = voxel count and a contact sheet, or the error and failing line), `edit`
  (exact old/new replacements on the program of the last version that ran, then run; the edited program is shown
  back), `revert` (back to the best version so far, sprite target), `look` (large render from any side) and `done`; budgets `agent_steps` (16), `agent_minutes` (90) and `agent_step_thinking`
  (30 000 per action). Each working run is saved; the program and brief stay in the source.
- Characters by parts (`spriteparts.py`, `partsgen.py`; generation `method: parts`, picked on its own when a reference
  has role `side` and is a sprite and no method is given; tools `sprite_parts_read`, `model_from_parts`,
  `model_parts_adjust`; UI method *Automático* / *Personaje por partes* and the panel *Partes del personaje*,
  `static/parts.js`). The side sprite is read into a profile: ink colours (dark and found on the figure's edge far
  more than their share), cells (patches of one colour between lines). A plan `{"facing", "parts": [...]}` gives each
  part a `type` and rough shapes in side-sprite columns and rows (`where`: rectangles, polygons, per-shape or per-part
  `colors` as map letters or hex; `cells`; `except`): a cell that is one part's but for a sliver (60% its own and no
  other part taking more than 12% or 6 pixels) goes to it whole, a cell several parts really share is split along the
  shapes. Where shapes overlap, one lying mostly inside another wins there (a wing over a body drawn as one rectangle);
  shapes that overlap only at their borders leave each cell with the part that has the rest of it; in a cell both really
  share, a polygon or a colour-filtered shape beats a bare rectangle (the later of two such wins) and two bare rectangles
  meet halfway; lines and leftovers join their neighbours. Types: `solid`
  (all solids are puffed up as one volume with circular sections scaled by `aspect` or to `width`, blended across joins),
  `fin` (`thickness`), `fan` (`base`, `width` or `spread`, `root`, `towards`, `rim`: rounded towards its upper and lower edges, by default over
  0.6 of its half thickness; flat end), `pair` (`offset`, `width` or
  `thickness`, `lean`, `side` both/near/far), `wings` (flat sheets hinged at `root`: `out_up`, `out_back` (when neither is given `tune_wings` builds the part at 30, 45, 60 and 75 degrees from straight up
  (straight up is a thin V, straight back a wall) and keeps the opening with the best mean silhouette overlap with the front and back sprites; plans written by the
  model never give them, except when carrying on with a request in words), `span`,
  `offset`, `thickness`), `decal` (exact pixels, shown only where the side cameras see), `skip` (not built; what lies
  under it is completed from the median parts). `far_copies`: inside a wings or pair part whose `far` is auto, a second
  area of the main area's colour family behind a line of its own (at least 0.3 of its size; the main area is the one with
  most pixels counting the other-coloured areas it carries) becomes a skip part named `<part> (far)` (so does a smaller such area that touches a skip part of the plan: what its
  rough shape left of the far copy); the report says so
  and `far: keep` builds it. Plan key `eye` ([column, row] of the side sprite, `_check_eye`): a plan whose eye lies off
  the figure, in a wings, fan, fin or skip part, or in the back third for its `facing` raises `PlanError` (the model reads
  the sprite again); a note warns when nothing around it looks like an eye. Other notes in the report: a skip part of colours no wings or
  pair part has (`_check_skips`), a narrow foot at the bottom rows owned by a solid (`_check_stand`). Nothing floats: voxels of a pair or wings part that
  cannot be reached from the solids through touching voxels (`_floating`, 26 neighbours) are moved towards the middle plane,
  each side as a block, until they touch the body or something that hangs from it (`settled`: how many voxels in; no bar and no
  note), and what still floats is said (`_check_floating`). A wings part with no `offset` starts where it meets the body in the
  profile, not out where the body is widest at its root. A sheet whose opening Seurat chose opens the way it is drawn from its
  root: downwards when it hangs below it (`hangs`), towards the front when it lies ahead of it (`forward`). A part made of lines
  only takes the colours beside it, and a piece of a part cut off from its colours takes the part's main colour (never the ink).
  What the plan's author reads back (the parts picture, `root`, the columns in the notes) uses the sprite's own columns also when
  the sprite faces left (the profile is kept facing right inside). Plans written by the model cannot
  force `paint` either. Plans written by the
  model never keep the far copy (`partsgen._drop_keep`); a person can, in the panel or with the tools. Carrying on (`partsgen._base_model`): `continue_from` (or `into` with
  start `current`) on a model whose source is a parts plan builds that plan as step 0 and opens the loop with the sprites
  kept in `source.entries`, the pictures of the model and the prompt as the request; UI *Pedir el cambio* in the parts
  panel. Loop action `revert` (`step`) rebuilds the plan of an earlier built step. Second opinion (`partsgen.second_opinion`,
  setting `parts_check`, on by default): after the first plan that builds, one call in a fresh context with the sprites, the
  parts picture and a list of each part's columns, rows and colours answers `{head, wrong[], missing[], verdict}`; faults
  (not the unsure ones, not those about parts Seurat left out itself) are appended to the loop's next message, the round
  keeps them in `check`, and the plan that answers them is checked once more. Guarantees: every sprite pixel that is not skipped is a column of voxels,
  nothing outside the silhouette, left-right symmetric, the first voxel from each side shows the sprite's pixel. A `width` given to a solid or a fan is met exactly. `paint_faces`: a solid takes
  the front or back sprite's own pixels on the voxels that camera sees and whose surface looks more at it than sideways
  (`surface_normals`), when that sprite's silhouette overlaps the model's by `FACE_IOU` (0.82) or the part asks (`paint`:
  auto, none, front, back, both); when the whole figure does not, each solid is tried on its own (`_band_pose`: over the
  rows the part takes from that camera the two silhouettes must overlap by `FACE_IOU` at a placement within 2 pixels of the
  whole figure's, which is then the part's own placement); small dark spots (pupils) kept.
  Parts placed in the other sprites (plan keys `front`, `back`: shapes in that sprite's own columns and rows; written by
  hand, the generation prompt does not ask for them): `read_view` reads that sprite like the side one, snaps the shapes to
  its patches (a patch mostly inside the shapes is the part's whole, a pixel away from every shape is nobody's) and keeps,
  per part and per row, how far from the sprite's symmetry axis it reaches (`View.inner`, `View.outer`, in voxels to each
  side). `build(..., sprites=)` uses the front view of a part, or the back one (`across`): a solid is as deep as the side
  sprite draws it and as wide as that view draws it at every height (its rows laid over the part's rows in the profile,
  an ellipse across each run of the row; `width` given by hand keeps the round puffed form), a pair stands as far from the
  middle and is as thick as its two copies there, a fin stays a thin slab and gets a second thin blade across it as wide as
  that view draws it, a fan takes its end width, and wings are laid out from their own drawing (`planform`: those sprites
  are drawn from above, so the part's pixels there are the wing's outline and markings; each pixel becomes voxels at its
  distance from the axis, along the body at one profile column per `scale` rows of the drawing with the drawing's inner end
  at `root`, lifted from the root row by `rise` at the tips: the height the side sprite raises the wing, negative when it
  hangs below its root; the grid grows along the body when the wing runs past the profile; its pixels in the profile build
  nothing, it keeps the drawing's colours and `tune_wings` leaves it alone). `paint_views` paints what each such solid or
  pair shows straight on to the front and back from its own place in that sprite (rows over rows, columns to each side of
  the middle plane); `paint_faces` is for parts with no place there. `reading_picture` shows every sprite next to its
  pixels by part, `sides_picture` each sprite next to the model from that side.
  Colours: the outline on the figure's edge takes its part's colour (`outline`: drop/keep/shade), inner ink follows `ink` (clean by default: a line of the drawing takes its part's colour and a small dark feature, an ink
  patch of up to 6 pixels off the outline, stays; soften, keep, drop), exposed voxels show the sprite's pixel and inner ones the part's own colour (also
  under what hides it). The game camera is oblique (`project`: heights kept, depth shifts the picture by `depth` pixels
  per voxel); `tune_camera` finds the depth of the front and back sprites; `compare` gives the silhouette overlap and a
  match that only counts pixels of the same colour family; `paint_caps` lays the back (or front) sprite's picture of a
  fan over the voxels that camera sees and the side cameras do not. The generation is an action loop (build, adjust,
  look, done; `parts_steps` 8 by default, the agent's minutes and thinking per step, twice the thinking for the first
  plan): after every build the model gets the report of each part, the parts picture, the sprite comparison and the lit
  views; each build is a version and a round (`render`, `sprite_render`, `parts_render`). The plan, the cameras and the
  reference roles stay in the model's source, so `model_parts_adjust` rebuilds it with changes.
- Sprite target (`target`: auto/sprite/free; auto = sprite when the front/back/side pictures are small pixel art): the
  sprites are the design to rebuild in 3D at one voxel per sprite pixel (`spritecam.grid_size`: X from the front/back
  width, Y and Z from the side sprite). The side sprite is a true profile (`sprite_side_pitch` 0), counts double in the
  score and reaches the program as `SIDE` (Z, Y) in grid coordinates with `SIDE_MASK`, `SPRITE_PALETTE`,
  `extrude_side`, `carve_side`, `paint_side` and `paint_view(g, role)` (`sandbox_runner.sprite_helpers`; data from
  `spritecam.program_data`); the prompt carries the palette and a pixel map of each sprite with the grid mapping.
  Front/back sprites are seen from `sprite_pitch` (45) degrees above; their horizontal axis is x. After every run
  `spritecam.match` projects the model from each camera at one voxel per pixel (eight samples per voxel, nearest wins,
  flat colours), places it on the sprite (box centres, then the best shift within 4 px) and returns silhouette IoU,
  missing/extra shares, colour difference (outline excluded) and the model's width/height relative to the sprite, plus
  a sheet (sprite, model, red/blue diff) the agent sees and the job keeps per round (`sprite_render`, `match`, `views`).
  With each view it also gets the biggest missing/extra areas in numbers (`spritecam.where`: grid z/y for the side,
  grid x and sprite rows for front/back).
  Models far from the sprite's size are scaled instead (`fit` auto/pixel/area). The side sprite is tried from both
  sides. `model_sprite_match` does the same for any model; `model_look` has views overworld_front/back/left/right
  (`render.camera`: orthographic from any yaw and pitch); `model_program` takes `sprites` for the same tools. The agent tracks its best
  sprite score: a worse run is flagged in the next message and the best version (grid, palette, program) is what the
  generation keeps. `continue_from` (model id, agent only) carries a generation on from that model's program and brief
  without a new brief; new versions go to that model.
- Sprite sheets and animations (`sheets.py`, tool `reference_frames`, REST `/api/references/<id>/frames` and
  `frames.png`, UI *Fotogramas* on every reference card): every frame of a GIF/animated WebP/APNG (up to 96; a flat
  backdrop is removed also when the frame mixes it with transparency) or every cell of a sheet (grid from the gaps, from
  the repeat of touching sprites over the occupied extent, or `cols` x `rows`; pixel-art scale undone once for the whole
  sheet; specks under 15% of a typical frame dropped). `guess_views`: sheets are grouped by rows or columns (whichever
  groups more alike frames), animations cut into equal runs when they turn; the asymmetric groups (outline and colours)
  are sides, the one with more inner detail at the top is the front, the plainest the back; the medoid (standing frame)
  comes first. Reference entries take role `sheet` (all sides, `views` to override, `cols`/`rows`), or front/back/side/
  `view` with `frames` (first = main, rest = poses); an animated picture with a plain role gives its standing frame and
  the others as poses (`Services.expand_references`). Poses are matched against the same render: a view's score is half
  its main frame, half the poses' mean; a role's weight is shared by its sprites (two side sprites weigh one side).
- Camera calibration (`spritecam.calibrate`, tool `model_sprite_calibrate`, REST `POST /api/models/<id>/calibrate`, UI
  *Calibrar con el modelo abierto*): a coarse sweep of yaw and pitch around each role's usual camera (front/back 0-80
  degrees above and 20 around, side 0-60 on both sides, `view` all around), then finer steps, comparing one-voxel-per-pixel
  silhouettes (pixel fit with the best shift, or area fit for models of another scale). `apply` saves the front/back
  and side heights as settings. In agent generations (setting `sprite_calibrate`, param `calibrate`) it runs after the
  first run, every third run and while a `view` has no camera, within 20-70 degrees and 10 around, and a sprite takes
  the new camera only when it matches 0.02 better; the program data, scores and the best version are recomputed with it.
- Lit renders (`lit.py`): key light with shadow rays marched through the grid at each face corner, sky occlusion from
  nine short rays per corner, a ground at y = 0 fading out around the model; lighting baked once per model, each camera
  only projects k x k samples per face in painter's order (supersampled 2x within 4 M samples). About 1 s for a 128 x 64 x
  128 scene of 130 k faces. Views `lit`, `lit_front_left`, `lit_back_left`, `lit_back_right`, `lit_front`, `lit_top`,
  `sheet_lit`, any `yaw`/`pitch` (`model_look`, `/render.png`, PNG export). The review sheets of every method and the
  agent's `look` of iso views use them (setting `review_render` lit/flat).
- Spins (`lit.spin`, tool `model_spin`, REST `/api/models/<id>/spin/{gif|webm}`, Export tab): the camera circles the model
  with the light fixed in the world, fixed canvas and centre; GIF via Pillow, WebM (VP9) via ffmpeg (`imageio-ffmpeg` or
  PATH). 24 frames of the 128-side scene take about 13 s.
- Rounds in 3-D (`rounds.py`): every recorded round with voxels is kept as `renders/<job>-r<n>.npz` (grid + palette);
  tool `generate_round` (lit picture; `save` makes a model, source kind `round`), REST `/api/jobs/<job>/rounds/<n>/model`
  (voxels for the editor) and `/save`; UI *Ver en 3D* opens it in the editor as a preview that is not autosaved (bar with
  *Guardar como modelo* / *Volver*) and *Guardar como modelo*.
- Scenes by parts (`scene.py`, `generate_start` method `scene`, UI method *Escena por partes*): plan JSON (objects with
  size, description, `ground`, placements with position and quarter-turn rotation; at most `scene_objects`), shown as
  hollow boxes (round 0); each object built by its own agent loop (`scene_steps` actions, the time budget shared) in its
  own grid (up to 128; the ground up to the scene's X x Z), recorded rounds tagged with the object; `compose` turns,
  places (ground first, the rest set down on what is under their lowest layer), merges palettes (equal colours shared,
  nearest past 255) and reports clipped placements; one layout check (lit view + top view, only placement changes) when
  rounds > 0. Scene grid `size` or setting `scene_size`, up to 256 a side. The model's source keeps the plan and every
  object's program.
- Voxel programs in fixed rounds (method `code`): first the model writes a design brief (what the subject is from its
  own knowledge and the description, parts with sizes and x/y/z positions, palette, what must read from each side;
  pictures are only hints, never traced), shown in the job and kept in `source.brief`; then it writes Python (`PALETTE` + `build(g)`, helpers ellipsoid,
  sphere, capsule, box, polygon, mirror_x, shade_by_height; numpy masks over xs/ys/zs), run by `program.py` in a separate
  `python -I` process (`sandbox_runner.py`): restricted builtins, imports limited to math/numpy/random/itertools/
  functools/colorsys, source checked for files/processes/dunder names, `program_timeout_s` limit. Each round renders the
  result and asks for a complete better program; failures and empty grids are fed back; the last working program is
  stored in the model's `source.program`. Grid `code_size` (64) or the call's `size`, up to 128.
- Run a voxel program directly (`model_program`): into a new model or replacing one (kept as a version); the program
  stays in the model's source. Code generations save each working round as they go (a restart loses nothing).
- Generations with the language model (`generate_start` → `generate_status`): DeepSeek V4.1 Flash on the Sparks by
  default (resolved through Hoard Link and Faustus's registry, also while Faustus runs on a Spark), or any
  OpenAI-compatible URL from Settings. By default it reasons at the highest level (effort `max`: up to 200 000 thinking
  tokens per round, counted exactly from the server's usage; when the thinking reaches that budget, or the server cuts
  it, a second call without thinking gets the whole reasoning and writes the answer, up to 24 000 tokens) and sculpts the shape from parts on an empty grid; Seurat then
  paints the front and back pictures onto the shape after every round (the side picture only with `paint_sides`), so the model works on
  volume and the colours come from the pictures. Answers are constrained to JSON, streamed (progress while thinking,
  instant cancel, loops in the answer or the thinking cut), and reviewed for rounds with a six-view contact sheet and the
  front-silhouette differences. A round that empties, only clears or destroys most of the model is undone and the model
  is told why. Cut answers keep their complete operations; a failure part-way saves what was built (or the instant
  model when nothing was). Start from auto (empty grid), inflate, hull, extrude, empty or the current model (refine).
- Generate from a description alone (no pictures): objects, characters or whole scenes in a grid of the given size
  (`generate_start` with `prompt` and `size`, no references).
- Run generations one after another (default, each thinks faster) or several at the same time (`parallel_jobs`).
- Export `.vox`, `.glb`, `.stl` (mm), PNG of a view, isometric sprite strip; import `.vox`.
- Software renders (no GPU): flat views, isometric views, labelled contact sheets.

## Cannot

- Generate textures or smooth meshes: everything is voxels with flat palette colours.
- Animate a model: the frames of a walk are poses that judge one static model (a neutral pose between them); there is no
  rig, no posed copies and no animated voxel export.
- Always guess the sides of a sheet right: front and back are told apart by detail at the top (a face), so a back with a
  busy pattern can pass for the front; 8-direction sheets get front, back and two sides, the diagonals stay unused
  unless set as `view` (the *Fotogramas* dialog changes any of it). Animations are cut only into equal runs.
- Calibrate against a bad model: the camera bends to whatever silhouette there is; during generations it is held near
  the usual camera for that reason.
- Make WebM without ffmpeg (GIF always works). Light the scene from more than one fixed light (no coloured or moving
  lights, no reflections).
- Move the objects of a generated scene afterwards as objects: once composed it is one voxel model (the plan and the
  programs stay in its source).
- Guarantee that the model's back and sides match what the artist imagined: they are invented from the front (add side,
  top or back pictures to constrain them).
- Be quick with the default settings: with effort `max` each round takes several minutes (a four-round generation can
  take half an hour or more). Effort `off` answers in seconds but sculpts much worse.
- Copy game sprites as exact projections: overworld sprites are drawn from above at an angle and by hand, so even with
  the sprite target only the side profile is exact; the front/back camera (45 degrees by default, or the calibrated one:
  the angle that fits the model best, not necessarily the artist's) is approximate, and the parts no sprite shows
  (underside, between the wings) are the model's own reading.

## Where things are

- Data: `data/` (models/<id>/model.json and versions/, references/, renders/ with each round's sheet and raw answer,
  exports/, settings.json, mcp-token). Trash: `data/trash/`.
- Voxel engine `voxels.py`; builders `builders.py`; pictures `images.py`; renders `render.py`; exports `exporters.py`;
  generations `generate.py`; sprite target, game cameras and calibration `spritecam.py`; sheets and animations
  `sheets.py`; lit renders and spins `lit.py`; scenes by parts `scene.py`; round snapshots `rounds.py` (files in
  `data/renders/`); model client `llm.py`; UI in `static/` (three.js r170 vendored).

## Skills for the agents that connect

Seurat takes from the skill library of the family's workspace the procedures its work needs (3D assets for the web, Blender assets, composing images, verifying a result) and hands them to every agent that connects: they are named in what the agent reads on connecting, read with `skills_list` and `skill_read`, and offered as prompts. The `skills` block of `faustus-plugin.json` says which; the copies live in `data/skills/` and follow the library when a skill is improved there.
