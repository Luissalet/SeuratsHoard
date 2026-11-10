# Seurat's Hoard

Voxel studio of the Hoard family. Draw voxel models in a 3D editor, turn pictures and sprites into voxel models, and
export them to MagicaVoxel, Blender/Godot/Unity, a 3D printer or back to sprites.

- **Editor** (browser or Hoard window, port 5250): add, erase, paint, fill, eyedropper and box tools, X/Z symmetry, a
  layer slicer to work inside the model, undo/redo, palette editing, autosave with versions you can restore.
- **Pictures to voxels**
  - *Instant* (no model): *sprite 3D* (the sprite as a voxel diorama piece, like the sprite-to-voxel game mods: the
    picture on front and back, a gentle rounded thickness, the 2D outline kept off the sides), *inflate* (the front puffed up like a cushion), *extrude* (a slab) or *silhouette* (front +
    side views carved into a block, coloured from each view).
  - *As an agent* (default): after the design brief, DeepSeek works in a loop like an agent harness: it runs its voxel
    program, gets the result or the error with the failing line, looks at the model from any side, fixes and refines
    (rewriting the program or editing parts of it, and going back to its best version when a change makes it worse),
    and says when it is finished (step and time budgets in Settings). Every run that works is saved.
  - *Characters by parts* (`method: parts`; what runs on its own when there is a side sprite): the side sprite is the
    model's exact profile (every pixel is a column of voxels) and a plan says what each part is across the body: `solid`
    (a rounded solid on the middle plane: body, head, thick tail), `fin` (a thin slab), `fan` (widens away from its root,
    rounded towards its edges; its end is painted from the back sprite), `pair` (legs, arms, ears: one on each side), `wings` (two flat sheets that
    leave the body, with a span and a direction), `decal` (a marking only on the flanks: eyes) and `skip` (what the side
    sprite shows only for looks, like the far wing peeking out: not built, the mirrored near one already is it). DeepSeek
    decides that structure and reads the widths off the front and back sprites; the geometry, the colours (the black
    outline goes, inner lines are toned down) and the comparison are Seurat's. When the front or the back sprite shows the same pose as the model (standing characters,
    chibis), what faces that camera is painted from it: the face from the front, the hair or the back from behind; the
    flanks stay the side sprite (`paint` on each solid: `auto`, `none`, `front`, `back`, `both`). After every plan it shows the model which
    pixels each part caught, each sprite next to the model from its camera (oblique: heights are kept and depth shifts;
    how much is found for each sprite) and the model lit from several sides, and the model adjusts or rewrites the plan.
    Shapes are rough (rectangles or polygons in the sprite's columns and rows, with a colour filter): they snap to the
    sprite's patches of colour and its lines. The plan stays with the model: the *Partes del personaje* panel changes
    widths, wingspan, how the wings open, how far apart the legs are or what each part is, and rebuilds; the whole plan
    can be edited there too.
  - *Game sprites as the target* (`target: sprite`, automatic for small pixel art): the front, back and side pictures
    are overworld sprites to rebuild in 3D, the way voxel mods do it (about one voxel per sprite pixel, the sprite's own
    colours and design). The side sprite is a true profile (no camera angle): the most important view, it counts double
    and the program gets it as an array in grid coordinates (`SIDE`, column = z, row = y) with tools to carve the model
    to it (`carve_side`), fill it (`extrude_side`) and paint from it (`paint_side`, `paint_view` for any sprite), plus the
    sprites' palette and pixel maps of every sprite in the prompt. Front and back sprites are drawn from about 45 degrees
    above (their horizontal axis is exactly x; Settings). After every run Seurat projects the model from each sprite's
    camera at one voxel per pixel and shows the model each sprite next to it with a silhouette diff (red missing, blue
    extra), match scores and how its size compares; the job panel shows them per round. The agent is told when a run
    is worse than its best so far and the best version is the one kept. `continue_from` carries a generation on from a
    model's program and brief (restore the step you want first).
  - *Sprite sheets and animated sprites*: every frame of a GIF (or animated WebP/PNG) and every cell of a sheet is read
    (the grid is found from the gaps between the sprites, from their repeat when they touch, or given as columns x
    rows). Seurat guesses which frames show which side (walk sheets: rows or columns are directions, the rest steps;
    an animation that turns is cut into its runs; asymmetric groups are sides, the one with a face is the front) and you
    can change it in *Fotogramas*. With the role *sheet* every side becomes another camera of the same model and its other
    frames count as poses: the standing frame is the main view, the steps share half of that view's score.
  - *Camera calibration*: after the first runs (and every third) the agent finds from which angle the game drew each
    sprite, comparing the model's silhouette from many heights and turns (within 20-70 degrees and 10 around while the
    model is being built, and only when a sprite matches clearly better), instead of the fixed 45/0 degrees; a side whose
    camera is unknown (role *view*) finds its turn all around. *Calibrar con el modelo abierto* (`model_sprite_calibrate`)
    does it for any model with a wide search, shows each sprite against the model from its camera and can save the
    heights found as settings.
  - *Scenes by parts* (`method: scene`, grids up to 256 a side): the model plans the scene as objects (a ground, then
    each object with its size and placements; repeated objects are built once and placed several times, turned), each
    object is built by its own agent loop in its own grid (up to 128), Seurat places them and sets them down on what is
    under them, merges their palettes, and the model checks the layout once from a lit view and the top. The plan is
    shown as boxes first; every object's programs are kept with the scene.
  - *Rounds in 3-D*: the voxels of every round of a generation are kept, so any intermediate round opens in the editor
    (*Ver en 3D*, a preview that is not saved) or becomes a model of its own (*Guardar como modelo*, `generate_round`).
  - *With the language model, as a program in fixed rounds*: DeepSeek V4.1 Flash first writes a design brief from what it
    knows about the subject and your description (parts, proportions, positions in 3D, palette; pictures are only
    hints, never traced), then a Python program that builds the
    voxels (blended ellipsoids, tapered capsules, loops of feathers, polygons, symmetry, numpy masks), Seurat runs it in a
    separate process with a time limit and no files or network, shows the result and asks for a better program round
    after round. The last working program is kept with the model. Grids up to 128 per side; scenes from a description
    alone or creatures from pictures (official artwork works far better than small game sprites).
  - *With the language model, as operations* (`method: ops`): it sees
    the reference pictures and their pixel grids, thinks at length (effort `max` by default, several minutes per round)
    and sculpts the shape from parts (ellipsoids, tapered capsules for necks and tails, polygon prisms for wings and
    fins). Seurat paints the pictures onto that shape (front and back; the side picture too if you switch it on), and the model reviews a contact sheet of
    its own result (six views, front-silhouette differences, operation errors) for as many rounds as you choose. A round
    that would empty or wreck the model is undone. Every round is shown with its render, notes and errors; the
    answers are kept as text next to the renders. Roles per picture: front, side, top, back, extra.
  - *From a description alone* (no pictures): the model builds an object, a character or a whole scene from the prompt
    in a grid of the chosen size, with the same review rounds.
  - Pixel art enlarged by an integer factor is read at its real size; a flat background colour becomes transparent.
- **Lit renders**: the 3-D views of the review sheets the model sees, `look` and `model_look` (`lit`, `lit_front_left`,
  `lit_back_left`, `lit_back_right`, `lit_front`, `lit_top`, any yaw/pitch, `sheet_lit`) have real light: a key light
  with cast shadows marched through the grid, ambient occlusion in the creases and a ground that catches the shadow,
  computed in numpy without a GPU (setting `review_render`: lit or flat).
- **Spin animations**: a turntable of the model with that light, as GIF or WebM (VP9 through ffmpeg), from the Export
  tab or `model_spin`.
- **Exports**: `.vox` (MagicaVoxel 150, round trip), `.glb` (vertex colours, greedy-meshed), binary `.stl` in millimetres
  (closed mesh, z up, setting `voxel_mm`), PNG of any view, and an isometric sprite strip from four sides. `.vox` files
  and Minecraft `.schem` schematics (WorldEdit) can be imported.
- **Assistants** (MCP, `mcp_server.py`, 36 tools): list/create/edit models with operations (`model_ops`), look at them
  (`model_look` returns a picture, also from the game overworld cameras and lit from any side), compare a model with game
  sprites from their camera (`model_sprite_match`, also with sheets and poses), calibrate those cameras
  (`model_sprite_calibrate`), read the frames of sheets and animations (`reference_frames`), run voxel programs with the
  sprite tools (`model_program` with `sprites`), add references, build or generate from pictures or as scenes by parts,
  follow and cancel generations, open or save any round (`generate_round`), export, spins (`model_spin`), versions,
  settings.

## The operation language

Axes: x right, y up, z towards the viewer. `c` is a palette index; `c: 0` erases.
`voxel`, `voxels`, `box`, `line` (with `r`/`r_to`: a tapered capsule), `prism` (a polygon on a plane with thickness:
wings, fins, ears), `sphere`/`ellipsoid`, `cylinder`, `layer` (pixel rows on the front, side or top
plane), `inflate` (pixel rows puffed into a volume), `paint` (recolour the surface seen from a side, pixel by pixel),
`lathe` (solid of revolution), `recolor`, `mirror`, `move`, `clear`. Details in `seurat_hoard/voxels.py`.

## Run

```
python -m venv .venv
.venv\Scripts\python -m pip install -r requirements.txt
.venv\Scripts\python -m seurat_hoard
```

Opens at http://127.0.0.1:5250. The Hoard Hub starts it like the rest of the family. Data (models, references, renders,
settings, `mcp-token`) in `data/`. Settings: reading size and colours, review rounds, reasoning effort and thinking
tokens, painting the pictures onto the shape, camera calibration, lit or flat review views, scenes (actions per
object, most objects, default size), a fixed OpenAI-compatible model URL instead of the family's, STL and GLB scale.
WebM spins need ffmpeg on the PATH (or the `imageio-ffmpeg` package); GIF needs nothing.

Tests: `.venv\Scripts\python -m pytest`.
