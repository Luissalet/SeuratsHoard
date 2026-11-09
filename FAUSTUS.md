# Seurat's Hoard — what it can and cannot do

App id `seurat`, service `seurats-hoard`, port 5250, MCP stdio bridge `mcp_server.py` (27 tools; `model_look` and
`reference_look` return MCP images). Connector preset: its own `faustus-plugin.json`.

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
- Instant models from pictures: inflate, extrude, hull (front + side + optional top).
- Generations with the language model (`generate_start` → `generate_status`): DeepSeek V4.1 Flash on the Sparks by
  default (resolved through Hoard Link and Faustus's registry, also while Faustus runs on a Spark), or any
  OpenAI-compatible URL from Settings. By default it reasons at the highest level (effort `max`: up to 100 000 thinking
  tokens per round, counted exactly from the server's usage; when the thinking reaches that budget, or the server cuts
  it, a second call without thinking gets the whole reasoning and writes the answer, up to 24 000 tokens) and sculpts the shape from parts on an empty grid; Seurat then
  paints the reference pictures onto the shape (front, back, both sides) after every round, so the model works on
  volume and the colours come from the pictures. Answers are constrained to JSON, streamed (progress while thinking,
  instant cancel, loops in the answer or the thinking cut), and reviewed for rounds with a six-view contact sheet and the
  front-silhouette differences. A round that empties, only clears or destroys most of the model is undone and the model
  is told why. Cut answers keep their complete operations; a failure part-way saves what was built (or the instant
  model when nothing was). Start from auto (empty grid), inflate, hull, extrude, empty or the current model (refine).
- Generate from a description alone (no pictures): objects, characters or whole scenes in a grid of the given size
  (`generate_start` with `prompt` and `size`, no references).
- Run several generations at the same time (`parallel_jobs`, 2 by default).
- Export `.vox`, `.glb`, `.stl` (mm), PNG of a view, isometric sprite strip; import `.vox`.
- Software renders (no GPU): flat views, isometric views, labelled contact sheets.

## Cannot

- Generate textures or smooth meshes: everything is voxels with flat palette colours.
- Read animated sprites beyond their first frame.
- Guarantee that the model's back and sides match what the artist imagined: they are invented from the front (add side,
  top or back pictures to constrain them).
- Be quick with the default settings: with effort `max` each round takes several minutes (a four-round generation can
  take half an hour or more). Effort `off` answers in seconds but sculpts much worse.
- Copy game sprites as exact projections: overworld sprites are drawn from above at an angle, so the side and top of
  the model are the model's own reading of the pictures.

## Where things are

- Data: `data/` (models/<id>/model.json and versions/, references/, renders/ with each round's sheet and raw answer,
  exports/, settings.json, mcp-token). Trash: `data/trash/`.
- Voxel engine `voxels.py`; builders `builders.py`; pictures `images.py`; renders `render.py`; exports `exporters.py`;
  generations `generate.py`; model client `llm.py`; UI in `static/` (three.js r170 vendored).
