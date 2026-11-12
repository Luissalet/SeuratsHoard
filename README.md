# Seurat's Hoard

A voxel studio: draw voxel models in a 3D editor, turn pictures and sprites into voxel models, and export them to MagicaVoxel, Blender/Godot/Unity, a 3D printer or back to sprites.

[Leer en español](README.es.md)

## What it is

Seurat's Hoard is a local app of the Hoard family (default port 5250). It has a voxel editor and a generation pipeline in which a language model designs and builds voxel models from reference pictures, sprites, sprite sheets or a plain description, while Seurat runs the geometry, colouring, rendering and comparison itself.

## What it does

**Editor** (browser or Hoard window)
- Add, erase, paint, fill, eyedropper and box tools, X/Z symmetry, a layer slicer to work inside the model, undo/redo, palette editing, and autosave with versions you can restore.

**Pictures to voxels**
- *Instant* (no model): *sprite 3D* (the sprite as a voxel diorama piece, picture on front and back with a rounded thickness), *inflate* (the front puffed up like a cushion), *extrude* (a slab) or *silhouette* (front and side views carved into a block, coloured from each view).
- *As an agent* (default): after a design brief, the language model works in a loop like an agent harness. It runs its voxel program, gets the result or the error with the failing line, looks at the model from any side, fixes and refines (rewriting the program or editing parts, and going back to its best version when a change makes it worse) and says when it is finished. Step and time budgets are in Settings, and every run that works is saved.
- *In fixed rounds*: the model writes a design brief from what it knows about the subject and your description (pictures are hints, never traced), then a Python program that builds the voxels (blended ellipsoids, tapered capsules, polygons, symmetry, numpy masks). Seurat runs it in a separate process with a time limit and no files or network, shows the result and asks for a better program each round. Grids up to 128 per side.
- *As operations* (`method: ops`): the model sees the reference pictures and their pixel grids, thinks at length and sculpts the shape from parts (ellipsoids, tapered capsules, polygon prisms). Seurat paints the pictures onto that shape, and the model reviews a contact sheet of its own result (six views, front-silhouette differences, operation errors) for as many rounds as you choose; a round that would empty or wreck the model is undone. Picture roles: front, side, top, back, extra.
- *From a description alone*: an object, a character or a whole scene built from the prompt in a grid of the chosen size, with the same review rounds.
- *Characters by parts* (`method: parts`, automatic when there is a side sprite): the side sprite is the model's exact profile and a plan says what each part is across the body (`solid`, `fin`, `fan`, `pair`, `wings`, `decal`, `skip`). The model decides the structure and reads widths off the front and back sprites; geometry, colours and comparison are Seurat's. It checks the plan (the eye must sit on the head, floating wings and legs are brought to the body), paints faces and backs from the front and back sprites when they show the same pose, shows the model which pixels each part caught and the result lit from several sides, and runs an independent check that each part name matches its pixels. The *Partes del personaje* panel changes widths, wingspan, how wings open, leg spacing or part types and rebuilds; *Pedir el cambio* (or `continue_from` with a text) lets the model carry on from the plan with a request in words.
- *Game sprites as the target* (`target: sprite`, automatic for small pixel art): the front, back and side pictures are overworld sprites rebuilt in 3D at about one voxel per sprite pixel, with the sprite's own colours. The side sprite is a true profile and counts double; the program gets it as an array with tools to carve, fill and paint from it (`carve_side`, `extrude_side`, `paint_side`, `paint_view`). After every run Seurat projects the model from each sprite's camera and shows each sprite next to it with a silhouette diff (red missing, blue extra) and match scores. The best version is the one kept.
- *Sprite sheets and animated sprites*: every frame of a GIF (or animated WebP/PNG) and every cell of a sheet is read (grid found from gaps or repeats, or given as columns × rows). Seurat guesses which frames show which side and you can change it in *Fotogramas*; with the role *sheet*, each side becomes another camera of the same model and other frames count as poses.
- *Camera calibration*: the agent finds from which angle the game drew each sprite instead of the fixed 45/0 degrees. *Calibrar con el modelo abierto* (`model_sprite_calibrate`) does it for any model and can save the heights found as settings.
- *Scenes by parts* (`method: scene`, grids up to 256 a side): the model plans the scene as objects (a ground, then each object with size and placements; repeated objects are built once), each object is built by its own agent loop in its own grid, Seurat places them on what is under them and merges palettes, and the model checks the layout once. The plan is shown as boxes first.
- *Rounds in 3-D*: every round of a generation is kept, so any round opens in the editor (*Ver en 3D*) or becomes a model of its own (*Guardar como modelo*, `generate_round`).
- Pixel art enlarged by an integer factor is read at its real size; a flat background colour becomes transparent.

**Lit renders and animation**
- The 3-D views of the review sheets (`look`, `model_look`: `lit`, `lit_front_left`, `lit_back_left`, `lit_back_right`, `lit_front`, `lit_top`, any yaw/pitch, `sheet_lit`) have real light: a key light with cast shadows, ambient occlusion in the creases and a ground that catches the shadow, computed without a GPU (setting `review_render`: lit or flat).
- Spin animations: a turntable of the model with that light, as GIF or WebM (VP9 through ffmpeg), from the Export tab or `model_spin`.

**Import and export**
- Export `.vox` (MagicaVoxel, round trip), `.glb` (vertex colours, greedy-meshed), binary `.stl` in millimetres (closed mesh, z up, setting `voxel_mm`), PNG of any view, and an isometric sprite strip from four sides.
- Import `.vox` files and Minecraft `.schem` schematics (WorldEdit).

**The operation language.** Axes: x right, y up, z towards the viewer; `c` is a palette index and `c: 0` erases. Operations: `voxel`, `voxels`, `box`, `line` (with `r`/`r_to`: a tapered capsule), `prism` (a polygon with thickness: wings, fins, ears), `sphere`/`ellipsoid`, `cylinder`, `layer` (pixel rows on the front, side or top plane), `inflate`, `paint` (recolour the surface seen from a side), `lathe`, `recolor`, `mirror`, `move`, `clear`. Details in `seurat_hoard/voxels.py`.

## How to run it

```
python -m venv .venv
.venv\Scripts\python -m pip install -r requirements.txt
.venv\Scripts\python -m seurat_hoard
```

It opens at http://127.0.0.1:5250. The Hoard Hub starts it like the rest of the family. Data (models, references, renders, settings, `mcp-token`) lives in `data/`. Settings cover reading size and colours, review rounds, reasoning effort and thinking tokens, painting the pictures onto the shape, camera calibration, lit or flat review views, scene limits (actions per object, most objects, default size), a fixed OpenAI-compatible model URL instead of the family's, and STL and GLB scale. WebM spins need ffmpeg on the PATH (or the `imageio-ffmpeg` package); GIF needs nothing.

## Connection with Faustus and the Hub

The plugin manifest is `faustus-plugin.json`. The MCP bridge (`mcp_server.py`) exposes the studio to assistants, by group:
- Models: `model_list`, `model_create`, `model_get`, `model_duplicate`, `model_rename`, `model_resize`, `model_delete`, `model_ops` (edit with operations), `model_program` (run a voxel program, with `sprites`), `model_palette`, `model_dump`, `model_export`, `model_import_vox`, `model_versions`, `model_restore`, `version_note`.
- Looking and comparing: `model_look` (returns a picture, also from the game overworld cameras and lit from any side), `model_sprite_match`, `model_sprite_calibrate`, `model_spin`.
- References: `reference_add`, `reference_list`, `reference_look`, `reference_grid`, `reference_frames`, `reference_sheet`, `reference_bytes`, `reference_delete`.
- Generation: `generate_start`, `generate_status`, `generate_list`, `generate_cancel`, `generate_round`, `model_from_parts`, `model_parts_adjust`, `model_rebuild`.
- Settings: `settings_get`, `settings_set`.

**Skills.** Seurat takes from the family's skill library the procedures its work needs (3D assets for the web, Blender assets, composing images, verifying a result) and hands them to every agent that connects: they are named in what the agent reads on connecting, read with `skills_list` and `skill_read`, and offered as prompts. The `skills` block of `faustus-plugin.json` says which; the copies live in `data/skills/` and follow the library.

## Project layout

| Path | Contents |
|---|---|
| `seurat_hoard/` | the app: store, voxel engine and operation language, generation loops, parts, scenes, sprite cameras, sheets, lit renderer, exporters, API, static web interface |
| `seurat_hoard/hoard_link/` | vendored family library |
| `mcp_server.py` | MCP bridge |
| `faustus-plugin.json` | plugin manifest |
| `tests/` | test suite |
| `data/` | models, references, renders, settings (not tracked) |

## Tests

```
.venv\Scripts\python -m pytest
```

## Licence

MIT. Third-party components are listed in `THIRD_PARTY_NOTICES.md`.
