# Seurat's Hoard

Estudio de voxels: dibuja modelos voxel en un editor 3D, convierte imágenes y sprites en modelos voxel y expórtalos a MagicaVoxel, Blender/Godot/Unity, una impresora 3D o de vuelta a sprites.

[Read in English](README.md)

## Qué es

Seurat's Hoard es una aplicación local de la familia Hoard (puerto 5250 por defecto). Tiene un editor de voxels y un proceso de generación en el que un modelo de lenguaje diseña y construye modelos voxel a partir de imágenes de referencia, sprites, hojas de sprites o una simple descripción, mientras Seurat se encarga de la geometría, el color, el renderizado y la comparación.

## Qué hace

**Editor** (navegador o ventana Hoard)
- Añadir, borrar, pintar, rellenar, cuentagotas y caja, simetría X/Z, corte por capas para trabajar por dentro, deshacer/rehacer, edición de la paleta y guardado automático con versiones que puedes recuperar.

**De imagen a voxels**
- *Rápido* (sin modelo): *sprite 3D* (el sprite como pieza de diorama voxel, la imagen delante y detrás con un grosor redondeado), *inflar* (el frente abombado como un cojín), *extruir* (una losa) o *silueta* (vistas de frente y lado talladas en un bloque, coloreadas desde cada vista).
- *Como agente* (por defecto): tras un informe de diseño, el modelo de lenguaje trabaja en bucle como un arnés de agente. Ejecuta su programa de voxels, recibe el resultado o el error con la línea que falla, mira el modelo desde cualquier lado, corrige y afina (reescribiendo el programa o editando partes, y volviendo a su mejor versión cuando un cambio empeora) y avisa cuando termina. Los presupuestos de pasos y tiempo están en Ajustes y cada ejecución que funciona se guarda.
- *En rondas fijas*: el modelo escribe un informe de diseño con lo que sabe del tema y tu descripción (las imágenes son pistas, nunca se calcan), y después un programa Python que construye los voxels (elipsoides fundidos, cápsulas cónicas, polígonos, simetría, máscaras numpy). Seurat lo ejecuta en un proceso aparte con límite de tiempo y sin ficheros ni red, muestra el resultado y pide un programa mejor en cada ronda. Cuadrículas de hasta 128 por lado.
- *Como operaciones* (`method: ops`): el modelo ve las imágenes de referencia y sus cuadrículas de píxeles, piensa a fondo y esculpe la forma con partes (elipsoides, cápsulas cónicas, prismas poligonales). Seurat pinta las imágenes sobre esa forma y el modelo revisa una hoja de contacto de su propio resultado (seis vistas, diferencias de silueta frontal, errores de operaciones) durante las rondas que elijas; una ronda que vaciaría o destrozaría el modelo se deshace. Papeles de imagen: frente, lado, arriba, atrás, extra.
- *Solo desde una descripción*: un objeto, un personaje o una escena completa construidos desde el texto en una cuadrícula del tamaño elegido, con las mismas rondas de revisión.
- *Personajes por partes* (`method: parts`, automático cuando hay un sprite lateral): el sprite lateral es el perfil exacto del modelo y un plan dice qué es cada parte a lo ancho del cuerpo (`solid`, `fin`, `fan`, `pair`, `wings`, `decal`, `skip`). El modelo decide la estructura y lee los anchos en los sprites de frente y espalda; la geometría, los colores y la comparación son de Seurat. Valida el plan (el ojo debe estar en la cabeza, las alas y patas flotantes se acercan al cuerpo), pinta caras y espaldas desde los sprites de frente y espalda cuando muestran la misma postura, enseña al modelo qué píxeles ha captado cada parte y el resultado iluminado desde varios lados, y hace una comprobación independiente de que el nombre de cada parte corresponde a sus píxeles. El panel *Partes del personaje* cambia anchos, envergadura, apertura de alas, separación de patas o tipos de parte y reconstruye; *Pedir el cambio* (o `continue_from` con un texto) deja que el modelo siga desde el plan con una petición en palabras.
- *Sprites de juego como objetivo* (`target: sprite`, automático para pixel art pequeño): las imágenes de frente, espalda y lado son sprites del mundo del juego que se reconstruyen en 3D a razón de un voxel por píxel de sprite, con los colores del propio sprite. El sprite lateral es un perfil verdadero y cuenta doble; el programa lo recibe como un array con herramientas para tallar, rellenar y pintar desde él (`carve_side`, `extrude_side`, `paint_side`, `paint_view`). Tras cada ejecución Seurat proyecta el modelo desde la cámara de cada sprite y los muestra juntos con una diferencia de silueta (rojo falta, azul sobra) y puntuaciones de coincidencia. Se conserva la mejor versión.
- *Hojas de sprites y sprites animados*: se lee cada fotograma de un GIF (o WebP/PNG animado) y cada celda de una hoja (la cuadrícula se halla por los huecos o por la repetición, o se da como columnas × filas). Seurat adivina qué fotogramas muestran qué lado y puedes cambiarlo en *Fotogramas*; con el papel *sheet*, cada lado pasa a ser otra cámara del mismo modelo y los demás fotogramas cuentan como poses.
- *Calibración de cámara*: el agente averigua desde qué ángulo dibujó el juego cada sprite en lugar de los 45/0 grados fijos. *Calibrar con el modelo abierto* (`model_sprite_calibrate`) lo hace para cualquier modelo y puede guardar las alturas halladas como ajustes.
- *Escenas por partes* (`method: scene`, cuadrículas de hasta 256 por lado): el modelo planifica la escena como objetos (un suelo y después cada objeto con su tamaño y ubicaciones; los objetos repetidos se construyen una vez), cada objeto lo construye su propio bucle de agente en su propia cuadrícula, Seurat los coloca sobre lo que hay debajo y fusiona las paletas, y el modelo revisa la composición una vez. El plan se muestra antes como cajas.
- *Rondas en 3D*: se conservan todas las rondas de una generación, así que cualquiera se abre en el editor (*Ver en 3D*) o se convierte en un modelo propio (*Guardar como modelo*, `generate_round`).
- El pixel art ampliado por un factor entero se lee a su tamaño real; un color de fondo plano pasa a ser transparente.

**Renderizado iluminado y animación**
- Las vistas 3D de las hojas de revisión (`look`, `model_look`: `lit`, `lit_front_left`, `lit_back_left`, `lit_back_right`, `lit_front`, `lit_top`, cualquier yaw/pitch, `sheet_lit`) tienen luz real: una luz principal con sombras proyectadas, oclusión ambiental en los pliegues y un suelo que recoge la sombra, calculado sin GPU (ajuste `review_render`: iluminado o plano).
- Animaciones de giro: un plato giratorio del modelo con esa luz, como GIF o WebM (VP9 mediante ffmpeg), desde la pestaña Exportar o `model_spin`.

**Importar y exportar**
- Exporta `.vox` (MagicaVoxel, ida y vuelta), `.glb` (colores por vértice, malla voraz), `.stl` binario en milímetros (malla cerrada, z arriba, ajuste `voxel_mm`), PNG de cualquier vista y una tira de sprites isométricos desde cuatro lados.
- Importa ficheros `.vox` y esquemáticos `.schem` de Minecraft (WorldEdit).

**El lenguaje de operaciones.** Ejes: x a la derecha, y arriba, z hacia el espectador; `c` es un índice de paleta y `c: 0` borra. Operaciones: `voxel`, `voxels`, `box`, `line` (con `r`/`r_to`: una cápsula cónica), `prism` (un polígono con grosor: alas, aletas, orejas), `sphere`/`ellipsoid`, `cylinder`, `layer` (filas de píxeles en el plano frontal, lateral o superior), `inflate`, `paint` (recolorea la superficie vista desde un lado), `lathe`, `recolor`, `mirror`, `move`, `clear`. Detalles en `seurat_hoard/voxels.py`.

## Cómo arrancarlo

```
python -m venv .venv
.venv\Scripts\python -m pip install -r requirements.txt
.venv\Scripts\python -m seurat_hoard
```

Se abre en http://127.0.0.1:5250. El Hoard Hub lo arranca como al resto de la familia. Los datos (modelos, referencias, renders, ajustes, `mcp-token`) están en `data/`. Los ajustes cubren tamaño y colores de lectura, rondas de revisión, esfuerzo de razonamiento y tokens de pensamiento, pintado de las imágenes sobre la forma, calibración de cámara, vistas de revisión iluminadas o planas, límites de escena (acciones por objeto, máximo de objetos, tamaño por defecto), una URL fija de modelo compatible con OpenAI en lugar de la de la familia, y la escala de STL y GLB. Los giros en WebM necesitan ffmpeg en el PATH (o el paquete `imageio-ffmpeg`); el GIF no necesita nada.

## Conexión con Faustus y el Hub

El manifiesto del plugin es `faustus-plugin.json`. El puente MCP (`mcp_server.py`) expone el estudio a los asistentes, por grupos:
- Modelos: `model_list`, `model_create`, `model_get`, `model_duplicate`, `model_rename`, `model_resize`, `model_delete`, `model_ops` (editar con operaciones), `model_program` (ejecutar un programa de voxels, con `sprites`), `model_palette`, `model_dump`, `model_export`, `model_import_vox`, `model_versions`, `model_restore`, `version_note`.
- Mirar y comparar: `model_look` (devuelve una imagen, también desde las cámaras del mundo del juego e iluminada desde cualquier lado), `model_sprite_match`, `model_sprite_calibrate`, `model_spin`.
- Referencias: `reference_add`, `reference_list`, `reference_look`, `reference_grid`, `reference_frames`, `reference_sheet`, `reference_bytes`, `reference_delete`.
- Generación: `generate_start`, `generate_status`, `generate_list`, `generate_cancel`, `generate_round`, `model_from_parts`, `model_parts_adjust`, `model_rebuild`.
- Ajustes: `settings_get`, `settings_set`.

**Skills.** Seurat toma de la biblioteca de skills de la familia los procedimientos que necesita su trabajo (recursos 3D para la web, recursos de Blender, composición de imágenes, verificación de un resultado) y se los entrega a cada agente que se conecta: se nombran en lo que el agente lee al conectar, se leen con `skills_list` y `skill_read` y se ofrecen como prompts. El bloque `skills` de `faustus-plugin.json` dice cuáles; las copias viven en `data/skills/` y siguen a la biblioteca.

## Estructura del proyecto

| Ruta | Contenido |
|---|---|
| `seurat_hoard/` | la app: almacén, motor de voxels y lenguaje de operaciones, bucles de generación, partes, escenas, cámaras de sprite, hojas, renderizador iluminado, exportadores, API, interfaz web estática |
| `seurat_hoard/hoard_link/` | biblioteca de la familia incluida |
| `mcp_server.py` | puente MCP |
| `faustus-plugin.json` | manifiesto del plugin |
| `tests/` | batería de pruebas |
| `data/` | modelos, referencias, renders, ajustes (no versionados) |

## Pruebas

```
.venv\Scripts\python -m pytest
```

## Licencia

MIT. Los componentes de terceros se listan en `THIRD_PARTY_NOTICES.md`.
