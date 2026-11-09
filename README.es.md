# Seurat's Hoard

Estudio de voxels de la familia Hoard. Dibuja modelos voxel en un editor 3D, convierte imágenes y sprites en modelos
voxel y expórtalos a MagicaVoxel, Blender/Godot/Unity, una impresora 3D o de vuelta a sprites.

- **Editor** (navegador o ventana Hoard, puerto 5250): añadir, borrar, pintar, rellenar, cuentagotas y caja, simetría X/Z,
  corte por capas para trabajar por dentro, deshacer/rehacer, edición de la paleta y guardado automático con versiones
  que puedes recuperar.
- **De imagen a voxels**
  - *Rápido* (sin modelo): *inflar* (el frente abombado como un cojín), *extruir* (una losa) o *silueta* (vistas de
    frente y lado talladas en un bloque, coloreadas desde cada vista).
  - *Como agente* (por defecto): tras el brief de diseño, DeepSeek trabaja en bucle como un harness agéntico: ejecuta su
    programa voxel, recibe el resultado o el error con la línea que falla, mira el modelo desde cualquier lado, corrige y
    refina, y dice cuándo ha terminado (acciones y minutos máximos en Ajustes). Cada ejecución que funciona se guarda.
  - *Con el modelo de lenguaje, como programa por rondas fijas*: DeepSeek V4.1 Flash escribe primero un brief de diseño con
    lo que sabe del tema y tu descripción (partes, proporciones, posición en 3D, paleta; las imágenes son solo pistas,
    nunca se calcan) y después un programa en Python que
    construye los voxels (elipsoides fundidos, cápsulas que se estrechan, bucles de plumas, polígonos, simetría, máscaras
    de numpy); Seurat lo ejecuta en un proceso aparte con tiempo límite y sin ficheros ni red, enseña el resultado y le
    pide un programa mejor ronda tras ronda. El último programa que funcionó se guarda con el modelo. Cuadrículas de
    hasta 128 de lado; escenas solo con descripción o criaturas desde imágenes (el artwork oficial funciona mucho mejor
    que los sprites pequeños de juego).
  - *Con el modelo de lenguaje, como operaciones* (`method: ops`): ve las
    imágenes de referencia y sus cuadrículas de píxeles, razona a fondo (esfuerzo `max` por defecto, varios minutos por
    ronda) y esculpe la forma por partes (elipsoides, cápsulas que se estrechan para cuellos y colas, prismas de polígono
    para alas y aletas). Seurat pinta las imágenes sobre esa forma (frente y espalda; también la lateral si lo activas) y el modelo revisa una
    hoja con seis vistas de su propio resultado (diferencias de silueta frontal y errores de sus operaciones) durante las
    rondas que elijas. Una ronda que vaciaría o destrozaría el modelo se deshace. Cada ronda se ve con su render,
    sus notas y sus errores; las respuestas se guardan como texto junto a los renders. Papel de cada imagen: frente,
    lado, arriba, detrás, extra.
  - *Solo con una descripción* (sin imágenes): el modelo construye un objeto, un personaje o una escena entera a partir
    del texto, en una cuadrícula del tamaño elegido y con las mismas rondas de revisión.
  - El pixel art ampliado por un factor entero se lee a su tamaño real; un fondo de un solo color pasa a transparente.
- **Exportar**: `.vox` (MagicaVoxel 150, ida y vuelta), `.glb` (colores por vértice, mallado agrupado), `.stl` binario
  en milímetros (malla cerrada, z arriba, ajuste `voxel_mm`), PNG de cualquier vista y una tira de sprites isométricos
  desde cuatro lados. Los `.vox` se pueden importar.
- **Asistentes** (MCP, `mcp_server.py`, 28 herramientas): listar, crear y editar modelos con operaciones (`model_ops`),
  verlos (`model_look` devuelve una imagen), añadir referencias, construir o generar desde imágenes, seguir y cancelar
  generaciones, exportar, versiones y ajustes.

## El lenguaje de operaciones

Ejes: x a la derecha, y arriba, z hacia quien mira. `c` es un índice de la paleta; `c: 0` borra.
`voxel`, `voxels`, `box`, `line` (con `r`/`r_to`: una cápsula que se estrecha), `prism` (un polígono en un plano con
grosor: alas, aletas, orejas), `sphere`/`ellipsoid`, `cylinder`, `layer` (filas de píxeles en el plano frontal,
lateral o superior), `inflate` (filas de píxeles abombadas en volumen), `paint` (repinta la superficie vista desde un
lado, píxel a píxel), `lathe` (sólido de revolución), `recolor`, `mirror`, `move`, `clear`. Detalle en
`seurat_hoard/voxels.py`.

## Arrancar

```
python -m venv .venv
.venv\Scripts\python -m pip install -r requirements.txt
.venv\Scripts\python -m seurat_hoard
```

Se abre en http://127.0.0.1:5250. El Hoard Hub lo arranca como al resto de la familia. Datos (modelos, referencias,
renders, ajustes, `mcp-token`) en `data/`. Ajustes: tamaño y colores de lectura, rondas de revisión, esfuerzo y tokens
de razonamiento, pintar las imágenes sobre la forma, una URL fija compatible con OpenAI en lugar del modelo de la familia, y escala de STL y GLB.

Tests: `.venv\Scripts\python -m pytest`.
