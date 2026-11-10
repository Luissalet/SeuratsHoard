# Seurat's Hoard

Estudio de voxels de la familia Hoard. Dibuja modelos voxel en un editor 3D, convierte imágenes y sprites en modelos
voxel y expórtalos a MagicaVoxel, Blender/Godot/Unity, una impresora 3D o de vuelta a sprites.

- **Editor** (navegador o ventana Hoard, puerto 5250): añadir, borrar, pintar, rellenar, cuentagotas y caja, simetría X/Z,
  corte por capas para trabajar por dentro, deshacer/rehacer, edición de la paleta y guardado automático con versiones
  que puedes recuperar.
- **De imagen a voxels**
  - *Rápido* (sin modelo): *sprite 3D* (el sprite como pieza de diorama voxel, como los mods que pasan sprites a voxel:
    la imagen delante y detrás, un grosor redondeado suave y el contorno 2D fuera de los lados), *inflar* (el frente abombado como un cojín), *extruir* (una losa) o *silueta* (vistas de
    frente y lado talladas en un bloque, coloreadas desde cada vista).
  - *Como agente* (por defecto): tras el brief de diseño, DeepSeek trabaja en bucle como un harness agéntico: ejecuta su
    programa voxel, recibe el resultado o el error con la línea que falla, mira el modelo desde cualquier lado, corrige y
    refina (reescribiendo el programa o editando partes, y volviendo a su mejor versión si un cambio empeora), y dice cuándo ha terminado (acciones y minutos máximos en Ajustes). Cada ejecución que funciona se guarda.
  - *Personajes por partes* (`method: parts`; es lo que se usa solo cuando hay un sprite lateral): el sprite lateral es el
    perfil exacto del modelo (cada píxel es una columna de voxels) y un plan dice qué es cada parte a lo ancho del cuerpo:
    `solid` (sólido redondeado en el plano medio: cuerpo, cabeza, cola gruesa), `fin` (aleta fina), `fan` (abanico que se
    ensancha, redondeado hacia sus bordes; su extremo se pinta desde el sprite trasero), `pair` (par simétrico: patas, brazos, orejas), `wings` (dos
    láminas planas que salen del cuerpo, con envergadura y dirección de apertura), `decal` (marca solo en los flancos:
    ojos) y `skip` (lo que el sprite lateral enseña solo por estética, como el ala de detrás asomando: no se construye,
    la cercana reflejada ya es la otra; dentro de unas alas o de un par Seurat la busca sola, otra zona de los mismos
    colores tras una línea propia, y `far: keep` la conserva). El plan dice dónde está el ojo (`eye`):
    la forma que lo lleva es la cabeza, y Seurat rechaza un plan que lo ponga en un ala o en el extremo contrario al que
    mira el personaje, y avisa cuando una parte «no construir» tiene colores que ningún ala o par tiene (una marca del
    cuerpo que se perdería) o cuando la figura queda de pie sobre una sola pata central. DeepSeek decide esa estructura y lee los anchos en los sprites de frente y
    espalda; la geometría, los colores (el contorno negro desaparece y las líneas interiores se suavizan) y la
    comparación son de Seurat. Cuando el sprite de frente o el de espalda enseñan la misma pose que el modelo (personajes
    de pie, chibis), lo que mira a esa cámara se pinta con ese sprite: la cara de frente, el pelo o el dorso por detrás;
    los flancos siguen siendo el sprite lateral (`paint` en cada sólido: `auto`, `none`, `front`, `back`, `both`). Tras cada plan le enseña qué píxeles cogió cada parte, cada sprite junto al modelo desde
    su cámara (oblicua: las alturas se conservan y la profundidad desplaza; cuánto, se busca solo para cada sprite) y el
    modelo con luz desde varios lados, y el modelo ajusta (`adjust`) o rehace el plan. Las formas del plan son
    aproximadas (rectángulos o polígonos en columnas y filas del sprite, con filtro de colores): se ajustan a las manchas
    de color y a las líneas del sprite. El plan se guarda con el modelo: el panel *Partes del personaje* cambia anchos,
    envergadura, apertura de las alas, separación de las patas o el tipo de cada parte y reconstruye, y también deja
    editar el plan entero.
  - *Sprites del juego como objetivo* (`target: sprite`, automático con pixel art pequeño): las imágenes frontal, trasera
    y lateral son sprites de overworld que hay que reconstruir en 3D como hacen los mods voxel (un voxel por píxel más o
    menos, los colores y el diseño del sprite). El sprite lateral es un perfil puro (sin ángulo de cámara): la vista más
    importante, cuenta doble y el programa lo recibe como matriz en coordenadas de la cuadrícula (`SIDE`, columna = z,
    fila = y) con herramientas para recortar el modelo a él (`carve_side`), rellenarlo (`extrude_side`) y pintar desde él
    (`paint_side`, y `paint_view` para cualquier sprite), además de la paleta de los sprites y sus mapas de píxeles en el
    prompt. Frente y espalda están dibujados desde unos 45 grados (su eje horizontal es exactamente x; en Ajustes). Tras
    cada ejecución Seurat proyecta el modelo desde la cámara de cada sprite a un voxel por píxel y le enseña cada sprite
    junto a él con la diferencia de silueta (rojo falta, azul sobra), la coincidencia y cómo le va de tamaño; el panel del
    trabajo las enseña por ronda. El agente sabe cuándo una ejecución es peor que su mejor versión y se guarda la mejor,
    no la última. `continue_from` continúa una generación desde el programa y el brief de un modelo (restaura antes el
    paso que quieras).
  - *Hojas de sprites y sprites animados*: se lee cada fotograma de un GIF (o WebP/PNG animado) y cada celda de una hoja
    (la rejilla sale de los huecos entre sprites, de su repetición cuando se tocan, o de columnas x filas que le des).
    Seurat adivina qué fotogramas enseñan cada lado (en las hojas de caminar, filas o columnas son direcciones y el resto
    pasos; una animación que gira se corta en sus tramos; los grupos asimétricos son laterales y el que tiene cara es el
    frente) y lo puedes cambiar en *Fotogramas*. Con el papel *Hoja o animación* cada lado es otra cámara del mismo
    modelo y sus otros fotogramas cuentan como poses: el fotograma de pie es la vista principal y los pasos se reparten la
    mitad de la nota de esa vista.
  - *Calibrar la cámara*: tras las primeras ejecuciones (y cada tres) el agente busca desde qué ángulo dibujó el juego
    cada sprite comparando la silueta del modelo a varias alturas y giros (entre 20 y 70 grados y 10 a cada lado mientras
    el modelo se construye, y solo si el sprite encaja claramente mejor), en lugar de los 45/0 grados fijos; un lado con
    cámara desconocida (papel *Vista libre*) busca su giro en toda la vuelta. *Calibrar con el modelo abierto*
    (`model_sprite_calibrate`) lo hace con cualquier modelo y una búsqueda amplia, enseña cada sprite junto al modelo
    desde su cámara y puede guardar en Ajustes las alturas encontradas.
  - *Escenas por partes* (`method: scene`, cuadrículas de hasta 256 de lado): el modelo planifica la escena en objetos
    (un suelo y cada objeto con su tamaño y sus colocaciones; los repetidos se construyen una vez y se colocan varias,
    girados), cada objeto lo construye su propio bucle de agente en su cuadrícula (hasta 128), Seurat los coloca y los
    asienta sobre lo que tienen debajo, junta sus paletas y el modelo revisa una vez la composición con una vista con luz
    y otra desde arriba. El plan se enseña antes como cajas; los programas de cada objeto se guardan con la escena.
  - *Rondas en 3D*: se guardan los voxels de cada ronda de una generación, así que cualquier ronda intermedia se abre en
    el editor (*Ver en 3D*, una vista previa que no se guarda) o se convierte en un modelo propio (*Guardar como modelo*,
    `generate_round`).
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
- **Render con luz**: las vistas 3D de las hojas que revisa el modelo, `look` y `model_look` (`lit`, `lit_front_left`,
  `lit_back_left`, `lit_back_right`, `lit_front`, `lit_top`, cualquier giro y altura, `sheet_lit`) tienen luz de verdad:
  una luz principal con sombras arrojadas trazadas por la cuadrícula, oclusión ambiental en los pliegues y un suelo que
  recoge la sombra, calculado con numpy sin GPU (ajuste `review_render`: con luz o plano).
- **Animaciones de giro**: el modelo girando con esa luz, en GIF o WebM (VP9 con ffmpeg), desde Exportar o con
  `model_spin`.
- **Exportar**: `.vox` (MagicaVoxel 150, ida y vuelta), `.glb` (colores por vértice, mallado agrupado), `.stl` binario
  en milímetros (malla cerrada, z arriba, ajuste `voxel_mm`), PNG de cualquier vista y una tira de sprites isométricos
  desde cuatro lados. Se pueden importar `.vox` y schematics `.schem` de Minecraft (WorldEdit).
- **Asistentes** (MCP, `mcp_server.py`, 36 herramientas): listar, crear y editar modelos con operaciones (`model_ops`),
  verlos (`model_look` devuelve una imagen, también desde las cámaras de overworld del juego y con luz desde cualquier
  lado), comparar un modelo con sprites del juego desde su cámara (`model_sprite_match`, también con hojas y poses),
  calibrar esas cámaras (`model_sprite_calibrate`), leer los fotogramas de hojas y animaciones (`reference_frames`),
  ejecutar programas voxel con las herramientas de sprites (`model_program` con `sprites`), añadir referencias,
  construir o generar desde imágenes o como escena por partes, seguir y cancelar generaciones, abrir o guardar cualquier
  ronda (`generate_round`), exportar, giros (`model_spin`), versiones y ajustes.

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
de razonamiento, pintar las imágenes sobre la forma, calibrar la cámara, vistas de revisión con luz o planas, escenas
(acciones por objeto, objetos como máximo, tamaño por defecto), una URL fija compatible con OpenAI en lugar del modelo de
la familia, y escala de STL y GLB. Los giros en WebM necesitan ffmpeg en el PATH (o el paquete `imageio-ffmpeg`); el GIF
no necesita nada.

Tests: `.venv\Scripts\python -m pytest`.
