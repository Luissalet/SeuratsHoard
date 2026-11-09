# Objetivos de Seurat's Hoard

- Calibrar la cámara de cada hoja de sprites: buscar sola los grados que mejor encajan (con un modelo de referencia o
  ajustando a la vez cámara y modelo) en lugar de los 45/20 fijos.
- Sprites de cuatro direcciones y de caminar: usar los fotogramas de andar como poses extra del mismo modelo.
- Escenas por partes: que el modelo planifique una escena grande en objetos y construya cada uno en su propio programa,
  componiendo el resultado (más allá de 128 de lado).
- Render con luz de verdad para las revisiones (sombras y oclusión como el editor three.js) en lugar del isométrico por
  software.
- Sprites animados: leer todos los fotogramas de un GIF o de una hoja de sprites y usar las poses como vistas extra.
- Vista de rondas en 3D: abrir en el editor cualquier ronda intermedia de una generación, no solo el resultado.
- Exportar animaciones de giro (GIF/WebM) desde los renders isométricos.
