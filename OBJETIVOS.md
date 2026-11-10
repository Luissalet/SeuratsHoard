# Objetivos de Seurat's Hoard

- Poses de verdad: con los fotogramas de caminar como guía, articular el modelo por partes (piernas, brazos, cola) para
  sacar copias en cada pose y exportar la animación (tira de sprites, GIF o `.vox` con varios modelos).
- Escenas editables por objetos: mover, girar o rehacer un objeto de una escena ya generada desde su plan y sus programas,
  sin volver a generar el resto.
- Partes: usar la vista de frente como perfil cuando no hay sprite lateral (personajes dibujados solo de frente).
- Partes: proponer solas las alas, las patas y los abanicos sin modelo de lenguaje, comparando hipótesis contra los
  sprites de frente y espalda.
- Partes: alas y aletas curvadas o plegadas (hoy son láminas planas) y extremidades lejanas con pose propia.
- Partes: que la generación con DeepSeek sitúe cada parte en los tres sprites (`front`, `back`) y juzgue mirando el
  modelo, no cifras de solape. El motor ya construye así los planes escritos a mano; con el prompt probado el 10-11
  DeepSeek leyó bien los tres sprites, pero el pájaro de prueba salió peor que con el perfil y los anchos: el cuerpo toma los hombros,
  la cola en abanico sale como una tabla y, con las alas tendidas desde su dibujo, el cuerpo que queda debajo es pequeño.
- Partes: profundidad propia para lo que el sprite de frente dibuja plano (una máscara, unas orejas): hoy la parte
  entera toma ese ancho en todo su fondo y las cabezas salen cuadradas.
- Partes: construir al doble de tamaño con los bordes redondeados (el plan y los sprites escalados): probado a mano da
  un acabado de escultura; falta integrarlo como opción de tamaño.
- Partes: sprites laterales dibujados desde arriba (cámara alta) como perfil: hoy se toman tal cual y el modelo sale
  algo más alto y grueso.
- Partes: pintar de frente y de espalda también cuando ese sprite está en otra pose que el lateral, deformando su dibujo
  parte a parte.
