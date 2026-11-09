// The voxel editor: the grid in memory, the 3D view (one merged mesh of the visible faces with baked ambient
// occlusion), picking by walking the grid along the mouse ray, the tools, symmetry, the layer slicer and undo/redo.
// Axes as in the server: x right, y up, z towards the viewer. A voxel (x, y, z) fills [x, x+1] x [y, y+1] x [z, z+1].

import * as THREE from 'three';
import { OrbitControls } from '/vendor/three/OrbitControls.js';

export const TOOLS = {
  add: { label: 'Añadir', key: 'A' },
  erase: { label: 'Borrar', key: 'B' },
  paint: { label: 'Pintar', key: 'P' },
  fill: { label: 'Relleno', key: 'R' },
  pick: { label: 'Cuentagotas', key: 'I' },
  box: { label: 'Caja', key: 'C' },
};

const HISTORY_LIMIT = 300;
const AO_LEVEL = [0.52, 0.68, 0.84, 1.0];

// The six faces: normal, and the four corners (offsets in the unit cube) in counter-clockwise order seen from outside.
const FACES = (() => {
  const out = [];
  for (let axis = 0; axis < 3; axis++) {
    for (const sign of [1, -1]) {
      const u = (axis + 1) % 3, v = (axis + 2) % 3;
      const n = [0, 0, 0]; n[axis] = sign;
      const base = sign > 0 ? 1 : 0;
      let corners = [[0, 0], [1, 0], [1, 1], [0, 1]].map(([a, b]) => {
        const c = [0, 0, 0]; c[axis] = base; c[u] = a; c[v] = b; return { c, a, b };
      });
      // u x v points along +axis; for the negative side reverse the order so the face looks outwards
      if (sign < 0) corners = corners.reverse();
      out.push({ axis, sign, u, v, n, corners });
    }
  }
  return out;
})();

function cssVar(name, fallback) {
  const v = getComputedStyle(document.documentElement).getPropertyValue(name).trim();
  return v || fallback;
}

export class Editor {
  constructor(container, hooks = {}) {
    this.container = container;
    this.hooks = hooks;               // onEdit(), onHover(info), onPick(c), onHistory(), onTool(tool)
    this.size = [16, 16, 16];
    this.grid = new Uint8Array(16 * 16 * 16);
    this.palette = ['#ffffff'];
    this.count = 0;
    this.tool = 'add';
    this.prevTool = 'add';
    this.color = 1;
    this.symX = false;
    this.symZ = false;
    this.limit = 16;                  // visible layers: y < limit
    this.boxDepth = 1;
    this.undoStack = [];
    this.redoStack = [];
    this.stroke = null;
    this.drag = null;
    this.modelId = null;
    this._renderQueued = false;
    this._meshDirty = true;
    this._hoverEvent = null;
    this._init();
  }

  // ---------------------------------------------------------------- set-up
  _init() {
    const renderer = new THREE.WebGLRenderer({ antialias: true, alpha: true, preserveDrawingBuffer: true });
    renderer.setPixelRatio(Math.min(window.devicePixelRatio || 1, 2));
    renderer.outputColorSpace = THREE.SRGBColorSpace;
    renderer.setClearColor(0x000000, 0);
    this.renderer = renderer;
    this.canvas = renderer.domElement;
    this.canvas.className = 'viewport-canvas';
    this.canvas.tabIndex = 0;
    this.container.prepend(this.canvas);

    const scene = new THREE.Scene();
    this.scene = scene;
    this.camera = new THREE.PerspectiveCamera(38, 1, 0.05, 4000);
    this.camera.position.set(40, 30, 50);

    scene.add(new THREE.HemisphereLight(0xfff6f0, 0x4a3e3a, 1.25));
    const key = new THREE.DirectionalLight(0xffffff, 2.1);
    key.position.set(0.42, 1, 0.62);
    scene.add(key);
    const fill = new THREE.DirectionalLight(0xffe6dc, 0.45);
    fill.position.set(-0.8, 0.25, -0.4);
    scene.add(fill);

    this.material = new THREE.MeshLambertMaterial({ vertexColors: true });
    this.mesh = new THREE.Mesh(new THREE.BufferGeometry(), this.material);
    scene.add(this.mesh);

    this.helpers = new THREE.Group();
    scene.add(this.helpers);

    // hover ghost (one cube, scaled for the box tool)
    const ghostGeo = new THREE.BoxGeometry(1, 1, 1);
    this.ghostFillMat = new THREE.MeshBasicMaterial({ color: 0xffffff, transparent: true, opacity: 0.35, depthWrite: false });
    this.ghostLineMat = new THREE.LineBasicMaterial({ color: 0xffffff, transparent: true, opacity: 0.95, depthTest: false });
    this.ghost = new THREE.Group();
    this.ghostFill = new THREE.Mesh(ghostGeo, this.ghostFillMat);
    this.ghostLine = new THREE.LineSegments(new THREE.EdgesGeometry(ghostGeo), this.ghostLineMat);
    this.ghostLine.renderOrder = 10;
    this.ghost.add(this.ghostFill, this.ghostLine);
    this.ghost.visible = false;
    scene.add(this.ghost);

    const controls = new OrbitControls(this.camera, this.canvas);
    controls.mouseButtons = { LEFT: null, MIDDLE: THREE.MOUSE.DOLLY, RIGHT: THREE.MOUSE.ROTATE };
    controls.touches = { ONE: THREE.TOUCH.ROTATE, TWO: THREE.TOUCH.DOLLY_PAN };
    controls.enableDamping = false;
    controls.zoomSpeed = 1.2;
    controls.rotateSpeed = 0.8;
    controls.screenSpacePanning = true;
    controls.addEventListener('change', () => this.requestRender());
    this.controls = controls;

    this.raycaster = new THREE.Raycaster();
    this.canvas.addEventListener('contextmenu', (e) => e.preventDefault());
    this.canvas.addEventListener('pointerdown', (e) => this._onDown(e));
    this.canvas.addEventListener('pointermove', (e) => this._onMove(e));
    this.canvas.addEventListener('pointerleave', () => { if (!this.drag && !this.stroke) { this.ghost.visible = false; this._hover(null); this.requestRender(); } });
    window.addEventListener('pointerup', (e) => this._onUp(e));
    window.addEventListener('pointercancel', () => this._cancelStroke());

    new ResizeObserver(() => this._resize()).observe(this.container);
    this._resize();
  }

  _resize() {
    const w = Math.max(1, this.container.clientWidth), h = Math.max(1, this.container.clientHeight);
    this.renderer.setSize(w, h, false);
    this.canvas.style.width = '100%';
    this.canvas.style.height = '100%';
    this.camera.aspect = w / h;
    this.camera.updateProjectionMatrix();
    this.requestRender();
  }

  requestRender() {
    if (this._renderQueued) return;
    this._renderQueued = true;
    requestAnimationFrame(() => {
      this._renderQueued = false;
      if (this._meshDirty) this._rebuildMesh();
      this.renderer.render(this.scene, this.camera);
    });
  }

  // ---------------------------------------------------------------- model in and out
  load(model, { keepCamera = false } = {}) {
    const [sx, sy, sz] = model.size;
    const sameShape = this.size[0] === sx && this.size[1] === sy && this.size[2] === sz;
    const sameModel = this.modelId === model.id;
    this.size = [sx, sy, sz];
    this.grid = new Uint8Array(sx * sy * sz);
    this.palette = (model.palette && model.palette.length ? model.palette : ['#ffffff']).slice(0, 255);
    let count = 0;
    for (const v of model.voxels || []) {
      const [x, y, z, c] = v;
      if (x < 0 || y < 0 || z < 0 || x >= sx || y >= sy || z >= sz || c <= 0) continue;
      const i = this.idx(x, y, z);
      if (!this.grid[i]) count++;
      this.grid[i] = c;
    }
    this.count = count;
    this.modelId = model.id;
    if (!sameModel || !sameShape) this.limit = sy;
    this.limit = Math.min(this.limit, sy);
    if (!sameModel) {
      // start with the colour the model uses most (or a mid-tone of the palette when it is empty)
      const used = new Uint32Array(256);
      for (let i = 0; i < this.grid.length; i++) used[this.grid[i]]++;
      used[0] = 0;
      let best = 0;
      for (let c = 1; c < 256; c++) if (used[c] > used[best]) best = c;
      this.color = best || Math.min(this.palette.length, 13);
    }
    if (this.color > this.palette.length) this.color = 1;
    this.undoStack = [];
    this.redoStack = [];
    this.stroke = null;
    this.drag = null;
    this._colorCache = null;
    this._buildHelpers();
    this._meshDirty = true;
    if (!(keepCamera && sameModel && sameShape)) this.setView(sameModel && sameShape ? null : 'iso');
    this.requestRender();
    this._emitHistory();
  }

  voxelList() {
    const [sx, sy, sz] = this.size;
    const out = [];
    const g = this.grid;
    for (let z = 0; z < sz; z++) for (let y = 0; y < sy; y++) {
      let i = sx * (y + sy * z);
      for (let x = 0; x < sx; x++, i++) if (g[i]) out.push([x, y, z, g[i]]);
    }
    return out;
  }

  idx(x, y, z) { return x + this.size[0] * (y + this.size[1] * z); }
  inside(x, y, z) { return x >= 0 && y >= 0 && z >= 0 && x < this.size[0] && y < this.size[1] && z < this.size[2]; }
  get(x, y, z) { return this.inside(x, y, z) ? this.grid[this.idx(x, y, z)] : 0; }
  // what the view shows: cells at or above the slicer are treated as empty
  visible(x, y, z) { return y < this.limit && this.inside(x, y, z) ? this.grid[this.idx(x, y, z)] : 0; }

  // ---------------------------------------------------------------- helpers: floor grid, bounds, slicer plane
  _buildHelpers() {
    for (const c of [...this.helpers.children]) { this.helpers.remove(c); c.geometry?.dispose(); c.material?.dispose(); }
    const [sx, sy, sz] = this.size;
    const accent = new THREE.Color(cssVar('--hoard-accent', '#ff9e7a'));

    const floor = new THREE.Mesh(new THREE.PlaneGeometry(sx, sz),
      new THREE.MeshBasicMaterial({ color: 0x2a221f, transparent: true, opacity: 0.85, depthWrite: false, side: THREE.DoubleSide }));
    floor.rotation.x = -Math.PI / 2;
    floor.position.set(sx / 2, -0.002, sz / 2);
    floor.renderOrder = -2;
    this.helpers.add(floor);

    const minor = [], major = [];
    const step = Math.max(sx, sz) > 96 ? 4 : 1;
    for (let x = 0; x <= sx; x += step) ((x % 8 === 0 || x === sx) ? major : minor).push(x, 0, 0, x, 0, sz);
    for (let z = 0; z <= sz; z += step) ((z % 8 === 0 || z === sz) ? major : minor).push(0, 0, z, sx, 0, z);
    const lines = (arr, color, opacity) => {
      const g = new THREE.BufferGeometry();
      g.setAttribute('position', new THREE.Float32BufferAttribute(arr, 3));
      const l = new THREE.LineSegments(g, new THREE.LineBasicMaterial({ color, transparent: true, opacity, depthWrite: false }));
      l.renderOrder = -1;
      return l;
    };
    this.helpers.add(lines(minor, 0x6a5d58, 0.35));
    this.helpers.add(lines(major, 0x8a7b75, 0.6));
    // the front edge (z = sz, towards the viewer) in the accent colour
    this.helpers.add(lines([0, 0.001, sz, sx, 0.001, sz], accent, 1));

    const box = new THREE.LineSegments(new THREE.EdgesGeometry(new THREE.BoxGeometry(sx, sy, sz)),
      new THREE.LineBasicMaterial({ color: 0x8a7b75, transparent: true, opacity: 0.22, depthWrite: false }));
    box.position.set(sx / 2, sy / 2, sz / 2);
    this.helpers.add(box);

    const slicer = new THREE.Group();
    const sp = new THREE.Mesh(new THREE.PlaneGeometry(sx, sz),
      new THREE.MeshBasicMaterial({ color: accent, transparent: true, opacity: 0.07, depthWrite: false, side: THREE.DoubleSide }));
    sp.rotation.x = -Math.PI / 2;
    sp.position.set(sx / 2, 0, sz / 2);
    const so = new THREE.LineSegments(new THREE.EdgesGeometry(new THREE.PlaneGeometry(sx, sz)),
      new THREE.LineBasicMaterial({ color: accent, transparent: true, opacity: 0.8 }));
    so.rotation.x = -Math.PI / 2;
    so.position.set(sx / 2, 0, sz / 2);
    slicer.add(sp, so);
    this.slicerPlane = slicer;
    this.helpers.add(slicer);
    this._placeSlicer();
  }

  _placeSlicer() {
    if (!this.slicerPlane) return;
    this.slicerPlane.visible = this.limit < this.size[1];
    this.slicerPlane.position.y = this.limit + 0.001;
  }

  // ---------------------------------------------------------------- the mesh
  _colors() {
    if (this._colorCache) return this._colorCache;
    const out = new Float32Array(256 * 3);
    const tmp = new THREE.Color();
    for (let i = 1; i < 256; i++) {
      const hex = this.palette[i - 1] || '#ff00ff';
      tmp.set(hex);   // converted to the linear working space
      out[i * 3] = tmp.r; out[i * 3 + 1] = tmp.g; out[i * 3 + 2] = tmp.b;
    }
    this._colorCache = out;
    return out;
  }

  _rebuildMesh() {
    this._meshDirty = false;
    const [sx, sy, sz] = this.size;
    const g = this.grid, lim = Math.min(this.limit, sy);
    const occ = (x, y, z) => (x >= 0 && y >= 0 && z >= 0 && x < sx && y < lim && z < sz && g[x + sx * (y + sy * z)] ? 1 : 0);
    // pass 1: count visible faces
    let faces = 0;
    const p = [0, 0, 0];
    for (let z = 0; z < sz; z++) for (let y = 0; y < lim; y++) for (let x = 0; x < sx; x++) {
      if (!g[x + sx * (y + sy * z)]) continue;
      if (!occ(x + 1, y, z)) faces++;
      if (!occ(x - 1, y, z)) faces++;
      if (!occ(x, y + 1, z)) faces++;
      if (!occ(x, y - 1, z)) faces++;
      if (!occ(x, y, z + 1)) faces++;
      if (!occ(x, y, z - 1)) faces++;
    }
    const pos = new Float32Array(faces * 12), nor = new Float32Array(faces * 12), col = new Float32Array(faces * 12);
    const index = new Uint32Array(faces * 6);
    const pal = this._colors();
    let f = 0;
    const ao = [0, 0, 0, 0];
    for (let z = 0; z < sz; z++) for (let y = 0; y < lim; y++) for (let x = 0; x < sx; x++) {
      const c = g[x + sx * (y + sy * z)];
      if (!c) continue;
      p[0] = x; p[1] = y; p[2] = z;
      for (const face of FACES) {
        const nx = x + face.n[0], ny = y + face.n[1], nz = z + face.n[2];
        if (occ(nx, ny, nz)) continue;
        const r = pal[c * 3], gg = pal[c * 3 + 1], b = pal[c * 3 + 2];
        for (let k = 0; k < 4; k++) {
          const { c: off, a, b: bb } = face.corners[k];
          const o = f * 12 + k * 3;
          pos[o] = x + off[0]; pos[o + 1] = y + off[1]; pos[o + 2] = z + off[2];
          nor[o] = face.n[0]; nor[o + 1] = face.n[1]; nor[o + 2] = face.n[2];
          // ambient occlusion from the three cells around this corner, in the layer outside the face
          const du = a ? 1 : -1, dv = bb ? 1 : -1;
          const q = [nx, ny, nz];
          const s1 = [...q]; s1[face.u] += du;
          const s2 = [...q]; s2[face.v] += dv;
          const cc = [...q]; cc[face.u] += du; cc[face.v] += dv;
          const o1 = occ(s1[0], s1[1], s1[2]), o2 = occ(s2[0], s2[1], s2[2]), o3 = occ(cc[0], cc[1], cc[2]);
          const level = o1 && o2 ? 0 : 3 - (o1 + o2 + o3);
          ao[k] = level;
          const m = AO_LEVEL[level];
          col[o] = r * m; col[o + 1] = gg * m; col[o + 2] = b * m;
        }
        const v0 = f * 4, ii = f * 6;
        if (ao[0] + ao[2] >= ao[1] + ao[3]) {
          index[ii] = v0; index[ii + 1] = v0 + 1; index[ii + 2] = v0 + 2;
          index[ii + 3] = v0; index[ii + 4] = v0 + 2; index[ii + 5] = v0 + 3;
        } else {
          index[ii] = v0 + 1; index[ii + 1] = v0 + 2; index[ii + 2] = v0 + 3;
          index[ii + 3] = v0 + 1; index[ii + 4] = v0 + 3; index[ii + 5] = v0;
        }
        f++;
      }
    }
    const geo = new THREE.BufferGeometry();
    geo.setAttribute('position', new THREE.BufferAttribute(pos, 3));
    geo.setAttribute('normal', new THREE.BufferAttribute(nor, 3));
    geo.setAttribute('color', new THREE.BufferAttribute(col, 3));
    geo.setIndex(new THREE.BufferAttribute(index, 1));
    geo.computeBoundingSphere();
    this.mesh.geometry.dispose();
    this.mesh.geometry = geo;
    this.faceCount = faces;
  }

  // ---------------------------------------------------------------- camera
  setView(name) {
    const [sx, sy, sz] = this.size;
    const target = new THREE.Vector3(sx / 2, sy * 0.42, sz / 2);
    const radius = Math.sqrt(sx * sx + sy * sy + sz * sz) / 2;
    const dist = Math.max(4, radius / Math.sin(THREE.MathUtils.degToRad(this.camera.fov / 2)) * 1.08);
    let dir;
    switch (name) {
      case 'front': dir = new THREE.Vector3(0, 0.0001, 1); target.y = sy / 2; break;
      case 'left': dir = new THREE.Vector3(-1, 0.0001, 0); target.y = sy / 2; break;
      case 'right': dir = new THREE.Vector3(1, 0.0001, 0); target.y = sy / 2; break;
      case 'back': dir = new THREE.Vector3(0, 0.0001, -1); target.y = sy / 2; break;
      case 'top': dir = new THREE.Vector3(0, 1, 0.0001); target.y = 0; break;
      case null: return;
      default: dir = new THREE.Vector3(0.75, 0.62, 1);
    }
    dir.normalize();
    this.camera.position.copy(target).addScaledVector(dir, dist);
    this.camera.near = Math.max(0.05, dist / 400);
    this.camera.far = dist * 20 + 100;
    this.camera.updateProjectionMatrix();
    this.controls.target.copy(target);
    this.controls.update();
    this.requestRender();
  }

  // ---------------------------------------------------------------- picking
  // Walks the grid cells along the mouse ray (Amanatides & Woo). Returns {hit:[x,y,z]|null, normal:[nx,ny,nz],
  // target:[x,y,z] (where an added voxel would go), floor:bool} or null.
  pick(clientX, clientY) {
    const rect = this.canvas.getBoundingClientRect();
    const ndc = new THREE.Vector2(((clientX - rect.left) / rect.width) * 2 - 1, -((clientY - rect.top) / rect.height) * 2 + 1);
    this.raycaster.setFromCamera(ndc, this.camera);
    return this._walk(this.raycaster.ray);
  }

  _walk(ray) {
    const o = [ray.origin.x, ray.origin.y, ray.origin.z], d = [ray.direction.x, ray.direction.y, ray.direction.z];
    const [sx, , sz] = this.size;
    const max = [sx, this.limit, sz];
    let tmin = -Infinity, tmax = Infinity, enterAxis = -1;
    for (let a = 0; a < 3; a++) {
      if (Math.abs(d[a]) < 1e-12) {
        if (o[a] < 0 || o[a] > max[a]) { tmin = Infinity; break; }
        continue;
      }
      let t1 = (0 - o[a]) / d[a], t2 = (max[a] - o[a]) / d[a];
      if (t1 > t2) [t1, t2] = [t2, t1];
      if (t1 > tmin) { tmin = t1; enterAxis = a; }
      if (t2 < tmax) tmax = t2;
    }
    if (this.limit > 0 && tmin <= tmax && tmax > 0) {
      const t0 = Math.max(tmin, 0);
      const cell = [0, 1, 2].map((a) => Math.min(max[a] - 1, Math.max(0, Math.floor(o[a] + d[a] * (t0 + 1e-7)))));
      const step = d.map((v) => (v > 0 ? 1 : v < 0 ? -1 : 0));
      const tDelta = d.map((v) => (v !== 0 ? Math.abs(1 / v) : Infinity));
      const tNext = [0, 1, 2].map((a) => {
        if (step[a] === 0) return Infinity;
        const bound = step[a] > 0 ? cell[a] + 1 : cell[a];
        return (bound - o[a]) / d[a];
      });
      let normal = [0, 0, 0];
      if (tmin > 0 && enterAxis >= 0) normal[enterAxis] = -step[enterAxis];
      else {   // the camera is inside the grid: face the main direction of the ray
        const a = Math.abs(d[0]) > Math.abs(d[1]) ? (Math.abs(d[0]) > Math.abs(d[2]) ? 0 : 2) : (Math.abs(d[1]) > Math.abs(d[2]) ? 1 : 2);
        normal[a] = -step[a] || 1;
      }
      for (let guard = 0; guard < 2000; guard++) {
        if (this.visible(cell[0], cell[1], cell[2])) {
          const target = [cell[0] + normal[0], cell[1] + normal[1], cell[2] + normal[2]];
          return { hit: cell.slice(), normal, target, floor: false };
        }
        let a = tNext[0] < tNext[1] ? (tNext[0] < tNext[2] ? 0 : 2) : (tNext[1] < tNext[2] ? 1 : 2);
        if (tNext[a] > tmax) break;
        cell[a] += step[a];
        tNext[a] += tDelta[a];
        normal = [0, 0, 0]; normal[a] = -step[a];
        if (cell[a] < 0 || cell[a] >= max[a]) break;
      }
    }
    // the floor (y = 0) inside the grid
    if (d[1] < 0 && o[1] > 0) {
      const t = -o[1] / d[1];
      const x = Math.floor(o[0] + d[0] * t), z = Math.floor(o[2] + d[2] * t);
      if (x >= 0 && z >= 0 && x < sx && z < sz) return { hit: null, normal: [0, 1, 0], target: [x, 0, z], floor: true };
    }
    return null;
  }

  // ---------------------------------------------------------------- editing primitives
  _set(x, y, z, c) {
    if (!this.inside(x, y, z) || y >= this.limit) return;
    const i = this.idx(x, y, z);
    const before = this.grid[i];
    if (before === c) return;
    if (!this.stroke) this.stroke = new Map();
    const rec = this.stroke.get(i);
    if (rec) rec[1] = c; else this.stroke.set(i, [before, c]);
    this.grid[i] = c;
    this.count += (before ? 0 : 1) - (c ? 0 : 1);
    this._meshDirty = true;
  }

  // the edit and its mirrors
  // `recolour`: the mirrors only repaint voxels that exist (painting must not create voxels on the other side)
  _setSym(x, y, z, c, recolour = false) {
    const [sx, , sz] = this.size;
    this._set(x, y, z, c);
    const put = (a, b, d) => { if (!recolour || this.get(a, b, d)) this._set(a, b, d, c); };
    if (this.symX) put(sx - 1 - x, y, z);
    if (this.symZ) put(x, y, sz - 1 - z);
    if (this.symX && this.symZ) put(sx - 1 - x, y, sz - 1 - z);
  }

  _commit(label) {
    const s = this.stroke;
    this.stroke = null;
    if (!s) return;
    const changes = [];
    for (const [i, [b, a]] of s) if (b !== a) changes.push(i, b, a);
    if (!changes.length) return;
    this._push({ kind: 'cells', changes: Int32Array.from(changes), label });
  }

  _push(entry) {
    this.undoStack.push(entry);
    if (this.undoStack.length > HISTORY_LIMIT) this.undoStack.shift();
    this.redoStack = [];
    this._edited();
  }

  _cancelStroke() {
    if (this.drag) { this.drag = null; this.ghost.visible = false; }
    if (this.stroke) this._commit('trazo');
    this.requestRender();
  }

  _edited() {
    this.requestRender();
    this._emitHistory();
    this.hooks.onEdit && this.hooks.onEdit();
  }

  _emitHistory() { this.hooks.onHistory && this.hooks.onHistory(this.undoStack.length, this.redoStack.length); }

  undo() { this._step(this.undoStack, this.redoStack, true); }
  redo() { this._step(this.redoStack, this.undoStack, false); }

  _step(from, to, back) {
    if (this.drag || this.stroke) return;
    const e = from.pop();
    if (!e) return;
    if (e.kind === 'cells') {
      const ch = e.changes;
      for (let k = 0; k < ch.length; k += 3) {
        const i = ch[k], v = back ? ch[k + 1] : ch[k + 2], prev = this.grid[i];
        this.grid[i] = v;
        this.count += (prev ? 0 : 1) - (v ? 0 : 1);
      }
      this._meshDirty = true;
    } else if (e.kind === 'palette') {
      this.palette = (back ? e.before : e.after).slice();
      this._colorCache = null;
      this._meshDirty = true;
      if (this.color > this.palette.length) this.color = this.palette.length;
      this.hooks.onPalette && this.hooks.onPalette();
    }
    to.push(e);
    this.requestRender();
    this._emitHistory();
    this.hooks.onEdit && this.hooks.onEdit();
  }

  // palette changes: preview without history while a colour input is being dragged, then commit with the old palette
  previewColor(index, hex) {
    if (index < 1 || index > this.palette.length) return;
    this.palette[index - 1] = hex;
    this._colorCache = null;
    this._meshDirty = true;
    this.requestRender();
  }

  commitPalette(before, label = 'paleta') {
    const after = this.palette.slice();
    if (before.length === after.length && before.every((c, i) => c === after[i])) return;
    this._colorCache = null;
    this._meshDirty = true;
    this._push({ kind: 'palette', before: before.slice(), after, label });
  }

  addColor(hex) {
    if (this.palette.length >= 255) return 0;
    const before = this.palette.slice();
    this.palette.push(hex);
    this.commitPalette(before, 'color nuevo');
    return this.palette.length;
  }

  // ---------------------------------------------------------------- state setters
  setTool(tool) {
    if (!TOOLS[tool] || tool === this.tool) return;
    if (tool === 'pick') this.prevTool = this.tool;
    this.tool = tool;
    this.ghost.visible = false;
    this.requestRender();
    this.hooks.onTool && this.hooks.onTool(tool);
  }

  setColor(c) {
    this.color = Math.max(1, Math.min(c, this.palette.length));
    this.hooks.onColor && this.hooks.onColor(this.color);
  }

  setLimit(n) {
    this.limit = Math.max(0, Math.min(n, this.size[1]));
    this.ghost.visible = false;
    this._meshDirty = true;
    this._placeSlicer();
    this.requestRender();
  }

  // ---------------------------------------------------------------- pointer
  _hover(info) { this.hooks.onHover && this.hooks.onHover(info); }

  _onDown(e) {
    if (e.button !== 0 || e.pointerType === 'touch') return;
    this.canvas.focus({ preventScroll: true });
    const p = this.pick(e.clientX, e.clientY);
    if (e.altKey || this.tool === 'pick') {
      if (p && p.hit) {
        this.setColor(this.get(...p.hit));
        this.hooks.onPick && this.hooks.onPick(this.color);
        if (this.tool === 'pick') this.setTool(this.prevTool && this.prevTool !== 'pick' ? this.prevTool : 'add');
      }
      return;
    }
    if (!p) return;
    e.preventDefault();
    const t = this.tool;
    if (t === 'box') {
      const erase = e.shiftKey;
      const start = erase ? (p.hit || p.target) : p.target;
      if (!start || !this.inside(...start)) return;
      const axis = Math.max(0, p.normal.findIndex((v) => v !== 0));
      const sign = p.normal[axis] || 1;
      // the plane of the clicked surface: between the hit voxel (or the floor) and the cell in front of it
      const surface = erase ? start[axis] + (sign > 0 ? 1 : 0) : start[axis] + (sign > 0 ? 0 : 1);
      this.drag = { start, end: start.slice(), axis, sign, erase, surface };
      this._showBox();
      return;
    }
    this.stroke = new Map();
    this.drag = { tool: t, plane: null, last: null };
    this._applyAt(p, e);
  }

  _onMove(e) {
    this._hoverEvent = e;
    if (this._moveQueued) return;
    this._moveQueued = true;
    requestAnimationFrame(() => {
      this._moveQueued = false;
      const ev = this._hoverEvent;
      if (!ev) return;
      if (this.drag && this.drag.start) { this._dragBox(ev); return; }
      const p = this.pick(ev.clientX, ev.clientY);
      if (this.drag && this.stroke) {
        if (ev.buttons & 1) this._applyAt(p, ev);
      }
      this._showGhost(p, ev);
    });
  }

  _onUp(e) {
    if (e.button !== 0) return;
    if (this.drag && this.drag.start) {
      const { start, end, erase } = this.drag;
      const [a, b] = this._boxRange(start, end);
      this.stroke = new Map();
      const c = erase ? 0 : this.color;
      for (let x = a[0]; x <= b[0]; x++) for (let y = a[1]; y <= b[1]; y++) for (let z = a[2]; z <= b[2]; z++) this._setSym(x, y, z, c);
      this.drag = null;
      this.ghost.visible = false;
      this._commit(erase ? 'caja (borrar)' : 'caja');
      this.requestRender();
      return;
    }
    if (this.stroke) {
      this.drag = null;
      this._commit(TOOLS[this.tool].label.toLowerCase());
      this.requestRender();
    }
  }

  _applyAt(p, e) {
    if (!p) return;
    const t = this.tool;
    const d = this.drag;
    if (t === 'add') {
      const tg = p.target;
      if (!tg || !this.inside(...tg) || tg[1] >= this.limit || this.get(...tg)) return;
      // a stroke stays on the layer of its first voxel so it does not climb towards the camera
      if (d.plane) { if (tg[d.plane.axis] !== d.plane.value) return; }
      else { const axis = p.normal.findIndex((v) => v !== 0); d.plane = { axis, value: tg[axis] }; }
      this._setSym(...tg, this.color);
    } else if (t === 'erase') {
      if (!p.hit) return;
      // like adding: a stroke keeps to the layer it started on instead of digging
      if (d.plane) { if (p.hit[d.plane.axis] !== d.plane.value) return; }
      else { const axis = p.normal.findIndex((v) => v !== 0); d.plane = { axis, value: p.hit[axis] }; }
      this._setSym(...p.hit, 0);
    } else if (t === 'paint') {
      if (!p.hit) return;
      this._setSym(...p.hit, this.color, true);
    } else if (t === 'fill') {
      if (!p.hit) return;
      this._fillPlane(p.hit, p.normal);
      this._commit('relleno');
      this.drag = null;
    }
    this.requestRender();
  }

  // recolour the connected patch of the same colour in the clicked layer (the plane of the clicked face)
  _fillPlane(start, normal) {
    const axis = Math.max(0, normal.findIndex((v) => v !== 0));
    const u = (axis + 1) % 3, v = (axis + 2) % 3;
    const from = this.get(...start);
    if (!from || from === this.color) return;
    const seen = new Set();
    const stack = [start];
    let n = 0;
    while (stack.length && n < 200000) {
      const c = stack.pop();
      const key = this.idx(...c);
      if (seen.has(key)) continue;
      seen.add(key);
      if (this.visible(...c) !== from) continue;
      // only cells whose clicked-side face is visible
      const out = c.slice(); out[axis] += normal[axis] || 1;
      if (this.visible(...out)) continue;
      this._setSym(c[0], c[1], c[2], this.color, true);
      n++;
      for (const [du, dv] of [[1, 0], [-1, 0], [0, 1], [0, -1]]) {
        const q = c.slice(); q[u] += du; q[v] += dv;
        if (this.inside(...q) && !seen.has(this.idx(...q))) stack.push(q);
      }
    }
  }

  // ---------------------------------------------------------------- ghosts
  _showGhost(p, e) {
    const t = this.tool;
    let cell = null, mode = 'add';
    if (p) {
      if (e && e.altKey) { cell = p.hit; mode = 'pick'; }
      else if (t === 'add' || (t === 'box' && !(e && e.shiftKey))) { cell = p.target; mode = 'add'; }
      else if (t === 'box') { cell = p.hit || p.target; mode = 'erase'; }
      else if (t === 'erase') { cell = p.hit; mode = 'erase'; }
      else if (t === 'paint' || t === 'fill') { cell = p.hit; mode = 'paint'; }
      else if (t === 'pick') { cell = p.hit; mode = 'pick'; }
    }
    if (cell && (!this.inside(...cell) || cell[1] >= this.limit)) cell = null;
    this._hover(p ? { cell: p.hit || p.target, floor: p.floor, color: p.hit ? this.get(...p.hit) : 0 } : null);
    if (!cell) { this.ghost.visible = false; this.requestRender(); return; }
    this._styleGhost(mode);
    this.ghost.scale.set(1.02, 1.02, 1.02);
    this.ghost.position.set(cell[0] + 0.5, cell[1] + 0.5, cell[2] + 0.5);
    this.ghost.visible = true;
    this.requestRender();
  }

  _styleGhost(mode) {
    const cur = this.palette[this.color - 1] || '#ffffff';
    if (mode === 'erase') {
      this.ghostFillMat.color.set('#ff5a4a'); this.ghostFillMat.opacity = 0.28;
      this.ghostLineMat.color.set('#ff7a6a');
    } else if (mode === 'pick') {
      this.ghostFillMat.opacity = 0; this.ghostLineMat.color.set('#ffffff');
    } else if (mode === 'paint') {
      this.ghostFillMat.color.set(cur); this.ghostFillMat.opacity = 0.75; this.ghostLineMat.color.set('#ffffff');
    } else {
      this.ghostFillMat.color.set(cur); this.ghostFillMat.opacity = 0.55; this.ghostLineMat.color.set('#ffffff');
    }
  }

  _boxRange(start, end) {
    const d = this.drag;
    const a = [0, 1, 2].map((k) => Math.min(start[k], end[k]));
    const b = [0, 1, 2].map((k) => Math.max(start[k], end[k]));
    const depth = Math.max(1, this.boxDepth | 0) - 1;
    if (depth > 0 && d) {
      const dir = d.erase ? -d.sign : d.sign;
      if (dir > 0) b[d.axis] += depth; else a[d.axis] -= depth;
    }
    for (let k = 0; k < 3; k++) {
      const hi = k === 1 ? this.limit - 1 : this.size[k] - 1;
      a[k] = Math.max(0, Math.min(a[k], hi)); b[k] = Math.max(0, Math.min(b[k], hi));
    }
    return [a, b];
  }

  _dragBox(e) {
    const d = this.drag;
    const rect = this.canvas.getBoundingClientRect();
    const ndc = new THREE.Vector2(((e.clientX - rect.left) / rect.width) * 2 - 1, -((e.clientY - rect.top) / rect.height) * 2 + 1);
    this.raycaster.setFromCamera(ndc, this.camera);
    const n = new THREE.Vector3(); n.setComponent(d.axis, 1);
    const plane = new THREE.Plane(n, -d.surface);
    const hit = new THREE.Vector3();
    if (this.raycaster.ray.intersectPlane(plane, hit)) {
      const end = [Math.floor(hit.x + 1e-6), Math.floor(hit.y + 1e-6), Math.floor(hit.z + 1e-6)];
      end[d.axis] = d.start[d.axis];
      for (let k = 0; k < 3; k++) end[k] = Math.max(0, Math.min(end[k], this.size[k] - 1));
      d.end = end;
      this._hover({ cell: end, floor: false, color: 0, box: this._boxRange(d.start, d.end) });
    }
    this._showBox();
  }

  _showBox() {
    const d = this.drag;
    const [a, b] = this._boxRange(d.start, d.end);
    this._styleGhost(d.erase ? 'erase' : 'add');
    const s = [b[0] - a[0] + 1, b[1] - a[1] + 1, b[2] - a[2] + 1];
    this.ghost.scale.set(s[0] + 0.02, s[1] + 0.02, s[2] + 0.02);
    this.ghost.position.set(a[0] + s[0] / 2, a[1] + s[1] / 2, a[2] + s[2] / 2);
    this.ghost.visible = true;
    this.requestRender();
  }

  snapshotPNG() { this.renderer.render(this.scene, this.camera); return this.canvas.toDataURL('image/png'); }
}
