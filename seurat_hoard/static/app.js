// Seurat's Hoard: the top bar (models), the editor and its toolbar, the palette, autosave and the status bar.
// The side panels (references, generation, export, settings) live in panels.js.

import { Editor, TOOLS } from './editor.js';
import { $, el, api, toast, fail, modal, confirmBox, askText, nf, fmtDate, store, isTyping } from './ui.js';
import { initPanels } from './panels.js';

const ICONS = {
  add: '<svg viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="1.8" stroke-linejoin="round"><path d="M12 3 4 7.5v9L12 21l8-4.5v-9z"/><path d="M12 9v6M9 12h6"/></svg>',
  erase: '<svg viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="1.8" stroke-linejoin="round"><path d="M12 3 4 7.5v9L12 21l8-4.5v-9z"/><path d="M9 12h6"/></svg>',
  paint: '<svg viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="1.8" stroke-linecap="round" stroke-linejoin="round"><path d="M19 3 10 12"/><path d="M10 12c-2.5-.5-5 1-5 4 0 1.5-1 2.5-2 3 3 1 7 .5 8.5-2 .8-1.4.5-3.3-1.5-5z"/></svg>',
  fill: '<svg viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="1.8" stroke-linejoin="round"><path d="m4 11 7-7 8 8-7 7z"/><path d="M4 11h15"/><path d="M20 15s1.5 2 1.5 3a1.5 1.5 0 0 1-3 0c0-1 1.5-3 1.5-3z"/></svg>',
  pick: '<svg viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="1.8" stroke-linecap="round" stroke-linejoin="round"><path d="m14 6 4 4"/><path d="M16.5 3.5a2.1 2.1 0 0 1 3 3L17 9l-3-3z"/><path d="M15 8 6 17l-1 3 3-1 9-9"/></svg>',
  box: '<svg viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="1.8" stroke-linejoin="round"><path d="M4 8h12v12H4z"/><path d="m4 8 4-4h12l-4 4M16 20l4-4V4"/></svg>',
  undo: '<svg viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="1.8" stroke-linecap="round" stroke-linejoin="round"><path d="M9 14 4 9l5-5"/><path d="M4 9h10a6 6 0 0 1 0 12h-3"/></svg>',
  redo: '<svg viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="1.8" stroke-linecap="round" stroke-linejoin="round"><path d="m15 14 5-5-5-5"/><path d="M20 9H10a6 6 0 0 0 0 12h3"/></svg>',
};

const TOOL_HELP = {
  add: 'Clic en una cara o en el suelo. Arrastra para seguir en la misma capa.',
  erase: 'Clic en un voxel para quitarlo. Arrastra para borrar en la misma capa.',
  paint: 'Clic o arrastra sobre los voxels para pintarlos.',
  fill: 'Clic en una cara: pinta la zona del mismo color en esa capa.',
  pick: 'Clic en un voxel para coger su color.',
  box: 'Arrastra sobre una cara o el suelo. Mayús: borrar.',
};

const state = {
  models: [],
  model: null,          // {id, name, size, source, ...} of the open model (voxels live in the editor)
  settings: null,
  status: null,
  editSeq: 0,
  savedSeq: 0,
  saving: null,
  saveTimer: null,
  saveError: false,
};

const bus = new EventTarget();
const emit = (type, detail) => bus.dispatchEvent(new CustomEvent(type, { detail }));

// ---------------------------------------------------------------- editor
const editor = new Editor($('viewport'), {
  onEdit: () => { markDirty(); updateStatus(); renderPaletteUsage(); },
  onHover: (info) => {
    if (!info) { $('st-cursor').textContent = 'Cursor: —'; return; }
    if (info.box) {
      const [a, b] = info.box;
      $('st-cursor').textContent = `Caja: ${b[0] - a[0] + 1} × ${b[1] - a[1] + 1} × ${b[2] - a[2] + 1}`;
      return;
    }
    const [x, y, z] = info.cell;
    $('st-cursor').textContent = info.floor ? `Suelo: x ${x} · z ${z}` : `Cursor: x ${x} · y ${y} · z ${z}${info.color ? ` · color ${info.color}` : ''}`;
  },
  onPick: (c) => toast(`Color ${c}: ${editor.palette[c - 1]}`),
  onHistory: (u, r) => { $('tb-undo').disabled = !u; $('tb-redo').disabled = !r; },
  onTool: () => renderTools(),
  onColor: () => renderCurrentColor(),
  onPalette: () => renderPalette(),
});

// ---------------------------------------------------------------- toolbar
function buildToolbar() {
  const tb = $('toolbar');
  for (const [id, t] of Object.entries(TOOLS)) {
    tb.append(el('button', { class: 'tool', id: `tool-${id}`, 'data-tool': id, title: `${t.label} (${t.key})`, 'aria-label': t.label,
      html: `${ICONS[id]}<span class="k">${t.key}</span>`, onclick: () => editor.setTool(id) }));
  }
  tb.append(el('div', { class: 'tb-sep' }));
  tb.append(el('span', { class: 'tb-label', text: 'Simetría' }));
  tb.append(el('button', { class: 'tool text', id: 'tb-symx', title: 'Simetría en X: refleja izquierda y derecha (X)', html: 'X<span class="k">X</span>',
    onclick: () => toggleSym('x') }));
  tb.append(el('button', { class: 'tool text', id: 'tb-symz', title: 'Simetría en Z: refleja delante y detrás (Z)', html: 'Z<span class="k">Z</span>',
    onclick: () => toggleSym('z') }));
  tb.append(el('div', { class: 'tb-sep' }));
  tb.append(el('button', { class: 'tool', id: 'tb-undo', title: 'Deshacer (Ctrl+Z)', 'aria-label': 'Deshacer', html: ICONS.undo, disabled: true, onclick: () => editor.undo() }));
  tb.append(el('button', { class: 'tool', id: 'tb-redo', title: 'Rehacer (Ctrl+Y)', 'aria-label': 'Rehacer', html: ICONS.redo, disabled: true, onclick: () => editor.redo() }));
  renderTools();
}

function toggleSym(axis) {
  if (axis === 'x') editor.symX = !editor.symX; else editor.symZ = !editor.symZ;
  renderTools();
}

function renderTools() {
  for (const b of document.querySelectorAll('.tool[data-tool]')) b.classList.toggle('active', b.dataset.tool === editor.tool);
  $('tb-symx').classList.toggle('active', editor.symX);
  $('tb-symz').classList.toggle('active', editor.symZ);
  const t = TOOLS[editor.tool];
  $('st-tool').textContent = `Herramienta: ${t.label}`;
  const sym = [editor.symX && 'X', editor.symZ && 'Z'].filter(Boolean);
  $('st-sym').textContent = sym.length ? `Simetría ${sym.join(' + ')}` : '';
  const opts = $('tool-options');
  opts.replaceChildren(el('div', { class: 'opt-pill' }, el('b', { text: t.label }), el('span', { text: TOOL_HELP[editor.tool] })));
  if (editor.tool === 'box') {
    const input = el('input', { type: 'number', min: '1', max: '256', value: String(editor.boxDepth), title: 'Grosor de la caja, hacia fuera de la cara' });
    input.addEventListener('input', () => { editor.boxDepth = Math.max(1, Math.min(256, parseInt(input.value, 10) || 1)); });
    opts.append(el('label', { class: 'opt-pill' }, 'Grosor', input));
  }
}

// ---------------------------------------------------------------- palette
let paletteBefore = null;
let editingIndex = 0;

function renderPalette() {
  const box = $('swatches');
  box.replaceChildren();
  editor.palette.forEach((hex, i) => {
    const b = el('button', { class: 'sw', 'data-i': String(i + 1), title: `${i + 1}: ${hex}`, style: `background:${hex}`, role: 'option' });
    b.addEventListener('click', () => editor.setColor(i + 1));
    b.addEventListener('dblclick', () => { editor.setColor(i + 1); editColor(i + 1); });
    box.append(b);
  });
  $('btn-add-color').disabled = editor.palette.length >= 255;
  renderCurrentColor();
  renderPaletteUsage();
}

function renderCurrentColor() {
  const c = editor.color, hex = editor.palette[c - 1] || '#000000';
  $('cc-swatch').style.background = hex;
  $('cc-index').textContent = String(c);
  $('cc-hex').textContent = hex;
  for (const b of $('swatches').children) b.classList.toggle('active', Number(b.dataset.i) === c);
  const act = $('swatches').querySelector('.sw.active');
  if (act) act.scrollIntoView({ block: 'nearest' });
}

let usageTimer = null;
function renderPaletteUsage() {
  clearTimeout(usageTimer);
  usageTimer = setTimeout(() => {
    const used = new Uint8Array(256);
    const g = editor.grid;
    for (let i = 0; i < g.length; i++) used[g[i]] = 1;
    for (const b of $('swatches').children) b.classList.toggle('used', !!used[Number(b.dataset.i)]);
  }, 200);
}

function editColor(index) {
  editingIndex = index;
  paletteBefore = editor.palette.slice();
  const input = $('color-input');
  input.value = editor.palette[index - 1];
  input.click();
}

$('color-input').addEventListener('input', (e) => {
  if (!editingIndex) return;
  editor.previewColor(editingIndex, e.target.value);
  const sw = $('swatches').querySelector(`.sw[data-i="${editingIndex}"]`);
  if (sw) { sw.style.background = e.target.value; sw.title = `${editingIndex}: ${e.target.value}`; }
  renderCurrentColor();
});
$('color-input').addEventListener('change', () => {
  if (!editingIndex || !paletteBefore) return;
  editor.commitPalette(paletteBefore, 'color');
  paletteBefore = null;
  renderPalette();
  $('color-input').blur();
});
$('current-color').addEventListener('click', () => editColor(editor.color));
$('btn-add-color').addEventListener('click', () => {
  const base = editor.palette[editor.color - 1] || '#888888';
  const n = editor.addColor(base);
  if (!n) { toast('La paleta ya tiene 255 colores.', 'err'); return; }
  editor.setColor(n);
  renderPalette();
  editColor(n);
});

// ---------------------------------------------------------------- models
function setSaveState(s, text) {
  $('save-state').dataset.state = s;
  $('save-text').textContent = text;
}

function markDirty() {
  if (!state.model) return;
  state.editSeq++;
  setSaveState('pending', 'Sin guardar');
  clearTimeout(state.saveTimer);
  state.saveTimer = setTimeout(() => saveNow(), 1500);
}

function payload() {
  return { voxels: editor.voxelList(), palette: editor.palette.slice(), size: editor.size.slice(), name: state.model.name, version_note: 'editor' };
}

async function saveNow() {
  clearTimeout(state.saveTimer);
  if (state.saving) await state.saving.catch(() => {});
  if (!state.model || state.editSeq === state.savedSeq) return;
  const seq = state.editSeq, id = state.model.id;
  setSaveState('saving', 'Guardando…');
  state.saving = api(`/api/models/${id}`, { method: 'PUT', body: payload() });
  try {
    const res = await state.saving;
    state.savedSeq = Math.max(state.savedSeq, seq);
    state.saveError = false;
    if (state.model && state.model.id === id) Object.assign(state.model, { updated: res.updated, count: res.count });
    const m = state.models.find((x) => x.id === id);
    if (m) { m.count = res.count; m.updated = res.updated; }
    emit('saved', { id });
  } catch (err) {
    state.saveError = true;
    setSaveState('error', 'Error al guardar');
    fail(err);
    state.saving = null;
    clearTimeout(state.saveTimer);
    state.saveTimer = setTimeout(() => saveNow(), 5000);
    return;
  }
  state.saving = null;
  if (state.editSeq !== state.savedSeq) { setSaveState('pending', 'Sin guardar'); state.saveTimer = setTimeout(() => saveNow(), 1500); }
  else setSaveState('saved', 'Guardado');
}

async function flushSave() {
  if (state.model && state.editSeq !== state.savedSeq) await saveNow();
  else if (state.saving) await state.saving.catch(() => {});
}

async function refreshModels() {
  const res = await api('/api/models');
  state.models = res.models || [];
  renderModelSelect();
  return state.models;
}

function renderModelSelect() {
  const sel = $('model-select');
  sel.replaceChildren(...state.models.map((m) => el('option', { value: m.id, text: `${m.name}  ·  ${(m.size || []).join('×')}` })));
  if (state.model) sel.value = state.model.id;
}

async function openModel(id, { keepCamera = false, quiet = false } = {}) {
  if (state.model && state.model.id !== id) await flushSave();
  const m = await api(`/api/models/${id}`);
  state.model = { id: m.id, name: m.name, size: m.size, source: m.source, count: m.count, updated: m.updated };
  state.editSeq = state.savedSeq = 0;
  clearTimeout(state.saveTimer);
  editor.load(m, { keepCamera });
  store('model', m.id);
  if (!state.models.find((x) => x.id === m.id)) await refreshModels();
  renderModelSelect();
  renderPalette();
  setupSlicer();
  updateStatus();
  setSaveState('saved', 'Guardado');
  emit('model', state.model);
  if (!quiet) document.title = `${m.name} · Seurat's Hoard`;
  return m;
}

async function createModel(name, size) {
  const m = await api('/api/models', { method: 'POST', body: { name, size } });
  await refreshModels();
  await openModel(m.id);
  return m;
}

function updateStatus() {
  const [sx, sy, sz] = editor.size;
  $('st-size').textContent = `Tamaño ${sx} × ${sy} × ${sz}`;
  $('st-count').textContent = `${nf.format(editor.count)} voxels`;
  $('empty-hint').hidden = editor.count > 0;
}

// ---------------------------------------------------------------- slicer
function setupSlicer() {
  const r = $('slicer-range');
  r.max = String(editor.size[1]);
  r.value = String(editor.limit);
  renderSlicer();
}
function renderSlicer() {
  const all = editor.limit >= editor.size[1];
  $('slicer-value').textContent = all ? 'todas' : `y<${editor.limit}`;
  $('slicer').classList.toggle('on', !all);
}
$('slicer-range').addEventListener('input', (e) => { editor.setLimit(parseInt(e.target.value, 10)); renderSlicer(); });
$('slicer').addEventListener('dblclick', () => { editor.setLimit(editor.size[1]); setupSlicer(); });

// ---------------------------------------------------------------- top bar actions
$('model-select').addEventListener('change', (e) => openModel(e.target.value).catch(fail));

$('btn-new').addEventListener('click', async () => {
  const def = (state.settings && state.settings.new_size) || [32, 32, 32];
  const name = el('input', { type: 'text', value: 'Sin título', maxlength: '120' });
  const ins = def.map((v) => el('input', { type: 'number', min: '1', max: '256', value: String(v) }));
  const presets = el('div', { class: 'presets' }, ...[16, 32, 48, 64, 96].map((n) =>
    el('button', { type: 'button', class: 'btn small', text: `${n}³`, onclick: () => ins.forEach((i) => { i.value = String(n); }) })));
  const r = await modal({
    title: 'Modelo nuevo',
    body: [el('label', { class: 'field' }, el('span', { text: 'Nombre' }), name),
      el('div', { class: 'size-row' }, ...['Ancho (x)', 'Alto (y)', 'Fondo (z)'].map((t, k) => el('label', {}, t, ins[k]))), presets],
    buttons: [{ label: 'Cancelar', value: null }, { label: 'Crear', value: 'ok', kind: 'primary' }],
    onOpen: () => { name.focus(); name.select(); },
    validate: () => (ins.every((i) => +i.value >= 1 && +i.value <= 256) ? '' : 'Cada lado va de 1 a 256.'),
  });
  if (!r) return;
  try {
    await createModel(name.value.trim() || 'Sin título', ins.map((i) => parseInt(i.value, 10)));
    toast('Modelo creado.', 'ok');
  } catch (err) { fail(err); }
});

$('btn-dup').addEventListener('click', async () => {
  if (!state.model) return;
  try {
    await flushSave();
    const m = await api(`/api/models/${state.model.id}/duplicate`, { method: 'POST' });
    await refreshModels();
    await openModel(m.id, { keepCamera: true });
    toast(`Copia creada: ${m.name}`, 'ok');
  } catch (err) { fail(err); }
});

$('btn-rename').addEventListener('click', async () => {
  if (!state.model) return;
  const name = await askText('Renombrar', 'Nombre del modelo', state.model.name, 'Renombrar');
  if (!name || name === state.model.name) return;
  try {
    await flushSave();
    await api(`/api/models/${state.model.id}/rename`, { method: 'POST', body: { name } });
    state.model.name = name;
    document.title = `${name} · Seurat's Hoard`;
    await refreshModels();
    emit('model', state.model);
  } catch (err) { fail(err); }
});

$('btn-delete').addEventListener('click', async () => {
  if (!state.model) return;
  const ok = await confirmBox('Borrar modelo', `«${state.model.name}» irá a la papelera de datos (data/trash). ¿Seguro?`, 'Borrar', true);
  if (!ok) return;
  try {
    const id = state.model.id;
    clearTimeout(state.saveTimer);
    if (state.saving) await state.saving.catch(() => {});
    await api(`/api/models/${id}`, { method: 'DELETE' });
    state.model = null;
    const list = await refreshModels();
    if (list.length) await openModel(list[0].id);
    else await createModel('Sin título', (state.settings && state.settings.new_size) || [32, 32, 32]);
    toast('Modelo enviado a la papelera.', 'ok');
  } catch (err) { fail(err); }
});

$('btn-versions').addEventListener('click', async () => {
  if (!state.model) return;
  try {
    await flushSave();
    const { versions } = await api(`/api/models/${state.model.id}/versions`);
    const list = el('ul', { class: 'versions' });
    if (!versions.length) list.append(el('li', {}, el('span', { class: 'vinfo', text: 'Todavía no hay versiones anteriores.' })));
    let chosen = null;
    let dlgRef = null;
    for (const v of versions) {
      list.append(el('li', {},
        el('div', { class: 'vinfo' }, el('span', { text: `${fmtDate(v.updated)} · ${nf.format(v.count)} voxels` }), el('small', { text: noteText(v.note) })),
        el('button', { type: 'button', class: 'btn small', text: 'Restaurar', onclick: () => { chosen = v; dlgRef.close(); } })));
    }
    await modal({ title: `Versiones de «${state.model.name}»`, body: [el('p', { text: 'Restaurar guarda antes el estado actual como otra versión.' }), list],
      onOpen: (d) => { dlgRef = d; } });
    if (!chosen) return;
    await api(`/api/models/${state.model.id}/restore`, { method: 'POST', body: { version: chosen.version } });
    await openModel(state.model.id, { keepCamera: true });
    await refreshModels();
    toast(`Restaurada la versión de ${fmtDate(chosen.updated)}.`, 'ok');
  } catch (err) { fail(err); }
});

function noteText(note) {
  if (!note) return 'sin nota';
  if (note === 'editor') return 'cambios en el editor';
  if (note.startsWith('before generation')) return 'antes de una generación';
  if (note.startsWith('before restoring')) return 'antes de restaurar otra versión';
  return note;
}

$('btn-import').addEventListener('click', () => $('file-vox').click());
$('file-vox').addEventListener('change', async (e) => {
  const f = e.target.files[0];
  e.target.value = '';
  if (f) await importVox(f);
});

async function importVox(file) {
  try {
    await flushSave();
    const name = file.name.replace(/\.vox$/i, '');
    const m = await api(`/api/models/import-vox?name=${encodeURIComponent(name)}`, { method: 'POST', raw: file });
    await refreshModels();
    await openModel(m.id);
    toast(`Importado: ${m.name} (${nf.format(m.count)} voxels)`, 'ok');
  } catch (err) { fail(err); }
}

// ---------------------------------------------------------------- views
for (const b of document.querySelectorAll('[data-view]')) b.addEventListener('click', () => editor.setView(b.dataset.view));

// ---------------------------------------------------------------- keyboard
const KEYS = { a: 'add', b: 'erase', p: 'paint', r: 'fill', i: 'pick', c: 'box' };
window.addEventListener('keydown', (e) => {
  if (isTyping(e) || document.querySelector('dialog[open]')) return;
  const k = e.key.toLowerCase();
  if ((e.ctrlKey || e.metaKey) && k === 'z') { e.preventDefault(); if (e.shiftKey) editor.redo(); else editor.undo(); return; }
  if ((e.ctrlKey || e.metaKey) && k === 'y') { e.preventDefault(); editor.redo(); return; }
  if ((e.ctrlKey || e.metaKey) && k === 's') { e.preventDefault(); saveNow(); return; }
  if (e.ctrlKey || e.metaKey || e.altKey) return;
  if (KEYS[k]) { editor.setTool(KEYS[k]); return; }
  if (k === 'x') { toggleSym('x'); return; }
  if (k === 'z') { toggleSym('z'); return; }
  if (e.key === 'Home') { editor.setView('iso'); return; }
  if (e.key === 'PageUp' || e.key === 'PageDown') {
    e.preventDefault();
    editor.setLimit(editor.limit + (e.key === 'PageUp' ? 1 : -1));
    $('slicer-range').value = String(editor.limit);
    renderSlicer();
  }
});
// Shift changes what the box tool will do: refresh the ghost
window.addEventListener('keyup', (e) => { if (e.key === 'Shift' || e.key === 'Alt') editor.requestRender(); });

window.addEventListener('beforeunload', (e) => {
  if (!state.model || state.editSeq === state.savedSeq) return;
  const body = JSON.stringify(payload());
  if (body.length < 60000) {
    fetch(`/api/models/${state.model.id}`, { method: 'PUT', body, headers: { 'Content-Type': 'application/json' }, keepalive: true });
  } else {
    e.preventDefault();
    e.returnValue = '';
  }
});
document.addEventListener('visibilitychange', () => { if (document.hidden) flushSave(); });

// ---------------------------------------------------------------- start
async function loadStatus() {
  try {
    state.status = await api('/api/status');
    state.settings = state.status.settings;
    emit('status', state.status);
  } catch (err) { fail(err); }
  return state.status;
}

async function start() {
  buildToolbar();
  await loadStatus();
  initPanels({
    bus, editor, state, openModel, refreshModels, flushSave, loadStatus,
    current: () => state.model,
    settings: () => state.settings,
    setSettings: (s) => { state.settings = s; },
    importVox,
  });
  try {
    const list = await refreshModels();
    if (!list.length) {
      await createModel('Sin título', (state.settings && state.settings.new_size) || [32, 32, 32]);
    } else {
      const last = store('model');
      await openModel(list.find((m) => m.id === last) ? last : list[0].id);
    }
  } catch (err) { fail(err); }
  setInterval(loadStatus, 60000);
}

start();

// exposed for quick checks from the browser console
window.seurat = { editor, state, openModel, flushSave };
