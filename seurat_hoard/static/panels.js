// The side panel: reference pictures, generation with the language model (and the instant builders), exports and
// settings. `ctx` comes from app.js (the editor, the open model, openModel, flushSave, the event bus...).

import { $, el, api, toast, fail, modal, confirmBox, nf, store } from './ui.js';

const ROLES = [
  ['front', 'Frente'], ['side', 'Lado (izquierdo)'], ['top', 'Arriba'], ['back', 'Detrás'], ['extra', 'Extra'], ['none', 'No usar'],
];
const ROLE_ORDER = { front: 0, side: 1, top: 2, back: 3, extra: 4 };
const MAX_REFS = 6;
const JOB_STATUS = { queued: 'En cola', running: 'Trabajando', done: 'Hecho', failed: 'Error', cancelled: 'Cancelado' };
const START_LABEL = { auto: 'automático', inflate: 'inflar', hull: 'silueta', extrude: 'extruir', empty: 'vacío', current: 'modelo actual' };

let ctx;
const refs = { list: [], roles: store('roles') || {} };
const job = { id: null, data: null, timer: null, watching: false, started: 0 };

export function initPanels(context) {
  ctx = context;
  initTabs();
  initReferences();
  initGenerate();
  initExport();
  initSettings();
  initGlobalDrop();
  ctx.bus.addEventListener('status', (e) => { renderLlm(e.detail); fillSettings(e.detail.settings, false); });
  if (ctx.state.status) { renderLlm(ctx.state.status); fillSettings(ctx.state.status.settings, true); }
  ctx.bus.addEventListener('model', () => { renderDest(); refreshExport(); });
  ctx.bus.addEventListener('saved', () => refreshExport());
  loadReferences();
  resumeJob();
}

// ================================================================= tabs
function initTabs() {
  const show = (name) => {
    for (const t of document.querySelectorAll('.tab')) t.classList.toggle('active', t.dataset.tab === name);
    for (const p of document.querySelectorAll('.pane')) p.classList.toggle('active', p.dataset.pane === name);
    store('tab', name);
    if (name === 'export') refreshExport(true);
    if (name === 'gen') { renderGenRefs(); ctx.loadStatus(); }
  };
  for (const t of document.querySelectorAll('.tab')) t.addEventListener('click', () => show(t.dataset.tab));
  document.addEventListener('click', (e) => {
    const g = e.target.closest('[data-goto]');
    if (g) { e.preventDefault(); show(g.dataset.goto); }
  });
  ctx.showTab = show;
  const last = store('tab');
  if (last && document.querySelector(`.tab[data-tab="${last}"]`)) show(last);
}

const paneVisible = (name) => document.querySelector(`.pane[data-pane="${name}"]`).classList.contains('active');

// ================================================================= references
function initReferences() {
  const drop = $('drop');
  drop.addEventListener('click', (e) => { if (e.target.id !== 'pick-refs') $('file-refs').click(); });
  $('pick-refs').addEventListener('click', (e) => { e.preventDefault(); $('file-refs').click(); });
  $('file-refs').addEventListener('change', (e) => { const files = [...e.target.files]; e.target.value = ''; uploadReferences(files); });
  for (const ev of ['dragenter', 'dragover']) drop.addEventListener(ev, (e) => { e.preventDefault(); drop.classList.add('over'); });
  for (const ev of ['dragleave', 'drop']) drop.addEventListener(ev, () => drop.classList.remove('over'));
}

async function loadReferences() {
  try {
    const res = await api('/api/references');
    refs.list = res.references || [];
    // roles of references that no longer exist go away; a first visit uses the newest picture as the front
    const ids = new Set(refs.list.map((r) => r.id));
    for (const k of Object.keys(refs.roles)) if (!ids.has(k)) delete refs.roles[k];
    if (refs.list.length && !refs.list.some((r) => refs.roles[r.id])) refs.roles[refs.list[0].id] = 'front';
    saveRoles();
  } catch (err) { fail(err); }
  renderReferences();
}

const roleOf = (id) => refs.roles[id] || 'none';
function saveRoles() { store('roles', refs.roles); }

function setRole(id, role) {
  refs.roles[id] = role;
  saveRoles();
  renderReferences();
}

function usedReferences() {
  return refs.list.filter((r) => roleOf(r.id) !== 'none')
    .sort((a, b) => ROLE_ORDER[roleOf(a.id)] - ROLE_ORDER[roleOf(b.id)]);
}

function roleSelect(r, cls = 'role') {
  const sel = el('select', { class: cls, 'aria-label': `Papel de ${r.name}` },
    ...ROLES.map(([v, label]) => el('option', { value: v, text: label })));
  sel.value = roleOf(r.id);
  sel.addEventListener('change', () => setRole(r.id, sel.value));
  return sel;
}

function renderReferences() {
  const grid = $('ref-grid');
  grid.replaceChildren();
  $('refs-empty').hidden = refs.list.length > 0;
  for (const r of refs.list) {
    const role = roleOf(r.id);
    const card = el('div', { class: `ref${role !== 'none' ? ' used' : ''}${role === 'front' ? ' front' : ''}`, 'data-ref': r.id },
      el('div', { class: 'ref-thumb checker', title: 'Ver grande y cómo la lee el modelo', onclick: () => showReference(r) },
        el('img', { src: `/references/${r.id}/thumb.png?side=256`, alt: r.name, loading: 'lazy' }),
        el('button', { class: 'ref-del', title: 'Borrar la referencia', 'aria-label': 'Borrar', text: '×',
          onclick: (e) => { e.stopPropagation(); deleteReference(r); } })),
      el('div', { class: 'ref-body' },
        el('div', { class: 'ref-name', title: r.name }, r.name, el('small', { text: `${r.width}×${r.height}` })),
        roleSelect(r)));
    grid.append(card);
  }
  renderGenRefs();
}

async function uploadReferences(files) {
  const images = files.filter((f) => /^image\//.test(f.type) || /\.(png|jpe?g|gif|webp|bmp)$/i.test(f.name));
  if (!images.length) { toast('Elige imágenes PNG o JPG.', 'err'); return; }
  const added = [];
  for (const f of images) {
    try {
      const meta = await api(`/api/references?name=${encodeURIComponent(f.name.replace(/\.[^.]+$/, ''))}`, { method: 'POST', raw: f, type: f.type || 'application/octet-stream' });
      added.push(meta);
    } catch (err) { fail(Object.assign(err, { message: `${f.name}: ${err.message}` })); }
  }
  if (!added.length) return;
  // the pictures just uploaded are the ones to use: the first is the front, the rest extra views
  for (const k of Object.keys(refs.roles)) refs.roles[k] = 'none';
  added.forEach((m, i) => { refs.roles[m.id] = i === 0 ? 'front' : (i < MAX_REFS ? 'extra' : 'none'); });
  saveRoles();
  await loadReferences();
  toast(added.length === 1 ? `Subida: ${added[0].name}` : `Subidas ${added.length} imágenes`, 'ok');
}

async function deleteReference(r) {
  if (!await confirmBox('Borrar referencia', `¿Borrar «${r.name}»? Los modelos hechos con ella no cambian.`, 'Borrar', true)) return;
  try {
    await api(`/api/references/${r.id}`, { method: 'DELETE' });
    delete refs.roles[r.id];
    saveRoles();
    await loadReferences();
  } catch (err) { fail(err); }
}

async function showReference(r) {
  const size = el('input', { type: 'range', min: '8', max: '96', value: $('gen-size').value });
  const colors = el('input', { type: 'range', min: '2', max: '48', value: $('gen-colors').value });
  const sizeOut = el('output', { text: size.value }), colorsOut = el('output', { text: colors.value });
  const indexed = el('img', { alt: 'Así lo lee el modelo' });
  const caption = el('figcaption');
  const name = el('input', { type: 'text', value: r.name, maxlength: '120' });
  let t = null;
  const update = () => {
    sizeOut.textContent = size.value; colorsOut.textContent = colors.value;
    caption.textContent = `Así lo lee el modelo: ${size.value} px como máximo, ${colors.value} colores`;
    clearTimeout(t);
    t = setTimeout(() => { indexed.src = `/api/references/${r.id}/indexed.png?max_side=${size.value}&colors=${colors.value}`; }, 120);
  };
  size.addEventListener('input', update);
  colors.addEventListener('input', update);
  update();
  const res = await modal({
    title: 'Referencia', wide: true,
    body: [
      el('label', { class: 'field' }, el('span', { text: 'Nombre' }), name),
      el('div', { class: 'ref-detail' },
        el('figure', {}, el('div', { class: 'imgbox checker' }, el('img', { src: `/references/${r.id}`, alt: r.name })),
          el('figcaption', { text: `Original: ${r.width} × ${r.height} px` })),
        el('figure', {}, el('div', { class: 'imgbox checker' }, indexed), caption)),
      el('div', { class: 'ref-controls' },
        el('label', { class: 'field' }, el('span', {}, 'Tamaño ', sizeOut, ' voxels'), size),
        el('label', { class: 'field' }, el('span', {}, 'Colores ', colorsOut), colors)),
      el('p', { text: 'Tamaño y colores se copian al formulario de Generar.' }),
    ],
    buttons: [{ label: 'Cerrar', value: null }, { label: 'Usar como frente', value: 'front' }, { label: 'Guardar', value: 'save', kind: 'primary' }],
  });
  // whatever the button, keep the size/colours chosen here
  $('gen-size').value = size.value; $('gen-colors').value = colors.value; syncOutputs();
  if (!res) return;
  try {
    const newName = name.value.trim();
    if (newName && newName !== r.name) await api(`/api/references/${r.id}/rename`, { method: 'POST', body: { name: newName } });
    if (res.value === 'front') {
      for (const x of refs.list) if (roleOf(x.id) === 'front') refs.roles[x.id] = 'extra';
      refs.roles[r.id] = 'front';
      saveRoles();
    }
    await loadReferences();
  } catch (err) { fail(err); }
}

// ================================================================= generate
function initGenerate() {
  for (const id of ['gen-size', 'gen-colors', 'gen-rounds']) $(id).addEventListener('input', syncOutputs);
  $('gen-start').addEventListener('change', renderDest);
  for (const r of document.querySelectorAll('input[name="gen-dest"]')) r.addEventListener('change', renderDest);
  $('gen-prompt').value = store('prompt') || '';
  $('gen-prompt').addEventListener('input', () => store('prompt', $('gen-prompt').value));
  $('btn-generate').addEventListener('click', () => startGeneration());
  for (const b of document.querySelectorAll('[data-build]')) b.addEventListener('click', () => quickBuild(b.dataset.build, b));
}

function syncOutputs() {
  $('gen-size-out').textContent = $('gen-size').value;
  $('gen-colors-out').textContent = $('gen-colors').value;
  $('gen-rounds-out').textContent = $('gen-rounds').value;
}

function renderDest() {
  const current = $('gen-start').value === 'current';
  const newRadio = document.querySelector('input[name="gen-dest"][value="new"]');
  const intoRadio = document.querySelector('input[name="gen-dest"][value="into"]');
  if (current) intoRadio.checked = true;
  newRadio.disabled = current;
  newRadio.parentElement.classList.toggle('disabled', current);
  const m = ctx.current();
  intoRadio.parentElement.lastChild.textContent = ` Nueva versión de «${m ? m.name : 'el modelo abierto'}»`;
  $('gen-name').disabled = intoRadio.checked;
}

function renderGenRefs() {
  const box = $('gen-refs');
  box.replaceChildren();
  const used = usedReferences();
  if (!used.length) {
    box.append(el('div', { class: 'empty' }, 'Ninguna referencia elegida: se construirá solo con la descripción (un objeto o una escena). ', el('a', { href: '#', 'data-goto': 'refs', text: 'Sube o elige una' }), '.'));
  }
  used.forEach((r, i) => {
    const row = el('div', { class: 'gen-ref' },
      el('img', { src: `/references/${r.id}/thumb.png?side=96`, alt: '' }),
      el('span', { title: r.name, text: r.name + (i >= MAX_REFS ? ' (sobra: máx. 6)' : '') }),
      roleSelect(r));
    box.append(row);
  });
  if (used.length && !used.some((r) => roleOf(r.id) === 'front')) {
    box.append(el('small', { text: 'Ninguna es «Frente»: se usará la primera como frente.' }));
  }
}

function renderLlm(status) {
  const llm = status && status.llm;
  const line = $('llm-line');
  if (!llm) { line.dataset.state = 'unknown'; $('llm-text').textContent = 'Sin datos del modelo.'; return; }
  if (llm.available) {
    line.dataset.state = 'ok';
    const who = [llm.model, llm.provider].filter(Boolean).join(' · ') || 'el modelo de la familia';
    $('llm-text').textContent = `Responderá ${who}${llm.url ? ` (${llm.url})` : ''}.`;
  } else {
    line.dataset.state = 'off';
    $('llm-text').textContent = `Ahora mismo no hay modelo disponible. ${llm.reason || ''}`.trim();
  }
}

function genParams() {
  const used = usedReferences().slice(0, MAX_REFS);
  const start = $('gen-start').value;
  const into = document.querySelector('input[name="gen-dest"]:checked').value === 'into' || start === 'current';
  const m = ctx.current();
  return {
    references: used.map((r) => ({ id: r.id, role: roleOf(r.id) })),
    prompt: $('gen-prompt').value.trim(),
    name: into ? '' : $('gen-name').value.trim(),
    into: into && m ? m.id : '',
    start,
    rounds: parseInt($('gen-rounds').value, 10),
    max_side: parseInt($('gen-size').value, 10),
    colors: parseInt($('gen-colors').value, 10),
    effort: $('gen-effort').value,
    ...(used.length ? {} : { size: Array(3).fill(parseInt($('gen-size').value, 10)) }),
  };
}

async function startGeneration(params) {
  params = params || genParams();
  if (!params.references.length && !params.prompt) { toast('Elige una referencia (pestaña Referencias) o describe qué construir.', 'err'); return; }
  if (!params.references.length && !['auto', 'empty', 'current'].includes(params.start)) params.start = 'auto';
  if (job.data && (job.data.status === 'running' || job.data.status === 'queued')) {
    toast('Ya hay una generación en marcha.', 'err'); return;
  }
  const btn = $('btn-generate');
  btn.disabled = true;
  try {
    if (params.into) await ctx.flushSave();
    const res = await api('/api/generate', { method: 'POST', body: params });
    // show the new job at once instead of the previous one until the first poll
    renderJob({ id: res.job_id, status: 'queued', step: 'en cola', log: [], rounds: [], created: Date.now() / 1000, params });
    renderBusy(true);
    watchJob(res.job_id, true);
  } catch (err) { fail(err); btn.disabled = false; }
}

async function quickBuild(mode, btn) {
  const used = usedReferences();
  const front = used.find((r) => roleOf(r.id) === 'front') || used[0];
  if (!front) { toast('Elige una referencia como «Frente» primero.', 'err'); return; }
  const side = used.find((r) => roleOf(r.id) === 'side');
  const top = used.find((r) => roleOf(r.id) === 'top');
  if (mode === 'hull' && !side) { toast('La silueta necesita además una referencia con papel «Lado».', 'err'); return; }
  btn.disabled = true;
  try {
    const body = { reference_id: front.id, mode, max_side: parseInt($('gen-size').value, 10), colors: parseInt($('gen-colors').value, 10) };
    if (side && mode === 'hull') body.side_id = side.id;
    if (top && mode === 'hull') body.top_id = top.id;
    const name = $('gen-name').value.trim();
    if (name) body.name = name;
    const m = await api('/api/build', { method: 'POST', body });
    await ctx.refreshModels();
    await ctx.openModel(m.id);
    toast(`Hecho: «${m.name}», ${nf.format(m.count)} voxels. Abierto en el editor.`, 'ok');
  } catch (err) { fail(err); }
  btn.disabled = false;
}

async function resumeJob() {
  try {
    const { jobs } = await api('/api/jobs');
    const live = (jobs || []).find((j) => j.status === 'running' || j.status === 'queued');
    if (live) { watchJob(live.id, true); return; }
    const recent = (jobs || [])[0];
    if (recent && Date.now() / 1000 - recent.created < 1800) watchJob(recent.id, false);
  } catch { /* the panel stays empty */ }
}

function watchJob(id, live) {
  job.id = id;
  job.watching = live;
  job.data = null;
  clearTimeout(job.timer);
  pollJob();
}

async function pollJob() {
  const id = job.id;
  let data;
  try { data = await api(`/api/jobs/${id}`); } catch (err) {
    if (err.status === 404) { $('job').hidden = true; job.id = null; renderBusy(false); return; }
    job.timer = setTimeout(pollJob, 3000);
    return;
  }
  if (id !== job.id) return;
  const before = job.data ? job.data.status : null;
  job.data = data;
  renderJob(data);
  const active = data.status === 'running' || data.status === 'queued';
  renderBusy(active);
  if (active) { job.timer = setTimeout(pollJob, 1000); return; }
  if (job.watching && before !== data.status) {
    job.watching = false;
    if (data.status === 'done' && data.model_id) {
      try {
        await ctx.refreshModels();
        const cur = ctx.current();
        await ctx.openModel(data.model_id, { keepCamera: !!(cur && cur.id === data.model_id) });
        toast(data.warning ? `El modelo falló a mitad; «${data.model_name}» guarda lo que había.` : `Listo: «${data.model_name}» está abierto en el editor.`, data.warning ? 'err' : 'ok', 5000);
      } catch (err) { fail(err); }
    } else if (data.status === 'failed') {
      toast('La generación ha fallado. Mira el detalle en Generar.', 'err');
    } else if (data.status === 'cancelled' && data.model_id) {
      await ctx.refreshModels();
      toast(`Cancelado. Se guardó lo que había en «${data.model_name}».`);
    }
  }
}

function renderBusy(active) {
  $('btn-generate').disabled = active;
  $('btn-generate').textContent = active ? 'Generando…' : 'Generar con DeepSeek';
  const tab = document.querySelector('.tab[data-tab="gen"]');
  const badge = tab.querySelector('.badge');
  if (active && !badge) tab.append(el('span', { class: 'badge', title: 'Generando' }));
  if (!active && badge) badge.remove();
}

function fmtElapsed(s) {
  s = Math.max(0, Math.round(s));
  return s < 60 ? `${s} s` : `${Math.floor(s / 60)} min ${String(s % 60).padStart(2, '0')} s`;
}

function renderJob(d) {
  const box = $('job');
  box.hidden = false;
  const logBox = box.querySelector('.job-log');
  const stick = !logBox || logBox.scrollTop + logBox.clientHeight >= logBox.scrollHeight - 8;
  const active = d.status === 'running' || d.status === 'queued';
  const elapsed = ((d.finished || Date.now() / 1000) - d.created);
  const head = el('div', { class: 'job-head' },
    el('span', { class: 'pill', 'data-s': d.status, text: JOB_STATUS[d.status] || d.status }),
    el('span', { class: 'step', title: d.step, text: d.step || '' }),
    el('span', { class: 'time', text: fmtElapsed(elapsed) }),
    active ? el('button', { class: 'btn small danger', text: 'Cancelar', onclick: cancelJob }) : null,
    !active ? el('button', { class: 'btn small ghost', text: 'Ocultar', onclick: () => { box.hidden = true; } }) : null);
  const parts = [head];
  if (d.status === 'failed') {
    const p = d.params || {};
    const retry = p.references && p.references.length && p.use_model !== false
      ? el('button', { class: 'btn small', text: 'Guardar solo el modelo inicial',
        title: 'Repite la generación sin el modelo de lenguaje: solo el modelo de partida',
        onclick: () => startGeneration({ ...p, use_model: false }) })
      : null;
    parts.push(el('div', { class: 'job-error' }, el('b', { text: 'No se pudo terminar. ' }), d.error || 'Error desconocido.',
      el('div', {}, 'No se ha guardado ningún modelo.'), retry));
  } else if (d.status === 'done' && d.model_id) {
    parts.push(el('div', { class: 'job-done' }, 'Guardado como ', el('b', { text: `«${d.model_name}»` }),
      d.llm ? ` · respondió ${d.llm}` : ' · sin modelo de lenguaje', ' · ',
      el('a', { href: '#', text: 'abrir', onclick: (e) => { e.preventDefault(); ctx.openModel(d.model_id).catch(fail); } })));
    if (d.warning) {
      parts.push(el('div', { class: 'job-error' }, el('b', { text: 'El modelo falló a mitad. ' }), d.warning,
        el('div', {}, 'Se guardó lo que había hasta ese momento.')));
    }
  }
  const log = el('div', { class: 'job-log' }, ...(d.log || []).map((l) =>
    el('div', {}, el('time', { text: new Date(l.t * 1000).toLocaleTimeString('es-ES') }), l.text)));
  parts.push(log);
  const rounds = el('div', { class: 'rounds' });
  for (const r of d.rounds || []) {
    const img = r.render
      ? el('img', { src: `/renders/${r.render}`, alt: `Ronda ${r.round}`, loading: 'lazy', onclick: () => showRender(r) })
      : el('div', { class: 'noimg', text: 'vacío' });
    const isStart = r.round === 0;
    rounds.append(el('div', { class: 'round' }, img,
      el('div', {},
        el('div', { class: 'round-title', text: isStart ? `Inicio (${START_LABEL[(r.notes || '').replace('inicio: ', '')] || r.notes})` : `Ronda ${r.round}` }),
        el('div', { class: 'round-meta', text: `${nf.format(r.count)} voxels${isStart ? '' : ` · ${r.applied} operaciones`}${r.size ? ` · ${r.size.join('×')}` : ''}` }),
        !isStart && r.notes ? el('div', { class: 'round-notes', text: r.notes }) : null,
        r.errors && r.errors.length ? el('ul', { class: 'round-errors' }, ...r.errors.map((e) => el('li', { text: e }))) : null)));
  }
  if ((d.rounds || []).length) parts.push(rounds);
  box.replaceChildren(...parts);
  if (stick) log.scrollTop = log.scrollHeight;
}

async function cancelJob() {
  if (!job.id) return;
  try { await api(`/api/jobs/${job.id}/cancel`, { method: 'POST' }); toast('Cancelando…'); } catch (err) { fail(err); }
}

function showRender(r) {
  modal({ title: r.round === 0 ? 'Modelo de partida' : `Ronda ${r.round}`, wide: true,
    body: [el('img', { class: 'big-render', src: `/renders/${r.render}`, alt: '' }), r.notes ? el('p', { text: r.notes }) : null] });
}

// ================================================================= export
let exportStale = true;

function initExport() {
  for (const b of document.querySelectorAll('[data-export]')) b.addEventListener('click', () => doExport(b.dataset.export, b));
  $('export-img').addEventListener('error', () => { $('export-img').hidden = true; $('export-empty').hidden = false; });
  $('export-img').addEventListener('load', () => { $('export-img').hidden = false; $('export-empty').hidden = true; });
}

function contentBounds() {
  const ed = ctx.editor;
  const [sx, sy] = ed.size;
  const lo = [Infinity, Infinity, Infinity], hi = [-1, -1, -1];
  const g = ed.grid;
  for (let i = 0; i < g.length; i++) {
    if (!g[i]) continue;
    const x = i % sx, y = Math.floor(i / sx) % sy, z = Math.floor(i / (sx * sy));
    if (x < lo[0]) lo[0] = x; if (y < lo[1]) lo[1] = y; if (z < lo[2]) lo[2] = z;
    if (x > hi[0]) hi[0] = x; if (y > hi[1]) hi[1] = y; if (z > hi[2]) hi[2] = z;
  }
  return hi[0] < 0 ? null : [hi[0] - lo[0] + 1, hi[1] - lo[1] + 1, hi[2] - lo[2] + 1];
}

const fmtNum = (v) => new Intl.NumberFormat('es-ES', { maximumFractionDigits: 2 }).format(v);

function refreshExport(force) {
  if (!force) { exportStale = true; if (!paneVisible('export')) return; }
  const m = ctx.current();
  if (!m) return;
  exportStale = false;
  const empty = ctx.editor.count === 0;
  for (const b of document.querySelectorAll('[data-export]')) b.disabled = empty;
  $('export-empty').hidden = !empty;
  $('export-img').hidden = empty;
  if (!empty) $('export-img').src = `/api/models/${m.id}/render.png?view=sheet&t=${Date.now()}`;
  const s = ctx.settings() || {};
  const ext = contentBounds();
  const mm = s.voxel_mm || 2, gm = s.glb_voxel_size || 0.1;
  $('stl-info').textContent = ext
    ? `.stl · ${fmtNum(mm)} mm por voxel → ${ext.map((v) => fmtNum(v * mm)).join(' × ')} mm`
    : `.stl · ${fmtNum(mm)} mm por voxel`;
  $('glb-info').textContent = ext
    ? `.glb · ${fmtNum(gm)} m por voxel → ${ext.map((v) => fmtNum(v * gm)).join(' × ')} m`
    : `.glb · ${fmtNum(gm)} m por voxel`;
}

async function doExport(fmt, btn) {
  const m = ctx.current();
  if (!m) return;
  btn.disabled = true;
  try {
    await ctx.flushSave();
    let url = `/api/models/${m.id}/export/${fmt}`;
    if (fmt === 'png') url += `?view=${$('png-view').value}`;
    // ask first so an error comes back as a message instead of a broken download
    const head = await fetch(url, { method: 'GET' });
    if (!head.ok) {
      let msg = `Error ${head.status}`;
      try { const j = await head.json(); msg = j.error || msg; } catch { /* not JSON */ }
      throw new Error(msg);
    }
    const blob = await head.blob();
    const cd = head.headers.get('Content-Disposition') || '';
    const match = /filename\*=UTF-8''([^;]+)/i.exec(cd);
    const filename = match ? decodeURIComponent(match[1]) : `${m.name}.${fmt === 'sprites' ? 'png' : fmt}`;
    const obj = URL.createObjectURL(blob);
    const a = el('a', { href: obj, download: filename });
    document.body.append(a); a.click(); a.remove();
    setTimeout(() => URL.revokeObjectURL(obj), 10000);
    toast(`Descargado: ${filename} (${fmtNum(blob.size / 1024)} KB)`, 'ok');
  } catch (err) { fail(err); }
  btn.disabled = false;
}

// ================================================================= settings
let settingsDirty = false;

function initSettings() {
  const form = $('settings-form');
  form.addEventListener('input', () => { settingsDirty = true; });
  form.addEventListener('submit', async (e) => {
    e.preventDefault();
    const f = form.elements;
    const body = {
      max_side: +f.max_side.value, colors: +f.colors.value, review_rounds: +f.review_rounds.value, effort: f.effort.value,
      max_tokens: +f.max_tokens.value, thinking_tokens: +f.thinking_tokens.value, timeout_s: +f.timeout_s.value,
      auto_paint: f.auto_paint.checked, parallel_jobs: +f.parallel_jobs.value, llm_url: f.llm_url.value.trim(), llm_model: f.llm_model.value.trim(),
      voxel_mm: +f.voxel_mm.value, glb_voxel_size: +f.glb_voxel_size.value,
      new_size: [+f.new_size_x.value, +f.new_size_y.value, +f.new_size_z.value],
    };
    try {
      const s = await api('/api/settings', { method: 'POST', body });
      ctx.setSettings(s);
      settingsDirty = false;
      fillSettings(s, true);
      await ctx.loadStatus();
      refreshExport();
      toast('Ajustes guardados.', 'ok');
    } catch (err) { fail(err); }
  });
  $('settings-reset').addEventListener('click', () => { settingsDirty = false; fillSettings(ctx.settings(), false); });
}

// `defaults`: also put the reading/generation values into the Generar form
function fillSettings(s, defaults) {
  if (!s) return;
  if (!settingsDirty) {
    const f = $('settings-form').elements;
    for (const k of ['max_side', 'colors', 'review_rounds', 'effort', 'max_tokens', 'thinking_tokens', 'parallel_jobs', 'timeout_s', 'llm_url', 'llm_model', 'voxel_mm', 'glb_voxel_size']) {
      if (f[k] && s[k] !== undefined) f[k].value = s[k];
    }
    if (f.auto_paint) f.auto_paint.checked = s.auto_paint !== false;
    const [x, y, z] = s.new_size || [32, 32, 32];
    f.new_size_x.value = x; f.new_size_y.value = y; f.new_size_z.value = z;
  }
  if (defaults) {
    $('gen-size').value = Math.max(8, Math.min(96, s.max_side));
    $('gen-colors').value = s.colors;
    $('gen-rounds').value = s.review_rounds;
    $('gen-effort').value = s.effort;
    syncOutputs();
  }
}

// ================================================================= drop anywhere
function initGlobalDrop() {
  const veil = $('dropveil');
  let depth = 0;
  const hasFiles = (e) => e.dataTransfer && [...e.dataTransfer.types].includes('Files');
  window.addEventListener('dragenter', (e) => { if (!hasFiles(e)) return; depth++; veil.hidden = false; });
  window.addEventListener('dragleave', () => { depth = Math.max(0, depth - 1); if (!depth) veil.hidden = true; });
  window.addEventListener('dragover', (e) => { if (hasFiles(e)) e.preventDefault(); });
  window.addEventListener('drop', async (e) => {
    if (!hasFiles(e)) return;
    e.preventDefault();
    depth = 0; veil.hidden = true;
    const files = [...e.dataTransfer.files];
    const vox = files.filter((f) => /\.vox$/i.test(f.name));
    const pics = files.filter((f) => !/\.vox$/i.test(f.name));
    for (const f of vox) await ctx.importVox(f);
    if (pics.length) { ctx.showTab('refs'); await uploadReferences(pics); }
  });
}

