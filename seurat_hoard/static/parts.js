// Parts of a character built from its sprites (the plan kept with the model): sizes to tune with a few fields, and
// the plan itself for whoever wants to rewrite it. Lives under the generation job in the side panel.

import { $, el, api, toast, fail } from './ui.js';

const TYPES = [['solid', 'Sólido redondeado'], ['fin', 'Aleta fina'], ['fan', 'Abanico'], ['pair', 'Par (uno a cada lado)'],
  ['wings', 'Alas (dos láminas)'], ['decal', 'Marca en los flancos'], ['skip', 'No construir (la de detrás)']];
const PAINT = [['auto', 'Automático'], ['none', 'Solo el sprite lateral'], ['front', 'Cara de frente'], ['back', 'Dorso'], ['both', 'Frente y dorso']];
const FIELDS = {
  solid: [['width', 'Ancho', 'voxels de lado a lado; vacío = tan ancho como grueso es de perfil', 2, 120, 1],
    ['paint', 'Caras', 'lo que mira a la cámara de frente o de espalda se pinta con ese sprite; automático = solo si ese sprite está en la misma pose que el modelo', PAINT]],
  fin: [['thickness', 'Grosor', 'voxels', 2, 12, 2]],
  fan: [['base', 'Ancho en la raíz', 'voxels', 2, 60, 1], ['width', 'Ancho al final', 'voxels', 2, 120, 1]],
  pair: [['offset', 'Separación', 'del centro del cuerpo al centro de cada uno', 0, 60, 0.5], ['width', 'Grosor', 'de cada uno, de lado a lado', 1, 60, 1]],
  wings: [['span', 'Envergadura', 'de punta a punta, en voxels', 4, 124, 1], ['out_up', 'Se abre al subir', '1 = cuanto más alto, más fuera (alas en V)', -2, 3, 0.1],
    ['out_back', 'Se abre hacia atrás', '1 = cuanto más atrás, más fuera (alas echadas atrás)', -2, 3, 0.1], ['thickness', 'Grosor', 'voxels', 2, 6, 1]],
  decal: [], skip: [],
};

let ctx;
let box;
let busy = false;

export function initParts(context) {
  ctx = context;
  const style = document.createElement('style');
  style.textContent = `
    .parts-box { margin-top: 14px; padding: 12px; border: 1px solid var(--line, #3a3640); border-radius: 10px; display: grid; gap: 10px; }
    .parts-box h3 { margin: 0; font-size: 14px; }
    .parts-box .hint { margin: 0; opacity: .75; font-size: 12px; }
    .part-row { display: grid; gap: 6px; padding: 8px; border-radius: 8px; background: rgba(255,255,255,.04); }
    .part-head { display: flex; gap: 8px; align-items: center; justify-content: space-between; }
    .part-head b { font-size: 13px; }
    .part-fields { display: grid; grid-template-columns: repeat(auto-fill, minmax(118px, 1fr)); gap: 6px; }
    .part-fields label { display: grid; gap: 2px; font-size: 11px; opacity: .9; }
    .part-fields input, .part-fields select, .part-head select { width: 100%; min-width: 0; padding: 4px 6px; border-radius: 6px; border: 1px solid var(--line, #3a3640);
      background: rgba(0,0,0,.25); color: inherit; font: inherit; font-size: 12px; }
    .part-head select { width: auto; max-width: 60%; }
    .parts-actions { display: flex; gap: 8px; flex-wrap: wrap; align-items: center; }
    .parts-box textarea { width: 100%; min-height: 160px; font: 12px/1.35 ui-monospace, Consolas, monospace; border-radius: 8px; padding: 8px;
      border: 1px solid var(--line, #3a3640); background: rgba(0,0,0,.25); color: inherit; resize: vertical; }
    .parts-report { white-space: pre-wrap; font: 11px/1.4 ui-monospace, Consolas, monospace; opacity: .85; margin: 0; }`;
  document.head.append(style);
  box = el('div', { class: 'parts-box', id: 'parts-box', hidden: true });
  const anchor = $('job');
  if (anchor && anchor.parentNode) anchor.parentNode.insertBefore(box, anchor.nextSibling);
  ctx.bus.addEventListener('model', render);
  render();
}

function planOf(m) { return m && m.source && m.source.method === 'parts' && m.source.plan ? m.source.plan : null; }

function render(report) {
  if (!box) return;
  const m = ctx.current();
  const plan = planOf(m);
  if (!plan) { box.hidden = true; box.replaceChildren(); return; }
  box.hidden = false;
  const rows = plan.parts.map((part) => {
    const type = el('select', { 'data-part': part.name, 'data-key': 'type', title: 'Qué es esta parte' },
      ...TYPES.map(([v, t]) => el('option', { value: v, text: t, selected: v === part.type })));
    const fields = el('div', { class: 'part-fields' });
    const fill = () => {
      fields.replaceChildren(...(FIELDS[type.value] || []).map(([key, label, hint, min, max, step]) =>
        el('label', { title: hint }, label, Array.isArray(min)
          ? el('select', { 'data-part': part.name, 'data-key': key },
            ...min.map(([v, t]) => el('option', { value: v, text: t, selected: v === (part[key] || min[0][0]) })))
          : el('input', { type: 'number', min, max, step, 'data-part': part.name, 'data-key': key, placeholder: 'auto',
            value: part[key] === null || part[key] === undefined ? '' : part[key] }))));
    };
    type.addEventListener('change', fill);
    fill();
    return el('div', { class: 'part-row' }, el('div', { class: 'part-head' }, el('b', { text: part.name }), type), fields);
  });
  const area = el('textarea', { spellcheck: false, hidden: true });
  area.value = JSON.stringify(plan, null, 1);
  const rebuild = el('button', { type: 'button', class: 'btn small primary', text: 'Reconstruir', title: 'Aplica los cambios y vuelve a construir el modelo (la versión anterior se guarda)',
    onclick: () => apply(area) });
  const toggle = el('button', { type: 'button', class: 'btn small', text: 'Editar el plan', title: 'El plan completo en JSON: formas, colores y tipos de cada parte',
    onclick: () => { area.hidden = !area.hidden; toggle.textContent = area.hidden ? 'Editar el plan' : 'Ocultar el plan'; } });
  box.replaceChildren(
    el('h3', { text: 'Partes del personaje' }),
    el('p', { class: 'hint', text: 'El perfil es el sprite lateral; aquí se ajusta cómo es cada parte a lo ancho. Vacío = automático.' }),
    ...rows,
    el('div', { class: 'parts-actions' }, rebuild, toggle),
    area,
    typeof report === 'string' && report ? el('pre', { class: 'parts-report', text: report }) : null);
}

async function apply(area) {
  if (busy) return;
  const m = ctx.current();
  const plan = planOf(m);
  if (!plan) return;
  const args = { model_id: m.id };
  if (!area.hidden) {
    try { args.plan = JSON.parse(area.value); } catch (err) { toast('El plan no es JSON válido: ' + err.message, 'err'); return; }
  } else {
    const set = {};
    for (const input of box.querySelectorAll('[data-part]')) {
      const part = plan.parts.find((p) => p.name === input.dataset.part);
      if (!part) continue;
      const key = input.dataset.key;
      let value = input.value;
      const choice = input.tagName === 'SELECT';
      if (!choice) value = value === '' ? null : Number(value);
      const before = part[key] === undefined ? (choice && key !== 'type' ? input.options[0].value : null) : part[key];
      if (value !== before && !(value === null && before === null)) (set[part.name] = set[part.name] || {})[key] = value;
    }
    if (!Object.keys(set).length) { toast('No hay cambios que aplicar.', 'ok'); return; }
    args.set = set;
  }
  busy = true;
  try {
    await ctx.flushSave();
    const res = await api('/api/ui/call', { method: 'POST', body: { name: 'model_parts_adjust', arguments: args } });
    await ctx.openModel(m.id, { keepCamera: true });
    render(res.report || '');
    toast('Modelo reconstruido.', 'ok');
  } catch (err) { fail(err); } finally { busy = false; }
}
