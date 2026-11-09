// Small helpers shared by every module: requests, toasts, dialogs and formatting.

export const $ = (id) => document.getElementById(id);

export function el(tag, attrs = {}, ...children) {
  const node = document.createElement(tag);
  for (const [k, v] of Object.entries(attrs || {})) {
    if (v === undefined || v === null || v === false) continue;
    if (k === 'class') node.className = v;
    else if (k === 'text') node.textContent = v;
    else if (k === 'html') node.innerHTML = v;
    else if (k.startsWith('on') && typeof v === 'function') node.addEventListener(k.slice(2), v);
    else if (k === 'dataset') Object.assign(node.dataset, v);
    else if (v === true) node.setAttribute(k, '');
    else node.setAttribute(k, v);
  }
  for (const c of children.flat()) {
    if (c === null || c === undefined || c === false) continue;
    node.append(c instanceof Node ? c : document.createTextNode(String(c)));
  }
  return node;
}

export class ApiError extends Error {
  constructor(message, hint = '', code = '', status = 0) {
    super(message);
    this.hint = hint; this.code = code; this.status = status;
  }
}

// JSON in and out; `raw` sends a Blob/ArrayBuffer as the body with its own content type.
export async function api(path, opts = {}) {
  const init = { method: opts.method || 'GET', headers: {} };
  if (opts.raw !== undefined) {
    init.body = opts.raw;
    init.headers['Content-Type'] = opts.type || 'application/octet-stream';
  } else if (opts.body !== undefined) {
    init.headers['Content-Type'] = 'application/json';
    init.body = JSON.stringify(opts.body);
  }
  if (opts.keepalive) init.keepalive = true;
  let res;
  try {
    res = await fetch(path, init);
  } catch (err) {
    throw new ApiError('No se puede contactar con el servidor.', '¿Sigue abierto Seurat\'s Hoard?', 'network');
  }
  const text = await res.text();
  let data = {};
  try { data = text ? JSON.parse(text) : {}; } catch { data = { error: text.slice(0, 300) }; }
  if (!res.ok) throw new ApiError(data.error || `Error ${res.status}`, data.hint || '', data.code || '', res.status);
  return data;
}

export function toast(text, kind = '', ms) {
  const box = $('toasts');
  const node = el('div', { class: `toast ${kind}`, role: kind === 'err' ? 'alert' : 'status' }, text);
  box.appendChild(node);
  while (box.children.length > 4) box.firstChild.remove();
  setTimeout(() => node.remove(), ms || (kind === 'err' ? 7000 : 3200));
}

export function fail(err) {
  console.warn(err);
  const msg = err && err.message ? err.message : String(err);
  toast(err && err.hint ? `${msg} — ${err.hint}` : msg, 'err');
}

// ------------------------------------------------------------------ dialogs
// A modal built on <dialog>: `body` is a node (or nodes), `buttons` [{label, value, kind}]; resolves the value
// (or null on Escape / Cancelar). `onOpen(dialog)` runs after it is shown.
export function modal({ title, body, buttons = [{ label: 'Cerrar', value: null }], wide = false, onOpen, validate }) {
  return new Promise((resolve) => {
    const form = el('form', { method: 'dialog', class: 'dlg-form' });
    form.append(el('h2', { text: title }));
    const content = el('div', { class: 'dlg-body' });
    for (const n of [].concat(body || [])) if (n) content.append(n);
    form.append(content);
    const actions = el('div', { class: 'dlg-actions' });
    for (const b of buttons) {
      actions.append(el('button', { class: `btn ${b.kind || ''}`, value: b.value === null ? '__null' : String(b.value), type: 'submit' }, b.label));
    }
    form.append(actions);
    const dlg = el('dialog', { class: `dialog${wide ? ' wide' : ''}` }, form);
    document.body.append(dlg);
    let result = null;
    form.addEventListener('submit', (ev) => {
      const v = ev.submitter ? ev.submitter.value : '__null';
      if (v !== '__null' && validate) {
        const problem = validate(form, v);
        if (problem) { ev.preventDefault(); toast(problem, 'err'); return; }
      }
      result = v === '__null' ? null : v;
    });
    dlg.addEventListener('close', () => { dlg.remove(); resolve(result === null ? null : { value: result, form }); });
    dlg.showModal();
    if (onOpen) onOpen(dlg);
  });
}

export async function confirmBox(title, text, okLabel = 'Aceptar', danger = false) {
  const r = await modal({ title, body: el('p', { text }), buttons: [
    { label: 'Cancelar', value: null }, { label: okLabel, value: 'ok', kind: danger ? 'danger-solid' : 'primary' }] });
  return !!r;
}

export async function askText(title, label, value = '', okLabel = 'Aceptar') {
  const input = el('input', { type: 'text', name: 'v', value, maxlength: '120', required: true, autocomplete: 'off' });
  const r = await modal({ title, body: el('label', { class: 'field' }, el('span', { text: label }), input),
    buttons: [{ label: 'Cancelar', value: null }, { label: okLabel, value: 'ok', kind: 'primary' }],
    onOpen: () => { input.focus(); input.select(); },
    validate: () => (input.value.trim() ? '' : 'Escribe un nombre.') });
  return r ? input.value.trim() : null;
}

// ------------------------------------------------------------------ formatting
export const nf = new Intl.NumberFormat('es-ES');

export function fmtDate(ts) {
  if (!ts) return '';
  const d = new Date(ts * 1000);
  const today = new Date();
  const hm = d.toLocaleTimeString('es-ES', { hour: '2-digit', minute: '2-digit', second: '2-digit' });
  return d.toDateString() === today.toDateString() ? `hoy ${hm}` : `${d.toLocaleDateString('es-ES')} ${hm}`;
}

export function download(url) {
  const a = el('a', { href: url, download: '' });
  document.body.append(a);
  a.click();
  a.remove();
}

export function store(key, value) {
  try {
    if (value === undefined) return JSON.parse(localStorage.getItem(`seurat.${key}`) || 'null');
    localStorage.setItem(`seurat.${key}`, JSON.stringify(value));
  } catch { /* private mode */ }
  return null;
}

export function isTyping(ev) {
  const t = ev.target;
  if (!t) return false;
  if (t.tagName === 'INPUT') return !['color', 'range', 'checkbox', 'radio', 'button', 'file'].includes(t.type);
  return t.tagName === 'TEXTAREA' || t.tagName === 'SELECT' || t.isContentEditable;
}
