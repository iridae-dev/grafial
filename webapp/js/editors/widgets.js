// Tiny DOM helpers shared by inspector editors.

export function el(parent, tag, className) {
  const node = document.createElement(tag);
  if (className) node.className = className;
  parent.appendChild(node);
  return node;
}

/** Labeled text input; returns the input element. */
export function field(parent, label, value) {
  const wrap = el(parent, 'div', 'field');
  el(wrap, 'label').textContent = label;
  const input = el(wrap, 'input');
  input.value = value ?? '';
  return input;
}

/** Section header with an optional "+" add button. */
export function sectionHeader(parent, title, onAdd) {
  const h = el(parent, 'div', 'section-h');
  const span = el(h, 'span');
  span.textContent = title;
  if (onAdd) {
    const btn = el(h, 'button');
    btn.textContent = '+';
    btn.addEventListener('click', onAdd);
  }
  return h;
}

export function rowButton(parent, text, onClick) {
  const btn = el(parent, 'button');
  btn.textContent = text;
  btn.addEventListener('click', onClick);
  return btn;
}

/** Apply / Delete button bar. */
export function buttonBar(parent, { apply, remove, applyLabel = 'Apply' }) {
  const bar = el(parent, 'div', 'btn-bar');
  if (apply) {
    const btn = el(bar, 'button', 'primary');
    btn.textContent = applyLabel;
    btn.addEventListener('click', () => {
      try { apply(); } catch (err) { showError(parent, err); }
    });
  }
  if (remove) {
    const btn = el(bar, 'button', 'danger');
    btn.textContent = 'Delete';
    btn.addEventListener('click', remove);
  }
  return bar;
}

export function showError(parent, err) {
  let note = parent.querySelector(':scope > .error-note');
  if (!note) note = el(parent, 'div', 'error-note');
  note.textContent = String(err?.message ?? err);
}

export function clearError(parent) {
  parent.querySelector(':scope > .error-note')?.remove();
}

/** Numeric input that parses on read; empty string means null. */
export function readNumber(input, what) {
  const raw = input.value.trim();
  if (raw === '') return null;
  const v = Number(raw);
  if (!Number.isFinite(v)) throw new Error(`${what}: not a number: '${raw}'`);
  return v;
}
