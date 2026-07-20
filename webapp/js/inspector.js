// Dockable, context-sensitive inspector. Renders the right editor for the
// current selection: form editors for schema/model, a spreadsheet for
// evidence, structural editors for rules and flows, and a posterior detail
// view for result-graph nodes.

import { state, select, setSource } from './state.js';
import { check } from './wasmapi.js';
import { replaceBlock, removeBlock, findDeclaration } from './blockedit.js';
import { renameAll } from './rename.js';
import { renderSchemaEditor } from './editors/schema.js';
import { renderModelEditor } from './editors/model.js';
import { renderEvidenceEditor } from './editors/evidence.js';
import { renderFlowEditor } from './editors/flow.js';
import { renderRuleEditor } from './editors/rule.js';
import { renderResultNode } from './editors/resultnode.js';

const KIND_TITLES = {
  schema: 'Schema', belief_model: 'Belief Model', evidence: 'Evidence',
  rule: 'Rule', flow: 'Flow', 'result-node': 'Node',
};

export function initInspector() {
  const layout = document.getElementById('layout');
  const inspector = document.getElementById('inspector');
  const divider = document.getElementById('divider');

  document.getElementById('inspector-close').addEventListener('click', () => select(null));
  document.getElementById('btn-dock').addEventListener('click', () => {
    layout.classList.toggle('dock-left');
    layout.classList.toggle('dock-right');
  });

  // Drag-resize. The divider sits between the panes and the inspector; when
  // docked left the drag direction inverts.
  divider.addEventListener('pointerdown', (down) => {
    down.preventDefault();
    divider.setPointerCapture(down.pointerId);
    const startX = down.clientX;
    const startW = inspector.getBoundingClientRect().width;
    const docked = layout.classList.contains('dock-left') ? -1 : 1;
    const move = (e) => {
      const w = Math.max(260, Math.min(window.innerWidth * 0.7, startW + docked * (startX - e.clientX)));
      document.documentElement.style.setProperty('--inspector-w', `${w}px`);
    };
    const up = () => {
      divider.removeEventListener('pointermove', move);
      divider.removeEventListener('pointerup', up);
    };
    divider.addEventListener('pointermove', move);
    divider.addEventListener('pointerup', up);
  });
}

export function renderInspector() {
  const inspector = document.getElementById('inspector');
  const title = document.getElementById('inspector-title');
  const body = document.getElementById('inspector-body');
  const sel = state.selection;

  if (!sel) {
    inspector.classList.add('collapsed');
    return;
  }
  inspector.classList.remove('collapsed');
  title.textContent = `${KIND_TITLES[sel.kind] ?? sel.kind}: ${sel.name ?? ''}`;
  body.innerHTML = '';

  const ctx = {
    commitBlock: (kind, oldName, newText, newSelection, opts) =>
      commit(kind, oldName, newText, newSelection, { validate: false, ...opts }),
    commitValidated: (kind, oldName, newText, newSelection, opts) =>
      commit(kind, oldName, newText, newSelection, { validate: true, ...opts }),
    deleteBlock,
    blockText,
    refresh: renderInspector,
  };

  try {
    if (sel.kind === 'schema') renderSchemaEditor(body, sel.name, ctx);
    else if (sel.kind === 'belief_model') renderModelEditor(body, sel.name, ctx);
    else if (sel.kind === 'evidence') renderEvidenceEditor(body, sel.name, ctx);
    else if (sel.kind === 'rule') renderRuleEditor(body, sel.name, ctx);
    else if (sel.kind === 'flow') renderFlowEditor(body, sel.name, ctx);
    else if (sel.kind === 'result-node') renderResultNode(body, sel);
    else body.innerHTML = '<div class="hint">Nothing to show.</div>';
  } catch (err) {
    const note = document.createElement('div');
    note.className = 'error-note';
    note.textContent = String(err.message ?? err);
    body.appendChild(note);
  }
}

/** Current text of a declaration block (or null if missing). */
function blockText(kind, name) {
  const decl = findDeclaration(state.source, kind, name);
  return decl ? state.source.slice(decl.start, decl.end) : null;
}

/**
 * Replaces a declaration with regenerated text, optionally cascading renames
 * through the rest of the program, then re-renders.
 *
 * - validate: pre-parse the candidate and refuse to commit on parse errors
 *   (used by editors with free-text expressions). Semantic errors never block
 *   — they surface in the status bar, since source is truth even mid-error.
 * - renames: [{from, to}] identifier renames applied program-wide (outside
 *   strings/comments) so references follow, with a confirm listing the count.
 */
function commit(kind, oldName, newText, newSelection, { validate = false, renames = [] } = {}) {
  let candidate = replaceBlock(state.source, kind, oldName, newText);

  const effective = renames.filter((r) => r.from && r.to && r.from !== r.to);
  if (effective.length > 0) {
    const { source: renamed, count } = renameAll(candidate, effective);
    if (count > 0) {
      const summary = effective.map((r) => `${r.from} → ${r.to}`).join(', ');
      if (!confirm(`Rename ${summary} across ${count} occurrence${count === 1 ? '' : 's'} in the rest of the program?`)) {
        return; // user declined the cascade; nothing committed
      }
      candidate = renamed;
    }
  }

  if (validate) {
    const diag = check(candidate);
    if (!diag.valid) {
      throw new Error(`Not applied — program would not parse:\n${diag.error}`);
    }
  }

  setSource(candidate);
  if (newSelection) select(newSelection);
  else renderInspector();
}

function deleteBlock(kind, name) {
  if (!confirm(`Delete ${kind.replace('_', ' ')} '${name}'?`)) return;
  setSource(removeBlock(state.source, kind, name));
  select(null);
}
