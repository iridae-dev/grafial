// Composer bootstrap and wiring.

import { initWasm, buildIdentity } from './wasmapi.js';
import { state, subscribe, setSource, undo, redo, loadInitial, select } from './state.js';
import { starterProgram } from './codegen.js';
import { renderMap } from './map.js';
import { initInspector, renderInspector } from './inspector.js';
import { initRun, refreshFlowPicker, renderResults } from './run.js';

const EXAMPLES = [
  'minimal', 'social', 'ab_testing', 'advanced_metrics', 'common_mistakes',
  'competing_choices', 'pipeline_composition', 'prior_sensitivity',
  'probabilistic_pattern_matching', 'soft_vs_hard_updates',
  'transitive_closure', 'uncertainty_propagation',
];

async function main() {
  let version;
  try {
    version = await initWasm();
  } catch (err) {
    console.error(err);
    document.getElementById('boot-error').hidden = false;
    return;
  }

  initAbout(version);
  initInspector();
  initTabs();
  initToolbar();
  initSourcePane();
  initRun({ onRan: () => { activateTab('results'); renderAll('run'); } });

  subscribe((event) => renderAll(event));
  loadInitial(starterProgram());
}


function initAbout(version) {
  const identity = buildIdentity();
  const short = `${identity.grafial_version || version} · ${(identity.git_commit || 'unknown').slice(0, 7)}`;
  const buildEl = document.getElementById('status-build');
  if (buildEl) buildEl.textContent = short;
  const dialog = document.getElementById('about-dialog');
  const pre = document.getElementById('about-identity');
  const btn = document.getElementById('btn-about');
  if (pre) pre.textContent = JSON.stringify(identity, null, 2);
  if (btn && dialog) {
    btn.addEventListener('click', () => dialog.showModal());
  }
}

// --- rendering -----------------------------------------------------------------

let activePane = 'program';

function renderAll(event) {
  if (event === 'source') {
    renderMap(document.getElementById('pane-program'));
    syncSourcePane();
    refreshFlowPicker();
    renderStatus();
    renderUndoButtons();
  }
  if (event === 'selection') {
    renderInspector();
    renderMap(document.getElementById('pane-program'));
    if (state.selection?.kind === 'result-node') renderResults(document.getElementById('pane-results'));
  }
  if (event === 'run') {
    renderResults(document.getElementById('pane-results'));
  }
}

function renderStatus() {
  const validity = document.getElementById('status-validity');
  const lints = document.getElementById('status-lints');
  const d = state.diagnostics;
  if (!d) return;
  validity.className = d.valid ? 'ok' : 'err';
  validity.textContent = d.valid ? '✓ valid' : `✗ ${firstLine(d.error)}`;
  const count = d.style_lints.length + d.statistical_lints.length;
  lints.textContent = count > 0 ? `${count} lint${count === 1 ? '' : 's'}` : '';
}

function renderUndoButtons() {
  document.getElementById('btn-undo').disabled = state.undo.length === 0;
  document.getElementById('btn-redo').disabled = state.redo.length === 0;
}

function firstLine(s) {
  return String(s ?? '').split('\n')[0].slice(0, 120);
}

// --- tabs -----------------------------------------------------------------------

function initTabs() {
  for (const tab of document.querySelectorAll('#tabs .tab')) {
    tab.addEventListener('click', () => activateTab(tab.dataset.pane));
  }
}

function activateTab(name) {
  activePane = name;
  for (const tab of document.querySelectorAll('#tabs .tab')) {
    tab.classList.toggle('active', tab.dataset.pane === name);
  }
  for (const pane of document.querySelectorAll('.pane')) {
    pane.classList.toggle('active', pane.id === `pane-${name}`);
  }
  if (name === 'program') renderMap(document.getElementById('pane-program'));
}

// --- source pane -----------------------------------------------------------------

function initSourcePane() {
  const textarea = document.getElementById('source-text');
  let debounce = null;
  textarea.addEventListener('input', () => {
    clearTimeout(debounce);
    debounce = setTimeout(() => setSource(textarea.value), 400);
  });
}

function syncSourcePane() {
  const textarea = document.getElementById('source-text');
  if (document.activeElement !== textarea) textarea.value = state.source;
  const host = document.getElementById('diagnostics');
  host.innerHTML = '';
  const d = state.diagnostics;
  if (!d) return;
  if (!d.valid) diag(host, 'err', d.error);
  for (const l of d.style_lints) diag(host, 'warn', `style ${l.code} @ ${l.range.start.line}:${l.range.start.column} — ${l.message}`);
  for (const l of d.statistical_lints) diag(host, l.severity === 'warning' ? 'warn' : 'info', `${l.code} @ ${l.range.start.line}:${l.range.start.column} — ${l.message}`);
}

function diag(host, cls, text) {
  const div = document.createElement('div');
  div.className = `diag ${cls}`;
  div.textContent = text;
  host.appendChild(div);
}

// --- toolbar ---------------------------------------------------------------------

function initToolbar() {
  document.getElementById('btn-new').addEventListener('click', () => {
    if (!confirm('Start a new program? Unsaved changes stay in undo history.')) return;
    setSource(starterProgram(), { resetSelection: true });
    select(null);
    activateTab('program');
  });

  document.getElementById('btn-undo').addEventListener('click', undo);
  document.getElementById('btn-redo').addEventListener('click', redo);
  document.addEventListener('keydown', (e) => {
    if ((e.metaKey || e.ctrlKey) && e.key.toLowerCase() === 'z') {
      e.preventDefault();
      if (e.shiftKey) redo(); else undo();
    }
  });

  // Open: File System Access API when available, else the hidden input.
  document.getElementById('btn-open').addEventListener('click', async () => {
    if (window.showOpenFilePicker) {
      try {
        const [handle] = await window.showOpenFilePicker({
          types: [{ description: 'Grafial', accept: { 'text/plain': ['.grafial'] } }],
        });
        const file = await handle.getFile();
        openText(await file.text());
      } catch { /* cancelled */ }
    } else {
      document.getElementById('file-open').click();
    }
  });
  document.getElementById('file-open').addEventListener('change', async (e) => {
    const file = e.target.files[0];
    e.target.value = '';
    if (file) openText(await file.text());
  });

  document.getElementById('btn-save').addEventListener('click', async () => {
    if (window.showSaveFilePicker) {
      try {
        const handle = await window.showSaveFilePicker({
          suggestedName: 'program.grafial',
          types: [{ description: 'Grafial', accept: { 'text/plain': ['.grafial'] } }],
        });
        const writable = await handle.createWritable();
        await writable.write(state.source);
        await writable.close();
        return;
      } catch { return; /* cancelled */ }
    }
    const blob = new Blob([state.source], { type: 'text/plain' });
    const a = document.createElement('a');
    a.href = URL.createObjectURL(blob);
    a.download = 'program.grafial';
    a.click();
    URL.revokeObjectURL(a.href);
  });

  // Repository examples (available when served from the repo root).
  const examplesSel = document.getElementById('sel-example');
  for (const name of EXAMPLES) {
    const opt = document.createElement('option');
    opt.value = name;
    opt.textContent = name;
    examplesSel.appendChild(opt);
  }
  examplesSel.addEventListener('change', async () => {
    const name = examplesSel.value;
    examplesSel.value = '';
    if (!name) return;
    try {
      // Bundled location first (static deploys); repo location as fallback
      // for ad-hoc dev servers pointed at the repository root.
      let resp = await fetch(`examples/${name}.grafial`);
      if (!resp.ok) resp = await fetch(`../crates/grafial-examples/${name}.grafial`);
      if (!resp.ok) throw new Error(`HTTP ${resp.status}`);
      openText(await resp.text());
    } catch (err) {
      alert(`Could not load example '${name}'. ${err}`);
    }
  });
}

function openText(text) {
  setSource(text, { resetSelection: true });
  select(null);
  activateTab('program');
}

main();
