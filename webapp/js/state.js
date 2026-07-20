// Application state. The Grafial source text is the single source of truth;
// everything else (structure, diagnostics) derives from it via the wasm API.

import { check, programStructure } from './wasmapi.js';

const AUTOSAVE_KEY = 'grafial.composer.source';

export const state = {
  source: '',
  /** Last successful program_structure() result (kept through invalid edits). */
  structure: null,
  /** Latest check() result for the current source. */
  diagnostics: null,
  /** Current inspector selection: {kind, name} | {kind: 'result-node', ...} | null */
  selection: null,
  /** Latest run result: {flow, result} | null */
  run: null,
  undo: [],
  redo: [],
};

const listeners = new Set();

/** Subscribes to state changes. Events: 'source' | 'selection' | 'run'. */
export function subscribe(fn) {
  listeners.add(fn);
  return () => listeners.delete(fn);
}

function emit(event) {
  for (const fn of listeners) fn(event, state);
}

function revalidate() {
  state.diagnostics = check(state.source);
  if (state.diagnostics.valid) {
    state.structure = programStructure(state.source);
  }
}

/**
 * Replaces the program source. Set record=false for undo/redo replays and
 * transient typing (the source editor records undo on commit instead).
 */
export function setSource(text, { record = true, resetSelection = false } = {}) {
  if (text === state.source) return;
  if (record) {
    state.undo.push(state.source);
    if (state.undo.length > 200) state.undo.shift();
    state.redo.length = 0;
  }
  state.source = text;
  revalidate();
  if (resetSelection) state.selection = null;
  try { localStorage.setItem(AUTOSAVE_KEY, text); } catch { /* storage full/blocked */ }
  emit('source');
}

export function undo() {
  if (state.undo.length === 0) return;
  state.redo.push(state.source);
  state.source = state.undo.pop();
  revalidate();
  try { localStorage.setItem(AUTOSAVE_KEY, state.source); } catch { /* ignore */ }
  emit('source');
}

export function redo() {
  if (state.redo.length === 0) return;
  state.undo.push(state.source);
  state.source = state.redo.pop();
  revalidate();
  try { localStorage.setItem(AUTOSAVE_KEY, state.source); } catch { /* ignore */ }
  emit('source');
}

export function select(selection) {
  state.selection = selection;
  emit('selection');
}

export function setRun(run) {
  state.run = run;
  emit('run');
}

/** Initial load: autosaved program if present, else the provided fallback. */
export function loadInitial(fallback) {
  let text = null;
  try { text = localStorage.getItem(AUTOSAVE_KEY); } catch { /* ignore */ }
  state.source = text || fallback;
  revalidate();
  emit('source');
}

/** Looks up helpers on the last-good structure. */
export function findSchema(name) {
  return state.structure?.schemas.find((s) => s.name === name) ?? null;
}
export function findModel(name) {
  return state.structure?.belief_models.find((m) => m.name === name) ?? null;
}
export function findEvidence(name) {
  return state.structure?.evidences.find((e) => e.name === name) ?? null;
}
export function findFlow(name) {
  return state.structure?.flows.find((f) => f.name === name) ?? null;
}
/** Schema backing a belief model name (or null). */
export function schemaForModel(modelName) {
  const model = findModel(modelName);
  return model ? findSchema(model.on_schema) : null;
}
