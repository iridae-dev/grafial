// Thin wrapper around the wasm-pack package (web target, built into
// webapp/pkg by scripts/serve_composer.sh). All results come back parsed.

let mod = null;

export async function initWasm() {
  mod = await import('../pkg/grafial_wasm.js');
  await mod.default();
  return mod.version();
}

export function check(source) {
  return JSON.parse(mod.check(source));
}

export function programStructure(source) {
  return JSON.parse(mod.program_structure(source));
}

export function runFlow(source, flowName) {
  return JSON.parse(mod.run_flow(source, flowName));
}

export function formatCanonical(source) {
  return mod.format_canonical(source);
}
