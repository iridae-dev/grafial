// Corpus round-trip: for EVERY repository example, regenerate each flow and
// rule from its program_structure() JSON via genFlow/genRule, splice the
// regenerated blocks back, and require identical execution results.
//
// This exercises the whole structural-editing path (Rust AST printer + wasm
// exposure + JS codegen + block splicing) against real programs.
//
// Also tests rename.js. Requires webapp/tests/pkg-node (see README); skips
// without it.

import { test } from 'node:test';
import assert from 'node:assert/strict';
import { createRequire } from 'node:module';
import { readdirSync, readFileSync } from 'node:fs';
import { join, dirname } from 'node:path';
import { fileURLToPath } from 'node:url';

import { genFlow, genRule } from '../js/codegen.js';
import { replaceBlock } from '../js/blockedit.js';
import { renameIdentifier, renameAll, countIdentifier } from '../js/rename.js';

const here = dirname(fileURLToPath(import.meta.url));
const examplesDir = join(here, '..', '..', 'crates', 'grafial-examples');

const require = createRequire(import.meta.url);
let wasm = null;
try {
  wasm = require('./pkg-node/grafial_wasm.js');
} catch { /* skip below */ }

const exampleFiles = readdirSync(examplesDir).filter((f) => f.endsWith('.grafial'));

function runAllFlows(source, structure) {
  const out = {};
  for (const flow of structure.flows) {
    out[flow.name] = wasm.run_flow(source, flow.name);
  }
  return out;
}

test('every example flow and rule regenerates losslessly', { skip: !wasm && 'pkg-node not built' }, () => {
  assert.ok(exampleFiles.length >= 10, 'examples corpus present');
  for (const file of exampleFiles) {
    const original = readFileSync(join(examplesDir, file), 'utf8');
    const structure = JSON.parse(wasm.program_structure(original));

    let regenerated = original;
    for (const flow of structure.flows) {
      regenerated = replaceBlock(regenerated, 'flow', flow.name, genFlow(flow));
    }
    for (const rule of structure.rules) {
      regenerated = replaceBlock(regenerated, 'rule', rule.name, genRule(rule));
    }

    const check = JSON.parse(wasm.check(regenerated));
    assert.equal(check.valid, true, `${file}: regenerated program invalid: ${check.error}`);
    assert.deepEqual(check.style_lints, [], `${file}: regenerated code must lint clean`);

    assert.deepEqual(
      runAllFlows(regenerated, structure),
      runAllFlows(original, structure),
      `${file}: execution results changed after flow/rule regeneration`,
    );
  }
});

test('renameIdentifier skips strings and comments, respects word boundaries', () => {
  const src = `// Person comment mentioning Person
schema S {
  node Person { value: Real } /* Person */
  edge KNOWS { }
}
evidence Ev on M {
  Person { "Person" { value: 1.0 } }
  KNOWS(Person -> Person) { "Person" -> "Personal" }
}
rule R on M {
  for (P:Person) where E[P.value] > 0.0 => {
    non_bayesian_nudge P.value to E[P.value] variance=preserve
  }
}
`;
  assert.equal(countIdentifier(src, 'Person'), 5); // schema node, evidence header, group header x2, rule pattern
  const { source: out, count } = renameIdentifier(src, 'Person', 'Human');
  assert.equal(count, 5);
  assert.ok(out.includes('node Human { value: Real }'));
  assert.ok(out.includes('KNOWS(Human -> Human)'));
  assert.ok(out.includes('(P:Human)'));
  // Strings and comments untouched; 'Personal' not clipped.
  assert.ok(out.includes('"Person" { value: 1.0 }'));
  assert.ok(out.includes('"Person" -> "Personal"'));
  assert.ok(out.includes('// Person comment mentioning Person'));
  assert.ok(out.includes('/* Person */'));
});

test('renameAll applies simultaneous renames without chaining', () => {
  const { source: out } = renameAll('a b c', [
    { from: 'a', to: 'b' },
    { from: 'b', to: 'c' },
  ]);
  assert.equal(out, 'b c c'); // 'a' never becomes 'c'
});

test('rename cascade keeps a real program valid and semantics intact',
  { skip: !wasm && 'pkg-node not built' }, () => {
  const original = readFileSync(join(examplesDir, 'social.grafial'), 'utf8');
  const structure = JSON.parse(wasm.program_structure(original));
  const flowName = structure.flows[0].name;
  const before = wasm.run_flow(original, flowName);

  // Rename the node type and an attribute everywhere.
  const { source: renamed } = renameAll(original, [
    { from: 'Person', to: 'Human' },
    { from: 'some_value', to: 'wealth' },
  ]);
  const check = JSON.parse(wasm.check(renamed));
  assert.equal(check.valid, true, check.error ?? '');
  // Same numbers, since only names changed.
  const after = wasm.run_flow(renamed, flowName);
  assert.deepEqual(JSON.parse(after).metrics, JSON.parse(before).metrics);
});
