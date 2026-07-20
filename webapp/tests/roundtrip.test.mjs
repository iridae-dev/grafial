// Round-trip test through the real engine: codegen output must parse,
// validate, lint clean, and execute via the wasm API.
//
// Requires a nodejs-target wasm build:
//   wasm-pack build crates/grafial-wasm --target nodejs --out-dir ../../webapp/tests/pkg-node
// Skips (with a note) when the package is absent, so `node --test` stays
// runnable without the wasm toolchain.

import { test } from 'node:test';
import assert from 'node:assert/strict';
import { createRequire } from 'node:module';

import { genSchema, genBeliefModel, genEvidence, evidenceDocFromStructure, starterProgram } from '../js/codegen.js';
import { replaceBlock } from '../js/blockedit.js';

const require = createRequire(import.meta.url);
let wasm = null;
try {
  wasm = require('./pkg-node/grafial_wasm.js');
} catch {
  // not built; tests below skip
}

test('generated program validates, lints clean, and runs', { skip: !wasm && 'pkg-node not built' }, () => {
  const source = [
    genSchema({
      name: 'S',
      nodes: [{ name: 'Person', attrs: [{ name: 'score', type: 'Real' }] }],
      edges: ['REL'],
    }),
    genBeliefModel({
      name: 'M', on_schema: 'S',
      nodes: [{
        node_type: 'Person',
        attrs: [{ name: 'score', posterior: { family: 'gaussian', params: { prior_mean: 0, prior_precision: 1 } } }],
      }],
      edges: [{ edge_type: 'REL', exist: { family: 'bernoulli', params: { prior: 0.5, pseudo_count: 2 } } }],
    }),
    genEvidence({
      name: 'Ev', on_model: 'M',
      nodeRows: [
        { type: 'Person', name: 'Alice', values: { score: { value: 1, precision: 10 } } },
        { type: 'Person', name: 'Bob', values: { score: { value: 3, precision: null } } },
      ],
      edgeRows: [
        { edge_type: 'REL', src_type: 'Person', src: 'Alice', dst_type: 'Person', dst: 'Bob', mode: 'present' },
        { edge_type: 'REL', src_type: 'Person', src: 'Bob', dst_type: 'Person', dst: 'Alice', mode: 'absent' },
      ],
      weightRows: [],
    }),
    `flow Demo on M {
  graph g = from_evidence Ev
  metric total = nodes(Person) |> sum(by=E[node.score])
  export g as "out"
}`,
  ].join('\n\n') + '\n';

  const check = JSON.parse(wasm.check(source));
  assert.equal(check.valid, true, check.error ?? '');
  assert.deepEqual(check.style_lints, [], 'generated code must be canonical-style clean');

  const result = JSON.parse(wasm.run_flow(source, 'Demo'));
  // Alice: (0*1 + 10*1)/(1+10) = 0.909..., Bob: (0 + 3)/2 = 1.5
  assert.ok(Math.abs(result.metrics.total - (10 / 11 + 1.5)) < 1e-9);
  const names = result.exports.out.nodes.map((n) => n.name).sort();
  assert.deepEqual(names, ['Alice', 'Bob']);
});

test('evidence spreadsheet round-trip: structure -> doc -> codegen -> same posteriors',
  { skip: !wasm && 'pkg-node not built' }, () => {
  const original = starterProgram();
  const structure = JSON.parse(wasm.program_structure(original));
  const evidence = structure.evidences[0];

  // Rebuild the evidence block from its structured observations.
  const doc = evidenceDocFromStructure(evidence);
  const regenerated = replaceBlock(original, 'evidence', evidence.name, genEvidence(doc));

  const check = JSON.parse(wasm.check(regenerated));
  assert.equal(check.valid, true, check.error ?? '');

  // Same flow, same posteriors: the round-trip must be semantically lossless.
  const flow = structure.flows[0].name;
  const before = JSON.parse(wasm.run_flow(original, flow));
  const after = JSON.parse(wasm.run_flow(regenerated, flow));
  assert.deepEqual(after.metrics, before.metrics);
  assert.deepEqual(after.exports, before.exports);
});
