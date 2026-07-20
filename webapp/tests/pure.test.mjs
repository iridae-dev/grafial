// Headless tests for the composer's pure modules. Run: node --test webapp/tests/
import { test } from 'node:test';
import assert from 'node:assert/strict';

import {
  findDeclarations, findDeclaration, replaceBlock, removeBlock, appendDeclaration,
} from '../js/blockedit.js';
import {
  num, genSchema, genBeliefModel, genEvidence, evidenceDocFromStructure, starterProgram,
} from '../js/codegen.js';
import { parseCSV, csvToNodeRows, parseCell } from '../js/csv.js';

const SOURCE = `// leading comment with a stray { brace
schema S {
  node Person { score: Real }
  edge REL { }
}

/* block comment { */
belief_model M on S {
  node Person { score ~ Gaussian(mean=0.0, precision=1.0) }
  edge REL { exist ~ Bernoulli(prior=0.5, weight=2.0) }
}

evidence Ev on M {
  Person { "Al{ice" { score: 1.0 } }
  REL(Person -> Person) { "Al{ice" -> "Al{ice" }
}

flow Demo on M {
  graph g = from_evidence Ev
  export g as "out"
}
`;

test('findDeclarations sees all blocks, unfazed by braces in comments/strings', () => {
  const decls = findDeclarations(SOURCE);
  assert.deepEqual(
    decls.map((d) => [d.kind, d.name, d.on]),
    [
      ['schema', 'S', null],
      ['belief_model', 'M', 'S'],
      ['evidence', 'Ev', 'M'],
      ['flow', 'Demo', 'M'],
    ],
  );
  // Block spans reproduce the declaration text exactly.
  const ev = decls[2];
  const text = SOURCE.slice(ev.start, ev.end);
  assert.ok(text.startsWith('evidence Ev on M {'));
  assert.ok(text.endsWith('}'));
});

test('replaceBlock swaps exactly one declaration', () => {
  const out = replaceBlock(SOURCE, 'evidence', 'Ev', 'evidence Ev on M {\n}');
  assert.ok(out.includes('evidence Ev on M {\n}'));
  assert.ok(out.includes('belief_model M on S'));
  assert.ok(!out.includes('"Al{ice" { score: 1.0 }'));
  assert.equal(findDeclarations(out).length, 4);
});

test('removeBlock deletes a declaration', () => {
  const out = removeBlock(SOURCE, 'flow', 'Demo');
  assert.equal(findDeclaration(out, 'flow', 'Demo'), null);
  assert.equal(findDeclarations(out).length, 3);
});

test('appendDeclaration groups with the same kind', () => {
  const out = appendDeclaration(SOURCE, 'evidence', 'evidence Ev2 on M {\n}');
  const decls = findDeclarations(out);
  const kinds = decls.map((d) => d.kind);
  // New evidence lands directly after the existing one, before the flow.
  assert.deepEqual(kinds, ['schema', 'belief_model', 'evidence', 'evidence', 'flow']);
  assert.equal(decls[3].name, 'Ev2');
});

test('num renders round-trippable Real literals', () => {
  assert.equal(num(1), '1.0');
  assert.equal(num(0.5), '0.5');
  assert.equal(num(-3), '-3.0');
  assert.equal(num(1e-9), '1e-9');
  assert.throws(() => num(NaN));
});

test('genSchema emits a parseable block shape', () => {
  const text = genSchema({
    name: 'S',
    nodes: [{ name: 'Person', attrs: [{ name: 'score', type: 'Real' }] }],
    edges: ['REL'],
  });
  assert.ok(text.includes('schema S {'));
  assert.ok(text.includes('node Person {'));
  assert.ok(text.includes('score: Real'));
  assert.ok(text.includes('edge REL { }'));
});

test('genBeliefModel emits canonical parameter aliases', () => {
  const text = genBeliefModel({
    name: 'M', on_schema: 'S',
    nodes: [{
      node_type: 'Person',
      attrs: [{ name: 'score', posterior: { family: 'gaussian', params: { prior_mean: 0, prior_precision: 0.1 } } }],
    }],
    edges: [{ edge_type: 'REL', exist: { family: 'bernoulli', params: { prior: 0.5, pseudo_count: 2 } } }],
  });
  assert.ok(text.includes('score ~ Gaussian(mean=0.0, precision=0.1)'));
  assert.ok(text.includes('exist ~ Bernoulli(prior=0.5, weight=2.0)'));
});

test('genEvidence groups nodes and edges, emits choice verbs and weights', () => {
  const text = genEvidence({
    name: 'Ev', on_model: 'M',
    nodeRows: [
      { type: 'Person', name: 'Alice', values: { score: { value: 1, precision: 10 } } },
      { type: 'Person', name: 'Bob', values: { score: { value: 2, precision: null } } },
    ],
    edgeRows: [
      { edge_type: 'REL', src_type: 'Person', src: 'Alice', dst_type: 'Person', dst: 'Bob', mode: 'present' },
      { edge_type: 'REL', src_type: 'Person', src: 'Bob', dst_type: 'Person', dst: 'Alice', mode: 'absent' },
      { edge_type: 'ROUTES', src_type: 'Person', src: 'Alice', dst_type: 'Person', dst: 'Bob', mode: 'chosen' },
    ],
    weightRows: [
      { edge_type: 'REL', src_type: 'Person', src: 'Alice', dst_type: 'Person', dst: 'Bob', value: 2.5, precision: 3 },
    ],
  });
  assert.ok(text.includes('"Alice" { score: 1.0 (precision=10.0) }'));
  assert.ok(text.includes('"Bob" { score: 2.0 }'));
  assert.ok(text.includes('"Alice" -> "Bob"'));
  assert.ok(text.includes('"Bob" -/> "Alice"'));
  assert.ok(text.includes('choose edge ROUTES(Person["Alice"], Person["Bob"])'));
  assert.ok(text.includes('observe edge REL(Person["Alice"], Person["Bob"]) weight=2.5 (precision=3.0)'));
});

test('evidenceDocFromStructure packs repeated observations into repeated rows', () => {
  const doc = evidenceDocFromStructure({
    name: 'Ev', on_model: 'M',
    observations: [
      { kind: 'attribute', node_type: 'Person', node: 'Alice', attr: 'score', value: 1, precision: null },
      { kind: 'attribute', node_type: 'Person', node: 'Alice', attr: 'other', value: 5, precision: null },
      { kind: 'attribute', node_type: 'Person', node: 'Alice', attr: 'score', value: 2, precision: 4 },
      { kind: 'edge', edge_type: 'REL', src_type: 'Person', src: 'Alice', dst_type: 'Person', dst: 'Bob', mode: 'present' },
    ],
  });
  // First row holds score+other; the repeated score observation opens row 2.
  assert.equal(doc.nodeRows.length, 2);
  assert.deepEqual(doc.nodeRows[0].values.score, { value: 1, precision: null });
  assert.deepEqual(doc.nodeRows[0].values.other, { value: 5, precision: null });
  assert.deepEqual(doc.nodeRows[1].values, { score: { value: 2, precision: 4 } });
  assert.equal(doc.edgeRows.length, 1);
});

test('evidence codegen round-trips through evidenceDocFromStructure', () => {
  const doc = {
    name: 'Ev', on_model: 'M',
    nodeRows: [
      { type: 'Person', name: 'Alice', values: { score: { value: 1, precision: null } } },
      { type: 'Person', name: 'Alice', values: { score: { value: 2, precision: null } } },
    ],
    edgeRows: [], weightRows: [],
  };
  const text = genEvidence(doc);
  // Both repeated observations survive in the generated text.
  assert.equal((text.match(/"Alice"/g) || []).length, 2);
});

test('parseCSV handles quotes, commas, CRLF', () => {
  const rows = parseCSV('name,score\r\n"Smith, A",1.5\n"say ""hi""",2\n');
  assert.deepEqual(rows, [['name', 'score'], ['Smith, A', '1.5'], ['say "hi"', '2']]);
});

test('csvToNodeRows maps columns to attributes and rejects unknown ones', () => {
  const rows = parseCSV('name,score\nAlice,1.0\nBob,2.5 @ 10\nCarol,\n');
  const nodeRows = csvToNodeRows(rows, 'Person', ['score']);
  assert.equal(nodeRows.length, 3);
  assert.deepEqual(nodeRows[0], { type: 'Person', name: 'Alice', values: { score: { value: 1, precision: null } } });
  assert.deepEqual(nodeRows[1].values.score, { value: 2.5, precision: 10 });
  assert.deepEqual(nodeRows[2].values, {});

  assert.throws(
    () => csvToNodeRows(parseCSV('name,bogus\nA,1\n'), 'Person', ['score']),
    /unknown attribute column/,
  );
});

test('parseCell validates numbers and precision', () => {
  assert.deepEqual(parseCell('1.5', 2, 'x'), { value: 1.5, precision: null });
  assert.deepEqual(parseCell('1.5 @ 10', 2, 'x'), { value: 1.5, precision: 10 });
  assert.throws(() => parseCell('abc', 2, 'x'), /not a number/);
  assert.throws(() => parseCell('1 @ -2', 2, 'x'), /positive/);
});

test('starterProgram declares all four kinds', () => {
  const decls = findDeclarations(starterProgram());
  assert.deepEqual(decls.map((d) => d.kind), ['schema', 'belief_model', 'evidence', 'flow']);
});

// --- genFlow / genRule / forceLayout (added with the structural editors) -----

test('genFlow emits every construct', async () => {
  const { genFlow } = await import('../js/codegen.js');
  const text = genFlow({
    name: 'F', on_model: 'M',
    metric_imports: [{ source_alias: 'stats', local_name: 'prior_rate' }],
    graphs: [
      { name: 'base', expr: { kind: 'from_evidence', evidence: 'Ev' } },
      { name: 'imported', expr: { kind: 'from_graph', alias: 'other_graph' } },
      {
        name: 'refined',
        expr: {
          kind: 'pipeline', start: 'base',
          transforms: [
            { kind: 'apply_rule', rule: 'R1' },
            { kind: 'apply_ruleset', rules: ['R1', 'R2'] },
            { kind: 'infer_beliefs' },
            { kind: 'snapshot', name: 'mid' },
            { kind: 'prune_edges', edge_type: 'REL', predicate: 'prob(edge) < 0.1' },
          ],
        },
      },
      { name: 'best', expr: { kind: 'select_model', candidates: ['base', 'refined'], criterion: 'edge_bic' } },
    ],
    metrics: [{ name: 'total', expr: 'nodes(N) |> count()' }],
    exports: [{ graph: 'refined', alias: 'out' }],
    metric_exports: [{ metric: 'total', alias: 'total_stat' }],
  });
  assert.ok(text.includes('import_metric stats as prior_rate'));
  assert.ok(text.includes('graph base = from_evidence Ev'));
  assert.ok(text.includes('graph imported = from_graph "other_graph"'));
  assert.ok(text.includes('base |> apply_rule R1 |> apply_ruleset { R1, R2 } |> infer_beliefs |> snapshot "mid" |> prune_edges REL where prob(edge) < 0.1'));
  assert.ok(text.includes('graph best = select_model { base, refined } by edge_bic'));
  assert.ok(text.includes('metric total = nodes(N) |> count()'));
  assert.ok(text.includes('export refined as "out"'));
  assert.ok(text.includes('export_metric total as "total_stat"'));
});

test('genRule emits pattern form and for-sugar form', async () => {
  const { genRule } = await import('../js/codegen.js');
  const pattern = genRule({
    name: 'R', on_model: 'M',
    patterns: [{
      src: { var: 'A', label: 'N' }, edge: { var: 'ab', type: 'REL' }, dst: { var: 'B', label: 'N' },
    }],
    where: 'prob(ab) >= 0.5',
    actions: ['delete ab confidence=high'],
    mode: 'for_each',
  });
  assert.ok(pattern.includes('(A:N)-[ab:REL]->(B:N)'));
  assert.ok(pattern.includes('where'));
  assert.ok(pattern.includes('mode: for_each'));

  const sugar = genRule({
    name: 'R2', on_model: 'M',
    patterns: [{
      src: { var: 'P', label: 'N' }, edge: { var: '__for_dummy', type: '__FOR_NODE__' }, dst: { var: 'P', label: 'N' },
    }],
    where: 'E[P.x] < 0.0',
    actions: ['P.x ~= 0.0 precision=0.1'],
    mode: null,
  });
  assert.ok(sugar.includes('for (P:N) where E[P.x] < 0.0 => {'));
  assert.ok(!sugar.includes('__FOR_NODE__'), 'sentinel must not leak into source');
});

test('forceLayout is deterministic and stays in bounds', async () => {
  const { forceLayout } = await import('../js/run.js');
  const nodes = Array.from({ length: 8 }, (_, i) => ({ id: i }));
  const edges = [
    { src: 0, dst: 1, prob: 0.9 }, { src: 1, dst: 2, prob: 0.7 },
    { src: 0, dst: 3, prob: 0.8 }, { src: 4, dst: 5, prob: 0.5 },
  ];
  const a = forceLayout(nodes, edges, 640, 480);
  const b = forceLayout(nodes, edges, 640, 480);
  for (const n of nodes) {
    assert.deepEqual(a.get(n.id), b.get(n.id), 'layout must be deterministic');
    const p = a.get(n.id);
    assert.ok(p.x >= 0 && p.x <= 640 && p.y >= 0 && p.y <= 480, `node ${n.id} out of bounds`);
  }
  // Connected nodes end up nearer than the strangers 0 and 5.
  const d01 = Math.hypot(a.get(0).x - a.get(1).x, a.get(0).y - a.get(1).y);
  const d05 = Math.hypot(a.get(0).x - a.get(5).x, a.get(0).y - a.get(5).y);
  assert.ok(d01 < d05, 'edge-connected nodes should sit closer');
});
