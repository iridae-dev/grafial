// Structural flow editor: graphs with pipeline builders (add / remove /
// reorder transforms), metrics with expressions, exports, and cross-flow
// imports. Generates the flow block via genFlow.

import { state, findFlow, findModel, findSchema } from '../state.js';
import { genFlow } from '../codegen.js';
import { el, field, sectionHeader, rowButton, buttonBar, clearError } from './widgets.js';

const TRANSFORM_KINDS = ['apply_rule', 'apply_ruleset', 'infer_beliefs', 'prune_edges', 'snapshot'];

export function renderFlowEditor(body, name, ctx) {
  const flow = findFlow(name);
  if (!flow) { body.innerHTML = '<div class="hint">Not found in last valid program.</div>'; return; }

  // Deep working copy of the structure JSON (same shape genFlow consumes).
  const doc = JSON.parse(JSON.stringify(flow));
  const model = findModel(doc.on_model);
  const schema = model ? findSchema(model.on_schema) : null;
  const ruleNames = (state.structure?.rules ?? []).filter((r) => r.on_model === doc.on_model).map((r) => r.name);
  const evidenceNames = (state.structure?.evidences ?? []).filter((e) => e.on_model === doc.on_model).map((e) => e.name);
  const edgeTypes = schema?.edges ?? [];
  // Aliases other flows export (for from_graph / import_metric pickers).
  const graphAliases = [];
  const metricAliases = [];
  for (const f of state.structure?.flows ?? []) {
    if (f.name === flow.name) continue;
    graphAliases.push(...f.exports.map((e) => e.alias));
    metricAliases.push(...f.metric_exports.map((e) => e.alias));
  }

  const nameInput = field(body, 'Name', doc.name);
  el(body, 'div', 'hint').textContent = `on ${doc.on_model}`;
  const readers = []; // deferred input -> doc appliers

  // --- imports -----------------------------------------------------------------
  const importsHost = el(body, 'div');
  const renderImports = () => {
    importsHost.innerHTML = '';
    sectionHeader(importsHost, 'Metric imports', () => {
      doc.metric_imports.push({ source_alias: metricAliases[0] ?? 'alias', local_name: 'imported' });
      renderImports();
    });
    if (doc.metric_imports.length === 0) return;
    const table = el(importsHost, 'table', 'sheet');
    table.innerHTML = '<tr><th>Exported alias</th><th>Local name</th><th></th></tr>';
    for (const imp of doc.metric_imports) {
      const tr = el(table, 'tr');
      const aliasIn = comboCell(tr, metricAliases, imp.source_alias);
      const localIn = inputCell(tr, imp.local_name);
      readers.push(() => { imp.source_alias = aliasIn.value.trim(); imp.local_name = localIn.value.trim(); });
      rowButton(el(tr, 'td', 'row-actions'), '✕', () => {
        doc.metric_imports = doc.metric_imports.filter((x) => x !== imp);
        renderImports();
      });
    }
  };
  renderImports();

  // --- graphs --------------------------------------------------------------------
  const graphsHost = el(body, 'div');
  const graphNames = () => doc.graphs.map((g) => g.name);

  const renderGraphs = () => {
    graphsHost.innerHTML = '';
    sectionHeader(graphsHost, 'Graphs', () => {
      const base = doc.graphs[0]?.name;
      doc.graphs.push({
        name: `g${doc.graphs.length + 1}`,
        expr: base
          ? { kind: 'pipeline', start: base, transforms: [] }
          : { kind: 'from_evidence', evidence: evidenceNames[0] ?? '' },
      });
      renderGraphs();
    });

    for (const g of doc.graphs) {
      const head = el(graphsHost, 'div', 'section-h');
      const nameIn = el(head, 'input');
      nameIn.value = g.name;
      readers.push(() => { g.name = nameIn.value.trim(); });
      const kindSel = el(head, 'select');
      for (const k of ['from_evidence', 'from_graph', 'pipeline', 'select_model']) {
        const o = el(kindSel, 'option');
        o.value = k; o.textContent = k;
      }
      kindSel.value = g.expr.kind;
      kindSel.addEventListener('change', () => {
        g.expr = defaultExpr(kindSel.value, doc, evidenceNames, graphAliases);
        renderGraphs();
      });
      rowButton(head, '✕', () => { doc.graphs = doc.graphs.filter((x) => x !== g); renderGraphs(); });

      const detail = el(graphsHost, 'div');
      renderGraphExpr(detail, g, {
        readers, evidenceNames, graphAliases, ruleNames, edgeTypes,
        otherGraphs: () => graphNames().filter((n) => n !== g.name),
        rerender: renderGraphs,
      });
    }
  };
  renderGraphs();

  // --- metrics ---------------------------------------------------------------------
  const metricsHost = el(body, 'div');
  const renderMetrics = () => {
    metricsHost.innerHTML = '';
    sectionHeader(metricsHost, 'Metrics', () => {
      doc.metrics.push({ name: `metric${doc.metrics.length + 1}`, expr: 'nodes(Type) |> count()' });
      renderMetrics();
    });
    if (doc.metrics.length === 0) return;
    const table = el(metricsHost, 'table', 'sheet');
    table.innerHTML = '<tr><th>Name</th><th>On graph</th><th>Expression</th><th></th></tr>';
    for (const m of doc.metrics) {
      const tr = el(table, 'tr');
      const nameIn = inputCell(tr, m.name);
      const graphIn = comboCell(tr, ['', ...graphNames()], m.on_graph ?? '');
      const exprIn = inputCell(tr, m.expr);
      readers.push(() => {
        m.name = nameIn.value.trim();
        m.on_graph = graphIn.value.trim() || null;
        m.expr = exprIn.value;
      });
      rowButton(el(tr, 'td', 'row-actions'), '✕', () => {
        doc.metrics = doc.metrics.filter((x) => x !== m);
        renderMetrics();
      });
    }
    el(metricsHost, 'div', 'hint').textContent =
      'Prefer `metric m on <graph> = ...`. Without an explicit target, metrics evaluate against the LAST graph. Note: avg() over an empty set is a runtime error — guard with an epsilon.';
  };
  renderMetrics();

  // --- exports ---------------------------------------------------------------------
  const exportsHost = el(body, 'div');
  const renderExports = () => {
    exportsHost.innerHTML = '';
    sectionHeader(exportsHost, 'Exports', () => {
      doc.exports.push({ graph: doc.graphs[doc.graphs.length - 1]?.name ?? '', alias: 'result' });
      renderExports();
    });
    if (doc.exports.length > 0) {
      const table = el(exportsHost, 'table', 'sheet');
      table.innerHTML = '<tr><th>Graph</th><th>Alias</th><th></th></tr>';
      for (const e of doc.exports) {
        const tr = el(table, 'tr');
        const graphIn = comboCell(tr, graphNames(), e.graph);
        const aliasIn = inputCell(tr, e.alias);
        readers.push(() => { e.graph = graphIn.value.trim(); e.alias = aliasIn.value.trim(); });
        rowButton(el(tr, 'td', 'row-actions'), '✕', () => {
          doc.exports = doc.exports.filter((x) => x !== e);
          renderExports();
        });
      }
    }
    sectionHeader(exportsHost, 'Metric exports', () => {
      doc.metric_exports.push({ metric: doc.metrics[0]?.name ?? '', alias: 'stat' });
      renderExports();
    });
    if (doc.metric_exports.length > 0) {
      const table = el(exportsHost, 'table', 'sheet');
      table.innerHTML = '<tr><th>Metric</th><th>Alias</th><th></th></tr>';
      for (const e of doc.metric_exports) {
        const tr = el(table, 'tr');
        const metricIn = comboCell(tr, doc.metrics.map((m) => m.name), e.metric);
        const aliasIn = inputCell(tr, e.alias);
        readers.push(() => { e.metric = metricIn.value.trim(); e.alias = aliasIn.value.trim(); });
        rowButton(el(tr, 'td', 'row-actions'), '✕', () => {
          doc.metric_exports = doc.metric_exports.filter((x) => x !== e);
          renderExports();
        });
      }
    }
  };
  renderExports();

  buttonBar(body, {
    apply: () => {
      clearError(body);
      for (const read of readers) read();
      doc.name = nameInput.value.trim();
      ctx.commitValidated('flow', name, genFlow(doc), { kind: 'flow', name: doc.name });
    },
    remove: () => ctx.deleteBlock('flow', name),
  });
}

// --- graph expression detail --------------------------------------------------------

function defaultExpr(kind, doc, evidenceNames, graphAliases) {
  switch (kind) {
    case 'from_evidence': return { kind, evidence: evidenceNames[0] ?? '' };
    case 'from_graph': return { kind, alias: graphAliases[0] ?? 'exported_graph' };
    case 'select_model': return { kind, candidates: doc.graphs.slice(0, 2).map((g) => g.name), criterion: 'edge_aic' };
    default: return { kind: 'pipeline', start: doc.graphs[0]?.name ?? '', transforms: [] };
  }
}

function renderGraphExpr(host, g, ctx2) {
  const { readers, evidenceNames, graphAliases, ruleNames, edgeTypes, otherGraphs, rerender } = ctx2;
  const expr = g.expr;

  if (expr.kind === 'from_evidence') {
    const line = el(host, 'div', 'field');
    el(line, 'label').textContent = 'from evidence';
    const sel = combo(line, evidenceNames, expr.evidence);
    readers.push(() => { expr.evidence = sel.value.trim(); });
    return;
  }
  if (expr.kind === 'from_graph') {
    const line = el(host, 'div', 'field');
    el(line, 'label').textContent = 'import exported graph (alias)';
    const sel = combo(line, graphAliases, expr.alias);
    readers.push(() => { expr.alias = sel.value.trim(); });
    return;
  }
  if (expr.kind === 'select_model') {
    const line = el(host, 'div', 'field');
    el(line, 'label').textContent = 'candidates (comma-separated graph names)';
    const input = el(line, 'input');
    input.value = expr.candidates.join(', ');
    const critLine = el(host, 'div', 'field');
    el(critLine, 'label').textContent = 'criterion';
    const crit = el(critLine, 'select');
    for (const c of ['edge_aic', 'edge_bic']) {
      const o = el(crit, 'option');
      o.value = c; o.textContent = c;
    }
    crit.value = expr.criterion;
    readers.push(() => {
      expr.candidates = input.value.split(',').map((s) => s.trim()).filter(Boolean);
      expr.criterion = crit.value;
    });
    return;
  }

  // pipeline
  const startLine = el(host, 'div', 'field');
  el(startLine, 'label').textContent = 'start graph';
  const startSel = combo(startLine, otherGraphs(), expr.start);
  readers.push(() => { expr.start = startSel.value.trim(); });

  const list = el(host, 'div');
  const renderTransforms = () => {
    list.innerHTML = '';
    expr.transforms.forEach((t, i) => {
      const row = el(list, 'div', 'section-h');
      el(row, 'span', 'chip-arrow').textContent = '|>';

      const kindSel = el(row, 'select');
      for (const k of TRANSFORM_KINDS) {
        const o = el(kindSel, 'option');
        o.value = k; o.textContent = k;
      }
      kindSel.value = t.kind;
      kindSel.addEventListener('change', () => {
        expr.transforms[i] = defaultTransform(kindSel.value, ruleNames, edgeTypes);
        renderTransforms();
      });

      if (t.kind === 'apply_rule') {
        const sel = combo(row, ruleNames, t.rule);
        readers.push(() => { t.rule = sel.value.trim(); });
      } else if (t.kind === 'apply_ruleset') {
        const input = el(row, 'input');
        input.value = (t.rules ?? []).join(', ');
        input.placeholder = 'RuleA, RuleB';
        readers.push(() => { t.rules = input.value.split(',').map((s) => s.trim()).filter(Boolean); });
      } else if (t.kind === 'snapshot') {
        const input = el(row, 'input');
        input.value = t.name ?? '';
        input.placeholder = 'snapshot name';
        readers.push(() => { t.name = input.value.trim(); });
      } else if (t.kind === 'prune_edges') {
        const sel = combo(row, edgeTypes, t.edge_type);
        const input = el(row, 'input');
        input.value = t.predicate ?? 'prob(edge) < 0.1';
        input.placeholder = 'prob(edge) < 0.1';
        readers.push(() => { t.edge_type = sel.value.trim(); t.predicate = input.value; });
      }

      rowButton(row, '↑', () => {
        if (i === 0) return;
        [expr.transforms[i - 1], expr.transforms[i]] = [expr.transforms[i], expr.transforms[i - 1]];
        rerender();
      });
      rowButton(row, '↓', () => {
        if (i === expr.transforms.length - 1) return;
        [expr.transforms[i + 1], expr.transforms[i]] = [expr.transforms[i], expr.transforms[i + 1]];
        rerender();
      });
      rowButton(row, '✕', () => { expr.transforms.splice(i, 1); rerender(); });
    });

    const addRow = el(list, 'div', 'section-h');
    rowButton(addRow, '+ transform', () => {
      expr.transforms.push(defaultTransform('apply_rule', ruleNames, edgeTypes));
      rerender();
    });
  };
  renderTransforms();
}

function defaultTransform(kind, ruleNames, edgeTypes) {
  switch (kind) {
    case 'apply_rule': return { kind, rule: ruleNames[0] ?? '' };
    case 'apply_ruleset': return { kind, rules: ruleNames.slice(0, 1) };
    case 'snapshot': return { kind, name: 'checkpoint' };
    case 'prune_edges': return { kind, edge_type: edgeTypes[0] ?? '', predicate: 'prob(edge) < 0.1' };
    default: return { kind: 'infer_beliefs' };
  }
}

// --- small widgets --------------------------------------------------------------------

/** Editable input with a datalist of suggestions. */
let comboSeq = 0;
function combo(parent, options, value) {
  const input = el(parent, 'input');
  const listId = `combo-${comboSeq += 1}`;
  const datalist = el(parent, 'datalist');
  datalist.id = listId;
  for (const opt of options) {
    const o = el(datalist, 'option');
    o.value = opt;
  }
  input.setAttribute('list', listId);
  input.value = value ?? '';
  return input;
}

function comboCell(tr, options, value) {
  return combo(el(tr, 'td'), options, value);
}

function inputCell(tr, value) {
  const input = el(el(tr, 'td'), 'input');
  input.value = value ?? '';
  return input;
}
