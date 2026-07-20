// Generates canonical-style Grafial declaration blocks from the composer's
// document models. Inverse of the shapes served by the wasm
// program_structure() API — parse gives us structure, codegen gives us text.

/** Formats a number so it round-trips as a Grafial Real literal. */
export function num(v) {
  if (!Number.isFinite(v)) throw new Error(`not a finite number: ${v}`);
  const s = String(v);
  return /[.eE]/.test(s) ? s : `${s}.0`;
}

const IDENT = /^[A-Za-z_][A-Za-z0-9_]*$/;

function ident(name, what) {
  if (!IDENT.test(name)) throw new Error(`invalid ${what} identifier: '${name}'`);
  return name;
}

function str(text) {
  return `"${String(text).replace(/\\/g, '\\\\').replace(/"/g, '\\"')}"`;
}

// --- schema ---------------------------------------------------------------

/**
 * schemaDoc: { name, nodes: [{name, attrs: [{name, type}]}], edges: [name] }
 */
export function genSchema(doc) {
  const lines = [`schema ${ident(doc.name, 'schema')} {`];
  for (const node of doc.nodes) {
    lines.push(`  node ${ident(node.name, 'node type')} {`);
    for (const attr of node.attrs) {
      lines.push(`    ${ident(attr.name, 'attribute')}: ${attr.type || 'Real'}`);
    }
    lines.push('  }');
  }
  for (const edge of doc.edges) {
    lines.push(`  edge ${ident(edge, 'edge type')} { }`);
  }
  lines.push('}');
  return lines.join('\n');
}

// --- belief model ----------------------------------------------------------

// Canonical parameter spellings for emission (the parser aliases these).
const PARAM_ALIASES = { prior_mean: 'mean', prior_precision: 'precision', pseudo_count: 'weight' };

function genParams(params) {
  return Object.entries(params)
    .map(([k, v]) => `${PARAM_ALIASES[k] ?? k}=${num(v)}`)
    .join(', ');
}

function genPosterior(p) {
  if (p.family === 'gaussian') return `Gaussian(${genParams(p.params)})`;
  if (p.family === 'bernoulli') return `Bernoulli(${genParams(p.params)})`;
  if (p.family === 'categorical') {
    const prior = p.prior ?? { kind: 'uniform', pseudo_count: 1.0 };
    if (prior.kind === 'uniform') {
      return `Categorical(group_by=source, prior=uniform, pseudo_count=${num(prior.pseudo_count)})`;
    }
    return `Categorical(group_by=source, prior=[${prior.concentrations.map(num).join(', ')}])`;
  }
  throw new Error(`unknown posterior family: ${p.family}`);
}

/**
 * modelDoc: {
 *   name, on_schema,
 *   nodes: [{node_type, attrs: [{name, posterior: {family, params}}]}],
 *   edges: [{edge_type, exist: {family, params|prior}}],
 * }
 */
export function genBeliefModel(doc) {
  const lines = [`belief_model ${ident(doc.name, 'belief model')} on ${ident(doc.on_schema, 'schema')} {`];
  for (const node of doc.nodes) {
    lines.push(`  node ${ident(node.node_type, 'node type')} {`);
    for (const attr of node.attrs) {
      lines.push(`    ${ident(attr.name, 'attribute')} ~ ${genPosterior(attr.posterior)}`);
    }
    lines.push('  }');
  }
  for (const edge of doc.edges) {
    lines.push(`  edge ${ident(edge.edge_type, 'edge type')} {`);
    lines.push(`    exist ~ ${genPosterior(edge.exist)}`);
    lines.push('  }');
  }
  lines.push('}');
  return lines.join('\n');
}

// --- evidence ---------------------------------------------------------------

/**
 * evidenceDoc: {
 *   name, on_model,
 *   nodeRows: [{type, name, values: {attr: {value, precision|null}}}],
 *   edgeRows: [{edge_type, src_type, src, dst_type, dst,
 *               mode: present|absent|chosen|unchosen|forced_choice}],
 *   weightRows: [{edge_type, src_type, src, dst_type, dst, value, precision|null}],
 * }
 *
 * Each nodeRow is one observation group — repeated rows for the same instance
 * are legal and accumulate Bayesian updates, matching grouped-evidence
 * semantics.
 */
export function genEvidence(doc) {
  const lines = [`evidence ${ident(doc.name, 'evidence')} on ${ident(doc.on_model, 'belief model')} {`];

  // Group node rows by node type, preserving row order within a type.
  const byType = new Map();
  for (const row of doc.nodeRows) {
    const entries = Object.entries(row.values).filter(([, cell]) => cell != null && cell.value != null);
    if (entries.length === 0) continue;
    if (!byType.has(row.type)) byType.set(row.type, []);
    const fields = entries
      .map(([attr, cell]) => {
        const precision = cell.precision != null ? ` (precision=${num(cell.precision)})` : '';
        return `${ident(attr, 'attribute')}: ${num(cell.value)}${precision}`;
      })
      .join(', ');
    byType.get(row.type).push(`    ${str(row.name)} { ${fields} }`);
  }
  for (const [type, rows] of byType) {
    lines.push(`  ${ident(type, 'node type')} {`);
    lines.push(rows.join(',\n'));
    lines.push('  }');
  }

  // Present/absent edges use grouped sugar; categorical modes use choice verbs.
  const grouped = new Map(); // "EDGE|SrcType|DstType" -> lines
  const verbs = [];
  for (const row of doc.edgeRows ?? []) {
    const key = `${row.edge_type}|${row.src_type}|${row.dst_type}`;
    if (row.mode === 'present' || row.mode === 'absent') {
      const op = row.mode === 'present' ? '->' : '-/>';
      if (!grouped.has(key)) grouped.set(key, []);
      grouped.get(key).push(`    ${str(row.src)} ${op} ${str(row.dst)}`);
    } else {
      const ref = `${ident(row.edge_type, 'edge type')}(${ident(row.src_type, 'node type')}[${str(row.src)}], ${ident(row.dst_type, 'node type')}[${str(row.dst)}])`;
      if (row.mode === 'chosen') verbs.push(`  choose edge ${ref}`);
      else if (row.mode === 'unchosen') verbs.push(`  unchoose edge ${ref}`);
      else if (row.mode === 'forced_choice') verbs.push(`  observe edge ${ref} forced_choice`);
      else throw new Error(`unknown edge mode: ${row.mode}`);
    }
  }
  for (const [key, rows] of grouped) {
    const [edgeType, srcType, dstType] = key.split('|');
    lines.push(`  ${edgeType}(${srcType} -> ${dstType}) {`);
    lines.push(rows.join(';\n') + '');
    lines.push('  }');
  }
  lines.push(...verbs);

  for (const row of doc.weightRows ?? []) {
    const precision = row.precision != null ? ` (precision=${num(row.precision)})` : '';
    lines.push(
      `  observe edge ${ident(row.edge_type, 'edge type')}(${ident(row.src_type, 'node type')}[${str(row.src)}], ` +
        `${ident(row.dst_type, 'node type')}[${str(row.dst)}]) weight=${num(row.value)}${precision}`,
    );
  }

  lines.push('}');
  return lines.join('\n');
}

/**
 * Converts the wasm program_structure() observation list into the evidence
 * document model that genEvidence() consumes (spreadsheet round-trip).
 *
 * Attribute observations pack into rows greedily: an observation lands on the
 * last row for its instance that doesn't yet have that attribute; otherwise it
 * opens a new row. Repeated observations therefore become repeated rows.
 */
export function evidenceDocFromStructure(evidence) {
  const nodeRows = [];
  const rowIndex = new Map(); // "type|name" -> indexes into nodeRows
  const edgeRows = [];
  const weightRows = [];

  for (const obs of evidence.observations) {
    if (obs.kind === 'attribute') {
      const key = `${obs.node_type}|${obs.node}`;
      if (!rowIndex.has(key)) rowIndex.set(key, []);
      const candidates = rowIndex.get(key);
      let row = null;
      for (const idx of candidates) {
        if (!(obs.attr in nodeRows[idx].values)) { row = nodeRows[idx]; break; }
      }
      if (!row) {
        row = { type: obs.node_type, name: obs.node, values: {} };
        candidates.push(nodeRows.length);
        nodeRows.push(row);
      }
      row.values[obs.attr] = { value: obs.value, precision: obs.precision ?? null };
    } else if (obs.kind === 'edge') {
      edgeRows.push({
        edge_type: obs.edge_type,
        src_type: obs.src_type, src: obs.src,
        dst_type: obs.dst_type, dst: obs.dst,
        mode: obs.mode,
      });
    } else if (obs.kind === 'edge_weight') {
      weightRows.push({
        edge_type: obs.edge_type,
        src_type: obs.src_type, src: obs.src,
        dst_type: obs.dst_type, dst: obs.dst,
        value: obs.value, precision: obs.precision ?? null,
      });
    }
  }

  return { name: evidence.name, on_model: evidence.on_model, nodeRows, edgeRows, weightRows };
}

// --- templates ---------------------------------------------------------------

/** A minimal complete starter program for "New". */
export function starterProgram() {
  return `// New Grafial program
schema MySchema {
  node Entity {
    value: Real
  }
  edge RELATES { }
}

belief_model MyBeliefs on MySchema {
  node Entity {
    value ~ Gaussian(mean=0.0, precision=0.1)
  }
  edge RELATES {
    exist ~ Bernoulli(prior=0.5, weight=2.0)
  }
}

evidence MyEvidence on MyBeliefs {
  Entity {
    "A" { value: 1.0 },
    "B" { value: 2.0 }
  }
  RELATES(Entity -> Entity) { "A" -> "B" }
}

flow Main on MyBeliefs {
  graph g = from_evidence MyEvidence
  metric total = nodes(Entity) |> sum(by=E[node.value])
  export g as "result"
}
`;
}

/** Template for a new rule block. */
export function ruleTemplate(name, onModel, nodeType, edgeType) {
  return `rule ${name} on ${onModel} {
  pattern
    (A:${nodeType})-[ab:${edgeType}]->(B:${nodeType})

  where
    prob(ab) >= 0.5

  action {
    non_bayesian_nudge B.value to E[B.value] variance=preserve
  }

  mode: for_each
}`;
}

/** Template for a new flow block. */
export function flowTemplate(name, onModel, evidenceName) {
  return `flow ${name} on ${onModel} {
  graph g = from_evidence ${evidenceName}
  export g as "${name.toLowerCase()}_result"
}`;
}

// --- flows ---------------------------------------------------------------------

function genGraphExpr(expr) {
  switch (expr.kind) {
    case 'from_evidence': return `from_evidence ${ident(expr.evidence, 'evidence')}`;
    case 'from_graph': return `from_graph ${JSON.stringify(expr.alias)}`;
    case 'select_model':
      return `select_model { ${expr.candidates.map((c) => ident(c, 'candidate')).join(', ')} } by ${expr.criterion}`;
    case 'pipeline': {
      const parts = [ident(expr.start, 'graph')];
      for (const t of expr.transforms) {
        if (t.kind === 'apply_rule') parts.push(`apply_rule ${ident(t.rule, 'rule')}`);
        else if (t.kind === 'apply_ruleset') parts.push(`apply_ruleset { ${t.rules.map((r) => ident(r, 'rule')).join(', ')} }`);
        else if (t.kind === 'snapshot') parts.push(`snapshot ${JSON.stringify(t.name)}`);
        else if (t.kind === 'infer_beliefs') parts.push('infer_beliefs');
        else if (t.kind === 'prune_edges') {
          if (!t.predicate || !t.predicate.trim()) throw new Error(`prune_edges ${t.edge_type}: predicate required`);
          parts.push(`prune_edges ${ident(t.edge_type, 'edge type')} where ${t.predicate.trim()}`);
        } else throw new Error(`unknown transform: ${t.kind}`);
      }
      return parts.join(' |> ');
    }
    default: throw new Error(`unknown graph expression: ${expr.kind}`);
  }
}

/**
 * flowDoc mirrors the program_structure() flow shape:
 * { name, on_model, metric_imports: [{source_alias, local_name}],
 *   graphs: [{name, expr}], metrics: [{name, expr}],
 *   exports: [{graph, alias}], metric_exports: [{metric, alias}] }
 */
export function genFlow(doc) {
  const lines = [`flow ${ident(doc.name, 'flow')} on ${ident(doc.on_model, 'belief model')} {`];
  for (const i of doc.metric_imports ?? []) {
    lines.push(`  import_metric ${ident(i.source_alias, 'metric alias')} as ${ident(i.local_name, 'metric name')}`);
  }
  for (const g of doc.graphs ?? []) {
    lines.push(`  graph ${ident(g.name, 'graph')} = ${genGraphExpr(g.expr)}`);
  }
  for (const m of doc.metrics ?? []) {
    if (!m.expr || !m.expr.trim()) throw new Error(`metric ${m.name}: expression required`);
    lines.push(`  metric ${ident(m.name, 'metric')} = ${m.expr.trim()}`);
  }
  for (const e of doc.exports ?? []) {
    lines.push(`  export ${ident(e.graph, 'graph')} as ${JSON.stringify(e.alias)}`);
  }
  for (const e of doc.metric_exports ?? []) {
    lines.push(`  export_metric ${ident(e.metric, 'metric')} as ${JSON.stringify(e.alias)}`);
  }
  lines.push('}');
  return lines.join('\n');
}

// --- rules ---------------------------------------------------------------------

/**
 * ruleDoc mirrors the program_structure() rule shape:
 * { name, on_model, patterns: [{src: {var, label}, edge: {var, type}, dst: {var, label}}],
 *   where: string|null, actions: [string], mode: string|null }
 */
export function genRule(doc) {
  if ((doc.patterns ?? []).length === 0) throw new Error('a rule needs at least one pattern');
  if ((doc.actions ?? []).length === 0) throw new Error('a rule needs at least one action');

  // Node-only iteration parses to a sentinel pattern; emit the `for` sugar
  // back rather than leaking the internal __FOR_NODE__ edge type.
  if (isForNodeRule(doc)) {
    const p = doc.patterns[0];
    const lines = [`rule ${ident(doc.name, 'rule')} on ${ident(doc.on_model, 'belief model')} {`];
    const whereStr = doc.where && doc.where.trim() ? ` where ${doc.where.trim()}` : '';
    lines.push(`  for (${ident(p.src.var, 'variable')}:${ident(p.src.label, 'label')})${whereStr} => {`);
    for (const action of doc.actions) {
      if (action.trim()) lines.push(`    ${action.trim()}`);
    }
    lines.push('  }');
    lines.push('}');
    return lines.join('\n');
  }

  const lines = [`rule ${ident(doc.name, 'rule')} on ${ident(doc.on_model, 'belief model')} {`];
  lines.push('  pattern');
  lines.push(doc.patterns
    .map((p) => `    (${ident(p.src.var, 'variable')}:${ident(p.src.label, 'label')})` +
      `-[${ident(p.edge.var, 'variable')}:${ident(p.edge.type, 'edge type')}]->` +
      `(${ident(p.dst.var, 'variable')}:${ident(p.dst.label, 'label')})`)
    .join(',\n'));
  if (doc.where && doc.where.trim()) {
    lines.push('');
    lines.push('  where');
    lines.push(`    ${doc.where.trim()}`);
  }
  lines.push('');
  lines.push('  action {');
  for (const action of doc.actions) {
    if (!action.trim()) continue;
    lines.push(`    ${action.trim()}`);
  }
  lines.push('  }');
  lines.push('');
  lines.push(`  mode: ${doc.mode || 'for_each'}`);
  lines.push('}');
  return lines.join('\n');
}

/** True when the rule is node-only iteration sugar (`for (P:Type)`). */
export function isForNodeRule(doc) {
  return (doc.patterns ?? []).length === 1 && doc.patterns[0].edge.type === '__FOR_NODE__';
}
