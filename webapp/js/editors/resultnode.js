// Inspector detail for a node clicked in a result graph: posterior table and
// incident edges with existence probabilities.

import { state } from '../state.js';
import { el } from './widgets.js';

export function renderResultNode(body, sel) {
  const graph = state.run?.result && lookupGraph(state.run.result, sel.graphName);
  const node = graph?.nodes.find((n) => n.id === sel.nodeId);
  if (!node) { body.innerHTML = '<div class="hint">Node not found (results changed?).</div>'; return; }

  el(body, 'div', 'hint').textContent = `${node.label} #${node.id} in ${sel.graphName} (flow ${state.run.flow})`;

  const attrs = el(body, 'table', 'kv');
  for (const [attrName, g] of Object.entries(node.attrs)) {
    const tr = el(attrs, 'tr');
    el(tr, 'td').textContent = attrName;
    el(tr, 'td').textContent = `E = ${fmt(g.mean)}  ·  σ² = ${fmt(g.variance)}`;
  }

  const incident = graph.edges.filter((e) => e.src === node.id || e.dst === node.id);
  if (incident.length > 0) {
    const h = el(body, 'div', 'section-h');
    h.textContent = 'Edges';
    const table = el(body, 'table', 'kv');
    const nameOf = (id) => {
      const n = graph.nodes.find((x) => x.id === id);
      return n?.name ?? `${n?.label ?? '?'} #${id}`;
    };
    for (const e of incident) {
      const tr = el(table, 'tr');
      el(tr, 'td').textContent = `${e.type}`;
      el(tr, 'td').textContent = `${nameOf(e.src)} → ${nameOf(e.dst)}  ·  P = ${fmt(e.prob)}`;
    }
  }
}

function lookupGraph(result, name) {
  return result.exports[name] ?? result.graphs[name] ?? result.snapshots[name] ?? null;
}

function fmt(v) {
  if (!Number.isFinite(v)) return String(v);
  return Math.abs(v) >= 1000 || (Math.abs(v) < 0.001 && v !== 0) ? v.toExponential(3) : v.toFixed(4).replace(/\.?0+$/, '');
}
