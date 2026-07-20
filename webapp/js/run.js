// Flow execution and results rendering: metrics table, belief-graph view
// (nodes labeled by evidence instance name, edge opacity by existence
// probability), rule-firing audit, and inference diagnostics.

import { state, setRun, select } from './state.js';
import { runFlow } from './wasmapi.js';
import { el } from './editors/widgets.js';

export function initRun({ onRan }) {
  document.getElementById('btn-run').addEventListener('click', () => {
    const flowName = document.getElementById('sel-flow').value;
    if (!flowName) return;
    try {
      const result = runFlow(state.source, flowName);
      setRun({ flow: flowName, result, graphName: null });
      onRan();
    } catch (err) {
      setRun({ flow: flowName, error: String(err.message ?? err) });
      onRan();
    }
  });
}

export function refreshFlowPicker() {
  const sel = document.getElementById('sel-flow');
  const current = sel.value;
  sel.innerHTML = '';
  for (const flow of state.structure?.flows ?? []) {
    const opt = el(sel, 'option');
    opt.value = flow.name;
    opt.textContent = flow.name;
  }
  if ([...sel.options].some((o) => o.value === current)) sel.value = current;
}

export function renderResults(container) {
  container.innerHTML = '';
  const run = state.run;
  if (!run) {
    container.innerHTML = '<div class="empty-note">Run a flow to see results.</div>';
    return;
  }
  if (run.error) {
    const block = el(container, 'div', 'results-block');
    el(block, 'h3').textContent = `Flow ${run.flow} failed`;
    el(block, 'div', 'error-note').textContent = run.error;
    return;
  }
  const result = run.result;

  const grid = el(container, 'div', 'results-grid');
  const left = el(grid, 'div');
  const right = el(grid, 'div');

  // --- graph view -----------------------------------------------------------
  const graphBlock = el(left, 'div', 'results-block');
  graphBlock.id = 'graph-view';
  const title = el(graphBlock, 'h3');
  title.textContent = `Belief graphs — flow ${run.flow}`;

  const graphNames = [
    ...Object.keys(result.exports).map((n) => ['export', n]),
    ...Object.keys(result.snapshots).map((n) => ['snapshot', n]),
    ...Object.keys(result.graphs).map((n) => ['graph', n]),
  ];
  if (graphNames.length === 0) {
    el(graphBlock, 'div', 'hint').textContent = 'No graphs in this flow.';
  } else {
    if (!run.graphName || !graphNames.some(([, n]) => n === run.graphName)) {
      run.graphName = graphNames[0][1];
    }
    const picker = el(graphBlock, 'select');
    for (const [kindLabel, n] of graphNames) {
      const opt = el(picker, 'option');
      opt.value = n;
      opt.textContent = `${n} (${kindLabel})`;
    }
    picker.value = run.graphName;
    picker.addEventListener('change', () => { run.graphName = picker.value; renderResults(container); });

    const graph = result.exports[run.graphName] ?? result.snapshots[run.graphName] ?? result.graphs[run.graphName];
    renderGraphSVG(graphBlock, graph, run.graphName);
  }

  // --- metrics ---------------------------------------------------------------
  const metricsBlock = el(right, 'div', 'results-block');
  el(metricsBlock, 'h3').textContent = 'Metrics';
  const entries = [...Object.entries(result.metrics)].sort(([a], [b]) => a.localeCompare(b));
  if (entries.length === 0) el(metricsBlock, 'div', 'hint').textContent = 'No metrics.';
  const mTable = el(metricsBlock, 'table', 'kv');
  for (const [k, v] of entries) {
    const tr = el(mTable, 'tr');
    el(tr, 'td').textContent = k;
    el(tr, 'td').textContent = fmt(v);
  }
  for (const [k, v] of Object.entries(result.metric_exports)) {
    const tr = el(mTable, 'tr');
    el(tr, 'td').textContent = `${k} (exported)`;
    el(tr, 'td').textContent = fmt(v);
  }

  // --- audits ------------------------------------------------------------------
  if (result.intervention_audit.length > 0) {
    const block = el(right, 'div', 'results-block');
    el(block, 'h3').textContent = 'Rule applications';
    const table = el(block, 'table', 'kv');
    for (const a of result.intervention_audit) {
      const tr = el(table, 'tr');
      el(tr, 'td').textContent = `${a.rule}`;
      const zero = a.matched_bindings === 0 ? ' — never fired!' : '';
      el(tr, 'td').textContent = `${a.graph}: matched ${a.matched_bindings}, actions ${a.actions_executed}${zero}`;
    }
  }
  if (result.inference_diagnostics.length > 0) {
    const block = el(right, 'div', 'results-block');
    el(block, 'h3').textContent = 'Inference diagnostics';
    const table = el(block, 'table', 'kv');
    for (const d of result.inference_diagnostics) {
      const tr = el(table, 'tr');
      el(tr, 'td').textContent = `${d.graph} (${d.algorithm})`;
      el(tr, 'td').textContent = `${d.converged ? 'converged' : 'NOT converged'} in ${d.iterations_run}/${d.max_iterations}, Δ=${d.final_max_message_delta.toExponential(2)}`;
    }
  }
}

// --- SVG belief-graph rendering ------------------------------------------------
//
// Deterministic force-directed layout (seeded from a circle, fixed iteration
// count, no randomness — same graph always lays out the same way). Edge
// opacity encodes existence probability; labels prefer instance names.

/** Fruchterman–Reingold-style layout; probability-weighted attraction. */
export function forceLayout(nodes, edges, W, H) {
  const n = nodes.length;
  const pos = new Map();
  const cx = W / 2;
  const cy = H / 2;
  const seedRadius = Math.min(W, H) / 2 - 80;
  nodes.forEach((node, i) => {
    const angle = (2 * Math.PI * i) / Math.max(1, n) - Math.PI / 2;
    pos.set(node.id, {
      x: cx + seedRadius * Math.cos(angle),
      y: cy + seedRadius * Math.sin(angle),
    });
  });
  if (n <= 1) return pos;

  const area = W * H;
  const k = Math.sqrt(area / n) * 0.6; // ideal spring length
  const iterations = 200;
  let temperature = Math.min(W, H) / 8;
  const cool = temperature / iterations;

  for (let iter = 0; iter < iterations; iter += 1) {
    const disp = new Map(nodes.map((node) => [node.id, { x: 0, y: 0 }]));

    // Pairwise repulsion.
    for (let i = 0; i < n; i += 1) {
      for (let j = i + 1; j < n; j += 1) {
        const a = pos.get(nodes[i].id);
        const b = pos.get(nodes[j].id);
        let dx = a.x - b.x;
        let dy = a.y - b.y;
        let d = Math.hypot(dx, dy);
        if (d < 0.01) { dx = 0.01 * (i - j); dy = 0.01; d = Math.hypot(dx, dy); }
        const f = (k * k) / d;
        const da = disp.get(nodes[i].id);
        const db = disp.get(nodes[j].id);
        da.x += (dx / d) * f; da.y += (dy / d) * f;
        db.x -= (dx / d) * f; db.y -= (dy / d) * f;
      }
    }
    // Spring attraction along edges, weighted by existence probability so
    // confident edges pull harder than doubtful ones.
    for (const e of edges) {
      if (e.src === e.dst) continue;
      const a = pos.get(e.src);
      const b = pos.get(e.dst);
      if (!a || !b) continue;
      const dx = a.x - b.x;
      const dy = a.y - b.y;
      const d = Math.max(0.01, Math.hypot(dx, dy));
      const f = ((d * d) / k) * (0.3 + 0.7 * e.prob);
      const da = disp.get(e.src);
      const db = disp.get(e.dst);
      da.x -= (dx / d) * f; da.y -= (dy / d) * f;
      db.x += (dx / d) * f; db.y += (dy / d) * f;
    }
    // Gentle gravity toward the center keeps disconnected parts on screen.
    for (const node of nodes) {
      const p = pos.get(node.id);
      const d = disp.get(node.id);
      d.x += (cx - p.x) * 0.03;
      d.y += (cy - p.y) * 0.03;
    }
    // Apply displacements, capped by the cooling temperature.
    for (const node of nodes) {
      const p = pos.get(node.id);
      const d = disp.get(node.id);
      const len = Math.max(0.01, Math.hypot(d.x, d.y));
      const cap = Math.min(len, temperature);
      p.x += (d.x / len) * cap;
      p.y += (d.y / len) * cap;
    }
    temperature = Math.max(0.5, temperature - cool);
  }

  // Fit to the viewport with padding.
  const xs = [...pos.values()].map((p) => p.x);
  const ys = [...pos.values()].map((p) => p.y);
  const pad = 70;
  const minX = Math.min(...xs); const maxX = Math.max(...xs);
  const minY = Math.min(...ys); const maxY = Math.max(...ys);
  const sx = (W - 2 * pad) / Math.max(1, maxX - minX);
  const sy = (H - 2 * pad) / Math.max(1, maxY - minY);
  const scale = Math.min(sx, sy, 1.5);
  for (const p of pos.values()) {
    p.x = pad + (p.x - minX) * scale + (W - 2 * pad - (maxX - minX) * scale) / 2;
    p.y = pad + (p.y - minY) * scale + (H - 2 * pad - (maxY - minY) * scale) / 2;
  }
  return pos;
}

function renderGraphSVG(parent, graph, graphName) {
  const W = 640;
  const H = Math.max(360, Math.min(640, 120 + graph.nodes.length * 34));

  const svg = document.createElementNS('http://www.w3.org/2000/svg', 'svg');
  svg.setAttribute('viewBox', `0 0 ${W} ${H}`);
  parent.appendChild(svg);

  if (graph.nodes.length === 0) {
    el(parent, 'div', 'hint').textContent = 'Empty graph.';
    return;
  }

  const pos = forceLayout(graph.nodes, graph.edges, W, H);

  const ns = 'http://www.w3.org/2000/svg';
  const displayName = (n) => n.name ?? `${n.label} #${n.id}`;

  for (const e of graph.edges) {
    const a = pos.get(e.src);
    const b = pos.get(e.dst);
    if (!a || !b) continue;
    const g = document.createElementNS(ns, 'g');
    g.setAttribute('class', 'gedge');
    g.setAttribute('opacity', String(0.25 + 0.75 * e.prob));

    const path = document.createElementNS(ns, 'path');
    let midX;
    let midY;
    if (e.src === e.dst) {
      // Self-loop: a small circle above the node.
      path.setAttribute('d', `M ${a.x - 8} ${a.y - 16} a 14 14 0 1 1 16 0`);
      midX = a.x; midY = a.y - 48;
    } else {
      // Slight curve so opposite-direction edges don't overlap.
      const dx = b.x - a.x; const dy = b.y - a.y;
      const len = Math.hypot(dx, dy) || 1;
      const off = 14;
      midX = (a.x + b.x) / 2 - (dy / len) * off;
      midY = (a.y + b.y) / 2 + (dx / len) * off;
      path.setAttribute('d', `M ${a.x} ${a.y} Q ${midX} ${midY} ${b.x} ${b.y}`);
      path.setAttribute('marker-end', 'url(#arrow)');
    }
    g.appendChild(path);

    const label = document.createElementNS(ns, 'text');
    label.setAttribute('x', midX);
    label.setAttribute('y', midY - 4);
    label.textContent = `${e.type} ${e.prob.toFixed(2)}`;
    g.appendChild(label);
    svg.appendChild(g);
  }

  // Arrowhead marker
  const defs = document.createElementNS(ns, 'defs');
  defs.innerHTML = '<marker id="arrow" viewBox="0 0 10 10" refX="18" refY="5" markerWidth="7" markerHeight="7" orient="auto-start-reverse"><path d="M 0 0 L 10 5 L 0 10 z" fill="currentColor"/></marker>';
  svg.appendChild(defs);

  for (const n of graph.nodes) {
    const p = pos.get(n.id);
    const g = document.createElementNS(ns, 'g');
    g.setAttribute('class', 'gnode');
    if (state.selection?.kind === 'result-node' && state.selection.nodeId === n.id
        && state.selection.graphName === graphName) {
      g.setAttribute('class', 'gnode selected');
    }

    const circle = document.createElementNS(ns, 'circle');
    circle.setAttribute('cx', p.x);
    circle.setAttribute('cy', p.y);
    circle.setAttribute('r', 16);
    g.appendChild(circle);

    const label = document.createElementNS(ns, 'text');
    label.setAttribute('x', p.x);
    label.setAttribute('y', p.y + 30);
    label.textContent = displayName(n);
    g.appendChild(label);

    g.addEventListener('click', () => {
      select({ kind: 'result-node', name: displayName(n), graphName, nodeId: n.id });
    });
    svg.appendChild(g);
  }
}

function fmt(v) {
  if (!Number.isFinite(v)) return String(v);
  return Math.abs(v) >= 10000 || (Math.abs(v) < 0.0001 && v !== 0) ? v.toExponential(4) : String(Math.round(v * 1e6) / 1e6);
}
