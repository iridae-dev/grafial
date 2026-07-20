// Program map: declaration cards in kind columns with dependency edges.
//
// Deliberately NOT a free-form node canvas: a Grafial program is a layered
// set of declarations (schema <- model <- evidence/rules/flows), so a fixed
// columnar map with drawn dependencies reads better than draggable boxes.

import { state, select } from './state.js';
import { appendDeclaration } from './blockedit.js';
import { setSource } from './state.js';
import { genSchema, genBeliefModel, genEvidence, ruleTemplate, flowTemplate } from './codegen.js';

const KINDS = [
  { kind: 'schema', title: 'Schemas' },
  { kind: 'belief_model', title: 'Belief Models' },
  { kind: 'evidence', title: 'Evidence' },
  { kind: 'rule', title: 'Rules' },
  { kind: 'flow', title: 'Flows' },
];

export function renderMap(container) {
  const s = state.structure;
  container.innerHTML = '';
  if (!s) {
    container.innerHTML = '<div class="empty-note">Program has errors — fix them in the Source tab to see the map.</div>';
    return;
  }

  const svg = document.createElementNS('http://www.w3.org/2000/svg', 'svg');
  svg.id = 'map-edges';
  container.appendChild(svg);

  const columns = document.createElement('div');
  columns.className = 'map-columns';
  container.appendChild(columns);

  const cards = new Map(); // "kind|name" -> element

  for (const { kind, title } of KINDS) {
    const col = document.createElement('div');
    col.className = 'map-column';
    const h = document.createElement('h3');
    h.textContent = title;
    const add = document.createElement('button');
    add.textContent = '+';
    add.title = `Add ${kind.replace('_', ' ')}`;
    add.addEventListener('click', () => addDeclaration(kind));
    h.appendChild(add);
    col.appendChild(h);

    for (const item of itemsOf(s, kind)) {
      const card = document.createElement('div');
      card.className = 'card';
      card.dataset.kind = kind;
      card.dataset.name = item.name;
      if (state.selection?.kind === kind && state.selection?.name === item.name) {
        card.classList.add('selected');
      }
      card.innerHTML = `<div class="name"></div><div class="meta"></div>`;
      card.querySelector('.name').textContent = item.name;
      card.querySelector('.meta').textContent = cardMeta(kind, item);
      card.addEventListener('click', () => select({ kind, name: item.name }));
      col.appendChild(card);
      cards.set(`${kind}|${item.name}`, card);
    }
    columns.appendChild(col);
  }

  requestAnimationFrame(() => drawEdges(container, svg, s, cards));
}

function itemsOf(s, kind) {
  switch (kind) {
    case 'schema': return s.schemas;
    case 'belief_model': return s.belief_models;
    case 'evidence': return s.evidences;
    case 'rule': return s.rules;
    case 'flow': return s.flows;
    default: return [];
  }
}

function cardMeta(kind, item) {
  switch (kind) {
    case 'schema':
      return `${item.nodes.length} node type${item.nodes.length === 1 ? '' : 's'}, ${item.edges.length} edge`;
    case 'belief_model': return `on ${item.on_schema}`;
    case 'evidence': return `on ${item.on_model} · ${item.observation_count} obs`;
    case 'rule': return `on ${item.on_model}${item.mode ? ` · ${item.mode}` : ''}`;
    case 'flow': {
      const deps = item.needs_prior ? ' · imports prior flows' : '';
      return `on ${item.on_model} · ${item.graphs.length} graph${item.graphs.length === 1 ? '' : 's'}${deps}`;
    }
    default: return '';
  }
}

/** Dependency edges: model->schema, evidence/rule/flow->model, flow->flow. */
function drawEdges(container, svg, s, cards) {
  const cRect = container.getBoundingClientRect();
  svg.setAttribute('width', container.scrollWidth);
  svg.setAttribute('height', container.scrollHeight);
  svg.innerHTML = '';

  const anchor = (key, side) => {
    const el = cards.get(key);
    if (!el) return null;
    const r = el.getBoundingClientRect();
    return {
      x: (side === 'left' ? r.left : r.right) - cRect.left + container.scrollLeft,
      y: r.top + r.height / 2 - cRect.top + container.scrollTop,
    };
  };
  const curve = (from, to, cls = '') => {
    if (!from || !to) return;
    const mx = (from.x + to.x) / 2;
    const p = document.createElementNS('http://www.w3.org/2000/svg', 'path');
    p.setAttribute('d', `M ${from.x} ${from.y} C ${mx} ${from.y}, ${mx} ${to.y}, ${to.x} ${to.y}`);
    if (cls) p.setAttribute('class', cls);
    svg.appendChild(p);
  };

  for (const m of s.belief_models) {
    curve(anchor(`schema|${m.on_schema}`, 'right'), anchor(`belief_model|${m.name}`, 'left'));
  }
  for (const e of s.evidences) {
    curve(anchor(`belief_model|${e.on_model}`, 'right'), anchor(`evidence|${e.name}`, 'left'));
  }
  for (const r of s.rules) {
    curve(anchor(`belief_model|${r.on_model}`, 'right'), anchor(`rule|${r.name}`, 'left'));
  }
  for (const f of s.flows) {
    // Evidence feeding this flow
    for (const g of f.graphs) {
      if (g.expr.kind === 'from_evidence') {
        curve(anchor(`evidence|${g.expr.evidence}`, 'right'), anchor(`flow|${f.name}`, 'left'));
      }
    }
    // Rules applied by this flow
    for (const rule of rulesUsedBy(f)) {
      curve(anchor(`rule|${rule}`, 'right'), anchor(`flow|${f.name}`, 'left'));
    }
  }
  // Cross-flow dataflow: exporter -> importer, matched by alias.
  for (const importer of s.flows) {
    const wanted = new Set([
      ...importer.graphs.filter((g) => g.expr.kind === 'from_graph').map((g) => g.expr.alias),
      ...importer.metric_imports.map((i) => i.source_alias),
    ]);
    if (wanted.size === 0) continue;
    for (const exporter of s.flows) {
      if (exporter.name === importer.name) continue;
      const provides = [
        ...exporter.exports.map((e) => e.alias),
        ...exporter.metric_exports.map((e) => e.alias),
      ];
      if (provides.some((a) => wanted.has(a))) {
        curve(anchor(`flow|${exporter.name}`, 'right'), anchor(`flow|${importer.name}`, 'left'), 'crossflow');
      }
    }
  }
}

export function rulesUsedBy(flow) {
  const rules = new Set();
  for (const g of flow.graphs) {
    if (g.expr.kind !== 'pipeline') continue;
    for (const t of g.expr.transforms) {
      if (t.kind === 'apply_rule') rules.add(t.rule);
      if (t.kind === 'apply_ruleset') for (const r of t.rules) rules.add(r);
    }
  }
  return [...rules];
}

// --- adding declarations ------------------------------------------------------

function freshName(base, taken) {
  if (!taken.includes(base)) return base;
  let i = 2;
  while (taken.includes(`${base}${i}`)) i += 1;
  return `${base}${i}`;
}

function addDeclaration(kind) {
  const s = state.structure;
  if (!s) return;
  const schema = s.schemas[0];
  const model = s.belief_models[0];
  let name, text;

  switch (kind) {
    case 'schema': {
      name = freshName('NewSchema', s.schemas.map((x) => x.name));
      text = genSchema({ name, nodes: [{ name: 'Entity', attrs: [{ name: 'value', type: 'Real' }] }], edges: ['RELATES'] });
      break;
    }
    case 'belief_model': {
      if (!schema) return alert('Define a schema first.');
      name = freshName('NewBeliefs', s.belief_models.map((x) => x.name));
      text = genBeliefModel({
        name, on_schema: schema.name,
        nodes: schema.nodes.map((n) => ({
          node_type: n.name,
          attrs: n.attrs.map((a) => ({ name: a.name, posterior: { family: 'gaussian', params: { prior_mean: 0, prior_precision: 0.1 } } })),
        })),
        edges: schema.edges.map((e) => ({ edge_type: e, exist: { family: 'bernoulli', params: { prior: 0.5, pseudo_count: 2 } } })),
      });
      break;
    }
    case 'evidence': {
      if (!model) return alert('Define a belief model first.');
      name = freshName('NewEvidence', s.evidences.map((x) => x.name));
      text = genEvidence({ name, on_model: model.name, nodeRows: [], edgeRows: [], weightRows: [] });
      break;
    }
    case 'rule': {
      if (!model) return alert('Define a belief model first.');
      const sch = s.schemas.find((x) => x.name === model.on_schema);
      const nodeType = sch?.nodes[0]?.name ?? 'Entity';
      const edgeType = sch?.edges[0] ?? 'RELATES';
      name = freshName('NewRule', s.rules.map((x) => x.name));
      text = ruleTemplate(name, model.name, nodeType, edgeType);
      break;
    }
    case 'flow': {
      if (!model) return alert('Define a belief model first.');
      const ev = s.evidences.find((x) => x.on_model === model.name);
      if (!ev) return alert('Define evidence first.');
      name = freshName('NewFlow', s.flows.map((x) => x.name));
      text = flowTemplate(name, model.name, ev.name);
      break;
    }
    default: return;
  }

  setSource(appendDeclaration(state.source, kind, text));
  select({ kind, name });
}
