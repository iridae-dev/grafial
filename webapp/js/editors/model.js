// Form editor for belief models: per-attribute posterior parameters and
// per-edge-type existence posteriors. The attribute list follows the backing
// schema; attributes the model doesn't declare yet get editable defaults.

import { findModel, findSchema } from '../state.js';
import { genBeliefModel } from '../codegen.js';
import { el, field, sectionHeader, buttonBar, readNumber } from './widgets.js';

const GAUSSIAN_DEFAULTS = { prior_mean: 0, prior_precision: 0.1 };
const BERNOULLI_DEFAULTS = { prior: 0.5, pseudo_count: 2 };

export function renderModelEditor(body, name, ctx) {
  const model = findModel(name);
  if (!model) { body.innerHTML = '<div class="hint">Not found in last valid program.</div>'; return; }
  const schema = findSchema(model.on_schema);

  const nameInput = field(body, 'Name', model.name);
  el(body, 'div', 'hint').textContent = `on schema ${model.on_schema}`;

  // Working doc: schema-driven attribute list, seeded from the model.
  const doc = {
    name: model.name,
    on_schema: model.on_schema,
    nodes: [],
    edges: [],
  };

  const nodeTypes = schema ? schema.nodes : model.nodes.map((n) => ({ name: n.node_type, attrs: n.attrs.map((a) => ({ name: a.name })) }));
  const paramInputs = []; // {get: () => void} appliers

  for (const nodeType of nodeTypes) {
    const declared = model.nodes.find((n) => n.node_type === nodeType.name);
    const nodeDoc = { node_type: nodeType.name, attrs: [] };
    doc.nodes.push(nodeDoc);

    sectionHeader(body, `node ${nodeType.name}`);
    for (const attr of nodeType.attrs) {
      const existing = declared?.attrs.find((a) => a.name === attr.name)?.posterior;
      const params = { ...GAUSSIAN_DEFAULTS, ...(existing?.params ?? {}) };
      const attrDoc = { name: attr.name, posterior: { family: 'gaussian', params } };
      nodeDoc.attrs.push(attrDoc);

      el(body, 'div', 'hint').textContent = `${attr.name} ~ Gaussian`;
      const table = el(body, 'table', 'sheet');
      table.innerHTML = '<tr><th>Parameter</th><th>Value</th></tr>';
      // Known-first ordering, then any extras (corr_*, observation_precision).
      const keys = ['prior_mean', 'prior_precision', ...Object.keys(params).filter((k) => !['prior_mean', 'prior_precision'].includes(k))];
      for (const key of keys) {
        const tr = el(table, 'tr');
        el(tr, 'td').textContent = key === 'prior_mean' ? 'mean' : key === 'prior_precision' ? 'precision' : key;
        const td = el(tr, 'td');
        const input = el(td, 'input');
        input.value = String(params[key]);
        paramInputs.push({
          apply: () => {
            const v = readNumber(input, `${nodeType.name}.${attr.name} ${key}`);
            if (v === null) delete attrDoc.posterior.params[key];
            else attrDoc.posterior.params[key] = v;
          },
        });
      }
      // Optional observation_precision if absent.
      if (!('observation_precision' in params)) {
        const tr = el(table, 'tr');
        el(tr, 'td').textContent = 'observation_precision';
        const td = el(tr, 'td');
        const input = el(td, 'input');
        input.placeholder = 'default 1.0';
        paramInputs.push({
          apply: () => {
            const v = readNumber(input, `${nodeType.name}.${attr.name} observation_precision`);
            if (v !== null) attrDoc.posterior.params.observation_precision = v;
          },
        });
      }
    }
  }

  const edgeTypes = schema ? schema.edges : model.edges.map((e) => e.edge_type);
  for (const edgeType of edgeTypes) {
    const declared = model.edges.find((e) => e.edge_type === edgeType)?.exist;
    const edgeDoc = { edge_type: edgeType, exist: null };
    doc.edges.push(edgeDoc);

    sectionHeader(body, `edge ${edgeType}`);
    const familySel = el(body, 'select');
    familySel.innerHTML = '<option value="bernoulli">Bernoulli (independent)</option><option value="categorical">Categorical (competing)</option>';
    familySel.value = declared?.family === 'categorical' ? 'categorical' : 'bernoulli';

    const paramsHost = el(body, 'div');
    const renderParams = () => {
      paramsHost.innerHTML = '';
      const table = el(paramsHost, 'table', 'sheet');
      table.innerHTML = '<tr><th>Parameter</th><th>Value</th></tr>';
      if (familySel.value === 'bernoulli') {
        const params = declared?.family === 'bernoulli' ? { ...BERNOULLI_DEFAULTS, ...declared.params } : { ...BERNOULLI_DEFAULTS };
        edgeDoc.exist = { family: 'bernoulli', params };
        for (const key of ['prior', 'pseudo_count']) {
          const tr = el(table, 'tr');
          el(tr, 'td').textContent = key === 'pseudo_count' ? 'weight (pseudo-count)' : key;
          const input = el(el(tr, 'td'), 'input');
          input.value = String(params[key]);
          input.dataset.key = key;
        }
      } else {
        const prior = declared?.family === 'categorical' && declared.prior?.kind === 'uniform'
          ? { ...declared.prior } : { kind: 'uniform', pseudo_count: 1 };
        edgeDoc.exist = { family: 'categorical', prior };
        const tr = el(table, 'tr');
        el(tr, 'td').textContent = 'pseudo_count (uniform prior)';
        const input = el(el(tr, 'td'), 'input');
        input.value = String(prior.pseudo_count);
        input.dataset.key = 'pseudo_count';
        if (declared?.family === 'categorical' && declared.prior?.kind === 'explicit') {
          el(paramsHost, 'div', 'hint').textContent =
            'This edge uses an explicit concentration prior; applying will rewrite it as uniform. Edit the source directly to keep explicit concentrations.';
        }
      }
    };
    familySel.addEventListener('change', renderParams);
    renderParams();

    paramInputs.push({
      apply: () => {
        for (const input of paramsHost.querySelectorAll('input[data-key]')) {
          const v = readNumber(input, `${edgeType} ${input.dataset.key}`);
          if (v === null) continue;
          if (edgeDoc.exist.family === 'bernoulli') edgeDoc.exist.params[input.dataset.key] = v;
          else edgeDoc.exist.prior.pseudo_count = v;
        }
      },
    });
  }

  buttonBar(body, {
    apply: () => {
      for (const p of paramInputs) p.apply();
      doc.name = nameInput.value.trim();
      ctx.commitBlock('belief_model', name, genBeliefModel(doc), { kind: 'belief_model', name: doc.name },
        { renames: [{ from: name, to: doc.name }] });
    },
    remove: () => ctx.deleteBlock('belief_model', name),
  });
}
