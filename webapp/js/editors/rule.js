// Structural rule editor: pattern rows (or node-only iteration), where
// expression, action statements, and mode. Where/actions remain expression
// text — that's the language — but the rule's shape is edited structurally
// and regenerated via genRule with validate-before-apply.

import { findModel, findSchema, state } from '../state.js';
import { genRule, isForNodeRule } from '../codegen.js';
import { el, field, sectionHeader, rowButton, buttonBar, clearError } from './widgets.js';

export function renderRuleEditor(body, name, ctx) {
  const rule = state.structure?.rules.find((r) => r.name === name);
  if (!rule) { body.innerHTML = '<div class="hint">Not found in last valid program.</div>'; return; }
  const model = findModel(rule.on_model);
  const schema = model ? findSchema(model.on_schema) : null;
  const labels = schema?.nodes.map((n) => n.name) ?? [];
  const edgeTypes = schema?.edges ?? [];

  const doc = JSON.parse(JSON.stringify(rule));
  const nameInput = field(body, 'Name', doc.name);
  el(body, 'div', 'hint').textContent = `on ${doc.on_model}`;
  const readers = [];

  // --- shape: node iteration vs edge patterns ------------------------------------
  const shapeLine = el(body, 'div', 'field');
  el(shapeLine, 'label').textContent = 'Match shape';
  const shapeSel = el(shapeLine, 'select');
  shapeSel.innerHTML =
    '<option value="patterns">edge patterns (join on shared variables)</option>' +
    '<option value="for">every node of a type</option>';
  shapeSel.value = isForNodeRule(doc) ? 'for' : 'patterns';

  const patternsHost = el(body, 'div');
  const renderPatterns = () => {
    patternsHost.innerHTML = '';
    if (shapeSel.value === 'for') {
      const p = isForNodeRule(doc) ? doc.patterns[0] : forPattern(doc.patterns[0], labels);
      doc.patterns = [p];
      const line = el(patternsHost, 'div', 'section-h');
      line.append('for (');
      const varIn = el(line, 'input');
      varIn.value = p.src.var;
      varIn.style.width = '60px';
      line.append(':');
      const labelSel = select(line, labels, p.src.label);
      line.append(')');
      readers.push(() => {
        const v = varIn.value.trim();
        const label = labelSel.value;
        doc.patterns = [{
          src: { var: v, label }, dst: { var: v, label },
          edge: { var: '__for_dummy', type: '__FOR_NODE__' },
        }];
      });
      return;
    }

    if (isForNodeRule(doc)) {
      doc.patterns = [defaultPattern(labels, edgeTypes)];
    }
    sectionHeader(patternsHost, 'Patterns', () => {
      doc.patterns.push(defaultPattern(labels, edgeTypes));
      renderPatterns();
    });
    const table = el(patternsHost, 'table', 'sheet');
    table.innerHTML = '<tr><th>Src var</th><th>:Type</th><th>Edge var</th><th>:Edge</th><th>Dst var</th><th>:Type</th><th></th></tr>';
    for (const p of doc.patterns) {
      const tr = el(table, 'tr');
      const srcVar = inputCell(tr, p.src.var);
      const srcLabel = selectCell(tr, labels, p.src.label);
      const edgeVar = inputCell(tr, p.edge.var);
      const edgeType = selectCell(tr, edgeTypes, p.edge.type);
      const dstVar = inputCell(tr, p.dst.var);
      const dstLabel = selectCell(tr, labels, p.dst.label);
      readers.push(() => {
        p.src = { var: srcVar.value.trim(), label: srcLabel.value };
        p.edge = { var: edgeVar.value.trim(), type: edgeType.value };
        p.dst = { var: dstVar.value.trim(), label: dstLabel.value };
      });
      rowButton(el(tr, 'td', 'row-actions'), '✕', () => {
        doc.patterns = doc.patterns.filter((x) => x !== p);
        renderPatterns();
      });
    }
    el(patternsHost, 'div', 'hint').textContent =
      'Reusing a variable name joins patterns on the same node. A repeated variable within one pattern matches only self-edges.';
  };
  shapeSel.addEventListener('change', renderPatterns);
  renderPatterns();

  // --- where ------------------------------------------------------------------------
  sectionHeader(body, 'Where (condition)');
  const whereWrap = el(body, 'div', 'block-editor');
  const whereText = el(whereWrap, 'textarea');
  whereText.style.minHeight = '60px';
  whereText.value = doc.where ?? '';
  whereText.placeholder = 'e.g. prob(ab) >= 0.7 and E[A.value] > E[B.value]';

  // --- actions ------------------------------------------------------------------------
  sectionHeader(body, 'Actions (one per line)');
  const actionsWrap = el(body, 'div', 'block-editor');
  const actionsText = el(actionsWrap, 'textarea');
  actionsText.style.minHeight = '90px';
  actionsText.value = (doc.actions ?? []).join('\n');
  actionsText.placeholder = 'non_bayesian_nudge B.value to E[A.value] variance=preserve';

  // --- mode ------------------------------------------------------------------------
  const modeLine = el(body, 'div', 'field');
  el(modeLine, 'label').textContent = 'Mode';
  const modeSel = el(modeLine, 'select');
  modeSel.innerHTML = '<option value="for_each">for_each</option><option value="fixpoint">fixpoint</option>';
  modeSel.value = doc.mode === 'fixpoint' ? 'fixpoint' : 'for_each';
  el(body, 'div', 'hint').textContent =
    'Flow transforms currently run rules with for_each semantics; fixpoint applies when a rule is run standalone. Apply a rule multiple times in a flow for multi-hop propagation.';

  buttonBar(body, {
    apply: () => {
      clearError(body);
      for (const read of readers) read();
      doc.name = nameInput.value.trim();
      doc.where = whereText.value.trim() || null;
      doc.actions = actionsText.value.split('\n').map((s) => s.trim()).filter(Boolean);
      doc.mode = shapeSel.value === 'for' ? null : modeSel.value;
      ctx.commitValidated('rule', name, genRule(doc), { kind: 'rule', name: doc.name },
        { renames: [{ from: name, to: doc.name }] });
    },
    remove: () => ctx.deleteBlock('rule', name),
  });
}

function defaultPattern(labels, edgeTypes) {
  return {
    src: { var: 'A', label: labels[0] ?? 'Type' },
    edge: { var: 'ab', type: edgeTypes[0] ?? 'EDGE' },
    dst: { var: 'B', label: labels[0] ?? 'Type' },
  };
}

function forPattern(existing, labels) {
  const label = existing?.src.label ?? labels[0] ?? 'Type';
  return {
    src: { var: 'P', label }, dst: { var: 'P', label },
    edge: { var: '__for_dummy', type: '__FOR_NODE__' },
  };
}

function select(parent, options, value) {
  const sel = el(parent, 'select');
  for (const opt of options) {
    const o = el(sel, 'option');
    o.value = opt; o.textContent = opt;
  }
  if (value && !options.includes(value)) {
    const o = el(sel, 'option');
    o.value = value; o.textContent = value;
  }
  sel.value = value || options[0] || '';
  return sel;
}

function selectCell(tr, options, value) {
  return select(el(tr, 'td'), options, value);
}

function inputCell(tr, value) {
  const input = el(el(tr, 'td'), 'input');
  input.value = value ?? '';
  return input;
}
