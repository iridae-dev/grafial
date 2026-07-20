// Form editor for schema declarations: node types with attributes, edge types.
//
// Renames cascade: original identifiers are tracked per row, and on Apply the
// old → new pairs are applied program-wide (outside strings/comments) so
// belief models, evidence, rules, and flows follow along.

import { findSchema } from '../state.js';
import { genSchema } from '../codegen.js';
import { el, field, sectionHeader, rowButton, buttonBar } from './widgets.js';

export function renderSchemaEditor(body, name, ctx) {
  const schema = findSchema(name);
  if (!schema) { body.innerHTML = '<div class="hint">Not found in last valid program.</div>'; return; }

  // Working copy; `orig` fields remember pre-edit identifiers for cascading.
  const doc = {
    name: schema.name,
    nodes: schema.nodes.map((n) => ({
      name: n.name, orig: n.name,
      attrs: n.attrs.map((a) => ({ name: a.name, orig: a.name, type: a.type })),
    })),
    edges: schema.edges.map((e) => ({ name: e, orig: e })),
  };

  const nameInput = field(body, 'Name', doc.name);

  const nodesHost = el(body, 'div');
  const renderNodes = () => {
    nodesHost.innerHTML = '';
    sectionHeader(nodesHost, 'Node types', () => {
      doc.nodes.push({ name: `Type${doc.nodes.length + 1}`, orig: null, attrs: [{ name: 'value', orig: null, type: 'Real' }] });
      renderNodes();
    });
    for (const node of doc.nodes) {
      const head = el(nodesHost, 'div', 'section-h');
      const nameIn = el(head, 'input');
      nameIn.value = node.name;
      nameIn.addEventListener('input', () => { node.name = nameIn.value.trim(); });
      rowButton(head, '+ attr', () => { node.attrs.push({ name: `attr${node.attrs.length + 1}`, orig: null, type: 'Real' }); renderNodes(); });
      rowButton(head, '✕', () => { doc.nodes = doc.nodes.filter((n) => n !== node); renderNodes(); });

      const table = el(nodesHost, 'table', 'sheet');
      table.innerHTML = '<tr><th>Attribute</th><th>Type</th><th></th></tr>';
      for (const attr of node.attrs) {
        const tr = el(table, 'tr');
        const nameField = el(el(tr, 'td'), 'input');
        nameField.value = attr.name;
        nameField.addEventListener('input', () => { attr.name = nameField.value.trim(); });
        const typeField = el(el(tr, 'td'), 'input');
        typeField.value = attr.type;
        typeField.addEventListener('input', () => { attr.type = typeField.value.trim() || 'Real'; });
        rowButton(el(tr, 'td', 'row-actions'), '✕', () => {
          node.attrs = node.attrs.filter((a) => a !== attr);
          renderNodes();
        });
      }
    }
  };
  renderNodes();

  const edgesHost = el(body, 'div');
  const renderEdges = () => {
    edgesHost.innerHTML = '';
    sectionHeader(edgesHost, 'Edge types', () => { doc.edges.push({ name: `EDGE${doc.edges.length + 1}`, orig: null }); renderEdges(); });
    for (const edge of doc.edges) {
      const row = el(edgesHost, 'div', 'section-h');
      const input = el(row, 'input');
      input.value = edge.name;
      input.addEventListener('input', () => { edge.name = input.value.trim(); });
      rowButton(row, '✕', () => { doc.edges = doc.edges.filter((e) => e !== edge); renderEdges(); });
    }
  };
  renderEdges();

  el(body, 'div', 'hint').textContent =
    'Renames cascade through belief models, evidence, rules, and flows (you will be asked to confirm). Removing types that are still referenced surfaces validation errors in the status bar.';

  buttonBar(body, {
    apply: () => {
      doc.name = nameInput.value.trim();

      const renames = [];
      if (doc.name !== name) renames.push({ from: name, to: doc.name });
      for (const node of doc.nodes) {
        if (node.orig && node.orig !== node.name) renames.push({ from: node.orig, to: node.name });
        for (const attr of node.attrs) {
          if (attr.orig && attr.orig !== attr.name) renames.push({ from: attr.orig, to: attr.name });
        }
      }
      for (const edge of doc.edges) {
        if (edge.orig && edge.orig !== edge.name) renames.push({ from: edge.orig, to: edge.name });
      }

      const genDoc = {
        name: doc.name,
        nodes: doc.nodes.map((n) => ({ name: n.name, attrs: n.attrs.map((a) => ({ name: a.name, type: a.type })) })),
        edges: doc.edges.map((e) => e.name),
      };
      ctx.commitBlock('schema', name, genSchema(genDoc), { kind: 'schema', name: doc.name }, { renames });
    },
    remove: () => ctx.deleteBlock('schema', name),
  });
}
