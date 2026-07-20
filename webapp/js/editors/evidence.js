// Evidence editor: a mini-spreadsheet per node type (rows = observation
// groups, columns = attributes), an edge-observation table, and CSV import.
//
// Cells accept "value" or "value @ precision". A blank cell means "no
// observation". Repeated rows for the same instance are legal — each row is
// one observation group and updates accumulate (proper Bayesian evidence).

import { state, findEvidence, schemaForModel } from '../state.js';
import { genEvidence, evidenceDocFromStructure } from '../codegen.js';
import { parseCSV, csvToNodeRows, parseCell } from '../csv.js';
import { el, field, sectionHeader, rowButton, buttonBar, showError, clearError } from './widgets.js';

export function renderEvidenceEditor(body, name, ctx) {
  const evidence = findEvidence(name);
  if (!evidence) { body.innerHTML = '<div class="hint">Not found in last valid program.</div>'; return; }
  const schema = schemaForModel(evidence.on_model);
  if (!schema) { body.innerHTML = '<div class="hint">The belief model / schema chain is broken; fix it first.</div>'; return; }

  const doc = evidenceDocFromStructure(evidence);
  const nameInput = field(body, 'Name', doc.name);
  el(body, 'div', 'hint').textContent = `on ${evidence.on_model} · cells accept "value" or "value @ precision"; blank = unobserved`;

  // --- node observation tables, one per schema node type -------------------
  const tablesHost = el(body, 'div');
  const cellReaders = [];

  const renderTables = () => {
    tablesHost.innerHTML = '';
    cellReaders.length = 0;

    for (const nodeType of schema.nodes) {
      const attrs = nodeType.attrs.map((a) => a.name);
      const rows = doc.nodeRows.filter((r) => r.type === nodeType.name);

      sectionHeader(tablesHost, `${nodeType.name} observations`, () => {
        doc.nodeRows.push({ type: nodeType.name, name: '', values: {} });
        renderTables();
      });

      const table = el(tablesHost, 'table', 'sheet');
      const head = el(table, 'tr');
      el(head, 'th').textContent = 'Instance';
      for (const attr of attrs) el(head, 'th').textContent = attr;
      el(head, 'th');

      for (const row of rows) {
        const tr = el(table, 'tr');
        const nameIn = el(el(tr, 'td'), 'input');
        nameIn.value = row.name;
        nameIn.placeholder = 'name';
        cellReaders.push(() => { row.name = nameIn.value.trim(); });

        for (const attr of attrs) {
          const input = el(el(tr, 'td'), 'input');
          const cell = row.values[attr];
          input.value = cell == null ? '' : cell.precision != null ? `${cell.value} @ ${cell.precision}` : String(cell.value);
          cellReaders.push(() => {
            const raw = input.value.trim();
            if (raw === '') delete row.values[attr];
            else row.values[attr] = parseCell(raw, '?', attr);
          });
        }
        rowButton(el(tr, 'td', 'row-actions'), '✕', () => {
          doc.nodeRows = doc.nodeRows.filter((r) => r !== row);
          renderTables();
        });
      }

      // CSV import for this node type.
      const csvBar = el(tablesHost, 'div', 'hint');
      rowButton(csvBar, `Import CSV → ${nodeType.name}`, () => importCSV(nodeType.name, attrs));
      csvBar.append(' first column = instance name; headers must match attributes');
    }
  };

  const importCSV = (nodeType, attrs) => {
    const picker = document.getElementById('file-csv');
    picker.onchange = async () => {
      const file = picker.files[0];
      picker.value = '';
      if (!file) return;
      clearError(body);
      try {
        const rows = csvToNodeRows(parseCSV(await file.text()), nodeType, attrs);
        readAllCells(); // keep in-progress edits
        doc.nodeRows.push(...rows);
        renderTables();
        renderEdges();
      } catch (err) {
        showError(body, err);
      }
    };
    picker.click();
  };

  // --- edge observations -----------------------------------------------------
  const edgesHost = el(body, 'div');
  const renderEdges = () => {
    edgesHost.innerHTML = '';
    sectionHeader(edgesHost, 'Edge observations', () => {
      doc.edgeRows.push({
        edge_type: schema.edges[0] ?? '',
        src_type: schema.nodes[0]?.name ?? '', src: '',
        dst_type: schema.nodes[0]?.name ?? '', dst: '',
        mode: 'present',
      });
      renderEdges();
    });

    if (schema.edges.length === 0) {
      el(edgesHost, 'div', 'hint').textContent = 'The schema declares no edge types.';
      return;
    }

    const table = el(edgesHost, 'table', 'sheet');
    table.innerHTML = '<tr><th>Edge</th><th>Src type</th><th>Src</th><th>Dst type</th><th>Dst</th><th>Mode</th><th></th></tr>';
    for (const row of doc.edgeRows) {
      const tr = el(table, 'tr');
      const typeSel = selectCell(tr, schema.edges, row.edge_type);
      const srcTypeSel = selectCell(tr, schema.nodes.map((n) => n.name), row.src_type);
      const srcIn = inputCell(tr, row.src);
      const dstTypeSel = selectCell(tr, schema.nodes.map((n) => n.name), row.dst_type);
      const dstIn = inputCell(tr, row.dst);
      const modeSel = selectCell(tr, ['present', 'absent', 'chosen', 'unchosen', 'forced_choice'], row.mode);
      cellReaders.push(() => {
        row.edge_type = typeSel.value; row.src_type = srcTypeSel.value; row.src = srcIn.value.trim();
        row.dst_type = dstTypeSel.value; row.dst = dstIn.value.trim(); row.mode = modeSel.value;
      });
      rowButton(el(tr, 'td', 'row-actions'), '✕', () => {
        doc.edgeRows = doc.edgeRows.filter((r) => r !== row);
        renderEdges();
      });
    }
  };

  // --- edge weight observations (shown only when present) --------------------
  const weightsHost = el(body, 'div');
  const renderWeights = () => {
    weightsHost.innerHTML = '';
    if ((doc.weightRows ?? []).length === 0) return;
    sectionHeader(weightsHost, 'Edge weight observations');
    const table = el(weightsHost, 'table', 'sheet');
    table.innerHTML = '<tr><th>Edge</th><th>Src</th><th>Dst</th><th>Weight</th><th></th></tr>';
    for (const row of doc.weightRows) {
      const tr = el(table, 'tr');
      el(tr, 'td').textContent = row.edge_type;
      el(tr, 'td').textContent = row.src;
      el(tr, 'td').textContent = row.dst;
      const input = el(el(tr, 'td'), 'input');
      input.value = row.precision != null ? `${row.value} @ ${row.precision}` : String(row.value);
      cellReaders.push(() => {
        const parsed = parseCell(input.value.trim(), '?', 'weight');
        row.value = parsed.value; row.precision = parsed.precision;
      });
      rowButton(el(tr, 'td', 'row-actions'), '✕', () => {
        doc.weightRows = doc.weightRows.filter((r) => r !== row);
        renderWeights();
      });
    }
  };

  const readAllCells = () => { for (const read of cellReaders) read(); };

  renderTables();
  renderEdges();
  renderWeights();

  buttonBar(body, {
    apply: () => {
      clearError(body);
      readAllCells();
      doc.nodeRows = doc.nodeRows.filter((r) => r.name !== '' || Object.keys(r.values).length > 0);
      for (const r of doc.nodeRows) {
        if (r.name === '') throw new Error('every observation row needs an instance name');
      }
      doc.edgeRows = doc.edgeRows.filter((r) => r.src !== '' || r.dst !== '');
      for (const r of doc.edgeRows) {
        if (r.src === '' || r.dst === '') throw new Error('edge observations need both src and dst instance names');
      }
      doc.name = nameInput.value.trim();
      ctx.commitBlock('evidence', name, genEvidence(doc), { kind: 'evidence', name: doc.name },
        { renames: [{ from: name, to: doc.name }] });
    },
    remove: () => ctx.deleteBlock('evidence', name),
  });
}

function selectCell(tr, options, value) {
  const sel = el(el(tr, 'td'), 'select');
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

function inputCell(tr, value) {
  const input = el(el(tr, 'td'), 'input');
  input.value = value ?? '';
  return input;
}
