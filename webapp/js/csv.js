// Minimal CSV parser for evidence import: quoted fields, embedded commas,
// doubled quotes, CRLF. Returns array of rows (arrays of strings).

export function parseCSV(text) {
  const rows = [];
  let row = [];
  let field = '';
  let inQuotes = false;
  let i = 0;
  const n = text.length;

  const endField = () => { row.push(field); field = ''; };
  const endRow = () => { endField(); rows.push(row); row = []; };

  while (i < n) {
    const c = text[i];
    if (inQuotes) {
      if (c === '"') {
        if (text[i + 1] === '"') { field += '"'; i += 2; continue; }
        inQuotes = false; i += 1; continue;
      }
      field += c; i += 1; continue;
    }
    if (c === '"') { inQuotes = true; i += 1; continue; }
    if (c === ',') { endField(); i += 1; continue; }
    if (c === '\r') { i += 1; continue; }
    if (c === '\n') { endRow(); i += 1; continue; }
    field += c; i += 1;
  }
  if (field !== '' || row.length > 0) endRow();
  // Drop trailing fully-empty rows
  while (rows.length > 0 && rows[rows.length - 1].every((f) => f === '')) rows.pop();
  return rows;
}

/**
 * Maps parsed CSV into evidence node rows.
 * The first column is the instance name; remaining headers must match
 * attribute names (unknown columns are reported, not silently dropped).
 * Blank cells mean "no observation". Values may carry per-cell precision as
 * `12.5 @ 10` (value @ precision).
 */
export function csvToNodeRows(rows, nodeType, knownAttrs) {
  if (rows.length < 2) throw new Error('CSV needs a header row and at least one data row');
  const [header, ...data] = rows;
  const attrCols = header.slice(1).map((h) => h.trim());
  const unknown = attrCols.filter((a) => !knownAttrs.includes(a));
  if (unknown.length > 0) {
    throw new Error(`unknown attribute column(s): ${unknown.join(', ')} — expected any of: ${knownAttrs.join(', ')}`);
  }

  const nodeRows = [];
  for (const [rowIdx, cells] of data.entries()) {
    const name = (cells[0] ?? '').trim();
    if (!name) throw new Error(`row ${rowIdx + 2}: missing instance name in first column`);
    const values = {};
    for (const [colIdx, attr] of attrCols.entries()) {
      const raw = (cells[colIdx + 1] ?? '').trim();
      if (raw === '') continue;
      values[attr] = parseCell(raw, rowIdx + 2, attr);
    }
    nodeRows.push({ type: nodeType, name, values });
  }
  return nodeRows;
}

/** Parses "value" or "value @ precision" into {value, precision}. */
export function parseCell(raw, rowNum, attr) {
  const parts = raw.split('@').map((p) => p.trim());
  const value = Number(parts[0]);
  if (!Number.isFinite(value)) {
    throw new Error(`row ${rowNum}, column '${attr}': not a number: '${parts[0]}'`);
  }
  if (parts.length === 1) return { value, precision: null };
  const precision = Number(parts[1]);
  if (!Number.isFinite(precision) || precision <= 0) {
    throw new Error(`row ${rowNum}, column '${attr}': precision must be a positive number, got '${parts[1]}'`);
  }
  return { value, precision };
}
