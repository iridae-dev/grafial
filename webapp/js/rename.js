// Identifier renaming across Grafial source, for cascading renames from the
// visual editors (schema/model/evidence/rule names, node/edge types,
// attribute names).
//
// Replacement is word-boundary exact and skips string literals and comments —
// instance names like "Alice" are strings and never touched. This is textual,
// not scope-aware: renaming an identifier that is also used for something
// unrelated (e.g. an attribute named `value` colliding with fold's `value`
// accumulator) renames both, so callers should surface the occurrence count
// to the user.

const WORD = /[A-Za-z0-9_]/;

/**
 * Counts code occurrences (outside strings/comments) of an identifier.
 */
export function countIdentifier(source, name) {
  return scan(source, name, null);
}

/**
 * Renames all code occurrences of `oldName` to `newName`.
 * Returns {source, count}.
 */
export function renameIdentifier(source, oldName, newName) {
  if (oldName === newName) return { source, count: 0 };
  const out = [];
  const count = scan(source, oldName, (start, end) => out.push([start, end]));
  let result = source;
  for (const [start, end] of out.reverse()) {
    result = result.slice(0, start) + newName + result.slice(end);
  }
  return { source: result, count };
}

/**
 * Applies a list of renames [{from, to}] in one pass order-independently:
 * all matches are located against the original text first, so renames cannot
 * cascade into each other (A->B, B->C never turns A into C).
 */
export function renameAll(source, renames) {
  const jobs = [];
  let total = 0;
  for (const { from, to } of renames) {
    if (from === to) continue;
    scan(source, from, (start, end) => jobs.push({ start, end, to }));
    total += 1;
  }
  if (total === 0 || jobs.length === 0) return { source, count: 0 };
  jobs.sort((a, b) => b.start - a.start);
  let result = source;
  for (const { start, end, to } of jobs) {
    result = result.slice(0, start) + to + result.slice(end);
  }
  return { source: result, count: jobs.length };
}

/** Core scanner: walks code (skipping strings/comments), invoking cb per hit. */
function scan(source, name, cb) {
  const n = source.length;
  let i = 0;
  let count = 0;
  while (i < n) {
    const c = source[i];
    if (c === '/' && source[i + 1] === '/') {
      const nl = source.indexOf('\n', i);
      i = nl === -1 ? n : nl + 1;
      continue;
    }
    if (c === '/' && source[i + 1] === '*') {
      const close = source.indexOf('*/', i + 2);
      i = close === -1 ? n : close + 2;
      continue;
    }
    if (c === '"') {
      i += 1;
      while (i < n && source[i] !== '"') i += source[i] === '\\' ? 2 : 1;
      i += 1;
      continue;
    }
    if (WORD.test(c)) {
      let j = i;
      while (j < n && WORD.test(source[j])) j += 1;
      if (j - i === name.length && source.slice(i, j) === name) {
        count += 1;
        if (cb) cb(i, j);
      }
      i = j;
      continue;
    }
    i += 1;
  }
  return count;
}
