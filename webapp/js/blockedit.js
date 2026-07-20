// Locate and replace top-level declaration blocks in Grafial source text.
//
// The composer treats source text as ground truth: visual editors regenerate
// one declaration block and splice it back. This module provides the
// text-surgery primitives, comment/string-aware so braces inside strings or
// comments don't confuse the matcher.

const DECL_KEYWORDS = ['schema', 'belief_model', 'evidence', 'rule', 'flow'];

/**
 * Scans source for top-level declarations.
 * Returns [{kind, name, on, start, end, headerStart}] where [start, end) spans
 * the whole block including the closing brace, and `on` is the `on X` target
 * (null for schemas).
 */
export function findDeclarations(source) {
  const decls = [];
  let i = 0;
  const n = source.length;

  while (i < n) {
    i = skipTrivia(source, i);
    if (i >= n) break;

    const word = readWord(source, i);
    if (!DECL_KEYWORDS.includes(word.text)) {
      // Not a declaration start; skip to next line to resync.
      const nl = source.indexOf('\n', i);
      i = nl === -1 ? n : nl + 1;
      continue;
    }

    const headerStart = i;
    let j = skipTrivia(source, word.end);
    const name = readWord(source, j);
    if (!name.text) { i = word.end; continue; }
    j = skipTrivia(source, name.end);

    let on = null;
    const maybeOn = readWord(source, j);
    if (maybeOn.text === 'on') {
      j = skipTrivia(source, maybeOn.end);
      const target = readWord(source, j);
      on = target.text || null;
      j = skipTrivia(source, target.end);
    }

    if (source[j] !== '{') { i = name.end; continue; }
    const close = matchBrace(source, j);
    if (close === -1) break; // unbalanced; give up on the rest

    decls.push({ kind: word.text, name: name.text, on, start: headerStart, end: close + 1 });
    i = close + 1;
  }
  return decls;
}

/** Finds one declaration by kind + name, or null. */
export function findDeclaration(source, kind, name) {
  return findDeclarations(source).find((d) => d.kind === kind && d.name === name) ?? null;
}

/**
 * Replaces the block identified by kind + name with newText.
 * Throws if the declaration is not found.
 */
export function replaceBlock(source, kind, name, newText) {
  const decl = findDeclaration(source, kind, name);
  if (!decl) throw new Error(`declaration ${kind} '${name}' not found`);
  return source.slice(0, decl.start) + newText.trim() + source.slice(decl.end);
}

/** Removes the block (and one trailing newline run) identified by kind + name. */
export function removeBlock(source, kind, name) {
  const decl = findDeclaration(source, kind, name);
  if (!decl) throw new Error(`declaration ${kind} '${name}' not found`);
  let end = decl.end;
  while (end < source.length && source[end] === '\n') end += 1;
  let start = decl.start;
  while (start > 0 && source[start - 1] === '\n') start -= 1;
  return source.slice(0, start) + '\n\n'.slice(0, start > 0 ? 2 : 0) + source.slice(end);
}

/**
 * Appends a new declaration, placed after the last declaration of the same
 * kind when one exists (keeps files grouped), else at the end of the file.
 */
export function appendDeclaration(source, kind, newText) {
  const decls = findDeclarations(source);
  const sameKind = decls.filter((d) => d.kind === kind);
  const text = newText.trim();
  if (sameKind.length > 0) {
    const at = sameKind[sameKind.length - 1].end;
    return source.slice(0, at) + '\n\n' + text + source.slice(at);
  }
  const base = source.trimEnd();
  return (base ? base + '\n\n' : '') + text + '\n';
}

// --- lexical helpers -------------------------------------------------------

function readWord(source, i) {
  let j = i;
  while (j < source.length && /[A-Za-z0-9_]/.test(source[j])) j += 1;
  return { text: source.slice(i, j), end: j };
}

/** Skips whitespace and comments. */
function skipTrivia(source, i) {
  const n = source.length;
  while (i < n) {
    const c = source[i];
    if (c === ' ' || c === '\t' || c === '\n' || c === '\r') { i += 1; continue; }
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
    break;
  }
  return i;
}

/** Given index of '{', returns index of its matching '}', or -1. */
function matchBrace(source, open) {
  let depth = 0;
  let i = open;
  const n = source.length;
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
    if (c === '{') depth += 1;
    if (c === '}') {
      depth -= 1;
      if (depth === 0) return i;
    }
    i += 1;
  }
  return -1;
}
