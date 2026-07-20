# Grafial Composer

A visual web app for loading, creating, editing, running, and saving Grafial
programs, powered entirely in-browser by the `grafial-wasm` engine build. No
server-side execution, no JS framework, no build step — plain ES modules.

```bash
./scripts/serve_composer.sh          # builds wasm, serves the repo root
# open http://localhost:8000/webapp/
```

## Design

A Grafial program is not one graph, so this is deliberately **not** a
free-form node-and-wire canvas. The declarations have different natural
shapes, and each gets the editor that fits:

| Declaration | Shape | Editor |
|---|---|---|
| `schema` | type definitions | form (node types, attributes, edge types) |
| `belief_model` | distributions over schema | form (posterior family + parameters per attribute / edge type) |
| `evidence` | tabular observations | **mini-spreadsheet** per node type + edge table + CSV import |
| `rule` | pattern + condition + actions | structural editor: pattern rows or node-iteration, where/actions as expressions, validate-before-apply |
| `flow` | linear pipelines | structural pipeline builder: add/remove/reorder transforms, metrics with expressions, exports/imports |
| results | posterior graphs + scalars | graph view, metrics table, rule audit |

The pieces:

- **Program map** (main view): declaration cards in kind columns —
  Schemas → Belief Models → Evidence / Rules → Flows — with drawn dependency
  edges (`on` relationships, evidence/rules feeding flows, and dashed
  cross-flow export→import edges). Click a card to open it in the inspector;
  `+` adds a new declaration of that kind.
- **Inspector** (dockable): context-sensitive editor for the selected card.
  Dock left/right with ⇄, resize by dragging the divider, close with ×.
- **Source tab**: the full program text with live diagnostics (parse errors,
  canonical-style lints, statistical guardrails). **Source text is the single
  source of truth** — every visual editor regenerates one declaration block
  and splices it back, so files stay git-friendly and CLI-compatible, and
  nothing the composer can't render is ever lost.
- **Results tab**: after Run — belief graphs in a deterministic force-directed
  layout (probability-weighted springs; same graph always lays out the same
  way), nodes labeled by evidence instance name, edge opacity encoding
  existence probability (click a node for its posteriors in the inspector),
  metrics, rule-firing audit (zero-match rules are called out), and inference
  diagnostics.
- **Rename cascade**: renaming a schema, node type, attribute, edge type,
  belief model, evidence, or rule from its editor renames references across
  the whole program (identifier-exact, strings and comments untouched), with a
  confirmation showing the occurrence count.

Structural editing is powered by an AST → source printer in the engine
(`grafial-frontend/src/printer.rs`): the wasm `program_structure()` API serves
metric expressions, prune predicates, and rule where/action bodies as
canonical text, and the editors regenerate declaration blocks from structured
documents. A corpus test regenerates every flow and rule in every repository
example and requires byte-for-byte identical execution results.

## Evidence spreadsheet

- One table per schema node type; rows are observation groups, columns are
  attributes. Cells take `value` or `value @ precision`; blank = unobserved.
- Repeated rows for the same instance are legal and accumulate updates —
  that's Bayesian evidence, not a keyed record store.
- Edge observations: src/dst instance names with mode
  (`present`/`absent`/`chosen`/`unchosen`/`forced_choice`).
- **CSV import** per node type: first column = instance name, remaining
  headers must match attribute names; cells may use `value @ precision`.

## Development

Pure logic (block splicing, code generation, CSV) lives in framework-free
modules with headless tests:

```bash
node --test webapp/tests/pure.test.mjs

# Round-trip tests through the real engine (build the node package first):
cd crates/grafial-wasm && wasm-pack build --target nodejs --out-dir ../../webapp/tests/pkg-node
node --test webapp/tests/roundtrip.test.mjs
```

Layout: `js/blockedit.js` (declaration-block text surgery), `js/codegen.js`
(document model → canonical Grafial text), `js/csv.js`, `js/state.js`
(source-as-truth store with undo + autosave), `js/map.js`, `js/inspector.js`
+ `js/editors/*`, `js/run.js`, `js/app.js`.
