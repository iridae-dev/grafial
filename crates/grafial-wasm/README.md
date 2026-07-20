# grafial-wasm

WebAssembly bindings for Grafial: parse, inspect, and execute Grafial programs
in the browser. This crate is the foundation for browser tooling — in
particular a visual, graph-based UI for composing Grafial programs.

## Building

```bash
rustup target add wasm32-unknown-unknown
cargo install wasm-pack

./scripts/build_wasm.sh            # ES module for browsers (default)
./scripts/build_wasm.sh bundler    # for webpack/vite
./scripts/build_wasm.sh nodejs     # CommonJS for Node
```

Output lands in `crates/grafial-wasm/pkg/`: the `.wasm` binary, JS glue code,
and TypeScript definitions.

## API

All functions take Grafial source text and return JSON strings (`JSON.parse`
on the JS side). Errors surface as thrown exceptions, except `check`, which
reports parse/validation failures *inside* its JSON so an editor can render
them as diagnostics.

| Function | Returns |
|---|---|
| `version()` | crate version string |
| `check(source)` | `{ valid, error, style_lints, statistical_lints }` — each lint has `code`, `message`, `range` (line/column), and style lints carry a `replacement` for quick fixes |
| `format_canonical(source)` | source rewritten to canonical style |
| `list_flows(source)` | flow names in program order |
| `program_structure(source)` | structural description of every declaration (see below) |
| `run_flow(source, flowName)` | full execution result (see below) |

### `program_structure` — the visual-composer surface

Describes the program as data a UI can render as a graph:

- `schemas`: node types with attributes, edge types
- `belief_models`: `on_schema` link, per-attribute posterior families and
  parameters, per-edge-type existence posteriors
- `evidences`: `on_model` link, observation counts
- `rules`: `on_model` link, pattern shapes (`src`/`edge`/`dst` variables and
  labels), mode, action count
- `flows`: per-graph expressions as tagged JSON (`from_evidence`,
  `from_graph`, `pipeline` with its transform list, `select_model`), metric
  names, exports/imports, and `needs_prior` — whether the flow consumes
  earlier flows' outputs

Cross-flow dataflow edges for the UI: connect a flow's `exports[].alias` /
`metric_exports[].alias` to later flows' `from_graph` aliases /
`metric_imports[].source_alias`.

### `run_flow` — execution results

Runs the named flow, automatically executing prerequisite flows in program
order when the target imports graphs or metrics (same semantics as the CLI).
Returns:

- `metrics`, `metric_exports`: scalar values
- `graphs`, `exports`, `snapshots`: every belief graph fully serialized —
  nodes (`id`, type `label`, evidence instance `name` like `"Alice"` or null,
  posterior `attrs` with `mean`/`variance`) and edges (`id`, `src`, `dst`,
  `type`, existence `prob`)
- `intervention_audit`: which rules fired, match and action counts
- `inference_diagnostics`: belief-propagation convergence data

## Browser usage

```html
<script type="module">
  import init, { check, program_structure, run_flow } from "./pkg/grafial_wasm.js";
  await init();

  const source = await (await fetch("./social.grafial")).text();

  const diagnostics = JSON.parse(check(source));
  if (!diagnostics.valid) throw new Error(diagnostics.error);

  const structure = JSON.parse(program_structure(source)); // render as node graph
  const result = JSON.parse(run_flow(source, "Demo"));     // render belief graphs
  console.log(result.metrics, result.exports);
</script>
```

## Notes and known gaps

- The engine's JIT (`jit`/`aot`, Cranelift) and `parallel` features are
  native-only; this crate uses the interpreter path, which is the default.
- Node `name` is the evidence instance label (`"Alice"`); nodes created by
  other means (e.g. programmatic construction) have `name: null`, so fall
  back to `label #id` when rendering.
- The pure-Rust API lives in `grafial_wasm::api` and returns
  `serde_json::Value`; it is unit-tested natively (`cargo test -p
  grafial-wasm`), so CI does not need a browser.
