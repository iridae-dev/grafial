# Grafial

Grafial is a domain-specific language and runtime for Bayesian reasoning over graphs.
It is designed for systems where nodes, edges, and attributes are uncertain and should
be updated as new evidence arrives.

This repository is a monorepo containing the language frontend, IR, execution engine,
CLI, Python bindings, WebAssembly bindings, a browser-based visual composer,
tests, and benchmarks.

[![All Tests](https://github.com/iridae-dev/grafial/actions/workflows/phase6-release-gate.yml/badge.svg)](https://github.com/iridae-dev/grafial/actions/workflows/phase6-release-gate.yml) [![Release](https://img.shields.io/github/v/release/iridae-dev/grafial)](
  https://github.com/iridae-dev/grafial/releases/latest)


## Install

**Try the Composer online:** [https://grafial.iridae.com/](https://grafial.iridae.com/)

**CLI:** Download the latest `grafial-<version>-<platform>.tar.gz` (or `.zip` on Windows) from [Releases](https://github.com/iridae-dev/grafial/releases), extract, and add the binary to your PATH.

**Python:** `pip install grafial`

**VS Code:** Install the Grafial extension from [`crates/grafial-vscode`](crates/grafial-vscode); download `grafial-lsp` from [Releases](https://github.com/iridae-dev/grafial/releases) or build with `cargo build -p grafial-lsp --release`.

## Grafial Composer (browser)

**Hosted:** [https://grafial.iridae.com/](https://grafial.iridae.com/)

A visual editor for loading, creating, editing, running, and saving Grafial
programs — entirely in the browser via the WebAssembly engine build:

```bash
./scripts/serve_composer.sh                # develop locally
# open http://localhost:8000/webapp/

./scripts/build_composer_dist.sh           # static bundle -> dist/composer,
                                           # upload to any web host
```

It provides a program map with dependency edges, a dockable inspector with
per-declaration editors (schema/model forms, a spreadsheet for evidence with
CSV import, structural pipeline/rule builders with program-wide rename
cascade), and a results view that renders posterior belief graphs
(force-directed, labeled by evidence names), metrics, and rule-firing audits. See
[webapp/README.md](webapp/README.md) for the design and
[crates/grafial-wasm/README.md](crates/grafial-wasm/README.md) for the
underlying JSON API.

## What Grafial Gives You

- A declarative DSL for probabilistic graph programs.
- Bayesian posteriors on graph structure and attributes.
- Rule and flow execution over uncertain graphs.
- Deterministic execution with optional performance features.
- Tooling for CLI, Python integration, testing, and benchmarking.
- A WebAssembly build and in-browser visual composer.

## Core Concepts

Grafial programs are organized around five building blocks:

1. `schema`: typed node/edge structure.
2. `belief_model`: priors over attributes, edge existence, and optional continuous edge weights.
3. `evidence`: observed data used to update beliefs.
4. `rule`: graph pattern + condition + actions.
5. `flow`: pipeline that builds graphs, applies transforms, computes metrics, and exports results.

Common flow transforms:

- `apply_rule RuleName`
- `apply_ruleset { RuleA, RuleB, ... }`
- `infer_beliefs` (deterministic loopy belief propagation on independent edges)
- `prune_edges EdgeType where prob(edge) < threshold` (or `weight(edge)` predicates for weighted edges)
- `snapshot "name"`

## Quick Start (build from source)

Prerequisites:

- Rust stable toolchain
- Cargo
- Optional: `nix-shell` (from [`shell.nix`](shell.nix)) for a pinned dev environment

Build and run the CLI:

```bash
cargo build --workspace
cargo install --path crates/grafial-cli

# Validate program and list available flows
grafial crates/grafial-examples/social.grafial --list-flows

# Execute a flow
grafial crates/grafial-examples/social.grafial --flow Demo

# JSON output
grafial crates/grafial-examples/social.grafial --flow Demo --output json
```

Style linting and canonical rewrites:

```bash
grafial crates/grafial-examples/social.grafial --lint-style
grafial crates/grafial-examples/social.grafial --fix-style
```

## Example Program

[`crates/grafial-examples/minimal.grafial`](crates/grafial-examples/minimal.grafial):

```grafial
schema Minimal {
  node Entity {
    value: Real
  }
  edge CONNECTED { }
}

belief_model MinimalBeliefs on Minimal {
  node Entity {
    value ~ Gaussian(mean=0.0, precision=0.01)
  }
  edge CONNECTED {
    exist ~ Bernoulli(prior=0.5, weight=2.0)
  }
}

evidence MinimalEvidence on MinimalBeliefs {
  Entity { "A" { value: 1.0 } }
  CONNECTED(Entity -> Entity) { "A" -> "B" }
}

flow MinimalFlow on MinimalBeliefs {
  graph g = from_evidence MinimalEvidence
  metric total = nodes(Entity) |> sum(by=E[node.value])
  export g as "output"
}
```

## Monorepo Layout

Key crates in [`crates/`](crates/):

- [`grafial-frontend`](crates/grafial-frontend/): parser, AST, validation, style linting/formatting.
- [`grafial-ir`](crates/grafial-ir/): lowered IR and optimization passes.
- [`grafial-core`](crates/grafial-core/): execution engine, graph model, rule/flow runtime, kernels.
- [`grafial-cli`](crates/grafial-cli/): `grafial` command-line tool.
- [`grafial-python`](crates/grafial-python/): PyO3 bindings.
- [`grafial-lsp`](crates/grafial-lsp/): language server implementation.
- [`grafial-tests`](crates/grafial-tests/): integration test crate.
- [`grafial-benches`](crates/grafial-benches/): benchmark crate.
- [`grafial-examples`](crates/grafial-examples/): sample `.grafial` programs.
- [`grafial-wasm`](crates/grafial-wasm/): WebAssembly bindings (JSON API for browser tooling).

Plus [`webapp/`](webapp/): the Grafial Composer, a no-build browser app on top of
[`grafial-wasm`](crates/grafial-wasm/). See also the [examples index](documentation/EXAMPLES.md).

## Engine Feature Flags (grafial-core)

[`crates/grafial-core/Cargo.toml`](crates/grafial-core/Cargo.toml) defines optional features. Maturity labels:

| Feature | Maturity | Notes |
|---|---|---|
| `parallel` | **supported** | Parallel evidence/metric paths; deterministic ruleset orchestration |
| `vectorized` | **supported** | Vectorized Bayesian evidence updates |
| `serde` / `bincode` / `tracing` | **supported** | Serialization / observability |
| `jit` | **experimental** | Cranelift JIT backend (native-only) |
| `aot` | **experimental** | Ahead-of-time flow artifacts with hash validation (native-only) |
| `simd-kernels` | **experimental** | SIMD numeric kernel dispatch |
| `gpu-kernels` | **scaffold/baseline only** | Host staging path; not a production GPU runtime |
| `storage-experimental` / `storage-dense-index` / `storage-soa` | **scaffold/baseline only** | Storage/index experiments |

Examples:

```bash
cargo test -p grafial-core --features parallel
cargo test -p grafial-core --features jit
cargo test -p grafial-core --features simd-kernels,gpu-kernels
cargo clippy -p grafial-core --all-targets --all-features -- -D warnings
```

## Programmatic Usage

Rust:

```rust
use grafial_core::{parse_and_validate, run_flow};

fn main() -> Result<(), Box<dyn std::error::Error>> {
    let source = std::fs::read_to_string("crates/grafial-examples/minimal.grafial")?;
    let program = parse_and_validate(&source)?;
    let result = run_flow(&program, "MinimalFlow", None)?;
    println!("metric exports: {:?}", result.metric_exports);
    Ok(())
}
```

Python (from `crates/grafial-python`):

```python
import pathlib
import grafial

source = pathlib.Path("crates/grafial-examples/minimal.grafial").read_text()
program = grafial.compile(source)
ctx = grafial.run_flow(program, "MinimalFlow")
print(ctx.metrics)
```

## Development Workflow

```bash
# Format check
cargo fmt --all -- --check

# Lints
cargo clippy --workspace --all-targets --all-features -- -D warnings

# Tests
cargo test --workspace
```

Lockfile policy:
- Root `Cargo.lock` is tracked in git for reproducible CLI/workspace builds.

Benchmarks:

```bash
# Workspace benchmark crate
cargo bench -p grafial-benches

# Core kernel/path benchmarks
cargo bench -p grafial-core --bench vectorized_evidence --features vectorized
cargo bench -p grafial-core --bench parallel_execution --features parallel
cargo bench -p grafial-core --bench numeric_kernels --features simd-kernels,gpu-kernels
```

## Documentation

- [Documentation index](documentation/README.md)
- [Build/install details](documentation/BUILDING.md)
- [Language guide](documentation/LANGUAGE_GUIDE.md)
- [Probabilistic semantics (normative)](documentation/PROBABILISTIC_SEMANTICS.md)
- [Examples index](documentation/EXAMPLES.md)
- [Engine internals](documentation/ENGINE_ARCHITECTURE.md)

## Contributing

See [CONTRIBUTING.md](CONTRIBUTING.md) for contribution and coding guidelines.

## License

[MIT](LICENSE)
