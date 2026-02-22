# Grafial

Grafial is a domain-specific language and runtime for Bayesian reasoning over graphs.
It is designed for systems where nodes, edges, and attributes are uncertain and should
be updated as new evidence arrives.

This repository is a monorepo containing the language frontend, IR, execution engine,
CLI, Python bindings, tests, and benchmarks.

## What Grafial Gives You

- A declarative DSL for probabilistic graph programs.
- Bayesian posteriors on graph structure and attributes.
- Rule and flow execution over uncertain graphs.
- Deterministic execution with optional performance features.
- Tooling for CLI, Python integration, testing, and benchmarking.

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

## Quick Start

Prerequisites:

- Rust stable toolchain
- Cargo
- Optional: `nix-shell` (from `shell.nix`) for a pinned dev environment

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

`crates/grafial-examples/minimal.grafial`:

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

Key crates in `crates/`:

- `grafial-frontend`: parser, AST, validation, style linting/formatting.
- `grafial-ir`: lowered IR and optimization passes.
- `grafial-core`: execution engine, graph model, rule/flow runtime, kernels.
- `grafial-cli`: `grafial` command-line tool.
- `grafial-python`: PyO3 bindings.
- `grafial-lsp`: language server implementation.
- `grafial-tests`: integration test crate.
- `grafial-benches`: benchmark crate.
- `grafial-examples`: sample `.grafial` programs.

## Engine Feature Flags (grafial-core)

`crates/grafial-core/Cargo.toml` defines optional features:

- `parallel`: parallel evidence and metric execution paths with deterministic ruleset orchestration hooks.
- `jit`: Cranelift-based JIT backend.
- `aot`: ahead-of-time flow artifact compilation with runtime hash validation and compiled entrypoint execution checks.
- `vectorized`: vectorized Bayesian evidence updates.
- `simd-kernels`: SIMD numeric kernel dispatch (feature-gated).
- `gpu-kernels`: GPU-staged kernel dispatch path (feature-gated host staging baseline).
- `serde`, `bincode`, `tracing`: serialization/observability support.
- `storage-experimental`, `storage-dense-index`, `storage-soa`: storage/index experimentation flags.

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

- Documentation index: `documentation/README.md`
- Build/install details: `documentation/BUILDING.md`
- Language guide: `documentation/LANGUAGE_GUIDE.md`
- Engine internals: `documentation/ENGINE_ARCHITECTURE.md`

## Contributing

See `CONTRIBUTING.md` (if present) for contribution and coding guidelines.

## License

MIT
