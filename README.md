# Grafial: A language for reasoning when connections aren't clear

Grafial is a domain-specific language for reasoning about **uncertain relationships**.  
It treats graphs not as fixed structures, but as **probabilistic systems** where nodes, edges, and attributes each carry **degrees of belief** rather than binary truth.  

Instead of demanding statistical expertise or hand-written Bayesian math, Grafial lets you **describe uncertain systems declaratively**. You define schemas, evidence, and rules; the engine maintains consistent posterior beliefs automatically.  

---

## 1. Overview

Grafial was built to make it practical to work with **graphs that represent Bayesian uncertainty**: systems where connections, properties, and outcomes are *partially observed, probabilistic, and evolving*.  

Where traditional graph engines deal in facts ("A is connected to B"), Grafial deals in beliefs ("A is probably connected to B, and that affects what we think about C").  

Example use cases:

- Bayesian A/B testing and decision analysis  
- Probabilistic social or influence networks  
- Causal reasoning and belief propagation  
- Uncertain graph querying and inference

---

## 2. Project Structure

This is a **monorepo** containing all Grafial components:

```
baygraph/
├── crates/
│   ├── grafial-core/          # Core engine (graph, rules, flows, metrics)
│   ├── grafial-frontend/      # Parser, AST, validation (grammar.pest)
│   ├── grafial-ir/            # Intermediate representation
│   ├── grafial-cli/           # Command-line interface
│   ├── grafial-python/        # Python bindings (PyO3)
│   ├── grafial-tests/         # Integration and unit tests
│   ├── grafial-benches/       # Performance benchmarks
│   ├── grafial-examples/      # Example .grafial programs (.grafial files only)
│   └── grafial-vscode/        # VSCode extension
├── documentation/             # Project documentation
├── Cargo.toml                 # Workspace configuration
└── README.md                  # This file
```

---

## 3. Language

The **Grafial language** defines graph schemas, probabilistic models, evidence updates, and reasoning rules. Its syntax looks familiar to anyone who's used declarative or rule-based systems, but it's explicitly designed for uncertainty.  

Key concepts:
- `belief_model` defines priors and posterior distributions for nodes and edges.  
- `evidence` updates beliefs from observations.  
- `rule` expresses probabilistic transformations and reasoning steps.  
- `flow` sequences updates and computes derived metrics.  

Example:

```Grafial
schema ABTest {
  node Variant {
    conversion_rate: Real
  }
  edge OUTPERFORMS { }
}

belief_model TestBeliefs on ABTest {
  node Variant {
    conversion_rate ~ GaussianPosterior(prior_mean=0.1, prior_precision=10.0)
  }
  edge OUTPERFORMS {
    exist ~ BernoulliPosterior(prior=0.5, pseudo_count=2.0)
  }
}

evidence VariantBData on TestBeliefs {
  observe Variant["B"].conversion_rate = 0.15
}
```

See **`documentation/LANGUAGE_GUIDE.md`** for the full grammar and semantics, or explore `crates/grafial-examples/` for complete working examples.

---

## 4. Building and Testing

### Prerequisites

- Rust (stable toolchain)
- For Python bindings: Python 3.8+ and `maturin`
- For development: `nix-shell` (see `shell.nix`)

### Build

Build all crates in the workspace:

```bash
cargo build --workspace --release
```

Build a specific crate:

```bash
cargo build -p grafial-core --release
cargo build -p grafial-cli --release
```

### Test

Run all tests:

```bash
cargo test --workspace
```

Run tests for a specific crate:

```bash
cargo test -p grafial-core
cargo test -p grafial-frontend
cargo test -p grafial-tests
```

### Benchmarks

Run performance benchmarks:

```bash
cargo bench -p grafial-benches
```

---

## 5. Examples

The `grafial-examples` crate contains `.grafial` programs demonstrating various language features and use cases.

### Running Examples

Use the CLI to execute example `.grafial` files:

```bash
# Run with the CLI
cargo run -p grafial-cli -- crates/grafial-examples/minimal.grafial

# Run a specific flow
cargo run -p grafial-cli -- crates/grafial-examples/social.grafial --flow Demo

# Get JSON output
cargo run -p grafial-cli -- crates/grafial-examples/ab_testing.grafial --flow ABTest -o json
```

### Available Examples
- `minimal.grafial` - Simplest possible Grafial program
- `social.grafial` - Social network with belief propagation
- `ab_testing.grafial` - Bayesian A/B testing
- `competing_choices.grafial` - Categorical distributions and forced choice
- `transitive_closure.grafial` - Fixpoint iteration rules
- And more in `crates/grafial-examples/`

---

## 6. CLI Tool

The Grafial CLI allows you to validate and execute Grafial programs from the command line:

```bash
# Build the CLI
cargo build -p grafial-cli --release

# Validate a program
cargo run --bin grafial -- crates/grafial-examples/minimal.grafial

# List available flows
cargo run --bin grafial -- crates/grafial-examples/social.grafial --list-flows

# Execute a flow
cargo run --bin grafial -- crates/grafial-examples/social.grafial --flow Demo

# Get JSON output
cargo run --bin grafial -- crates/grafial-examples/social.grafial --flow Demo -o json

# Get debug output
cargo run --bin grafial -- crates/grafial-examples/social.grafial --flow Demo -o debug
```

After building, the binary is available at `target/release/grafial`:

```bash
./target/release/grafial crates/grafial-examples/minimal.grafial
```

See **`documentation/BUILDING.md`** for detailed installation and build instructions.

---

## 7. Python Bindings

The **Python interface** exposes Grafial's engine using PyO3 bindings. It allows you to load, run, and analyze probabilistic graphs interactively: perfect for notebooks, pipelines, or decision systems.

### Install

```bash
cd crates/grafial-python
maturin develop --release
```

Or using `uv` (recommended):

```bash
cd crates/grafial-python
uv venv
source .venv/bin/activate  # On Windows: .venv\Scripts\activate
uv pip install -e .
```

### Example

```python
from grafial import GraphRuntime

runtime = GraphRuntime()
runtime.load("crates/grafial-examples/ab_testing.grafial")
runtime.run()
```

See **`documentation/PYTHON_PLAN.md`** for integration details and usage examples.

---

## 8. VSCode Extension

The VSCode extension provides syntax highlighting for Grafial files.

### Install

```bash
cd crates/grafial-vscode
npm install
npm run package
code --install-extension grafial-0.1.0.vsix
```

See **`crates/grafial-vscode/README.md`** for details.

---

## 9. Documentation

- **`documentation/LANGUAGE_GUIDE.md`** - Complete language reference
- **`documentation/ENGINE_ARCHITECTURE.md`** - Runtime architecture and API
- **`documentation/BUILDING.md`** - Build and development setup
- **`documentation/PYTHON_PLAN.md`** - Python integration guide
- **`documentation/ROADMAP.md`** - Project roadmap

---

## 10. Development

### Workspace Structure

This monorepo uses Cargo workspaces to manage multiple related crates:

- **Core crates** (`grafial-core`, `grafial-frontend`, `grafial-ir`) - Library code
- **Application crates** (`grafial-cli`) - Executables
- **Bindings** (`grafial-python`) - Language bindings
- **Examples** (`grafial-examples`) - Example `.grafial` programs demonstrating features
- **Tooling** (`grafial-tests`, `grafial-benches`) - Development tools
- **Extensions** (`grafial-vscode`) - Editor support

### Adding a New Crate

1. Create the crate directory in `crates/`
2. Add it to `Cargo.toml` workspace members
3. Configure dependencies in the crate's `Cargo.toml`

### Code Style

- Format with `cargo fmt --all`
- Lint with `cargo clippy --all-targets --all-features -- -D warnings`
- Follow Rust edition 2021 conventions

See **`AGENTS.md`** for detailed development guidelines.

---

## 11. License

MIT OR Apache-2.0
