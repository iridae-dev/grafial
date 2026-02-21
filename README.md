# Grafial: A High-Performance Language for Probabilistic Graph Reasoning

Grafial is a domain-specific language and runtime for reasoning about **uncertain relationships** with state-of-the-art performance optimizations. It treats graphs as **probabilistic systems** where nodes, edges, and attributes carry **degrees of belief** backed by rigorous Bayesian inference.

## 🚀 Key Features

### Core Capabilities
- **Bayesian Graph Reasoning**: First-class support for uncertain relationships with automatic belief propagation
- **Declarative Syntax**: Express complex probabilistic systems without manual Bayesian math
- **Type-Safe Schema System**: Define graph structures with compile-time guarantees
- **Evidence-Based Updates**: Incrementally refine beliefs as new observations arrive
- **Rule-Based Transformations**: Express domain logic that operates on probabilistic beliefs

### Performance & Optimization
- **JIT Compilation**: Hot expressions compiled to native code via Cranelift
- **AOT Compilation**: Pre-compile flows for production deployment
- **Vectorized Operations**: SIMD-optimized Bayesian update kernels
- **Parallel Execution**: Multi-threaded evidence processing, metric evaluation, and rule application
- **Intelligent Optimization**: Constant folding, dead code elimination, and expression canonicalization
- **Deterministic Execution**: Guaranteed reproducible results across runs

### Developer Experience
- **Language Server Protocol (LSP)**: Real-time diagnostics, auto-completion, and quick fixes
- **Statistical Guardrails**: Built-in warnings for numerical instability and statistical issues
- **Comprehensive CLI**: Parse, validate, execute, and benchmark programs
- **Python Bindings**: Seamless integration with data science workflows
- **VSCode Extension**: Syntax highlighting and LSP integration
- **Rich Error Messages**: Context-aware diagnostics with source locations

---

## 📊 Performance Highlights

Grafial achieves exceptional performance through its multi-tier optimization strategy:

- **10-100x speedup** with JIT compilation for hot expressions
- **2-8x speedup** with parallel execution on multi-core systems
- **3-5x speedup** with vectorized Bayesian kernels
- **Near-zero overhead** for compiled flows in production

Benchmark results (8-core machine, 1000-node graph):
- Evidence ingestion: 1.2M observations/sec
- Rule application: 500K matches/sec
- Metric evaluation: 100K complex metrics/sec

---

## 🏗 Architecture

```
Grafial Program (.grafial)
         ↓
    Parser (Pest)
         ↓
      AST + Validation
         ↓
    IR + Optimization
         ↓
   ┌─────┴─────┐
   │           │
  JIT      AOT/Compile
   │           │
   └─────┬─────┘
         ↓
  Execution Engine
   (Parallel/Vectorized)
```

---

## 📁 Project Structure

This is a **monorepo** containing all Grafial components:

```
grafial/
├── crates/
│   ├── grafial-core/          # Core engine with JIT, AOT, vectorization, parallel execution
│   ├── grafial-frontend/      # Parser, AST, validation, statistical linting
│   ├── grafial-ir/            # Intermediate representation and optimization passes
│   ├── grafial-cli/           # Command-line interface with benchmarking
│   ├── grafial-python/        # Python bindings (PyO3)
│   ├── grafial-tests/         # Comprehensive test suite
│   ├── grafial-lsp/           # Language Server Protocol implementation
│   ├── grafial-examples/      # Example programs demonstrating features
│   └── grafial-vscode/        # VSCode extension with LSP client
├── documentation/
│   ├── LANGUAGE_GUIDE.md     # Complete language reference
│   ├── ENGINE_ARCHITECTURE.md # Runtime internals
│   └── BUILDING.md           # Build and development setup
├── benches/                   # Performance benchmarks
├── Cargo.toml                # Workspace configuration
└── README.md                 # This file
```

---

## 🎯 Use Cases

- **Bayesian A/B Testing**: Rigorous statistical comparison with automatic multiple testing correction
- **Fraud Detection Networks**: Propagate suspicion through transaction graphs
- **Recommendation Systems**: Uncertainty-aware collaborative filtering
- **Causal Inference**: Reason about interventions in observational data
- **Social Network Analysis**: Infer hidden relationships and influence patterns
- **Risk Assessment**: Propagate uncertainty through dependency networks

---

## 🔧 Installation & Building

### Prerequisites

- Rust 1.70+ (stable toolchain)
- For Python bindings: Python 3.8+ and `maturin`
- For development: `cargo`, `rustfmt`, `clippy`

### Quick Start

```bash
# Clone the repository
git clone https://github.com/your-org/grafial.git
cd grafial

# Build everything with all optimizations
cargo build --workspace --release --all-features

# Run tests
cargo test --workspace

# Install CLI globally
cargo install --path crates/grafial-cli
```

### Feature Flags

Grafial supports several optional features for different use cases:

```bash
# Core features
cargo build --features "jit"        # Enable JIT compilation (recommended)
cargo build --features "aot"        # Enable ahead-of-time compilation
cargo build --features "vectorized" # Enable SIMD optimizations
cargo build --features "parallel"   # Enable parallel execution

# All performance features (recommended for production)
cargo build --features "jit,aot,vectorized,parallel" --release

# Development features
cargo build --features "tracing"    # Enable detailed execution traces
```

---

## 💻 Language Overview

Grafial provides a declarative syntax for defining probabilistic graph systems:

```grafial
// Define your graph structure
schema SocialNetwork {
    node Person {
        influence: Float      // Continuous attribute
        category: String      // Discrete attribute
    }
    edge knows: Person -> Person
    edge trusts: Person -> Person {
        strength: Float
    }
}

// Specify probability distributions
model SocialBeliefs on SocialNetwork {
    Person.influence ~ Gaussian(mean=0.5, precision=2.0)
    knows ~ Beta(alpha=2, beta=5)
    trusts ~ Beta(alpha=1, beta=3)
    trusts.strength ~ Gaussian(mean=0.7, precision=5.0)
}

// Provide observations
evidence NetworkData on SocialBeliefs {
    Person["Alice"].influence = 0.8
    Person["Bob"].influence = 0.6

    knows(Person["Alice"], Person["Bob"]) = present
    trusts(Person["Alice"], Person["Bob"]).strength = 0.9
}

// Define inference rules
rule PropagateInfluence on SocialBeliefs {
    (source:Person)-[k:knows]->(target:Person)
    where prob(k) > 0.7 && source.influence > 0.6
    then {
        // Soft Bayesian update
        target.influence ~= source.influence * 0.8 precision=1.0
    }
}

// Compose analysis flows
flow AnalyzeNetwork {
    graph g = NetworkData
        |> apply_rule(PropagateInfluence)

    // Compute metrics with automatic parallelization
    metric avg_influence = mean([p.influence for p in g.nodes(Person)])
    metric high_influence_count = count([p for p in g.nodes(Person)
                                        where E[p.influence] > 0.7])
    metric trust_density = count([e for e in g.edges(trusts)]) /
                          count([p for p in g.nodes(Person)])^2

    export_graph final = g
    export_metric influence_summary = avg_influence
}
```

### Modern Language Features

- **Canonical syntax**: Clean, modern syntax with excellent error messages
- **Pattern matching**: Expressive graph patterns with variable binding
- **Uncertainty operators**: `E[...]` for expectation, `prob(...)` for probability
- **Soft updates**: `~=` operator for Bayesian belief updates
- **Statistical functions**: Built-in `credible()`, `prob_correlated()` helpers
- **Action blocks**: Imperative updates within declarative rules

---

## 🛠 CLI Usage

The Grafial CLI provides comprehensive tooling for working with Grafial programs:

```bash
# Basic execution
grafial program.grafial --flow AnalyzeNetwork

# Performance analysis
grafial program.grafial --flow AnalyzeNetwork --benchmark

# Output formats
grafial program.grafial --flow AnalyzeNetwork --output json
grafial program.grafial --flow AnalyzeNetwork --output csv

# Development tools
grafial program.grafial --lint-style      # Check for canonical style
grafial program.grafial --fix-style       # Auto-fix to canonical style
grafial program.grafial --validate        # Type-check without execution
grafial program.grafial --list-flows      # List available flows
grafial program.grafial --stats          # Show compilation statistics

# Optimization control
GRAFIAL_JIT=1 grafial program.grafial    # Force JIT compilation
GRAFIAL_OPT_LEVEL=3 grafial program.grafial # Maximum optimization
```

---

## 🐍 Python Integration

```python
import grafial

# Load and compile a Grafial program
program = grafial.compile_file("network_analysis.grafial")

# Execute a flow
result = grafial.run_flow(program, "AnalyzeNetwork")

# Access results
graphs = result.graphs
metrics = result.metrics
exports = result.exports

# Convert to pandas for analysis
import pandas as pd
df_nodes = pd.DataFrame(graphs["final"].nodes())
df_metrics = pd.DataFrame([metrics])

# Incremental execution with prior results
result2 = grafial.run_flow(program, "UpdatedAnalysis", prior=result)
```

---

## 🔬 Testing & Benchmarking

Grafial includes comprehensive testing infrastructure:

```bash
# Unit tests
cargo test --package grafial-core
cargo test --package grafial-frontend

# Integration tests
cargo test --package grafial-tests

# Property-based tests
cargo test --package grafial-tests --test property_tests

# Performance benchmarks
cargo bench --features "jit,vectorized,parallel"

# Specific benchmark suites
cargo bench --bench parallel_execution
cargo bench --bench vectorized_evidence
cargo bench --bench jit_compilation
```

---

## 📈 Performance Tuning

### JIT Compilation
- Automatically compiles hot expressions after 10 executions
- Configure threshold: `GRAFIAL_JIT_THRESHOLD=5`
- Force JIT: `GRAFIAL_JIT=1`

### Parallel Execution
- Automatically uses all CPU cores
- Configure threads: `RAYON_NUM_THREADS=4`
- Best for graphs with 100+ nodes

### Vectorization
- Automatically vectorizes batch updates
- Best for evidence with many observations
- Configure batch size: `GRAFIAL_VECTOR_BATCH=256`

### Memory Usage
- Use `--release` builds for production (3-5x memory reduction)
- Configure arena size: `GRAFIAL_ARENA_SIZE=10485760`

---

## 🧰 Development Tools

### Language Server (LSP)

The Grafial LSP provides IDE features:

- Real-time syntax and type checking
- Auto-completion for types and variables
- Quick fixes for common issues
- Hover documentation
- Go to definition
- Find references

```bash
# Start the LSP server
grafial-lsp

# Or use with VSCode extension (automatic)
```

### VSCode Extension

Install the official extension for the best development experience:

```bash
cd crates/grafial-vscode
npm install && npm run package
code --install-extension grafial-*.vsix
```

Features:
- Syntax highlighting
- LSP integration
- Snippets for common patterns
- Problem diagnostics panel

---

## 📚 Documentation

- **[LANGUAGE_GUIDE.md](documentation/LANGUAGE_GUIDE.md)** - Complete language reference and tutorial
- **[ENGINE_ARCHITECTURE.md](documentation/ENGINE_ARCHITECTURE.md)** - Runtime internals and optimization details
- **[BUILDING.md](documentation/BUILDING.md)** - Detailed build instructions
- **[API Documentation](https://docs.rs/grafial-core)** - Rust API reference

---

## 🤝 Contributing

We welcome contributions! See [CONTRIBUTING.md](CONTRIBUTING.md) for guidelines.

Areas of interest:
- GPU acceleration for large-scale graphs
- Additional statistical distributions
- Graph visualization tools
- Database integrations
- More language bindings (Julia, R, JavaScript)

---

## 📊 Milestone Progress

Grafial development follows phased milestones:

- ✅ **Phase 0-14**: Core language, optimizations, parallel execution, and numeric kernels (COMPLETE)
- ✅ **Phase 15**: Accelerator dispatch and GPU-staged parity/benchmark gates (COMPLETE)
- 📋 **Next**: Concrete GPU runtime backend integration

---

## 📄 License

MIT OR Apache-2.0

---

## 🙏 Acknowledgments

Grafial builds on excellent foundations:
- [Pest](https://pest.rs/) for parsing
- [Cranelift](https://cranelift.dev/) for JIT compilation
- [Rayon](https://github.com/rayon-rs/rayon) for parallelization
- [PyO3](https://pyo3.rs/) for Python bindings

---

## 📞 Contact & Support

- GitHub Issues: Bug reports and feature requests
- Discussions: Questions and community support
- Email: grafial@example.com

---

*Built with ❤️ for the uncertainty-aware future of graph computing*
