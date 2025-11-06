# Python Bindings Implementation Plan

**Goal:** Make Grafial usable without writing Rust, providing a complete Python API for Bayesian belief graph inference.

**Status:** Planning phase - bindings crate exists at `crates/grafial-python/` with placeholder implementation.

---

## Overview

The Python bindings will expose Grafial's core functionality through PyO3, allowing Python users to:
- Compile Grafial programs from source
- Build and apply evidence dynamically
- Execute flows and inspect results
- Access graphs, metrics, and posterior beliefs
- Export to pandas/NetworkX for analysis

---

## Core API Design

### Module-Level Functions

```python
import grafial

# Compile a Grafial program from source
program = grafial.compile(source: str) -> Program

# Run a flow (uses static evidence from program)
ctx = grafial.run_flow(program: Program, flow_name: str) -> Context

# Run a flow with runtime evidence
ctx = grafial.run_flow_with_evidence(
    program: Program,
    flow_name: str,
    evidence: Evidence,
    ctx: Optional[Context] = None
) -> Context

# Chain flows (pass context with graphs/metrics between flows)
ctx = grafial.run_flow_with_context(
    program: Program,
    flow_name: str,
    ctx: Context
) -> Context
```

### Python Classes

#### `Program`
Represents a compiled Grafial program (schema, belief model, evidence, rules, flows).

**Methods:**
- `get_flow_names() -> List[str]` - List all flow names in the program
- `get_schema_names() -> List[str]` - List all schema names
- `get_belief_model_names() -> List[str]` - List all belief model names

#### `Evidence`
Builder for runtime evidence (observations not in the `.grafial` file).

**Constructor:**
```python
evidence = grafial.Evidence(
    name: str,
    model: str  # Belief model name
)
```

**Methods:**
- `observe_edge(node_type: str, src_id: str, edge_type: str, dst_type: str, dst_id: str, present: bool) -> None`
- `observe_edge_chosen(node_type: str, src_id: str, edge_type: str, dst_type: str, dst_id: str) -> None`  # For competing edges
- `observe_edge_unchosen(node_type: str, src_id: str, edge_type: str, dst_type: str, dst_id: str) -> None`  # For competing edges
- `observe_edge_forced_choice(node_type: str, src_id: str, edge_type: str, dst_type: str, dst_id: str) -> None`  # For competing edges
- `observe_numeric(node_type: str, node_id: str, attr: str, value: float) -> None`
- `clear() -> None` - Clear all observations

#### `Context`
Result of running a flow, containing graphs and metrics.

**Properties:**
- `graphs: Dict[str, BeliefGraph]` - Named graphs exported from flow
- `metrics: Dict[str, float]` - Metrics exported from flow

**Methods:**
- `get_graph(name: str) -> Optional[BeliefGraph]`
- `get_metric(name: str) -> Optional[float]`

#### `BeliefGraph`
Python wrapper for `BeliefGraph` - represents a probabilistic graph state.

**Methods:**
- `nodes(label: Optional[str] = None) -> Iterator[NodeView]` - Iterate over nodes (optionally filtered by label)
- `edges(edge_type: Optional[str] = None) -> Iterator[EdgeView]` - Iterate over edges (optionally filtered by type)
- `competing_groups(edge_type: Optional[str] = None) -> Iterator[CompetingGroup]` - Iterate over competing edge groups
- `to_pandas() -> Tuple[pd.DataFrame, pd.DataFrame]` - Export to pandas (returns (nodes_df, edges_df))
- `to_networkx(threshold: float = 0.0) -> nx.Graph` - Export to NetworkX (filter edges by probability threshold)

#### `NodeView`
Read-only view of a node in the graph.

**Properties:**
- `id: str` - Node identifier
- `label: str` - Node type label

**Methods:**
- `E(attr: str) -> float` - Get expected value (mean) of attribute
- `Var(attr: str) -> float` - Get variance of attribute (1/precision for Gaussian)
- `has_attr(attr: str) -> bool` - Check if attribute exists

#### `EdgeView`
Read-only view of an edge in the graph.

**Properties:**
- `src: str` - Source node ID
- `dst: str` - Destination node ID
- `type: str` - Edge type
- `prob: float` - Mean probability of existence (E[p] for independent, E[π_k] for competing)
- `forced_state: Optional[str]` - "present", "absent", or None if not forced

**Methods:**
- `is_competing() -> bool` - Whether this edge is part of a competing group
- `is_independent() -> bool` - Whether this edge has independent posterior

#### `CompetingGroup`
Represents a group of competing edges (CategoricalPosterior).

**Properties:**
- `source_node: str` - Source node ID
- `edge_type: str` - Edge type
- `categories: List[str]` - Destination node IDs (categories)
- `probabilities: List[float]` - Mean probabilities E[π_k] for each category
- `entropy: float` - Shannon entropy of the distribution

**Methods:**
- `winner(epsilon: float = 0.01) -> Optional[str]` - Destination with highest probability, or None if tied
- `prob_vector() -> List[float]` - Full probability vector

---

## Implementation Details

### PyO3 Integration

**Module Structure:**
```
crates/grafial-python/
├── src/
│   └── lib.rs              # Main PyO3 module
├── Cargo.toml              # Rust dependencies
└── pyproject.toml          # Python package metadata
```

The implementation will be in `crates/grafial-python/src/lib.rs` with helper modules as needed.

### GIL Management

**Critical:** Release the GIL for long-running operations to allow Python threads to run concurrently.

**Strategy:**
```rust
use pyo3::prelude::*;
use pyo3::types::PyDict;

// Long-running operations (compile, run_flow, etc.)
#[pyfunction]
fn run_flow(py: Python, program: &PyProgram, flow_name: &str) -> PyResult<PyContext> {
    py.allow_threads(|| {
        // Heavy computation here - GIL released
        let ctx = program.inner.run_flow(flow_name)?;
        Ok(PyContext::new(ctx))
    })
}

// Short getters (keep GIL)
#[pymethods]
impl BeliefGraph {
    fn node_count(&self) -> usize {
        self.inner.nodes().len()  // Fast, keep GIL
    }
}
```

**Guidelines:**
- Release GIL for: `compile()`, `run_flow()`, `run_flow_with_evidence()`, `run_flow_with_context()`
- Keep GIL for: simple property access, small lookups, iteration setup

### Error Handling

**Strategy:** Convert Rust `ExecError` to rich Python exceptions.

**Error Conversion:**
```rust
use pyo3::exceptions::{PyValueError, PyRuntimeError, PyTypeError};

impl From<grafial_core::ExecError> for PyErr {
    fn from(err: grafial_core::ExecError) -> Self {
        match err {
            ExecError::ParseError(msg) => PyValueError::new_err(format!("Parse error: {}", msg)),
            ExecError::ValidationError(msg) => PyValueError::new_err(format!("Validation error: {}", msg)),
            ExecError::Execution(msg) => PyRuntimeError::new_err(format!("Execution error: {}", msg)),
            ExecError::Numerical(msg) => PyRuntimeError::new_err(format!("Numerical error: {}", msg)),
            ExecError::Internal(msg) => PyRuntimeError::new_err(format!("Internal error: {}", msg)),
        }
    }
}
```

**Do NOT:**
- Expose `anyhow`/`thiserror` types directly to Python
- Use generic `PyException` - use specific exception types
- Panic on user errors - always return `PyResult`

### Immutability Semantics

**Critical:** Python API must preserve Rust's immutable graph semantics.

**Strategy:**
- Methods that "mutate" graphs return new graph handles
- Old graphs remain valid via `Arc` sharing
- No in-place mutation methods exposed

```rust
#[pymethods]
impl BeliefGraph {
    // This would be a "mutation" - returns new graph
    fn apply_rule(&self, py: Python, rule_name: &str) -> PyResult<BeliefGraph> {
        py.allow_threads(|| {
            let new_graph = self.inner.apply_rule(rule_name)?;
            Ok(BeliefGraph::new(new_graph))  // New handle, old still valid
        })
    }
}
```

### Zero-Copy Considerations

**Strategy:** Prefer correctness over zero-copy for complex structures.

**Where zero-copy is safe:**
- Simple numeric arrays (if using numpy): `PyReadonlyArray1<f64>`
- String slices (if stable): `&str` → `PyString`

**Where to avoid zero-copy:**
- Complex nested structures (graphs, nodes, edges)
- Data that might be mutated on Rust side
- Structures that need lifetime management

**Recommendation:** Start with owned copies, optimize later if profiling shows bottlenecks.

### Data Structure Wrappers

**BeliefGraph Wrapper:**
```rust
#[pyclass]
pub struct BeliefGraph {
    inner: Arc<grafial_core::engine::graph::BeliefGraph>,
}

#[pymethods]
impl BeliefGraph {
    fn nodes(&self, label: Option<&str>) -> PyResult<Vec<NodeView>> {
        // Convert to Python-friendly format
    }
    
    fn edges(&self, edge_type: Option<&str>) -> PyResult<Vec<EdgeView>> {
        // Convert to Python-friendly format
    }
}
```

**NodeView/EdgeView:**
- Thin wrappers around Rust structs
- Compute properties on-demand (E[], Var[], prob)
- Cache expensive computations if needed

### Pandas/NetworkX Integration

**Pandas Export:**
```python
def to_pandas(self) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """
    Export graph to pandas DataFrames.
    
    Returns:
        (nodes_df, edges_df) where:
        - nodes_df: columns [id, label, attr1_mean, attr1_var, attr2_mean, ...]
        - edges_df: columns [src, dst, type, prob, forced_state]
    """
```

**NetworkX Export:**
```python
def to_networkx(self, threshold: float = 0.0) -> nx.Graph:
    """
    Export graph to NetworkX Graph.
    
    Args:
        threshold: Only include edges with prob >= threshold
    
    Returns:
        NetworkX Graph with node/edge attributes
    """
```

**Dependencies:**
- `pandas` and `networkx` are optional dependencies
- Use feature flags: `[features] default = [] python = ["pyo3"]`
- Check if available at runtime before exposing methods

---

## Build and Distribution

### Maturin Setup

**Cargo.toml** (already configured in `crates/grafial-python/Cargo.toml`):
```toml
[lib]
name = "grafial"
crate-type = ["cdylib", "rlib"]

[dependencies]
grafial-core = { path = "../grafial-core" }
pyo3 = { version = "0.21", features = ["auto-initialize"] }
```

**pyproject.toml** (already exists in `crates/grafial-python/pyproject.toml`):
```toml
[build-system]
requires = ["maturin>=1.0,<2.0"]
build-backend = "maturin"

[project]
name = "grafial"
requires-python = ">=3.8"
```

### Development Build

**Using maturin:**
```bash
cd crates/grafial-python
maturin develop --release
```

**Using uv (recommended):**
```bash
cd crates/grafial-python
uv venv
source .venv/bin/activate  # On Windows: .venv\Scripts\activate
uv pip install -e .
```

**In nix-shell:**
```bash
cd crates/grafial-python
# uv is available if installed separately
uv venv && uv pip install -e .
# Or use maturin if available
maturin develop --release
```

### Testing

**Python Tests:**
Create `crates/grafial-python/tests/` directory:
```python
# crates/grafial-python/tests/test_basic.py
import grafial

def test_compile():
    source = """
    schema Test { node Person { } edge REL { } }
    belief_model TestBeliefs on Test {
        edge REL { exist ~ BernoulliPosterior(prior=0.5, pseudo_count=2.0) }
    }
    """
    program = grafial.compile(source)
    assert program is not None

def test_run_flow():
    # ... test flow execution
```

**Run tests:**
```bash
# From workspace root
cargo test -p grafial-python

# From Python side (after installation)
cd crates/grafial-python
pytest tests/
```

---

## Example Usage

### Basic Usage

```python
import grafial

# Compile program
with open("crates/grafial-examples/social.grafial", "r") as f:
    source = f.read()
program = grafial.compile(source)

# Run flow with static evidence
ctx = grafial.run_flow(program, "Demo")
print(f"Result metric: {ctx.metrics['my_metric']}")

# Get exported graph
graph = ctx.graphs["result"]
for node in graph.nodes():
    print(f"Node {node.id}: score = {node.E('score')}")
```

### Dynamic Evidence

```python
# Build runtime evidence
evidence = grafial.Evidence("RuntimeEvidence", model="SocialBeliefs")
evidence.observe_edge("Person", "Alice", "KNOWS", "Person", "Bob", present=True)
evidence.observe_numeric("Person", "Alice", "score", 10.0)

# Run flow with evidence
ctx = grafial.run_flow_with_evidence(program, "ComputeBudget", evidence)
```

### Competing Edges

```python
# Observe competing edge choices
evidence = grafial.Evidence("RoutingEvidence", model="NetworkBeliefs")
evidence.observe_edge_chosen("Server", "S1", "ROUTES_TO", "Server", "S2")
evidence.observe_edge_chosen("Server", "S1", "ROUTES_TO", "S2")  # Again
evidence.observe_edge_chosen("Server", "S1", "ROUTES_TO", "S3")  # Different choice

# Inspect competing groups
graph = ctx.graphs["result"]
for group in graph.competing_groups("ROUTES_TO"):
    if group.entropy < 0.5:
        winner = group.winner()
        print(f"Low diversity at {group.source_node}: winner is {winner}")
```

### Pandas/NetworkX Export

```python
# Export to pandas
nodes_df, edges_df = graph.to_pandas()
print(nodes_df.head())
print(edges_df[edges_df['prob'] > 0.5])  # Filter high-probability edges

# Export to NetworkX
import networkx as nx
G = graph.to_networkx(threshold=0.3)  # Only edges with prob >= 0.3
nx.draw(G, with_labels=True)
```

---

## Implementation Phases

### Phase 1: Core Infrastructure
- [ ] Set up PyO3 module structure in `crates/grafial-python/src/lib.rs`
- [ ] Implement `Program` wrapper
- [ ] Implement `compile()` function
- [ ] Basic error conversion (`ExecError` → Python exceptions)
- [ ] GIL management for compile

### Phase 2: Flow Execution
- [ ] Implement `Context` wrapper
- [ ] Implement `run_flow()` function
- [ ] Implement `run_flow_with_evidence()` function
- [ ] Implement `run_flow_with_context()` function
- [ ] GIL management for flow execution

### Phase 3: Evidence Builder
- [ ] Implement `Evidence` class
- [ ] Implement `observe_edge()` methods
- [ ] Implement `observe_numeric()` method
- [ ] Implement competing edge evidence methods
- [ ] Validation and error handling

### Phase 4: Graph Inspection
- [ ] Implement `BeliefGraph` wrapper
- [ ] Implement `NodeView` and `EdgeView`
- [ ] Implement `nodes()` and `edges()` iterators
- [ ] Implement `E()` and `Var()` methods
- [ ] Implement `competing_groups()` iterator
- [ ] Implement `CompetingGroup` wrapper

### Phase 5: Convenience Exports
- [ ] Implement `to_pandas()` (optional pandas dependency)
- [ ] Implement `to_networkx()` (optional networkx dependency)
- [ ] Handle missing dependencies gracefully

### Phase 6: Testing and Documentation
- [ ] Python unit tests for all APIs
- [ ] Integration tests with example programs
- [ ] API documentation (docstrings)
- [ ] Usage examples in docs
- [ ] Performance benchmarks

---

## Exit Criteria

**Complete when:**

1. **From Python, you can:**
   - `compile()` a `.grafial` file source string
   - Build an `Evidence` object with runtime observations
   - `run_flow()` with static or dynamic evidence
   - `run_flow_with_context()` to chain flows
   - Read metrics from `Context.metrics`
   - Inspect graphs via `nodes()` and `edges()` iterators
   - Access posterior beliefs via `E()` and `Var()`
   - Inspect competing edge groups
   - Export to pandas/NetworkX

2. **Performance:**
   - GIL released for long-running operations
   - No unnecessary Python ↔ Rust round trips
   - Memory usage reasonable (no leaks)

3. **Error Handling:**
   - All Rust errors converted to appropriate Python exceptions
   - Clear error messages with context
   - No panics reach Python code

4. **Testing:**
   - Python unit tests pass
   - Integration tests with real `.grafial` files
   - Edge cases handled (empty graphs, missing nodes, etc.)

---

## References

- **PyO3 Documentation:** https://pyo3.rs/
- **Maturin Documentation:** https://maturin.rs/
- **uv Documentation:** https://github.com/astral-sh/uv
- **Rust Engine:** `crates/grafial-core/src/engine/` - Core types to wrap
- **Flow Execution:** `crates/grafial-core/src/engine/flow_exec.rs` - `run_flow` implementation
- **Errors:** `crates/grafial-core/src/engine/errors.rs` - `ExecError` enum to convert
- **Frontend:** `crates/grafial-frontend/` - Parser and AST types

---

## Notes

- **Naming:** Python API uses `snake_case` (Python convention), Rust uses `snake_case` for functions and `PascalCase` for types
- **Type Hints:** Add Python type hints for better IDE support
- **Docstrings:** All public methods should have Google-style docstrings
- **Versioning:** Python package version should match Rust crate version
- **Distribution:** Eventually publish to PyPI as `grafial` (check availability)
- **Module Name:** Python module is `grafial` (lowercase), not `Grafial`
