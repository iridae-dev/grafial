# Python Bindings Implementation Plan

**Goal:** Make Baygraph usable without writing Rust, providing a complete Python API for Bayesian belief graph inference.

**Status:** Planning phase - bindings module exists as placeholder (`src/bindings/mod.rs`)

---

## Overview

The Python bindings will expose Baygraph's core functionality through PyO3, allowing Python users to:
- Compile Baygraph programs from source
- Build and apply evidence dynamically
- Execute flows and inspect results
- Access graphs, metrics, and posterior beliefs
- Export to pandas/NetworkX for analysis

---

## Core API Design

### Module-Level Functions

```python
import baygraph

# Compile a Baygraph program from source
program = baygraph.compile(source: str) -> Program

# Run a flow (uses static evidence from program)
ctx = baygraph.run_flow(program: Program, flow_name: str) -> Context

# Run a flow with runtime evidence
ctx = baygraph.run_flow_with_evidence(
    program: Program,
    flow_name: str,
    evidence: Evidence,
    ctx: Optional[Context] = None
) -> Context

# Chain flows (pass context with graphs/metrics between flows)
ctx = baygraph.run_flow_with_context(
    program: Program,
    flow_name: str,
    ctx: Context
) -> Context
```

### Python Classes

#### `Program`
Represents a compiled Baygraph program (schema, belief model, evidence, rules, flows).

**Methods:**
- `get_flow_names() -> List[str]` - List all flow names in the program
- `get_schema_names() -> List[str]` - List all schema names
- `get_belief_model_names() -> List[str]` - List all belief model names

#### `Evidence`
Builder for runtime evidence (observations not in the `.bg` file).

**Constructor:**
```python
evidence = baygraph.Evidence(
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
src/bindings/
├── mod.rs              # Main module
├── python/
│   ├── mod.rs          # Python module initialization
│   ├── program.rs      # Program wrapper
│   ├── evidence.rs     # Evidence builder
│   ├── context.rs      # Context wrapper
│   ├── graph.rs        # BeliefGraph wrapper
│   ├── node_view.rs    # NodeView wrapper
│   ├── edge_view.rs    # EdgeView wrapper
│   └── competing_group.rs  # CompetingGroup wrapper
└── errors.rs           # Error conversion utilities
```

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

impl From<ExecError> for PyErr {
    fn from(err: ExecError) -> Self {
        match err {
            ExecError::ParseError(msg) => PyValueError::new_err(format!("Parse error: {}", msg)),
            ExecError::ValidationError(msg) => PyValueError::new_err(format!("Validation error: {}", msg)),
            ExecError::TypeError(msg) => PyTypeError::new_err(format!("Type error: {}", msg)),
            ExecError::Internal(msg) => PyRuntimeError::new_err(format!("Internal error: {}", msg)),
            // ... map all variants
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
    inner: Arc<engine::graph::BeliefGraph>,
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

**Cargo.toml additions:**
```toml
[lib]
name = "baygraph"
crate-type = ["cdylib", "rlib"]

[dependencies]
pyo3 = { version = "0.21", features = ["auto-initialize"] }

[features]
python = ["pyo3"]
```

**pyproject.toml (create):**
```toml
[build-system]
requires = ["maturin>=1.0,<2.0"]
build-backend = "maturin"

[project]
name = "baygraph"
requires-python = ">=3.8"
classifiers = [
    "Programming Language :: Rust",
    "Programming Language :: Python :: Implementation :: CPython",
    "Programming Language :: Python :: Implementation :: PyPy",
]
```

### Development Build

```bash
# Install maturin
pip install maturin

# Develop mode (editable install)
maturin develop --release

# Or in nix-shell
cd /path/to/baygraph
maturin develop --release
```

### Testing

**Python Tests:**
Create `tests/python/` directory:
```python
# tests/python/test_basic.py
import baygraph

def test_compile():
    source = """
    schema Test { node Person { } edge REL { } }
    belief_model TestBeliefs on Test {
        edge REL { exist ~ BernoulliPosterior() }
    }
    """
    program = baygraph.compile(source)
    assert program is not None

def test_run_flow():
    # ... test flow execution
```

**Run tests:**
```bash
# From Rust side
cargo test --features python

# From Python side (after maturin develop)
pytest tests/python/
```

---

## Example Usage

### Basic Usage

```python
import baygraph

# Compile program
with open("model.bg", "r") as f:
    source = f.read()
program = baygraph.compile(source)

# Run flow with static evidence
ctx = baygraph.run_flow(program, "MyFlow")
print(f"Result metric: {ctx.metrics['my_metric']}")

# Get exported graph
graph = ctx.graphs["result"]
for node in graph.nodes():
    print(f"Node {node.id}: score = {node.E('score')}")
```

### Dynamic Evidence

```python
# Build runtime evidence
evidence = baygraph.Evidence("RuntimeEvidence", model="SocialBeliefs")
evidence.observe_edge("Person", "Alice", "KNOWS", "Person", "Bob", present=True)
evidence.observe_numeric("Person", "Alice", "score", 10.0)

# Run flow with evidence
ctx = baygraph.run_flow_with_evidence(program, "ComputeBudget", evidence)
```

### Competing Edges

```python
# Observe competing edge choices
evidence = baygraph.Evidence("RoutingEvidence", model="NetworkBeliefs")
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

### Phase 1: Core Infrastructure (Week 1)
- [ ] Set up PyO3 project structure (`src/bindings/python/`)
- [ ] Implement `Program` wrapper
- [ ] Implement `compile()` function
- [ ] Basic error conversion (`ExecError` → Python exceptions)
- [ ] GIL management for compile

### Phase 2: Flow Execution (Week 1-2)
- [ ] Implement `Context` wrapper
- [ ] Implement `run_flow()` function
- [ ] Implement `run_flow_with_evidence()` function
- [ ] Implement `run_flow_with_context()` function
- [ ] GIL management for flow execution

### Phase 3: Evidence Builder (Week 2)
- [ ] Implement `Evidence` class
- [ ] Implement `observe_edge()` methods
- [ ] Implement `observe_numeric()` method
- [ ] Implement competing edge evidence methods
- [ ] Validation and error handling

### Phase 4: Graph Inspection (Week 2-3)
- [ ] Implement `BeliefGraph` wrapper
- [ ] Implement `NodeView` and `EdgeView`
- [ ] Implement `nodes()` and `edges()` iterators
- [ ] Implement `E()` and `Var()` methods
- [ ] Implement `competing_groups()` iterator
- [ ] Implement `CompetingGroup` wrapper

### Phase 5: Convenience Exports (Week 3)
- [ ] Implement `to_pandas()` (optional pandas dependency)
- [ ] Implement `to_networkx()` (optional networkx dependency)
- [ ] Handle missing dependencies gracefully

### Phase 6: Testing and Documentation (Week 3-4)
- [ ] Python unit tests for all APIs
- [ ] Integration tests with example programs
- [ ] API documentation (docstrings)
- [ ] Usage examples in docs
- [ ] Performance benchmarks

---

## Exit Criteria

✅ **Complete when:**

1. **From Python, you can:**
   - ✅ `compile()` a `.bg` file source string
   - ✅ Build an `Evidence` object with runtime observations
   - ✅ `run_flow()` with static or dynamic evidence
   - ✅ `run_flow_with_context()` to chain flows
   - ✅ Read metrics from `Context.metrics`
   - ✅ Inspect graphs via `nodes()` and `edges()` iterators
   - ✅ Access posterior beliefs via `E()` and `Var()`
   - ✅ Inspect competing edge groups
   - ✅ Export to pandas/NetworkX

2. **Performance:**
   - ✅ GIL released for long-running operations
   - ✅ No unnecessary Python ↔ Rust round trips
   - ✅ Memory usage reasonable (no leaks)

3. **Error Handling:**
   - ✅ All Rust errors converted to appropriate Python exceptions
   - ✅ Clear error messages with context
   - ✅ No panics reach Python code

4. **Testing:**
   - ✅ Python unit tests pass
   - ✅ Integration tests with real `.bg` files
   - ✅ Edge cases handled (empty graphs, missing nodes, etc.)

---

## References

- **PyO3 Documentation:** https://pyo3.rs/
- **Maturin Documentation:** https://maturin.rs/
- **Design Doc Section 5.11:** Python FFI guidelines (GIL, errors, zero-copy)
- **Design Doc Section 6:** Python integration examples and API table
- **Rust Engine:** `src/engine/` - Core types to wrap
- **Flow Execution:** `src/engine/flow_exec.rs` - `run_flow` implementation
- **Errors:** `src/engine/errors.rs` - `ExecError` enum to convert

---

## Notes

- **Naming:** Python API uses `snake_case` (Python convention), Rust uses `snake_case` for functions and `PascalCase` for types
- **Type Hints:** Add Python type hints for better IDE support
- **Docstrings:** All public methods should have Google-style docstrings
- **Versioning:** Python package version should match Rust crate version
- **Distribution:** Eventually publish to PyPI as `baygraph` (check availability)
