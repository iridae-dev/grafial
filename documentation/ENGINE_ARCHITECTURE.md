# Grafial Engine Architecture

A concise guide to the Rust engine's internal structure for contributors and maintainers.

---

## Monorepo Structure

```
baygraph/
├── crates/
│   ├── grafial-frontend/    # Parser, AST, validation
│   ├── grafial-ir/          # Intermediate representation
│   ├── grafial-core/        # Core engine (graph, rules, flows, metrics)
│   ├── grafial-cli/         # Command-line interface
│   ├── grafial-python/      # Python bindings (PyO3)
│   ├── grafial-tests/       # Integration and unit tests
│   ├── grafial-benches/     # Performance benchmarks
│   └── grafial-vscode/      # VSCode extension
└── documentation/           # Project documentation
```

---

## Crate Overview

### grafial-frontend

**Location:** `crates/grafial-frontend/`

**Modules:**
- **Parser** (`src/parser.rs`): pest grammar → typed AST
- **AST** (`src/ast.rs`): Typed representation of schemas, rules, flows, metrics
- **Validation** (`src/validate.rs`): Semantic validation and type checking
- **Errors** (`src/errors.rs`): Frontend error types (`FrontendError`)

**Responsibilities:**
- Parse Grafial DSL source code
- Build typed AST from parse tree
- Validate schemas, belief models, and references
- Produce `ProgramAst` for execution

**Public API:**
- `parse_program(source: &str) -> Result<ProgramAst, FrontendError>`
- `validate_program(ast: &ProgramAst) -> Result<(), FrontendError>`

### grafial-ir

**Location:** `crates/grafial-ir/`

**Modules:**
- **RuleIR** (`src/rule.rs`): Lowered rule representation
- **FlowIR** (`src/flow.rs`): Lowered flow representation
- **ProgramIR** (`src/program.rs`): Complete program IR

**Purpose:** Canonical lowered representation designed to decouple frontend from engine.

**Status:** Infrastructure exists with lowering functions, but engine currently uses AST directly. See ROADMAP.md for migration plans.

### grafial-core

**Location:** `crates/grafial-core/`

**Modules:**
- **`engine/graph.rs`**: `BeliefGraph` data structure with Bayesian posteriors
- **`engine/evidence.rs`**: Evidence ingestion and graph building
- **`engine/rule_exec.rs`**: Pattern matching and rule execution
- **`engine/flow_exec.rs`**: Flow transform interpreter and pipeline execution
- **`engine/expr_eval.rs`**: Expression evaluation core
- **`engine/expr_utils.rs`**: Expression utility functions
- **`engine/query_plan.rs`**: Query plan optimization and caching
- **`engine/snapshot.rs`**: Graph snapshot serialization
- **`engine/errors.rs`**: Error types (`ExecError`)
- **`metrics/mod.rs`**: Metric function registry
- **`storage/mod.rs`**: Storage utilities (currently minimal)

**Design Goals:**
- Zero-copy between phases where feasible
- Avoid panics in library code; return `Result<T, ExecError>`
- Thread-safe (`Send + Sync`) structures for parallel evaluation
- Immutable graphs between transforms

**Public API:**
- `parse_and_validate(source: &str) -> Result<ProgramAst, ExecError>`
- `run_flow(program: &ProgramAst, flow_name: &str, evidence: &EvidenceDef) -> Result<FlowResult, ExecError>`

---

## Data Model and Storage

### Stable Identifiers

```rust
#[repr(transparent)]
#[derive(Copy, Clone, Eq, PartialEq, Hash, Ord, PartialOrd, Debug)]
pub struct NodeId(pub u32);

#[repr(transparent)]
#[derive(Copy, Clone, Eq, PartialEq, Hash, Ord, PartialOrd, Debug)]
pub struct EdgeId(pub u32);
```

**Design decisions:**
- `u32` for efficient storage (supports up to 4B nodes/edges)
- `Ord` implementation ensures stable, deterministic iteration
- `#[repr(transparent)]` for zero-cost abstraction

### Storage Layout

**Structure-of-Arrays (SoA) style:**
- Contiguous vectors for nodes/edges (cache-friendly)
- Hot data (IDs, endpoints, posterior handles) in separate arrays
- Adjacency index with offset ranges for O(1) neighborhood access

**Adjacency Index:**
- Precomputed and cached for fast queries
- Maps `(NodeId, EdgeType)` → `(start_offset, end_offset)` in edge ID array
- Enables O(1) access to all edges from a node of a given type

### Immutability and Structural Sharing

**Graph representation:**
```rust
pub struct BeliefGraph {
    inner: Arc<BeliefGraphInner>,  // Shared base graph
    delta: SmallVec<GraphDelta>,   // Copy-on-write modifications
}
```

**Design:**
- Base graph is `Arc`-shared for efficient cloning
- Mutations recorded in `delta` (copy-on-write)
- `ensure_owned()` commits delta to base when needed
- Most operations are O(changes) not O(graph size)

**Fine-grained deltas:**
- Store only changed posterior parameters (not full node/edge clones)
- Reduces memory usage and improves commit performance
- Accessor methods apply deltas on-the-fly for reads

### Posterior Storage

**Posterior types:**
- `GaussianPosterior`: `{ mean: f64, precision: f64 }`
- `BetaPosterior`: `{ alpha: f64, beta: f64 }`
- `DirichletPosterior`: `{ concentrations: Vec<f64> }`

**Edge posteriors:**
```rust
pub enum EdgePosterior {
    Independent(BetaPosterior),
    Competing(CompetingEdgeRef),  // Reference to shared group
}
```

**Competing edges:**
- Shared `DirichletPosterior` groups indexed by `(source_node, edge_type)`
- Edges reference group via `CompetingEdgeRef { group_id, category_index }`
- Groups stored in `BeliefGraphInner.competing_groups: FxHashMap<...>`

---

## Determinism and Parallelism

### Determinism Guarantees

- **Stable iteration order**: Iterate by sorted `NodeId`/`EdgeId` (not hash order)
- **Deterministic pattern matching**: Query plans use stable ordering
- **Pairwise summation**: For parallel reductions, use pairwise sum to minimize floating-point error drift

### Parallelism Strategy

- **Optional parallelism**: Use `rayon` behind `#[cfg(feature = "rayon")]`
- **Thread safety**: All shared state is `Send + Sync`
- **Immutable references**: Pass `&BeliefGraph` (immutable) to parallel workers
- **No globals**: Pass `Arc` handles through execution contexts

**Example (metrics):**
```rust
#[cfg(feature = "rayon")]
{
    let nodes: Vec<_> = nodes_sorted_by_id(graph.nodes())  // Stable order
        .into_iter()
        .filter(|n| n.label.as_ref() == label)
        .collect();
    
    let terms: Vec<f64> = nodes.par_iter()  // Parallel processing
        .map(|n| /* compute */)
        .collect();
    
    pairwise_sum(&terms)  // Deterministic reduction
}
```

---

## Error Handling

### Error Types

**Frontend errors** (`grafial-frontend`):
```rust
#[non_exhaustive]
pub enum FrontendError {
    ParseError(String),
    ValidationError(String),
}
```

**Engine errors** (`grafial-core`):
```rust
#[non_exhaustive]
pub enum ExecError {
    ParseError(String),
    ValidationError(String),
    Execution(String),
    Numerical(String),
    Internal(String),
}
```

**Design principles:**
- All public APIs return `Result<T, ExecError>`
- No panics in library code (except debug assertions)
- Validate user inputs at boundaries (parser/typechecker)
- Use `thiserror` for automatic error formatting
- Frontend errors convert to `ExecError` via `From` trait

**Error conversion:**
- Python bindings: Convert `ExecError` to appropriate Python exceptions
- CLI: Format errors with context for user feedback

---

## Rule Engine and Pattern Matching

### Pattern Matching

- **Query plans**: Compile patterns to optimized query plans
- **Caching**: Cache query plans by pattern signature for reuse
- **Selectivity**: Order joins by selectivity (most selective first)
- **Delta-aware**: Pattern matching considers both base graph and pending deltas

### Rule Execution

- **Working copy**: Mutations applied to delta (original graph unchanged)
- **Commit**: Delta committed at end of rule application
- **Fixpoint mode**: Iterate until convergence (`max_change < tolerance`)
- **Deterministic**: Pattern matching uses stable EdgeId ordering

### Where Clauses

- **Short-circuiting**: Evaluate conditions with early exit
- **Hoisting**: Lift invariant expressions out of inner loops
- **Context**: Evaluate against input graph (consistent state)

---

## Extensibility

### Extension Mechanisms

| Area | Extension Mechanism |
|------|---------------------|
| **Graph transforms** | Register new `Transform` in engine (Rust) |
| **Metrics / aggregates** | Implement `MetricFn` trait and register |
| **Built-in functions** | Extend intrinsic function table in `expr_eval` |
| **Python interop** | Call `run_flow` / inspect graphs / metrics |
| **UI parity** | Structured editor over AST; text ⇄ AST round-trip |

### Design Philosophy

**Core principle:** New capability means new **function**, not new **syntax**.

- Extend via registries (metrics, functions) not syntax changes
- Keep DSL minimal and composable
- Add new functionality as Rust implementations, not language constructs

**Example: Adding a new metric**

```rust
// In crates/grafial-core/src/metrics/mod.rs
pub struct MyCustomMetric;

impl MetricFn for MyCustomMetric {
    fn eval(&self, graph: &BeliefGraph, args: &MetricArgs, ctx: &MetricContext) -> Result<f64, ExecError> {
        // Implementation
    }
}

// Register in metric registry
registry.insert("my_metric", Arc::new(MyCustomMetric));
```

No DSL changes required - users can call `my_metric(...)` in their `.grafial` files.

### Metric Function Registry

```rust
pub trait MetricFn: Send + Sync + 'static {
    fn eval(
        &self,
        graph: &BeliefGraph,
        args: &MetricArgs,
        ctx: &MetricContext,
    ) -> Result<f64, ExecError>;
}
```

**Registry design:**
- Immutable `Arc<HashMap<String, Arc<dyn MetricFn>>>`
- Constructed at engine initialization
- Passed through execution contexts (no globals)
- Enables testing and determinism

---

## Testing Strategy

### Unit Tests

- Co-located with modules (`#[cfg(test)]` blocks)
- Test posterior update invariants
- Test rule execution edge cases
- Test numerical stability

### Integration Tests

- Located in `crates/grafial-tests/tests/`
- End-to-end flow execution
- Evidence building and application
- Rule pattern matching

### Property Tests

- Use `proptest` for posterior invariants
- Test determinism (same input → same output)
- Test numerical stability bounds

### Benchmarks

- Located in `crates/grafial-benches/benches/`
- Use `criterion` for performance tracking
- Bench evidence application, rule evaluation, metric scans
- Track allocations with profiling tools

---

## Performance Optimizations

**Implemented:**
- Fine-grained delta compression (store only changed values)
- String interning (`Arc<str>` for edge types, node labels)
- FxHashMap for integer-keyed maps (faster hashing)
- Query plan caching (avoid re-analysis)
- Copy-on-write semantics (reduce cloning)
- Iterator optimizations (unstable sort, pre-allocated capacity)

**Future:** See `ROADMAP.md` for planned optimizations including parallelization, algorithmic improvements, and memory layout optimizations.

---

## Python Bindings

**Location:** `crates/grafial-python/`

**Design principles:**
- Thin wrappers over stable Rust types
- Don't expose internal enums directly
- Immutable graph semantics (mutations return new handles)
- Release GIL for long-running operations
- Convert `ExecError` to Python exceptions

**Status:** Placeholder implementation exists. See `PYTHON_PLAN.md` for detailed implementation plan.

---

## Key Design Rules

1. **Graphs are immutable values** between transforms
2. **Rules and metrics are pure** functions of current graph + context
3. **Flows define dataflow**, not control flow
4. **Metrics are the only sanctioned way** to handle global scalars
5. **All extensions come from function registries**, not syntax inflation

---

## References

- **Language Guide**: `LANGUAGE_GUIDE.md` - User-facing language documentation
- **Roadmap**: `ROADMAP.md` - Future work and performance improvements
- **Python Plan**: `PYTHON_PLAN.md` - Python bindings implementation plan
- **Building**: `BUILDING.md` - Build and development setup
