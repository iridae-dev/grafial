# Rust Best Practices for Bayesian Belief Graph Language

This document outlines Rust development best practices specific to the Bayesian Belief Graph language project. These practices are tailored to the unique requirements of probabilistic graph processing: determinism, numerical stability, and immutability.

---

## 1. Error Handling

### 1.1 Use `Result<T, ExecError>` for All Public APIs

**Principle**: Library code should never panic. All public APIs return `Result<T, ExecError>`.

**Implementation**:
```rust
use crate::engine::errors::ExecError;

pub fn update_posterior(&mut self, value: f64) -> Result<(), ExecError> {
    // Validate inputs
    if value.is_nan() || value.is_infinite() {
        return Err(ExecError::ValidationError("invalid value".into()));
    }
    // ... update logic
    Ok(())
}
```

**Rationale**: See `baygraph_design.md:533-541`. User-facing code must handle errors gracefully; panics are only for unrecoverable programmer errors (use `debug_assert!` for those).

### 1.2 Non-Exhaustive Error Enum

Use `#[non_exhaustive]` on `ExecError` to allow future error variants without breaking changes:

```rust
#[non_exhaustive]
#[derive(Debug, Error)]
pub enum ExecError {
    #[error("parse error: {0}")]
    ParseError(String),
    #[error("validation error: {0}")]
    ValidationError(String),
    // ... more variants
}
```

**Reference**: `baygraph_design.md:539`

### 1.3 Validate at Boundaries

Validate user inputs at parser/typechecker boundaries, not deep in the engine:

```rust
// ✅ Good: validate early
pub fn parse_program(source: &str) -> Result<ProgramAst, ExecError> {
    // Parse and validate immediately
}

// ❌ Bad: validate in hot path
pub fn compute_metric(graph: &BeliefGraph) -> Result<f64, ExecError> {
    // Don't validate here - should be validated earlier
}
```

---

## 2. Determinism

### 2.1 Stable Identifiers with `Ord` Trait

Use newtype wrappers with `Ord` for stable ordering:

```rust
#[repr(transparent)]
#[derive(Copy, Clone, Eq, PartialEq, Hash, Ord, PartialOrd, Debug)]
pub struct NodeId(pub u32);

#[repr(transparent)]
#[derive(Copy, Clone, Eq, PartialEq, Hash, Ord, PartialOrd, Debug)]
pub struct EdgeId(pub u32);
```

**Reference**: `baygraph_design.md:471-486`

### 2.2 Always Iterate in Stable Order

**Never rely on hash iteration order**. Always sort by stable IDs:

```rust
// ✅ Good: sorted iteration
pub fn iterate_nodes(&self) -> impl Iterator<Item = &NodeData> {
    let mut nodes: Vec<_> = self.nodes.iter().collect();
    nodes.sort_by_key(|n| n.id);  // Stable order
    nodes.into_iter()
}

// ❌ Bad: hash map iteration
for (id, node) in &self.node_map {  // Order is non-deterministic!
    // ...
}
```

**Reference**: `baygraph_design.md:517-518, 523`

### 2.3 Deterministic Parallelism

When using `rayon` for parallel operations, ensure deterministic results:

```rust
use rayon::prelude::*;

// ✅ Good: sorted before parallel reduction
let sorted_ids: Vec<NodeId> = {
    let mut ids: Vec<_> = graph.nodes.iter().map(|n| n.id).collect();
    ids.sort();  // Stable order
    ids
};

let result = sorted_ids
    .par_iter()
    .map(|id| compute_value(graph, *id))
    .reduce(|| 0.0, |a, b| a + b);  // Pairwise reduction for stability
```

**Reference**: `baygraph_design.md:520-524`

---

## 3. Numerical Stability

### 3.1 Precision Clipping for Gaussian Posteriors

Always clip extremely small precisions to prevent division by zero:

```rust
const MIN_PRECISION: f64 = 1e-6;

pub fn update(&mut self, x: f64, tau_obs: f64) {
    let tau_old = self.precision;
    let tau_obs = tau_obs.max(1e-12);  // Prevent zero observation precision
    let tau_new = (tau_old + tau_obs).max(MIN_PRECISION);  // Clip minimum
    // ... rest of update
}
```

**Reference**: `baygraph_design.md:203`

### 3.2 Large-Finite Values for Force Operations

Use large but finite values, not infinity:

```rust
const FORCE_PRECISION: f64 = 1_000_000.0;  // Not f64::INFINITY

pub fn force_value(&mut self, x: f64) {
    self.mean = x;
    self.precision = FORCE_PRECISION;  // Large but finite
}
```

**Reference**: `baygraph_design.md:107, 133-135, 199-200`

### 3.3 Stable Summation for Large Aggregates

Use Kahan or pairwise summation for large reductions:

```rust
// ✅ Good: Kahan summation for large sums
pub fn sum_nodes_stable(nodes: &[NodeData]) -> f64 {
    let mut sum = 0.0;
    let mut c = 0.0;  // Compensation term
    
    for node in nodes.iter().sorted_by_key(|n| n.id) {
        let term = compute_contribution(node);
        let y = term - c;
        let t = sum + y;
        c = (t - sum) - y;  // Track lost precision
        sum = t;
    }
    sum
}

// For parallel: use pairwise summation
let result = sorted_chunks
    .par_iter()
    .map(|chunk| chunk.iter().sum::<f64>())
    .reduce(|| 0.0, |a, b| a + b);
```

**Reference**: `baygraph_design.md:523`

### 3.4 Validate Beta Parameters

Enforce minimum values for Beta distribution parameters:

```rust
const MIN_BETA_PARAM: f64 = 0.01;

pub fn observe(&mut self, present: bool) {
    if present {
        self.alpha += 1.0;
    } else {
        self.beta += 1.0;
    }
    // Ensure proper prior
    self.alpha = self.alpha.max(MIN_BETA_PARAM);
    self.beta = self.beta.max(MIN_BETA_PARAM);
}
```

**Reference**: `baygraph_design.md:217`

---

## 4. Immutability and Structural Sharing

### 4.1 Immutable Graphs Between Transforms

Graphs are immutable values between transforms. Use `Arc` + copy-on-write:

```rust
use std::sync::Arc;
use std::collections::HashMap;

pub struct BeliefGraph {
    nodes: Arc<Vec<NodeData>>,
    edges: Arc<Vec<EdgeData>>,
    // Indexes for O(1) lookup
    node_index: Arc<HashMap<NodeId, usize>>,
}

impl BeliefGraph {
    pub fn apply_transform(&self, transform: Transform) -> Result<Self, ExecError> {
        // Clone Arc (cheap) and modify only what changed
        let mut new_nodes = Arc::try_unwrap(self.nodes.clone())
            .unwrap_or_else(|arc| (*arc).clone());
        // Apply changes to new_nodes
        // ...
        Ok(Self { nodes: Arc::new(new_nodes), ..self })
    }
}
```

**Reference**: `baygraph_design.md:468, 637`

### 4.2 Copy-on-Write for Deltas

Represent graph views as `{ base: Arc<BeliefGraphInner>, delta: SmallVec<...> }`:

```rust
use smallvec::SmallVec;

pub struct GraphView {
    base: Arc<BeliefGraphInner>,
    delta: SmallVec<[(NodeId, NodeDelta); 8]>,  // Most changes are small
}
```

**Reference**: `baygraph_design.md:474`

### 4.3 Working Copies for Rule Execution

Keep inputs immutable; apply side-effects to working copy:

```rust
pub fn run_rule_for_each(input: &BeliefGraph, rule: &RuleDef) -> Result<BeliefGraph, ExecError> {
    let mut work = input.clone();  // Working copy
    
    for match_binding in find_matches(input, rule)? {
        // Apply actions to working copy, not input
        execute_actions(&mut work, &rule.actions, &match_binding)?;
    }
    
    Ok(work)  // Return new graph
}
```

**Reference**: `baygraph_design.md:527, 533`

---

## 5. Type Safety and Performance

### 5.1 Prefer Enums Over Trait Objects in Hot Paths

Use enums for posterior types to avoid vtable indirection:

```rust
// ✅ Good: enum for hot path
pub enum PosteriorState {
    Gaussian(GaussianPosterior),
    Bernoulli(BetaPosterior),
    Categorical(DirichletPosterior),
}

// ❌ Bad: trait object in hot loop
pub trait Posterior: Send + Sync {
    fn mean(&self) -> f64;
}
```

**Reference**: `baygraph_design.md:473`

### 5.2 Expose Trait Objects at Registry Boundaries Only

Use trait objects for extensibility, but only at the registry boundary:

```rust
// Registry (extensibility boundary)
pub trait MetricFn: Send + Sync + 'static {
    fn eval(&self, graph: &BeliefGraph, args: &MetricArgs, ctx: &MetricContext) 
        -> Result<f64, ExecError>;
}

// Registry implementation
pub struct MetricRegistry {
    functions: Arc<HashMap<String, Arc<dyn MetricFn>>>,
}

// Hot path: use concrete types
impl BeliefGraph {
    pub fn expectation(&self, node: NodeId, attr: &str) -> Result<f64, ExecError> {
        // Direct access to GaussianPosterior, no trait dispatch
        let gaussian = self.get_gaussian(node, attr)?;
        Ok(gaussian.mean)
    }
}
```

**Reference**: `baygraph_design.md:442-465`

### 5.3 Use Struct-of-Arrays for Hot Data

Store hot data in contiguous vectors for cache locality:

```rust
pub struct BeliefGraph {
    // SoA-style: hot data together
    node_ids: Vec<NodeId>,
    node_labels: Vec<String>,
    node_attrs: Vec<HashMap<String, GaussianPosterior>>,
    
    edge_ids: Vec<EdgeId>,
    edge_srcs: Vec<NodeId>,
    edge_dsts: Vec<NodeId>,
    edge_types: Vec<String>,
    edge_exist: Vec<BetaPosterior>,
}
```

**Reference**: `baygraph_design.md:472`

---

## 6. Memory Management

### 6.1 Zero-Copy Where Feasible

Allocate once, reuse across phases:

```rust
// ✅ Good: reuse AST across phases
pub fn compile(ast: &ProgramAst) -> Result<CompiledProgram, ExecError> {
    // AST is already parsed, no re-parsing
    let ir = lower_to_ir(ast)?;
    Ok(CompiledProgram { ast: Arc::new(ast.clone()), ir })
}
```

**Reference**: `baygraph_design.md:438`

### 6.2 Use `Arc` for Shared Immutable Data

Share immutable data structures with `Arc`:

```rust
pub struct ExecutionContext {
    registry: Arc<MetricRegistry>,  // Shared, immutable
    graphs: HashMap<String, Arc<BeliefGraph>>,  // Shared graphs
}
```

**Reference**: `baygraph_design.md:465, 524`

### 6.3 Avoid Global Mutable State

Pass handles through contexts instead of using globals:

```rust
// ✅ Good: pass registry through context
pub fn eval_metric(
    metric: &MetricDef,
    graph: &BeliefGraph,
    registry: &Arc<MetricRegistry>,
    ctx: &MetricContext,
) -> Result<f64, ExecError> {
    // ...
}

// ❌ Bad: global registry
lazy_static! {
    static ref REGISTRY: Mutex<MetricRegistry> = Mutex::new(MetricRegistry::new());
}
```

**Reference**: `baygraph_design.md:465, 524`

---

## 7. Concurrency and Thread Safety

### 7.1 Ensure `Send + Sync` for Shared Types

All types used in parallel contexts must be `Send + Sync`:

```rust
pub struct BeliefGraph {
    // All fields must be Send + Sync
    nodes: Vec<NodeData>,  // ✅ Vec is Send + Sync
    edges: Vec<EdgeData>,  // ✅ Vec is Send + Sync
}

// Verify in tests
#[test]
fn belief_graph_is_send_sync() {
    fn assert_send_sync<T: Send + Sync>() {}
    assert_send_sync::<BeliefGraph>();
}
```

**Reference**: `baygraph_design.md:440, 524`

### 7.2 Shard Mutable State in Parallel Contexts

When parallelizing, shard mutable state per thread:

```rust
use rayon::prelude::*;

pub fn apply_evidence_parallel(
    graph: &BeliefGraph,
    evidence: &[EvidenceItem],
) -> Result<BeliefGraph, ExecError> {
    // Shard evidence by node/edge
    let chunks: Vec<_> = evidence.chunks(evidence.len() / num_threads).collect();
    
    chunks.par_iter()
        .map(|chunk| {
            let mut local_graph = graph.clone();  // Per-thread copy
            for item in chunk {
                local_graph.apply_evidence(item)?;
            }
            Ok(local_graph)
        })
        .reduce(|| Ok(graph.clone()), |a, b| merge_graphs(a?, b?))
}
```

**Reference**: `baygraph_design.md:522`

---

## 8. API Design

### 8.1 Explicit Methods for Force Operations

Provide explicit methods rather than generic "update" with flags:

```rust
impl GaussianPosterior {
    pub fn update(&mut self, x: f64, tau_obs: f64) {
        // Normal Bayesian update
    }
    
    pub fn force_value(&mut self, x: f64) {
        // Force with large precision
        self.mean = x;
        self.precision = FORCE_PRECISION;
    }
}

impl BetaPosterior {
    pub fn observe(&mut self, present: bool) {
        // Normal update
    }
    
    pub fn force_present(&mut self) {
        self.alpha = FORCE_ALPHA;
        self.beta = 1.0;
    }
    
    pub fn force_absent(&mut self) {
        self.alpha = 1.0;
        self.beta = FORCE_BETA;
    }
}
```

**Reference**: `baygraph_design.md:510, 516`

### 8.2 Distribution-Specific Methods in Hot Loops

Keep hot loops monomorphic by using concrete types:

```rust
// ✅ Good: monomorphic hot loop
pub fn update_attributes(graph: &mut BeliefGraph, observations: &[Observation]) {
    for obs in observations {
        match obs {
            Observation::Gaussian { node, attr, value, tau } => {
                let gaussian = graph.get_gaussian_mut(*node, attr).unwrap();
                gaussian.update(*value, *tau);  // Direct call, no trait dispatch
            }
            // ...
        }
    }
}

// ❌ Bad: trait dispatch in hot loop
pub fn update_attributes(graph: &mut BeliefGraph, observations: &[Observation]) {
    for obs in observations {
        let posterior: &mut dyn Posterior = graph.get_posterior_mut(obs.node, obs.attr)?;
        posterior.update(obs.value);  // Vtable lookup every iteration
    }
}
```

**Reference**: `baygraph_design.md:508-514`

---

## 9. Testing

### 9.1 Property Tests for Posterior Invariants

Use `proptest` for testing mathematical invariants:

```rust
use proptest::prelude::*;

proptest! {
    #[test]
    fn gaussian_precision_monotonic(
        mean in -100.0..100.0,
        precision in 0.01..1000.0,
        obs in -100.0..100.0,
        tau_obs in 0.01..1000.0,
    ) {
        let mut g = GaussianPosterior { mean, precision };
        let old_precision = g.precision;
        g.update(obs, tau_obs);
        assert!(g.precision >= old_precision, "precision should increase");
    }
    
    #[test]
    fn beta_mean_bounded(
        alpha in 0.01..1000.0,
        beta in 0.01..1000.0,
    ) {
        let b = BetaPosterior { alpha, beta };
        let mean = b.mean();
        assert!(mean >= 0.0 && mean <= 1.0, "mean must be in [0, 1]");
    }
}
```

**Reference**: `baygraph_design.md:557, 563`

### 9.2 Determinism Tests

Test that operations are deterministic:

```rust
#[test]
fn rule_execution_is_deterministic() {
    let graph = create_test_graph();
    let rule = create_test_rule();
    
    let result1 = run_rule_for_each(&graph, &rule).unwrap();
    let result2 = run_rule_for_each(&graph, &rule).unwrap();
    
    // Should be bitwise identical
    assert_eq!(result1, result2);
}
```

### 9.3 Name Tests with Intent

Use descriptive test names that explain what is being tested:

```rust
// ✅ Good: descriptive name
#[test]
fn where_prob_blocks_action_when_below_threshold() {
    // ...
}

// ❌ Bad: vague name
#[test]
fn test_rule() {
    // ...
}
```

**Reference**: `AGENTS.md:27`

---

## 10. Documentation

### 10.1 Reference Design Doc Sections

Always reference relevant sections of `baygraph_design.md` in code comments:

```rust
// baygraph_design.md:93-101 — Normal-Normal update
pub fn update(&mut self, x: f64, tau_obs: f64) {
    // ...
}

// baygraph_design.md:133-135 — force_absent sets α=1, β=1e6
pub fn force_absent(&mut self) {
    self.alpha = 1.0;
    self.beta = FORCE_BETA;
}
```

### 10.2 Document Numerical Constants

Explain why specific numerical values are chosen:

```rust
/// High precision value for force operations (baygraph_design.md:107, 133-135)
/// 
/// Uses 1e6 instead of infinity to:
/// - Avoid infinities in subsequent calculations
/// - Maintain numerical stability
/// - Allow further (negligible) updates if needed
const FORCE_PRECISION: f64 = 1_000_000.0;

/// Minimum precision for Gaussian posteriors (baygraph_design.md:203)
/// 
/// Prevents division by zero in variance = 1/τ calculations
/// when prior is extremely vague.
const MIN_PRECISION: f64 = 1e-6;
```

---

## 11. Python Bindings (PyO3)

### 11.1 Release GIL for Long Operations

Always release the GIL for computationally expensive operations:

```rust
use pyo3::prelude::*;

#[pymethods]
impl BeliefGraph {
    fn run_flow(&self, flow_name: &str, py: Python) -> PyResult<PyObject> {
        py.allow_threads(|| {
            // Long-running computation
            self.execute_flow(flow_name)
        })
    }
}
```

**Reference**: `baygraph_design.md:555`

### 11.2 Convert Errors to Python Exceptions

Don't expose Rust error types directly:

```rust
use pyo3::exceptions::PyValueError;

pub fn to_python_result<T>(result: Result<T, ExecError>) -> PyResult<T> {
    result.map_err(|e| PyValueError::new_err(format!("{}", e)))
}
```

**Reference**: `baygraph_design.md:556`

---

## 12. Code Organization

### 12.1 Module Structure

Follow the established module structure:

```
src/
├── frontend/    # Parser + AST (pest grammar)
├── ir/          # Intermediate representations
├── engine/      # Belief graph + rule + flow execution
├── metrics/     # Metric function registry
├── storage/     # Graph storage, IDs, indices
└── bindings/    # Python + CLI integration
```

**Reference**: `baygraph_design.md:401-409, AGENTS.md:4`

### 12.2 Naming Conventions

- Modules/dirs: `snake_case`
- Types/traits: `CamelCase`
- Functions/variables: `snake_case`
- Constants: `SCREAMING_SNAKE_CASE`

**Reference**: `AGENTS.md:19`

---

## 13. Performance Considerations

### 13.1 Profile Before Optimizing

Don't optimize prematurely. Profile with `criterion` or `perf` first:

```rust
use criterion::{black_box, criterion_group, criterion_main, Criterion};

fn benchmark_rule_execution(c: &mut Criterion) {
    let graph = create_large_graph();
    let rule = create_test_rule();
    
    c.bench_function("run_rule_for_each", |b| {
        b.iter(|| run_rule_for_each(black_box(&graph), black_box(&rule)))
    });
}
```

**Reference**: `baygraph_design.md:559`

### 13.2 Optimize Hot Paths

Focus optimization on hot paths (rule execution, metric evaluation):

```rust
// Hot path: use direct access, avoid allocations
pub fn prob_mean(&self, edge: EdgeId) -> Result<f64, ExecError> {
    let e = self.edges.get(edge.0 as usize)?;  // Direct index
    let a = e.exist.alpha.max(0.0);
    let b = e.exist.beta.max(0.0);
    Ok(a / (a + b))
}
```

---

## 14. Common Pitfalls to Avoid

### 14.1 ❌ Don't Rely on Hash Iteration Order

```rust
// ❌ Bad
for (id, node) in &self.node_map {
    // Order is non-deterministic!
}

// ✅ Good
let mut sorted: Vec<_> = self.node_map.iter().collect();
sorted.sort_by_key(|(id, _)| *id);
for (id, node) in sorted {
    // Deterministic order
}
```

### 14.2 ❌ Don't Use `unwrap()` in Library Code

```rust
// ❌ Bad
pub fn get_node(&self, id: NodeId) -> &NodeData {
    self.nodes.get(id.0 as usize).unwrap()  // May panic!
}

// ✅ Good
pub fn get_node(&self, id: NodeId) -> Result<&NodeData, ExecError> {
    self.nodes.get(id.0 as usize)
        .ok_or_else(|| ExecError::Internal(format!("missing node {:?}", id)))
}
```

### 14.3 ❌ Don't Use Floating-Point Equality

```rust
// ❌ Bad
if value == 0.0 {  // Floating-point comparison
    // ...
}

// ✅ Good
const EPSILON: f64 = 1e-12;
if value.abs() < EPSILON {
    // ...
}
```

### 14.4 ❌ Don't Mutate Shared State in Parallel

```rust
// ❌ Bad
let mut shared_graph = graph.clone();
observations.par_iter().for_each(|obs| {
    shared_graph.apply_evidence(obs);  // Data race!
});

// ✅ Good
let results: Vec<_> = observations.par_iter()
    .map(|obs| {
        let mut local = graph.clone();
        local.apply_evidence(obs)?;
        Ok(local)
    })
    .collect::<Result<_, _>>()?;
```

---

## Summary

Key principles for this project:

1. **Determinism**: Always sort by stable IDs; never rely on hash order
2. **Numerical Stability**: Clip precisions, use stable summation, validate bounds
3. **Immutability**: Graphs are immutable between transforms; use `Arc` + CoW
4. **Error Handling**: `Result<T, ExecError>` everywhere; no panics in library code
5. **Performance**: Prefer enums in hot paths, SoA storage, zero-copy where possible
6. **Thread Safety**: Ensure `Send + Sync`, shard mutable state in parallel
7. **Documentation**: Reference design doc sections, explain numerical constants

Follow these practices to ensure the codebase remains correct, maintainable, and performant as the project evolves.

