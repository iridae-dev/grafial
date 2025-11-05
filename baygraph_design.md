# Bayesian Belief Graph Language — Design Specification

**A language for reasoning when connections aren't clear.**

**Version:** Draft 2.0  
**Scope:** Probabilistic, rule-based graph language with declarative queries and dataflow, plus constrained imperative updates  
**Target Implementation:** Rust engine with optional Python + UI integration

---

## 1. Purpose

Define a minimal, expressive **probabilistic, rule-based graph language** with declarative queries and dataflow, plus constrained imperative updates inside rules.

The language addresses the challenge of reasoning about uncertain graph structures: when relationships between entities are probabilistic rather than deterministic, and when evidence must be combined with prior beliefs to make inferences.

The language operates on **Bayesian belief graphs** — graphs whose nodes, edges, and attributes hold probabilistic states. It allows users to:

- Declare graph schemas and belief models  
- Ingest evidence  
- Express probabilistic, rule-based transformations with declarative pattern matching
- Chain transformations as declarative flows  
- Compute and transfer scalar metrics between flows
- Apply constrained imperative updates (expectation setting, edge forcing) within rule actions

All inference and execution run on a **Rust engine**; the language acts as a serialization + rule definition layer.

---

## 2. Conceptual Layers

| Layer | Responsibility |
|-------|----------------|
| **Schema** | Defines graph structure and attribute types |
| **Belief Model** | Defines posterior families for each element |
| **Evidence** | Updates belief state from observations |
| **Rules** | Declarative probabilistic transformations |
| **Flows** | Composable pipelines of transformations |
| **Metrics** | Scalar or aggregate computations over graphs |

---

## 3. Language Constructs

### 3.1 Schema

```bayscript
schema Social {
  node Person {
    some_value: Real
    other_value: Real
  }

  edge REL { }
}
```

Defines node and edge types with attributes.

---

### 3.2 Belief Model

```bayscript
belief_model SocialBeliefs on Social {
  edge REL {
    exist ~ BernoulliPosterior(prior = 0.1, pseudo_count = 2.0)
  }

  node Person {
    some_value  ~ GaussianPosterior(prior_mean = 0.0, prior_precision = 0.01)
    other_value ~ GaussianPosterior(prior_mean = 0.0, prior_precision = 0.01)
  }
}
```

Each attribute or edge has a posterior distribution type.

---

### 3.2.1 Posterior Parametrization

Unless stated otherwise, posterior families are chosen to be conjugate to the evidence likelihood so that closed‑form updates exist and are numerically stable.

#### Gaussian Posterior (Normal–Normal)

A `GaussianPosterior()` assumes observations come from a Normal with known precision (likelihood precision τ_obs). It maintains sufficient statistics: mean μ and precision τ = 1/σ².

```bayscript
GaussianPosterior(
  prior_mean = 0.0,
  prior_precision = 0.01,  // weak prior: high variance (σ² = 100)
  observation_precision = 1.0  // likelihood precision for evidence
)
```

**Default prior**: μ₀ = 0.0, τ₀ = 0.01 (variance = 100)

**Bayesian update** for one observation x with precision τ_obs (Normal–Normal conjugacy):
- τ_new = τ_old + τ_obs
- μ_new = (τ_old × μ_old + τ_obs × x) / τ_new

For a batch of independent observations {x_i} with the same τ_obs, the update is equivalent to adding n × τ_obs to precision and a weighted sum to the mean numerator:
- τ_new = τ_old + n × τ_obs
- μ_new = (τ_old × μ_old + τ_obs × Σ x_i) / τ_new

Posterior variance is σ²_post = 1/τ_new.

Posterior predictive for a future observation (with the same τ_obs) has mean μ_new and variance σ²_pred = 1/τ_new + 1/τ_obs.

**Forcing a value** (via deterministic evidence):
- Sets μ = observed_value
- Sets τ = 10⁶ (large but finite precision to avoid infinities)
This corresponds to conditioning on an observation with effectively infinite precision; subsequent finite‑precision updates will have negligible effect but remain well‑defined.

#### Bernoulli Posterior (Beta–Bernoulli)

A `BernoulliPosterior()` maintains a Beta distribution over probability p ∈ [0,1]:

```bayscript
BernoulliPosterior(
  prior = 0.1,  // maps to Beta(α₀, β₀) via method of moments
  pseudo_count = 2.0  // total pseudo-observations
)
```

**Prior parametrization**: Given prior mean p₀ and pseudo_count n₀ > 0 (total pseudo‑observations):
- α₀ = p₀ × n₀
- β₀ = (1 - p₀) × n₀

**Default**: prior = 0.5, pseudo_count = 2.0 → Beta(1, 1) (uniform)

**Bayesian update** (Beta–Bernoulli conjugacy):
- Observe "present": α_new = α_old + 1, β_new = β_old
- Observe "absent": α_new = α_old, β_new = β_old + 1
- Posterior mean: p = α / (α + β)
- Posterior variance: Var[p] = αβ / [(α + β)² (α + β + 1)]

**Forcing states** (via `force_present`/`force_absent`):
- `force_present`: α = 10⁶, β = 1 (mean ≈ 0.999999)
- `force_absent`: α = 1, β = 10⁶ (mean ≈ 0.000001)
These are numerically stable approximations to degenerate beliefs; they avoid infinities while making further single observations inconsequential.

#### Categorical Posterior (Dirichlet–Categorical)

For mutually exclusive choices among K alternatives (e.g., "one outgoing edge must be chosen" per source), use a `CategoricalPosterior` with a Dirichlet prior over category probabilities π ∈ Δ^{K−1}.

**When to use:**
- Routing: one outgoing path per packet
- Resource allocation: one task assignment per worker
- State machines: one transition per state
- Tournament brackets: one winner per match

**When NOT to use:**
- Social networks (users can follow multiple people) → use independent Bernoulli
- Multi-label classification (multiple labels can apply) → use independent Bernoulli

**Syntax:**

```bayscript
belief_model NetworkBeliefs on NetworkSchema {
  edge ROUTES_TO {
    exist ~ CategoricalPosterior(
      group_by = "source",           // Required: partition by "source" or "destination"
      prior = uniform,                // Keyword: uniform Dirichlet (all α_k equal)
      pseudo_count = 3.0              // Total Σ α_k = pseudo_count
    )
  }
}
```

**Explicit prior per category** (when categories known statically):

```bayscript
belief_model WorkflowBeliefs on WorkflowSchema {
  edge ASSIGNS {
    exist ~ CategoricalPosterior(
      group_by = "source",
      prior = [2.0, 3.0, 5.0],       // Biased: α = [2, 3, 5], favors third option
      categories = ["Task_A", "Task_B", "Task_C"]  // Optional: for validation
    )
  }
}
```

**Mathematical foundation:**
- Edges from source u form a group with shared belief: π ~ Dirichlet(α_1, ..., α_K)
- Constraint: Σ_k π_k = 1 (exactly one choice)
- Evidence about choosing edge (u, v_k) increments α_k

**Bayesian update** (Dirichlet–Categorical conjugacy):
- Observe category k: α_k,new = α_k,old + 1
- Posterior mean: E[π_k] = α_k / Σ_j α_j
- Posterior variance: Var[π_k] = (α_k (α_0 - α_k)) / (α_0² (α_0 + 1)) where α_0 = Σ_j α_j

**Forcing a choice** (via deterministic evidence):
- `force_choice` for category k: α_k = 10⁶, α_j = 1.0 for j ≠ k
- This is a numerically stable approximation to a degenerate distribution

**Validation rules:**
- `group_by` must be "source" or "destination"
- If `prior = uniform`, must specify `pseudo_count`
- If `prior = [...]` (array), pseudo_count is sum of array
- All values in `prior` array must be > 0 (proper Dirichlet)
- Enforce α_k ≥ 0.01 for all k to prevent improper priors

When modeling "competing" edges (e.g., exactly one destination per source), prefer `CategoricalPosterior` grouped by source over renormalizing independent Bernoulli probabilities, which is not Bayesian.

---

### 3.2.2 Prior Sensitivity and Specification

**Prior choice significantly impacts small-sample inference.** Users must understand prior influence.

#### Recommended Prior Strategies

**Weakly informative priors** (default):
```bayscript
node Person {
  age ~ GaussianPosterior(
    prior_mean = 35.0,           // reasonable center
    prior_precision = 0.01       // SD = 10, weakly informative
  )
}
```

**Informative priors** (domain knowledge):
```bayscript
node Person {
  height_cm ~ GaussianPosterior(
    prior_mean = 170.0,
    prior_precision = 0.1        // SD = 3.16, stronger belief
  )
}
```

**Skeptical priors** (high evidence threshold):
```bayscript
edge FRAUD {
  exist ~ BernoulliPosterior(
    prior = 0.01,                // rare event
    pseudo_count = 100           // requires strong evidence to shift
  )
}
```

---

### 3.2.3 Numerical Stability and Edge Cases

#### Gaussian Posterior Edge Cases

**Infinite precision** (degenerate posteriors):
- Represented as τ = 10⁶, not true infinity
- Operations checking "is fixed?" use threshold: τ > 10⁵

**Numerical stability (small precision)**:
- When τ is extremely small (very vague prior), clip to τ_min = 10⁻⁶ to prevent division by zero in variance = 1/τ and loss of significance.

**Extreme observations**:
- If |x - μ| > 10 × σ, issue warning (potential outlier or data error)
- User can set `outlier_threshold` in belief model

#### Bernoulli Posterior Edge Cases

**Near-deterministic beliefs**:
- α or β > 10⁶ treated as “effectively fixed” for control‑flow checks
- Queries return posterior means (near 0 or 1); round to {0,1} only if explicitly requested by the caller to avoid masking uncertainty

**Zero/invalid pseudo-counts**:
- Improper prior (α ≤ 0 or β ≤ 0) is forbidden
- Enforce α ≥ 0.01, β ≥ 0.01 as a numeric floor for stability

**Numerical precision**:
- For α, β > 10⁴, compute Beta/Gamma terms in log‑space using stable log‑Gamma; use Stirling or Lanczos approximations as appropriate

---

### 3.3 Evidence

**Independent edges** (Bernoulli posterior):

```bayscript
evidence SocialEvidence on SocialBeliefs {
  observe edge REL(Person["Alice"], Person["Bob"]) present
  observe edge REL(Person["Alice"], Person["Carol"]) absent

  observe Person["Alice"].some_value = 10.0
  observe Person["Bob"].other_value  = 5.0
}
```

**Competing edges** (Categorical posterior):

```bayscript
evidence NetworkEvidence on NetworkBeliefs {
  // Competing edges: explicit choices
  observe edge ROUTES_TO(Server["S1"], Server["S2"]) chosen
  observe edge ROUTES_TO(Server["S3"], Server["S5"]) chosen

  // Negative evidence (rarely used)
  observe edge ROUTES_TO(Server["S7"], Server["S8"]) unchosen

  // Deterministic choice
  observe edge ROUTES_TO(Server["S9"], Server["S10"]) forced_choice
}
```

**Evidence modes and semantics:**

| Edge Type | Statement | Posterior Update | Notes |
|-----------|-----------|------------------|-------|
| Independent | `observe edge E(A,B) present` | Beta: α += 1 | "Edge exists" |
| Independent | `observe edge E(A,B) absent` | Beta: β += 1 | "Edge doesn't exist" |
| Competing | `observe edge E(A,B) chosen` | Dirichlet: α_B += 1 | "A chose B from all options" |
| Competing | `observe edge E(A,B) unchosen` | Dirichlet: α_k += 1/(K-1) for k≠B | "A didn't choose B" (rare usage) |
| Competing | `observe edge E(A,B) forced_choice` | Dirichlet: α_B = 1e6, others = 1.0 | "A deterministically chose B" |

Evidence updates posteriors (Bayesian update or forced states). The evidence mode must match the edge posterior type (independent vs. competing); type checking occurs at runtime (or compile-time if evidence is static).

---

### 3.4 Rules

```bayscript
rule TransferAndDisconnect on SocialBeliefs {

  pattern
    (A:Person)-[ab:REL]->(B:Person),
    (B)-[bc:REL]->(C:Person)

  where
    prob(ab) >= 0.9 and prob(bc) >= 0.9
    and exists (A)-[ax:REL]->(X)
      where prob(ax) >= 0.9 and X != B and X != C
    and not exists (A)-[ac:REL]->(C) where prob(ac) >= 0.5
    and degree(C, min_prob=0.9) == 1

  action {
    let v_ab = E[A.some_value] / 2
    set_expectation A.some_value = E[A.some_value] - v_ab
    set_expectation B.some_value = E[B.some_value] + v_ab

    let v_bc = E[B.other_value] / 2
    set_expectation B.other_value = E[B.other_value] - v_bc
    set_expectation C.other_value = E[C.other_value] + v_bc

    force_absent bc
  }

  mode: for_each
}
```

Semantics:

- Pattern binds variables over uncertain graph structure.  
- `where` filters by posterior probabilities or expectations.  
- `action` mutates belief summaries deterministically: for Gaussian posteriors, `set_expectation X = v` sets the posterior mean μ := v while leaving precision τ unchanged; for Bernoulli, force operations set near‑deterministic Beta parameters as specified above.  
- `mode` defines iteration behavior (`for_each`, `fixpoint`).

**Query functions for competing edges:**

The following functions work with both independent and competing edges, with semantics that adapt to the edge type:

- **`prob(edge)`**: For independent edges, returns E[p] = α / (α + β). For competing edges, returns E[π_k] = α_k / Σ_j α_j.
- **`degree(node, edge_type, min_prob)`**: For independent edges, counts edges with E[p] ≥ threshold. For competing edges, counts categories with E[π_k] ≥ threshold. Since Σ π_k = 1 for competing edges, at most one category can exceed 0.5 (unless K=2 and both ≈ 0.5).

**Competing-edge-specific query functions:**

- **`winner(node, edge_type, direction="outgoing", epsilon=0.01)`**: Returns the edge with maximum E[π_k], or `null` if tied within `epsilon`. Useful for deterministic routing decisions.

```bayscript
rule CheckWinner on NetworkBeliefs {
  pattern
    (A:Server)
  where
    winner(A, outgoing=ROUTES_TO) == B
    // B is the clear winner (max E[π_k] with no ties)
  action { /* ... */ }
}
```

- **`entropy(node, edge_type, direction="outgoing")`**: Returns Shannon entropy H(π) = -Σ π_k log(π_k) in nats. Range: [0, log(K)] where K = number of categories. H = 0 means deterministic (one π_k ≈ 1), H = log(K) means uniform (all π_k ≈ 1/K). Useful for detecting high-uncertainty routing decisions.

```bayscript
rule UncertainRoute on NetworkBeliefs {
  pattern
    (A:Server)
  where
    entropy(A, outgoing=ROUTES_TO) > 1.5
    // High uncertainty: no clear winner
  action {
    // Flag for manual review or collect more evidence
  }
}
```

- **`prob_vector(node, edge_type, direction="outgoing")`**: Returns [E[π_1], E[π_2], ..., E[π_K]] for all categories. Primarily for metrics and export, not pattern matching.

---

### 3.5 Flows

```bayscript
flow DefaultPipeline on SocialBeliefs {

  graph base =
    from_evidence SocialEvidence

  graph cleaned =
    base
      |> apply_ruleset { TransferAndDisconnect }
      |> prune_edges REL where prob(edge) < 0.05
      |> snapshot "post_cleanup"

  metric avg_degree = avg_degree(Person, REL, min_prob=0.8)
  export_metric avg_degree as "avg_deg"

  export cleaned as "cleaned_graph"
}
```

Each step is a pure transform; graphs are immutable pipeline values.

Note on edge competition: If outgoing edges from a node are mutually exclusive or should sum to one, model them with a `CategoricalPosterior` grouped by source (Dirichlet–Categorical) rather than renormalizing independent Bernoulli posteriors. Renormalizing Bernoulli probabilities across a group is not Bayesian and should be avoided for posterior updates (it can be used for ranking/visualization only).

---

### 3.6 Metrics

Metrics are scalar expressions evaluated over graphs.  
They are the **canonical way to compute and transfer global values**.

#### 3.6.1 Simple metric

Expected fraction of active nodes (uses posterior probabilities as soft weights):

```bayscript
metric active_ratio =
  sum_nodes(label=Person, contrib=P(node.active == true)) /
  count_nodes(label=Person)
```

#### 3.6.2 Concurrent aggregation — `sum_nodes`

```bayscript
metric global_risk =
  sum_nodes(
    label   = Person,
    where   = P(node.active == true) > 0.7,
    contrib = E[node.local_risk]
  )
```

#### 3.6.3 Sequential aggregation — `fold_nodes`

```bayscript
metric final_budget =
  fold_nodes(
    label    = Person,
    where    = P(node.active == true) > 0.5,
    order_by = node.stage_index ASC,
    init     = base_budget,
    step     = value * E[node.multiplier]
  )
```

Both return scalars and can be used or exported between flows.

---

### 3.7 Cross-Flow Metric Transfer

```bayscript
flow ComputeBudget on SocialBeliefs {
  metric base_budget = 1000.0

  graph g = from_evidence SocialEvidence
    |> apply_rule CleanUp

  metric final_budget =
    fold_nodes(
      label    = Person,
      order_by = node.stage_index ASC,
      init     = base_budget,
      step     = value * E[node.multiplier]
    )

  export_metric final_budget as "scenario_budget"
}

flow ApplyBudget on SocialBeliefs {
  import_metric scenario_budget as budget

  graph result =
    from_evidence SocialEvidence
      |> apply_rule AllocateResources(budget_limit = budget)

  export result as "allocation_graph"
}
```

---

## 4. Core Semantics Summary

| Construct | Scope / Effect |
|------------|----------------|
| `schema` | Structural definition |
| `belief_model` | Statistical representation |
| `evidence` | Bayesian update input |
| `rule` | Pattern + condition + action; modifies beliefs |
| `flow` | Sequence of transforms over graphs |
| `metric` | Expression producing a scalar |
| `export_metric` / `import_metric` | Transfer scalars between flows |
| `export` / `from_graph` | Transfer graphs between flows |

---

## 5. Rust Engine Architecture

```
baygraph/
├── frontend/       # pest parser + AST builder
├── ir/             # intermediate representations
├── engine/         # belief graph + rule + flow execution
├── metrics/        # registry of metric/aggregator functions
├── storage/        # graph storage, ids, arenas, indices
└── bindings/       # Python + CLI integration
```

### 5.1 Parser (frontend/)

- Grammar: `pest`
- Produces typed AST for schemas, rules, flows, metrics.

### 5.2 IR

- Canonical lowered representation (`GraphIR`, `RuleIR`, `FlowIR`).
- Expression trees with typed nodes.

### 5.3 Engine

Implements:

- `BeliefGraph` (nodes, edges, posteriors)
- Evidence ingestion
- Pattern matching and rule execution
- Flow transform interpreter
- Metric evaluation engine

Design goals:
- Zero‑copy between phases where feasible; allocate once, reuse.
- Avoid panics in library code; return `Result<_, Error>`.
- Thread‑safe (Send + Sync) structures to allow parallel evaluation.

### 5.4 Metric Function Registry

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

Built-ins:

- `sum_nodes`
- `fold_nodes`
- `count_nodes`
- `avg_degree`
- etc.

Extensible by registering new `MetricFn` in Rust; no DSL changes required.

Implement the registry as an immutable map `Arc<HashMap<Symbol, Arc<dyn MetricFn>>>` constructed at engine init; pass a handle through execution contexts for testability and determinism (avoid globals).

---

### 5.5 Data Model and Storage (engine/, storage/)

- Stable identifiers: newtype wrappers `NodeId(u32)`, `EdgeId(u32)` implement `Copy + Eq + Hash + Ord` to ensure stable ordering and deterministic iteration.
- Storage choice: contiguous vectors for nodes/edges with SoA‑style fields for hot data (ids, endpoints, posterior handles) to improve cache locality. Maintain adjacency via offset ranges per node for O(1) neighborhood access.
- Posteriors: store as enum `PosteriorState` with small variants (`Gaussian`, `Bernoulli`, `Categorical`) and boxed payloads if needed. Prefer enum over trait objects in the hot path to minimize vtable indirection; expose trait objects at the registry boundary only.
- Snapshots: graphs are immutable between transforms via structural sharing (e.g., `Arc` + copy‑on‑write of modified vectors). Represent a `GraphView` as `{ base: Arc<BeliefGraphInner>, delta: SmallVec<...> }` so most steps are O(changes).

Example identifiers:

```rust
#[repr(transparent)]
#[derive(Copy, Clone, Eq, PartialEq, Hash, Ord, PartialOrd, Debug)]
pub struct NodeId(pub u32);

#[repr(transparent)]
#[derive(Copy, Clone, Eq, PartialEq, Hash, Ord, PartialOrd, Debug)]
pub struct EdgeId(pub u32);
```

---

### 5.6 Posterior API

Expose a unified API for updates and summaries without over‑abstracting:

```rust
pub enum PosteriorState {
    Gaussian(GaussianPosterior),
    Bernoulli(BetaPosterior),
    Categorical(DirichletPosterior),
}

impl PosteriorState {
    pub fn mean_scalar(&self) -> Option<f64>;    // Some for Gaussian/Bernoulli
    pub fn mean_vector(&self) -> Option<&[f64]>; // Some for Categorical
    pub fn variance_scalar(&self) -> Option<f64>;
}
```

Updates are distribution‑specific methods on concrete types to keep hot loops monomorphic:

```rust
impl GaussianPosterior { pub fn update(&mut self, x: f64, tau_obs: f64) { /* as spec */ } }
impl BetaPosterior     { pub fn observe(&mut self, present: bool) { /* as spec */ } }
impl DirichletPosterior{ pub fn observe(&mut self, k: usize) { /* increment α_k */ } }
```

Force operations set large‑finite precision/concentration as specified in Section 3.2.1; provide explicit methods `force_value`, `force_present`, `force_absent`, `force_choice`.

**Competing edges data structures:**

For competing edges, store references to shared Dirichlet groups:

```rust
pub enum EdgePosterior {
    Independent(BetaPosterior),
    Competing(CompetingEdgeRef),
}

pub struct CompetingEdgeRef {
    group_id: GroupId,
    category_index: usize,
}

pub struct DirichletGroupPosterior {
    source_node: NodeId,
    edge_type: EdgeTypeId,
    categories: Vec<NodeId>,        // destination nodes
    concentrations: Vec<f64>,       // α_k parameters
}
```

Groups are indexed by (source_node, edge_type, direction) for O(1) lookup. The `prob()` function for competing edges retrieves the group and computes E[π_k] = α_k / Σ_j α_j.

---

### 5.7 Determinism and Parallelism

- Parallelism: use `rayon` for embarrassingly parallel passes (evidence application, metric scans). Ensure all shared state is behind immutable references or interior mutability that is sharded per thread.
- Determinism: when parallel, reduce over sorted stable IDs; never rely on hash iteration order. Use pairwise or Kahan summation for large reductions to minimize floating‑point error drift.
- Send/Sync: ensure `BeliefGraph` and registry types implement `Send + Sync`. Avoid global mutable singletons; pass `Arc` handles.

---

### 5.8 Rule Engine and Pattern Matching

- Compile patterns to an indexed query plan over adjacency lists, using join ordering heuristics (start with most selective predicate).
- Evaluate `where` clauses with short‑circuiting, hoisting invariant expressions out of inner loops.
- `mode: fixpoint` executes until no changes above a configurable tolerance; detect convergence via a batched delta set.
- Side‑effects in actions operate on a working copy (delta) to preserve referential transparency of inputs; commit at the end of the rule application step.

---

### 5.9 Errors, Results, and Logging

- Public API returns `Result<T, ExecError>`; define a `#[non_exhaustive]` error enum with variants like `Parse`, `Type`, `Exec`, `Numerical`, `Oom` using `thiserror`.
- Avoid panics except for unrecoverable programmer errors (debug‑assert only). Validate user inputs at the boundary (parser/typechecker).
- Use `tracing` for structured logs with spans per flow/rule; expose a feature flag `tracing` to opt‑in at compile time.

---

### 5.10 Serialization and Reproducibility

- Derive `serde::{Serialize, Deserialize}` for IR and posterior types for snapshots/checkpointing.
- Record engine/version metadata and registry hashes inside snapshots for compatibility checks.
- If any randomized transforms are added later, route RNG via an explicit `Seed` in `ExecutionContext` for reproducibility.

---

### 5.11 Python FFI (PyO3)

- Release the GIL for long‑running computations (`Python::allow_threads`).
- Convert Rust errors into rich Python exceptions; do not expose `anyhow`/`thiserror` types directly.
- Zero‑copy where possible (e.g., expose arrays via `PyReadonlyArray1<f64>` if using `numpy`), but prefer correctness over zero‑copy for complex structures.

---

### 5.12 Testing and Benchmarks

- Golden tests for parsing/typechecking from DSL files; property tests (`proptest`) for posterior update invariants (monotonic precision, bounds 0–1, etc.).
- Differential tests of rule engine on small graphs against a reference Python implementation if available.
- Benchmarks with `criterion` for evidence application, rule evaluation, and metric scans; track allocations with `track_caller` + `heaptrack` in CI as needed.

---

## 6. Python Integration (via PyO3)

```python
import baygraph

program = baygraph.compile(open("model.bg").read())

evidence = baygraph.Evidence("RuntimeEvidence", model="SocialBeliefs")
evidence.observe_edge("Person", "Alice", "REL", "Person", "Bob", present=True)
evidence.observe_numeric("Person", "Alice", "some_value", 10.0)

ctx = baygraph.Context()
ctx = baygraph.run_flow_with_evidence(program, "ComputeBudget", evidence, ctx)
ctx = baygraph.run_flow_with_context(program, "ApplyBudget", ctx)

print(ctx.metrics["scenario_budget"])
```

Python APIs:

| Function | Purpose |
|-----------|----------|
| `compile(source)` | Parse and compile DSL |
| `run_flow(program, flow_name)` | Run flow from static evidence |
| `run_flow_with_evidence(...)` | Inject runtime evidence |
| `run_flow_with_context(...)` | Chain flows (graphs + metrics) |
| `BeliefGraph` | Inspect posterior beliefs; export to pandas/networkx |

Rust API guidelines for Python wrappers:
- Thin wrappers over stable Rust types; avoid exposing internal enums directly.
- Methods that mutate graphs return new handles (immutable graph semantics) and keep old versions alive via `Arc` sharing.
- Long‑running methods release the GIL; short getters keep it.

---

## 7. Extensibility Principles

| Area | Extension Mechanism |
|-------|---------------------|
| **Graph transforms** | Register new `Transform` in engine (Rust) |
| **Metrics / aggregates** | Implement new `MetricFn` |
| **Built-in functions in expressions** | Extend intrinsic function table |
| **Python interop** | Call `run_flow` / inspect graphs / metrics |
| **UI parity** | Structured editor over AST; text ⇄ AST round-trip |

Design never adds new top-level constructs unless absolutely necessary — new capability means new **function**, not new **syntax**.

---

## 8. Implementation Phases

1. **v0 Parser & AST**  
   - pest grammar → typed AST (schema, rule, flow, metric)  
2. **v0 Engine**  
   - Simple belief graph + rule interpreter  
   - Implement `sum_nodes`, `fold_nodes`  
3. **v1 Flow Runner**  
   - Immutable graph pipelines, snapshots, exports  
4. **v1 Python Binding**  
   - PyO3 + `maturin` build; expose compile/run/inspect  
5. **v2 Competing Edges** (Phase 7+)
   - CategoricalPosterior with Dirichlet groups
   - Evidence modes: `chosen`, `unchosen`, `forced_choice`
   - Query functions: `winner()`, `entropy()`, `prob_vector()`
   - Python API: `graph.competing_groups()`
6. **v2 UI**  
   - Structured editor for schema/rule/flow/metric  
   - Graph + belief visualizations  

---

## 9. Key Design Rules

- Graphs are **immutable values** between transforms.  
- Rules and metrics are **pure** functions of current graph + context.  
- Flows define **dataflow**, not control flow.  
- Metrics are the only sanctioned way to handle **global scalars**.  
- All extensions come from **function registries**, not syntax inflation.

---

## 10. Example End-to-End

```bayscript
schema Social {
  node Person { multiplier: Real }
  edge REL { }
}

belief_model SocialBeliefs on Social {
  node Person { multiplier ~ GaussianPosterior() }
  edge REL { exist ~ BernoulliPosterior(prior = 0.2) }
}

evidence SocialEvidence on SocialBeliefs {
  observe edge REL(Person["A"], Person["B"]) present
  observe Person["A"].multiplier = 1.2
  observe Person["B"].multiplier = 0.8
}

flow Compute on SocialBeliefs {
  metric base = 100.0
  metric result = fold_nodes(
    label = Person,
    order_by = node.id ASC,
    init = base,
    step = value * E[node.multiplier]
  )
  export_metric result as "final_value"
}
```

Run from Python:

```python
prog = baygraph.compile(open("social.bg").read())
ctx  = baygraph.run_flow(prog, "Compute")
print(ctx.metrics["final_value"])
```

---

## 11. Example: Competing Edges (Network Packet Routing)

### Scenario: Network Packet Routing

**Schema:**
```bayscript
schema Network {
  node Server {
    load: Real
  }

  edge ROUTES_TO { }
}
```

**Belief Model:**
```bayscript
belief_model NetworkBeliefs on Network {
  node Server {
    load ~ GaussianPosterior(prior_mean = 0.0, prior_precision = 0.01)
  }

  edge ROUTES_TO {
    exist ~ CategoricalPosterior(
      group_by = "source",
      prior = uniform,
      pseudo_count = 5.0
    )
  }
}
```

**Evidence:**
```bayscript
evidence Observations on NetworkBeliefs {
  // Observed routing choices
  observe edge ROUTES_TO(Server["S1"], Server["S2"]) chosen
  observe edge ROUTES_TO(Server["S1"], Server["S2"]) chosen
  observe edge ROUTES_TO(Server["S1"], Server["S3"]) chosen

  // Server loads
  observe Server["S2"].load = 0.7
  observe Server["S3"].load = 0.3
}
```

**Rule: Load Balancing Alert**
```bayscript
rule LoadBalanceAlert on NetworkBeliefs {
  pattern
    (A:Server)

  where
    entropy(A, outgoing=ROUTES_TO) < 0.5
    and exists (A)-[ax:ROUTES_TO]->(X)
      where prob(ax) >= 0.6 and E[X.load] > 0.6
    // A routes mostly to one destination X, and X is overloaded

  action {
    // Flag for rebalancing
  }
}
```

**Flow:**
```bayscript
flow AnalyzeRouting on NetworkBeliefs {
  graph base = from_evidence Observations

  graph flagged = base
    |> apply_rule LoadBalanceAlert

  metric total_entropy = sum_nodes(
    label = Server,
    contrib = entropy(node, outgoing=ROUTES_TO)
  )

  export_metric total_entropy as "routing_diversity"
  export flagged as "result"
}
```

**Python usage:**
```python
import baygraph

prog = baygraph.compile(open("network.bg").read())
ctx = baygraph.run_flow(prog, "AnalyzeRouting")

print(f"Total routing diversity: {ctx.metrics['routing_diversity']:.2f}")

graph = ctx.graphs["result"]
for group in graph.competing_groups(type="ROUTES_TO"):
    if group.entropy < 0.5:
        print(f"Warning: {group.source_node} has low routing diversity")
        winner_edge = group.winner
        print(f"  Dominant route: {winner_edge.dst_node} ({winner_edge.prob:.1%})")
```

---

### This document defines:

- Core semantics  
- Complete syntax surface  
- Engine + interop architecture  
- Extensibility model  
- Competing edges (CategoricalPosterior) design and semantics

Use it as the foundation for the initial Rust prototype (`baygraph_core`).
