# Competing Edges — Design Addendum for Phase 5.5

**Status:** Design locked for Phase 7+ implementation
**Version:** 1.0
**Date:** 2025-11-05
**Parent Document:** baygraph_design.md

---

## 1. Problem Statement

When modeling scenarios where a node must choose **exactly one** alternative from a set of options (mutually exclusive edges), independent Bernoulli posteriors are inadequate:

- **Problem 1**: Independent edges can sum to probabilities > 1 or < 1
- **Problem 2**: Renormalizing Bernoulli posteriors to sum to 1 is not Bayesian (violates conjugacy)
- **Problem 3**: Evidence about one edge doesn't inform beliefs about competing alternatives

**Examples requiring competing edges:**
- Message routing: one outgoing path per packet
- Resource allocation: one task assignment per worker
- State machines: one transition per state
- Tournament brackets: one winner per match

**Existing design note** (baygraph_design.md:144-158, 310):
> For mutually exclusive choices among K alternatives, use a `CategoricalPosterior` with a Dirichlet prior over category probabilities π ∈ Δ^{K−1}.

This document provides **concrete syntax, semantics, and implementation guidance** for competing edges.

---

## 2. Conceptual Model

### 2.1 Mathematical Foundation

**Independent edges** (current):
- Each edge (u, v) has independent belief: p ~ Beta(α, β)
- Edges from the same source can have Σ_v E[p_uv] ≠ 1

**Competing edges** (new):
- Edges from source u form a group with shared belief: π ~ Dirichlet(α_1, ..., α_K)
- Constraint: Σ_k π_k = 1 (exactly one choice)
- Evidence about choosing edge (u, v_k) increments α_k

### 2.2 When to Use Each Model

| Scenario | Model | Rationale |
|----------|-------|-----------|
| Social network "follows" | Independent | Users can follow multiple people |
| Router forwarding table | Competing | Packet goes to exactly one next hop |
| Budget allocation (strict) | Competing | Budget must sum to 100% |
| Budget allocation (flexible) | Independent | Overspend/underspend allowed |
| Probabilistic workflow | Competing | One next step is chosen |
| Multi-label classification | Independent | Multiple labels can apply |

---

## 3. Syntax Design

### 3.1 Belief Model Declaration

**Independent edges** (unchanged):
```bayscript
belief_model MyBeliefs on MySchema {
  edge FOLLOWS {
    exist ~ BernoulliPosterior(prior = 0.3, pseudo_count = 2.0)
  }
}
```

**Competing edges** (new):
```bayscript
belief_model NetworkBeliefs on NetworkSchema {
  edge ROUTES_TO {
    exist ~ CategoricalPosterior(
      group_by = "source",           // Required: partition by source or destination
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

### 3.2 Grammar Additions

```pest
// In posterior_type rule:
posterior_type = {
  gaussian_posterior |
  bernoulli_posterior |
  categorical_posterior    // NEW
}

categorical_posterior = {
  "CategoricalPosterior" ~ "(" ~
    categorical_param ~ ("," ~ categorical_param)* ~
  ")"
}

categorical_param = {
  ("group_by" ~ "=" ~ string_lit) |
  ("prior" ~ "=" ~ (prior_array | "uniform")) |
  ("pseudo_count" ~ "=" ~ float_lit) |
  ("categories" ~ "=" ~ "[" ~ string_lit ~ ("," ~ string_lit)* ~ "]")
}

prior_array = { "[" ~ float_lit ~ ("," ~ float_lit)* ~ "]" }
```

### 3.3 Validation Rules

At parse/compile time, enforce:

1. **Required parameters:**
   - `group_by` must be "source" or "destination"

2. **Mutual exclusivity:**
   - If `prior = uniform`, must specify `pseudo_count`
   - If `prior = [...]` (array), pseudo_count is sum of array

3. **Category consistency:**
   - If `categories = [...]` specified, must match schema declared destinations
   - All values in `prior` array must be > 0 (proper Dirichlet)

4. **Per-edge uniqueness:**
   - An edge type is either independent OR competing, never both

Example validation error:
```
Error: Edge type 'ROUTES_TO' is declared as CategoricalPosterior but missing required parameter 'group_by'
  --> model.bg:12:5
   |
12 |     exist ~ CategoricalPosterior(prior = uniform, pseudo_count = 3.0)
   |             ^^^^^^^^^^^^^^^^^^^^^ requires group_by = "source" or "destination"
```

---

## 4. Evidence Syntax and Semantics

### 4.1 New Evidence Keywords

Extend the `evidence_mode` grammar:

```pest
evidence_mode = {
  "present" | "absent" |           // Independent edges only
  "chosen" | "unchosen" |          // Competing edges only
  "forced_choice"                   // Competing edges (deterministic)
}
```

### 4.2 Semantic Interpretation

| Edge Type | Statement | Posterior Update | Notes |
|-----------|-----------|------------------|-------|
| Independent | `observe edge E(A,B) present` | Beta: α += 1 | "Edge exists" |
| Independent | `observe edge E(A,B) absent` | Beta: β += 1 | "Edge doesn't exist" |
| Competing | `observe edge E(A,B) chosen` | Dirichlet: α_B += 1 | "A chose B from all options" |
| Competing | `observe edge E(A,B) unchosen` | Dirichlet: α_k += 1/(K-1) for k≠B | "A didn't choose B" (rare usage) |
| Competing | `observe edge E(A,B) forced_choice` | Dirichlet: α_B = 1e6, others = 1.0 | "A deterministically chose B" |

### 4.3 Example Evidence Blocks

```bayscript
evidence NetworkEvidence on NetworkBeliefs {
  // Independent edges (unchanged)
  observe edge FOLLOWS(Person["Alice"], Person["Bob"]) present
  observe edge FOLLOWS(Person["Alice"], Person["Carol"]) absent

  // Competing edges: explicit choices
  observe edge ROUTES_TO(Server["S1"], Server["S2"]) chosen
  observe edge ROUTES_TO(Server["S3"], Server["S5"]) chosen

  // Negative evidence (rarely used)
  observe edge ROUTES_TO(Server["S7"], Server["S8"]) unchosen

  // Deterministic choice
  observe edge ROUTES_TO(Server["S9"], Server["S10"]) forced_choice
}
```

### 4.4 Type Checking Evidence

At runtime (or compile-time if evidence is static), validate evidence mode matches edge posterior type:

```rust
fn validate_evidence_mode(
    edge_type: &EdgeType,
    mode: EvidenceMode,
    belief_model: &BeliefModel,
) -> Result<(), ExecError> {
    let posterior = belief_model.get_edge_posterior(edge_type)?;

    match (posterior, mode) {
        (PosteriorFamily::Independent, EvidenceMode::Present | EvidenceMode::Absent) => Ok(()),
        (PosteriorFamily::Competing, EvidenceMode::Chosen | EvidenceMode::Unchosen | EvidenceMode::ForcedChoice) => Ok(()),

        (PosteriorFamily::Independent, EvidenceMode::Chosen | EvidenceMode::Unchosen | EvidenceMode::ForcedChoice) => {
            Err(ExecError::Type(format!(
                "Edge '{}' is independent; use 'present' or 'absent', not '{:?}'",
                edge_type.name, mode
            )))
        }

        (PosteriorFamily::Competing, EvidenceMode::Present | EvidenceMode::Absent) => {
            Err(ExecError::Type(format!(
                "Edge '{}' has competing posterior; use 'chosen', 'unchosen', or 'forced_choice', not '{:?}'",
                edge_type.name, mode
            )))
        }
    }
}
```

---

## 5. Query Semantics

### 5.1 Core Query Functions

#### `prob(edge)` — Edge Probability

**Signature:** `prob(edge_var) -> f64`

**Semantics:**
- **Independent:** Returns E[p] = α / (α + β) from Beta posterior
- **Competing:** Returns E[π_k] = α_k / Σ_j α_j from Dirichlet posterior

**Always returns:** Scalar in [0, 1]

**Example:**
```bayscript
rule HighProbabilityRoute on NetworkBeliefs {
  pattern
    (A:Server)-[ab:ROUTES_TO]->(B:Server)

  where
    prob(ab) >= 0.6
    // Matches if E[π_B] >= 0.6 (B is likely winner)

  action {
    // ...
  }
}
```

**Implementation:**
```rust
impl BeliefGraph {
    pub fn prob(&self, edge_id: EdgeId) -> f64 {
        let edge = &self.edges[edge_id];
        match &edge.posterior {
            EdgePosterior::Independent(beta) => {
                beta.mean()  // α / (α + β)
            }
            EdgePosterior::Competing(group_ref) => {
                let group = &self.competing_groups[group_ref.group_id];
                let cat_idx = group_ref.category_index;
                group.dirichlet.mean_at(cat_idx)  // α_k / Σ α_j
            }
        }
    }
}
```

#### `degree(node, edge_type, min_prob)` — Degree Count

**Signature:** `degree(node, outgoing=EdgeType, min_prob=threshold) -> usize`

**Semantics:**
- **Independent:** Count edges with E[p] ≥ threshold
- **Competing:** Count categories with E[π_k] ≥ threshold

**Key difference for competing edges:**
Since Σ π_k = 1, at most one category can exceed 0.5 (unless K=2 and both ≈ 0.5).

**Example:**
```bayscript
rule SingleRoute on NetworkBeliefs {
  pattern
    (A:Server)-[ab:ROUTES_TO]->(B:Server)

  where
    degree(A, outgoing=ROUTES_TO, min_prob=0.8) == 1
    // True if exactly one destination has E[π_k] >= 0.8
    // (i.e., confident single winner)

  action {
    // Process confident route
  }
}
```

**Implementation:**
```rust
pub fn degree(
    &self,
    node_id: NodeId,
    edge_type: EdgeTypeId,
    direction: Direction,
    min_prob: f64,
) -> usize {
    let edges = self.adjacency(node_id, edge_type, direction);

    // Check if these edges form a competing group
    if let Some(group_id) = self.get_competing_group(node_id, edge_type, direction) {
        // Competing: count categories above threshold
        let group = &self.competing_groups[group_id];
        group.dirichlet.mean_probs()
            .iter()
            .filter(|&&p| p >= min_prob)
            .count()
    } else {
        // Independent: count individual edges above threshold
        edges.iter()
            .filter(|&&eid| self.prob(eid) >= min_prob)
            .count()
    }
}
```

#### `exists` — Subquery Matching

**Signature:** `exists (pattern) where (condition)`

**Semantics:** Same as independent edges; evaluates `prob(edge)` for each match.

**Note:** For competing edges, `exists ... where prob(edge) >= 0.5` is typically true (winner has π_k near 1 if evidence strong, or 1/K if weak). This is correct but less discriminative than for independent edges.

**Example:**
```bayscript
rule HasAlternative on NetworkBeliefs {
  pattern
    (A:Server)-[ab:ROUTES_TO]->(B:Server)

  where
    prob(ab) >= 0.6
    and exists (A)-[ax:ROUTES_TO]->(X)
      where prob(ax) >= 0.2 and X != B
    // Checks if there's a backup route with non-negligible probability

  action {
    // ...
  }
}
```

### 5.2 New Query Functions (Competing-Specific)

Add these intrinsic functions to the engine registry:

#### `winner(node, edge_type)` — Dominant Choice

**Signature:** `winner(node, edge_type, direction="outgoing", epsilon=0.01) -> EdgeId?`

**Returns:** Edge with max E[π_k], or `null` if tied within `epsilon`

**Example:**
```bayscript
rule CheckWinner on NetworkBeliefs {
  pattern
    (A:Server)

  where
    winner(A, outgoing=ROUTES_TO) == B
    // B is the clear winner (max E[π_k] with no ties)

  action {
    // ...
  }
}
```

**Implementation:**
```rust
pub fn winner(
    &self,
    node_id: NodeId,
    edge_type: EdgeTypeId,
    direction: Direction,
    epsilon: f64,
) -> Option<EdgeId> {
    let group_id = self.get_competing_group(node_id, edge_type, direction)?;
    let group = &self.competing_groups[group_id];

    let probs = group.dirichlet.mean_probs();
    let (max_idx, &max_prob) = probs.iter()
        .enumerate()
        .max_by(|(_, a), (_, b)| a.partial_cmp(b).unwrap())?;

    // Check for ties within epsilon
    let num_ties = probs.iter()
        .filter(|&&p| (p - max_prob).abs() < epsilon)
        .count();

    if num_ties > 1 {
        None  // Ambiguous: multiple categories within epsilon of max
    } else {
        Some(group.categories[max_idx])
    }
}
```

#### `entropy(node, edge_type)` — Uncertainty Measure

**Signature:** `entropy(node, edge_type, direction="outgoing") -> f64`

**Returns:** Shannon entropy H(π) = -Σ π_k log(π_k) in nats (or bits if using log2)

**Range:** [0, log(K)] where K = number of categories
- H = 0: Deterministic (one π_k ≈ 1)
- H = log(K): Uniform (all π_k ≈ 1/K)

**Example:**
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

**Implementation:**
```rust
pub fn entropy(
    &self,
    node_id: NodeId,
    edge_type: EdgeTypeId,
    direction: Direction,
) -> f64 {
    let group_id = self.get_competing_group(node_id, edge_type, direction)
        .expect("entropy() only valid for competing edges");
    let group = &self.competing_groups[group_id];

    let probs = group.dirichlet.mean_probs();

    // Shannon entropy: -Σ p_k log(p_k)
    probs.iter()
        .filter(|&&p| p > 1e-10)  // Avoid log(0)
        .map(|&p| -p * p.ln())
        .sum()
}
```

#### `prob_vector(node, edge_type)` — Full Distribution

**Signature:** `prob_vector(node, edge_type, direction="outgoing") -> Vec<f64>`

**Returns:** [E[π_1], E[π_2], ..., E[π_K]] for all categories

**Use case:** Export full distribution for external analysis or plotting

**Note:** This returns a vector, not a scalar. Primarily for metrics and export, not pattern matching.

**Example (in metric context):**
```bayscript
flow AnalyzeRoutes on NetworkBeliefs {
  graph g = from_evidence E

  metric route_uncertainty =
    sum_nodes(
      label = Server,
      contrib = entropy(node, outgoing=ROUTES_TO)
    )

  export_metric route_uncertainty as "total_uncertainty"
}
```

---

## 6. Open Design Questions — Resolved

### Q1: Static vs. Dynamic Group Discovery

**Decision:** Static declaration by default; optional dynamic discovery via schema flag.

**Rationale:** Type safety, early validation, aligns with schema-first design.

**Implementation:**
```bayscript
schema Network {
  edge ROUTES_TO {
    allow_dynamic_categories = false  // Default: strict validation
  }
}
```

If `allow_dynamic_categories = true`, new destinations observed in evidence are added dynamically with default prior.

---

### Q2: Sources with No Observed Edges

**Decision:** Return prior mean E[π_k] = α_k / Σ α_j until first observation.

**Rationale:** Consistent with Bayesian semantics; prior is a valid belief.

**Example:** If prior = [2, 3, 5] with pseudo_count = 10:
- Before any evidence: E[π] = [0.2, 0.3, 0.5]
- After observing category 2: α = [2, 4, 5], E[π] = [0.18, 0.36, 0.45]

---

### Q3: Partial Competition (Soft Constraints)

**Decision:** Not supported in Phase 7; defer to Phase 8+ as separate model.

**Rationale:** Dirichlet–Categorical enforces Σ π_k = 1 strictly. Soft constraints require different parameterization (e.g., logit-normal).

**Workaround for v1:** Use independent Bernoulli with validation metrics.

---

### Q4: Evidence for Non-Existent Categories

**Decision:** Error in strict mode (default); allow in dynamic mode.

**Strict mode (default):**
```rust
Err(ExecError::Exec(format!(
    "Node {:?} is not a declared category for edge {} from {:?}",
    dst, edge_type, src
)))
```

**Dynamic mode:**
```rust
// Add new category with default prior
let alpha_default = belief_model.default_category_prior(edge_type);
group.add_category(dst, alpha_default);
```

---

### Q5: Visualization and User Feedback

**Decision:** Expose competing groups as hierarchical structures in Python API and JSON export.

**Python API:**
```python
for group in graph.competing_groups(type="ROUTES_TO"):
    print(f"Source: {group.source_node}")
    print(f"  Entropy: {group.entropy:.3f}")
    for cat in group.categories:
        print(f"    -> {cat.dst_node}: {cat.prob:.3f} (α={cat.alpha:.1f})")
```

**JSON Export:**
```json
{
  "competing_groups": [
    {
      "source_node": "Server_A",
      "edge_type": "ROUTES_TO",
      "entropy": 0.82,
      "winner": "Server_B",
      "categories": [
        {"dst": "Server_B", "prob": 0.55, "alpha": 8.5},
        {"dst": "Server_C", "prob": 0.35, "alpha": 5.5}
      ]
    }
  ]
}
```

---

## 7. Implementation Phases

### Phase 5.5 (Current): Design Only ✓

**Deliverables:**
- This document (competing_edges_design.md)
- Locked syntax and semantics
- Resolved open questions
- No code changes to engine

**Status:** Complete

---

### Phase 7: Competing Edges Implementation

**Prerequisites:** Phase 6 (Python bindings) complete

**Tasks:**

1. **Data structures** (engine/posterior.rs):
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

2. **Parser extensions** (frontend/parser.rs):
   - Add `categorical_posterior` rule to pest grammar
   - Parse `CategoricalPosterior(...)` into AST
   - Validate required parameters

3. **Evidence handling** (engine/evidence.rs):
   - Implement `observe_chosen`, `observe_unchosen`, `observe_forced_choice`
   - Type-check evidence mode vs. posterior family
   - Handle dynamic category discovery if enabled

4. **Query functions** (engine/queries.rs):
   - Update `prob()` to handle competing edges
   - Update `degree()` to count categories
   - Implement `winner()`, `entropy()`, `prob_vector()`

5. **Tests** (tests/competing_edges_tests.rs):
   - Basic Dirichlet update correctness
   - Evidence type checking (reject wrong modes)
   - Query semantics (prob, degree, winner)
   - Pattern matching with competing edges
   - Cross-flow metric transfer with competing edges

6. **Python API** (bindings/python.rs):
   - Expose `graph.competing_groups(type=...)` iterator
   - Add `group.entropy`, `group.winner` properties
   - JSON export for competing groups

**Exit criteria:**
- All tests pass
- Example flows using competing edges run successfully
- Python API documented with examples

---

## 8. Example: Complete Use Case

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

## 9. Migration Guide (for Phase 7+)

When upgrading existing models to use competing edges:

### Step 1: Identify Competing Patterns

Review edge types where:
- Probabilities from a source should sum to ~1.0
- Edges represent mutually exclusive choices
- Evidence about one edge should inform beliefs about alternatives

### Step 2: Update Belief Model

**Before:**
```bayscript
belief_model Old on Schema {
  edge ASSIGNS {
    exist ~ BernoulliPosterior(prior = 0.3, pseudo_count = 2.0)
  }
}
```

**After:**
```bayscript
belief_model New on Schema {
  edge ASSIGNS {
    exist ~ CategoricalPosterior(
      group_by = "source",
      prior = uniform,
      pseudo_count = 5.0  // Adjust based on prior strength
    )
  }
}
```

### Step 3: Update Evidence

**Before:**
```bayscript
evidence E on Old {
  observe edge ASSIGNS(Worker["W1"], Task["T1"]) present
}
```

**After:**
```bayscript
evidence E on New {
  observe edge ASSIGNS(Worker["W1"], Task["T1"]) chosen
}
```

### Step 4: Update Rules (if needed)

Most rules work unchanged, but consider:
- `degree(node, ...)` now counts categories, not edges
- Add `winner(node, edge_type)` checks for deterministic routing
- Use `entropy(node, edge_type)` to detect high-uncertainty cases

### Step 5: Test and Validate

Run existing flows and verify:
- Posterior probabilities sum to ~1.0 per source
- Evidence updates propagate correctly
- Metrics reflect new semantics

---

## 10. References

- **Parent design:** baygraph_design.md (sections 3.2.1, 3.3, 3.4)
- **Dirichlet–Categorical conjugacy:** Murphy, "Machine Learning: A Probabilistic Perspective", §3.4
- **Design principles:** baygraph_design.md:636-641 (immutability, minimalism, extensibility)

---

## Appendix A: Posterior Comparison Table

| Aspect | Independent (Beta–Bernoulli) | Competing (Dirichlet–Categorical) |
|--------|------------------------------|-----------------------------------|
| **Belief representation** | p ~ Beta(α, β) per edge | π ~ Dirichlet(α_1, ..., α_K) per group |
| **Constraint** | 0 ≤ p ≤ 1 | Σ π_k = 1, π_k ≥ 0 |
| **Evidence** | "present" / "absent" | "chosen" / "unchosen" / "forced_choice" |
| **Update (positive)** | α += 1 | α_k += 1 |
| **Update (negative)** | β += 1 | Distribute to other categories |
| **Query: prob(edge)** | E[p] = α/(α+β) | E[π_k] = α_k / Σ α_j |
| **Query: degree(node)** | Count edges with p ≥ t | Count categories with π_k ≥ t |
| **Forcing** | α=1e6, β=1 or vice versa | α_k=1e6, others=1 |
| **Typical use** | Social networks, multi-label | Routing, allocation, state machines |

---

## Appendix B: Dirichlet Posterior Formulas

**Prior:** π ~ Dirichlet(α_1, α_2, ..., α_K)

**Posterior mean:** E[π_k] = α_k / (Σ_j α_j)

**Posterior variance:** Var[π_k] = (α_k (α_0 - α_k)) / (α_0² (α_0 + 1))
where α_0 = Σ_j α_j

**Shannon entropy (of mean):** H = -Σ_k E[π_k] log(E[π_k])

**Conjugate update:** Observe category k ⇒ α_k := α_k + 1

**Force choice k:** α_k := 1e6, α_j := 1.0 for j ≠ k
(numerically stable approximation to degenerate distribution)

**Stability:** Enforce α_k ≥ 0.01 for all k to prevent improper priors

---

**End of Design Addendum**
