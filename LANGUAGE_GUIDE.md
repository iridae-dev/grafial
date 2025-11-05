# Baygraph Language Guide

**A practical guide to the Baygraph probabilistic graph language.**

This guide documents the language as actually implemented. Every feature described here has been validated against the codebase.

---

## Table of Contents

1. [Quick Start](#quick-start)
2. [Schema Definitions](#schema-definitions)
3. [Belief Models](#belief-models)
4. [Evidence](#evidence)
5. [Rules](#rules)
6. [Flows](#flows)
7. [Metrics](#metrics)
8. [Expressions](#expressions)
9. [Built-in Functions](#built-in-functions)

---

## Quick Start

A minimal Baygraph program:

```bayscript
schema Social {
  node Person {
    score: Real
  }
  edge REL { }
}

belief_model SocialBeliefs on Social {
  node Person {
    score ~ GaussianPosterior(prior_mean=0.0, prior_precision=0.01)
  }
  edge REL {
    exist ~ BernoulliPosterior(prior=0.1, pseudo_count=2.0)
  }
}

evidence SocialEvidence on SocialBeliefs {
  observe Person["Alice"].score = 10.0
  observe edge REL(Person["Alice"], Person["Bob"]) present
}

flow Demo on SocialBeliefs {
  graph base = from_evidence SocialEvidence
  metric total_score = sum_nodes(label=Person, contrib=E[node.score])
  export base as "demo"
}
```

---

## Schema Definitions

Schemas define the structure of your graph: what node types and edge types exist, and what attributes they have.

### Syntax

```bayscript
schema SchemaName {
  node NodeType {
    attribute_name: Real
    another_attr: Real
  }
  
  edge EdgeType { }
}
```

### Node Types

Node types define entities in your graph. Each node type can have zero or more attributes:

```bayscript
node Person {
  age: Real
  salary: Real
}
```

**Attribute types:**
- `Real` - Floating-point numeric values (f64)

### Edge Types

Edge types define relationships. Edges themselves don't have attributes (only existence probabilities):

```bayscript
edge KNOWS { }
edge WORKS_FOR { }
```

**Note:** Edge attributes are not currently supported. If you need edge attributes, model them as nodes with relationships.

---

## Belief Models

Belief models define the probabilistic inference models for each schema element. They specify which posterior distribution family to use for each attribute and edge.

### Syntax

```bayscript
belief_model ModelName on SchemaName {
  node NodeType {
    attribute_name ~ PosteriorType(...)
  }
  
  edge EdgeType {
    exist ~ PosteriorType(...)
  }
}
```

### Posterior Types

#### GaussianPosterior (Normal-Normal)

For continuous numeric attributes. Assumes observations come from a Normal distribution with known precision.

```bayscript
node Person {
  age ~ GaussianPosterior(
    prior_mean = 35.0,
    prior_precision = 0.01,
    observation_precision = 1.0
  )
}
```

**Parameters:**
- `prior_mean` (default: 0.0) - Prior expected value
- `prior_precision` (default: 0.01) - Prior precision (τ = 1/σ²)
- `observation_precision` (default: 1.0) - Likelihood precision for evidence

**When to use:** Continuous measurements (age, salary, temperature, etc.)

#### BernoulliPosterior (Beta-Bernoulli)

For independent edge existence probabilities. Each edge exists or doesn't exist independently.

```bayscript
edge KNOWS {
  exist ~ BernoulliPosterior(
    prior = 0.1,
    pseudo_count = 2.0
  )
}
```

**Parameters:**
- `prior` (required) - Prior probability in [0, 1]
- `pseudo_count` (required) - Strength of prior (α = prior × pseudo_count, β = (1-prior) × pseudo_count)

**When to use:** Independent relationships (friendship, follows, etc.)

#### CategoricalPosterior (Dirichlet-Categorical)

For competing edges where exactly one destination is chosen per source. Probabilities sum to 1 within each group.

```bayscript
edge ROUTES_TO {
  exist ~ CategoricalPosterior(
    group_by = "source",
    prior = uniform,
    pseudo_count = 1.0
  )
}
```

**Parameters:**
- `group_by` (required) - Must be `"source"` or `"destination"`
- `prior` - Either `uniform` or an array `[0.2, 0.3, 0.5]` (must sum to 1)
- `pseudo_count` (required) - Strength of prior
- `categories` (optional) - Explicit list of category strings (for explicit priors only)

**When to use:** Mutually exclusive choices (routing, classification, single assignment)

**When NOT to use:** Independent relationships (use `BernoulliPosterior` instead)

**Example with explicit prior:**
```bayscript
edge ROUTES_TO {
  exist ~ CategoricalPosterior(
    group_by = "source",
    prior = [0.5, 0.3, 0.2],
    pseudo_count = 10.0,
    categories = ["Server1", "Server2", "Server3"]
  )
}
```

**Dynamic category discovery:** For uniform priors, new categories are automatically discovered when evidence is observed. For explicit priors, categories must be declared upfront.

---

## Evidence

Evidence updates belief states from observations. It's how you inject data into your Bayesian graph.

### Syntax

```bayscript
evidence EvidenceName on BeliefModelName {
  observe Person["Alice"].score = 10.0
  observe edge REL(Person["Alice"], Person["Bob"]) present
}
```

### Attribute Observations

Update node attribute values:

```bayscript
observe Person["Alice"].age = 30.0
observe Person["Bob"].salary = 75000.0
```

These update the Gaussian posterior for the attribute using Bayesian inference.

### Edge Observations

#### Independent Edges (BernoulliPosterior)

```bayscript
observe edge KNOWS(Person["Alice"], Person["Bob"]) present
observe edge KNOWS(Person["Alice"], Person["Carol"]) absent
```

**Modes:**
- `present` - Edge exists (α += 1)
- `absent` - Edge doesn't exist (β += 1)

#### Competing Edges (CategoricalPosterior)

```bayscript
observe edge ROUTES_TO(Server["S1"], Server["S2"]) chosen
observe edge ROUTES_TO(Server["S3"], Server["S5"]) chosen
observe edge ROUTES_TO(Server["S7"], Server["S8"]) unchosen
observe edge ROUTES_TO(Server["S9"], Server["S10"]) forced_choice
```

**Modes:**
- `chosen` - Source chose this destination (α_k += 1)
- `unchosen` - Source didn't choose this destination (α_j += 1/(K-1) for j≠k)
- `forced_choice` - Deterministic choice (α_k = 1e6, others = 1.0)

**Note:** Evidence mode must match the edge posterior type. You can't use `chosen` on a `BernoulliPosterior` edge.

---

## Rules

Rules define pattern-based graph transformations. They match patterns, filter with conditions, and execute actions.

### Syntax

```bayscript
rule RuleName on BeliefModelName {
  pattern
    (A:Person)-[ab:REL]->(B:Person),
    (B:Person)-[bc:REL]->(C:Person)
  
  where
    prob(ab) >= 0.9 and prob(bc) >= 0.9
  
  action {
    let v = E[A.score] / 2
    set_expectation A.score = E[A.score] - v
    set_expectation B.score = E[B.score] + v
    force_absent bc
  }
  
  mode: for_each
}
```

### Patterns

Patterns bind variables to graph elements:

```bayscript
pattern
  (A:Person)-[ab:REL]->(B:Person)
```

- `(A:Person)` - Node variable `A` of type `Person`
- `[ab:REL]` - Edge variable `ab` of type `REL`
- `(B:Person)` - Node variable `B` of type `Person`

Multiple patterns create joins:

```bayscript
pattern
  (A:Person)-[ab:REL]->(B:Person),
  (B:Person)-[bc:REL]->(C:Person)
```

This finds all paths of length 2 where intermediate node `B` is shared.

### Where Clauses

Filter matches using boolean expressions:

```bayscript
where
  prob(ab) >= 0.9
  and degree(B, min_prob=0.8) > 2
  and exists (A)-[ax:REL]->(X) where prob(ax) >= 0.9 and X != B
```

**Available operators:**
- Arithmetic: `+`, `-`, `*`, `/`
- Comparison: `<`, `<=`, `>`, `>=`, `==`, `!=`
- Logical: `and`, `or`, `not`

See [Built-in Functions](#built-in-functions) for available functions.

### Actions

Actions mutate the graph deterministically:

```bayscript
action {
  let v = E[A.score] / 2
  set_expectation A.score = E[A.score] - v
  set_expectation B.score = E[B.score] + v
  force_absent bc
}
```

**Available actions:**
- `let var = expr` - Define local variable
- `set_expectation node.attr = expr` - Update attribute mean (soft update)
- `force_absent edge_var` - Force edge to be absent (high certainty)

**Note:** `set_expectation` updates the posterior mean while preserving precision. It's a "soft" update, not a hard constraint.

### Rule Modes

- `mode: for_each` - Execute action for each matching pattern (default)
- `mode: fixpoint` - Execute until convergence (not yet implemented)

---

## Flows

Flows define pipelines of graph transformations. Each step produces a new immutable graph.

### Syntax

```bayscript
flow FlowName on BeliefModelName {
  graph base = from_evidence EvidenceName
  graph cleaned = base |> apply_rule RuleName |> prune_edges REL where prob(edge) < 0.1
  metric avg_deg = avg_degree(Person, REL, min_prob=0.8)
  export cleaned as "output"
}
```

### Graph Sources

#### From Evidence

```bayscript
graph base = from_evidence EvidenceName
```

Builds a graph from evidence observations.

#### From Graph (Cross-Flow)

```bayscript
graph base = from_graph "export_name"
```

Loads a graph exported from a previous flow.

### Transforms

Transforms are chained with `|>`:

```bayscript
graph result = 
  base
    |> apply_rule RuleName
    |> apply_ruleset { Rule1, Rule2, Rule3 }
    |> prune_edges REL where prob(edge) < 0.1
    |> snapshot "checkpoint"
```

**Available transforms:**
- `apply_rule RuleName` - Apply a single rule
- `apply_ruleset { Rule1, Rule2, ... }` - Apply multiple rules in order
- `prune_edges EdgeType where expr` - Remove edges matching condition
- `snapshot "name"` - Save graph checkpoint (for debugging or cross-flow access)

### Metrics

Metrics compute scalar values over graphs:

```bayscript
metric total_score = sum_nodes(label=Person, contrib=E[node.score])
metric avg_deg = avg_degree(Person, REL, min_prob=0.8)
```

See [Metrics](#metrics) section for details.

### Exports

```bayscript
export graph_name as "export_name"
export_metric metric_name as "export_name"
```

Export graphs and metrics for use in other flows or external systems.

### Imports

```bayscript
import_metric exported_name as local_name
```

Import metrics from previous flows.

---

## Metrics

Metrics are scalar expressions evaluated over graphs. They're the canonical way to compute global values.

### Built-in Metric Functions

#### count_nodes

Count nodes matching criteria:

```bayscript
metric active_count = count_nodes(label=Person, where=E[node.active] > 0.5)
```

**Parameters:**
- `label` (required) - Node type to count
- `where` (optional) - Filter expression

#### sum_nodes

Sum contributions from nodes:

```bayscript
metric total_risk = sum_nodes(
  label=Person,
  where=prob(node.active) > 0.7,
  contrib=E[node.risk_score]
)
```

**Parameters:**
- `label` (required) - Node type
- `where` (optional) - Filter expression
- `contrib` (required) - Expression to sum

#### fold_nodes

Sequential aggregation with ordering:

```bayscript
metric final_budget = fold_nodes(
  label=Person,
  where=E[node.active] > 0.5,
  order_by=E[node.stage] ASC,
  init=1000.0,
  step=value * E[node.multiplier]
)
```

**Parameters:**
- `label` (required) - Node type
- `where` (optional) - Filter expression
- `order_by` (optional) - Ordering expression (`ASC` or `DESC`)
- `init` (required) - Initial accumulator value
- `step` (required) - Step expression (use `value` variable for accumulator)

**Note:** `value` is a special variable available only in `fold_nodes` step expressions.

#### avg_degree

Average degree of nodes:

```bayscript
metric avg_connections = avg_degree(Person, REL, min_prob=0.8)
```

**Parameters:**
- First positional: Node type label
- Second positional: Edge type name
- `min_prob` (optional, default: 0.0) - Minimum edge probability threshold

### Metric Cross-References

Metrics can reference other metrics:

```bayscript
metric base_budget = 1000.0
metric final_budget = fold_nodes(
  label=Person,
  init=base_budget,
  step=value * E[node.multiplier]
)
```

---

## Expressions

Expressions are used in `where` clauses, actions, and metrics.

### Literals

```bayscript
10.0
3.14
true
false
```

### Variables

Pattern variables are available in where clauses and actions:

```bayscript
where prob(ab) >= 0.9 and E[A.score] > 50.0
```

In metric contexts, `node` refers to the current node being processed:

```bayscript
sum_nodes(label=Person, contrib=E[node.score])
```

### Arithmetic

```bayscript
E[A.score] + 10.0
E[A.score] * 2.0
E[A.score] / 2.0
E[A.score] - 5.0
```

### Comparisons

```bayscript
prob(ab) >= 0.9
E[A.score] < 100.0
degree(B) == 2
```

### Logical

```bayscript
prob(ab) >= 0.9 and prob(bc) >= 0.9
prob(ab) < 0.1 or prob(ab) > 0.9
not exists (A)-[ax:REL]->(X) where prob(ax) >= 0.5
```

### Field Access

Access node attributes:

```bayscript
E[A.score]
E[node.age]
```

The `E[]` operator extracts the expected value (mean) from a Gaussian posterior.

### Exists Subqueries

Check if a pattern exists:

```bayscript
exists (A)-[ax:REL]->(X) where prob(ax) >= 0.9 and X != B
```

`not exists` negates the check.

**Note:** Exists subqueries are not supported in metric expressions.

---

## Built-in Functions

### prob(edge)

Returns the posterior mean probability of an edge existing.

```bayscript
prob(ab)  // Returns E[p] = α / (α + β) for BernoulliPosterior
          // Returns E[π_k] = α_k / Σ_j α_j for CategoricalPosterior
```

**Arguments:**
- `edge` - Edge variable from pattern

### degree(node, min_prob=0.0)

Counts outgoing edges with probability >= threshold.

```bayscript
degree(B, min_prob=0.8)
```

**Arguments:**
- `node` - Node variable from pattern
- `min_prob` (optional) - Minimum probability threshold

**Returns:** Number of edges (f64 for use in expressions)

### winner(node, edge_type, epsilon=0.01)

For competing edges, returns the destination node with maximum probability, or null if tied.

```bayscript
where winner(A, ROUTES_TO, epsilon=0.01) == B
```

**Arguments:**
- `node` - Node variable from pattern
- `edge_type` - Edge type identifier
- `epsilon` (optional) - Tolerance for tie detection

**Returns:** Node ID as f64 (or -1.0 if None/null)

**Note:** Only valid for competing edges (CategoricalPosterior).

### entropy(node, edge_type)

Computes Shannon entropy of competing edge probabilities.

```bayscript
where entropy(A, ROUTES_TO) > 1.5
```

**Arguments:**
- `node` - Node variable from pattern
- `edge_type` - Edge type identifier

**Returns:** Entropy in nats (range: [0, log(K)] where K = number of categories)

**Note:** Only valid for competing edges (CategoricalPosterior).

### prob_vector(node, edge_type)

Returns the full probability vector for competing edges.

```bayscript
// Returns [E[π_1], E[π_2], ..., E[π_K]]
```

**Note:** Primarily for metrics and export, not pattern matching.

### E[node.attr]

Extracts the expected value (mean) from a Gaussian posterior attribute.

```bayscript
E[A.score]
E[node.age]
```

**Arguments:**
- `node` - Node variable or `node` in metric contexts
- `attr` - Attribute name

**Returns:** Posterior mean (f64)

---

## Complete Example

```bayscript
schema Network {
  node Server {
    load: Real
  }
  edge ROUTES_TO { }
}

belief_model NetworkBeliefs on Network {
  node Server {
    load ~ GaussianPosterior(prior_mean=0.5, prior_precision=1.0)
  }
  edge ROUTES_TO {
    exist ~ CategoricalPosterior(
      group_by="source",
      prior=uniform,
      pseudo_count=1.0
    )
  }
}

evidence NetworkEvidence on NetworkBeliefs {
  observe Server["S1"].load = 0.8
  observe Server["S2"].load = 0.3
  observe edge ROUTES_TO(Server["S1"], Server["S2"]) chosen
  observe edge ROUTES_TO(Server["S1"], Server["S3"]) chosen
}

rule BalanceLoad on NetworkBeliefs {
  pattern
    (A:Server)-[ab:ROUTES_TO]->(B:Server)
  
  where
    E[A.load] > 0.7
    and winner(A, ROUTES_TO) == B
    and entropy(A, ROUTES_TO) < 0.5
  
  action {
    set_expectation B.load = E[B.load] + 0.1
  }
  
  mode: for_each
}

flow Pipeline on NetworkBeliefs {
  graph base = from_evidence NetworkEvidence
  graph balanced = base |> apply_rule BalanceLoad
  metric avg_load = sum_nodes(label=Server, contrib=E[node.load]) / count_nodes(label=Server)
  export balanced as "final"
}
```

---

## Validation Notes

All features described in this guide have been validated against the implementation:

- ✅ Schema definitions (nodes, edges, attributes)
- ✅ Belief models (GaussianPosterior, BernoulliPosterior, CategoricalPosterior)
- ✅ Evidence modes (present, absent, chosen, unchosen, forced_choice)
- ✅ Rules (patterns, where clauses, actions, modes)
- ✅ Flow transforms (from_evidence, from_graph, apply_rule, apply_ruleset, snapshot, prune_edges)
- ✅ Metrics (count_nodes, sum_nodes, fold_nodes, avg_degree)
- ✅ Expression functions (prob, degree, winner, entropy, E[])
- ✅ Exists/not exists subqueries

For implementation details, see `baygraph_design.md`. For performance considerations, see `performance_ideas.md`.

