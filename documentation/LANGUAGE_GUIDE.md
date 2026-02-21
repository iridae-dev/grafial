# Grafial Language Guide

This guide documents the Grafial language as currently implemented.

## 1. Program Structure

A program contains these declaration kinds:

- `schema`
- `belief_model`
- `evidence`
- `rule`
- `flow`

Example:

```grafial
schema Social {
  node Person { score: Real }
  edge REL { }
}

belief_model SocialBeliefs on Social {
  node Person { score ~ Gaussian(mean=0.0, precision=0.1) }
  edge REL { exist ~ Bernoulli(prior=0.5, weight=2.0) }
}

evidence SocialEvidence on SocialBeliefs {
  Person { "Alice" { score: 1.0 }, "Bob" { score: 2.0 } }
  REL(Person -> Person) { "Alice" -> "Bob" }
}

flow Demo on SocialBeliefs {
  graph base = from_evidence SocialEvidence
  metric avg_score = nodes(Person) |> avg(by=E[node.score])
  export base as "demo"
}
```

## 2. Syntax Basics

- Blocks use `{ ... }`.
- Statement separators are optional (`;` is accepted in many places).
- Comments: `// ...` and `/* ... */`.
- Strings: double-quoted only.
- Booleans: `true`, `false`.
- Logical operators: `and`, `or`, `not`.

## 3. Schemas

Schemas define node/edge types and node attributes.

```grafial
schema MySchema {
  node Entity {
    value: Real
    weight: Real
  }

  edge CONNECTS { }
}
```

Notes:

- Edge attributes are not supported.
- Priors are not declared in `schema`; they are declared in `belief_model`.

## 4. Belief Models

Belief models assign posterior families to schema fields.

```grafial
belief_model MyBeliefs on MySchema {
  node Entity {
    value ~ Gaussian(mean=0.0, precision=0.01)
    weight ~ GaussianPosterior(prior_mean=1.0, prior_precision=1.0)
  }

  edge CONNECTS {
    exist ~ Bernoulli(prior=0.3, weight=2.0)
  }
}
```

Supported posterior names:

- `Gaussian` or `GaussianPosterior`
- `Bernoulli` or `BernoulliPosterior`
- `Categorical` or `CategoricalPosterior`

Parameter aliases accepted by parser:

- Gaussian: `mean` -> `prior_mean`, `precision` -> `prior_precision`
- Bernoulli: `weight` -> `pseudo_count`

### 4.1 Categorical Edges

```grafial
belief_model RoutingBeliefs on Routing {
  edge ROUTES_TO {
    exist ~ Categorical(group_by=source, prior=uniform, pseudo_count=1.0)
  }
}
```

Notes:

- Parser accepts `group_by=source` or `group_by="source"`.
- Validation accepts `group_by="source"` and `group_by="destination"`.
- Runtime currently supports only `group_by=source` (quoted or unquoted forms both parse to the same value).
- Dynamic category discovery is supported for `prior=uniform`.
- Dynamic category discovery is not supported for explicit `prior=[...]`.

## 5. Evidence

Evidence applies observations against a belief model.

### 5.1 Grouped Evidence (recommended)

```grafial
evidence Ev on MyBeliefs {
  Entity {
    "A" { value: 1.0 (precision=10.0), weight: 2.0 }
    "B" { value: 0.2 }
  }

  CONNECTS(Entity -> Entity) {
    "A" -> "B";
    "B" -/> "A"
  }
}
```

For multi-edge grouped blocks, separate entries with `;` or `,`.

### 5.2 Explicit Observe Syntax

```grafial
evidence Ev2 on MyBeliefs {
  observe Entity["A"].value = 1.0
  observe Entity("B").value = 0.5 (precision=5.0)

  observe edge CONNECTS(Entity["A"], Entity["B"]) present
  observe edge CONNECTS(Entity["B"], Entity["A"]) absent
}
```

Supported edge modes:

- `present`
- `absent`
- `chosen`
- `unchosen`
- `forced_choice`

### 5.3 Categorical Choice Verbs

```grafial
evidence RoutingEv on RoutingBeliefs {
  choose edge ROUTES_TO(Router["R1"], Router["R2"])
  unchoose edge ROUTES_TO(Router["R1"], Router["R3"])
}
```

## 6. Rules

Rules match graph patterns, optionally filter in `where`, then run actions.

### 6.1 Full Form

```grafial
rule Transfer on SocialBeliefs {
  pattern
    (A:Person)-[ab:REL]->(B:Person)

  where
    prob(ab) >= 0.8 and E[A.score] > E[B.score]

  action {
    let delta = (E[A.score] - E[B.score]) * 0.1
    B.score ~= (E[B.score] + delta) precision=0.2 count=2
  }

  mode: for_each
}
```

### 6.2 Sugar Forms

Pattern -> action:

```grafial
rule FastTransfer on SocialBeliefs {
  (A:Person)-[ab:REL]->(B:Person)
  where prob(ab) >= 0.8 => {
    non_bayesian_nudge B.score to (E[B.score] + 0.1) variance=preserve
  }
}
```

Node-only form:

```grafial
rule BoostLow on SocialBeliefs {
  for (P:Person) where E[P.score] < 0.0 => {
    P.score ~= 0.0 precision=0.1
  }
}
```

### 6.3 Rule Modes (Current Runtime Behavior)

- Parser accepts `mode: <ident>` on rules.
- The standalone rule runner supports `for_each` (default) and `fixpoint`.
- Flow transforms (`apply_rule`, `apply_ruleset`) currently execute rules with `for_each` semantics regardless of `mode:`.

## 7. Actions

Supported action statements:

- `let x = expr`
- `non_bayesian_nudge A.attr to expr variance=...`
- `A.attr ~= expr precision=... count=...`
- `delete e confidence=low|high`
- `suppress e weight=...`

Legacy action keywords are no longer accepted:

- `set_expectation A.attr = expr`
- `force_absent e`

Use `grafial <file> --fix-style` (or the LSP quick fix) to rewrite legacy files automatically.

Soft-update and edge-action argument forms:

- Canonical inline args:
  - `A.x ~= 1.0 precision=0.2 count=3`
  - `delete e confidence=high`
  - `suppress e weight=10`
- Compatibility args (accepted):
  - `A.x ~= 1.0 (precision=0.2, count=3)`
  - `delete e (confidence=high)`
  - `suppress e (weight=10)`

Semantics summary:

- `non_bayesian_nudge`:
  - `preserve`: set mean only; keep precision
  - `increase(f)`: multiply precision by `f` (default `0.5`)
  - `decrease(f)`: multiply precision by `f` (default `2.0`)
- `~=`: Normal-Normal soft update with optional `count` multiplier.
- `delete`: independent edges only; near-zero prior with confidence scaling.
- `suppress`: independent edges only; moderate prior against existence.

## 8. Expressions

Expression forms:

- Numbers, booleans, vars
- Arithmetic: `+ - * /`
- Comparison: `== != < <= > >=`
- Logical: `and or not`
- Field access: `A.score`
- Function calls: `name(...)`
- Expectation bracket form: `E[A.score]`
- Exists subqueries:
  - `exists (A:Person)-[ab:REL]->(B:Person) where prob(ab) > 0.5`
  - `not exists ...`

## 9. Built-in Functions

### 9.1 Rule/Where Context

Supported:

- `E[A.attr]`
- `prob(edge_var)`
- `prob(A.attr > B.attr)` (supports `< <= > >=` comparisons)
- `prob_correlated(A.attr > B.attr, rho=...)` (correlation-aware comparison probability)
- `credible(event, p=0.95, rho=0.0)` (returns true when posterior event probability is at least `p`)
- `degree(A, min_prob=0.5)`
- `winner(A, ROUTES_TO, epsilon=0.01)`
- `entropy(A, ROUTES_TO)`
- `variance(A.attr)`
- `stddev(A.attr)`
- `ci_lo(A.attr, p)`
- `ci_hi(A.attr, p)`
- `effective_n(A.attr)`
- `quantile(A.attr, p)`

### 9.2 Metric Node-Expression Context

Inside metric filters/contrib/step expressions (`node` is the bound row variable), supported:

- `E[node.attr]`
- `degree(node, min_prob=...)`
- `entropy(node, EDGE_TYPE)`
- `variance(node.attr)`
- `stddev(node.attr)`
- `ci_lo(node.attr, p)`
- `ci_hi(node.attr, p)`
- `effective_n(node.attr)`
- `quantile(node.attr, p)`

## 10. Validation Rules You Should Expect

- Bare field access is rejected in rule and metric contexts.
  - Use `E[Node.attr]` or `prob(...)`.
- `prob(...)` in rule `where` must be either:
  - edge variable, or
  - supported comparison form.
- `prob_correlated(...)` in rule `where` must be:
  - a supported comparison form with optional `rho=...` argument (`rho` in `[-1, 1]`).
- `credible(...)` in rule `where` must be:
  - an edge variable event, or
  - a supported comparison form, with optional `p` and `rho` named args.
- `prune_edges ... where ...` predicates are restricted to `prob(edge)`-style checks.
- Metric expressions reject `exists` subqueries.
- Builder `order_by(...)` is only valid with `fold(...)`.

## 11. Flows

Flows define graph pipelines and metric/export surfaces.

```grafial
flow Analysis on SocialBeliefs {
  graph base = from_evidence SocialEvidence

  graph cleaned = base
    |> apply_rule Transfer
    |> prune_edges REL where prob(edge) < 0.1
    |> snapshot "after_clean"

  metric avg_score = nodes(Person) |> avg(by=E[node.score])
  export cleaned as "output"
  export_metric avg_score as "avg_output"
}
```

Graph expression forms:

- `from_evidence EvidenceName`
- `from_graph "alias"`
- `existing_graph |> transform |> transform ...`

Transforms:

- `apply_rule RuleName`
- `apply_ruleset { RuleA, RuleB, ... }`
- `snapshot "name"`
- `prune_edges EdgeType where expr`

Metric sharing between flows:

```grafial
flow F1 on M {
  graph g = from_evidence Ev
  metric m = nodes(N) |> count()
  export_metric m as "saved_m"
}

flow F2 on M {
  import_metric saved_m as budget
  graph g = from_evidence Ev
  metric out = fold_nodes(label=N, init=budget, step=value)
}
```

## 12. Metrics

Core metric functions:

- `count_nodes(label=..., where=...)`
- `sum_nodes(label=..., where=..., contrib=...)`
- `fold_nodes(label=..., where=..., order_by=..., init=..., step=...)`
- `avg_degree(label=..., edge_type=..., min_prob=...)`

Metric builder pipeline (desugars to core functions):

- `nodes(Label) |> where(expr) |> count()`
- `nodes(Label) |> where(expr) |> sum(by=expr)`
- `nodes(Label) |> where(expr) |> avg(by=expr)`
- `nodes(Label) |> where(expr) |> order_by(expr) |> fold(init=expr, step=expr)`

Notes:

- In `fold(...)`, `value` is the accumulator variable.
- Metrics can reference previously computed metrics and imported metrics by name.

## 13. Determinism and Execution Model

- Graph transforms produce immutable snapshots (copy-on-write under the hood).
- Flow planning and execution order are deterministic.
- Rule matching and metric evaluation are deterministic by stable ID order.
- Runtime flow outputs include `intervention_audit` events for `apply_rule`/`apply_ruleset`
  transforms (rule name, match count, action count) for reproducibility/traceability hooks.

## 14. Current Non-Goals / Not Implemented in Syntax

These are not part of the current parser/runtime surface:

- Inline priors directly in `schema` declarations.
- Fluent metric syntax like `Person.where(...).sum(...)`.
- Direct flow terminal syntax like bare `from_evidence ... |> ... |> export ...` without `graph` statements.
- `note "..."` flow annotations.
- Single-quoted strings.
- Numeric literals with `_` separators.

## 15. Tooling for Canonical Style

CLI style tooling:

```bash
grafial file.grafial --lint-style
grafial file.grafial --fix-style
```

Current canonical-style rewrites cover:

- legacy action keywords:
  - `set_expectation` -> `non_bayesian_nudge ... variance=preserve`
  - `force_absent` -> `delete ... confidence=high`
- compatibility argument forms:
  - soft updates (`~=`)
  - `delete`
  - `suppress`

Stable canonical-style lint codes:

- `canonical_set_expectation`
- `canonical_force_absent`
- `canonical_inline_args`

Refer to repository release notes and commit history for migration details.

## 16. Statistical Lints and Suppression Pragmas

Frontend/LSP tooling emits non-fatal statistical guardrail lints with stable codes:

- `stat_variance_collapse`
- `stat_prior_dominance`
- `stat_precision_outlier`
- `stat_prior_data_conflict`
- `stat_numerical_instability`
- `stat_multiple_testing`
- `stat_circular_update`
- `stat_delete_explanation`
- `stat_suppress_explanation`

You can suppress specific lint codes in scoped regions with pragmas:

```grafial
// grafial-lint: ignore(stat_multiple_testing)
rule RiskyButIntentional on MyBeliefs {
  ...
}
```

Scope behavior:

- A pragma inside a declaration applies from that line to the end of that declaration.
- A pragma before a declaration applies to the next declaration block.
