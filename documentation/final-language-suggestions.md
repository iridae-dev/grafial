# Grafial — Final Language Suggestions (Opinionated v1)

This document merges and reconciles prior proposals into a single, principled design for Grafial. We prioritize fluency, clarity, and a cohesive “one right way” over multiple equivalent forms. Backwards compatibility is not a constraint.

## Design Principles

- Fluent by default: common cases are terse; explicitness available when needed.
- Probabilistic first: uncertainty is a first‑class citizen, never bolted on.
- One canonical style: avoid parallel syntaxes that fragment usage.
- Readable ASCII: no reliance on Unicode symbols; easy to type and diff.
- Tooling‑friendly: predictable structure for IDEs, formatters, and linters.

## Canonical Surface Syntax (At a Glance)

- Blocks use braces `{ ... }`; newlines terminate statements; semicolons optional.
- Comments: `//` and `/* ... */`; docs: `///` (item), `//!` (file/module).
- Node refs: `Person("Alice")`; allow bare idents `Person(Alice)` and numbers `Person(42)` when valid.
- Priors inline in schema; posterior names are short: `Gaussian`, `Bernoulli`, `Categorical`.
- Evidence is grouped; edges default to present unless marked absent; categorical uses verbs.
- Rules use compact pattern → action form with an optional `where` guard; node‑only “for each” has a dedicated form.
- Expressions default to expectations/probabilities for arithmetic, but Boolean comparisons in `where` must be explicit: use `E(...)`, `prob(...)`, or `credible(...)`.
- Metrics use a builder style: `Person.where(cond).sum(expr)`.
- Flows compose with pipes `|>`; `snapshot` and `export` are terminal operations in the pipeline.

## 1) Schema & Priors

Canonical (inline priors):

```grafial
schema Social {
  node Person {
    score: Real ~ Gaussian(mean=0.0, precision=0.01)
    trust: Real ~ Gaussian(mean=0.5)         // precision defaulted
  }
  edge KNOWS ~ Bernoulli(prior=0.1, weight=2.0)
  edge ROUTES_TO ~ Categorical(group_by=source, prior=uniform, weight=1.0)
}
```

- Posterior type names are concise and explicit: `Gaussian(mean, precision)`, `Bernoulli(prior, weight)`, `Categorical(group_by, prior, weight)`.
- Inline priors are the default. An explicit model is optional for advanced scenarios:

```grafial
model SocialBeliefs on Social { /* override priors if needed */ }
```

Defaults and inference:
- Omit `precision` to use a weak, documented default.
- `group_by` uses enums: `source` or `target`, not strings.

## 2) Evidence

Grouped and concise (with optional per-observation precision):

```grafial
evidence SocialData on Social {
  Person {
    "Alice" { score: 0.8 (precision=10.0), trust: 0.6 }
    "Bob"   { score: 0.5 }
  }

  KNOWS {
    "Alice" -> "Bob"    // present implied
    "Bob" -/> "Eve"     // explicitly absent
  }

  // Categorical choices (Dirichlet updates)
  choose ROUTES_TO("R1", "R2")
  unchoose ROUTES_TO("R1", "R3")  // optional verb for negative signal
}
```

- `observe` is implicit in evidence blocks; the verbs `choose`/`unchoose` are used for categorical.
- Bulk import patterns (CSV/JSON) are allowed but are tooling features, not syntax core.

## 3) Rules & Patterns

Compact, readable patterns with an optional `where` guard and block action:

```grafial
rule Transfer on Social {
  (A)-[ab:KNOWS]->(B)
  where prob(ab) >= 0.9 and E(B.score) < E(A.score) => {
    let v = (E(A.score) - E(B.score)) * 0.1
    non_bayesian_nudge B.score to (E(B.score) + v) variance=preserve
  }
}
```

Node‑only form (no dummy self‑edges):

```grafial
rule BoostHighTrust on Social {
  for (P:Person) where P.trust > 0.7 => {
    P.score *= 1.05
  }
}
```

Path patterns and alternation:

```grafial
// Multi‑hop with anonymous intermediates
(A:Person)-[:KNOWS]->()-[:KNOWS]->(C:Person)
where prob(A.score > C.score) > 0.8 => {
  non_bayesian_nudge C.trust to (E(C.trust) + 0.05) variance=preserve
}

// Alternation for similar edges
(A)-[e:KNOWS|LIKES]->(B)
where prob(e) >= 0.8 => {
  non_bayesian_nudge B.score to (E(B.score) + E(A.score) * 0.1) variance=preserve
}
```

Fixpoint:

```grafial
@fixpoint
rule Propagate on Social {
  (A)-[ab:KNOWS]->(B)
  where prob(ab) >= 0.8 and E(A.trust) > E(B.trust) + 0.05 => {
    B.trust ~= E(A.trust) * 0.9 precision=0.2
  }
}
```

Action semantics (opinionated and explicit):
- Non‑Bayesian intervention: `non_bayesian_nudge A.attr to expr variance=preserve|increase|decrease(factor=k)` sets the posterior mean to `expr` with explicit variance handling.
  - Semantics: `preserve` keeps precision τ unchanged; `increase(f)` sets τ_new = τ_old × f (f < 1 increases variance); `decrease(f)` sets τ_new = τ_old × f (f > 1 decreases variance).
- Bayesian soft update: `A.attr ~= value precision=τ` adds weak evidence (conjugate update) toward `value`.
  - Convenience: `count=n` multiplies the observation precision (effective τ_obs = τ × n).
- Edge deletion and suppression:
  - `delete e [confidence=low|high]` sets a near‑zero Beta prior (default Beta(1,1e6); low≈1e3, high≈1e9). IDEs should display “observations required to undelete to p≈0.5.”
  - `suppress e [weight=k]` sets a moderate Beta prior against existence (e.g., Beta(1,10) or Beta(1,k)).
  - Optional framing: allow `certainty=x` sugar that maps to Beta parameters under a documented model.

## 4) Expressions & Semantics

Context and comparisons:
- Arithmetic defaults: inside arithmetic expressions, `A.attr` is `E(A.attr)` and edge variables default to `prob(edge)`.
- Boolean comparisons in `where` must be explicit: use `E(...)`, `prob(...)`, or `credible(lhs > rhs, p)`.
- Uncertainty primitives are available: `variance(x)`, `stddev(x)`, `ci(x, p)` (returns lo, hi), `prob(predicate)`, `effective_n(x)`, `quantile(x, p)`.
- Deterministic vs uncertain: even when comparing to scalars, wrap the uncertain side (`E(A.score) > 0.5`). Mixed explicit/implicit forms are lint errors.
- Probability calculations should respect posterior correlations where supported; future variants may expose `prob_independent`/`prob_correlated`.

Logical operators are words: `and`, `or`, `not`. Arithmetic/comparison use symbols.

## 5) Metrics

Builder pattern (single canonical style):

```grafial
metric total = Person.sum(score)
metric premium = Person.where(score > 60 and trust > 0.6).count()
metric weighted = Person.sum(score * trust)
metric avg_score = Person.avg(score)
metric strong_degree = Person.avg_degree(KNOWS, min_prob=0.8)
```

- Within metric expressions, `node` is implicit; refer to attributes directly (`score`, `trust`).
- Provide standard stats: `.avg`, `.sum`, `.count`, `.var`, `.std`, `.median`, `.percentile(p)`, `.range()`.
- Network helpers: `.degree(edge, min_prob=...)`, `.avg_degree(edge, min_prob=...)`.
- Uncertainty helpers: `.avg_variance(attr)`, and accessors `variance(x)`, `stddev(x)`, `ci(x,p)`, `effective_n(x)`, `quantile(x,p)` for use in expressions.

Rationale: the builder reads naturally and avoids repeating labels and boilerplate.

## 6) Flows & Pipelines

Pipeline composition with pipes and terminal ops:

```grafial
flow SocialPipeline on Social {
  from_evidence SocialData
    |> Transfer
    |> snapshot cleaned
    |> BoostHighTrust
    |> export "final_social"
}
```

- `from_evidence Name` starts from an evidence set; `from_graph "name"` composes across flows.
- `snapshot name` stores a checkpoint; `export "name"` emits the current graph.
- Use graph identifiers only when needed for diffs or comparisons; otherwise prefer a linear pipeline.

## 7) Strings, Idents, Numbers

- Strings: allow both single and double quotes with identical escaping rules.
- Idents in node refs: allow bare idents when unambiguous (`Person(Alice)`); otherwise use quotes.
- Numbers: support `_` separators in numeric literals (e.g., `1_000.0`).

## 8) Comments, Docs, and Notes

- Use `//` and `/* ... */` for comments.
- Doc comments: `///` for items, `//!` for file/module docs; tooling surfaces these.
- Optional in‑flow annotations: `note "text"` is allowed in flows to attach narrative to snapshots/exports/metrics; it has no semantic effect but is available to tooling/UX.

## 9) Patterns We Deliberately Avoid

- No indentation‑sensitive blocks: braces are canonical (simpler parsing, fewer surprises).
- No symbolic `&&`/`||`: `and`, `or`, `not` are clearer in a DSL.
- No competing canonical forms for metrics or rules: builder and arrow→action forms are the standard.
- No hard constraints (exact probability 0/1) in the core: we keep probabilistic semantics; `delete` is an extremely strong prior, not a hard truth.

## 10) End‑to‑End Example

```grafial
schema ABTest {
  node Variant {
    conversion: Real ~ Gaussian(mean=0.10, precision=10.0)
    traffic: Real ~ Gaussian(mean=1000.0)
  }
  edge OUTPERFORMS ~ Bernoulli(prior=0.5, weight=2.0)
}

evidence Data on ABTest {
  Variant {
    "A" { conversion: 0.12, traffic: 1000 }
    "B" { conversion: 0.15, traffic: 1000 }
  }
  OUTPERFORMS { "A" -> "B" } // present implied
}

rule Decide on ABTest {
  (A:Variant)-[e:OUTPERFORMS]->(B:Variant)
    where prob(e) >= 0.6 and prob(B.conversion > A.conversion) > 0.9 => {
      non_bayesian_nudge B.conversion to (E(B.conversion) * 1.01) variance=preserve
    }
}

flow Analysis on ABTest {
  from_evidence Data
    |> Decide
    |> snapshot decided

  metric avg = Variant.avg(conversion)
  metric strong = Variant.count(E(conversion) > 0.12)

  note "B beats A but lift is marginal"
  export "winner"
}
```

## 11) Tooling, Lints & Statistical Guardrails (Style Enforced)

- Prefer inline priors in `schema`; use `model` only for overrides.
- In expressions, rely on contextual defaults for arithmetic only; in `where`, require explicit `E(...)`/`prob(...)`/`credible(...)`.
- Use builder metrics exclusively; avoid legacy `sum_nodes`/`count_nodes` forms.
- Use arrow→action rule form and `for (...) => {}` for node‑only loops.
- Prefer call‑style node refs `Label("id")` in examples and docs.
- Encourage trailing commas and stable formatting; provide a formatter to normalize layout.

Statistical lints and annotations:
- Lints: variance collapse (repeated nudges), prior dominance (τ too strong), multiple testing (many ORs), implicit comparisons (error), undelete guidance after `delete`, circular update detection across rules, precision outlier warnings, prior‑data conflict checks, numerical instability (τ→0 or ∞), and flags for over‑deterministic thresholds.
- Annotations: `@track_uncertainty_budget(max_comparisons=...)`, `@lint_prior_dominance(threshold=...)`, `@counterfactual` on rules with interventions, `@audit_trail` for flows; `@seed`, `@deterministic` for reproducibility.

Domain priors & sensitivity:
- Standard library suggestions: `priors::weak_informative`, `priors::beta_uniform`, `priors::jeffrey_scale`, `priors::zero_centered(scale=1)`, `priors::ab_testing`, `priors::social_network`.
- Sensitivity: `@sensitivity_analysis(priors=[...], metric="...")` to visualize robustness of conclusions.

---

This “Final” proposal yields a cohesive, modern DSL that reads like a principled query‑and‑rules language for uncertain graphs: compact where it should be, explicit when it matters, and consistent end‑to‑end for a great authoring experience.
