# Grafial Final Language — Implementation Roadmap

This roadmap operationalizes `final-language-suggestions.md` with a concrete, staged plan to implement the language, upgrade the engine and tools, migrate examples, and enforce statistical guardrails.

## Goals

- Implement the canonical syntax and semantics described in `final-language-suggestions.md`.
- Make uncertainty explicit where it matters and prevent common statistical mistakes.
- Keep project deterministic and well‑tested at every step.

## Status Summary

- Phase 0: Planned
- Phase 1 (Grammar + AST): Completed
- Phase 2 (Engine semantics): In progress — variance/stddev/quantile done; ci/effective_n + action semantics next
- Phase 3 (LSP & lints): Not started
- Phase 4 (Examples & docs): Not started
- Phase 5 (Cleanup & release): Not started

## Scope (Deliverables)

- Parser/grammar updates (inline priors, grouped evidence, new rule/action forms, builder metrics)
- Engine features (non_bayesian_nudge, `~=` Bayesian soft updates, delete/suppress, uncertainty primitives)
- Frontend validation (explicit comparisons in `where`, ambiguity checks)
- Metrics runtime (builder API, new stats and probability helpers)
- LSP diagnostics + quick fixes (lints, code actions, hover docs)
- Examples migrated to new syntax; docs updated; CLI stays stable

## Architecture Touchpoints

- Frontend: `crates/grafial-frontend/grammar.pest`, `parser.rs`, AST types, validations
- Core engine: `crates/grafial-core/engine/{graph.rs,evidence.rs,rule_exec.rs,metrics.rs}`
- CLI: `crates/grafial-cli/`
- LSP/VS Code: `crates/grafial-vscode/` (diagnostics, code actions, hover)
- Examples: `crates/grafial-examples/*.grafial`
- Docs: `documentation/*.md`, `README.md`, language guide

## Phased Plan (Opinionated)

Phase 0 — Planning & Guardrails (1–2 days)
- Freeze current main with a branch `syntax-v1` for the new language.
- Add CI lanes to run parser + engine tests and examples with “v1” feature.
- Define acceptance tests for: examples parse+run; uncertainty primitives; rule fixpoint convergence.

Phase 1 — Grammar + AST (inline priors, evidence, rules) (1–1.5 weeks)
Status: Completed
- Inline priors in schema
  - Add `Gaussian`, `Bernoulli`, `Categorical` tokens and argument schemas.
  - Allow `attr: Real ~ Gaussian(...)` in `node` and `edge ~ Bernoulli(...)` forms.
- Evidence grouping and categorical verbs
  - Blocks for nodes/edges; `->` present, `-/>` absent; `choose`/`unchoose` for categorical.
  - Optional per‑observation `(precision=...)` after literal values.
- Rule syntax
  - Pattern → action form with `where` and `=>` block.
  - Node‑only `for (Label)` form.
- Actions
  - Parse `non_bayesian_nudge X to expr [variance=...]`.
  - Parse `X ~= value precision=τ [count=n]`.
  - Parse `delete e [confidence=...]` and `suppress e [weight=...]`.
- Expressions & comparisons
  - Keep arithmetic defaults for `E(...)`/`prob(...)` in expressions.
  - Add frontend validation to require explicit wrappers in any Boolean comparison within `where`.
- Metrics builder
  - Grammar for `Label.where(...).sum(expr)`, `.avg`, `.count`, `.var`, `.std`, `.median`, `.percentile(p)`, `.range()`.
- AST + IR changes and visitors to support new constructs.

Issue breakdown (Phase 1):
- frontend/grammar
  - [x] Add `Gaussian`, `Bernoulli`, `Categorical` productions with named args (short names accepted; parameter name normalization in parser).
  - [x] Support `Label("id")` and `Label(id)` node refs; keep strings and numeric literals.
  - [x] Evidence blocks: node group, edge group, `->`/`-/>` blocks (grouped form).
  - [x] `(precision=...)` after attribute literals (per-observation).
  - [x] `choose`/`unchoose` tokens for categorical edges.
  - [x] Pattern → action `=>` form; node-only `for (Label)`.
  - [x] Pattern → action `=>` form.
  - [x] Node-only `for (Label)` form (parse sugar; engine iteration to follow in Phase 2).
  - [x] Actions: `non_bayesian_nudge ... variance=...`, `~=` with `precision=` and optional `count=`, `delete ... [confidence=]`, `suppress ... [weight=]`.
  - [x] Parse actions: `non_bayesian_nudge ... variance=...`, `~=` with `precision=`/`count=`, `delete ... [confidence=]`, `suppress ... [weight=]` (engine semantics pending in Phase 2).
  - [x] Metrics builder pipeline: `nodes(Label) |> where(...) |> sum(by=...) | count() | avg(by=...) | fold(init, step, order_by)` (desugars to existing functions)
- parser/AST/validation
  - [x] Extend AST nodes for new constructs and attach source spans.
  - [x] Validation: require explicit uncertainty in `where` (reject bare field access; enforce E[NodeVar.attr] or prob(...)).
  - [ ] Validation: disallow mixed implicit/explicit comparisons; require `prob(e)` for edges in `where`.
  - [ ] Determinism: ensure declaration lists, evidence entries, and pattern lists preserve stable order.

Definition of done (Phase 1):
- [x] All new grammar forms parse; round-trip tests pass for seed examples.
- [x] Violations of explicit-where rule produce precise diagnostics (line/col).
- [x] Minimal seed example compiles with inline priors, grouped evidence, and new rule/action syntax.

Acceptance:
- Unit tests for grammar productions and round‑trip AST.
- Parse all updated minimal examples using new syntax (create a small seed set).

Phase 2 — Engine Semantics (1–1.5 weeks)
- Implement `non_bayesian_nudge` with variance semantics
  - `preserve`: τ unchanged; `increase(f)`: τ = τ×f (f<1 increases variance); `decrease(f)`: τ = τ×f (f>1 decreases variance).
  - Audit trail hooks for interventions (`@counterfactual`).
- Implement Bayesian soft update `~=` (Normal‑Normal)
  - τ_new = τ_old + τ_obs; μ_new = (τ_old×μ_old + τ_obs×v)/τ_new.
  - Support `count=n` → τ_obs_eff = τ×n.
- Edge operations
  - `delete e` → Beta(1, 1e6) by default; map `confidence` to {1e3, 1e9};
  - `suppress e` → Beta(1, weight) with default 10.
  - Provide helper to compute “observations to p≈0.5” for diagnostics.
- Uncertainty primitives
  - `variance(x)`, `stddev(x)`, `ci(x,p)`, `quantile(x,p)`, `effective_n(x)`.
  - Probability comparisons: `prob(A > B)` for Gaussians via analytic CDF; document independence assumption initially.
- Metrics runtime
  - Evaluate builder API and primitives; ensure deterministic accumulation order.

Issue breakdown (Phase 2):
- core/engine
  - [x] Implement `non_bayesian_nudge` with variance strategies (preserve/increase/decrease).
  - [x] Implement soft update `~=` (Normal–Normal), including `count`.
  - [x] Implement `delete` (Beta(1,1e6) default) with `confidence` mapping; `suppress` (Beta(1,weight)).
  - [x] Helper: compute observations required to reach p≈0.5 after `delete`.
  - [x] Primitive: `variance(x)`
  - [x] Primitive: `stddev(x)`
  - [x] Primitive: `ci(x,p)`
  - [x] Primitive: `quantile(x,p)`
  - [x] Primitive: `effective_n(x)`
  - [x] Probability comparison for Gaussians: `prob(A > B)` via analytic CDF; document independence.
- metrics runtime
  - [ ] Execute builder chains; verify stable iteration and accumulation.

Definition of done (Phase 2):
- [ ] Unit tests for each action and primitive; property tests for conjugacy and numeric stability.
- [ ] Benchmarks run within 10% of baseline; document any changes.
- [ ] Diagnostics surfaced for undelete analysis (API ready for LSP).

Acceptance:
- Property tests for conjugate updates, soft updates, and invariants.
- Unit tests for `delete/suppress` parameterizations and undelete math.
- Tests for probability comparison and CI/quantile correctness.

Phase 3 — LSP & Lints (1 week)
- Diagnostics
  - Error on implicit comparisons in `where` (suggest quick fixes to wrap with `E(...)` or `prob(...)`).
  - Warnings: variance collapse (repeated nudges), prior dominance, precision outliers, multiple testing, prior‑data conflict, numerical instability, circular updates.
  - Edge deletion tooltip: undelete observation analysis; optional `certainty=` sugar hint.
- Code actions
  - “Modernize syntax”: convert legacy forms to v1 (internal use).
  - Quick‑wrap uncertain comparisons; convert `+=` legacy to `non_bayesian_nudge` or `~=` with `precision=`.
- Hovers
  - Show posterior summary (μ, σ, CI); show effective_n; short distribution cheatsheet.

Acceptance:
- LSP tests for diagnostics positions and quick‑fix edits.

Issue breakdown (Phase 3):
- diagnostics
  - [ ] Error: implicit comparisons in `where` with quick-fix to `E(...)`/`prob(...)`.
  - [ ] Warn: variance collapse (repeated nudges on same attr) with suggestion to increase variance or switch to `~=`.
  - [ ] Warn: prior dominance with quantitative guidance (prior vs typical τ_obs).
  - [ ] Warn: precision outliers; prior–data conflict; numerical instability (τ→0 or ∞); multiple testing; circular updates across rules.
  - [ ] Tooltip: `delete` → show undelete observation counts; suggest `confidence=`.
- code actions
  - [ ] Modernize legacy syntax; convert `+=` to `non_bayesian_nudge` or `~=`.
  - [ ] Wrap uncertain comparisons automatically.
- hovers
  - [ ] Show μ, σ, CI, effective_n; quick links to docs.

Phase 4 — Examples & Docs Migration (3–5 days)
- Rewrite all `crates/grafial-examples/*.grafial` to v1 syntax (explicit comparisons, evidence grouping, verbs, builder metrics).
- Update `documentation/LANGUAGE_GUIDE.md`, `README.md` snippets, and tutorial flows.
- Add a “Statistical Semantics” section mirroring review mitigations.

Acceptance:
- All examples parse and run; CI verifies metric outputs are sensible (golden files or assertions).

Migration checklist (Phase 4):
- [x] Convert `examples/minimal.grafial` to inline priors, grouped evidence, builder metrics.
- [x] Convert `examples/social.grafial` with explicit `where` comparisons and `non_bayesian_nudge/~=`.
- [x] Convert `examples/ab_testing.grafial` with probability comparisons and builder metrics.
- [x] Convert remaining examples; remove legacy constructs (set_expectation/force_absent).
- [x] Update `README.md` snippets and `documentation/LANGUAGE_GUIDE.md` with new syntax.

Phase 5 — Cleanup & Release (2–3 days)
- Remove legacy code paths not needed for v1.
- Finalize formatting rules; run `cargo fmt`, `clippy` clean.
- Tag release; publish notes highlighting statistical guardrails.

## Risks & Mitigations

- Grammar churn breaking downstream: implement on `syntax-v1` branch; migrate examples in the same PR to keep CI green.
- Probability comparisons under correlation: document independence assumption now; plan `prob_correlated` later.
- Lint noise: tune thresholds and provide disable pragmas (`// grafial-lint: ignore(rule)` per line).
- Performance regressions: add benches for new primitives and ensure no slowdown >10% on representative graphs.

## Testing Strategy

- Frontend: golden AST snapshots per new syntax form; fuzz tests on parser (Pest).
- Engine: proptests for conjugacy and numeric stability; unit tests for each primitive.
- Integration: run examples end‑to‑end and assert metric values / topology properties.
- LSP: fixture files with expected diagnostics and code actions.

Immediate test plan seeds:
- [ ] Grammar golden for: inline priors, evidence `->`/`-/>`, `choose`, `unchoose`, rule `=>`, node-only `for`, actions, builder metrics.
- [ ] Engine property tests: Normal–Normal update math; Beta updates; undelete math.
- [ ] Metrics: `quantile`, `ci`, `effective_n` sanity against closed form.
- [ ] LSP: implicit-where error quick-fix; undelete tooltip presence.

## Acceptance Criteria (Exit Checklist)

- Grammar supports all canonical forms; old forms removed or gated off.
- Engine implements `non_bayesian_nudge`, `~=`; delete/suppress; uncertainty primitives.
- `where` comparisons enforced explicit; lints active with actionable messages.
- Metrics builder API stable; docs include full reference and examples.
- All examples and docs updated; CI green (build, tests, clippy, fmt).

## Timeline (Rough)

- Phase 0–1: 1.5 weeks
- Phase 2: 1.5 weeks
- Phase 3: 1 week
- Phase 4: 1 week
- Phase 5: 0.5 week

Total: ~4.5 weeks (buffer to 5–6 weeks accounting for review iterations).

Near-term schedule (by week):
- Week 1: Grammar + AST; seed example parses; explicit-where validation.
- Week 2: Actions + primitives in engine; tests for updates and delete/suppress.
- Week 3: Metrics builder runtime; LSP diagnostics skeleton.
- Week 4: LSP code actions + hovers; migrate core examples and docs.
- Week 5 (buffer): Cleanup, polish, release prep.

## Ownership & Coordination

- Frontend/grammar/AST: Parser owner
- Engine semantics/metrics: Core engine owner
- LSP diagnostics & quick fixes: Tools owner
- Docs/examples migration: Docs owner
- Project lead: coordinates release, reviews, and statistical sign‑off

## Out‑of‑Scope (Follow‑ups)

- Correlation‑aware `prob()` and predictive checks
- Advanced metrics (entropy, KL divergence)
- Causal annotations and do‑calculus semantics
- Prior sensitivity wizards and UI

---

This plan delivers the opinionated v1 language with strong statistical foundations and developer ergonomics, in a sequence that keeps the codebase stable and reviewable.
