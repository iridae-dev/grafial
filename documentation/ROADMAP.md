# ROADMAP

This is the canonical compiler/runtime roadmap for Grafial.

## Status

- Phase 0: Completed
- Phase 1: Completed
- Phase 2: Completed
- Phase 3: Completed
- Phase 4: Completed
- Phase 5: Completed
- Phase 6: Completed
- Phase 7: Completed
- Phase 8: Completed
- Phase 9: Completed
- Phase 10: Future
- Phase 11: Future
- Phase 12: Future
- Phase 13: Future
- Phase 14: Future

## Execution Order

1. Phase 0 - Language Contract Freeze
2. Phase 1 - Frontend Correctness Recovery
3. Phase 2 - Semantic Checker Upgrade
4. Phase 3 - Real Compiler IR Pipeline
5. Phase 4 - Optimization + Determinism
6. Phase 5 - Tooling Lift (LSP, quick fixes, formatting/linting)
7. Phase 6 - Hardening + Release Gate
8. Phase 7 - Statistical Guardrails
9. Phase 8 - Canonicalization Completion
10. Phase 9 - Advanced Probabilistic Semantics
11. Phase 10 - JIT Backend Prototype
12. Phase 11 - AOT + Vectorized Runtime
13. Phase 12 - Parallel Engine Execution
14. Phase 13 - Graph Storage + Indexing
15. Phase 14 - Numeric Kernels + Equivalence

## Phase 0 - Language Contract Freeze

Goal: define one canonical syntax contract and remove parser/spec drift.

Contract decisions for v1 compiler input:
- Soft updates: canonical inline args are accepted (`A.x ~= 1.0 precision=0.2 count=3`).
- Soft updates: parenthesized args are also accepted for compatibility (`A.x ~= 1.0 (precision=0.2, count=3)`).
- Edge operations: inline args are accepted (`delete e confidence=high`, `suppress e weight=10`).
- Edge operations: parenthesized args are also accepted for compatibility.
- Categorical `group_by`: both quoted and identifier forms are accepted (`group_by="source"` and `group_by=source`).
- Metric builder usage: builder pipelines can be used in expression context (not only as a top-level metric RHS).

Deliverables:
- Grammar aligned with contract.
- Parser aligned with contract.
- Regression tests for each accepted form.

## Phase 1 - Frontend Correctness Recovery

Goal: fix concrete parser/frontend correctness bugs already observed in examples/tests.

Target fixes:
- Action blocks must populate `RuleDef.actions` reliably.
- `for (Var:Label)` sugar must bind `Var`/`Label` correctly.
- Soft update/delete/suppress argument parsing must handle canonical and compatibility forms.
- Categorical parsing must accept `group_by=source` form.
- Builder pipelines in nested expression contexts must parse.

Validation gates:
- `cargo test -p grafial-frontend`
- `cargo test -p grafial-tests --test examples_tests`
- `cargo run -p grafial-cli --bin grafial -- crates/grafial-examples/*.grafial` parse/validate sweep (batched)

Completion notes (this change):
- Fixed action parsing so action blocks populate `RuleDef.actions`.
- Fixed `for (Var:Label)` sugar variable/label binding.
- Added canonical + compatibility parsing for soft updates, `delete`, and `suppress`.
- Added categorical `group_by=source` parsing support.
- Enabled nested metric-builder expressions and fixed expression operator token parsing.
- Added/strengthened parser regression tests and updated brittle integration expectation for `social.grafial`.

## Phase 2 - Semantic Checker Upgrade

- Introduce validation contexts (`rule where`, `metric expr`, `action expr`).
- Improve semantic diagnostics and variable-scope checks.
- Attach precise span/range information for diagnostics.

Completion notes (this change):
- Added rich validation diagnostics with optional context and source ranges.
- Added `validate_program_with_source(...)` and source-span indexing for rule where/action, metric expressions, and prune predicates.
- Upgraded rule validation with scoped variable checking across where clauses, exists subqueries, and sequential action statements (`let` locals included).
- Upgraded metric validation with contextual checks for flow metric expressions, imported metric scope, node-expression contexts (`node`), and fold-step accumulator variable (`value`).
- Upgraded LSP diagnostic mapping to use semantic validation ranges instead of always pinning to document start.

## Phase 3 - Real Compiler IR Pipeline

- Add `ExprIR` and `EvidenceIR`.
- Lower `ProgramAst -> ProgramIR` with normalized typed nodes.
- Add core entrypoints that execute IR directly.
- Keep AST-based wrappers temporarily for compatibility.

Completion notes (this change):
- Added first-class `ExprIR` (with unary/binary ops + call args) and `EvidenceIR` (typed observations + modes) plus round-trip helpers.
- Upgraded `RuleIR` to use `ExprIR`/`ActionIR` and upgraded `FlowIR` to carry typed metric/export/import surfaces and `ExprIR` predicates.
- Upgraded `ProgramIR` to lower evidences via `EvidenceIR` and provide IR<->AST conversion helpers.
- Added `grafial_core::parse_validate_and_lower(...)` and `grafial_core::run_flow_ir(...)` as IR-native public entrypoints.
- Migrated `flow_exec` to execute `ProgramIR`/`FlowIR`/`RuleIR` directly while preserving AST compatibility wrappers (`run_flow`, `run_flow_with_builder`).
- Added IR-path parity test in `crates/grafial-tests/tests/examples_tests.rs` (`example_minimal_ir_entrypoint_matches_ast_wrapper`).

## Phase 4 - Optimization + Determinism

- Constant folding and canonicalization passes.
- Dead metric/rule elimination where safe.
- Deterministic compile and execution planning order guarantees.

Completion notes (this change):
- Added IR optimizer (`grafial_ir::optimize_program`, `ProgramIR::optimized`) with constant folding and canonicalization across rule expressions, action expressions, metric expressions, and prune predicates.
- Added safe dead-rule elimination in the optimizer (unreferenced rules are removed from optimized execution IR) and runtime dead-rule indexing (only flow-referenced rules are converted/indexed for execution).
- Added safe dead-metric optimization in flow execution: non-live metrics that fold to constants bypass runtime metric evaluation while preserving `FlowResult.metrics` values.
- Added transform-level no-op elimination and canonicalization:
  - drops empty `apply_ruleset`
  - rewrites single-rule `apply_ruleset` to `apply_rule`
  - removes `prune_edges` with constant-false predicates
  - canonicalizes constant-truthy prune predicates to `true`
- Added deterministic dependency-driven graph execution planning in `flow_exec`:
  - explicit graph-plan builder with stable ordering
  - support for out-of-order pipeline chains as long as dependencies are satisfiable
  - deterministic unresolved dependency errors
- Wired optimization into compiler/runtime entrypoints:
  - `parse_validate_and_lower(...)` now returns optimized IR
  - `run_flow_ir(...)` and `run_flow_with_builder(...)` execute optimized IR
- Added/updated tests for optimizer behavior and deterministic planning:
  - `crates/grafial-ir/src/optimize.rs` unit tests
  - `crates/grafial-core/src/engine/flow_exec.rs` tests for graph-plan determinism, out-of-order pipeline execution, and live-metric dependency tracking
  - existing Phase 4 exit criteria and example parity tests remain green.

## Phase 5 - Tooling Lift

- LSP diagnostics with precise ranges.
- Quick-fixes for uncertainty wrappers and syntax modernization.
- Formatter/lint rules for canonical style.

Completion notes (this change):
- Added canonical-style tooling in frontend:
  - `lint_canonical_style(...)` to detect compatibility syntax forms.
  - `format_canonical_style(...)` to rewrite compatibility syntax to canonical inline args.
  - Lint coverage includes parenthesized compatibility forms for:
    - `soft_update ... (~precision/count...)`
    - `delete ... (confidence=...)`
    - `suppress ... (weight=...)`
- Upgraded LSP capabilities with `textDocument/codeAction` quick fixes:
  - Canonical syntax modernization quick-fix from style diagnostics.
  - Uncertainty-wrapper quick-fix that wraps bare field expressions in `E[...]` when validator diagnostics require explicit uncertainty wrappers.
- LSP diagnostics now include canonical-style warning diagnostics with precise ranges and embedded fix payloads.
- Added CLI canonical-style tooling:
  - `--lint-style` to report canonical-style compatibility warnings.
  - `--fix-style` to rewrite source in-place to canonical style.
- Added tests for the new tooling surfaces:
  - Frontend style lint/format unit tests (`crates/grafial-frontend/src/style.rs`).
  - LSP quick-fix plumbing tests (`crates/grafial-lsp/src/main.rs`).

## Phase 6 - Hardening + Release Gate

- CI gates for examples parse+validate+flow execution.
- Property tests for Bayesian update invariants and determinism.
- Roadmap completion tied to CI checks, not manual checkboxes.

Completion notes (this change):
- Added CI release-gate automation:
  - `scripts/phase6_release_gate.sh` runs the Phase 6 checks end-to-end.
  - `.github/workflows/phase6-release-gate.yml` runs the release gate on push/PR.
- Added full example execution gate:
  - `crates/grafial-tests/tests/phase6_release_gate.rs` parses/validates every `.grafial` example and executes every declared flow.
  - Gate verifies AST and IR entrypoints produce matching metric/export surfaces for every example flow.
- Expanded property-test hardening:
  - `crates/grafial-tests/tests/property_tests.rs` now includes Bayesian monotonicity and boundedness invariants for Beta/Gaussian updates.
  - Added determinism property coverage for flow execution under different edge insertion orders.
- Roadmap completion is now tied to CI pass/fail:
  - Phase 6 checks are encoded as executable test+workflow gates rather than manual checklist updates.


## Phase 7 - Statistical Guardrails

- Add statistical diagnostics in frontend/LSP with stable lint codes:
  - variance collapse (repeated non-Bayesian nudges),
  - prior dominance and precision outliers,
  - prior-data conflict and numerical instability,
  - multiple testing and circular update patterns.
- Add delete/suppress explanatory diagnostics (including undelete observation count estimates).
- Add scoped lint suppression pragmas for noisy-but-intentional cases (for example `// grafial-lint: ignore(<code>)`).

Completion notes (this change):
- Added frontend statistical lint pass with stable codes:
  - `stat_variance_collapse`
  - `stat_prior_dominance`
  - `stat_precision_outlier`
  - `stat_prior_data_conflict`
  - `stat_numerical_instability`
  - `stat_multiple_testing`
  - `stat_circular_update`
  - `stat_delete_explanation`
  - `stat_suppress_explanation`
- Added explanatory diagnostics for `delete`/`suppress`, including observation-count estimates to recover edge probability above 0.5.
- Added scoped lint suppression pragmas:
  - `// grafial-lint: ignore(<code>)`
  - supports declaration-scoped suppression for both canonical-style and statistical lints in tooling surfaces.
- Wired Phase 7 lints into LSP diagnostics and added regression tests in frontend/LSP crates.

## Phase 8 - Canonicalization Completion

- Add modernization quick fixes for legacy constructs:
  - `set_expectation` -> `non_bayesian_nudge ... variance=preserve`
  - `force_absent` -> `delete ... confidence=high`
- Expand canonical style lint/fix coverage for remaining compatibility forms.
- Migrate examples/docs to canonical syntax-first presentation and gate/retire remaining legacy parser paths.
- We don't have to support anything "legacy" since we haven't yet released Grafial to the public. Ensure the right way is the only way. 

Completion notes (this change):
- Added canonical modernization lint/fix coverage for legacy actions:
  - `canonical_set_expectation`
  - `canonical_force_absent`
  - existing `canonical_inline_args` coverage retained
- Updated style tooling rewrites:
  - `set_expectation A.attr = expr` -> `non_bayesian_nudge A.attr to expr variance=preserve`
  - `force_absent e` -> `delete e confidence=high`
  - parenthesized compatibility args remain auto-rewritten to inline canonical form
- Retired legacy parser paths:
  - removed `set_expectation` and `force_absent` from grammar + parser action parsing
  - canonical action syntax is now the only accepted source syntax for these operations
- Migrated shipped examples and language docs to canonical syntax-first usage.
- Added migration documentation in `documentation/MIGRATION_GUIDE.md`.
- Strengthened gates/tests:
  - phase release-gate now enforces zero canonical-style compatibility lints across all shipped examples
  - added frontend/LSP tests for legacy modernization quick-fix payloads and rewrites

## Phase 9 - Advanced Probabilistic Semantics

- Add correlation-aware probability comparison support (for example `prob_correlated`) while retaining explicit independence semantics where used.
- Evaluate and, if justified, add `credible(...)` as a first-class comparison helper with clear semantics.
- Add intervention audit metadata hooks for reproducibility and traceability in compiler/runtime outputs.

Completion notes (this change):
- Added `prob_correlated(...)` in rule evaluation:
  - explicit correlation-aware Gaussian comparison probability (`rho` in `[-1, 1]`)
  - existing `prob(...)` comparison behavior remains the explicit independence form
- Added `credible(...)` as a first-class probability-threshold helper:
  - supports edge events (`credible(e, p=...)`)
  - supports comparison events with optional correlation (`credible(A.x > B.x, p=..., rho=...)`)
- Added validation and integration coverage for new semantics in frontend/core/test crates.
- Added runtime intervention audit hooks in flow outputs:
  - `FlowResult.intervention_audit` now records per-transform rule execution metadata
    (flow, graph, transform id, rule name, mode, matched bindings, actions executed)
  - CLI summary/JSON output now surfaces intervention audit events.

## Phase 10 - JIT Backend Prototype

- Define an IR-level codegen backend boundary (`ProgramIR`/`ExprIR`/`RuleIR`) so execution can swap interpreter vs JIT without frontend changes.
- Prototype expression JIT for hot metric and predicate expressions with compilation caching and deterministic fallback to interpreter.
- Add hot-path thresholds and profiling hooks so JIT only triggers when execution count amortizes compile cost.
- Run backend spike comparison (LLVM vs Cranelift) and select default backend based on compile latency, runtime speed, complexity, and maintenance cost.

## Phase 11 - AOT + Vectorized Runtime

- Lower selected rule predicates/actions from IR to native kernels while keeping dynamic graph traversal in the runtime planner.
- Add vectorized Bayesian update kernels for batched evidence ingestion where deterministic ordering is preserved.
- Add optional flow AOT build path (artifact generation for precompiled execution units) with strict feature gating.
- Add benchmark and correctness gates:
  - no semantic drift vs interpreter path,
  - deterministic parity across backends,
  - documented performance targets on representative workloads.

## Phase 12 - Parallel Engine Execution

- Add feature-gated parallel evidence ingestion:
  - process independent observations in parallel,
  - merge/apply deltas in deterministic order.
- Add dependency-aware parallel metric evaluation:
  - build metric dependency graph,
  - execute independent levels in parallel while preserving dependent ordering.
- Evaluate safe parallel rule application for non-overlapping match partitions with deterministic merge semantics.

## Phase 13 - Graph Storage + Indexing

- Add incremental adjacency index maintenance with rebuild heuristics for large delta batches.
- Add lazy adjacency invalidation/versioning and sparse per-node delta indexes for degree and neighborhood queries.
- Reduce allocation pressure in hot paths:
  - arena allocation for temporary rule-match structures,
  - inline storage for small posterior/attribute collections.
- Evaluate optional storage-model experiments behind feature flags:
  - deeper SoA layouts,
  - workload-specialized graph representations.

## Phase 14 - Numeric Kernels + Equivalence

- Add SIMD-accelerated probability kernels where data shape and platform support justify it (for example larger categorical posterior vectors).
- Require strict numerical equivalence gates for all optimized paths:
  - posterior parity vs reference implementation (within documented epsilon),
  - deterministic parity across repeated runs and backend selections.
- Gate risky optimizations behind feature flags and benchmark thresholds before default enablement.
