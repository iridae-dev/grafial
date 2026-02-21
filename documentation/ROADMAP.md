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
- Phase 10: Completed
- Phase 11: Completed
- Phase 12: Completed ✅ (Fully integrated with parallel evidence, metrics, and rules)
- Phase 13: Completed ✅ (Storage/indexing optimizations integrated with experimental storage models)
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
11. Phase 10 - JIT Backend Finalization
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

## Phase 10 - JIT Backend Finalization

- Define an IR-level backend boundary (`ProgramIR`/`ExprIR`/`RuleIR`) so execution can swap interpreter vs JIT without frontend changes.
- Implement hot-expression JIT for metric and prune expressions with caching and deterministic interpreter fallback for unsupported forms.
- Select and standardize on a single JIT backend to reduce maintenance and avoid parallel half-implemented paths.

Completion notes (this change):
- Added the IR execution backend boundary in `flow_exec`:
  - `IrExecutionBackend` trait
  - `InterpreterExecutionBackend` for explicit interpreter execution
  - `run_flow_ir_with_backend(...)` explicit backend entrypoint
- Finalized on Cranelift JIT (`--features jit`):
  - `CraneliftJitExecutionBackend` with `JitConfig` and `JitProfile`
  - `run_flow_ir(...)` now dispatches through Cranelift JIT by default
  - parity/fallback behavior remains deterministic via interpreter fallback on unsupported expressions
- Implemented real Cranelift native code generation in `engine/jit_backend.rs`:
  - `cranelift-codegen`, `cranelift-frontend`, `cranelift-jit`, `cranelift-module`, `cranelift-native`
    are now feature-gated under `grafial-core/features = ["jit"]`
  - `compile_metric_expr(expr)` → `CompiledMetricFn`: lowers `ExprIR` (numbers, bools, vars,
    unary/binary ops) to a native `fn(*const f64, usize) -> f64` via Cranelift; variables are
    resolved to flat-array indices at compile time
  - `compile_prune_expr(expr)` → `CompiledPruneFn`: lowers prune predicates (numbers, bools,
    `prob(edge)`, unary/binary ops) to a native `fn(f64) -> f64` via Cranelift
  - Pre-check pass validates expression support before any Cranelift state is allocated
  - `CompiledMetricFn`/`CompiledPruneFn` hold `Arc<JITModule>` to keep JIT memory alive
  - `unsafe impl Send + Sync` on wrappers: sound because compiled code is immutable after
    `finalize_definitions()` and is called with thread-local argument slices only
  - Feature-gated dispatch in `flow_exec`: `jit_compile_metric`/`jit_compile_prune` and
    `jit_eval_metric`/`jit_eval_prune` select real Cranelift or bytecode fallback at compile time
  - Without `--features jit` the legacy register-bytecode interpreter path is used unchanged
- Removed Phase 10 decision artifacts and alternate candidate scaffolding:
  - removed LLVM candidate backend path
  - removed backend spike/matrix harnesses and generated reports
  - removed Phase 10 benchmark scripts and decision docs

## Phase 11 - AOT + Vectorized Runtime

**Status: Complete ✅**

### Completed Components:

#### 1. Vectorized Bayesian Update Kernels ✅

- **Vectorized evidence ingestion** (`crates/grafial-core/src/engine/vectorized.rs`)
  - Implemented `gaussian_batch_update` for batched Gaussian posterior updates
  - Implemented `beta_batch_update` for batched Beta posterior updates
  - Mathematically equivalent to sequential updates (preserves deterministic ordering)
  - Auto-vectorization friendly loops for SIMD optimization

- **Feature-gated integration** (`vectorized` feature flag)
  - Modified `evidence.rs` to use vectorized batch updates when feature enabled
  - Added `observe_edge_batch` and `observe_attr_batch` methods to `BeliefGraph`
  - Falls back to sequential scalar updates when feature disabled

- **Testing and benchmarks**
  - Created comprehensive parity tests (`tests/vectorized_parity.rs`)
  - Verified identical results between scalar and vectorized paths
  - Added benchmarks comparing performance (`benches/vectorized_evidence.rs`)

#### 2. Native Rule Kernel Compilation ✅

- **Rule predicate JIT compilation** (`crates/grafial-core/src/engine/rule_kernels.rs`)
  - Implemented Cranelift-based compilation for simple rule predicates
  - Compiles arithmetic and logical expressions to native x86_64 code
  - Hot predicates automatically compiled after threshold (10 executions)
  - Complex expressions (field access, functions) fall back to interpreter

- **Hybrid execution model** (`rule_exec.rs` integration)
  - Pattern matching and graph traversal remain interpreted for flexibility
  - Where clause predicates compiled to native code when hot
  - Seamless fallback to interpreter for complex or cold expressions
  - Cache management for compiled kernels with execution counting

- **Comprehensive testing** (`tests/rule_kernels_test.rs`)
  - Tests for simple arithmetic predicates
  - Tests for logical operators (AND, OR, NOT)
  - Verification of compilation threshold mechanism
  - Fallback testing for complex expressions

#### 3. AOT Flow Compilation ✅

- **AOT compilation infrastructure** (`crates/grafial-core/src/engine/aot_flows.rs`)
  - Implemented `FlowCompiler` that compiles FlowIR to native object files
  - Uses Cranelift to generate optimized x86_64 machine code
  - Support for different optimization levels (None, Speed, SpeedAndSize)
  - Generates relocatable object files for linking

- **Build-time integration** (`crates/grafial-core/build.rs`)
  - Build script for compiling flows at build time
  - Automatic discovery of `.gf` flow files
  - Manifest generation for tracking compiled flows
  - Environment variable passing for runtime discovery

- **Runtime integration** (`crates/grafial-core/src/engine/aot_integration.rs`)
  - Global AOT registry for managing compiled flows
  - Automatic fallback to interpreter when compiled version unavailable
  - Hash-based verification of compiled flow validity
  - Runtime JIT compilation support for frequently-used flows

- **Comprehensive testing** (`tests/aot_flows_test.rs`)
  - Tests for flow compilation with different optimization levels
  - Dependency extraction and tracking tests
  - Empty flow edge case testing
  - Metadata generation verification

### All Phase 11 objectives completed!

## Phase 12 - Parallel Engine Execution

**Status: Complete ✅**

### Completed Components:

#### 1. Feature-gated Parallel Evidence Ingestion ✅

- **Parallel observation processing** (`crates/grafial-core/src/engine/parallel_evidence.rs`)
  - Implemented partitioning of observations by target (node/edge)
  - Process partitions in parallel using Rayon thread pool
  - Apply deltas in deterministic order using BTreeMap

- **Integration with evidence module** (`evidence.rs`)
  - Added conditional compilation for parallel path
  - Sequential fallback when feature disabled
  - Maintains deterministic results

#### 2. Dependency-aware Parallel Metric Evaluation ✅

- **Metric dependency analysis** (`crates/grafial-core/src/engine/parallel_metrics.rs`)
  - Build dependency graph from metric definitions
  - Topological sorting for evaluation order
  - Execute independent levels in parallel

- **Integration with flow execution** (`flow_exec.rs`)
  - Modified `evaluate_metrics()` to use parallel evaluation
  - Preserves metric dependencies and ordering
  - Handles circular dependency detection

#### 3. Safe Parallel Rule Application ✅

- **Non-overlapping match detection** (`crates/grafial-core/src/engine/parallel_rules.rs`)
  - Identify rule matches that don't conflict
  - Batch non-overlapping matches for parallel execution
  - Sequential fallback for conflicting matches

- **Integration with transform pipeline** (`flow_exec.rs`)
  - Modified `apply_transform()` for parallel ruleset application
  - Maintains audit trail for debugging
  - Deterministic merge semantics

### Testing and Benchmarks:

- **Integration tests** (`tests/parallel_execution_test.rs`)
  - Verify parallel/sequential parity
  - Test determinism across multiple runs
  - Large-scale batch processing tests

- **Performance benchmarks** (`benches/parallel_execution.rs`)
  - Evidence processing with varying sizes
  - Complex metric evaluation scenarios
  - Rule application scaling tests

### All Phase 12 objectives completed!

## Phase 13 - Graph Storage + Indexing

- Add incremental adjacency index maintenance with rebuild heuristics for large delta batches.
- Add lazy adjacency invalidation/versioning and sparse per-node delta indexes for degree and neighborhood queries.
- Reduce allocation pressure in hot paths:
  - arena allocation for temporary rule-match structures,
  - inline storage for small posterior/attribute collections.
- Evaluate optional storage-model experiments behind feature flags:
  - deeper SoA layouts,
  - workload-specialized graph representations.

Completion notes (this change):
- Integrated incremental adjacency maintenance into `BeliefGraph` with lazy invalidation and rebuild heuristics:
  - enhanced index updates are applied during `apply_delta` structural commits,
  - legacy adjacency cache is invalidated on structural changes and rebuilt lazily.
- Added adjacency query surfaces backed by incremental indexes:
  - `get_outgoing_edges(...)` now prefers the enhanced index,
  - added `out_degree(...)` and `neighbors(...)` index-backed helpers.
- Reduced rule-matching allocation pressure in hot multi-pattern joins:
  - switched recursive temporary match expansion to arena-backed bindings (`Arena`/`ArenaMatchBindings`),
  - conversion to `MatchBindings` now happens only at terminal match emission.
- Added opt-in storage experiment feature flags in `grafial-core`:
  - `storage-experimental`
  - `storage-dense-index` (includes optional dense adjacency construction for high-density graphs),
  - `storage-soa` (adds an SoA edge-index representation for adjacency query evaluation).

### All Phase 13 objectives completed!

## Phase 14 - Numeric Kernels + Equivalence

- Add SIMD-accelerated probability kernels where data shape and platform support justify it (for example larger categorical posterior vectors).
- Require strict numerical equivalence gates for all optimized paths:
  - posterior parity vs reference implementation (within documented epsilon),
  - deterministic parity across repeated runs and backend selections.
- Gate risky optimizations behind feature flags and benchmark thresholds before default enablement.
