# Work in Progress: Grafial Language Redesign

**Status**: Mid-migration — Grammar and parser updated, examples partially migrated, engine semantics in progress

**Date**: Current session

## Overview

We are in the middle of implementing a major language redesign based on `final-language-suggestions.md` and tracked in `final-language-implementation-roadmap.md`. The goal is to make Grafial more fluent, probabilistic-first, and tooling-friendly.

## What's Been Completed

### Phase 1: Grammar + AST (✅ Completed)

**Grammar Changes (`grammar.pest`)**:
- ✅ Shortened posterior type names: `GaussianPosterior` → `Gaussian`, `BernoulliPosterior` → `Bernoulli`, `CategoricalPosterior` → `Categorical`
- ✅ Parameter name normalization: `prior_mean` → `mean`, `prior_precision` → `precision`, `pseudo_count` → `weight`
- ✅ Node reference syntax: Added support for `Label("id")`, `Label(id)`, and `Label[42]` forms (keeping original bracket syntax)
- ✅ Evidence grouping syntax:
  - Node groups: `Entity { "A" { value: 1.0 } }`
  - Edge groups: `CONNECTED(Entity -> Entity) { "A" -> "B" }` (present) and `"A" -/> "B"` (absent)
  - Precision annotations: `value: 1.0 (precision=10.0)`
- ✅ Categorical evidence verbs: `choose` and `unchoose` statements
- ✅ Rule syntax sugar:
  - Pattern → action form: `pattern => { actions }`
  - Node-only iteration: `for (Var:Label) [where expr] => { actions }`
- ✅ New action statements:
  - `non_bayesian_nudge X to expr [variance=preserve|increase|decrease]`
  - `X ~= value [precision=τ] [count=n]` (Bayesian soft update)
  - `delete e [confidence=...]`
  - `suppress e [weight=...]`
- ✅ Metrics builder pipeline: `nodes(Label) |> where(...) |> sum(by=...) |> count() |> avg(by=...) |> fold(...)`

**Parser/AST Changes**:
- ✅ Extended AST nodes for all new constructs
- ✅ Source span tracking for diagnostics
- ✅ Validation: Require explicit uncertainty in `where` clauses (reject bare field access)

**Examples Migration**:
- ✅ All example files updated to new syntax:
  - Posterior types shortened (`Gaussian` instead of `GaussianPosterior`)
  - Parameter names normalized (`mean` instead of `prior_mean`)
  - Evidence syntax changed to grouped form
  - Metrics updated to builder pipeline syntax

**Documentation**:
- ✅ `LANGUAGE_GUIDE.md` updated with new syntax examples
- ✅ `final-language-suggestions.md` created (design document)
- ✅ `final-language-implementation-roadmap.md` created (implementation plan)

### VS Code Extension

**New Files**:
- ✅ `extension.js` - Main extension entry point with LSP client
- ✅ `webpack.config.js` - Build configuration
- ✅ `package.json` - Updated with LSP dependencies
- ✅ `package-lock.json` - Dependency lockfile
- ✅ Documentation files: `CURSOR_SETUP.md`, `DEBUG_LSP.md`, `QUICK_TEST.md`, `TESTING_STEPS.md`, `TROUBLESHOOTING.md`

**Changes**:
- ✅ Extension now uses `vscode-languageclient` for LSP integration
- ✅ Server path resolution supports `~` expansion and relative paths
- ✅ Packaged as `grafial-0.1.0.vsix`

**Removed**:
- ❌ `install-cursor.sh` (replaced by VSIX installation)

## What's In Progress

### Phase 2: Engine Semantics (🚧 In Progress)

**Status**: Variance/stddev/quantile done; CI/effective_n + action semantics next

**Engine Changes (`crates/grafial-core/src/engine/`)**:
- 🚧 `graph.rs` - Major refactoring (392 lines changed)
- 🚧 `rule_exec.rs` - Extended for new action types (313 lines added)
- 🚧 `evidence.rs` - Updated for grouped evidence syntax
- 🚧 `expr_utils.rs` - New utility functions (87 lines added)
- 🚧 `metrics/mod.rs` - Builder API and new statistical functions (146 lines added)

**Pending Implementation**:
- [ ] `non_bayesian_nudge` variance semantics (`preserve`, `increase(f)`, `decrease(f)`)
- [ ] Bayesian soft update `~=` (Normal-Normal updates)
- [ ] `delete` and `suppress` edge operations
- [ ] Uncertainty primitives: `prob()`, `credible()`, `ci()`, `effective_n()`
- [ ] Metrics builder runtime (desugaring to existing functions)

## What's Not Started

### Phase 3: LSP & Lints
- [ ] LSP diagnostics for new syntax
- [ ] Quick fixes for common mistakes
- [ ] Hover documentation
- [ ] Code actions

### Phase 4: Examples & Docs
- [ ] All examples fully migrated and tested
- [ ] Documentation complete
- [ ] Migration guide for existing code

### Phase 5: Cleanup & Release
- [ ] Remove deprecated syntax
- [ ] Final testing
- [ ] Release preparation

## Breaking Changes Summary

### Syntax Changes

1. **Posterior Types**:
   - `GaussianPosterior` → `Gaussian`
   - `BernoulliPosterior` → `Bernoulli`
   - `CategoricalPosterior` → `Categorical`

2. **Parameter Names**:
   - `prior_mean` → `mean`
   - `prior_precision` → `precision`
   - `pseudo_count` → `weight`

3. **Evidence Syntax**:
   - Old: `observe Entity["A"].value = 1.0`
   - New: `Entity { "A" { value: 1.0 } }`
   - Old: `observe edge CONNECTED(Entity["A"], Entity["B"]) present`
   - New: `CONNECTED(Entity -> Entity) { "A" -> "B" }`

4. **Metrics**:
   - Old: `sum_nodes(label=Entity, contrib=E[node.value])`
   - New: `nodes(Entity) |> sum(by=E[node.value])`

5. **Rules**:
   - New sugar forms: `pattern => { actions }` and `for (Var:Label) => { actions }`

## Files Changed

### Core Language
- `crates/grafial-frontend/grammar.pest` - Grammar definitions
- `crates/grafial-frontend/src/parser.rs` - Parser implementation
- `crates/grafial-frontend/src/ast.rs` - AST types
- `crates/grafial-frontend/src/validate.rs` - Validation logic

### Engine
- `crates/grafial-core/src/engine/graph.rs` - Graph operations
- `crates/grafial-core/src/engine/rule_exec.rs` - Rule execution
- `crates/grafial-core/src/engine/evidence.rs` - Evidence handling
- `crates/grafial-core/src/engine/expr_utils.rs` - Expression utilities
- `crates/grafial-core/src/metrics/mod.rs` - Metrics system

### Examples (All Updated)
- All `.grafial` files in `crates/grafial-examples/`

### Documentation
- `documentation/LANGUAGE_GUIDE.md` - Updated with new syntax
- `documentation/final-language-suggestions.md` - Design document (new)
- `documentation/final-language-implementation-roadmap.md` - Implementation plan (new)
- `documentation/llvm-optimization-guide.md` - Optimization guide (new)

### VS Code Extension
- `crates/grafial-vscode/extension.js` - Main extension (new)
- `crates/grafial-vscode/webpack.config.js` - Build config (new)
- `crates/grafial-vscode/package.json` - Updated dependencies
- `crates/grafial-vscode/README.md` - Updated documentation
- Various setup/troubleshooting docs (new)

### Other
- `Cargo.toml` - Dependency updates
- `README.md` - Updated examples
- `crates/grafial-python/src/lib.rs` - Minor updates
- `crates/grafial-tests/tests/evidence_building_tests.rs` - Test updates

## Next Steps

1. **Complete Phase 2**: Finish engine semantics for new actions and uncertainty primitives
2. **Test Migration**: Ensure all examples parse and run correctly with new syntax
3. **LSP Integration**: Add diagnostics and code actions for new syntax
4. **Documentation**: Complete migration guide and update all docs
5. **Cleanup**: Remove deprecated syntax and finalize API

## Notes

- The codebase is in a semi-broken state during migration
- Examples have been updated to new syntax but may not all run correctly yet
- Engine changes are substantial and may need additional testing
- VS Code extension has been rebuilt but LSP features are not yet complete
- This is a breaking change - no backwards compatibility is maintained

## Commit Strategy

When ready to commit:
1. Stage all changes (grammar, parser, examples, docs, engine)
2. Use conventional commit: `feat(lang): implement v1 language redesign`
3. Include this WIP document in the commit message or as a reference
4. Consider creating a branch for the migration if not already done

