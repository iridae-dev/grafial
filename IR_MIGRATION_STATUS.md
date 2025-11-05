# IR Migration Status and Roadmap

**Last Updated:** 2025-01-XX  
**Status:** IR types implemented, engine still uses AST directly

---

## Current State

### What's Implemented

✅ **IR Types Created:**
- `RuleIR` in `src/ir/rule.rs` - Lowered rule representation
- `FlowIR` in `src/ir/flow.rs` - Lowered flow representation (existed previously)
- `ProgramIR` in `src/ir/program.rs` - Complete program IR aggregating all components

✅ **Lowering Functions:**
- `RuleIR::from(&RuleDef)` - Converts AST rules to IR
- `FlowIR::from(&FlowDef)` - Converts AST flows to IR
- `ProgramIR::from(&ProgramAst)` - Converts entire programs to IR

✅ **Tests:**
- Comprehensive tests for IR lowering
- All IR tests passing (14 tests)

### What's NOT Implemented

❌ **Engine Migration:**
- `flow_exec.rs` still uses `ProgramAst`, `FlowDef`, `GraphExpr`, `Transform` directly
- `rule_exec.rs` still uses `RuleDef`, `PatternItem`, `ActionStmt`, `ExprAst` directly
- `evidence.rs` still uses `EvidenceDef`, `ObserveStmt` directly
- `expr_eval.rs` uses `ExprAst` throughout (no `ExprIR` exists)

❌ **Expression IR:**
- No `ExprIR` type exists
- Expressions are deeply embedded in rules, flows, and evidence
- Would require significant refactoring

---

## Architecture Decision

### Current Architecture

```
Parser → AST → Engine (direct)
         ↓
        IR (available but unused by engine)
```

The engine currently bypasses IR and uses AST directly.

### Proposed Architecture (Future)

```
Parser → AST → IR (lowering) → Engine
                ↑
          Optimizations here
```

The engine would use IR as its stable interface.

---

## Pros and Cons Analysis

### ✅ Pros of Migrating to IR

1. **Decoupling from Frontend**
   - Engine becomes independent of parser/AST changes
   - Frontend can evolve without breaking engine
   - Enables multiple frontends (alternative syntaxes, visual builders)

2. **Optimization Surface**
   - IR can be optimized before execution
   - Can add query plans, constant folding, dead code elimination
   - AST retains parsing artifacts (source locations, etc.)

3. **Stable Interface**
   - IR provides a contract between frontend and engine
   - AST changes don't break engine
   - Easier to version and maintain compatibility

4. **Performance Opportunities**
   - IR can be execution-friendly (pre-indexed patterns)
   - Can eliminate parsing overhead in repeated execution
   - Potential for IR caching/serialization

5. **Testing and Debugging**
   - Can test engine with IR directly (bypass parser)
   - IR is closer to execution semantics, easier to reason about

### ❌ Cons of Migrating to IR

1. **Current Duplication**
   - IR currently mirrors AST (shallow clone)
   - No immediate benefit; adds complexity without gains
   - Current code works fine with AST directly

2. **Significant Migration Effort**
   - Update `flow_exec.rs` (uses `ProgramAst`, `FlowDef`, `GraphExpr`, `Transform`)
   - Update `rule_exec.rs` (uses `RuleDef`, `PatternItem`, `ActionStmt`, `ExprAst`)
   - Update `evidence.rs` (uses `EvidenceDef`, `ObserveStmt`)
   - Expression evaluation (`expr_eval.rs`) uses `ExprAst` throughout
   - Many function signatures and internal logic would change

3. **Expression IR Missing**
   - `ExprAst` is used extensively in evaluation
   - IR doesn't have `ExprIR` yet (would need to create it)
   - Expressions are deeply embedded in rules, flows, evidence
   - This is the largest change surface

4. **Maintenance Burden**
   - Two parallel type systems to maintain
   - Need to keep IR in sync with AST
   - Lowering functions become another surface to test

5. **Limited Immediate Value**
   - No multiple frontends planned currently
   - No IR-level optimizations implemented yet
   - AST is already typed and structured
   - Current performance is acceptable

6. **Breaking Changes**
   - Public APIs (`run_flow`, `run_rule_*`) would change
   - Tests would need significant updates
   - Risk of introducing bugs during migration

---

## Recommended Approach

### Phase 1: Current State (✅ Complete)
- IR types exist with lowering functions
- Engine uses AST directly
- IR available for future use

### Phase 2: Gradual Migration (When Needed)

**Option A: Full Migration (Recommended When)**
- Multiple frontends needed
- IR-level optimizations required (query plans, constant folding)
- AST changes frequently breaking engine

**Option B: Hybrid Approach (Pragmatic)**
- Keep AST for expressions (most complex, least benefit)
- Migrate only hot paths (pattern matching, rule execution)
- Use IR for new features that need optimization
- Gradually migrate as needed

### Phase 3: IR Optimizations (Future)
- Query plan optimization using IR
- Constant folding
- Dead code elimination
- Pattern matching optimization

---

## Migration Roadmap (If Needed)

### Step 1: Expression IR (Prerequisite)
- Create `ExprIR` type
- Implement `ExprAst` → `ExprIR` lowering
- Update expression evaluation to handle both AST and IR
- **Estimated Effort:** High (expressions are pervasive)

### Step 2: Evidence IR
- Create `EvidenceIR` type
- Update `evidence.rs` to use `EvidenceIR`
- **Estimated Effort:** Medium

### Step 3: Rule Execution Migration
- Update `rule_exec.rs` to use `RuleIR` instead of `RuleDef`
- Update pattern matching to use IR
- Update action execution to use IR
- **Estimated Effort:** High

### Step 4: Flow Execution Migration
- Update `flow_exec.rs` to use `FlowIR` and `ProgramIR`
- Update graph expression evaluation
- Update transform application
- **Estimated Effort:** Medium

### Step 5: Public API Updates
- Update public functions to accept IR
- Provide convenience functions that accept AST and lower internally
- Update all tests
- **Estimated Effort:** Medium

### Total Estimated Effort: **High** (several weeks of focused work)

---

## Decision Criteria

### Migrate to IR When:
- ✅ Multiple frontends are needed
- ✅ IR-level optimizations are required
- ✅ AST changes are frequently breaking engine
- ✅ Performance benefits from IR optimization are significant
- ✅ Team has capacity for large refactoring

### Defer Migration When:
- ❌ Single frontend is sufficient
- ❌ AST works well for current needs
- ❌ No immediate performance concerns
- ❌ Team focused on feature development
- ❌ Breaking changes are unacceptable

---

## Current Recommendation

**Defer migration for now.** The IR infrastructure exists and is tested, but the engine works well with AST directly. The migration would be a significant effort with limited immediate benefit.

**Use IR when:**
1. Query plan optimization is implemented (Priority 4.1)
2. Multiple frontends are needed
3. AST changes are causing engine breakage

**Alternative:**
- Keep IR as-is (foundation exists)
- Use IR only for new features that need optimization
- Gradually migrate hot paths (e.g., pattern matching) to IR-specific optimized forms
- Consider IR as an optimization layer rather than a replacement

---

## Related Documents

- `baygraph_design.md:422-426` - IR design specification
- `baygraph_roadmap_remaining.md` - Priority 7: IR Lowering (marked complete)
- `src/ir/` - IR implementation

---

## Notes

- IR lowering functions are complete and tested
- IR types are ready for use
- Engine can be migrated incrementally when needed
- No breaking changes required to adopt IR gradually
- IR provides a clean separation of concerns for future optimization

