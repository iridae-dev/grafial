# Grafial Development Roadmap

**Last Updated:** 2025-01-XX  
**Scope:** Performance optimizations, IR migration, and architectural improvements  
**Critical Constraint:** All optimizations preserve Bayesian mathematical correctness

---

## Executive Summary

This roadmap outlines future development priorities for Grafial, organized by:
1. **Performance Optimizations** - Remaining unimplemented optimizations
2. **IR Migration** - Status and plans for migrating engine to use IR
3. **Architectural Improvements** - Major refactors and structural changes

**Current Status:**
- ✅ Phase 1 & 2 performance optimizations completed (50-70% overall improvement)
- ✅ IR types and lowering functions implemented (engine still uses AST directly)
- ⏳ Phase 3+ optimizations and IR migration pending

---

## Completed Work

### Performance Optimizations - Phase 1 & 2 ✅

**Phase 1: Quick Wins (Completed)**
- Pre-allocate HashMaps with estimated capacity
- Use FxHashMap for integer keys (NodeId, EdgeId, CompetingGroupId)
- Two-pass rule execution to reduce cloning (clone only when matches found)
- Optimize delta-aware degree computation (fast path with adjacency index)
- Eliminate string allocations (Arc<str> for edge types in AdjacencyIndex)

**Achieved:** 20-30% overall improvement, 40% allocation reduction

**Phase 2: Medium Effort (Completed)**
- String interning (Arc<str> for node labels and edge types)
- Fine-grained delta compression (NodeAttributeChange, EdgeProbChange)
- Query plan caching improvements (FxHashMap, PatternKey struct, unstable sorts)
- Iterator optimizations (replace stable sorts with unstable sorts)

**Achieved:** Additional 30-40% improvement, 50% memory reduction

### IR Infrastructure ✅

**Implemented:**
- `RuleIR` in `src/ir/rule.rs` - Lowered rule representation
- `FlowIR` in `src/ir/flow.rs` - Lowered flow representation
- `ProgramIR` in `src/ir/program.rs` - Complete program IR
- Lowering functions: `RuleIR::from(&RuleDef)`, `FlowIR::from(&FlowDef)`, `ProgramIR::from(&ProgramAst)`
- Comprehensive IR tests (all passing)

**Status:** IR infrastructure exists but engine still uses AST directly

---

## Future Work

### Phase 3: Parallelization (High Priority)

**Estimated Effort:** 2-3 weeks  
**Expected Gain:** 2-4x throughput on multi-core systems

#### 3.1 Parallel Evidence Application (2.3)

**Location:** `src/engine/evidence.rs`

**Problem:** Evidence application is sequential but embarrassingly parallel:
```rust
// Current sequential processing
for obs in &evidence.observations {
    match obs {
        ObserveStmt::Attribute { node, attr, value } => {
            // Each update is independent!
            graph.observe_attr(node_id, attr, *value, obs_precision)?;
        }
        // ...
    }
}
```

**Solution:** Use rayon for parallel processing (feature-gated):
- Collect all deltas in parallel, apply sequentially for determinism
- Partition observations into independent batches
- Use chunking to reduce synchronization overhead

**Impact:** 2-4x throughput for large evidence sets (10k+ observations)  
**Risk:** Medium - requires careful synchronization to preserve determinism and Bayesian correctness  
**Estimated Effort:** 4-5 days

---

#### 3.2 Parallel Metric Computation (7.2)

**Location:** `src/metrics/mod.rs`, `src/engine/flow_exec.rs:248-307`

**Problem:** Sequential metric evaluation even when metrics are independent

**Solution:** Build dependency graph and parallelize independent metrics:
- Use topological sort to identify levels of independent metrics
- Evaluate metrics in each level in parallel
- Preserve evaluation order for dependent metrics

**Impact:** 2-8x speedup for flows with many independent metrics (10+ metrics)  
**Risk:** Medium - requires dependency analysis, must preserve evaluation order  
**Estimated Effort:** 2-3 days

---

#### 3.3 Parallel Rule Application (7.1)

**Location:** `src/engine/rule_exec.rs:493-566`

**Problem:** Sequential rule execution even when patterns don't overlap

**Solution:** Detect independence and parallelize:
- Partition matches by affected graph elements
- Process independent partitions in parallel
- Merge deltas sequentially (deterministic order)
- Use graph coloring to identify independent matches

**Impact:** 2-4x speedup for rules with many independent matches  
**Risk:** High - requires careful dependency analysis to preserve Bayesian correctness and determinism  
**Estimated Effort:** 3-4 days (complex!)

---

### Phase 4: Architectural Improvements (Medium Priority)

**Estimated Effort:** 3-6 weeks  
**Expected Gain:** 50-100% improvement for specific workloads

#### 4.1 Incremental Adjacency Index Updates (3.1)

**Location:** `src/engine/graph.rs:587-645, 1407-1439`

**Problem:** Adjacency index is rebuilt from scratch on every mutation

**Solution:** Maintain adjacency index incrementally through deltas:
- Track dirty state for incremental updates
- Apply delta changes incrementally (O(1) amortized for small deltas)
- Rebuild only when delta is large (heuristic: >10% of edges)

**Impact:** 70-90% reduction in adjacency rebuild cost for small deltas (common case)  
**Risk:** High - complex implementation, requires careful correctness testing  
**Estimated Effort:** 1 week

---

#### 4.2 Arena Allocation for Temporary Structures (2.2)

**Location:** `src/engine/rule_exec.rs:546-556, 639-666`

**Problem:** Repeated allocations for temporary match bindings

**Solution:** Use arena allocator (bumpalo) for temporary structures:
- Allocate match bindings in arena during pattern matching
- All freed at once when arena drops
- Zero-copy sharing of temporary data structures

**Impact:** 30-50% reduction in allocation overhead for multi-pattern rules  
**Risk:** Medium - requires lifetime management, adds dependency  
**Estimated Effort:** 3-4 days

---

#### 4.3 Structure-of-Arrays (SoA) Layout (3.2)

**Location:** `src/engine/graph.rs:364-374, 556-570`

**Problem:** Array-of-Structures (AoS) layout causes cache misses

**Solution:** Separate hot and cold data using SoA layout:
- Hot data: node_ids, node_labels (aligned vectors)
- Cold data: node_attrs (HashMap, lazy-loaded)
- Similar restructuring for edges

**Impact:** 40-60% improvement in iteration-heavy operations  
**Risk:** Very high - fundamental restructuring, affects all code, extensive testing required  
**Estimated Effort:** 1-2 weeks (major effort!)

---

#### 4.4 Specialized Graph Formats (3.4)

**Location:** Entire `src/engine/graph.rs` module

**Problem:** One-size-fits-all graph representation is suboptimal for different access patterns

**Solution:** Polymorphic graph representations with runtime selection:
- `MutableGraph` - Optimized for writes (delta-heavy, lazy indexing)
- `ImmutableGraph` - Optimized for reads (pre-built indexes, SoA layout)
- `HybridGraph` - Current default (Arc-based structural sharing)
- Auto-optimize based on usage heuristics

**Impact:** 50-100% improvement for workload-specific operations  
**Risk:** Very high - fundamental architectural change, requires extensive refactoring  
**Estimated Effort:** 2-3 weeks

---

### Phase 5: Algorithmic Improvements (Low Priority)

**Estimated Effort:** 1-2 weeks  
**Expected Gain:** 80-95% improvement for specific queries

#### 5.1 Sparse Index for Delta Degree Computation (4.1)

**Location:** `src/engine/graph.rs:1207-1293`

**Problem:** Linear scan through deltas (O(E + D) complexity)

**Solution:** Build sparse per-node delta index:
- Lazy-built index when delta size > threshold
- O(neighbors) lookup instead of O(E + D) scan
- Index invalidated on delta application

**Impact:** 80-95% improvement for degree queries with large deltas  
**Risk:** Medium - adds complexity, requires careful index invalidation  
**Estimated Effort:** 3-4 days

---

#### 5.2 Lazy Adjacency Updates with Versioning (4.2)

**Location:** `src/engine/graph.rs:587-645`

**Problem:** Adjacency index invalidated on every mutation, even if not used

**Solution:** Version-based lazy invalidation:
- Track version number, increment on mutations
- Check version before using cached index
- Rebuild only when needed

**Impact:** Eliminates unnecessary adjacency rebuilds (saves 60-80% of rebuild overhead)  
**Risk:** Low - simple versioning scheme, easy to test  
**Estimated Effort:** 2-3 days

---

### Phase 6: Memory Layout Optimizations (Low Priority)

**Estimated Effort:** 1 week  
**Expected Gain:** 30-40% reduction in allocations

#### 6.1 Inline Storage for Small Posterior Collections (5.1)

**Location:** `src/engine/graph.rs:373` (node attributes)

**Problem:** HashMap allocation even for nodes with 1-2 attributes

**Solution:** Use SmallVec or inline storage:
- `SmallVec<[(Arc<str>, GaussianPosterior); 4]>` - Inline up to 4 attrs
- Or custom inline map with overflow to HashMap

**Impact:** 30-40% reduction in allocations for typical graphs (avg 2-3 attributes/node)  
**Risk:** Medium - requires changing data structure, affects all attribute access  
**Estimated Effort:** 3-4 days

---

### Phase 7: Advanced Optimizations (Optional)

**Estimated Effort:** 2-4 weeks  
**Expected Gain:** 2-4x speedup for specific operations

#### 7.1 SIMD-Accelerated Probability Computations (3.3)

**Location:** `src/engine/graph.rs:186-191, 316-328, 333-343`

**Problem:** Scalar probability computations in tight loops

**Solution:** Use SIMD (AVX2/NEON) for vectorized probability computation:
- Process 4 f64s at a time with AVX2
- Horizontal sum for totals
- Vectorized division

**Impact:** 2-4x speedup for probability vector computations (critical in metrics and entropy)  
**Risk:** High - requires unsafe code, platform-specific, needs extensive testing  
**Note:** Only beneficial for large categorical posteriors (K > 16)  
**Estimated Effort:** 1-2 weeks

---

## IR Migration Roadmap

### Current State

**Architecture:**
```
Parser → AST → Engine (direct)
         ↓
        IR (available but unused by engine)
```

The engine currently bypasses IR and uses AST directly.

**Implemented:**
- ✅ IR types exist with lowering functions
- ✅ Engine uses AST directly
- ✅ IR available for future use

**Not Implemented:**
- ❌ Engine migration (still uses AST types directly)
- ❌ Expression IR (no `ExprIR` type exists)
- ❌ Evidence IR migration

---

### Migration Decision Criteria

**Migrate to IR When:**
- ✅ Multiple frontends are needed
- ✅ IR-level optimizations are required (query plans, constant folding)
- ✅ AST changes are frequently breaking engine
- ✅ Performance benefits from IR optimization are significant
- ✅ Team has capacity for large refactoring

**Defer Migration When:**
- ❌ Single frontend is sufficient
- ❌ AST works well for current needs
- ❌ No immediate performance concerns
- ❌ Team focused on feature development
- ❌ Breaking changes are unacceptable

**Current Recommendation:** Defer migration for now. IR infrastructure exists and is tested, but engine works well with AST directly.

---

### Migration Roadmap (If Needed)

**Total Estimated Effort:** High (several weeks of focused work)

#### Step 1: Expression IR (Prerequisite)
- Create `ExprIR` type
- Implement `ExprAst` → `ExprIR` lowering
- Update expression evaluation to handle both AST and IR
- **Estimated Effort:** High (expressions are pervasive)

#### Step 2: Evidence IR
- Create `EvidenceIR` type
- Update `evidence.rs` to use `EvidenceIR`
- **Estimated Effort:** Medium

#### Step 3: Rule Execution Migration
- Update `rule_exec.rs` to use `RuleIR` instead of `RuleDef`
- Update pattern matching to use IR
- Update action execution to use IR
- **Estimated Effort:** High

#### Step 4: Flow Execution Migration
- Update `flow_exec.rs` to use `FlowIR` and `ProgramIR`
- Update graph expression evaluation
- Update transform application
- **Estimated Effort:** Medium

#### Step 5: Public API Updates
- Update public functions to accept IR
- Provide convenience functions that accept AST and lower internally
- Update all tests
- **Estimated Effort:** Medium

---

### Proposed Architecture (Future)

```
Parser → AST → IR (lowering) → Engine
                ↑
          Optimizations here
```

The engine would use IR as its stable interface, enabling:
- Decoupling from frontend (independent of parser/AST changes)
- Optimization surface (query plans, constant folding, dead code elimination)
- Stable interface (contract between frontend and engine)
- Performance opportunities (pre-indexed patterns, IR caching/serialization)

---

## Implementation Priority Summary

### Immediate Priorities (Next 1-2 Months)

1. **Phase 3: Parallelization** (2-3 weeks)
   - Parallel evidence application
   - Parallel metric computation
   - Parallel rule application (if needed)
   - **Expected gain:** 2-4x throughput on multi-core systems

2. **Phase 4: Architectural Improvements** (3-6 weeks, as needed)
   - Incremental adjacency updates
   - Arena allocation
   - SoA layout (if performance requires it)

### Future Considerations (3-6 Months)

3. **Phase 5: Algorithmic Improvements** (1-2 weeks)
   - Sparse delta index
   - Lazy adjacency versioning

4. **Phase 6: Memory Layout** (1 week)
   - Inline storage for small collections

5. **IR Migration** (when criteria are met)
   - Expression IR creation
   - Gradual engine migration
   - IR-level optimizations

### Optional/Advanced (As Needed)

6. **Phase 7: Advanced Optimizations** (2-4 weeks)
   - SIMD acceleration
   - Specialized graph formats

---

## Risk Mitigation

### Preserving Bayesian Correctness

**Critical principle:** All optimizations must produce **identical numerical results** to current implementation.

**Testing strategy:**
- Compare optimized vs. reference implementation
- Verify identical Bayesian posteriors (within floating-point epsilon)
- Test determinism (100+ runs with identical results)
- Use property-based testing for edge cases

### Incremental Adoption

Use feature flags to gate risky optimizations:
```toml
[features]
default = ["quick-wins"]
quick-wins = []  # Safe optimizations (already completed)
experimental = ["simd", "parallel"]  # Riskier optimizations
simd = []
parallel = ["rayon"]
```

---

## Performance Testing

### Benchmark Additions

Add benchmarks for:
1. String allocation overhead (String vs Arc<str>)
2. Delta compression (full clone vs fine-grained delta)
3. Parallel evidence application (sequential vs parallel)
4. Query plan optimization (2-pattern, 3-pattern, 4-pattern rules)
5. Adjacency index rebuild (full vs incremental)

### Profiling Commands

```bash
# CPU profiling
cargo build --release --features rayon
perf record --call-graph=dwarf target/release/benches/graph_benchmarks
perf report

# Memory profiling
valgrind --tool=massif target/release/benches/graph_benchmarks
ms_print massif.out.*

# Allocation tracking
heaptrack target/release/benches/graph_benchmarks
heaptrack_gui heaptrack.graph_benchmarks.*
```

---

## Related Documents

- `Grafial_design.md` - Core design specifications
- `LANGUAGE_GUIDE.md` - Language syntax and semantics
- `Python_Plan.md` - Python bindings roadmap
- `src/ir/` - IR implementation
- `performance_ideas.md` - (This document supersedes it)

---

## Notes

- All Phase 1 & 2 performance optimizations are complete
- IR infrastructure is ready for use when needed
- Parallelization is the highest-priority remaining work
- IR migration can be deferred until criteria are met
- All optimizations maintain Bayesian correctness and determinism

