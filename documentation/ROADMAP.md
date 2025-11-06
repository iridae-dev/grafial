# Grafial Development Roadmap

## High Priority

### Parallel Evidence Application

**Location:** `crates/grafial-core/src/engine/evidence.rs`

**Problem:** Evidence application is sequential but embarrassingly parallel. Each observation update is independent.

**Solution:** Use rayon for parallel processing (feature-gated). Collect all deltas in parallel, apply sequentially for determinism. Partition observations into independent batches.

**Impact:** 2-4x throughput for large evidence sets (10k+ observations)  
**Risk:** Medium - requires careful synchronization to preserve determinism and Bayesian correctness

---

### Parallel Metric Computation

**Location:** `crates/grafial-core/src/metrics/mod.rs`, `crates/grafial-core/src/engine/flow_exec.rs`

**Problem:** Sequential metric evaluation even when metrics are independent.

**Solution:** Build dependency graph and parallelize independent metrics. Use topological sort to identify levels of independent metrics. Evaluate metrics in each level in parallel. Preserve evaluation order for dependent metrics.

**Impact:** 2-8x speedup for flows with many independent metrics (10+ metrics)  
**Risk:** Medium - requires dependency analysis, must preserve evaluation order

---

### Parallel Rule Application

**Location:** `crates/grafial-core/src/engine/rule_exec.rs`

**Problem:** Sequential rule execution even when patterns don't overlap.

**Solution:** Detect independence and parallelize. Partition matches by affected graph elements. Process independent partitions in parallel. Merge deltas sequentially (deterministic order). Use graph coloring to identify independent matches.

**Impact:** 2-4x speedup for rules with many independent matches  
**Risk:** High - requires careful dependency analysis to preserve Bayesian correctness and determinism

---

## Medium Priority

### Incremental Adjacency Index Updates

**Location:** `crates/grafial-core/src/engine/graph.rs`

**Problem:** Adjacency index is rebuilt from scratch on every mutation.

**Solution:** Maintain adjacency index incrementally through deltas. Track dirty state for incremental updates. Apply delta changes incrementally (O(1) amortized for small deltas). Rebuild only when delta is large (heuristic: >10% of edges).

**Impact:** 70-90% reduction in adjacency rebuild cost for small deltas (common case)  
**Risk:** High - complex implementation, requires careful correctness testing

---

### Arena Allocation for Temporary Structures

**Location:** `crates/grafial-core/src/engine/rule_exec.rs`

**Problem:** Repeated allocations for temporary match bindings.

**Solution:** Use arena allocator (bumpalo) for temporary structures. Allocate match bindings in arena during pattern matching. All freed at once when arena drops. Zero-copy sharing of temporary data structures.

**Impact:** 30-50% reduction in allocation overhead for multi-pattern rules  
**Risk:** Medium - requires lifetime management, adds dependency

---

### Structure-of-Arrays (SoA) Layout

**Location:** `crates/grafial-core/src/engine/graph.rs`

**Problem:** Array-of-Structures (AoS) layout causes cache misses.

**Solution:** Separate hot and cold data using SoA layout. Hot data: node_ids, node_labels (aligned vectors). Cold data: node_attrs (HashMap, lazy-loaded). Similar restructuring for edges.

**Impact:** 40-60% improvement in iteration-heavy operations  
**Risk:** Very high - fundamental restructuring, affects all code, extensive testing required

---

### Specialized Graph Formats

**Location:** `crates/grafial-core/src/engine/graph.rs`

**Problem:** One-size-fits-all graph representation is suboptimal for different access patterns.

**Solution:** Polymorphic graph representations with runtime selection. `MutableGraph` - Optimized for writes (delta-heavy, lazy indexing). `ImmutableGraph` - Optimized for reads (pre-built indexes, SoA layout). `HybridGraph` - Current default (Arc-based structural sharing). Auto-optimize based on usage heuristics.

**Impact:** 50-100% improvement for workload-specific operations  
**Risk:** Very high - fundamental architectural change, requires extensive refactoring

---

## Low Priority

### Sparse Index for Delta Degree Computation

**Location:** `crates/grafial-core/src/engine/graph.rs`

**Problem:** Linear scan through deltas (O(E + D) complexity).

**Solution:** Build sparse per-node delta index. Lazy-built index when delta size > threshold. O(neighbors) lookup instead of O(E + D) scan. Index invalidated on delta application.

**Impact:** 80-95% improvement for degree queries with large deltas  
**Risk:** Medium - adds complexity, requires careful index invalidation

---

### Lazy Adjacency Updates with Versioning

**Location:** `crates/grafial-core/src/engine/graph.rs`

**Problem:** Adjacency index invalidated on every mutation, even if not used.

**Solution:** Version-based lazy invalidation. Track version number, increment on mutations. Check version before using cached index. Rebuild only when needed.

**Impact:** Eliminates unnecessary adjacency rebuilds (saves 60-80% of rebuild overhead)  
**Risk:** Low - simple versioning scheme, easy to test

---

### Inline Storage for Small Posterior Collections

**Location:** `crates/grafial-core/src/engine/graph.rs`

**Problem:** HashMap allocation even for nodes with 1-2 attributes.

**Solution:** Use SmallVec or inline storage. `SmallVec<[(Arc<str>, GaussianPosterior); 4]>` - Inline up to 4 attrs. Or custom inline map with overflow to HashMap.

**Impact:** 30-40% reduction in allocations for typical graphs (avg 2-3 attributes/node)  
**Risk:** Medium - requires changing data structure, affects all attribute access

---

## Optional

### SIMD-Accelerated Probability Computations

**Location:** `crates/grafial-core/src/engine/graph.rs`

**Problem:** Scalar probability computations in tight loops.

**Solution:** Use SIMD (AVX2/NEON) for vectorized probability computation. Process 4 f64s at a time with AVX2. Horizontal sum for totals. Vectorized division.

**Impact:** 2-4x speedup for probability vector computations (critical in metrics and entropy)  
**Risk:** High - requires unsafe code, platform-specific, needs extensive testing  
**Note:** Only beneficial for large categorical posteriors (K > 16)

---

## IR Migration

### Current State

IR types exist with lowering functions, but engine still uses AST directly. IR infrastructure is available for future use.

### Migration Decision

**Migrate to IR when:**
- Multiple frontends are needed
- IR-level optimizations are required (query plans, constant folding)
- AST changes are frequently breaking engine
- Performance benefits from IR optimization are significant

**Defer migration when:**
- Single frontend is sufficient
- AST works well for current needs
- No immediate performance concerns
- Team focused on feature development

**Current recommendation:** Defer migration. IR infrastructure exists and is tested, but engine works well with AST directly.

### Migration Steps (If Needed)

1. **Expression IR (Prerequisite)**: Create `ExprIR` type, implement `ExprAst` → `ExprIR` lowering, update expression evaluation to handle both AST and IR
2. **Evidence IR**: Create `EvidenceIR` type, update `evidence.rs` to use `EvidenceIR`
3. **Rule Execution Migration**: Update `rule_exec.rs` to use `RuleIR` instead of `RuleDef`, update pattern matching and action execution to use IR
4. **Flow Execution Migration**: Update `flow_exec.rs` to use `FlowIR` and `ProgramIR`, update graph expression evaluation and transform application
5. **Public API Updates**: Update public functions to accept IR, provide convenience functions that accept AST and lower internally, update all tests

### Proposed Architecture

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

## Critical Constraint

All optimizations must produce **identical numerical results** to current implementation. Preserve Bayesian mathematical correctness and determinism.

**Testing strategy:**
- Compare optimized vs. reference implementation
- Verify identical Bayesian posteriors (within floating-point epsilon)
- Test determinism (100+ runs with identical results)
- Use property-based testing for edge cases
- Use feature flags to gate risky optimizations
