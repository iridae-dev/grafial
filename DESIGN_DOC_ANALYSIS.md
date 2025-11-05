

### 2. Prior Sensitivity Analysis ⚠️ **UNIQUE**

**Design Doc Contains (Section 3.2.2):**
- Recommended prior strategies:
  - Weakly informative priors (default)
  - Informative priors (domain knowledge)
  - Skeptical priors (high evidence threshold)
- Rationale for prior choices
- Examples with specific parameter values

**Current Coverage:**
- ❌ Not in LANGUAGE_GUIDE.md
- ❌ Not in examples (examples show syntax but not prior strategy)
- ⚠️ Some guidance in code comments but not systematic

**Recommendation:** **Medium risk** - This is valuable guidance for users. Could:
- Add a "Prior Selection Guide" section to LANGUAGE_GUIDE.md, OR
- Create a separate `PRIOR_GUIDE.md` if extensive

---

### 3. Numerical Stability Details ⚠️ **PARTIALLY COVERED**

**Design Doc Contains (Section 3.2.3):**
- Edge cases for Gaussian posteriors:
  - Infinite precision handling (τ = 10⁶ threshold)
  - Small precision clipping (τ_min = 10⁻⁶)
  - Outlier detection (|x - μ| > 10 × σ)
- Edge cases for Bernoulli posteriors:
  - Near-deterministic beliefs (α or β > 10⁶)
  - Zero/invalid pseudo-counts (enforce α ≥ 0.01, β ≥ 0.01)
  - Log-space computations for large parameters
- Specific threshold values

**Current Coverage:**
- ✅ Thresholds are in code constants (FORCE_PRECISION, MIN_PRECISION, etc.)
- ✅ Edge case handling is in implementation
- ❌ Not documented for users in LANGUAGE_GUIDE.md

**Recommendation:** Low risk - implementation handles this. Could add brief note to LANGUAGE_GUIDE if users need to understand edge cases.

---

### 4. Rust Engine Architecture ⚠️ **UNIQUE**

**Design Doc Contains (Section 5):**
- Detailed module structure and responsibilities
- Data structure design decisions:
  - Stable identifiers (NodeId, EdgeId)
  - Storage layout (SoA-style, contiguous vectors)
  - Structural sharing (Arc + copy-on-write)
- Posterior API design
- Determinism and parallelism strategy
- Error handling strategy
- Serialization approach
- Testing strategy

**Current Coverage:**
- ❌ Not in ROADMAP.md (roadmap focuses on future work, not current architecture)
- ✅ Partially in code comments
- ❌ No architectural overview document

**Recommendation:** **Medium risk** - Valuable for contributors/maintainers. Could:
- Extract to `ARCHITECTURE.md` if team needs it, OR
- Keep minimal version in code comments, OR
- Remove if only end-users need docs (not implementers)

---

### 5. Extensibility Principles ⚠️ **UNIQUE**

**Design Doc Contains (Section 7):**
- Extension mechanisms table
- Design rule: "new capability means new function, not new syntax"
- Registry-based extensibility model

**Current Coverage:**
- ❌ Not documented elsewhere
- ⚠️ Implicit in code structure but not explicit

**Recommendation:** Low risk - mostly for implementers. Could add to ROADMAP.md or ARCHITECTURE.md if created.

---

### 6. Implementation Phases ⚠️ **OUTDATED**

**Design Doc Contains (Section 8):**
- Historical implementation phases (v0, v1, v2)
- Phases are mostly complete

**Current Coverage:**
- ✅ ROADMAP.md has current/future phases
- ❌ No historical record of what was completed

**Recommendation:** Low risk - historical record not essential. Could add brief "Completed" section to ROADMAP.md if needed.

---

### 7. Key Design Rules ⚠️ **UNIQUE**

**Design Doc Contains (Section 9):**
- Graphs are immutable values
- Rules/metrics are pure functions
- Flows define dataflow, not control flow
- Metrics are only sanctioned way for global scalars
- All extensions via function registries

**Current Coverage:**
- ❌ Not explicitly documented elsewhere
- ⚠️ Implicit in language design but not stated

**Recommendation:** Low risk - mostly philosophical. Could add to LANGUAGE_GUIDE.md introduction if helpful.



## Code References to Design Doc

The codebase has **19 references** to `baygraph_design.md`:
- Formula references (update equations)
- Constant definitions (FORCE_PRECISION, MIN_PRECISION)
- Design decision references

**Action Required:** If removing design doc, these references should be updated to point to:
- Code comments (for formulas)
- LANGUAGE_GUIDE.md (for user-facing docs)
- ARCHITECTURE.md (for implementation details, if created)

-