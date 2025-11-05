## 1. Competing Edges, Generally as designed in competing_edges_design, but also:
Dirichlet–Categorical posterior (v1): (see competing_edges_design.md, baygraph_design.md:144-158)
	- Add Categorical edges grouped by source (see competing_edges_design.md:§3, §7)
	- Implement `CategoricalPosterior` with Dirichlet updates (see competing_edges_design.md:§3.1)
	- Update query functions (`prob`, `degree`) to handle competing edges (see competing_edges_design.md:§5.1)
	- Add new intrinsics: `winner()`, `entropy()`, `prob_vector()` (see competing_edges_design.md:§5.2)
	- Implement evidence keywords: `chosen`, `unchosen`, `forced_choice` (see competing_edges_design.md:§4)
	- Update metrics and queries to surface vector means when applicable.



## 2. Python bindings

Goal: Make this usable without writing Rust.

Tasks:

- Add bindings::python module using pyo3:
	- pyclass Program
	- pyclass BeliefGraph
	- pyclass Evidence
	- pyclass Context

Expose:

```
compile(source: str) -> Program
run_flow(program, flow_name: str) -> Context
run_flow_with_evidence(program, flow_name: str, evidence, ctx=None) -> Context
```

- BeliefGraph Python API:
	- nodes(label=None) → iterable of node views.
	- edges(type=None) → iterable of edge views.
	- Node view: .id, .E(attr), .Var(attr).
	- Edge view: .src, .dst, .prob, .forced_state.

- Convenience exports:
	- to_pandas() for nodes/edges.
	- to_networkx(threshold=...) for quick analysis.
- Build and test via maturin.

Implementation details:

- Release the GIL for long‑running operations; map `ExecError` to Python exceptions.
- Preserve immutability semantics: mutating methods return new graph handles; old ones stay valid.

Exit criteria:

- From a Python script, you can:
	- compile() a .bg file,
	- build an Evidence object,
	- run a flow,
	- read metrics and inspect resulting graph.


## 3. Snapshots and serialization: (see baygraph_design.md:539-544)
	- Graphs are immutable between transforms via structural sharing (Arc + copy-on-write). (see baygraph_design.md:468-474)
	- Represent `GraphView` as `{ base: Arc<BeliefGraphInner>, delta: SmallVec<...> }` so most steps are O(changes).
	- Derive `serde::{Serialize, Deserialize}` for IR and posterior types for snapshots/checkpointing. (see baygraph_design.md:541)
	- Record engine/version metadata and registry hashes inside snapshots for compatibility checks. (see baygraph_design.md:542)
	- If randomized transforms added later, route RNG via explicit `Seed` in `ExecutionContext`. (see baygraph_design.md:543-544)