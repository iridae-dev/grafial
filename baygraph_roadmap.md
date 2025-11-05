## Python bindings

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
