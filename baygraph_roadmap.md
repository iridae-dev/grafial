# Bayesian Belief Graph DSL — High Level Roadmap

I’ll assume:

- Engine in Rust
- Python used for orchestration/experimentation
- UI is “nice to have later”, not first

1. Parse + AST
2. Expressions → ExprAst
3. BeliefGraph + basic rule execution
4. Flows + transforms
5. Metrics (sum_nodes, fold_nodes) + cross-flow metrics
6. Python bindings
7. Perf + robustness
8. UI

If you follow that order and don’t expand the scope mid-phase, this is actually buildable without you disappearing into a “design forever” hole.

I’d break it down like this:

## Phase 0 – Repo + project skeleton

Status: Complete (2025-11-04)

Goal: Have a clean Rust crate that builds, with a place for grammar, AST, engine, and bindings

Tasks:

- Create repro with something like:

```
baygraph/
  Cargo.toml
  grammar/
    baygraph.pest
  src/
    lib.rs
    frontend/
      mod.rs
      parser.rs
      ast.rs
    ir/
      mod.rs
    engine/
      mod.rs
      graph.rs
      rule_exec.rs
      flow_exec.rs
    metrics/
      mod.rs
  examples/
    minimal.bg
    social.bg
  tests/
    parser_tests.rs
    engine_tests.rs
```
- Hook up pest and a dummy parser that just verifies program = SOI ~ EOI.

- Establish core crates/deps: `thiserror` for errors, `tracing` (feature‑gated), `serde` (for future snapshots), and `rayon` (introduced later, behind a feature flag).

- Create foundational types:
  - Stable ID newtypes: `NodeId(u32)`, `EdgeId(u32)` with `Copy + Eq + Hash + Ord`.
  - Skeleton `ExecError` enum in `engine::errors` with `#[non_exhaustive]`.

Exit criteria:

cargo test and cargo build succeed; project layout is stable enough to not move files around every day.


## Phase 1 – Parser + AST only

Goal: Parse the DSL into a typed AST, no semantics yet.

Status: Complete (2025-11-04)

Tasks:

- Define minimal grammar in pest for:
	- schema (see baygraph_design.md:39)
	- belief_model (you can initially treat body as “opaque”) (see baygraph_design.md:56)
	- evidence (also mostly opaque at first) (see baygraph_design.md:224)
	- rule with: (see baygraph_design.md:240)
		- basic pattern (nodes + edges)
		- stub where/action as raw text
	- flow with: (see baygraph_design.md:281)
		- graph definitions
		- apply_rule
		- prune_edges
		- export
	- metric definitions as simple ident = <expr-text> initially (see baygraph_design.md:308)
- Define AST structs in frontend::ast: (see baygraph_design.md:411, baygraph_design.md:384)
	- Program, Schema, NodeDef, EdgeDef
	- BeliefModel, EvidenceDef
	- RuleDef (pattern structure real, expressions can be strings for now)
	- FlowDef, GraphDef, FlowStep
	- MetricDef
- Write tree-walkers from Pair<Rule> to AST: (see baygraph_design.md:411-415)
	- No IR, no evaluation. Just building Rust structs for the whole file.
- Tests: (see baygraph_design.md:555-557)
	- A couple of .bg examples in examples/, assert parsed AST structure in parser_tests.

Exit criteria:
- You can parse a non-trivial social.bg that uses schema, one rule, one flow, one metric (see baygraph_design.md:642).
- AST looks sane when you println!("{:#?}", ast).

Don’t touch semantics until this is solid.

## Phase 2 – Expressions + lowered IR

Status: Complete (2025-11-05)


Goal: Stop treating expressions as opaque strings; introduce a real expression tree and basic type-checking.

Tasks:

- Extend grammar for expressions:
	- numeric literals, identifiers, calls (foo(...))
	- basic operators: + - * /
	- comparisons: == != < <= > >=
	- logical: and, or, not
	- simple field access: node.attr, E[node.attr], prob(edge) etc.
- Add ExprAst enum in frontend::ast:
	- Literal, Var, Call, Binary, Unary, FieldAccess, etc.
- Parse:
	- where clauses into ExprAst
	- action expressions into ExprAst
	- metric RHS into ExprAst
	- transform predicates (prune_edges ... where ...) into ExprAst
- Add a simple expression IR (you can reuse the AST at first).
- Basic validation/type checks:
	- prob(...) is only called on edge vars.
	- E[...] only on numeric attrs.
	- sum_nodes(...) / fold_nodes(...) have correct argument shapes (syntactic checks).

Exit criteria:

- You can compile expressions to AST/IR without hacks.
- If someone writes nonsense like prob(A.some_value), you can reject it early.

This is where a lot of subtle bugs show up; fail fast.

## Phase 3 – Minimal engine + belief graph model

Status: Complete (2025-11-05)


Goal: Have an in-memory belief graph and execute the simplest rules and flows against it.

Scope: hard limit for v0:

- One label (Person). (see baygraph_design.md:42)
- One edge type (REL). (see baygraph_design.md:48)
- Numeric attributes only. (see baygraph_design.md:81, baygraph_design.md:65-66)
- Bernoulli edge existence with prob: f32. (see baygraph_design.md:110)
- Gaussian-ish attributes with just (mean, var). (see baygraph_design.md:93-101)

Tasks:

- BeliefGraph struct in engine::graph: (see baygraph_design.md:425-427)
	- Node store: Vec<NodeData> with id, label, attr map.
	- Edge store: Vec<EdgeData> with src, dst, type, prob, forced.
	- Adjacency lists for outgoing edges by type. (see baygraph_design.md:466)
- Evidence application: (see baygraph_design.md:224-236)
	- present → Beta–Bernoulli force via large finite α ≫ β. (see baygraph_design.md:133-135)
	- absent → Beta–Bernoulli force via large finite β ≫ α. (see baygraph_design.md:133-135)
	- observe attr = value → Normal–Normal update; `force` uses large finite precision (τ ≈ 1e6). (see baygraph_design.md:93-101, baygraph_design.md:107)
	- Clip extremely small precisions to τ_min ≈ 1e-6 for stability. (see baygraph_design.md:203)
- Basic API:
	- BeliefGraph::from_evidence(schema, belief_model, evidence) → build a tiny graph instance. (see baygraph_design.md:287, baygraph_design.md:426)
	- For now, you can hard-code node ids (“Alice”, “Bob”) in tests.
	- Deterministic iteration over nodes/edges (sorted by stable IDs); avoid panics, return `Result`. (see baygraph_design.md:465, baygraph_design.md:517, baygraph_design.md:533-534)
- Rule execution: minimal subset:
	- pattern (A:Person)-[ab:REL]->(B:Person). (see baygraph_design.md:245)
	- where with prob(ab) >= t and degree(B, min_prob=t2). (see baygraph_design.md:250, baygraph_design.md:254)
	- action with:
		- let (locals) (see baygraph_design.md:257, baygraph_design.md:261)
		- set_expectation Node.attr = Expr (see baygraph_design.md:258-263, baygraph_design.md:275-276)
		- force_absent edgeVar (see baygraph_design.md:265, baygraph_design.md:133-135)
	- Only mode: for_each. (see baygraph_design.md:268)
- Matcher: (see baygraph_design.md:524-527)
	- Iterate edges to bind (A,B) for patterns of that shape.
	- Evaluate where per match.
	- Apply action per match.
	- Keep inputs immutable; apply side‑effects to a working copy and commit after each rule. (see baygraph_design.md:468, baygraph_design.md:527)
- Tests:
	- Tiny graphs where rules:
		- zero out edge probabilities,
		- move values between two nodes. (see baygraph_design.md:557-561)

Exit criteria:

You can run a single rule on a small in-memory graph and see expected changes in prob and mean.

No flows yet, just direct rule execution in tests.

## Phase 4 – Flows + transforms

Status: Complete (2025-11-05)


Goal: Immutable-ish graph pipelines with basic transforms + rule application. (see baygraph_design.md:38, baygraph_design.md:287-306, baygraph_design.md:637)

Scope v0 transforms:

- from_evidence (see baygraph_design.md:293)
- apply_rule (see baygraph_design.md:297)
- prune_edges REL where <expr> (see baygraph_design.md:298)
- export (see baygraph_design.md:304, baygraph_design.md:395)

Tasks:

- Flow IR in ir::flow: (see baygraph_design.md:422-425)
	- GraphExpr with:
		- FromEvidence(name)
		- FromGraph(name)
		- Pipeline(start_graph, Vec<TransformIR>)
	- TransformIR variants:
		- ApplyRule(rule_id, mode_override)
		- PruneEdges(edge_type, predicate_expr)
- Flow executor: (see baygraph_design.md:434)
	- Evaluate graph expressions into concrete BeliefGraph instances.
	- Each transform clones or reuses graph as needed (you can be inefficient for now, just be correct).
	- Graphs are immutable between transforms. (see baygraph_design.md:468, baygraph_design.md:637)
- Context / result:
	- FlowResult with:
		- named graphs,
		- metrics (stub for now).
- DSL wiring:
	- Parse flow blocks into FlowDef → IR. (see baygraph_design.md:398)
	- Support:

```
flow Demo on SocialBeliefs {
  graph base = from_evidence SocialEvidence
  graph cleaned = base |> apply_rule TransferAndDisconnect |> prune_edges REL where prob(edge) < 0.1
  export cleaned as "demo"
}
```
- Tests:
	- End-to-end: parse → compile → run Demo → inspect output graph (edges dropped, rule applied).	

Exit criteria:

You can run a simple flow, produce a BeliefGraph, and verify structure/beliefs.

## Phase 5 – Metrics + sum_nodes / fold_nodes

Goal: Implement metrics as real scalar expressions, including graph-aware aggregates and cross-flow transfer. (see baygraph_design.md:314-354, baygraph_design.md:442-460)

Tasks:

- Metric evaluation context: (see baygraph_design.md:450, baygraph_design.md:435, baygraph_design.md:316)
	- MetricContext with:
		- current graph,
		- known metrics (for cross-reference),
		- optional node and value bindings for aggregators.
- Metric function registry in metrics::: (see baygraph_design.md:442, baygraph_design.md:465, baygraph_design.md:457-460)
	- Provide a registry mapping names → `Arc<dyn MetricFn + Send + Sync>`.
	- `MetricFn::eval(&BeliefGraph, &Args, &Context) -> Result<f64, ExecError>`.
	- Built-ins: sum_nodes(label, where, contrib); fold_nodes(label, where, order_by, init, step); count_nodes(label, where); avg_degree(label, edge_type, min_prob).
- Eval implementation: (see baygraph_design.md:329-351)
	- sum_nodes: (see baygraph_design.md:333-338, baygraph_design.md:523)
		- Iterate nodes of label.
		- Filter with where.
		- Sum contrib with numerically stable reduction (pairwise/Kahan when large).
	- fold_nodes: (see baygraph_design.md:344-351, baygraph_design.md:520)
		- Filter nodes.
		- Sort by order_by (stable, deterministic).
		- Evaluate init.
		- Fold with step using value and node.
- Integrate metrics with flows: (see baygraph_design.md:301-306, baygraph_design.md:357-386, baygraph_design.md:400)
	- Evaluate metric definitions after graph expressions in a flow.
	- Store metrics in a Context object.
	- Implement export_metric / import_metric.
- Cross-flow context: (see baygraph_design.md:582, baygraph_design.md:594)
	- run_flow_with_context(program, flow_name, ctx) which:
	- takes in previous Context (graphs + metrics).
	- returns updated Context.
- Tests: (see baygraph_design.md:563, baygraph_design.md:520, baygraph_design.md:666-672)
	- Compute sum_nodes and fold_nodes on small graphs.
	- Compute final_budget and use it in another flow’s rule predicate.
	- Property tests for posterior invariants (Beta bounds, precision monotonicity) and metric determinism.

Exit criteria:

- Metrics work, including sequential transforms (fold_nodes). (see baygraph_design.md:353)
- You can pass scalar results between flows. (see baygraph_design.md:400, baygraph_design.md:374, baygraph_design.md:378)

### Phase 5.5 – Planning for competing edges (design only)

Goal: Lock in semantics for mutually exclusive edges without implementation pressure.

Notes:

- Prefer Dirichlet–Categorical grouped by source over any renormalization of independent Bernoulli edges.
- Defer implementation to Phase 7+; keep the DSL surface ready (no syntax changes required).

## Phase 6 – Python bindings

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

## Phase 7 – Hardening and performance

Goal: Make it not fall over in real graph sizes.

Tasks:

- Optimize adjacency representation.
- Improve matcher (avoid insane N³ behavior).
- Add iteration caps / safety to fixpoint rules.
- Better error messages in parser and runtime.
- Benchmarks: basic scale tests on 10k–100k nodes.

- Numerical stability:
  - Log‑space evaluation for Beta/Dirichlet functions; stable log‑Gamma (Lanczos/Stirling).
  - Consistent small‑precision clipping (τ_min) and large‑finite “force” thresholds.

- Determinism + parallelism:
  - `rayon` for parallel scans with stable reductions; no reliance on hash order.
  - Kahan/pairwise summation for large aggregates.

- Snapshots and serialization:
  - `Arc` + copy‑on‑write deltas for immutable graph pipelines.
  - `serde` snapshots with engine/registry metadata for reproducibility.

- Feature: Dirichlet–Categorical posterior (v1):
  - Add Categorical edges grouped by source; update metrics and queries to surface vector means when applicable.

This is where you profile and fix hot paths, not before.

## Phase 8 - UI

Only after the engine + Python are sane.

High-level:

- Backend: small HTTP/JSON wrapper around the Rust engine (or Python host).
- Frontend: React-based structured editor:
	- schema editor,
	- rule editor (pattern canvas + condition/action builders),
	- flow editor (pipeline view),
	- graph/metric inspector.

But you can ignore this until the engine proves itself useful via Python.
