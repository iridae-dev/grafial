//! Flow execution engine.
//!
//! Implements Phase 4 features:
//! - Evaluate graph expressions (`from_evidence`, pipelines)
//! - Apply transforms (`apply_rule`, `prune_edges`)
//! - Keep graphs immutable between transforms (clone-on-write for now)
//! - Collect named graphs and exports in a `FlowResult`

use std::collections::HashMap;

use crate::engine::errors::ExecError;
use crate::engine::expr_eval::{eval_expr_core, ExprContext};
use crate::engine::graph::{BeliefGraph, EdgeId};
use crate::engine::rule_exec::run_rule_for_each_with_globals;
use crate::frontend::ast::{CallArg, ExprAst, ProgramAst, Transform};
use crate::metrics::{eval_metric_expr, MetricContext, MetricRegistry};

/// The result of running a flow: named graphs and exported aliases.
#[derive(Debug, Clone)]
pub struct FlowResult {
    /// All graphs defined in the flow by variable name
    pub graphs: HashMap<String, BeliefGraph>,
    /// Exported graphs by alias string
    pub exports: HashMap<String, BeliefGraph>,
    /// Computed metrics (scalars) by metric variable name
    pub metrics: HashMap<String, f64>,
    /// Exported metrics by alias string
    pub metric_exports: HashMap<String, f64>,
}

impl Default for FlowResult {
    fn default() -> Self {
        Self { graphs: HashMap::new(), exports: HashMap::new(), metrics: HashMap::new(), metric_exports: HashMap::new() }
    }
}

/// Callback that constructs a BeliefGraph from a specific evidence definition.
pub type EvidenceBuilder<'a> = dyn Fn(&'a crate::frontend::ast::EvidenceDef) -> Result<BeliefGraph, ExecError>;

/// Run a named flow from a parsed and validated program.
///
/// Graphs are treated as immutable between transforms: each transform produces a new
/// BeliefGraph instance. Implementation is correctness-first; optimization comes later.
pub fn run_flow<'a>(
    program: &'a ProgramAst,
    flow_name: &str,
    evidence_builder: &EvidenceBuilder<'a>,
) -> Result<FlowResult, ExecError> {
    run_flow_with_context(program, flow_name, evidence_builder, None)
}

/// Run a named flow with an optional prior context for metrics.
///
/// Prior metrics (when provided) are available as variables during metric
/// evaluation, enabling simple cross-flow scalar transfer.
pub fn run_flow_with_context<'a>(
    program: &'a ProgramAst,
    flow_name: &str,
    evidence_builder: &EvidenceBuilder<'a>,
    prior: Option<&FlowResult>,
) -> Result<FlowResult, ExecError> {
    let flow = program
        .flows
        .iter()
        .find(|f| f.name == flow_name)
        .ok_or_else(|| ExecError::Internal(format!("unknown flow '{}'", flow_name)))?;

    // Index helpers for resolution by name
    let evidence_by_name: HashMap<&str, &crate::frontend::ast::EvidenceDef> =
        program.evidences.iter().map(|e| (e.name.as_str(), e)).collect();
    let rules_by_name: HashMap<&str, &crate::frontend::ast::RuleDef> =
        program.rules.iter().map(|r| (r.name.as_str(), r)).collect();

    let mut result = FlowResult::default();
    if let Some(p) = prior {
        // Carry forward prior exported metrics by alias for import
        result.metric_exports.extend(p.metric_exports.clone());
    }

    // Build rule globals from imported metrics
    let mut rule_globals: HashMap<String, f64> = HashMap::new();
    if let Some(p) = prior {
        for imp in &flow.metric_imports {
            if let Some(v) = p.metric_exports.get(&imp.source_alias) {
                rule_globals.insert(imp.local_name.clone(), *v);
            }
        }
    }

    // Evaluate graph definitions in order
    for g in &flow.graphs {
        match &g.expr {
            crate::frontend::ast::GraphExpr::FromEvidence { evidence } => {
                let ev = evidence_by_name
                    .get(evidence.as_str())
                    .ok_or_else(|| ExecError::Internal(format!("unknown evidence '{}'", evidence)))?;
                let graph = evidence_builder(ev)?;
                result.graphs.insert(g.name.clone(), graph);
            }
            crate::frontend::ast::GraphExpr::Pipeline { start, transforms } => {
                let mut current = result
                    .graphs
                    .get(start)
                    .cloned()
                    .ok_or_else(|| ExecError::Internal(format!("unknown start graph '{}'", start)))?;

                for t in transforms {
                    current = match t {
                        Transform::ApplyRule { rule } => {
                            let r = rules_by_name
                                .get(rule.as_str())
                                .ok_or_else(|| ExecError::Internal(format!("unknown rule '{}'", rule)))?;
                            // Phase 4 scope: only for_each supported
                            run_rule_for_each_with_globals(&current, r, &rule_globals)?
                        }
                        Transform::PruneEdges { edge_type, predicate } => {
                            prune_edges(&current, edge_type, predicate)?
                        }
                    };
                }

                result.graphs.insert(g.name.clone(), current);
            }
        }
    }

    // Evaluate metrics against the last defined graph (if any), in order.
    if !flow.metrics.is_empty() || !flow.metric_exports.is_empty() || !flow.metric_imports.is_empty() {
        let last_graph_name = flow
            .graphs
            .last()
            .ok_or_else(|| ExecError::Internal("no graphs defined for metric evaluation".into()))?
            .name
            .clone();
        let target_graph = result
            .graphs
            .get(&last_graph_name)
            .ok_or_else(|| ExecError::Internal("metric target graph missing".into()))?;

        let registry = MetricRegistry::with_builtins();
        // Seed context with imported metrics from prior exports
        let mut ctx = MetricContext { metrics: HashMap::new() };
        if let Some(p) = prior {
            for imp in &flow.metric_imports {
                if let Some(v) = p.metric_exports.get(&imp.source_alias) {
                    ctx.metrics.insert(imp.local_name.clone(), *v);
                }
            }
        }
        // Also expose any earlier metrics from this flow if present
        for (k, v) in &result.metrics { ctx.metrics.insert(k.clone(), *v); }
        for m in &flow.metrics {
            let v = eval_metric_expr(&m.expr, target_graph, &registry, &ctx)?;
            result.metrics.insert(m.name.clone(), v);
            ctx.metrics.insert(m.name.clone(), v);
        }
        // Handle metric exports for this flow
        for mex in &flow.metric_exports {
            let val = result
                .metrics
                .get(&mex.metric)
                .copied()
                .ok_or_else(|| ExecError::Internal(format!("unknown metric '{}' in export_metric", mex.metric)))?;
            result.metric_exports.insert(mex.alias.clone(), val);
        }
    }

    // Handle exports (graphs by alias)
    for ex in &flow.exports {
        let g = result
            .graphs
            .get(&ex.graph)
            .cloned()
            .ok_or_else(|| ExecError::Internal(format!("unknown graph '{}' in export", ex.graph)))?;
        result.exports.insert(ex.alias.clone(), g);
    }

    Ok(result)
}

fn prune_edges(input: &BeliefGraph, edge_type: &str, predicate: &ExprAst) -> Result<BeliefGraph, ExecError> {
    // Evaluate predicate per edge, removing edges where predicate evaluates to true
    // Build list of edges to keep deterministically by EdgeId
    let mut keep: Vec<EdgeId> = input
        .edges
        .iter()
        .filter(|e| e.ty != edge_type)
        .map(|e| e.id)
        .collect();

    let mut candidates: Vec<EdgeId> = input
        .edges
        .iter()
        .filter(|e| e.ty == edge_type)
        .map(|e| e.id)
        .collect();
    candidates.sort();

    for eid in candidates {
        let should_drop = eval_prune_predicate(predicate, input, eid)? != 0.0;
        if !should_drop {
            keep.push(eid);
        }
    }
    keep.sort();

    // Rebuild graph with kept edges only
    input.rebuild_with_edges(&input.nodes, &keep)
}

/// Expression evaluation context for prune predicates.
///
/// Only allows `prob(edge)` function calls and prohibits variables/fields.
struct PruneExprContext {
    edge: EdgeId,
}

impl ExprContext for PruneExprContext {
    fn resolve_var(&self, _name: &str) -> Option<f64> {
        None // No variables allowed in prune predicates
    }

    fn eval_function(
        &self,
        name: &str,
        pos_args: &[ExprAst],
        all_args: &[CallArg],
        graph: &BeliefGraph,
    ) -> Result<f64, ExecError> {
        match name {
            "prob" => {
                if !all_args.is_empty() && matches!(all_args[0], CallArg::Named { .. }) {
                    return Err(ExecError::ValidationError("prob() does not accept named arguments".into()));
                }
                if pos_args.len() != 1 {
                    return Err(ExecError::ValidationError("prob(): expected single positional argument".into()));
                }
                match &pos_args[0] {
                    ExprAst::Var(v) if v == "edge" => graph.prob_mean(self.edge),
                    _ => Err(ExecError::ValidationError("prob(): argument must be 'edge' in prune predicate".into())),
                }
            }
            _ => Err(ExecError::ValidationError(format!("unsupported function '{}' in prune predicate", name))),
        }
    }

    fn eval_field(&self, _target: &ExprAst, _field: &str, _graph: &BeliefGraph) -> Result<f64, ExecError> {
        Err(ExecError::ValidationError("field access not allowed in prune predicate".into()))
    }
}

fn eval_prune_predicate(expr: &ExprAst, graph: &BeliefGraph, edge: EdgeId) -> Result<f64, ExecError> {
    let ctx = PruneExprContext { edge };
    eval_expr_core(expr, graph, &ctx)
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::engine::graph::{BetaPosterior, EdgeData, GaussianPosterior, NodeData, NodeId, EdgeId};
    use crate::frontend::ast::*;

    fn build_simple_graph() -> BeliefGraph {
        let mut g = BeliefGraph::default();
        g.insert_node(NodeData {
            id: NodeId(1),
            label: "Person".into(),
            attrs: HashMap::from([("value".into(), GaussianPosterior { mean: 10.0, precision: 1.0 })]),
        });
        g.insert_node(NodeData {
            id: NodeId(2),
            label: "Person".into(),
            attrs: HashMap::from([("value".into(), GaussianPosterior { mean: 20.0, precision: 1.0 })]),
        });
        g.insert_edge(EdgeData {
            id: EdgeId(1),
            src: NodeId(1),
            dst: NodeId(2),
            ty: "REL".into(),
            exist: BetaPosterior { alpha: 8.0, beta: 2.0 },
        });
        g.insert_edge(EdgeData {
            id: EdgeId(2),
            src: NodeId(2),
            dst: NodeId(1),
            ty: "REL".into(),
            exist: BetaPosterior { alpha: 1.0, beta: 9.0 },
        });
        g
    }

    #[test]
    fn eval_prune_predicate_number_literal() {
        let g = build_simple_graph();
        let result = eval_prune_predicate(&ExprAst::Number(42.5), &g, EdgeId(1)).unwrap();
        assert_eq!(result, 42.5);
    }

    #[test]
    fn eval_prune_predicate_bool_true() {
        let g = build_simple_graph();
        let result = eval_prune_predicate(&ExprAst::Bool(true), &g, EdgeId(1)).unwrap();
        assert_eq!(result, 1.0);
    }

    #[test]
    fn eval_prune_predicate_bool_false() {
        let g = build_simple_graph();
        let result = eval_prune_predicate(&ExprAst::Bool(false), &g, EdgeId(1)).unwrap();
        assert_eq!(result, 0.0);
    }

    #[test]
    fn eval_prune_predicate_prob_edge() {
        let g = build_simple_graph();
        let expr = ExprAst::Call {
            name: "prob".into(),
            args: vec![CallArg::Positional(ExprAst::Var("edge".into()))],
        };
        let result = eval_prune_predicate(&expr, &g, EdgeId(1)).unwrap();
        assert!((result - 0.8).abs() < 0.01); // alpha=8, beta=2 -> 8/10=0.8
    }

    #[test]
    fn eval_prune_predicate_prob_requires_edge_var() {
        let g = build_simple_graph();
        let expr = ExprAst::Call {
            name: "prob".into(),
            args: vec![CallArg::Positional(ExprAst::Var("other".into()))],
        };
        let result = eval_prune_predicate(&expr, &g, EdgeId(1));
        assert!(result.is_err());
    }

    #[test]
    fn eval_prune_predicate_bare_var_fails() {
        let g = build_simple_graph();
        let result = eval_prune_predicate(&ExprAst::Var("x".into()), &g, EdgeId(1));
        assert!(result.is_err());
    }

    #[test]
    fn eval_prune_predicate_field_access_fails() {
        let g = build_simple_graph();
        let expr = ExprAst::Field {
            target: Box::new(ExprAst::Var("edge".into())),
            field: "prob".into(),
        };
        let result = eval_prune_predicate(&expr, &g, EdgeId(1));
        assert!(result.is_err());
    }

    #[test]
    fn eval_prune_predicate_unary_neg() {
        let g = build_simple_graph();
        let expr = ExprAst::Unary {
            op: UnaryOp::Neg,
            expr: Box::new(ExprAst::Number(5.0)),
        };
        let result = eval_prune_predicate(&expr, &g, EdgeId(1)).unwrap();
        assert_eq!(result, -5.0);
    }

    #[test]
    fn eval_prune_predicate_unary_not() {
        let g = build_simple_graph();
        let expr = ExprAst::Unary {
            op: UnaryOp::Not,
            expr: Box::new(ExprAst::Number(0.0)),
        };
        let result = eval_prune_predicate(&expr, &g, EdgeId(1)).unwrap();
        assert_eq!(result, 1.0);
    }

    #[test]
    fn eval_prune_predicate_binary_comparison() {
        let g = build_simple_graph();
        let expr = ExprAst::Binary {
            op: BinaryOp::Lt,
            left: Box::new(ExprAst::Call {
                name: "prob".into(),
                args: vec![CallArg::Positional(ExprAst::Var("edge".into()))],
            }),
            right: Box::new(ExprAst::Number(0.5)),
        };
        let result = eval_prune_predicate(&expr, &g, EdgeId(1)).unwrap();
        assert_eq!(result, 0.0); // 0.8 < 0.5 is false
    }

    #[test]
    fn eval_prune_predicate_binary_arithmetic() {
        let g = build_simple_graph();
        let expr = ExprAst::Binary {
            op: BinaryOp::Add,
            left: Box::new(ExprAst::Number(3.0)),
            right: Box::new(ExprAst::Number(4.0)),
        };
        let result = eval_prune_predicate(&expr, &g, EdgeId(1)).unwrap();
        assert_eq!(result, 7.0);
    }

    #[test]
    fn prune_edges_removes_matching_edges() {
        let g = build_simple_graph();
        // Prune REL edges where prob(edge) < 0.5
        let predicate = ExprAst::Binary {
            op: BinaryOp::Lt,
            left: Box::new(ExprAst::Call {
                name: "prob".into(),
                args: vec![CallArg::Positional(ExprAst::Var("edge".into()))],
            }),
            right: Box::new(ExprAst::Number(0.5)),
        };
        let result = prune_edges(&g, "REL", &predicate).unwrap();

        // Should remove EdgeId(2) which has alpha=1, beta=9 (prob=0.1)
        // Should keep EdgeId(1) which has alpha=8, beta=2 (prob=0.8)
        assert_eq!(result.edges.len(), 1);
        assert_eq!(result.edges[0].id, EdgeId(1));
    }

    #[test]
    fn prune_edges_keeps_non_matching_type() {
        let mut g = build_simple_graph();
        g.insert_edge(EdgeData {
            id: EdgeId(3),
            src: NodeId(1),
            dst: NodeId(2),
            ty: "OTHER".into(),
            exist: BetaPosterior { alpha: 1.0, beta: 99.0 },
        });

        let predicate = ExprAst::Bool(true); // Always prune
        let result = prune_edges(&g, "REL", &predicate).unwrap();

        // Should remove both REL edges but keep OTHER edge
        assert_eq!(result.edges.len(), 1);
        assert_eq!(result.edges[0].ty, "OTHER");
    }

    #[test]
    fn prune_edges_with_constant_false_keeps_all() {
        let g = build_simple_graph();
        let predicate = ExprAst::Bool(false);
        let result = prune_edges(&g, "REL", &predicate).unwrap();
        assert_eq!(result.edges.len(), 2);
    }

    #[test]
    fn prune_edges_with_constant_true_removes_all_of_type() {
        let g = build_simple_graph();
        let predicate = ExprAst::Bool(true);
        let result = prune_edges(&g, "REL", &predicate).unwrap();
        assert_eq!(result.edges.len(), 0);
    }

    #[test]
    fn flow_result_default_is_empty() {
        let result = FlowResult::default();
        assert!(result.graphs.is_empty());
        assert!(result.exports.is_empty());
        assert!(result.metrics.is_empty());
        assert!(result.metric_exports.is_empty());
    }

    #[test]
    fn run_flow_evaluates_metrics_on_last_graph() {
        // Build a tiny program with a flow that creates a graph and computes a metric
        let program = ProgramAst {
            schemas: vec![], belief_models: vec![], evidences: vec![EvidenceDef { name: "Ev".into(), on_model: "M".into(), body_src: "".into() }], rules: vec![],
            flows: vec![FlowDef {
                name: "F".into(), on_model: "M".into(),
                graphs: vec![GraphDef { name: "g".into(), expr: GraphExpr::FromEvidence { evidence: "Ev".into() } }],
                metrics: vec![MetricDef {
                    name: "total".into(),
                    expr: ExprAst::Call {
                        name: "sum_nodes".into(),
                        args: vec![
                            CallArg::Named { name: "label".into(), value: ExprAst::Var("Person".into()) },
                            CallArg::Named { name: "contrib".into(), value: ExprAst::Call { name: "E".into(), args: vec![CallArg::Positional(ExprAst::Field { target: Box::new(ExprAst::Var("node".into())), field: "value".into() })] } },
                        ],
                    },
                }],
                exports: vec![],
                metric_exports: vec![],
                metric_imports: vec![],
            }],
        };

        // Evidence builder returns a small graph with two Person nodes and values 10 and 20
        let evidence_builder: &EvidenceBuilder = &|_ev| {
            let mut g = BeliefGraph::default();
            g.insert_node(NodeData {
                id: NodeId(1),
                label: "Person".into(),
                attrs: HashMap::from([("value".into(), GaussianPosterior { mean: 10.0, precision: 1.0 })]),
            });
            g.insert_node(NodeData {
                id: NodeId(2),
                label: "Person".into(),
                attrs: HashMap::from([("value".into(), GaussianPosterior { mean: 20.0, precision: 1.0 })]),
            });
            Ok(g)
        };

        let result = run_flow(&program, "F", evidence_builder).unwrap();
        assert!(result.metrics.contains_key("total"));
        assert!((result.metrics["total"] - 30.0).abs() < 1e-9);
    }

    #[test]
    fn run_flow_with_context_imports_metric() {
        // Program with two flows: Producer exports a metric, Consumer imports and uses it
        let program = ProgramAst {
            schemas: vec![],
            belief_models: vec![],
            evidences: vec![EvidenceDef { name: "Ev".into(), on_model: "M".into(), body_src: "".into() }],
            rules: vec![],
            flows: vec![
                FlowDef {
                    name: "Producer".into(), on_model: "M".into(),
                    graphs: vec![GraphDef { name: "g".into(), expr: GraphExpr::FromEvidence { evidence: "Ev".into() } }],
                    metrics: vec![MetricDef { name: "base".into(), expr: ExprAst::Number(100.0) }],
                    exports: vec![],
                    metric_exports: vec![MetricExportDef { metric: "base".into(), alias: "scenario_budget".into() }],
                    metric_imports: vec![],
                },
                FlowDef {
                    name: "Consumer".into(), on_model: "M".into(),
                    graphs: vec![GraphDef { name: "g".into(), expr: GraphExpr::FromEvidence { evidence: "Ev".into() } }],
                    metrics: vec![MetricDef { name: "fin".into(), expr: ExprAst::Call { name: "fold_nodes".into(), args: vec![
                        CallArg::Named { name: "label".into(), value: ExprAst::Var("Person".into()) },
                        CallArg::Named { name: "init".into(), value: ExprAst::Var("budget".into()) },
                        CallArg::Named { name: "step".into(), value: ExprAst::Var("value".into()) },
                    ] } }],
                    exports: vec![],
                    metric_exports: vec![],
                    metric_imports: vec![MetricImportDef { source_alias: "scenario_budget".into(), local_name: "budget".into() }],
                }
            ],
        };

        // Evidence builder creates a graph with one Person; fold identity keeps init
        let evidence_builder: &EvidenceBuilder = &|_ev| {
            let mut g = BeliefGraph::default();
            g.insert_node(NodeData { id: NodeId(1), label: "Person".into(), attrs: HashMap::new() });
            Ok(g)
        };

        let prod = run_flow(&program, "Producer", evidence_builder).unwrap();
        assert_eq!(prod.metric_exports.get("scenario_budget"), Some(&100.0));

        let cons = run_flow_with_context(&program, "Consumer", evidence_builder, Some(&prod)).unwrap();
        assert!((cons.metrics.get("fin").copied().unwrap_or(-1.0) - 100.0).abs() < 1e-9);
    }

    #[test]
    fn rule_predicate_uses_imported_metric_threshold() {
        // Producer exports a threshold; Consumer imports it and uses in rule predicate
        let rule = RuleDef {
            name: "RemoveBelow".into(), on_model: "M".into(), mode: Some("for_each".into()),
            patterns: vec![PatternItem {
                src: NodePattern { var: "A".into(), label: "P".into() },
                edge: EdgePattern { var: "e".into(), ty: "R".into() },
                dst: NodePattern { var: "B".into(), label: "P".into() },
            }],
            where_expr: Some(ExprAst::Binary {
                op: BinaryOp::Lt,
                left: Box::new(ExprAst::Call { name: "prob".into(), args: vec![CallArg::Positional(ExprAst::Var("e".into()))] }),
                right: Box::new(ExprAst::Var("threshold".into())),
            }),
            actions: vec![ActionStmt::ForceAbsent { edge_var: "e".into() }],
        };

        let program = ProgramAst {
            schemas: vec![], belief_models: vec![], evidences: vec![EvidenceDef { name: "Ev".into(), on_model: "M".into(), body_src: "".into() }],
            rules: vec![rule],
            flows: vec![
                FlowDef {
                    name: "Producer".into(), on_model: "M".into(),
                    graphs: vec![GraphDef { name: "g".into(), expr: GraphExpr::FromEvidence { evidence: "Ev".into() } }],
                    metrics: vec![MetricDef { name: "th".into(), expr: ExprAst::Number(0.7) }],
                    exports: vec![], metric_exports: vec![MetricExportDef { metric: "th".into(), alias: "threshold".into() }], metric_imports: vec![],
                },
                FlowDef {
                    name: "Consumer".into(), on_model: "M".into(),
                    graphs: vec![
                        GraphDef { name: "g".into(), expr: GraphExpr::FromEvidence { evidence: "Ev".into() } },
                        GraphDef { name: "out".into(), expr: GraphExpr::Pipeline { start: "g".into(), transforms: vec![Transform::ApplyRule { rule: "RemoveBelow".into() }] } }
                    ],
                    metrics: vec![], exports: vec![], metric_exports: vec![], metric_imports: vec![MetricImportDef { source_alias: "threshold".into(), local_name: "threshold".into() }],
                }
            ],
        };

        let evidence_builder: &EvidenceBuilder = &|_ev| {
            let mut g = BeliefGraph::default();
            g.insert_node(NodeData { id: NodeId(1), label: "P".into(), attrs: HashMap::new() });
            g.insert_node(NodeData { id: NodeId(2), label: "P".into(), attrs: HashMap::new() });
            g.insert_edge(EdgeData { id: EdgeId(1), src: NodeId(1), dst: NodeId(2), ty: "R".into(), exist: BetaPosterior { alpha: 3.0, beta: 7.0 } }); // prob 0.3
            Ok(g)
        };

        let prod = run_flow(&program, "Producer", evidence_builder).unwrap();
        let cons = run_flow_with_context(&program, "Consumer", evidence_builder, Some(&prod)).unwrap();
        // After rule, the low-probability edge should be forced absent
        let out = cons.graphs.get("out").unwrap();
        let p = out.prob_mean(EdgeId(1)).unwrap();
        assert!(p < 1e-5);
    }
}
