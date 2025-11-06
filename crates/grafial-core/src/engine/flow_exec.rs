//! Flow execution engine.
//!
//! Executes flows: sequences of graph transformations that produce named graphs and metrics.
//! Graphs are immutable between transforms (each transform produces a new graph), enabling
//! safe parallel execution and snapshotting.

use std::collections::HashMap;

use crate::engine::errors::ExecError;
use crate::engine::evidence::build_graph_from_evidence;
use crate::engine::expr_eval::{eval_expr_core, ExprContext};
use crate::engine::graph::{BeliefGraph, EdgeId};
use crate::engine::rule_exec::run_rule_for_each_with_globals;
use crate::metrics::{eval_metric_expr, MetricContext, MetricRegistry};
use grafial_frontend::{CallArg, ExprAst, GraphExpr, ProgramAst, Transform};

/// Graph builder trait for abstracting evidence-to-graph construction.
///
/// Allows `run_flow_internal` to work with both production evidence building and test-only
/// custom builders, eliminating duplication between `run_flow` and `run_flow_with_builder`.
trait GraphBuilder {
    fn build_graph(
        &self,
        evidence: &grafial_frontend::ast::EvidenceDef,
        program: &ProgramAst,
    ) -> Result<BeliefGraph, ExecError>;
}

/// Production graph builder using standard evidence building.
struct StandardGraphBuilder;

impl GraphBuilder for StandardGraphBuilder {
    fn build_graph(
        &self,
        evidence: &grafial_frontend::ast::EvidenceDef,
        program: &ProgramAst,
    ) -> Result<BeliefGraph, ExecError> {
        build_graph_from_evidence(evidence, program)
    }
}

/// Custom graph builder for testing.
struct CustomGraphBuilder<'a> {
    builder:
        &'a (dyn Fn(&grafial_frontend::ast::EvidenceDef) -> Result<BeliefGraph, ExecError> + 'a),
}

impl<'a> GraphBuilder for CustomGraphBuilder<'a> {
    fn build_graph(
        &self,
        evidence: &grafial_frontend::ast::EvidenceDef,
        _program: &ProgramAst,
    ) -> Result<BeliefGraph, ExecError> {
        (self.builder)(evidence)
    }
}

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
    /// Named graph snapshots saved during pipeline execution
    pub snapshots: HashMap<String, BeliefGraph>,
}

impl Default for FlowResult {
    fn default() -> Self {
        Self {
            graphs: HashMap::new(),
            exports: HashMap::new(),
            metrics: HashMap::new(),
            metric_exports: HashMap::new(),
            snapshots: HashMap::new(),
        }
    }
}

/// Runs a named flow from a parsed and validated program.
///
/// Each transform produces a new graph (immutability), enabling safe snapshotting and
/// parallel execution. Metrics are evaluated after all graph transformations complete.
pub fn run_flow(
    program: &ProgramAst,
    flow_name: &str,
    prior: Option<&FlowResult>,
) -> Result<FlowResult, ExecError> {
    let builder = StandardGraphBuilder;
    run_flow_internal(program, flow_name, prior, &builder)
}

/// Test-only helper that allows custom evidence builders for testing.
///
/// Production code should use `run_flow()`. This exists to support tests that need
/// custom graph construction without going through the standard evidence building path.
pub fn run_flow_with_builder<'a>(
    program: &'a ProgramAst,
    flow_name: &str,
    evidence_builder: &'a (dyn Fn(&grafial_frontend::ast::EvidenceDef) -> Result<BeliefGraph, ExecError>
             + 'a),
    prior: Option<&FlowResult>,
) -> Result<FlowResult, ExecError> {
    let builder = CustomGraphBuilder {
        builder: evidence_builder,
    };
    run_flow_internal(program, flow_name, prior, &builder)
}

/// Internal flow execution logic shared between `run_flow` and `run_flow_with_builder`.
fn run_flow_internal<B: GraphBuilder>(
    program: &ProgramAst,
    flow_name: &str,
    prior: Option<&FlowResult>,
    graph_builder: &B,
) -> Result<FlowResult, ExecError> {
    let flow = find_flow(program, flow_name)?;

    let evidence_by_name = build_evidence_index(&program.evidences);
    let rules_by_name = build_rules_index(&program.rules);

    let mut result = initialize_flow_result(prior);
    let rule_globals = build_rule_globals(flow, prior)?;

    // Evaluate graph definitions in order
    for g in &flow.graphs {
        match &g.expr {
            GraphExpr::Pipeline { .. } => {
                // Pipelines are handled after all initial graphs are created
                // Skip for now - will process in next loop
            }
            _ => {
                let graph =
                    eval_graph_expr(&g.expr, &evidence_by_name, prior, graph_builder, program)?;
                result.graphs.insert(g.name.clone(), graph);
            }
        }
    }

    // Execute pipelines (which may create additional graphs)
    for g in &flow.graphs {
        if let GraphExpr::Pipeline { start, transforms } = &g.expr {
            let mut current = result
                .graphs
                .get(start)
                .ok_or_else(|| ExecError::Internal(format!("unknown start graph '{}'", start)))?
                .clone();

            for t in transforms {
                current = apply_transform(t, &current, &rules_by_name, &rule_globals, &mut result)?;
            }

            result.graphs.insert(g.name.clone(), current);
        }
    }

    evaluate_metrics(flow, prior, &mut result)?;
    handle_exports(flow, &mut result)?;

    Ok(result)
}

/// Find a flow by name in the program
fn find_flow<'a>(
    program: &'a ProgramAst,
    flow_name: &str,
) -> Result<&'a grafial_frontend::ast::FlowDef, ExecError> {
    program
        .flows
        .iter()
        .find(|f| f.name == flow_name)
        .ok_or_else(|| ExecError::Internal(format!("unknown flow '{}'", flow_name)))
}

/// Build an index of evidences by name for O(1) lookup
fn build_evidence_index(
    evidences: &[grafial_frontend::ast::EvidenceDef],
) -> HashMap<&str, &grafial_frontend::ast::EvidenceDef> {
    evidences.iter().map(|e| (e.name.as_str(), e)).collect()
}

/// Build an index of rules by name for O(1) lookup
fn build_rules_index(
    rules: &[grafial_frontend::ast::RuleDef],
) -> HashMap<&str, &grafial_frontend::ast::RuleDef> {
    rules.iter().map(|r| (r.name.as_str(), r)).collect()
}

/// Initialize flow result with prior metrics if available
fn initialize_flow_result(prior: Option<&FlowResult>) -> FlowResult {
    let mut result = FlowResult::default();
    if let Some(p) = prior {
        result.metric_exports.extend(p.metric_exports.clone());
    }
    result
}

/// Build rule globals from imported metrics.
fn build_rule_globals(
    flow: &grafial_frontend::ast::FlowDef,
    prior: Option<&FlowResult>,
) -> Result<HashMap<String, f64>, ExecError> {
    let mut rule_globals = HashMap::new();
    if let Some(p) = prior {
        for imp in &flow.metric_imports {
            if let Some(v) = p.metric_exports.get(&imp.source_alias) {
                rule_globals.insert(imp.local_name.clone(), *v);
            }
        }
    }
    Ok(rule_globals)
}

/// Evaluate a graph expression to produce a BeliefGraph.
fn eval_graph_expr<B: GraphBuilder>(
    expr: &GraphExpr,
    evidence_by_name: &HashMap<&str, &grafial_frontend::ast::EvidenceDef>,
    prior: Option<&FlowResult>,
    graph_builder: &B,
    program: &ProgramAst,
) -> Result<BeliefGraph, ExecError> {
    match expr {
        GraphExpr::FromEvidence { evidence } => {
            let ev = evidence_by_name
                .get(evidence.as_str())
                .ok_or_else(|| ExecError::Internal(format!("unknown evidence '{}'", evidence)))?;
            graph_builder.build_graph(ev, program)
        }
        GraphExpr::FromGraph { alias } => lookup_graph_from_prior(alias, prior),
        GraphExpr::Pipeline { .. } => {
            // Pipelines require the start graph to already exist, so they're handled
            // separately in run_flow_internal after initial graphs are created
            Err(ExecError::Internal(
                "pipeline evaluation should be handled separately".into(),
            ))
        }
    }
}

/// Look up a graph from prior flow's exports or snapshots.
fn lookup_graph_from_prior(
    alias: &str,
    prior: Option<&FlowResult>,
) -> Result<BeliefGraph, ExecError> {
    prior
        .and_then(|p| {
            // First try exports, then snapshots
            p.exports.get(alias).or_else(|| p.snapshots.get(alias))
        })
        .cloned()
        .ok_or_else(|| {
            ExecError::Internal(format!(
                "graph '{}' not found in prior flow exports or snapshots",
                alias
            ))
        })
}

/// Applies a single transform to a graph, returning a new graph.
fn apply_transform(
    transform: &Transform,
    graph: &BeliefGraph,
    rules_by_name: &HashMap<&str, &grafial_frontend::ast::RuleDef>,
    rule_globals: &HashMap<String, f64>,
    result: &mut FlowResult,
) -> Result<BeliefGraph, ExecError> {
    match transform {
        Transform::ApplyRule { rule } => {
            let r = rules_by_name
                .get(rule.as_str())
                .ok_or_else(|| ExecError::Internal(format!("unknown rule '{}'", rule)))?;
            run_rule_for_each_with_globals(graph, r, rule_globals)
        }
        Transform::ApplyRuleset { rules } => {
            // Sequential application: each rule receives the previous rule's output
            let mut current = graph.clone();
            for rule_name in rules {
                let r = rules_by_name.get(rule_name.as_str()).ok_or_else(|| {
                    ExecError::Internal(format!("unknown rule '{}' in ruleset", rule_name))
                })?;
                current = run_rule_for_each_with_globals(&current, r, rule_globals)?;
            }
            Ok(current)
        }
        Transform::Snapshot { name } => {
            // Ensure deltas are applied before snapshotting
            let mut snapshot_graph = graph.clone();
            snapshot_graph.ensure_owned();
            result.snapshots.insert(name.clone(), snapshot_graph);
            Ok(graph.clone())
        }
        Transform::PruneEdges {
            edge_type,
            predicate,
        } => prune_edges(graph, edge_type, predicate),
    }
}

/// Evaluates metrics against the last defined graph.
///
/// Metrics are evaluated in dependency order (earlier metrics are available to later ones).
/// Imported metrics from prior flows are available to all metric expressions.
fn evaluate_metrics(
    flow: &grafial_frontend::ast::FlowDef,
    prior: Option<&FlowResult>,
    result: &mut FlowResult,
) -> Result<(), ExecError> {
    if flow.metrics.is_empty() && flow.metric_exports.is_empty() && flow.metric_imports.is_empty() {
        return Ok(());
    }

    let last_graph_name = flow
        .graphs
        .last()
        .ok_or_else(|| ExecError::Internal("no graphs defined for metric evaluation".into()))?
        .name
        .as_str();
    let target_graph = result
        .graphs
        .get(last_graph_name)
        .ok_or_else(|| ExecError::Internal("metric target graph missing".into()))?;

    let registry = MetricRegistry::with_builtins();
    let mut ctx = MetricContext {
        metrics: HashMap::new(),
    };

    // Imported metrics from prior flows are available to all expressions
    if let Some(p) = prior {
        for imp in &flow.metric_imports {
            if let Some(v) = p.metric_exports.get(&imp.source_alias) {
                ctx.metrics.insert(imp.local_name.clone(), *v);
            }
        }
    }

    // Earlier metrics in this flow are available to later ones
    for (k, v) in &result.metrics {
        ctx.metrics.insert(k.clone(), *v);
    }

    // Evaluate each metric in order
    for m in &flow.metrics {
        let v = eval_metric_expr(&m.expr, target_graph, &registry, &ctx)?;
        result.metrics.insert(m.name.clone(), v);
        ctx.metrics.insert(m.name.clone(), v);
    }

    // Handle metric exports for this flow
    for mex in &flow.metric_exports {
        let val = result.metrics.get(&mex.metric).copied().ok_or_else(|| {
            ExecError::Internal(format!("unknown metric '{}' in export_metric", mex.metric))
        })?;
        result.metric_exports.insert(mex.alias.clone(), val);
    }

    Ok(())
}

/// Handle graph exports by alias.
fn handle_exports(
    flow: &grafial_frontend::ast::FlowDef,
    result: &mut FlowResult,
) -> Result<(), ExecError> {
    for ex in &flow.exports {
        let g = result
            .graphs
            .get(&ex.graph)
            .ok_or_else(|| ExecError::Internal(format!("unknown graph '{}' in export", ex.graph)))?
            .clone();
        result.exports.insert(ex.alias.clone(), g);
    }
    Ok(())
}

fn prune_edges(
    input: &BeliefGraph,
    edge_type: &str,
    predicate: &ExprAst,
) -> Result<BeliefGraph, ExecError> {
    let (mut keep, mut candidates) = classify_edges(input, edge_type);
    candidates.sort();

    for eid in candidates {
        let should_keep = eval_prune_predicate(predicate, input, eid)? == 0.0;
        if should_keep {
            keep.push(eid);
        }
    }
    keep.sort();

    rebuild_pruned_graph(input, &keep)
}

/// Classify edges into those to keep (wrong type) and candidates for pruning (matching type)
fn classify_edges(input: &BeliefGraph, edge_type: &str) -> (Vec<EdgeId>, Vec<EdgeId>) {
    let mut keep = Vec::new();
    let mut candidates = Vec::new();
    let mut seen_edge_ids = std::collections::HashSet::new();

    for edge in input.edges() {
        seen_edge_ids.insert(edge.id);
        if edge.ty.as_ref() == edge_type {
            candidates.push(edge.id);
        } else {
            keep.push(edge.id);
        }
    }

    process_delta_changes(
        input.delta(),
        edge_type,
        &mut seen_edge_ids,
        &mut keep,
        &mut candidates,
    );

    (keep, candidates)
}

/// Process delta changes to update edge classifications
fn process_delta_changes(
    delta: &[crate::engine::graph::GraphDelta],
    edge_type: &str,
    seen_edge_ids: &mut std::collections::HashSet<EdgeId>,
    keep: &mut Vec<EdgeId>,
    candidates: &mut Vec<EdgeId>,
) {
    use crate::engine::graph::GraphDelta;

    for change in delta {
        match change {
            GraphDelta::EdgeChange { id, edge } => {
                if seen_edge_ids.insert(*id) {
                    // New edge from delta
                    if edge.ty.as_ref() == edge_type {
                        candidates.push(*id);
                    } else {
                        keep.push(*id);
                    }
                } else {
                    // Modified edge - reclassify
                    keep.retain(|&eid| eid != *id);
                    candidates.retain(|&eid| eid != *id);
                    if edge.ty.as_ref() == edge_type {
                        candidates.push(*id);
                    } else {
                        keep.push(*id);
                    }
                }
            }
            GraphDelta::EdgeRemoved { id } => {
                keep.retain(|&eid| eid != *id);
                candidates.retain(|&eid| eid != *id);
            }
            _ => {} // Node changes don't affect edge classification
        }
    }
}

/// Rebuild graph with only the specified edges
fn rebuild_pruned_graph(
    input: &BeliefGraph,
    keep_edges: &[EdgeId],
) -> Result<BeliefGraph, ExecError> {
    let mut input_mut = input.clone();
    input_mut.ensure_owned();
    input_mut.rebuild_with_edges(input_mut.nodes(), keep_edges)
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
                    return Err(ExecError::ValidationError(
                        "prob() does not accept named arguments".into(),
                    ));
                }
                if pos_args.len() != 1 {
                    return Err(ExecError::ValidationError(
                        "prob(): expected single positional argument".into(),
                    ));
                }
                match &pos_args[0] {
                    ExprAst::Var(v) if v == "edge" => graph.prob_mean(self.edge),
                    _ => Err(ExecError::ValidationError(
                        "prob(): argument must be 'edge' in prune predicate".into(),
                    )),
                }
            }
            _ => Err(ExecError::ValidationError(format!(
                "unsupported function '{}' in prune predicate",
                name
            ))),
        }
    }

    fn eval_field(
        &self,
        _target: &ExprAst,
        _field: &str,
        _graph: &BeliefGraph,
    ) -> Result<f64, ExecError> {
        Err(ExecError::ValidationError(
            "field access not allowed in prune predicate".into(),
        ))
    }
}

fn eval_prune_predicate(
    expr: &ExprAst,
    graph: &BeliefGraph,
    edge: EdgeId,
) -> Result<f64, ExecError> {
    let ctx = PruneExprContext { edge };
    eval_expr_core(expr, graph, &ctx)
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::engine::graph::{BetaPosterior, EdgeId, GaussianPosterior, NodeData, NodeId};
    use grafial_frontend::ast::*;

    fn build_simple_graph() -> BeliefGraph {
        let mut g = BeliefGraph::default();
        g.insert_node(NodeData {
            id: NodeId(1),
            label: "Person".into(),
            attrs: HashMap::from([(
                "value".into(),
                GaussianPosterior {
                    mean: 10.0,
                    precision: 1.0,
                },
            )]),
        });
        g.insert_node(NodeData {
            id: NodeId(2),
            label: "Person".into(),
            attrs: HashMap::from([(
                "value".into(),
                GaussianPosterior {
                    mean: 20.0,
                    precision: 1.0,
                },
            )]),
        });
        g.insert_edge(BeliefGraph::test_edge_with_beta(
            EdgeId(1),
            NodeId(1),
            NodeId(2),
            "REL".into(),
            BetaPosterior {
                alpha: 8.0,
                beta: 2.0,
            },
        ));
        g.insert_edge(BeliefGraph::test_edge_with_beta(
            EdgeId(2),
            NodeId(2),
            NodeId(1),
            "REL".into(),
            BetaPosterior {
                alpha: 1.0,
                beta: 9.0,
            },
        ));
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
            left: Box::new(ExprAst::Number(0.5)),
            right: Box::new(ExprAst::Number(1.0)),
        };
        let result = eval_prune_predicate(&expr, &g, EdgeId(1)).unwrap();
        assert_eq!(result, 1.0);
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
        // Edge 2 has prob < 0.5, so should be removed
        assert_eq!(result.edges().len(), 1);
        assert_eq!(result.edges()[0].id, EdgeId(1));
    }

    #[test]
    fn prune_edges_keeps_non_matching_type() {
        let mut g = build_simple_graph();
        g.insert_edge(BeliefGraph::test_edge_with_beta(
            EdgeId(3),
            NodeId(1),
            NodeId(2),
            "OTHER".into(),
            BetaPosterior {
                alpha: 1.0,
                beta: 1.0,
            },
        ));
        let predicate = ExprAst::Bool(true);
        let result = prune_edges(&g, "REL", &predicate).unwrap();
        // Should keep OTHER edge
        assert_eq!(result.edges().len(), 1);
        assert_eq!(result.edges()[0].ty.as_ref(), "OTHER");
    }

    #[test]
    fn prune_edges_with_constant_false_keeps_all() {
        let g = build_simple_graph();
        let predicate = ExprAst::Bool(false);
        let result = prune_edges(&g, "REL", &predicate).unwrap();
        assert_eq!(result.edges().len(), 2);
    }

    #[test]
    fn prune_edges_with_constant_true_removes_all_of_type() {
        let g = build_simple_graph();
        let predicate = ExprAst::Bool(true);
        let result = prune_edges(&g, "REL", &predicate).unwrap();
        assert_eq!(result.edges().len(), 0);
    }
}
