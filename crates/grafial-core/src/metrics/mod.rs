//! Graph metrics computation module.
//!
//! Implements Phase 5 built-in metrics and a simple registry:
//! - `count_nodes(label, where=...)`
//! - `sum_nodes(label, where=..., contrib=...)`
//! - `avg_degree(label, edge_type, min_prob=0.0)`
//! - `fold_nodes(label, where=..., order_by=..., init, step)`
//!
//! Notes:
//! - Deterministic evaluation: stable iteration order by `NodeId` and `EdgeId`.
//! - Numeric stability: Kahan summation for `sum_nodes` when many terms.
//! - Expression evaluation supports: numbers, bools, arithmetic, comparisons,
//!   logical ops, `E[node.attr]`, and `degree(node, min_prob=...)` in per-node contexts.

use std::collections::HashMap;
use std::sync::Arc;

use crate::engine::errors::ExecError;
use crate::engine::expr_eval::{eval_expr_core, ExprContext};
use crate::engine::graph::{BeliefGraph, NodeId};
use grafial_frontend::ast::{BinaryOp, CallArg, ExprAst, UnaryOp};

#[cfg(feature = "rayon")]
use rayon::prelude::*;

/// Context passed during metric evaluation.
#[derive(Debug, Default, Clone)]
pub struct MetricContext {
    /// Previously computed metrics available by name (for cross-references)
    pub metrics: HashMap<String, f64>,
}

/// Arguments passed to metric functions (positional + named)
#[derive(Debug, Default, Clone)]
pub struct MetricArgs<'a> {
    pub pos: Vec<&'a ExprAst>,
    pub named: HashMap<&'a str, &'a ExprAst>,
}

impl<'a> MetricArgs<'a> {
    pub fn from_call_args(args: &'a [CallArg]) -> Self {
        let mut out = MetricArgs::default();
        for a in args {
            match a {
                CallArg::Positional(e) => out.pos.push(e),
                CallArg::Named { name, value } => {
                    out.named.insert(name.as_str(), value);
                }
            }
        }
        out
    }

    pub fn get_named(&self, key: &str) -> Option<&&'a ExprAst> {
        self.named.get(key)
    }
}

/// Trait for metric function implementations.
pub trait MetricFn: Send + Sync + 'static {
    fn eval(
        &self,
        graph: &BeliefGraph,
        args: &MetricArgs<'_>,
        ctx: &MetricContext,
    ) -> Result<f64, ExecError>;
}

/// Built-in registry mapping names to metric implementations.
#[derive(Default, Clone)]
pub struct MetricRegistry {
    inner: HashMap<String, Arc<dyn MetricFn>>,
}

impl MetricRegistry {
    pub fn with_builtins() -> Self {
        let mut r = MetricRegistry {
            inner: HashMap::new(),
        };
        r.register("count_nodes", Arc::new(CountNodes));
        r.register("sum_nodes", Arc::new(SumNodes));
        r.register("fold_nodes", Arc::new(FoldNodes));
        r.register("avg_degree", Arc::new(AvgDegree));
        r
    }

    pub fn register(&mut self, name: &str, f: Arc<dyn MetricFn>) {
        self.inner.insert(name.to_string(), f);
    }

    pub fn get(&self, name: &str) -> Option<Arc<dyn MetricFn>> {
        self.inner.get(name).cloned()
    }
}

/// Evaluate a metric expression to a scalar, using the registry.
pub fn eval_metric_expr(
    expr: &ExprAst,
    graph: &BeliefGraph,
    reg: &MetricRegistry,
    ctx: &MetricContext,
) -> Result<f64, ExecError> {
    match expr {
        ExprAst::Exists { .. } => {
            // Exists subqueries are not supported in metric expressions
            return Err(ExecError::ValidationError(
                "exists subqueries not supported in metric expressions".into(),
            ));
        }
        ExprAst::Number(v) => Ok(*v),
        ExprAst::Bool(b) => Ok(if *b { 1.0 } else { 0.0 }),
        ExprAst::Var(name) => ctx.metrics.get(name).copied().ok_or_else(|| {
            ExecError::ValidationError(format!("unknown metric variable '{}'", name))
        }),
        ExprAst::Field { .. } => Err(ExecError::ValidationError(
            "bare field access not supported in metric; use E[node.attr]".into(),
        )),
        ExprAst::Unary { op, expr } => {
            let v = eval_metric_expr(expr, graph, reg, ctx)?;
            Ok(match op {
                UnaryOp::Neg => -v,
                UnaryOp::Not => {
                    if v == 0.0 {
                        1.0
                    } else {
                        0.0
                    }
                }
            })
        }
        ExprAst::Binary { op, left, right } => {
            let l = eval_metric_expr(left, graph, reg, ctx)?;
            let r = eval_metric_expr(right, graph, reg, ctx)?;
            let result = match op {
                BinaryOp::Add => l + r,
                BinaryOp::Sub => l - r,
                BinaryOp::Mul => l * r,
                BinaryOp::Div => {
                    if r.abs() < 1e-15 {
                        return Err(ExecError::ValidationError(
                            "division by zero in metric expression".into(),
                        ));
                    }
                    l / r
                }
                BinaryOp::Eq => {
                    if (l - r).abs() < 1e-12 {
                        1.0
                    } else {
                        0.0
                    }
                }
                BinaryOp::Ne => {
                    if (l - r).abs() >= 1e-12 {
                        1.0
                    } else {
                        0.0
                    }
                }
                BinaryOp::Lt => {
                    if l < r {
                        1.0
                    } else {
                        0.0
                    }
                }
                BinaryOp::Le => {
                    if l <= r {
                        1.0
                    } else {
                        0.0
                    }
                }
                BinaryOp::Gt => {
                    if l > r {
                        1.0
                    } else {
                        0.0
                    }
                }
                BinaryOp::Ge => {
                    if l >= r {
                        1.0
                    } else {
                        0.0
                    }
                }
                BinaryOp::And => {
                    if (l != 0.0) && (r != 0.0) {
                        1.0
                    } else {
                        0.0
                    }
                }
                BinaryOp::Or => {
                    if (l != 0.0) || (r != 0.0) {
                        1.0
                    } else {
                        0.0
                    }
                }
            };
            if !result.is_finite() {
                return Err(ExecError::ValidationError(format!(
                    "metric expression produced non-finite value: {}",
                    result
                )));
            }
            Ok(result)
        }
        ExprAst::Call { name, args } => {
            if let Some(f) = reg.get(name) {
                let a = MetricArgs::from_call_args(args);
                f.eval(graph, &a, ctx)
            } else {
                Err(ExecError::ValidationError(format!(
                    "unknown metric function '{}'",
                    name
                )))
            }
        }
    }
}

// ----------------------------------------------------------------------------
// Built-in implementations
// ----------------------------------------------------------------------------

struct CountNodes;
impl MetricFn for CountNodes {
    fn eval(
        &self,
        graph: &BeliefGraph,
        args: &MetricArgs<'_>,
        _ctx: &MetricContext,
    ) -> Result<f64, ExecError> {
        // label required; optional where filter
        let label = parse_label(args)?;
        let where_expr = args.named.get("where").copied();
        let mut count = 0usize;
        for n in nodes_sorted_by_id(graph.nodes()) {
            if n.label.as_ref() != label {
                continue;
            }
            if let Some(w) = where_expr {
                let wv = eval_node_expr(w, graph, n.id, _ctx, None)?;
                if wv == 0.0 {
                    continue;
                }
            }
            count += 1;
        }
        Ok(count as f64)
    }
}

struct SumNodes;
impl MetricFn for SumNodes {
    fn eval(
        &self,
        graph: &BeliefGraph,
        args: &MetricArgs<'_>,
        ctx: &MetricContext,
    ) -> Result<f64, ExecError> {
        let label = parse_label(args)?;
        let where_expr = args.named.get("where").copied();
        let contrib_expr = args
            .named
            .get("contrib")
            .copied()
            .or_else(|| args.pos.get(2).copied())
            .ok_or_else(|| {
                ExecError::ValidationError("sum_nodes: missing 'contrib' argument".into())
            })?;

        #[cfg(feature = "rayon")]
        {
            // Phase 7: Parallel implementation with deterministic pairwise summation
            // Process nodes in stable ID order, then use pairwise summation to reduce
            // floating-point error accumulation compared to linear summation.
            let nodes: Vec<_> = nodes_sorted_by_id(graph.nodes())
                .into_iter()
                .filter(|n| n.label.as_ref() == label)
                .collect();

            let terms: Result<Vec<f64>, ExecError> = nodes
                .par_iter()
                .map(|n| {
                    let nid = n.id;
                    if let Some(w) = where_expr {
                        let wv = eval_node_expr(w, graph, nid, ctx, None)?;
                        if wv == 0.0 {
                            return Ok(0.0);
                        }
                    }
                    eval_node_expr(contrib_expr, graph, nid, ctx, None)
                })
                .collect();

            let terms = terms?;
            // Deterministic pairwise summation for numerical stability
            Ok(pairwise_sum(&terms))
        }

        #[cfg(not(feature = "rayon"))]
        {
            // Sequential Kahan compensated summation for numerical stability
            // See: Kahan, W. (1965). "Further remarks on reducing truncation errors"
            let mut sum = 0.0f64;
            let mut c = 0.0f64; // Running compensation for lost low-order bits
            for n in nodes_sorted_by_id(graph.nodes()) {
                if n.label.as_ref() != label {
                    continue;
                }
                let nid = n.id;
                if let Some(w) = where_expr {
                    let wv = eval_node_expr(w, graph, nid, ctx, None)?;
                    if wv == 0.0 {
                        continue;
                    }
                }
                let term = eval_node_expr(contrib_expr, graph, nid, ctx, None)?;
                // Kahan algorithm: compensation term c persists across iterations
                let y = term - c;
                let t = sum + y;
                c = (t - sum) - y;
                sum = t;
            }
            Ok(sum)
        }
    }
}

/// Pairwise summation for better numerical accuracy in parallel reductions.
///
/// Recursively divides the array in half and sums each half, then combines.
/// This reduces floating-point error accumulation compared to linear summation.
///
/// Uses a divide-and-conquer approach: recursively sum left and right halves,
/// then combine. This improves numerical stability for large arrays by reducing
/// the number of additions that accumulate rounding error.
#[allow(dead_code)] // Reserved for future parallel implementation
fn pairwise_sum(values: &[f64]) -> f64 {
    match values.len() {
        0 => 0.0,
        1 => values[0],
        2 => values[0] + values[1],
        n => {
            let mid = n / 2;
            pairwise_sum(&values[..mid]) + pairwise_sum(&values[mid..])
        }
    }
}

struct FoldNodes;
impl MetricFn for FoldNodes {
    fn eval(
        &self,
        graph: &BeliefGraph,
        args: &MetricArgs<'_>,
        ctx: &MetricContext,
    ) -> Result<f64, ExecError> {
        let label = parse_label(args)?;
        let where_expr = args.named.get("where").copied();
        let order_by = args.named.get("order_by").copied();
        let init_expr = args
            .named
            .get("init")
            .copied()
            .or_else(|| args.pos.get(3).copied())
            .ok_or_else(|| {
                ExecError::ValidationError("fold_nodes: missing 'init' argument".into())
            })?;
        let step_expr = args
            .named
            .get("step")
            .copied()
            .or_else(|| args.pos.get(4).copied())
            .ok_or_else(|| {
                ExecError::ValidationError("fold_nodes: missing 'step' argument".into())
            })?;

        // Collect candidate nodes (filtered), with order key
        let mut items: Vec<(NodeId, f64)> = Vec::new();
        for n in nodes_sorted_by_id(graph.nodes()) {
            if n.label.as_ref() != label {
                continue;
            }
            let nid = n.id;
            if let Some(w) = where_expr {
                let wv = eval_node_expr(w, graph, nid, ctx, None)?;
                if wv == 0.0 {
                    continue;
                }
            }
            let key = if let Some(ob) = order_by {
                eval_node_expr(ob, graph, nid, ctx, None)?
            } else {
                nid.0 as f64
            };
            items.push((nid, key));
        }
        // Stable sort by key, then by id to ensure determinism
        // NaN values sort to end to maintain determinism
        items.sort_by(|a, b| {
            match (a.1.is_nan(), b.1.is_nan()) {
                (true, true) => a.0.cmp(&b.0),                // Both NaN: sort by ID
                (true, false) => std::cmp::Ordering::Greater, // NaN sorts last
                (false, true) => std::cmp::Ordering::Less,
                (false, false) => {
                    a.1.partial_cmp(&b.1)
                        .unwrap_or(std::cmp::Ordering::Equal)
                        .then(a.0.cmp(&b.0))
                }
            }
        });

        // Evaluate init (no node binding)
        let mut acc = eval_metric_expr(init_expr, graph, &MetricRegistry::with_builtins(), ctx)?;
        if !acc.is_finite() {
            return Err(ExecError::ValidationError(format!(
                "fold_nodes: init expression produced non-finite value: {}",
                acc
            )));
        }
        for (nid, _) in items {
            // Provide 'value' binding to the step expression
            acc = eval_node_expr(step_expr, graph, nid, ctx, Some(acc))?;
            if !acc.is_finite() {
                return Err(ExecError::ValidationError(format!(
                    "fold_nodes: step expression produced non-finite value: {} for node {:?}",
                    acc, nid
                )));
            }
        }
        Ok(acc)
    }
}

struct AvgDegree;
impl MetricFn for AvgDegree {
    fn eval(
        &self,
        graph: &BeliefGraph,
        args: &MetricArgs<'_>,
        _ctx: &MetricContext,
    ) -> Result<f64, ExecError> {
        // avg over nodes of specified label, degree counted for specified edge_type with min_prob
        let label = parse_label(args)?;
        // edge_type: positional 2 or named "edge_type"
        let et_expr = args
            .named
            .get("edge_type")
            .copied()
            .or_else(|| args.pos.get(1).copied())
            .ok_or_else(|| {
                ExecError::ValidationError("avg_degree: missing 'edge_type' argument".into())
            })?;
        let edge_type = parse_ident_like(et_expr)?;
        let min_prob = if let Some(mp) = args.named.get("min_prob").copied() {
            parse_scalar(mp)?
        } else {
            0.0
        };

        let mut count_nodes = 0usize;
        let mut sum_deg = 0usize;

        // Build adjacency by type for efficient lookup (once)
        let adj = graph.adjacency_outgoing_by_type();
        // Cache the key to avoid cloning edge_type for each node (convert to Arc<str> once)
        use std::sync::Arc;
        let edge_type_arc: Arc<str> = Arc::from(edge_type);
        for n in nodes_sorted_by_id(graph.nodes()) {
            if n.label.as_ref() != label {
                continue;
            }
            count_nodes += 1;
            let mut d = 0usize;
            // Clone Arc<str> only once per loop (cheap, just reference count increment)
            let key = (n.id, edge_type_arc.clone());
            if let Some(edges) = adj.get(&key) {
                for &eid in edges {
                    let p = graph.prob_mean(eid)?;
                    if p >= min_prob {
                        d += 1;
                    }
                }
            }
            sum_deg += d;
        }

        if count_nodes == 0 {
            return Ok(0.0);
        }
        Ok((sum_deg as f64) / (count_nodes as f64))
    }
}

// ----------------------------------------------------------------------------
// Helpers
// ----------------------------------------------------------------------------

fn parse_label(args: &MetricArgs<'_>) -> Result<String, ExecError> {
    let e = args
        .named
        .get("label")
        .copied()
        .or_else(|| args.pos.get(0).copied())
        .ok_or_else(|| ExecError::ValidationError("missing 'label' argument".into()))?;
    parse_ident_like(e)
}

fn parse_ident_like(e: &ExprAst) -> Result<String, ExecError> {
    match e {
        ExprAst::Var(s) => Ok(s.clone()),
        _ => Err(ExecError::ValidationError(
            "expected identifier-like argument".into(),
        )),
    }
}

fn parse_scalar(e: &ExprAst) -> Result<f64, ExecError> {
    match e {
        ExprAst::Number(v) => Ok(*v),
        _ => Err(ExecError::ValidationError(
            "expected numeric literal for argument".into(),
        )),
    }
}

/// Expression evaluation context for metric node expressions.
///
/// Provides variable resolution from metric context and accumulator value,
/// and implements metric-specific functions (E, degree).
struct MetricExprContext<'a> {
    node: NodeId,
    metric_ctx: &'a MetricContext,
    acc_value: Option<f64>,
}

impl<'a> ExprContext for MetricExprContext<'a> {
    fn resolve_var(&self, name: &str) -> Option<f64> {
        if name == "value" {
            self.acc_value
        } else if name == "node" {
            None // Special error case handled below
        } else {
            self.metric_ctx.metrics.get(name).copied()
        }
    }

    fn eval_function(
        &self,
        name: &str,
        pos_args: &[ExprAst],
        all_args: &[CallArg],
        graph: &BeliefGraph,
    ) -> Result<f64, ExecError> {
        match name {
            // E[Var.attr] is represented as Call("E", [Field(Var(..), attr)])
            "E" => {
                if !all_args.is_empty() && matches!(all_args[0], CallArg::Named { .. }) {
                    return Err(ExecError::ValidationError(
                        "E[] does not accept named arguments".into(),
                    ));
                }
                if pos_args.len() != 1 {
                    return Err(ExecError::ValidationError(
                        "E[] expects one positional argument".into(),
                    ));
                }
                match &pos_args[0] {
                    ExprAst::Field { target, field } => match &**target {
                        ExprAst::Var(v) if v == "node" => graph.expectation(self.node, field),
                        _ => Err(ExecError::ValidationError(
                            "E[] requires node.attr with 'node' variable".into(),
                        )),
                    },
                    _ => Err(ExecError::ValidationError(
                        "E[] requires a field expression".into(),
                    )),
                }
            }
            "entropy" => {
                // entropy(node, edge_type)
                if pos_args.len() < 2 {
                    return Err(ExecError::ValidationError(
                        "entropy(): requires node and edge_type arguments".into(),
                    ));
                }
                match &pos_args[0] {
                    ExprAst::Var(v) if v == "node" => {}
                    _ => {
                        return Err(ExecError::ValidationError(
                            "entropy(): first argument must be 'node' in metric expressions".into(),
                        ))
                    }
                }
                let edge_type = match &pos_args[1] {
                    ExprAst::Var(v) => v.clone(),
                    _ => {
                        return Err(ExecError::ValidationError(
                            "entropy(): edge_type must be an identifier".into(),
                        ))
                    }
                };
                if let Some(group) = graph.get_competing_group(self.node, &edge_type) {
                    Ok(group.posterior.entropy())
                } else {
                    Ok(0.0)
                }
            }
            "degree" => {
                if pos_args.is_empty() {
                    return Err(ExecError::ValidationError(
                        "degree(): missing node argument".into(),
                    ));
                }
                // Only allow degree(node, min_prob=...)
                let mut min_prob = 0.0;
                for a in all_args {
                    if let CallArg::Named { name: n, value } = a {
                        if n == "min_prob" {
                            let ctx = MetricExprContext {
                                node: self.node,
                                metric_ctx: self.metric_ctx,
                                acc_value: self.acc_value,
                            };
                            min_prob = eval_expr_core(value, graph, &ctx)?;
                        }
                    }
                }
                match &pos_args[0] {
                    ExprAst::Var(v) if v == "node" => {
                        Ok(graph.degree_outgoing(self.node, min_prob) as f64)
                    }
                    _ => Err(ExecError::ValidationError(
                        "degree(): first argument must be 'node'".into(),
                    )),
                }
            }
            other => Err(ExecError::ValidationError(format!(
                "unsupported function '{}' in node metric expression",
                other
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
            "bare field access not supported in metric; use E[node.attr]".into(),
        ))
    }
}

/// Evaluates an expression in the context of a specific node during metric computation.
///
/// If `acc_value` is Some(v), a Var("value") resolves to v (used in fold_nodes).
fn eval_node_expr(
    expr: &ExprAst,
    graph: &BeliefGraph,
    node: NodeId,
    ctx: &MetricContext,
    acc_value: Option<f64>,
) -> Result<f64, ExecError> {
    let eval_ctx = MetricExprContext {
        node,
        metric_ctx: ctx,
        acc_value,
    };

    // Special handling for "node" variable (not allowed)
    if let ExprAst::Var(name) = expr {
        if name == "node" {
            return Err(ExecError::ValidationError(
                "bare 'node' variable not allowed; use E[node.attr] or degree(node, ...)".into(),
            ));
        }
    }

    // Special handling for "value" variable (must be in fold_nodes context)
    if let ExprAst::Var(name) = expr {
        if name == "value" && acc_value.is_none() {
            return Err(ExecError::ValidationError(
                "'value' is only valid inside fold_nodes step".into(),
            ));
        }
    }

    eval_expr_core(expr, graph, &eval_ctx)
}

/// Helper function to iterate nodes in deterministic order by NodeId.
/// Returns a sorted vector of node references for stable, reproducible iteration.
fn nodes_sorted_by_id(
    nodes: &[crate::engine::graph::NodeData],
) -> Vec<&crate::engine::graph::NodeData> {
    let mut sorted: Vec<_> = nodes.iter().collect();
    sorted.sort_by_key(|n| n.id);
    sorted
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::engine::graph::{BeliefGraph, BetaPosterior, EdgeId, GaussianPosterior, NodeData};

    fn demo_graph() -> BeliefGraph {
        let mut g = demo_graph_impl();
        // Apply delta to ensure all items are in base for iteration
        g.ensure_owned();
        g
    }

    fn demo_graph_impl() -> BeliefGraph {
        use std::collections::HashMap;
        let mut g = BeliefGraph::default();
        g.insert_node(NodeData {
            id: NodeId(1),
            label: "Person".into(),
            attrs: HashMap::from([
                (
                    "a".into(),
                    GaussianPosterior {
                        mean: 1.0,
                        precision: 1.0,
                    },
                ),
                (
                    "b".into(),
                    GaussianPosterior {
                        mean: 2.0,
                        precision: 1.0,
                    },
                ),
            ]),
        });
        g.insert_node(NodeData {
            id: NodeId(2),
            label: "Person".into(),
            attrs: HashMap::from([
                (
                    "a".into(),
                    GaussianPosterior {
                        mean: 3.0,
                        precision: 1.0,
                    },
                ),
                (
                    "b".into(),
                    GaussianPosterior {
                        mean: 4.0,
                        precision: 1.0,
                    },
                ),
            ]),
        });
        g.insert_edge(BeliefGraph::test_edge_with_beta(
            EdgeId(1),
            NodeId(1),
            NodeId(2),
            "REL".into(),
            BetaPosterior {
                alpha: 9.0,
                beta: 1.0,
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
    fn sum_nodes_person_a() {
        let g = demo_graph();
        let reg = MetricRegistry::with_builtins();
        let ctx = MetricContext::default();
        let expr = ExprAst::Call {
            name: "sum_nodes".into(),
            args: vec![
                CallArg::Named {
                    name: "label".into(),
                    value: ExprAst::Var("Person".into()),
                },
                CallArg::Named {
                    name: "contrib".into(),
                    value: ExprAst::Call {
                        name: "E".into(),
                        args: vec![CallArg::Positional(ExprAst::Field {
                            target: Box::new(ExprAst::Var("node".into())),
                            field: "a".into(),
                        })],
                    },
                },
            ],
        };
        let v = eval_metric_expr(&expr, &g, &reg, &ctx).unwrap();
        assert!((v - 4.0).abs() < 1e-9);
    }

    #[test]
    fn fold_nodes_multiply_chain() {
        let g = demo_graph();
        let reg = MetricRegistry::with_builtins();
        let mut ctx = MetricContext::default();
        ctx.metrics.insert("base".into(), 10.0);
        let expr = ExprAst::Call {
            name: "fold_nodes".into(),
            args: vec![
                CallArg::Named {
                    name: "label".into(),
                    value: ExprAst::Var("Person".into()),
                },
                CallArg::Named {
                    name: "order_by".into(),
                    value: ExprAst::Call {
                        name: "E".into(),
                        args: vec![CallArg::Positional(ExprAst::Field {
                            target: Box::new(ExprAst::Var("node".into())),
                            field: "a".into(),
                        })],
                    },
                },
                CallArg::Named {
                    name: "init".into(),
                    value: ExprAst::Var("base".into()),
                },
                CallArg::Named {
                    name: "step".into(),
                    value: ExprAst::Binary {
                        op: BinaryOp::Mul,
                        left: Box::new(ExprAst::Var("value".into())),
                        right: Box::new(ExprAst::Call {
                            name: "E".into(),
                            args: vec![CallArg::Positional(ExprAst::Field {
                                target: Box::new(ExprAst::Var("node".into())),
                                field: "b".into(),
                            })],
                        }),
                    },
                },
            ],
        };
        let v = eval_metric_expr(&expr, &g, &reg, &ctx).unwrap();
        // Order by E[node.a] ascending: node 1 (a=1), then node 2 (a=3)
        // init=10, step: value * E[node.b]; node1 b=2 => 20; node2 b=4 => 80
        assert!((v - 80.0).abs() < 1e-9);
    }

    #[test]
    fn avg_degree_rel_min_prob() {
        let g = demo_graph();
        let reg = MetricRegistry::with_builtins();
        let ctx = MetricContext::default();
        let expr = ExprAst::Call {
            name: "avg_degree".into(),
            args: vec![
                CallArg::Positional(ExprAst::Var("Person".into())),
                CallArg::Positional(ExprAst::Var("REL".into())),
                CallArg::Named {
                    name: "min_prob".into(),
                    value: ExprAst::Number(0.5),
                },
            ],
        };
        let v = eval_metric_expr(&expr, &g, &reg, &ctx).unwrap();
        // node1 out edge prob 0.9 -> counted; node2 out edge prob 0.1 -> not counted
        // avg = (1 + 0) / 2 = 0.5
        assert!((v - 0.5).abs() < 1e-9);
    }

    // === NEW TESTS FOR EDGE CASES ===

    #[test]
    fn sum_nodes_with_where_filter() {
        let g = demo_graph();
        let reg = MetricRegistry::with_builtins();
        let ctx = MetricContext::default();
        // Sum contrib=1.0 for nodes where E[node.a] > 2.0
        let expr = ExprAst::Call {
            name: "sum_nodes".into(),
            args: vec![
                CallArg::Named {
                    name: "label".into(),
                    value: ExprAst::Var("Person".into()),
                },
                CallArg::Named {
                    name: "where".into(),
                    value: ExprAst::Binary {
                        op: BinaryOp::Gt,
                        left: Box::new(ExprAst::Call {
                            name: "E".into(),
                            args: vec![CallArg::Positional(ExprAst::Field {
                                target: Box::new(ExprAst::Var("node".into())),
                                field: "a".into(),
                            })],
                        }),
                        right: Box::new(ExprAst::Number(2.0)),
                    },
                },
                CallArg::Named {
                    name: "contrib".into(),
                    value: ExprAst::Number(1.0),
                },
            ],
        };
        let v = eval_metric_expr(&expr, &g, &reg, &ctx).unwrap();
        // Only node 2 (a=3.0) passes the filter
        assert_eq!(v, 1.0);
    }

    #[test]
    fn sum_nodes_with_where_all_filtered() {
        let g = demo_graph();
        let reg = MetricRegistry::with_builtins();
        let ctx = MetricContext::default();
        // Sum where E[node.a] > 100.0 - no nodes match
        let expr = ExprAst::Call {
            name: "sum_nodes".into(),
            args: vec![
                CallArg::Named {
                    name: "label".into(),
                    value: ExprAst::Var("Person".into()),
                },
                CallArg::Named {
                    name: "where".into(),
                    value: ExprAst::Binary {
                        op: BinaryOp::Gt,
                        left: Box::new(ExprAst::Call {
                            name: "E".into(),
                            args: vec![CallArg::Positional(ExprAst::Field {
                                target: Box::new(ExprAst::Var("node".into())),
                                field: "a".into(),
                            })],
                        }),
                        right: Box::new(ExprAst::Number(100.0)),
                    },
                },
                CallArg::Named {
                    name: "contrib".into(),
                    value: ExprAst::Number(1.0),
                },
            ],
        };
        let v = eval_metric_expr(&expr, &g, &reg, &ctx).unwrap();
        assert_eq!(v, 0.0); // No nodes match filter
    }

    #[test]
    fn count_nodes_empty_graph() {
        let g = BeliefGraph::default();
        let reg = MetricRegistry::with_builtins();
        let ctx = MetricContext::default();
        let expr = ExprAst::Call {
            name: "count_nodes".into(),
            args: vec![CallArg::Named {
                name: "label".into(),
                value: ExprAst::Var("Person".into()),
            }],
        };
        let v = eval_metric_expr(&expr, &g, &reg, &ctx).unwrap();
        assert_eq!(v, 0.0);
    }

    #[test]
    fn count_nodes_no_matches() {
        let g = demo_graph();
        let reg = MetricRegistry::with_builtins();
        let ctx = MetricContext::default();
        let expr = ExprAst::Call {
            name: "count_nodes".into(),
            args: vec![CallArg::Named {
                name: "label".into(),
                value: ExprAst::Var("NonExistent".into()),
            }],
        };
        let v = eval_metric_expr(&expr, &g, &reg, &ctx).unwrap();
        assert_eq!(v, 0.0);
    }

    #[test]
    fn count_nodes_with_where_filter() {
        let g = demo_graph();
        let reg = MetricRegistry::with_builtins();
        let ctx = MetricContext::default();
        // Count nodes where E[node.a] < 2.0
        let expr = ExprAst::Call {
            name: "count_nodes".into(),
            args: vec![
                CallArg::Named {
                    name: "label".into(),
                    value: ExprAst::Var("Person".into()),
                },
                CallArg::Named {
                    name: "where".into(),
                    value: ExprAst::Binary {
                        op: BinaryOp::Lt,
                        left: Box::new(ExprAst::Call {
                            name: "E".into(),
                            args: vec![CallArg::Positional(ExprAst::Field {
                                target: Box::new(ExprAst::Var("node".into())),
                                field: "a".into(),
                            })],
                        }),
                        right: Box::new(ExprAst::Number(2.0)),
                    },
                },
            ],
        };
        let v = eval_metric_expr(&expr, &g, &reg, &ctx).unwrap();
        // Only node 1 (a=1.0) passes the filter
        assert_eq!(v, 1.0);
    }

    #[test]
    fn fold_nodes_with_where_filter() {
        let g = demo_graph();
        let reg = MetricRegistry::with_builtins();
        let ctx = MetricContext::default();
        // Fold only nodes where E[node.a] >= 2.0
        let expr = ExprAst::Call {
            name: "fold_nodes".into(),
            args: vec![
                CallArg::Named {
                    name: "label".into(),
                    value: ExprAst::Var("Person".into()),
                },
                CallArg::Named {
                    name: "where".into(),
                    value: ExprAst::Binary {
                        op: BinaryOp::Ge,
                        left: Box::new(ExprAst::Call {
                            name: "E".into(),
                            args: vec![CallArg::Positional(ExprAst::Field {
                                target: Box::new(ExprAst::Var("node".into())),
                                field: "a".into(),
                            })],
                        }),
                        right: Box::new(ExprAst::Number(2.0)),
                    },
                },
                CallArg::Named {
                    name: "init".into(),
                    value: ExprAst::Number(0.0),
                },
                CallArg::Named {
                    name: "step".into(),
                    value: ExprAst::Binary {
                        op: BinaryOp::Add,
                        left: Box::new(ExprAst::Var("value".into())),
                        right: Box::new(ExprAst::Call {
                            name: "E".into(),
                            args: vec![CallArg::Positional(ExprAst::Field {
                                target: Box::new(ExprAst::Var("node".into())),
                                field: "b".into(),
                            })],
                        }),
                    },
                },
            ],
        };
        let v = eval_metric_expr(&expr, &g, &reg, &ctx).unwrap();
        // Only node 2 (a=3.0 >= 2.0) passes filter, b=4.0
        assert_eq!(v, 4.0);
    }

    #[test]
    fn metric_cross_reference() {
        let g = demo_graph();
        let reg = MetricRegistry::with_builtins();
        let mut ctx = MetricContext::default();

        // First compute sum_a
        let sum_a_expr = ExprAst::Call {
            name: "sum_nodes".into(),
            args: vec![
                CallArg::Named {
                    name: "label".into(),
                    value: ExprAst::Var("Person".into()),
                },
                CallArg::Named {
                    name: "contrib".into(),
                    value: ExprAst::Call {
                        name: "E".into(),
                        args: vec![CallArg::Positional(ExprAst::Field {
                            target: Box::new(ExprAst::Var("node".into())),
                            field: "a".into(),
                        })],
                    },
                },
            ],
        };
        let sum_a = eval_metric_expr(&sum_a_expr, &g, &reg, &ctx).unwrap();
        ctx.metrics.insert("sum_a".into(), sum_a);

        // Now compute sum_b + sum_a (cross-reference)
        let combined_expr = ExprAst::Binary {
            op: BinaryOp::Add,
            left: Box::new(ExprAst::Call {
                name: "sum_nodes".into(),
                args: vec![
                    CallArg::Named {
                        name: "label".into(),
                        value: ExprAst::Var("Person".into()),
                    },
                    CallArg::Named {
                        name: "contrib".into(),
                        value: ExprAst::Call {
                            name: "E".into(),
                            args: vec![CallArg::Positional(ExprAst::Field {
                                target: Box::new(ExprAst::Var("node".into())),
                                field: "b".into(),
                            })],
                        },
                    },
                ],
            }),
            right: Box::new(ExprAst::Var("sum_a".into())),
        };
        let v = eval_metric_expr(&combined_expr, &g, &reg, &ctx).unwrap();
        // sum_a = 1+3=4, sum_b = 2+4=6, total = 10
        assert!((v - 10.0).abs() < 1e-9);
    }

    #[test]
    fn division_by_zero_error() {
        let g = demo_graph();
        let reg = MetricRegistry::with_builtins();
        let ctx = MetricContext::default();
        let expr = ExprAst::Binary {
            op: BinaryOp::Div,
            left: Box::new(ExprAst::Number(1.0)),
            right: Box::new(ExprAst::Number(0.0)),
        };
        let result = eval_metric_expr(&expr, &g, &reg, &ctx);
        assert!(result.is_err());
        assert!(result.unwrap_err().to_string().contains("division by zero"));
    }

    #[test]
    fn fold_nodes_overflow_detection() {
        let g = demo_graph();
        let reg = MetricRegistry::with_builtins();
        let ctx = MetricContext::default();
        // Multiply by very large numbers to trigger overflow
        let expr = ExprAst::Call {
            name: "fold_nodes".into(),
            args: vec![
                CallArg::Named {
                    name: "label".into(),
                    value: ExprAst::Var("Person".into()),
                },
                CallArg::Named {
                    name: "init".into(),
                    value: ExprAst::Number(1e308),
                },
                CallArg::Named {
                    name: "step".into(),
                    value: ExprAst::Binary {
                        op: BinaryOp::Mul,
                        left: Box::new(ExprAst::Var("value".into())),
                        right: Box::new(ExprAst::Number(10.0)),
                    },
                },
            ],
        };
        let result = eval_metric_expr(&expr, &g, &reg, &ctx);
        // Should error on non-finite value
        assert!(result.is_err());
        assert!(result.unwrap_err().to_string().contains("non-finite"));
    }

    #[test]
    fn fold_nodes_nan_in_order_by() {
        use std::collections::HashMap;
        let mut g = BeliefGraph::default();
        g.insert_node(NodeData {
            id: NodeId(1),
            label: "Test".into(),
            attrs: HashMap::from([(
                "x".into(),
                GaussianPosterior {
                    mean: 1.0,
                    precision: 1.0,
                },
            )]),
        });
        g.insert_node(NodeData {
            id: NodeId(2),
            label: "Test".into(),
            attrs: HashMap::from([(
                "x".into(),
                GaussianPosterior {
                    mean: 2.0,
                    precision: 1.0,
                },
            )]),
        });

        // Apply delta to ensure all items are in base for iteration
        g.ensure_owned();

        let reg = MetricRegistry::with_builtins();
        let ctx = MetricContext::default();

        // Order by division that produces NaN for one node
        let expr = ExprAst::Call {
            name: "fold_nodes".into(),
            args: vec![
                CallArg::Named {
                    name: "label".into(),
                    value: ExprAst::Var("Test".into()),
                },
                CallArg::Named {
                    name: "order_by".into(),
                    value: ExprAst::Binary {
                        op: BinaryOp::Div,
                        left: Box::new(ExprAst::Number(0.0)),
                        right: Box::new(ExprAst::Number(0.0)),
                    },
                },
                CallArg::Named {
                    name: "init".into(),
                    value: ExprAst::Number(0.0),
                },
                CallArg::Named {
                    name: "step".into(),
                    value: ExprAst::Binary {
                        op: BinaryOp::Add,
                        left: Box::new(ExprAst::Var("value".into())),
                        right: Box::new(ExprAst::Number(1.0)),
                    },
                },
            ],
        };
        // This should error during order_by evaluation (division by zero)
        let result = eval_metric_expr(&expr, &g, &reg, &ctx);
        assert!(result.is_err());
    }

    #[test]
    fn avg_degree_empty_graph() {
        let g = BeliefGraph::default();
        let reg = MetricRegistry::with_builtins();
        let ctx = MetricContext::default();
        let expr = ExprAst::Call {
            name: "avg_degree".into(),
            args: vec![
                CallArg::Positional(ExprAst::Var("Person".into())),
                CallArg::Positional(ExprAst::Var("REL".into())),
            ],
        };
        let v = eval_metric_expr(&expr, &g, &reg, &ctx).unwrap();
        assert_eq!(v, 0.0); // Returns 0 for empty graph
    }
}
