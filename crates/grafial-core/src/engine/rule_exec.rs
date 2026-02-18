//! # Rule Execution Engine
//!
//! This module implements the core rule execution logic for Grafial, including:
//!
//! - **Pattern matching**: Finding subgraph matches in the belief graph
//! - **Expression evaluation**: Computing values for where clauses and action expressions
//! - **Action execution**: Applying graph updates (set_expectation, force_absent, etc.)
//!
//! ## Rule Execution Model
//!
//! Rules follow a pattern-match-execute model:
//! 1. Match patterns in the graph to find variable bindings
//! 2. Evaluate the optional `where` clause with bound variables
//! 3. Execute actions for each successful match (for_each mode) or first match
//!
//! Rules operate on immutable graphs: mutations are applied to a working copy (delta),
//! and the original graph remains unchanged. The working copy is committed at the end
//! of rule execution to produce the output graph.
//!
//! ## Expression Language
//!
//! Supports:
//! - Arithmetic: `+`, `-`, `*`, `/`
//! - Comparisons: `<`, `<=`, `>`, `>=`, `==`, `!=`
//! - Logical: `and`, `or`, `not`
//! - Special functions: `prob(edge)`, `prob_correlated(A.attr > B.attr, rho=...)`,
//!   `credible(event, p=...)`, `degree(node)`, `E[node.attr]`
//!
//! ## Performance
//!
//! Uses O(1) hash lookups for nodes/edges and eager evaluation with
//! local variable binding for efficient execution.
//!
//! ## Determinism
//!
//! Pattern matching iterates edges in stable order (sorted by EdgeId) to ensure
//! deterministic rule execution. This guarantees that the same input graph and rule
//! always produce the same output, regardless of hash table iteration order.

use std::collections::HashMap;

use crate::engine::errors::ExecError;
use crate::engine::expr_eval::{eval_binary_op, eval_expr_core, eval_unary_op, ExprContext};
use crate::engine::expr_utils::inv_norm_cdf;
use crate::engine::graph::{BeliefGraph, EdgeId, NodeId};
use crate::engine::query_plan::QueryPlanCache;
use grafial_frontend::ast::{ActionStmt, CallArg, ExprAst, PatternItem, RuleDef};
use grafial_ir::RuleIR;

// Phase 7: Fixpoint iteration configuration
/// Maximum iterations for fixpoint rules to prevent infinite loops.
///
/// If a fixpoint rule doesn't converge within this many iterations, execution
/// terminates with an error to prevent infinite loops.
const MAX_FIXPOINT_ITERATIONS: usize = 1000;

/// Convergence tolerance for fixpoint rules
/// If changes are smaller than this threshold, consider converged
const FIXPOINT_TOLERANCE: f64 = 1e-6;

/// Variable bindings for a single pattern match.
///
/// Maps pattern variables (from rule patterns) to concrete graph elements.
/// For example, pattern `(A:Person)-[e:KNOWS]->(B:Person)` creates bindings
/// for variables A, B (nodes) and e (edge).
#[derive(Debug, Clone, Default)]
pub struct MatchBindings {
    /// Maps node variable names to node IDs
    pub node_vars: HashMap<String, NodeId>,
    /// Maps edge variable names to edge IDs
    pub edge_vars: HashMap<String, EdgeId>,
}

impl MatchBindings {
    /// Creates bindings with pre-allocated capacity based on expected pattern count.
    ///
    /// Estimates: 2 nodes per pattern (src + dst), 1 edge per pattern.
    /// This avoids multiple reallocations during pattern matching.
    pub fn with_capacity(patterns: &[PatternItem]) -> Self {
        let estimated_nodes = patterns.len() * 2; // src + dst per pattern
        let estimated_edges = patterns.len();
        Self {
            node_vars: HashMap::with_capacity(estimated_nodes),
            edge_vars: HashMap::with_capacity(estimated_edges),
        }
    }
}

/// Local variable storage for rule action execution.
///
/// Tracks variables created by `let` statements within action blocks.
struct Locals(HashMap<String, f64>);

impl Locals {
    fn new() -> Self {
        Self(HashMap::new())
    }

    fn get(&self, name: &str) -> Option<f64> {
        self.0.get(name).copied()
    }

    fn set(&mut self, name: String, value: f64) {
        self.0.insert(name, value);
    }
}

/// Expression evaluation context for rule execution.
///
/// Provides variable resolution from locals, globals, and pattern bindings,
/// and implements rule-specific functions (prob, degree, E).
struct RuleExprContext<'a> {
    bindings: &'a MatchBindings,
    locals: &'a Locals,
    globals: &'a HashMap<String, f64>,
}

impl<'a> ExprContext for RuleExprContext<'a> {
    fn resolve_var(&self, name: &str) -> Option<f64> {
        // Check locals first, then globals, then node/edge variables
        self.locals
            .get(name)
            .or_else(|| self.globals.get(name).copied())
            .or_else(|| {
                // Node variables: convert NodeId to f64 for comparisons
                self.bindings.node_vars.get(name).map(|nid| nid.0 as f64)
            })
            .or_else(|| {
                // Edge variables: convert EdgeId to f64 for comparisons
                self.bindings.edge_vars.get(name).map(|eid| eid.0 as f64)
            })
    }

    fn eval_function(
        &self,
        name: &str,
        pos_args: &[ExprAst],
        all_args: &[CallArg],
        graph: &BeliefGraph,
    ) -> Result<f64, ExecError> {
        match name {
            "E" => self.eval_expectation_function(pos_args, all_args, graph),
            "prob" => self.eval_prob_function(pos_args, all_args, graph),
            "prob_correlated" => self.eval_prob_correlated_function(pos_args, all_args, graph),
            "credible" => self.eval_credible_function(pos_args, all_args, graph),
            "variance" => self.eval_variance_function(pos_args, all_args, graph),
            "stddev" => self.eval_stddev_function(pos_args, all_args, graph),
            "ci_lo" => self.eval_ci_function(pos_args, all_args, graph, true),
            "ci_hi" => self.eval_ci_function(pos_args, all_args, graph, false),
            "effective_n" => self.eval_effective_n_function(pos_args, all_args, graph),
            "degree" => self.eval_degree_function(pos_args, all_args, graph),
            "winner" => self.eval_winner_function(pos_args, all_args, graph),
            "entropy" => self.eval_entropy_function(pos_args, all_args, graph),
            "quantile" => self.eval_quantile_function(pos_args, all_args, graph),
            other => Err(ExecError::Internal(format!("unknown function '{}'", other))),
        }
    }

    fn eval_field(
        &self,
        target: &ExprAst,
        _field: &str,
        _graph: &BeliefGraph,
    ) -> Result<f64, ExecError> {
        Err(ExecError::Internal(format!(
            "bare field access not supported; use E[Node.attr]: {:?}",
            target
        )))
    }
}

impl<'a> RuleExprContext<'a> {
    /// Helper: Extract node ID from node variable name
    fn resolve_node_var(&self, var_name: &str) -> Result<NodeId, ExecError> {
        self.bindings
            .node_vars
            .get(var_name)
            .copied()
            .ok_or_else(|| ExecError::Internal(format!("unknown node var '{}'", var_name)))
    }

    /// Evaluates quantile(NodeVar.attr, p) assuming Normal posterior
    fn eval_quantile_function(
        &self,
        pos_args: &[ExprAst],
        all_args: &[CallArg],
        graph: &BeliefGraph,
    ) -> Result<f64, ExecError> {
        // Only positional args supported: (field, p)
        if !all_args.is_empty() && matches!(all_args[0], CallArg::Named { .. }) {
            return Err(ExecError::Internal(
                "quantile(): named args not supported".into(),
            ));
        }
        if pos_args.len() < 2 {
            return Err(ExecError::Internal(
                "quantile(): requires field and p arguments".into(),
            ));
        }
        let p = match pos_args[1] {
            ExprAst::Number(v) => v,
            _ => {
                return Err(ExecError::ValidationError(
                    "quantile(): p must be numeric".into(),
                ))
            }
        };
        if p <= 0.0 || p >= 1.0 {
            return Err(ExecError::ValidationError(
                "quantile(): p must be in (0,1)".into(),
            ));
        }
        match &pos_args[0] {
            ExprAst::Field { target, field } => match &**target {
                ExprAst::Var(var_name) => {
                    let nid = self.resolve_node_var(var_name)?;
                    let mu = graph.expectation(nid, field)?;
                    let sigma = graph.stddev(nid, field)?;
                    let z = inv_norm_cdf(p);
                    Ok(mu + z * sigma)
                }
                _ => Err(ExecError::Internal(
                    "quantile(): first argument must be NodeVar.attr".into(),
                )),
            },
            _ => Err(ExecError::Internal(
                "quantile(): first argument must be a field expression".into(),
            )),
        }
    }

    /// Evaluates variance(NodeVar.attr) - returns posterior variance for an attribute
    fn eval_variance_function(
        &self,
        pos_args: &[ExprAst],
        all_args: &[CallArg],
        graph: &BeliefGraph,
    ) -> Result<f64, ExecError> {
        self.ensure_positional_only(all_args, "variance")?;
        if pos_args.len() != 1 {
            return Err(ExecError::Internal(
                "variance() expects one positional argument".into(),
            ));
        }
        match &pos_args[0] {
            ExprAst::Field { target, field } => match &**target {
                ExprAst::Var(var_name) => {
                    let nid = self.resolve_node_var(var_name)?;
                    graph.variance(nid, field)
                }
                _ => Err(ExecError::Internal(
                    "variance() requires NodeVar.attr".into(),
                )),
            },
            _ => Err(ExecError::Internal(
                "variance() requires a field expression".into(),
            )),
        }
    }

    /// Evaluates stddev(NodeVar.attr) - returns posterior standard deviation
    fn eval_stddev_function(
        &self,
        pos_args: &[ExprAst],
        all_args: &[CallArg],
        graph: &BeliefGraph,
    ) -> Result<f64, ExecError> {
        self.ensure_positional_only(all_args, "stddev")?;
        if pos_args.len() != 1 {
            return Err(ExecError::Internal(
                "stddev() expects one positional argument".into(),
            ));
        }
        match &pos_args[0] {
            ExprAst::Field { target, field } => match &**target {
                ExprAst::Var(var_name) => {
                    let nid = self.resolve_node_var(var_name)?;
                    graph.stddev(nid, field)
                }
                _ => Err(ExecError::Internal("stddev() requires NodeVar.attr".into())),
            },
            _ => Err(ExecError::Internal(
                "stddev() requires a field expression".into(),
            )),
        }
    }

    /// Evaluates ci_lo/ci_hi(NodeVar.attr, p) assuming Normal posterior
    fn eval_ci_function(
        &self,
        pos_args: &[ExprAst],
        all_args: &[CallArg],
        graph: &BeliefGraph,
        lower: bool,
    ) -> Result<f64, ExecError> {
        self.ensure_positional_only(all_args, if lower { "ci_lo" } else { "ci_hi" })?;
        if pos_args.len() != 2 {
            return Err(ExecError::Internal(
                "ci_lo/ci_hi() expects field and p".into(),
            ));
        }
        let p = match pos_args[1] {
            ExprAst::Number(v) => v,
            _ => return Err(ExecError::ValidationError("ci(): p must be numeric".into())),
        };
        if !(0.0 < p && p < 1.0) {
            return Err(ExecError::ValidationError(
                "ci(): p must be in (0,1)".into(),
            ));
        }
        match &pos_args[0] {
            ExprAst::Field { target, field } => match &**target {
                ExprAst::Var(var_name) => {
                    let nid = self.resolve_node_var(var_name)?;
                    let mean = graph.expectation(nid, field)?;
                    let sigma = graph.stddev(nid, field)?;
                    // Convert central coverage p into one-sided quantile
                    let q = (1.0 + p) / 2.0;
                    let z = crate::engine::expr_utils::inv_norm_cdf(q);
                    if lower {
                        Ok(mean - z * sigma)
                    } else {
                        Ok(mean + z * sigma)
                    }
                }
                _ => Err(ExecError::Internal("ci() requires NodeVar.attr".into())),
            },
            _ => Err(ExecError::Internal(
                "ci() requires a field expression".into(),
            )),
        }
    }

    /// Evaluates effective_n(NodeVar.attr) as posterior precision (τ)
    fn eval_effective_n_function(
        &self,
        pos_args: &[ExprAst],
        all_args: &[CallArg],
        graph: &BeliefGraph,
    ) -> Result<f64, ExecError> {
        self.ensure_positional_only(all_args, "effective_n")?;
        if pos_args.len() != 1 {
            return Err(ExecError::Internal(
                "effective_n() expects one positional argument".into(),
            ));
        }
        match &pos_args[0] {
            ExprAst::Field { target, field } => match &**target {
                ExprAst::Var(var_name) => {
                    let nid = self.resolve_node_var(var_name)?;
                    graph.precision(nid, field)
                }
                _ => Err(ExecError::Internal(
                    "effective_n() requires NodeVar.attr".into(),
                )),
            },
            _ => Err(ExecError::Internal(
                "effective_n() requires a field expression".into(),
            )),
        }
    }

    /// Helper: Extract edge ID from edge variable name
    fn resolve_edge_var(&self, var_name: &str) -> Result<EdgeId, ExecError> {
        self.bindings
            .edge_vars
            .get(var_name)
            .copied()
            .ok_or_else(|| ExecError::Internal(format!("unknown edge var '{}'", var_name)))
    }

    /// Helper: Extract named argument value by name
    fn extract_named_arg(
        &self,
        all_args: &[CallArg],
        name: &str,
        graph: &BeliefGraph,
    ) -> Result<Option<f64>, ExecError> {
        for arg in all_args {
            if let CallArg::Named {
                name: arg_name,
                value,
            } = arg
            {
                if arg_name == name {
                    return eval_expr_core(value, graph, self).map(Some);
                }
            }
        }
        Ok(None)
    }

    /// Helper: Validate positional-only arguments
    fn ensure_positional_only(&self, all_args: &[CallArg], fn_name: &str) -> Result<(), ExecError> {
        if !all_args.is_empty() && matches!(all_args[0], CallArg::Named { .. }) {
            return Err(ExecError::Internal(format!(
                "{}() does not accept named arguments",
                fn_name
            )));
        }
        Ok(())
    }

    /// Evaluates E[node.attr] - returns posterior mean for a node attribute
    fn eval_expectation_function(
        &self,
        pos_args: &[ExprAst],
        all_args: &[CallArg],
        graph: &BeliefGraph,
    ) -> Result<f64, ExecError> {
        self.ensure_positional_only(all_args, "E")?;

        if pos_args.len() != 1 {
            return Err(ExecError::Internal(
                "E[] expects one positional argument".into(),
            ));
        }

        match &pos_args[0] {
            ExprAst::Field { target, field } => match &**target {
                ExprAst::Var(var_name) => {
                    let nid = self.resolve_node_var(var_name)?;
                    graph.expectation(nid, field)
                }
                _ => Err(ExecError::Internal("E[] requires NodeVar.attr".into())),
            },
            _ => Err(ExecError::Internal(
                "E[] requires a field expression".into(),
            )),
        }
    }

    /// Evaluates prob(edge) or prob(lhs > rhs).
    ///
    /// Comparison forms assume independence between operands.
    fn eval_prob_function(
        &self,
        pos_args: &[ExprAst],
        all_args: &[CallArg],
        graph: &BeliefGraph,
    ) -> Result<f64, ExecError> {
        self.ensure_positional_only(all_args, "prob")?;

        if pos_args.len() != 1 {
            return Err(ExecError::Internal(
                "prob() expects one positional argument".into(),
            ));
        }

        match &pos_args[0] {
            ExprAst::Var(var_name) => {
                let eid = self.resolve_edge_var(var_name)?;
                let result = graph.prob_mean(eid)?;
                Ok(result)
            }
            ExprAst::Binary { op, left, right } => {
                self.eval_prob_comparison(*op, left, right, 0.0, "prob", graph)
            }
            _ => Err(ExecError::Internal(
                "prob(): argument must be an edge variable or comparison".into(),
            )),
        }
    }

    /// Evaluates prob_correlated(lhs > rhs, rho=...) for Gaussian comparison probabilities.
    ///
    /// `prob(...)` remains the independence form; use this for explicit non-zero correlation.
    fn eval_prob_correlated_function(
        &self,
        pos_args: &[ExprAst],
        all_args: &[CallArg],
        graph: &BeliefGraph,
    ) -> Result<f64, ExecError> {
        if pos_args.len() != 1 {
            return Err(ExecError::Internal(
                "prob_correlated() expects one positional comparison argument".into(),
            ));
        }

        let mut rho = 0.0f64;
        for arg in all_args {
            match arg {
                CallArg::Positional(_) => {}
                CallArg::Named { name, value } if name == "rho" => {
                    rho = eval_expr_core(value, graph, self)?;
                }
                CallArg::Named { name, .. } => {
                    return Err(ExecError::ValidationError(format!(
                        "prob_correlated(): unknown named argument '{}'; expected rho",
                        name
                    )))
                }
            }
        }

        if !rho.is_finite() || !(-1.0..=1.0).contains(&rho) {
            return Err(ExecError::ValidationError(
                "prob_correlated(): rho must be finite and in [-1, 1]".into(),
            ));
        }

        match &pos_args[0] {
            ExprAst::Binary { op, left, right } => {
                self.eval_prob_comparison(*op, left, right, rho, "prob_correlated", graph)
            }
            _ => Err(ExecError::ValidationError(
                "prob_correlated(): argument must be a comparison like A.x > B.x".into(),
            )),
        }
    }

    /// Evaluates Gaussian comparison probability `P(lhs OP rhs)` with optional correlation.
    fn eval_prob_comparison(
        &self,
        op: grafial_frontend::ast::BinaryOp,
        left: &ExprAst,
        right: &ExprAst,
        rho: f64,
        fn_name: &str,
        graph: &BeliefGraph,
    ) -> Result<f64, ExecError> {
        use grafial_frontend::ast::BinaryOp::*;
        match op {
            Gt | Ge | Lt | Le => {
                let (m_l, v_l) = self.extract_mean_var(left, graph)?;
                let (m_r, v_r) = self.extract_mean_var(right, graph)?;
                let m = m_l - m_r;
                let cov = rho * (v_l.max(0.0) * v_r.max(0.0)).sqrt();
                let s2 = v_l + v_r - (2.0 * cov);

                if !s2.is_finite() || s2 < -1e-12 {
                    return Err(ExecError::ValidationError(format!(
                        "{}(): implied variance is invalid (check rho and operand variances)",
                        fn_name
                    )));
                }
                if s2 <= 1e-18 {
                    // Deterministic fallback for near-zero variance.
                    let val = match op {
                        Gt | Ge => (m > 0.0) as i32 as f64,
                        Lt | Le => (m < 0.0) as i32 as f64,
                        _ => unreachable!(),
                    };
                    return Ok(val);
                }

                let z = m / s2.sqrt();
                let p_gt = crate::engine::expr_utils::norm_cdf(z);
                let p = match op {
                    Gt | Ge => p_gt,
                    Lt | Le => 1.0 - p_gt,
                    _ => unreachable!(),
                };
                Ok(p)
            }
            _ => Err(ExecError::ValidationError(format!(
                "{}(): only supports comparisons like {}(A > B)",
                fn_name, fn_name
            ))),
        }
    }

    /// Evaluates credible(event, p=..., rho=...) as a boolean (1.0/0.0).
    ///
    /// Semantics:
    /// - Compute event probability:
    ///   - edge variable: `P(edge exists)`
    ///   - comparison: Gaussian comparison probability (with optional `rho`)
    /// - Return 1.0 when `P(event) >= p`, else 0.0.
    ///
    /// Defaults: `p=0.95`, `rho=0.0`.
    fn eval_credible_function(
        &self,
        pos_args: &[ExprAst],
        all_args: &[CallArg],
        graph: &BeliefGraph,
    ) -> Result<f64, ExecError> {
        if pos_args.len() != 1 {
            return Err(ExecError::ValidationError(
                "credible(): expects one positional event argument".into(),
            ));
        }

        let mut threshold = 0.95f64;
        let mut rho = 0.0f64;
        for arg in all_args {
            match arg {
                CallArg::Positional(_) => {}
                CallArg::Named { name, value } if name == "p" => {
                    threshold = eval_expr_core(value, graph, self)?;
                }
                CallArg::Named { name, value } if name == "rho" => {
                    rho = eval_expr_core(value, graph, self)?;
                }
                CallArg::Named { name, .. } => {
                    return Err(ExecError::ValidationError(format!(
                        "credible(): unknown named argument '{}'; expected p or rho",
                        name
                    )))
                }
            }
        }

        if !threshold.is_finite() || !(0.0..=1.0).contains(&threshold) {
            return Err(ExecError::ValidationError(
                "credible(): p must be finite and in [0, 1]".into(),
            ));
        }
        if !rho.is_finite() || !(-1.0..=1.0).contains(&rho) {
            return Err(ExecError::ValidationError(
                "credible(): rho must be finite and in [-1, 1]".into(),
            ));
        }

        let event_prob = match &pos_args[0] {
            ExprAst::Var(var_name) => {
                let eid = self.resolve_edge_var(var_name)?;
                graph.prob_mean(eid)?
            }
            ExprAst::Binary { op, left, right } => {
                self.eval_prob_comparison(*op, left, right, rho, "credible", graph)?
            }
            _ => {
                return Err(ExecError::ValidationError(
                    "credible(): event must be an edge variable or comparison".into(),
                ))
            }
        };

        Ok((event_prob >= threshold) as i32 as f64)
    }

    /// Helper: extract mean and variance for simple expressions: node.attr or number
    fn extract_mean_var(
        &self,
        expr: &ExprAst,
        graph: &BeliefGraph,
    ) -> Result<(f64, f64), ExecError> {
        match expr {
            ExprAst::Number(x) => Ok((*x, 0.0)),
            ExprAst::Field { target, field } => match &**target {
                ExprAst::Var(var_name) => {
                    let nid = self.resolve_node_var(var_name)?;
                    let m = graph.expectation(nid, field)?;
                    let v = graph.variance(nid, field)?;
                    Ok((m, v))
                }
                _ => Err(ExecError::ValidationError(
                    "prob(): comparison must reference node variables".into(),
                )),
            },
            _ => Err(ExecError::ValidationError(
                "prob(): only supports node.attr or numeric in comparisons".into(),
            )),
        }
    }

    /// Evaluates degree(node, min_prob=0.0) - counts outgoing edges above threshold
    fn eval_degree_function(
        &self,
        pos_args: &[ExprAst],
        all_args: &[CallArg],
        graph: &BeliefGraph,
    ) -> Result<f64, ExecError> {
        if pos_args.is_empty() {
            return Err(ExecError::Internal(
                "degree(): missing node argument".into(),
            ));
        }

        let min_prob = self
            .extract_named_arg(all_args, "min_prob", graph)?
            .unwrap_or(0.0);

        match &pos_args[0] {
            ExprAst::Var(var_name) => {
                let nid = self.resolve_node_var(var_name)?;
                Ok(graph.degree_outgoing(nid, min_prob) as f64)
            }
            _ => Err(ExecError::Internal(
                "degree(): first argument must be a node variable".into(),
            )),
        }
    }

    /// Evaluates winner(node, edge_type, epsilon=0.01) - finds winning edge in competitive group
    fn eval_winner_function(
        &self,
        pos_args: &[ExprAst],
        all_args: &[CallArg],
        graph: &BeliefGraph,
    ) -> Result<f64, ExecError> {
        if pos_args.len() < 2 {
            return Err(ExecError::Internal(
                "winner(): requires node and edge_type arguments".into(),
            ));
        }

        let epsilon = self
            .extract_named_arg(all_args, "epsilon", graph)?
            .unwrap_or(0.01);

        let node_var = match &pos_args[0] {
            ExprAst::Var(v) => v,
            _ => {
                return Err(ExecError::Internal(
                    "winner(): first argument must be a node variable".into(),
                ))
            }
        };

        let edge_type = match &pos_args[1] {
            ExprAst::Var(v) => v.clone(),
            _ => {
                return Err(ExecError::Internal(
                    "winner(): edge_type must be an identifier".into(),
                ))
            }
        };

        let nid = self.resolve_node_var(node_var)?;

        match graph.winner(nid, &edge_type, epsilon) {
            Some(winner_node_id) => Ok(winner_node_id.0 as f64),
            None => Ok(-1.0),
        }
    }

    /// Evaluates entropy(node, edge_type) - computes entropy of competing edge distribution
    fn eval_entropy_function(
        &self,
        pos_args: &[ExprAst],
        _all_args: &[CallArg],
        graph: &BeliefGraph,
    ) -> Result<f64, ExecError> {
        if pos_args.len() < 2 {
            return Err(ExecError::Internal(
                "entropy(): requires node and edge_type arguments".into(),
            ));
        }

        let node_var = match &pos_args[0] {
            ExprAst::Var(v) => v,
            _ => {
                return Err(ExecError::Internal(
                    "entropy(): first argument must be a node variable".into(),
                ))
            }
        };

        let edge_type = match &pos_args[1] {
            ExprAst::Var(v) => v.clone(),
            _ => {
                return Err(ExecError::Internal(
                    "entropy(): edge_type must be an identifier".into(),
                ))
            }
        };

        let nid = self.resolve_node_var(node_var)?;
        // Gracefully handle nodes without competing groups: treat entropy as 0.0
        if let Some(group) = graph.get_competing_group(nid, &edge_type) {
            Ok(group.posterior.entropy())
        } else {
            Ok(0.0)
        }
    }

    /// Evaluates an exists subquery expression.
    ///
    /// Searches for a single edge matching the pattern with compatible variable bindings.
    /// Returns 1.0 if found (or 0.0 if negated), 0.0 if not found (or 1.0 if negated).
    ///
    /// Variables in the exists pattern can reference parent rule variables - if a variable
    /// is already bound, it must match the same graph element. This enables correlated
    /// subqueries like `exists (A)-[e:KNOWS]->(B) where prob(e) > 0.5`.
    pub fn eval_exists(
        &self,
        pattern: &grafial_frontend::ast::PatternItem,
        where_expr: &Option<Box<ExprAst>>,
        negated: bool,
        graph: &BeliefGraph,
    ) -> Result<f64, ExecError> {
        // Need to ensure graph has deltas applied so find_candidate_edges can see all edges
        // But we can't mutate graph, so we need find_candidate_edges to handle deltas
        // Actually, find_candidate_edges uses graph.edges() which only returns base edges
        // So we need to make find_candidate_edges delta-aware, or ensure graph has deltas applied
        let candidates = find_candidate_edges(graph, &pattern.edge.ty);

        for edge in candidates {
            if !matches_node_labels(graph, edge, pattern)? {
                continue;
            }

            // Build subquery bindings, validating compatibility with parent bindings
            // Subquery has single pattern, so estimate 2 nodes + 1 edge
            let mut subquery_bindings = MatchBindings {
                node_vars: HashMap::with_capacity(2),
                edge_vars: HashMap::with_capacity(1),
            };

            // Source node: must match parent binding if already bound
            if let Some(&parent_node) = self.bindings.node_vars.get(&pattern.src.var) {
                if parent_node != edge.src {
                    continue;
                }
            }
            subquery_bindings
                .node_vars
                .insert(pattern.src.var.clone(), edge.src);

            // Destination node: must match parent binding if already bound
            if let Some(&parent_node) = self.bindings.node_vars.get(&pattern.dst.var) {
                if parent_node != edge.dst {
                    continue;
                }
            }
            subquery_bindings
                .node_vars
                .insert(pattern.dst.var.clone(), edge.dst);

            // Edge variable: must match parent binding if already bound
            if let Some(&parent_edge) = self.bindings.edge_vars.get(&pattern.edge.var) {
                if parent_edge != edge.id {
                    continue;
                }
            }
            subquery_bindings
                .edge_vars
                .insert(pattern.edge.var.clone(), edge.id);

            // Merge with parent bindings for where clause evaluation (subquery can reference both)
            let mut merged_bindings = self.bindings.clone();
            merged_bindings
                .node_vars
                .extend(subquery_bindings.node_vars);
            merged_bindings
                .edge_vars
                .extend(subquery_bindings.edge_vars);

            // Evaluate where clause if present
            if let Some(w) = where_expr {
                let subquery_ctx = RuleExprContext {
                    bindings: &merged_bindings,
                    locals: self.locals,
                    globals: self.globals,
                };
                let v = eval_where_with_exists(w, graph, &subquery_ctx)?;
                if v == 0.0 {
                    continue;
                }
            }

            return Ok(if negated { 0.0 } else { 1.0 });
        }

        Ok(if negated { 1.0 } else { 0.0 })
    }
}

/// Evaluates a where clause with bound variables, returning true if it passes.
///
/// Exists subqueries require special handling (pattern matching), so we use
/// a separate evaluation path rather than the standard expression evaluator.
fn evaluate_where_clause(
    where_expr: &Option<ExprAst>,
    bindings: &MatchBindings,
    globals: &HashMap<String, f64>,
    graph: &BeliefGraph,
) -> Result<bool, ExecError> {
    let Some(expr) = where_expr else {
        return Ok(true);
    };

    let where_locals = Locals::new();
    let where_ctx = RuleExprContext {
        bindings,
        locals: &where_locals,
        globals,
    };

    // DEBUG: Print edge probabilities for diagnosis
    let v = eval_where_with_exists(expr, graph, &where_ctx)?;

    Ok(v != 0.0)
}

/// Evaluates a where clause expression, handling exists subqueries.
///
/// Exists expressions require pattern matching, so they're handled specially
/// rather than going through the standard expression evaluator.
fn eval_where_with_exists(
    expr: &ExprAst,
    graph: &BeliefGraph,
    ctx: &RuleExprContext,
) -> Result<f64, ExecError> {
    match expr {
        ExprAst::Exists {
            pattern,
            where_expr,
            negated,
        } => ctx.eval_exists(pattern, where_expr, *negated, graph),
        ExprAst::Binary { op, left, right } => {
            let l = eval_where_with_exists(left, graph, ctx)?;
            let r = eval_where_with_exists(right, graph, ctx)?;
            eval_binary_op(*op, l, r)
        }
        ExprAst::Unary { op, expr } => {
            let v = eval_where_with_exists(expr, graph, ctx)?;
            Ok(eval_unary_op(*op, v))
        }
        _ => {
            // For all other expression types, use the standard evaluator
            eval_expr_core(expr, graph, ctx)
        }
    }
}

/// Executes action statements against a belief graph.
///
/// Actions are executed in order, with local variables from `let` statements
/// available to subsequent actions.
///
/// # Action Semantics
///
/// - `Let`: Evaluates an expression and stores it in a local variable
/// - `SetExpectation`: Updates a node attribute's expected value (soft update)
///   - Adjusts mean without changing precision (variance)
/// - `ForceAbsent`: Forces an edge to be absent with high certainty
///   - Sets Beta parameters to α=1, β=1e6 (mean ≈ 0.000001)
pub fn execute_actions(
    graph: &mut BeliefGraph,
    actions: &[ActionStmt],
    bindings: &MatchBindings,
    globals: &HashMap<String, f64>,
) -> Result<(), ExecError> {
    let mut locals = Locals::new();

    for action in actions {
        let ctx = RuleExprContext {
            bindings,
            locals: &locals,
            globals,
        };

        match action {
            ActionStmt::Let { name, expr } => {
                let value = eval_expr_core(expr, graph, &ctx)?;
                locals.set(name.clone(), value);
            }
            ActionStmt::SetExpectation {
                node_var,
                attr,
                expr,
            } => {
                let nid = ctx.resolve_node_var(node_var)?;
                let value = eval_expr_core(expr, graph, &ctx)?;
                graph.set_expectation(nid, attr, value)?;
            }
            ActionStmt::ForceAbsent { edge_var } => {
                let eid = ctx.resolve_edge_var(edge_var)?;
                graph.force_absent(eid)?;
            }
            ActionStmt::NonBayesianNudge {
                node_var,
                attr,
                expr,
                variance,
            } => {
                let nid = ctx.resolve_node_var(node_var)?;
                let value = eval_expr_core(expr, graph, &ctx)?;
                let variance = variance
                    .as_ref()
                    .unwrap_or(&grafial_frontend::ast::VarianceSpec::Preserve);
                graph.non_bayesian_nudge(nid, attr, value, variance)?;
            }
            ActionStmt::SoftUpdate {
                node_var,
                attr,
                expr,
                precision,
                count,
            } => {
                let nid = ctx.resolve_node_var(node_var)?;
                let value = eval_expr_core(expr, graph, &ctx)?;
                let tau = precision.unwrap_or(1.0);
                let c = count.unwrap_or(1.0);
                graph.soft_update(nid, attr, value, tau, c)?;
            }
            ActionStmt::DeleteEdge {
                edge_var,
                confidence,
            } => {
                let eid = ctx.resolve_edge_var(edge_var)?;
                graph.delete_edge(eid, confidence.as_deref())?;
            }
            ActionStmt::SuppressEdge { edge_var, weight } => {
                let eid = ctx.resolve_edge_var(edge_var)?;
                graph.suppress_edge(eid, *weight)?;
            }
        }
    }
    Ok(())
}

/// Computes the maximum absolute change between two graphs.
///
/// Used for fixpoint convergence detection. Compares:
/// - Node attribute means (Gaussian posteriors)
/// - Edge existence probabilities (Beta posteriors)
///
/// # Arguments
///
/// * `old` - The previous graph state
/// * `new` - The new graph state
///
/// # Returns
///
/// The maximum absolute difference across all attributes and probabilities.
/// Returns 0.0 if graphs are identical or if either graph is empty.
///
/// Used to determine if a fixpoint rule has converged. Compares posterior means
/// and probabilities between iterations to detect when changes fall below the
/// convergence threshold.
fn compute_max_change(old: &BeliefGraph, new: &BeliefGraph) -> f64 {
    let mut max_delta: f64 = 0.0;

    // Compare node attributes
    for (old_node, new_node) in old.nodes().iter().zip(new.nodes().iter()) {
        if old_node.id != new_node.id {
            // Graph structure changed, return large delta
            return f64::INFINITY;
        }
        for (attr_name, old_attr) in &old_node.attrs {
            if let Some(new_attr) = new_node.attrs.get(attr_name) {
                let delta = (old_attr.mean - new_attr.mean).abs();
                max_delta = max_delta.max(delta);
            }
        }
    }

    // Compare edge probabilities
    for (old_edge, new_edge) in old.edges().iter().zip(new.edges().iter()) {
        if old_edge.id != new_edge.id {
            // Graph structure changed, return large delta
            return f64::INFINITY;
        }
        let old_prob = old_edge
            .exist
            .mean_probability(old.competing_groups())
            .unwrap_or(0.0);
        let new_prob = new_edge
            .exist
            .mean_probability(new.competing_groups())
            .unwrap_or(0.0);
        let delta = (old_prob - new_prob).abs();
        max_delta = max_delta.max(delta);
    }

    max_delta
}

/// Executes a rule in "for_each" mode over all matching patterns.
///
/// Finds all graph patterns matching the rule, evaluates the optional `where` clause,
/// and applies actions to a working copy for each successful match.
///
/// # Current Limitations
///
/// - Supports exactly one pattern item (no multi-pattern rules yet)
///
/// Execution model:
/// 1. Find all pattern matches (deterministic order by EdgeId)
/// 2. Filter matches by where clause
/// 3. Apply actions to working copy (immutable input preserved)
/// 4. Return new graph with changes applied
pub fn run_rule_for_each(input: &BeliefGraph, rule: &RuleDef) -> Result<BeliefGraph, ExecError> {
    run_rule_for_each_with_globals(input, rule, &HashMap::new())
}

/// Executes a rule with access to global scalar variables (e.g., imported metrics).
///
/// Supports both single-pattern and multi-pattern rules. For multi-pattern rules,
/// finds all combinations of edges that match all patterns with shared variables.
/// Uses deterministic iteration over sorted EdgeIds for reproducibility.
pub fn run_rule_for_each_with_globals(
    input: &BeliefGraph,
    rule: &RuleDef,
    globals: &HashMap<String, f64>,
) -> Result<BeliefGraph, ExecError> {
    if rule.patterns.is_empty() {
        return Err(ExecError::ValidationError(
            "rule must have at least one pattern".into(),
        ));
    }

    // Single-pattern path: optimized for the common case (no join needed)
    if rule.patterns.len() == 1 {
        let pat = &rule.patterns[0];
        // Special-case: node-only iteration sugar uses a dummy edge type
        if pat.edge.ty == "__FOR_NODE__" {
            // First pass: collect matching nodes by label
            let mut matches = Vec::new();
            for node in input.nodes() {
                if node.label.as_ref() == pat.src.label.as_str() {
                    let mut bindings = MatchBindings::with_capacity(&rule.patterns);
                    // Bind both src and dst to the same node
                    bindings.node_vars.insert(pat.src.var.clone(), node.id);
                    bindings.node_vars.insert(pat.dst.var.clone(), node.id);
                    matches.push(bindings);
                }
            }

            // Filter matches by where clause
            let mut filtered = Vec::new();
            for b in matches.into_iter() {
                if evaluate_where_clause(&rule.where_expr, &b, globals, input)? {
                    filtered.push(b);
                }
            }

            if filtered.is_empty() {
                return Ok(input.clone());
            }

            let mut work = input.clone();
            work.ensure_owned();
            for b in filtered.iter() {
                execute_actions(&mut work, &rule.actions, b, globals)?;
            }
            return Ok(work);
        }
        // Use input graph for candidate finding (we want to iterate over original edges)
        let candidates = find_candidate_edges(input, &pat.edge.ty);

        // First pass: collect matches (read-only, no clone needed)
        let mut matches = Vec::new();
        for edge in candidates {
            if !matches_node_labels(input, edge, pat)? {
                continue;
            }

            let mut bindings = MatchBindings::with_capacity(&rule.patterns);
            bindings.node_vars.insert(pat.src.var.clone(), edge.src);
            bindings.node_vars.insert(pat.dst.var.clone(), edge.dst);
            bindings.edge_vars.insert(pat.edge.var.clone(), edge.id);

            // Evaluate where clause against input graph (original state)
            // This ensures where clause evaluation is consistent and doesn't depend on action order
            if !evaluate_where_clause(&rule.where_expr, &bindings, globals, input)? {
                continue;
            }

            matches.push(bindings);
        }

        // If no matches, return input graph (no clone needed)
        if matches.is_empty() {
            return Ok(input.clone());
        }

        // Second pass: clone only if we have matches to process
        // Clone and ensure deltas are applied so find_candidate_edges can see all edges
        let mut work = input.clone();
        work.ensure_owned();

        // Apply actions for all matches
        for bindings in matches {
            execute_actions(&mut work, &rule.actions, &bindings, globals)?;
        }

        return Ok(work);
    }

    // Multi-pattern matching: use query plan to optimize join order
    let mut plan_cache = QueryPlanCache::new();
    let plan = plan_cache.get_or_create(&rule.patterns, input);
    let ordered_patterns: Vec<PatternItem> = plan
        .ordered_patterns
        .iter()
        .map(|&idx| rule.patterns[idx].clone())
        .collect();

    // First pass: collect all matches (read-only, no clone needed)
    let mut matches = Vec::new();
    let initial_bindings = MatchBindings::with_capacity(&rule.patterns);
    find_multi_pattern_matches(input, &ordered_patterns, 0, &initial_bindings, &mut matches)?;

    // Deterministic ordering: sort by EdgeId tuples for reproducibility
    // Use unstable sort (faster) since EdgeId ordering is stable and deterministic
    matches.sort_unstable_by(|a, b| {
        // Collect and sort edge IDs for comparison
        let mut a_edges: Vec<_> = a.edge_vars.values().cloned().collect();
        let mut b_edges: Vec<_> = b.edge_vars.values().cloned().collect();
        a_edges.sort_unstable();
        b_edges.sort_unstable();
        a_edges.cmp(&b_edges)
    });

    // Filter matches by where clause (still read-only)
    // Debug logging removed for clean output

    let mut filtered_matches = Vec::new();
    for bindings in &matches {
        if evaluate_where_clause(&rule.where_expr, bindings, globals, input)? {
            filtered_matches.push(bindings.clone());
        }
    }

    // If no matches, return input graph (no clone needed)
    if filtered_matches.is_empty() {
        return Ok(input.clone());
    }

    // Debug logging removed for clean output

    // Second pass: clone only if we have matches to process
    let mut work = input.clone();
    work.ensure_owned();

    // Apply actions to working copy
    for bindings in &filtered_matches {
        execute_actions(&mut work, &rule.actions, bindings, globals)?;
    }

    Ok(work)
}

/// Finds candidate edges matching a pattern's edge type, sorted deterministically.
///
/// Deterministic ordering by EdgeId ensures reproducible rule execution across runs.
fn find_candidate_edges<'a>(
    graph: &'a BeliefGraph,
    edge_type: &str,
) -> Vec<&'a crate::engine::graph::EdgeData> {
    // Collect edges from both base and delta
    let mut candidates: Vec<_> = Vec::new();

    // First, collect all edge IDs we've seen (to avoid duplicates)
    let mut seen_ids = std::collections::HashSet::new();

    // Arc<str> implements PartialEq with &str, so we can compare directly using .as_ref()
    // Add edges from base graph
    for edge in graph.edges() {
        if edge.ty.as_ref() == edge_type {
            seen_ids.insert(edge.id);
            candidates.push(edge);
        }
    }

    // Add edges from delta (checking for duplicates)
    for change in graph.delta() {
        if let crate::engine::graph::GraphDelta::EdgeChange { id, edge } = change {
            if edge.ty.as_ref() == edge_type && !seen_ids.contains(id) {
                seen_ids.insert(*id);
                candidates.push(edge);
            }
        }
    }

    // Use unstable sort (faster) since EdgeId ordering is stable and deterministic
    candidates.sort_unstable_by_key(|e| e.id);
    candidates
}

/// Checks if an edge matches a pattern's node labels.
fn matches_node_labels(
    graph: &BeliefGraph,
    edge: &crate::engine::graph::EdgeData,
    pattern: &PatternItem,
) -> Result<bool, ExecError> {
    let src = graph
        .node(edge.src)
        .ok_or_else(|| ExecError::Internal("missing src node".into()))?;
    let dst = graph
        .node(edge.dst)
        .ok_or_else(|| ExecError::Internal("missing dst node".into()))?;
    Ok(src.label.as_ref() == pattern.src.label.as_str()
        && dst.label.as_ref() == pattern.dst.label.as_str())
}

/// Attempts to extend bindings with a pattern match, validating variable compatibility.
///
/// Returns `Some(new_bindings)` if the edge matches and all variables are compatible,
/// `None` if variables conflict (e.g., shared variable bound to different node).
///
/// Shared variables across patterns must bind to the same graph element - this
/// enforces the join semantics in multi-pattern rules.
fn try_extend_bindings(
    current_bindings: &MatchBindings,
    pattern: &PatternItem,
    edge: &crate::engine::graph::EdgeData,
) -> Option<MatchBindings> {
    let mut new_bindings = current_bindings.clone();

    // Validate source node variable compatibility
    if let Some(&existing) = current_bindings.node_vars.get(&pattern.src.var) {
        if existing != edge.src {
            return None;
        }
    }
    new_bindings
        .node_vars
        .insert(pattern.src.var.clone(), edge.src);

    // Validate destination node variable compatibility
    if let Some(&existing) = current_bindings.node_vars.get(&pattern.dst.var) {
        if existing != edge.dst {
            return None;
        }
    }
    new_bindings
        .node_vars
        .insert(pattern.dst.var.clone(), edge.dst);

    // Edge variables must be unique per pattern (no conflicts possible in well-formed rules)
    if current_bindings.edge_vars.contains_key(&pattern.edge.var) {
        return None;
    }
    new_bindings
        .edge_vars
        .insert(pattern.edge.var.clone(), edge.id);

    Some(new_bindings)
}

/// Recursively finds all matches for multi-pattern rules.
///
/// Builds up bindings pattern-by-pattern, ensuring shared variables (from previous
/// patterns) match consistently. This implements the join semantics for multi-pattern rules.
fn find_multi_pattern_matches(
    graph: &BeliefGraph,
    patterns: &[PatternItem],
    pattern_idx: usize,
    current_bindings: &MatchBindings,
    matches: &mut Vec<MatchBindings>,
) -> Result<(), ExecError> {
    if pattern_idx >= patterns.len() {
        matches.push(current_bindings.clone());
        return Ok(());
    }

    let pat = &patterns[pattern_idx];
    let candidates = find_candidate_edges(graph, &pat.edge.ty);

    for edge in candidates {
        if !matches_node_labels(graph, edge, pat)? {
            continue;
        }

        let Some(new_bindings) = try_extend_bindings(current_bindings, pat, edge) else {
            continue;
        };

        find_multi_pattern_matches(graph, patterns, pattern_idx + 1, &new_bindings, matches)?;
    }

    Ok(())
}

/// Executes a rule in "fixpoint" mode until convergence or iteration limit.
///
/// Applies the rule repeatedly until convergence (changes < FIXPOINT_TOLERANCE)
/// or MAX_FIXPOINT_ITERATIONS is reached.
///
/// Fixpoint iteration strategy:
/// - Run rule, compare output to input
/// - If max change < FIXPOINT_TOLERANCE, converged
/// - Otherwise, use output as new input and repeat
/// - Terminates after MAX_FIXPOINT_ITERATIONS to prevent infinite loops
pub fn run_rule_fixpoint(
    input: &BeliefGraph,
    rule: &RuleDef,
    globals: &HashMap<String, f64>,
) -> Result<BeliefGraph, ExecError> {
    let mut current = input.clone();

    for _iteration in 0..MAX_FIXPOINT_ITERATIONS {
        let next = run_rule_for_each_with_globals(&current, rule, globals)?;
        let max_change = compute_max_change(&current, &next);

        // Check for convergence
        if max_change < FIXPOINT_TOLERANCE {
            #[cfg(feature = "tracing")]
            tracing::debug!(
                "Fixpoint converged after {} iterations (max_change = {:.2e})",
                _iteration + 1,
                max_change
            );
            return Ok(next);
        }

        current = next;
    }

    // Iteration limit reached without convergence
    Err(ExecError::Execution(format!(
        "Fixpoint rule '{}' did not converge after {} iterations",
        rule.name, MAX_FIXPOINT_ITERATIONS
    )))
}

/// Public API for running rules with automatic mode detection.
///
/// Dispatches to `run_rule_for_each_with_globals` or `run_rule_fixpoint` based on mode.
pub fn run_rule(
    input: &BeliefGraph,
    rule: &RuleDef,
    globals: &HashMap<String, f64>,
) -> Result<BeliefGraph, ExecError> {
    match rule.mode.as_deref() {
        Some("fixpoint") => run_rule_fixpoint(input, rule, globals),
        Some("for_each") | None => run_rule_for_each_with_globals(input, rule, globals),
        Some(other) => Err(ExecError::ValidationError(format!(
            "unknown rule mode: '{}' (expected 'for_each' or 'fixpoint')",
            other
        ))),
    }
}

/// Executes an IR rule in "for_each" mode with global scalar bindings.
pub fn run_rule_ir_for_each_with_globals(
    input: &BeliefGraph,
    rule: &RuleIR,
    globals: &HashMap<String, f64>,
) -> Result<BeliefGraph, ExecError> {
    let ast_rule = rule.to_ast();
    run_rule_for_each_with_globals(input, &ast_rule, globals)
}

/// Executes an IR rule using its configured mode with global scalar bindings.
pub fn run_rule_ir(
    input: &BeliefGraph,
    rule: &RuleIR,
    globals: &HashMap<String, f64>,
) -> Result<BeliefGraph, ExecError> {
    let ast_rule = rule.to_ast();
    run_rule(input, &ast_rule, globals)
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::engine::graph::{
        BeliefGraph, BetaPosterior, EdgeId, GaussianPosterior, NodeData, NodeId,
    };
    use grafial_frontend::ast::{BinaryOp, CallArg, EdgePattern, NodePattern};
    use std::sync::Arc;

    // Helper to create a simple test graph
    fn create_test_graph() -> BeliefGraph {
        let mut g = BeliefGraph::default();
        g.insert_node(NodeData {
            id: NodeId(1),
            label: Arc::from("Person"),
            attrs: HashMap::from([(
                "x".into(),
                GaussianPosterior {
                    mean: 10.0,
                    precision: 1.0,
                },
            )]),
        });
        g.insert_node(NodeData {
            id: NodeId(2),
            label: Arc::from("Person"),
            attrs: HashMap::from([(
                "x".into(),
                GaussianPosterior {
                    mean: 5.0,
                    precision: 1.0,
                },
            )]),
        });
        g.insert_edge(BeliefGraph::test_edge_with_beta(
            EdgeId(1),
            NodeId(1),
            NodeId(2),
            "KNOWS".into(),
            BetaPosterior {
                alpha: 5.0,
                beta: 5.0,
            },
        ));
        // Apply delta to ensure items are in base for iteration
        g.ensure_owned();
        g
    }

    fn create_uncertain_compare_graph() -> BeliefGraph {
        let mut g = BeliefGraph::default();
        g.insert_node(NodeData {
            id: NodeId(1),
            label: Arc::from("Person"),
            attrs: HashMap::from([(
                "x".into(),
                GaussianPosterior {
                    mean: 0.1,
                    precision: 1.0, // var = 1.0
                },
            )]),
        });
        g.insert_node(NodeData {
            id: NodeId(2),
            label: Arc::from("Person"),
            attrs: HashMap::from([(
                "x".into(),
                GaussianPosterior {
                    mean: 0.0,
                    precision: 1.0, // var = 1.0
                },
            )]),
        });
        g.ensure_owned();
        g
    }

    // ============================================================================
    // eval_expr Tests (now using shared evaluator)
    // ============================================================================

    #[test]
    fn eval_expr_number() {
        let g = BeliefGraph::default();
        let bindings = MatchBindings::default();
        let locals = Locals::new();
        let ctx = RuleExprContext {
            bindings: &bindings,
            locals: &locals,
            globals: &HashMap::new(),
        };

        let result = eval_expr_core(&ExprAst::Number(42.5), &g, &ctx).unwrap();
        assert!((result - 42.5).abs() < 1e-9);
    }

    #[test]
    fn eval_expr_bool() {
        let g = BeliefGraph::default();
        let bindings = MatchBindings::default();
        let locals = Locals::new();
        let ctx = RuleExprContext {
            bindings: &bindings,
            locals: &locals,
            globals: &HashMap::new(),
        };

        let result = eval_expr_core(&ExprAst::Bool(true), &g, &ctx).unwrap();
        assert_eq!(result, 1.0);
        let result = eval_expr_core(&ExprAst::Bool(false), &g, &ctx).unwrap();
        assert_eq!(result, 0.0);
    }

    #[test]
    fn eval_expr_var() {
        let g = BeliefGraph::default();
        let mut locals = Locals::new();
        locals.set("x".into(), 10.0);
        let bindings = MatchBindings::default();
        let ctx = RuleExprContext {
            bindings: &bindings,
            locals: &locals,
            globals: &HashMap::new(),
        };

        let result = eval_expr_core(&ExprAst::Var("x".into()), &g, &ctx).unwrap();
        assert_eq!(result, 10.0);
    }

    #[test]
    fn eval_expr_binary_arithmetic() {
        let g = BeliefGraph::default();
        let bindings = MatchBindings::default();
        let locals = Locals::new();
        let ctx = RuleExprContext {
            bindings: &bindings,
            locals: &locals,
            globals: &HashMap::new(),
        };

        use grafial_frontend::ast::BinaryOp;
        let expr = ExprAst::Binary {
            op: BinaryOp::Add,
            left: Box::new(ExprAst::Number(3.0)),
            right: Box::new(ExprAst::Number(4.0)),
        };
        let result = eval_expr_core(&expr, &g, &ctx).unwrap();
        assert_eq!(result, 7.0);

        let expr = ExprAst::Binary {
            op: BinaryOp::Mul,
            left: Box::new(ExprAst::Number(3.0)),
            right: Box::new(ExprAst::Number(4.0)),
        };
        let result = eval_expr_core(&expr, &g, &ctx).unwrap();
        assert_eq!(result, 12.0);
    }

    #[test]
    fn eval_expr_binary_comparison() {
        let g = BeliefGraph::default();
        let bindings = MatchBindings::default();
        let locals = Locals::new();
        let ctx = RuleExprContext {
            bindings: &bindings,
            locals: &locals,
            globals: &HashMap::new(),
        };

        use grafial_frontend::ast::BinaryOp;
        let expr = ExprAst::Binary {
            op: BinaryOp::Eq,
            left: Box::new(ExprAst::Number(5.0)),
            right: Box::new(ExprAst::Number(5.0)),
        };
        assert_eq!(eval_expr_core(&expr, &g, &ctx).unwrap(), 1.0);

        let expr = ExprAst::Binary {
            op: BinaryOp::Lt,
            left: Box::new(ExprAst::Number(3.0)),
            right: Box::new(ExprAst::Number(5.0)),
        };
        assert_eq!(eval_expr_core(&expr, &g, &ctx).unwrap(), 1.0);
    }

    #[test]
    fn eval_expr_unary() {
        let g = BeliefGraph::default();
        let bindings = MatchBindings::default();
        let locals = Locals::new();
        let ctx = RuleExprContext {
            bindings: &bindings,
            locals: &locals,
            globals: &HashMap::new(),
        };

        use grafial_frontend::ast::UnaryOp;
        let expr = ExprAst::Unary {
            op: UnaryOp::Neg,
            expr: Box::new(ExprAst::Number(5.0)),
        };
        assert_eq!(eval_expr_core(&expr, &g, &ctx).unwrap(), -5.0);

        let expr = ExprAst::Unary {
            op: UnaryOp::Not,
            expr: Box::new(ExprAst::Number(0.0)),
        };
        assert_eq!(eval_expr_core(&expr, &g, &ctx).unwrap(), 1.0);
    }

    #[test]
    fn eval_expr_logical_ops() {
        let g = BeliefGraph::default();
        let bindings = MatchBindings::default();
        let locals = Locals::new();
        let ctx = RuleExprContext {
            bindings: &bindings,
            locals: &locals,
            globals: &HashMap::new(),
        };

        use grafial_frontend::ast::BinaryOp;
        let expr = ExprAst::Binary {
            op: BinaryOp::And,
            left: Box::new(ExprAst::Number(1.0)),
            right: Box::new(ExprAst::Number(1.0)),
        };
        assert_eq!(eval_expr_core(&expr, &g, &ctx).unwrap(), 1.0);

        let expr = ExprAst::Binary {
            op: BinaryOp::And,
            left: Box::new(ExprAst::Number(1.0)),
            right: Box::new(ExprAst::Number(0.0)),
        };
        assert_eq!(eval_expr_core(&expr, &g, &ctx).unwrap(), 0.0);

        let expr = ExprAst::Binary {
            op: BinaryOp::Or,
            left: Box::new(ExprAst::Number(0.0)),
            right: Box::new(ExprAst::Number(1.0)),
        };
        assert_eq!(eval_expr_core(&expr, &g, &ctx).unwrap(), 1.0);
    }

    #[test]
    fn eval_prob_correlated_changes_comparison_probability() {
        let g = create_uncertain_compare_graph();
        let mut bindings = MatchBindings::default();
        bindings.node_vars.insert("A".into(), NodeId(1));
        bindings.node_vars.insert("B".into(), NodeId(2));
        let locals = Locals::new();
        let ctx = RuleExprContext {
            bindings: &bindings,
            locals: &locals,
            globals: &HashMap::new(),
        };

        let cmp = ExprAst::Binary {
            op: BinaryOp::Gt,
            left: Box::new(ExprAst::Field {
                target: Box::new(ExprAst::Var("A".into())),
                field: "x".into(),
            }),
            right: Box::new(ExprAst::Field {
                target: Box::new(ExprAst::Var("B".into())),
                field: "x".into(),
            }),
        };

        let p_ind = eval_expr_core(
            &ExprAst::Call {
                name: "prob".into(),
                args: vec![CallArg::Positional(cmp.clone())],
            },
            &g,
            &ctx,
        )
        .expect("prob");

        let p_pos_corr = eval_expr_core(
            &ExprAst::Call {
                name: "prob_correlated".into(),
                args: vec![
                    CallArg::Positional(cmp.clone()),
                    CallArg::Named {
                        name: "rho".into(),
                        value: ExprAst::Number(0.9),
                    },
                ],
            },
            &g,
            &ctx,
        )
        .expect("prob_correlated positive rho");

        let p_neg_corr = eval_expr_core(
            &ExprAst::Call {
                name: "prob_correlated".into(),
                args: vec![
                    CallArg::Positional(cmp),
                    CallArg::Named {
                        name: "rho".into(),
                        value: ExprAst::Number(-0.9),
                    },
                ],
            },
            &g,
            &ctx,
        )
        .expect("prob_correlated negative rho");

        assert!(
            p_pos_corr > p_ind,
            "positive correlation should raise P(A > B)"
        );
        assert!(
            p_neg_corr < p_ind,
            "negative correlation should lower P(A > B)"
        );
    }

    #[test]
    fn eval_prob_correlated_rejects_invalid_rho() {
        let g = create_uncertain_compare_graph();
        let mut bindings = MatchBindings::default();
        bindings.node_vars.insert("A".into(), NodeId(1));
        bindings.node_vars.insert("B".into(), NodeId(2));
        let locals = Locals::new();
        let ctx = RuleExprContext {
            bindings: &bindings,
            locals: &locals,
            globals: &HashMap::new(),
        };

        let expr = ExprAst::Call {
            name: "prob_correlated".into(),
            args: vec![
                CallArg::Positional(ExprAst::Binary {
                    op: BinaryOp::Gt,
                    left: Box::new(ExprAst::Field {
                        target: Box::new(ExprAst::Var("A".into())),
                        field: "x".into(),
                    }),
                    right: Box::new(ExprAst::Field {
                        target: Box::new(ExprAst::Var("B".into())),
                        field: "x".into(),
                    }),
                }),
                CallArg::Named {
                    name: "rho".into(),
                    value: ExprAst::Number(1.2),
                },
            ],
        };

        let err = eval_expr_core(&expr, &g, &ctx).expect_err("invalid rho should fail");
        assert!(err.to_string().contains("rho"));
    }

    #[test]
    fn eval_credible_on_edge_and_comparison_events() {
        let g = create_test_graph();
        let mut bindings = MatchBindings::default();
        bindings.node_vars.insert("A".into(), NodeId(1));
        bindings.node_vars.insert("B".into(), NodeId(2));
        bindings.edge_vars.insert("e".into(), EdgeId(1));
        let locals = Locals::new();
        let ctx = RuleExprContext {
            bindings: &bindings,
            locals: &locals,
            globals: &HashMap::new(),
        };

        let edge_true = eval_expr_core(
            &ExprAst::Call {
                name: "credible".into(),
                args: vec![
                    CallArg::Positional(ExprAst::Var("e".into())),
                    CallArg::Named {
                        name: "p".into(),
                        value: ExprAst::Number(0.4),
                    },
                ],
            },
            &g,
            &ctx,
        )
        .expect("credible edge true");
        assert_eq!(edge_true, 1.0);

        let edge_false = eval_expr_core(
            &ExprAst::Call {
                name: "credible".into(),
                args: vec![
                    CallArg::Positional(ExprAst::Var("e".into())),
                    CallArg::Named {
                        name: "p".into(),
                        value: ExprAst::Number(0.6),
                    },
                ],
            },
            &g,
            &ctx,
        )
        .expect("credible edge false");
        assert_eq!(edge_false, 0.0);

        let cmp_true = eval_expr_core(
            &ExprAst::Call {
                name: "credible".into(),
                args: vec![
                    CallArg::Positional(ExprAst::Binary {
                        op: BinaryOp::Gt,
                        left: Box::new(ExprAst::Field {
                            target: Box::new(ExprAst::Var("A".into())),
                            field: "x".into(),
                        }),
                        right: Box::new(ExprAst::Field {
                            target: Box::new(ExprAst::Var("B".into())),
                            field: "x".into(),
                        }),
                    }),
                    CallArg::Named {
                        name: "p".into(),
                        value: ExprAst::Number(0.95),
                    },
                ],
            },
            &g,
            &ctx,
        )
        .expect("credible comparison true");
        assert_eq!(cmp_true, 1.0);
    }

    #[test]
    fn eval_credible_rejects_invalid_threshold() {
        let g = create_test_graph();
        let mut bindings = MatchBindings::default();
        bindings.edge_vars.insert("e".into(), EdgeId(1));
        let locals = Locals::new();
        let ctx = RuleExprContext {
            bindings: &bindings,
            locals: &locals,
            globals: &HashMap::new(),
        };

        let expr = ExprAst::Call {
            name: "credible".into(),
            args: vec![
                CallArg::Positional(ExprAst::Var("e".into())),
                CallArg::Named {
                    name: "p".into(),
                    value: ExprAst::Number(1.5),
                },
            ],
        };
        let err = eval_expr_core(&expr, &g, &ctx).expect_err("invalid p should fail");
        assert!(err.to_string().contains("credible(): p"));
    }

    #[test]
    fn eval_expr_globals() {
        let g = BeliefGraph::default();
        let bindings = MatchBindings::default();
        let locals = Locals::new();
        let mut globals = HashMap::new();
        globals.insert("global_var".into(), 99.0);
        let ctx = RuleExprContext {
            bindings: &bindings,
            locals: &locals,
            globals: &globals,
        };

        let result = eval_expr_core(&ExprAst::Var("global_var".into()), &g, &ctx).unwrap();
        assert_eq!(result, 99.0);
    }

    // ============================================================================
    // execute_actions Tests
    // ============================================================================

    #[test]
    fn actions_set_expectation_and_force_absent() {
        let mut g = create_test_graph();
        let mut bindings = MatchBindings::default();
        bindings.node_vars.insert("A".into(), NodeId(1));
        bindings.edge_vars.insert("e".into(), EdgeId(1));

        let actions = vec![
            ActionStmt::Let {
                name: "v_ab".into(),
                expr: ExprAst::Binary {
                    op: grafial_frontend::ast::BinaryOp::Div,
                    left: Box::new(ExprAst::Call {
                        name: "E".into(),
                        args: vec![CallArg::Positional(ExprAst::Field {
                            target: Box::new(ExprAst::Var("A".into())),
                            field: "x".into(),
                        })],
                    }),
                    right: Box::new(ExprAst::Number(2.0)),
                },
            },
            ActionStmt::SetExpectation {
                node_var: "A".into(),
                attr: "x".into(),
                expr: ExprAst::Var("v_ab".into()),
            },
            ActionStmt::ForceAbsent {
                edge_var: "e".into(),
            },
        ];

        execute_actions(&mut g, &actions, &bindings, &HashMap::new()).unwrap();

        let prob = g.prob_mean(EdgeId(1)).unwrap();
        assert!(prob < 0.01, "edge should be forced absent");
    }

    #[test]
    fn run_rule_for_each_multi_pattern_chain() {
        // Test: (A)-[ab]->(B), (B)-[bc]->(C) - find chains of two edges
        let mut g = BeliefGraph::default();

        // Create nodes
        let n1 = g.add_node("Person".into(), HashMap::new());
        let n2 = g.add_node("Person".into(), HashMap::new());
        let n3 = g.add_node("Person".into(), HashMap::new());

        // Create edges: n1 -> n2 -> n3 (chain)
        let e1 = g.add_edge(
            n1,
            n2,
            "REL".into(),
            BetaPosterior {
                alpha: 9.0,
                beta: 1.0,
            },
        );
        let e2 = g.add_edge(
            n2,
            n3,
            "REL".into(),
            BetaPosterior {
                alpha: 8.0,
                beta: 2.0,
            },
        );

        // Create another edge that doesn't form a chain
        let _e3 = g.add_edge(
            n1,
            n3,
            "REL".into(),
            BetaPosterior {
                alpha: 1.0,
                beta: 9.0,
            },
        );

        // Apply delta to ensure all items are in base for iteration
        g.ensure_owned();

        let rule = RuleDef {
            name: "ChainRule".into(),
            on_model: "M".into(),
            patterns: vec![
                PatternItem {
                    src: NodePattern {
                        var: "A".into(),
                        label: "Person".into(),
                    },
                    edge: EdgePattern {
                        var: "ab".into(),
                        ty: "REL".into(),
                    },
                    dst: NodePattern {
                        var: "B".into(),
                        label: "Person".into(),
                    },
                },
                PatternItem {
                    src: NodePattern {
                        var: "B".into(),
                        label: "Person".into(),
                    },
                    edge: EdgePattern {
                        var: "bc".into(),
                        ty: "REL".into(),
                    },
                    dst: NodePattern {
                        var: "C".into(),
                        label: "Person".into(),
                    },
                },
            ],
            where_expr: None,
            actions: vec![ActionStmt::ForceAbsent {
                edge_var: "bc".into(),
            }],
            mode: Some("for_each".into()),
        };

        let result = run_rule_for_each_with_globals(&g, &rule, &HashMap::new()).unwrap();

        // Should have matched the chain (e1 -> e2) and forced e2 absent
        assert!(result.prob_mean(e2).unwrap() < 1e-5);
        // e1 should still be present
        assert!(result.prob_mean(e1).unwrap() > 0.5);
    }

    #[test]
    fn run_rule_for_each_multi_pattern_with_where_clause() {
        // Test multi-pattern with where clause filtering
        let mut g = BeliefGraph::default();

        let n1 = g.add_node("Person".into(), HashMap::new());
        let n2 = g.add_node("Person".into(), HashMap::new());
        let n3 = g.add_node("Person".into(), HashMap::new());

        // High probability chain
        let _e1 = g.add_edge(
            n1,
            n2,
            "REL".to_string(),
            BetaPosterior {
                alpha: 9.0,
                beta: 1.0,
            },
        );
        let e2 = g.add_edge(
            n2,
            n3,
            "REL".to_string(),
            BetaPosterior {
                alpha: 8.0,
                beta: 2.0,
            },
        );

        // Low probability chain (should not match)
        let n4 = g.add_node("Person".into(), HashMap::new());
        let _e3 = g.add_edge(
            n1,
            n4,
            "REL".to_string(),
            BetaPosterior {
                alpha: 1.0,
                beta: 9.0,
            },
        );
        let e4 = g.add_edge(
            n4,
            n3,
            "REL".to_string(),
            BetaPosterior {
                alpha: 2.0,
                beta: 8.0,
            },
        );

        // Apply delta to ensure all items are in base for iteration
        g.ensure_owned();

        let rule = RuleDef {
            name: "HighProbChain".into(),
            on_model: "M".into(),
            patterns: vec![
                PatternItem {
                    src: NodePattern {
                        var: "A".into(),
                        label: "Person".into(),
                    },
                    edge: EdgePattern {
                        var: "ab".into(),
                        ty: "REL".into(),
                    },
                    dst: NodePattern {
                        var: "B".into(),
                        label: "Person".into(),
                    },
                },
                PatternItem {
                    src: NodePattern {
                        var: "B".into(),
                        label: "Person".into(),
                    },
                    edge: EdgePattern {
                        var: "bc".into(),
                        ty: "REL".into(),
                    },
                    dst: NodePattern {
                        var: "C".into(),
                        label: "Person".into(),
                    },
                },
            ],
            where_expr: Some(ExprAst::Binary {
                op: BinaryOp::And,
                left: Box::new(ExprAst::Binary {
                    op: BinaryOp::Ge,
                    left: Box::new(ExprAst::Call {
                        name: "prob".into(),
                        args: vec![CallArg::Positional(ExprAst::Var("ab".into()))],
                    }),
                    right: Box::new(ExprAst::Number(0.8)),
                }),
                right: Box::new(ExprAst::Binary {
                    op: BinaryOp::Ge,
                    left: Box::new(ExprAst::Call {
                        name: "prob".into(),
                        args: vec![CallArg::Positional(ExprAst::Var("bc".into()))],
                    }),
                    right: Box::new(ExprAst::Number(0.7)),
                }),
            }),
            actions: vec![ActionStmt::ForceAbsent {
                edge_var: "bc".into(),
            }],
            mode: Some("for_each".into()),
        };

        let result = run_rule_for_each_with_globals(&g, &rule, &HashMap::new()).unwrap();

        // High probability chain should match and e2 should be forced absent
        assert!(result.prob_mean(e2).unwrap() < 1e-5);
        // Low probability chain should not match, e4 should remain
        assert!(result.prob_mean(e4).unwrap() > 0.1);
    }

    #[test]
    fn run_rule_for_each_multi_pattern_no_matches() {
        // Test multi-pattern that finds no matches
        let mut g = BeliefGraph::default();

        let n1 = g.add_node("Person".into(), HashMap::new());
        let n2 = g.add_node("Person".into(), HashMap::new());
        let n3 = g.add_node("Person".into(), HashMap::new());

        // Create edges but no chain
        let _e1 = g.add_edge(
            n1,
            n2,
            "REL".into(),
            BetaPosterior {
                alpha: 9.0,
                beta: 1.0,
            },
        );
        let _e2 = g.add_edge(
            n3,
            n2,
            "REL".into(),
            BetaPosterior {
                alpha: 8.0,
                beta: 2.0,
            },
        ); // Wrong direction

        // Apply delta to ensure all items are in base for iteration
        g.ensure_owned();

        let rule = RuleDef {
            name: "NoMatch".into(),
            on_model: "M".into(),
            patterns: vec![
                PatternItem {
                    src: NodePattern {
                        var: "A".into(),
                        label: "Person".into(),
                    },
                    edge: EdgePattern {
                        var: "ab".into(),
                        ty: "REL".into(),
                    },
                    dst: NodePattern {
                        var: "B".into(),
                        label: "Person".into(),
                    },
                },
                PatternItem {
                    src: NodePattern {
                        var: "B".into(),
                        label: "Person".into(),
                    },
                    edge: EdgePattern {
                        var: "bc".into(),
                        ty: "REL".into(),
                    },
                    dst: NodePattern {
                        var: "C".into(),
                        label: "Person".into(),
                    },
                },
            ],
            where_expr: None,
            actions: vec![ActionStmt::ForceAbsent {
                edge_var: "bc".into(),
            }],
            mode: Some("for_each".into()),
        };

        let result = run_rule_for_each_with_globals(&g, &rule, &HashMap::new()).unwrap();

        // Graph should be unchanged (no matches)
        assert_eq!(result.edges().len(), g.edges().len());
    }

    #[test]
    fn run_rule_for_each_single_pattern() {
        let g = create_test_graph();
        let rule = RuleDef {
            name: "TestRule".into(),
            on_model: "M".into(),
            patterns: vec![PatternItem {
                src: NodePattern {
                    var: "A".into(),
                    label: "Person".into(),
                },
                edge: EdgePattern {
                    var: "e".into(),
                    ty: "KNOWS".into(),
                },
                dst: NodePattern {
                    var: "B".into(),
                    label: "Person".into(),
                },
            }],
            where_expr: Some(ExprAst::Call {
                name: "prob".into(),
                args: vec![CallArg::Positional(ExprAst::Var("e".into()))],
            }),
            actions: vec![ActionStmt::ForceAbsent {
                edge_var: "e".into(),
            }],
            mode: Some("for_each".into()),
        };

        let result = run_rule_for_each(&g, &rule).unwrap();
        let prob = result.prob_mean(EdgeId(1)).unwrap();
        assert!(prob < 0.01, "edge should be forced absent");
    }
}
