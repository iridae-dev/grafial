//! # Rule Execution Engine
//!
//! This module implements the core rule execution logic for Baygraph, including:
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
//! See baygraph_design.md:524-527 for the execution model and baygraph_design.md:527
//! for immutability requirements (working copies).
//!
//! ## Expression Language
//!
//! Supports:
//! - Arithmetic: `+`, `-`, `*`, `/`
//! - Comparisons: `<`, `<=`, `>`, `>=`, `==`, `!=`
//! - Logical: `and`, `or`, `not`
//! - Special functions: `prob(edge)`, `degree(node)`, `E[node.attr]`
//!
//! ## Performance
//!
//! Uses O(1) hash lookups for nodes/edges and eager evaluation with
//! local variable binding for efficient execution.
//!
//! ## Determinism
//!
//! Pattern matching iterates edges in stable order (sorted by EdgeId) to ensure
//! deterministic rule execution. See baygraph_design.md:517-518.

use std::collections::HashMap;

use crate::engine::errors::ExecError;
use crate::engine::expr_eval::{eval_expr_core, ExprContext};
use crate::engine::graph::{BeliefGraph, EdgeId, NodeId};
use crate::frontend::ast::{ActionStmt, CallArg, ExprAst, PatternItem, RuleDef};

/// Variable bindings for a single pattern match.
///
/// Maps pattern variables (from rule patterns) to concrete graph elements.
/// For example, pattern `(A:Person)-[e:KNOWS]->(B:Person)` creates bindings
/// for variables A, B (nodes) and e (edge).
#[derive(Debug, Default, Clone)]
pub struct MatchBindings {
    /// Maps node variable names to node IDs
    pub node_vars: HashMap<String, NodeId>,
    /// Maps edge variable names to edge IDs
    pub edge_vars: HashMap<String, EdgeId>,
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
        self.locals.get(name).or_else(|| self.globals.get(name).copied())
    }

    fn eval_function(
        &self,
        name: &str,
        pos_args: &[ExprAst],
        all_args: &[CallArg],
        graph: &BeliefGraph,
    ) -> Result<f64, ExecError> {
        match name {
            // E[Var.attr] is represented as Call("E", [Field(Var(..), attr)]) by the parser
            "E" => {
                if !all_args.is_empty() && matches!(all_args[0], CallArg::Named { .. }) {
                    return Err(ExecError::Internal("E[] does not accept named arguments".into()));
                }
                if pos_args.len() != 1 {
                    return Err(ExecError::Internal("E[] expects one positional argument".into()));
                }
                match &pos_args[0] {
                    ExprAst::Field { target, field } => match &**target {
                        ExprAst::Var(vn) => {
                            let nid = *self
                                .bindings
                                .node_vars
                                .get(vn)
                                .ok_or_else(|| ExecError::Internal(format!("unknown node var '{}'", vn)))?;
                            graph.expectation(nid, field)
                        }
                        _ => Err(ExecError::Internal("E[] requires NodeVar.attr".into())),
                    },
                    _ => Err(ExecError::Internal("E[] requires a field expression".into())),
                }
            }
            "prob" => {
                if !all_args.is_empty() && matches!(all_args[0], CallArg::Named { .. }) {
                    return Err(ExecError::Internal("prob() does not accept named arguments".into()));
                }
                if pos_args.len() != 1 {
                    return Err(ExecError::Internal("prob() expects one positional argument".into()));
                }
                match &pos_args[0] {
                    ExprAst::Var(v) => {
                        let eid = *self
                            .bindings
                            .edge_vars
                            .get(v)
                            .ok_or_else(|| ExecError::Internal(format!("unknown edge var '{}'", v)))?;
                        graph.prob_mean(eid)
                    }
                    _ => Err(ExecError::Internal("prob(): argument must be an edge variable".into())),
                }
            }
            "degree" => {
                if pos_args.len() < 1 {
                    return Err(ExecError::Internal("degree(): missing node argument".into()));
                }
                let mut min_prob = 0.0;
                for a in all_args {
                    if let CallArg::Named { name, value } = a {
                        if name == "min_prob" {
                            min_prob = eval_expr_core(value, graph, self)?;
                        }
                    }
                }
                match &pos_args[0] {
                    ExprAst::Var(v) => {
                        let nid = *self
                            .bindings
                            .node_vars
                            .get(v)
                            .ok_or_else(|| ExecError::Internal(format!("unknown node var '{}'", v)))?;
                        Ok(graph.degree_outgoing(nid, min_prob) as f64)
                    }
                    _ => Err(ExecError::Internal("degree(): first argument must be a node variable".into())),
                }
            }
            other => Err(ExecError::Internal(format!("unknown function '{}'", other))),
        }
    }

    fn eval_field(&self, target: &ExprAst, _field: &str, _graph: &BeliefGraph) -> Result<f64, ExecError> {
        Err(ExecError::Internal(format!(
            "bare field access not supported; use E[Node.attr]: {:?}",
            target
        )))
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
/// - `ForceAbsent`: Forces an edge to be absent with high certainty
///
/// See baygraph_design.md:275-276 (set_expectation) and 133-135 (force_absent).
pub fn execute_actions(
    graph: &mut BeliefGraph,
    actions: &[ActionStmt],
    bindings: &MatchBindings,
    globals: &HashMap<String, f64>,
) -> Result<(), ExecError> {
    let mut locals = Locals::new();

    for a in actions {
        // Create context for this action (reborrow locals)
        let ctx = RuleExprContext {
            bindings,
            locals: &locals,
            globals,
        };

        match a {
            ActionStmt::Let { name, expr } => {
                let v = eval_expr_core(expr, graph, &ctx)?;
                locals.set(name.clone(), v);
            }
            ActionStmt::SetExpectation { node_var, attr, expr } => {
                let nid = *bindings
                    .node_vars
                    .get(node_var)
                    .ok_or_else(|| ExecError::Internal(format!("unknown node var '{}'", node_var)))?;
                let v = eval_expr_core(expr, graph, &ctx)?;
                graph.set_expectation(nid, attr, v)?;
            }
            ActionStmt::ForceAbsent { edge_var } => {
                let eid = *bindings
                    .edge_vars
                    .get(edge_var)
                    .ok_or_else(|| ExecError::Internal(format!("unknown edge var '{}'", edge_var)))?;
                graph.force_absent(eid)?;
            }
        }
    }
    Ok(())
}

/// Executes a rule in "for_each" mode over all matching patterns.
///
/// This is the core rule execution function for Phase 3. It:
/// 1. Finds all graph patterns matching the rule's pattern
/// 2. Evaluates the optional `where` clause for each match
/// 3. Applies actions to a working copy for each successful match
/// 4. Returns the modified graph
///
/// # Current Limitations (Phase 3)
///
/// - Supports exactly one pattern item (no multi-pattern rules yet)
/// - Only "for_each" execution mode
///
/// # Arguments
///
/// * `input` - The input belief graph (read-only)
/// * `rule` - The rule definition to execute
///
/// # Returns
///
/// * `Ok(BeliefGraph)` - The transformed graph after applying all matches
/// * `Err(ExecError)` - Validation error or execution failure
///
/// # Example
///
/// ```rust,ignore
/// // Rule: for each KNOWS edge with prob >= 0.5, force it absent
/// let output = run_rule_for_each(&graph, &rule)?;
/// ```
///
/// See baygraph_design.md:524-527 for the execution model.
pub fn run_rule_for_each(input: &BeliefGraph, rule: &RuleDef) -> Result<BeliefGraph, ExecError> {
    run_rule_for_each_with_globals(input, rule, &HashMap::new())
}

/// Executes a rule with access to global scalar variables (e.g., imported metrics).
pub fn run_rule_for_each_with_globals(
    input: &BeliefGraph,
    rule: &RuleDef,
    globals: &HashMap<String, f64>,
) -> Result<BeliefGraph, ExecError> {
    if rule.patterns.len() != 1 {
        return Err(ExecError::ValidationError(
            "only single pattern item supported in Phase 3".into(),
        ));
    }
    let pat: &PatternItem = &rule.patterns[0];
    let mut work = input.clone();

    // Iterate edges in stable order (sorted by EdgeId) for determinism
    let mut idxs: Vec<usize> = (0..input.edges.len()).collect();
    idxs.sort_by_key(|&i| input.edges[i].id);

    for i in idxs {
        let e = &input.edges[i];
        if e.ty != pat.edge.ty { continue; }
        let src = input
            .node(e.src)
            .ok_or_else(|| ExecError::Internal("missing src".into()))?;
        let dst = input
            .node(e.dst)
            .ok_or_else(|| ExecError::Internal("missing dst".into()))?;
        if src.label != pat.src.label || dst.label != pat.dst.label { continue; }

        let mut bindings = MatchBindings::default();
        bindings.node_vars.insert(pat.src.var.clone(), e.src);
        bindings.node_vars.insert(pat.dst.var.clone(), e.dst);
        bindings.edge_vars.insert(pat.edge.var.clone(), e.id);

        // where clause
        if let Some(w) = &rule.where_expr {
            let where_locals = Locals::new();
            let where_ctx = RuleExprContext {
                bindings: &bindings,
                locals: &where_locals,
                globals,
            };
            let v = eval_expr_core(w, input, &where_ctx)?;
            let is_true = v != 0.0;
            if !is_true { continue; }
        }

        // Apply actions to working copy
        execute_actions(&mut work, &rule.actions, &bindings, globals)?;
    }

    Ok(work)
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::engine::graph::{BeliefGraph, BetaPosterior, EdgeData, GaussianPosterior, NodeData};
    use crate::frontend::ast::{EdgePattern, NodePattern};

    // Helper to create a simple test graph
    fn create_test_graph() -> BeliefGraph {
        let mut g = BeliefGraph::default();
        g.insert_node(NodeData {
            id: NodeId(1),
            label: "Person".into(),
            attrs: HashMap::from([("x".into(), GaussianPosterior { mean: 10.0, precision: 1.0 })]),
        });
        g.insert_node(NodeData {
            id: NodeId(2),
            label: "Person".into(),
            attrs: HashMap::from([("x".into(), GaussianPosterior { mean: 5.0, precision: 1.0 })]),
        });
        g.insert_edge(EdgeData {
            id: EdgeId(1),
            src: NodeId(1),
            dst: NodeId(2),
            ty: "KNOWS".into(),
            exist: BetaPosterior { alpha: 5.0, beta: 5.0 },
        });
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

        use crate::frontend::ast::BinaryOp;
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

        use crate::frontend::ast::BinaryOp;
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

        use crate::frontend::ast::UnaryOp;
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

        use crate::frontend::ast::BinaryOp;
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
                    op: crate::frontend::ast::BinaryOp::Div,
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
            ActionStmt::ForceAbsent { edge_var: "e".into() },
        ];

        execute_actions(&mut g, &actions, &bindings, &HashMap::new()).unwrap();

        let prob = g.prob_mean(EdgeId(1)).unwrap();
        assert!(prob < 0.01, "edge should be forced absent");
    }

    #[test]
    fn run_rule_for_each_single_pattern() {
        let g = create_test_graph();
        let rule = RuleDef {
            name: "TestRule".into(),
            on_model: "M".into(),
            patterns: vec![PatternItem {
                src: NodePattern { var: "A".into(), label: "Person".into() },
                edge: EdgePattern { var: "e".into(), ty: "KNOWS".into() },
                dst: NodePattern { var: "B".into(), label: "Person".into() },
            }],
            where_expr: Some(ExprAst::Call {
                name: "prob".into(),
                args: vec![CallArg::Positional(ExprAst::Var("e".into()))],
            }),
            actions: vec![ActionStmt::ForceAbsent { edge_var: "e".into() }],
            mode: Some("for_each".into()),
        };

        let result = run_rule_for_each(&g, &rule).unwrap();
        let prob = result.prob_mean(EdgeId(1)).unwrap();
        assert!(prob < 0.01, "edge should be forced absent");
    }
}
