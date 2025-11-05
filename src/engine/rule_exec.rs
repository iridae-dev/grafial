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

use std::collections::HashMap;

use crate::engine::errors::ExecError;
use crate::engine::graph::{BeliefGraph, EdgeId, NodeId};
use crate::frontend::ast::{ActionStmt, BinaryOp, CallArg, ExprAst, PatternItem, RuleDef, UnaryOp};

/// Epsilon for floating-point equality comparisons
const FLOAT_EPSILON: f64 = 1e-12;

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

#[derive(Debug, Default, Clone)]
struct Locals(HashMap<String, f64>);

impl Locals {
    fn get(&self, name: &str) -> Option<f64> { self.0.get(name).copied() }
    fn set(&mut self, name: String, v: f64) { self.0.insert(name, v); }
}

/// Executes a sequence of action statements on a graph.
///
/// Actions are executed in order, with `let` statements creating local variables
/// that can be used in subsequent actions. All actions operate on the provided
/// graph using the pattern match bindings.
///
/// # Arguments
///
/// * `graph` - The graph to modify (typically a working copy)
/// * `actions` - The actions to execute in order
/// * `bindings` - Variable bindings from pattern matching
///
/// # Returns
///
/// * `Ok(())` - All actions executed successfully
/// * `Err(ExecError)` - An action failed (invalid variable, etc.)
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
    let mut locals = Locals::default();
    for a in actions {
        match a {
            ActionStmt::Let { name, expr } => {
                let v = eval_expr(expr, graph, bindings, &locals, globals)?;
                locals.set(name.clone(), v);
            }
            ActionStmt::SetExpectation { node_var, attr, expr } => {
                let nid = *bindings
                    .node_vars
                    .get(node_var)
                    .ok_or_else(|| ExecError::Internal(format!("unknown node var '{}'", node_var)))?;
                let v = eval_expr(expr, graph, bindings, &locals, globals)?;
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

fn eval_expr(
    expr: &ExprAst,
    graph: &BeliefGraph,
    bindings: &MatchBindings,
    locals: &Locals,
    globals: &HashMap<String, f64>,
) -> Result<f64, ExecError> {
    match expr {
        ExprAst::Number(value) => Ok(*value),
        ExprAst::Bool(b) => Ok(if *b { 1.0 } else { 0.0 }),
        ExprAst::Var(name) => locals
            .get(name)
            .or_else(|| globals.get(name).copied())
            .ok_or_else(|| ExecError::Internal(format!("unbound variable '{}'", name))),
        ExprAst::Field { .. } => Err(ExecError::Internal("bare field access not supported; use E[Node.attr]".into())),
        ExprAst::Call { name, args } => match name.as_str() {
            // E[Var.attr] is represented as Call("E", [Field(Var(..), attr)]) by the parser
            "E" => {
                let (pos, named) = split_args(args);
                if !named.is_empty() || pos.len() != 1 {
                    return Err(ExecError::Internal("E[] expects one positional argument".into()));
                }
                match pos[0] {
                    ExprAst::Field { target, field } => match &**target {
                        ExprAst::Var(vn) => {
                            let nid = *bindings
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
                let (pos, named) = split_args(args);
                if !named.is_empty() || pos.len() != 1 {
                    return Err(ExecError::Internal("prob() expects one positional argument".into()));
                }
                match pos[0] {
                    ExprAst::Var(v) => {
                        let eid = *bindings
                            .edge_vars
                            .get(v)
                            .ok_or_else(|| ExecError::Internal(format!("unknown edge var '{}'", v)))?;
                        graph.prob_mean(eid)
                    }
                    _ => Err(ExecError::Internal("prob(): argument must be an edge variable".into())),
                }
            }
            "degree" => {
                let (pos, named) = split_args(args);
                if pos.len() < 1 {
                    return Err(ExecError::Internal("degree(): missing node argument".into()));
                }
                let mut min_prob = 0.0;
                for (k, v) in named {
                    if k == "min_prob" { min_prob = eval_expr(v, graph, bindings, locals, globals)?; }
                }
                match pos[0] {
                    ExprAst::Var(v) => {
                        let nid = *bindings
                            .node_vars
                            .get(v)
                            .ok_or_else(|| ExecError::Internal(format!("unknown node var '{}'", v)))?;
                        Ok(graph.degree_outgoing(nid, min_prob) as f64)
                    }
                    _ => Err(ExecError::Internal("degree(): first argument must be a node variable".into())),
                }
            }
            other => Err(ExecError::Internal(format!("unknown function '{}'", other))),
        },
        ExprAst::Unary { op, expr } => {
            let v = eval_expr(expr, graph, bindings, locals, globals)?;
            Ok(match op { UnaryOp::Neg => -v, UnaryOp::Not => if v == 0.0 { 1.0 } else { 0.0 } })
        }
        ExprAst::Binary { op, left, right } => {
            let l = eval_expr(left, graph, bindings, locals, globals)?;
            let r = eval_expr(right, graph, bindings, locals, globals)?;
            Ok(match op {
                BinaryOp::Add => l + r,
                BinaryOp::Sub => l - r,
                BinaryOp::Mul => l * r,
                BinaryOp::Div => l / r,
                BinaryOp::Eq => if (l - r).abs() < FLOAT_EPSILON { 1.0 } else { 0.0 },
                BinaryOp::Ne => if (l - r).abs() >= FLOAT_EPSILON { 1.0 } else { 0.0 },
                BinaryOp::Lt => if l < r { 1.0 } else { 0.0 },
                BinaryOp::Le => if l <= r { 1.0 } else { 0.0 },
                BinaryOp::Gt => if l > r { 1.0 } else { 0.0 },
                BinaryOp::Ge => if l >= r { 1.0 } else { 0.0 },
                BinaryOp::And => if (l != 0.0) && (r != 0.0) { 1.0 } else { 0.0 },
                BinaryOp::Or => if (l != 0.0) || (r != 0.0) { 1.0 } else { 0.0 },
            })
        }
    }
}

fn split_args<'a>(args: &'a [CallArg]) -> (Vec<&'a ExprAst>, Vec<(&'a str, &'a ExprAst)>) {
    let mut pos = Vec::new();
    let mut named = Vec::new();
    for a in args {
        match a {
            CallArg::Positional(e) => pos.push(e),
            CallArg::Named { name, value } => named.push((name.as_str(), value)),
        }
    }
    (pos, named)
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
            let v = eval_expr(w, input, &bindings, &Locals::default(), globals)?;
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
    // eval_expr Tests
    // ============================================================================

    #[test]
    fn eval_expr_number() {
        let g = BeliefGraph::default();
        let bindings = MatchBindings::default();
        let locals = Locals::default();

        let result = eval_expr(&ExprAst::Number(42.5), &g, &bindings, &locals, &HashMap::new()).unwrap();
        assert!((result - 42.5).abs() < 1e-9);
    }

    #[test]
    fn eval_expr_bool_true() {
        let g = BeliefGraph::default();
        let bindings = MatchBindings::default();
        let locals = Locals::default();

        let result = eval_expr(&ExprAst::Bool(true), &g, &bindings, &locals, &HashMap::new()).unwrap();
        assert_eq!(result, 1.0);
    }

    #[test]
    fn eval_expr_bool_false() {
        let g = BeliefGraph::default();
        let bindings = MatchBindings::default();
        let locals = Locals::default();

        let result = eval_expr(&ExprAst::Bool(false), &g, &bindings, &locals, &HashMap::new()).unwrap();
        assert_eq!(result, 0.0);
    }

    #[test]
    fn eval_expr_var_from_locals() {
        let g = BeliefGraph::default();
        let bindings = MatchBindings::default();
        let mut locals = Locals::default();
        locals.set("x".into(), 100.0);

        let result = eval_expr(&ExprAst::Var("x".into()), &g, &bindings, &locals, &HashMap::new()).unwrap();
        assert_eq!(result, 100.0);
    }

    #[test]
    fn eval_expr_var_unbound_fails() {
        let g = BeliefGraph::default();
        let bindings = MatchBindings::default();
        let locals = Locals::default();

        let result = eval_expr(&ExprAst::Var("undefined".into()), &g, &bindings, &locals, &HashMap::new());
        assert!(result.is_err());
    }

    #[test]
    fn eval_expr_binary_add() {
        let g = BeliefGraph::default();
        let bindings = MatchBindings::default();
        let locals = Locals::default();

        let expr = ExprAst::Binary {
            op: BinaryOp::Add,
            left: Box::new(ExprAst::Number(10.0)),
            right: Box::new(ExprAst::Number(5.0)),
        };

        let result = eval_expr(&expr, &g, &bindings, &locals, &HashMap::new()).unwrap();
        assert_eq!(result, 15.0);
    }

    #[test]
    fn eval_expr_binary_sub() {
        let g = BeliefGraph::default();
        let bindings = MatchBindings::default();
        let locals = Locals::default();

        let expr = ExprAst::Binary {
            op: BinaryOp::Sub,
            left: Box::new(ExprAst::Number(10.0)),
            right: Box::new(ExprAst::Number(3.0)),
        };

        let result = eval_expr(&expr, &g, &bindings, &locals, &HashMap::new()).unwrap();
        assert_eq!(result, 7.0);
    }

    #[test]
    fn eval_expr_binary_mul() {
        let g = BeliefGraph::default();
        let bindings = MatchBindings::default();
        let locals = Locals::default();

        let expr = ExprAst::Binary {
            op: BinaryOp::Mul,
            left: Box::new(ExprAst::Number(4.0)),
            right: Box::new(ExprAst::Number(5.0)),
        };

        let result = eval_expr(&expr, &g, &bindings, &locals, &HashMap::new()).unwrap();
        assert_eq!(result, 20.0);
    }

    #[test]
    fn eval_expr_binary_div() {
        let g = BeliefGraph::default();
        let bindings = MatchBindings::default();
        let locals = Locals::default();

        let expr = ExprAst::Binary {
            op: BinaryOp::Div,
            left: Box::new(ExprAst::Number(10.0)),
            right: Box::new(ExprAst::Number(2.0)),
        };

        let result = eval_expr(&expr, &g, &bindings, &locals, &HashMap::new()).unwrap();
        assert_eq!(result, 5.0);
    }

    #[test]
    fn eval_expr_binary_comparisons() {
        let g = BeliefGraph::default();
        let bindings = MatchBindings::default();
        let locals = Locals::default();

        let test_cases = vec![
            (BinaryOp::Lt, 5.0, 10.0, 1.0),
            (BinaryOp::Lt, 10.0, 5.0, 0.0),
            (BinaryOp::Le, 5.0, 5.0, 1.0),
            (BinaryOp::Gt, 10.0, 5.0, 1.0),
            (BinaryOp::Gt, 5.0, 10.0, 0.0),
            (BinaryOp::Ge, 5.0, 5.0, 1.0),
            (BinaryOp::Eq, 5.0, 5.0, 1.0),
            (BinaryOp::Eq, 5.0, 6.0, 0.0),
            (BinaryOp::Ne, 5.0, 6.0, 1.0),
        ];

        for (op, left, right, expected) in test_cases {
            let expr = ExprAst::Binary {
                op,
                left: Box::new(ExprAst::Number(left)),
                right: Box::new(ExprAst::Number(right)),
            };
            let result = eval_expr(&expr, &g, &bindings, &locals, &HashMap::new()).unwrap();
            assert_eq!(result, expected, "Failed for {:?} {} {}", op, left, right);
        }
    }

    #[test]
    fn eval_expr_binary_logical_and() {
        let g = BeliefGraph::default();
        let bindings = MatchBindings::default();
        let locals = Locals::default();

        let expr = ExprAst::Binary {
            op: BinaryOp::And,
            left: Box::new(ExprAst::Bool(true)),
            right: Box::new(ExprAst::Bool(true)),
        };
        assert_eq!(eval_expr(&expr, &g, &bindings, &locals, &HashMap::new()).unwrap(), 1.0);

        let expr = ExprAst::Binary {
            op: BinaryOp::And,
            left: Box::new(ExprAst::Bool(true)),
            right: Box::new(ExprAst::Bool(false)),
        };
        assert_eq!(eval_expr(&expr, &g, &bindings, &locals, &HashMap::new()).unwrap(), 0.0);
    }

    #[test]
    fn eval_expr_binary_logical_or() {
        let g = BeliefGraph::default();
        let bindings = MatchBindings::default();
        let locals = Locals::default();

        let expr = ExprAst::Binary {
            op: BinaryOp::Or,
            left: Box::new(ExprAst::Bool(false)),
            right: Box::new(ExprAst::Bool(true)),
        };
        assert_eq!(eval_expr(&expr, &g, &bindings, &locals, &HashMap::new()).unwrap(), 1.0);

        let expr = ExprAst::Binary {
            op: BinaryOp::Or,
            left: Box::new(ExprAst::Bool(false)),
            right: Box::new(ExprAst::Bool(false)),
        };
        assert_eq!(eval_expr(&expr, &g, &bindings, &locals, &HashMap::new()).unwrap(), 0.0);
    }

    #[test]
    fn eval_expr_unary_neg() {
        let g = BeliefGraph::default();
        let bindings = MatchBindings::default();
        let locals = Locals::default();

        let expr = ExprAst::Unary {
            op: UnaryOp::Neg,
            expr: Box::new(ExprAst::Number(42.0)),
        };

        let result = eval_expr(&expr, &g, &bindings, &locals, &HashMap::new()).unwrap();
        assert_eq!(result, -42.0);
    }

    #[test]
    fn eval_expr_unary_not() {
        let g = BeliefGraph::default();
        let bindings = MatchBindings::default();
        let locals = Locals::default();

        let expr = ExprAst::Unary {
            op: UnaryOp::Not,
            expr: Box::new(ExprAst::Bool(true)),
        };
        assert_eq!(eval_expr(&expr, &g, &bindings, &locals, &HashMap::new()).unwrap(), 0.0);

        let expr = ExprAst::Unary {
            op: UnaryOp::Not,
            expr: Box::new(ExprAst::Bool(false)),
        };
        assert_eq!(eval_expr(&expr, &g, &bindings, &locals, &HashMap::new()).unwrap(), 1.0);
    }

    #[test]
    fn eval_expr_call_e_retrieves_expectation() {
        let g = create_test_graph();
        let mut bindings = MatchBindings::default();
        bindings.node_vars.insert("A".into(), NodeId(1));

        let expr = ExprAst::Call {
            name: "E".into(),
            args: vec![CallArg::Positional(ExprAst::Field {
                target: Box::new(ExprAst::Var("A".into())),
                field: "x".into(),
            })],
        };

        let result = eval_expr(&expr, &g, &bindings, &Locals::default(), &HashMap::new()).unwrap();
        assert_eq!(result, 10.0);
    }

    #[test]
    fn eval_expr_call_prob_retrieves_edge_probability() {
        let g = create_test_graph();
        let mut bindings = MatchBindings::default();
        bindings.edge_vars.insert("e".into(), EdgeId(1));

        let expr = ExprAst::Call {
            name: "prob".into(),
            args: vec![CallArg::Positional(ExprAst::Var("e".into()))],
        };

        let result = eval_expr(&expr, &g, &bindings, &Locals::default(), &HashMap::new()).unwrap();
        assert!((result - 0.5).abs() < 1e-9); // Beta(5,5) mean is 0.5
    }

    #[test]
    fn eval_expr_call_degree_counts_edges() {
        let g = create_test_graph();
        let mut bindings = MatchBindings::default();
        bindings.node_vars.insert("A".into(), NodeId(1));

        let expr = ExprAst::Call {
            name: "degree".into(),
            args: vec![
                CallArg::Positional(ExprAst::Var("A".into())),
                CallArg::Named {
                    name: "min_prob".into(),
                    value: ExprAst::Number(0.3),
                },
            ],
        };

        let result = eval_expr(&expr, &g, &bindings, &Locals::default(), &HashMap::new()).unwrap();
        assert_eq!(result, 1.0);
    }

    // ============================================================================
    // execute_actions Tests
    // ============================================================================

    #[test]
    fn execute_actions_let_stores_variable() {
        let mut g = create_test_graph();
        let bindings = MatchBindings::default();

        let actions = vec![ActionStmt::Let {
            name: "temp".into(),
            expr: ExprAst::Number(99.0),
        }];

        execute_actions(&mut g, &actions, &bindings, &HashMap::new()).unwrap();
        // No direct way to verify locals, but it shouldn't error
    }

    #[test]
    fn execute_actions_set_expectation_updates_node() {
        let mut g = create_test_graph();
        let mut bindings = MatchBindings::default();
        bindings.node_vars.insert("A".into(), NodeId(1));

        let actions = vec![ActionStmt::SetExpectation {
            node_var: "A".into(),
            attr: "x".into(),
            expr: ExprAst::Number(20.0),
        }];

        execute_actions(&mut g, &actions, &bindings, &HashMap::new()).unwrap();

        let mean = g.expectation(NodeId(1), "x").unwrap();
        assert_eq!(mean, 20.0);
    }

    #[test]
    fn execute_actions_force_absent_updates_edge() {
        let mut g = create_test_graph();
        let mut bindings = MatchBindings::default();
        bindings.edge_vars.insert("e".into(), EdgeId(1));

        let actions = vec![ActionStmt::ForceAbsent {
            edge_var: "e".into(),
        }];

        execute_actions(&mut g, &actions, &bindings, &HashMap::new()).unwrap();

        let prob = g.prob_mean(EdgeId(1)).unwrap();
        assert!(prob < 1e-5);
    }

    #[test]
    fn execute_actions_let_then_use() {
        let mut g = create_test_graph();
        let mut bindings = MatchBindings::default();
        bindings.node_vars.insert("A".into(), NodeId(1));

        let actions = vec![
            ActionStmt::Let {
                name: "temp".into(),
                expr: ExprAst::Number(15.0),
            },
            ActionStmt::SetExpectation {
                node_var: "A".into(),
                attr: "x".into(),
                expr: ExprAst::Var("temp".into()),
            },
        ];

        execute_actions(&mut g, &actions, &bindings, &HashMap::new()).unwrap();

        let mean = g.expectation(NodeId(1), "x").unwrap();
        assert_eq!(mean, 15.0);
    }

    // ============================================================================
    // run_rule_for_each Tests
    // ============================================================================

    #[test]
    fn run_rule_for_each_applies_action_to_matching_edge() {
        let g = create_test_graph();

        let rule = RuleDef {
            name: "TestRule".into(),
            on_model: "M".into(),
            mode: Some("for_each".into()),
            patterns: vec![PatternItem {
                src: NodePattern { var: "A".into(), label: "Person".into() },
                edge: EdgePattern { var: "e".into(), ty: "KNOWS".into() },
                dst: NodePattern { var: "B".into(), label: "Person".into() },
            }],
            where_expr: None,
            actions: vec![ActionStmt::ForceAbsent { edge_var: "e".into() }],
        };

        let result = run_rule_for_each(&g, &rule).unwrap();

        let prob = result.prob_mean(EdgeId(1)).unwrap();
        assert!(prob < 1e-5);
    }

    #[test]
    fn run_rule_for_each_with_where_clause_filters() {
        let g = create_test_graph();

        let rule = RuleDef {
            name: "TestRule".into(),
            on_model: "M".into(),
            mode: Some("for_each".into()),
            patterns: vec![PatternItem {
                src: NodePattern { var: "A".into(), label: "Person".into() },
                edge: EdgePattern { var: "e".into(), ty: "KNOWS".into() },
                dst: NodePattern { var: "B".into(), label: "Person".into() },
            }],
            where_expr: Some(ExprAst::Binary {
                op: BinaryOp::Gt,
                left: Box::new(ExprAst::Call {
                    name: "prob".into(),
                    args: vec![CallArg::Positional(ExprAst::Var("e".into()))],
                }),
                right: Box::new(ExprAst::Number(0.9)),
            }),
            actions: vec![ActionStmt::ForceAbsent { edge_var: "e".into() }],
        };

        let result = run_rule_for_each(&g, &rule).unwrap();

        // prob(e) = 0.5, which is not > 0.9, so action should not apply
        let prob = result.prob_mean(EdgeId(1)).unwrap();
        assert!((prob - 0.5).abs() < 1e-9);
    }

    #[test]
    fn run_rule_for_each_rejects_multiple_patterns() {
        let g = create_test_graph();

        let rule = RuleDef {
            name: "TestRule".into(),
            on_model: "M".into(),
            mode: Some("for_each".into()),
            patterns: vec![
                PatternItem {
                    src: NodePattern { var: "A".into(), label: "Person".into() },
                    edge: EdgePattern { var: "e1".into(), ty: "KNOWS".into() },
                    dst: NodePattern { var: "B".into(), label: "Person".into() },
                },
                PatternItem {
                    src: NodePattern { var: "B".into(), label: "Person".into() },
                    edge: EdgePattern { var: "e2".into(), ty: "KNOWS".into() },
                    dst: NodePattern { var: "C".into(), label: "Person".into() },
                },
            ],
            where_expr: None,
            actions: vec![],
        };

        let result = run_rule_for_each(&g, &rule);
        assert!(result.is_err());
    }
}
