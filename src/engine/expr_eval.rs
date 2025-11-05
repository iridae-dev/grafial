//! Shared expression evaluation core.
//!
//! This module provides a unified expression evaluation system that eliminates
//! duplication across rule execution, flow transforms, and metrics.
//!
//! ## Design Principles
//!
//! - **DRY**: Binary/unary operations evaluated once, reused everywhere
//! - **SOLID**: Context-specific behavior via traits (variable resolution, function calls)
//! - **Extensibility**: New evaluation contexts can be added without modifying core logic
//!
//! ## Architecture
//!
//! The evaluation system uses a context trait pattern:
//! - `ExprContext` trait: Defines how variables and functions are resolved
//! - Core evaluator: Handles literals, operators, and delegates context-specific parts
//!
//! See baygraph_design.md:275-276 for expression semantics.

use crate::engine::errors::ExecError;
use crate::engine::graph::BeliefGraph;
use crate::frontend::ast::{BinaryOp, ExprAst, UnaryOp};

/// Epsilon for floating-point equality comparisons
const FLOAT_EPSILON: f64 = 1e-12;

/// Context for expression evaluation that provides variable resolution and function calls.
///
/// Different contexts (rules, metrics, prune predicates) implement this trait
/// to customize how variables and functions are resolved while sharing the core
/// evaluation logic.
pub trait ExprContext {
    /// Resolve a variable name to a numeric value.
    ///
    /// Returns `None` if the variable doesn't exist in this context.
    fn resolve_var(&self, name: &str) -> Option<f64>;

    /// Evaluate a function call in this context.
    ///
    /// Returns the function result or an error if the function is unknown or
    /// arguments are invalid.
    ///
    /// * `name` - Function name
    /// * `pos_args` - Positional arguments (as ExprAst)
    /// * `all_args` - All arguments including named (for contexts that need them)
    /// * `graph` - The belief graph for graph queries
    fn eval_function(
        &self,
        name: &str,
        pos_args: &[ExprAst],
        all_args: &[crate::frontend::ast::CallArg],
        graph: &BeliefGraph,
    ) -> Result<f64, ExecError>;

    /// Evaluate a field access in this context.
    ///
    /// This is used for expressions like `E[node.attr]` where the target
    /// needs context-specific resolution.
    fn eval_field(&self, target: &ExprAst, field: &str, graph: &BeliefGraph) -> Result<f64, ExecError>;
}

/// Evaluates a binary operation on two numeric values.
///
/// This is the single source of truth for binary operation semantics,
/// ensuring consistent behavior across all evaluation contexts.
pub fn eval_binary_op(op: BinaryOp, left: f64, right: f64) -> Result<f64, ExecError> {
    let result = match op {
        BinaryOp::Add => left + right,
        BinaryOp::Sub => left - right,
        BinaryOp::Mul => left * right,
        BinaryOp::Div => {
            if right.abs() < FLOAT_EPSILON {
                return Err(ExecError::ValidationError("division by zero".into()));
            }
            left / right
        }
        BinaryOp::Eq => if (left - right).abs() < FLOAT_EPSILON { 1.0 } else { 0.0 },
        BinaryOp::Ne => if (left - right).abs() >= FLOAT_EPSILON { 1.0 } else { 0.0 },
        BinaryOp::Lt => if left < right { 1.0 } else { 0.0 },
        BinaryOp::Le => if left <= right { 1.0 } else { 0.0 },
        BinaryOp::Gt => if left > right { 1.0 } else { 0.0 },
        BinaryOp::Ge => if left >= right { 1.0 } else { 0.0 },
        BinaryOp::And => if (left != 0.0) && (right != 0.0) { 1.0 } else { 0.0 },
        BinaryOp::Or => if (left != 0.0) || (right != 0.0) { 1.0 } else { 0.0 },
    };

    // Validate result is finite
    if !result.is_finite() {
        return Err(ExecError::ValidationError(
            format!("binary operation produced non-finite value: {} {:?} {}", left, op, right)
        ));
    }

    Ok(result)
}

/// Evaluates a unary operation on a numeric value.
///
/// This is the single source of truth for unary operation semantics.
pub fn eval_unary_op(op: UnaryOp, value: f64) -> f64 {
    match op {
        UnaryOp::Neg => -value,
        UnaryOp::Not => if value == 0.0 { 1.0 } else { 0.0 },
    }
}

/// Core expression evaluator that delegates context-specific behavior.
///
/// This function handles the common structure of expression evaluation:
/// - Literals (numbers, booleans)
/// - Binary and unary operations
/// - Delegates to context for variables, fields, and functions
pub fn eval_expr_core<C: ExprContext>(
    expr: &ExprAst,
    graph: &BeliefGraph,
    ctx: &C,
) -> Result<f64, ExecError> {
    match expr {
        ExprAst::Number(value) => Ok(*value),
        ExprAst::Bool(b) => Ok(if *b { 1.0 } else { 0.0 }),
        ExprAst::Var(name) => ctx
            .resolve_var(name)
            .ok_or_else(|| ExecError::Internal(format!("unbound variable '{}'", name))),
        ExprAst::Field { target, field } => ctx.eval_field(target, field, graph),
        ExprAst::Call { name, args } => {
            // Extract positional arguments only (named args handled by context)
            let pos_args: Vec<ExprAst> = args.iter()
                .filter_map(|a| match a {
                    crate::frontend::ast::CallArg::Positional(e) => Some(e.clone()),
                    crate::frontend::ast::CallArg::Named { .. } => None,
                })
                .collect();
            // Pass both positional args and full args for context to handle named args
            ctx.eval_function(name, &pos_args, args, graph)
        }
        ExprAst::Unary { op, expr } => {
            let v = eval_expr_core(expr, graph, ctx)?;
            Ok(eval_unary_op(*op, v))
        }
        ExprAst::Binary { op, left, right } => {
            let l = eval_expr_core(left, graph, ctx)?;
            let r = eval_expr_core(right, graph, ctx)?;
            eval_binary_op(*op, l, r)
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    struct TestContext;

    impl ExprContext for TestContext {
        fn resolve_var(&self, name: &str) -> Option<f64> {
            match name {
                "x" => Some(10.0),
                "y" => Some(5.0),
                _ => None,
            }
        }

        fn eval_function(
            &self,
            _name: &str,
            _pos_args: &[ExprAst],
            _all_args: &[crate::frontend::ast::CallArg],
            _graph: &BeliefGraph,
        ) -> Result<f64, ExecError> {
            Err(ExecError::Internal("no functions in test context".into()))
        }

        fn eval_field(&self, _target: &ExprAst, _field: &str, _graph: &BeliefGraph) -> Result<f64, ExecError> {
            Err(ExecError::Internal("no fields in test context".into()))
        }
    }

    #[test]
    fn eval_binary_op_arithmetic() {
        assert_eq!(eval_binary_op(BinaryOp::Add, 3.0, 4.0).unwrap(), 7.0);
        assert_eq!(eval_binary_op(BinaryOp::Sub, 10.0, 3.0).unwrap(), 7.0);
        assert_eq!(eval_binary_op(BinaryOp::Mul, 3.0, 4.0).unwrap(), 12.0);
        assert_eq!(eval_binary_op(BinaryOp::Div, 12.0, 4.0).unwrap(), 3.0);
    }

    #[test]
    fn eval_binary_op_division_by_zero() {
        assert!(eval_binary_op(BinaryOp::Div, 10.0, 0.0).is_err());
    }

    #[test]
    fn eval_binary_op_comparison() {
        assert_eq!(eval_binary_op(BinaryOp::Eq, 5.0, 5.0).unwrap(), 1.0);
        assert_eq!(eval_binary_op(BinaryOp::Eq, 5.0, 5.0 + 1e-13).unwrap(), 1.0); // Within epsilon
        assert_eq!(eval_binary_op(BinaryOp::Eq, 5.0, 5.0000001).unwrap(), 0.0); // Outside epsilon
        assert_eq!(eval_binary_op(BinaryOp::Ne, 5.0, 6.0).unwrap(), 1.0);
        assert_eq!(eval_binary_op(BinaryOp::Lt, 3.0, 5.0).unwrap(), 1.0);
        assert_eq!(eval_binary_op(BinaryOp::Gt, 5.0, 3.0).unwrap(), 1.0);
    }

    #[test]
    fn test_eval_unary_op() {
        assert_eq!(eval_unary_op(UnaryOp::Neg, 5.0), -5.0);
        assert_eq!(eval_unary_op(UnaryOp::Not, 0.0), 1.0);
        assert_eq!(eval_unary_op(UnaryOp::Not, 1.0), 0.0);
    }

    #[test]
    fn eval_expr_core_literals() {
        let ctx = TestContext;
        let graph = crate::engine::graph::BeliefGraph::default();

        assert_eq!(eval_expr_core(&ExprAst::Number(42.0), &graph, &ctx).unwrap(), 42.0);
        assert_eq!(eval_expr_core(&ExprAst::Bool(true), &graph, &ctx).unwrap(), 1.0);
        assert_eq!(eval_expr_core(&ExprAst::Bool(false), &graph, &ctx).unwrap(), 0.0);
    }

    #[test]
    fn eval_expr_core_variables() {
        let ctx = TestContext;
        let graph = crate::engine::graph::BeliefGraph::default();

        assert_eq!(eval_expr_core(&ExprAst::Var("x".into()), &graph, &ctx).unwrap(), 10.0);
        assert!(eval_expr_core(&ExprAst::Var("z".into()), &graph, &ctx).is_err());
    }

    #[test]
    fn eval_expr_core_binary() {
        let ctx = TestContext;
        let graph = crate::engine::graph::BeliefGraph::default();

        let expr = ExprAst::Binary {
            op: BinaryOp::Add,
            left: Box::new(ExprAst::Var("x".into())),
            right: Box::new(ExprAst::Var("y".into())),
        };
        assert_eq!(eval_expr_core(&expr, &graph, &ctx).unwrap(), 15.0);
    }
}

