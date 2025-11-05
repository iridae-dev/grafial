//! Utility functions for expression argument parsing.
//!
//! Provides reusable functions for extracting positional and named arguments
//! from CallArg slices, eliminating duplication across different evaluation contexts.

use crate::frontend::ast::{CallArg, ExprAst};

/// Splits call arguments into positional and named arguments.
///
/// This is a utility function used by multiple expression evaluators to
/// parse function call arguments consistently.
///
/// # Returns
///
/// A tuple of:
/// - `Vec<&ExprAst>`: Positional arguments in order
/// - `Vec<(&str, &ExprAst)>`: Named arguments as (name, value) pairs
pub fn split_args<'a>(args: &'a [CallArg]) -> (Vec<&'a ExprAst>, Vec<(&'a str, &'a ExprAst)>) {
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

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn split_args_positional_only() {
        let args = vec![
            CallArg::Positional(ExprAst::Number(1.0)),
            CallArg::Positional(ExprAst::Number(2.0)),
        ];
        let (pos, named) = split_args(&args);
        assert_eq!(pos.len(), 2);
        assert_eq!(named.len(), 0);
    }

    #[test]
    fn split_args_mixed() {
        let args = vec![
            CallArg::Positional(ExprAst::Number(1.0)),
            CallArg::Named {
                name: "x".into(),
                value: ExprAst::Number(2.0),
            },
            CallArg::Positional(ExprAst::Number(3.0)),
        ];
        let (pos, named) = split_args(&args);
        assert_eq!(pos.len(), 2);
        assert_eq!(named.len(), 1);
        assert_eq!(named[0].0, "x");
    }
}

