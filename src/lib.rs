//! # Baygraph - Bayesian Belief Graph DSL
//!
//! Baygraph is a domain-specific language for probabilistic reasoning over graphs,
//! combining graph pattern matching with Bayesian inference.
//!
//! ## Architecture
//!
//! The system is organized into several modules:
//!
//! - **frontend**: Parser and AST for the Baygraph DSL
//! - **engine**: Core execution engine with Bayesian graph operations
//! - **ir**: Intermediate representation (future use)
//! - **metrics**: Graph metrics computation
//! - **storage**: Persistence layer
//! - **bindings**: Language bindings for external use
//!
//! ## Usage
//!
//! ```rust,ignore
//! use baygraph::{parse_and_validate, ast::ProgramAst};
//!
//! let source = r#"
//!     schema Social {
//!         node Person { age: Real }
//!         edge Knows {}
//!     }
//!     belief_model M on Social {}
//! "#;
//!
//! let ast = parse_and_validate(source).expect("valid program");
//! ```

#![forbid(unsafe_code)]

pub mod frontend;
pub mod ir;
pub mod engine;
pub mod metrics;
pub mod storage;
pub mod bindings;

// Re-export commonly used types
pub use frontend::ast;
pub use frontend::validate;

/// Parses Baygraph DSL source code into an unvalidated AST.
///
/// This function performs syntactic parsing only. The resulting AST may contain
/// semantic errors. Use [`parse_and_validate`] for full validation.
///
/// # Arguments
///
/// * `source` - The Baygraph DSL source code to parse
///
/// # Returns
///
/// * `Ok(ProgramAst)` - Successfully parsed AST
/// * `Err(ExecError::ParseError)` - Syntax error in the source
///
/// # Example
///
/// ```rust,ignore
/// use baygraph::parse_program;
///
/// let source = "schema S { node N {} edge E {} }";
/// let ast = parse_program(source)?;
/// ```
pub fn parse_program(source: &str) -> Result<ast::ProgramAst, engine::errors::ExecError> {
    frontend::parser::parse_program(source)
}

/// Parses and validates Baygraph DSL source code.
///
/// This function performs both syntactic parsing and semantic validation,
/// ensuring the program is well-formed and ready for execution.
///
/// # Validation Checks
///
/// - `prob()` is only called on edge variables
/// - `E[node.attr]` uses valid node variables and attributes
/// - `degree()` uses valid node variables
/// - Metric functions have required arguments (label, contrib, etc.)
/// - Prune predicates use the correct `edge` variable
///
/// # Arguments
///
/// * `source` - The Baygraph DSL source code to parse and validate
///
/// # Returns
///
/// * `Ok(ProgramAst)` - Successfully parsed and validated AST
/// * `Err(ExecError)` - Parse error or validation error
///
/// # Example
///
/// ```rust,ignore
/// use baygraph::parse_and_validate;
///
/// let source = r#"
///     schema S {
///         node N { x: Real }
///         edge E {}
///     }
///     belief_model M on S {}
/// "#;
/// let ast = parse_and_validate(source)?;
/// ```
pub fn parse_and_validate(source: &str) -> Result<ast::ProgramAst, engine::errors::ExecError> {
    let ast = parse_program(source)?;
    validate::validate_program(&ast)?;
    Ok(ast)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn parse_program_parses_minimal_schema() {
        let src = "schema S { node N {} edge E {} }";
        let result = parse_program(src);

        assert!(result.is_ok());
        let ast = result.unwrap();
        assert_eq!(ast.schemas.len(), 1);
        assert_eq!(ast.schemas[0].name, "S");
    }

    #[test]
    fn parse_program_returns_error_on_invalid_syntax() {
        let src = "this is not valid syntax";
        let result = parse_program(src);

        assert!(result.is_err());
    }

    #[test]
    fn parse_program_parses_complete_program() {
        let src = r#"
            schema S {
                node N { x: Real }
                edge E {}
            }
            belief_model M on S {}
            rule R on M {
                pattern (A:N)-[e:E]->(B:N)
            }
            flow F on M {
                graph g = from_evidence Ev
            }
        "#;

        let result = parse_program(src);
        assert!(result.is_ok());

        let ast = result.unwrap();
        assert_eq!(ast.schemas.len(), 1);
        assert_eq!(ast.belief_models.len(), 1);
        assert_eq!(ast.rules.len(), 1);
        assert_eq!(ast.flows.len(), 1);
    }

    #[test]
    fn parse_and_validate_accepts_valid_program() {
        let src = r#"
            schema S {
                node N { x: Real }
                edge E {}
            }
            belief_model M on S {}
            rule R on M {
                pattern (A:N)-[e:E]->(B:N)
                where prob(e) >= 0.5
            }
            flow F on M {
                graph g = from_evidence Ev
            }
        "#;

        let result = parse_and_validate(src);
        assert!(result.is_ok());
    }

    #[test]
    fn parse_and_validate_rejects_invalid_validation() {
        let src = r#"
            schema S {
                node N { x: Real }
                edge E {}
            }
            belief_model M on S {}
            rule R on M {
                pattern (A:N)-[e:E]->(B:N)
                where prob(A) >= 0.5
            }
        "#;

        // prob(A) is invalid - should be prob(e)
        let result = parse_and_validate(src);
        assert!(result.is_err());
    }

    #[test]
    fn parse_and_validate_rejects_invalid_syntax() {
        let src = "invalid { syntax }}";

        let result = parse_and_validate(src);
        assert!(result.is_err());
    }

    #[test]
    fn parse_program_handles_empty_string() {
        let src = "";
        let result = parse_program(src);

        // Empty input should either parse as empty AST or error
        let _ = result;
    }

    #[test]
    fn parse_program_preserves_schema_details() {
        let src = r#"
            schema TestSchema {
                node Person { age: Real score: Real }
                edge Knows {}
            }
        "#;

        let result = parse_program(src).unwrap();
        let schema = &result.schemas[0];

        assert_eq!(schema.name, "TestSchema");
        assert_eq!(schema.nodes.len(), 1);
        assert_eq!(schema.nodes[0].name, "Person");
        assert_eq!(schema.nodes[0].attrs.len(), 2);
        assert_eq!(schema.edges.len(), 1);
        assert_eq!(schema.edges[0].name, "Knows");
    }

    #[test]
    fn parse_program_handles_multiple_schemas() {
        let src = r#"
            schema S1 { node N1 {} edge E1 {} }
            schema S2 { node N2 {} edge E2 {} }
        "#;

        let result = parse_program(src).unwrap();
        assert_eq!(result.schemas.len(), 2);
        assert_eq!(result.schemas[0].name, "S1");
        assert_eq!(result.schemas[1].name, "S2");
    }

    #[test]
    fn parse_and_validate_integration_with_public_api() {
        // Test that the public API functions work together
        let src = r#"
            schema S { node N { x: Real } edge E {} }
            belief_model M on S {}
        "#;

        // First parse
        let ast1 = parse_program(src).unwrap();
        assert_eq!(ast1.schemas.len(), 1);

        // Then parse and validate
        let ast2 = parse_and_validate(src).unwrap();
        assert_eq!(ast2.schemas.len(), 1);

        // Both should produce equivalent results
        assert_eq!(ast1.schemas[0].name, ast2.schemas[0].name);
    }
}
