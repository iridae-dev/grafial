//! Error types for Baygraph execution.

use thiserror::Error;

/// Errors that can occur during parsing, validation, or execution.
///
/// This enum is marked `#[non_exhaustive]` to allow adding new error variants
/// in the future without breaking changes.
#[non_exhaustive]
#[derive(Debug, Error)]
pub enum ExecError {
    /// Syntax error during parsing.
    ///
    /// Contains a human-readable description of the parse error,
    /// typically with line/column information from Pest.
    #[error("parse error: {0}")]
    ParseError(String),

    /// Semantic validation error after parsing.
    ///
    /// Indicates the program is syntactically valid but semantically incorrect,
    /// such as using `prob()` on a non-edge variable.
    #[error("validation error: {0}")]
    ValidationError(String),

    /// Internal execution error.
    ///
    /// Indicates an unexpected condition during execution, such as
    /// missing nodes/edges or invalid graph state.
    #[error("internal error: {0}")]
    Internal(String),
}
