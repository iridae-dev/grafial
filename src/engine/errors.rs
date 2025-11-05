//! Error types for Baygraph execution.

use thiserror::Error;

/// Errors that can occur during parsing, validation, or execution.
///
/// This enum is marked `#[non_exhaustive]` to allow adding new error variants
/// in the future without breaking changes.
///
/// # Phase 7 Error Improvements
///
/// Enhanced error types for better debugging and user feedback.
/// See baygraph_design.md:533-536 for error handling strategy.
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

    /// Runtime execution error.
    ///
    /// Indicates an error during rule or flow execution, such as:
    /// - Fixpoint rules that don't converge within iteration limits
    /// - Invalid graph transformations
    /// - Constraint violations
    ///
    /// Phase 7: Added for better error categorization
    #[error("execution error: {0}")]
    Execution(String),

    /// Numerical stability error.
    ///
    /// Indicates numerical issues such as:
    /// - NaN or Inf values in computations
    /// - Precision underflow/overflow
    /// - Invalid probability values outside [0, 1]
    ///
    /// Phase 7: Added for numerical robustness
    #[error("numerical error: {0}")]
    Numerical(String),

    /// Internal execution error.
    ///
    /// Indicates an unexpected condition during execution, such as
    /// missing nodes/edges or invalid graph state.
    /// This should be used only for programmer errors, not user errors.
    #[error("internal error: {0}")]
    Internal(String),
}
