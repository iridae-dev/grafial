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
    #[error("parse error: {0}")]
    ParseError(String),

    /// Semantic validation error (e.g., using `prob()` on a non-edge variable).
    #[error("validation error: {0}")]
    ValidationError(String),

    /// Runtime execution error (e.g., fixpoint non-convergence, invalid transformations).
    #[error("execution error: {0}")]
    Execution(String),

    /// Numerical stability error (NaN/Inf, precision issues, invalid probabilities).
    #[error("numerical error: {0}")]
    Numerical(String),

    /// Internal execution error (programmer error, not user error).
    #[error("internal error: {0}")]
    Internal(String),
}
