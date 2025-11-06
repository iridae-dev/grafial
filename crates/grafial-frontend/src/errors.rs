//! Error types for parsing and validation.

use thiserror::Error;

/// Errors that can occur during parsing or validation.
#[non_exhaustive]
#[derive(Debug, Error)]
pub enum FrontendError {
    /// Syntax error during parsing.
    #[error("parse error: {0}")]
    ParseError(String),

    /// Semantic validation error.
    #[error("validation error: {0}")]
    ValidationError(String),
}
