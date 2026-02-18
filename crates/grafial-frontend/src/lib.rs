//! # Grafial Frontend
//!
//! Parser, AST, and validation for the Grafial DSL.

pub mod ast;
pub mod errors;
pub mod parser;
pub mod style;
pub mod validate;

// Re-export commonly used types
pub use ast::*;
pub use errors::{
    FrontendError, SourcePosition, SourceRange, ValidationContext, ValidationDiagnostic,
};
pub use parser::parse_program;
pub use style::{format_canonical_style, lint_canonical_style, CanonicalStyleLint};
pub use validate::{validate_program, validate_program_with_source};
