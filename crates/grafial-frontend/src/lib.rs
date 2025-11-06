//! # Grafial Frontend
//!
//! Parser, AST, and validation for the Grafial DSL.

pub mod ast;
pub mod errors;
pub mod parser;
pub mod validate;

// Re-export commonly used types
pub use ast::*;
pub use errors::FrontendError;
pub use parser::parse_program;
pub use validate::validate_program;
