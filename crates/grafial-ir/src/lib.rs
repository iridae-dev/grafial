//! # Grafial IR
//!
//! Intermediate representation for Grafial programs.

pub mod flow;
pub mod program;
pub mod rule;

// Re-export commonly used types
pub use flow::{FlowIR, GraphDefIR, GraphExprIR, TransformIR};
pub use program::ProgramIR;
pub use rule::RuleIR;
