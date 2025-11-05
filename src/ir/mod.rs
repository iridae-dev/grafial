//! Intermediate representation module.
//!
//! This module provides a canonical lowered representation (GraphIR, RuleIR, FlowIR)
//! that decouples execution from parsing structures. The IR is a stable interface
//! between frontend and engine.
//!
//! Lowering is a shallow clone from AST to IR, but provides a clean boundary
//! for future optimizations or multiple frontends.

pub mod flow;
pub mod rule;
pub mod program;

// Re-export commonly used types
pub use flow::{FlowIR, GraphDefIR, GraphExprIR, TransformIR};
pub use rule::RuleIR;
pub use program::ProgramIR;

