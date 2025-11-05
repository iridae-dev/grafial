//! Intermediate representation module.
//!
//! For Phase 4 we introduce a minimal IR for flows that mirrors the
//! already-typed AST but gives the engine a stable surface to depend on.
//! Lowering is a shallow clone from AST to IR.

pub mod flow;

