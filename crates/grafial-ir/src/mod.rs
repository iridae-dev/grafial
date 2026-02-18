//! Intermediate representation module.
//!
//! NOTE: `lib.rs` is the crate root; this module mirrors exports for compatibility.

pub mod evidence;
pub mod expr;
pub mod flow;
pub mod optimize;
pub mod program;
pub mod rule;

pub use evidence::{EvidenceIR, EvidenceModeIR, NodeRefIR, ObserveStmtIR};
pub use expr::{BinaryOpIR, CallArgIR, ExprIR, UnaryOpIR};
pub use flow::{
    ExportDefIR, FlowIR, GraphDefIR, GraphExprIR, MetricDefIR, MetricExportDefIR,
    MetricImportDefIR, TransformIR,
};
pub use optimize::optimize_program;
pub use program::ProgramIR;
pub use rule::{ActionIR, RuleIR, VarianceSpecIR};
