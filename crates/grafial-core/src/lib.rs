//! # Grafial Core
//!
//! Core engine for Grafial Bayesian belief graphs.

pub mod engine;
pub mod metrics;
pub mod storage;

// Re-export commonly used types
pub use engine::errors::ExecError;
pub use engine::flow_exec::{
    run_flow, run_flow_ir, run_flow_ir_with_backend, InterpreterExecutionBackend,
    IrExecutionBackend,
};
pub use engine::graph::BeliefGraph;
pub use grafial_ir::ProgramIR;

/// Parse and validate a Grafial program.
///
/// This is a convenience function that combines parsing and validation,
/// converting frontend errors to core errors.
pub fn parse_and_validate(source: &str) -> Result<grafial_frontend::ProgramAst, ExecError> {
    let ast = grafial_frontend::parse_program(source)?;
    grafial_frontend::validate_program_with_source(&ast, source)?;
    Ok(ast)
}

/// Parse, validate, and lower a Grafial program to IR.
pub fn parse_validate_and_lower(source: &str) -> Result<ProgramIR, ExecError> {
    let ast = parse_and_validate(source)?;
    Ok(ProgramIR::from(&ast).optimized())
}
