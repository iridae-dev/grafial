//! # Grafial Core
//!
//! Core engine for Grafial Bayesian belief graphs.

pub mod engine;
pub mod metrics;
pub mod storage;

// Re-export commonly used types
pub use engine::errors::ExecError;
pub use engine::flow_exec::run_flow;
pub use engine::graph::BeliefGraph;

/// Parse and validate a Grafial program.
///
/// This is a convenience function that combines parsing and validation,
/// converting frontend errors to core errors.
pub fn parse_and_validate(source: &str) -> Result<grafial_frontend::ProgramAst, ExecError> {
    let ast = grafial_frontend::parse_program(source)?;
    grafial_frontend::validate_program(&ast)?;
    Ok(ast)
}
