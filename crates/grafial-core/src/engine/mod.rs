//! The execution engine for Grafial belief graphs.
//!
//! This module provides:
//! - **errors**: Error types for execution failures
//! - **graph**: Core belief graph data structure with Bayesian inference
//! - **expr_eval**: Shared expression evaluation core (DRY principle)
//! - **rule_exec**: Pattern matching and rule execution engine
//! - **flow_exec**: Flow execution and graph transformation pipelines

pub mod errors;
pub mod evidence;
pub mod expr_eval;
pub mod expr_utils;
pub mod flow_exec;
pub mod graph;
pub mod query_plan;
pub mod rule_exec;
pub mod snapshot;
