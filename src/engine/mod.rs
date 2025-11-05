//! The execution engine for Baygraph belief graphs.
//!
//! This module provides:
//! - **errors**: Error types for execution failures
//! - **graph**: Core belief graph data structure with Bayesian inference
//! - **rule_exec**: Pattern matching and rule execution engine
//! - **flow_exec**: Flow execution and graph transformation pipelines

pub mod errors;
pub mod graph;
pub mod rule_exec;
pub mod flow_exec;

