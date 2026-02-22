//! The execution engine for Grafial belief graphs.
//!
//! This module provides:
//! - **errors**: Error types for execution failures
//! - **graph**: Core belief graph data structure with Bayesian inference
//! - **expr_eval**: Shared expression evaluation core (DRY principle)
//! - **rule_exec**: Pattern matching and rule execution engine
//! - **flow_exec**: Flow execution and graph transformation pipelines

pub mod adjacency_index;
#[cfg(any(feature = "aot", test))]
pub mod aot_flows;
#[cfg(feature = "aot")]
pub mod aot_integration;
pub mod arena_allocator;
pub mod belief_propagation;
pub mod errors;
pub mod evidence;
pub mod expr_eval;
pub mod expr_utils;
pub mod flow_exec;
pub mod graph;
#[cfg(feature = "jit")]
pub mod jit_backend;
pub mod model_selection;
pub mod numeric_kernels;
#[cfg(feature = "parallel")]
pub mod parallel_evidence;
#[cfg(feature = "parallel")]
pub mod parallel_flow;
#[cfg(feature = "parallel")]
pub mod parallel_graph;
#[cfg(feature = "parallel")]
pub mod parallel_metrics;
#[cfg(feature = "parallel")]
pub mod parallel_rules;
pub mod query_plan;
pub mod rule_exec;
#[cfg(feature = "jit")]
pub mod rule_kernels;
pub mod snapshot;
#[cfg(feature = "vectorized")]
pub mod vectorized;
