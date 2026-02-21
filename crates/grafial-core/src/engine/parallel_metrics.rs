//! Parallel metric evaluation with dependency analysis.
//!
//! This module provides parallel computation of metrics while respecting
//! dependencies between them. Metrics that don't depend on each other
//! are computed in parallel for improved performance.
//!
//! ## Architecture
//!
//! - **Dependency analysis**: Builds DAG of metric dependencies
//! - **Topological sorting**: Determines evaluation order
//! - **Parallel evaluation**: Metrics in same level evaluated concurrently
//! - **Result caching**: Avoids redundant computation
//!
//! ## Feature gating
//!
//! Parallel metrics are behind the `parallel` feature flag. When disabled,
//! metrics are evaluated sequentially.

#[cfg(feature = "parallel")]
use rayon::prelude::*;
use std::collections::{HashMap, HashSet};

use crate::engine::errors::ExecError;
use crate::engine::graph::BeliefGraph;
use grafial_ir::{ExprIR, MetricDefIR};

/// Result of parallel metric evaluation.
#[derive(Debug)]
pub struct ParallelMetricResult {
    /// Computed metric values
    pub values: HashMap<String, f64>,
    /// Statistics about parallel execution
    pub stats: ParallelMetricStats,
}

/// Statistics about parallel metric evaluation.
#[derive(Debug, Default)]
pub struct ParallelMetricStats {
    /// Number of metrics evaluated
    pub metrics_evaluated: usize,
    /// Number of parallel levels in dependency graph
    pub parallel_levels: usize,
    /// Maximum parallelism achieved
    pub max_parallelism: usize,
}

/// Dependency graph for metrics.
#[derive(Debug)]
struct MetricDependencyGraph {
    /// Adjacency list: metric -> set of metrics it depends on
    dependencies: HashMap<String, HashSet<String>>,
    /// Reverse adjacency list: metric -> set of metrics that depend on it
    dependents: HashMap<String, HashSet<String>>,
}

impl MetricDependencyGraph {
    /// Build dependency graph from metric definitions.
    fn from_metrics(metrics: &[MetricDefIR]) -> Result<Self, ExecError> {
        let mut dependencies = HashMap::new();
        let mut dependents = HashMap::new();

        for metric in metrics {
            let deps = extract_dependencies(&metric.expr);
            dependencies.insert(metric.name.clone(), deps.clone());

            // Update reverse dependencies
            for dep in deps {
                dependents
                    .entry(dep)
                    .or_insert_with(HashSet::new)
                    .insert(metric.name.clone());
            }
        }

        Ok(Self {
            dependencies,
            dependents,
        })
    }

    /// Compute topological levels for parallel evaluation.
    ///
    /// Returns a vector where each element is a set of metrics that can be
    /// evaluated in parallel (they have no dependencies on each other).
    fn topological_levels(&self) -> Result<Vec<Vec<String>>, ExecError> {
        let mut levels = Vec::new();
        let mut in_degree: HashMap<String, usize> = HashMap::new();
        let mut processed = HashSet::new();

        // Calculate in-degrees
        for (metric, deps) in &self.dependencies {
            in_degree.insert(metric.clone(), deps.len());
        }

        // Process metrics level by level
        loop {
            let mut current_level = Vec::new();

            // Find all metrics with in-degree 0 that haven't been processed
            for (metric, &degree) in &in_degree {
                if degree == 0 && !processed.contains(metric) {
                    current_level.push(metric.clone());
                }
            }

            if current_level.is_empty() {
                break;
            }

            // Mark current level as processed
            for metric in &current_level {
                processed.insert(metric.clone());

                // Reduce in-degree of dependent metrics
                if let Some(deps) = self.dependents.get(metric) {
                    for dep in deps {
                        if let Some(degree) = in_degree.get_mut(dep) {
                            *degree = degree.saturating_sub(1);
                        }
                    }
                }
            }

            levels.push(current_level);
        }

        // Check for cycles
        if processed.len() != self.dependencies.len() {
            return Err(ExecError::ValidationError(
                "Circular dependency detected in metrics".to_string(),
            ));
        }

        Ok(levels)
    }
}

/// Extract metric dependencies from an expression.
fn extract_dependencies(expr: &ExprIR) -> HashSet<String> {
    let mut deps = HashSet::new();

    // For now, we'll just extract variable references
    // A full implementation would recursively walk the expression tree
    match expr {
        ExprIR::Var(name) => {
            // Assume variables starting with "metric_" are metric references
            // This avoids confusion with metric names that start with "m_"
            if name.starts_with("metric_") {
                // Extract the actual metric name after "metric_"
                let metric_name = name.strip_prefix("metric_").unwrap_or(name);
                deps.insert(metric_name.to_string());
            }
        }
        _ => {
            // TODO: Recursively extract from other expression types
        }
    }

    deps
}

/// Evaluate metrics in parallel respecting dependencies.
#[cfg(feature = "parallel")]
pub fn evaluate_metrics_parallel(
    graph: &BeliefGraph,
    metrics: &[MetricDefIR],
) -> Result<ParallelMetricResult, ExecError> {
    // Build dependency graph
    let dep_graph = MetricDependencyGraph::from_metrics(metrics)?;

    // Get topological levels for parallel evaluation
    let levels = dep_graph.topological_levels()?;

    let mut values = HashMap::new();
    let mut max_parallelism = 0;

    // Build metric map for quick lookup
    let metric_map: HashMap<String, &MetricDefIR> =
        metrics.iter().map(|m| (m.name.clone(), m)).collect();

    // Evaluate metrics level by level
    for level in &levels {
        max_parallelism = max_parallelism.max(level.len());

        // Evaluate all metrics in this level in parallel
        let level_results: Vec<_> = level
            .par_iter()
            .map(|metric_name| {
                let metric_def = metric_map.get(metric_name).ok_or_else(|| {
                    ExecError::ValidationError(format!("Metric {} not found", metric_name))
                })?;

                let value = evaluate_metric_expr(graph, &metric_def.expr, &values)?;
                Ok((metric_name.clone(), value))
            })
            .collect::<Result<Vec<_>, ExecError>>()?;

        // Store results from this level
        for (name, value) in level_results {
            values.insert(name, value);
        }
    }

    Ok(ParallelMetricResult {
        values,
        stats: ParallelMetricStats {
            metrics_evaluated: metrics.len(),
            parallel_levels: levels.len(),
            max_parallelism,
        },
    })
}

/// Sequential fallback for metric evaluation.
#[cfg(not(feature = "parallel"))]
pub fn evaluate_metrics_parallel(
    graph: &BeliefGraph,
    metrics: &[MetricDefIR],
) -> Result<ParallelMetricResult, ExecError> {
    let mut values = HashMap::new();

    // Simple sequential evaluation
    for metric in metrics {
        let value = evaluate_metric_expr(graph, &metric.expr, &values)?;
        values.insert(metric.name.clone(), value);
    }

    Ok(ParallelMetricResult {
        values,
        stats: ParallelMetricStats {
            metrics_evaluated: metrics.len(),
            parallel_levels: 1, // Sequential = 1 level
            max_parallelism: 1, // Sequential = no parallelism
        },
    })
}

/// Evaluate a single metric expression.
fn evaluate_metric_expr(
    _graph: &BeliefGraph,
    expr: &ExprIR,
    computed_values: &HashMap<String, f64>,
) -> Result<f64, ExecError> {
    // Simplified evaluation for now
    // A full implementation would use the expr_eval module
    match expr {
        ExprIR::Number(n) => Ok(*n),
        ExprIR::Var(name) => computed_values
            .get(name)
            .copied()
            .ok_or_else(|| ExecError::ValidationError(format!("Variable {} not found", name))),
        _ => {
            // TODO: Implement full expression evaluation
            Ok(0.0)
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_dependency_extraction() {
        let expr = ExprIR::Var("metric_m1".to_string());

        let deps = extract_dependencies(&expr);
        assert_eq!(deps.len(), 1);
        assert!(deps.contains("m1")); // Should extract "m1" after the prefix
    }

    #[test]
    fn test_topological_levels() {
        let metrics = vec![
            MetricDefIR {
                name: "m1".to_string(),
                expr: ExprIR::Number(1.0),
            },
            MetricDefIR {
                name: "m2".to_string(),
                expr: ExprIR::Number(2.0),
            },
            MetricDefIR {
                name: "m3".to_string(),
                expr: ExprIR::Var("metric_m1".to_string()), // Depends on m1 via metric_ prefix
            },
        ];

        let dep_graph = MetricDependencyGraph::from_metrics(&metrics).unwrap();
        let levels = dep_graph.topological_levels().unwrap();

        // m1 and m2 should be in the first level (no dependencies)
        // m3 should be in the second level (depends on m1)
        assert_eq!(levels.len(), 2);
        assert_eq!(levels[0].len(), 2); // m1 and m2
        assert_eq!(levels[1].len(), 1); // m3
        assert!(levels[1].contains(&"m3".to_string()));
    }

    #[test]
    fn test_circular_dependency_detection() {
        let metrics = vec![
            MetricDefIR {
                name: "m1".to_string(),
                expr: ExprIR::Var("metric_m2".to_string()), // m1 depends on m2
            },
            MetricDefIR {
                name: "m2".to_string(),
                expr: ExprIR::Var("metric_m1".to_string()), // m2 depends on m1 - circular!
            },
        ];

        let dep_graph = MetricDependencyGraph::from_metrics(&metrics).unwrap();
        let result = dep_graph.topological_levels();

        assert!(result.is_err());
        if let Err(ExecError::ValidationError(msg)) = result {
            assert!(msg.contains("Circular dependency"));
        }
    }
}
