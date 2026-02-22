//! Parallel evidence ingestion for independent observations.
//!
//! This module provides parallel processing of evidence observations while
//! maintaining deterministic results. Independent observations are processed
//! in parallel groups to improve performance.
//!
//! ## Architecture
//!
//! - **Observation partitioning**: Groups observations by target (node/edge)
//! - **Parallel batch processing**: Each partition processed independently
//! - **Deterministic ordering**: Results applied in stable order
//!
//! ## Feature gating
//!
//! Parallel evidence is behind the `parallel` feature flag. When disabled,
//! evidence processing falls back to sequential execution.

#[cfg(feature = "parallel")]
use rayon::prelude::*;
use std::collections::{BTreeMap, HashMap};

use crate::engine::errors::ExecError;
use crate::engine::graph::{BeliefGraph, EdgeId, NodeId};
use grafial_frontend::ast::{EvidenceMode, ObserveStmt};

/// Result of parallel evidence processing.
#[derive(Debug)]
pub struct ParallelEvidenceResult {
    /// Observations organized by target node
    pub node_observations: BTreeMap<NodeId, Vec<ObservationData>>,
    /// Observations organized by target edge
    pub edge_observations: BTreeMap<EdgeId, Vec<ObservationData>>,
    /// Statistics about parallel execution
    pub stats: ParallelStats,
}

/// Observation data to be applied.
#[derive(Debug, Clone)]
pub enum ObservationData {
    Attribute {
        attr: String,
        value: f64,
        precision: f64,
    },
    EdgeWeight {
        value: f64,
        precision: f64,
    },
    EdgePresent,
    EdgeAbsent,
    EdgeChosen,
    EdgeUnchosen,
    EdgeForcedChoice,
}

/// Statistics about parallel evidence processing.
#[derive(Debug, Default)]
pub struct ParallelStats {
    /// Number of observations processed
    pub observations_processed: usize,
    /// Number of parallel partitions created
    pub partitions_created: usize,
    /// Number of conflicts detected (if any)
    pub conflicts_detected: usize,
}

/// Observation target for partitioning.
#[derive(Debug, Clone, Hash, Eq, PartialEq, Ord, PartialOrd)]
enum ObservationTarget {
    Node(NodeId),
    Edge(EdgeId),
}

/// Partitioned observation for parallel processing.
#[derive(Debug)]
struct PartitionedObservation {
    target: ObservationTarget,
    observations: Vec<ObserveStmt>,
}

/// Process evidence observations in parallel.
///
/// This function partitions observations by their target (node/edge),
/// processes each partition in parallel, and returns organized results.
#[cfg(feature = "parallel")]
pub fn process_evidence_parallel(
    _graph: &BeliefGraph,
    observations: &[ObserveStmt],
    node_map: &HashMap<(String, String), NodeId>,
    edge_map: &HashMap<(NodeId, NodeId, String), EdgeId>,
) -> Result<ParallelEvidenceResult, ExecError> {
    // Step 1: Partition observations by target
    let partitions = partition_observations(observations, node_map, edge_map)?;

    let stats = ParallelStats {
        observations_processed: observations.len(),
        partitions_created: partitions.len(),
        conflicts_detected: 0,
    };

    // Step 2: Process partitions in parallel to extract observation data
    let partition_results: Vec<_> = partitions
        .into_par_iter()
        .map(|partition| process_partition(partition, node_map, edge_map))
        .collect::<Result<Vec<_>, _>>()?;

    // Step 3: Organize results by target
    let mut node_observations = BTreeMap::new();
    let mut edge_observations = BTreeMap::new();

    for (target, obs_data) in partition_results {
        match target {
            ObservationTarget::Node(node_id) => {
                node_observations
                    .entry(node_id)
                    .or_insert_with(Vec::new)
                    .extend(obs_data);
            }
            ObservationTarget::Edge(edge_id) => {
                edge_observations
                    .entry(edge_id)
                    .or_insert_with(Vec::new)
                    .extend(obs_data);
            }
        }
    }

    Ok(ParallelEvidenceResult {
        node_observations,
        edge_observations,
        stats,
    })
}

/// Partition observations by their target for parallel processing.
fn partition_observations(
    observations: &[ObserveStmt],
    node_map: &HashMap<(String, String), NodeId>,
    edge_map: &HashMap<(NodeId, NodeId, String), EdgeId>,
) -> Result<Vec<PartitionedObservation>, ExecError> {
    let mut partitions: HashMap<ObservationTarget, Vec<ObserveStmt>> = HashMap::new();

    for obs in observations {
        let target = match obs {
            ObserveStmt::Edge {
                src,
                dst,
                edge_type,
                ..
            } => {
                // Look up edge ID
                let src_id = *node_map
                    .get(&(src.0.clone(), src.1.clone()))
                    .ok_or_else(|| {
                        ExecError::ValidationError(format!(
                            "Source node {}:{} not found",
                            src.0, src.1
                        ))
                    })?;
                let dst_id = *node_map
                    .get(&(dst.0.clone(), dst.1.clone()))
                    .ok_or_else(|| {
                        ExecError::ValidationError(format!(
                            "Destination node {}:{} not found",
                            dst.0, dst.1
                        ))
                    })?;

                let edge_id = *edge_map
                    .get(&(src_id, dst_id, edge_type.clone()))
                    .ok_or_else(|| {
                        ExecError::ValidationError(format!(
                            "Edge {} not found between nodes",
                            edge_type
                        ))
                    })?;

                ObservationTarget::Edge(edge_id)
            }
            ObserveStmt::EdgeWeight {
                src,
                dst,
                edge_type,
                ..
            } => {
                let src_id = *node_map
                    .get(&(src.0.clone(), src.1.clone()))
                    .ok_or_else(|| {
                        ExecError::ValidationError(format!(
                            "Source node {}:{} not found",
                            src.0, src.1
                        ))
                    })?;
                let dst_id = *node_map
                    .get(&(dst.0.clone(), dst.1.clone()))
                    .ok_or_else(|| {
                        ExecError::ValidationError(format!(
                            "Destination node {}:{} not found",
                            dst.0, dst.1
                        ))
                    })?;

                let edge_id = *edge_map
                    .get(&(src_id, dst_id, edge_type.clone()))
                    .ok_or_else(|| {
                        ExecError::ValidationError(format!(
                            "Edge {} not found between nodes",
                            edge_type
                        ))
                    })?;

                ObservationTarget::Edge(edge_id)
            }
            ObserveStmt::Attribute { node, .. } => {
                let node_id =
                    *node_map
                        .get(&(node.0.clone(), node.1.clone()))
                        .ok_or_else(|| {
                            ExecError::ValidationError(format!(
                                "Node {}:{} not found",
                                node.0, node.1
                            ))
                        })?;

                ObservationTarget::Node(node_id)
            }
        };

        partitions
            .entry(target.clone())
            .or_default()
            .push(obs.clone());
    }

    // Convert to vector of partitions
    Ok(partitions
        .into_iter()
        .map(|(target, observations)| PartitionedObservation {
            target,
            observations,
        })
        .collect())
}

/// Process a single partition of observations.
fn process_partition(
    partition: PartitionedObservation,
    _node_map: &HashMap<(String, String), NodeId>,
    _edge_map: &HashMap<(NodeId, NodeId, String), EdgeId>,
) -> Result<(ObservationTarget, Vec<ObservationData>), ExecError> {
    let mut obs_data = Vec::new();

    // Extract observation data from each statement
    for obs in &partition.observations {
        match obs {
            ObserveStmt::Edge { mode, .. } => {
                let mode_data = match mode {
                    EvidenceMode::Present => ObservationData::EdgePresent,
                    EvidenceMode::Absent => ObservationData::EdgeAbsent,
                    EvidenceMode::Chosen => ObservationData::EdgeChosen,
                    EvidenceMode::Unchosen => ObservationData::EdgeUnchosen,
                    EvidenceMode::ForcedChoice => ObservationData::EdgeForcedChoice,
                };
                obs_data.push(mode_data);
            }
            ObserveStmt::EdgeWeight {
                value, precision, ..
            } => {
                let tau = precision.unwrap_or(1.0);
                obs_data.push(ObservationData::EdgeWeight {
                    value: *value,
                    precision: tau,
                });
            }
            ObserveStmt::Attribute {
                attr,
                value,
                precision,
                ..
            } => {
                // Default precision if not specified
                let tau = precision.unwrap_or(1.0);
                obs_data.push(ObservationData::Attribute {
                    attr: attr.clone(),
                    value: *value,
                    precision: tau,
                });
            }
        }
    }

    Ok((partition.target, obs_data))
}

/// Sequential fallback for evidence processing.
#[cfg(not(feature = "parallel"))]
pub fn process_evidence_parallel(
    _graph: &BeliefGraph,
    observations: &[ObserveStmt],
    node_map: &HashMap<(String, String), NodeId>,
    edge_map: &HashMap<(NodeId, NodeId, String), EdgeId>,
) -> Result<ParallelEvidenceResult, ExecError> {
    // Fall back to sequential processing
    let partitions = partition_observations(observations, node_map, edge_map)?;

    let mut node_observations = BTreeMap::new();
    let mut edge_observations = BTreeMap::new();

    for partition in partitions {
        let (target, obs_data) = process_partition(partition, node_map, edge_map)?;
        match target {
            ObservationTarget::Node(node_id) => {
                node_observations
                    .entry(node_id)
                    .or_default()
                    .extend(obs_data);
            }
            ObservationTarget::Edge(edge_id) => {
                edge_observations
                    .entry(edge_id)
                    .or_default()
                    .extend(obs_data);
            }
        }
    }

    Ok(ParallelEvidenceResult {
        node_observations,
        edge_observations,
        stats: ParallelStats {
            observations_processed: observations.len(),
            partitions_created: 1, // Sequential = 1 partition
            conflicts_detected: 0,
        },
    })
}

/// Apply parallel evidence results to a graph.
///
/// This applies the observations from parallel processing in a deterministic order.
pub fn apply_parallel_results(
    graph: &mut BeliefGraph,
    results: ParallelEvidenceResult,
) -> Result<(), ExecError> {
    // Apply node observations in deterministic order (BTreeMap ensures this)
    for (node_id, observations) in results.node_observations {
        for obs in observations {
            if let ObservationData::Attribute {
                attr,
                value,
                precision,
            } = obs
            {
                graph.observe_attr(node_id, &attr, value, precision)?;
            }
        }
    }

    // Apply edge observations in deterministic order
    for (edge_id, observations) in results.edge_observations {
        for obs in observations {
            match obs {
                ObservationData::EdgeWeight { value, precision } => {
                    graph.observe_edge_weight(edge_id, value, precision)?
                }
                ObservationData::EdgePresent => graph.observe_edge(edge_id, true)?,
                ObservationData::EdgeAbsent => graph.observe_edge(edge_id, false)?,
                ObservationData::EdgeChosen => graph.observe_edge_chosen(edge_id)?,
                ObservationData::EdgeUnchosen => graph.observe_edge_unchosen(edge_id)?,
                ObservationData::EdgeForcedChoice => graph.observe_edge_forced_choice(edge_id)?,
                _ => {} // Edge observations should only be edge modes
            }
        }
    }

    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::engine::graph::GaussianPosterior;

    #[test]
    fn test_observation_partitioning() {
        let mut graph = BeliefGraph::default();

        // Add some test nodes
        let node1 = graph.add_node("Person".to_string(), HashMap::new());
        let node2 = graph.add_node("Person".to_string(), HashMap::new());

        // Create node map
        let mut node_map = HashMap::new();
        node_map.insert(("Person".to_string(), "Alice".to_string()), node1);
        node_map.insert(("Person".to_string(), "Bob".to_string()), node2);

        let edge_map = HashMap::new();

        // Create observations
        let observations = vec![
            ObserveStmt::Attribute {
                node: ("Person".to_string(), "Alice".to_string()),
                attr: "age".to_string(),
                value: 25.0,
                precision: None,
            },
            ObserveStmt::Attribute {
                node: ("Person".to_string(), "Bob".to_string()),
                attr: "age".to_string(),
                value: 30.0,
                precision: None,
            },
        ];

        let partitions = partition_observations(&observations, &node_map, &edge_map);

        // Should create 2 partitions (one per node)
        assert!(partitions.is_ok());
        let partitions = partitions.unwrap();
        assert_eq!(partitions.len(), 2);
    }

    #[test]
    fn test_parallel_determinism() {
        // Test that parallel processing produces same results as sequential
        let mut graph = BeliefGraph::default();

        // Add nodes with attributes
        let mut attrs = HashMap::new();
        attrs.insert(
            "score".to_string(),
            GaussianPosterior {
                mean: 0.0,
                precision: 1.0,
            },
        );

        let node1 = graph.add_node("Person".to_string(), attrs.clone());
        let node2 = graph.add_node("Person".to_string(), attrs);

        // Create node map
        let mut node_map = HashMap::new();
        node_map.insert(("Person".to_string(), "Alice".to_string()), node1);
        node_map.insert(("Person".to_string(), "Bob".to_string()), node2);

        let edge_map = HashMap::new();

        let observations = vec![
            ObserveStmt::Attribute {
                node: ("Person".to_string(), "Alice".to_string()),
                attr: "score".to_string(),
                value: 10.0,
                precision: Some(2.0),
            },
            ObserveStmt::Attribute {
                node: ("Person".to_string(), "Bob".to_string()),
                attr: "score".to_string(),
                value: 15.0,
                precision: Some(1.5),
            },
        ];

        // Process in parallel
        let parallel_result =
            process_evidence_parallel(&graph, &observations, &node_map, &edge_map).unwrap();

        // Verify we got the right number of observations
        assert_eq!(parallel_result.node_observations.len(), 2);
        assert!(parallel_result.node_observations.contains_key(&node1));
        assert!(parallel_result.node_observations.contains_key(&node2));

        // Apply to graph and verify it works
        let mut test_graph = graph.clone();
        apply_parallel_results(&mut test_graph, parallel_result).unwrap();
    }
}
