//! Parallel graph operations with fine-grained locking.
//!
//! This module provides thread-safe graph operations that allow multiple
//! readers and writers to work on different parts of the graph concurrently.
//!
//! ## Architecture
//!
//! - **Segment-based locking**: Graph divided into segments for parallel access
//! - **Read-write locks**: Multiple readers, exclusive writers per segment
//! - **Lock-free reads**: Common operations use lock-free data structures
//! - **Batch operations**: Amortize locking overhead with batched updates
//!
//! ## Feature gating
//!
//! Parallel graph operations are behind the `parallel` feature flag.

#[cfg(feature = "parallel")]
use rayon::prelude::*;
use std::collections::{HashMap, HashSet};
use std::sync::{Arc, Mutex, RwLock};

use crate::engine::errors::ExecError;
use crate::engine::graph::{EdgeId, GaussianPosterior, NodeId};

/// Number of segments to divide the graph into for parallel access.
const GRAPH_SEGMENTS: usize = 64;

/// A thread-safe wrapper around BeliefGraph with segment-based locking.
#[cfg(feature = "parallel")]
pub struct ParallelGraph {
    /// Segments of the graph, each with its own lock
    segments: Vec<Arc<RwLock<GraphSegment>>>,
    /// Global metadata protected by a single lock
    metadata: Arc<RwLock<GraphMetadata>>,
    /// Work queue for async operations
    work_queue: Arc<Mutex<WorkQueue>>,
}

/// A segment of the graph that can be locked independently.
#[derive(Debug, Default)]
struct GraphSegment {
    /// Nodes in this segment
    nodes: HashMap<NodeId, NodeData>,
    /// Edges in this segment
    edges: HashMap<EdgeId, EdgeData>,
}

/// Node data stored in a segment.
#[derive(Debug, Clone)]
pub struct NodeData {
    #[allow(dead_code)]
    node_type: String,
    pub attributes: HashMap<String, GaussianPosterior>,
}

/// Edge data stored in a segment.
#[derive(Debug, Clone)]
#[allow(dead_code)]
struct EdgeData {
    src: NodeId,
    dst: NodeId,
    edge_type: String,
    probability: f64,
}

/// Global graph metadata.
#[derive(Debug, Default)]
struct GraphMetadata {
    /// Total number of nodes
    node_count: usize,
    /// Total number of edges
    edge_count: usize,
    /// Node type index
    node_types: HashMap<String, HashSet<NodeId>>,
    /// Edge type index
    edge_types: HashMap<String, HashSet<EdgeId>>,
}

/// Work queue for async operations.
#[derive(Debug)]
struct WorkQueue {
    /// Pending operations
    pending: Vec<WorkItem>,
    /// Completed operations
    completed: Vec<WorkResult>,
}

/// A unit of work to be executed.
#[derive(Debug, Clone)]
pub enum WorkItem {
    #[allow(dead_code)]
    NodeUpdate {
        id: NodeId,
        attr: String,
        value: f64,
    },
    #[allow(dead_code)]
    EdgeUpdate { id: EdgeId, probability: f64 },
    #[allow(dead_code)]
    BatchObserve { observations: Vec<ObservationWork> },
}

/// Result of a work item execution.
#[derive(Debug, Clone)]
pub enum WorkResult {
    #[allow(dead_code)]
    Success { item: WorkItem },
    #[allow(dead_code)]
    Failure { item: WorkItem, error: String },
}

/// An observation to be processed.
#[derive(Debug, Clone)]
pub struct ObservationWork {
    pub target_id: NodeId,
    pub attr: String,
    pub value: f64,
    pub precision: f64,
}

#[cfg(feature = "parallel")]
impl ParallelGraph {
    /// Create a new parallel graph with the specified number of segments.
    pub fn new() -> Self {
        let mut segments = Vec::with_capacity(GRAPH_SEGMENTS);
        for _ in 0..GRAPH_SEGMENTS {
            segments.push(Arc::new(RwLock::new(GraphSegment::default())));
        }

        Self {
            segments,
            metadata: Arc::new(RwLock::new(GraphMetadata::default())),
            work_queue: Arc::new(Mutex::new(WorkQueue {
                pending: Vec::new(),
                completed: Vec::new(),
            })),
        }
    }

    /// Get the segment index for a node ID.
    fn node_segment(&self, id: NodeId) -> usize {
        (id.0 as usize) % GRAPH_SEGMENTS
    }

    /// Get the segment index for an edge ID.
    fn edge_segment(&self, id: EdgeId) -> usize {
        (id.0 as usize) % GRAPH_SEGMENTS
    }

    /// Add a node to the graph.
    pub fn add_node(
        &self,
        id: NodeId,
        node_type: String,
        attrs: HashMap<String, GaussianPosterior>,
    ) {
        let segment_idx = self.node_segment(id);
        let mut segment = self.segments[segment_idx].write().unwrap();

        segment.nodes.insert(
            id,
            NodeData {
                node_type: node_type.clone(),
                attributes: attrs,
            },
        );

        // Update metadata
        let mut metadata = self.metadata.write().unwrap();
        metadata.node_count += 1;
        metadata.node_types.entry(node_type).or_default().insert(id);
    }

    /// Add an edge to the graph.
    pub fn add_edge(&self, id: EdgeId, src: NodeId, dst: NodeId, edge_type: String, prob: f64) {
        let segment_idx = self.edge_segment(id);
        let mut segment = self.segments[segment_idx].write().unwrap();

        segment.edges.insert(
            id,
            EdgeData {
                src,
                dst,
                edge_type: edge_type.clone(),
                probability: prob,
            },
        );

        // Update metadata
        let mut metadata = self.metadata.write().unwrap();
        metadata.edge_count += 1;
        metadata.edge_types.entry(edge_type).or_default().insert(id);
    }

    /// Parallel node attribute update.
    ///
    /// Updates multiple node attributes in parallel across segments.
    pub fn update_attributes_parallel(&self, updates: Vec<(NodeId, String, f64, f64)>) {
        // Group updates by segment
        let mut updates_by_segment: HashMap<usize, Vec<_>> = HashMap::new();
        for (node_id, attr, value, precision) in updates {
            let segment_idx = self.node_segment(node_id);
            updates_by_segment
                .entry(segment_idx)
                .or_insert_with(Vec::new)
                .push((node_id, attr, value, precision));
        }

        // Process each segment in parallel
        updates_by_segment
            .into_par_iter()
            .for_each(|(segment_idx, segment_updates)| {
                let mut segment = self.segments[segment_idx].write().unwrap();

                for (node_id, attr, value, precision) in segment_updates {
                    if let Some(node) = segment.nodes.get_mut(&node_id) {
                        if let Some(posterior) = node.attributes.get_mut(&attr) {
                            // Bayesian update
                            let new_precision = posterior.precision + precision;
                            let new_mean = (posterior.mean * posterior.precision
                                + value * precision)
                                / new_precision;
                            *posterior = GaussianPosterior {
                                mean: new_mean,
                                precision: new_precision,
                            };
                        }
                    }
                }
            });
    }

    /// Parallel edge probability update.
    pub fn update_edges_parallel(&self, updates: Vec<(EdgeId, f64)>) {
        // Group updates by segment
        let mut updates_by_segment: HashMap<usize, Vec<_>> = HashMap::new();
        for (edge_id, prob) in updates {
            let segment_idx = self.edge_segment(edge_id);
            updates_by_segment
                .entry(segment_idx)
                .or_insert_with(Vec::new)
                .push((edge_id, prob));
        }

        // Process each segment in parallel
        updates_by_segment
            .into_par_iter()
            .for_each(|(segment_idx, segment_updates)| {
                let mut segment = self.segments[segment_idx].write().unwrap();

                for (edge_id, prob) in segment_updates {
                    if let Some(edge) = segment.edges.get_mut(&edge_id) {
                        edge.probability = prob;
                    }
                }
            });
    }

    /// Submit work items for async execution.
    pub fn submit_work(&self, items: Vec<WorkItem>) {
        let mut queue = self.work_queue.lock().unwrap();
        queue.pending.extend(items);
    }

    /// Process pending work items in parallel.
    pub fn process_work_queue(&self) -> Vec<WorkResult> {
        let items = {
            let mut queue = self.work_queue.lock().unwrap();
            std::mem::take(&mut queue.pending)
        };

        if items.is_empty() {
            return Vec::new();
        }

        // Process work items in parallel
        let results: Vec<WorkResult> = items
            .into_par_iter()
            .map(|item| match self.process_work_item(item.clone()) {
                Ok(()) => WorkResult::Success { item },
                Err(e) => WorkResult::Failure {
                    item,
                    error: e.to_string(),
                },
            })
            .collect();

        // Store results
        let mut queue = self.work_queue.lock().unwrap();
        queue.completed.extend(results.clone());

        results
    }

    /// Process a single work item.
    fn process_work_item(&self, item: WorkItem) -> Result<(), ExecError> {
        match item {
            WorkItem::NodeUpdate { id, attr, value } => {
                let segment_idx = self.node_segment(id);
                let mut segment = self.segments[segment_idx].write().unwrap();

                if let Some(node) = segment.nodes.get_mut(&id) {
                    if let Some(posterior) = node.attributes.get_mut(&attr) {
                        posterior.mean = value;
                    }
                }
                Ok(())
            }
            WorkItem::EdgeUpdate { id, probability } => {
                let segment_idx = self.edge_segment(id);
                let mut segment = self.segments[segment_idx].write().unwrap();

                if let Some(edge) = segment.edges.get_mut(&id) {
                    edge.probability = probability;
                }
                Ok(())
            }
            WorkItem::BatchObserve { observations } => {
                // Group by segment and process in parallel
                let mut obs_by_segment: HashMap<usize, Vec<_>> = HashMap::new();
                for obs in observations {
                    let segment_idx = self.node_segment(obs.target_id);
                    obs_by_segment
                        .entry(segment_idx)
                        .or_insert_with(Vec::new)
                        .push(obs);
                }

                for (segment_idx, segment_obs) in obs_by_segment {
                    let mut segment = self.segments[segment_idx].write().unwrap();
                    for obs in segment_obs {
                        if let Some(node) = segment.nodes.get_mut(&obs.target_id) {
                            if let Some(posterior) = node.attributes.get_mut(&obs.attr) {
                                // Bayesian update
                                let new_precision = posterior.precision + obs.precision;
                                let new_mean = (posterior.mean * posterior.precision
                                    + obs.value * obs.precision)
                                    / new_precision;
                                *posterior = GaussianPosterior {
                                    mean: new_mean,
                                    precision: new_precision,
                                };
                            }
                        }
                    }
                }
                Ok(())
            }
        }
    }

    /// Parallel graph traversal with visitor pattern.
    pub fn parallel_visit_nodes<F>(&self, visitor: F)
    where
        F: Fn(&NodeId, &NodeData) + Sync + Send,
    {
        self.segments.par_iter().for_each(|segment| {
            let segment = segment.read().unwrap();
            for (id, data) in &segment.nodes {
                visitor(id, data);
            }
        });
    }

    /// Map-reduce operation over nodes.
    pub fn map_reduce_nodes<M, R, T>(&self, mapper: M, reducer: R, initial: T) -> T
    where
        M: Fn(&NodeId, &NodeData) -> T + Sync + Send,
        R: Fn(T, T) -> T + Sync + Send,
        T: Send + Clone + Sync,
    {
        self.segments
            .par_iter()
            .map(|segment| {
                let segment = segment.read().unwrap();
                segment
                    .nodes
                    .iter()
                    .map(|(id, data)| mapper(id, data))
                    .fold(initial.clone(), &reducer)
            })
            .reduce(|| initial.clone(), &reducer)
    }

    /// Get statistics about the graph.
    pub fn stats(&self) -> GraphStats {
        let metadata = self.metadata.read().unwrap();

        // Calculate segment balance
        let node_distribution: Vec<usize> = self
            .segments
            .iter()
            .map(|s| s.read().unwrap().nodes.len())
            .collect();

        let edge_distribution: Vec<usize> = self
            .segments
            .iter()
            .map(|s| s.read().unwrap().edges.len())
            .collect();

        GraphStats {
            total_nodes: metadata.node_count,
            total_edges: metadata.edge_count,
            num_segments: GRAPH_SEGMENTS,
            node_distribution,
            edge_distribution,
            node_types: metadata.node_types.len(),
            edge_types: metadata.edge_types.len(),
        }
    }
}

#[cfg(feature = "parallel")]
impl Default for ParallelGraph {
    fn default() -> Self {
        Self::new()
    }
}

/// Statistics about the parallel graph.
#[derive(Debug)]
pub struct GraphStats {
    pub total_nodes: usize,
    pub total_edges: usize,
    pub num_segments: usize,
    pub node_distribution: Vec<usize>,
    pub edge_distribution: Vec<usize>,
    pub node_types: usize,
    pub edge_types: usize,
}

impl GraphStats {
    /// Calculate the imbalance factor for load distribution.
    pub fn imbalance_factor(&self) -> f64 {
        let avg_nodes = self.total_nodes as f64 / self.num_segments as f64;
        let max_nodes = *self.node_distribution.iter().max().unwrap_or(&0) as f64;

        if avg_nodes > 0.0 {
            max_nodes / avg_nodes
        } else {
            1.0
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[cfg(feature = "parallel")]
    #[test]
    fn test_parallel_graph_creation() {
        let graph = ParallelGraph::new();
        let stats = graph.stats();

        assert_eq!(stats.total_nodes, 0);
        assert_eq!(stats.total_edges, 0);
        assert_eq!(stats.num_segments, GRAPH_SEGMENTS);
    }

    #[cfg(feature = "parallel")]
    #[test]
    fn test_parallel_node_updates() {
        let graph = ParallelGraph::new();

        // Add some nodes
        for i in 0..100 {
            let mut attrs = HashMap::new();
            attrs.insert(
                "value".to_string(),
                GaussianPosterior {
                    mean: 0.0,
                    precision: 1.0,
                },
            );
            graph.add_node(NodeId(i), "TestNode".to_string(), attrs);
        }

        // Prepare parallel updates
        let updates: Vec<_> = (0..100)
            .map(|i| (NodeId(i), "value".to_string(), 10.0, 2.0))
            .collect();

        // Apply updates in parallel
        graph.update_attributes_parallel(updates);

        let stats = graph.stats();
        assert_eq!(stats.total_nodes, 100);
    }

    #[cfg(feature = "parallel")]
    #[test]
    fn test_work_queue() {
        let graph = ParallelGraph::new();

        // Add a node
        let mut attrs = HashMap::new();
        attrs.insert(
            "score".to_string(),
            GaussianPosterior {
                mean: 5.0,
                precision: 1.0,
            },
        );
        graph.add_node(NodeId(0), "TestNode".to_string(), attrs);

        // Submit work
        let work = vec![WorkItem::NodeUpdate {
            id: NodeId(0),
            attr: "score".to_string(),
            value: 15.0,
        }];
        graph.submit_work(work);

        // Process work
        let results = graph.process_work_queue();
        assert_eq!(results.len(), 1);

        match &results[0] {
            WorkResult::Success { .. } => {}
            WorkResult::Failure { error, .. } => panic!("Work failed: {}", error),
        }
    }

    #[cfg(feature = "parallel")]
    #[test]
    fn test_map_reduce() {
        let graph = ParallelGraph::new();

        // Add nodes
        for i in 0..10 {
            let mut attrs = HashMap::new();
            attrs.insert(
                "value".to_string(),
                GaussianPosterior {
                    mean: i as f64,
                    precision: 1.0,
                },
            );
            graph.add_node(NodeId(i), "TestNode".to_string(), attrs);
        }

        // Count nodes using map-reduce
        let count = graph.map_reduce_nodes(|_id, _data| 1, |a, b| a + b, 0);

        assert_eq!(count, 10);
    }
}
