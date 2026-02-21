//! Enhanced adjacency index with incremental updates and lazy invalidation.
//!
//! This module implements Phase 13 optimizations for graph adjacency queries,
//! providing efficient neighborhood lookups with incremental maintenance.

use rustc_hash::FxHashMap;
use smallvec::SmallVec;
use std::collections::HashSet;
use std::sync::Arc;

use super::graph::{EdgeData, EdgeId, NodeId};

/// Version number for tracking index updates
type Version = u64;

/// Threshold for triggering full index rebuild
const DELTA_REBUILD_THRESHOLD: usize = 1000;

/// Maximum size for inline storage in SmallVec
const INLINE_VEC_SIZE: usize = 8;

/// Enhanced adjacency index with incremental updates and versioning.
///
/// Features:
/// - Incremental index maintenance for small deltas
/// - Lazy invalidation with versioning
/// - Sparse per-node delta indexes
/// - Automatic rebuild heuristics for large deltas
#[derive(Debug, Clone)]
pub struct EnhancedAdjacencyIndex {
    /// Primary adjacency data: (NodeId, EdgeType) -> list of edges
    /// Using SmallVec for inline storage of small edge lists
    primary: FxHashMap<(NodeId, Arc<str>), SmallVec<[EdgeId; INLINE_VEC_SIZE]>>,

    /// Reverse index: EdgeId -> (source, target) for fast edge lookups
    reverse: FxHashMap<EdgeId, (NodeId, NodeId)>,

    /// Per-node delta indexes for incremental updates
    /// Tracks added/removed edges since last full rebuild
    node_deltas: FxHashMap<NodeId, NodeDelta>,

    /// Version counter for lazy invalidation
    version: Version,

    /// Node versions for tracking staleness
    node_versions: FxHashMap<NodeId, Version>,

    /// Number of pending delta operations
    pending_delta_count: usize,

    /// Statistics for monitoring performance
    stats: IndexStats,
}

/// Delta changes for a single node
#[derive(Debug, Clone, Default)]
struct NodeDelta {
    /// Edges added since last rebuild
    _added: SmallVec<[(Arc<str>, EdgeId); 4]>,
    /// Edges removed since last rebuild
    removed: SmallVec<[(Arc<str>, EdgeId); 4]>,
    /// Version when delta was created
    version: Version,
}

/// Statistics for index performance monitoring
#[derive(Debug, Clone, Default)]
pub struct IndexStats {
    /// Number of full rebuilds performed
    pub full_rebuilds: usize,
    /// Number of incremental updates
    pub incremental_updates: usize,
    /// Number of lazy invalidations
    pub invalidations: usize,
    /// Average delta size at rebuild
    pub avg_delta_size: f64,
}

impl EnhancedAdjacencyIndex {
    /// Creates a new empty adjacency index.
    pub fn new() -> Self {
        Self {
            primary: FxHashMap::default(),
            reverse: FxHashMap::default(),
            node_deltas: FxHashMap::default(),
            version: 0,
            node_versions: FxHashMap::default(),
            pending_delta_count: 0,
            stats: IndexStats::default(),
        }
    }

    /// Builds the index from a collection of edges (full rebuild).
    pub fn build_from_edges(&mut self, edges: &[(EdgeId, &EdgeData)]) {
        // Clear existing data
        self.primary.clear();
        self.reverse.clear();
        self.node_deltas.clear();

        // Build primary and reverse indexes
        for (edge_id, edge_data) in edges {
            let key = (edge_data.src, edge_data.ty.clone());
            self.primary.entry(key).or_default().push(*edge_id);

            self.reverse
                .insert(*edge_id, (edge_data.src, edge_data.dst));
        }

        // Update version and stats
        self.version += 1;
        self.pending_delta_count = 0;
        self.stats.full_rebuilds += 1;

        // Update all node versions
        for (node_id, _) in self.primary.keys() {
            self.node_versions.insert(*node_id, self.version);
        }
    }

    /// Incrementally adds an edge to the index.
    pub fn add_edge(&mut self, edge_id: EdgeId, edge_data: &EdgeData) {
        // Add to primary index immediately for fast queries
        let key = (edge_data.src, edge_data.ty.clone());
        self.primary.entry(key.clone()).or_default().push(edge_id);

        // Add to reverse index
        self.reverse.insert(edge_id, (edge_data.src, edge_data.dst));

        // Don't record in delta since we've already updated the primary index
        // Delta is only for tracking pending changes not yet in primary

        // Update counters
        self.pending_delta_count += 1;
        self.stats.incremental_updates += 1;

        // Check if rebuild is needed
        if self.pending_delta_count > DELTA_REBUILD_THRESHOLD {
            self.trigger_rebuild();
        }
    }

    /// Incrementally removes an edge from the index.
    pub fn remove_edge(&mut self, edge_id: EdgeId) -> bool {
        // Look up edge info from reverse index
        let Some((src, _dst)) = self.reverse.remove(&edge_id) else {
            return false;
        };

        // Find and remove from primary index
        // This is O(n) for the edge list, but lists are typically small
        let mut removed = false;
        self.primary.retain(|(node, _edge_type), edges| {
            if *node == src {
                let old_len = edges.len();
                edges.retain(|e| *e != edge_id);
                if edges.len() < old_len {
                    removed = true;

                    // Record in delta
                    let delta = self.node_deltas.entry(src).or_default();
                    delta.removed.push((Arc::from(""), edge_id)); // Edge type not critical for removal
                    delta.version = self.version;
                }
            }
            !edges.is_empty() // Remove empty entries
        });

        if removed {
            self.pending_delta_count += 1;
            self.stats.incremental_updates += 1;

            // Check if rebuild is needed
            if self.pending_delta_count > DELTA_REBUILD_THRESHOLD {
                self.trigger_rebuild();
            }
        }

        removed
    }

    /// Gets edges for a node and edge type, applying deltas if needed.
    pub fn get_edges(&self, node: NodeId, edge_type: &str) -> Vec<EdgeId> {
        let key = (node, Arc::from(edge_type));

        // Get base edges
        let mut edges = self
            .primary
            .get(&key)
            .map(|v| v.to_vec())
            .unwrap_or_default();

        // Apply delta if exists - only remove deleted edges since added are already in primary
        if let Some(delta) = self.node_deltas.get(&node) {
            // Remove edges marked for deletion
            let removed: HashSet<EdgeId> = delta
                .removed
                .iter()
                .filter(|(et, _)| et.is_empty() || &**et == edge_type)
                .map(|(_, e)| *e)
                .collect();
            edges.retain(|e| !removed.contains(e));
        }

        edges
    }

    /// Invalidates the index for a specific node (lazy invalidation).
    pub fn invalidate_node(&mut self, node: NodeId) {
        self.version += 1;
        self.node_versions.insert(node, 0); // Mark as invalid
        self.stats.invalidations += 1;
    }

    /// Checks if a node's index is stale.
    pub fn is_node_stale(&self, node: NodeId) -> bool {
        self.node_versions
            .get(&node)
            .map(|&v| v < self.version)
            .unwrap_or(true)
    }

    /// Gets the number of pending delta operations.
    pub fn pending_deltas(&self) -> usize {
        self.pending_delta_count
    }

    /// Forces a full rebuild of the index.
    pub fn force_rebuild(&mut self, edges: &[(EdgeId, &EdgeData)]) {
        let avg_delta = if self.stats.full_rebuilds > 0 {
            (self.stats.avg_delta_size * self.stats.full_rebuilds as f64
                + self.pending_delta_count as f64)
                / (self.stats.full_rebuilds + 1) as f64
        } else {
            self.pending_delta_count as f64
        };

        self.stats.avg_delta_size = avg_delta;
        self.build_from_edges(edges);
    }

    /// Internal method to trigger rebuild when threshold is reached.
    fn trigger_rebuild(&mut self) {
        // In a real implementation, this would coordinate with the graph
        // to get all edges and rebuild. For now, we just reset the counter.
        // The actual rebuild would be triggered by the graph when it notices
        // the high delta count.

        // Log that rebuild is needed
        #[cfg(feature = "tracing")]
        tracing::info!(
            "Adjacency index rebuild needed: {} pending deltas",
            self.pending_delta_count
        );
    }

    /// Gets statistics about the index.
    pub fn stats(&self) -> &IndexStats {
        &self.stats
    }

    /// Gets the degree of a node (number of outgoing edges).
    pub fn out_degree(&self, node: NodeId) -> usize {
        let mut degree = 0;

        // Count edges in primary index
        for ((n, _), edges) in &self.primary {
            if *n == node {
                degree += edges.len();
            }
        }

        // Apply deltas - only subtract removed edges since added are already in primary
        if let Some(delta) = self.node_deltas.get(&node) {
            degree = degree.saturating_sub(delta.removed.len());
        }

        degree
    }

    /// Gets all neighbors of a node (targets of outgoing edges).
    pub fn get_neighbors(&self, node: NodeId) -> HashSet<NodeId> {
        let mut neighbors = HashSet::new();

        // Collect from primary index
        for ((n, _), edges) in &self.primary {
            if *n == node {
                for edge_id in edges {
                    if let Some((_, dst)) = self.reverse.get(edge_id) {
                        neighbors.insert(*dst);
                    }
                }
            }
        }

        // Apply deltas - only remove deleted edges since added are already in primary
        if let Some(delta) = self.node_deltas.get(&node) {
            // Remove neighbors from deleted edges
            for (_, edge_id) in &delta.removed {
                if let Some((_, dst)) = self.reverse.get(edge_id) {
                    neighbors.remove(dst);
                }
            }
        }

        neighbors
    }
}

impl Default for EnhancedAdjacencyIndex {
    fn default() -> Self {
        Self::new()
    }
}

/// Specialized adjacency structure for dense graphs.
///
/// Uses a more compact representation for graphs with high edge density.
pub struct DenseAdjacencyIndex {
    /// Bit matrix representation for edge existence
    /// Row = source node, Column = destination node
    adjacency_matrix: Vec<Vec<bool>>,
    /// Node ID to matrix index mapping
    node_to_index: FxHashMap<NodeId, usize>,
    /// Index to node ID mapping
    index_to_node: Vec<NodeId>,
    /// Edge type filtering
    edge_types: HashSet<Arc<str>>,
}

impl DenseAdjacencyIndex {
    /// Creates a new dense adjacency index for a fixed set of nodes.
    pub fn new(nodes: &[NodeId]) -> Self {
        let n = nodes.len();
        let mut node_to_index = FxHashMap::default();
        let mut index_to_node = Vec::with_capacity(n);

        for (i, &node) in nodes.iter().enumerate() {
            node_to_index.insert(node, i);
            index_to_node.push(node);
        }

        Self {
            adjacency_matrix: vec![vec![false; n]; n],
            node_to_index,
            index_to_node,
            edge_types: HashSet::new(),
        }
    }

    /// Adds an edge to the dense index.
    pub fn add_edge(&mut self, src: NodeId, dst: NodeId, edge_type: &str) -> bool {
        self.edge_types.insert(Arc::from(edge_type));

        if let (Some(&i), Some(&j)) = (self.node_to_index.get(&src), self.node_to_index.get(&dst)) {
            if !self.adjacency_matrix[i][j] {
                self.adjacency_matrix[i][j] = true;
                return true;
            }
        }
        false
    }

    /// Checks if an edge exists.
    pub fn has_edge(&self, src: NodeId, dst: NodeId) -> bool {
        if let (Some(&i), Some(&j)) = (self.node_to_index.get(&src), self.node_to_index.get(&dst)) {
            self.adjacency_matrix[i][j]
        } else {
            false
        }
    }

    /// Gets all neighbors of a node.
    pub fn get_neighbors(&self, node: NodeId) -> Vec<NodeId> {
        if let Some(&i) = self.node_to_index.get(&node) {
            self.adjacency_matrix[i]
                .iter()
                .enumerate()
                .filter_map(|(j, &has_edge)| {
                    if has_edge {
                        Some(self.index_to_node[j])
                    } else {
                        None
                    }
                })
                .collect()
        } else {
            Vec::new()
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_incremental_updates() {
        let mut index = EnhancedAdjacencyIndex::new();

        // Add some edges
        let edge1 = EdgeId(1);
        let _edge2 = EdgeId(2);
        let node1 = NodeId(1);
        let node2 = NodeId(2);

        let edge_data1 = EdgeData {
            id: edge1,
            src: node1,
            dst: node2,
            ty: Arc::from("knows"),
            exist: Default::default(),
        };

        index.add_edge(edge1, &edge_data1);

        // Check edge retrieval
        let edges = index.get_edges(node1, "knows");
        assert_eq!(edges.len(), 1);
        assert_eq!(edges[0], edge1);

        // Check neighbors
        let neighbors = index.get_neighbors(node1);
        assert!(neighbors.contains(&node2));
    }

    #[test]
    fn test_lazy_invalidation() {
        let mut index = EnhancedAdjacencyIndex::new();
        let node = NodeId(1);

        assert!(index.is_node_stale(node));

        index.node_versions.insert(node, index.version);
        assert!(!index.is_node_stale(node));

        index.invalidate_node(node);
        assert!(index.is_node_stale(node));
    }

    #[test]
    fn test_dense_index() {
        let nodes = vec![NodeId(1), NodeId(2), NodeId(3)];
        let mut index = DenseAdjacencyIndex::new(&nodes);

        index.add_edge(NodeId(1), NodeId(2), "knows");
        index.add_edge(NodeId(2), NodeId(3), "knows");

        assert!(index.has_edge(NodeId(1), NodeId(2)));
        assert!(!index.has_edge(NodeId(1), NodeId(3)));

        let neighbors = index.get_neighbors(NodeId(1));
        assert_eq!(neighbors.len(), 1);
        assert!(neighbors.contains(&NodeId(2)));
    }
}
