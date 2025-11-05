//! # Bayesian Belief Graph
//!
//! This module implements the core Bayesian belief graph data structure and operations.
//!
//! ## Key Components
//!
//! - **GaussianPosterior**: Maintains Bayesian posterior over continuous node attributes
//!   using conjugate Normal-Normal updates (mean and precision τ = 1/σ²)
//!
//! - **BetaPosterior**: Maintains Bayesian posterior over edge existence using
//!   conjugate Beta-Binomial updates (α, β parameters)
//!
//! - **BeliefGraph**: Main graph structure with nodes (typed entities with attributes)
//!   and edges (relationships with existence probability)
//!
//! ## Design
//!
//! The implementation follows the Bayesian inference specifications in baygraph_design.md:
//! - Conjugate priors for efficient updates
//! - Force operations for hard constraints (e.g., force_absent, force_value)
//! - O(1) node/edge lookups via HashMap indexes
//!
//! ## Example
//!
//! ```rust,ignore
//! use baygraph::engine::graph::*;
//! use std::collections::HashMap;
//!
//! let mut graph = BeliefGraph::default();
//! let node_id = graph.add_node("Person".into(), HashMap::new());
//! let edge_id = graph.add_edge(node_id, node_id, "KNOWS".into(),
//!     BetaPosterior { alpha: 1.0, beta: 1.0 });
//! ```

use std::collections::HashMap;

use crate::engine::errors::ExecError;

// Bayesian inference constants per baygraph_design.md
/// High precision value for force operations (baygraph_design.md:107, 133-135)
const FORCE_PRECISION: f64 = 1_000_000.0;

/// Minimum precision for Gaussian posteriors (baygraph_design.md:203)
const MIN_PRECISION: f64 = 1e-6;

/// Minimum observation precision to prevent numerical issues
const MIN_OBS_PRECISION: f64 = 1e-12;

/// Minimum Beta parameter value to enforce proper prior (baygraph_design.md:223)
/// Beta distribution requires α > 0 and β > 0 strictly. We enforce α ≥ 0.01, β ≥ 0.01
/// as a numeric floor for stability and to prevent improper priors.
const MIN_BETA_PARAM: f64 = 0.01;

/// Maximum allowed deviation in standard deviations for outlier warnings
/// See baygraph_design.md:211-213
const OUTLIER_THRESHOLD_SIGMA: f64 = 10.0;

/// A unique identifier for a node in the belief graph.
///
/// See baygraph_design.md:471-486 for stable identifier requirements.
#[repr(transparent)]
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Ord, PartialOrd)]
#[cfg_attr(feature = "serde", derive(serde::Serialize, serde::Deserialize))]
pub struct NodeId(pub u32);

/// A unique identifier for an edge in the belief graph.
///
/// See baygraph_design.md:471-486 for stable identifier requirements.
#[repr(transparent)]
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Ord, PartialOrd)]
#[cfg_attr(feature = "serde", derive(serde::Serialize, serde::Deserialize))]
pub struct EdgeId(pub u32);

/// A Gaussian posterior distribution for continuous node attributes.
///
/// Uses the precision parameterization (τ = 1/σ²) for efficient conjugate updates.
/// See baygraph_design.md sections 93-101 for the update equations.
#[derive(Debug, Clone)]
#[cfg_attr(feature = "serde", derive(serde::Serialize, serde::Deserialize))]
pub struct GaussianPosterior {
    /// The posterior mean (μ)
    pub mean: f64,
    /// The posterior precision (τ = 1/σ²)
    pub precision: f64,
}

impl GaussianPosterior {
    /// Updates the expected value without changing precision.
    ///
    /// Soft update that adjusts the mean without increasing certainty.
    /// See baygraph_design.md:275-276.
    pub fn set_expectation(&mut self, v: f64) {
        self.mean = v;
    }

    /// Performs a Bayesian update with a new observation.
    ///
    /// Uses Normal-Normal conjugate update formulas:
    /// ```text
    /// τ_new = τ_old + τ_obs
    /// μ_new = (τ_old * μ_old + τ_obs * x) / τ_new
    /// ```
    ///
    /// # Arguments
    ///
    /// * `x` - The observed value
    /// * `tau_obs` - The observation precision (1/σ²_obs)
    ///
    /// See baygraph_design.md:93-101.
    pub fn update(&mut self, x: f64, tau_obs: f64) {
        let tau_old = self.precision;
        let tau_obs = tau_obs.max(MIN_OBS_PRECISION);
        let tau_new = (tau_old + tau_obs).max(MIN_PRECISION);
        let mu_num = tau_old * self.mean + tau_obs * x;
        let mu_new = mu_num / tau_new;
        self.mean = mu_new;
        self.precision = tau_new;
    }

    /// Forces the attribute to a specific value with very high certainty.
    ///
    /// Sets precision to FORCE_PRECISION (1e6), effectively creating a hard constraint.
    /// See baygraph_design.md:107.
    ///
    /// # Arguments
    ///
    /// * `x` - The value to force
    pub fn force_value(&mut self, x: f64) {
        self.mean = x;
        self.precision = FORCE_PRECISION;
    }
}

/// A Beta posterior distribution for edge existence probability.
///
/// Uses the Beta(α, β) parameterization for conjugate Bernoulli updates.
/// The mean probability is α/(α+β). See baygraph_design.md:127-135.
#[derive(Debug, Clone)]
#[cfg_attr(feature = "serde", derive(serde::Serialize, serde::Deserialize))]
pub struct BetaPosterior {
    /// The alpha parameter (pseudo-count of successes)
    pub alpha: f64,
    /// The beta parameter (pseudo-count of failures)
    pub beta: f64,
}

impl BetaPosterior {
    /// Forces the edge to be absent (α=1, β=1e6).
    ///
    /// See baygraph_design.md:133-135.
    pub fn force_absent(&mut self) {
        self.alpha = 1.0;
        self.beta = FORCE_PRECISION;
    }

    /// Forces the edge to be present (α=1e6, β=1).
    ///
    /// See baygraph_design.md:133-135.
    pub fn force_present(&mut self) {
        self.alpha = FORCE_PRECISION;
        self.beta = 1.0;
    }

    /// Performs a conjugate Beta-Bernoulli update.
    ///
    /// Increments α if present, β if absent. See baygraph_design.md:127.
    ///
    /// # Arguments
    ///
    /// * `present` - Whether the edge was observed to be present
    pub fn observe(&mut self, present: bool) {
        if present { self.alpha += 1.0; } else { self.beta += 1.0; }
    }

    /// Computes the posterior mean probability E[p] = α / (α + β).
    ///
    /// Applies MIN_BETA_PARAM floor to both parameters for numerical stability.
    /// This ensures consistent probability calculations across the codebase.
    pub fn mean_probability(&self) -> f64 {
        let a = self.alpha.max(MIN_BETA_PARAM);
        let b = self.beta.max(MIN_BETA_PARAM);
        a / (a + b)
    }
}

/// A node in the belief graph with typed attributes.
#[derive(Debug, Clone)]
#[cfg_attr(feature = "serde", derive(serde::Serialize, serde::Deserialize))]
pub struct NodeData {
    /// The unique node identifier
    pub id: NodeId,
    /// The node type label (e.g., "Person", "Company")
    pub label: String,
    /// Gaussian posteriors for continuous attributes
    pub attrs: HashMap<String, GaussianPosterior>,
}

/// A directed edge in the belief graph with existence probability.
#[derive(Debug, Clone)]
#[cfg_attr(feature = "serde", derive(serde::Serialize, serde::Deserialize))]
pub struct EdgeData {
    /// The unique edge identifier
    pub id: EdgeId,
    /// The source node ID
    pub src: NodeId,
    /// The destination node ID
    pub dst: NodeId,
    /// The edge type (e.g., "KNOWS", "LIKES")
    pub ty: String,
    /// Beta posterior for edge existence probability
    pub exist: BetaPosterior,
}

/// Optimized adjacency list for fast neighborhood queries.
///
/// Uses offset-based indexing into a contiguous edge ID array for O(1) access.
/// This structure is precomputed and cached for performance.
///
/// See baygraph_design.md:466-474 for the adjacency optimization strategy.
#[derive(Debug, Clone)]
pub struct AdjacencyIndex {
    /// Maps (NodeId, EdgeType) to (start_offset, end_offset) in edge_ids
    ranges: HashMap<(NodeId, String), (usize, usize)>,
    /// Contiguous sorted array of EdgeIds
    edge_ids: Vec<EdgeId>,
}

impl AdjacencyIndex {
    /// Creates a new empty adjacency index.
    pub fn new() -> Self {
        Self {
            ranges: HashMap::new(),
            edge_ids: Vec::new(),
        }
    }

    /// Builds an adjacency index from edges.
    ///
    /// Groups edges by (src, type) and stores them in sorted order for determinism.
    pub fn from_edges(edges: &[EdgeData]) -> Self {
        let mut map: HashMap<(NodeId, String), Vec<EdgeId>> = HashMap::new();
        for e in edges {
            map.entry((e.src, e.ty.clone())).or_default().push(e.id);
        }

        // Sort each adjacency list for determinism
        for v in map.values_mut() {
            v.sort();
        }

        // Flatten into contiguous storage with offset ranges
        let mut edge_ids = Vec::new();
        let mut ranges = HashMap::new();
        for (key, ids) in map {
            let start = edge_ids.len();
            edge_ids.extend_from_slice(&ids);
            let end = edge_ids.len();
            ranges.insert(key, (start, end));
        }

        Self { ranges, edge_ids }
    }

    /// Gets outgoing edge IDs for a node and edge type.
    ///
    /// Returns an empty slice if no edges exist.
    pub fn get_edges(&self, node: NodeId, edge_type: &str) -> &[EdgeId] {
        if let Some(&(start, end)) = self.ranges.get(&(node, edge_type.to_string())) {
            &self.edge_ids[start..end]
        } else {
            &[]
        }
    }

    /// Gets all outgoing edge IDs for a node across all edge types.
    pub fn get_all_edges(&self, node: NodeId) -> Vec<EdgeId> {
        let mut result = Vec::new();
        for ((n, _), (start, end)) in &self.ranges {
            if *n == node {
                result.extend_from_slice(&self.edge_ids[*start..*end]);
            }
        }
        result.sort(); // Maintain deterministic order
        result
    }
}

/// A belief graph with Bayesian inference over nodes and edges.
///
/// This is the core data structure for Baygraph, maintaining:
/// - Nodes with continuous-valued attributes (Gaussian posteriors)
/// - Directed edges with existence probabilities (Beta posteriors)
/// - O(1) lookup indexes for efficient access
/// - Optimized adjacency structure for fast neighborhood queries
///
/// All graph modifications preserve Bayesian consistency through
/// conjugate prior updates and force operations.
///
/// # Performance (Phase 7)
///
/// The graph uses Structure-of-Arrays (SoA) style organization for hot data:
/// - Contiguous node/edge storage for cache efficiency
/// - Precomputed adjacency index with offset ranges for O(1) neighborhood access
/// - Stable IDs (NodeId, EdgeId) for deterministic iteration
///
/// See baygraph_design.md:466-474 for optimization details.
#[derive(Debug, Clone)]
pub struct BeliefGraph {
    /// All nodes in the graph
    pub nodes: Vec<NodeData>,
    /// All edges in the graph
    pub edges: Vec<EdgeData>,
    /// Index mapping NodeId to position in nodes vector
    node_index: HashMap<NodeId, usize>,
    /// Index mapping EdgeId to position in edges vector
    edge_index: HashMap<EdgeId, usize>,
    /// Cached adjacency index for fast neighborhood queries (Phase 7)
    adjacency: Option<AdjacencyIndex>,
}

impl Default for BeliefGraph {
    fn default() -> Self {
        Self {
            nodes: Vec::new(),
            edges: Vec::new(),
            node_index: HashMap::new(),
            edge_index: HashMap::new(),
            adjacency: None,
        }
    }
}

impl BeliefGraph {
    /// Looks up a node by ID.
    ///
    /// # Arguments
    ///
    /// * `id` - The node ID to look up
    ///
    /// # Returns
    ///
    /// * `Some(&NodeData)` - The node if it exists
    /// * `None` - If the node ID is not in the graph
    pub fn node(&self, id: NodeId) -> Option<&NodeData> {
        self.node_index.get(&id).and_then(|&idx| self.nodes.get(idx))
    }

    /// Looks up a node by ID with mutable access.
    pub fn node_mut(&mut self, id: NodeId) -> Option<&mut NodeData> {
        self.node_index.get(&id).and_then(|&idx| self.nodes.get_mut(idx))
    }

    /// Looks up an edge by ID.
    pub fn edge(&self, id: EdgeId) -> Option<&EdgeData> {
        self.edge_index.get(&id).and_then(|&idx| self.edges.get(idx))
    }

    /// Looks up an edge by ID with mutable access.
    pub fn edge_mut(&mut self, id: EdgeId) -> Option<&mut EdgeData> {
        self.edge_index.get(&id).and_then(|&idx| self.edges.get_mut(idx))
    }

    /// Adds a new node to the graph and returns its ID.
    ///
    /// The node ID is assigned automatically based on the current node count.
    ///
    /// # Arguments
    ///
    /// * `label` - The node type label
    /// * `attrs` - Initial Gaussian posteriors for node attributes
    ///
    /// # Returns
    ///
    /// The newly assigned NodeId
    pub fn add_node(&mut self, label: String, attrs: HashMap<String, GaussianPosterior>) -> NodeId {
        let id = NodeId(self.nodes.len() as u32);
        let idx = self.nodes.len();
        self.nodes.push(NodeData { id, label, attrs });
        self.node_index.insert(id, idx);
        id
    }

    /// Add an edge and update the index. Returns the EdgeId.
    pub fn add_edge(&mut self, src: NodeId, dst: NodeId, ty: String, exist: BetaPosterior) -> EdgeId {
        let id = EdgeId(self.edges.len() as u32);
        let idx = self.edges.len();
        self.edges.push(EdgeData { id, src, dst, ty, exist });
        self.edge_index.insert(id, idx);
        id
    }

    /// Internal helper to add a node with a specific ID and update the index.
    /// Used for testing and deserialization. Caller must ensure ID uniqueness.
    ///
    /// # Warning
    /// This is an internal API and should not be used in production code.
    /// Use `add_node()` instead for normal usage.
    pub fn insert_node(&mut self, node: NodeData) {
        let idx = self.nodes.len();
        self.node_index.insert(node.id, idx);
        self.nodes.push(node);
    }

    /// Internal helper to add an edge with a specific ID and update the index.
    /// Used for testing and deserialization. Caller must ensure ID uniqueness.
    ///
    /// # Warning
    /// This is an internal API and should not be used in production code.
    /// Use `add_edge()` instead for normal usage.
    pub fn insert_edge(&mut self, edge: EdgeData) {
        let idx = self.edges.len();
        self.edge_index.insert(edge.id, idx);
        self.edges.push(edge);
    }

    /// Rebuilds a graph from a subset of nodes and edges.
    ///
    /// Used by graph transformations (e.g., prune_edges) to create a new graph
    /// with filtered elements.
    ///
    /// # Arguments
    ///
    /// * `nodes` - All nodes to include in the rebuilt graph
    /// * `edge_ids` - Edge IDs to include (must exist in the original graph)
    ///
    /// # Returns
    ///
    /// * `Ok(BeliefGraph)` - The rebuilt graph
    /// * `Err(ExecError)` - If any edge ID doesn't exist
    pub fn rebuild_with_edges(&self, nodes: &[NodeData], edge_ids: &[EdgeId]) -> Result<Self, ExecError> {
        let mut rebuilt = BeliefGraph::default();
        for node in nodes {
            rebuilt.insert_node(node.clone());
        }
        for eid in edge_ids {
            let e = self
                .edge(*eid)
                .ok_or_else(|| ExecError::Internal("missing edge during rebuild".into()))?
                .clone();
            rebuilt.insert_edge(e);
        }
        Ok(rebuilt)
    }

    pub fn expectation(&self, node: NodeId, attr: &str) -> Result<f64, ExecError> {
        let n = self.node(node).ok_or_else(|| ExecError::Internal("missing node".into()))?;
        let g = n
            .attrs
            .get(attr)
            .ok_or_else(|| ExecError::Internal(format!("missing attr '{}'", attr)))?;
        Ok(g.mean)
    }

    pub fn set_expectation(&mut self, node: NodeId, attr: &str, value: f64) -> Result<(), ExecError> {
        let n = self
            .node_mut(node)
            .ok_or_else(|| ExecError::Internal("missing node".into()))?;
        let g = n
            .attrs
            .get_mut(attr)
            .ok_or_else(|| ExecError::Internal(format!("missing attr '{}'", attr)))?;
        g.set_expectation(value);
        Ok(())
    }

    pub fn force_absent(&mut self, edge: EdgeId) -> Result<(), ExecError> {
        let e = self
            .edge_mut(edge)
            .ok_or_else(|| ExecError::Internal("missing edge".into()))?;
        e.exist.force_absent();
        Ok(())
    }

    /// Computes the posterior mean probability for an edge's existence.
    ///
    /// Returns the expected value of the Beta posterior: E[p] = α / (α + β).
    /// This is the Bayesian point estimate under squared error loss.
    ///
    /// # Arguments
    ///
    /// * `edge` - The edge ID to query
    ///
    /// # Returns
    ///
    /// * `Ok(f64)` - The posterior mean probability in [0, 1]
    /// * `Err(ExecError)` - If edge doesn't exist
    pub fn prob_mean(&self, edge: EdgeId) -> Result<f64, ExecError> {
        let e = self
            .edge(edge)
            .ok_or_else(|| ExecError::Internal("missing edge".into()))?;
        Ok(e.exist.mean_probability())
    }

    /// Counts outgoing edges from a node that meet a minimum probability threshold.
    ///
    /// # Arguments
    ///
    /// * `node` - The source node ID
    /// * `min_prob` - Minimum probability threshold in [0, 1]
    ///
    /// # Returns
    ///
    /// Number of outgoing edges meeting the probability threshold
    pub fn degree_outgoing(&self, node: NodeId, min_prob: f64) -> usize {
        self.edges
            .iter()
            .filter(|e| e.src == node && e.exist.mean_probability() >= min_prob)
            .count()
    }

    /// Builds or rebuilds the adjacency index for fast neighborhood queries.
    ///
    /// Call after bulk graph modifications. The index is cached until invalidated.
    /// Building is O(E log E) due to sorting; subsequent queries are O(1).
    pub fn build_adjacency(&mut self) {
        self.adjacency = Some(AdjacencyIndex::from_edges(&self.edges));
    }

    /// Ensures adjacency index is built, building it lazily if needed.
    fn ensure_adjacency(&mut self) {
        if self.adjacency.is_none() {
            self.build_adjacency();
        }
    }

    /// Gets outgoing edges for a node and edge type.
    ///
    /// Builds the adjacency index lazily if not already built. Returns edge IDs
    /// in deterministic sorted order.
    pub fn get_outgoing_edges(&mut self, node: NodeId, edge_type: &str) -> Vec<EdgeId> {
        self.ensure_adjacency();
        self.adjacency
            .as_ref()
            .unwrap()
            .get_edges(node, edge_type)
            .to_vec()
    }

    pub fn adjacency_outgoing_by_type(&self) -> std::collections::HashMap<(NodeId, String), Vec<EdgeId>> {
        // Build on demand. Could use cached adjacency index if available.
        let mut map: std::collections::HashMap<(NodeId, String), Vec<EdgeId>> = std::collections::HashMap::new();
        for e in &self.edges {
            map.entry((e.src, e.ty.clone())).or_default().push(e.id);
        }
        // Keep deterministic order by stable EdgeId
        for v in map.values_mut() { v.sort(); }
        map
    }

    pub fn observe_edge(&mut self, edge: EdgeId, present: bool) -> Result<(), ExecError> {
        let e = self
            .edge_mut(edge)
            .ok_or_else(|| ExecError::Internal("missing edge".into()))?;
        e.exist.observe(present);
        Ok(())
    }

    pub fn force_edge_present(&mut self, edge: EdgeId) -> Result<(), ExecError> {
        let e = self
            .edge_mut(edge)
            .ok_or_else(|| ExecError::Internal("missing edge".into()))?;
        e.exist.force_present();
        Ok(())
    }

    pub fn observe_attr(&mut self, node: NodeId, attr: &str, x: f64, tau_obs: f64) -> Result<(), ExecError> {
        let n = self
            .node_mut(node)
            .ok_or_else(|| ExecError::Internal("missing node".into()))?;
        let g = n
            .attrs
            .get_mut(attr)
            .ok_or_else(|| ExecError::Internal(format!("missing attr '{}'", attr)))?;
        g.update(x, tau_obs);
        Ok(())
    }

    pub fn force_attr_value(&mut self, node: NodeId, attr: &str, x: f64) -> Result<(), ExecError> {
        let n = self
            .node_mut(node)
            .ok_or_else(|| ExecError::Internal("missing node".into()))?;
        let g = n
            .attrs
            .get_mut(attr)
            .ok_or_else(|| ExecError::Internal(format!("missing attr '{}'", attr)))?;
        g.force_value(x);
        Ok(())
    }

    /// Validates numerical stability of the graph.
    ///
    /// Checks for NaN/Inf values, invalid precisions, Beta parameters outside
    /// valid range, and outlier values (> 10σ from mean).
    ///
    /// See baygraph_design.md:199-227 for numerical stability requirements.
    ///
    /// # Returns
    ///
    /// - `Ok(Vec<String>)` - List of warnings (empty if no issues)
    /// - `Err(ExecError::Numerical)` - Critical numerical error that makes graph invalid
    pub fn validate_numerical_stability(&self) -> Result<Vec<String>, ExecError> {
        let mut warnings = Vec::new();

        // Check node attributes
        for node in &self.nodes {
            for (attr_name, attr) in &node.attrs {
                // Check for NaN/Inf
                if !attr.mean.is_finite() {
                    return Err(ExecError::Numerical(format!(
                        "Node {:?} attribute '{}' has non-finite mean: {}",
                        node.id, attr_name, attr.mean
                    )));
                }
                if !attr.precision.is_finite() || attr.precision <= 0.0 {
                    return Err(ExecError::Numerical(format!(
                        "Node {:?} attribute '{}' has invalid precision: {}",
                        node.id, attr_name, attr.precision
                    )));
                }

                // Warn on very small precision (large uncertainty)
                if attr.precision < MIN_PRECISION * 10.0 {
                    warnings.push(format!(
                        "Node {:?} attribute '{}' has very low precision: {:.2e}",
                        node.id, attr_name, attr.precision
                    ));
                }

                // Warn on extreme outliers (|x - μ| > 10σ would be unusual)
                let std_dev = (1.0 / attr.precision).sqrt();
                // This check is more for data quality than numerical stability
                if std_dev.is_finite() {
                    let z_score = attr.mean.abs() / std_dev;
                    if z_score > OUTLIER_THRESHOLD_SIGMA {
                        warnings.push(format!(
                            "Node {:?} attribute '{}' has extreme value: mean={:.2e}, σ={:.2e}",
                            node.id, attr_name, attr.mean, std_dev
                        ));
                    }
                }
            }
        }

        // Check edge posteriors
        for edge in &self.edges {
            // Check for NaN/Inf in Beta parameters
            if !edge.exist.alpha.is_finite() || !edge.exist.beta.is_finite() {
                return Err(ExecError::Numerical(format!(
                    "Edge {:?} has non-finite Beta parameters: α={}, β={}",
                    edge.id, edge.exist.alpha, edge.exist.beta
                )));
            }

            // Enforce minimum Beta parameters (proper priors)
            if edge.exist.alpha < MIN_BETA_PARAM || edge.exist.beta < MIN_BETA_PARAM {
                warnings.push(format!(
                    "Edge {:?} has Beta parameters below minimum (α={:.2e}, β={:.2e}), will be clipped to {:.2e}",
                    edge.id, edge.exist.alpha, edge.exist.beta, MIN_BETA_PARAM
                ));
            }

            // Compute probability for range check
            let a = edge.exist.alpha.max(MIN_BETA_PARAM);
            let b = edge.exist.beta.max(MIN_BETA_PARAM);
            let prob = a / (a + b);

            // Probability must be in [0, 1]
            if !(0.0..=1.0).contains(&prob) {
                return Err(ExecError::Numerical(format!(
                    "Edge {:?} has invalid probability: {} (from α={}, β={})",
                    edge.id, prob, edge.exist.alpha, edge.exist.beta
                )));
            }
        }

        Ok(warnings)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    // ============================================================================
    // GaussianPosterior Unit Tests
    // ============================================================================

    #[test]
    fn gaussian_set_expectation_updates_mean_only() {
        let mut g = GaussianPosterior { mean: 0.0, precision: 1.0 };
        g.set_expectation(5.0);
        assert_eq!(g.mean, 5.0);
        assert_eq!(g.precision, 1.0); // precision unchanged
    }

    #[test]
    fn gaussian_update_combines_prior_and_observation() {
        // Prior: N(0, τ=1), Observation: x=10 with τ_obs=1
        // Expected posterior: N(5, τ=2)
        let mut g = GaussianPosterior { mean: 0.0, precision: 1.0 };
        g.update(10.0, 1.0);

        assert!((g.mean - 5.0).abs() < 1e-9, "mean should be 5.0");
        assert!((g.precision - 2.0).abs() < 1e-9, "precision should be 2.0");
    }

    #[test]
    fn gaussian_update_increases_precision() {
        let mut g = GaussianPosterior { mean: 0.0, precision: 1.0 };
        let initial_precision = g.precision;

        g.update(5.0, 2.0);

        assert!(g.precision > initial_precision, "precision should increase after observation");
    }

    #[test]
    fn gaussian_update_weighted_by_precision() {
        // High precision observation should dominate
        let mut g = GaussianPosterior { mean: 0.0, precision: 0.01 };
        g.update(100.0, 10.0); // high precision observation

        // Mean should be much closer to observation than prior
        assert!(g.mean > 90.0, "mean should be close to high-precision observation");
    }

    #[test]
    fn gaussian_update_clips_minimum_precision() {
        let mut g = GaussianPosterior { mean: 0.0, precision: 0.0 };
        g.update(5.0, 0.0);

        // Should clip to minimum τ=1e-6 per design doc
        assert!(g.precision >= 1e-6, "precision should be clipped to minimum");
    }

    #[test]
    fn gaussian_update_clips_negative_tau_obs() {
        let mut g = GaussianPosterior { mean: 0.0, precision: 1.0 };
        g.update(5.0, -1.0); // negative observation precision

        // Should clip tau_obs to 1e-12
        assert!(g.precision >= 1.0, "should handle negative tau_obs");
    }

    #[test]
    fn gaussian_force_value_sets_high_precision() {
        let mut g = GaussianPosterior { mean: 0.0, precision: 1.0 };
        g.force_value(42.0);

        assert_eq!(g.mean, 42.0, "mean should be forced to exact value");
        assert_eq!(g.precision, FORCE_PRECISION, "precision should be very high");
    }

    #[test]
    fn gaussian_force_value_represents_certainty() {
        let mut g = GaussianPosterior { mean: 10.0, precision: 2.0 };
        g.force_value(5.0);

        // Variance = 1/precision = 1/1e6, so std dev is ~0.001
        let variance = 1.0 / g.precision;
        assert!(variance < 1e-5, "forced value should have very low variance");
    }

    // ============================================================================
    // BetaPosterior Unit Tests
    // ============================================================================

    #[test]
    fn beta_force_absent_sets_near_zero_probability() {
        let mut b = BetaPosterior { alpha: 5.0, beta: 5.0 };
        b.force_absent();

        assert_eq!(b.alpha, 1.0);
        assert_eq!(b.beta, FORCE_PRECISION);

        let mean = b.alpha / (b.alpha + b.beta);
        assert!(mean < 1e-5, "forced absent should have near-zero mean");
    }

    #[test]
    fn beta_force_present_sets_near_one_probability() {
        let mut b = BetaPosterior { alpha: 5.0, beta: 5.0 };
        b.force_present();

        assert_eq!(b.alpha, FORCE_PRECISION);
        assert_eq!(b.beta, 1.0);

        let mean = b.alpha / (b.alpha + b.beta);
        assert!(mean > 0.99999, "forced present should have near-one mean");
    }

    #[test]
    fn beta_observe_present_increments_alpha() {
        let mut b = BetaPosterior { alpha: 2.0, beta: 3.0 };
        b.observe(true);

        assert_eq!(b.alpha, 3.0);
        assert_eq!(b.beta, 3.0);
    }

    #[test]
    fn beta_observe_absent_increments_beta() {
        let mut b = BetaPosterior { alpha: 2.0, beta: 3.0 };
        b.observe(false);

        assert_eq!(b.alpha, 2.0);
        assert_eq!(b.beta, 4.0);
    }

    #[test]
    fn beta_multiple_observations_accumulate() {
        let mut b = BetaPosterior { alpha: 1.0, beta: 1.0 };

        b.observe(true);
        b.observe(true);
        b.observe(false);

        assert_eq!(b.alpha, 3.0, "two present observations");
        assert_eq!(b.beta, 2.0, "one absent observation");
    }

    #[test]
    fn beta_mean_calculation_uniform_prior() {
        let b = BetaPosterior { alpha: 1.0, beta: 1.0 };
        let mean = b.alpha / (b.alpha + b.beta);

        assert!((mean - 0.5).abs() < 1e-9, "uniform prior should have mean 0.5");
    }

    #[test]
    fn beta_mean_calculation_biased_prior() {
        let b = BetaPosterior { alpha: 8.0, beta: 2.0 };
        let mean = b.alpha / (b.alpha + b.beta);

        assert!((mean - 0.8).abs() < 1e-9, "Beta(8,2) should have mean 0.8");
    }

    // ============================================================================
    // BeliefGraph Unit Tests
    // ============================================================================

    #[test]
    fn belief_graph_default_is_empty() {
        let g = BeliefGraph::default();
        assert_eq!(g.nodes.len(), 0);
        assert_eq!(g.edges.len(), 0);
    }

    #[test]
    fn belief_graph_node_lookup_by_id() {
        let mut g = BeliefGraph::default();
        g.insert_node(NodeData {
            id: NodeId(1),
            label: "Person".into(),
            attrs: HashMap::new(),
        });

        assert!(g.node(NodeId(1)).is_some());
        assert!(g.node(NodeId(999)).is_none());
    }

    #[test]
    fn belief_graph_edge_lookup_by_id() {
        let mut g = BeliefGraph::default();
        g.insert_edge(EdgeData {
            id: EdgeId(42),
            src: NodeId(1),
            dst: NodeId(2),
            ty: "REL".into(),
            exist: BetaPosterior { alpha: 1.0, beta: 1.0 },
        });

        assert!(g.edge(EdgeId(42)).is_some());
        assert!(g.edge(EdgeId(999)).is_none());
    }

    #[test]
    fn belief_graph_expectation_retrieves_mean() {
        let mut g = BeliefGraph::default();
        g.insert_node(NodeData {
            id: NodeId(1),
            label: "Person".into(),
            attrs: HashMap::from([
                ("age".into(), GaussianPosterior { mean: 25.5, precision: 1.0 }),
            ]),
        });

        let mean = g.expectation(NodeId(1), "age").unwrap();
        assert!((mean - 25.5).abs() < 1e-9);
    }

    #[test]
    fn belief_graph_expectation_missing_node_errors() {
        let g = BeliefGraph::default();
        let result = g.expectation(NodeId(999), "age");

        assert!(result.is_err());
    }

    #[test]
    fn belief_graph_expectation_missing_attr_errors() {
        let mut g = BeliefGraph::default();
        g.insert_node(NodeData {
            id: NodeId(1),
            label: "Person".into(),
            attrs: HashMap::new(),
        });

        let result = g.expectation(NodeId(1), "nonexistent");
        assert!(result.is_err());
    }

    #[test]
    fn belief_graph_set_expectation_updates_mean() {
        let mut g = BeliefGraph::default();
        g.insert_node(NodeData {
            id: NodeId(1),
            label: "Person".into(),
            attrs: HashMap::from([
                ("x".into(), GaussianPosterior { mean: 0.0, precision: 1.0 }),
            ]),
        });

        g.set_expectation(NodeId(1), "x", 10.0).unwrap();

        let mean = g.expectation(NodeId(1), "x").unwrap();
        assert!((mean - 10.0).abs() < 1e-9);
    }

    #[test]
    fn belief_graph_prob_mean_calculates_beta_mean() {
        let mut g = BeliefGraph::default();
        g.insert_edge(EdgeData {
            id: EdgeId(1),
            src: NodeId(1),
            dst: NodeId(2),
            ty: "REL".into(),
            exist: BetaPosterior { alpha: 3.0, beta: 7.0 },
        });

        let prob = g.prob_mean(EdgeId(1)).unwrap();
        assert!((prob - 0.3).abs() < 1e-9); // 3/(3+7) = 0.3
    }

    #[test]
    fn belief_graph_prob_mean_clips_negative_parameters() {
        let mut g = BeliefGraph::default();
        g.insert_edge(EdgeData {
            id: EdgeId(1),
            src: NodeId(1),
            dst: NodeId(2),
            ty: "REL".into(),
            exist: BetaPosterior { alpha: -1.0, beta: 2.0 },
        });

        // Should clip negative alpha to MIN_BETA_PARAM (0.01) per design spec
        // Expected: 0.01 / (0.01 + 2.0) ≈ 0.00497...
        let prob = g.prob_mean(EdgeId(1)).unwrap();
        let expected = MIN_BETA_PARAM / (MIN_BETA_PARAM + 2.0);
        assert!((prob - expected).abs() < 1e-9, "alpha should clip to MIN_BETA_PARAM, not 0");
    }

    #[test]
    fn belief_graph_prob_mean_handles_zero_parameters() {
        let mut g = BeliefGraph::default();
        g.insert_edge(EdgeData {
            id: EdgeId(1),
            src: NodeId(1),
            dst: NodeId(2),
            ty: "REL".into(),
            exist: BetaPosterior { alpha: 0.0, beta: 0.0 },
        });

        // With clipping to MIN_BETA_PARAM, even (0,0) becomes valid:
        // 0.01 / (0.01 + 0.01) = 0.5
        let prob = g.prob_mean(EdgeId(1)).unwrap();
        assert!((prob - 0.5).abs() < 1e-9, "Both params clip to 0.01, giving mean 0.5");
    }

    #[test]
    fn belief_graph_degree_outgoing_counts_high_prob_edges() {
        let mut g = BeliefGraph::default();
        g.insert_node(NodeData { id: NodeId(1), label: "N".into(), attrs: HashMap::new() });

        // Add edges with different probabilities
        g.insert_edge(EdgeData {
            id: EdgeId(1), src: NodeId(1), dst: NodeId(2), ty: "R".into(),
            exist: BetaPosterior { alpha: 8.0, beta: 2.0 }, // p=0.8
        });
        g.insert_edge(EdgeData {
            id: EdgeId(2), src: NodeId(1), dst: NodeId(3), ty: "R".into(),
            exist: BetaPosterior { alpha: 2.0, beta: 8.0 }, // p=0.2
        });
        g.insert_edge(EdgeData {
            id: EdgeId(3), src: NodeId(1), dst: NodeId(4), ty: "R".into(),
            exist: BetaPosterior { alpha: 6.0, beta: 4.0 }, // p=0.6
        });

        assert_eq!(g.degree_outgoing(NodeId(1), 0.7), 1, "only one edge >= 0.7");
        assert_eq!(g.degree_outgoing(NodeId(1), 0.5), 2, "two edges >= 0.5");
        assert_eq!(g.degree_outgoing(NodeId(1), 0.1), 3, "all edges >= 0.1");
    }

    #[test]
    fn belief_graph_degree_outgoing_filters_by_source() {
        let mut g = BeliefGraph::default();
        g.insert_edge(EdgeData {
            id: EdgeId(1), src: NodeId(1), dst: NodeId(2), ty: "R".into(),
            exist: BetaPosterior { alpha: 5.0, beta: 5.0 },
        });
        g.insert_edge(EdgeData {
            id: EdgeId(2), src: NodeId(2), dst: NodeId(3), ty: "R".into(),
            exist: BetaPosterior { alpha: 5.0, beta: 5.0 },
        });

        assert_eq!(g.degree_outgoing(NodeId(1), 0.0), 1);
        assert_eq!(g.degree_outgoing(NodeId(2), 0.0), 1);
        assert_eq!(g.degree_outgoing(NodeId(3), 0.0), 0);
    }

    #[test]
    fn belief_graph_adjacency_outgoing_groups_by_node_and_type() {
        let mut g = BeliefGraph::default();
        g.insert_edge(EdgeData {
            id: EdgeId(1), src: NodeId(1), dst: NodeId(2), ty: "LIKES".into(),
            exist: BetaPosterior { alpha: 1.0, beta: 1.0 },
        });
        g.insert_edge(EdgeData {
            id: EdgeId(2), src: NodeId(1), dst: NodeId(3), ty: "LIKES".into(),
            exist: BetaPosterior { alpha: 1.0, beta: 1.0 },
        });
        g.insert_edge(EdgeData {
            id: EdgeId(3), src: NodeId(1), dst: NodeId(4), ty: "KNOWS".into(),
            exist: BetaPosterior { alpha: 1.0, beta: 1.0 },
        });

        let adj = g.adjacency_outgoing_by_type();

        let likes = adj.get(&(NodeId(1), "LIKES".into())).unwrap();
        assert_eq!(likes.len(), 2);
        assert!(likes.contains(&EdgeId(1)));
        assert!(likes.contains(&EdgeId(2)));

        let knows = adj.get(&(NodeId(1), "KNOWS".into())).unwrap();
        assert_eq!(knows.len(), 1);
        assert!(knows.contains(&EdgeId(3)));
    }

    #[test]
    fn belief_graph_adjacency_outgoing_returns_sorted_edges() {
        let mut g = BeliefGraph::default();
        g.insert_edge(EdgeData {
            id: EdgeId(5), src: NodeId(1), dst: NodeId(2), ty: "R".into(),
            exist: BetaPosterior { alpha: 1.0, beta: 1.0 },
        });
        g.insert_edge(EdgeData {
            id: EdgeId(2), src: NodeId(1), dst: NodeId(3), ty: "R".into(),
            exist: BetaPosterior { alpha: 1.0, beta: 1.0 },
        });
        g.insert_edge(EdgeData {
            id: EdgeId(8), src: NodeId(1), dst: NodeId(4), ty: "R".into(),
            exist: BetaPosterior { alpha: 1.0, beta: 1.0 },
        });

        let adj = g.adjacency_outgoing_by_type();
        let edges = adj.get(&(NodeId(1), "R".into())).unwrap();

        assert_eq!(edges, &vec![EdgeId(2), EdgeId(5), EdgeId(8)], "should be sorted");
    }

    #[test]
    fn belief_graph_observe_edge_updates_beta_posterior() {
        let mut g = BeliefGraph::default();
        g.insert_edge(EdgeData {
            id: EdgeId(1), src: NodeId(1), dst: NodeId(2), ty: "R".into(),
            exist: BetaPosterior { alpha: 1.0, beta: 1.0 },
        });

        g.observe_edge(EdgeId(1), true).unwrap();

        let edge = g.edge(EdgeId(1)).unwrap();
        assert_eq!(edge.exist.alpha, 2.0);
        assert_eq!(edge.exist.beta, 1.0);
    }

    #[test]
    fn belief_graph_force_edge_present_sets_high_alpha() {
        let mut g = BeliefGraph::default();
        g.insert_edge(EdgeData {
            id: EdgeId(1), src: NodeId(1), dst: NodeId(2), ty: "R".into(),
            exist: BetaPosterior { alpha: 1.0, beta: 1.0 },
        });

        g.force_edge_present(EdgeId(1)).unwrap();

        let edge = g.edge(EdgeId(1)).unwrap();
        assert_eq!(edge.exist.alpha, FORCE_PRECISION);
        assert_eq!(edge.exist.beta, 1.0);
    }

    #[test]
    fn belief_graph_observe_attr_updates_gaussian() {
        let mut g = BeliefGraph::default();
        g.insert_node(NodeData {
            id: NodeId(1),
            label: "N".into(),
            attrs: HashMap::from([
                ("x".into(), GaussianPosterior { mean: 0.0, precision: 1.0 }),
            ]),
        });

        g.observe_attr(NodeId(1), "x", 10.0, 1.0).unwrap();

        let mean = g.expectation(NodeId(1), "x").unwrap();
        assert!((mean - 5.0).abs() < 1e-9); // posterior mean should be 5.0
    }

    #[test]
    fn belief_graph_force_attr_value_sets_precise_value() {
        let mut g = BeliefGraph::default();
        g.insert_node(NodeData {
            id: NodeId(1),
            label: "N".into(),
            attrs: HashMap::from([
                ("x".into(), GaussianPosterior { mean: 0.0, precision: 1.0 }),
            ]),
        });

        g.force_attr_value(NodeId(1), "x", 42.0).unwrap();

        let mean = g.expectation(NodeId(1), "x").unwrap();
        assert!((mean - 42.0).abs() < 1e-9);

        let node = g.node(NodeId(1)).unwrap();
        let attr = node.attrs.get("x").unwrap();
        assert_eq!(attr.precision, FORCE_PRECISION);
    }

    // ============================================================================
    // Additional Mathematical Correctness Tests
    // ============================================================================

    #[test]
    fn beta_posterior_mean_formula_verification() {
        // Verify the Beta posterior mean formula: E[p] = α / (α + β)
        let test_cases = vec![
            (1.0, 1.0, 0.5),      // Uniform: Beta(1,1) → 0.5
            (2.0, 8.0, 0.2),      // Skewed: Beta(2,8) → 0.2
            (8.0, 2.0, 0.8),      // Skewed: Beta(8,2) → 0.8
            (10.0, 10.0, 0.5),    // Symmetric: Beta(10,10) → 0.5
            (100.0, 1.0, 0.99),   // Near certain: Beta(100,1) ≈ 0.99
        ];

        for (alpha, beta, expected_mean) in test_cases {
            let mut g = BeliefGraph::default();
            g.insert_edge(EdgeData {
                id: EdgeId(1),
                src: NodeId(1),
                dst: NodeId(2),
                ty: "R".into(),
                exist: BetaPosterior { alpha, beta },
            });

            let prob = g.prob_mean(EdgeId(1)).unwrap();
            let expected = alpha / (alpha + beta);
            assert!(
                (prob - expected).abs() < 1e-9,
                "Beta({},{}) should have mean {} but got {}",
                alpha, beta, expected, prob
            );
            assert!(
                (prob - expected_mean).abs() < 0.01,
                "Beta({},{}) should be approximately {}",
                alpha, beta, expected_mean
            );
        }
    }

    #[test]
    fn beta_posterior_update_maintains_proper_parameters() {
        // Verify that Beta updates always maintain α > 0 and β > 0
        let mut beta = BetaPosterior { alpha: 1.0, beta: 1.0 };

        // Multiple observations
        for _ in 0..10 {
            beta.observe(true);
        }
        assert!(beta.alpha > 0.0, "Alpha must stay positive");
        assert!(beta.beta > 0.0, "Beta must stay positive");

        for _ in 0..10 {
            beta.observe(false);
        }
        assert!(beta.alpha > 0.0, "Alpha must stay positive");
        assert!(beta.beta > 0.0, "Beta must stay positive");
    }

    #[test]
    fn gaussian_update_precision_never_decreases() {
        // Fundamental property: observing data should never reduce precision
        let mut g = GaussianPosterior { mean: 0.0, precision: 1.0 };
        let initial_precision = g.precision;

        // Single observation
        g.update(5.0, 1.0);
        assert!(
            g.precision >= initial_precision,
            "Precision should never decrease after observation"
        );

        // Another observation
        let mid_precision = g.precision;
        g.update(3.0, 0.5);
        assert!(
            g.precision >= mid_precision,
            "Precision should continue to increase"
        );
    }

    #[test]
    fn gaussian_update_mean_is_precision_weighted_average() {
        // Prior: N(μ₀=10, τ₀=2)
        // Observation: x=20 with τ_obs=3
        // Expected: μ_new = (2×10 + 3×20) / (2+3) = (20 + 60) / 5 = 16
        let mut g = GaussianPosterior { mean: 10.0, precision: 2.0 };
        g.update(20.0, 3.0);

        let expected_precision = 2.0 + 3.0;
        let expected_mean = (2.0 * 10.0 + 3.0 * 20.0) / expected_precision;

        assert!(
            (g.precision - expected_precision).abs() < 1e-9,
            "Precision should be sum: τ_new = τ_old + τ_obs"
        );
        assert!(
            (g.mean - expected_mean).abs() < 1e-9,
            "Mean should be precision-weighted average"
        );
        assert!((g.mean - 16.0).abs() < 1e-9, "Expected mean 16.0");
    }

    #[test]
    fn degree_outgoing_consistent_with_prob_mean() {
        // Ensure degree_outgoing uses same probability calculation as prob_mean
        let mut g = BeliefGraph::default();
        g.insert_node(NodeData {
            id: NodeId(1),
            label: "N".into(),
            attrs: HashMap::new(),
        });

        // Add edge with Beta(3, 7) → mean = 0.3
        g.insert_edge(EdgeData {
            id: EdgeId(1),
            src: NodeId(1),
            dst: NodeId(2),
            ty: "R".into(),
            exist: BetaPosterior { alpha: 3.0, beta: 7.0 },
        });

        let prob = g.prob_mean(EdgeId(1)).unwrap();
        assert!((prob - 0.3).abs() < 1e-9, "Probability should be 0.3");

        // degree_outgoing should count it if min_prob <= 0.3
        assert_eq!(g.degree_outgoing(NodeId(1), 0.25), 1, "Should count at 0.25 threshold");
        assert_eq!(g.degree_outgoing(NodeId(1), 0.30), 1, "Should count at 0.30 threshold");
        assert_eq!(g.degree_outgoing(NodeId(1), 0.35), 0, "Should not count at 0.35 threshold");
    }

    #[test]
    fn force_operations_create_near_deterministic_beliefs() {
        // force_present should create near-1 probability
        let mut b = BetaPosterior { alpha: 1.0, beta: 1.0 };
        b.force_present();
        let mean = b.alpha / (b.alpha + b.beta);
        assert!(mean > 0.999, "force_present should give probability > 0.999");

        // force_absent should create near-0 probability
        b.force_absent();
        let mean = b.alpha / (b.alpha + b.beta);
        assert!(mean < 0.001, "force_absent should give probability < 0.001");

        // force_value should create very small variance
        let mut g = GaussianPosterior { mean: 0.0, precision: 1.0 };
        g.force_value(42.0);
        let variance = 1.0 / g.precision;
        assert!(variance < 1e-5, "force_value should give variance < 1e-5");
        assert_eq!(g.mean, 42.0, "force_value should set exact mean");
    }

    // ============================================================================
    // Adjacency Index Tests (Phase 7)
    // ============================================================================

    #[test]
    fn adjacency_index_groups_edges_by_node_and_type() {
        let mut g = BeliefGraph::default();
        g.insert_edge(EdgeData {
            id: EdgeId(1), src: NodeId(1), dst: NodeId(2), ty: "LIKES".into(),
            exist: BetaPosterior { alpha: 1.0, beta: 1.0 },
        });
        g.insert_edge(EdgeData {
            id: EdgeId(2), src: NodeId(1), dst: NodeId(3), ty: "LIKES".into(),
            exist: BetaPosterior { alpha: 1.0, beta: 1.0 },
        });
        g.insert_edge(EdgeData {
            id: EdgeId(3), src: NodeId(1), dst: NodeId(4), ty: "KNOWS".into(),
            exist: BetaPosterior { alpha: 1.0, beta: 1.0 },
        });
        g.insert_edge(EdgeData {
            id: EdgeId(4), src: NodeId(2), dst: NodeId(3), ty: "LIKES".into(),
            exist: BetaPosterior { alpha: 1.0, beta: 1.0 },
        });

        g.build_adjacency();

        let likes = g.get_outgoing_edges(NodeId(1), "LIKES");
        assert_eq!(likes.len(), 2);
        assert!(likes.contains(&EdgeId(1)));
        assert!(likes.contains(&EdgeId(2)));

        let knows = g.get_outgoing_edges(NodeId(1), "KNOWS");
        assert_eq!(knows.len(), 1);
        assert!(knows.contains(&EdgeId(3)));

        let node2_likes = g.get_outgoing_edges(NodeId(2), "LIKES");
        assert_eq!(node2_likes.len(), 1);
        assert!(node2_likes.contains(&EdgeId(4)));
    }

    #[test]
    fn adjacency_index_returns_sorted_edges() {
        let mut g = BeliefGraph::default();
        // Insert edges in non-sorted order
        g.insert_edge(EdgeData {
            id: EdgeId(5), src: NodeId(1), dst: NodeId(2), ty: "R".into(),
            exist: BetaPosterior { alpha: 1.0, beta: 1.0 },
        });
        g.insert_edge(EdgeData {
            id: EdgeId(2), src: NodeId(1), dst: NodeId(3), ty: "R".into(),
            exist: BetaPosterior { alpha: 1.0, beta: 1.0 },
        });
        g.insert_edge(EdgeData {
            id: EdgeId(8), src: NodeId(1), dst: NodeId(4), ty: "R".into(),
            exist: BetaPosterior { alpha: 1.0, beta: 1.0 },
        });

        g.build_adjacency();
        let edges = g.get_outgoing_edges(NodeId(1), "R");

        // Should be sorted by EdgeId
        assert_eq!(edges, vec![EdgeId(2), EdgeId(5), EdgeId(8)]);
    }

    #[test]
    fn adjacency_index_handles_empty_neighborhoods() {
        let mut g = BeliefGraph::default();
        g.insert_edge(EdgeData {
            id: EdgeId(1), src: NodeId(1), dst: NodeId(2), ty: "R".into(),
            exist: BetaPosterior { alpha: 1.0, beta: 1.0 },
        });

        g.build_adjacency();

        // Query non-existent node
        let edges = g.get_outgoing_edges(NodeId(999), "R");
        assert_eq!(edges.len(), 0);

        // Query non-existent edge type
        let edges = g.get_outgoing_edges(NodeId(1), "NONEXISTENT");
        assert_eq!(edges.len(), 0);
    }

    #[test]
    fn adjacency_index_lazy_build() {
        let mut g = BeliefGraph::default();
        g.insert_edge(EdgeData {
            id: EdgeId(1), src: NodeId(1), dst: NodeId(2), ty: "R".into(),
            exist: BetaPosterior { alpha: 1.0, beta: 1.0 },
        });

        // Don't explicitly call build_adjacency
        let edges = g.get_outgoing_edges(NodeId(1), "R");

        // Should build index lazily and return correct result
        assert_eq!(edges.len(), 1);
        assert_eq!(edges[0], EdgeId(1));
    }
}
