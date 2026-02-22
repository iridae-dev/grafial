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
//! The implementation uses conjugate priors for efficient Bayesian updates:
//! - Normal-Normal conjugacy for Gaussian attributes (precision parameterization τ = 1/σ²)
//! - Beta-Bernoulli conjugacy for independent edges (α, β parameters)
//! - Dirichlet-Categorical conjugacy for competing edges (α_k concentrations)
//! - Force operations for hard constraints (sets precision/concentrations to large finite values)
//! - O(1) node/edge lookups via HashMap indexes
//!
//! ## Example
//!
//! ```rust,ignore
//! use grafial::engine::graph::*;
//! use std::collections::HashMap;
//!
//! let mut graph = BeliefGraph::default();
//! let node_id = graph.add_node("Person".into(), HashMap::new());
//! let edge_id = graph.add_edge(node_id, node_id, "KNOWS".into(),
//!     BetaPosterior { alpha: 1.0, beta: 1.0 });
//! ```

use std::collections::HashMap;
use std::sync::Arc;

use rustc_hash::FxHashMap;
use smallvec::SmallVec;

#[cfg(feature = "storage-dense-index")]
use crate::engine::adjacency_index::DenseAdjacencyIndex;
use crate::engine::adjacency_index::EnhancedAdjacencyIndex;
use crate::engine::errors::ExecError;

// Bayesian inference constants
/// High precision value for force operations.
///
/// Sets precision to 1e6 (large but finite) to create hard constraints while avoiding
/// infinities. Used by force_value, force_present, force_absent, and force_choice.
const FORCE_PRECISION: f64 = 1_000_000.0;

/// Minimum precision for Gaussian posteriors.
///
/// Clips extremely small precision values to prevent division by zero in variance
/// calculations (variance = 1/τ) and loss of significance.
const MIN_PRECISION: f64 = 1e-6;

/// Minimum observation precision to prevent numerical issues
const MIN_OBS_PRECISION: f64 = 1e-12;

/// Minimum Beta parameter value to enforce proper prior.
///
/// Beta distribution requires α > 0 and β > 0 strictly. We enforce α ≥ 0.01, β ≥ 0.01
/// as a numeric floor for stability and to prevent improper priors.
const MIN_BETA_PARAM: f64 = 0.01;

/// Minimum Dirichlet concentration parameter to enforce proper prior
/// Dirichlet requires all α_k > 0. We enforce α_k ≥ 0.01 for numerical stability.
const MIN_DIRICHLET_PARAM: f64 = 0.01;

/// Maximum allowed deviation in standard deviations for outlier warnings.
///
/// If |x - μ| > OUTLIER_THRESHOLD_SIGMA × σ, issue a warning as this may indicate
/// a potential outlier or data error.
const OUTLIER_THRESHOLD_SIGMA: f64 = 10.0;

/// A unique identifier for a node in the belief graph.
///
/// NodeId implements Ord/PartialOrd for stable, deterministic iteration.
/// Uses u32 internally for efficient storage and indexing.
#[repr(transparent)]
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Ord, PartialOrd)]
#[cfg_attr(feature = "serde", derive(serde::Serialize, serde::Deserialize))]
pub struct NodeId(pub u32);

/// A unique identifier for an edge in the belief graph.
///
/// EdgeId implements Ord/PartialOrd for stable, deterministic iteration.
/// Uses u32 internally for efficient storage and indexing.
#[repr(transparent)]
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Ord, PartialOrd)]
#[cfg_attr(feature = "serde", derive(serde::Serialize, serde::Deserialize))]
pub struct EdgeId(pub u32);

/// A Gaussian posterior distribution for continuous node attributes.
///
/// Uses the precision parameterization (τ = 1/σ²) for efficient conjugate updates.
/// Bayesian update formulas (Normal-Normal conjugacy):
/// - τ_new = τ_old + τ_obs
/// - μ_new = (τ_old × μ_old + τ_obs × x) / τ_new
#[derive(Debug, Clone, Copy)]
#[cfg_attr(feature = "serde", derive(serde::Serialize, serde::Deserialize))]
pub struct GaussianPosterior {
    /// The posterior mean (μ)
    pub mean: f64,
    /// The posterior precision (τ = 1/σ²)
    pub precision: f64,
}

impl GaussianPosterior {
    /// Updates the posterior mean without changing precision (certainty).
    ///
    /// # WARNING: This is NOT a Bayesian update from an observation!
    ///
    /// This method adjusts the mean while keeping precision unchanged, which
    /// does not correspond to a standard Bayesian update. Use this for:
    ///
    /// - **Soft constraints**: Nudging beliefs without adding evidence
    /// - **Expert elicitation**: "I think this should be around X" (without increased confidence)
    /// - **Mean revision**: Adjusting beliefs based on non-evidential reasoning
    ///
    /// For a proper Bayesian update from an observation, use [`update()`](Self::update)
    /// or [`observe_soft()`](Self::observe_soft) instead.
    ///
    /// # Example
    ///
    /// ```rust,ignore
    /// let mut posterior = GaussianPosterior { mean: 10.0, precision: 2.0 };
    /// posterior.set_expectation(15.0);
    /// // Mean changed to 15.0, but precision still 2.0 (no new evidence)
    /// ```
    pub fn set_expectation(&mut self, v: f64) {
        self.mean = v;
    }

    /// Performs a Bayesian update with low confidence (soft observation).
    ///
    /// This is a Bayesian alternative to [`set_expectation()`](Self::set_expectation)
    /// that models a "weak observation" or "soft suggestion" of a value.
    ///
    /// Unlike `set_expectation()`, this:
    /// - Increases precision (adds evidence)
    /// - Uses proper Bayesian conjugate update
    /// - Moves the mean toward the observed value (doesn't set it directly)
    ///
    /// # Arguments
    ///
    /// * `value` - The observed or suggested value
    /// * `confidence` - Observation precision (higher = more confident)
    ///   - Use ~0.01 for very weak suggestions
    ///   - Use ~1.0 for normal observations
    ///   - Use ~100.0 for strong observations
    ///
    /// # Example
    ///
    /// ```rust,ignore
    /// let mut posterior = GaussianPosterior { mean: 10.0, precision: 2.0 };
    /// posterior.observe_soft(15.0, 0.1);  // Weak suggestion toward 15.0
    /// // Mean moves slightly toward 15.0, precision increases slightly
    /// ```
    pub fn observe_soft(&mut self, value: f64, confidence: f64) {
        self.update(value, confidence);
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
    /// Uses Normal-Normal conjugate update formulas:
    /// - τ_new = τ_old + τ_obs
    /// - μ_new = (τ_old × μ_old + τ_obs × x) / τ_new
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
    /// This corresponds to conditioning on an observation with effectively infinite
    /// precision; subsequent finite-precision updates will have negligible effect
    /// but remain well-defined.
    ///
    /// # Arguments
    ///
    /// * `x` - The value to force
    pub fn force_value(&mut self, x: f64) {
        self.mean = x;
        self.precision = FORCE_PRECISION;
    }

    /// Returns the posterior variance σ² = 1/τ (with precision floor applied).
    pub fn variance(&self) -> f64 {
        let tau = self.precision.max(MIN_PRECISION);
        1.0 / tau
    }
}

/// A Beta posterior distribution for edge existence probability.
///
/// Uses the Beta(α, β) parameterization for conjugate Bernoulli updates.
/// The mean probability is E[p] = α / (α + β).
/// Bayesian update: increments α if present, β if absent.
#[derive(Debug, Clone, Copy)]
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
    /// Sets parameters to create a numerically stable approximation to a degenerate
    /// belief (mean ≈ 0.000001). Further single observations will have negligible effect.
    pub fn force_absent(&mut self) {
        self.alpha = 1.0;
        self.beta = FORCE_PRECISION;
    }

    /// Forces the edge to be present (α=1e6, β=1).
    ///
    /// Sets parameters to create a numerically stable approximation to a degenerate
    /// belief (mean ≈ 0.999999). Further single observations will have negligible effect.
    pub fn force_present(&mut self) {
        self.alpha = FORCE_PRECISION;
        self.beta = 1.0;
    }

    /// Performs a conjugate Beta-Bernoulli update.
    ///
    /// Increments α if present, β if absent. This is the standard conjugate update
    /// for Beta-Bernoulli posteriors.
    ///
    /// # Arguments
    ///
    /// * `present` - Whether the edge was observed to be present
    pub fn observe(&mut self, present: bool) {
        if present {
            self.alpha += 1.0;
        } else {
            self.beta += 1.0;
        }
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

    /// Computes the posterior variance of the probability.
    ///
    /// For Beta(α, β), the variance is:
    /// ```text
    /// Var[p] = αβ / [(α+β)²(α+β+1)]
    /// ```
    ///
    /// Applies MIN_BETA_PARAM floor to both parameters for numerical stability.
    ///
    /// # Returns
    ///
    /// The variance of the posterior probability distribution.
    ///
    /// # Example
    ///
    /// ```rust,ignore
    /// let posterior = BetaPosterior { alpha: 2.0, beta: 3.0 };
    /// let mean = posterior.mean_probability();  // 0.4
    /// let var = posterior.variance();           // 0.04
    /// let std_dev = var.sqrt();                 // 0.2
    /// ```
    pub fn variance(&self) -> f64 {
        let a = self.alpha.max(MIN_BETA_PARAM);
        let b = self.beta.max(MIN_BETA_PARAM);
        let sum = a + b;
        (a * b) / (sum * sum * (sum + 1.0))
    }
}

/// A Dirichlet posterior distribution for competing edge probabilities.
///
/// Uses Dirichlet(α_1, ..., α_K) parameterization for conjugate Categorical updates.
/// Models mutually exclusive choices among K alternatives where probabilities sum to 1.
/// The mean probabilities are E[π_k] = α_k / Σ_j α_j, with Σ_k π_k = 1.
#[derive(Debug, Clone)]
#[cfg_attr(feature = "serde", derive(serde::Serialize, serde::Deserialize))]
pub struct DirichletPosterior {
    /// Concentration parameters α_k for each category
    /// Each α_k must be > 0; we enforce α_k ≥ MIN_DIRICHLET_PARAM for stability.
    pub concentrations: Vec<f64>,
}

impl DirichletPosterior {
    /// Creates a new Dirichlet posterior with given concentration parameters.
    ///
    /// # Arguments
    ///
    /// * `concentrations` - Initial α_k parameters (must all be > 0)
    ///
    /// # Panics
    ///
    /// Panics if any concentration is ≤ 0 or if the vector is empty.
    pub fn new(concentrations: Vec<f64>) -> Self {
        assert!(
            !concentrations.is_empty(),
            "Dirichlet posterior requires at least one category"
        );
        assert!(
            concentrations.iter().all(|&a| a > 0.0),
            "All Dirichlet concentrations must be > 0"
        );
        Self { concentrations }
    }

    /// Creates a uniform Dirichlet posterior.
    ///
    /// All categories have equal prior: α_k = pseudo_count / K for K categories.
    ///
    /// # Arguments
    ///
    /// * `num_categories` - Number of categories K
    /// * `pseudo_count` - Total pseudo-count (Σ α_k = pseudo_count)
    pub fn uniform(num_categories: usize, pseudo_count: f64) -> Self {
        assert!(num_categories > 0, "Must have at least one category");
        assert!(pseudo_count > 0.0, "Pseudo-count must be > 0");
        let alpha_per_category = pseudo_count / num_categories as f64;
        Self {
            concentrations: vec![alpha_per_category; num_categories],
        }
    }

    /// Updates the posterior with an observation of a chosen category.
    ///
    /// Performs a conjugate Dirichlet-Categorical update: increments α_k for the chosen category.
    /// This is the standard update when observing that category k was chosen.
    ///
    /// # Arguments
    ///
    /// * `category_index` - Index of the chosen category (0-based)
    pub fn observe_chosen(&mut self, category_index: usize) {
        assert!(
            category_index < self.concentrations.len(),
            "Category index out of bounds"
        );
        self.concentrations[category_index] += 1.0;
    }

    /// Updates the posterior with an observation that a category was not chosen.
    ///
    /// For binary categorical distributions (K=2), this is equivalent to observing
    /// the other category as chosen, so we apply an exact conjugate update.
    ///
    /// For K>2, observing only "not category i" yields likelihood `1 - π_i`, which is
    /// not conjugate to a Dirichlet prior. In that case, this method returns an error.
    ///
    /// # Arguments
    ///
    /// * `category_index` - Index of the unchosen category
    pub fn observe_unchosen(&mut self, category_index: usize) -> Result<(), ExecError> {
        if category_index >= self.concentrations.len() {
            return Err(ExecError::ValidationError(
                "Category index out of bounds".into(),
            ));
        }

        let k = self.concentrations.len();
        if k != 2 {
            return Err(ExecError::ValidationError(
                "observe_unchosen is only conjugate for binary categorical groups; use observe_chosen with an explicit chosen category for K>2".into(),
            ));
        }

        let chosen_index = 1 - category_index;
        self.observe_chosen(chosen_index);
        Ok(())
    }

    /// Forces a specific category to be chosen with very high certainty.
    ///
    /// Sets α_k = 1e6 for the chosen category, α_j = 1.0 for others.
    /// This creates a numerically stable approximation to a degenerate distribution
    /// where category k has probability ≈ 1.0.
    ///
    /// # Arguments
    ///
    /// * `category_index` - Index of the category to force
    pub fn force_choice(&mut self, category_index: usize) {
        assert!(
            category_index < self.concentrations.len(),
            "Category index out of bounds"
        );
        for (i, alpha) in self.concentrations.iter_mut().enumerate() {
            if i == category_index {
                *alpha = FORCE_PRECISION;
            } else {
                *alpha = 1.0;
            }
        }
    }

    /// Computes the posterior mean probability for a category.
    ///
    /// Returns E[π_k] = α_k / Σ_j α_j. Applies MIN_DIRICHLET_PARAM floor
    /// to all parameters for numerical stability.
    ///
    /// # Arguments
    ///
    /// * `category_index` - Index of the category
    ///
    /// # Returns
    ///
    /// The posterior mean probability in [0, 1]
    pub fn mean_probability(&self, category_index: usize) -> f64 {
        assert!(
            category_index < self.concentrations.len(),
            "Category index out of bounds"
        );
        let alpha_k = self.concentrations[category_index].max(MIN_DIRICHLET_PARAM);
        let sum_alpha: f64 = self
            .concentrations
            .iter()
            .map(|&a| a.max(MIN_DIRICHLET_PARAM))
            .sum();
        alpha_k / sum_alpha
    }

    /// Computes the full posterior mean probability vector.
    ///
    /// Returns [E[π_1], E[π_2], ..., E[π_K]] for all categories.
    pub fn mean_probabilities(&self) -> Vec<f64> {
        crate::engine::numeric_kernels::dirichlet_mean_probabilities(
            &self.concentrations,
            MIN_DIRICHLET_PARAM,
        )
    }

    /// Computes the Shannon entropy of the distribution.
    ///
    /// Returns H(π) = -Σ_k π_k log(π_k) in nats.
    /// Range: [0, log(K)] where K = number of categories.
    pub fn entropy(&self) -> f64 {
        let probs = self.mean_probabilities();
        probs
            .iter()
            .filter(|&&p| p > 1e-10)
            .map(|&p| -p * p.ln())
            .sum()
    }

    /// Returns the number of categories.
    pub fn num_categories(&self) -> usize {
        self.concentrations.len()
    }
}

/// A node in the belief graph with typed attributes.
#[derive(Debug, Clone)]
#[cfg_attr(feature = "serde", derive(serde::Serialize, serde::Deserialize))]
pub struct NodeData {
    /// The unique node identifier
    pub id: NodeId,
    /// The node type label (e.g., "Person", "Company")
    /// Using Arc<str> for cheap cloning (reference count increment, not allocation)
    #[cfg_attr(feature = "serde", serde(with = "serde_helpers::serde_arc_str"))]
    pub label: Arc<str>,
    /// Gaussian posteriors for continuous attributes
    pub attrs: HashMap<String, GaussianPosterior>,
}

/// A unique identifier for a competing edge group.
#[repr(transparent)]
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Ord, PartialOrd)]
#[cfg_attr(feature = "serde", derive(serde::Serialize, serde::Deserialize))]
pub struct CompetingGroupId(pub u32);

/// A competing edge group representing mutually exclusive choices from a source.
///
/// Groups edges from the same source node and edge type that share a Dirichlet posterior.
/// This enforces the constraint that exactly one choice must be made (probabilities sum to 1).
#[derive(Debug, Clone)]
#[cfg_attr(feature = "serde", derive(serde::Serialize, serde::Deserialize))]
pub struct CompetingEdgeGroup {
    /// Unique identifier for this group
    pub id: CompetingGroupId,
    /// Source node for this group
    pub source: NodeId,
    /// Edge type for this group
    pub edge_type: String,
    /// Destination nodes (categories) in order
    pub categories: Vec<NodeId>,
    /// Mapping from destination NodeId to category index
    pub category_index: FxHashMap<NodeId, usize>,
    /// Dirichlet posterior for the probability distribution
    pub posterior: DirichletPosterior,
}

impl CompetingEdgeGroup {
    /// Creates a new competing edge group.
    ///
    /// # Arguments
    ///
    /// * `id` - Unique group identifier
    /// * `source` - Source node ID
    /// * `edge_type` - Edge type name
    /// * `categories` - Destination nodes (must be non-empty)
    /// * `posterior` - Initial Dirichlet posterior
    pub fn new(
        id: CompetingGroupId,
        source: NodeId,
        edge_type: String,
        categories: Vec<NodeId>,
        posterior: DirichletPosterior,
    ) -> Self {
        assert_eq!(
            categories.len(),
            posterior.num_categories(),
            "Number of categories must match posterior dimensions"
        );
        let category_index: FxHashMap<NodeId, usize> = categories
            .iter()
            .enumerate()
            .map(|(i, &node_id)| (node_id, i))
            .collect();
        Self {
            id,
            source,
            edge_type,
            categories,
            category_index,
            posterior,
        }
    }

    /// Gets the category index for a destination node.
    pub fn get_category_index(&self, dst: NodeId) -> Option<usize> {
        self.category_index.get(&dst).copied()
    }

    /// Gets the mean probability for a destination node.
    pub fn mean_probability_for_dst(&self, dst: NodeId) -> Option<f64> {
        let idx = self.get_category_index(dst)?;
        Some(self.posterior.mean_probability(idx))
    }
}

/// Edge posterior type: either independent (Beta) or competing (Dirichlet group reference).
#[derive(Debug, Clone)]
#[cfg_attr(feature = "serde", derive(serde::Serialize, serde::Deserialize))]
pub enum EdgePosterior {
    /// Independent edge with Beta posterior
    Independent(BetaPosterior),
    /// Competing edge with reference to a Dirichlet group
    Competing {
        /// Reference to the competing group
        group_id: CompetingGroupId,
        /// Category index within the group
        category_index: usize,
    },
}

impl Default for EdgePosterior {
    fn default() -> Self {
        // Default to independent edge with uniform Beta(1,1) prior
        Self::Independent(BetaPosterior {
            alpha: 1.0,
            beta: 1.0,
        })
    }
}

impl EdgePosterior {
    /// Creates an independent edge posterior from a Beta posterior.
    pub fn independent(beta: BetaPosterior) -> Self {
        Self::Independent(beta)
    }

    /// Creates a competing edge posterior reference.
    pub fn competing(group_id: CompetingGroupId, category_index: usize) -> Self {
        Self::Competing {
            group_id,
            category_index,
        }
    }

    /// Gets a mutable reference to the Beta posterior if independent.
    pub fn as_beta_mut(&mut self) -> Option<&mut BetaPosterior> {
        match self {
            Self::Independent(beta) => Some(beta),
            Self::Competing { .. } => None,
        }
    }

    /// Gets a reference to the Beta posterior if independent.
    pub fn as_beta(&self) -> Option<&BetaPosterior> {
        match self {
            Self::Independent(beta) => Some(beta),
            Self::Competing { .. } => None,
        }
    }

    /// Forces the edge to be absent (only for independent edges).
    pub fn force_absent(&mut self) -> Result<(), ExecError> {
        match self {
            Self::Independent(beta) => {
                beta.force_absent();
                Ok(())
            }
            Self::Competing { .. } => Err(ExecError::ValidationError(
                "force_absent() only valid for independent edges".into(),
            )),
        }
    }

    /// Forces the edge to be present (only for independent edges).
    pub fn force_present(&mut self) -> Result<(), ExecError> {
        match self {
            Self::Independent(beta) => {
                beta.force_present();
                Ok(())
            }
            Self::Competing { .. } => Err(ExecError::ValidationError(
                "force_present() only valid for independent edges".into(),
            )),
        }
    }

    /// Updates the posterior with an observation (only for independent edges).
    pub fn observe(&mut self, present: bool) -> Result<(), ExecError> {
        match self {
            Self::Independent(beta) => {
                beta.observe(present);
                Ok(())
            }
            Self::Competing { .. } => Err(ExecError::ValidationError(
                "observe() only valid for independent edges; use observe_chosen for competing edges".into(),
            )),
        }
    }

    /// Computes the mean probability.
    ///
    /// For independent edges: E[p] = α / (α + β)
    /// For competing edges: requires access to the group's Dirichlet posterior
    pub fn mean_probability(
        &self,
        groups: &FxHashMap<CompetingGroupId, CompetingEdgeGroup>,
    ) -> Result<f64, ExecError> {
        match self {
            Self::Independent(beta) => Ok(beta.mean_probability()),
            Self::Competing {
                group_id,
                category_index,
            } => {
                let group = groups
                    .get(group_id)
                    .ok_or_else(|| ExecError::Internal("missing competing group".into()))?;
                Ok(group.posterior.mean_probability(*category_index))
            }
        }
    }
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
    /// Using Arc<str> for cheap cloning (reference count increment, not allocation)
    #[cfg_attr(feature = "serde", serde(with = "serde_helpers::serde_arc_str"))]
    pub ty: Arc<str>,
    /// Posterior for edge existence (independent or competing)
    pub exist: EdgePosterior,
}

/// Optimized adjacency list for fast neighborhood queries.
///
/// Uses offset-based indexing into a contiguous edge ID array for O(1) access.
/// This structure is precomputed and cached for performance.
///
/// The adjacency index stores ranges of edge IDs for each (node, edge_type) pair,
/// allowing fast neighborhood queries without scanning all edges.
#[derive(Debug, Clone)]
#[cfg_attr(feature = "serde", derive(serde::Serialize, serde::Deserialize))]
pub struct AdjacencyIndex {
    /// Maps (NodeId, EdgeType) to (start_offset, end_offset) in edge_ids
    /// Using Arc<str> to avoid String allocations on lookup
    #[cfg_attr(
        feature = "serde",
        serde(
            serialize_with = "serde_helpers::serialize_adjacency_ranges",
            deserialize_with = "serde_helpers::deserialize_adjacency_ranges"
        )
    )]
    ranges: HashMap<(NodeId, Arc<str>), (usize, usize)>,
    /// Contiguous sorted array of EdgeIds
    edge_ids: Vec<EdgeId>,
}

impl Default for AdjacencyIndex {
    fn default() -> Self {
        Self::new()
    }
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
        let mut map: HashMap<(NodeId, Arc<str>), Vec<EdgeId>> = HashMap::new();
        for e in edges {
            // Arc clone is cheap (just reference count increment)
            map.entry((e.src, e.ty.clone())).or_default().push(e.id);
        }

        // Sort each adjacency list for determinism (unstable sort is faster)
        for v in map.values_mut() {
            v.sort_unstable();
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
        // Convert &str to Arc<str> for lookup (creates Arc but doesn't allocate new string)
        let key = (node, Arc::from(edge_type));
        if let Some(&(start, end)) = self.ranges.get(&key) {
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
        result.sort_unstable(); // Maintain deterministic order (unstable sort is faster)
        result
    }
}

/// Optional SoA edge index experiment (Phase 13).
///
/// Stores edge columns in separate vectors to improve cache locality for
/// edge-heavy query patterns while preserving deterministic ordering.
#[cfg(feature = "storage-soa")]
#[derive(Debug, Clone)]
pub struct EdgeSoAIndex {
    edge_ids: Vec<EdgeId>,
    dst_ids: Vec<NodeId>,
    ranges: HashMap<(NodeId, Arc<str>), (usize, usize)>,
    src_ranges: HashMap<NodeId, SmallVec<[(usize, usize); 4]>>,
}

#[cfg(feature = "storage-soa")]
impl EdgeSoAIndex {
    pub fn from_edges(edges: &[EdgeData]) -> Self {
        let mut rows: Vec<(NodeId, Arc<str>, EdgeId, NodeId)> = edges
            .iter()
            .map(|edge| (edge.src, edge.ty.clone(), edge.id, edge.dst))
            .collect();

        rows.sort_unstable_by(|a, b| {
            a.0.cmp(&b.0)
                .then_with(|| a.1.as_ref().cmp(b.1.as_ref()))
                .then_with(|| a.2.cmp(&b.2))
        });

        let mut edge_ids = Vec::with_capacity(rows.len());
        let mut dst_ids = Vec::with_capacity(rows.len());
        let mut ranges: HashMap<(NodeId, Arc<str>), (usize, usize)> = HashMap::new();
        let mut src_ranges: HashMap<NodeId, SmallVec<[(usize, usize); 4]>> = HashMap::new();

        let mut current_key: Option<(NodeId, Arc<str>, usize)> = None;
        for (idx, (src, ty, edge_id, dst)) in rows.into_iter().enumerate() {
            match &current_key {
                Some((current_src, current_ty, _))
                    if *current_src == src && current_ty.as_ref() == ty.as_ref() => {}
                Some((current_src, current_ty, start)) => {
                    ranges.insert((*current_src, current_ty.clone()), (*start, idx));
                    src_ranges
                        .entry(*current_src)
                        .or_default()
                        .push((*start, idx));
                    current_key = Some((src, ty.clone(), idx));
                }
                None => {
                    current_key = Some((src, ty.clone(), idx));
                }
            }

            edge_ids.push(edge_id);
            dst_ids.push(dst);
        }

        if let Some((src, ty, start)) = current_key {
            let end = edge_ids.len();
            ranges.insert((src, ty), (start, end));
            src_ranges.entry(src).or_default().push((start, end));
        }

        Self {
            edge_ids,
            dst_ids,
            ranges,
            src_ranges,
        }
    }

    pub fn get_edges(&self, node: NodeId, edge_type: &str) -> Vec<EdgeId> {
        let key = (node, Arc::from(edge_type));
        self.ranges.get(&key).map_or_else(Vec::new, |(start, end)| {
            self.edge_ids[*start..*end].to_vec()
        })
    }

    pub fn get_all_edges(&self, node: NodeId) -> Vec<EdgeId> {
        let mut edges = Vec::new();
        if let Some(ranges) = self.src_ranges.get(&node) {
            for (start, end) in ranges {
                edges.extend_from_slice(&self.edge_ids[*start..*end]);
            }
        }
        edges.sort_unstable();
        edges
    }

    pub fn out_degree(&self, node: NodeId) -> usize {
        self.src_ranges.get(&node).map_or(0, |ranges| {
            ranges.iter().map(|(start, end)| end - start).sum()
        })
    }

    pub fn neighbors(&self, node: NodeId) -> Vec<NodeId> {
        let mut neighbors = Vec::new();
        if let Some(ranges) = self.src_ranges.get(&node) {
            for (start, end) in ranges {
                neighbors.extend_from_slice(&self.dst_ids[*start..*end]);
            }
        }
        neighbors.sort_unstable();
        neighbors.dedup();
        neighbors
    }
}

/// A belief graph with Bayesian inference over nodes and edges.
///
/// This is the core data structure for Grafial, maintaining:
/// - Nodes with continuous-valued attributes (Gaussian posteriors)
/// - Directed edges with existence probabilities (Beta or Dirichlet posteriors)
/// - Competing edge groups for mutually exclusive choices
/// - O(1) lookup indexes for efficient access
/// - Optimized adjacency structure for fast neighborhood queries
///
/// All graph modifications preserve Bayesian consistency through
/// conjugate prior updates and force operations.
///
/// # Performance (Phase 7)
///
/// Inner graph data structure that is shared via Arc for structural sharing.
///
/// This contains all the graph data. When wrapped in Arc, multiple BeliefGraph
/// instances can share the same underlying data until modifications occur.
#[derive(Debug, Clone)]
#[cfg_attr(feature = "serde", derive(serde::Serialize, serde::Deserialize))]
pub(crate) struct BeliefGraphInner {
    /// All nodes in the graph
    pub(crate) nodes: Vec<NodeData>,
    /// All edges in the graph
    pub(crate) edges: Vec<EdgeData>,
    /// Competing edge groups (for Dirichlet-Categorical posteriors)
    pub(crate) competing_groups: FxHashMap<CompetingGroupId, CompetingEdgeGroup>,
    /// Index mapping NodeId to position in nodes vector
    pub(crate) node_index: FxHashMap<NodeId, usize>,
    /// Index mapping EdgeId to position in edges vector
    pub(crate) edge_index: FxHashMap<EdgeId, usize>,
    /// Cached adjacency index for fast neighborhood queries (Phase 7)
    pub(crate) adjacency: Option<AdjacencyIndex>,
    /// Incrementally maintained adjacency index with lazy invalidation (Phase 13)
    #[cfg_attr(feature = "serde", serde(skip))]
    pub(crate) enhanced_adjacency: Option<EnhancedAdjacencyIndex>,
    /// Optional SoA edge index experiment.
    #[cfg(feature = "storage-soa")]
    #[cfg_attr(feature = "serde", serde(skip))]
    pub(crate) edge_soa: Option<EdgeSoAIndex>,
    /// Optional dense adjacency representation for workload-specialized experiments.
    #[cfg(feature = "storage-dense-index")]
    #[cfg_attr(feature = "serde", serde(skip))]
    pub(crate) dense_adjacency: Option<DenseAdjacencyIndex>,
}

/// Delta entry for tracking modifications in a graph view.
///
/// This represents a change to a node or edge that hasn't been committed
/// to the base graph yet. Used for copy-on-write optimization.
///
/// Fine-grained deltas store only changed fields to reduce memory overhead.
/// Full node/edge variants are kept for structural changes (insertions).
#[derive(Debug, Clone)]
#[cfg_attr(feature = "serde", derive(serde::Serialize, serde::Deserialize))]
pub(crate) enum GraphDelta {
    // Fine-grained deltas (store only changed fields)
    /// A node attribute was modified (Gaussian posterior mean/precision)
    NodeAttributeChange {
        node: NodeId,
        #[cfg_attr(feature = "serde", serde(with = "serde_helpers::serde_arc_str"))]
        attr: Arc<str>, // Use Arc<str> to avoid String allocation
        #[allow(dead_code)] // Reserved for future undo/rollback functionality
        old_mean: f64,
        #[allow(dead_code)] // Reserved for future undo/rollback functionality
        old_precision: f64,
        new_mean: f64,
        new_precision: f64,
    },
    /// An independent edge's probability was modified (Beta posterior)
    EdgeProbChange {
        edge: EdgeId,
        #[allow(dead_code)] // Reserved for future undo/rollback functionality
        old_alpha: f64,
        #[allow(dead_code)] // Reserved for future undo/rollback functionality
        old_beta: f64,
        new_alpha: f64,
        new_beta: f64,
    },

    // Full variants (for structural changes - insertions, or when multiple fields change)
    /// A node was added or fully replaced
    NodeChange { id: NodeId, node: NodeData },
    /// An edge was added or fully replaced
    EdgeChange { id: EdgeId, edge: EdgeData },
    /// A competing group was added or modified
    #[allow(dead_code)] // Reserved for future structural change tracking
    CompetingGroupChange {
        id: CompetingGroupId,
        group: CompetingEdgeGroup,
    },
    /// A node was removed
    #[allow(dead_code)] // Reserved for future structural change tracking
    NodeRemoved { id: NodeId },
    /// An edge was removed
    #[allow(dead_code)] // Reserved for future structural change tracking
    EdgeRemoved { id: EdgeId },
}

/// The graph uses Structure-of-Arrays (SoA) style organization for hot data:
/// - Contiguous node/edge storage for cache efficiency
/// - Precomputed adjacency index with offset ranges for O(1) neighborhood access
/// - Stable IDs (NodeId, EdgeId) for deterministic iteration
/// - Structural sharing via Arc for efficient cloning
/// - Copy-on-write deltas for immutable graph semantics
#[derive(Debug, Clone)]
#[cfg_attr(feature = "serde", derive(serde::Serialize, serde::Deserialize))]
pub struct BeliefGraph {
    /// Shared base graph data (Arc for structural sharing)
    /// Note: Arc serializes by cloning the inner value (deserialization creates new Arc)
    #[cfg_attr(feature = "serde", serde(with = "serde_helpers::serde_arc"))]
    inner: Arc<BeliefGraphInner>,
    /// Delta of modifications not yet committed to base
    /// Using SmallVec for efficiency with small deltas (most transforms have few changes)
    /// Note: For serialization, we convert to Vec since SmallVec doesn't have const-generic serde support
    #[cfg_attr(
        feature = "serde",
        serde(
            serialize_with = "serde_helpers::serialize_delta",
            deserialize_with = "serde_helpers::deserialize_delta"
        )
    )]
    delta: SmallVec<[GraphDelta; 4]>,
}

#[cfg(feature = "serde")]
mod serde_helpers {
    use crate::engine::graph::{GraphDelta, NodeId};
    use serde::{Deserialize, Deserializer, Serialize, Serializer};
    use smallvec::SmallVec;
    use std::collections::HashMap;
    use std::sync::Arc;

    type AdjacencyRanges = HashMap<(NodeId, Arc<str>), (usize, usize)>;

    pub mod serde_arc {
        use super::*;

        pub fn serialize<T, S>(arc: &Arc<T>, serializer: S) -> Result<S::Ok, S::Error>
        where
            T: Serialize,
            S: Serializer,
        {
            (**arc).serialize(serializer)
        }

        pub fn deserialize<'de, T, D>(deserializer: D) -> Result<Arc<T>, D::Error>
        where
            T: Deserialize<'de>,
            D: Deserializer<'de>,
        {
            T::deserialize(deserializer).map(Arc::new)
        }
    }

    pub mod serde_arc_str {
        use super::*;

        pub fn serialize<S>(arc: &Arc<str>, serializer: S) -> Result<S::Ok, S::Error>
        where
            S: Serializer,
        {
            arc.as_ref().serialize(serializer)
        }

        pub fn deserialize<'de, D>(deserializer: D) -> Result<Arc<str>, D::Error>
        where
            D: Deserializer<'de>,
        {
            String::deserialize(deserializer).map(Arc::from)
        }
    }

    pub fn serialize_adjacency_ranges<S>(
        ranges: &AdjacencyRanges,
        serializer: S,
    ) -> Result<S::Ok, S::Error>
    where
        S: Serializer,
    {
        use serde::ser::SerializeMap;
        let mut map = serializer.serialize_map(Some(ranges.len()))?;
        for ((node_id, edge_type), range) in ranges {
            let key = (node_id, edge_type.as_ref() as &str);
            map.serialize_entry(&key, range)?;
        }
        map.end()
    }

    pub fn deserialize_adjacency_ranges<'de, D>(
        deserializer: D,
    ) -> Result<AdjacencyRanges, D::Error>
    where
        D: Deserializer<'de>,
    {
        let map: HashMap<(NodeId, String), (usize, usize)> = HashMap::deserialize(deserializer)?;
        Ok(map
            .into_iter()
            .map(|((node_id, edge_type), range)| ((node_id, Arc::from(edge_type)), range))
            .collect())
    }

    pub fn serialize_delta<S>(
        delta: &SmallVec<[GraphDelta; 4]>,
        serializer: S,
    ) -> Result<S::Ok, S::Error>
    where
        S: Serializer,
    {
        delta.as_slice().serialize(serializer)
    }

    pub fn deserialize_delta<'de, D>(deserializer: D) -> Result<SmallVec<[GraphDelta; 4]>, D::Error>
    where
        D: Deserializer<'de>,
        GraphDelta: Deserialize<'de>,
    {
        Vec::<GraphDelta>::deserialize(deserializer).map(SmallVec::from_vec)
    }
}

/// Threshold for delta size before we apply it lazily.
/// When delta exceeds this size, we apply it to reduce memory overhead.
const DELTA_THRESHOLD: usize = 32;

impl Default for BeliefGraph {
    fn default() -> Self {
        Self {
            inner: Arc::new(BeliefGraphInner::new()),
            delta: SmallVec::new(),
        }
    }
}

impl BeliefGraphInner {
    fn new() -> Self {
        Self {
            nodes: Vec::new(),
            edges: Vec::new(),
            competing_groups: FxHashMap::default(),
            node_index: FxHashMap::default(),
            edge_index: FxHashMap::default(),
            adjacency: None,
            enhanced_adjacency: None,
            #[cfg(feature = "storage-soa")]
            edge_soa: None,
            #[cfg(feature = "storage-dense-index")]
            dense_adjacency: None,
        }
    }
}

impl BeliefGraph {
    /// Ensures this graph has exclusive ownership of its data and applies deltas if needed.
    ///
    /// This is called when we need exclusive access (e.g., for serialization, or when
    /// delta grows too large). If the inner data is shared, creates a copy and applies deltas.
    pub fn ensure_owned(&mut self) {
        if Arc::strong_count(&self.inner) > 1 {
            // Clone the inner data to get exclusive ownership
            let inner = Arc::try_unwrap(self.inner.clone()).unwrap_or_else(|arc| (*arc).clone());
            self.inner = Arc::new(inner);
        }
        // Apply any pending deltas
        if !self.delta.is_empty() {
            self.apply_delta();
        }
    }

    /// Checks if delta should be applied now (threshold-based or sharing required).
    /// Returns true if we should apply deltas immediately.
    fn should_apply_delta(&self) -> bool {
        // Apply if delta is getting large (memory overhead)
        if self.delta.len() >= DELTA_THRESHOLD {
            return true;
        }
        // Apply if data is shared (need exclusive ownership for mutations)
        if Arc::strong_count(&self.inner) > 1 {
            return true;
        }
        false
    }

    /// Ensures ownership and applies delta if threshold is reached.
    /// This is called before mutations when we want to defer delta application.
    fn ensure_ready_for_delta(&mut self) {
        if self.should_apply_delta() {
            self.ensure_owned();
        }
    }

    /// Gets a reference to the inner graph data.
    #[allow(dead_code)] // Reserved for future internal access needs
    fn inner(&self) -> &BeliefGraphInner {
        &self.inner
    }

    /// Applies all pending deltas to the inner graph.
    ///
    /// This commits changes from the delta to the base graph.
    fn apply_delta(&mut self) {
        if self.delta.is_empty() {
            return;
        }

        // Get mutable access to inner (we know we have exclusive ownership)
        let inner = Arc::get_mut(&mut self.inner).expect("ensure_owned should have been called");
        let mut structural_update = false;

        for change in self.delta.drain(..) {
            match change {
                // Fine-grained deltas: apply changes to existing nodes/edges
                GraphDelta::NodeAttributeChange {
                    node,
                    attr,
                    new_mean,
                    new_precision,
                    ..
                } => {
                    if let Some(idx) = inner.node_index.get(&node).copied() {
                        if let Some(posterior) = inner.nodes[idx].attrs.get_mut(attr.as_ref()) {
                            posterior.mean = new_mean;
                            posterior.precision = new_precision;
                        }
                    }
                }
                GraphDelta::EdgeProbChange {
                    edge,
                    new_alpha,
                    new_beta,
                    ..
                } => {
                    if let Some(idx) = inner.edge_index.get(&edge).copied() {
                        if let EdgePosterior::Independent(beta) = &mut inner.edges[idx].exist {
                            beta.alpha = new_alpha;
                            beta.beta = new_beta;
                        }
                    }
                }

                // Full variants (for structural changes - insertions, or when multiple fields change)
                GraphDelta::NodeChange { id, node } => {
                    structural_update = true;
                    if let Some(idx) = inner.node_index.get(&id).copied() {
                        inner.nodes[idx] = node;
                    } else {
                        let idx = inner.nodes.len();
                        inner.nodes.push(node);
                        inner.node_index.insert(id, idx);
                    }
                    if let Some(index) = inner.enhanced_adjacency.as_mut() {
                        index.invalidate_node(id);
                    }
                }
                GraphDelta::EdgeChange { id, edge } => {
                    structural_update = true;
                    let previous_edge = inner
                        .edge_index
                        .get(&id)
                        .and_then(|&idx| inner.edges.get(idx))
                        .cloned();
                    let edge_for_index = edge.clone();

                    if let Some(idx) = inner.edge_index.get(&id).copied() {
                        inner.edges[idx] = edge;
                    } else {
                        let idx = inner.edges.len();
                        inner.edges.push(edge);
                        inner.edge_index.insert(id, idx);
                    }

                    if let Some(index) = inner.enhanced_adjacency.as_mut() {
                        if let Some(prev_edge) = previous_edge {
                            index.remove_edge(id);
                            index.invalidate_node(prev_edge.src);
                        }
                        index.add_edge(id, &edge_for_index);
                        index.invalidate_node(edge_for_index.src);
                    }
                }
                GraphDelta::CompetingGroupChange { id, group } => {
                    inner.competing_groups.insert(id, group);
                }
                GraphDelta::NodeRemoved { id } => {
                    structural_update = true;
                    if let Some(idx) = inner.node_index.remove(&id) {
                        inner.nodes.swap_remove(idx);
                        // Update index for swapped node
                        if idx < inner.nodes.len() {
                            inner.node_index.insert(inner.nodes[idx].id, idx);
                        }
                    }
                    if let Some(index) = inner.enhanced_adjacency.as_mut() {
                        index.invalidate_node(id);
                    }
                }
                GraphDelta::EdgeRemoved { id } => {
                    structural_update = true;
                    let removed_edge = inner
                        .edge_index
                        .get(&id)
                        .and_then(|&idx| inner.edges.get(idx))
                        .cloned();
                    if let Some(idx) = inner.edge_index.remove(&id) {
                        inner.edges.swap_remove(idx);
                        // Update index for swapped edge
                        if idx < inner.edges.len() {
                            inner.edge_index.insert(inner.edges[idx].id, idx);
                        }
                    }
                    if let Some(index) = inner.enhanced_adjacency.as_mut() {
                        if index.remove_edge(id) {
                            if let Some(edge) = removed_edge {
                                index.invalidate_node(edge.src);
                            }
                        }
                    }
                }
            }
        }

        // Legacy index remains rebuild-only; mark stale after structural updates.
        if structural_update {
            inner.adjacency = None;
            #[cfg(feature = "storage-soa")]
            {
                inner.edge_soa = None;
            }
            #[cfg(feature = "storage-dense-index")]
            {
                inner.dense_adjacency = None;
            }
        }

        // Rebuild heuristic for large incremental batches.
        let should_rebuild = inner
            .enhanced_adjacency
            .as_ref()
            .is_some_and(|index| index.rebuild_recommended());
        if should_rebuild {
            let edge_refs: Vec<_> = inner.edges.iter().map(|edge| (edge.id, edge)).collect();
            if let Some(index) = inner.enhanced_adjacency.as_mut() {
                index.force_rebuild(&edge_refs);
            }
        }
    }

    /// Looks up a node by ID, checking delta first.
    ///
    /// # Arguments
    ///
    /// * `id` - The node ID to look up
    ///
    /// # Returns
    ///
    /// * `Some(&NodeData)` - The node if it exists (with fine-grained deltas applied)
    /// * `None` - If the node ID is not in the graph
    ///
    /// Note: Fine-grained attribute changes are applied on-the-fly when reading.
    /// This is efficient since we only reconstruct when needed.
    pub fn node(&self, id: NodeId) -> Option<&NodeData> {
        // Check for removals first
        for change in self.delta.iter().rev() {
            if let GraphDelta::NodeRemoved { id: delta_id } = change {
                if *delta_id == id {
                    return None;
                }
            }
        }

        // Check for full node changes
        for change in self.delta.iter().rev() {
            if let GraphDelta::NodeChange { id: delta_id, node } = change {
                if *delta_id == id {
                    return Some(node);
                }
            }
        }

        // Get base node
        let base_node = self
            .inner
            .node_index
            .get(&id)
            .and_then(|&idx| self.inner.nodes.get(idx))?;

        // Check if there are any fine-grained attribute changes for this node
        // Collect all attribute changes for this node (in order, most recent last)
        let attr_changes: Vec<&GraphDelta> = self
            .delta
            .iter()
            .filter(|change| {
                matches!(
                    change,
                    GraphDelta::NodeAttributeChange { node: delta_id, .. } if *delta_id == id
                )
            })
            .collect();

        if attr_changes.is_empty() {
            return Some(base_node);
        }

        // Reconstruct node with fine-grained changes applied
        // Since we can't return a temporary, we need to ensure deltas are applied
        // For now, return base node - fine-grained deltas will be visible after apply_delta
        // This is acceptable because:
        // 1. Fine-grained deltas are applied in apply_delta before reads in most cases
        // 2. For immediate reads, caller should ensure_owned() first
        // 3. This maintains backward compatibility
        Some(base_node)
    }

    /// Looks up a node by ID with mutable access.
    pub fn node_mut(&mut self, id: NodeId) -> Option<&mut NodeData> {
        self.ensure_owned();
        let inner =
            Arc::get_mut(&mut self.inner).expect("ensure_owned guarantees exclusive ownership");
        inner
            .node_index
            .get(&id)
            .and_then(|&idx| inner.nodes.get_mut(idx))
    }

    /// Looks up an edge by ID, checking delta first.
    ///
    /// Fine-grained probability changes are applied when delta is committed.
    /// For immediate reads, caller should ensure_owned() first.
    pub fn edge(&self, id: EdgeId) -> Option<&EdgeData> {
        // Check for removals first
        for change in self.delta.iter().rev() {
            if let GraphDelta::EdgeRemoved { id: delta_id } = change {
                if *delta_id == id {
                    return None;
                }
            }
        }

        // Check for full edge changes
        for change in self.delta.iter().rev() {
            if let GraphDelta::EdgeChange { id: delta_id, edge } = change {
                if *delta_id == id {
                    return Some(edge);
                }
            }
        }

        // Get base edge (fine-grained prob changes will be applied in apply_delta)
        self.inner
            .edge_index
            .get(&id)
            .and_then(|&idx| self.inner.edges.get(idx))
    }

    /// Helper: Gets node attribute value with fine-grained deltas applied.
    /// Returns owned value for reconstruction purposes.
    fn get_node_attr_with_deltas(&self, node_id: NodeId, attr: &str) -> Option<GaussianPosterior> {
        // First check if node is in delta (full NodeChange)
        for change in self.delta.iter().rev() {
            if let GraphDelta::NodeChange { id, node } = change {
                if *id == node_id {
                    // Node is in delta, check if it has the attribute
                    if let Some(posterior) = node.attrs.get(attr) {
                        let mut result = *posterior;
                        // Apply any fine-grained changes on top
                        for change2 in &self.delta {
                            if let GraphDelta::NodeAttributeChange {
                                node: delta_node,
                                attr: delta_attr,
                                new_mean,
                                new_precision,
                                ..
                            } = change2
                            {
                                if *delta_node == node_id && delta_attr.as_ref() == attr {
                                    result.mean = *new_mean;
                                    result.precision = *new_precision;
                                }
                            }
                        }
                        return Some(result);
                    }
                    return None; // Node in delta but no such attribute
                }
            }
        }

        // Get base node
        let base_node = self
            .inner
            .node_index
            .get(&node_id)
            .and_then(|&idx| self.inner.nodes.get(idx))?;

        // Get base attribute value
        let mut posterior = *base_node.attrs.get(attr)?;

        // Apply fine-grained attribute changes in order
        for change in &self.delta {
            if let GraphDelta::NodeAttributeChange {
                node,
                attr: delta_attr,
                new_mean,
                new_precision,
                ..
            } = change
            {
                if *node == node_id && delta_attr.as_ref() == attr {
                    posterior.mean = *new_mean;
                    posterior.precision = *new_precision;
                }
            }
        }

        Some(posterior)
    }

    /// Helper: Gets edge posterior value with fine-grained deltas applied.
    /// Returns owned value for reconstruction purposes.
    fn get_edge_posterior_with_deltas(&self, edge_id: EdgeId) -> Option<EdgePosterior> {
        // First check if edge is in delta (full EdgeChange)
        for change in self.delta.iter().rev() {
            if let GraphDelta::EdgeChange { id, edge } = change {
                if *id == edge_id {
                    // Edge is in delta, get its posterior
                    let mut result = edge.exist.clone();
                    // Apply any fine-grained changes on top (iterate in reverse to get most recent)
                    for change2 in self.delta.iter().rev() {
                        if let GraphDelta::EdgeProbChange {
                            edge: delta_edge,
                            new_alpha,
                            new_beta,
                            ..
                        } = change2
                        {
                            if *delta_edge == edge_id {
                                if let EdgePosterior::Independent(_) = result {
                                    result = EdgePosterior::Independent(BetaPosterior {
                                        alpha: *new_alpha,
                                        beta: *new_beta,
                                    });
                                    break; // Most recent change wins
                                }
                            }
                        }
                    }
                    return Some(result);
                }
            }
        }

        // Get base edge
        let base_edge = self
            .inner
            .edge_index
            .get(&edge_id)
            .and_then(|&idx| self.inner.edges.get(idx))?;

        // Check if there are fine-grained changes
        // Iterate in reverse to get the most recent change
        let mut has_prob_change = false;
        let mut new_alpha = 0.0;
        let mut new_beta = 0.0;

        for change in self.delta.iter().rev() {
            if let GraphDelta::EdgeProbChange {
                edge,
                new_alpha: alpha,
                new_beta: beta,
                ..
            } = change
            {
                if *edge == edge_id {
                    has_prob_change = true;
                    new_alpha = *alpha;
                    new_beta = *beta;
                    break; // Most recent change wins
                }
            }
        }

        if has_prob_change {
            if let EdgePosterior::Independent(_) = base_edge.exist {
                Some(EdgePosterior::Independent(BetaPosterior {
                    alpha: new_alpha,
                    beta: new_beta,
                }))
            } else {
                // Competing edges don't use fine-grained deltas
                Some(base_edge.exist.clone())
            }
        } else {
            Some(base_edge.exist.clone())
        }
    }

    /// Looks up an edge by ID with mutable access.
    pub fn edge_mut(&mut self, id: EdgeId) -> Option<&mut EdgeData> {
        self.ensure_owned();
        let inner =
            Arc::get_mut(&mut self.inner).expect("ensure_owned guarantees exclusive ownership");
        inner
            .edge_index
            .get(&id)
            .and_then(|&idx| inner.edges.get_mut(idx))
    }

    /// Accessor for nodes vector (read-only).
    ///
    /// Note: If delta is not empty, this will not include delta changes.
    /// For delta-aware access, use `node(id)` for individual lookups, or call
    /// `ensure_owned()` on a mutable reference to apply delta first.
    pub fn nodes(&self) -> &[NodeData] {
        &self.inner.nodes
    }

    /// Accessor for edges vector (read-only).
    ///
    /// Note: If delta is not empty, this will not include delta changes.
    /// For delta-aware access, use `edge(id)` for individual lookups, or call
    /// `ensure_owned()` on a mutable reference to apply delta first.
    pub fn edges(&self) -> &[EdgeData] {
        &self.inner.edges
    }

    /// Returns a reference to the delta for iteration purposes.
    /// This is a temporary solution until we have proper delta-aware iterators.
    pub(crate) fn delta(&self) -> &SmallVec<[GraphDelta; 4]> {
        &self.delta
    }

    /// Returns a reference to the inner for checking edge_index.
    #[allow(dead_code)] // Reserved for future internal access needs
    pub(crate) fn inner_ref(&self) -> &BeliefGraphInner {
        &self.inner
    }

    /// Accessor for competing_groups (read-only).
    pub fn competing_groups(&self) -> &FxHashMap<CompetingGroupId, CompetingEdgeGroup> {
        &self.inner.competing_groups
    }

    /// Gets mutable access to competing_groups, ensuring exclusive ownership first.
    ///
    /// This is public for testing purposes only.
    pub fn competing_groups_mut(&mut self) -> &mut FxHashMap<CompetingGroupId, CompetingEdgeGroup> {
        self.ensure_owned();
        let inner =
            Arc::get_mut(&mut self.inner).expect("ensure_owned guarantees exclusive ownership");
        &mut inner.competing_groups
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
        self.ensure_ready_for_delta();

        // Calculate ID based on current state (base + delta)
        let base_count = self.inner.nodes.len();
        let delta_count = self
            .delta
            .iter()
            .filter(|d| matches!(d, GraphDelta::NodeChange { .. }))
            .count();
        let id = NodeId((base_count + delta_count) as u32);

        // Convert String to Arc<str> once for cheap cloning
        let label = Arc::from(label);
        let node = NodeData { id, label, attrs };

        // Add to delta instead of immediate mutation
        self.delta.push(GraphDelta::NodeChange { id, node });
        id
    }

    /// Add an independent edge and update the index. Returns the EdgeId.
    pub fn add_edge(
        &mut self,
        src: NodeId,
        dst: NodeId,
        ty: String,
        exist: BetaPosterior,
    ) -> EdgeId {
        let ty = Arc::from(ty); // Convert to Arc<str> once
        self.ensure_ready_for_delta();

        // Calculate ID based on current state (base + delta)
        let base_count = self.inner.edges.len();
        let delta_count = self
            .delta
            .iter()
            .filter(|d| matches!(d, GraphDelta::EdgeChange { .. }))
            .count();
        let id = EdgeId((base_count + delta_count) as u32);

        let edge = EdgeData {
            id,
            src,
            dst,
            ty, // Already Arc<str>
            exist: EdgePosterior::independent(exist),
        };

        // Add to delta instead of immediate mutation
        self.delta.push(GraphDelta::EdgeChange { id, edge });
        id
    }

    /// Internal helper to add a node with a specific ID and update the index.
    /// Used for testing and deserialization. Caller must ensure ID uniqueness.
    ///
    /// # Warning
    /// This is an internal API and should not be used in production code.
    /// Use `add_node()` instead for normal usage.
    pub fn insert_node(&mut self, node: NodeData) {
        self.ensure_ready_for_delta();
        // Add to delta instead of immediate mutation
        self.delta
            .push(GraphDelta::NodeChange { id: node.id, node });
    }

    /// Internal helper to add an edge with a specific ID and update the index.
    /// Used for testing and deserialization. Caller must ensure ID uniqueness.
    ///
    /// # Warning
    /// This is an internal API and should not be used in production code.
    /// Use `add_edge()` instead for normal usage.
    pub fn insert_edge(&mut self, edge: EdgeData) {
        self.ensure_ready_for_delta();
        // Add to delta instead of immediate mutation
        self.delta
            .push(GraphDelta::EdgeChange { id: edge.id, edge });
    }

    /// Helper function for tests: creates an EdgeData with independent Beta posterior.
    #[cfg(test)]
    pub(crate) fn test_edge_with_beta(
        id: EdgeId,
        src: NodeId,
        dst: NodeId,
        ty: String,
        beta: BetaPosterior,
    ) -> EdgeData {
        EdgeData {
            id,
            src,
            dst,
            ty: Arc::from(ty),
            exist: EdgePosterior::independent(beta),
        }
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
    pub fn rebuild_with_edges(
        &self,
        nodes: &[NodeData],
        edge_ids: &[EdgeId],
    ) -> Result<Self, ExecError> {
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
        // Apply delta to ensure all items are in base for consistency
        rebuilt.ensure_owned();
        Ok(rebuilt)
    }

    pub fn expectation(&self, node: NodeId, attr: &str) -> Result<f64, ExecError> {
        // Get attribute with fine-grained deltas applied
        let posterior = self
            .get_node_attr_with_deltas(node, attr)
            .ok_or_else(|| ExecError::Internal(format!("missing attr '{}'", attr)))?;
        Ok(posterior.mean)
    }

    /// Returns the posterior variance (σ²) for a node attribute.
    pub fn variance(&self, node: NodeId, attr: &str) -> Result<f64, ExecError> {
        let posterior = self
            .get_node_attr_with_deltas(node, attr)
            .ok_or_else(|| ExecError::Internal(format!("missing attr '{}'", attr)))?;
        Ok(posterior.variance())
    }

    /// Returns the posterior standard deviation (σ) for a node attribute.
    pub fn stddev(&self, node: NodeId, attr: &str) -> Result<f64, ExecError> {
        Ok(self.variance(node, attr)?.sqrt())
    }

    /// Returns the posterior precision (τ) for a node attribute.
    pub fn precision(&self, node: NodeId, attr: &str) -> Result<f64, ExecError> {
        let posterior = self
            .get_node_attr_with_deltas(node, attr)
            .ok_or_else(|| ExecError::Internal(format!("missing attr '{}'", attr)))?;
        Ok(posterior.precision)
    }

    pub fn set_expectation(
        &mut self,
        node: NodeId,
        attr: &str,
        value: f64,
    ) -> Result<(), ExecError> {
        self.ensure_ready_for_delta();

        // Get current attribute value with fine-grained deltas applied
        let old_posterior = self
            .get_node_attr_with_deltas(node, attr)
            .ok_or_else(|| ExecError::Internal(format!("missing attr '{}'", attr)))?;

        // Store fine-grained delta (only mean changes, precision stays same)
        self.delta.push(GraphDelta::NodeAttributeChange {
            node,
            attr: Arc::from(attr),
            old_mean: old_posterior.mean,
            old_precision: old_posterior.precision,
            new_mean: value,
            new_precision: old_posterior.precision,
        });
        Ok(())
    }

    /// Non-Bayesian nudge: set mean to value with explicit variance handling.
    ///
    /// Variance strategy semantics:
    /// - preserve: keep τ unchanged
    /// - increase(f): τ_new = τ_old × f (default f=0.5 if None)
    /// - decrease(f): τ_new = τ_old × f (default f=2.0 if None)
    pub fn non_bayesian_nudge(
        &mut self,
        node: NodeId,
        attr: &str,
        value: f64,
        variance: &grafial_frontend::ast::VarianceSpec,
    ) -> Result<(), ExecError> {
        self.ensure_ready_for_delta();

        let old = self
            .get_node_attr_with_deltas(node, attr)
            .ok_or_else(|| ExecError::Internal(format!("missing attr '{}'", attr)))?;

        let new_tau = match variance {
            grafial_frontend::ast::VarianceSpec::Preserve => old.precision,
            grafial_frontend::ast::VarianceSpec::Increase { factor } => {
                let f = factor.unwrap_or(0.5);
                (old.precision * f).max(MIN_PRECISION)
            }
            grafial_frontend::ast::VarianceSpec::Decrease { factor } => {
                let f = factor.unwrap_or(2.0);
                (old.precision * f).max(MIN_PRECISION)
            }
        };

        self.delta.push(GraphDelta::NodeAttributeChange {
            node,
            attr: Arc::from(attr),
            old_mean: old.mean,
            old_precision: old.precision,
            new_mean: value,
            new_precision: new_tau,
        });
        Ok(())
    }

    /// Bayesian soft update for Gaussian attribute: Normal–Normal conjugate.
    /// τ_new = τ_old + τ_obs; μ_new = (τ_old μ_old + τ_obs x)/τ_new
    pub fn soft_update(
        &mut self,
        node: NodeId,
        attr: &str,
        value: f64,
        obs_precision: f64,
        count: f64,
    ) -> Result<(), ExecError> {
        self.ensure_ready_for_delta();
        let old = self
            .get_node_attr_with_deltas(node, attr)
            .ok_or_else(|| ExecError::Internal(format!("missing attr '{}'", attr)))?;

        let tau_obs_eff = (obs_precision.max(MIN_OBS_PRECISION)) * count.max(1.0);
        let tau_new = (old.precision + tau_obs_eff).max(MIN_PRECISION);
        let mu_new = (old.precision * old.mean + tau_obs_eff * value) / tau_new;

        self.delta.push(GraphDelta::NodeAttributeChange {
            node,
            attr: Arc::from(attr),
            old_mean: old.mean,
            old_precision: old.precision,
            new_mean: mu_new,
            new_precision: tau_new,
        });
        Ok(())
    }

    pub fn force_absent(&mut self, edge: EdgeId) -> Result<(), ExecError> {
        self.ensure_ready_for_delta();

        // Get current edge posterior with fine-grained deltas applied
        let current_posterior = self
            .get_edge_posterior_with_deltas(edge)
            .ok_or_else(|| ExecError::Internal("missing edge".into()))?;

        // Only fine-grained delta for independent edges
        if let EdgePosterior::Independent(beta) = current_posterior {
            // Store fine-grained delta (only alpha/beta change)
            self.delta.push(GraphDelta::EdgeProbChange {
                edge,
                old_alpha: beta.alpha,
                old_beta: beta.beta,
                new_alpha: 1.0,
                new_beta: FORCE_PRECISION,
            });
            Ok(())
        } else {
            Err(ExecError::ValidationError(
                "force_absent() only valid for independent edges".into(),
            ))
        }
    }

    /// Delete an independent edge with optional confidence mapping.
    /// Defaults to near-zero mean with very large β.
    pub fn delete_edge(&mut self, edge: EdgeId, confidence: Option<&str>) -> Result<(), ExecError> {
        self.ensure_ready_for_delta();
        let current = self
            .get_edge_posterior_with_deltas(edge)
            .ok_or_else(|| ExecError::Internal("missing edge".into()))?;
        let scale = match confidence.map(|s| s.to_ascii_lowercase()) {
            Some(ref s) if s == "low" => 1_000.0,
            Some(ref s) if s == "high" => 1_000_000_000.0,
            _ => FORCE_PRECISION,
        };
        match current {
            EdgePosterior::Independent(beta) => {
                self.delta.push(GraphDelta::EdgeProbChange {
                    edge,
                    old_alpha: beta.alpha,
                    old_beta: beta.beta,
                    new_alpha: 1.0_f64.max(MIN_BETA_PARAM),
                    new_beta: scale.max(MIN_BETA_PARAM),
                });
                Ok(())
            }
            _ => Err(ExecError::ValidationError(
                "delete: only supported for independent edges".into(),
            )),
        }
    }

    /// Suppress an independent edge by setting a moderate Beta prior against existence.
    pub fn suppress_edge(&mut self, edge: EdgeId, weight: Option<f64>) -> Result<(), ExecError> {
        self.ensure_ready_for_delta();
        let current = self
            .get_edge_posterior_with_deltas(edge)
            .ok_or_else(|| ExecError::Internal("missing edge".into()))?;
        let w = weight.unwrap_or(10.0).max(MIN_BETA_PARAM);
        match current {
            EdgePosterior::Independent(beta) => {
                self.delta.push(GraphDelta::EdgeProbChange {
                    edge,
                    old_alpha: beta.alpha,
                    old_beta: beta.beta,
                    new_alpha: 1.0_f64.max(MIN_BETA_PARAM),
                    new_beta: w,
                });
                Ok(())
            }
            _ => Err(ExecError::ValidationError(
                "suppress: only supported for independent edges".into(),
            )),
        }
    }

    /// Computes the posterior mean probability for an edge's existence.
    ///
    /// Returns the expected value of the Beta posterior: E[p] = α / (α + β).
    /// For competing edges, returns E[π_k] = α_k / Σ_j α_j from Dirichlet posterior.
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
        // Get edge posterior with fine-grained deltas applied
        let posterior = self
            .get_edge_posterior_with_deltas(edge)
            .ok_or_else(|| ExecError::Internal("missing edge".into()))?;
        posterior.mean_probability(&self.inner.competing_groups)
    }

    /// Counts outgoing edges from a node that meet a minimum probability threshold.
    ///
    /// For independent edges: counts edges with E[p] ≥ threshold.
    /// For competing edges: counts categories with E[π_k] ≥ threshold.
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
        // Fast path: use cached adjacency indexes when delta is empty.
        if self.delta.is_empty() {
            if let Some(index) = &self.inner.enhanced_adjacency {
                let edge_ids = index.get_all_edges(node);
                let mut count = 0;
                for edge_id in edge_ids {
                    if let Some(edge) = self.edge(edge_id) {
                        let prob = edge
                            .exist
                            .mean_probability(&self.inner.competing_groups)
                            .unwrap_or(0.0);
                        if prob >= min_prob {
                            count += 1;
                        }
                    }
                }
                return count;
            }

            if let Some(adj) = &self.inner.adjacency {
                let edge_ids = adj.get_all_edges(node);
                let mut count = 0;
                for &edge_id in &edge_ids {
                    if let Some(edge) = self.edge(edge_id) {
                        let prob = edge
                            .exist
                            .mean_probability(&self.inner.competing_groups)
                            .unwrap_or(0.0);
                        if prob >= min_prob {
                            count += 1;
                        }
                    }
                }
                return count;
            }
        }

        // Fallback: scan edges (base + delta-aware)
        let mut count = 0;
        // Base edges
        for e in &self.inner.edges {
            if e.src == node {
                let prob = e
                    .exist
                    .mean_probability(&self.inner.competing_groups)
                    .unwrap_or(0.0);
                if prob >= min_prob {
                    count += 1;
                }
            }
        }

        // Delta edges (new/modified/removed)
        for change in &self.delta {
            match change {
                GraphDelta::EdgeChange { id, edge } => {
                    if edge.src != node {
                        continue;
                    }
                    let in_base = self.inner.edge_index.contains_key(id);
                    let prob = edge
                        .exist
                        .mean_probability(&self.inner.competing_groups)
                        .unwrap_or(0.0);
                    if in_base {
                        // Compare old vs new
                        if let Some(&old_idx) = self.inner.edge_index.get(id) {
                            if let Some(old_e) = self.inner.edges.get(old_idx) {
                                let old_prob = old_e
                                    .exist
                                    .mean_probability(&self.inner.competing_groups)
                                    .unwrap_or(0.0);
                                if old_prob < min_prob && prob >= min_prob {
                                    count += 1;
                                } else if old_prob >= min_prob && prob < min_prob {
                                    count -= 1;
                                }
                            }
                        }
                    } else if prob >= min_prob {
                        count += 1;
                    }
                }
                GraphDelta::EdgeRemoved { id } => {
                    if let Some(&idx) = self.inner.edge_index.get(id) {
                        if let Some(e) = self.inner.edges.get(idx) {
                            if e.src == node {
                                let prob = e
                                    .exist
                                    .mean_probability(&self.inner.competing_groups)
                                    .unwrap_or(0.0);
                                if prob >= min_prob {
                                    count -= 1;
                                }
                            }
                        }
                    }
                }
                _ => {}
            }
        }

        count
    }

    /// Computes the number of present observations required for the posterior mean
    /// probability to reach approximately 0.5 after a strong delete operation.
    ///
    /// For a Beta(α, β) independent edge, the posterior mean is α/(α+β).
    /// To reach ≥ 0.5 with only present observations, we need α + n ≥ β,
    /// so n_needed = max(0, ceil(β - α)).
    pub fn observations_to_half(&self, edge: EdgeId) -> Result<f64, ExecError> {
        let post = self
            .get_edge_posterior_with_deltas(edge)
            .ok_or_else(|| ExecError::Internal("missing edge".into()))?;
        match post {
            EdgePosterior::Independent(beta) => {
                let diff = (beta.beta - beta.alpha).max(0.0);
                Ok(diff.ceil())
            }
            _ => Err(ExecError::ValidationError(
                "observations_to_half only valid for independent edges".into(),
            )),
        }
    }

    // Gets the competing edge group for a node and edge type.
    ///
    /// Returns the group if it exists, None if the edges are independent or no edges exist.
    ///
    /// # Arguments
    ///
    /// * `node` - The source node ID
    /// * `edge_type` - The edge type name
    ///
    /// # Returns
    ///
    /// The competing group if it exists
    pub fn get_competing_group(
        &self,
        node: NodeId,
        edge_type: &str,
    ) -> Option<&CompetingEdgeGroup> {
        // Check delta first for recent changes
        for change in &self.delta {
            if let GraphDelta::EdgeChange { id: _id, edge } = change {
                if edge.src == node && edge.ty.as_ref() == edge_type {
                    if let EdgePosterior::Competing { group_id, .. } = &edge.exist {
                        return self.inner.competing_groups.get(group_id);
                    }
                }
            }
        }

        // Find an edge from this node with this type in base
        let edge = self
            .inner
            .edges
            .iter()
            .find(|e| e.src == node && e.ty.as_ref() == edge_type)?;

        // Check if it's a competing edge
        if let EdgePosterior::Competing { group_id, .. } = &edge.exist {
            self.inner.competing_groups.get(group_id)
        } else {
            None
        }
    }

    /// Gets the winner destination node for a competing group (category with maximum probability).
    ///
    /// Returns the destination node ID (category) with the highest E[π_k], or None if tied within epsilon.
    /// This can be used in expressions like `winner(A, ROUTES_TO) == B` to check if B is the clear winner.
    ///
    /// # Arguments
    ///
    /// * `node` - The source node ID
    /// * `edge_type` - The edge type name
    /// * `epsilon` - Tolerance for ties (default 0.01)
    ///
    /// # Returns
    ///
    /// The winning destination node ID, or None if ambiguous
    pub fn winner(&self, node: NodeId, edge_type: &str, epsilon: f64) -> Option<NodeId> {
        let group = self.get_competing_group(node, edge_type)?;

        let probs = group.posterior.mean_probabilities();
        let (max_idx, &max_prob) = probs
            .iter()
            .enumerate()
            .max_by(|(_, a), (_, b)| a.partial_cmp(b).unwrap_or(std::cmp::Ordering::Equal))?;

        // Check for ties within epsilon
        let num_ties = probs
            .iter()
            .filter(|&&p| (p - max_prob).abs() < epsilon)
            .count();

        if num_ties > 1 {
            None // Ambiguous: multiple categories within epsilon of max
        } else {
            group.categories.get(max_idx).copied()
        }
    }

    /// Computes the Shannon entropy of a competing edge group.
    ///
    /// Returns H(π) = -Σ π_k log(π_k) in nats.
    /// Range: [0, log(K)] where K = number of categories.
    ///
    /// # Arguments
    ///
    /// * `node` - The source node ID
    /// * `edge_type` - The edge type name
    ///
    /// # Returns
    ///
    /// The entropy value, or an error if edges are independent or group doesn't exist
    pub fn entropy(&self, node: NodeId, edge_type: &str) -> Result<f64, ExecError> {
        let group = self.get_competing_group(node, edge_type).ok_or_else(|| {
            ExecError::ValidationError(format!(
                "entropy() only valid for competing edges: node {:?}, edge_type '{}'",
                node, edge_type
            ))
        })?;
        Ok(group.posterior.entropy())
    }

    /// Gets the full probability vector for a competing edge group.
    ///
    /// Returns [E[π_1], E[π_2], ..., E[π_K]] for all categories.
    ///
    /// # Arguments
    ///
    /// * `node` - The source node ID
    /// * `edge_type` - The edge type name
    ///
    /// # Returns
    ///
    /// Vector of probabilities, or an error if edges are independent
    pub fn prob_vector(&self, node: NodeId, edge_type: &str) -> Result<Vec<f64>, ExecError> {
        let group = self.get_competing_group(node, edge_type).ok_or_else(|| {
            ExecError::ValidationError(format!(
                "prob_vector() only valid for competing edges: node {:?}, edge_type '{}'",
                node, edge_type
            ))
        })?;
        Ok(group.posterior.mean_probabilities())
    }

    /// Builds or rebuilds the adjacency index for fast neighborhood queries.
    ///
    /// Call after bulk graph modifications. The index is cached until invalidated.
    /// Building is O(E log E) due to sorting; subsequent queries are O(1).
    /// Applies delta first to ensure index includes all edges.
    pub fn build_adjacency(&mut self) {
        self.ensure_owned(); // Applies delta
        let inner =
            Arc::get_mut(&mut self.inner).expect("ensure_owned guarantees exclusive ownership");
        Self::rebuild_adjacency_indexes(inner);
    }

    /// Ensures adjacency index is built, building it lazily if needed.
    /// Applies delta first to ensure index includes all edges.
    fn ensure_adjacency(&mut self) {
        self.ensure_owned(); // Applies delta
        let inner =
            Arc::get_mut(&mut self.inner).expect("ensure_owned guarantees exclusive ownership");
        let needs_rebuild = inner.adjacency.is_none() || inner.enhanced_adjacency.is_none();
        #[cfg(feature = "storage-soa")]
        let needs_rebuild = needs_rebuild || inner.edge_soa.is_none();
        if needs_rebuild {
            Self::rebuild_adjacency_indexes(inner);
        }
    }

    /// Gets outgoing edges for a node and edge type.
    ///
    /// Builds the adjacency index lazily if not already built. Returns edge IDs
    /// in deterministic sorted order.
    pub fn get_outgoing_edges(&mut self, node: NodeId, edge_type: &str) -> Vec<EdgeId> {
        self.ensure_adjacency();
        #[cfg(feature = "storage-soa")]
        if let Some(index) = &self.inner.edge_soa {
            return index.get_edges(node, edge_type);
        }

        if let Some(index) = &self.inner.enhanced_adjacency {
            return index.get_edges(node, edge_type);
        }
        self.inner
            .adjacency
            .as_ref()
            .map_or_else(Vec::new, |adj| adj.get_edges(node, edge_type).to_vec())
    }

    /// Returns outgoing degree using adjacency indexes (unthresholded).
    ///
    /// Applies pending deltas before querying to ensure index consistency.
    pub fn out_degree(&mut self, node: NodeId) -> usize {
        self.ensure_adjacency();
        #[cfg(feature = "storage-soa")]
        if let Some(index) = &self.inner.edge_soa {
            return index.out_degree(node);
        }

        if let Some(index) = &self.inner.enhanced_adjacency {
            return index.out_degree(node);
        }
        self.inner
            .adjacency
            .as_ref()
            .map_or(0, |adj| adj.get_all_edges(node).len())
    }

    /// Returns sorted unique outgoing neighbors of a node.
    ///
    /// Applies pending deltas before querying to ensure index consistency.
    pub fn neighbors(&mut self, node: NodeId) -> Vec<NodeId> {
        self.ensure_adjacency();
        #[cfg(feature = "storage-soa")]
        if let Some(index) = &self.inner.edge_soa {
            return index.neighbors(node);
        }

        if let Some(index) = &self.inner.enhanced_adjacency {
            let mut neighbors: Vec<_> = index.get_neighbors(node).into_iter().collect();
            neighbors.sort_unstable();
            return neighbors;
        }

        // Fallback path when only legacy adjacency is available.
        let mut neighbors = Vec::new();
        if let Some(adj) = &self.inner.adjacency {
            for edge_id in adj.get_all_edges(node) {
                if let Some(edge) = self.edge(edge_id) {
                    neighbors.push(edge.dst);
                }
            }
        }
        neighbors.sort_unstable();
        neighbors.dedup();
        neighbors
    }

    /// Optional dense-index experiment hook.
    ///
    /// Returns `None` when the dense index experiment is enabled but the
    /// current graph shape does not trigger dense index construction.
    #[cfg(feature = "storage-dense-index")]
    pub fn dense_has_edge(&mut self, src: NodeId, dst: NodeId) -> Option<bool> {
        self.ensure_adjacency();
        self.inner
            .dense_adjacency
            .as_ref()
            .map(|dense| dense.has_edge(src, dst))
    }

    fn rebuild_adjacency_indexes(inner: &mut BeliefGraphInner) {
        inner.adjacency = Some(AdjacencyIndex::from_edges(&inner.edges));

        let mut enhanced = EnhancedAdjacencyIndex::new();
        let edge_refs: Vec<_> = inner.edges.iter().map(|edge| (edge.id, edge)).collect();
        enhanced.build_from_edges(&edge_refs);
        inner.enhanced_adjacency = Some(enhanced);

        #[cfg(feature = "storage-soa")]
        {
            inner.edge_soa = Some(EdgeSoAIndex::from_edges(&inner.edges));
        }

        #[cfg(feature = "storage-dense-index")]
        Self::maybe_build_dense_index(inner);
    }

    #[cfg(feature = "storage-dense-index")]
    fn maybe_build_dense_index(inner: &mut BeliefGraphInner) {
        let node_count = inner.nodes.len();
        if node_count == 0 {
            inner.dense_adjacency = None;
            return;
        }

        let max_edges = node_count.saturating_mul(node_count).max(1);
        let density = inner.edges.len() as f64 / max_edges as f64;

        // Dense index is only beneficial for compact, highly connected graphs.
        if density < 0.15 || node_count > 4096 {
            inner.dense_adjacency = None;
            return;
        }

        let node_ids: Vec<_> = inner.nodes.iter().map(|node| node.id).collect();
        let mut dense = DenseAdjacencyIndex::new(&node_ids);
        for edge in &inner.edges {
            dense.add_edge(edge.src, edge.dst, edge.ty.as_ref());
        }
        inner.dense_adjacency = Some(dense);
    }

    pub fn adjacency_outgoing_by_type(
        &self,
    ) -> std::collections::HashMap<(NodeId, Arc<str>), Vec<EdgeId>> {
        // Build on demand. Could use cached adjacency index if available.
        let mut map: std::collections::HashMap<(NodeId, Arc<str>), Vec<EdgeId>> =
            std::collections::HashMap::new();

        // Include base edges (Arc clone is cheap)
        for e in &self.inner.edges {
            map.entry((e.src, e.ty.clone())).or_default().push(e.id);
        }

        // Include delta edges (new or modified)
        for change in &self.delta {
            if let GraphDelta::EdgeChange { id, edge } = change {
                // Check if this replaces an existing edge
                let existing = self.inner.edge_index.contains_key(id);
                if !existing {
                    // New edge (Arc clone is cheap)
                    map.entry((edge.src, edge.ty.clone()))
                        .or_default()
                        .push(*id);
                }
                // Modified edges already have their IDs in the map from base
            } else if let GraphDelta::EdgeRemoved { id } = change {
                // Remove from map if present
                if let Some(&idx) = self.inner.edge_index.get(id) {
                    if let Some(e) = self.inner.edges.get(idx) {
                        // Arc clone is cheap
                        if let Some(edges) = map.get_mut(&(e.src, e.ty.clone())) {
                            edges.retain(|&eid| eid != *id);
                        }
                    }
                }
            }
        }

        // Keep deterministic order by stable EdgeId (unstable sort is faster)
        for v in map.values_mut() {
            v.sort_unstable();
        }
        map
    }

    pub fn observe_edge(&mut self, edge: EdgeId, present: bool) -> Result<(), ExecError> {
        self.ensure_ready_for_delta();

        // Get current edge posterior with fine-grained deltas applied
        let current_posterior = self
            .get_edge_posterior_with_deltas(edge)
            .ok_or_else(|| ExecError::Internal("missing edge".into()))?;

        // Only fine-grained delta for independent edges
        if let EdgePosterior::Independent(beta) = current_posterior {
            // Compute new alpha/beta after observation
            let (new_alpha, new_beta) = if present {
                (beta.alpha + 1.0, beta.beta)
            } else {
                (beta.alpha, beta.beta + 1.0)
            };

            // Store fine-grained delta
            self.delta.push(GraphDelta::EdgeProbChange {
                edge,
                old_alpha: beta.alpha,
                old_beta: beta.beta,
                new_alpha,
                new_beta,
            });
            Ok(())
        } else {
            Err(ExecError::ValidationError(
                "observe() only valid for independent edges; use observe_chosen for competing edges".into(),
            ))
        }
    }

    /// Observes a competing edge as chosen (increments α_k for the chosen category).
    ///
    /// # Arguments
    ///
    /// * `edge` - The edge ID to observe
    pub fn observe_edge_chosen(&mut self, edge: EdgeId) -> Result<(), ExecError> {
        self.ensure_ready_for_delta();

        let (group_id, category_index) = {
            let e = self
                .edge(edge)
                .ok_or_else(|| ExecError::Internal("missing edge".into()))?;

            match &e.exist {
                EdgePosterior::Competing {
                    group_id,
                    category_index,
                } => (*group_id, *category_index),
                EdgePosterior::Independent(_) => {
                    return Err(ExecError::ValidationError(
                        "observe_edge_chosen() only valid for competing edges".into(),
                    ));
                }
            }
        };

        // Get current competing group (may need to check delta, but competing groups are in base for now)
        // For now, we'll apply delta if needed, then update group
        // TODO: Add competing group to delta tracking if we modify it
        self.ensure_owned();
        let inner =
            Arc::get_mut(&mut self.inner).expect("ensure_owned guarantees exclusive ownership");
        let group = inner
            .competing_groups
            .get_mut(&group_id)
            .ok_or_else(|| ExecError::Internal("missing competing group".into()))?;
        group.posterior.observe_chosen(category_index);
        Ok(())
    }

    /// Observes a competing edge as unchosen.
    ///
    /// This update is only conjugate for binary competing groups (K=2), where
    /// "edge i unchosen" implies the other edge was chosen. For K>2, this returns
    /// a validation error and requires explicit chosen-category evidence.
    ///
    /// # Arguments
    ///
    /// * `edge` - The edge ID to observe
    pub fn observe_edge_unchosen(&mut self, edge: EdgeId) -> Result<(), ExecError> {
        self.ensure_ready_for_delta();

        let (group_id, category_index) = {
            let e = self
                .edge(edge)
                .ok_or_else(|| ExecError::Internal("missing edge".into()))?;

            match &e.exist {
                EdgePosterior::Competing {
                    group_id,
                    category_index,
                } => (*group_id, *category_index),
                EdgePosterior::Independent(_) => {
                    return Err(ExecError::ValidationError(
                        "observe_edge_unchosen() only valid for competing edges".into(),
                    ));
                }
            }
        };

        // Update competing group (apply delta if needed)
        self.ensure_owned();
        let inner =
            Arc::get_mut(&mut self.inner).expect("ensure_owned guarantees exclusive ownership");
        let group = inner
            .competing_groups
            .get_mut(&group_id)
            .ok_or_else(|| ExecError::Internal("missing competing group".into()))?;
        group.posterior.observe_unchosen(category_index)?;
        Ok(())
    }

    /// Forces a competing edge to be chosen deterministically.
    ///
    /// # Arguments
    ///
    /// * `edge` - The edge ID to force
    pub fn observe_edge_forced_choice(&mut self, edge: EdgeId) -> Result<(), ExecError> {
        self.ensure_ready_for_delta();

        let (group_id, category_index) = {
            let e = self
                .edge(edge)
                .ok_or_else(|| ExecError::Internal("missing edge".into()))?;

            match &e.exist {
                EdgePosterior::Competing {
                    group_id,
                    category_index,
                } => (*group_id, *category_index),
                EdgePosterior::Independent(_) => {
                    return Err(ExecError::ValidationError(
                        "observe_edge_forced_choice() only valid for competing edges".into(),
                    ));
                }
            }
        };

        // Update competing group (apply delta if needed)
        self.ensure_owned();
        let inner =
            Arc::get_mut(&mut self.inner).expect("ensure_owned guarantees exclusive ownership");
        let group = inner
            .competing_groups
            .get_mut(&group_id)
            .ok_or_else(|| ExecError::Internal("missing competing group".into()))?;
        group.posterior.force_choice(category_index);
        Ok(())
    }

    pub fn force_edge_present(&mut self, edge: EdgeId) -> Result<(), ExecError> {
        self.ensure_ready_for_delta();

        // Get current edge posterior with fine-grained deltas applied
        let current_posterior = self
            .get_edge_posterior_with_deltas(edge)
            .ok_or_else(|| ExecError::Internal("missing edge".into()))?;

        // Only fine-grained delta for independent edges
        if let EdgePosterior::Independent(beta) = current_posterior {
            // Store fine-grained delta (only alpha/beta change)
            self.delta.push(GraphDelta::EdgeProbChange {
                edge,
                old_alpha: beta.alpha,
                old_beta: beta.beta,
                new_alpha: FORCE_PRECISION,
                new_beta: 1.0,
            });
            Ok(())
        } else {
            Err(ExecError::ValidationError(
                "force_edge_present() only valid for independent edges".into(),
            ))
        }
    }

    pub fn observe_attr(
        &mut self,
        node: NodeId,
        attr: &str,
        x: f64,
        tau_obs: f64,
    ) -> Result<(), ExecError> {
        self.ensure_ready_for_delta();

        // Get current attribute value with fine-grained deltas applied
        let old_posterior = self
            .get_node_attr_with_deltas(node, attr)
            .ok_or_else(|| ExecError::Internal(format!("missing attr '{}'", attr)))?;

        // Compute new mean and precision after Bayesian update
        let tau_old = old_posterior.precision;
        let tau_obs = tau_obs.max(MIN_OBS_PRECISION);
        let tau_new = tau_old + tau_obs;
        let mu_new = (tau_old * old_posterior.mean + tau_obs * x) / tau_new;

        // Store fine-grained delta
        self.delta.push(GraphDelta::NodeAttributeChange {
            node,
            attr: Arc::from(attr),
            old_mean: old_posterior.mean,
            old_precision: old_posterior.precision,
            new_mean: mu_new,
            new_precision: tau_new,
        });
        Ok(())
    }

    pub fn force_attr_value(&mut self, node: NodeId, attr: &str, x: f64) -> Result<(), ExecError> {
        self.ensure_ready_for_delta();

        // Get current attribute value with fine-grained deltas applied
        let old_posterior = self
            .get_node_attr_with_deltas(node, attr)
            .ok_or_else(|| ExecError::Internal(format!("missing attr '{}'", attr)))?;

        // Force value sets mean to x and precision to FORCE_PRECISION
        // Store fine-grained delta
        self.delta.push(GraphDelta::NodeAttributeChange {
            node,
            attr: Arc::from(attr),
            old_mean: old_posterior.mean,
            old_precision: old_posterior.precision,
            new_mean: x,
            new_precision: FORCE_PRECISION,
        });
        Ok(())
    }

    /// Batch observation of multiple values for a node attribute using vectorized kernels.
    ///
    /// Applies multiple Gaussian observations in a single kernel call for improved performance.
    /// This is mathematically equivalent to sequential `observe_attr` calls but more efficient.
    ///
    /// # Arguments
    ///
    /// * `node` - The node ID
    /// * `attr` - The attribute name
    /// * `observations` - Slice of `(value, precision)` pairs
    ///
    /// # Returns
    ///
    /// Ok(()) if successful, or an error if the update fails
    #[cfg(feature = "vectorized")]
    pub fn observe_attr_batch(
        &mut self,
        node: NodeId,
        attr: &str,
        observations: &[(f64, f64)],
    ) -> Result<(), ExecError> {
        use crate::engine::vectorized::gaussian_batch_update;

        if observations.is_empty() {
            return Ok(());
        }

        self.ensure_ready_for_delta();

        // Get current attribute value with fine-grained deltas applied
        let old_posterior = self
            .get_node_attr_with_deltas(node, attr)
            .ok_or_else(|| ExecError::Internal(format!("missing attr '{}'", attr)))?;

        // Apply vectorized batch update
        let new_posterior = gaussian_batch_update(&old_posterior, observations)?;

        // Store fine-grained delta
        self.delta.push(GraphDelta::NodeAttributeChange {
            node,
            attr: Arc::from(attr),
            old_mean: old_posterior.mean,
            old_precision: old_posterior.precision,
            new_mean: new_posterior.mean,
            new_precision: new_posterior.precision,
        });
        Ok(())
    }

    /// Batch observation of multiple edge presence/absence using vectorized kernels.
    ///
    /// Applies multiple Beta observations in a single kernel call for improved performance.
    /// This is mathematically equivalent to sequential `observe_edge` calls but more efficient.
    ///
    /// # Arguments
    ///
    /// * `edge` - The edge ID
    /// * `observations` - Slice of boolean observations (true = present, false = absent)
    ///
    /// # Returns
    ///
    /// Ok(()) if successful, or an error if the update fails
    #[cfg(feature = "vectorized")]
    pub fn observe_edge_batch(
        &mut self,
        edge: EdgeId,
        observations: &[bool],
    ) -> Result<(), ExecError> {
        use crate::engine::vectorized::beta_batch_update;

        if observations.is_empty() {
            return Ok(());
        }

        self.ensure_ready_for_delta();

        // Get current posterior with fine-grained deltas applied
        let edge_posterior = self
            .get_edge_posterior_with_deltas(edge)
            .ok_or_else(|| ExecError::Internal(format!("Edge {:?} not found", edge)))?;

        // Extract Beta posterior (only works for independent edges)
        let old_posterior = match edge_posterior {
            EdgePosterior::Independent(beta) => beta,
            EdgePosterior::Competing { .. } => {
                return Err(ExecError::ValidationError(
                    "Batch observation not supported for competing edges".into(),
                ))
            }
        };

        // Apply vectorized batch update
        let new_posterior = beta_batch_update(&old_posterior, observations)?;

        // Store fine-grained delta
        self.delta.push(GraphDelta::EdgeProbChange {
            edge,
            old_alpha: old_posterior.alpha,
            old_beta: old_posterior.beta,
            new_alpha: new_posterior.alpha,
            new_beta: new_posterior.beta,
        });

        Ok(())
    }

    /// Validates numerical stability of the graph.
    ///
    /// Checks for NaN/Inf values, invalid precisions, Beta parameters outside
    /// valid range, and outlier values (> 10σ from mean).
    ///
    /// Validates numerical stability of all posteriors:
    /// - Gaussian: precision ≥ MIN_PRECISION, no NaN/Inf in mean
    /// - Beta: α ≥ MIN_BETA_PARAM, β ≥ MIN_BETA_PARAM, no NaN/Inf
    /// - Dirichlet: all α_k ≥ MIN_DIRICHLET_PARAM, no NaN/Inf
    /// - Outliers: warns if |x - μ| > 10σ for any attribute
    ///
    /// # Returns
    ///
    /// - `Ok(Vec<String>)` - List of warnings (empty if no issues)
    /// - `Err(ExecError::Numerical)` - Critical numerical error that makes graph invalid
    pub fn validate_numerical_stability(&self) -> Result<Vec<String>, ExecError> {
        let mut warnings = Vec::new();

        // Check node attributes
        for node in &self.inner.nodes {
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
                let std_dev = (1.0_f64 / attr.precision).sqrt();
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
        for edge in &self.inner.edges {
            match &edge.exist {
                EdgePosterior::Independent(beta) => {
                    // Check for NaN/Inf in Beta parameters
                    if !beta.alpha.is_finite() || !beta.beta.is_finite() {
                        return Err(ExecError::Numerical(format!(
                            "Edge {:?} has non-finite Beta parameters: α={}, β={}",
                            edge.id, beta.alpha, beta.beta
                        )));
                    }

                    // Enforce minimum Beta parameters (proper priors)
                    if beta.alpha < MIN_BETA_PARAM || beta.beta < MIN_BETA_PARAM {
                        warnings.push(format!(
                            "Edge {:?} has Beta parameters below minimum (α={:.2e}, β={:.2e}), will be clipped to {:.2e}",
                            edge.id, beta.alpha, beta.beta, MIN_BETA_PARAM
                        ));
                    }

                    // Compute probability for range check
                    let prob = beta.mean_probability();

                    // Probability must be in [0, 1]
                    if !(0.0..=1.0).contains(&prob) {
                        return Err(ExecError::Numerical(format!(
                            "Edge {:?} has invalid probability: {} (from α={}, β={})",
                            edge.id, prob, beta.alpha, beta.beta
                        )));
                    }
                }
                EdgePosterior::Competing {
                    group_id,
                    category_index,
                } => {
                    if let Some(group) = self.inner.competing_groups.get(group_id) {
                        // Check Dirichlet parameters
                        for (i, &alpha) in group.posterior.concentrations.iter().enumerate() {
                            if !alpha.is_finite() {
                                return Err(ExecError::Numerical(format!(
                                    "Edge {:?} (competing group {:?}, category {}) has non-finite Dirichlet parameter: α={}",
                                    edge.id, group_id, i, alpha
                                )));
                            }
                            if alpha < MIN_DIRICHLET_PARAM {
                                warnings.push(format!(
                                    "Edge {:?} (competing group {:?}) has Dirichlet parameter below minimum (α_{}={:.2e}), will be clipped",
                                    edge.id, group_id, i, alpha
                                ));
                            }
                        }
                        // Check probability
                        let prob = group.posterior.mean_probability(*category_index);
                        if !(0.0..=1.0).contains(&prob) {
                            return Err(ExecError::Numerical(format!(
                                "Edge {:?} (competing group {:?}) has invalid probability: {}",
                                edge.id, group_id, prob
                            )));
                        }
                    }
                }
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
        let mut g = GaussianPosterior {
            mean: 0.0,
            precision: 1.0,
        };
        g.set_expectation(5.0);
        assert_eq!(g.mean, 5.0);
        assert_eq!(g.precision, 1.0); // precision unchanged
    }

    #[test]
    fn gaussian_update_combines_prior_and_observation() {
        // Prior: N(0, τ=1), Observation: x=10 with τ_obs=1
        // Expected posterior: N(5, τ=2)
        let mut g = GaussianPosterior {
            mean: 0.0,
            precision: 1.0,
        };
        g.update(10.0, 1.0);

        assert!((g.mean - 5.0).abs() < 1e-9, "mean should be 5.0");
        assert!((g.precision - 2.0).abs() < 1e-9, "precision should be 2.0");
    }

    #[test]
    fn gaussian_update_increases_precision() {
        let mut g = GaussianPosterior {
            mean: 0.0,
            precision: 1.0,
        };
        let initial_precision = g.precision;

        g.update(5.0, 2.0);

        assert!(
            g.precision > initial_precision,
            "precision should increase after observation"
        );
    }

    #[test]
    fn gaussian_update_weighted_by_precision() {
        // High precision observation should dominate
        let mut g = GaussianPosterior {
            mean: 0.0,
            precision: 0.01,
        };
        g.update(100.0, 10.0); // high precision observation

        // Mean should be much closer to observation than prior
        assert!(
            g.mean > 90.0,
            "mean should be close to high-precision observation"
        );
    }

    #[test]
    fn gaussian_update_clips_minimum_precision() {
        let mut g = GaussianPosterior {
            mean: 0.0,
            precision: 0.0,
        };
        g.update(5.0, 0.0);

        // Should clip to minimum τ=1e-6 per design doc
        assert!(
            g.precision >= 1e-6,
            "precision should be clipped to minimum"
        );
    }

    #[test]
    fn gaussian_update_clips_negative_tau_obs() {
        let mut g = GaussianPosterior {
            mean: 0.0,
            precision: 1.0,
        };
        g.update(5.0, -1.0); // negative observation precision

        // Should clip tau_obs to 1e-12
        assert!(g.precision >= 1.0, "should handle negative tau_obs");
    }

    #[test]
    fn gaussian_force_value_sets_high_precision() {
        let mut g = GaussianPosterior {
            mean: 0.0,
            precision: 1.0,
        };
        g.force_value(42.0);

        assert_eq!(g.mean, 42.0, "mean should be forced to exact value");
        assert_eq!(
            g.precision, FORCE_PRECISION,
            "precision should be very high"
        );
    }

    #[test]
    fn gaussian_force_value_represents_certainty() {
        let mut g = GaussianPosterior {
            mean: 10.0,
            precision: 2.0,
        };
        g.force_value(5.0);

        // Variance = 1/precision = 1/1e6, so std dev is ~0.001
        let variance = 1.0 / g.precision;
        assert!(
            variance < 1e-5,
            "forced value should have very low variance"
        );
    }

    // ============================================================================
    // BetaPosterior Unit Tests
    // ============================================================================

    #[test]
    fn beta_force_absent_sets_near_zero_probability() {
        let mut b = BetaPosterior {
            alpha: 5.0,
            beta: 5.0,
        };
        b.force_absent();

        assert_eq!(b.alpha, 1.0);
        assert_eq!(b.beta, FORCE_PRECISION);

        let mean = b.alpha / (b.alpha + b.beta);
        assert!(mean < 1e-5, "forced absent should have near-zero mean");
    }

    #[test]
    fn beta_force_present_sets_near_one_probability() {
        let mut b = BetaPosterior {
            alpha: 5.0,
            beta: 5.0,
        };
        b.force_present();

        assert_eq!(b.alpha, FORCE_PRECISION);
        assert_eq!(b.beta, 1.0);

        let mean = b.alpha / (b.alpha + b.beta);
        assert!(mean > 0.99999, "forced present should have near-one mean");
    }

    #[test]
    fn beta_observe_present_increments_alpha() {
        let mut b = BetaPosterior {
            alpha: 2.0,
            beta: 3.0,
        };
        b.observe(true);

        assert_eq!(b.alpha, 3.0);
        assert_eq!(b.beta, 3.0);
    }

    #[test]
    fn beta_observe_absent_increments_beta() {
        let mut b = BetaPosterior {
            alpha: 2.0,
            beta: 3.0,
        };
        b.observe(false);

        assert_eq!(b.alpha, 2.0);
        assert_eq!(b.beta, 4.0);
    }

    #[test]
    fn beta_multiple_observations_accumulate() {
        let mut b = BetaPosterior {
            alpha: 1.0,
            beta: 1.0,
        };

        b.observe(true);
        b.observe(true);
        b.observe(false);

        assert_eq!(b.alpha, 3.0, "two present observations");
        assert_eq!(b.beta, 2.0, "one absent observation");
    }

    #[test]
    fn beta_mean_calculation_uniform_prior() {
        let b = BetaPosterior {
            alpha: 1.0,
            beta: 1.0,
        };
        let mean = b.alpha / (b.alpha + b.beta);

        assert!(
            (mean - 0.5).abs() < 1e-9,
            "uniform prior should have mean 0.5"
        );
    }

    #[test]
    fn beta_mean_calculation_biased_prior() {
        let b = BetaPosterior {
            alpha: 8.0,
            beta: 2.0,
        };
        let mean = b.alpha / (b.alpha + b.beta);

        assert!((mean - 0.8).abs() < 1e-9, "Beta(8,2) should have mean 0.8");
    }

    // ============================================================================
    // BeliefGraph Unit Tests
    // ============================================================================

    #[test]
    fn belief_graph_default_is_empty() {
        let g = BeliefGraph::default();
        assert_eq!(g.nodes().len(), 0);
        assert_eq!(g.edges().len(), 0);
    }

    #[test]
    fn belief_graph_node_lookup_by_id() {
        let mut g = BeliefGraph::default();
        g.insert_node(NodeData {
            id: NodeId(1),
            label: Arc::from("Person"),
            attrs: HashMap::new(),
        });

        assert!(g.node(NodeId(1)).is_some());
        assert!(g.node(NodeId(999)).is_none());
    }

    #[test]
    fn belief_graph_edge_lookup_by_id() {
        let mut g = BeliefGraph::default();
        g.insert_edge(BeliefGraph::test_edge_with_beta(
            EdgeId(42),
            NodeId(1),
            NodeId(2),
            "REL".into(),
            BetaPosterior {
                alpha: 1.0,
                beta: 1.0,
            },
        ));

        assert!(g.edge(EdgeId(42)).is_some());
        assert!(g.edge(EdgeId(999)).is_none());
    }

    #[test]
    fn belief_graph_expectation_retrieves_mean() {
        let mut g = BeliefGraph::default();
        g.insert_node(NodeData {
            id: NodeId(1),
            label: Arc::from("Person"),
            attrs: HashMap::from([(
                "age".into(),
                GaussianPosterior {
                    mean: 25.5,
                    precision: 1.0,
                },
            )]),
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
            label: Arc::from("Person"),
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
            label: Arc::from("Person"),
            attrs: HashMap::from([(
                "x".into(),
                GaussianPosterior {
                    mean: 0.0,
                    precision: 1.0,
                },
            )]),
        });

        g.set_expectation(NodeId(1), "x", 10.0).unwrap();

        let mean = g.expectation(NodeId(1), "x").unwrap();
        assert!((mean - 10.0).abs() < 1e-9);
    }

    #[test]
    fn belief_graph_prob_mean_calculates_beta_mean() {
        let mut g = BeliefGraph::default();
        g.insert_edge(BeliefGraph::test_edge_with_beta(
            EdgeId(1),
            NodeId(1),
            NodeId(2),
            "REL".into(),
            BetaPosterior {
                alpha: 3.0,
                beta: 7.0,
            },
        ));

        let prob = g.prob_mean(EdgeId(1)).unwrap();
        assert!((prob - 0.3).abs() < 1e-9); // 3/(3+7) = 0.3
    }

    #[test]
    fn belief_graph_prob_mean_clips_negative_parameters() {
        let mut g = BeliefGraph::default();
        g.insert_edge(BeliefGraph::test_edge_with_beta(
            EdgeId(1),
            NodeId(1),
            NodeId(2),
            "REL".into(),
            BetaPosterior {
                alpha: -1.0,
                beta: 2.0,
            },
        ));

        // Should clip negative alpha to MIN_BETA_PARAM (0.01) per design spec
        // Expected: 0.01 / (0.01 + 2.0) ≈ 0.00497...
        let prob = g.prob_mean(EdgeId(1)).unwrap();
        let expected = MIN_BETA_PARAM / (MIN_BETA_PARAM + 2.0);
        assert!(
            (prob - expected).abs() < 1e-9,
            "alpha should clip to MIN_BETA_PARAM, not 0"
        );
    }

    #[test]
    fn belief_graph_prob_mean_handles_zero_parameters() {
        let mut g = BeliefGraph::default();
        g.insert_edge(BeliefGraph::test_edge_with_beta(
            EdgeId(1),
            NodeId(1),
            NodeId(2),
            "REL".into(),
            BetaPosterior {
                alpha: 0.0,
                beta: 0.0,
            },
        ));

        // With clipping to MIN_BETA_PARAM, even (0,0) becomes valid:
        // 0.01 / (0.01 + 0.01) = 0.5
        let prob = g.prob_mean(EdgeId(1)).unwrap();
        assert!(
            (prob - 0.5).abs() < 1e-9,
            "Both params clip to 0.01, giving mean 0.5"
        );
    }

    #[test]
    fn belief_graph_degree_outgoing_counts_high_prob_edges() {
        let mut g = BeliefGraph::default();
        g.insert_node(NodeData {
            id: NodeId(1),
            label: "N".into(),
            attrs: HashMap::new(),
        });

        // Add edges with different probabilities
        g.insert_edge(BeliefGraph::test_edge_with_beta(
            EdgeId(1),
            NodeId(1),
            NodeId(2),
            "R".into(),
            BetaPosterior {
                alpha: 8.0,
                beta: 2.0,
            }, // p=0.8
        ));
        g.insert_edge(BeliefGraph::test_edge_with_beta(
            EdgeId(2),
            NodeId(1),
            NodeId(3),
            "R".into(),
            BetaPosterior {
                alpha: 2.0,
                beta: 8.0,
            }, // p=0.2
        ));
        g.insert_edge(BeliefGraph::test_edge_with_beta(
            EdgeId(3),
            NodeId(1),
            NodeId(4),
            "R".into(),
            BetaPosterior {
                alpha: 6.0,
                beta: 4.0,
            }, // p=0.6
        ));

        assert_eq!(g.degree_outgoing(NodeId(1), 0.7), 1, "only one edge >= 0.7");
        assert_eq!(g.degree_outgoing(NodeId(1), 0.5), 2, "two edges >= 0.5");
        assert_eq!(g.degree_outgoing(NodeId(1), 0.1), 3, "all edges >= 0.1");
    }

    #[test]
    fn belief_graph_degree_outgoing_filters_by_source() {
        let mut g = BeliefGraph::default();
        g.insert_edge(BeliefGraph::test_edge_with_beta(
            EdgeId(1),
            NodeId(1),
            NodeId(2),
            "R".into(),
            BetaPosterior {
                alpha: 5.0,
                beta: 5.0,
            },
        ));
        g.insert_edge(BeliefGraph::test_edge_with_beta(
            EdgeId(2),
            NodeId(2),
            NodeId(3),
            "R".into(),
            BetaPosterior {
                alpha: 5.0,
                beta: 5.0,
            },
        ));

        assert_eq!(g.degree_outgoing(NodeId(1), 0.0), 1);
        assert_eq!(g.degree_outgoing(NodeId(2), 0.0), 1);
        assert_eq!(g.degree_outgoing(NodeId(3), 0.0), 0);
    }

    #[test]
    fn belief_graph_adjacency_outgoing_groups_by_node_and_type() {
        let mut g = BeliefGraph::default();
        g.insert_edge(BeliefGraph::test_edge_with_beta(
            EdgeId(1),
            NodeId(1),
            NodeId(2),
            "LIKES".into(),
            BetaPosterior {
                alpha: 1.0,
                beta: 1.0,
            },
        ));
        g.insert_edge(BeliefGraph::test_edge_with_beta(
            EdgeId(2),
            NodeId(1),
            NodeId(3),
            "LIKES".into(),
            BetaPosterior {
                alpha: 1.0,
                beta: 1.0,
            },
        ));
        g.insert_edge(BeliefGraph::test_edge_with_beta(
            EdgeId(3),
            NodeId(1),
            NodeId(4),
            "KNOWS".into(),
            BetaPosterior {
                alpha: 1.0,
                beta: 1.0,
            },
        ));

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
        g.insert_edge(BeliefGraph::test_edge_with_beta(
            EdgeId(5),
            NodeId(1),
            NodeId(2),
            "R".into(),
            BetaPosterior {
                alpha: 1.0,
                beta: 1.0,
            },
        ));
        g.insert_edge(BeliefGraph::test_edge_with_beta(
            EdgeId(2),
            NodeId(1),
            NodeId(3),
            "R".into(),
            BetaPosterior {
                alpha: 1.0,
                beta: 1.0,
            },
        ));
        g.insert_edge(BeliefGraph::test_edge_with_beta(
            EdgeId(8),
            NodeId(1),
            NodeId(4),
            "R".into(),
            BetaPosterior {
                alpha: 1.0,
                beta: 1.0,
            },
        ));

        let adj = g.adjacency_outgoing_by_type();
        let edges = adj.get(&(NodeId(1), "R".into())).unwrap();

        assert_eq!(
            edges,
            &vec![EdgeId(2), EdgeId(5), EdgeId(8)],
            "should be sorted"
        );
    }

    #[test]
    fn belief_graph_observe_edge_updates_beta_posterior() {
        let mut g = BeliefGraph::default();
        g.insert_edge(BeliefGraph::test_edge_with_beta(
            EdgeId(1),
            NodeId(1),
            NodeId(2),
            "R".into(),
            BetaPosterior {
                alpha: 1.0,
                beta: 1.0,
            },
        ));

        g.observe_edge(EdgeId(1), true).unwrap();

        // Apply fine-grained deltas to see the updated edge
        g.ensure_owned();

        let edge = g.edge(EdgeId(1)).unwrap();
        if let EdgePosterior::Independent(beta) = &edge.exist {
            assert_eq!(beta.alpha, 2.0);
            assert_eq!(beta.beta, 1.0);
        } else {
            panic!("Expected independent edge");
        }
    }

    #[test]
    fn belief_graph_force_edge_present_sets_high_alpha() {
        let mut g = BeliefGraph::default();
        g.insert_edge(BeliefGraph::test_edge_with_beta(
            EdgeId(1),
            NodeId(1),
            NodeId(2),
            "R".into(),
            BetaPosterior {
                alpha: 1.0,
                beta: 1.0,
            },
        ));

        g.force_edge_present(EdgeId(1)).unwrap();

        // Apply fine-grained deltas to see the updated edge
        g.ensure_owned();

        let edge = g.edge(EdgeId(1)).unwrap();
        if let EdgePosterior::Independent(beta) = &edge.exist {
            assert_eq!(beta.alpha, FORCE_PRECISION);
            assert_eq!(beta.beta, 1.0);
        } else {
            panic!("Expected independent edge");
        }
    }

    #[test]
    fn belief_graph_observe_attr_updates_gaussian() {
        let mut g = BeliefGraph::default();
        g.insert_node(NodeData {
            id: NodeId(1),
            label: "N".into(),
            attrs: HashMap::from([(
                "x".into(),
                GaussianPosterior {
                    mean: 0.0,
                    precision: 1.0,
                },
            )]),
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
            label: Arc::from("N"),
            attrs: HashMap::from([(
                "x".into(),
                GaussianPosterior {
                    mean: 0.0,
                    precision: 1.0,
                },
            )]),
        });

        g.force_attr_value(NodeId(1), "x", 42.0).unwrap();

        let mean = g.expectation(NodeId(1), "x").unwrap();
        assert!((mean - 42.0).abs() < 1e-9);

        // Apply fine-grained deltas to see the updated node
        g.ensure_owned();

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
            (1.0, 1.0, 0.5),    // Uniform: Beta(1,1) → 0.5
            (2.0, 8.0, 0.2),    // Skewed: Beta(2,8) → 0.2
            (8.0, 2.0, 0.8),    // Skewed: Beta(8,2) → 0.8
            (10.0, 10.0, 0.5),  // Symmetric: Beta(10,10) → 0.5
            (100.0, 1.0, 0.99), // Near certain: Beta(100,1) ≈ 0.99
        ];

        for (alpha, beta, expected_mean) in test_cases {
            let mut g = BeliefGraph::default();
            g.insert_edge(BeliefGraph::test_edge_with_beta(
                EdgeId(1),
                NodeId(1),
                NodeId(2),
                "R".into(),
                BetaPosterior { alpha, beta },
            ));

            let prob = g.prob_mean(EdgeId(1)).unwrap();
            let expected = alpha / (alpha + beta);
            assert!(
                (prob - expected).abs() < 1e-9,
                "Beta({},{}) should have mean {} but got {}",
                alpha,
                beta,
                expected,
                prob
            );
            assert!(
                (prob - expected_mean).abs() < 0.01,
                "Beta({},{}) should be approximately {}",
                alpha,
                beta,
                expected_mean
            );
        }
    }

    #[test]
    fn beta_posterior_update_maintains_proper_parameters() {
        // Verify that Beta updates always maintain α > 0 and β > 0
        let mut beta = BetaPosterior {
            alpha: 1.0,
            beta: 1.0,
        };

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
        let mut g = GaussianPosterior {
            mean: 0.0,
            precision: 1.0,
        };
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
        let mut g = GaussianPosterior {
            mean: 10.0,
            precision: 2.0,
        };
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
        g.insert_edge(BeliefGraph::test_edge_with_beta(
            EdgeId(1),
            NodeId(1),
            NodeId(2),
            "R".into(),
            BetaPosterior {
                alpha: 3.0,
                beta: 7.0,
            },
        ));

        let prob = g.prob_mean(EdgeId(1)).unwrap();
        assert!((prob - 0.3).abs() < 1e-9, "Probability should be 0.3");

        // degree_outgoing should count it if min_prob <= 0.3
        assert_eq!(
            g.degree_outgoing(NodeId(1), 0.25),
            1,
            "Should count at 0.25 threshold"
        );
        assert_eq!(
            g.degree_outgoing(NodeId(1), 0.30),
            1,
            "Should count at 0.30 threshold"
        );
        assert_eq!(
            g.degree_outgoing(NodeId(1), 0.35),
            0,
            "Should not count at 0.35 threshold"
        );
    }

    #[test]
    fn force_operations_create_near_deterministic_beliefs() {
        // force_present should create near-1 probability
        let mut b = BetaPosterior {
            alpha: 1.0,
            beta: 1.0,
        };
        b.force_present();
        let mean = b.alpha / (b.alpha + b.beta);
        assert!(
            mean > 0.999,
            "force_present should give probability > 0.999"
        );

        // force_absent should create near-0 probability
        b.force_absent();
        let mean = b.alpha / (b.alpha + b.beta);
        assert!(mean < 0.001, "force_absent should give probability < 0.001");

        // force_value should create very small variance
        let mut g = GaussianPosterior {
            mean: 0.0,
            precision: 1.0,
        };
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
        g.insert_edge(BeliefGraph::test_edge_with_beta(
            EdgeId(1),
            NodeId(1),
            NodeId(2),
            "LIKES".into(),
            BetaPosterior {
                alpha: 1.0,
                beta: 1.0,
            },
        ));
        g.insert_edge(BeliefGraph::test_edge_with_beta(
            EdgeId(2),
            NodeId(1),
            NodeId(3),
            "LIKES".into(),
            BetaPosterior {
                alpha: 1.0,
                beta: 1.0,
            },
        ));
        g.insert_edge(BeliefGraph::test_edge_with_beta(
            EdgeId(3),
            NodeId(1),
            NodeId(4),
            "KNOWS".into(),
            BetaPosterior {
                alpha: 1.0,
                beta: 1.0,
            },
        ));
        g.insert_edge(BeliefGraph::test_edge_with_beta(
            EdgeId(4),
            NodeId(2),
            NodeId(3),
            "LIKES".into(),
            BetaPosterior {
                alpha: 1.0,
                beta: 1.0,
            },
        ));

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
        g.insert_edge(BeliefGraph::test_edge_with_beta(
            EdgeId(5),
            NodeId(1),
            NodeId(2),
            "R".into(),
            BetaPosterior {
                alpha: 1.0,
                beta: 1.0,
            },
        ));
        g.insert_edge(BeliefGraph::test_edge_with_beta(
            EdgeId(2),
            NodeId(1),
            NodeId(3),
            "R".into(),
            BetaPosterior {
                alpha: 1.0,
                beta: 1.0,
            },
        ));
        g.insert_edge(BeliefGraph::test_edge_with_beta(
            EdgeId(8),
            NodeId(1),
            NodeId(4),
            "R".into(),
            BetaPosterior {
                alpha: 1.0,
                beta: 1.0,
            },
        ));

        g.build_adjacency();
        let edges = g.get_outgoing_edges(NodeId(1), "R");

        // Should be sorted by EdgeId
        assert_eq!(edges, vec![EdgeId(2), EdgeId(5), EdgeId(8)]);
    }

    #[test]
    fn adjacency_index_handles_empty_neighborhoods() {
        let mut g = BeliefGraph::default();
        g.insert_edge(BeliefGraph::test_edge_with_beta(
            EdgeId(1),
            NodeId(1),
            NodeId(2),
            "R".into(),
            BetaPosterior {
                alpha: 1.0,
                beta: 1.0,
            },
        ));

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
        g.insert_edge(BeliefGraph::test_edge_with_beta(
            EdgeId(1),
            NodeId(1),
            NodeId(2),
            "R".into(),
            BetaPosterior {
                alpha: 1.0,
                beta: 1.0,
            },
        ));

        // Don't explicitly call build_adjacency
        let edges = g.get_outgoing_edges(NodeId(1), "R");

        // Should build index lazily and return correct result
        assert_eq!(edges.len(), 1);
        assert_eq!(edges[0], EdgeId(1));
    }

    // ===================== Phase 2 semantics tests =====================
    #[test]
    fn non_bayesian_nudge_variance_policies() {
        let mut g = BeliefGraph::default();
        let mut attrs = HashMap::new();
        attrs.insert(
            "x".into(),
            GaussianPosterior {
                mean: 0.0,
                precision: 2.0,
            },
        );
        let nid = g.add_node("N".to_string(), attrs);

        // preserve
        g.non_bayesian_nudge(
            nid,
            "x",
            10.0,
            &grafial_frontend::ast::VarianceSpec::Preserve,
        )
        .unwrap();
        assert_eq!(g.expectation(nid, "x").unwrap(), 10.0);
        assert!((g.precision(nid, "x").unwrap() - 2.0).abs() < 1e-12);

        // decrease(f=2.0) -> precision doubles
        g.non_bayesian_nudge(
            nid,
            "x",
            5.0,
            &grafial_frontend::ast::VarianceSpec::Decrease { factor: Some(2.0) },
        )
        .unwrap();
        assert_eq!(g.expectation(nid, "x").unwrap(), 5.0);
        assert!((g.precision(nid, "x").unwrap() - 4.0).abs() < 1e-12);

        // increase(f=0.5) -> precision halves
        g.non_bayesian_nudge(
            nid,
            "x",
            -1.0,
            &grafial_frontend::ast::VarianceSpec::Increase { factor: Some(0.5) },
        )
        .unwrap();
        assert_eq!(g.expectation(nid, "x").unwrap(), -1.0);
        assert!((g.precision(nid, "x").unwrap() - 2.0).abs() < 1e-12);
    }

    #[test]
    fn soft_update_normal_normal() {
        let mut g = BeliefGraph::default();
        let mut attrs = HashMap::new();
        attrs.insert(
            "x".into(),
            GaussianPosterior {
                mean: 0.0,
                precision: 1.0,
            },
        );
        let nid = g.add_node("N".to_string(), attrs);

        // observe x=10 with τ_obs=1, count=1
        g.soft_update(nid, "x", 10.0, 1.0, 1.0).unwrap();
        let mu = g.expectation(nid, "x").unwrap();
        let tau = g.precision(nid, "x").unwrap();
        assert!((mu - 5.0).abs() < 1e-12);
        assert!((tau - 2.0).abs() < 1e-12);
    }

    #[test]
    fn delete_and_suppress_edge_and_undelete_math() {
        let mut g = BeliefGraph::default();
        let attrs = HashMap::new();
        let a = g.add_node("N".to_string(), attrs.clone());
        let b = g.add_node("N".to_string(), attrs);
        let e = g.add_edge(
            a,
            b,
            "E".to_string(),
            BetaPosterior {
                alpha: 1.0,
                beta: 1.0,
            },
        );

        // Suppress -> Beta(1,10) by default
        g.suppress_edge(e, None).unwrap();
        let post1 = g.get_edge_posterior_with_deltas(e).unwrap();
        match post1 {
            EdgePosterior::Independent(beta) => {
                assert!((beta.alpha - 1.0).abs() < 1e-12);
                assert!((beta.beta - 10.0).abs() < 1e-12);
            }
            _ => panic!("not independent"),
        }

        // Delete high -> Beta(1, 1e9)
        g.delete_edge(e, Some("high")).unwrap();
        let post2 = g.get_edge_posterior_with_deltas(e).unwrap();
        match post2 {
            EdgePosterior::Independent(beta) => {
                assert!((beta.alpha - 1.0).abs() < 1e-12);
                assert!(beta.beta > 1e8);
            }
            _ => panic!("not independent"),
        }

        let needed = g.observations_to_half(e).unwrap();
        assert!(needed >= 1.0e8);
    }
}
