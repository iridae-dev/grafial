//! Evidence building module.
//!
//! Converts evidence definitions to BeliefGraph instances by:
//! - Creating nodes and edges on demand from observations
//! - Initializing posteriors from belief model declarations
//! - Applying evidence observations to update posteriors
//! - Managing competing edge groups for categorical posteriors

use std::collections::HashMap;
use std::sync::Arc;

#[cfg(feature = "rayon")]
use rayon::prelude::*;

use crate::engine::errors::ExecError;
use crate::engine::graph::{
    BeliefGraph, BetaPosterior, CompetingEdgeGroup, CompetingGroupId, DirichletPosterior,
    EdgeData, EdgeId, EdgePosterior, GaussianPosterior, NodeId,
};
use crate::frontend::ast::{
    BeliefModel, CategoricalPrior, EdgeBeliefDecl, EvidenceDef, EvidenceMode,
    ObserveStmt, PosteriorType, Schema,
};

/// Builds a BeliefGraph from an evidence definition.
///
/// Nodes and edges are created on demand from observations. Posteriors are initialized
/// from the belief model and updated with evidence observations. Competing edge groups
/// are managed automatically for categorical posteriors.
pub fn build_graph_from_evidence(
    evidence: &EvidenceDef,
    program: &crate::frontend::ast::ProgramAst,
) -> Result<BeliefGraph, ExecError> {
    // Find the belief model this evidence applies to
    let model = program
        .belief_models
        .iter()
        .find(|m| m.name == evidence.on_model)
        .ok_or_else(|| ExecError::ValidationError(
            format!("Evidence '{}' references unknown belief model '{}'", evidence.name, evidence.on_model)
        ))?;

    // Find the schema this model operates on
    let schema = program
        .schemas
        .iter()
        .find(|s| s.name == model.on_schema)
        .ok_or_else(|| ExecError::ValidationError(
            format!("Belief model '{}' references unknown schema '{}'", model.name, model.on_schema)
        ))?;

    let mut graph = BeliefGraph::default();
    
    // Index maps for efficient node/edge/group lookup during construction
    let mut node_map: HashMap<(String, String), NodeId> = HashMap::new();
    let mut edge_map: HashMap<(NodeId, NodeId, String), EdgeId> = HashMap::new();
    let mut competing_group_map: HashMap<(NodeId, String), CompetingGroupId> = HashMap::new();
    let mut next_group_id = 0u32;

    // First pass: collect all nodes referenced in observations to ensure graph structure
    // is complete before applying updates (needed for parallel evidence application)
    let mut nodes_to_create: Vec<(String, String)> = Vec::new();
    
    for obs in &evidence.observations {
        match obs {
            ObserveStmt::Edge { src, dst, .. } => {
                nodes_to_create.push((src.0.clone(), src.1.clone()));
                nodes_to_create.push((dst.0.clone(), dst.1.clone()));
            }
            ObserveStmt::Attribute { node, .. } => {
                nodes_to_create.push((node.0.clone(), node.1.clone()));
            }
        }
    }
    
    // Deduplicate nodes
    nodes_to_create.sort();
    nodes_to_create.dedup();
    
    // Create all nodes first
    for (node_type, label) in nodes_to_create {
        let node_id = create_node_if_needed(
            &mut graph,
            &mut node_map,
            &schema,
            &model,
            &node_type,
            &label,
        )?;
        // Ensure node_id is in map (should already be, but be safe)
        node_map.insert((node_type, label), node_id);
    }
    
    // Second pass: create edges and collect observations for parallel processing
    // We need to create all edges first, then apply observations in parallel
    let mut edge_observations: Vec<(EdgeId, EvidenceMode)> = Vec::new();
    let mut attr_observations: Vec<(NodeId, String, f64, f64)> = Vec::new(); // (node_id, attr, value, tau_obs)
    
    for obs in &evidence.observations {
        match obs {
            ObserveStmt::Edge {
                edge_type,
                src,
                dst,
                mode,
            } => {
                let src_id = node_map
                    .get(&(src.0.clone(), src.1.clone()))
                    .ok_or_else(|| ExecError::Internal(format!("Missing source node: {}[\"{}\"]", src.0, src.1)))?;
                let dst_id = node_map
                    .get(&(dst.0.clone(), dst.1.clone()))
                    .ok_or_else(|| ExecError::Internal(format!("Missing destination node: {}[\"{}\"]", dst.0, dst.1)))?;
                
                // Find edge declaration in belief model
                let edge_decl = model.edges.iter().find(|e| e.edge_type == *edge_type)
                    .ok_or_else(|| ExecError::ValidationError(
                        format!("Edge type '{}' not declared in belief model '{}'", edge_type, model.name)
                    ))?;
                
                // Create edge if it doesn't exist
                let edge_id = if let Some(eid) = edge_map.get(&(*src_id, *dst_id, edge_type.clone())) {
                    *eid
                } else {
                    create_edge_with_posterior(
                        &mut graph,
                        &mut edge_map,
                        &mut competing_group_map,
                        &mut next_group_id,
                        *src_id,
                        *dst_id,
                        edge_type.clone(),
                        edge_decl,
                        &schema,
                    )?
                };
                
                // Collect observation for parallel processing
                edge_observations.push((edge_id, mode.clone()));
            }
            ObserveStmt::Attribute { node, attr, value } => {
                let node_id = node_map
                    .get(&(node.0.clone(), node.1.clone()))
                    .ok_or_else(|| ExecError::Internal(format!("Missing node: {}[\"{}\"]", node.0, node.1)))?;
                
                // Find attribute declaration in belief model
                let node_decl = model.nodes.iter().find(|n| n.node_type == node.0)
                    .ok_or_else(|| ExecError::ValidationError(
                        format!("Node type '{}' not declared in belief model '{}'", node.0, model.name)
                    ))?;
                
                let (_, posterior_type) = node_decl.attrs.iter()
                    .find(|(name, _)| name == attr)
                    .ok_or_else(|| ExecError::ValidationError(
                        format!("Attribute '{}' not declared for node type '{}' in belief model", attr, node.0)
                    ))?;
                
                // Get observation precision from posterior type
                let tau_obs = match posterior_type {
                    PosteriorType::Gaussian { params } => {
                        params.iter()
                            .find(|(name, _)| name == "observation_precision")
                            .map(|(_, v)| *v)
                            .unwrap_or(1.0) // Default observation precision
                    }
                    _ => 1.0,
                };
                
                // Collect observation for parallel processing
                attr_observations.push((*node_id, attr.clone(), *value, tau_obs));
            }
        }
    }
    
    // Apply observations in parallel batches
    // Group by target for deterministic ordering, then process batches in parallel
    apply_observations_parallel(&mut graph, edge_observations, attr_observations)?;
    
    // Apply any pending deltas before returning to ensure nodes() and edges() work correctly
    graph.ensure_owned();
    
    Ok(graph)
}

/// Creates a node if it doesn't exist, initializing attributes from the belief model.
fn create_node_if_needed(
    graph: &mut BeliefGraph,
    node_map: &mut HashMap<(String, String), NodeId>,
    schema: &Schema,
    model: &BeliefModel,
    node_type: &str,
    label: &str,
) -> Result<NodeId, ExecError> {
    // Check if node already exists
    if let Some(&node_id) = node_map.get(&(node_type.to_string(), label.to_string())) {
        return Ok(node_id);
    }
    
    // Find node definition in schema
    let node_def = schema.nodes.iter()
        .find(|n| n.name == node_type)
        .ok_or_else(|| ExecError::ValidationError(
            format!("Node type '{}' not found in schema '{}'", node_type, schema.name)
        ))?;
    
    // Find node belief declaration in model
    let node_belief = model.nodes.iter()
        .find(|n| n.node_type == node_type)
        .ok_or_else(|| ExecError::ValidationError(
            format!("Node type '{}' not declared in belief model '{}'", node_type, model.name)
        ))?;
    
    // Initialize attributes from belief model
    let mut attrs = HashMap::new();
    for attr_def in &node_def.attrs {
        let (_, posterior_type) = node_belief.attrs.iter()
            .find(|(name, _)| name == &attr_def.name)
            .ok_or_else(|| ExecError::ValidationError(
                format!("Attribute '{}' not declared for node type '{}' in belief model", attr_def.name, node_type)
            ))?;
        
        let posterior = match posterior_type {
            PosteriorType::Gaussian { params } => {
                let prior_mean = params.iter()
                    .find(|(name, _)| name == "prior_mean")
                    .map(|(_, v)| *v)
                    .unwrap_or(0.0);
                let prior_precision = params.iter()
                    .find(|(name, _)| name == "prior_precision")
                    .map(|(_, v)| *v)
                    .unwrap_or(0.01);
                GaussianPosterior {
                    mean: prior_mean,
                    precision: prior_precision,
                }
            }
            _ => return Err(ExecError::ValidationError(
                format!("Node attributes must use GaussianPosterior, got {:?}", posterior_type)
            )),
        };
        
        attrs.insert(attr_def.name.clone(), posterior);
    }
    
    // Create node
    let node_id = graph.add_node(node_type.to_string(), attrs);
    node_map.insert((node_type.to_string(), label.to_string()), node_id);
    
    Ok(node_id)
}

/// Creates an edge with the appropriate posterior type from the belief model.
fn create_edge_with_posterior(
    graph: &mut BeliefGraph,
    edge_map: &mut HashMap<(NodeId, NodeId, String), EdgeId>,
    competing_group_map: &mut HashMap<(NodeId, String), CompetingGroupId>,
    next_group_id: &mut u32,
    src: NodeId,
    dst: NodeId,
    edge_type: String,
    edge_decl: &EdgeBeliefDecl,
    schema: &Schema,
) -> Result<EdgeId, ExecError> {
    // Check if edge already exists
    if let Some(&edge_id) = edge_map.get(&(src, dst, edge_type.clone())) {
        return Ok(edge_id);
    }
    
    // Validate edge type exists in schema
    if !schema.edges.iter().any(|e| e.name == edge_type) {
        return Err(ExecError::ValidationError(
            format!("Edge type '{}' not found in schema", edge_type)
        ));
    }
    
    match &edge_decl.exist {
        PosteriorType::Bernoulli { params } => {
            // Independent edge with Beta posterior
            let prior = params.iter()
                .find(|(name, _)| name == "prior")
                .map(|(_, v)| *v)
                .unwrap_or(0.5);
            let pseudo_count = params.iter()
                .find(|(name, _)| name == "pseudo_count")
                .map(|(_, v)| *v)
                .unwrap_or(2.0);
            
            let alpha = prior * pseudo_count;
            let beta = (1.0 - prior) * pseudo_count;
            
            let beta_posterior = BetaPosterior { alpha, beta };
            let edge_id = graph.add_edge(src, dst, edge_type.clone(), beta_posterior);
            edge_map.insert((src, dst, edge_type), edge_id);
            Ok(edge_id)
        }
        PosteriorType::Categorical { group_by, prior, .. } => {
            // Competing edge with Dirichlet posterior
            if group_by != "source" {
                return Err(ExecError::ValidationError(
                    format!("CategoricalPosterior with group_by='{}' not yet supported (only 'source')", group_by)
                ));
            }
            
            // Get or create competing group for this (src, edge_type)
            let group_id = if let Some(&gid) = competing_group_map.get(&(src, edge_type.clone())) {
                gid
            } else {
                let new_group_id = CompetingGroupId(*next_group_id);
                *next_group_id += 1;
                competing_group_map.insert((src, edge_type.clone()), new_group_id);
                new_group_id
            };
            
            // Get the category index (create group if new, or get existing)
            let category_idx = {
                let competing_groups = graph.competing_groups_mut();
                if let Some(existing_group) = competing_groups.get_mut(&group_id) {
                // Check if this destination is already in the group
                if let Some(&idx) = existing_group.category_index.get(&dst) {
                    idx // Already exists, return existing index
                } else {
                    // Add new category to existing group
                    let category_index = existing_group.categories.len();
                    existing_group.categories.push(dst);
                    existing_group.category_index.insert(dst, category_index);
                    
                    // Update Dirichlet posterior to include new category
                    // For uniform prior: add new category with its prior concentration,
                    // preserving existing posterior concentrations (Bayesian principle)
                    let old_concentrations = existing_group.posterior.concentrations.clone();
                    let new_concentrations = if let CategoricalPrior::Uniform { pseudo_count } = prior {
                        // Correct Bayesian approach: preserve existing posteriors, add new category with prior
                        // For uniform prior with K existing categories: prior per category was pseudo_count / K
                        // For new category (K+1): prior concentration is pseudo_count / (K+1)
                        // However, we want to be consistent with the original uniform allocation
                        let k_old = old_concentrations.len();
                        let prior_alpha_new = pseudo_count / (k_old + 1) as f64;

                        let mut new = old_concentrations.clone();
                        new.push(prior_alpha_new);
                        new
                    } else {
                        // For explicit prior, we can't add new categories dynamically
                        return Err(ExecError::ValidationError(
                            "Dynamic category discovery not supported for explicit CategoricalPrior".into()
                        ));
                    };
                    existing_group.posterior = DirichletPosterior::new(new_concentrations);
                    
                    category_index // Return the index we just added
                }
            } else {
                let competing_groups = graph.competing_groups_mut();
                // Create new group
                let initial_concentrations = match prior {
                    CategoricalPrior::Uniform { pseudo_count } => {
                        vec![pseudo_count / 1.0] // Start with one category
                    }
                    CategoricalPrior::Explicit { concentrations } => {
                        if concentrations.len() != 1 {
                            return Err(ExecError::ValidationError(
                                format!("Explicit CategoricalPrior must match number of categories (got {} for first edge)", concentrations.len())
                            ));
                        }
                        concentrations.clone()
                    }
                };
                
                let dirichlet = DirichletPosterior::new(initial_concentrations);
                let new_group = CompetingEdgeGroup::new(
                    group_id,
                    src,
                    edge_type.clone(),
                    vec![dst],
                    dirichlet,
                );
                    competing_groups.insert(group_id, new_group);
                    0 // First category
                }
            };
            
            // Create edge referencing the competing group
            let edge_id = EdgeId(graph.edges().len() as u32);
            let edge = EdgeData {
                id: edge_id,
                src,
                dst,
                ty: Arc::from(edge_type.clone()),
                exist: EdgePosterior::Competing {
                    group_id,
                    category_index: category_idx,
                },
            };
            graph.insert_edge(edge);
            edge_map.insert((src, dst, edge_type), edge_id);
            Ok(edge_id)
        }
        _ => Err(ExecError::ValidationError(
            format!("Edge posterior type not supported: {:?}", edge_decl.exist)
        )),
    }
}

/// Applies observations in parallel batches for better performance.
///
/// Groups observations by target (EdgeId or (NodeId, attr)) for deterministic ordering,
/// then processes batches in parallel where targets don't conflict.
/// Uses a mutex for thread-safe access to the graph when parallel processing is enabled.
fn apply_observations_parallel(
    graph: &mut BeliefGraph,
    edge_observations: Vec<(EdgeId, EvidenceMode)>,
    attr_observations: Vec<(NodeId, String, f64, f64)>,
) -> Result<(), ExecError> {
    // Group edge observations by EdgeId for deterministic ordering
    // Multiple observations on the same edge must be applied sequentially
    let mut edge_groups: HashMap<EdgeId, Vec<EvidenceMode>> = HashMap::new();
    for (edge_id, mode) in edge_observations {
        edge_groups.entry(edge_id).or_insert_with(Vec::new).push(mode);
    }
    
    // Group attribute observations by (NodeId, attr) for deterministic ordering
    // Multiple observations on the same attribute must be applied sequentially
    let mut attr_groups: HashMap<(NodeId, String), Vec<(f64, f64)>> = HashMap::new();
    for (node_id, attr, value, tau_obs) in attr_observations {
        attr_groups.entry((node_id, attr)).or_insert_with(Vec::new).push((value, tau_obs));
    }
    
    // Convert to sorted vectors for deterministic processing
    let mut edge_groups_vec: Vec<(EdgeId, Vec<EvidenceMode>)> = edge_groups.into_iter().collect();
    edge_groups_vec.sort_by_key(|(edge_id, _)| *edge_id);
    
    let mut attr_groups_vec: Vec<((NodeId, String), Vec<(f64, f64)>)> = attr_groups.into_iter().collect();
    attr_groups_vec.sort_by_key(|((node_id, attr), _)| (*node_id, attr.clone()));
    
    // Apply observations: sequential within groups (order matters for Bayesian updates),
    // but groups can be processed in parallel if they don't conflict
    #[cfg(feature = "rayon")]
    {
        use std::sync::{Arc, Mutex};
        
        // Wrap graph in Arc<Mutex> for thread-safe parallel access
        // Note: This serializes access to the graph, but we group by target to minimize contention
        // Each group processes all its observations in a single lock, reducing lock overhead
        let graph_mutex = Arc::new(Mutex::new(std::mem::take(graph)));
        
        // Process edge observations in parallel batches
        edge_groups_vec
            .par_iter()
            .try_for_each(|(edge_id, modes)| -> Result<(), ExecError> {
                // Apply all observations for this edge sequentially in a single lock
                let mut graph_guard = graph_mutex.lock().unwrap();
                for mode in modes {
                    apply_edge_observation_single(&mut *graph_guard, *edge_id, mode.clone())?;
                }
                Ok(())
            })?;
        
        // Process attribute observations in parallel batches
        attr_groups_vec
            .par_iter()
            .try_for_each(|((node_id, attr), values)| -> Result<(), ExecError> {
                // Apply all observations for this attribute sequentially in a single lock
                let mut graph_guard = graph_mutex.lock().unwrap();
                for (value, tau_obs) in values {
                    graph_guard.observe_attr(*node_id, attr, *value, *tau_obs)?;
                }
                Ok(())
            })?;
        
        // Extract graph back from mutex
        *graph = Arc::try_unwrap(graph_mutex)
            .unwrap()
            .into_inner()
            .unwrap();
    }
    
    #[cfg(not(feature = "rayon"))]
    {
        // Sequential fallback when rayon is not available
        for (edge_id, modes) in edge_groups_vec {
            for mode in modes {
                apply_edge_observation_single(graph, edge_id, mode)?;
            }
        }
        
        for ((node_id, attr), values) in attr_groups_vec {
            for (value, tau_obs) in values {
                graph.observe_attr(node_id, &attr, value, tau_obs)?;
            }
        }
    }
    
    Ok(())
}

/// Applies a single edge observation to update the posterior.
fn apply_edge_observation_single(
    graph: &mut BeliefGraph,
    edge_id: EdgeId,
    mode: EvidenceMode,
) -> Result<(), ExecError> {
    match mode {
        EvidenceMode::Present => {
            graph.observe_edge(edge_id, true)?;
        }
        EvidenceMode::Absent => {
            graph.observe_edge(edge_id, false)?;
        }
        EvidenceMode::Chosen => {
            graph.observe_edge_chosen(edge_id)?;
        }
        EvidenceMode::Unchosen => {
            graph.observe_edge_unchosen(edge_id)?;
        }
        EvidenceMode::ForcedChoice => {
            graph.observe_edge_forced_choice(edge_id)?;
        }
    }
    Ok(())
}

