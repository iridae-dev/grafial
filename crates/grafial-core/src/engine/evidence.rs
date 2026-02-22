//! Evidence building module.
//!
//! Converts evidence definitions to BeliefGraph instances by:
//! - Creating nodes and edges on demand from observations
//! - Initializing posteriors from belief model declarations
//! - Applying evidence observations to update posteriors
//! - Managing competing edge groups for categorical posteriors

use std::collections::{HashMap, HashSet};
use std::sync::Arc;

#[cfg(all(not(feature = "parallel"), feature = "rayon"))]
use rayon::prelude::*;

#[cfg(feature = "parallel")]
use super::parallel_evidence::{apply_parallel_results, process_evidence_parallel};
use crate::engine::errors::ExecError;
use crate::engine::graph::{
    BeliefGraph, BetaPosterior, CompetingEdgeGroup, CompetingGroupId, DirichletPosterior, EdgeData,
    EdgeId, EdgePosterior, GaussianPosterior, NodeId,
};
use grafial_frontend::ast::{
    BeliefModel, CategoricalPrior, EdgeBeliefDecl, EvidenceDef, EvidenceMode, NodeBeliefDecl,
    ObserveStmt, PosteriorType, Schema,
};
use grafial_ir::{EvidenceIR, ProgramIR};

struct EvidenceBuildContext<'a> {
    model: &'a BeliefModel,
    schema: &'a Schema,
}

struct EdgeBuildInput {
    src: NodeId,
    dst: NodeId,
    edge_type: String,
}

#[cfg(not(feature = "parallel"))]
type EdgeObservation = (EdgeId, EvidenceMode);
#[cfg(not(feature = "parallel"))]
type AttrObservationEntry = (NodeId, String, f64, f64);
#[cfg(not(feature = "parallel"))]
type AttrObservation = (f64, f64);
#[cfg(not(feature = "parallel"))]
type AttrObservationGroup = ((NodeId, String), Vec<AttrObservation>);
#[cfg(not(feature = "parallel"))]
type PreparedObservationSplit = (Vec<EdgeObservation>, Vec<AttrObservationEntry>);

#[derive(Debug, Clone)]
enum PreparedObservation {
    Edge {
        edge_id: EdgeId,
        edge_type: String,
        src: (String, String),
        dst: (String, String),
        mode: EvidenceMode,
    },
    Attribute {
        node_id: NodeId,
        node: (String, String),
        attr: String,
        value: f64,
        precision: f64,
    },
}

/// Builds a BeliefGraph from an evidence definition.
///
/// Nodes and edges are created on demand from observations. Posteriors are initialized
/// from the belief model and updated with evidence observations. Competing edge groups
/// are managed automatically for categorical posteriors.
pub fn build_graph_from_evidence(
    evidence: &EvidenceDef,
    program: &grafial_frontend::ast::ProgramAst,
) -> Result<BeliefGraph, ExecError> {
    let context = resolve_evidence_context(
        &evidence.name,
        &evidence.on_model,
        &program.belief_models,
        &program.schemas,
    )?;
    build_graph_from_evidence_with_context(evidence, context)
}

/// Builds a BeliefGraph from an IR evidence definition.
///
/// This adapter keeps evidence construction logic centralized in one implementation.
pub fn build_graph_from_evidence_ir(
    evidence: &EvidenceIR,
    program: &ProgramIR,
) -> Result<BeliefGraph, ExecError> {
    let evidence_ast = evidence.to_ast();
    let context = resolve_evidence_context(
        &evidence_ast.name,
        &evidence_ast.on_model,
        &program.belief_models,
        &program.schemas,
    )?;
    build_graph_from_evidence_with_context(&evidence_ast, context)
}

fn resolve_evidence_context<'a>(
    evidence_name: &str,
    on_model: &str,
    belief_models: &'a [BeliefModel],
    schemas: &'a [Schema],
) -> Result<EvidenceBuildContext<'a>, ExecError> {
    let model = belief_models
        .iter()
        .find(|m| m.name == on_model)
        .ok_or_else(|| {
            ExecError::ValidationError(format!(
                "Evidence '{}' references unknown belief model '{}'",
                evidence_name, on_model
            ))
        })?;

    let schema = schemas
        .iter()
        .find(|s| s.name == model.on_schema)
        .ok_or_else(|| {
            ExecError::ValidationError(format!(
                "Belief model '{}' references unknown schema '{}'",
                model.name, model.on_schema
            ))
        })?;

    Ok(EvidenceBuildContext { model, schema })
}

fn build_graph_from_evidence_with_context(
    evidence: &EvidenceDef,
    context: EvidenceBuildContext<'_>,
) -> Result<BeliefGraph, ExecError> {
    let model = context.model;
    let schema = context.schema;
    let mut graph = BeliefGraph::default();

    // Index maps for efficient node/edge/group lookup during construction
    let mut node_map: HashMap<(String, String), NodeId> = HashMap::new();
    let mut edge_map: HashMap<(NodeId, NodeId, String), EdgeId> = HashMap::new();
    let mut competing_group_map: HashMap<(NodeId, String), CompetingGroupId> = HashMap::new();

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
        let node_id =
            create_node_if_needed(&mut graph, &mut node_map, schema, model, &node_type, &label)?;
        // Ensure node_id is in map (should already be, but be safe)
        node_map.insert((node_type, label), node_id);
    }

    // Precompute fixed category sets for competing groups before any posterior updates.
    // This avoids order-dependent retroactive changes to Dirichlet interpretation.
    let planned_competing_categories = plan_competing_group_categories(evidence, model, &node_map)?;

    // Second pass: create edges and normalize observations once.
    let mut prepared_observations = Vec::with_capacity(evidence.observations.len());

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
                    .ok_or_else(|| {
                        ExecError::Internal(format!(
                            "Missing source node: {}[\"{}\"]",
                            src.0, src.1
                        ))
                    })?;
                let dst_id = node_map
                    .get(&(dst.0.clone(), dst.1.clone()))
                    .ok_or_else(|| {
                        ExecError::Internal(format!(
                            "Missing destination node: {}[\"{}\"]",
                            dst.0, dst.1
                        ))
                    })?;

                // Find edge declaration in belief model
                let edge_decl = model
                    .edges
                    .iter()
                    .find(|e| e.edge_type == *edge_type)
                    .ok_or_else(|| {
                        ExecError::ValidationError(format!(
                            "Edge type '{}' not declared in belief model '{}'",
                            edge_type, model.name
                        ))
                    })?;

                // Create edge if it doesn't exist
                let edge_id =
                    if let Some(eid) = edge_map.get(&(*src_id, *dst_id, edge_type.clone())) {
                        *eid
                    } else {
                        create_edge_with_posterior(
                            &mut graph,
                            &mut edge_map,
                            &mut competing_group_map,
                            EdgeBuildInput {
                                src: *src_id,
                                dst: *dst_id,
                                edge_type: edge_type.clone(),
                            },
                            edge_decl,
                            schema,
                            &planned_competing_categories,
                        )?
                    };

                prepared_observations.push(PreparedObservation::Edge {
                    edge_id,
                    edge_type: edge_type.clone(),
                    src: src.clone(),
                    dst: dst.clone(),
                    mode: mode.clone(),
                });
            }
            ObserveStmt::Attribute {
                node,
                attr,
                value,
                precision,
            } => {
                let node_id = node_map
                    .get(&(node.0.clone(), node.1.clone()))
                    .ok_or_else(|| {
                        ExecError::Internal(format!("Missing node: {}[\"{}\"]", node.0, node.1))
                    })?;

                // Find attribute declaration in belief model
                let node_decl = model
                    .nodes
                    .iter()
                    .find(|n| n.node_type == node.0)
                    .ok_or_else(|| {
                        ExecError::ValidationError(format!(
                            "Node type '{}' not declared in belief model '{}'",
                            node.0, model.name
                        ))
                    })?;

                let (_, posterior_type) = node_decl
                    .attrs
                    .iter()
                    .find(|(name, _)| name == attr)
                    .ok_or_else(|| {
                        ExecError::ValidationError(format!(
                            "Attribute '{}' not declared for node type '{}' in belief model",
                            attr, node.0
                        ))
                    })?;

                // Get observation precision from posterior type
                let tau_obs = if let Some(p) = precision {
                    *p
                } else {
                    match posterior_type {
                        PosteriorType::Gaussian { params } => {
                            params
                                .iter()
                                .find(|(name, _)| name == "observation_precision")
                                .map(|(_, v)| *v)
                                .unwrap_or(1.0) // Default observation precision
                        }
                        _ => 1.0,
                    }
                };

                prepared_observations.push(PreparedObservation::Attribute {
                    node_id: *node_id,
                    node: node.clone(),
                    attr: attr.clone(),
                    value: *value,
                    precision: tau_obs,
                });
            }
        }
    }

    // Apply observations in parallel batches when parallel feature is enabled
    #[cfg(feature = "parallel")]
    {
        let parallel_observations = prepared_to_parallel_observations(&prepared_observations);
        let parallel_result =
            process_evidence_parallel(&graph, &parallel_observations, &node_map, &edge_map)?;

        // Apply parallel results to the graph
        apply_parallel_results(&mut graph, parallel_result)?;
    }

    #[cfg(not(feature = "parallel"))]
    {
        let (edge_observations, attr_observations) =
            split_prepared_observations(prepared_observations);
        // Sequential fallback: group by target for deterministic ordering
        apply_observations_sequential(&mut graph, edge_observations, attr_observations)?;
    }

    // Apply any pending deltas before returning to ensure nodes() and edges() work correctly
    graph.ensure_owned();

    Ok(graph)
}

fn plan_competing_group_categories(
    evidence: &EvidenceDef,
    model: &BeliefModel,
    node_map: &HashMap<(String, String), NodeId>,
) -> Result<HashMap<(NodeId, String), Vec<NodeId>>, ExecError> {
    let mut planned: HashMap<(NodeId, String), Vec<NodeId>> = HashMap::new();

    for obs in &evidence.observations {
        let ObserveStmt::Edge {
            edge_type,
            src,
            dst,
            ..
        } = obs
        else {
            continue;
        };

        let edge_decl = model
            .edges
            .iter()
            .find(|e| e.edge_type == *edge_type)
            .ok_or_else(|| {
                ExecError::ValidationError(format!(
                    "Edge type '{}' not declared in belief model '{}'",
                    edge_type, model.name
                ))
            })?;

        let PosteriorType::Categorical { categories, .. } = &edge_decl.exist else {
            continue;
        };

        if let Some(allowed_categories) = categories {
            if !allowed_categories.iter().any(|name| name == &dst.1) {
                return Err(ExecError::ValidationError(format!(
                    "Destination '{}' is not allowed for categorical edge '{}'; expected one of {:?}",
                    dst.1, edge_type, allowed_categories
                )));
            }
        }

        let src_id = *node_map
            .get(&(src.0.clone(), src.1.clone()))
            .ok_or_else(|| {
                ExecError::Internal(format!("Missing source node: {}[\"{}\"]", src.0, src.1))
            })?;
        let dst_id = *node_map
            .get(&(dst.0.clone(), dst.1.clone()))
            .ok_or_else(|| {
                ExecError::Internal(format!(
                    "Missing destination node: {}[\"{}\"]",
                    dst.0, dst.1
                ))
            })?;

        planned
            .entry((src_id, edge_type.clone()))
            .or_default()
            .push(dst_id);
    }

    for categories in planned.values_mut() {
        categories.sort();
        categories.dedup();
    }

    Ok(planned)
}

#[cfg(feature = "parallel")]
fn prepared_to_parallel_observations(
    prepared_observations: &[PreparedObservation],
) -> Vec<ObserveStmt> {
    prepared_observations
        .iter()
        .map(|obs| match obs {
            PreparedObservation::Edge {
                edge_id,
                edge_type,
                src,
                dst,
                mode,
            } => {
                let _ = edge_id;
                ObserveStmt::Edge {
                    edge_type: edge_type.clone(),
                    src: src.clone(),
                    dst: dst.clone(),
                    mode: mode.clone(),
                }
            }
            PreparedObservation::Attribute {
                node_id,
                node,
                attr,
                value,
                precision,
            } => {
                let _ = node_id;
                ObserveStmt::Attribute {
                    node: node.clone(),
                    attr: attr.clone(),
                    value: *value,
                    precision: Some(*precision),
                }
            }
        })
        .collect()
}

#[cfg(not(feature = "parallel"))]
fn split_prepared_observations(
    prepared_observations: Vec<PreparedObservation>,
) -> PreparedObservationSplit {
    let mut edge_observations = Vec::new();
    let mut attr_observations = Vec::new();

    for obs in prepared_observations {
        match obs {
            PreparedObservation::Edge {
                edge_id,
                edge_type,
                src,
                dst,
                mode,
            } => {
                let _ = (edge_type, src, dst);
                edge_observations.push((edge_id, mode));
            }
            PreparedObservation::Attribute {
                node_id,
                node,
                attr,
                value,
                precision,
            } => {
                let _ = node;
                attr_observations.push((node_id, attr, value, precision));
            }
        }
    }

    (edge_observations, attr_observations)
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
    let node_def = schema
        .nodes
        .iter()
        .find(|n| n.name == node_type)
        .ok_or_else(|| {
            ExecError::ValidationError(format!(
                "Node type '{}' not found in schema '{}'",
                node_type, schema.name
            ))
        })?;

    // Find node belief declaration in model
    let node_belief = model
        .nodes
        .iter()
        .find(|n| n.node_type == node_type)
        .ok_or_else(|| {
            ExecError::ValidationError(format!(
                "Node type '{}' not declared in belief model '{}'",
                node_type, model.name
            ))
        })?;
    let fixed_correlations = extract_fixed_gaussian_correlations(node_belief, &model.name)?;

    // Initialize attributes from belief model
    let mut attrs = HashMap::new();
    for attr_def in &node_def.attrs {
        let (_, posterior_type) = node_belief
            .attrs
            .iter()
            .find(|(name, _)| name == &attr_def.name)
            .ok_or_else(|| {
                ExecError::ValidationError(format!(
                    "Attribute '{}' not declared for node type '{}' in belief model",
                    attr_def.name, node_type
                ))
            })?;

        let posterior = match posterior_type {
            PosteriorType::Gaussian { params } => {
                let prior_mean = params
                    .iter()
                    .find(|(name, _)| name == "prior_mean")
                    .map(|(_, v)| *v)
                    .unwrap_or(0.0);
                let prior_precision = params
                    .iter()
                    .find(|(name, _)| name == "prior_precision")
                    .map(|(_, v)| *v)
                    .unwrap_or(0.01);
                GaussianPosterior {
                    mean: prior_mean,
                    precision: prior_precision,
                }
            }
            _ => {
                return Err(ExecError::ValidationError(format!(
                    "Node attributes must use GaussianPosterior, got {:?}",
                    posterior_type
                )))
            }
        };

        attrs.insert(attr_def.name.clone(), posterior);
    }

    // Create node
    let node_id = graph.add_node(node_type.to_string(), attrs);
    node_map.insert((node_type.to_string(), label.to_string()), node_id);
    for ((left_attr, right_attr), rho) in fixed_correlations {
        graph.set_attr_correlation(node_id, &left_attr, &right_attr, rho)?;
    }

    Ok(node_id)
}

fn extract_fixed_gaussian_correlations(
    node_belief: &NodeBeliefDecl,
    model_name: &str,
) -> Result<HashMap<(String, String), f64>, ExecError> {
    let declared_attrs: HashSet<&str> = node_belief
        .attrs
        .iter()
        .map(|(attr_name, _)| attr_name.as_str())
        .collect();
    let mut correlations: HashMap<(String, String), f64> = HashMap::new();

    for (attr_name, posterior) in &node_belief.attrs {
        let PosteriorType::Gaussian { params } = posterior else {
            continue;
        };

        for (param_name, value) in params {
            let Some(other_attr) = param_name.strip_prefix("corr_") else {
                continue;
            };

            if other_attr.is_empty() {
                return Err(ExecError::ValidationError(format!(
                    "Invalid Gaussian correlation parameter '{}' on '{}.{}' in belief model '{}'",
                    param_name, node_belief.node_type, attr_name, model_name
                )));
            }
            if !declared_attrs.contains(other_attr) {
                return Err(ExecError::ValidationError(format!(
                    "Correlation parameter '{}.{} = {}' references unknown attribute '{}' in belief model '{}'",
                    node_belief.node_type, param_name, value, other_attr, model_name
                )));
            }
            if !value.is_finite() || !(-1.0..=1.0).contains(value) {
                return Err(ExecError::ValidationError(format!(
                    "Correlation '{}.{}' must be finite and in [-1, 1], got {}",
                    node_belief.node_type, param_name, value
                )));
            }
            if other_attr == attr_name && (*value - 1.0).abs() > 1e-12 {
                return Err(ExecError::ValidationError(format!(
                    "Self-correlation '{}.{}' must be exactly 1, got {}",
                    node_belief.node_type, attr_name, value
                )));
            }

            let pair_key = if attr_name.as_str() <= other_attr {
                (attr_name.clone(), other_attr.to_string())
            } else {
                (other_attr.to_string(), attr_name.clone())
            };

            if let Some(prev) = correlations.insert(pair_key.clone(), *value) {
                if (prev - value).abs() > 1e-12 {
                    return Err(ExecError::ValidationError(format!(
                        "Conflicting correlation declarations for '{}.{}' and '{}.{}' in belief model '{}': {} vs {}",
                        node_belief.node_type,
                        pair_key.0,
                        node_belief.node_type,
                        pair_key.1,
                        model_name,
                        prev,
                        value
                    )));
                }
            }
        }
    }

    Ok(correlations)
}

/// Creates an edge with the appropriate posterior type from the belief model.
fn create_edge_with_posterior(
    graph: &mut BeliefGraph,
    edge_map: &mut HashMap<(NodeId, NodeId, String), EdgeId>,
    competing_group_map: &mut HashMap<(NodeId, String), CompetingGroupId>,
    edge: EdgeBuildInput,
    edge_decl: &EdgeBeliefDecl,
    schema: &Schema,
    planned_competing_categories: &HashMap<(NodeId, String), Vec<NodeId>>,
) -> Result<EdgeId, ExecError> {
    let EdgeBuildInput {
        src,
        dst,
        edge_type,
    } = edge;

    // Check if edge already exists
    if let Some(&edge_id) = edge_map.get(&(src, dst, edge_type.clone())) {
        return Ok(edge_id);
    }

    // Validate edge type exists in schema
    if !schema.edges.iter().any(|e| e.name == edge_type) {
        return Err(ExecError::ValidationError(format!(
            "Edge type '{}' not found in schema",
            edge_type
        )));
    }

    match &edge_decl.exist {
        PosteriorType::Bernoulli { params } => {
            // Independent edge with Beta posterior
            let prior = params
                .iter()
                .find(|(name, _)| name == "prior")
                .map(|(_, v)| *v)
                .unwrap_or(0.5);
            let pseudo_count = params
                .iter()
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
        PosteriorType::Categorical {
            group_by, prior, ..
        } => {
            // Competing edge with Dirichlet posterior
            if group_by != "source" {
                return Err(ExecError::ValidationError(format!(
                    "CategoricalPosterior with group_by='{}' not yet supported (only 'source')",
                    group_by
                )));
            }

            // Get or create competing group for this (src, edge_type)
            let group_id = if let Some(&gid) = competing_group_map.get(&(src, edge_type.clone())) {
                gid
            } else {
                let new_group_id = CompetingGroupId(competing_group_map.len() as u32);
                competing_group_map.insert((src, edge_type.clone()), new_group_id);
                new_group_id
            };

            // Get the category index (create group if new, or use existing fixed group)
            let category_idx = {
                let competing_groups = graph.competing_groups_mut();
                if let Some(existing_group) = competing_groups.get_mut(&group_id) {
                    if let Some(&idx) = existing_group.category_index.get(&dst) {
                        idx
                    } else {
                        return Err(ExecError::ValidationError(format!(
                            "Dynamic category discovery is not supported for competing edge '{}': destination {:?} is outside the fixed category set",
                            edge_type, dst
                        )));
                    }
                } else {
                    let competing_groups = graph.competing_groups_mut();

                    let mut categories = planned_competing_categories
                        .get(&(src, edge_type.clone()))
                        .cloned()
                        .unwrap_or_else(|| vec![dst]);
                    categories.sort();
                    categories.dedup();
                    if !categories.contains(&dst) {
                        return Err(ExecError::Internal(format!(
                            "Destination {:?} missing from planned category set for edge '{}'",
                            dst, edge_type
                        )));
                    }

                    let initial_concentrations = match prior {
                        CategoricalPrior::Uniform { pseudo_count } => {
                            vec![pseudo_count / categories.len() as f64; categories.len()]
                        }
                        CategoricalPrior::Explicit { concentrations } => {
                            if concentrations.len() != categories.len() {
                                return Err(ExecError::ValidationError(
                                format!(
                                    "Explicit CategoricalPrior must match fixed category count (got {}, expected {})",
                                    concentrations.len(),
                                    categories.len()
                                )
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
                        categories,
                        dirichlet,
                    );
                    let category_index = new_group.get_category_index(dst).ok_or_else(|| {
                        ExecError::Internal(format!(
                            "Destination {:?} missing from new competing group for edge '{}'",
                            dst, edge_type
                        ))
                    })?;
                    competing_groups.insert(group_id, new_group);
                    category_index
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
        _ => Err(ExecError::ValidationError(format!(
            "Edge posterior type not supported: {:?}",
            edge_decl.exist
        ))),
    }
}

/// Applies observations sequentially with optional vectorized batching.
///
/// Groups observations by target (EdgeId or (NodeId, attr)) for deterministic ordering,
/// then applies updates either using vectorized kernels (when available) or sequentially.
#[cfg(not(feature = "parallel"))]
fn apply_observations_sequential(
    graph: &mut BeliefGraph,
    edge_observations: Vec<(EdgeId, EvidenceMode)>,
    attr_observations: Vec<(NodeId, String, f64, f64)>,
) -> Result<(), ExecError> {
    // Group edge observations by EdgeId for deterministic ordering
    // Multiple observations on the same edge must be applied sequentially
    let mut edge_groups: HashMap<EdgeId, Vec<EvidenceMode>> = HashMap::new();
    for (edge_id, mode) in edge_observations {
        edge_groups.entry(edge_id).or_default().push(mode);
    }

    // Group attribute observations by (NodeId, attr) for deterministic ordering
    // Multiple observations on the same attribute must be applied sequentially
    let mut attr_groups: HashMap<(NodeId, String), Vec<AttrObservation>> = HashMap::new();
    for (node_id, attr, value, tau_obs) in attr_observations {
        attr_groups
            .entry((node_id, attr))
            .or_default()
            .push((value, tau_obs));
    }

    // Convert to sorted vectors for deterministic processing
    let mut edge_groups_vec: Vec<(EdgeId, Vec<EvidenceMode>)> = edge_groups.into_iter().collect();
    edge_groups_vec.sort_by_key(|(edge_id, _)| *edge_id);

    let mut attr_groups_vec: Vec<AttrObservationGroup> = attr_groups.into_iter().collect();
    attr_groups_vec.sort_by_key(|((node_id, attr), _)| (*node_id, attr.clone()));

    // Apply observations using vectorized kernels when available
    #[cfg(feature = "vectorized")]
    {
        // Vectorized path: batch observations per target for efficient kernel execution
        for (edge_id, modes) in edge_groups_vec {
            // Separate Present/Absent observations for batching from other modes
            let mut beta_observations = Vec::new();
            let mut other_modes = Vec::new();

            for mode in modes {
                match mode {
                    EvidenceMode::Present => beta_observations.push(true),
                    EvidenceMode::Absent => beta_observations.push(false),
                    _ => other_modes.push(mode),
                }
            }

            // Apply batched Beta updates if any
            if !beta_observations.is_empty() {
                graph.observe_edge_batch(edge_id, &beta_observations)?;
            }

            // Handle non-Beta updates separately (they don't batch)
            for mode in other_modes {
                match mode {
                    EvidenceMode::Chosen => graph.observe_edge_chosen(edge_id)?,
                    EvidenceMode::Unchosen => graph.observe_edge_unchosen(edge_id)?,
                    EvidenceMode::ForcedChoice => graph.observe_edge_forced_choice(edge_id)?,
                    _ => {} // Should not happen
                }
            }
        }

        // Batch attribute observations for Gaussian updates
        for ((node_id, attr), values) in attr_groups_vec {
            graph.observe_attr_batch(node_id, &attr, &values)?;
        }
    }

    #[cfg(all(not(feature = "vectorized"), feature = "rayon"))]
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
                    apply_edge_observation_single(&mut graph_guard, *edge_id, mode.clone())?;
                }
                Ok(())
            })?;

        // Process attribute observations in parallel batches
        attr_groups_vec.par_iter().try_for_each(
            |((node_id, attr), values)| -> Result<(), ExecError> {
                // Apply all observations for this attribute sequentially in a single lock
                let mut graph_guard = graph_mutex.lock().unwrap();
                for (value, tau_obs) in values {
                    graph_guard.observe_attr(*node_id, attr, *value, *tau_obs)?;
                }
                Ok(())
            },
        )?;

        // Extract graph back from mutex
        *graph = Arc::try_unwrap(graph_mutex).unwrap().into_inner().unwrap();
    }

    #[cfg(all(not(feature = "vectorized"), not(feature = "rayon")))]
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
#[cfg(all(not(feature = "parallel"), not(feature = "vectorized")))]
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
