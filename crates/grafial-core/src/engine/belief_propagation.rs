//! Belief propagation transform for flow pipelines.
//!
//! This module implements deterministic loopy sum-product belief propagation over
//! independent edge variables. Pairwise factors are induced between edges that
//! share a node and edge type, which provides local smoothing while preserving
//! each edge's effective sample size.

use std::collections::HashMap;
use std::sync::Arc;

use crate::engine::errors::ExecError;
use crate::engine::graph::{BeliefGraph, EdgeId, EdgePosterior, NodeId};

const MIN_BETA_PARAM: f64 = 0.01;
const MIN_PROBABILITY: f64 = 1e-6;

/// Configuration for loopy belief propagation.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct BeliefPropagationConfig {
    /// Maximum synchronous message-passing iterations.
    pub max_iterations: usize,
    /// Damping factor in [0, 1). Higher values slow updates and improve stability.
    pub damping: f64,
    /// Convergence threshold on max absolute message delta.
    pub convergence_tolerance: f64,
    /// Coupling strength between neighboring edges. Positive favors agreement.
    pub coupling_strength: f64,
}

impl Default for BeliefPropagationConfig {
    fn default() -> Self {
        Self {
            max_iterations: 32,
            damping: 0.35,
            convergence_tolerance: 1e-5,
            coupling_strength: 0.6,
        }
    }
}

impl BeliefPropagationConfig {
    fn validate(self) -> Result<Self, ExecError> {
        if self.max_iterations == 0 {
            return Err(ExecError::ValidationError(
                "infer_beliefs: max_iterations must be > 0".into(),
            ));
        }
        if !(0.0..1.0).contains(&self.damping) {
            return Err(ExecError::ValidationError(
                "infer_beliefs: damping must be in [0, 1)".into(),
            ));
        }
        if self.convergence_tolerance <= 0.0 || !self.convergence_tolerance.is_finite() {
            return Err(ExecError::ValidationError(
                "infer_beliefs: convergence_tolerance must be finite and > 0".into(),
            ));
        }
        if !self.coupling_strength.is_finite() {
            return Err(ExecError::ValidationError(
                "infer_beliefs: coupling_strength must be finite".into(),
            ));
        }
        Ok(self)
    }
}

#[derive(Debug, Clone)]
struct EdgeVariable {
    edge_id: EdgeId,
    src: NodeId,
    dst: NodeId,
    edge_type: Arc<str>,
    prior_alpha: f64,
    prior_beta: f64,
}

/// Runtime diagnostics emitted by loopy belief propagation.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct BeliefPropagationDiagnostics {
    /// Iteration limit configured for this run.
    pub max_iterations: usize,
    /// Number of synchronous iterations actually executed.
    pub iterations_run: usize,
    /// Whether convergence tolerance was reached before the iteration limit.
    pub converged: bool,
    /// Final max absolute message delta after the last iteration.
    pub final_max_message_delta: f64,
    /// Number of independent edge variables considered.
    pub variable_count: usize,
    /// Number of variables with at least one neighbor in the factor graph.
    pub connected_variable_count: usize,
}

/// Runs loopy sum-product belief propagation with default configuration.
pub fn run_loopy_belief_propagation(input: &BeliefGraph) -> Result<BeliefGraph, ExecError> {
    run_loopy_belief_propagation_with_config(input, BeliefPropagationConfig::default())
}

/// Runs loopy sum-product belief propagation with explicit configuration.
pub fn run_loopy_belief_propagation_with_config(
    input: &BeliefGraph,
    config: BeliefPropagationConfig,
) -> Result<BeliefGraph, ExecError> {
    run_loopy_belief_propagation_with_config_diagnostics(input, config).map(|(graph, _)| graph)
}

/// Runs loopy sum-product belief propagation with default configuration and diagnostics.
pub fn run_loopy_belief_propagation_with_diagnostics(
    input: &BeliefGraph,
) -> Result<(BeliefGraph, BeliefPropagationDiagnostics), ExecError> {
    run_loopy_belief_propagation_with_config_diagnostics(input, BeliefPropagationConfig::default())
}

/// Runs loopy sum-product belief propagation with explicit configuration and diagnostics.
pub fn run_loopy_belief_propagation_with_config_diagnostics(
    input: &BeliefGraph,
    config: BeliefPropagationConfig,
) -> Result<(BeliefGraph, BeliefPropagationDiagnostics), ExecError> {
    let config = config.validate()?;

    let mut output = input.clone();
    output.ensure_owned();

    let variables = collect_independent_edge_variables(&output);
    let variable_count = variables.len();
    let mut diagnostics = BeliefPropagationDiagnostics {
        max_iterations: config.max_iterations,
        iterations_run: 0,
        converged: true,
        final_max_message_delta: 0.0,
        variable_count,
        connected_variable_count: 0,
    };

    if variables.len() <= 1 {
        return Ok((output, diagnostics));
    }

    let neighbors = build_neighborhood_graph(&variables);
    diagnostics.connected_variable_count = neighbors.iter().filter(|adj| !adj.is_empty()).count();
    if neighbors.iter().all(|n| n.is_empty()) {
        return Ok((output, diagnostics));
    }
    let reverse = build_reverse_neighbor_index(&neighbors)?;
    let unary = build_unary_log_potentials(&variables);

    let mut messages: Vec<Vec<[f64; 2]>> = neighbors
        .iter()
        .map(|n| vec![[0.5, 0.5]; n.len()])
        .collect();
    let mut next_messages = messages.clone();

    let log_same = config.coupling_strength;
    let log_diff = -config.coupling_strength;

    diagnostics.converged = false;
    for iteration in 0..config.max_iterations {
        let mut max_delta = 0.0_f64;

        for (src_idx, src_neighbors) in neighbors.iter().enumerate() {
            for (neighbor_slot, _) in src_neighbors.iter().enumerate() {
                let mut log_prod_absent = unary[src_idx][0];
                let mut log_prod_present = unary[src_idx][1];

                for (other_slot, &other_idx) in src_neighbors.iter().enumerate() {
                    if other_slot == neighbor_slot {
                        continue;
                    }
                    let reverse_slot = reverse[src_idx][other_slot];
                    let incoming = messages[other_idx][reverse_slot];
                    log_prod_absent += incoming[0].max(MIN_PROBABILITY).ln();
                    log_prod_present += incoming[1].max(MIN_PROBABILITY).ln();
                }

                let log_msg_absent =
                    log_sum_exp(log_same + log_prod_absent, log_diff + log_prod_present);
                let log_msg_present =
                    log_sum_exp(log_diff + log_prod_absent, log_same + log_prod_present);
                let log_norm = log_sum_exp(log_msg_absent, log_msg_present);

                let fresh = [
                    (log_msg_absent - log_norm)
                        .exp()
                        .clamp(MIN_PROBABILITY, 1.0 - MIN_PROBABILITY),
                    (log_msg_present - log_norm)
                        .exp()
                        .clamp(MIN_PROBABILITY, 1.0 - MIN_PROBABILITY),
                ];
                let current = messages[src_idx][neighbor_slot];
                let updated = damped_normalize(current, fresh, config.damping);

                max_delta = max_delta.max(
                    (updated[0] - current[0])
                        .abs()
                        .max((updated[1] - current[1]).abs()),
                );
                next_messages[src_idx][neighbor_slot] = updated;
            }
        }

        std::mem::swap(&mut messages, &mut next_messages);
        diagnostics.iterations_run = iteration + 1;
        diagnostics.final_max_message_delta = max_delta;
        if max_delta < config.convergence_tolerance {
            diagnostics.converged = true;
            break;
        }
    }

    let marginals = compute_marginals(&unary, &neighbors, &reverse, &messages);
    for (variable, p_present) in variables.iter().zip(marginals.into_iter()) {
        write_posterior_mean_preserving_strength(
            &mut output,
            variable.edge_id,
            variable.prior_alpha,
            variable.prior_beta,
            p_present,
        )?;
    }

    Ok((output, diagnostics))
}

fn collect_independent_edge_variables(graph: &BeliefGraph) -> Vec<EdgeVariable> {
    let mut variables = Vec::new();
    for edge in graph.edges() {
        if let EdgePosterior::Independent(beta) = edge.exist {
            variables.push(EdgeVariable {
                edge_id: edge.id,
                src: edge.src,
                dst: edge.dst,
                edge_type: edge.ty.clone(),
                prior_alpha: beta.alpha,
                prior_beta: beta.beta,
            });
        }
    }
    variables.sort_unstable_by_key(|v| v.edge_id);
    variables
}

fn build_neighborhood_graph(variables: &[EdgeVariable]) -> Vec<Vec<usize>> {
    let mut buckets: HashMap<(NodeId, Arc<str>, bool), Vec<usize>> = HashMap::new();
    for (idx, variable) in variables.iter().enumerate() {
        buckets
            .entry((variable.src, variable.edge_type.clone(), true))
            .or_default()
            .push(idx);
        buckets
            .entry((variable.dst, variable.edge_type.clone(), false))
            .or_default()
            .push(idx);
    }

    let mut neighbors = vec![Vec::new(); variables.len()];
    for mut members in buckets.into_values() {
        members.sort_unstable();
        members.dedup();
        for (offset, &lhs) in members.iter().enumerate() {
            for &rhs in &members[offset + 1..] {
                neighbors[lhs].push(rhs);
                neighbors[rhs].push(lhs);
            }
        }
    }

    for adjacent in &mut neighbors {
        adjacent.sort_unstable();
        adjacent.dedup();
    }
    neighbors
}

fn build_reverse_neighbor_index(neighbors: &[Vec<usize>]) -> Result<Vec<Vec<usize>>, ExecError> {
    let mut reverse: Vec<Vec<usize>> = neighbors
        .iter()
        .map(|adjacent| Vec::with_capacity(adjacent.len()))
        .collect();

    for (idx, adjacent) in neighbors.iter().enumerate() {
        for &other in adjacent {
            let position = neighbors[other].binary_search(&idx).map_err(|_| {
                ExecError::Internal(
                    "infer_beliefs: inconsistent neighborhood graph while indexing reverse edges"
                        .into(),
                )
            })?;
            reverse[idx].push(position);
        }
    }

    Ok(reverse)
}

fn build_unary_log_potentials(variables: &[EdgeVariable]) -> Vec<[f64; 2]> {
    variables
        .iter()
        .map(|variable| {
            let alpha = variable.prior_alpha.max(MIN_BETA_PARAM);
            let beta = variable.prior_beta.max(MIN_BETA_PARAM);
            let strength = alpha + beta;
            let prior_present = (alpha / strength).clamp(MIN_PROBABILITY, 1.0 - MIN_PROBABILITY);
            [(1.0 - prior_present).ln(), prior_present.ln()]
        })
        .collect()
}

fn compute_marginals(
    unary: &[[f64; 2]],
    neighbors: &[Vec<usize>],
    reverse: &[Vec<usize>],
    messages: &[Vec<[f64; 2]>],
) -> Vec<f64> {
    let mut marginals = Vec::with_capacity(unary.len());

    for (idx, adjacent) in neighbors.iter().enumerate() {
        let mut log_absent = unary[idx][0];
        let mut log_present = unary[idx][1];

        for (slot, &other_idx) in adjacent.iter().enumerate() {
            let reverse_slot = reverse[idx][slot];
            let incoming = messages[other_idx][reverse_slot];
            log_absent += incoming[0].max(MIN_PROBABILITY).ln();
            log_present += incoming[1].max(MIN_PROBABILITY).ln();
        }

        let norm = log_sum_exp(log_absent, log_present);
        let p_present = (log_present - norm)
            .exp()
            .clamp(MIN_PROBABILITY, 1.0 - MIN_PROBABILITY);
        marginals.push(p_present);
    }

    marginals
}

fn write_posterior_mean_preserving_strength(
    graph: &mut BeliefGraph,
    edge_id: EdgeId,
    prior_alpha: f64,
    prior_beta: f64,
    p_present: f64,
) -> Result<(), ExecError> {
    let p = p_present.clamp(MIN_PROBABILITY, 1.0 - MIN_PROBABILITY);
    let strength = (prior_alpha.max(MIN_BETA_PARAM) + prior_beta.max(MIN_BETA_PARAM))
        .max(2.0 * MIN_BETA_PARAM);

    let mut alpha = (p * strength).max(MIN_BETA_PARAM);
    let mut beta = ((1.0 - p) * strength).max(MIN_BETA_PARAM);
    let renormalize = strength / (alpha + beta);
    alpha *= renormalize;
    beta *= renormalize;

    let edge = graph.edge_mut(edge_id).ok_or_else(|| {
        ExecError::Internal(format!(
            "infer_beliefs: missing edge {:?} while writing posterior",
            edge_id
        ))
    })?;
    match &mut edge.exist {
        EdgePosterior::Independent(posterior) => {
            *posterior = crate::engine::graph::BetaPosterior { alpha, beta };
            Ok(())
        }
        EdgePosterior::Competing { .. } => Err(ExecError::Internal(format!(
            "infer_beliefs: edge {:?} changed posterior type during execution",
            edge_id
        ))),
    }
}

#[inline]
fn log_sum_exp(a: f64, b: f64) -> f64 {
    let m = a.max(b);
    if !m.is_finite() {
        return m;
    }
    m + ((a - m).exp() + (b - m).exp()).ln()
}

#[inline]
fn damped_normalize(previous: [f64; 2], fresh: [f64; 2], damping: f64) -> [f64; 2] {
    let mut absent = damping * previous[0] + (1.0 - damping) * fresh[0];
    let mut present = damping * previous[1] + (1.0 - damping) * fresh[1];
    let norm = (absent + present).max(f64::MIN_POSITIVE);
    absent /= norm;
    present /= norm;
    [absent, present]
}

#[cfg(test)]
mod tests {
    use std::collections::HashMap;

    use super::*;
    use crate::engine::graph::{BetaPosterior, EdgeData, GaussianPosterior, NodeData};

    fn simple_graph() -> BeliefGraph {
        let mut graph = BeliefGraph::default();
        for id in 1..=3_u32 {
            graph.insert_node(NodeData {
                id: NodeId(id),
                label: Arc::from("N"),
                attrs: HashMap::from([(
                    "x".into(),
                    GaussianPosterior {
                        mean: id as f64,
                        precision: 1.0,
                    },
                )]),
            });
        }
        graph.ensure_owned();
        graph
    }

    #[test]
    fn infer_beliefs_noop_on_empty_graph() {
        let graph = BeliefGraph::default();
        let output = run_loopy_belief_propagation(&graph).expect("inference");
        assert_eq!(output.edges().len(), 0);
    }

    #[test]
    fn infer_beliefs_keeps_isolated_edge_probability() {
        let mut graph = simple_graph();
        graph.insert_edge(EdgeData {
            id: EdgeId(1),
            src: NodeId(1),
            dst: NodeId(2),
            ty: Arc::from("REL"),
            exist: EdgePosterior::independent(BetaPosterior {
                alpha: 8.0,
                beta: 2.0,
            }),
        });
        graph.insert_edge(EdgeData {
            id: EdgeId(2),
            src: NodeId(1),
            dst: NodeId(3),
            ty: Arc::from("OTHER"),
            exist: EdgePosterior::independent(BetaPosterior {
                alpha: 3.0,
                beta: 7.0,
            }),
        });
        graph.ensure_owned();

        let before_1 = graph.prob_mean(EdgeId(1)).expect("edge 1 prob");
        let before_2 = graph.prob_mean(EdgeId(2)).expect("edge 2 prob");

        let output = run_loopy_belief_propagation(&graph).expect("inference");
        let after_1 = output.prob_mean(EdgeId(1)).expect("edge 1 prob");
        let after_2 = output.prob_mean(EdgeId(2)).expect("edge 2 prob");

        assert!((after_1 - before_1).abs() < 1e-9);
        assert!((after_2 - before_2).abs() < 1e-9);
    }

    #[test]
    fn infer_beliefs_smooths_connected_edges() {
        let mut graph = simple_graph();
        graph.insert_edge(EdgeData {
            id: EdgeId(1),
            src: NodeId(1),
            dst: NodeId(2),
            ty: Arc::from("REL"),
            exist: EdgePosterior::independent(BetaPosterior {
                alpha: 9.0,
                beta: 1.0,
            }),
        });
        graph.insert_edge(EdgeData {
            id: EdgeId(2),
            src: NodeId(1),
            dst: NodeId(3),
            ty: Arc::from("REL"),
            exist: EdgePosterior::independent(BetaPosterior {
                alpha: 1.0,
                beta: 9.0,
            }),
        });
        graph.ensure_owned();

        let before_high = graph.prob_mean(EdgeId(1)).expect("edge 1");
        let before_low = graph.prob_mean(EdgeId(2)).expect("edge 2");
        let output = run_loopy_belief_propagation(&graph).expect("inference");
        let after_high = output.prob_mean(EdgeId(1)).expect("edge 1");
        let after_low = output.prob_mean(EdgeId(2)).expect("edge 2");

        assert!(after_high < before_high);
        assert!(after_low > before_low);
        assert!(after_high > 0.5);
        assert!(after_low < 0.5);
    }

    #[test]
    fn infer_beliefs_is_deterministic() {
        let mut graph = simple_graph();
        graph.insert_edge(EdgeData {
            id: EdgeId(1),
            src: NodeId(1),
            dst: NodeId(2),
            ty: Arc::from("REL"),
            exist: EdgePosterior::independent(BetaPosterior {
                alpha: 7.0,
                beta: 3.0,
            }),
        });
        graph.insert_edge(EdgeData {
            id: EdgeId(2),
            src: NodeId(2),
            dst: NodeId(3),
            ty: Arc::from("REL"),
            exist: EdgePosterior::independent(BetaPosterior {
                alpha: 2.0,
                beta: 8.0,
            }),
        });
        graph.ensure_owned();

        let first = run_loopy_belief_propagation(&graph).expect("first");
        let second = run_loopy_belief_propagation(&graph).expect("second");

        let first_p1 = first.prob_mean(EdgeId(1)).expect("first edge 1");
        let first_p2 = first.prob_mean(EdgeId(2)).expect("first edge 2");
        let second_p1 = second.prob_mean(EdgeId(1)).expect("second edge 1");
        let second_p2 = second.prob_mean(EdgeId(2)).expect("second edge 2");

        assert!((first_p1 - second_p1).abs() < 1e-12);
        assert!((first_p2 - second_p2).abs() < 1e-12);
    }

    #[test]
    fn infer_beliefs_reports_convergence_diagnostics() {
        let mut graph = simple_graph();
        graph.insert_edge(EdgeData {
            id: EdgeId(1),
            src: NodeId(1),
            dst: NodeId(2),
            ty: Arc::from("REL"),
            exist: EdgePosterior::independent(BetaPosterior {
                alpha: 9.0,
                beta: 1.0,
            }),
        });
        graph.insert_edge(EdgeData {
            id: EdgeId(2),
            src: NodeId(1),
            dst: NodeId(3),
            ty: Arc::from("REL"),
            exist: EdgePosterior::independent(BetaPosterior {
                alpha: 1.0,
                beta: 9.0,
            }),
        });
        graph.ensure_owned();

        let (_out, diagnostics) =
            run_loopy_belief_propagation_with_diagnostics(&graph).expect("inference");
        assert_eq!(diagnostics.variable_count, 2);
        assert_eq!(diagnostics.connected_variable_count, 2);
        assert!(diagnostics.iterations_run > 0);
        assert!(diagnostics.iterations_run <= diagnostics.max_iterations);
        assert!(diagnostics.final_max_message_delta.is_finite());
    }
}
