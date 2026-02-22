//! Model-selection helpers for comparing candidate graph structures.
//!
//! The current implementation focuses on edge-structure criteria:
//! - `edge_aic`
//! - `edge_bic`
//!
//! Scores are computed from posterior concentration parameters:
//! - Independent edges: Beta(α, β) terms
//! - Competing groups: Dirichlet(α_1..α_K) terms
//!
//! This gives a deterministic, lightweight criterion suitable for choosing
//! among alternative graph structures produced in flow pipelines.

use std::collections::HashSet;

use crate::engine::errors::ExecError;
use crate::engine::graph::{BeliefGraph, EdgePosterior};

const MIN_WEIGHT: f64 = 1e-9;
const MIN_PROBABILITY: f64 = 1e-12;
const SCORE_EPSILON: f64 = 1e-12;

/// Edge-model selection criterion.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum EdgeModelCriterion {
    /// Akaike Information Criterion: `2k - 2 ln L`
    Aic,
    /// Bayesian Information Criterion: `ln(n)k - 2 ln L`
    Bic,
}

/// Score details for a candidate graph under an edge-model criterion.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct EdgeModelScore {
    /// Maximized log-likelihood approximation over edge posteriors.
    pub log_likelihood: f64,
    /// Effective number of model parameters.
    pub num_parameters: f64,
    /// Effective sample size from posterior concentration mass.
    pub effective_sample_size: f64,
    /// Final criterion score (lower is better).
    pub score: f64,
}

/// Selection result for candidate graphs.
#[derive(Debug, Clone, PartialEq)]
pub struct SelectedGraphModel {
    /// Candidate graph variable name.
    pub name: String,
    /// Score details for the selected candidate.
    pub details: EdgeModelScore,
}

/// Compute an edge-model score (`AIC`/`BIC`) for a graph.
pub fn score_graph_edges(
    graph: &BeliefGraph,
    criterion: EdgeModelCriterion,
) -> Result<EdgeModelScore, ExecError> {
    let mut owned = graph.clone();
    owned.ensure_owned();

    let mut log_likelihood = 0.0;
    let mut num_parameters = 0.0;
    let mut effective_sample_size = 0.0;
    let mut seen_groups = HashSet::new();

    for edge in owned.edges() {
        match &edge.exist {
            EdgePosterior::Independent(beta) => {
                let alpha = beta.alpha.max(MIN_WEIGHT);
                let beta_param = beta.beta.max(MIN_WEIGHT);
                let total = alpha + beta_param;
                let p = clamp_probability(alpha / total);
                let q = clamp_probability(1.0 - p);

                log_likelihood += alpha * p.ln() + beta_param * q.ln();
                num_parameters += 1.0;
                effective_sample_size += total;
            }
            EdgePosterior::Competing { group_id, .. } => {
                if !seen_groups.insert(*group_id) {
                    continue;
                }
                let group = owned.competing_groups().get(group_id).ok_or_else(|| {
                    ExecError::Internal(format!(
                        "missing competing group {:?} while scoring model",
                        group_id
                    ))
                })?;

                let concentrations: Vec<f64> = group
                    .posterior
                    .concentrations
                    .iter()
                    .map(|v| v.max(MIN_WEIGHT))
                    .collect();
                let total: f64 = concentrations.iter().sum();
                for alpha in concentrations {
                    let p = clamp_probability(alpha / total);
                    log_likelihood += alpha * p.ln();
                }

                num_parameters += (group.posterior.num_categories().saturating_sub(1)) as f64;
                effective_sample_size += total;
            }
        }
    }

    let score = match criterion {
        EdgeModelCriterion::Aic => 2.0 * num_parameters - 2.0 * log_likelihood,
        EdgeModelCriterion::Bic => {
            let n = effective_sample_size.max(1.0);
            n.ln() * num_parameters - 2.0 * log_likelihood
        }
    };

    Ok(EdgeModelScore {
        log_likelihood,
        num_parameters,
        effective_sample_size,
        score,
    })
}

/// Select the best candidate graph name under the specified criterion.
///
/// Lower score is better. Ties are broken deterministically by candidate name.
pub fn select_best_graph<'a>(
    candidates: impl IntoIterator<Item = (&'a str, &'a BeliefGraph)>,
    criterion: EdgeModelCriterion,
) -> Result<SelectedGraphModel, ExecError> {
    let mut scored = Vec::new();
    for (name, graph) in candidates {
        let details = score_graph_edges(graph, criterion)?;
        scored.push((name.to_string(), details));
    }

    if scored.is_empty() {
        return Err(ExecError::ValidationError(
            "select_model: candidate list must not be empty".into(),
        ));
    }

    if scored
        .iter()
        .any(|(_, details)| details.effective_sample_size <= MIN_WEIGHT)
    {
        return Err(ExecError::ValidationError(
            "select_model: all candidates must have positive effective sample size".into(),
        ));
    }

    let reference_n = scored[0].1.effective_sample_size;
    let tolerance = reference_n.abs().max(1.0) * 1e-9;
    for (name, details) in &scored[1..] {
        if (details.effective_sample_size - reference_n).abs() > tolerance {
            return Err(ExecError::ValidationError(format!(
                "select_model: candidates must share effective sample size; '{}' has n={:.6} vs reference n={:.6}",
                name, details.effective_sample_size, reference_n
            )));
        }
    }

    let mut best: Option<SelectedGraphModel> = None;
    for (name, details) in scored {
        match &best {
            None => best = Some(SelectedGraphModel { name, details }),
            Some(current) => {
                let better_score = details.score < (current.details.score - SCORE_EPSILON);
                let tied_score = (details.score - current.details.score).abs() <= SCORE_EPSILON;
                if better_score || (tied_score && name < current.name) {
                    best = Some(SelectedGraphModel { name, details });
                }
            }
        }
    }

    best.ok_or_else(|| {
        ExecError::ValidationError("select_model: candidate list must not be empty".into())
    })
}

fn clamp_probability(v: f64) -> f64 {
    v.clamp(MIN_PROBABILITY, 1.0 - MIN_PROBABILITY)
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::engine::graph::{BeliefGraph, BetaPosterior, EdgeId, NodeData, NodeId};
    use std::collections::HashMap;

    fn graph_with_single_edge(alpha: f64, beta: f64) -> BeliefGraph {
        let mut graph = BeliefGraph::default();
        graph.insert_node(NodeData {
            id: NodeId(1),
            label: "N".into(),
            attrs: HashMap::new(),
        });
        graph.insert_node(NodeData {
            id: NodeId(2),
            label: "N".into(),
            attrs: HashMap::new(),
        });
        graph.insert_edge(BeliefGraph::test_edge_with_beta(
            EdgeId(1),
            NodeId(1),
            NodeId(2),
            "REL".into(),
            BetaPosterior { alpha, beta },
        ));
        graph
    }

    #[test]
    fn score_graph_edges_handles_empty_graph() {
        let graph = BeliefGraph::default();
        let score = score_graph_edges(&graph, EdgeModelCriterion::Bic).expect("score");
        assert!(score.log_likelihood.is_finite());
        assert_eq!(score.num_parameters, 0.0);
        assert_eq!(score.effective_sample_size, 0.0);
        assert_eq!(score.score, 0.0);
    }

    #[test]
    fn select_best_graph_prefers_lower_score() {
        let strong_signal = graph_with_single_edge(30.0, 2.0);
        let weak_signal = graph_with_single_edge(16.0, 16.0);

        let selected = select_best_graph(
            [("strong", &strong_signal), ("weak", &weak_signal)],
            EdgeModelCriterion::Bic,
        )
        .expect("select");

        assert_eq!(selected.name, "strong");
        assert!(selected.details.score.is_finite());
    }

    #[test]
    fn select_best_graph_breaks_ties_by_name() {
        let g1 = graph_with_single_edge(10.0, 5.0);
        let g2 = graph_with_single_edge(10.0, 5.0);

        let selected = select_best_graph(
            [("b_graph", &g1), ("a_graph", &g2)],
            EdgeModelCriterion::Aic,
        )
        .expect("select");

        assert_eq!(selected.name, "a_graph");
    }

    #[test]
    fn select_best_graph_rejects_mismatched_effective_sample_size() {
        let g1 = graph_with_single_edge(10.0, 5.0);
        let g2 = graph_with_single_edge(20.0, 5.0);

        let err = select_best_graph([("g1", &g1), ("g2", &g2)], EdgeModelCriterion::Bic)
            .expect_err("must reject mismatched n");

        assert!(
            err.to_string().contains("share effective sample size"),
            "unexpected error: {}",
            err
        );
    }
}
