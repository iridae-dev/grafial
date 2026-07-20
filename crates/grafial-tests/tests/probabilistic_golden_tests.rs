//! Independent numerical golden tests for probabilistic semantics.
//!
//! Expected values are derived analytically or from a *separate* reference
//! recurrence in this file — not by comparing two Grafial execution paths.

use grafial_core::engine::belief_propagation::{
    run_loopy_belief_propagation_with_config_diagnostics, BeliefPropagationConfig,
};
use grafial_core::engine::graph::{
    BeliefGraph, BetaPosterior, DirichletPosterior, EdgePosterior, GaussianPosterior,
};
use grafial_core::engine::model_selection::{score_graph_edges, EdgeModelCriterion};
use grafial_core::parse_and_validate;
use grafial_core::run_flow;
use std::collections::HashMap;

const MIN_P: f64 = 1e-6;

fn assert_close(actual: f64, expected: f64, tol: f64, label: &str) {
    assert!(
        (actual - expected).abs() <= tol,
        "{label}: expected {expected:.12}, got {actual:.12}, diff={:.3e}",
        (actual - expected).abs()
    );
}

fn edge_mean(graph: &BeliefGraph, edge_id: grafial_core::engine::graph::EdgeId) -> f64 {
    match &graph.edge(edge_id).expect("edge").exist {
        EdgePosterior::Independent(beta) => beta.mean_probability(),
        other => panic!("expected independent edge posterior, got {other:?}"),
    }
}

fn edge_strength(graph: &BeliefGraph, edge_id: grafial_core::engine::graph::EdgeId) -> f64 {
    match &graph.edge(edge_id).expect("edge").exist {
        EdgePosterior::Independent(beta) => beta.alpha + beta.beta,
        other => panic!("expected independent edge posterior, got {other:?}"),
    }
}

fn log_sum_exp(a: f64, b: f64) -> f64 {
    let m = a.max(b);
    m + ((a - m).exp() + (b - m).exp()).ln()
}

#[derive(Clone, Copy)]
struct SiblingBpParams {
    alpha0: f64,
    beta0: f64,
    alpha1: f64,
    beta1: f64,
    damping: f64,
    kappa: f64,
    max_iterations: usize,
    tol: f64,
}

/// Independent reference for 2-variable fan-out loopy BP (same equations as the engine).
///
/// Returns converged present-probabilities for (edge0, edge1).
fn reference_two_sibling_bp(p: SiblingBpParams) -> (f64, f64, usize, bool) {
    let unary = |alpha: f64, beta: f64| {
        let mean = (alpha / (alpha + beta)).clamp(MIN_P, 1.0 - MIN_P);
        [(1.0 - mean).ln(), mean.ln()]
    };
    let u0 = unary(p.alpha0, p.beta0);
    let u1 = unary(p.alpha1, p.beta1);
    let log_same = p.kappa;
    let log_diff = -p.kappa;

    // messages[src][slot] where each has one neighbor → slot 0 only
    let mut m01 = [0.5, 0.5]; // 0 → 1
    let mut m10 = [0.5, 0.5]; // 1 → 0
    let mut iters = 0;
    let mut converged = false;
    for i in 0..p.max_iterations {
        iters = i + 1;
        // message 0→1: no other neighbors besides 1
        let log_prod0_abs = u0[0];
        let log_prod0_pre = u0[1];
        let fresh01 = {
            let la = log_sum_exp(log_same + log_prod0_abs, log_diff + log_prod0_pre);
            let lp = log_sum_exp(log_diff + log_prod0_abs, log_same + log_prod0_pre);
            let n = log_sum_exp(la, lp);
            [
                (la - n).exp().clamp(MIN_P, 1.0 - MIN_P),
                (lp - n).exp().clamp(MIN_P, 1.0 - MIN_P),
            ]
        };
        // message 1→0
        let log_prod1_abs = u1[0];
        let log_prod1_pre = u1[1];
        let fresh10 = {
            let la = log_sum_exp(log_same + log_prod1_abs, log_diff + log_prod1_pre);
            let lp = log_sum_exp(log_diff + log_prod1_abs, log_same + log_prod1_pre);
            let n = log_sum_exp(la, lp);
            [
                (la - n).exp().clamp(MIN_P, 1.0 - MIN_P),
                (lp - n).exp().clamp(MIN_P, 1.0 - MIN_P),
            ]
        };

        let damp = |cur: [f64; 2], fresh: [f64; 2]| {
            let mixed = [
                (1.0 - p.damping) * fresh[0] + p.damping * cur[0],
                (1.0 - p.damping) * fresh[1] + p.damping * cur[1],
            ];
            let s = mixed[0] + mixed[1];
            [mixed[0] / s, mixed[1] / s]
        };
        let next01 = damp(m01, fresh01);
        let next10 = damp(m10, fresh10);
        let delta = (next01[0] - m01[0])
            .abs()
            .max((next01[1] - m01[1]).abs())
            .max((next10[0] - m10[0]).abs())
            .max((next10[1] - m10[1]).abs());
        m01 = next01;
        m10 = next10;
        if delta < p.tol {
            converged = true;
            break;
        }
    }

    let marginal = |unary: [f64; 2], incoming: [f64; 2]| {
        let la = unary[0] + incoming[0].max(MIN_P).ln();
        let lp = unary[1] + incoming[1].max(MIN_P).ln();
        let n = log_sum_exp(la, lp);
        (lp - n).exp().clamp(MIN_P, 1.0 - MIN_P)
    };
    (marginal(u0, m10), marginal(u1, m01), iters, converged)
}

#[test]
fn soft_beta_update_matches_weighted_pseudo_count() {
    let mut posterior = BetaPosterior {
        alpha: 2.0,
        beta: 5.0,
    };
    posterior.observe_weighted(true, 0.25);
    assert_close(posterior.alpha, 2.25, 1e-12, "soft present alpha");
    assert_close(posterior.beta, 5.0, 1e-12, "soft present beta");

    posterior.observe_weighted(false, 0.5);
    assert_close(posterior.alpha, 2.25, 1e-12, "soft absent alpha");
    assert_close(posterior.beta, 5.5, 1e-12, "soft absent beta");
}

#[test]
fn categorical_competition_means_match_dirichlet_counts() {
    let mut posterior = DirichletPosterior::new(vec![1.0, 1.0, 1.0]);
    posterior.observe_chosen(1);
    posterior.observe_chosen(1);
    posterior.observe_chosen(2);
    let means = posterior.mean_probabilities();
    assert_close(means[0], 1.0 / 6.0, 1e-12, "cat0");
    assert_close(means[1], 3.0 / 6.0, 1e-12, "cat1");
    assert_close(means[2], 2.0 / 6.0, 1e-12, "cat2");
}

#[test]
fn categorical_unchosen_k2_is_other_chosen_and_k3_errors() {
    let mut binary = DirichletPosterior::new(vec![2.0, 5.0]);
    binary.observe_unchosen(0).unwrap();
    assert_eq!(binary.concentrations, vec![2.0, 6.0]);

    let mut ternary = DirichletPosterior::new(vec![1.0, 1.0, 1.0]);
    let err = ternary.observe_unchosen(1).unwrap_err();
    assert!(
        err.to_string().contains("only conjugate for binary"),
        "unexpected error: {err}"
    );
}

#[test]
fn loopy_bp_two_siblings_match_independent_reference_posteriors() {
    let alpha0 = 9.0;
    let beta0 = 1.0;
    let alpha1 = 1.0;
    let beta1 = 9.0;
    let damping = 0.2;
    let kappa = 1.0;
    let max_iterations = 64;
    let tol = 1e-8;

    let (ref0, ref1, ref_iters, ref_conv) = reference_two_sibling_bp(SiblingBpParams {
        alpha0,
        beta0,
        alpha1,
        beta1,
        damping,
        kappa,
        max_iterations,
        tol,
    });
    assert!(ref_conv, "reference failed to converge");

    let mut g = BeliefGraph::default();
    let a = g.add_node("N".into(), HashMap::new());
    let b = g.add_node("N".into(), HashMap::new());
    let c = g.add_node("N".into(), HashMap::new());
    let e0 = g.add_edge(
        a,
        b,
        "E".into(),
        BetaPosterior {
            alpha: alpha0,
            beta: beta0,
        },
    );
    let e1 = g.add_edge(
        a,
        c,
        "E".into(),
        BetaPosterior {
            alpha: alpha1,
            beta: beta1,
        },
    );
    g.ensure_owned();

    let cfg = BeliefPropagationConfig {
        max_iterations,
        damping,
        convergence_tolerance: tol,
        coupling_strength: kappa,
    };
    let (out, diag) = run_loopy_belief_propagation_with_config_diagnostics(&g, cfg).expect("bp");
    assert!(diag.converged);
    assert_eq!(diag.iterations_run, ref_iters);

    // Strength preserved; means match reference marginals.
    assert_close(edge_strength(&out, e0), alpha0 + beta0, 1e-9, "strength0");
    assert_close(edge_strength(&out, e1), alpha1 + beta1, 1e-9, "strength1");
    assert_close(edge_mean(&out, e0), ref0, 1e-9, "mean0 vs reference");
    assert_close(edge_mean(&out, e1), ref1, 1e-9, "mean1 vs reference");
}

#[test]
fn edge_aic_bic_match_closed_form_for_single_beta_edge() {
    // One independent edge Beta(3,1):
    // p=0.75, q=0.25
    // ll = 3 ln p + 1 ln q
    // k=1, n=4
    // AIC = 2k - 2 ll
    // BIC = ln(n) k - 2 ll
    let mut g = BeliefGraph::default();
    let a = g.add_node("N".into(), HashMap::new());
    let b = g.add_node("N".into(), HashMap::new());
    g.add_edge(
        a,
        b,
        "E".into(),
        BetaPosterior {
            alpha: 3.0,
            beta: 1.0,
        },
    );
    g.ensure_owned();

    let p = 0.75_f64;
    let q = 0.25_f64;
    let ll = 3.0 * p.ln() + 1.0 * q.ln();
    let aic = 2.0 * 1.0 - 2.0 * ll;
    let bic = 4.0_f64.ln() * 1.0 - 2.0 * ll;

    let aic_score = score_graph_edges(&g, EdgeModelCriterion::Aic).unwrap();
    let bic_score = score_graph_edges(&g, EdgeModelCriterion::Bic).unwrap();
    assert_close(aic_score.log_likelihood, ll, 1e-12, "ll");
    assert_close(aic_score.num_parameters, 1.0, 1e-12, "k");
    assert_close(aic_score.effective_sample_size, 4.0, 1e-12, "n");
    assert_close(aic_score.score, aic, 1e-12, "aic");
    assert_close(bic_score.score, bic, 1e-12, "bic");
}

#[test]
fn gaussian_attr_correlation_covariance_matches_closed_form() {
    let mut g = BeliefGraph::default();
    let mut attrs = HashMap::new();
    attrs.insert(
        "x".into(),
        GaussianPosterior {
            mean: 0.0,
            precision: 4.0, // var = 0.25
        },
    );
    attrs.insert(
        "y".into(),
        GaussianPosterior {
            mean: 1.0,
            precision: 1.0, // var = 1.0
        },
    );
    let n = g.add_node("N".into(), attrs);
    g.set_attr_correlation(n, "x", "y", 0.5).unwrap();

    // Cov = rho * sigma_x * sigma_y = 0.5 * 0.5 * 1.0 = 0.25
    let cov = g.attr_covariance(n, "x", "y").unwrap();
    assert_close(cov, 0.25, 1e-12, "cov(x,y)");
    assert_eq!(g.attr_correlation(n, "x", "y").unwrap(), 0.5);
}

#[test]
fn end_to_end_gaussian_precision_update_from_program() {
    let src = r#"
schema S { node Entity { value: Real } edge CONNECTED {} }
belief_model M on S {
  node Entity { value ~ Gaussian(mean=0.0, precision=1.0) }
  edge CONNECTED { exist ~ Bernoulli(prior=0.5, weight=2.0) }
}
evidence Ev on M {
  Entity { "A" { value: 2.0 (precision=3.0) } }
}
flow F on M {
  graph g = from_evidence Ev
  metric m = nodes(Entity) |> sum(by=E[node.value])
  export_metric m as "m"
  export g as "g"
}
"#;
    let program = parse_and_validate(src).expect("parse");
    let result = run_flow(&program, "F", None).expect("run");
    assert_close(
        *result.metric_exports.get("m").unwrap(),
        1.5,
        1e-9,
        "gaussian mean",
    );
}

#[test]
fn metric_on_graph_binds_explicit_target() {
    let src = r#"
schema S { node Entity { value: Real } edge CONNECTED {} }
belief_model M on S {
  node Entity { value ~ Gaussian(mean=0.0, precision=1.0) }
  edge CONNECTED { exist ~ Bernoulli(prior=0.9, weight=2.0) }
}
evidence Ev on M {
  Entity { "A" { value: 1.0 (precision=1000.0) }, "B" { value: 3.0 (precision=1000.0) } }
  CONNECTED(Entity -> Entity) { "A" -> "B" }
}
flow F on M {
  graph raw = from_evidence Ev
  graph pruned = raw |> prune_edges CONNECTED where prob(edge) < 1.0
  metric deg_raw on raw = avg_degree(label=Entity, edge_type=CONNECTED, min_prob=0.0)
  metric deg_pruned on pruned = avg_degree(label=Entity, edge_type=CONNECTED, min_prob=0.0)
  export_metric deg_raw as "deg_raw"
  export_metric deg_pruned as "deg_pruned"
}
"#;
    let program = parse_and_validate(src).expect("parse");
    let flow = program.flows.iter().find(|f| f.name == "F").unwrap();
    assert_eq!(flow.metrics[0].on_graph.as_deref(), Some("raw"));
    assert_eq!(flow.metrics[1].on_graph.as_deref(), Some("pruned"));
    let result = run_flow(&program, "F", None).expect("run");
    let deg_raw = *result.metric_exports.get("deg_raw").unwrap();
    let deg_pruned = *result.metric_exports.get("deg_pruned").unwrap();
    assert!(
        deg_raw > deg_pruned + 1e-6,
        "explicit metric graph binding failed: deg_raw={deg_raw}, deg_pruned={deg_pruned}"
    );
}
