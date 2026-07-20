//! Probe harness that records wall-time / convergence baselines for CI gates.
//!
//! Run: `cargo run -p grafial-benches --bin baseline_probe --release`
//! Output: JSON array of probe results to stdout.

use grafial_core::engine::belief_propagation::{
    run_loopy_belief_propagation_with_config_diagnostics, BeliefPropagationConfig,
};
use grafial_core::engine::graph::{BeliefGraph, BetaPosterior};
use grafial_core::engine::model_selection::{score_graph_edges, EdgeModelCriterion};
use std::collections::HashMap;
use std::time::Instant;

struct Probe {
    name: &'static str,
    n_nodes: usize,
    n_edges: usize,
    density: f64,
    transform: &'static str,
    backend: &'static str,
    wall_ms: f64,
    bp_iterations: Option<usize>,
    bp_converged: Option<bool>,
    score: Option<f64>,
}

fn fanout_graph(n_leaves: usize, alpha: f64, beta: f64) -> BeliefGraph {
    let mut g = BeliefGraph::default();
    let src = g.add_node("N".into(), HashMap::new());
    for _ in 0..n_leaves {
        let dst = g.add_node("N".into(), HashMap::new());
        g.add_edge(src, dst, "E".into(), BetaPosterior { alpha, beta });
    }
    g.ensure_owned();
    g
}

fn main() {
    let mut probes = Vec::new();

    for &(name, n_leaves, alpha, beta, max_iter) in &[
        ("bp_fanout_n8", 8usize, 4.0, 2.0, 64usize),
        ("bp_fanout_n64", 64, 3.0, 3.0, 32),
    ] {
        let g = fanout_graph(n_leaves, alpha, beta);
        let cfg = BeliefPropagationConfig {
            max_iterations: max_iter,
            damping: 0.35,
            convergence_tolerance: 1e-6,
            coupling_strength: 0.6,
        };
        let start = Instant::now();
        let (_out, diag) =
            run_loopy_belief_propagation_with_config_diagnostics(&g, cfg).expect("bp");
        let wall_ms = start.elapsed().as_secs_f64() * 1000.0;
        let n_nodes = n_leaves + 1;
        probes.push(Probe {
            name,
            n_nodes,
            n_edges: n_leaves,
            density: n_leaves as f64 / (n_nodes * n_leaves.max(1)) as f64,
            transform: "infer_beliefs",
            backend: "loopy_sum_product",
            wall_ms,
            bp_iterations: Some(diag.iterations_run),
            bp_converged: Some(diag.converged),
            score: None,
        });
    }

    for &(name, criterion) in &[
        ("edge_aic_n32", EdgeModelCriterion::Aic),
        ("edge_bic_n32", EdgeModelCriterion::Bic),
    ] {
        let g = fanout_graph(32, 5.0, 1.0);
        let start = Instant::now();
        let details = score_graph_edges(&g, criterion).expect("score");
        let wall_ms = start.elapsed().as_secs_f64() * 1000.0;
        probes.push(Probe {
            name,
            n_nodes: 33,
            n_edges: 32,
            density: 32.0 / (33.0 * 32.0),
            transform: match criterion {
                EdgeModelCriterion::Aic => "select_model/edge_aic",
                EdgeModelCriterion::Bic => "select_model/edge_bic",
            },
            backend: "model_selection",
            wall_ms,
            bp_iterations: None,
            bp_converged: None,
            score: Some(details.score),
        });
    }

    println!("[");
    for (i, p) in probes.iter().enumerate() {
        if i > 0 {
            println!(",");
        }
        print!(
            r#"  {{"name":"{}","n_nodes":{},"n_edges":{},"density":{:.6},"transform":"{}","backend":"{}","wall_ms":{:.3}"#,
            p.name, p.n_nodes, p.n_edges, p.density, p.transform, p.backend, p.wall_ms
        );
        if let Some(it) = p.bp_iterations {
            print!(r#","bp_iterations":{it}"#);
        }
        if let Some(c) = p.bp_converged {
            print!(r#","bp_converged":{c}"#);
        }
        if let Some(s) = p.score {
            print!(r#","score":{s:.6}"#);
        }
        print!("}}");
    }
    println!();
    println!("]");
}
