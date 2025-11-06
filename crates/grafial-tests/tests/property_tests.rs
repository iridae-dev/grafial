//! Property tests for posterior invariants and metric determinism (Phase 5)

use grafial_core::engine::graph::{
    BeliefGraph, BetaPosterior, EdgeData, EdgeId, EdgePosterior, GaussianPosterior, NodeData,
    NodeId,
};
use proptest::prelude::*;
use std::collections::HashMap;

proptest! {
    #[test]
    fn beta_posterior_mean_within_unit_interval(alpha in 0f64..1e6, beta in 0f64..1e6) {
        let mut g = BeliefGraph::default();
        g.insert_node(NodeData { id: NodeId(1), label: "N".into(), attrs: HashMap::new() });
        g.insert_node(NodeData { id: NodeId(2), label: "N".into(), attrs: HashMap::new() });
        g.insert_edge(EdgeData { id: EdgeId(1), src: NodeId(1), dst: NodeId(2), ty: "E".into(), exist: EdgePosterior::Independent(BetaPosterior { alpha, beta }) });
        let p = g.prob_mean(EdgeId(1)).unwrap();
        prop_assert!(p >= 0.0 && p <= 1.0);
    }

    #[test]
    fn gaussian_precision_monotone_on_update(mu in -1e3f64..1e3, tau0 in 1e-6f64..1e3, x in -1e3f64..1e3, tau_obs in 1e-12f64..1e3) {
        let mut gp = GaussianPosterior { mean: mu, precision: tau0 };
        let before = gp.precision;
        gp.update(x, tau_obs);
        let after = gp.precision;
        prop_assert!(after >= before, "precision should not decrease");
    }
}
