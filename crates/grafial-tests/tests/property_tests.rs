//! Property tests for Bayesian update invariants and execution determinism (Phase 6).

use grafial_core::engine::flow_exec::run_flow_with_builder;
use grafial_core::engine::graph::{
    BeliefGraph, BetaPosterior, EdgeData, EdgeId, EdgePosterior, GaussianPosterior, NodeData,
    NodeId,
};
use grafial_frontend::ast::*;
use proptest::prelude::*;
use proptest::test_runner::Config as ProptestConfig;
use std::collections::HashMap;

const DETERMINISM_ENDPOINTS: [(u32, u32); 6] = [(1, 2), (2, 3), (3, 4), (4, 1), (1, 3), (2, 4)];

fn build_determinism_program() -> ProgramAst {
    let schema = Schema {
        name: "DeterminismSchema".into(),
        nodes: vec![NodeDef {
            name: "Node".into(),
            attrs: vec![AttrDef {
                name: "score".into(),
                ty: "Real".into(),
            }],
        }],
        edges: vec![EdgeDef { name: "REL".into() }],
    };

    let belief_model = BeliefModel {
        name: "DeterminismBeliefs".into(),
        on_schema: "DeterminismSchema".into(),
        nodes: vec![],
        edges: vec![],
        body_src: "".into(),
    };

    let evidence = EvidenceDef {
        name: "DeterminismEvidence".into(),
        on_model: "DeterminismBeliefs".into(),
        observations: vec![],
        body_src: "".into(),
    };

    let rule = RuleDef {
        name: "CullUnlikelyEdges".into(),
        on_model: "DeterminismBeliefs".into(),
        patterns: vec![PatternItem {
            src: NodePattern {
                var: "A".into(),
                label: "Node".into(),
            },
            edge: EdgePattern {
                var: "e".into(),
                ty: "REL".into(),
            },
            dst: NodePattern {
                var: "B".into(),
                label: "Node".into(),
            },
        }],
        where_expr: Some(ExprAst::Binary {
            op: BinaryOp::Lt,
            left: Box::new(ExprAst::Call {
                name: "prob".into(),
                args: vec![CallArg::Positional(ExprAst::Var("e".into()))],
            }),
            right: Box::new(ExprAst::Number(0.4)),
        }),
        actions: vec![ActionStmt::DeleteEdge {
            edge_var: "e".into(),
            confidence: Some("high".into()),
        }],
        mode: Some("for_each".into()),
    };

    let flow = FlowDef {
        name: "DeterminismFlow".into(),
        on_model: "DeterminismBeliefs".into(),
        graphs: vec![
            GraphDef {
                name: "base".into(),
                expr: GraphExpr::FromEvidence {
                    evidence: "DeterminismEvidence".into(),
                },
            },
            GraphDef {
                name: "cleaned".into(),
                expr: GraphExpr::Pipeline {
                    start: "base".into(),
                    transforms: vec![
                        Transform::ApplyRule {
                            rule: "CullUnlikelyEdges".into(),
                        },
                        Transform::PruneEdges {
                            edge_type: "REL".into(),
                            predicate: ExprAst::Binary {
                                op: BinaryOp::Lt,
                                left: Box::new(ExprAst::Call {
                                    name: "prob".into(),
                                    args: vec![CallArg::Positional(ExprAst::Var("edge".into()))],
                                }),
                                right: Box::new(ExprAst::Number(0.01)),
                            },
                        },
                    ],
                },
            },
        ],
        metrics: vec![],
        exports: vec![ExportDef {
            graph: "cleaned".into(),
            alias: "stable".into(),
        }],
        metric_exports: vec![],
        metric_imports: vec![],
    };

    ProgramAst {
        schemas: vec![schema],
        belief_models: vec![belief_model],
        evidences: vec![evidence],
        rules: vec![rule],
        flows: vec![flow],
    }
}

fn build_graph_with_edge_order(edges: &[EdgeData], order: &[usize]) -> BeliefGraph {
    let mut graph = BeliefGraph::default();

    for node_id in 1u32..=4 {
        graph.insert_node(NodeData {
            id: NodeId(node_id),
            label: "Node".into(),
            attrs: HashMap::from([(
                "score".to_string(),
                GaussianPosterior {
                    mean: node_id as f64,
                    precision: 1.0,
                },
            )]),
        });
    }

    for idx in order {
        graph.insert_edge(edges[*idx].clone());
    }

    graph
}

fn graph_signature(mut graph: BeliefGraph) -> Vec<String> {
    graph.ensure_owned();

    let mut signature = Vec::new();

    for node in graph.nodes() {
        let mut attrs: Vec<_> = node.attrs.iter().collect();
        attrs.sort_by(|(left, _), (right, _)| left.cmp(right));
        for (name, posterior) in attrs {
            signature.push(format!(
                "N|{}|{}|{}|{:.12}|{:.12}",
                node.id.0, node.label, name, posterior.mean, posterior.precision
            ));
        }
    }

    for edge in graph.edges() {
        let prob = graph
            .prob_mean(edge.id)
            .expect("edge probability should exist");
        signature.push(format!(
            "E|{}|{}|{}|{}|{:.12}",
            edge.id.0, edge.src.0, edge.dst.0, edge.ty, prob
        ));
    }

    signature.sort();
    signature
}

proptest! {
    #[test]
    fn beta_posterior_mean_within_unit_interval(alpha in 0.01f64..1e6, beta in 0.01f64..1e6) {
        let mut g = BeliefGraph::default();
        g.insert_node(NodeData { id: NodeId(1), label: "N".into(), attrs: HashMap::new() });
        g.insert_node(NodeData { id: NodeId(2), label: "N".into(), attrs: HashMap::new() });
        g.insert_edge(EdgeData {
            id: EdgeId(1),
            src: NodeId(1),
            dst: NodeId(2),
            ty: "E".into(),
            exist: EdgePosterior::Independent(BetaPosterior { alpha, beta }),
        });
        let p = g.prob_mean(EdgeId(1)).unwrap();
        prop_assert!(p >= 0.0 && p <= 1.0);
    }

    #[test]
    fn beta_observe_present_never_decreases_mean(alpha in 0.01f64..1e6, beta in 0.01f64..1e6) {
        let mut posterior = BetaPosterior { alpha, beta };
        let before = posterior.alpha / (posterior.alpha + posterior.beta);
        posterior.observe(true);
        let after = posterior.alpha / (posterior.alpha + posterior.beta);
        prop_assert!(after >= before, "mean should be monotone on present observation");
    }

    #[test]
    fn beta_observe_absent_never_increases_mean(alpha in 0.01f64..1e6, beta in 0.01f64..1e6) {
        let mut posterior = BetaPosterior { alpha, beta };
        let before = posterior.alpha / (posterior.alpha + posterior.beta);
        posterior.observe(false);
        let after = posterior.alpha / (posterior.alpha + posterior.beta);
        prop_assert!(after <= before, "mean should be monotone on absent observation");
    }

    #[test]
    fn gaussian_precision_monotone_on_update(
        mu in -1e3f64..1e3,
        tau0 in 1e-6f64..1e3,
        x in -1e3f64..1e3,
        tau_obs in 1e-12f64..1e3
    ) {
        let mut gp = GaussianPosterior { mean: mu, precision: tau0 };
        let before = gp.precision;
        gp.update(x, tau_obs);
        let after = gp.precision;
        prop_assert!(after >= before, "precision should not decrease");
    }

    #[test]
    fn gaussian_update_mean_stays_between_prior_and_observation(
        mu in -1e3f64..1e3,
        tau0 in 1e-6f64..1e3,
        x in -1e3f64..1e3,
        tau_obs in 1e-12f64..1e3
    ) {
        let mut gp = GaussianPosterior { mean: mu, precision: tau0 };
        gp.update(x, tau_obs);
        let lo = mu.min(x) - 1e-12;
        let hi = mu.max(x) + 1e-12;
        prop_assert!(gp.mean >= lo && gp.mean <= hi, "posterior mean should stay in convex hull");
    }
}

proptest! {
    #![proptest_config(ProptestConfig::with_cases(48))]

    #[test]
    fn flow_execution_deterministic_across_edge_insertion_order(
        alphas in prop::collection::vec(0.1f64..20.0, DETERMINISM_ENDPOINTS.len()),
        betas in prop::collection::vec(0.1f64..20.0, DETERMINISM_ENDPOINTS.len())
    ) {
        let program = build_determinism_program();

        let edges: Vec<EdgeData> = DETERMINISM_ENDPOINTS
            .iter()
            .enumerate()
            .map(|(idx, (src, dst))| EdgeData {
                id: EdgeId((idx + 1) as u32),
                src: NodeId(*src),
                dst: NodeId(*dst),
                ty: "REL".into(),
                exist: EdgePosterior::Independent(BetaPosterior {
                    alpha: alphas[idx],
                    beta: betas[idx],
                }),
            })
            .collect();

        let forward_order: Vec<usize> = (0..edges.len()).collect();
        let reverse_order: Vec<usize> = (0..edges.len()).rev().collect();

        let forward_edges = edges.clone();
        let reverse_edges = edges.clone();

        let forward_builder = move |_evidence: &EvidenceDef| {
            Ok(build_graph_with_edge_order(&forward_edges, &forward_order))
        };
        let reverse_builder = move |_evidence: &EvidenceDef| {
            Ok(build_graph_with_edge_order(&reverse_edges, &reverse_order))
        };

        let forward_result = run_flow_with_builder(&program, "DeterminismFlow", &forward_builder, None)
            .expect("forward execution should succeed");
        let reverse_result = run_flow_with_builder(&program, "DeterminismFlow", &reverse_builder, None)
            .expect("reverse execution should succeed");

        let forward_graph = forward_result
            .exports
            .get("stable")
            .expect("forward export should exist")
            .clone();
        let reverse_graph = reverse_result
            .exports
            .get("stable")
            .expect("reverse export should exist")
            .clone();

        prop_assert_eq!(graph_signature(forward_graph), graph_signature(reverse_graph));
    }
}
