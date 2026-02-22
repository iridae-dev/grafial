//! Integration tests for competing edges functionality.
//!
//! Tests Dirichlet-Categorical posteriors, competing edge groups, and query functions.

use grafial_core::engine::errors::ExecError;
use grafial_core::engine::graph::*;
use grafial_core::engine::rule_exec::*;
use grafial_frontend::ast::*;
use rustc_hash::FxHashMap;
use std::collections::HashMap;

// ============================================================================
// DirichletPosterior Tests
// ============================================================================

#[test]
fn dirichlet_posterior_uniform_creation() {
    let dirichlet = DirichletPosterior::uniform(3, 6.0);
    assert_eq!(dirichlet.concentrations.len(), 3);
    assert_eq!(dirichlet.concentrations[0], 2.0);
    assert_eq!(dirichlet.concentrations[1], 2.0);
    assert_eq!(dirichlet.concentrations[2], 2.0);
}

#[test]
fn dirichlet_posterior_mean_probabilities() {
    let dirichlet = DirichletPosterior::new(vec![1.0, 2.0, 3.0]);
    let means = dirichlet.mean_probabilities();
    let sum: f64 = means.iter().sum();

    // Should sum to 1.0
    assert!((sum - 1.0).abs() < 1e-10);

    // Check individual means
    assert!((means[0] - 1.0 / 6.0).abs() < 1e-10);
    assert!((means[1] - 2.0 / 6.0).abs() < 1e-10);
    assert!((means[2] - 3.0 / 6.0).abs() < 1e-10);
}

#[test]
fn dirichlet_posterior_observe_chosen() {
    let mut dirichlet = DirichletPosterior::uniform(3, 3.0);

    // Initial: [1.0, 1.0, 1.0]
    dirichlet.observe_chosen(0);
    // After: [2.0, 1.0, 1.0]

    assert_eq!(dirichlet.concentrations[0], 2.0);
    assert_eq!(dirichlet.concentrations[1], 1.0);
    assert_eq!(dirichlet.concentrations[2], 1.0);

    let means = dirichlet.mean_probabilities();
    assert!((means[0] - 2.0 / 4.0).abs() < 1e-10);
}

#[test]
fn dirichlet_posterior_observe_unchosen() {
    let mut dirichlet = DirichletPosterior::new(vec![1.0, 1.0, 1.0]);

    // For K>2 this update is non-conjugate and should be rejected.
    let result = dirichlet.observe_unchosen(0);
    assert!(result.is_err());
    assert_eq!(dirichlet.concentrations, vec![1.0, 1.0, 1.0]);
}

#[test]
fn dirichlet_posterior_observe_unchosen_binary_group() {
    let mut dirichlet = DirichletPosterior::new(vec![2.0, 2.0]);

    // In a binary group, "category 0 unchosen" means category 1 was chosen.
    dirichlet.observe_unchosen(0).unwrap();
    assert_eq!(dirichlet.concentrations, vec![2.0, 3.0]);
}

#[test]
fn dirichlet_posterior_force_choice() {
    let mut dirichlet = DirichletPosterior::uniform(3, 3.0);

    // Force category 1
    dirichlet.force_choice(1);

    // Category 1 should have very high concentration (FORCE_PRECISION)
    assert!(dirichlet.concentrations[1] > 1e5);
    // Others are set to 1.0 (not preserved)
    assert_eq!(dirichlet.concentrations[0], 1.0);
    assert_eq!(dirichlet.concentrations[2], 1.0);

    let means = dirichlet.mean_probabilities();
    // Mean should be FORCE_PRECISION / (FORCE_PRECISION + 1.0 + 1.0) ≈ 1.0
    assert!(means[1] > 0.9999); // Should be very close to 1.0
}

#[test]
fn dirichlet_posterior_entropy() {
    // Uniform distribution should have maximum entropy
    let uniform = DirichletPosterior::uniform(3, 3.0);
    let h_uniform = uniform.entropy();

    // For uniform 3-category: H = log(3) ≈ 1.099
    assert!((h_uniform - 3.0_f64.ln()).abs() < 0.1);

    // Deterministic (forced choice) should have near-zero entropy
    let mut deterministic = DirichletPosterior::uniform(3, 3.0);
    deterministic.force_choice(0);
    let h_deterministic = deterministic.entropy();
    assert!(h_deterministic < 0.01);
}

#[test]
fn dirichlet_posterior_num_categories() {
    let dirichlet = DirichletPosterior::new(vec![1.0, 2.0, 3.0, 4.0]);
    assert_eq!(dirichlet.num_categories(), 4);
}

// ============================================================================
// CompetingEdgeGroup Tests
// ============================================================================

#[test]
fn competing_edge_group_creation() {
    let posterior = DirichletPosterior::uniform(2, 4.0);
    let group = CompetingEdgeGroup::new(
        CompetingGroupId(1),
        NodeId(10),
        "ROUTES_TO".into(),
        vec![NodeId(20), NodeId(30)],
        posterior,
    );

    assert_eq!(group.id, CompetingGroupId(1));
    assert_eq!(group.source, NodeId(10));
    assert_eq!(group.edge_type, "ROUTES_TO");
    assert_eq!(group.categories.len(), 2);
    assert_eq!(group.get_category_index(NodeId(20)), Some(0));
    assert_eq!(group.get_category_index(NodeId(30)), Some(1));
}

#[test]
fn competing_edge_group_mean_probability_for_dst() {
    let posterior = DirichletPosterior::new(vec![1.0, 2.0, 3.0]);
    let group = CompetingEdgeGroup::new(
        CompetingGroupId(1),
        NodeId(10),
        "ROUTES_TO".into(),
        vec![NodeId(20), NodeId(30), NodeId(40)],
        posterior,
    );

    // E[π_0] = 1/6
    assert!((group.mean_probability_for_dst(NodeId(20)).unwrap() - 1.0 / 6.0).abs() < 1e-10);
    // E[π_1] = 2/6
    assert!((group.mean_probability_for_dst(NodeId(30)).unwrap() - 2.0 / 6.0).abs() < 1e-10);
}

// ============================================================================
// BeliefGraph Competing Edges Tests
// ============================================================================

fn create_test_graph_with_competing_edges() -> BeliefGraph {
    let mut graph = BeliefGraph::default();

    // Create nodes
    graph.insert_node(NodeData {
        id: NodeId(1),
        label: "Server".into(),
        attrs: HashMap::new(),
    });
    graph.insert_node(NodeData {
        id: NodeId(2),
        label: "Server".into(),
        attrs: HashMap::new(),
    });
    graph.insert_node(NodeData {
        id: NodeId(3),
        label: "Server".into(),
        attrs: HashMap::new(),
    });

    // Create competing edge group
    let posterior = DirichletPosterior::uniform(2, 4.0);
    let group = CompetingEdgeGroup::new(
        CompetingGroupId(1),
        NodeId(1),
        "ROUTES_TO".into(),
        vec![NodeId(2), NodeId(3)],
        posterior,
    );
    graph
        .competing_groups_mut()
        .insert(CompetingGroupId(1), group);

    // Create competing edges
    graph.insert_edge(EdgeData {
        id: EdgeId(1),
        src: NodeId(1),
        dst: NodeId(2),
        ty: "ROUTES_TO".into(),
        exist: EdgePosterior::Competing {
            group_id: CompetingGroupId(1),
            category_index: 0,
        },
    });
    graph.insert_edge(EdgeData {
        id: EdgeId(2),
        src: NodeId(1),
        dst: NodeId(3),
        ty: "ROUTES_TO".into(),
        exist: EdgePosterior::Competing {
            group_id: CompetingGroupId(1),
            category_index: 1,
        },
    });

    graph
}

fn create_test_graph_with_three_competing_edges() -> BeliefGraph {
    let mut graph = BeliefGraph::default();

    graph.insert_node(NodeData {
        id: NodeId(1),
        label: "Server".into(),
        attrs: HashMap::new(),
    });
    graph.insert_node(NodeData {
        id: NodeId(2),
        label: "Server".into(),
        attrs: HashMap::new(),
    });
    graph.insert_node(NodeData {
        id: NodeId(3),
        label: "Server".into(),
        attrs: HashMap::new(),
    });
    graph.insert_node(NodeData {
        id: NodeId(4),
        label: "Server".into(),
        attrs: HashMap::new(),
    });

    let posterior = DirichletPosterior::uniform(3, 3.0);
    let group = CompetingEdgeGroup::new(
        CompetingGroupId(1),
        NodeId(1),
        "ROUTES_TO".into(),
        vec![NodeId(2), NodeId(3), NodeId(4)],
        posterior,
    );
    graph
        .competing_groups_mut()
        .insert(CompetingGroupId(1), group);

    graph.insert_edge(EdgeData {
        id: EdgeId(1),
        src: NodeId(1),
        dst: NodeId(2),
        ty: "ROUTES_TO".into(),
        exist: EdgePosterior::Competing {
            group_id: CompetingGroupId(1),
            category_index: 0,
        },
    });
    graph.insert_edge(EdgeData {
        id: EdgeId(2),
        src: NodeId(1),
        dst: NodeId(3),
        ty: "ROUTES_TO".into(),
        exist: EdgePosterior::Competing {
            group_id: CompetingGroupId(1),
            category_index: 1,
        },
    });
    graph.insert_edge(EdgeData {
        id: EdgeId(3),
        src: NodeId(1),
        dst: NodeId(4),
        ty: "ROUTES_TO".into(),
        exist: EdgePosterior::Competing {
            group_id: CompetingGroupId(1),
            category_index: 2,
        },
    });

    graph
}

#[test]
fn belief_graph_get_competing_group() {
    let graph = create_test_graph_with_competing_edges();

    let group = graph.get_competing_group(NodeId(1), "ROUTES_TO");
    assert!(group.is_some());
    assert_eq!(group.unwrap().id, CompetingGroupId(1));
}

#[test]
fn belief_graph_get_competing_group_returns_none_for_independent() {
    let mut graph = BeliefGraph::default();
    graph.insert_node(NodeData {
        id: NodeId(1),
        label: "Person".into(),
        attrs: HashMap::new(),
    });
    graph.insert_node(NodeData {
        id: NodeId(2),
        label: "Person".into(),
        attrs: HashMap::new(),
    });
    graph.insert_edge(EdgeData {
        id: EdgeId(1),
        src: NodeId(1),
        dst: NodeId(2),
        ty: "KNOWS".into(),
        exist: EdgePosterior::Independent(BetaPosterior {
            alpha: 1.0,
            beta: 1.0,
        }),
    });

    let group = graph.get_competing_group(NodeId(1), "KNOWS");
    assert!(group.is_none());
}

#[test]
fn belief_graph_winner() {
    let mut graph = create_test_graph_with_competing_edges();

    // Initially uniform, so no clear winner (within epsilon)
    let winner = graph.winner(NodeId(1), "ROUTES_TO", 0.01);
    assert!(winner.is_none()); // Tied within epsilon

    // Observe category 0 as chosen multiple times
    graph.observe_edge_chosen(EdgeId(1)).unwrap();
    graph.observe_edge_chosen(EdgeId(1)).unwrap();

    // Now category 0 should be the winner
    let winner = graph.winner(NodeId(1), "ROUTES_TO", 0.01);
    assert_eq!(winner, Some(NodeId(2))); // NodeId(2) is category 0
}

#[test]
fn belief_graph_entropy() {
    let graph = create_test_graph_with_competing_edges();

    // Uniform distribution should have high entropy
    let entropy = graph.entropy(NodeId(1), "ROUTES_TO").unwrap();
    assert!(entropy > 0.5);

    // Entropy should be less than log(2) ≈ 0.693 for 2 categories
    assert!(entropy < 1.0);
}

#[test]
fn belief_graph_entropy_errors_for_independent() {
    let mut graph = BeliefGraph::default();
    graph.insert_node(NodeData {
        id: NodeId(1),
        label: "Person".into(),
        attrs: HashMap::new(),
    });
    graph.insert_node(NodeData {
        id: NodeId(2),
        label: "Person".into(),
        attrs: HashMap::new(),
    });
    graph.insert_edge(EdgeData {
        id: EdgeId(1),
        src: NodeId(1),
        dst: NodeId(2),
        ty: "KNOWS".into(),
        exist: EdgePosterior::Independent(BetaPosterior {
            alpha: 1.0,
            beta: 1.0,
        }),
    });

    let result = graph.entropy(NodeId(1), "KNOWS");
    assert!(result.is_err());
    assert!(result.unwrap_err().to_string().contains("competing edges"));
}

#[test]
fn belief_graph_prob_vector() {
    let graph = create_test_graph_with_competing_edges();

    let probs = graph.prob_vector(NodeId(1), "ROUTES_TO").unwrap();
    assert_eq!(probs.len(), 2);
    assert!((probs[0] + probs[1] - 1.0).abs() < 1e-10);
    assert!((probs[0] - 0.5).abs() < 0.1); // Uniform
    assert!((probs[1] - 0.5).abs() < 0.1);
}

#[test]
fn belief_graph_observe_edge_chosen() {
    let mut graph = create_test_graph_with_competing_edges();

    // Initial probabilities should be uniform
    let probs_before = graph.prob_vector(NodeId(1), "ROUTES_TO").unwrap();
    assert!((probs_before[0] - 0.5).abs() < 0.1);

    // Observe edge 1 (category 0) as chosen
    graph.observe_edge_chosen(EdgeId(1)).unwrap();

    // Category 0 should now have higher probability
    let probs_after = graph.prob_vector(NodeId(1), "ROUTES_TO").unwrap();
    assert!(probs_after[0] > probs_before[0]);
    assert!(probs_after[1] < probs_before[1]);
}

#[test]
fn belief_graph_observe_edge_unchosen() {
    let mut graph = create_test_graph_with_competing_edges();

    // Binary case: edge 1 unchosen implies edge 2 was chosen.
    graph.observe_edge_unchosen(EdgeId(1)).unwrap();

    // Category 0 should have lower probability, category 1 should have higher
    let probs = graph.prob_vector(NodeId(1), "ROUTES_TO").unwrap();
    assert!(probs[0] < probs[1]);
}

#[test]
fn belief_graph_observe_edge_unchosen_rejects_nonbinary_group() {
    let mut graph = create_test_graph_with_three_competing_edges();
    let probs_before = graph.prob_vector(NodeId(1), "ROUTES_TO").unwrap();

    let result = graph.observe_edge_unchosen(EdgeId(1));
    assert!(result.is_err());

    // No posterior update should occur on rejected non-conjugate evidence.
    let probs_after = graph.prob_vector(NodeId(1), "ROUTES_TO").unwrap();
    assert_eq!(probs_before.len(), probs_after.len());
    for (before, after) in probs_before.iter().zip(probs_after.iter()) {
        assert!((before - after).abs() < 1e-12);
    }
}

#[test]
fn belief_graph_observe_edge_forced_choice() {
    let mut graph = create_test_graph_with_competing_edges();

    // Force edge 1 (category 0) to be chosen
    graph.observe_edge_forced_choice(EdgeId(1)).unwrap();

    // Category 0 should have probability very close to 1.0
    let probs = graph.prob_vector(NodeId(1), "ROUTES_TO").unwrap();
    assert!(probs[0] > 0.99);
    assert!(probs[1] < 0.01);
}

#[test]
fn belief_graph_observe_edge_chosen_errors_for_independent() {
    let mut graph = BeliefGraph::default();
    graph.insert_node(NodeData {
        id: NodeId(1),
        label: "Person".into(),
        attrs: HashMap::new(),
    });
    graph.insert_node(NodeData {
        id: NodeId(2),
        label: "Person".into(),
        attrs: HashMap::new(),
    });
    graph.insert_edge(EdgeData {
        id: EdgeId(1),
        src: NodeId(1),
        dst: NodeId(2),
        ty: "KNOWS".into(),
        exist: EdgePosterior::Independent(BetaPosterior {
            alpha: 1.0,
            beta: 1.0,
        }),
    });

    let result = graph.observe_edge_chosen(EdgeId(1));
    assert!(result.is_err());
    assert!(result.unwrap_err().to_string().contains("competing edges"));
}

#[test]
fn belief_graph_prob_mean_for_competing_edge() {
    let graph = create_test_graph_with_competing_edges();

    // Both edges should have probability 0.5 (uniform)
    let prob1 = graph.prob_mean(EdgeId(1)).unwrap();
    let prob2 = graph.prob_mean(EdgeId(2)).unwrap();

    assert!((prob1 - 0.5).abs() < 0.1);
    assert!((prob2 - 0.5).abs() < 0.1);
    assert!((prob1 + prob2 - 1.0).abs() < 0.1);
}

#[test]
fn belief_graph_degree_outgoing_for_competing_edges() {
    let graph = create_test_graph_with_competing_edges();

    // Both categories should meet min_prob=0.3
    let degree = graph.degree_outgoing(NodeId(1), 0.3);
    assert_eq!(degree, 2);

    // Only one should meet min_prob=0.6 (after observing one as chosen)
    let mut graph2 = create_test_graph_with_competing_edges();
    graph2.observe_edge_chosen(EdgeId(1)).unwrap();
    graph2.observe_edge_chosen(EdgeId(1)).unwrap();
    let degree = graph2.degree_outgoing(NodeId(1), 0.6);
    assert_eq!(degree, 1);
}

// ============================================================================
// Rule Execution Context Tests (winner, entropy)
// ============================================================================

#[test]
fn rule_context_winner_function() {
    let mut graph = create_test_graph_with_competing_edges();

    // Make category 0 (NodeId(2)) the clear winner
    graph.observe_edge_chosen(EdgeId(1)).unwrap();
    graph.observe_edge_chosen(EdgeId(1)).unwrap();

    // Test winner() via execute_actions which uses RuleExprContext internally
    // We can test it indirectly by using it in a rule where clause
    let bindings = MatchBindings {
        node_vars: HashMap::from([("A".into(), NodeId(1))]),
        edge_vars: HashMap::new(),
    };

    // Create a simple test context that mimics RuleExprContext
    use grafial_core::engine::expr_eval::{eval_expr_core, ExprContext};

    // Create a minimal context that implements ExprContext
    struct TestContext {
        bindings: MatchBindings,
    }

    impl ExprContext for TestContext {
        fn resolve_var(&self, name: &str) -> Option<f64> {
            self.bindings
                .node_vars
                .get(name)
                .map(|node_id| node_id.0 as f64)
        }

        fn eval_function(
            &self,
            name: &str,
            pos_args: &[ExprAst],
            _all_args: &[CallArg],
            graph: &BeliefGraph,
        ) -> Result<f64, ExecError> {
            match name {
                "winner" => {
                    if pos_args.len() < 2 {
                        return Err(ExecError::Internal(
                            "winner(): requires node and edge_type arguments".into(),
                        ));
                    }
                    let node_var = match &pos_args[0] {
                        ExprAst::Var(v) => v,
                        _ => {
                            return Err(ExecError::Internal(
                                "winner(): first argument must be a node variable".into(),
                            ))
                        }
                    };
                    let edge_type = match &pos_args[1] {
                        ExprAst::Var(v) => v.clone(),
                        _ => {
                            return Err(ExecError::Internal(
                                "winner(): edge_type must be an identifier".into(),
                            ))
                        }
                    };

                    let nid = *self.bindings.node_vars.get(node_var).ok_or_else(|| {
                        ExecError::Internal(format!("unknown node var '{}'", node_var))
                    })?;

                    match graph.winner(nid, &edge_type, 0.01) {
                        Some(winner_node_id) => Ok(winner_node_id.0 as f64),
                        None => Ok(-1.0),
                    }
                }
                "entropy" => {
                    if pos_args.len() < 2 {
                        return Err(ExecError::Internal(
                            "entropy(): requires node and edge_type arguments".into(),
                        ));
                    }
                    let node_var = match &pos_args[0] {
                        ExprAst::Var(v) => v,
                        _ => {
                            return Err(ExecError::Internal(
                                "entropy(): first argument must be a node variable".into(),
                            ))
                        }
                    };
                    let edge_type = match &pos_args[1] {
                        ExprAst::Var(v) => v.clone(),
                        _ => {
                            return Err(ExecError::Internal(
                                "entropy(): edge_type must be an identifier".into(),
                            ))
                        }
                    };

                    let nid = *self.bindings.node_vars.get(node_var).ok_or_else(|| {
                        ExecError::Internal(format!("unknown node var '{}'", node_var))
                    })?;

                    graph.entropy(nid, &edge_type)
                }
                _ => Err(ExecError::Internal(format!("unknown function '{}'", name))),
            }
        }

        fn eval_field(
            &self,
            _target: &ExprAst,
            _field: &str,
            _graph: &BeliefGraph,
        ) -> Result<f64, ExecError> {
            Err(ExecError::Internal(
                "field access not supported in test context".into(),
            ))
        }
    }

    let ctx = TestContext { bindings };

    // winner(A, ROUTES_TO) should return NodeId(2) as f64
    let winner_expr = ExprAst::Call {
        name: "winner".into(),
        args: vec![
            CallArg::Positional(ExprAst::Var("A".into())),
            CallArg::Positional(ExprAst::Var("ROUTES_TO".into())),
        ],
    };

    let result = eval_expr_core(&winner_expr, &graph, &ctx).unwrap();
    assert!((result - NodeId(2).0 as f64).abs() < 0.1);
}

#[test]
fn rule_context_entropy_function() {
    let graph = create_test_graph_with_competing_edges();

    let bindings = MatchBindings {
        node_vars: HashMap::from([("A".into(), NodeId(1))]),
        edge_vars: HashMap::new(),
    };

    use grafial_core::engine::expr_eval::{eval_expr_core, ExprContext};

    struct TestContext {
        bindings: MatchBindings,
    }

    impl ExprContext for TestContext {
        fn resolve_var(&self, name: &str) -> Option<f64> {
            self.bindings
                .node_vars
                .get(name)
                .map(|node_id| node_id.0 as f64)
        }

        fn eval_function(
            &self,
            name: &str,
            pos_args: &[ExprAst],
            _all_args: &[CallArg],
            graph: &BeliefGraph,
        ) -> Result<f64, ExecError> {
            match name {
                "entropy" => {
                    let node_var = match &pos_args[0] {
                        ExprAst::Var(v) => v,
                        _ => {
                            return Err(ExecError::Internal(
                                "entropy(): first argument must be a node variable".into(),
                            ))
                        }
                    };
                    let edge_type = match &pos_args[1] {
                        ExprAst::Var(v) => v.clone(),
                        _ => {
                            return Err(ExecError::Internal(
                                "entropy(): edge_type must be an identifier".into(),
                            ))
                        }
                    };

                    let nid = *self.bindings.node_vars.get(node_var).ok_or_else(|| {
                        ExecError::Internal(format!("unknown node var '{}'", node_var))
                    })?;

                    graph.entropy(nid, &edge_type)
                }
                _ => Err(ExecError::Internal(format!("unknown function '{}'", name))),
            }
        }

        fn eval_field(
            &self,
            _target: &ExprAst,
            _field: &str,
            _graph: &BeliefGraph,
        ) -> Result<f64, ExecError> {
            Err(ExecError::Internal(
                "field access not supported in test context".into(),
            ))
        }
    }

    let ctx = TestContext { bindings };

    let entropy_expr = ExprAst::Call {
        name: "entropy".into(),
        args: vec![
            CallArg::Positional(ExprAst::Var("A".into())),
            CallArg::Positional(ExprAst::Var("ROUTES_TO".into())),
        ],
    };

    let result = eval_expr_core(&entropy_expr, &graph, &ctx).unwrap();

    // Uniform distribution with 2 categories: entropy ≈ 0.693
    assert!(result > 0.5);
    assert!(result < 1.0);
}

// ============================================================================
// EdgePosterior Helper Methods Tests
// ============================================================================

#[test]
fn edge_posterior_independent_mean_probability() {
    let beta = BetaPosterior {
        alpha: 2.0,
        beta: 3.0,
    };
    let posterior = EdgePosterior::Independent(beta);
    let groups = FxHashMap::default();

    let prob = posterior.mean_probability(&groups).unwrap();
    assert!((prob - 2.0 / 5.0).abs() < 1e-10);
}

#[test]
fn edge_posterior_competing_mean_probability() {
    let mut groups = FxHashMap::default();
    let dirichlet = DirichletPosterior::new(vec![1.0, 2.0, 3.0]);
    let group = CompetingEdgeGroup::new(
        CompetingGroupId(1),
        NodeId(1),
        "ROUTES_TO".into(),
        vec![NodeId(2), NodeId(3), NodeId(4)],
        dirichlet,
    );
    groups.insert(CompetingGroupId(1), group);

    let posterior = EdgePosterior::Competing {
        group_id: CompetingGroupId(1),
        category_index: 1, // Second category
    };

    let prob = posterior.mean_probability(&groups).unwrap();
    // E[π_1] = 2 / 6
    assert!((prob - 2.0 / 6.0).abs() < 1e-10);
}

#[test]
fn edge_posterior_competing_mean_probability_errors_on_missing_group() {
    let groups = FxHashMap::default();
    let posterior = EdgePosterior::Competing {
        group_id: CompetingGroupId(999),
        category_index: 0,
    };

    let result = posterior.mean_probability(&groups);
    assert!(result.is_err());
}

// ============================================================================
// Validation Tests
// ============================================================================

#[test]
fn validation_rejects_chosen_evidence_for_independent_edge() {
    use grafial_frontend::ast::*;
    use grafial_frontend::validate_program;

    let program = ProgramAst {
        schemas: vec![Schema {
            name: "S".into(),
            nodes: vec![NodeDef {
                name: "N".into(),
                attrs: vec![],
            }],
            edges: vec![EdgeDef { name: "E".into() }],
        }],
        belief_models: vec![BeliefModel {
            name: "M".into(),
            on_schema: "S".into(),
            nodes: vec![],
            edges: vec![EdgeBeliefDecl {
                edge_type: "E".into(),
                exist: PosteriorType::Bernoulli {
                    params: vec![("prior".into(), 0.5), ("pseudo_count".into(), 2.0)],
                },
            }],
            body_src: "".into(),
        }],
        evidences: vec![EvidenceDef {
            name: "Ev".into(),
            on_model: "M".into(),
            observations: vec![ObserveStmt::Edge {
                edge_type: "E".into(),
                src: ("S".into(), "N1".into()),
                dst: ("S".into(), "N2".into()),
                mode: EvidenceMode::Chosen,
            }],
            body_src: "".into(),
        }],
        rules: vec![],
        flows: vec![],
    };

    let result = validate_program(&program);
    assert!(result.is_err());
    assert!(result.unwrap_err().to_string().contains("independent"));
}

#[test]
fn validation_rejects_present_evidence_for_competing_edge() {
    use grafial_frontend::ast::*;
    use grafial_frontend::validate_program;

    let program = ProgramAst {
        schemas: vec![Schema {
            name: "S".into(),
            nodes: vec![NodeDef {
                name: "N".into(),
                attrs: vec![],
            }],
            edges: vec![EdgeDef { name: "E".into() }],
        }],
        belief_models: vec![BeliefModel {
            name: "M".into(),
            on_schema: "S".into(),
            nodes: vec![],
            edges: vec![EdgeBeliefDecl {
                edge_type: "E".into(),
                exist: PosteriorType::Categorical {
                    group_by: "source".into(),
                    prior: CategoricalPrior::Uniform { pseudo_count: 3.0 },
                    categories: None,
                },
            }],
            body_src: "".into(),
        }],
        evidences: vec![EvidenceDef {
            name: "Ev".into(),
            on_model: "M".into(),
            observations: vec![ObserveStmt::Edge {
                edge_type: "E".into(),
                src: ("S".into(), "N1".into()),
                dst: ("S".into(), "N2".into()),
                mode: EvidenceMode::Present,
            }],
            body_src: "".into(),
        }],
        rules: vec![],
        flows: vec![],
    };

    let result = validate_program(&program);
    assert!(result.is_err());
    assert!(result.unwrap_err().to_string().contains("competing"));
}

#[test]
fn validation_accepts_chosen_evidence_for_competing_edge() {
    use grafial_frontend::ast::*;
    use grafial_frontend::validate_program;

    let program = ProgramAst {
        schemas: vec![Schema {
            name: "S".into(),
            nodes: vec![NodeDef {
                name: "N".into(),
                attrs: vec![],
            }],
            edges: vec![EdgeDef { name: "E".into() }],
        }],
        belief_models: vec![BeliefModel {
            name: "M".into(),
            on_schema: "S".into(),
            nodes: vec![],
            edges: vec![EdgeBeliefDecl {
                edge_type: "E".into(),
                exist: PosteriorType::Categorical {
                    group_by: "source".into(),
                    prior: CategoricalPrior::Uniform { pseudo_count: 3.0 },
                    categories: None,
                },
            }],
            body_src: "".into(),
        }],
        evidences: vec![EvidenceDef {
            name: "Ev".into(),
            on_model: "M".into(),
            observations: vec![ObserveStmt::Edge {
                edge_type: "E".into(),
                src: ("S".into(), "N1".into()),
                dst: ("S".into(), "N2".into()),
                mode: EvidenceMode::Chosen,
            }],
            body_src: "".into(),
        }],
        rules: vec![],
        flows: vec![],
    };

    let result = validate_program(&program);
    assert!(result.is_ok());
}

// ============================================================================
// Dynamic Category Discovery Tests
// ============================================================================

#[test]
fn competing_category_sets_are_fixed_before_updates() {
    use grafial_core::engine::evidence::build_graph_from_evidence;
    use grafial_frontend::ast::*;

    // Create a program with competing edges using uniform prior
    let program = ProgramAst {
        schemas: vec![Schema {
            name: "Network".into(),
            nodes: vec![NodeDef {
                name: "Server".into(),
                attrs: vec![],
            }],
            edges: vec![EdgeDef {
                name: "ROUTES_TO".into(),
            }],
        }],
        belief_models: vec![BeliefModel {
            name: "NetworkBeliefs".into(),
            on_schema: "Network".into(),
            nodes: vec![NodeBeliefDecl {
                node_type: "Server".into(),
                attrs: vec![],
            }],
            edges: vec![EdgeBeliefDecl {
                edge_type: "ROUTES_TO".into(),
                exist: PosteriorType::Categorical {
                    group_by: "source".into(),
                    prior: CategoricalPrior::Uniform { pseudo_count: 6.0 },
                    categories: None,
                },
            }],
            body_src: "".into(),
        }],
        evidences: vec![],
        rules: vec![],
        flows: vec![],
    };

    // Evidence with two destinations in a single competing group.
    let evidence_a = EvidenceDef {
        name: "EvidenceA".into(),
        on_model: "NetworkBeliefs".into(),
        observations: vec![
            ObserveStmt::Edge {
                edge_type: "ROUTES_TO".into(),
                src: ("Server".into(), "S1".into()),
                dst: ("Server".into(), "S2".into()),
                mode: EvidenceMode::Chosen,
            },
            ObserveStmt::Edge {
                edge_type: "ROUTES_TO".into(),
                src: ("Server".into(), "S1".into()),
                dst: ("Server".into(), "S3".into()),
                mode: EvidenceMode::Chosen,
            },
        ],
        body_src: "".into(),
    };

    let graph_a = build_graph_from_evidence(&evidence_a, &program).unwrap();
    let edges_a = graph_a.edges();
    let s1_id_a = edges_a[0].src;
    let group_a = graph_a.get_competing_group(s1_id_a, "ROUTES_TO").unwrap();

    assert_eq!(group_a.categories.len(), 2);
    assert_eq!(group_a.posterior.num_categories(), 2);
    assert_eq!(group_a.posterior.concentrations.len(), 2);
    for alpha in &group_a.posterior.concentrations {
        assert!(
            (*alpha - 4.0).abs() < 1e-10,
            "Expected fixed-set initialization then one observation per category (alpha=4.0), got {}",
            alpha
        );
    }

    let probs_a = group_a.posterior.mean_probabilities();
    let sum_a: f64 = probs_a.iter().sum();
    assert!(
        (sum_a - 1.0).abs() < 1e-10,
        "Probabilities should sum to 1.0, got {}",
        sum_a
    );
    assert!((probs_a[0] - 0.5).abs() < 1e-10);
    assert!((probs_a[1] - 0.5).abs() < 1e-10);

    // Same observations in reverse order should produce identical posteriors.
    let evidence_b = EvidenceDef {
        name: "EvidenceB".into(),
        on_model: "NetworkBeliefs".into(),
        observations: vec![
            ObserveStmt::Edge {
                edge_type: "ROUTES_TO".into(),
                src: ("Server".into(), "S1".into()),
                dst: ("Server".into(), "S3".into()),
                mode: EvidenceMode::Chosen,
            },
            ObserveStmt::Edge {
                edge_type: "ROUTES_TO".into(),
                src: ("Server".into(), "S1".into()),
                dst: ("Server".into(), "S2".into()),
                mode: EvidenceMode::Chosen,
            },
        ],
        body_src: "".into(),
    };
    let graph_b = build_graph_from_evidence(&evidence_b, &program).unwrap();
    let edges_b = graph_b.edges();
    let s1_id_b = edges_b[0].src;
    let group_b = graph_b.get_competing_group(s1_id_b, "ROUTES_TO").unwrap();

    assert_eq!(
        group_a.posterior.concentrations,
        group_b.posterior.concentrations
    );
}

#[test]
fn dynamic_category_discovery_rejects_explicit_prior() {
    use grafial_core::engine::evidence::build_graph_from_evidence;
    use grafial_frontend::ast::*;

    // Create a program with competing edges using explicit prior
    let program = ProgramAst {
        schemas: vec![Schema {
            name: "Network".into(),
            nodes: vec![NodeDef {
                name: "Server".into(),
                attrs: vec![],
            }],
            edges: vec![EdgeDef {
                name: "ROUTES_TO".into(),
            }],
        }],
        belief_models: vec![BeliefModel {
            name: "NetworkBeliefs".into(),
            on_schema: "Network".into(),
            nodes: vec![NodeBeliefDecl {
                node_type: "Server".into(),
                attrs: vec![],
            }],
            edges: vec![EdgeBeliefDecl {
                edge_type: "ROUTES_TO".into(),
                exist: PosteriorType::Categorical {
                    group_by: "source".into(),
                    prior: CategoricalPrior::Explicit {
                        concentrations: vec![2.0, 3.0],
                    },
                    categories: Some(vec!["S2".into(), "S3".into()]),
                },
            }],
            body_src: "".into(),
        }],
        evidences: vec![],
        rules: vec![],
        flows: vec![],
    };

    // Try to observe edge to a new destination S4 (should fail)
    let evidence = EvidenceDef {
        name: "Evidence".into(),
        on_model: "NetworkBeliefs".into(),
        observations: vec![
            ObserveStmt::Edge {
                edge_type: "ROUTES_TO".into(),
                src: ("Server".into(), "S1".into()),
                dst: ("Server".into(), "S2".into()),
                mode: EvidenceMode::Chosen,
            },
            ObserveStmt::Edge {
                edge_type: "ROUTES_TO".into(),
                src: ("Server".into(), "S1".into()),
                dst: ("Server".into(), "S4".into()), // New destination not in explicit prior
                mode: EvidenceMode::Chosen,
            },
        ],
        body_src: "".into(),
    };

    let result = build_graph_from_evidence(&evidence, &program);
    assert!(
        result.is_err(),
        "Should reject dynamic category discovery for explicit prior"
    );
    let err_msg = result.unwrap_err().to_string();
    // The error may mention fixed category constraints or prior/category mismatch.
    assert!(
        err_msg.contains("not allowed for categorical edge")
            || err_msg.contains("fixed category")
            || err_msg.contains("Dynamic category discovery not supported")
            || err_msg.contains("must match number of categories")
            || err_msg.contains("must match fixed category count")
            || err_msg.contains("not found"),
        "Error should mention fixed categories or category mismatch, got: {}",
        err_msg
    );
}
