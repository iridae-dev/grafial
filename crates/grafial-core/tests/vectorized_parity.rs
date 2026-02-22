//! Tests verifying deterministic parity between scalar and vectorized evidence paths.
//!
//! Ensures that batched vectorized updates produce identical results to sequential scalar updates.

#[cfg(feature = "vectorized")]
use grafial_core::engine::graph::{BetaPosterior, GaussianPosterior};
#[cfg(feature = "vectorized")]
use grafial_core::engine::vectorized::{beta_batch_update, gaussian_batch_update};
use grafial_core::engine::{evidence::build_graph_from_evidence, graph::NodeId};
use grafial_frontend::ast::{
    BeliefModel, EdgeBeliefDecl, EvidenceDef, EvidenceMode, NodeBeliefDecl, ObserveStmt,
    PosteriorType, ProgramAst, Schema,
};

/// Test that vectorized Gaussian updates match sequential updates exactly
#[test]
#[cfg(feature = "vectorized")]
fn test_gaussian_parity() {
    let test_cases = vec![
        // Single observation
        vec![(10.0, 1.0)],
        // Multiple observations with same precision
        vec![(5.0, 1.0), (10.0, 1.0), (15.0, 1.0)],
        // Varying precisions
        vec![(8.0, 0.5), (12.0, 2.0), (10.0, 1.5)],
        // Large batch
        (0..100)
            .map(|i| (i as f64, 1.0 + i as f64 / 100.0))
            .collect(),
    ];

    for observations in test_cases {
        let prior = GaussianPosterior {
            mean: 5.0,
            precision: 2.0,
        };

        // Sequential scalar updates
        let mut scalar_posterior = prior;
        for (value, tau_obs) in &observations {
            let tau_old = scalar_posterior.precision;
            let tau = tau_obs.max(1e-12); // MIN_OBS_PRECISION from graph.rs
            let tau_new = tau_old + tau;
            let mu_new = (tau_old * scalar_posterior.mean + tau * value) / tau_new;
            scalar_posterior = GaussianPosterior {
                mean: mu_new,
                precision: tau_new,
            };
        }

        // Vectorized batch update
        let vectorized_posterior = gaussian_batch_update(&prior, &observations).unwrap();

        // Compare results (should be identical)
        assert!(
            (scalar_posterior.mean - vectorized_posterior.mean).abs() < 1e-12,
            "Mean mismatch: scalar={}, vectorized={}",
            scalar_posterior.mean,
            vectorized_posterior.mean
        );
        assert!(
            (scalar_posterior.precision - vectorized_posterior.precision).abs() < 1e-12,
            "Precision mismatch: scalar={}, vectorized={}",
            scalar_posterior.precision,
            vectorized_posterior.precision
        );
    }
}

/// Test that vectorized Beta updates match sequential updates exactly
#[test]
#[cfg(feature = "vectorized")]
fn test_beta_parity() {
    let test_cases = vec![
        // All present
        vec![true, true, true],
        // All absent
        vec![false, false, false],
        // Mixed
        vec![true, false, true, false, true],
        // Large batch
        (0..100).map(|i| i % 3 != 0).collect(),
    ];

    for observations in test_cases {
        let prior = BetaPosterior {
            alpha: 2.0,
            beta: 3.0,
        };

        // Sequential scalar updates
        let mut scalar_posterior = prior;
        for present in &observations {
            if *present {
                scalar_posterior.alpha += 1.0;
            } else {
                scalar_posterior.beta += 1.0;
            }
        }

        // Vectorized batch update
        let vectorized_posterior = beta_batch_update(&prior, &observations).unwrap();

        // Compare results (should be identical)
        assert_eq!(
            scalar_posterior.alpha, vectorized_posterior.alpha,
            "Alpha mismatch: scalar={}, vectorized={}",
            scalar_posterior.alpha, vectorized_posterior.alpha
        );
        assert_eq!(
            scalar_posterior.beta, vectorized_posterior.beta,
            "Beta mismatch: scalar={}, vectorized={}",
            scalar_posterior.beta, vectorized_posterior.beta
        );
    }
}

/// Test that the evidence ingestion pipeline produces identical results
/// whether using scalar or vectorized paths
#[test]
fn test_evidence_pipeline_parity() {
    // Create a test program with belief model and evidence
    let schema = Schema {
        name: "test_schema".to_string(),
        nodes: vec![grafial_frontend::ast::NodeDef {
            name: "Person".to_string(),
            attrs: vec![
                grafial_frontend::ast::AttrDef {
                    name: "age".to_string(),
                    ty: "Real".to_string(),
                },
                grafial_frontend::ast::AttrDef {
                    name: "height".to_string(),
                    ty: "Real".to_string(),
                },
            ],
        }],
        edges: vec![grafial_frontend::ast::EdgeDef {
            name: "knows".to_string(),
        }],
    };

    let belief_model = BeliefModel {
        name: "test_model".to_string(),
        on_schema: "test_schema".to_string(),
        nodes: vec![NodeBeliefDecl {
            node_type: "Person".to_string(),
            attrs: vec![
                (
                    "age".to_string(),
                    PosteriorType::Gaussian {
                        params: vec![
                            ("prior_mean".to_string(), 30.0),
                            ("prior_precision".to_string(), 0.01),
                            ("observation_precision".to_string(), 1.0),
                        ],
                    },
                ),
                (
                    "height".to_string(),
                    PosteriorType::Gaussian {
                        params: vec![
                            ("prior_mean".to_string(), 170.0),
                            ("prior_precision".to_string(), 0.01),
                            ("observation_precision".to_string(), 0.5),
                        ],
                    },
                ),
            ],
        }],
        edges: vec![EdgeBeliefDecl {
            edge_type: "knows".to_string(),
            exist: PosteriorType::Bernoulli {
                params: vec![
                    ("prior".to_string(), 0.3),
                    ("pseudo_count".to_string(), 2.0),
                ],
            },
            weight: None,
        }],
        body_src: String::new(), // Not needed for testing
    };

    // Create evidence with multiple observations on the same targets
    let evidence = EvidenceDef {
        name: "test_evidence".to_string(),
        on_model: "test_model".to_string(),
        body_src: String::new(), // Not needed for testing
        observations: vec![
            // Multiple age observations for Alice
            ObserveStmt::Attribute {
                node: ("Person".to_string(), "Alice".to_string()),
                attr: "age".to_string(),
                value: 25.0,
                precision: None,
            },
            ObserveStmt::Attribute {
                node: ("Person".to_string(), "Alice".to_string()),
                attr: "age".to_string(),
                value: 26.0,
                precision: None,
            },
            ObserveStmt::Attribute {
                node: ("Person".to_string(), "Alice".to_string()),
                attr: "age".to_string(),
                value: 24.5,
                precision: Some(2.0),
            },
            // Multiple height observations for Alice
            ObserveStmt::Attribute {
                node: ("Person".to_string(), "Alice".to_string()),
                attr: "height".to_string(),
                value: 165.0,
                precision: None,
            },
            ObserveStmt::Attribute {
                node: ("Person".to_string(), "Alice".to_string()),
                attr: "height".to_string(),
                value: 166.0,
                precision: None,
            },
            // Multiple edge observations
            ObserveStmt::Edge {
                edge_type: "knows".to_string(),
                src: ("Person".to_string(), "Alice".to_string()),
                dst: ("Person".to_string(), "Bob".to_string()),
                mode: EvidenceMode::Present,
            },
            ObserveStmt::Edge {
                edge_type: "knows".to_string(),
                src: ("Person".to_string(), "Alice".to_string()),
                dst: ("Person".to_string(), "Charlie".to_string()),
                mode: EvidenceMode::Absent,
            },
            ObserveStmt::Edge {
                edge_type: "knows".to_string(),
                src: ("Person".to_string(), "Alice".to_string()),
                dst: ("Person".to_string(), "Charlie".to_string()),
                mode: EvidenceMode::Present,
            },
        ],
    };

    let program = ProgramAst {
        schemas: vec![schema],
        belief_models: vec![belief_model],
        evidences: vec![evidence.clone()],
        rules: vec![],
        flows: vec![],
    };

    // Build graph with vectorized path (if feature enabled)
    let graph = build_graph_from_evidence(&evidence, &program).unwrap();

    // Get Alice node
    let alice_id = NodeId(0); // First node created

    // Check age posterior using expectation and precision methods
    let age_mean = graph.expectation(alice_id, "age").unwrap();
    let age_precision = graph.precision(alice_id, "age").unwrap();

    // Calculate expected result manually (sequential updates)
    // Prior: mean=30.0, precision=0.01
    // Obs 1: value=25.0, tau=1.0
    //   tau_new = 0.01 + 1.0 = 1.01
    //   mu_new = (0.01 * 30.0 + 1.0 * 25.0) / 1.01 = 25.0495...
    // Obs 2: value=26.0, tau=1.0
    //   tau_new = 1.01 + 1.0 = 2.01
    //   mu_new = (1.01 * 25.0495... + 1.0 * 26.0) / 2.01 = 25.525...
    // Obs 3: value=24.5, tau=2.0
    //   tau_new = 2.01 + 2.0 = 4.01
    //   mu_new = (2.01 * 25.525... + 2.0 * 24.5) / 4.01 = 25.018...

    let expected_age_mean = {
        let mut mean = 30.0;
        let mut precision = 0.01;

        // Obs 1
        let tau_new = precision + 1.0;
        mean = (precision * mean + 1.0 * 25.0) / tau_new;
        precision = tau_new;

        // Obs 2
        let tau_new = precision + 1.0;
        mean = (precision * mean + 1.0 * 26.0) / tau_new;
        precision = tau_new;

        // Obs 3
        let tau_new = precision + 2.0;
        mean = (precision * mean + 2.0 * 24.5) / tau_new;
        // precision = tau_new; // Not needed for final calculation

        mean
    };

    assert!(
        (age_mean - expected_age_mean).abs() < 1e-10,
        "Age mean mismatch: got={}, expected={}",
        age_mean,
        expected_age_mean
    );
    assert!(
        (age_precision - 4.01).abs() < 1e-10,
        "Age precision mismatch: got={}, expected=4.01",
        age_precision
    );

    // Check height posterior
    let height_precision = graph.precision(alice_id, "height").unwrap();
    assert!(
        (height_precision - 1.01).abs() < 1e-10,
        "Height precision mismatch: got={}, expected=1.01",
        height_precision
    );
}

/// Test edge cases and boundary conditions
#[test]
#[cfg(feature = "vectorized")]
fn test_vectorized_edge_cases() {
    // Empty observations
    {
        let prior = GaussianPosterior {
            mean: 5.0,
            precision: 2.0,
        };
        let result = gaussian_batch_update(&prior, &[]).unwrap();
        assert_eq!(result.mean, prior.mean);
        assert_eq!(result.precision, prior.precision);
    }

    // Very small precision (should be clamped to MIN_OBS_PRECISION)
    {
        let prior = GaussianPosterior {
            mean: 0.0,
            precision: 1.0,
        };
        let observations = vec![(10.0, 1e-15), (20.0, 1e-20)];
        let result = gaussian_batch_update(&prior, &observations).unwrap();

        // Both observations should be clamped to MIN_OBS_PRECISION (1e-12)
        let expected_precision = 1.0 + 2.0 * 1e-12;
        assert!((result.precision - expected_precision).abs() < 1e-14);
    }

    // Beta with minimum parameter clamping
    {
        let prior = BetaPosterior {
            alpha: 0.001,
            beta: 0.001,
        };
        let observations = vec![true, false];
        let result = beta_batch_update(&prior, &observations).unwrap();

        // Should be clamped to at least MIN_BETA_PARAM (0.01)
        assert!(result.alpha >= 0.01);
        assert!(result.beta >= 0.01);
    }
}

/// Test that vectorized updates handle numerical stability correctly
#[test]
#[cfg(feature = "vectorized")]
fn test_vectorized_numerical_stability() {
    // Large batch with accumulating precision
    {
        let prior = GaussianPosterior {
            mean: 0.0,
            precision: 1.0,
        };

        // 10000 observations should not cause numerical issues
        let observations: Vec<(f64, f64)> = (0..10000).map(|i| (i as f64 / 1000.0, 1.0)).collect();

        let result = gaussian_batch_update(&prior, &observations).unwrap();
        assert!(result.mean.is_finite());
        assert!(result.precision.is_finite());
        assert!(result.precision > 0.0);
    }

    // Beta with large counts
    {
        let prior = BetaPosterior {
            alpha: 1.0,
            beta: 1.0,
        };

        // 10000 observations
        let observations: Vec<bool> = (0..10000).map(|i| i % 2 == 0).collect();

        let result = beta_batch_update(&prior, &observations).unwrap();
        assert!(result.alpha.is_finite());
        assert!(result.beta.is_finite());
        assert_eq!(result.alpha, 5001.0); // 1 + 5000 true observations
        assert_eq!(result.beta, 5001.0); // 1 + 5000 false observations
    }
}
