//! Tests for evidence building functionality.

use grafial_core::engine::evidence::build_graph_from_evidence;
use grafial_frontend::ast::*;

fn create_test_program() -> ProgramAst {
    ProgramAst {
        schemas: vec![Schema {
            name: "Social".into(),
            nodes: vec![NodeDef {
                name: "Person".into(),
                attrs: vec![AttrDef {
                    name: "score".into(),
                    ty: "Real".into(),
                }],
            }],
            edges: vec![EdgeDef {
                name: "KNOWS".into(),
            }],
        }],
        belief_models: vec![BeliefModel {
            name: "SocialBeliefs".into(),
            on_schema: "Social".into(),
            nodes: vec![NodeBeliefDecl {
                node_type: "Person".into(),
                attrs: vec![(
                    "score".into(),
                    PosteriorType::Gaussian {
                        params: vec![
                            ("prior_mean".into(), 0.0),
                            ("prior_precision".into(), 0.01),
                            ("observation_precision".into(), 1.0),
                        ],
                    },
                )],
            }],
            edges: vec![EdgeBeliefDecl {
                edge_type: "KNOWS".into(),
                exist: PosteriorType::Bernoulli {
                    params: vec![("prior".into(), 0.5), ("pseudo_count".into(), 2.0)],
                },
            }],
            body_src: "".into(),
        }],
        evidences: vec![],
        rules: vec![],
        flows: vec![],
    }
}

#[test]
fn build_graph_from_evidence_creates_nodes() {
    let mut program = create_test_program();

    let evidence = EvidenceDef {
        name: "TestEvidence".into(),
        on_model: "SocialBeliefs".into(),
        observations: vec![ObserveStmt::Attribute {
            node: ("Person".into(), "Alice".into()),
            attr: "score".into(),
            value: 10.0,
        }],
        body_src: "".into(),
    };
    program.evidences.push(evidence.clone());

    let graph = build_graph_from_evidence(&evidence, &program).unwrap();

    // Should have created one node
    assert_eq!(graph.nodes().len(), 1);
    let node = &graph.nodes()[0];
    assert_eq!(node.label.as_ref(), "Person");

    // Check that attribute was observed (mean should be updated from 0.0)
    let score = graph.expectation(node.id, "score").unwrap();
    assert!(
        score > 0.0,
        "Expected score > 0 after observing 10.0, got {}",
        score
    );
}

#[test]
fn build_graph_from_evidence_creates_edges() {
    let mut program = create_test_program();

    let evidence = EvidenceDef {
        name: "TestEvidence".into(),
        on_model: "SocialBeliefs".into(),
        observations: vec![ObserveStmt::Edge {
            edge_type: "KNOWS".into(),
            src: ("Person".into(), "Alice".into()),
            dst: ("Person".into(), "Bob".into()),
            mode: EvidenceMode::Present,
        }],
        body_src: "".into(),
    };
    program.evidences.push(evidence.clone());

    let graph = build_graph_from_evidence(&evidence, &program).unwrap();

    // Should have created 2 nodes and 1 edge
    assert_eq!(graph.nodes().len(), 2);
    assert_eq!(graph.edges().len(), 1);

    let edge = &graph.edges()[0];
    assert_eq!(edge.ty.as_ref(), "KNOWS");

    // After observing present, probability should be > 0.5
    let prob = graph.prob_mean(edge.id).unwrap();
    assert!(
        prob > 0.5,
        "Expected prob > 0.5 after observing present, got {}",
        prob
    );
}

#[test]
fn build_graph_from_evidence_handles_multiple_observations() {
    let mut program = create_test_program();

    let evidence = EvidenceDef {
        name: "TestEvidence".into(),
        on_model: "SocialBeliefs".into(),
        observations: vec![
            ObserveStmt::Attribute {
                node: ("Person".into(), "Alice".into()),
                attr: "score".into(),
                value: 10.0,
            },
            ObserveStmt::Attribute {
                node: ("Person".into(), "Bob".into()),
                attr: "score".into(),
                value: 5.0,
            },
            ObserveStmt::Edge {
                edge_type: "KNOWS".into(),
                src: ("Person".into(), "Alice".into()),
                dst: ("Person".into(), "Bob".into()),
                mode: EvidenceMode::Present,
            },
        ],
        body_src: "".into(),
    };
    program.evidences.push(evidence.clone());

    let graph = build_graph_from_evidence(&evidence, &program).unwrap();

    assert_eq!(graph.nodes().len(), 2);
    assert_eq!(graph.edges().len(), 1);
}

#[test]
fn build_graph_from_evidence_handles_absent_edge() {
    let mut program = create_test_program();

    let evidence = EvidenceDef {
        name: "TestEvidence".into(),
        on_model: "SocialBeliefs".into(),
        observations: vec![ObserveStmt::Edge {
            edge_type: "KNOWS".into(),
            src: ("Person".into(), "Alice".into()),
            dst: ("Person".into(), "Bob".into()),
            mode: EvidenceMode::Absent,
        }],
        body_src: "".into(),
    };
    program.evidences.push(evidence.clone());

    let graph = build_graph_from_evidence(&evidence, &program).unwrap();

    let edge = &graph.edges()[0];
    // After observing absent, probability should be < 0.5
    let prob = graph.prob_mean(edge.id).unwrap();
    assert!(
        prob < 0.5,
        "Expected prob < 0.5 after observing absent, got {}",
        prob
    );
}
