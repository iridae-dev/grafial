//! Phase 4 Exit Criteria Test
//!
//! This test demonstrates that Phase 4 is complete:
//! "You can run a simple flow, produce a BeliefGraph, and verify structure/beliefs."

use std::collections::HashMap;

use baygraph::engine::errors::ExecError;
use baygraph::engine::flow_exec::run_flow_with_builder;
use baygraph::engine::graph::{BeliefGraph, BetaPosterior, EdgeData, EdgePosterior, GaussianPosterior, NodeData, NodeId, EdgeId};
use baygraph::frontend::ast::*;

/// End-to-end test: Parse → Run Flow → Verify BeliefGraph structure and beliefs
#[test]
fn phase4_exit_criteria_complete_flow_execution() {
    // Build a program with schema, belief model, evidence, rule, and flow
    let program = build_complete_program();

    // Run the flow with a custom evidence builder
    let result = run_flow_with_builder(&program, "Demo", &build_evidence_graph, None).expect("flow execution succeeds");

    // VERIFY 1: Can produce a BeliefGraph from evidence
    let base_graph = result.graphs.get("base").expect("base graph exists");
    assert_eq!(base_graph.nodes().len(), 3, "base graph has 3 nodes");
    assert_eq!(base_graph.edges().len(), 3, "base graph has 3 edges initially");

    // VERIFY 2: Can apply transforms in a pipeline
    let cleaned_graph = result.graphs.get("cleaned").expect("cleaned graph exists");
    assert_eq!(cleaned_graph.nodes().len(), 3, "nodes preserved through pipeline");

    // VERIFY 3: Can verify structure - edges pruned correctly
    assert!(cleaned_graph.edges().len() < base_graph.edges().len(), "edges were pruned");
    assert_eq!(cleaned_graph.edges().len(), 1, "only high-probability edge remains");

    // VERIFY 4: Can verify beliefs - check edge probabilities
    let remaining_edge = &cleaned_graph.edges()[0];
    let prob = cleaned_graph.prob_mean(remaining_edge.id).expect("can read probability");
    assert!(prob > 0.8, "remaining edge has high probability ({})", prob);

    // VERIFY 5: Can verify node beliefs
    let _node = cleaned_graph.node(NodeId(1)).expect("node 1 exists");
    let expectation = cleaned_graph.expectation(NodeId(1), "score").expect("can read expectation");
    assert_eq!(expectation, 100.0, "node expectation is correct");

    // VERIFY 6: Export works - graph available by alias
    let exported = result.exports.get("demo_output").expect("exported graph exists");
    assert_eq!(exported.edges().len(), cleaned_graph.edges().len(), "exported graph matches cleaned graph");

    println!("✓ Phase 4 Exit Criteria Met:");
    println!("  - Can run a simple flow");
    println!("  - Produces BeliefGraph instances");
    println!("  - Can verify graph structure (nodes, edges)");
    println!("  - Can verify beliefs (probabilities, expectations)");
}

/// Builds a complete Baygraph program demonstrating all Phase 4 features
fn build_complete_program() -> ProgramAst {
    // Schema definition
    let schema = Schema {
        name: "SocialNetwork".into(),
        nodes: vec![NodeDef {
            name: "Person".into(),
            attrs: vec![AttrDef {
                name: "score".into(),
                ty: "Real".into(),
            }],
        }],
        edges: vec![EdgeDef {
            name: "Knows".into(),
        }],
    };

    // Belief model
    let belief_model = BeliefModel {
        name: "SocialBeliefs".into(),
        on_schema: "SocialNetwork".into(),
        nodes: vec![],
        edges: vec![],
        body_src: "".into(),
    };

    // Evidence
    let evidence = EvidenceDef {
        name: "SocialEvidence".into(),
        on_model: "SocialBeliefs".into(),
        observations: vec![],
        body_src: "".into(),
    };

    // Rule: Force low-probability edges to be absent
    let rule = RuleDef {
        name: "RemoveLowProbEdges".into(),
        on_model: "SocialBeliefs".into(),
        mode: Some("for_each".into()),
        patterns: vec![PatternItem {
            src: NodePattern {
                var: "A".into(),
                label: "Person".into(),
            },
            edge: EdgePattern {
                var: "e".into(),
                ty: "Knows".into(),
            },
            dst: NodePattern {
                var: "B".into(),
                label: "Person".into(),
            },
        }],
        where_expr: Some(ExprAst::Binary {
            op: BinaryOp::Lt,
            left: Box::new(ExprAst::Call {
                name: "prob".into(),
                args: vec![CallArg::Positional(ExprAst::Var("e".into()))],
            }),
            right: Box::new(ExprAst::Number(0.5)),
        }),
        actions: vec![ActionStmt::ForceAbsent { edge_var: "e".into() }],
    };

    // Flow: Load evidence, apply rule, prune, export
    let flow = FlowDef {
        name: "Demo".into(),
        on_model: "SocialBeliefs".into(),
        graphs: vec![
            // Load from evidence
            GraphDef {
                name: "base".into(),
                expr: GraphExpr::FromEvidence {
                    evidence: "SocialEvidence".into(),
                },
            },
            // Apply pipeline transforms
            GraphDef {
                name: "cleaned".into(),
                expr: GraphExpr::Pipeline {
                    start: "base".into(),
                    transforms: vec![
                        // Apply the rule
                        Transform::ApplyRule {
                            rule: "RemoveLowProbEdges".into(),
                        },
                        // Prune edges with very low probability
                        Transform::PruneEdges {
                            edge_type: "Knows".into(),
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
            alias: "demo_output".into(),
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

/// Evidence builder that creates a test social network graph
fn build_evidence_graph(_evidence: &EvidenceDef) -> Result<BeliefGraph, ExecError> {
    let mut graph = BeliefGraph::default();

    // Create 3 Person nodes with different scores
    graph.insert_node(NodeData {
        id: NodeId(1),
        label: "Person".into(),
        attrs: HashMap::from([("score".into(), GaussianPosterior {
            mean: 100.0,
            precision: 10.0,
        })]),
    });
    graph.insert_node(NodeData {
        id: NodeId(2),
        label: "Person".into(),
        attrs: HashMap::from([("score".into(), GaussianPosterior {
            mean: 75.0,
            precision: 10.0,
        })]),
    });
    graph.insert_node(NodeData {
        id: NodeId(3),
        label: "Person".into(),
        attrs: HashMap::from([("score".into(), GaussianPosterior {
            mean: 50.0,
            precision: 10.0,
        })]),
    });

    // Create 3 Knows edges with different probabilities
    // High probability edge (will survive)
    graph.insert_edge(EdgeData {
        id: EdgeId(1),
        src: NodeId(1),
        dst: NodeId(2),
        ty: "Knows".into(),
        exist: EdgePosterior::Independent(BetaPosterior {
            alpha: 9.0,
            beta: 1.0,
        }), // prob ≈ 0.9
    });

    // Medium probability edge (will be forced absent by rule, then pruned)
    graph.insert_edge(EdgeData {
        id: EdgeId(2),
        src: NodeId(2),
        dst: NodeId(3),
        ty: "Knows".into(),
        exist: EdgePosterior::Independent(BetaPosterior {
            alpha: 3.0,
            beta: 7.0,
        }), // prob = 0.3 < 0.5
    });

    // Low probability edge (will be forced absent by rule, then pruned)
    graph.insert_edge(EdgeData {
        id: EdgeId(3),
        src: NodeId(1),
        dst: NodeId(3),
        ty: "Knows".into(),
        exist: EdgePosterior::Independent(BetaPosterior {
            alpha: 1.0,
            beta: 19.0,
        }), // prob = 0.05 < 0.5
    });

    // Apply deltas before returning to ensure nodes() and edges() work correctly
    graph.ensure_owned();

    Ok(graph)
}

#[test]
fn phase4_demonstrates_immutable_graph_transforms() {
    // Demonstrate that transforms create new graphs (immutability)
    let program = build_complete_program();
    let result = run_flow_with_builder(&program, "Demo", &build_evidence_graph, None).expect("flow succeeds");

    let base = result.graphs.get("base").expect("base exists");
    let cleaned = result.graphs.get("cleaned").expect("cleaned exists");

    // Base graph unchanged - has all original edges
    assert_eq!(base.edges().len(), 3, "base graph unchanged");

    // Cleaned graph is different - edges removed
    assert_eq!(cleaned.edges().len(), 1, "cleaned graph has fewer edges");

    // This demonstrates immutability between transforms
    println!("✓ Graph immutability demonstrated:");
    println!("  - Base graph: {} edges", base.edges().len());
    println!("  - Cleaned graph: {} edges", cleaned.edges().len());
}

#[test]
fn phase4_demonstrates_multiple_transform_pipeline() {
    // Demonstrate that multiple transforms can be chained
    let program = build_complete_program();
    let result = run_flow_with_builder(&program, "Demo", &build_evidence_graph, None).expect("flow succeeds");

    let cleaned = result.graphs.get("cleaned").expect("cleaned exists");

    // Pipeline consisted of:
    // 1. Load from evidence (3 edges)
    // 2. Apply rule (forces edges with prob < 0.5 to near-zero)
    // 3. Prune edges (removes edges with prob < 0.01)

    // Result: only the high-prob edge remains
    assert_eq!(cleaned.edges().len(), 1);

    println!("✓ Multi-transform pipeline demonstrated:");
    println!("  - from_evidence: 3 edges");
    println!("  - apply_rule: forced 2 edges to low prob");
    println!("  - prune_edges: removed edges with prob < 0.01");
    println!("  - final result: 1 edge");
}
