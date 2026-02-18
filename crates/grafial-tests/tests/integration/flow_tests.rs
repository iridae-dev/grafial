use std::collections::HashMap;

use grafial_core::engine::errors::ExecError;
use grafial_core::engine::flow_exec::{run_flow_with_builder, FlowResult};
use grafial_core::engine::graph::{
    BeliefGraph, BetaPosterior, EdgeData, EdgeId, EdgePosterior, GaussianPosterior, NodeData,
    NodeId,
};
use grafial_frontend::ast::*;

fn build_test_program() -> ProgramAst {
    // schema + model names are carried only for clarity in this phase
    let rules = vec![RuleDef {
        name: "ForceLowProb".into(),
        on_model: "M".into(),
        mode: Some("for_each".into()),
        patterns: vec![PatternItem {
            src: NodePattern {
                var: "A".into(),
                label: "Person".into(),
            },
            edge: EdgePattern {
                var: "e".into(),
                ty: "REL".into(),
            },
            dst: NodePattern {
                var: "B".into(),
                label: "Person".into(),
            },
        }],
        // where prob(e) >= 0.5
        where_expr: Some(ExprAst::Binary {
            op: BinaryOp::Ge,
            left: Box::new(ExprAst::Call {
                name: "prob".into(),
                args: vec![CallArg::Positional(ExprAst::Var("e".into()))],
            }),
            right: Box::new(ExprAst::Number(0.5)),
        }),
        // action: force_absent e
        actions: vec![ActionStmt::ForceAbsent {
            edge_var: "e".into(),
        }],
    }];

    let flows = vec![FlowDef {
        name: "Demo".into(),
        on_model: "M".into(),
        graphs: vec![
            GraphDef {
                name: "base".into(),
                expr: GraphExpr::FromEvidence {
                    evidence: "Ev".into(),
                },
            },
            GraphDef {
                name: "cleaned".into(),
                expr: GraphExpr::Pipeline {
                    start: "base".into(),
                    transforms: vec![
                        Transform::ApplyRule {
                            rule: "ForceLowProb".into(),
                        },
                        Transform::PruneEdges {
                            edge_type: "REL".into(),
                            predicate: ExprAst::Binary {
                                op: BinaryOp::Lt,
                                left: Box::new(ExprAst::Call {
                                    name: "prob".into(),
                                    args: vec![CallArg::Positional(ExprAst::Var("edge".into()))],
                                }),
                                right: Box::new(ExprAst::Number(0.1)),
                            },
                        },
                    ],
                },
            },
        ],
        metrics: vec![],
        exports: vec![ExportDef {
            graph: "cleaned".into(),
            alias: "demo".into(),
        }],
        metric_exports: vec![],
        metric_imports: vec![],
    }];

    ProgramAst {
        schemas: vec![],
        belief_models: vec![],
        evidences: vec![EvidenceDef {
            name: "Ev".into(),
            on_model: "M".into(),
            body_src: "".into(),
            observations: vec![],
        }],
        rules,
        flows,
    }
}

fn evidence_builder(_: &EvidenceDef) -> Result<BeliefGraph, ExecError> {
    use std::sync::Arc;
    // Build a tiny graph: Person(1) -[REL]-> Person(2) with p=0.8
    let mut g = BeliefGraph::default();
    g.insert_node(NodeData {
        id: NodeId(1),
        label: Arc::from("Person"),
        attrs: HashMap::from([(
            "some_value".into(),
            GaussianPosterior {
                mean: 10.0,
                precision: 1.0,
            },
        )]),
    });
    g.insert_node(NodeData {
        id: NodeId(2),
        label: Arc::from("Person"),
        attrs: HashMap::from([(
            "some_value".into(),
            GaussianPosterior {
                mean: 0.0,
                precision: 1.0,
            },
        )]),
    });
    g.insert_edge(EdgeData {
        id: EdgeId(1),
        src: NodeId(1),
        dst: NodeId(2),
        ty: Arc::from("REL"),
        exist: EdgePosterior::independent(BetaPosterior {
            alpha: 8.0,
            beta: 2.0,
        }),
    });
    g.ensure_owned();
    Ok(g)
}

#[test]
fn run_flow_demo_applies_rule_and_prunes() {
    let prog = build_test_program();
    let result: FlowResult =
        run_flow_with_builder(&prog, "Demo", &evidence_builder, None).expect("run flow");

    // base graph should have one edge
    let base = result.graphs.get("base").expect("base graph");
    assert_eq!(base.edges().len(), 1);

    // cleaned graph exported as "demo" should have zero edges after force_absent + prune
    let exported = result.exports.get("demo").expect("exported graph");
    assert_eq!(exported.edges().len(), 0);

    // intervention audit hook should include the apply_rule transform event
    assert_eq!(result.intervention_audit.len(), 1);
    let event = &result.intervention_audit[0];
    assert_eq!(event.flow, "Demo");
    assert_eq!(event.graph, "cleaned");
    assert_eq!(event.rule, "ForceLowProb");
    assert!(event.matched_bindings >= 1);
    assert!(event.actions_executed >= 1);
}

#[test]
fn run_flow_from_evidence_loads_graph() {
    let prog = ProgramAst {
        schemas: vec![],
        belief_models: vec![],
        evidences: vec![EvidenceDef {
            name: "Ev".into(),
            on_model: "M".into(),
            body_src: "".into(),
            observations: vec![],
        }],
        rules: vec![],
        flows: vec![FlowDef {
            name: "Simple".into(),
            on_model: "M".into(),
            graphs: vec![GraphDef {
                name: "g".into(),
                expr: GraphExpr::FromEvidence {
                    evidence: "Ev".into(),
                },
            }],
            metrics: vec![],
            exports: vec![],
            metric_exports: vec![],
            metric_imports: vec![],
        }],
    };

    let result = run_flow_with_builder(&prog, "Simple", &evidence_builder, None).expect("run flow");
    let g = result.graphs.get("g").expect("graph g");
    assert_eq!(g.nodes().len(), 2);
    assert_eq!(g.edges().len(), 1);
}

#[test]
fn run_flow_exports_named_graph() {
    let prog = ProgramAst {
        schemas: vec![],
        belief_models: vec![],
        evidences: vec![EvidenceDef {
            name: "Ev".into(),
            on_model: "M".into(),
            body_src: "".into(),
            observations: vec![],
        }],
        rules: vec![],
        flows: vec![FlowDef {
            name: "ExportTest".into(),
            on_model: "M".into(),
            graphs: vec![GraphDef {
                name: "g".into(),
                expr: GraphExpr::FromEvidence {
                    evidence: "Ev".into(),
                },
            }],
            metrics: vec![],
            exports: vec![ExportDef {
                graph: "g".into(),
                alias: "exported".into(),
            }],
            metric_exports: vec![],
            metric_imports: vec![],
        }],
    };

    let result =
        run_flow_with_builder(&prog, "ExportTest", &evidence_builder, None).expect("run flow");
    assert!(result.exports.contains_key("exported"));
    assert_eq!(result.exports.get("exported").unwrap().edges().len(), 1);
}

#[test]
fn run_flow_errors_on_unknown_rule() {
    let prog = ProgramAst {
        schemas: vec![],
        belief_models: vec![],
        evidences: vec![EvidenceDef {
            name: "Ev".into(),
            on_model: "M".into(),
            body_src: "".into(),
            observations: vec![],
        }],
        rules: vec![],
        flows: vec![FlowDef {
            name: "BadRule".into(),
            on_model: "M".into(),
            graphs: vec![
                GraphDef {
                    name: "g".into(),
                    expr: GraphExpr::FromEvidence {
                        evidence: "Ev".into(),
                    },
                },
                GraphDef {
                    name: "out".into(),
                    expr: GraphExpr::Pipeline {
                        start: "g".into(),
                        transforms: vec![Transform::ApplyRule {
                            rule: "NonExistentRule".into(),
                        }],
                    },
                },
            ],
            metrics: vec![],
            exports: vec![],
            metric_exports: vec![],
            metric_imports: vec![],
        }],
    };

    let result = run_flow_with_builder(&prog, "BadRule", &evidence_builder, None);
    assert!(matches!(result, Err(ExecError::Internal(_))));
}

#[test]
fn run_flow_multiple_transforms_in_pipeline() {
    // Build a graph with 3 edges: high prob, medium prob, low prob
    fn multi_edge_builder(_: &EvidenceDef) -> Result<BeliefGraph, ExecError> {
        use std::sync::Arc;
        let mut g = BeliefGraph::default();
        g.insert_node(NodeData {
            id: NodeId(1),
            label: Arc::from("Node"),
            attrs: HashMap::new(),
        });
        g.insert_node(NodeData {
            id: NodeId(2),
            label: Arc::from("Node"),
            attrs: HashMap::new(),
        });
        g.insert_node(NodeData {
            id: NodeId(3),
            label: Arc::from("Node"),
            attrs: HashMap::new(),
        });
        // High probability edge
        g.insert_edge(EdgeData {
            id: EdgeId(1),
            src: NodeId(1),
            dst: NodeId(2),
            ty: Arc::from("LINK"),
            exist: EdgePosterior::independent(BetaPosterior {
                alpha: 9.0,
                beta: 1.0,
            }),
        });
        // Medium probability edge
        g.insert_edge(EdgeData {
            id: EdgeId(2),
            src: NodeId(2),
            dst: NodeId(3),
            ty: Arc::from("LINK"),
            exist: EdgePosterior::independent(BetaPosterior {
                alpha: 5.0,
                beta: 5.0,
            }),
        });
        // Low probability edge
        g.insert_edge(EdgeData {
            id: EdgeId(3),
            src: NodeId(1),
            dst: NodeId(3),
            ty: Arc::from("LINK"),
            exist: EdgePosterior::independent(BetaPosterior {
                alpha: 1.0,
                beta: 9.0,
            }),
        });
        g.ensure_owned();
        Ok(g)
    }

    let prog = ProgramAst {
        schemas: vec![],
        belief_models: vec![],
        evidences: vec![EvidenceDef {
            name: "Ev".into(),
            on_model: "M".into(),
            body_src: "".into(),
            observations: vec![],
        }],
        rules: vec![],
        flows: vec![FlowDef {
            name: "MultiTransform".into(),
            on_model: "M".into(),
            graphs: vec![
                GraphDef {
                    name: "base".into(),
                    expr: GraphExpr::FromEvidence {
                        evidence: "Ev".into(),
                    },
                },
                GraphDef {
                    name: "filtered".into(),
                    expr: GraphExpr::Pipeline {
                        start: "base".into(),
                        transforms: vec![
                            // First prune: remove edges with prob < 0.3
                            Transform::PruneEdges {
                                edge_type: "LINK".into(),
                                predicate: ExprAst::Binary {
                                    op: BinaryOp::Lt,
                                    left: Box::new(ExprAst::Call {
                                        name: "prob".into(),
                                        args: vec![CallArg::Positional(ExprAst::Var(
                                            "edge".into(),
                                        ))],
                                    }),
                                    right: Box::new(ExprAst::Number(0.3)),
                                },
                            },
                            // Second prune: remove edges with prob > 0.7
                            Transform::PruneEdges {
                                edge_type: "LINK".into(),
                                predicate: ExprAst::Binary {
                                    op: BinaryOp::Gt,
                                    left: Box::new(ExprAst::Call {
                                        name: "prob".into(),
                                        args: vec![CallArg::Positional(ExprAst::Var(
                                            "edge".into(),
                                        ))],
                                    }),
                                    right: Box::new(ExprAst::Number(0.7)),
                                },
                            },
                        ],
                    },
                },
            ],
            metrics: vec![],
            exports: vec![],
            metric_exports: vec![],
            metric_imports: vec![],
        }],
    };

    let result = run_flow_with_builder(&prog, "MultiTransform", &multi_edge_builder, None)
        .expect("run flow");

    // Base should have all 3 edges
    let base = result.graphs.get("base").expect("base graph");
    assert_eq!(base.edges().len(), 3);

    // Filtered should only have the medium prob edge (0.5)
    let filtered = result.graphs.get("filtered").expect("filtered graph");
    assert_eq!(filtered.edges().len(), 1);
    assert_eq!(filtered.edges()[0].id, EdgeId(2));
}

#[test]
fn run_flow_pipeline_preserves_nodes() {
    let prog = build_test_program();
    let result = run_flow_with_builder(&prog, "Demo", &evidence_builder, None).expect("run flow");

    let base = result.graphs.get("base").expect("base graph");
    let cleaned = result.graphs.get("cleaned").expect("cleaned graph");

    // Nodes should be preserved through pipeline
    assert_eq!(base.nodes().len(), cleaned.nodes().len());
    assert_eq!(cleaned.nodes().len(), 2);
}

#[test]
fn run_flow_empty_pipeline_clones_graph() {
    let prog = ProgramAst {
        schemas: vec![],
        belief_models: vec![],
        evidences: vec![EvidenceDef {
            name: "Ev".into(),
            on_model: "M".into(),
            body_src: "".into(),
            observations: vec![],
        }],
        rules: vec![],
        flows: vec![FlowDef {
            name: "Clone".into(),
            on_model: "M".into(),
            graphs: vec![
                GraphDef {
                    name: "base".into(),
                    expr: GraphExpr::FromEvidence {
                        evidence: "Ev".into(),
                    },
                },
                GraphDef {
                    name: "copy".into(),
                    expr: GraphExpr::Pipeline {
                        start: "base".into(),
                        transforms: vec![],
                    },
                },
            ],
            metrics: vec![],
            exports: vec![],
            metric_exports: vec![],
            metric_imports: vec![],
        }],
    };

    let result = run_flow_with_builder(&prog, "Clone", &evidence_builder, None).expect("run flow");

    let base = result.graphs.get("base").expect("base graph");
    let copy = result.graphs.get("copy").expect("copy graph");

    assert_eq!(base.nodes().len(), copy.nodes().len());
    assert_eq!(base.edges().len(), copy.edges().len());
}

#[test]
fn run_flow_errors_on_bad_export() {
    let prog = ProgramAst {
        schemas: vec![],
        belief_models: vec![],
        evidences: vec![],
        rules: vec![],
        flows: vec![FlowDef {
            name: "BadExport".into(),
            on_model: "M".into(),
            graphs: vec![],
            metrics: vec![],
            exports: vec![ExportDef {
                graph: "missing".into(),
                alias: "output".into(),
            }],
            metric_exports: vec![],
            metric_imports: vec![],
        }],
    };

    let result = run_flow_with_builder(&prog, "BadExport", &evidence_builder, None);
    assert!(result.is_err());
}

#[test]
fn apply_ruleset_applies_rules_sequentially() {
    // Create two rules that modify the graph in sequence
    let rules = vec![
        RuleDef {
            name: "ForceFirst".into(),
            on_model: "M".into(),
            mode: Some("for_each".into()),
            patterns: vec![PatternItem {
                src: NodePattern {
                    var: "A".into(),
                    label: "Person".into(),
                },
                edge: EdgePattern {
                    var: "e".into(),
                    ty: "REL".into(),
                },
                dst: NodePattern {
                    var: "B".into(),
                    label: "Person".into(),
                },
            }],
            where_expr: Some(ExprAst::Binary {
                op: BinaryOp::Eq,
                left: Box::new(ExprAst::Var("A".into())),
                right: Box::new(ExprAst::Number(1.0)),
            }),
            actions: vec![ActionStmt::ForceAbsent {
                edge_var: "e".into(),
            }],
        },
        RuleDef {
            name: "ForceSecond".into(),
            on_model: "M".into(),
            mode: Some("for_each".into()),
            patterns: vec![PatternItem {
                src: NodePattern {
                    var: "A".into(),
                    label: "Person".into(),
                },
                edge: EdgePattern {
                    var: "e".into(),
                    ty: "REL".into(),
                },
                dst: NodePattern {
                    var: "B".into(),
                    label: "Person".into(),
                },
            }],
            where_expr: Some(ExprAst::Binary {
                op: BinaryOp::Eq,
                left: Box::new(ExprAst::Var("B".into())),
                right: Box::new(ExprAst::Number(3.0)),
            }),
            actions: vec![ActionStmt::ForceAbsent {
                edge_var: "e".into(),
            }],
        },
    ];

    fn evidence_builder(_: &EvidenceDef) -> Result<BeliefGraph, ExecError> {
        let mut g = BeliefGraph::default();
        // Create 3 nodes with edges: 1->2, 1->3, 2->3
        use std::sync::Arc;
        g.insert_node(NodeData {
            id: NodeId(1),
            label: Arc::from("Person"),
            attrs: HashMap::new(),
        });
        g.insert_node(NodeData {
            id: NodeId(2),
            label: Arc::from("Person"),
            attrs: HashMap::new(),
        });
        g.insert_node(NodeData {
            id: NodeId(3),
            label: Arc::from("Person"),
            attrs: HashMap::new(),
        });
        g.insert_edge(EdgeData {
            id: EdgeId(1),
            src: NodeId(1),
            dst: NodeId(2),
            ty: Arc::from("REL"),
            exist: EdgePosterior::independent(BetaPosterior {
                alpha: 5.0,
                beta: 5.0,
            }),
        });
        g.insert_edge(EdgeData {
            id: EdgeId(2),
            src: NodeId(1),
            dst: NodeId(3),
            ty: Arc::from("REL"),
            exist: EdgePosterior::independent(BetaPosterior {
                alpha: 5.0,
                beta: 5.0,
            }),
        });
        g.insert_edge(EdgeData {
            id: EdgeId(3),
            src: NodeId(2),
            dst: NodeId(3),
            ty: Arc::from("REL"),
            exist: EdgePosterior::independent(BetaPosterior {
                alpha: 5.0,
                beta: 5.0,
            }),
        });
        g.ensure_owned();
        Ok(g)
    }

    let flows = vec![FlowDef {
        name: "TestRuleset".into(),
        on_model: "M".into(),
        graphs: vec![
            GraphDef {
                name: "base".into(),
                expr: GraphExpr::FromEvidence {
                    evidence: "Ev".into(),
                },
            },
            GraphDef {
                name: "result".into(),
                expr: GraphExpr::Pipeline {
                    start: "base".into(),
                    transforms: vec![Transform::ApplyRuleset {
                        rules: vec!["ForceFirst".into(), "ForceSecond".into()],
                    }],
                },
            },
        ],
        metrics: vec![],
        exports: vec![ExportDef {
            graph: "result".into(),
            alias: "out".into(),
        }],
        metric_exports: vec![],
        metric_imports: vec![],
    }];

    let prog = ProgramAst {
        schemas: vec![],
        belief_models: vec![],
        evidences: vec![EvidenceDef {
            name: "Ev".into(),
            on_model: "M".into(),
            body_src: "".into(),
            observations: vec![],
        }],
        rules,
        flows,
    };

    let result =
        run_flow_with_builder(&prog, "TestRuleset", &evidence_builder, None).expect("run flow");
    let output = result.exports.get("out").expect("exported graph");

    // Both rules should have been applied:
    // - ForceFirst forces absent edges where src=1: 1->2 (EdgeId(1)) and 1->3 (EdgeId(2))
    // - ForceSecond forces absent edges where dst=3: 1->3 (EdgeId(2)) and 2->3 (EdgeId(3))
    // So all three edges should have low probability
    assert_eq!(output.edges().len(), 3); // All edges still exist
                                         // Check that all edges have low probability
    let prob_1_2 = output.prob_mean(EdgeId(1)).unwrap();
    let prob_2_3 = output.prob_mean(EdgeId(3)).unwrap();
    let prob_1_3 = output.prob_mean(EdgeId(2)).unwrap();
    assert!(prob_1_2 < 1e-5, "edge 1->2 should be forced absent");
    assert!(prob_2_3 < 1e-5, "edge 2->3 should be forced absent");
    assert!(
        prob_1_3 < 1e-5,
        "edge 1->3 should be forced absent by both rules"
    );
}

#[test]
fn apply_ruleset_with_single_rule_works() {
    let rules = vec![RuleDef {
        name: "SingleRule".into(),
        on_model: "M".into(),
        mode: Some("for_each".into()),
        patterns: vec![PatternItem {
            src: NodePattern {
                var: "A".into(),
                label: "Person".into(),
            },
            edge: EdgePattern {
                var: "e".into(),
                ty: "REL".into(),
            },
            dst: NodePattern {
                var: "B".into(),
                label: "Person".into(),
            },
        }],
        where_expr: None,
        actions: vec![ActionStmt::ForceAbsent {
            edge_var: "e".into(),
        }],
    }];

    fn evidence_builder(_: &EvidenceDef) -> Result<BeliefGraph, ExecError> {
        use std::sync::Arc;
        let mut g = BeliefGraph::default();
        g.insert_node(NodeData {
            id: NodeId(1),
            label: Arc::from("Person"),
            attrs: HashMap::new(),
        });
        g.insert_node(NodeData {
            id: NodeId(2),
            label: Arc::from("Person"),
            attrs: HashMap::new(),
        });
        g.insert_edge(EdgeData {
            id: EdgeId(1),
            src: NodeId(1),
            dst: NodeId(2),
            ty: Arc::from("REL"),
            exist: EdgePosterior::independent(BetaPosterior {
                alpha: 1.0,
                beta: 1.0,
            }),
        });
        g.ensure_owned();
        Ok(g)
    }

    let flows = vec![FlowDef {
        name: "TestSingle".into(),
        on_model: "M".into(),
        graphs: vec![
            GraphDef {
                name: "base".into(),
                expr: GraphExpr::FromEvidence {
                    evidence: "Ev".into(),
                },
            },
            GraphDef {
                name: "result".into(),
                expr: GraphExpr::Pipeline {
                    start: "base".into(),
                    transforms: vec![Transform::ApplyRuleset {
                        rules: vec!["SingleRule".into()],
                    }],
                },
            },
        ],
        metrics: vec![],
        exports: vec![],
        metric_exports: vec![],
        metric_imports: vec![],
    }];

    let prog = ProgramAst {
        schemas: vec![],
        belief_models: vec![],
        evidences: vec![EvidenceDef {
            name: "Ev".into(),
            on_model: "M".into(),
            body_src: "".into(),
            observations: vec![],
        }],
        rules,
        flows,
    };

    let result =
        run_flow_with_builder(&prog, "TestSingle", &evidence_builder, None).expect("run flow");
    let output = result.graphs.get("result").expect("result graph");

    // Rule should have been applied (force_absent sets probability near 0)
    assert_eq!(output.edges().len(), 1);
    // After force_absent, probability should be near zero
    let prob = output.prob_mean(EdgeId(1)).unwrap();
    assert!(prob < 1e-5, "edge should be forced absent");
}

#[test]
fn apply_ruleset_errors_on_unknown_rule() {
    let rules = vec![RuleDef {
        name: "KnownRule".into(),
        on_model: "M".into(),
        mode: Some("for_each".into()),
        patterns: vec![PatternItem {
            src: NodePattern {
                var: "A".into(),
                label: "Person".into(),
            },
            edge: EdgePattern {
                var: "e".into(),
                ty: "REL".into(),
            },
            dst: NodePattern {
                var: "B".into(),
                label: "Person".into(),
            },
        }],
        where_expr: None,
        actions: vec![],
    }];

    fn evidence_builder(_: &EvidenceDef) -> Result<BeliefGraph, ExecError> {
        Ok(BeliefGraph::default())
    }

    let flows = vec![FlowDef {
        name: "TestError".into(),
        on_model: "M".into(),
        graphs: vec![
            GraphDef {
                name: "base".into(),
                expr: GraphExpr::FromEvidence {
                    evidence: "Ev".into(),
                },
            },
            GraphDef {
                name: "result".into(),
                expr: GraphExpr::Pipeline {
                    start: "base".into(),
                    transforms: vec![Transform::ApplyRuleset {
                        rules: vec!["KnownRule".into(), "UnknownRule".into()],
                    }],
                },
            },
        ],
        metrics: vec![],
        exports: vec![],
        metric_exports: vec![],
        metric_imports: vec![],
    }];

    let prog = ProgramAst {
        schemas: vec![],
        belief_models: vec![],
        evidences: vec![EvidenceDef {
            name: "Ev".into(),
            on_model: "M".into(),
            body_src: "".into(),
            observations: vec![],
        }],
        rules,
        flows,
    };

    let result = run_flow_with_builder(&prog, "TestError", &evidence_builder, None);
    assert!(matches!(result, Err(ExecError::Internal(_))));
    if let Err(ExecError::Internal(msg)) = result {
        assert!(msg.contains("unknown rule 'UnknownRule' in ruleset"));
    }
}

#[test]
fn snapshot_transform_saves_graph_state() {
    let rules = vec![RuleDef {
        name: "ForceLowProb".into(),
        on_model: "M".into(),
        mode: Some("for_each".into()),
        patterns: vec![PatternItem {
            src: NodePattern {
                var: "A".into(),
                label: "Person".into(),
            },
            edge: EdgePattern {
                var: "e".into(),
                ty: "REL".into(),
            },
            dst: NodePattern {
                var: "B".into(),
                label: "Person".into(),
            },
        }],
        where_expr: Some(ExprAst::Binary {
            op: BinaryOp::Ge,
            left: Box::new(ExprAst::Call {
                name: "prob".into(),
                args: vec![CallArg::Positional(ExprAst::Var("e".into()))],
            }),
            right: Box::new(ExprAst::Number(0.5)),
        }),
        actions: vec![ActionStmt::ForceAbsent {
            edge_var: "e".into(),
        }],
    }];

    fn evidence_builder(_: &EvidenceDef) -> Result<BeliefGraph, ExecError> {
        use std::sync::Arc;
        let mut g = BeliefGraph::default();
        g.insert_node(NodeData {
            id: NodeId(1),
            label: Arc::from("Person"),
            attrs: HashMap::new(),
        });
        g.insert_node(NodeData {
            id: NodeId(2),
            label: Arc::from("Person"),
            attrs: HashMap::new(),
        });
        g.insert_edge(EdgeData {
            id: EdgeId(1),
            src: NodeId(1),
            dst: NodeId(2),
            ty: Arc::from("REL"),
            exist: EdgePosterior::independent(BetaPosterior {
                alpha: 8.0,
                beta: 2.0,
            }),
        });
        g.ensure_owned();
        Ok(g)
    }

    let flows = vec![FlowDef {
        name: "SnapshotTest".into(),
        on_model: "M".into(),
        graphs: vec![
            GraphDef {
                name: "base".into(),
                expr: GraphExpr::FromEvidence {
                    evidence: "Ev".into(),
                },
            },
            GraphDef {
                name: "cleaned".into(),
                expr: GraphExpr::Pipeline {
                    start: "base".into(),
                    transforms: vec![
                        Transform::Snapshot {
                            name: "before_cleanup".into(),
                        },
                        Transform::ApplyRule {
                            rule: "ForceLowProb".into(),
                        },
                        Transform::Snapshot {
                            name: "after_cleanup".into(),
                        },
                    ],
                },
            },
        ],
        metrics: vec![],
        exports: vec![],
        metric_exports: vec![],
        metric_imports: vec![],
    }];

    let prog = ProgramAst {
        schemas: vec![],
        belief_models: vec![],
        evidences: vec![EvidenceDef {
            name: "Ev".into(),
            on_model: "M".into(),
            body_src: "".into(),
            observations: vec![],
        }],
        rules,
        flows,
    };

    let result =
        run_flow_with_builder(&prog, "SnapshotTest", &evidence_builder, None).expect("run flow");

    // Verify snapshots were saved
    assert!(result.snapshots.contains_key("before_cleanup"));
    assert!(result.snapshots.contains_key("after_cleanup"));

    // Verify snapshot states differ
    let before = result.snapshots.get("before_cleanup").unwrap();
    let after = result.snapshots.get("after_cleanup").unwrap();

    // Before snapshot should have the edge with high probability
    assert_eq!(before.edges().len(), 1);
    // After snapshot should have the edge with low probability (force_absent was applied)
    assert_eq!(after.edges().len(), 1);

    // The edge probability should be lower after force_absent
    // Use prob_mean which is delta-aware
    let before_prob = before.prob_mean(EdgeId(1)).unwrap();
    let after_prob = after.prob_mean(EdgeId(1)).unwrap();
    assert!(
        after_prob < before_prob,
        "after_prob ({}) should be less than before_prob ({})",
        after_prob,
        before_prob
    );
}

#[test]
fn snapshot_multiple_snapshots_in_pipeline() {
    fn evidence_builder(_: &EvidenceDef) -> Result<BeliefGraph, ExecError> {
        use std::sync::Arc;
        let mut g = BeliefGraph::default();
        g.insert_node(NodeData {
            id: NodeId(1),
            label: Arc::from("Person"),
            attrs: HashMap::new(),
        });
        g.insert_node(NodeData {
            id: NodeId(2),
            label: Arc::from("Person"),
            attrs: HashMap::new(),
        });
        g.insert_edge(EdgeData {
            id: EdgeId(1),
            src: NodeId(1),
            dst: NodeId(2),
            ty: Arc::from("REL"),
            exist: EdgePosterior::independent(BetaPosterior {
                alpha: 5.0,
                beta: 5.0,
            }),
        });
        g.ensure_owned();
        Ok(g)
    }

    let flows = vec![FlowDef {
        name: "MultiSnapshot".into(),
        on_model: "M".into(),
        graphs: vec![
            GraphDef {
                name: "base".into(),
                expr: GraphExpr::FromEvidence {
                    evidence: "Ev".into(),
                },
            },
            GraphDef {
                name: "result".into(),
                expr: GraphExpr::Pipeline {
                    start: "base".into(),
                    transforms: vec![
                        Transform::Snapshot {
                            name: "step1".into(),
                        },
                        Transform::Snapshot {
                            name: "step2".into(),
                        },
                        Transform::Snapshot {
                            name: "step3".into(),
                        },
                    ],
                },
            },
        ],
        metrics: vec![],
        exports: vec![],
        metric_exports: vec![],
        metric_imports: vec![],
    }];

    let prog = ProgramAst {
        schemas: vec![],
        belief_models: vec![],
        evidences: vec![EvidenceDef {
            name: "Ev".into(),
            on_model: "M".into(),
            body_src: "".into(),
            observations: vec![],
        }],
        rules: vec![],
        flows,
    };

    let result =
        run_flow_with_builder(&prog, "MultiSnapshot", &evidence_builder, None).expect("run flow");

    // All three snapshots should be saved
    assert!(result.snapshots.contains_key("step1"));
    assert!(result.snapshots.contains_key("step2"));
    assert!(result.snapshots.contains_key("step3"));

    // All snapshots should have the same graph state (no transforms between them)
    let step1 = result.snapshots.get("step1").unwrap();
    let step2 = result.snapshots.get("step2").unwrap();
    let step3 = result.snapshots.get("step3").unwrap();

    assert_eq!(step1.edges().len(), step2.edges().len());
    assert_eq!(step2.edges().len(), step3.edges().len());
}

#[test]
fn snapshot_overwrites_previous_snapshot_with_same_name() {
    fn evidence_builder(_: &EvidenceDef) -> Result<BeliefGraph, ExecError> {
        let mut g = BeliefGraph::default();
        g.insert_node(NodeData {
            id: NodeId(1),
            label: "Person".into(),
            attrs: HashMap::new(),
        });
        g.ensure_owned();
        Ok(g)
    }

    let flows = vec![FlowDef {
        name: "OverwriteSnapshot".into(),
        on_model: "M".into(),
        graphs: vec![
            GraphDef {
                name: "base".into(),
                expr: GraphExpr::FromEvidence {
                    evidence: "Ev".into(),
                },
            },
            GraphDef {
                name: "result".into(),
                expr: GraphExpr::Pipeline {
                    start: "base".into(),
                    transforms: vec![
                        Transform::Snapshot {
                            name: "checkpoint".into(),
                        },
                        Transform::Snapshot {
                            name: "checkpoint".into(),
                        },
                    ],
                },
            },
        ],
        metrics: vec![],
        exports: vec![],
        metric_exports: vec![],
        metric_imports: vec![],
    }];

    let prog = ProgramAst {
        schemas: vec![],
        belief_models: vec![],
        evidences: vec![EvidenceDef {
            name: "Ev".into(),
            on_model: "M".into(),
            body_src: "".into(),
            observations: vec![],
        }],
        rules: vec![],
        flows,
    };

    let result = run_flow_with_builder(&prog, "OverwriteSnapshot", &evidence_builder, None)
        .expect("run flow");

    // Only one snapshot should exist (the second one overwrites the first)
    assert_eq!(result.snapshots.len(), 1);
    assert!(result.snapshots.contains_key("checkpoint"));
}

#[test]
fn from_graph_imports_from_prior_flow_exports() {
    fn evidence_builder(_: &EvidenceDef) -> Result<BeliefGraph, ExecError> {
        use std::sync::Arc;
        let mut g = BeliefGraph::default();
        g.insert_node(NodeData {
            id: NodeId(1),
            label: Arc::from("Person"),
            attrs: HashMap::new(),
        });
        g.insert_node(NodeData {
            id: NodeId(2),
            label: Arc::from("Person"),
            attrs: HashMap::new(),
        });
        g.insert_edge(EdgeData {
            id: EdgeId(1),
            src: NodeId(1),
            dst: NodeId(2),
            ty: Arc::from("REL"),
            exist: EdgePosterior::independent(BetaPosterior {
                alpha: 5.0,
                beta: 5.0,
            }),
        });
        g.ensure_owned();
        Ok(g)
    }

    // First flow: exports a graph
    let flows = vec![
        FlowDef {
            name: "Producer".into(),
            on_model: "M".into(),
            graphs: vec![GraphDef {
                name: "base".into(),
                expr: GraphExpr::FromEvidence {
                    evidence: "Ev".into(),
                },
            }],
            metrics: vec![],
            exports: vec![ExportDef {
                graph: "base".into(),
                alias: "exported_graph".into(),
            }],
            metric_exports: vec![],
            metric_imports: vec![],
        },
        FlowDef {
            name: "Consumer".into(),
            on_model: "M".into(),
            graphs: vec![GraphDef {
                name: "imported".into(),
                expr: GraphExpr::FromGraph {
                    alias: "exported_graph".into(),
                },
            }],
            metrics: vec![],
            exports: vec![],
            metric_exports: vec![],
            metric_imports: vec![],
        },
    ];

    let prog = ProgramAst {
        schemas: vec![],
        belief_models: vec![],
        evidences: vec![EvidenceDef {
            name: "Ev".into(),
            on_model: "M".into(),
            body_src: "".into(),
            observations: vec![],
        }],
        rules: vec![],
        flows,
    };

    // Run first flow
    let prior = run_flow_with_builder(&prog, "Producer", &evidence_builder, None)
        .expect("run producer flow");

    // Run second flow with prior result
    let result = run_flow_with_builder(&prog, "Consumer", &evidence_builder, Some(&prior))
        .expect("run consumer flow");

    // Verify imported graph matches exported graph
    let imported = result.graphs.get("imported").expect("imported graph");
    let exported = prior.exports.get("exported_graph").expect("exported graph");

    assert_eq!(imported.nodes().len(), exported.nodes().len());
    assert_eq!(imported.edges().len(), exported.edges().len());
}

#[test]
fn from_graph_imports_from_prior_flow_snapshots() {
    fn evidence_builder(_: &EvidenceDef) -> Result<BeliefGraph, ExecError> {
        use std::sync::Arc;
        let mut g = BeliefGraph::default();
        g.insert_node(NodeData {
            id: NodeId(1),
            label: Arc::from("Person"),
            attrs: HashMap::new(),
        });
        g.insert_node(NodeData {
            id: NodeId(2),
            label: Arc::from("Person"),
            attrs: HashMap::new(),
        });
        g.insert_edge(EdgeData {
            id: EdgeId(1),
            src: NodeId(1),
            dst: NodeId(2),
            ty: Arc::from("REL"),
            exist: EdgePosterior::independent(BetaPosterior {
                alpha: 5.0,
                beta: 5.0,
            }),
        });
        g.ensure_owned();
        Ok(g)
    }

    // First flow: creates a snapshot
    let flows = vec![
        FlowDef {
            name: "Producer".into(),
            on_model: "M".into(),
            graphs: vec![
                GraphDef {
                    name: "base".into(),
                    expr: GraphExpr::FromEvidence {
                        evidence: "Ev".into(),
                    },
                },
                GraphDef {
                    name: "result".into(),
                    expr: GraphExpr::Pipeline {
                        start: "base".into(),
                        transforms: vec![Transform::Snapshot {
                            name: "checkpoint".into(),
                        }],
                    },
                },
            ],
            metrics: vec![],
            exports: vec![],
            metric_exports: vec![],
            metric_imports: vec![],
        },
        FlowDef {
            name: "Consumer".into(),
            on_model: "M".into(),
            graphs: vec![GraphDef {
                name: "imported".into(),
                expr: GraphExpr::FromGraph {
                    alias: "checkpoint".into(),
                },
            }],
            metrics: vec![],
            exports: vec![],
            metric_exports: vec![],
            metric_imports: vec![],
        },
    ];

    let prog = ProgramAst {
        schemas: vec![],
        belief_models: vec![],
        evidences: vec![EvidenceDef {
            name: "Ev".into(),
            on_model: "M".into(),
            body_src: "".into(),
            observations: vec![],
        }],
        rules: vec![],
        flows,
    };

    // Run first flow
    let prior = run_flow_with_builder(&prog, "Producer", &evidence_builder, None)
        .expect("run producer flow");

    // Run second flow with prior result
    let result = run_flow_with_builder(&prog, "Consumer", &evidence_builder, Some(&prior))
        .expect("run consumer flow");

    // Verify imported graph matches snapshot
    let imported = result.graphs.get("imported").expect("imported graph");
    let snapshot = prior.snapshots.get("checkpoint").expect("snapshot");

    assert_eq!(imported.nodes().len(), snapshot.nodes().len());
    assert_eq!(imported.edges().len(), snapshot.edges().len());
}

#[test]
fn from_graph_errors_on_missing_graph() {
    fn evidence_builder(_: &EvidenceDef) -> Result<BeliefGraph, ExecError> {
        Ok(BeliefGraph::default())
    }

    // Flow that tries to import a non-existent graph
    let flows = vec![FlowDef {
        name: "Consumer".into(),
        on_model: "M".into(),
        graphs: vec![GraphDef {
            name: "imported".into(),
            expr: GraphExpr::FromGraph {
                alias: "missing_graph".into(),
            },
        }],
        metrics: vec![],
        exports: vec![],
        metric_exports: vec![],
        metric_imports: vec![],
    }];

    let prog = ProgramAst {
        schemas: vec![],
        belief_models: vec![],
        evidences: vec![EvidenceDef {
            name: "Ev".into(),
            on_model: "M".into(),
            body_src: "".into(),
            observations: vec![],
        }],
        rules: vec![],
        flows,
    };

    // Run flow with empty prior (no exports or snapshots)
    let prior = FlowResult::default();
    let result = run_flow_with_builder(&prog, "Consumer", &evidence_builder, Some(&prior));

    // Should error because graph doesn't exist
    assert!(matches!(result, Err(ExecError::Internal(_))));
    if let Err(ExecError::Internal(msg)) = result {
        assert!(msg.contains("graph 'missing_graph' not found"));
    }
}

#[test]
fn from_graph_prefers_exports_over_snapshots() {
    // If both exports and snapshots have the same name, exports should take precedence
    fn evidence_builder(_: &EvidenceDef) -> Result<BeliefGraph, ExecError> {
        use std::sync::Arc;
        let mut g1 = BeliefGraph::default();
        g1.insert_node(NodeData {
            id: NodeId(1),
            label: Arc::from("Person"),
            attrs: HashMap::new(),
        });
        g1.ensure_owned();

        // Return different graphs so we can tell which one was used
        Ok(g1)
    }

    let flows = vec![
        FlowDef {
            name: "Producer".into(),
            on_model: "M".into(),
            graphs: vec![
                GraphDef {
                    name: "export_graph".into(),
                    expr: GraphExpr::FromEvidence {
                        evidence: "Ev".into(),
                    },
                },
                GraphDef {
                    name: "snapshot_graph".into(),
                    expr: GraphExpr::Pipeline {
                        start: "export_graph".into(),
                        transforms: vec![Transform::Snapshot {
                            name: "shared_name".into(),
                        }],
                    },
                },
            ],
            metrics: vec![],
            exports: vec![ExportDef {
                graph: "export_graph".into(),
                alias: "shared_name".into(),
            }],
            metric_exports: vec![],
            metric_imports: vec![],
        },
        FlowDef {
            name: "Consumer".into(),
            on_model: "M".into(),
            graphs: vec![GraphDef {
                name: "imported".into(),
                expr: GraphExpr::FromGraph {
                    alias: "shared_name".into(),
                },
            }],
            metrics: vec![],
            exports: vec![],
            metric_exports: vec![],
            metric_imports: vec![],
        },
    ];

    let prog = ProgramAst {
        schemas: vec![],
        belief_models: vec![],
        evidences: vec![EvidenceDef {
            name: "Ev".into(),
            on_model: "M".into(),
            body_src: "".into(),
            observations: vec![],
        }],
        rules: vec![],
        flows,
    };

    // Run first flow
    let prior = run_flow_with_builder(&prog, "Producer", &evidence_builder, None)
        .expect("run producer flow");

    // Verify both export and snapshot exist with same name
    assert!(prior.exports.contains_key("shared_name"));
    assert!(prior.snapshots.contains_key("shared_name"));

    // Run second flow
    let result = run_flow_with_builder(&prog, "Consumer", &evidence_builder, Some(&prior))
        .expect("run consumer flow");

    // Verify imported graph matches the exported graph (not the snapshot)
    let imported = result.graphs.get("imported").expect("imported graph");
    let exported = prior.exports.get("shared_name").expect("exported graph");

    assert_eq!(imported.nodes().len(), exported.nodes().len());
}
