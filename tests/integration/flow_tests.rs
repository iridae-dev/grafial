use std::collections::HashMap;

use baygraph::engine::errors::ExecError;
use baygraph::engine::flow_exec::{run_flow, FlowResult};
use baygraph::engine::graph::{BeliefGraph, BetaPosterior, EdgeData, GaussianPosterior, NodeData, NodeId, EdgeId};
use baygraph::frontend::ast::*;

fn build_test_program() -> ProgramAst {
    // schema + model names are carried only for clarity in this phase
    let rules = vec![RuleDef {
        name: "ForceLowProb".into(),
        on_model: "M".into(),
        mode: Some("for_each".into()),
        patterns: vec![PatternItem {
            src: NodePattern { var: "A".into(), label: "Person".into() },
            edge: EdgePattern { var: "e".into(), ty: "REL".into() },
            dst: NodePattern { var: "B".into(), label: "Person".into() },
        }],
        // where prob(e) >= 0.5
        where_expr: Some(ExprAst::Binary {
            op: BinaryOp::Ge,
            left: Box::new(ExprAst::Call { name: "prob".into(), args: vec![CallArg::Positional(ExprAst::Var("e".into()))] }),
            right: Box::new(ExprAst::Number(0.5)),
        }),
        // action: force_absent e
        actions: vec![ActionStmt::ForceAbsent { edge_var: "e".into() }],
    }];

    let flows = vec![FlowDef {
        name: "Demo".into(),
        on_model: "M".into(),
        graphs: vec![
            GraphDef { name: "base".into(), expr: GraphExpr::FromEvidence { evidence: "Ev".into() } },
            GraphDef {
                name: "cleaned".into(),
                expr: GraphExpr::Pipeline {
                    start: "base".into(),
                    transforms: vec![
                        Transform::ApplyRule { rule: "ForceLowProb".into() },
                        Transform::PruneEdges {
                            edge_type: "REL".into(),
                            predicate: ExprAst::Binary {
                                op: BinaryOp::Lt,
                                left: Box::new(ExprAst::Call { name: "prob".into(), args: vec![CallArg::Positional(ExprAst::Var("edge".into()))] }),
                                right: Box::new(ExprAst::Number(0.1)),
                            },
                        },
                    ],
                },
            },
        ],
        metrics: vec![],
        exports: vec![ExportDef { graph: "cleaned".into(), alias: "demo".into() }],
    }];

    ProgramAst {
        schemas: vec![],
        belief_models: vec![],
        evidences: vec![EvidenceDef { name: "Ev".into(), on_model: "M".into(), body_src: "".into() }],
        rules,
        flows,
    }
}

fn evidence_builder(_: &EvidenceDef) -> Result<BeliefGraph, ExecError> {
    // Build a tiny graph: Person(1) -[REL]-> Person(2) with p=0.8
    let mut g = BeliefGraph::default();
    g.insert_node(NodeData { id: NodeId(1), label: "Person".into(), attrs: HashMap::from([
        ("some_value".into(), GaussianPosterior { mean: 10.0, precision: 1.0 }),
    ]) });
    g.insert_node(NodeData { id: NodeId(2), label: "Person".into(), attrs: HashMap::from([
        ("some_value".into(), GaussianPosterior { mean: 0.0, precision: 1.0 }),
    ]) });
    g.insert_edge(EdgeData { id: EdgeId(1), src: NodeId(1), dst: NodeId(2), ty: "REL".into(), exist: BetaPosterior { alpha: 8.0, beta: 2.0 } });
    Ok(g)
}

#[test]
fn run_flow_demo_applies_rule_and_prunes() {
    let prog = build_test_program();
    let result: FlowResult = run_flow(&prog, "Demo", &evidence_builder).expect("run flow");

    // base graph should have one edge
    let base = result.graphs.get("base").expect("base graph");
    assert_eq!(base.edges.len(), 1);

    // cleaned graph exported as "demo" should have zero edges after force_absent + prune
    let exported = result.exports.get("demo").expect("exported graph");
    assert_eq!(exported.edges.len(), 0);
}

#[test]
fn run_flow_from_evidence_loads_graph() {
    let prog = ProgramAst {
        schemas: vec![],
        belief_models: vec![],
        evidences: vec![EvidenceDef { name: "Ev".into(), on_model: "M".into(), body_src: "".into() }],
        rules: vec![],
        flows: vec![FlowDef {
            name: "Simple".into(),
            on_model: "M".into(),
            graphs: vec![GraphDef { name: "g".into(), expr: GraphExpr::FromEvidence { evidence: "Ev".into() } }],
            metrics: vec![],
            exports: vec![],
        }],
    };

    let result = run_flow(&prog, "Simple", &evidence_builder).expect("run flow");
    let g = result.graphs.get("g").expect("graph g");
    assert_eq!(g.nodes.len(), 2);
    assert_eq!(g.edges.len(), 1);
}

#[test]
fn run_flow_unknown_flow_fails() {
    let prog = build_test_program();
    let result = run_flow(&prog, "NonExistent", &evidence_builder);
    assert!(result.is_err());
}

#[test]
fn run_flow_unknown_evidence_fails() {
    let prog = ProgramAst {
        schemas: vec![],
        belief_models: vec![],
        evidences: vec![],
        rules: vec![],
        flows: vec![FlowDef {
            name: "BadFlow".into(),
            on_model: "M".into(),
            graphs: vec![GraphDef { name: "g".into(), expr: GraphExpr::FromEvidence { evidence: "Missing".into() } }],
            metrics: vec![],
            exports: vec![],
        }],
    };

    let result = run_flow(&prog, "BadFlow", &evidence_builder);
    assert!(result.is_err());
}

#[test]
fn run_flow_unknown_start_graph_fails() {
    let prog = ProgramAst {
        schemas: vec![],
        belief_models: vec![],
        evidences: vec![],
        rules: vec![],
        flows: vec![FlowDef {
            name: "BadPipeline".into(),
            on_model: "M".into(),
            graphs: vec![GraphDef {
                name: "g".into(),
                expr: GraphExpr::Pipeline {
                    start: "nonexistent".into(),
                    transforms: vec![],
                },
            }],
            metrics: vec![],
            exports: vec![],
        }],
    };

    let result = run_flow(&prog, "BadPipeline", &evidence_builder);
    assert!(result.is_err());
}

#[test]
fn run_flow_unknown_rule_fails() {
    let prog = ProgramAst {
        schemas: vec![],
        belief_models: vec![],
        evidences: vec![EvidenceDef { name: "Ev".into(), on_model: "M".into(), body_src: "".into() }],
        rules: vec![],
        flows: vec![FlowDef {
            name: "BadRule".into(),
            on_model: "M".into(),
            graphs: vec![
                GraphDef { name: "base".into(), expr: GraphExpr::FromEvidence { evidence: "Ev".into() } },
                GraphDef {
                    name: "transformed".into(),
                    expr: GraphExpr::Pipeline {
                        start: "base".into(),
                        transforms: vec![Transform::ApplyRule { rule: "NonExistentRule".into() }],
                    },
                },
            ],
            metrics: vec![],
            exports: vec![],
        }],
    };

    let result = run_flow(&prog, "BadRule", &evidence_builder);
    assert!(result.is_err());
}

#[test]
fn run_flow_export_binds_alias() {
    let prog = ProgramAst {
        schemas: vec![],
        belief_models: vec![],
        evidences: vec![EvidenceDef { name: "Ev".into(), on_model: "M".into(), body_src: "".into() }],
        rules: vec![],
        flows: vec![FlowDef {
            name: "ExportTest".into(),
            on_model: "M".into(),
            graphs: vec![GraphDef { name: "g".into(), expr: GraphExpr::FromEvidence { evidence: "Ev".into() } }],
            metrics: vec![],
            exports: vec![ExportDef { graph: "g".into(), alias: "output".into() }],
        }],
    };

    let result = run_flow(&prog, "ExportTest", &evidence_builder).expect("run flow");
    assert!(result.exports.contains_key("output"));
    assert_eq!(result.exports.get("output").unwrap().nodes.len(), 2);
}

#[test]
fn run_flow_export_unknown_graph_fails() {
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
            exports: vec![ExportDef { graph: "missing".into(), alias: "output".into() }],
        }],
    };

    let result = run_flow(&prog, "BadExport", &evidence_builder);
    assert!(result.is_err());
}

#[test]
fn run_flow_multiple_transforms_in_pipeline() {
    // Build a graph with 3 edges: high prob, medium prob, low prob
    fn multi_edge_builder(_: &EvidenceDef) -> Result<BeliefGraph, ExecError> {
        let mut g = BeliefGraph::default();
        g.insert_node(NodeData {
            id: NodeId(1),
            label: "Node".into(),
            attrs: HashMap::new(),
        });
        g.insert_node(NodeData {
            id: NodeId(2),
            label: "Node".into(),
            attrs: HashMap::new(),
        });
        g.insert_node(NodeData {
            id: NodeId(3),
            label: "Node".into(),
            attrs: HashMap::new(),
        });
        // High probability edge
        g.insert_edge(EdgeData {
            id: EdgeId(1),
            src: NodeId(1),
            dst: NodeId(2),
            ty: "LINK".into(),
            exist: BetaPosterior { alpha: 9.0, beta: 1.0 },
        });
        // Medium probability edge
        g.insert_edge(EdgeData {
            id: EdgeId(2),
            src: NodeId(2),
            dst: NodeId(3),
            ty: "LINK".into(),
            exist: BetaPosterior { alpha: 5.0, beta: 5.0 },
        });
        // Low probability edge
        g.insert_edge(EdgeData {
            id: EdgeId(3),
            src: NodeId(1),
            dst: NodeId(3),
            ty: "LINK".into(),
            exist: BetaPosterior { alpha: 1.0, beta: 9.0 },
        });
        Ok(g)
    }

    let prog = ProgramAst {
        schemas: vec![],
        belief_models: vec![],
        evidences: vec![EvidenceDef { name: "Ev".into(), on_model: "M".into(), body_src: "".into() }],
        rules: vec![],
        flows: vec![FlowDef {
            name: "MultiTransform".into(),
            on_model: "M".into(),
            graphs: vec![
                GraphDef { name: "base".into(), expr: GraphExpr::FromEvidence { evidence: "Ev".into() } },
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
                                        args: vec![CallArg::Positional(ExprAst::Var("edge".into()))],
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
                                        args: vec![CallArg::Positional(ExprAst::Var("edge".into()))],
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
        }],
    };

    let result = run_flow(&prog, "MultiTransform", &multi_edge_builder).expect("run flow");

    // Base should have all 3 edges
    let base = result.graphs.get("base").expect("base graph");
    assert_eq!(base.edges.len(), 3);

    // Filtered should only have the medium prob edge (0.5)
    let filtered = result.graphs.get("filtered").expect("filtered graph");
    assert_eq!(filtered.edges.len(), 1);
    assert_eq!(filtered.edges[0].id, EdgeId(2));
}

#[test]
fn run_flow_pipeline_preserves_nodes() {
    let prog = build_test_program();
    let result = run_flow(&prog, "Demo", &evidence_builder).expect("run flow");

    let base = result.graphs.get("base").expect("base graph");
    let cleaned = result.graphs.get("cleaned").expect("cleaned graph");

    // Nodes should be preserved through pipeline
    assert_eq!(base.nodes.len(), cleaned.nodes.len());
    assert_eq!(cleaned.nodes.len(), 2);
}

#[test]
fn run_flow_empty_pipeline_clones_graph() {
    let prog = ProgramAst {
        schemas: vec![],
        belief_models: vec![],
        evidences: vec![EvidenceDef { name: "Ev".into(), on_model: "M".into(), body_src: "".into() }],
        rules: vec![],
        flows: vec![FlowDef {
            name: "Clone".into(),
            on_model: "M".into(),
            graphs: vec![
                GraphDef { name: "base".into(), expr: GraphExpr::FromEvidence { evidence: "Ev".into() } },
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
        }],
    };

    let result = run_flow(&prog, "Clone", &evidence_builder).expect("run flow");

    let base = result.graphs.get("base").expect("base graph");
    let copy = result.graphs.get("copy").expect("copy graph");

    assert_eq!(base.nodes.len(), copy.nodes.len());
    assert_eq!(base.edges.len(), copy.edges.len());
}

