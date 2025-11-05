//! Flow IR for Baygraph
//!
//! This IR closely follows the AST shapes but decouples execution from
//! parsing structures. It is intentionally minimal for Phase 4.

use crate::frontend::ast::ExprAst;

/// A graph expression IR.
#[derive(Debug, Clone, PartialEq)]
pub enum GraphExprIR {
    /// Build a graph from named evidence.
    FromEvidence(String),
    /// Import a graph exported from another flow.
    FromGraph(String),
    /// Start from an existing named graph within the flow and apply transforms.
    Pipeline { start_graph: String, transforms: Vec<TransformIR> },
}

/// Transform IR variants in flow pipelines.
#[derive(Debug, Clone, PartialEq)]
pub enum TransformIR {
    /// Apply a rule by name (mode may be overridden in future phases).
    ApplyRule { rule: String, mode_override: Option<String> },
    /// Apply multiple rules sequentially.
    ApplyRuleset { rules: Vec<String> },
    /// Save a snapshot of the current graph state.
    Snapshot { name: String },
    /// Prune edges of a given type based on a predicate expression.
    PruneEdges { edge_type: String, predicate: ExprAst },
}

/// A named graph definition in the flow IR.
#[derive(Debug, Clone, PartialEq)]
pub struct GraphDefIR {
    pub name: String,
    pub expr: GraphExprIR,
}

/// A lowered flow in IR form.
#[derive(Debug, Clone, PartialEq)]
pub struct FlowIR {
    pub name: String,
    pub on_model: String,
    pub graphs: Vec<GraphDefIR>,
}

impl From<&crate::frontend::ast::FlowDef> for FlowIR {
    fn from(f: &crate::frontend::ast::FlowDef) -> Self {
        let graphs = f
            .graphs
            .iter()
            .map(|g| GraphDefIR {
                name: g.name.clone(),
                expr: match &g.expr {
                    crate::frontend::ast::GraphExpr::FromEvidence { evidence } => {
                        GraphExprIR::FromEvidence(evidence.clone())
                    }
                    crate::frontend::ast::GraphExpr::FromGraph { alias } => {
                        GraphExprIR::FromGraph(alias.clone())
                    }
                    crate::frontend::ast::GraphExpr::Pipeline { start, transforms } => {
                        let ts = transforms
                            .iter()
                            .map(|t| match t {
                                crate::frontend::ast::Transform::ApplyRule { rule } => {
                                    TransformIR::ApplyRule { rule: rule.clone(), mode_override: None }
                                }
                                crate::frontend::ast::Transform::ApplyRuleset { rules } => {
                                    TransformIR::ApplyRuleset { rules: rules.clone() }
                                }
                                crate::frontend::ast::Transform::Snapshot { name } => {
                                    TransformIR::Snapshot { name: name.clone() }
                                }
                                crate::frontend::ast::Transform::PruneEdges { edge_type, predicate } => {
                                    TransformIR::PruneEdges { edge_type: edge_type.clone(), predicate: predicate.clone() }
                                }
                            })
                            .collect();
                        GraphExprIR::Pipeline { start_graph: start.clone(), transforms: ts }
                    }
                },
            })
            .collect();

        FlowIR { name: f.name.clone(), on_model: f.on_model.clone(), graphs }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::frontend::ast::*;

    #[test]
    fn flow_ir_from_evidence_conversion() {
        let flow_def = FlowDef {
            name: "TestFlow".into(),
            on_model: "TestModel".into(),
            graphs: vec![GraphDef {
                name: "g".into(),
                expr: GraphExpr::FromEvidence {
                    evidence: "Ev1".into(),
                },
            }],
            metrics: vec![],
            exports: vec![],
            metric_exports: vec![],
            metric_imports: vec![],
        };

        let ir = FlowIR::from(&flow_def);

        assert_eq!(ir.name, "TestFlow");
        assert_eq!(ir.on_model, "TestModel");
        assert_eq!(ir.graphs.len(), 1);
        assert_eq!(ir.graphs[0].name, "g");
        assert!(matches!(
            &ir.graphs[0].expr,
            GraphExprIR::FromEvidence(ev) if ev == "Ev1"
        ));
    }

    #[test]
    fn flow_ir_pipeline_conversion() {
        let flow_def = FlowDef {
            name: "PipelineFlow".into(),
            on_model: "Model".into(),
            graphs: vec![GraphDef {
                name: "result".into(),
                expr: GraphExpr::Pipeline {
                    start: "base".into(),
                    transforms: vec![
                        Transform::ApplyRule {
                            rule: "Rule1".into(),
                        },
                        Transform::PruneEdges {
                            edge_type: "LINK".into(),
                            predicate: ExprAst::Bool(true),
                        },
                    ],
                },
            }],
            metrics: vec![],
            exports: vec![],
            metric_exports: vec![],
            metric_imports: vec![],
        };

        let ir = FlowIR::from(&flow_def);

        assert_eq!(ir.graphs.len(), 1);
        assert_eq!(ir.graphs[0].name, "result");

        if let GraphExprIR::Pipeline { start_graph, transforms } = &ir.graphs[0].expr {
            assert_eq!(start_graph, "base");
            assert_eq!(transforms.len(), 2);

            assert!(matches!(
                &transforms[0],
                TransformIR::ApplyRule { rule, mode_override } if rule == "Rule1" && mode_override.is_none()
            ));

            assert!(matches!(
                &transforms[1],
                TransformIR::PruneEdges { edge_type, .. } if edge_type == "LINK"
            ));
        } else {
            panic!("Expected Pipeline expression");
        }
    }

    #[test]
    fn flow_ir_multiple_graphs() {
        let flow_def = FlowDef {
            name: "MultiGraph".into(),
            on_model: "Model".into(),
            graphs: vec![
                GraphDef {
                    name: "g1".into(),
                    expr: GraphExpr::FromEvidence {
                        evidence: "Ev1".into(),
                    },
                },
                GraphDef {
                    name: "g2".into(),
                    expr: GraphExpr::FromEvidence {
                        evidence: "Ev2".into(),
                    },
                },
            ],
            metrics: vec![],
            exports: vec![],
            metric_exports: vec![],
            metric_imports: vec![],
        };

        let ir = FlowIR::from(&flow_def);

        assert_eq!(ir.graphs.len(), 2);
        assert_eq!(ir.graphs[0].name, "g1");
        assert_eq!(ir.graphs[1].name, "g2");
    }

    #[test]
    fn transform_ir_apply_rule_has_none_mode_override() {
        let transform = Transform::ApplyRule {
            rule: "TestRule".into(),
        };

        let flow_def = FlowDef {
            name: "Test".into(),
            on_model: "M".into(),
            graphs: vec![GraphDef {
                name: "g".into(),
                expr: GraphExpr::Pipeline {
                    start: "base".into(),
                    transforms: vec![transform],
                },
            }],
            metrics: vec![],
            exports: vec![],
            metric_exports: vec![],
            metric_imports: vec![],
        };

        let ir = FlowIR::from(&flow_def);

        if let GraphExprIR::Pipeline { transforms, .. } = &ir.graphs[0].expr {
            if let TransformIR::ApplyRule { mode_override, .. } = &transforms[0] {
                assert!(mode_override.is_none());
            } else {
                panic!("Expected ApplyRule transform");
            }
        } else {
            panic!("Expected Pipeline");
        }
    }

    #[test]
    fn transform_ir_prune_edges_preserves_predicate() {
        let predicate = ExprAst::Binary {
            op: BinaryOp::Lt,
            left: Box::new(ExprAst::Number(1.0)),
            right: Box::new(ExprAst::Number(2.0)),
        };

        let flow_def = FlowDef {
            name: "Test".into(),
            on_model: "M".into(),
            graphs: vec![GraphDef {
                name: "g".into(),
                expr: GraphExpr::Pipeline {
                    start: "base".into(),
                    transforms: vec![Transform::PruneEdges {
                        edge_type: "LINK".into(),
                        predicate: predicate.clone(),
                    }],
                },
            }],
            metrics: vec![],
            exports: vec![],
            metric_exports: vec![],
            metric_imports: vec![],
        };

        let ir = FlowIR::from(&flow_def);

        if let GraphExprIR::Pipeline { transforms, .. } = &ir.graphs[0].expr {
            if let TransformIR::PruneEdges { edge_type, predicate: ir_pred } = &transforms[0] {
                assert_eq!(edge_type, "LINK");
                assert_eq!(ir_pred, &predicate);
            } else {
                panic!("Expected PruneEdges transform");
            }
        } else {
            panic!("Expected Pipeline");
        }
    }

    #[test]
    fn graph_def_ir_clone_works() {
        let def = GraphDefIR {
            name: "test".into(),
            expr: GraphExprIR::FromEvidence("ev".into()),
        };

        let cloned = def.clone();
        assert_eq!(def.name, cloned.name);
        assert_eq!(def.expr, cloned.expr);
    }

    #[test]
    fn flow_ir_clone_works() {
        let flow = FlowIR {
            name: "test".into(),
            on_model: "model".into(),
            graphs: vec![GraphDefIR {
                name: "g".into(),
                expr: GraphExprIR::FromEvidence("ev".into()),
            }],
        };

        let cloned = flow.clone();
        assert_eq!(flow.name, cloned.name);
        assert_eq!(flow.on_model, cloned.on_model);
        assert_eq!(flow.graphs.len(), cloned.graphs.len());
    }
}
