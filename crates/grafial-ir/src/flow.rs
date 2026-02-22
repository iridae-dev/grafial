//! Flow IR for Grafial.
//!
//! This IR captures graph expressions, transforms, and metric/export surfaces
//! with expression nodes normalized to `ExprIR`.

use crate::expr::ExprIR;
use grafial_frontend::ast::{
    ExportDef, FlowDef, GraphExpr, MetricDef, MetricExportDef, MetricImportDef, Transform,
};

/// A graph expression IR.
#[derive(Debug, Clone, PartialEq)]
pub enum GraphExprIR {
    /// Build a graph from named evidence.
    FromEvidence(String),
    /// Import a graph exported from another flow.
    FromGraph(String),
    /// Start from an existing named graph and apply transforms.
    Pipeline {
        start_graph: String,
        transforms: Vec<TransformIR>,
    },
}

impl GraphExprIR {
    /// Convert this IR graph expression back to frontend AST.
    pub fn to_ast(&self) -> GraphExpr {
        match self {
            Self::FromEvidence(evidence) => GraphExpr::FromEvidence {
                evidence: evidence.clone(),
            },
            Self::FromGraph(alias) => GraphExpr::FromGraph {
                alias: alias.clone(),
            },
            Self::Pipeline {
                start_graph,
                transforms,
            } => GraphExpr::Pipeline {
                start: start_graph.clone(),
                transforms: transforms.iter().map(TransformIR::to_ast).collect(),
            },
        }
    }
}

/// Transform IR variants in flow pipelines.
#[derive(Debug, Clone, PartialEq)]
pub enum TransformIR {
    /// Apply a rule by name (mode may be overridden in future phases).
    ApplyRule {
        rule: String,
        mode_override: Option<String>,
    },
    /// Apply multiple rules sequentially.
    ApplyRuleset { rules: Vec<String> },
    /// Save a snapshot of the current graph state.
    Snapshot { name: String },
    /// Run loopy belief propagation over independent edges.
    InferBeliefs,
    /// Prune edges of a given type based on a predicate expression.
    PruneEdges {
        edge_type: String,
        predicate: ExprIR,
    },
}

impl TransformIR {
    /// Convert this IR transform back to frontend AST.
    pub fn to_ast(&self) -> Transform {
        match self {
            Self::ApplyRule { rule, .. } => Transform::ApplyRule { rule: rule.clone() },
            Self::ApplyRuleset { rules } => Transform::ApplyRuleset {
                rules: rules.clone(),
            },
            Self::Snapshot { name } => Transform::Snapshot { name: name.clone() },
            Self::InferBeliefs => Transform::InferBeliefs,
            Self::PruneEdges {
                edge_type,
                predicate,
            } => Transform::PruneEdges {
                edge_type: edge_type.clone(),
                predicate: predicate.to_ast(),
            },
        }
    }
}

/// A named graph definition in flow IR.
#[derive(Debug, Clone, PartialEq)]
pub struct GraphDefIR {
    pub name: String,
    pub expr: GraphExprIR,
}

impl GraphDefIR {
    /// Convert this IR graph def back to frontend AST.
    pub fn to_ast(&self) -> grafial_frontend::ast::GraphDef {
        grafial_frontend::ast::GraphDef {
            name: self.name.clone(),
            expr: self.expr.to_ast(),
        }
    }
}

/// A metric definition in flow IR.
#[derive(Debug, Clone, PartialEq)]
pub struct MetricDefIR {
    pub name: String,
    pub expr: ExprIR,
}

impl MetricDefIR {
    /// Convert this IR metric def back to frontend AST.
    pub fn to_ast(&self) -> MetricDef {
        MetricDef {
            name: self.name.clone(),
            expr: self.expr.to_ast(),
        }
    }
}

/// A graph export definition in flow IR.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct ExportDefIR {
    pub graph: String,
    pub alias: String,
}

impl ExportDefIR {
    /// Convert this IR export def back to frontend AST.
    pub fn to_ast(&self) -> ExportDef {
        ExportDef {
            graph: self.graph.clone(),
            alias: self.alias.clone(),
        }
    }
}

/// A metric export definition in flow IR.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct MetricExportDefIR {
    pub metric: String,
    pub alias: String,
}

impl MetricExportDefIR {
    /// Convert this IR metric export def back to frontend AST.
    pub fn to_ast(&self) -> MetricExportDef {
        MetricExportDef {
            metric: self.metric.clone(),
            alias: self.alias.clone(),
        }
    }
}

/// A metric import definition in flow IR.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct MetricImportDefIR {
    pub source_alias: String,
    pub local_name: String,
}

impl MetricImportDefIR {
    /// Convert this IR metric import def back to frontend AST.
    pub fn to_ast(&self) -> MetricImportDef {
        MetricImportDef {
            source_alias: self.source_alias.clone(),
            local_name: self.local_name.clone(),
        }
    }
}

/// A lowered flow in IR form.
#[derive(Debug, Clone, PartialEq)]
pub struct FlowIR {
    pub name: String,
    pub on_model: String,
    pub graphs: Vec<GraphDefIR>,
    pub metrics: Vec<MetricDefIR>,
    pub exports: Vec<ExportDefIR>,
    pub metric_exports: Vec<MetricExportDefIR>,
    pub metric_imports: Vec<MetricImportDefIR>,
}

impl FlowIR {
    /// Convert this IR flow back to frontend AST.
    pub fn to_ast(&self) -> FlowDef {
        FlowDef {
            name: self.name.clone(),
            on_model: self.on_model.clone(),
            graphs: self.graphs.iter().map(GraphDefIR::to_ast).collect(),
            metrics: self.metrics.iter().map(MetricDefIR::to_ast).collect(),
            exports: self.exports.iter().map(ExportDefIR::to_ast).collect(),
            metric_exports: self
                .metric_exports
                .iter()
                .map(MetricExportDefIR::to_ast)
                .collect(),
            metric_imports: self
                .metric_imports
                .iter()
                .map(MetricImportDefIR::to_ast)
                .collect(),
        }
    }
}

impl From<&FlowDef> for FlowIR {
    fn from(f: &FlowDef) -> Self {
        let graphs = f
            .graphs
            .iter()
            .map(|g| GraphDefIR {
                name: g.name.clone(),
                expr: match &g.expr {
                    GraphExpr::FromEvidence { evidence } => {
                        GraphExprIR::FromEvidence(evidence.clone())
                    }
                    GraphExpr::FromGraph { alias } => GraphExprIR::FromGraph(alias.clone()),
                    GraphExpr::Pipeline { start, transforms } => GraphExprIR::Pipeline {
                        start_graph: start.clone(),
                        transforms: transforms
                            .iter()
                            .map(|t| match t {
                                Transform::ApplyRule { rule } => TransformIR::ApplyRule {
                                    rule: rule.clone(),
                                    mode_override: None,
                                },
                                Transform::ApplyRuleset { rules } => TransformIR::ApplyRuleset {
                                    rules: rules.clone(),
                                },
                                Transform::Snapshot { name } => {
                                    TransformIR::Snapshot { name: name.clone() }
                                }
                                Transform::InferBeliefs => TransformIR::InferBeliefs,
                                Transform::PruneEdges {
                                    edge_type,
                                    predicate,
                                } => TransformIR::PruneEdges {
                                    edge_type: edge_type.clone(),
                                    predicate: ExprIR::from(predicate),
                                },
                            })
                            .collect(),
                    },
                },
            })
            .collect();

        Self {
            name: f.name.clone(),
            on_model: f.on_model.clone(),
            graphs,
            metrics: f
                .metrics
                .iter()
                .map(|m| MetricDefIR {
                    name: m.name.clone(),
                    expr: ExprIR::from(&m.expr),
                })
                .collect(),
            exports: f
                .exports
                .iter()
                .map(|ex| ExportDefIR {
                    graph: ex.graph.clone(),
                    alias: ex.alias.clone(),
                })
                .collect(),
            metric_exports: f
                .metric_exports
                .iter()
                .map(|mex| MetricExportDefIR {
                    metric: mex.metric.clone(),
                    alias: mex.alias.clone(),
                })
                .collect(),
            metric_imports: f
                .metric_imports
                .iter()
                .map(|imp| MetricImportDefIR {
                    source_alias: imp.source_alias.clone(),
                    local_name: imp.local_name.clone(),
                })
                .collect(),
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use grafial_frontend::ast::{BinaryOp, CallArg, ExprAst};

    #[test]
    fn flow_ir_from_evidence_conversion() {
        let flow_def = FlowDef {
            name: "TestFlow".into(),
            on_model: "TestModel".into(),
            graphs: vec![grafial_frontend::ast::GraphDef {
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
            graphs: vec![grafial_frontend::ast::GraphDef {
                name: "result".into(),
                expr: GraphExpr::Pipeline {
                    start: "base".into(),
                    transforms: vec![
                        Transform::ApplyRule {
                            rule: "Rule1".into(),
                        },
                        Transform::InferBeliefs,
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

        if let GraphExprIR::Pipeline {
            start_graph,
            transforms,
        } = &ir.graphs[0].expr
        {
            assert_eq!(start_graph, "base");
            assert_eq!(transforms.len(), 3);

            assert!(matches!(
                &transforms[0],
                TransformIR::ApplyRule { rule, mode_override } if rule == "Rule1" && mode_override.is_none()
            ));

            assert!(matches!(&transforms[1], TransformIR::InferBeliefs));

            assert!(matches!(
                &transforms[2],
                TransformIR::PruneEdges { edge_type, predicate } if edge_type == "LINK" && matches!(predicate, ExprIR::Bool(true))
            ));
        } else {
            panic!("Expected Pipeline expression");
        }
    }

    #[test]
    fn flow_ir_preserves_metrics_and_exports() {
        let flow_def = FlowDef {
            name: "M".into(),
            on_model: "Model".into(),
            graphs: vec![grafial_frontend::ast::GraphDef {
                name: "g".into(),
                expr: GraphExpr::FromGraph {
                    alias: "prior".into(),
                },
            }],
            metrics: vec![MetricDef {
                name: "score".into(),
                expr: ExprAst::Call {
                    name: "mean".into(),
                    args: vec![
                        CallArg::Positional(ExprAst::Var("node".into())),
                        CallArg::Named {
                            name: "label".into(),
                            value: ExprAst::Var("Person".into()),
                        },
                    ],
                },
            }],
            exports: vec![ExportDef {
                graph: "g".into(),
                alias: "out".into(),
            }],
            metric_exports: vec![MetricExportDef {
                metric: "score".into(),
                alias: "score_out".into(),
            }],
            metric_imports: vec![MetricImportDef {
                source_alias: "baseline".into(),
                local_name: "prior_score".into(),
            }],
        };

        let ir = FlowIR::from(&flow_def);
        assert_eq!(ir.metrics.len(), 1);
        assert_eq!(ir.exports.len(), 1);
        assert_eq!(ir.metric_exports.len(), 1);
        assert_eq!(ir.metric_imports.len(), 1);
    }

    #[test]
    fn flow_ir_roundtrips_to_ast() {
        let flow_def = FlowDef {
            name: "RoundTrip".into(),
            on_model: "Model".into(),
            graphs: vec![grafial_frontend::ast::GraphDef {
                name: "g".into(),
                expr: GraphExpr::Pipeline {
                    start: "base".into(),
                    transforms: vec![Transform::PruneEdges {
                        edge_type: "REL".into(),
                        predicate: ExprAst::Binary {
                            op: BinaryOp::Gt,
                            left: Box::new(ExprAst::Number(1.0)),
                            right: Box::new(ExprAst::Number(0.0)),
                        },
                    }],
                },
            }],
            metrics: vec![MetricDef {
                name: "m".into(),
                expr: ExprAst::Bool(true),
            }],
            exports: vec![ExportDef {
                graph: "g".into(),
                alias: "out".into(),
            }],
            metric_exports: vec![MetricExportDef {
                metric: "m".into(),
                alias: "mo".into(),
            }],
            metric_imports: vec![MetricImportDef {
                source_alias: "in".into(),
                local_name: "local".into(),
            }],
        };

        let ir = FlowIR::from(&flow_def);
        let back = ir.to_ast();
        assert_eq!(flow_def, back);
    }
}
