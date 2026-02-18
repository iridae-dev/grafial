//! Rule IR for Grafial.
//!
//! This representation removes direct dependence on frontend expression/action
//! AST nodes while preserving rule semantics.

use crate::expr::ExprIR;
use grafial_frontend::ast::{ActionStmt, PatternItem, RuleDef, VarianceSpec};

/// Variance strategy in IR form for non-Bayesian nudges.
#[derive(Debug, Clone, PartialEq)]
pub enum VarianceSpecIR {
    Preserve,
    Increase { factor: Option<f64> },
    Decrease { factor: Option<f64> },
}

impl VarianceSpecIR {
    fn to_ast(&self) -> VarianceSpec {
        match self {
            Self::Preserve => VarianceSpec::Preserve,
            Self::Increase { factor } => VarianceSpec::Increase { factor: *factor },
            Self::Decrease { factor } => VarianceSpec::Decrease { factor: *factor },
        }
    }
}

impl From<&VarianceSpec> for VarianceSpecIR {
    fn from(value: &VarianceSpec) -> Self {
        match value {
            VarianceSpec::Preserve => Self::Preserve,
            VarianceSpec::Increase { factor } => Self::Increase { factor: *factor },
            VarianceSpec::Decrease { factor } => Self::Decrease { factor: *factor },
        }
    }
}

/// Action statement in IR form.
#[derive(Debug, Clone, PartialEq)]
pub enum ActionIR {
    Let {
        name: String,
        expr: ExprIR,
    },
    SetExpectation {
        node_var: String,
        attr: String,
        expr: ExprIR,
    },
    ForceAbsent {
        edge_var: String,
    },
    NonBayesianNudge {
        node_var: String,
        attr: String,
        expr: ExprIR,
        variance: Option<VarianceSpecIR>,
    },
    SoftUpdate {
        node_var: String,
        attr: String,
        expr: ExprIR,
        precision: Option<f64>,
        count: Option<f64>,
    },
    DeleteEdge {
        edge_var: String,
        confidence: Option<String>,
    },
    SuppressEdge {
        edge_var: String,
        weight: Option<f64>,
    },
}

impl ActionIR {
    /// Convert this IR action back to frontend AST.
    pub fn to_ast(&self) -> ActionStmt {
        match self {
            Self::Let { name, expr } => ActionStmt::Let {
                name: name.clone(),
                expr: expr.to_ast(),
            },
            Self::SetExpectation {
                node_var,
                attr,
                expr,
            } => ActionStmt::SetExpectation {
                node_var: node_var.clone(),
                attr: attr.clone(),
                expr: expr.to_ast(),
            },
            Self::ForceAbsent { edge_var } => ActionStmt::ForceAbsent {
                edge_var: edge_var.clone(),
            },
            Self::NonBayesianNudge {
                node_var,
                attr,
                expr,
                variance,
            } => ActionStmt::NonBayesianNudge {
                node_var: node_var.clone(),
                attr: attr.clone(),
                expr: expr.to_ast(),
                variance: variance.as_ref().map(VarianceSpecIR::to_ast),
            },
            Self::SoftUpdate {
                node_var,
                attr,
                expr,
                precision,
                count,
            } => ActionStmt::SoftUpdate {
                node_var: node_var.clone(),
                attr: attr.clone(),
                expr: expr.to_ast(),
                precision: *precision,
                count: *count,
            },
            Self::DeleteEdge {
                edge_var,
                confidence,
            } => ActionStmt::DeleteEdge {
                edge_var: edge_var.clone(),
                confidence: confidence.clone(),
            },
            Self::SuppressEdge { edge_var, weight } => ActionStmt::SuppressEdge {
                edge_var: edge_var.clone(),
                weight: *weight,
            },
        }
    }
}

impl From<&ActionStmt> for ActionIR {
    fn from(value: &ActionStmt) -> Self {
        match value {
            ActionStmt::Let { name, expr } => Self::Let {
                name: name.clone(),
                expr: ExprIR::from(expr),
            },
            ActionStmt::SetExpectation {
                node_var,
                attr,
                expr,
            } => Self::SetExpectation {
                node_var: node_var.clone(),
                attr: attr.clone(),
                expr: ExprIR::from(expr),
            },
            ActionStmt::ForceAbsent { edge_var } => Self::ForceAbsent {
                edge_var: edge_var.clone(),
            },
            ActionStmt::NonBayesianNudge {
                node_var,
                attr,
                expr,
                variance,
            } => Self::NonBayesianNudge {
                node_var: node_var.clone(),
                attr: attr.clone(),
                expr: ExprIR::from(expr),
                variance: variance.as_ref().map(VarianceSpecIR::from),
            },
            ActionStmt::SoftUpdate {
                node_var,
                attr,
                expr,
                precision,
                count,
            } => Self::SoftUpdate {
                node_var: node_var.clone(),
                attr: attr.clone(),
                expr: ExprIR::from(expr),
                precision: *precision,
                count: *count,
            },
            ActionStmt::DeleteEdge {
                edge_var,
                confidence,
            } => Self::DeleteEdge {
                edge_var: edge_var.clone(),
                confidence: confidence.clone(),
            },
            ActionStmt::SuppressEdge { edge_var, weight } => Self::SuppressEdge {
                edge_var: edge_var.clone(),
                weight: *weight,
            },
        }
    }
}

/// A lowered rule in IR form.
#[derive(Debug, Clone, PartialEq)]
pub struct RuleIR {
    pub name: String,
    pub on_model: String,
    pub patterns: Vec<PatternItem>,
    pub where_expr: Option<ExprIR>,
    pub actions: Vec<ActionIR>,
    pub mode: Option<String>,
}

impl RuleIR {
    /// Convert this IR rule back to frontend AST.
    pub fn to_ast(&self) -> RuleDef {
        RuleDef {
            name: self.name.clone(),
            on_model: self.on_model.clone(),
            patterns: self.patterns.clone(),
            where_expr: self.where_expr.as_ref().map(ExprIR::to_ast),
            actions: self.actions.iter().map(ActionIR::to_ast).collect(),
            mode: self.mode.clone(),
        }
    }
}

impl From<&RuleDef> for RuleIR {
    fn from(rule: &RuleDef) -> Self {
        Self {
            name: rule.name.clone(),
            on_model: rule.on_model.clone(),
            patterns: rule.patterns.clone(),
            where_expr: rule.where_expr.as_ref().map(ExprIR::from),
            actions: rule.actions.iter().map(ActionIR::from).collect(),
            mode: rule.mode.clone(),
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use grafial_frontend::ast::*;

    #[test]
    fn rule_ir_from_rule_def() {
        let rule_def = RuleDef {
            name: "TestRule".into(),
            on_model: "TestModel".into(),
            patterns: vec![PatternItem {
                src: NodePattern {
                    var: "n1".into(),
                    label: "Person".into(),
                },
                edge: EdgePattern {
                    var: "e1".into(),
                    ty: "REL".into(),
                },
                dst: NodePattern {
                    var: "n2".into(),
                    label: "Person".into(),
                },
            }],
            where_expr: Some(ExprAst::Bool(true)),
            actions: vec![ActionStmt::ForceAbsent {
                edge_var: "e1".into(),
            }],
            mode: None,
        };

        let ir = RuleIR::from(&rule_def);

        assert_eq!(ir.name, "TestRule");
        assert_eq!(ir.on_model, "TestModel");
        assert_eq!(ir.patterns.len(), 1);
        assert_eq!(ir.where_expr, Some(ExprIR::Bool(true)));
        assert_eq!(ir.actions.len(), 1);
    }

    #[test]
    fn rule_ir_clone_works() {
        let rule = RuleIR {
            name: "test".into(),
            on_model: "model".into(),
            patterns: vec![],
            where_expr: None,
            actions: vec![],
            mode: None,
        };

        let cloned = rule.clone();
        assert_eq!(rule.name, cloned.name);
        assert_eq!(rule.on_model, cloned.on_model);
    }

    #[test]
    fn rule_ir_roundtrips_to_ast() {
        let rule_def = RuleDef {
            name: "RoundTrip".into(),
            on_model: "M".into(),
            patterns: vec![PatternItem {
                src: NodePattern {
                    var: "a".into(),
                    label: "A".into(),
                },
                edge: EdgePattern {
                    var: "e".into(),
                    ty: "R".into(),
                },
                dst: NodePattern {
                    var: "b".into(),
                    label: "B".into(),
                },
            }],
            where_expr: Some(ExprAst::Binary {
                op: BinaryOp::Gt,
                left: Box::new(ExprAst::Number(2.0)),
                right: Box::new(ExprAst::Number(1.0)),
            }),
            actions: vec![ActionStmt::SetExpectation {
                node_var: "a".into(),
                attr: "x".into(),
                expr: ExprAst::Number(1.0),
            }],
            mode: Some("for_each".into()),
        };

        let ir = RuleIR::from(&rule_def);
        let back = ir.to_ast();
        assert_eq!(rule_def, back);
    }
}
