//! Program IR for Grafial.
//!
//! Aggregates all lowered IR components into a complete program representation.

use crate::evidence::EvidenceIR;
use crate::flow::FlowIR;
use crate::optimize::optimize_program;
use crate::rule::RuleIR;
use grafial_frontend::ast::ProgramAst;

/// A lowered program in IR form.
#[derive(Debug, Clone, PartialEq)]
pub struct ProgramIR {
    /// Schema definitions are copied directly from frontend AST.
    pub schemas: Vec<grafial_frontend::ast::Schema>,
    /// Belief models are copied directly from frontend AST.
    pub belief_models: Vec<grafial_frontend::ast::BeliefModel>,
    /// Lowered evidence definitions.
    pub evidences: Vec<EvidenceIR>,
    /// Lowered rules.
    pub rules: Vec<RuleIR>,
    /// Lowered flows.
    pub flows: Vec<FlowIR>,
}

impl ProgramIR {
    /// Convert this IR program back to frontend AST.
    pub fn to_ast(&self) -> ProgramAst {
        ProgramAst {
            schemas: self.schemas.clone(),
            belief_models: self.belief_models.clone(),
            evidences: self.evidences.iter().map(EvidenceIR::to_ast).collect(),
            rules: self.rules.iter().map(RuleIR::to_ast).collect(),
            flows: self.flows.iter().map(FlowIR::to_ast).collect(),
        }
    }

    /// Returns an optimized copy of this program IR.
    pub fn optimized(&self) -> Self {
        optimize_program(self)
    }
}

impl From<&ProgramAst> for ProgramIR {
    fn from(ast: &ProgramAst) -> Self {
        Self {
            schemas: ast.schemas.clone(),
            belief_models: ast.belief_models.clone(),
            evidences: ast.evidences.iter().map(EvidenceIR::from).collect(),
            rules: ast.rules.iter().map(RuleIR::from).collect(),
            flows: ast.flows.iter().map(FlowIR::from).collect(),
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use grafial_frontend::ast::*;

    #[test]
    fn program_ir_from_program_ast() {
        let ast = ProgramAst {
            schemas: vec![Schema {
                name: "TestSchema".into(),
                nodes: vec![],
                edges: vec![],
            }],
            belief_models: vec![],
            evidences: vec![],
            rules: vec![RuleDef {
                name: "Rule1".into(),
                on_model: "Model1".into(),
                patterns: vec![],
                where_expr: None,
                actions: vec![],
                mode: None,
            }],
            flows: vec![FlowDef {
                name: "Flow1".into(),
                on_model: "Model1".into(),
                graphs: vec![],
                metrics: vec![],
                exports: vec![],
                metric_exports: vec![],
                metric_imports: vec![],
            }],
        };

        let ir = ProgramIR::from(&ast);

        assert_eq!(ir.schemas.len(), 1);
        assert_eq!(ir.schemas[0].name, "TestSchema");
        assert_eq!(ir.rules.len(), 1);
        assert_eq!(ir.rules[0].name, "Rule1");
        assert_eq!(ir.flows.len(), 1);
        assert_eq!(ir.flows[0].name, "Flow1");
    }

    #[test]
    fn program_ir_preserves_all_components() {
        let ast = ProgramAst {
            schemas: vec![],
            belief_models: vec![BeliefModel {
                name: "Model1".into(),
                on_schema: "TestSchema".into(),
                nodes: vec![],
                edges: vec![],
                body_src: "".into(),
            }],
            evidences: vec![EvidenceDef {
                name: "Ev1".into(),
                on_model: "Model1".into(),
                body_src: "".into(),
                observations: vec![],
            }],
            rules: vec![],
            flows: vec![],
        };

        let ir = ProgramIR::from(&ast);

        assert_eq!(ir.belief_models.len(), 1);
        assert_eq!(ir.belief_models[0].name, "Model1");
        assert_eq!(ir.evidences.len(), 1);
        assert_eq!(ir.evidences[0].name, "Ev1");
    }

    #[test]
    fn program_ir_roundtrips_to_ast() {
        let ast = ProgramAst {
            schemas: vec![],
            belief_models: vec![],
            evidences: vec![EvidenceDef {
                name: "Ev".into(),
                on_model: "M".into(),
                observations: vec![ObserveStmt::Attribute {
                    node: ("N".into(), "x".into()),
                    attr: "a".into(),
                    value: 1.0,
                    precision: None,
                }],
                body_src: "raw".into(),
            }],
            rules: vec![RuleDef {
                name: "R".into(),
                on_model: "M".into(),
                patterns: vec![],
                where_expr: Some(ExprAst::Bool(true)),
                actions: vec![ActionStmt::ForceAbsent {
                    edge_var: "e".into(),
                }],
                mode: Some("for_each".into()),
            }],
            flows: vec![FlowDef {
                name: "F".into(),
                on_model: "M".into(),
                graphs: vec![],
                metrics: vec![],
                exports: vec![],
                metric_exports: vec![],
                metric_imports: vec![],
            }],
        };

        let ir = ProgramIR::from(&ast);
        let back = ir.to_ast();
        assert_eq!(ast, back);
    }
}
