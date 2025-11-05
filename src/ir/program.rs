//! Program IR for Baygraph
//!
//! Aggregates all IR types into a complete program representation.

use crate::ir::flow::FlowIR;
use crate::ir::rule::RuleIR;


/// A lowered program in IR form.
///
/// This is the stable interface between frontend and engine.
/// The engine should depend on ProgramIR, not ProgramAst.
#[derive(Debug, Clone, PartialEq)]
pub struct ProgramIR {
    /// Schema definitions (unchanged from AST - no need to lower)
    pub schemas: Vec<crate::frontend::ast::Schema>,
    /// Belief models (unchanged from AST - no need to lower)
    pub belief_models: Vec<crate::frontend::ast::BeliefModel>,
    /// Evidence definitions (unchanged from AST - no need to lower)
    pub evidences: Vec<crate::frontend::ast::EvidenceDef>,
    /// Lowered rules
    pub rules: Vec<RuleIR>,
    /// Lowered flows
    pub flows: Vec<FlowIR>,
}

impl From<&crate::frontend::ast::ProgramAst> for ProgramIR {
    fn from(ast: &crate::frontend::ast::ProgramAst) -> Self {
        Self {
            schemas: ast.schemas.clone(),
            belief_models: ast.belief_models.clone(),
            evidences: ast.evidences.clone(),
            rules: ast.rules.iter().map(RuleIR::from).collect(),
            flows: ast.flows.iter().map(FlowIR::from).collect(),
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::frontend::ast::*;

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
}

