//! Rule IR for Grafial
//!
//! This IR closely follows the AST shapes but decouples execution from
//! parsing structures. It is intentionally minimal.

use grafial_frontend::ast::{ActionStmt, ExprAst, PatternItem};

// Note: PatternItem is actually a struct in AST, but we keep it as-is for now
// since it's already a stable structure. In the future, we could create PatternItemIR
// if we need to optimize pattern representation.

/// A lowered rule in IR form.
#[derive(Debug, Clone, PartialEq)]
pub struct RuleIR {
    /// The rule name
    pub name: String,
    /// The belief model this rule operates on
    pub on_model: String,
    /// Graph patterns to match
    pub patterns: Vec<PatternItem>,
    /// Optional where clause for filtering matches
    pub where_expr: Option<ExprAst>,
    /// Actions to execute for each match
    pub actions: Vec<ActionStmt>,
    /// Execution mode (e.g., "for_each", "fixpoint")
    pub mode: Option<String>,
}

impl From<&grafial_frontend::ast::RuleDef> for RuleIR {
    fn from(rule: &grafial_frontend::ast::RuleDef) -> Self {
        Self {
            name: rule.name.clone(),
            on_model: rule.on_model.clone(),
            patterns: rule.patterns.clone(),
            where_expr: rule.where_expr.clone(),
            actions: rule.actions.clone(),
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
        assert_eq!(ir.where_expr, Some(ExprAst::Bool(true)));
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
}
