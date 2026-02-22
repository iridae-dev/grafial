//! IR optimization and canonicalization passes.
//!
//! Phase 4 focuses on safe compile-time rewrites:
//! - constant folding on expression trees
//! - canonicalization of equivalent forms
//! - elimination of unreferenced rules
//! - removal of no-op transforms

use std::collections::{HashMap, HashSet};

use crate::expr::{BinaryOpIR, CallArgIR, ExprIR, UnaryOpIR};
use crate::flow::{FlowIR, GraphExprIR, TransformIR};
use crate::program::ProgramIR;
use crate::rule::{ActionIR, RuleIR};

const FLOAT_EPSILON: f64 = 1e-12;

/// Optimize a lowered program IR with semantics-preserving rewrites.
pub fn optimize_program(program: &ProgramIR) -> ProgramIR {
    let mut optimized = program.clone();

    optimize_rules(&mut optimized.rules);
    optimize_flows(&mut optimized.flows);
    eliminate_unreferenced_rules(&mut optimized.rules, &optimized.flows);

    optimized
}

fn optimize_rules(rules: &mut [RuleIR]) {
    let const_env = HashMap::new();
    for rule in rules {
        if let Some(where_expr) = &rule.where_expr {
            rule.where_expr = Some(fold_expr(where_expr, &const_env));
        }
        for action in &mut rule.actions {
            optimize_action(action);
        }
    }
}

fn optimize_action(action: &mut ActionIR) {
    let const_env = HashMap::new();
    match action {
        ActionIR::Let { expr, .. } => *expr = fold_expr(expr, &const_env),
        ActionIR::SetExpectation { expr, .. } => *expr = fold_expr(expr, &const_env),
        ActionIR::NonBayesianNudge { expr, .. } => *expr = fold_expr(expr, &const_env),
        ActionIR::SoftUpdate { expr, .. } => *expr = fold_expr(expr, &const_env),
        ActionIR::ForceAbsent { .. } => {}
        ActionIR::DeleteEdge { .. } => {}
        ActionIR::SuppressEdge { .. } => {}
    }
}

fn optimize_flows(flows: &mut [FlowIR]) {
    for flow in flows {
        optimize_flow_graphs(flow);
        optimize_flow_metrics(flow);
    }
}

fn optimize_flow_graphs(flow: &mut FlowIR) {
    for graph in &mut flow.graphs {
        if let GraphExprIR::Pipeline { transforms, .. } = &mut graph.expr {
            let mut optimized_transforms = Vec::with_capacity(transforms.len());
            for transform in transforms.iter() {
                if let Some(optimized) = optimize_transform(transform) {
                    optimized_transforms.push(optimized);
                }
            }
            *transforms = optimized_transforms;
        }
    }
}

fn optimize_transform(transform: &TransformIR) -> Option<TransformIR> {
    match transform {
        TransformIR::ApplyRule {
            rule,
            mode_override,
        } => Some(TransformIR::ApplyRule {
            rule: rule.clone(),
            mode_override: mode_override.clone(),
        }),
        TransformIR::ApplyRuleset { rules } => match rules.as_slice() {
            [] => None,
            [single_rule] => Some(TransformIR::ApplyRule {
                rule: single_rule.clone(),
                mode_override: None,
            }),
            _ => Some(TransformIR::ApplyRuleset {
                rules: rules.clone(),
            }),
        },
        TransformIR::Snapshot { name } => Some(TransformIR::Snapshot { name: name.clone() }),
        TransformIR::InferBeliefs => Some(TransformIR::InferBeliefs),
        TransformIR::PruneEdges {
            edge_type,
            predicate,
        } => {
            let const_env = HashMap::new();
            let folded_predicate = fold_expr(predicate, &const_env);
            match literal_truthiness(&folded_predicate) {
                Some(false) => None,
                Some(true) => Some(TransformIR::PruneEdges {
                    edge_type: edge_type.clone(),
                    predicate: ExprIR::Bool(true),
                }),
                None => Some(TransformIR::PruneEdges {
                    edge_type: edge_type.clone(),
                    predicate: folded_predicate,
                }),
            }
        }
    }
}

fn optimize_flow_metrics(flow: &mut FlowIR) {
    // Constant propagation for flow-local metric variables in declaration order.
    let mut metric_constants: HashMap<String, f64> = HashMap::new();
    for metric in &mut flow.metrics {
        metric.expr = fold_expr(&metric.expr, &metric_constants);
        if let Some(v) = literal_numeric(&metric.expr) {
            metric_constants.insert(metric.name.clone(), v);
        } else {
            metric_constants.remove(metric.name.as_str());
        }
    }
}

fn eliminate_unreferenced_rules(rules: &mut Vec<RuleIR>, flows: &[FlowIR]) {
    let referenced = collect_referenced_rules(flows);
    rules.retain(|rule| referenced.contains(rule.name.as_str()));
}

fn collect_referenced_rules(flows: &[FlowIR]) -> HashSet<&str> {
    let mut referenced = HashSet::new();
    for flow in flows {
        for graph in &flow.graphs {
            if let GraphExprIR::Pipeline { transforms, .. } = &graph.expr {
                for transform in transforms {
                    match transform {
                        TransformIR::ApplyRule { rule, .. } => {
                            referenced.insert(rule.as_str());
                        }
                        TransformIR::ApplyRuleset { rules } => {
                            for rule in rules {
                                referenced.insert(rule.as_str());
                            }
                        }
                        TransformIR::Snapshot { .. }
                        | TransformIR::InferBeliefs
                        | TransformIR::PruneEdges { .. } => {}
                    }
                }
            }
        }
    }
    referenced
}

fn fold_expr(expr: &ExprIR, const_env: &HashMap<String, f64>) -> ExprIR {
    match expr {
        ExprIR::Number(v) => ExprIR::Number(*v),
        ExprIR::Bool(v) => ExprIR::Bool(*v),
        ExprIR::Var(name) => {
            if let Some(v) = const_env.get(name) {
                ExprIR::Number(*v)
            } else {
                ExprIR::Var(name.clone())
            }
        }
        ExprIR::Field { target, field } => ExprIR::Field {
            target: Box::new(fold_expr(target, const_env)),
            field: field.clone(),
        },
        ExprIR::Call { name, args } => ExprIR::Call {
            name: name.clone(),
            args: args
                .iter()
                .map(|arg| fold_call_arg(arg, const_env))
                .collect(),
        },
        ExprIR::Unary { op, expr } => {
            let folded_expr = fold_expr(expr, const_env);

            if let Some(value) = literal_numeric(&folded_expr) {
                return match op {
                    UnaryOpIR::Neg => ExprIR::Number(-value),
                    UnaryOpIR::Not => ExprIR::Bool(value == 0.0),
                };
            }

            match (*op, folded_expr) {
                (
                    UnaryOpIR::Neg,
                    ExprIR::Unary {
                        op: UnaryOpIR::Neg,
                        expr,
                    },
                ) => *expr,
                (op, expr) => ExprIR::Unary {
                    op,
                    expr: Box::new(expr),
                },
            }
        }
        ExprIR::Binary { op, left, right } => {
            let folded_left = fold_expr(left, const_env);
            let folded_right = fold_expr(right, const_env);

            if let (Some(l), Some(r)) = (
                literal_numeric(&folded_left),
                literal_numeric(&folded_right),
            ) {
                if let Some(v) = fold_binary_constants(*op, l, r) {
                    return if binary_result_is_bool(*op) {
                        ExprIR::Bool(v != 0.0)
                    } else {
                        ExprIR::Number(v)
                    };
                }
            }

            ExprIR::Binary {
                op: *op,
                left: Box::new(folded_left),
                right: Box::new(folded_right),
            }
        }
        ExprIR::Exists {
            pattern,
            where_expr,
            negated,
        } => {
            let folded_where = where_expr
                .as_ref()
                .map(|expr| Box::new(fold_expr(expr, const_env)));

            if let Some(where_expr) = &folded_where {
                if let Some(truthy) = literal_truthiness(where_expr) {
                    if truthy {
                        return ExprIR::Exists {
                            pattern: pattern.clone(),
                            where_expr: None,
                            negated: *negated,
                        };
                    }
                    return ExprIR::Bool(*negated);
                }
            }

            ExprIR::Exists {
                pattern: pattern.clone(),
                where_expr: folded_where,
                negated: *negated,
            }
        }
    }
}

fn fold_call_arg(arg: &CallArgIR, const_env: &HashMap<String, f64>) -> CallArgIR {
    match arg {
        CallArgIR::Positional(expr) => CallArgIR::Positional(fold_expr(expr, const_env)),
        CallArgIR::Named { name, value } => CallArgIR::Named {
            name: name.clone(),
            value: fold_expr(value, const_env),
        },
    }
}

fn literal_numeric(expr: &ExprIR) -> Option<f64> {
    match expr {
        ExprIR::Number(v) => Some(*v),
        ExprIR::Bool(b) => Some(if *b { 1.0 } else { 0.0 }),
        _ => None,
    }
}

fn literal_truthiness(expr: &ExprIR) -> Option<bool> {
    literal_numeric(expr).map(|v| v != 0.0)
}

fn fold_binary_constants(op: BinaryOpIR, left: f64, right: f64) -> Option<f64> {
    let result = match op {
        BinaryOpIR::Add => left + right,
        BinaryOpIR::Sub => left - right,
        BinaryOpIR::Mul => left * right,
        BinaryOpIR::Div => {
            if right.abs() < FLOAT_EPSILON {
                return None;
            }
            left / right
        }
        BinaryOpIR::Eq => {
            if (left - right).abs() < FLOAT_EPSILON {
                1.0
            } else {
                0.0
            }
        }
        BinaryOpIR::Ne => {
            if (left - right).abs() >= FLOAT_EPSILON {
                1.0
            } else {
                0.0
            }
        }
        BinaryOpIR::Lt => {
            if left < right {
                1.0
            } else {
                0.0
            }
        }
        BinaryOpIR::Le => {
            if left <= right {
                1.0
            } else {
                0.0
            }
        }
        BinaryOpIR::Gt => {
            if left > right {
                1.0
            } else {
                0.0
            }
        }
        BinaryOpIR::Ge => {
            if left >= right {
                1.0
            } else {
                0.0
            }
        }
        BinaryOpIR::And => {
            if (left != 0.0) && (right != 0.0) {
                1.0
            } else {
                0.0
            }
        }
        BinaryOpIR::Or => {
            if (left != 0.0) || (right != 0.0) {
                1.0
            } else {
                0.0
            }
        }
    };

    if result.is_finite() {
        Some(result)
    } else {
        None
    }
}

fn binary_result_is_bool(op: BinaryOpIR) -> bool {
    matches!(
        op,
        BinaryOpIR::Eq
            | BinaryOpIR::Ne
            | BinaryOpIR::Lt
            | BinaryOpIR::Le
            | BinaryOpIR::Gt
            | BinaryOpIR::Ge
            | BinaryOpIR::And
            | BinaryOpIR::Or
    )
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::flow::{GraphDefIR, MetricDefIR};
    use crate::rule::ActionIR;
    use grafial_frontend::ast::{EdgePattern, NodePattern, PatternItem};

    fn empty_program(rules: Vec<RuleIR>, flows: Vec<FlowIR>) -> ProgramIR {
        ProgramIR {
            schemas: vec![],
            belief_models: vec![],
            evidences: vec![],
            rules,
            flows,
        }
    }

    fn dummy_pattern() -> PatternItem {
        PatternItem {
            src: NodePattern {
                var: "a".into(),
                label: "A".into(),
            },
            edge: EdgePattern {
                var: "e".into(),
                ty: "REL".into(),
            },
            dst: NodePattern {
                var: "b".into(),
                label: "B".into(),
            },
        }
    }

    #[test]
    fn optimize_folds_metric_constants_with_propagation() {
        let flow = FlowIR {
            name: "F".into(),
            on_model: "M".into(),
            graphs: vec![],
            metrics: vec![
                MetricDefIR {
                    name: "m1".into(),
                    expr: ExprIR::Binary {
                        op: BinaryOpIR::Add,
                        left: Box::new(ExprIR::Number(2.0)),
                        right: Box::new(ExprIR::Number(3.0)),
                    },
                },
                MetricDefIR {
                    name: "m2".into(),
                    expr: ExprIR::Binary {
                        op: BinaryOpIR::Mul,
                        left: Box::new(ExprIR::Var("m1".into())),
                        right: Box::new(ExprIR::Number(4.0)),
                    },
                },
            ],
            exports: vec![],
            metric_exports: vec![],
            metric_imports: vec![],
        };

        let optimized = optimize_program(&empty_program(vec![], vec![flow]));
        assert_eq!(optimized.flows[0].metrics[0].expr, ExprIR::Number(5.0));
        assert_eq!(optimized.flows[0].metrics[1].expr, ExprIR::Number(20.0));
    }

    #[test]
    fn optimize_simplifies_pipeline_transforms() {
        let flow = FlowIR {
            name: "F".into(),
            on_model: "M".into(),
            graphs: vec![GraphDefIR {
                name: "g".into(),
                expr: GraphExprIR::Pipeline {
                    start_graph: "base".into(),
                    transforms: vec![
                        TransformIR::ApplyRuleset { rules: vec![] },
                        TransformIR::ApplyRuleset {
                            rules: vec!["R1".into()],
                        },
                        TransformIR::InferBeliefs,
                        TransformIR::PruneEdges {
                            edge_type: "REL".into(),
                            predicate: ExprIR::Bool(false),
                        },
                        TransformIR::PruneEdges {
                            edge_type: "REL".into(),
                            predicate: ExprIR::Number(1.0),
                        },
                    ],
                },
            }],
            metrics: vec![],
            exports: vec![],
            metric_exports: vec![],
            metric_imports: vec![],
        };

        let optimized = optimize_program(&empty_program(vec![], vec![flow]));
        let GraphExprIR::Pipeline { transforms, .. } = &optimized.flows[0].graphs[0].expr else {
            panic!("expected pipeline");
        };

        assert_eq!(transforms.len(), 3);
        assert!(matches!(
            &transforms[0],
            TransformIR::ApplyRule { rule, mode_override } if rule == "R1" && mode_override.is_none()
        ));
        assert!(matches!(&transforms[1], TransformIR::InferBeliefs));
        assert!(matches!(
            &transforms[2],
            TransformIR::PruneEdges { edge_type, predicate } if edge_type == "REL" && *predicate == ExprIR::Bool(true)
        ));
    }

    #[test]
    fn optimize_eliminates_unreferenced_rules() {
        let used_rule = RuleIR {
            name: "Used".into(),
            on_model: "M".into(),
            patterns: vec![dummy_pattern()],
            where_expr: None,
            actions: vec![ActionIR::ForceAbsent {
                edge_var: "e".into(),
            }],
            mode: Some("for_each".into()),
        };
        let dead_rule = RuleIR {
            name: "Dead".into(),
            on_model: "M".into(),
            patterns: vec![dummy_pattern()],
            where_expr: None,
            actions: vec![ActionIR::ForceAbsent {
                edge_var: "e".into(),
            }],
            mode: Some("for_each".into()),
        };
        let flow = FlowIR {
            name: "F".into(),
            on_model: "M".into(),
            graphs: vec![GraphDefIR {
                name: "g".into(),
                expr: GraphExprIR::Pipeline {
                    start_graph: "base".into(),
                    transforms: vec![TransformIR::ApplyRule {
                        rule: "Used".into(),
                        mode_override: None,
                    }],
                },
            }],
            metrics: vec![],
            exports: vec![],
            metric_exports: vec![],
            metric_imports: vec![],
        };

        let optimized = optimize_program(&empty_program(vec![used_rule, dead_rule], vec![flow]));
        assert_eq!(optimized.rules.len(), 1);
        assert_eq!(optimized.rules[0].name, "Used");
    }

    #[test]
    fn optimize_simplifies_exists_with_constant_where_clause() {
        let rule = RuleIR {
            name: "R".into(),
            on_model: "M".into(),
            patterns: vec![dummy_pattern()],
            where_expr: Some(ExprIR::Exists {
                pattern: dummy_pattern(),
                where_expr: Some(Box::new(ExprIR::Bool(false))),
                negated: true,
            }),
            actions: vec![],
            mode: Some("for_each".into()),
        };

        let flow = FlowIR {
            name: "F".into(),
            on_model: "M".into(),
            graphs: vec![GraphDefIR {
                name: "g".into(),
                expr: GraphExprIR::Pipeline {
                    start_graph: "base".into(),
                    transforms: vec![TransformIR::ApplyRule {
                        rule: "R".into(),
                        mode_override: None,
                    }],
                },
            }],
            metrics: vec![],
            exports: vec![],
            metric_exports: vec![],
            metric_imports: vec![],
        };

        let optimized = optimize_program(&empty_program(vec![rule], vec![flow]));
        assert_eq!(optimized.rules[0].where_expr, Some(ExprIR::Bool(true)));
    }
}
