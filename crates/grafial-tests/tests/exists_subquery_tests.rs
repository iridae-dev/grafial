//! Tests for exists/not exists subquery functionality.

use grafial_core::engine::graph::*;
use grafial_core::engine::rule_exec::run_rule_for_each;
use grafial_frontend::ast::*;
use std::collections::HashMap;

#[test]
fn exists_subquery_finds_matching_edge() {
    let mut g = BeliefGraph::default();

    let n1 = g.add_node("Person".into(), HashMap::new());
    let n2 = g.add_node("Person".into(), HashMap::new());
    let n3 = g.add_node("Person".into(), HashMap::new());

    // Create edges: n1 -> n2 (high prob), n1 -> n3 (low prob)
    let e1 = g.add_edge(
        n1,
        n2,
        "REL".into(),
        BetaPosterior {
            alpha: 9.0,
            beta: 1.0,
        },
    );
    let _e2 = g.add_edge(
        n1,
        n3,
        "REL".into(),
        BetaPosterior {
            alpha: 1.0,
            beta: 9.0,
        },
    );

    let rule = RuleDef {
        name: "HasHighProbEdge".into(),
        on_model: "M".into(),
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
            op: BinaryOp::And,
            left: Box::new(ExprAst::Binary {
                op: BinaryOp::Ge,
                left: Box::new(ExprAst::Call {
                    name: "prob".into(),
                    args: vec![CallArg::Positional(ExprAst::Var("e".into()))],
                }),
                right: Box::new(ExprAst::Number(0.8)),
            }),
            right: Box::new(ExprAst::Exists {
                pattern: PatternItem {
                    src: NodePattern {
                        var: "A".into(),
                        label: "Person".into(),
                    },
                    edge: EdgePattern {
                        var: "ax".into(),
                        ty: "REL".into(),
                    },
                    dst: NodePattern {
                        var: "X".into(),
                        label: "Person".into(),
                    },
                },
                where_expr: Some(Box::new(ExprAst::Binary {
                    op: BinaryOp::Ge,
                    left: Box::new(ExprAst::Call {
                        name: "prob".into(),
                        args: vec![CallArg::Positional(ExprAst::Var("ax".into()))],
                    }),
                    right: Box::new(ExprAst::Number(0.9)),
                })),
                negated: false,
            }),
        }),
        actions: vec![ActionStmt::ForceAbsent {
            edge_var: "e".into(),
        }],
        mode: Some("for_each".into()),
    };

    let result = run_rule_for_each(&g, &rule).unwrap();

    // Only e1 should match (has high prob and exists subquery finds n1 -> n2 with prob >= 0.9)
    let prob = result.prob_mean(e1).unwrap();
    assert!(prob < 1e-5, "Expected e1 prob < 1e-5, got {}", prob);
}

#[test]
fn not_exists_subquery_filters_matches() {
    let mut g = BeliefGraph::default();

    let n1 = g.add_node("Person".into(), HashMap::new());
    let n2 = g.add_node("Person".into(), HashMap::new());
    let n3 = g.add_node("Person".into(), HashMap::new());

    // Create edges: n1 -> n2, n1 -> n3
    let e1 = g.add_edge(
        n1,
        n2,
        "REL".into(),
        BetaPosterior {
            alpha: 9.0,
            beta: 1.0,
        },
    );
    let _e2 = g.add_edge(
        n1,
        n3,
        "REL".into(),
        BetaPosterior {
            alpha: 8.0,
            beta: 2.0,
        },
    );

    let rule = RuleDef {
        name: "NoOtherEdge".into(),
        on_model: "M".into(),
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
        where_expr: Some(ExprAst::Exists {
            pattern: PatternItem {
                src: NodePattern {
                    var: "A".into(),
                    label: "Person".into(),
                },
                edge: EdgePattern {
                    var: "ac".into(),
                    ty: "REL".into(),
                },
                dst: NodePattern {
                    var: "C".into(),
                    label: "Person".into(),
                },
            },
            where_expr: Some(Box::new(ExprAst::Binary {
                op: BinaryOp::Ge,
                left: Box::new(ExprAst::Call {
                    name: "prob".into(),
                    args: vec![CallArg::Positional(ExprAst::Var("ac".into()))],
                }),
                right: Box::new(ExprAst::Number(0.5)),
            })),
            negated: true, // not exists
        }),
        actions: vec![ActionStmt::ForceAbsent {
            edge_var: "e".into(),
        }],
        mode: Some("for_each".into()),
    };

    let result = run_rule_for_each(&g, &rule).unwrap();

    // e1 should match (n1 -> n2, and not exists n1 -> C with prob >= 0.5 where C != n2)
    // Actually wait - the pattern doesn't exclude B, so let me check the logic
    // The not exists checks for A -> C with prob >= 0.5, but C is a new variable
    // So it will match any C != B. Since we have n1 -> n3 with prob 0.8, the not exists should fail
    // So e1 should NOT match

    // Actually, let me reconsider: the not exists pattern has A -> C, and C is a new variable
    // So it will find n1 -> n3 (since C can be n3, and n3 != n2). Since n1 -> n3 exists with prob >= 0.5,
    // the not exists subquery returns false, so the where clause is false, so e1 should NOT be forced absent

    // So e1 should still have high probability
    assert!(result.prob_mean(e1).unwrap() > 0.5);
}

#[test]
fn exists_subquery_with_variable_constraint() {
    let mut g = BeliefGraph::default();

    let n1 = g.add_node("Person".into(), HashMap::new());
    let n2 = g.add_node("Person".into(), HashMap::new());
    let n3 = g.add_node("Person".into(), HashMap::new());

    // Create edges: n1 -> n2, n1 -> n3
    let e1 = g.add_edge(
        n1,
        n2,
        "REL".into(),
        BetaPosterior {
            alpha: 9.0,
            beta: 1.0,
        },
    );
    let e2 = g.add_edge(
        n1,
        n3,
        "REL".into(),
        BetaPosterior {
            alpha: 8.0,
            beta: 2.0,
        },
    );

    let rule = RuleDef {
        name: "ExistsWithConstraint".into(),
        on_model: "M".into(),
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
        where_expr: Some(ExprAst::Exists {
            pattern: PatternItem {
                src: NodePattern {
                    var: "A".into(),
                    label: "Person".into(),
                },
                edge: EdgePattern {
                    var: "ax".into(),
                    ty: "REL".into(),
                },
                dst: NodePattern {
                    var: "X".into(),
                    label: "Person".into(),
                },
            },
            where_expr: Some(Box::new(ExprAst::Binary {
                op: BinaryOp::And,
                left: Box::new(ExprAst::Binary {
                    op: BinaryOp::Ge,
                    left: Box::new(ExprAst::Call {
                        name: "prob".into(),
                        args: vec![CallArg::Positional(ExprAst::Var("ax".into()))],
                    }),
                    right: Box::new(ExprAst::Number(0.8)),
                }),
                // Compare node variables: X != B
                // Node variables are converted to their numeric IDs (NodeId.0 as f64) for comparison
                right: Box::new(ExprAst::Binary {
                    op: BinaryOp::Ne,
                    left: Box::new(ExprAst::Var("X".into())),
                    right: Box::new(ExprAst::Var("B".into())),
                }),
            })),
            negated: false,
        }),
        actions: vec![ActionStmt::ForceAbsent {
            edge_var: "e".into(),
        }],
        mode: Some("for_each".into()),
    };

    let result = run_rule_for_each(&g, &rule).unwrap();

    // For e1 (A=n1, B=n2): exists A -> X with prob >= 0.8 and X != B
    // This should find n1 -> n3 (e2) with prob 0.8 >= 0.8 and n3 != n2, so exists returns true
    // So e1 should be forced absent

    // For e2 (A=n1, B=n3): exists A -> X with prob >= 0.8 and X != B
    // This should find n1 -> n2 (e1) with prob 0.9 >= 0.8 and n2 != n3, so exists returns true
    // So e2 should also be forced absent

    assert!(result.prob_mean(e1).unwrap() < 1e-5);
    assert!(result.prob_mean(e2).unwrap() < 1e-5);
}
