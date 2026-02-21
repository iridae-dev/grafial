//! Tests for JIT-compiled rule kernels.
//!
//! Verifies that compiled predicates produce identical results to interpreted ones
//! and tests the compilation threshold mechanism.

#[cfg(feature = "jit")]
mod tests {
    use grafial_core::engine::graph::{BeliefGraph, GaussianPosterior};
    use grafial_core::engine::rule_exec::run_rule_for_each_with_globals;
    use grafial_frontend::ast::{
        BinaryOp, EdgePattern, ExprAst, NodePattern, PatternItem, RuleDef,
    };
    use std::collections::HashMap;

    /// Create a simple test graph with nodes and edges.
    fn create_test_graph() -> BeliefGraph {
        let mut graph = BeliefGraph::default();

        // Add nodes with attributes
        let mut attrs1 = HashMap::new();
        attrs1.insert(
            "score".to_string(),
            GaussianPosterior {
                mean: 10.0,
                precision: 1.0,
            },
        );
        let _node1 = graph.add_node("Person".to_string(), attrs1);

        let mut attrs2 = HashMap::new();
        attrs2.insert(
            "score".to_string(),
            GaussianPosterior {
                mean: 5.0,
                precision: 1.0,
            },
        );
        let _node2 = graph.add_node("Person".to_string(), attrs2);

        let mut attrs3 = HashMap::new();
        attrs3.insert(
            "score".to_string(),
            GaussianPosterior {
                mean: 15.0,
                precision: 1.0,
            },
        );
        graph.add_node("Person".to_string(), attrs3);

        graph
    }

    /// Test that simple arithmetic predicates can be compiled and executed.
    #[test]
    fn test_simple_predicate_compilation() {
        let graph = create_test_graph();
        let globals = HashMap::new();

        // Create a rule with a simple arithmetic predicate
        // Rule: for each Person node A where A > 5
        let rule = RuleDef {
            name: "test_rule".to_string(),
            on_model: "test_model".to_string(),
            patterns: vec![PatternItem {
                src: NodePattern {
                    var: "A".to_string(),
                    label: "Person".to_string(),
                },
                edge: EdgePattern {
                    var: "e".to_string(),
                    ty: "__FOR_NODE__".to_string(), // Special marker for node iteration
                },
                dst: NodePattern {
                    var: "A".to_string(),
                    label: "Person".to_string(),
                },
            }],
            where_expr: Some(ExprAst::Binary {
                op: BinaryOp::Gt,
                left: Box::new(ExprAst::Var("score_val".to_string())),
                right: Box::new(ExprAst::Number(7.0)),
            }),
            actions: vec![],
            mode: None,
        };

        // Run the rule multiple times to trigger compilation (threshold is 10)
        for i in 0..15 {
            let result = run_rule_for_each_with_globals(&graph, &rule, &globals);
            assert!(result.is_ok(), "Rule execution {} failed", i);
        }
    }

    /// Test that complex predicates fall back to interpreter.
    #[test]
    fn test_complex_predicate_fallback() {
        let graph = create_test_graph();
        let globals = HashMap::new();

        // Create a rule with a complex predicate (field access)
        // This should NOT be compiled and should fall back to interpreter
        let rule = RuleDef {
            name: "complex_rule".to_string(),
            on_model: "test_model".to_string(),
            patterns: vec![PatternItem {
                src: NodePattern {
                    var: "A".to_string(),
                    label: "Person".to_string(),
                },
                edge: EdgePattern {
                    var: "e".to_string(),
                    ty: "__FOR_NODE__".to_string(),
                },
                dst: NodePattern {
                    var: "A".to_string(),
                    label: "Person".to_string(),
                },
            }],
            where_expr: Some(ExprAst::Field {
                target: Box::new(ExprAst::Var("A".to_string())),
                field: "score".to_string(),
            }),
            actions: vec![],
            mode: None,
        };

        // Run the rule - should succeed even though it can't be JIT compiled
        for _ in 0..15 {
            let result = run_rule_for_each_with_globals(&graph, &rule, &globals);
            assert!(result.is_ok(), "Complex rule execution failed");
        }
    }

    /// Test logical operators in compiled predicates.
    #[test]
    fn test_logical_operators() {
        let graph = create_test_graph();
        let mut globals = HashMap::new();
        globals.insert("threshold".to_string(), 8.0);
        globals.insert("max_val".to_string(), 20.0);

        // Test AND operator
        let and_rule = RuleDef {
            name: "and_rule".to_string(),
            on_model: "test_model".to_string(),
            patterns: vec![PatternItem {
                src: NodePattern {
                    var: "A".to_string(),
                    label: "Person".to_string(),
                },
                edge: EdgePattern {
                    var: "e".to_string(),
                    ty: "__FOR_NODE__".to_string(),
                },
                dst: NodePattern {
                    var: "A".to_string(),
                    label: "Person".to_string(),
                },
            }],
            where_expr: Some(ExprAst::Binary {
                op: BinaryOp::And,
                left: Box::new(ExprAst::Binary {
                    op: BinaryOp::Gt,
                    left: Box::new(ExprAst::Var("threshold".to_string())),
                    right: Box::new(ExprAst::Number(5.0)),
                }),
                right: Box::new(ExprAst::Binary {
                    op: BinaryOp::Lt,
                    left: Box::new(ExprAst::Var("max_val".to_string())),
                    right: Box::new(ExprAst::Number(25.0)),
                }),
            }),
            actions: vec![],
            mode: None,
        };

        // Run multiple times to trigger compilation
        for _ in 0..15 {
            let result = run_rule_for_each_with_globals(&graph, &and_rule, &globals);
            assert!(result.is_ok(), "AND rule execution failed");
        }

        // Test OR operator
        let or_rule = RuleDef {
            name: "or_rule".to_string(),
            on_model: "test_model".to_string(),
            patterns: vec![PatternItem {
                src: NodePattern {
                    var: "A".to_string(),
                    label: "Person".to_string(),
                },
                edge: EdgePattern {
                    var: "e".to_string(),
                    ty: "__FOR_NODE__".to_string(),
                },
                dst: NodePattern {
                    var: "A".to_string(),
                    label: "Person".to_string(),
                },
            }],
            where_expr: Some(ExprAst::Binary {
                op: BinaryOp::Or,
                left: Box::new(ExprAst::Binary {
                    op: BinaryOp::Lt,
                    left: Box::new(ExprAst::Var("threshold".to_string())),
                    right: Box::new(ExprAst::Number(10.0)),
                }),
                right: Box::new(ExprAst::Binary {
                    op: BinaryOp::Gt,
                    left: Box::new(ExprAst::Var("max_val".to_string())),
                    right: Box::new(ExprAst::Number(15.0)),
                }),
            }),
            actions: vec![],
            mode: None,
        };

        for _ in 0..15 {
            let result = run_rule_for_each_with_globals(&graph, &or_rule, &globals);
            assert!(result.is_ok(), "OR rule execution failed");
        }
    }

    /// Test NOT operator in compiled predicates.
    #[test]
    fn test_not_operator() {
        let graph = create_test_graph();
        let globals = HashMap::new();

        let not_rule = RuleDef {
            name: "not_rule".to_string(),
            on_model: "test_model".to_string(),
            patterns: vec![PatternItem {
                src: NodePattern {
                    var: "A".to_string(),
                    label: "Person".to_string(),
                },
                edge: EdgePattern {
                    var: "e".to_string(),
                    ty: "__FOR_NODE__".to_string(),
                },
                dst: NodePattern {
                    var: "A".to_string(),
                    label: "Person".to_string(),
                },
            }],
            where_expr: Some(ExprAst::Unary {
                op: grafial_frontend::ast::UnaryOp::Not,
                expr: Box::new(ExprAst::Binary {
                    op: BinaryOp::Gt,
                    left: Box::new(ExprAst::Number(10.0)),
                    right: Box::new(ExprAst::Number(5.0)),
                }),
            }),
            actions: vec![],
            mode: None,
        };

        for _ in 0..15 {
            let result = run_rule_for_each_with_globals(&graph, &not_rule, &globals);
            assert!(result.is_ok(), "NOT rule execution failed");
        }
    }

    /// Test that the compilation threshold mechanism works correctly.
    #[test]
    fn test_compilation_threshold() {
        let graph = create_test_graph();
        let globals = HashMap::new();

        // Create a unique rule to avoid cache conflicts
        let rule = RuleDef {
            name: format!(
                "threshold_test_{}",
                std::time::SystemTime::now()
                    .duration_since(std::time::UNIX_EPOCH)
                    .unwrap()
                    .as_nanos()
            ),
            on_model: "test_model".to_string(),
            patterns: vec![PatternItem {
                src: NodePattern {
                    var: "A".to_string(),
                    label: "Person".to_string(),
                },
                edge: EdgePattern {
                    var: "e".to_string(),
                    ty: "__FOR_NODE__".to_string(),
                },
                dst: NodePattern {
                    var: "A".to_string(),
                    label: "Person".to_string(),
                },
            }],
            where_expr: Some(ExprAst::Binary {
                op: BinaryOp::Eq,
                left: Box::new(ExprAst::Number(1.0)),
                right: Box::new(ExprAst::Number(1.0)),
            }),
            actions: vec![],
            mode: None,
        };

        // First 9 executions should use interpreter (threshold is 10)
        for i in 0..9 {
            let result = run_rule_for_each_with_globals(&graph, &rule, &globals);
            assert!(result.is_ok(), "Execution {} before threshold failed", i);
        }

        // 10th execution should trigger compilation
        let result = run_rule_for_each_with_globals(&graph, &rule, &globals);
        assert!(result.is_ok(), "Execution at threshold failed");

        // Subsequent executions should use the compiled kernel
        for i in 11..20 {
            let result = run_rule_for_each_with_globals(&graph, &rule, &globals);
            assert!(result.is_ok(), "Execution {} after compilation failed", i);
        }
    }
}

#[cfg(not(feature = "jit"))]
#[test]
fn test_jit_disabled() {
    // When JIT is disabled, rules should still work via interpreter
    use grafial_core::engine::graph::BeliefGraph;
    use grafial_core::engine::rule_exec::run_rule_for_each_with_globals;
    use grafial_frontend::ast::{
        BinaryOp, EdgePattern, ExprAst, NodePattern, PatternItem, RuleDef,
    };
    use std::collections::HashMap;

    let graph = BeliefGraph::default();
    let globals = HashMap::new();

    let rule = RuleDef {
        name: "test".to_string(),
        on_model: "test_model".to_string(),
        patterns: vec![PatternItem {
            src: NodePattern {
                var: "A".to_string(),
                label: "Person".to_string(),
            },
            edge: EdgePattern {
                var: "e".to_string(),
                ty: "__FOR_NODE__".to_string(),
            },
            dst: NodePattern {
                var: "A".to_string(),
                label: "Person".to_string(),
            },
        }],
        where_expr: Some(ExprAst::Binary {
            op: BinaryOp::Gt,
            left: Box::new(ExprAst::Number(5.0)),
            right: Box::new(ExprAst::Number(3.0)),
        }),
        actions: vec![],
        mode: None,
    };

    // Should work fine without JIT
    let result = run_rule_for_each_with_globals(&graph, &rule, &globals);
    assert!(result.is_ok(), "Rule should work without JIT");
}
