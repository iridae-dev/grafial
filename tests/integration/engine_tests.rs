#[test]
fn placeholder_engine_compiles() {
    // Placeholder: ensure engine types are visible and basic newtypes exist
    let _nid = baygraph::engine::graph::NodeId(1);
    let _eid = baygraph::engine::graph::EdgeId(1);
    assert_eq!(_nid.0, 1);
    assert_eq!(_eid.0, 1);
}

#[test]
fn actions_set_expectation_and_force_absent() {
    use baygraph::engine::graph::*;
    use baygraph::engine::rule_exec::*;
    use baygraph::ast::{ActionStmt, ExprAst, BinaryOp, CallArg};
    use std::collections::HashMap;

    // Build a tiny graph with two nodes A, B and one edge bc.
    let mut g = BeliefGraph::default();
    g.insert_node(NodeData {
        id: NodeId(1),
        label: "Person".into(),
        attrs: HashMap::from([
            (
                "some_value".into(),
                GaussianPosterior { mean: 10.0, precision: 1.0 },
            ),
            (
                "other_value".into(),
                GaussianPosterior { mean: 0.0, precision: 1.0 },
            ),
        ]),
    });
    g.insert_node(NodeData {
        id: NodeId(2),
        label: "Person".into(),
        attrs: HashMap::from([
            (
                "some_value".into(),
                GaussianPosterior { mean: 0.0, precision: 1.0 },
            ),
            (
                "other_value".into(),
                GaussianPosterior { mean: 0.0, precision: 1.0 },
            ),
        ]),
    });
    g.insert_edge(EdgeData {
        id: EdgeId(42),
        src: NodeId(2),
        dst: NodeId(1),
        ty: "REL".into(),
        exist: BetaPosterior { alpha: 1.0, beta: 1.0 },
    });

    // let v_ab = E[A.some_value] / 2
    let v_ab_expr = ExprAst::Binary {
        op: BinaryOp::Div,
        left: Box::new(ExprAst::Call {
            name: "E".into(),
            args: vec![CallArg::Positional(ExprAst::Field {
                target: Box::new(ExprAst::Var("A".into())),
                field: "some_value".into(),
            })],
        }),
        right: Box::new(ExprAst::Number(2.0)),
    };
    let actions = vec![
        ActionStmt::Let { name: "v_ab".into(), expr: v_ab_expr.clone() },
        ActionStmt::SetExpectation {
            node_var: "A".into(),
            attr: "some_value".into(),
            expr: ExprAst::Binary {
                op: BinaryOp::Sub,
                left: Box::new(ExprAst::Call {
                    name: "E".into(),
                    args: vec![CallArg::Positional(ExprAst::Field {
                        target: Box::new(ExprAst::Var("A".into())),
                        field: "some_value".into(),
                    })],
                }),
                right: Box::new(ExprAst::Var("v_ab".into())),
            },
        },
        ActionStmt::SetExpectation {
            node_var: "B".into(),
            attr: "some_value".into(),
            expr: ExprAst::Binary {
                op: BinaryOp::Add,
                left: Box::new(ExprAst::Call {
                    name: "E".into(),
                    args: vec![CallArg::Positional(ExprAst::Field {
                        target: Box::new(ExprAst::Var("B".into())),
                        field: "some_value".into(),
                    })],
                }),
                right: Box::new(ExprAst::Var("v_ab".into())),
            },
        },
        ActionStmt::ForceAbsent { edge_var: "bc".into() },
    ];

    let mut bindings = MatchBindings::default();
    bindings.node_vars.insert("A".into(), NodeId(1));
    bindings.node_vars.insert("B".into(), NodeId(2));
    bindings.edge_vars.insert("bc".into(), EdgeId(42));

    execute_actions(&mut g, &actions, &bindings).expect("actions");

    // A.some_value: 10 -> 5, B.some_value: 0 -> 5
    assert!((g.expectation(NodeId(1), "some_value").unwrap() - 5.0).abs() < 1e-9);
    assert!((g.expectation(NodeId(2), "some_value").unwrap() - 5.0).abs() < 1e-9);
    // Edge 42 forced absent
    let e = g.edge(EdgeId(42)).unwrap();
    assert_eq!(e.exist.alpha, 1.0);
    assert!(e.exist.beta >= 1e6 - 1.0);
}

#[test]
fn run_rule_for_each_single_pattern() {
    use baygraph::engine::graph::*;
    use baygraph::engine::rule_exec::run_rule_for_each;
    use baygraph::ast::{ActionStmt, ExprAst, BinaryOp, CallArg, RuleDef, PatternItem, NodePattern, EdgePattern};
    use std::collections::HashMap;

    // Graph: A(1) -> B(2) edge e with prob ~ 0.6
    let mut g = BeliefGraph::default();
    g.insert_node(NodeData {
        id: NodeId(1),
        label: "Person".into(),
        attrs: HashMap::from([
            ("some_value".into(), GaussianPosterior { mean: 10.0, precision: 1.0 })
        ]),
    });
    g.insert_node(NodeData {
        id: NodeId(2),
        label: "Person".into(),
        attrs: HashMap::from([("some_value".into(), GaussianPosterior { mean: 0.0, precision: 1.0 })]),
    });
    g.insert_edge(EdgeData {
        id: EdgeId(7), src: NodeId(1), dst: NodeId(2), ty: "REL".into(),
        exist: BetaPosterior { alpha: 3.0, beta: 2.0 }, // mean 0.6
    });

    // Rule: pattern (A:Person)-[e:REL]->(B:Person)
    // where prob(e) >= 0.5
    // action: set_expectation A.some_value = E[A.some_value] - 1; force_absent e
    let rule = RuleDef {
        name: "R".into(), on_model: "M".into(), mode: Some("for_each".into()),
        patterns: vec![PatternItem {
            src: NodePattern { var: "A".into(), label: "Person".into() },
            edge: EdgePattern { var: "e".into(), ty: "REL".into() },
            dst: NodePattern { var: "B".into(), label: "Person".into() },
        }],
        where_expr: Some(ExprAst::Binary {
            op: BinaryOp::Ge,
            left: Box::new(ExprAst::Call { name: "prob".into(), args: vec![CallArg::Positional(ExprAst::Var("e".into()))] }),
            right: Box::new(ExprAst::Number(0.5)),
        }),
        actions: vec![
            ActionStmt::SetExpectation {
                node_var: "A".into(), attr: "some_value".into(),
                expr: ExprAst::Binary {
                    op: BinaryOp::Sub,
                    left: Box::new(ExprAst::Call {
                        name: "E".into(),
                        args: vec![CallArg::Positional(ExprAst::Field {
                            target: Box::new(ExprAst::Var("A".into())), field: "some_value".into()
                        })]
                    }),
                    right: Box::new(ExprAst::Number(1.0)),
                },
            },
            ActionStmt::ForceAbsent { edge_var: "e".into() },
        ],
    };

    let out = run_rule_for_each(&g, &rule).expect("run rule");
    // Mean adjusted
    assert!((out.expectation(NodeId(1), "some_value").unwrap() - 9.0).abs() < 1e-9);
    // Edge forced absent
    let e = out.edge(EdgeId(7)).unwrap();
    assert_eq!(e.exist.alpha, 1.0);
    assert!(e.exist.beta >= 1e6 - 1.0);
}

#[test]
fn evidence_application_updates_posteriors() {
    use baygraph::engine::graph::*;
    use std::collections::HashMap;

    let mut g = BeliefGraph::default();
    g.insert_node(NodeData {
        id: NodeId(1),
        label: "Person".into(),
        attrs: HashMap::from([("x".into(), GaussianPosterior { mean: 0.0, precision: 0.01 })]),
    });
    g.insert_edge(EdgeData {
        id: EdgeId(10), src: NodeId(1), dst: NodeId(1), ty: "REL".into(),
        exist: BetaPosterior { alpha: 1.0, beta: 1.0 },
    });

    // Observe attr value: prior mean 0, precision 0.01; observe x=10 with tau_obs=1 => mean > 0
    g.observe_attr(NodeId(1), "x", 10.0, 1.0).expect("observe attr");
    let m = g.expectation(NodeId(1), "x").unwrap();
    assert!(m > 0.0);

    // Force value clamps precision large
    g.force_attr_value(NodeId(1), "x", 5.0).expect("force attr");
    assert!((g.expectation(NodeId(1), "x").unwrap() - 5.0).abs() < 1e-9);

    // Observe edge present increments alpha
    g.observe_edge(EdgeId(10), true).expect("present");
    assert!(g.prob_mean(EdgeId(10)).unwrap() > 0.5);

    // Force absent
    g.force_absent(EdgeId(10)).expect("force absent");
    let p = g.prob_mean(EdgeId(10)).unwrap();
    assert!(p < 1e-5);
}
