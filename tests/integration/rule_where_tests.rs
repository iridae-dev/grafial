use baygraph::engine::graph::*;
use baygraph::engine::rule_exec::run_rule_for_each;
use baygraph::ast::{ActionStmt, ExprAst, BinaryOp, CallArg, RuleDef, PatternItem, NodePattern, EdgePattern};
use std::collections::HashMap;

fn mk_graph_for_degree() -> BeliefGraph {
    // Node B has two outgoing edges e1 (p~0.8) and e2 (p~0.4)
    let mut g = BeliefGraph::default();
    g.insert_node(NodeData { id: NodeId(1), label: "Person".into(), attrs: HashMap::new() }); // A
    g.insert_node(NodeData { id: NodeId(2), label: "Person".into(), attrs: HashMap::new() }); // B
    g.insert_node(NodeData { id: NodeId(3), label: "Person".into(), attrs: HashMap::new() }); // C
    g.insert_node(NodeData { id: NodeId(4), label: "Person".into(), attrs: HashMap::new() }); // D
    // Pattern edge from A->B
    g.insert_edge(EdgeData { id: EdgeId(10), src: NodeId(1), dst: NodeId(2), ty: "REL".into(), exist: BetaPosterior { alpha: 1.0, beta: 1.0 } });
    // B -> C with high prob ~ 0.8
    g.insert_edge(EdgeData { id: EdgeId(11), src: NodeId(2), dst: NodeId(3), ty: "REL".into(), exist: BetaPosterior { alpha: 8.0, beta: 2.0 } });
    // B -> D with lower prob ~ 0.4
    g.insert_edge(EdgeData { id: EdgeId(12), src: NodeId(2), dst: NodeId(4), ty: "REL".into(), exist: BetaPosterior { alpha: 2.0, beta: 3.0 } });
    g
}

#[test]
fn where_degree_filters_matches() {
    let g = mk_graph_for_degree();

    // Rule matches (A)-[e]->(B); where degree(B, min_prob=0.7) >= 1
    let rule = RuleDef {
        name: "Rdeg".into(), on_model: "M".into(), mode: Some("for_each".into()),
        patterns: vec![PatternItem {
            src: NodePattern { var: "A".into(), label: "Person".into() },
            edge: EdgePattern { var: "e".into(), ty: "REL".into() },
            dst: NodePattern { var: "B".into(), label: "Person".into() },
        }],
        where_expr: Some(ExprAst::Binary {
            op: BinaryOp::Ge,
            left: Box::new(ExprAst::Call { name: "degree".into(), args: vec![
                CallArg::Positional(ExprAst::Var("B".into())),
                CallArg::Named { name: "min_prob".into(), value: ExprAst::Number(0.7) },
            ] }),
            right: Box::new(ExprAst::Number(1.0)),
        }),
        actions: vec![ActionStmt::ForceAbsent { edge_var: "e".into() }],
    };

    let out = run_rule_for_each(&g, &rule).expect("run rule");
    // Only (A=1,B=2) pattern exists and B meets degree condition; edge 10 should be forced absent
    let e = out.edge(EdgeId(10)).unwrap();
    assert!(e.exist.beta >= 1e6 - 1.0);
}

#[test]
fn where_prob_blocks_action_when_below_threshold() {
    // Same graph, but require prob(e) >= 0.9; e has prob 0.5 initially, so no action
    let mut g = mk_graph_for_degree();
    // Ensure e has Beta(1,1) -> p=0.5
    let e0 = g.edge(EdgeId(10)).unwrap();
    assert!(((e0.exist.alpha / (e0.exist.alpha + e0.exist.beta)) - 0.5).abs() < 1e-9);

    let rule = RuleDef {
        name: "Rprob".into(), on_model: "M".into(), mode: Some("for_each".into()),
        patterns: vec![PatternItem {
            src: NodePattern { var: "A".into(), label: "Person".into() },
            edge: EdgePattern { var: "e".into(), ty: "REL".into() },
            dst: NodePattern { var: "B".into(), label: "Person".into() },
        }],
        where_expr: Some(ExprAst::Binary {
            op: BinaryOp::Ge,
            left: Box::new(ExprAst::Call { name: "prob".into(), args: vec![CallArg::Positional(ExprAst::Var("e".into()))] }),
            right: Box::new(ExprAst::Number(0.9)),
        }),
        actions: vec![ActionStmt::ForceAbsent { edge_var: "e".into() }],
    };

    let out = run_rule_for_each(&g, &rule).expect("run rule");
    // Edge should remain unchanged (not forced absent)
    let e = out.edge(EdgeId(10)).unwrap();
    assert!(e.exist.beta < 1e6 - 1.0);
}

