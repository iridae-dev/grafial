use baygraph::parse_program;

#[test]
fn parses_min_schema() {
    let src = "schema X { node N { a: Real } edge E {} }";
    let ast = parse_program(src).expect("parse min schema");
    assert_eq!(ast.schemas.len(), 1);
}

#[test]
fn parses_social_example() {
    let src = std::fs::read_to_string("examples/social.bg").expect("read example");
    let ast = parse_program(&src).expect("parse");
    // Basic shape checks
    assert_eq!(ast.schemas.len(), 1);
    assert_eq!(ast.belief_models.len(), 1);
    assert_eq!(ast.evidences.len(), 1);
    assert_eq!(ast.rules.len(), 1);
    assert_eq!(ast.flows.len(), 1);

    let schema = &ast.schemas[0];
    assert_eq!(schema.name, "Social");
    assert_eq!(schema.nodes.len(), 1);
    assert_eq!(schema.edges.len(), 1);

    let rule = &ast.rules[0];
    assert_eq!(rule.name, "TransferAndDisconnect");
    assert_eq!(rule.patterns.len(), 2);
    assert!(rule.where_expr.is_some());

    let flow = &ast.flows[0];
    assert_eq!(flow.name, "Demo");
    assert_eq!(flow.graphs.len(), 2);
    assert_eq!(flow.metrics.len(), 1);
    assert_eq!(flow.exports.len(), 1);
    // Check metric and transform expressions parsed
    assert!(!flow.metrics[0].name.is_empty());
    match &flow.graphs[1].expr {
        baygraph::ast::GraphExpr::Pipeline { transforms, .. } => {
            assert_eq!(transforms.len(), 2);
        }
        _ => panic!("expected pipeline"),
    }
}
