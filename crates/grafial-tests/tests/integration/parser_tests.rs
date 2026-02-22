use grafial_frontend::parse_program;

#[test]
fn parses_min_schema() {
    let src = "schema X { node N { a: Real } edge E {} }";
    let ast = parse_program(src).expect("parse min schema");
    assert_eq!(ast.schemas.len(), 1);
}

#[test]
fn parses_social_example() {
    let src = include_str!("../../../grafial-examples/social.grafial");
    let ast = parse_program(src).expect("parse");
    // Basic shape checks
    assert_eq!(ast.schemas.len(), 1);
    assert_eq!(ast.belief_models.len(), 1);
    assert_eq!(ast.evidences.len(), 1);
    assert!(!ast.rules.is_empty());
    assert_eq!(ast.flows.len(), 1);

    let schema = &ast.schemas[0];
    assert_eq!(schema.name, "Social");
    assert_eq!(schema.nodes.len(), 1);
    assert_eq!(schema.edges.len(), 1);

    let rule = ast
        .rules
        .iter()
        .find(|r| r.name == "TransferAndDisconnect")
        .expect("expected TransferAndDisconnect rule");
    assert_eq!(rule.name, "TransferAndDisconnect");
    assert_eq!(rule.patterns.len(), 2);
    assert!(rule.where_expr.is_some());

    let flow = &ast.flows[0];
    assert_eq!(flow.name, "Demo");
    assert_eq!(flow.graphs.len(), 3); // base, transferred, cleaned
    assert_eq!(flow.metrics.len(), 1);
    assert_eq!(flow.exports.len(), 1);
    // Check metric and transform expressions parsed
    assert!(!flow.metrics[0].name.is_empty());
    // flow.graphs[1] is "transferred" which has 1 transform (apply_rule)
    // flow.graphs[2] is "cleaned" which also has 1 transform (prune_edges)
    match &flow.graphs[1].expr {
        grafial_frontend::ast::GraphExpr::Pipeline { transforms, .. } => {
            assert_eq!(transforms.len(), 1); // transferred has one transform
        }
        _ => panic!("expected pipeline"),
    }
}

#[test]
fn parses_exists_and_not_exists_syntax() {
    // Ensure textual `exists` and `not exists` in where clauses parse correctly
    let src = include_str!("../../../grafial-examples/probabilistic_pattern_matching.grafial");
    let ast = parse_program(src).expect("parse probabilistic_pattern_matching");
    assert!(ast.rules.iter().any(|r| r.name == "IndirectInfluence"));
}
