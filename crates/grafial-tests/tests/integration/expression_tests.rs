use grafial_frontend::{parse_program, validate_program};

#[test]
fn invalid_prob_on_node_var_in_rule_where() {
    let src = r#"
schema S { node N { x: Real } edge E {} }
belief_model M on S { }
rule R on M {
  pattern (A:N)-[e:E]->(B:N)
  where prob(A) >= 0.5
}
flow F on M { graph g = from_evidence X }
"#;
    let ast = parse_program(src).expect("parse");
    let err = validate_program(&ast).expect_err("should fail validation");
    let msg = format!("{}", err);
    assert!(msg.contains("prob()"));
}

#[test]
fn invalid_prune_edges_prob_arg() {
    let src = r#"
schema S { node N { x: Real } edge E {} }
belief_model M on S { }
rule R on M { pattern (A:N)-[e:E]->(B:N) }
flow F on M {
  graph base = from_evidence X
  graph c = base |> prune_edges E where prob(e) < 0.1
}
"#;
    let ast = parse_program(src).expect("parse");
    let err = validate_program(&ast).expect_err("should fail validation");
    let msg = format!("{}", err);
    assert!(msg.contains("prune_edges predicate"));
}

#[test]
fn invalid_sum_nodes_missing_label() {
    let src = r#"
schema S { node N { x: Real } edge E {} }
belief_model M on S { }
flow F on M {
  graph g = from_evidence X
  metric m = sum_nodes(contrib=1)
}
"#;
    let ast = parse_program(src).expect("parse");
    let err = validate_program(&ast).expect_err("should fail validation");
    assert!(format!("{}", err).contains("sum_nodes"));
}

#[test]
fn valid_avg_degree_metric_and_rule_where() {
    let src = include_str!("../../../grafial-examples/social.grafial");
    let ast = parse_program(src).expect("parse");
    validate_program(&ast).expect("validation");
}

#[test]
fn valid_prob_correlated_rule_where() {
    let src = r#"
schema S { node N { x: Real } edge E {} }
belief_model M on S { }
rule R on M {
  pattern (A:N)-[e:E]->(B:N)
  where prob_correlated(A.x > B.x, rho=0.3) >= 0.5
}
flow F on M { graph g = from_evidence X }
"#;
    let ast = parse_program(src).expect("parse");
    validate_program(&ast).expect("validation");
}

#[test]
fn valid_credible_rule_where() {
    let src = r#"
schema S { node N { x: Real } edge E {} }
belief_model M on S { }
rule R on M {
  pattern (A:N)-[e:E]->(B:N)
  where credible(e, p=0.8) and credible(A.x > B.x, p=0.7, rho=0.2)
}
flow F on M { graph g = from_evidence X }
"#;
    let ast = parse_program(src).expect("parse");
    validate_program(&ast).expect("validation");
}
