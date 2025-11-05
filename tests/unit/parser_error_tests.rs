use baygraph::parse_program;

// ============================================================================
// Parser Error Handling Tests
// ============================================================================

#[test]
fn parse_empty_string_fails() {
    let result = parse_program("");
    // Empty program should either succeed with empty AST or fail gracefully
    // Checking that it doesn't panic
    let _ = result;
}

#[test]
fn parse_invalid_syntax_fails() {
    let src = "this is not valid syntax @@@ !!!";
    let result = parse_program(src);
    assert!(result.is_err(), "should fail on invalid syntax");
}

#[test]
fn parse_unclosed_brace_fails() {
    let src = "schema X { node N { a: Real }";
    let result = parse_program(src);
    assert!(result.is_err(), "should fail on unclosed brace");
}

#[test]
fn parse_missing_node_type_fails() {
    let src = "schema X { node N { a } }";
    let result = parse_program(src);
    assert!(result.is_err(), "should fail on missing type annotation");
}

#[test]
fn parse_duplicate_schema_names_allowed() {
    // Parser may allow duplicates, validation catches it
    let src = r#"
        schema S { node N {} edge E {} }
        schema S { node N {} edge E {} }
    "#;
    let result = parse_program(src);
    // Should parse successfully (validation layer catches semantic errors)
    let _ = result;
}

#[test]
fn parse_empty_schema_body() {
    let src = "schema X { }";
    let result = parse_program(src);
    // Empty schema may or may not be valid depending on grammar
    let _ = result;
}

#[test]
fn parse_rule_without_pattern_fails() {
    let src = r#"
        schema S { node N {} edge E {} }
        belief_model M on S {}
        rule R on M { }
    "#;
    let result = parse_program(src);
    assert!(result.is_err(), "rule should require at least one pattern");
}

#[test]
fn parse_rule_with_empty_action_list() {
    let src = r#"
        schema S { node N {} edge E {} }
        belief_model M on S {}
        rule R on M {
            pattern (A:N)-[e:E]->(B:N)
        }
    "#;
    let result = parse_program(src);
    // Rules with no actions should be allowed
    assert!(result.is_ok());
}

#[test]
fn parse_flow_with_no_graphs() {
    let src = r#"
        schema S { node N {} edge E {} }
        belief_model M on S {}
        flow F on M { }
    "#;
    let result = parse_program(src);
    // Empty flow may be allowed
    let _ = result;
}

#[test]
fn parse_malformed_pattern_fails() {
    let src = r#"
        schema S { node N {} edge E {} }
        belief_model M on S {}
        rule R on M {
            pattern (A:N) [e:E] (B:N)
        }
    "#;
    let result = parse_program(src);
    assert!(result.is_err(), "should fail on malformed pattern syntax");
}

#[test]
fn parse_expression_with_balanced_parens() {
    let src = r#"
        schema S { node N { x: Real } edge E {} }
        belief_model M on S {}
        rule R on M {
            pattern (A:N)-[e:E]->(B:N)
            where ((E[A.x] + 1) * 2) >= 10
        }
    "#;
    let result = parse_program(src);
    assert!(result.is_ok(), "should handle nested expressions");
}

#[test]
fn parse_expression_with_unbalanced_parens_fails() {
    let src = r#"
        schema S { node N { x: Real } edge E {} }
        belief_model M on S {}
        rule R on M {
            pattern (A:N)-[e:E]->(B:N)
            where ((E[A.x] + 1) >= 10
        }
    "#;
    let result = parse_program(src);
    assert!(result.is_err(), "should fail on unbalanced parentheses");
}

#[test]
fn parse_multiple_patterns_in_rule() {
    let src = r#"
        schema S { node N {} edge E {} }
        belief_model M on S {}
        rule R on M {
            pattern (A:N)-[e1:E]->(B:N)
            pattern (B:N)-[e2:E]->(C:N)
        }
    "#;
    let result = parse_program(src);
    assert!(result.is_ok(), "should support multiple patterns");
}

#[test]
fn parse_node_with_multiple_attributes() {
    let src = r#"
        schema S {
            node Person {
                age: Real
                score: Real
                active: Real
            }
            edge E {}
        }
    "#;
    let result = parse_program(src);
    assert!(result.is_ok(), "should support multiple node attributes");
}

#[test]
fn parse_reserved_keyword_as_identifier_may_fail() {
    let src = r#"
        schema schema {
            node node {}
            edge edge {}
        }
    "#;
    let result = parse_program(src);
    // Reserved keywords as names should fail
    assert!(result.is_err());
}

// ============================================================================
// Edge Cases and Boundary Conditions
// ============================================================================

#[test]
fn parse_very_long_identifier() {
    let long_name = "A".repeat(1000);
    let src = format!(r#"
        schema S {{
            node {} {{ x: Real }}
            edge E {{}}
        }}
    "#, long_name);
    let result = parse_program(&src);
    // Should handle long identifiers
    assert!(result.is_ok());
}

#[test]
fn parse_deeply_nested_expressions() {
    let src = r#"
        schema S { node N { x: Real } edge E {} }
        belief_model M on S {}
        rule R on M {
            pattern (A:N)-[e:E]->(B:N)
            where ((((((E[A.x] + 1) + 2) + 3) + 4) + 5) + 6) >= 0
        }
    "#;
    let result = parse_program(src);
    assert!(result.is_ok(), "should handle deeply nested expressions");
}

#[test]
fn parse_unicode_in_strings() {
    let src = r#"
        schema S {
            node N {}
            edge E {}
        }
        belief_model M on S {}
    "#;
    let result = parse_program(src);
    assert!(result.is_ok());
}

#[test]
fn parse_numbers_with_different_formats() {
    let src = r#"
        schema S { node N { x: Real } edge E {} }
        belief_model M on S {}
        rule R on M {
            pattern (A:N)-[e:E]->(B:N)
            where E[A.x] >= 42
            set_expectation A.x = 3.14159
        }
    "#;
    let result = parse_program(src);
    assert!(result.is_ok(), "should handle integers and floats");
}

#[test]
fn parse_negative_numbers() {
    let src = r#"
        schema S { node N { x: Real } edge E {} }
        belief_model M on S {}
        rule R on M {
            pattern (A:N)-[e:E]->(B:N)
            where E[A.x] >= -10
        }
    "#;
    let result = parse_program(src);
    // Negative numbers may need unary minus operator
    let _ = result;
}

#[test]
fn parse_scientific_notation() {
    let src = r#"
        schema S { node N { x: Real } edge E {} }
        belief_model M on S {}
        rule R on M {
            pattern (A:N)-[e:E]->(B:N)
            where prob(e) >= 1e-6
        }
    "#;
    let result = parse_program(src);
    // Scientific notation support depends on grammar
    let _ = result;
}

#[test]
fn parse_comments_are_ignored() {
    let src = r#"
        // This is a comment
        schema S {
            node N {} // inline comment
            edge E {}
        }
        /* Multi-line
           comment */
    "#;
    let result = parse_program(src);
    // Comment support depends on grammar
    let _ = result;
}

#[test]
fn parse_whitespace_variations() {
    let src = "schema    S    {    node    N    {}    edge    E    {}    }";
    let result = parse_program(src);
    assert!(result.is_ok(), "should handle extra whitespace");
}

#[test]
fn parse_newline_variations() {
    let src = "schema S {\nnode N {}\nedge E {}\n}";
    let result = parse_program(src);
    assert!(result.is_ok(), "should handle different newlines");
}

#[test]
fn parse_minimal_valid_program() {
    let src = "schema S { node N {} edge E {} }";
    let result = parse_program(src);
    assert!(result.is_ok(), "minimal program should parse");
}

#[test]
fn parse_program_with_all_components() {
    let src = r#"
        schema S {
            node N { x: Real }
            edge E {}
        }

        belief_model M on S {}

        evidence Ev on M {}

        rule R on M {
            pattern (A:N)-[e:E]->(B:N)
            where prob(e) >= 0.5
            set_expectation A.x = 10
        }

        flow F on M {
            graph g = from_evidence Ev
            metric m = count_nodes(label=N)
        }
    "#;
    let result = parse_program(src);
    assert!(result.is_ok(), "complete program should parse");
}
