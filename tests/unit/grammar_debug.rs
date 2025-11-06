use grafial::parse_program;

#[test]
fn test_program_rule() {
    let src = "schema X { node N { a: Real } edge E {} }";
    let result = parse_program(src);
    match result {
        Ok(ast) => {
            println!("Success! Parsed program");
            println!("  Schemas: {}", ast.schemas.len());
            if let Some(schema) = ast.schemas.first() {
                println!("  First schema: {}", schema.name);
                println!("  Nodes: {}", schema.nodes.len());
                println!("  Edges: {}", schema.edges.len());
            }
        }
        Err(e) => {
            println!("Error parsing program: {}", e);
            panic!("Failed to parse program: {}", e);
        }
    }
}

#[test]
fn test_minimal_schema() {
    let src = "schema X { node N { a: Real } edge E {} }";
    let result = parse_program(src);
    assert!(result.is_ok(), "Should parse minimal schema: {:?}", result.err());
    let ast = result.unwrap();
    assert_eq!(ast.schemas.len(), 1);
    assert_eq!(ast.schemas[0].name, "X");
}

