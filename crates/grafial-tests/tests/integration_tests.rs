//! Integration tests module that includes all integration test files.

#[path = "integration/engine_tests.rs"]
mod engine_tests;

#[path = "integration/expression_tests.rs"]
mod expression_tests;

#[path = "integration/flow_tests.rs"]
mod flow_tests;

#[path = "integration/parser_tests.rs"]
mod parser_tests;

#[path = "integration/rule_where_tests.rs"]
mod rule_where_tests;

#[path = "integration/snapshot_tests.rs"]
mod snapshot_tests;
