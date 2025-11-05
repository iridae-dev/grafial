//! The frontend module handles parsing and validation of Baygraph DSL source code.
//!
//! This module provides:
//! - **parser**: Transforms source text into an Abstract Syntax Tree (AST)
//! - **ast**: Type definitions for the AST structure
//! - **validate**: Semantic validation of parsed programs

pub mod parser;
pub mod ast;
pub mod validate;
