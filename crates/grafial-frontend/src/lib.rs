//! # Grafial Frontend
//!
//! Parser, AST, and validation for the Grafial DSL.

pub mod ast;
pub mod errors;
pub mod lint;
pub mod parser;
pub mod style;
pub mod validate;

// Re-export commonly used types
pub use ast::*;
pub use errors::{
    FrontendError, SourcePosition, SourceRange, ValidationContext, ValidationDiagnostic,
};
pub use lint::{
    collect_lint_suppressions, lint_is_suppressed, lint_statistical_guardrails, LintSeverity,
    LintSuppression, StatisticalLint, LINT_STAT_CIRCULAR_UPDATE, LINT_STAT_DELETE_EXPLANATION,
    LINT_STAT_MULTIPLE_TESTING, LINT_STAT_NUMERICAL_INSTABILITY, LINT_STAT_PRECISION_OUTLIER,
    LINT_STAT_PRIOR_DATA_CONFLICT, LINT_STAT_PRIOR_DOMINANCE, LINT_STAT_SUPPRESS_EXPLANATION,
    LINT_STAT_VARIANCE_COLLAPSE,
};
pub use parser::parse_program;
pub use style::{format_canonical_style, lint_canonical_style, CanonicalStyleLint};
pub use validate::{validate_program, validate_program_with_source};
