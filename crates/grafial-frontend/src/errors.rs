//! Error types for parsing and validation.

use std::fmt;

use thiserror::Error;

/// 1-based source position.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct SourcePosition {
    pub line: u32,
    pub column: u32,
}

/// Source range with inclusive start and end positions.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct SourceRange {
    pub start: SourcePosition,
    pub end: SourcePosition,
}

/// Semantic validation context.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum ValidationContext {
    BeliefModel {
        model: String,
    },
    Evidence {
        evidence: String,
    },
    RuleWhere {
        rule: String,
    },
    RuleAction {
        rule: String,
        action: String,
    },
    MetricExpr {
        flow: String,
        metric: String,
    },
    PrunePredicate {
        flow: String,
        graph: String,
        edge_type: String,
    },
}

impl fmt::Display for ValidationContext {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::BeliefModel { model } => write!(f, "belief_model '{}'", model),
            Self::Evidence { evidence } => write!(f, "evidence '{}'", evidence),
            Self::RuleWhere { rule } => write!(f, "rule '{}' where clause", rule),
            Self::RuleAction { rule, action } => {
                write!(f, "rule '{}' action '{}'", rule, action)
            }
            Self::MetricExpr { flow, metric } => {
                write!(f, "flow '{}' metric '{}'", flow, metric)
            }
            Self::PrunePredicate {
                flow,
                graph,
                edge_type,
            } => write!(
                f,
                "flow '{}' graph '{}' prune_edges {} predicate",
                flow, graph, edge_type
            ),
        }
    }
}

/// Rich semantic validation diagnostic.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct ValidationDiagnostic {
    pub message: String,
    pub context: Option<ValidationContext>,
    pub range: Option<SourceRange>,
}

impl fmt::Display for ValidationDiagnostic {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "validation error")?;
        if let Some(ctx) = &self.context {
            write!(f, " [{}]", ctx)?;
        }
        write!(f, ": {}", self.message)?;
        if let Some(range) = self.range {
            write!(
                f,
                " (at {}:{}-{}:{})",
                range.start.line, range.start.column, range.end.line, range.end.column
            )?;
        }
        Ok(())
    }
}

/// Errors that can occur during parsing or validation.
#[non_exhaustive]
#[derive(Debug, Error)]
pub enum FrontendError {
    /// Syntax error during parsing.
    #[error("parse error: {0}")]
    ParseError(String),

    /// Semantic validation error.
    #[error("validation error: {0}")]
    ValidationError(String),

    /// Semantic validation error with context and optional source range.
    #[error("{0}")]
    ValidationDiagnostic(ValidationDiagnostic),
}

impl FrontendError {
    /// Build a context-aware validation diagnostic.
    pub fn validation(
        message: impl Into<String>,
        context: Option<ValidationContext>,
        range: Option<SourceRange>,
    ) -> Self {
        Self::ValidationDiagnostic(ValidationDiagnostic {
            message: message.into(),
            context,
            range,
        })
    }

    /// Returns the rich validation diagnostic if present.
    pub fn validation_diagnostic(&self) -> Option<&ValidationDiagnostic> {
        match self {
            Self::ValidationDiagnostic(diag) => Some(diag),
            _ => None,
        }
    }
}
