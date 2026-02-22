//! # Semantic Validation
//!
//! This module performs semantic validation on the parsed AST, checking:
//!
//! - **Rule validation**: Ensures `prob()` is only used on edge variables,
//!   `prob_correlated()` is used on supported comparison forms,
//!   `E[]` is used on node attributes, and expressions are well-formed
//!
//! - **Metric validation**: Validates metric functions like `sum_nodes`,
//!   `count_nodes`, `avg_degree`, and `fold_nodes` have required arguments
//!
//! - **Prune predicate validation**: Ensures prune predicates use appropriate
//!   edge variable references
//!
//! ## Validation Rules
//!
//! - `prob(var)` requires `var` to be an edge variable from pattern
//! - `prob_correlated(lhs <op> rhs, rho=...)` requires node-attribute comparison operands
//! - `credible(event, p=...)` supports edge-variable or comparison events
//! - `E[node.attr]` requires `node` to be a node variable from pattern
//! - `degree(node)` requires `node` to be a node variable
//! - Metric functions require specific named arguments (label, contrib, etc.)
//!
//! Validation is separate from parsing to provide clear, actionable error messages.

use std::collections::HashSet;

use pest::Parser;

use crate::ast::*;
use crate::errors::{FrontendError, SourcePosition, SourceRange, ValidationContext};
use crate::parser::{BayGraphParser, Rule};

/// Performs semantic validation on a parsed program.
///
/// This validates that the program is semantically correct beyond syntax:
/// - Rules use variables correctly (prob on edges, E on node attributes)
/// - Metrics have required arguments
/// - Prune predicates use the `edge` variable
///
/// # Arguments
///
/// * `ast` - The parsed program to validate
///
/// # Returns
///
/// * `Ok(())` - Program is valid
/// * `Err(FrontendError::ValidationError | FrontendError::ValidationDiagnostic)` - Semantic error
///
/// # Example
///
/// ```rust,ignore
/// use grafial::frontend::{parser, validate};
///
/// let ast = parser::parse_program(source)?;
/// validate::validate_program(&ast)?; // Ensure semantic correctness
/// ```
pub fn validate_program(ast: &ProgramAst) -> Result<(), FrontendError> {
    validate_program_internal(ast, None)
}

/// Performs semantic validation with source-aware diagnostics.
///
/// This variant attaches precise ranges to validation diagnostics when possible.
pub fn validate_program_with_source(ast: &ProgramAst, source: &str) -> Result<(), FrontendError> {
    let source_map = build_source_map(source);
    validate_program_internal(ast, source_map.as_ref())
}

#[derive(Debug, Default, Clone)]
struct RuleSourceEntry {
    where_expr: Option<SourceRange>,
    action_exprs: Vec<SourceRange>,
}

#[derive(Debug, Default, Clone)]
struct FlowSourceEntry {
    metric_exprs: Vec<SourceRange>,
    prune_predicates: Vec<SourceRange>,
}

#[derive(Debug, Default, Clone)]
struct ValidationSourceMap {
    belief_models: Vec<SourceRange>,
    evidences: Vec<SourceRange>,
    rules: Vec<RuleSourceEntry>,
    flows: Vec<FlowSourceEntry>,
}

fn validate_program_internal(
    ast: &ProgramAst,
    source_map: Option<&ValidationSourceMap>,
) -> Result<(), FrontendError> {
    // Validate belief models (competing edge groups, posterior parameters)
    for (idx, model) in ast.belief_models.iter().enumerate() {
        let range = source_map.and_then(|m| m.belief_models.get(idx).copied());
        validate_belief_model(model, &ast.schemas, range)?;
    }

    // Validate evidence (evidence mode matches posterior type)
    for (idx, evidence) in ast.evidences.iter().enumerate() {
        let range = source_map.and_then(|m| m.evidences.get(idx).copied());
        validate_evidence(evidence, &ast.belief_models, range)?;
    }

    for (idx, rule) in ast.rules.iter().enumerate() {
        let source = source_map.and_then(|m| m.rules.get(idx));
        validate_rule_inner(rule, source)?;
    }

    for (idx, flow) in ast.flows.iter().enumerate() {
        let source = source_map.and_then(|m| m.flows.get(idx));
        validate_flow_inner(flow, source)?;
    }

    Ok(())
}

fn as_source_range(span: pest::Span<'_>) -> SourceRange {
    let (start_line, start_col) = span.start_pos().line_col();
    let (end_line, end_col) = span.end_pos().line_col();
    SourceRange {
        start: SourcePosition {
            line: start_line as u32,
            column: start_col as u32,
        },
        end: SourcePosition {
            line: end_line as u32,
            column: end_col as u32,
        },
    }
}

fn build_source_map(source: &str) -> Option<ValidationSourceMap> {
    let mut map = ValidationSourceMap::default();
    let mut pairs = BayGraphParser::parse(Rule::program, source).ok()?;
    let program_pair = pairs.next()?;
    for item in program_pair.into_inner() {
        match item.as_rule() {
            Rule::decl => {
                for decl in item.into_inner() {
                    collect_decl_source(decl, &mut map);
                }
            }
            Rule::belief_model_decl | Rule::evidence_decl | Rule::rule_decl | Rule::flow_decl => {
                collect_decl_source(item, &mut map);
            }
            _ => {}
        }
    }
    Some(map)
}

fn collect_decl_source(decl: pest::iterators::Pair<Rule>, map: &mut ValidationSourceMap) {
    match decl.as_rule() {
        Rule::belief_model_decl => map.belief_models.push(as_source_range(decl.as_span())),
        Rule::evidence_decl => map.evidences.push(as_source_range(decl.as_span())),
        Rule::rule_decl => map.rules.push(collect_rule_source(decl)),
        Rule::flow_decl => map.flows.push(collect_flow_source(decl)),
        _ => {}
    }
}

fn collect_rule_source(rule_decl: pest::iterators::Pair<Rule>) -> RuleSourceEntry {
    let mut entry = RuleSourceEntry::default();
    for p in rule_decl.into_inner() {
        if p.as_rule() != Rule::rule_body {
            continue;
        }
        for b in p.into_inner() {
            match b.as_rule() {
                Rule::sugar_rule | Rule::for_sugar => {
                    for part in b.into_inner() {
                        match part.as_rule() {
                            Rule::expr => {
                                if entry.where_expr.is_none() {
                                    entry.where_expr = Some(as_source_range(part.as_span()));
                                }
                            }
                            Rule::action_block => {
                                collect_action_spans(part, &mut entry.action_exprs);
                            }
                            _ => {}
                        }
                    }
                }
                Rule::where_clause => {
                    for part in b.into_inner() {
                        match part.as_rule() {
                            Rule::expr => entry.where_expr = Some(as_source_range(part.as_span())),
                            Rule::action_block => {
                                collect_action_spans(part, &mut entry.action_exprs);
                            }
                            _ => {}
                        }
                    }
                }
                Rule::action_clause => {
                    for part in b.into_inner() {
                        if part.as_rule() == Rule::action_block {
                            collect_action_spans(part, &mut entry.action_exprs);
                        }
                    }
                }
                _ => {}
            }
        }
    }
    entry
}

fn collect_action_spans(action_block: pest::iterators::Pair<Rule>, out: &mut Vec<SourceRange>) {
    for stmt in action_block.into_inner() {
        match stmt.as_rule() {
            Rule::action_stmt => {
                if let Some(inner) = stmt.clone().into_inner().next() {
                    out.push(as_source_range(inner.as_span()));
                } else {
                    out.push(as_source_range(stmt.as_span()));
                }
            }
            Rule::let_stmt
            | Rule::nbnudge_stmt
            | Rule::soft_update_stmt
            | Rule::delete_stmt
            | Rule::suppress_stmt => out.push(as_source_range(stmt.as_span())),
            _ => {}
        }
    }
}

fn collect_flow_source(flow_decl: pest::iterators::Pair<Rule>) -> FlowSourceEntry {
    let mut entry = FlowSourceEntry::default();
    for p in flow_decl.into_inner() {
        if p.as_rule() != Rule::flow_body {
            continue;
        }
        for b in p.into_inner() {
            match b.as_rule() {
                Rule::metric_stmt => {
                    if let Some(span) = find_metric_expr_span(b) {
                        entry.metric_exprs.push(span);
                    }
                }
                Rule::graph_stmt => {
                    collect_prune_predicate_spans(b, &mut entry.prune_predicates);
                }
                _ => {}
            }
        }
    }
    entry
}

fn find_metric_expr_span(metric_stmt: pest::iterators::Pair<Rule>) -> Option<SourceRange> {
    for p in metric_stmt.into_inner() {
        if matches!(p.as_rule(), Rule::expr | Rule::metric_builder_expr) {
            return Some(as_source_range(p.as_span()));
        }
    }
    None
}

fn collect_prune_predicate_spans(pair: pest::iterators::Pair<Rule>, out: &mut Vec<SourceRange>) {
    match pair.as_rule() {
        Rule::prune_edges_tr => {
            for p in pair.into_inner() {
                if p.as_rule() == Rule::expr {
                    out.push(as_source_range(p.as_span()));
                    break;
                }
            }
        }
        _ => {
            for p in pair.into_inner() {
                collect_prune_predicate_spans(p, out);
            }
        }
    }
}

fn validation_error(
    message: impl Into<String>,
    context: Option<&ValidationContext>,
    range: Option<SourceRange>,
) -> FrontendError {
    FrontendError::validation(message, context.cloned(), range)
}

fn validate_belief_model(
    model: &BeliefModel,
    schemas: &[Schema],
    range: Option<SourceRange>,
) -> Result<(), FrontendError> {
    let context = ValidationContext::BeliefModel {
        model: model.name.clone(),
    };

    // Find the schema this model operates on
    let schema = schemas
        .iter()
        .find(|s| s.name == model.on_schema)
        .ok_or_else(|| {
            validation_error(
                format!(
                    "Belief model '{}' references unknown schema '{}'",
                    model.name, model.on_schema
                ),
                Some(&context),
                range,
            )
        })?;

    // Validate node belief declarations.
    for node_decl in &model.nodes {
        let schema_node = schema
            .nodes
            .iter()
            .find(|n| n.name == node_decl.node_type)
            .ok_or_else(|| {
                validation_error(
                    format!(
                        "Node type '{}' not found in schema '{}'",
                        node_decl.node_type, model.on_schema
                    ),
                    Some(&context),
                    range,
                )
            })?;

        let mut attrs_seen = HashSet::new();
        for (attr_name, posterior) in &node_decl.attrs {
            if !schema_node.attrs.iter().any(|a| a.name == *attr_name) {
                return Err(validation_error(
                    format!(
                        "Attribute '{}.{}' not found in schema '{}'",
                        node_decl.node_type, attr_name, model.on_schema
                    ),
                    Some(&context),
                    range,
                ));
            }
            if !attrs_seen.insert(attr_name.as_str()) {
                return Err(validation_error(
                    format!(
                        "Attribute '{}.{}' declared multiple times in belief model '{}'",
                        node_decl.node_type, attr_name, model.name
                    ),
                    Some(&context),
                    range,
                ));
            }

            let target = format!("node attribute '{}.{}'", node_decl.node_type, attr_name);
            match posterior {
                PosteriorType::Gaussian { params } => {
                    validate_gaussian_params(params, &target, &context, range)?;
                }
                other => {
                    return Err(validation_error(
                        format!("{} must use GaussianPosterior, got {:?}", target, other),
                        Some(&context),
                        range,
                    ));
                }
            }
        }
    }

    // Track edge types to ensure uniqueness (independent vs competing)
    let mut edge_types_seen = std::collections::HashMap::new();

    // Validate edge declarations
    for edge_decl in &model.edges {
        // Check if edge type exists in schema
        if !schema.edges.iter().any(|e| e.name == edge_decl.edge_type) {
            return Err(validation_error(
                format!(
                    "Edge type '{}' not found in schema '{}'",
                    edge_decl.edge_type, model.on_schema
                ),
                Some(&context),
                range,
            ));
        }

        // Validate posterior type
        match &edge_decl.exist {
            PosteriorType::Categorical {
                group_by,
                prior,
                categories,
            } => {
                // Check group_by is valid
                if group_by != "source" && group_by != "destination" {
                    return Err(validation_error(
                        format!(
                            "CategoricalPosterior 'group_by' must be 'source' or 'destination', got '{}'",
                            group_by
                        ),
                        Some(&context),
                        range,
                    ));
                }

                // Check prior specification
                match prior {
                    CategoricalPrior::Uniform { pseudo_count } => {
                        if !pseudo_count.is_finite() || *pseudo_count <= 0.0 {
                            return Err(validation_error(
                                format!(
                                    "CategoricalPosterior 'pseudo_count' must be finite and > 0, got {}",
                                    pseudo_count
                                ),
                                Some(&context),
                                range,
                            ));
                        }
                    }
                    CategoricalPrior::Explicit { concentrations } => {
                        if concentrations.is_empty() {
                            return Err(validation_error(
                                "CategoricalPosterior explicit prior array cannot be empty",
                                Some(&context),
                                range,
                            ));
                        }
                        for (i, &alpha) in concentrations.iter().enumerate() {
                            if !alpha.is_finite() || alpha <= 0.0 {
                                return Err(validation_error(
                                    format!(
                                        "CategoricalPosterior prior array element {} must be finite and > 0, got {}",
                                        i, alpha
                                    ),
                                    Some(&context),
                                    range,
                                ));
                            }
                        }
                    }
                }

                if let Some(category_names) = categories {
                    if category_names.is_empty() {
                        return Err(validation_error(
                            "CategoricalPosterior 'categories' cannot be empty when provided",
                            Some(&context),
                            range,
                        ));
                    }
                    let mut seen = HashSet::new();
                    for category in category_names {
                        if category.trim().is_empty() {
                            return Err(validation_error(
                                "CategoricalPosterior 'categories' cannot contain empty names",
                                Some(&context),
                                range,
                            ));
                        }
                        if !seen.insert(category.as_str()) {
                            return Err(validation_error(
                                format!(
                                    "CategoricalPosterior 'categories' contains duplicate '{}'",
                                    category
                                ),
                                Some(&context),
                                range,
                            ));
                        }
                    }
                }

                // Note: categories validation against schema would require knowing all possible destinations
                // This is deferred to runtime when edges are actually created
            }
            PosteriorType::Bernoulli { params } => {
                let target = format!("edge '{}.exist'", edge_decl.edge_type);
                validate_bernoulli_params(params, &target, &context, range)?;
            }
            PosteriorType::Gaussian { params } => {
                let target = format!("edge '{}.exist'", edge_decl.edge_type);
                validate_gaussian_params(params, &target, &context, range)?;
                return Err(validation_error(
                    format!(
                        "{} must use BernoulliPosterior or CategoricalPosterior for edge existence",
                        target
                    ),
                    Some(&context),
                    range,
                ));
            }
        }

        // Check for duplicate edge type declarations
        if let Some(prev_type) = edge_types_seen.get(&edge_decl.edge_type) {
            return Err(validation_error(
                format!(
                    "Edge type '{}' declared multiple times in belief model '{}' (previous: {:?})",
                    edge_decl.edge_type, model.name, prev_type
                ),
                Some(&context),
                range,
            ));
        }
        edge_types_seen.insert(edge_decl.edge_type.clone(), &edge_decl.exist);
    }

    Ok(())
}

fn validate_gaussian_params(
    params: &[(String, f64)],
    target: &str,
    context: &ValidationContext,
    range: Option<SourceRange>,
) -> Result<(), FrontendError> {
    let mut seen_prior_mean = false;
    let mut seen_prior_precision = false;

    for (name, val) in params {
        if !val.is_finite() {
            return Err(validation_error(
                format!(
                    "{} GaussianPosterior parameter '{}' must be finite, got {}",
                    target, name, val
                ),
                Some(context),
                range,
            ));
        }

        match name.as_str() {
            "prior_mean" => {
                if seen_prior_mean {
                    return Err(validation_error(
                        format!(
                            "{} GaussianPosterior parameter 'prior_mean' declared multiple times",
                            target
                        ),
                        Some(context),
                        range,
                    ));
                }
                seen_prior_mean = true;
            }
            "prior_precision" => {
                if seen_prior_precision {
                    return Err(validation_error(
                        format!(
                            "{} GaussianPosterior parameter 'prior_precision' declared multiple times",
                            target
                        ),
                        Some(context),
                        range,
                    ));
                }
                if *val <= 0.0 {
                    return Err(validation_error(
                        format!(
                            "{} GaussianPosterior 'prior_precision' must be > 0, got {}",
                            target, val
                        ),
                        Some(context),
                        range,
                    ));
                }
                seen_prior_precision = true;
            }
            other => {
                return Err(validation_error(
                    format!(
                        "{} GaussianPosterior has unknown parameter '{}'",
                        target, other
                    ),
                    Some(context),
                    range,
                ));
            }
        }
    }

    Ok(())
}

fn validate_bernoulli_params(
    params: &[(String, f64)],
    target: &str,
    context: &ValidationContext,
    range: Option<SourceRange>,
) -> Result<(), FrontendError> {
    let mut seen_prior = false;
    let mut seen_pseudo_count = false;

    for (name, val) in params {
        if !val.is_finite() {
            return Err(validation_error(
                format!(
                    "{} BernoulliPosterior parameter '{}' must be finite, got {}",
                    target, name, val
                ),
                Some(context),
                range,
            ));
        }

        match name.as_str() {
            "prior" => {
                if seen_prior {
                    return Err(validation_error(
                        format!(
                            "{} BernoulliPosterior parameter 'prior' declared multiple times",
                            target
                        ),
                        Some(context),
                        range,
                    ));
                }
                if !(*val > 0.0 && *val < 1.0) {
                    return Err(validation_error(
                        format!(
                            "{} BernoulliPosterior 'prior' must be in (0, 1), got {}",
                            target, val
                        ),
                        Some(context),
                        range,
                    ));
                }
                seen_prior = true;
            }
            "pseudo_count" => {
                if seen_pseudo_count {
                    return Err(validation_error(
                        format!(
                            "{} BernoulliPosterior parameter 'pseudo_count' declared multiple times",
                            target
                        ),
                        Some(context),
                        range,
                    ));
                }
                if *val <= 0.0 {
                    return Err(validation_error(
                        format!(
                            "{} BernoulliPosterior 'pseudo_count' must be > 0, got {}",
                            target, val
                        ),
                        Some(context),
                        range,
                    ));
                }
                seen_pseudo_count = true;
            }
            other => {
                return Err(validation_error(
                    format!(
                        "{} BernoulliPosterior has unknown parameter '{}'",
                        target, other
                    ),
                    Some(context),
                    range,
                ));
            }
        }
    }

    Ok(())
}

fn validate_evidence(
    evidence: &EvidenceDef,
    belief_models: &[BeliefModel],
    range: Option<SourceRange>,
) -> Result<(), FrontendError> {
    let context = ValidationContext::Evidence {
        evidence: evidence.name.clone(),
    };

    // Find the belief model this evidence applies to
    let model = belief_models
        .iter()
        .find(|m| m.name == evidence.on_model)
        .ok_or_else(|| {
            validation_error(
                format!(
                    "Evidence '{}' references unknown belief model '{}'",
                    evidence.name, evidence.on_model
                ),
                Some(&context),
                range,
            )
        })?;

    // Validate each observation statement
    for obs in &evidence.observations {
        match obs {
            ObserveStmt::Edge {
                edge_type, mode, ..
            } => {
                // Find the edge declaration in the belief model
                let edge_decl = model.edges.iter().find(|e| e.edge_type == *edge_type);

                if let Some(edge_decl) = edge_decl {
                    // Check evidence mode matches posterior type
                    match (&edge_decl.exist, mode) {
                        (
                            PosteriorType::Categorical { .. },
                            EvidenceMode::Present | EvidenceMode::Absent,
                        ) => {
                            return Err(validation_error(
                                format!(
                                    "Edge '{}' has competing posterior; use 'chosen', 'unchosen', or 'forced_choice', not '{:?}'",
                                    edge_type, mode
                                ),
                                Some(&context),
                                range,
                            ));
                        }
                        (
                            PosteriorType::Bernoulli { .. } | PosteriorType::Gaussian { .. },
                            EvidenceMode::Chosen
                            | EvidenceMode::Unchosen
                            | EvidenceMode::ForcedChoice,
                        ) => {
                            return Err(validation_error(
                                format!(
                                    "Edge '{}' is independent; use 'present' or 'absent', not '{:?}'",
                                    edge_type, mode
                                ),
                                Some(&context),
                                range,
                            ));
                        }
                        _ => {} // Valid combinations
                    }
                } else {
                    // Edge type not declared in belief model - this might be okay if it's optional
                    // For now, we'll allow it (could be validated later when building the graph)
                }
            }
            ObserveStmt::Attribute { .. } => {
                // Attribute observations are validated when building the graph
            }
        }
    }

    Ok(())
}

#[allow(dead_code)]
fn validate_rule(rule: &RuleDef) -> Result<(), FrontendError> {
    validate_rule_inner(rule, None)
}

#[derive(Debug, Clone, Default)]
struct RuleScope {
    node_vars: HashSet<String>,
    edge_vars: HashSet<String>,
    locals: HashSet<String>,
}

impl RuleScope {
    fn from_patterns(patterns: &[PatternItem]) -> Self {
        let mut scope = Self::default();
        for p in patterns {
            scope.node_vars.insert(p.src.var.clone());
            scope.node_vars.insert(p.dst.var.clone());
            scope.edge_vars.insert(p.edge.var.clone());
        }
        scope
    }

    fn contains_var(&self, name: &str) -> bool {
        self.locals.contains(name) || self.node_vars.contains(name) || self.edge_vars.contains(name)
    }
}

fn validate_rule_inner(
    rule: &RuleDef,
    source: Option<&RuleSourceEntry>,
) -> Result<(), FrontendError> {
    let where_ctx = ValidationContext::RuleWhere {
        rule: rule.name.clone(),
    };
    let mut scope = RuleScope::from_patterns(&rule.patterns);

    if let Some(expr) = &rule.where_expr {
        let where_range = source.and_then(|s| s.where_expr);
        validate_rule_expr(expr, &scope, true, &where_ctx, where_range)?;
    }

    for (idx, action) in rule.actions.iter().enumerate() {
        let action_range = source.and_then(|s| s.action_exprs.get(idx).copied());
        let action_ctx = ValidationContext::RuleAction {
            rule: rule.name.clone(),
            action: action_kind(action).to_string(),
        };
        validate_action_stmt(action, &mut scope, &action_ctx, action_range)?;
    }

    Ok(())
}

fn action_kind(action: &ActionStmt) -> &'static str {
    match action {
        ActionStmt::Let { .. } => "let",
        ActionStmt::SetExpectation { .. } => "set_expectation",
        ActionStmt::ForceAbsent { .. } => "force_absent",
        ActionStmt::NonBayesianNudge { .. } => "non_bayesian_nudge",
        ActionStmt::SoftUpdate { .. } => "soft_update",
        ActionStmt::DeleteEdge { .. } => "delete",
        ActionStmt::SuppressEdge { .. } => "suppress",
    }
}

fn validate_action_stmt(
    action: &ActionStmt,
    scope: &mut RuleScope,
    context: &ValidationContext,
    range: Option<SourceRange>,
) -> Result<(), FrontendError> {
    match action {
        ActionStmt::Let { name, expr } => {
            validate_rule_expr(expr, scope, false, context, range)?;
            if scope.contains_var(name) {
                return Err(validation_error(
                    format!("let variable '{}' shadows an existing binding", name),
                    Some(context),
                    range,
                ));
            }
            scope.locals.insert(name.clone());
        }
        ActionStmt::SetExpectation { node_var, expr, .. }
        | ActionStmt::NonBayesianNudge { node_var, expr, .. }
        | ActionStmt::SoftUpdate { node_var, expr, .. } => {
            if !scope.node_vars.contains(node_var) {
                return Err(validation_error(
                    format!("unknown node variable '{}'", node_var),
                    Some(context),
                    range,
                ));
            }
            validate_rule_expr(expr, scope, false, context, range)?;
        }
        ActionStmt::ForceAbsent { edge_var }
        | ActionStmt::DeleteEdge { edge_var, .. }
        | ActionStmt::SuppressEdge { edge_var, .. } => {
            if !scope.edge_vars.contains(edge_var) {
                return Err(validation_error(
                    format!("unknown edge variable '{}'", edge_var),
                    Some(context),
                    range,
                ));
            }
        }
    }
    Ok(())
}

fn validate_rule_expr(
    expr: &ExprAst,
    scope: &RuleScope,
    allow_exists: bool,
    context: &ValidationContext,
    range: Option<SourceRange>,
) -> Result<(), FrontendError> {
    match expr {
        ExprAst::Number(_) | ExprAst::Bool(_) => Ok(()),
        ExprAst::Var(v) => {
            if scope.contains_var(v) {
                Ok(())
            } else {
                Err(validation_error(
                    format!("unknown variable '{}'", v),
                    Some(context),
                    range,
                ))
            }
        }
        ExprAst::Field { .. } => Err(validation_error(
            "Bare field access in rule expression is not allowed; use E[NodeVar.attr] or prob(...)",
            Some(context),
            range,
        )),
        ExprAst::Unary { expr, .. } => {
            validate_rule_expr(expr, scope, allow_exists, context, range)
        }
        ExprAst::Binary { left, right, .. } => {
            validate_rule_expr(left, scope, allow_exists, context, range)?;
            validate_rule_expr(right, scope, allow_exists, context, range)
        }
        ExprAst::Exists {
            pattern,
            where_expr,
            ..
        } => {
            if !allow_exists {
                return Err(validation_error(
                    "exists subqueries are only allowed in rule where clauses",
                    Some(context),
                    range,
                ));
            }
            let mut sub_scope = scope.clone();
            sub_scope.node_vars.insert(pattern.src.var.clone());
            sub_scope.node_vars.insert(pattern.dst.var.clone());
            sub_scope.edge_vars.insert(pattern.edge.var.clone());
            if let Some(where_expr) = where_expr {
                validate_rule_expr(where_expr, &sub_scope, true, context, range)?;
            }
            Ok(())
        }
        ExprAst::Call { name, args } => validate_rule_call(name, args, scope, context, range),
    }
}

fn validate_rule_call(
    name: &str,
    args: &[CallArg],
    scope: &RuleScope,
    context: &ValidationContext,
    range: Option<SourceRange>,
) -> Result<(), FrontendError> {
    let (pos, named) = split_args(args);
    match name {
        "prob" => {
            if !named.map.is_empty() || pos.len() != 1 {
                return Err(validation_error(
                    "prob(): expected single positional argument",
                    Some(context),
                    range,
                ));
            }
            match pos[0] {
                ExprAst::Var(v) if scope.edge_vars.contains(v) => Ok(()),
                ExprAst::Binary {
                    op: BinaryOp::Gt | BinaryOp::Ge | BinaryOp::Lt | BinaryOp::Le,
                    left,
                    right,
                } => {
                    validate_prob_operand(left, scope, context, range)?;
                    validate_prob_operand(right, scope, context, range)
                }
                ExprAst::Binary { .. } => Err(validation_error(
                    "prob(): only supports comparisons like prob(A > B)",
                    Some(context),
                    range,
                )),
                _ => Err(validation_error(
                    "prob(): argument must be an edge variable",
                    Some(context),
                    range,
                )),
            }
        }
        "prob_correlated" => {
            if pos.len() != 1 {
                return Err(validation_error(
                    "prob_correlated(): expected single positional comparison argument",
                    Some(context),
                    range,
                ));
            }
            for arg_name in named.map.keys() {
                if arg_name != "rho" {
                    return Err(validation_error(
                        format!(
                            "prob_correlated(): unknown named argument '{}'; expected rho",
                            arg_name
                        ),
                        Some(context),
                        range,
                    ));
                }
            }
            if let Some(rho_expr) = named.get("rho") {
                validate_rule_expr(rho_expr, scope, false, context, range)?;
            }
            match pos[0] {
                ExprAst::Binary {
                    op: BinaryOp::Gt | BinaryOp::Ge | BinaryOp::Lt | BinaryOp::Le,
                    left,
                    right,
                } => {
                    validate_prob_operand(left, scope, context, range)?;
                    validate_prob_operand(right, scope, context, range)
                }
                ExprAst::Binary { .. } => Err(validation_error(
                    "prob_correlated(): only supports comparisons like prob_correlated(A > B)",
                    Some(context),
                    range,
                )),
                _ => Err(validation_error(
                    "prob_correlated(): argument must be a comparison expression",
                    Some(context),
                    range,
                )),
            }
        }
        "credible" => {
            if pos.len() != 1 {
                return Err(validation_error(
                    "credible(): expected single positional event argument",
                    Some(context),
                    range,
                ));
            }
            for arg_name in named.map.keys() {
                if arg_name != "p" && arg_name != "rho" {
                    return Err(validation_error(
                        format!(
                            "credible(): unknown named argument '{}'; expected p or rho",
                            arg_name
                        ),
                        Some(context),
                        range,
                    ));
                }
            }
            if let Some(p_expr) = named.get("p") {
                validate_rule_expr(p_expr, scope, false, context, range)?;
            }
            if let Some(rho_expr) = named.get("rho") {
                validate_rule_expr(rho_expr, scope, false, context, range)?;
            }
            match pos[0] {
                ExprAst::Var(v) if scope.edge_vars.contains(v) => Ok(()),
                ExprAst::Binary {
                    op: BinaryOp::Gt | BinaryOp::Ge | BinaryOp::Lt | BinaryOp::Le,
                    left,
                    right,
                } => {
                    validate_prob_operand(left, scope, context, range)?;
                    validate_prob_operand(right, scope, context, range)
                }
                ExprAst::Binary { .. } => Err(validation_error(
                    "credible(): only supports comparisons like credible(A > B)",
                    Some(context),
                    range,
                )),
                _ => Err(validation_error(
                    "credible(): argument must be an edge variable or comparison expression",
                    Some(context),
                    range,
                )),
            }
        }
        "E" => validate_node_field_arg(
            "E[]",
            pos.as_slice(),
            &named,
            &scope.node_vars,
            context,
            range,
        ),
        "variance" | "stddev" | "effective_n" => validate_node_field_arg(
            &format!("{}()", name),
            pos.as_slice(),
            &named,
            &scope.node_vars,
            context,
            range,
        ),
        "ci_lo" | "ci_hi" | "quantile" => {
            if !named.map.is_empty() || pos.len() != 2 {
                return Err(validation_error(
                    format!("{}(): expected field and numeric p", name),
                    Some(context),
                    range,
                ));
            }
            validate_node_field_expr(pos[0], &scope.node_vars, context, range, name)?;
            if !matches!(pos[1], ExprAst::Number(_)) {
                return Err(validation_error(
                    format!("{}(): p must be numeric", name),
                    Some(context),
                    range,
                ));
            }
            Ok(())
        }
        "degree" => {
            if pos.is_empty() {
                return Err(validation_error(
                    "degree(): missing node argument",
                    Some(context),
                    range,
                ));
            }
            match pos[0] {
                ExprAst::Var(v) if scope.node_vars.contains(v) => {}
                _ => {
                    return Err(validation_error(
                        "degree(): first argument must be a node variable",
                        Some(context),
                        range,
                    ))
                }
            }
            for e in pos.iter().skip(1) {
                validate_rule_expr(e, scope, false, context, range)?;
            }
            for value in named.map.values() {
                validate_rule_expr(value, scope, false, context, range)?;
            }
            Ok(())
        }
        "winner" | "entropy" => {
            if pos.len() < 2 {
                return Err(validation_error(
                    format!("{}(): requires node and edge_type arguments", name),
                    Some(context),
                    range,
                ));
            }
            match pos[0] {
                ExprAst::Var(v) if scope.node_vars.contains(v) => {}
                _ => {
                    return Err(validation_error(
                        format!("{}(): first argument must be a node variable", name),
                        Some(context),
                        range,
                    ))
                }
            }
            if !matches!(pos[1], ExprAst::Var(_)) {
                return Err(validation_error(
                    format!("{}(): edge_type must be an identifier", name),
                    Some(context),
                    range,
                ));
            }
            for e in pos.iter().skip(2) {
                validate_rule_expr(e, scope, false, context, range)?;
            }
            for value in named.map.values() {
                validate_rule_expr(value, scope, false, context, range)?;
            }
            Ok(())
        }
        _ => {
            for a in args {
                match a {
                    CallArg::Positional(e) => {
                        validate_rule_expr(e, scope, allow_exists_in_call(name), context, range)?
                    }
                    CallArg::Named { value, .. } => validate_rule_expr(
                        value,
                        scope,
                        allow_exists_in_call(name),
                        context,
                        range,
                    )?,
                }
            }
            Ok(())
        }
    }
}

fn allow_exists_in_call(_name: &str) -> bool {
    // Exists nodes can only appear in rule where expressions and are handled
    // by the parent expression context. Inside normal calls we keep this disabled.
    false
}

fn validate_prob_operand(
    expr: &ExprAst,
    scope: &RuleScope,
    context: &ValidationContext,
    range: Option<SourceRange>,
) -> Result<(), FrontendError> {
    match expr {
        ExprAst::Number(_) => Ok(()),
        ExprAst::Field { target, .. } => match &**target {
            ExprAst::Var(v) if scope.node_vars.contains(v) => Ok(()),
            _ => Err(validation_error(
                "prob(): comparison must reference node variables",
                Some(context),
                range,
            )),
        },
        _ => Err(validation_error(
            "prob(): only supports node.attr or numeric in comparisons",
            Some(context),
            range,
        )),
    }
}

fn validate_node_field_arg(
    fn_name: &str,
    pos: &[&ExprAst],
    named: &ArgNames<'_>,
    node_vars: &HashSet<String>,
    context: &ValidationContext,
    range: Option<SourceRange>,
) -> Result<(), FrontendError> {
    if !named.map.is_empty() || pos.len() != 1 {
        return Err(validation_error(
            format!("{}: expected single positional argument", fn_name),
            Some(context),
            range,
        ));
    }
    validate_node_field_expr(pos[0], node_vars, context, range, fn_name)
}

fn validate_node_field_expr(
    expr: &ExprAst,
    node_vars: &HashSet<String>,
    context: &ValidationContext,
    range: Option<SourceRange>,
    fn_name: &str,
) -> Result<(), FrontendError> {
    match expr {
        ExprAst::Field { target, .. } => match &**target {
            ExprAst::Var(v) if node_vars.contains(v) => Ok(()),
            _ => Err(validation_error(
                format!("{}: must be NodeVar.attr", fn_name),
                Some(context),
                range,
            )),
        },
        _ => Err(validation_error(
            format!("{}: must be a field access expression", fn_name),
            Some(context),
            range,
        )),
    }
}

#[allow(dead_code)]
fn validate_flow(flow: &FlowDef) -> Result<(), FrontendError> {
    validate_flow_inner(flow, None)
}

fn validate_flow_inner(
    flow: &FlowDef,
    source: Option<&FlowSourceEntry>,
) -> Result<(), FrontendError> {
    let mut prune_idx = 0usize;
    for g in &flow.graphs {
        if let GraphExpr::Pipeline { transforms, .. } = &g.expr {
            for t in transforms {
                match t {
                    Transform::ApplyRule { .. } => {}
                    Transform::ApplyRuleset { .. } => {}
                    Transform::Snapshot { .. } => {}
                    Transform::InferBeliefs => {}
                    Transform::PruneEdges {
                        edge_type,
                        predicate,
                    } => {
                        let range = source.and_then(|s| s.prune_predicates.get(prune_idx).copied());
                        prune_idx += 1;
                        let ctx = ValidationContext::PrunePredicate {
                            flow: flow.name.clone(),
                            graph: g.name.clone(),
                            edge_type: edge_type.clone(),
                        };
                        validate_prune_predicate_with_ctx(predicate, Some(&ctx), range)?;
                    }
                }
            }
        }
    }

    let mut metric_scope: HashSet<String> = flow
        .metric_imports
        .iter()
        .map(|i| i.local_name.clone())
        .collect();
    for (idx, m) in flow.metrics.iter().enumerate() {
        let range = source.and_then(|s| s.metric_exprs.get(idx).copied());
        let ctx = ValidationContext::MetricExpr {
            flow: flow.name.clone(),
            metric: m.name.clone(),
        };
        validate_metric_shape(&m.expr, Some(&ctx), range)?;
        validate_metric_expr_scoped(
            &m.expr,
            &metric_scope,
            MetricExprEnv::top_level(),
            &ctx,
            range,
        )?;
        metric_scope.insert(m.name.clone());
    }

    Ok(())
}

#[allow(dead_code)]
fn validate_metric(expr: &ExprAst) -> Result<(), FrontendError> {
    validate_metric_shape(expr, None, None)
}

fn validate_metric_shape(
    expr: &ExprAst,
    context: Option<&ValidationContext>,
    range: Option<SourceRange>,
) -> Result<(), FrontendError> {
    // Only shape checks for known metric calls; others pass through.
    if let ExprAst::Call { name, args } = expr {
        match name.as_str() {
            "sum_nodes" => {
                let (pos, named) = split_args(args);
                let has_label = named.contains("label") || !pos.is_empty();
                let has_contrib = named.contains("contrib") || pos.len() >= 3; // label, where, contrib
                if !has_label {
                    return Err(validation_error(
                        "sum_nodes: missing 'label' argument",
                        context,
                        range,
                    ));
                }
                if !has_contrib {
                    return Err(validation_error(
                        "sum_nodes: missing 'contrib' argument",
                        context,
                        range,
                    ));
                }
            }
            "fold_nodes" => {
                let (pos, named) = split_args(args);
                let has_label = named.contains("label") || !pos.is_empty();
                let has_init = named.contains("init") || pos.len() >= 4;
                let has_step = named.contains("step") || pos.len() >= 5;
                if !has_label {
                    return Err(validation_error(
                        "fold_nodes: missing 'label' argument",
                        context,
                        range,
                    ));
                }
                if !has_init {
                    return Err(validation_error(
                        "fold_nodes: missing 'init' argument",
                        context,
                        range,
                    ));
                }
                if !has_step {
                    return Err(validation_error(
                        "fold_nodes: missing 'step' argument",
                        context,
                        range,
                    ));
                }
            }
            "count_nodes" => {
                let (pos, named) = split_args(args);
                let has_label = named.contains("label") || !pos.is_empty();
                if !has_label {
                    return Err(validation_error(
                        "count_nodes: missing 'label' argument",
                        context,
                        range,
                    ));
                }
            }
            "avg_degree" => {
                let (pos, named) = split_args(args);
                let has_label = !pos.is_empty() || named.contains("label");
                let has_edge_type = pos.len() >= 2 || named.contains("edge_type");
                if !has_label {
                    return Err(validation_error(
                        "avg_degree: missing 'label' argument",
                        context,
                        range,
                    ));
                }
                if !has_edge_type {
                    return Err(validation_error(
                        "avg_degree: missing 'edge_type' argument",
                        context,
                        range,
                    ));
                }
            }
            _ => {}
        }
    }
    Ok(())
}

fn split_args(args: &[CallArg]) -> (Vec<&ExprAst>, ArgNames<'_>) {
    let mut pos = Vec::new();
    let mut names = ArgNames::default();
    for a in args {
        match a {
            CallArg::Positional(e) => pos.push(e),
            CallArg::Named { name, value } => {
                names.insert(name.clone(), value);
            }
        }
    }
    (pos, names)
}

#[derive(Default)]
struct ArgNames<'a> {
    map: std::collections::HashMap<String, &'a ExprAst>,
}

impl<'a> ArgNames<'a> {
    fn insert(&mut self, name: String, value: &'a ExprAst) {
        self.map.insert(name, value);
    }
    fn contains(&self, name: &str) -> bool {
        self.map.contains_key(name)
    }
    fn get(&self, name: &str) -> Option<&'a ExprAst> {
        self.map.get(name).copied()
    }
}

#[allow(dead_code)]
fn validate_prune_predicate(expr: &ExprAst) -> Result<(), FrontendError> {
    validate_prune_predicate_with_ctx(expr, None, None)
}

fn validate_prune_predicate_with_ctx(
    expr: &ExprAst,
    context: Option<&ValidationContext>,
    range: Option<SourceRange>,
) -> Result<(), FrontendError> {
    // Ensure any prob(...) refers to 'edge'
    walk_expr(expr, &mut |e| {
        if let ExprAst::Call { name, args } = e {
            if name == "prob" {
                let (pos, named) = split_args(args);
                if !named.map.is_empty() || pos.len() != 1 {
                    return Err(validation_error(
                        "prob(): expected single positional argument",
                        context,
                        range,
                    ));
                }
                match pos[0] {
                    ExprAst::Var(v) if v == "edge" => Ok(()),
                    _ => Err(validation_error(
                        "prob(): argument must be 'edge' in prune_edges predicate",
                        context,
                        range,
                    )),
                }
            } else {
                Ok(())
            }
        } else {
            Ok(())
        }
    })
}

#[derive(Clone, Copy)]
struct MetricExprEnv {
    allow_node: bool,
    allow_value: bool,
    allow_ident_literal: bool,
}

impl MetricExprEnv {
    fn top_level() -> Self {
        Self {
            allow_node: false,
            allow_value: false,
            allow_ident_literal: false,
        }
    }

    fn node_context(allow_value: bool) -> Self {
        Self {
            allow_node: true,
            allow_value,
            allow_ident_literal: false,
        }
    }

    fn ident_slot() -> Self {
        Self {
            allow_node: false,
            allow_value: false,
            allow_ident_literal: true,
        }
    }
}

fn validate_metric_expr_scoped(
    expr: &ExprAst,
    metric_scope: &HashSet<String>,
    env: MetricExprEnv,
    context: &ValidationContext,
    range: Option<SourceRange>,
) -> Result<(), FrontendError> {
    match expr {
        ExprAst::Number(_) | ExprAst::Bool(_) => Ok(()),
        ExprAst::Var(name) => {
            if env.allow_ident_literal {
                return Ok(());
            }
            if name == "node" {
                return Err(validation_error(
                    "bare 'node' variable not allowed; use E[node.attr] or degree(node, ...)",
                    Some(context),
                    range,
                ));
            }
            if name == "value" {
                if env.allow_value {
                    return Ok(());
                }
                return Err(validation_error(
                    "'value' is only valid inside fold_nodes step",
                    Some(context),
                    range,
                ));
            }
            if metric_scope.contains(name) {
                Ok(())
            } else {
                Err(validation_error(
                    format!("unknown metric variable '{}'", name),
                    Some(context),
                    range,
                ))
            }
        }
        ExprAst::Field { .. } => Err(validation_error(
            "bare field access not supported in metric; use E[node.attr]",
            Some(context),
            range,
        )),
        ExprAst::Exists { .. } => Err(validation_error(
            "exists subqueries not supported in metric expressions",
            Some(context),
            range,
        )),
        ExprAst::Unary { expr, .. } => {
            validate_metric_expr_scoped(expr, metric_scope, env, context, range)
        }
        ExprAst::Binary { left, right, .. } => {
            validate_metric_expr_scoped(left, metric_scope, env, context, range)?;
            validate_metric_expr_scoped(right, metric_scope, env, context, range)
        }
        ExprAst::Call { name, args } => {
            validate_metric_call(name, args, metric_scope, env, context, range)
        }
    }
}

fn validate_metric_call(
    name: &str,
    args: &[CallArg],
    metric_scope: &HashSet<String>,
    env: MetricExprEnv,
    context: &ValidationContext,
    range: Option<SourceRange>,
) -> Result<(), FrontendError> {
    let (pos, named) = split_args(args);
    match name {
        "count_nodes" => {
            let label_expr = named
                .get("label")
                .or_else(|| pos.first().copied())
                .ok_or_else(|| {
                    validation_error(
                        "count_nodes: missing 'label' argument",
                        Some(context),
                        range,
                    )
                })?;
            validate_metric_expr_scoped(
                label_expr,
                metric_scope,
                MetricExprEnv::ident_slot(),
                context,
                range,
            )?;
            if let Some(where_expr) = named.get("where").or_else(|| pos.get(1).copied()) {
                validate_metric_expr_scoped(
                    where_expr,
                    metric_scope,
                    MetricExprEnv::node_context(false),
                    context,
                    range,
                )?;
            }
            Ok(())
        }
        "sum_nodes" => {
            let label_expr = named
                .get("label")
                .or_else(|| pos.first().copied())
                .ok_or_else(|| {
                    validation_error("sum_nodes: missing 'label' argument", Some(context), range)
                })?;
            let contrib_expr = named
                .get("contrib")
                .or_else(|| pos.get(2).copied())
                .ok_or_else(|| {
                    validation_error(
                        "sum_nodes: missing 'contrib' argument",
                        Some(context),
                        range,
                    )
                })?;
            validate_metric_expr_scoped(
                label_expr,
                metric_scope,
                MetricExprEnv::ident_slot(),
                context,
                range,
            )?;
            if let Some(where_expr) = named.get("where").or_else(|| pos.get(1).copied()) {
                validate_metric_expr_scoped(
                    where_expr,
                    metric_scope,
                    MetricExprEnv::node_context(false),
                    context,
                    range,
                )?;
            }
            validate_metric_expr_scoped(
                contrib_expr,
                metric_scope,
                MetricExprEnv::node_context(false),
                context,
                range,
            )
        }
        "fold_nodes" => {
            let label_expr = named
                .get("label")
                .or_else(|| pos.first().copied())
                .ok_or_else(|| {
                    validation_error("fold_nodes: missing 'label' argument", Some(context), range)
                })?;
            let init_expr = named
                .get("init")
                .or_else(|| pos.get(3).copied())
                .ok_or_else(|| {
                    validation_error("fold_nodes: missing 'init' argument", Some(context), range)
                })?;
            let step_expr = named
                .get("step")
                .or_else(|| pos.get(4).copied())
                .ok_or_else(|| {
                    validation_error("fold_nodes: missing 'step' argument", Some(context), range)
                })?;
            validate_metric_expr_scoped(
                label_expr,
                metric_scope,
                MetricExprEnv::ident_slot(),
                context,
                range,
            )?;
            if let Some(where_expr) = named.get("where").or_else(|| pos.get(1).copied()) {
                validate_metric_expr_scoped(
                    where_expr,
                    metric_scope,
                    MetricExprEnv::node_context(false),
                    context,
                    range,
                )?;
            }
            if let Some(order_expr) = named.get("order_by").or_else(|| pos.get(2).copied()) {
                validate_metric_expr_scoped(
                    order_expr,
                    metric_scope,
                    MetricExprEnv::node_context(false),
                    context,
                    range,
                )?;
            }
            validate_metric_expr_scoped(
                init_expr,
                metric_scope,
                MetricExprEnv::top_level(),
                context,
                range,
            )?;
            validate_metric_expr_scoped(
                step_expr,
                metric_scope,
                MetricExprEnv::node_context(true),
                context,
                range,
            )
        }
        "avg_degree" => {
            let label_expr = named
                .get("label")
                .or_else(|| pos.first().copied())
                .ok_or_else(|| {
                    validation_error("avg_degree: missing 'label' argument", Some(context), range)
                })?;
            let edge_type_expr = named
                .get("edge_type")
                .or_else(|| pos.get(1).copied())
                .ok_or_else(|| {
                    validation_error(
                        "avg_degree: missing 'edge_type' argument",
                        Some(context),
                        range,
                    )
                })?;
            validate_metric_expr_scoped(
                label_expr,
                metric_scope,
                MetricExprEnv::ident_slot(),
                context,
                range,
            )?;
            validate_metric_expr_scoped(
                edge_type_expr,
                metric_scope,
                MetricExprEnv::ident_slot(),
                context,
                range,
            )?;
            if let Some(min_prob) = named.get("min_prob") {
                if !matches!(min_prob, ExprAst::Number(_)) {
                    return Err(validation_error(
                        "avg_degree: 'min_prob' must be numeric",
                        Some(context),
                        range,
                    ));
                }
            }
            Ok(())
        }
        // Node-level metric expression functions
        "E" | "variance" | "stddev" | "effective_n" => {
            if env.allow_node {
                if !named.map.is_empty() || pos.len() != 1 {
                    return Err(validation_error(
                        format!("{}[] expects one positional argument", name),
                        Some(context),
                        range,
                    ));
                }
                validate_metric_node_field_arg(pos[0], context, range)
            } else {
                for a in args {
                    match a {
                        CallArg::Positional(e) => {
                            validate_metric_expr_scoped(e, metric_scope, env, context, range)?
                        }
                        CallArg::Named { value, .. } => {
                            validate_metric_expr_scoped(value, metric_scope, env, context, range)?
                        }
                    }
                }
                Ok(())
            }
        }
        "quantile" | "ci_lo" | "ci_hi" => {
            if env.allow_node {
                if pos.len() != 2 {
                    return Err(validation_error(
                        format!("{}(): requires node.attr and numeric p", name),
                        Some(context),
                        range,
                    ));
                }
                validate_metric_node_field_arg(pos[0], context, range)?;
                if !matches!(pos[1], ExprAst::Number(_)) {
                    return Err(validation_error(
                        format!("{}(): p must be numeric", name),
                        Some(context),
                        range,
                    ));
                }
                Ok(())
            } else {
                for a in args {
                    match a {
                        CallArg::Positional(e) => {
                            validate_metric_expr_scoped(e, metric_scope, env, context, range)?
                        }
                        CallArg::Named { value, .. } => {
                            validate_metric_expr_scoped(value, metric_scope, env, context, range)?
                        }
                    }
                }
                Ok(())
            }
        }
        "degree" => {
            if env.allow_node {
                if pos.is_empty() || !matches!(pos[0], ExprAst::Var(v) if v == "node") {
                    return Err(validation_error(
                        "degree(): first argument must be 'node'",
                        Some(context),
                        range,
                    ));
                }
                for e in pos.iter().skip(1) {
                    validate_metric_expr_scoped(
                        e,
                        metric_scope,
                        MetricExprEnv::node_context(env.allow_value),
                        context,
                        range,
                    )?;
                }
                for value in named.map.values() {
                    validate_metric_expr_scoped(
                        value,
                        metric_scope,
                        MetricExprEnv::node_context(env.allow_value),
                        context,
                        range,
                    )?;
                }
                Ok(())
            } else {
                for a in args {
                    match a {
                        CallArg::Positional(e) => {
                            validate_metric_expr_scoped(e, metric_scope, env, context, range)?
                        }
                        CallArg::Named { value, .. } => {
                            validate_metric_expr_scoped(value, metric_scope, env, context, range)?
                        }
                    }
                }
                Ok(())
            }
        }
        "entropy" => {
            if env.allow_node {
                if pos.len() < 2 {
                    return Err(validation_error(
                        "entropy(): requires node and edge_type arguments",
                        Some(context),
                        range,
                    ));
                }
                if !matches!(pos[0], ExprAst::Var(v) if v == "node") {
                    return Err(validation_error(
                        "entropy(): first argument must be 'node' in metric expressions",
                        Some(context),
                        range,
                    ));
                }
                if !matches!(pos[1], ExprAst::Var(_)) {
                    return Err(validation_error(
                        "entropy(): edge_type must be an identifier",
                        Some(context),
                        range,
                    ));
                }
                Ok(())
            } else {
                for a in args {
                    match a {
                        CallArg::Positional(e) => {
                            validate_metric_expr_scoped(e, metric_scope, env, context, range)?
                        }
                        CallArg::Named { value, .. } => {
                            validate_metric_expr_scoped(value, metric_scope, env, context, range)?
                        }
                    }
                }
                Ok(())
            }
        }
        other => {
            if env.allow_node {
                Err(validation_error(
                    format!("unsupported function '{}' in node metric expression", other),
                    Some(context),
                    range,
                ))
            } else {
                // Leave unknown top-level metric functions for runtime/registry.
                for a in args {
                    match a {
                        CallArg::Positional(e) => {
                            validate_metric_expr_scoped(e, metric_scope, env, context, range)?
                        }
                        CallArg::Named { value, .. } => {
                            validate_metric_expr_scoped(value, metric_scope, env, context, range)?
                        }
                    }
                }
                Ok(())
            }
        }
    }
}

fn validate_metric_node_field_arg(
    expr: &ExprAst,
    context: &ValidationContext,
    range: Option<SourceRange>,
) -> Result<(), FrontendError> {
    match expr {
        ExprAst::Field { target, .. } if matches!(&**target, ExprAst::Var(v) if v == "node") => {
            Ok(())
        }
        ExprAst::Field { .. } => Err(validation_error(
            "metric node function requires node.attr with 'node' variable",
            Some(context),
            range,
        )),
        _ => Err(validation_error(
            "metric node function requires a field expression",
            Some(context),
            range,
        )),
    }
}

/// Recursively walks an expression tree, applying a visitor function to each node.
///
/// This performs a pre-order traversal, visiting parent nodes before children.
/// The visitor can return an error to short-circuit the traversal.
///
/// # Arguments
///
/// * `expr` - The expression to walk
/// * `f` - A visitor function called on each expression node
///
/// # Returns
///
/// * `Ok(())` - Successfully visited all nodes
/// * `Err(FrontendError)` - Visitor function returned an error
fn walk_expr<F>(expr: &ExprAst, f: &mut F) -> Result<(), FrontendError>
where
    F: FnMut(&ExprAst) -> Result<(), FrontendError>,
{
    f(expr)?;
    match expr {
        ExprAst::Unary { expr, .. } => walk_expr(expr, f)?,
        ExprAst::Binary { left, right, .. } => {
            walk_expr(left, f)?;
            walk_expr(right, f)?;
        }
        ExprAst::Field { target, .. } => walk_expr(target, f)?,
        ExprAst::Call { args, .. } => {
            for a in args {
                match a {
                    CallArg::Positional(e) => walk_expr(e, f)?,
                    CallArg::Named { value, .. } => walk_expr(value, f)?,
                }
            }
        }
        _ => {}
    }
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;

    // ============================================================================
    // Metric Validation Tests
    // ============================================================================

    #[test]
    fn validate_sum_nodes_with_all_required_args() {
        let expr = ExprAst::Call {
            name: "sum_nodes".into(),
            args: vec![
                CallArg::Named {
                    name: "label".into(),
                    value: ExprAst::Var("Person".into()),
                },
                CallArg::Named {
                    name: "contrib".into(),
                    value: ExprAst::Number(1.0),
                },
            ],
        };

        assert!(validate_metric(&expr).is_ok());
    }

    #[test]
    fn validate_sum_nodes_missing_label_fails() {
        let expr = ExprAst::Call {
            name: "sum_nodes".into(),
            args: vec![CallArg::Named {
                name: "contrib".into(),
                value: ExprAst::Number(1.0),
            }],
        };

        let result = validate_metric(&expr);
        assert!(result.is_err());
        assert!(result.unwrap_err().to_string().contains("label"));
    }

    #[test]
    fn validate_sum_nodes_missing_contrib_fails() {
        let expr = ExprAst::Call {
            name: "sum_nodes".into(),
            args: vec![CallArg::Named {
                name: "label".into(),
                value: ExprAst::Var("Person".into()),
            }],
        };

        let result = validate_metric(&expr);
        assert!(result.is_err());
        assert!(result.unwrap_err().to_string().contains("contrib"));
    }

    #[test]
    fn validate_count_nodes_with_label() {
        let expr = ExprAst::Call {
            name: "count_nodes".into(),
            args: vec![CallArg::Named {
                name: "label".into(),
                value: ExprAst::Var("Person".into()),
            }],
        };

        assert!(validate_metric(&expr).is_ok());
    }

    #[test]
    fn validate_count_nodes_missing_label_fails() {
        let expr = ExprAst::Call {
            name: "count_nodes".into(),
            args: vec![],
        };

        let result = validate_metric(&expr);
        assert!(result.is_err());
        assert!(result.unwrap_err().to_string().contains("label"));
    }

    #[test]
    fn validate_avg_degree_with_all_args() {
        let expr = ExprAst::Call {
            name: "avg_degree".into(),
            args: vec![
                CallArg::Named {
                    name: "label".into(),
                    value: ExprAst::Var("Person".into()),
                },
                CallArg::Named {
                    name: "edge_type".into(),
                    value: ExprAst::Var("KNOWS".into()),
                },
            ],
        };

        assert!(validate_metric(&expr).is_ok());
    }

    #[test]
    fn validate_avg_degree_missing_label_fails() {
        let expr = ExprAst::Call {
            name: "avg_degree".into(),
            args: vec![CallArg::Named {
                name: "edge_type".into(),
                value: ExprAst::Var("KNOWS".into()),
            }],
        };

        let result = validate_metric(&expr);
        assert!(result.is_err());
        assert!(result.unwrap_err().to_string().contains("label"));
    }

    #[test]
    fn validate_avg_degree_missing_edge_type_fails() {
        let expr = ExprAst::Call {
            name: "avg_degree".into(),
            args: vec![CallArg::Named {
                name: "label".into(),
                value: ExprAst::Var("Person".into()),
            }],
        };

        let result = validate_metric(&expr);
        assert!(result.is_err());
        assert!(result.unwrap_err().to_string().contains("edge_type"));
    }

    #[test]
    fn validate_fold_nodes_with_all_required_args() {
        let expr = ExprAst::Call {
            name: "fold_nodes".into(),
            args: vec![
                CallArg::Named {
                    name: "label".into(),
                    value: ExprAst::Var("Person".into()),
                },
                CallArg::Named {
                    name: "init".into(),
                    value: ExprAst::Number(0.0),
                },
                CallArg::Named {
                    name: "step".into(),
                    value: ExprAst::Var("acc".into()),
                },
            ],
        };

        assert!(validate_metric(&expr).is_ok());
    }

    #[test]
    fn validate_fold_nodes_missing_init_fails() {
        let expr = ExprAst::Call {
            name: "fold_nodes".into(),
            args: vec![
                CallArg::Named {
                    name: "label".into(),
                    value: ExprAst::Var("Person".into()),
                },
                CallArg::Named {
                    name: "step".into(),
                    value: ExprAst::Var("acc".into()),
                },
            ],
        };

        let result = validate_metric(&expr);
        assert!(result.is_err());
        assert!(result.unwrap_err().to_string().contains("init"));
    }

    #[test]
    fn validate_unknown_metric_passes_through() {
        let expr = ExprAst::Call {
            name: "custom_metric".into(),
            args: vec![],
        };

        assert!(validate_metric(&expr).is_ok());
    }

    // ============================================================================
    // Rule Validation Tests
    // ============================================================================

    #[test]
    fn validate_rule_with_prob_on_edge_var() {
        let rule = RuleDef {
            name: "R".into(),
            on_model: "M".into(),
            mode: Some("for_each".into()),
            patterns: vec![PatternItem {
                src: NodePattern {
                    var: "A".into(),
                    label: "N".into(),
                },
                edge: EdgePattern {
                    var: "e".into(),
                    ty: "E".into(),
                },
                dst: NodePattern {
                    var: "B".into(),
                    label: "N".into(),
                },
            }],
            where_expr: Some(ExprAst::Binary {
                op: BinaryOp::Ge,
                left: Box::new(ExprAst::Call {
                    name: "prob".into(),
                    args: vec![CallArg::Positional(ExprAst::Var("e".into()))],
                }),
                right: Box::new(ExprAst::Number(0.5)),
            }),
            actions: vec![],
        };

        assert!(validate_rule(&rule).is_ok());
    }

    #[test]
    fn validate_rule_with_prob_on_node_var_fails() {
        let rule = RuleDef {
            name: "R".into(),
            on_model: "M".into(),
            mode: Some("for_each".into()),
            patterns: vec![PatternItem {
                src: NodePattern {
                    var: "A".into(),
                    label: "N".into(),
                },
                edge: EdgePattern {
                    var: "e".into(),
                    ty: "E".into(),
                },
                dst: NodePattern {
                    var: "B".into(),
                    label: "N".into(),
                },
            }],
            where_expr: Some(ExprAst::Call {
                name: "prob".into(),
                args: vec![CallArg::Positional(ExprAst::Var("A".into()))],
            }),
            actions: vec![],
        };

        let result = validate_rule(&rule);
        assert!(result.is_err());
        assert!(result.unwrap_err().to_string().contains("edge variable"));
    }

    #[test]
    fn validate_rule_with_prob_correlated_on_comparison() {
        let rule = RuleDef {
            name: "R".into(),
            on_model: "M".into(),
            mode: Some("for_each".into()),
            patterns: vec![PatternItem {
                src: NodePattern {
                    var: "A".into(),
                    label: "N".into(),
                },
                edge: EdgePattern {
                    var: "e".into(),
                    ty: "E".into(),
                },
                dst: NodePattern {
                    var: "B".into(),
                    label: "N".into(),
                },
            }],
            where_expr: Some(ExprAst::Binary {
                op: BinaryOp::Ge,
                left: Box::new(ExprAst::Call {
                    name: "prob_correlated".into(),
                    args: vec![
                        CallArg::Positional(ExprAst::Binary {
                            op: BinaryOp::Gt,
                            left: Box::new(ExprAst::Field {
                                target: Box::new(ExprAst::Var("A".into())),
                                field: "x".into(),
                            }),
                            right: Box::new(ExprAst::Field {
                                target: Box::new(ExprAst::Var("B".into())),
                                field: "x".into(),
                            }),
                        }),
                        CallArg::Named {
                            name: "rho".into(),
                            value: ExprAst::Number(0.3),
                        },
                    ],
                }),
                right: Box::new(ExprAst::Number(0.5)),
            }),
            actions: vec![],
        };

        assert!(validate_rule(&rule).is_ok());
    }

    #[test]
    fn validate_rule_with_prob_correlated_on_edge_var_fails() {
        let rule = RuleDef {
            name: "R".into(),
            on_model: "M".into(),
            mode: Some("for_each".into()),
            patterns: vec![PatternItem {
                src: NodePattern {
                    var: "A".into(),
                    label: "N".into(),
                },
                edge: EdgePattern {
                    var: "e".into(),
                    ty: "E".into(),
                },
                dst: NodePattern {
                    var: "B".into(),
                    label: "N".into(),
                },
            }],
            where_expr: Some(ExprAst::Call {
                name: "prob_correlated".into(),
                args: vec![CallArg::Positional(ExprAst::Var("e".into()))],
            }),
            actions: vec![],
        };

        let result = validate_rule(&rule);
        assert!(result.is_err());
        assert!(result
            .unwrap_err()
            .to_string()
            .contains("comparison expression"));
    }

    #[test]
    fn validate_rule_with_credible_on_edge_var() {
        let rule = RuleDef {
            name: "R".into(),
            on_model: "M".into(),
            mode: Some("for_each".into()),
            patterns: vec![PatternItem {
                src: NodePattern {
                    var: "A".into(),
                    label: "N".into(),
                },
                edge: EdgePattern {
                    var: "e".into(),
                    ty: "E".into(),
                },
                dst: NodePattern {
                    var: "B".into(),
                    label: "N".into(),
                },
            }],
            where_expr: Some(ExprAst::Call {
                name: "credible".into(),
                args: vec![
                    CallArg::Positional(ExprAst::Var("e".into())),
                    CallArg::Named {
                        name: "p".into(),
                        value: ExprAst::Number(0.8),
                    },
                ],
            }),
            actions: vec![],
        };

        assert!(validate_rule(&rule).is_ok());
    }

    #[test]
    fn validate_rule_with_credible_on_comparison_and_rho() {
        let rule = RuleDef {
            name: "R".into(),
            on_model: "M".into(),
            mode: Some("for_each".into()),
            patterns: vec![PatternItem {
                src: NodePattern {
                    var: "A".into(),
                    label: "N".into(),
                },
                edge: EdgePattern {
                    var: "e".into(),
                    ty: "E".into(),
                },
                dst: NodePattern {
                    var: "B".into(),
                    label: "N".into(),
                },
            }],
            where_expr: Some(ExprAst::Call {
                name: "credible".into(),
                args: vec![
                    CallArg::Positional(ExprAst::Binary {
                        op: BinaryOp::Gt,
                        left: Box::new(ExprAst::Field {
                            target: Box::new(ExprAst::Var("A".into())),
                            field: "x".into(),
                        }),
                        right: Box::new(ExprAst::Field {
                            target: Box::new(ExprAst::Var("B".into())),
                            field: "x".into(),
                        }),
                    }),
                    CallArg::Named {
                        name: "p".into(),
                        value: ExprAst::Number(0.9),
                    },
                    CallArg::Named {
                        name: "rho".into(),
                        value: ExprAst::Number(0.2),
                    },
                ],
            }),
            actions: vec![],
        };

        assert!(validate_rule(&rule).is_ok());
    }

    #[test]
    fn validate_rule_with_credible_unknown_named_arg_fails() {
        let rule = RuleDef {
            name: "R".into(),
            on_model: "M".into(),
            mode: Some("for_each".into()),
            patterns: vec![PatternItem {
                src: NodePattern {
                    var: "A".into(),
                    label: "N".into(),
                },
                edge: EdgePattern {
                    var: "e".into(),
                    ty: "E".into(),
                },
                dst: NodePattern {
                    var: "B".into(),
                    label: "N".into(),
                },
            }],
            where_expr: Some(ExprAst::Call {
                name: "credible".into(),
                args: vec![
                    CallArg::Positional(ExprAst::Var("e".into())),
                    CallArg::Named {
                        name: "threshold".into(),
                        value: ExprAst::Number(0.8),
                    },
                ],
            }),
            actions: vec![],
        };

        let result = validate_rule(&rule);
        assert!(result.is_err());
        assert!(result
            .unwrap_err()
            .to_string()
            .contains("expected p or rho"));
    }

    #[test]
    fn validate_rule_with_e_on_valid_field() {
        let rule = RuleDef {
            name: "R".into(),
            on_model: "M".into(),
            mode: Some("for_each".into()),
            patterns: vec![PatternItem {
                src: NodePattern {
                    var: "A".into(),
                    label: "N".into(),
                },
                edge: EdgePattern {
                    var: "e".into(),
                    ty: "E".into(),
                },
                dst: NodePattern {
                    var: "B".into(),
                    label: "N".into(),
                },
            }],
            where_expr: Some(ExprAst::Binary {
                op: BinaryOp::Gt,
                left: Box::new(ExprAst::Call {
                    name: "E".into(),
                    args: vec![CallArg::Positional(ExprAst::Field {
                        target: Box::new(ExprAst::Var("A".into())),
                        field: "x".into(),
                    })],
                }),
                right: Box::new(ExprAst::Number(5.0)),
            }),
            actions: vec![],
        };

        assert!(validate_rule(&rule).is_ok());
    }

    #[test]
    fn validate_rule_bare_field_in_where_fails() {
        let rule = RuleDef {
            name: "R".into(),
            on_model: "M".into(),
            patterns: vec![PatternItem {
                src: NodePattern {
                    var: "A".into(),
                    label: "N".into(),
                },
                edge: EdgePattern {
                    var: "e".into(),
                    ty: "E".into(),
                },
                dst: NodePattern {
                    var: "B".into(),
                    label: "N".into(),
                },
            }],
            where_expr: Some(ExprAst::Binary {
                op: BinaryOp::Gt,
                left: Box::new(ExprAst::Field {
                    target: Box::new(ExprAst::Var("A".into())),
                    field: "x".into(),
                }),
                right: Box::new(ExprAst::Number(0.0)),
            }),
            actions: vec![],
            mode: None,
        };
        let result = validate_rule(&rule);
        assert!(result.is_err());
        let msg = result.unwrap_err().to_string();
        assert!(msg.contains("Bare field access"));
    }

    #[test]
    fn validate_rule_with_e_on_invalid_var_fails() {
        let rule = RuleDef {
            name: "R".into(),
            on_model: "M".into(),
            mode: Some("for_each".into()),
            patterns: vec![PatternItem {
                src: NodePattern {
                    var: "A".into(),
                    label: "N".into(),
                },
                edge: EdgePattern {
                    var: "e".into(),
                    ty: "E".into(),
                },
                dst: NodePattern {
                    var: "B".into(),
                    label: "N".into(),
                },
            }],
            where_expr: Some(ExprAst::Call {
                name: "E".into(),
                args: vec![CallArg::Positional(ExprAst::Field {
                    target: Box::new(ExprAst::Var("Unknown".into())),
                    field: "x".into(),
                })],
            }),
            actions: vec![],
        };

        let result = validate_rule(&rule);
        assert!(result.is_err());
        assert!(result.unwrap_err().to_string().contains("NodeVar"));
    }

    #[test]
    fn validate_rule_with_e_on_non_field_fails() {
        let rule = RuleDef {
            name: "R".into(),
            on_model: "M".into(),
            mode: Some("for_each".into()),
            patterns: vec![PatternItem {
                src: NodePattern {
                    var: "A".into(),
                    label: "N".into(),
                },
                edge: EdgePattern {
                    var: "e".into(),
                    ty: "E".into(),
                },
                dst: NodePattern {
                    var: "B".into(),
                    label: "N".into(),
                },
            }],
            where_expr: Some(ExprAst::Call {
                name: "E".into(),
                args: vec![CallArg::Positional(ExprAst::Var("A".into()))],
            }),
            actions: vec![],
        };

        let result = validate_rule(&rule);
        assert!(result.is_err());
        assert!(result.unwrap_err().to_string().contains("field access"));
    }

    // ============================================================================
    // Prune Predicate Validation Tests
    // ============================================================================

    #[test]
    fn validate_prune_predicate_with_edge_var() {
        let expr = ExprAst::Binary {
            op: BinaryOp::Lt,
            left: Box::new(ExprAst::Call {
                name: "prob".into(),
                args: vec![CallArg::Positional(ExprAst::Var("edge".into()))],
            }),
            right: Box::new(ExprAst::Number(0.1)),
        };

        assert!(validate_prune_predicate(&expr).is_ok());
    }

    #[test]
    fn validate_prune_predicate_with_non_edge_var_fails() {
        let expr = ExprAst::Call {
            name: "prob".into(),
            args: vec![CallArg::Positional(ExprAst::Var("e".into()))],
        };

        let result = validate_prune_predicate(&expr);
        assert!(result.is_err());
        assert!(result.unwrap_err().to_string().contains("'edge'"));
    }

    #[test]
    fn validate_prune_predicate_with_multiple_args_fails() {
        let expr = ExprAst::Call {
            name: "prob".into(),
            args: vec![
                CallArg::Positional(ExprAst::Var("edge".into())),
                CallArg::Positional(ExprAst::Number(0.5)),
            ],
        };

        let result = validate_prune_predicate(&expr);
        assert!(result.is_err());
        assert!(result
            .unwrap_err()
            .to_string()
            .contains("single positional"));
    }

    #[test]
    fn validate_prune_predicate_without_prob_passes() {
        let expr = ExprAst::Binary {
            op: BinaryOp::Gt,
            left: Box::new(ExprAst::Var("x".into())),
            right: Box::new(ExprAst::Number(5.0)),
        };

        assert!(validate_prune_predicate(&expr).is_ok());
    }

    // ============================================================================
    // Walk Expression Tests
    // ============================================================================

    #[test]
    fn walk_expr_visits_all_nodes() {
        let expr = ExprAst::Binary {
            op: BinaryOp::Add,
            left: Box::new(ExprAst::Number(1.0)),
            right: Box::new(ExprAst::Binary {
                op: BinaryOp::Mul,
                left: Box::new(ExprAst::Number(2.0)),
                right: Box::new(ExprAst::Number(3.0)),
            }),
        };

        let mut count = 0;
        walk_expr(&expr, &mut |_| {
            count += 1;
            Ok(())
        })
        .unwrap();

        assert_eq!(count, 5); // 1 + (2 * 3) = 5 nodes total
    }

    #[test]
    fn walk_expr_propagates_errors() {
        let expr = ExprAst::Binary {
            op: BinaryOp::Add,
            left: Box::new(ExprAst::Var("bad".into())),
            right: Box::new(ExprAst::Number(1.0)),
        };

        let result = walk_expr(&expr, &mut |e| {
            if let ExprAst::Var(v) = e {
                if v == "bad" {
                    return Err(FrontendError::ValidationError("found bad var".into()));
                }
            }
            Ok(())
        });

        assert!(result.is_err());
        assert!(result.unwrap_err().to_string().contains("bad var"));
    }

    // ============================================================================
    // Integration Tests
    // ============================================================================

    #[test]
    fn validate_program_with_valid_rule_and_flow() {
        let ast = ProgramAst {
            schemas: vec![],
            belief_models: vec![],
            evidences: vec![],
            rules: vec![RuleDef {
                name: "R".into(),
                on_model: "M".into(),
                mode: Some("for_each".into()),
                patterns: vec![PatternItem {
                    src: NodePattern {
                        var: "A".into(),
                        label: "N".into(),
                    },
                    edge: EdgePattern {
                        var: "e".into(),
                        ty: "E".into(),
                    },
                    dst: NodePattern {
                        var: "B".into(),
                        label: "N".into(),
                    },
                }],
                where_expr: Some(ExprAst::Binary {
                    op: BinaryOp::Ge,
                    left: Box::new(ExprAst::Call {
                        name: "prob".into(),
                        args: vec![CallArg::Positional(ExprAst::Var("e".into()))],
                    }),
                    right: Box::new(ExprAst::Number(0.5)),
                }),
                actions: vec![],
            }],
            flows: vec![],
        };

        assert!(validate_program(&ast).is_ok());
    }

    #[test]
    fn validate_program_with_invalid_metric_fails() {
        let ast = ProgramAst {
            schemas: vec![],
            belief_models: vec![],
            evidences: vec![],
            rules: vec![],
            flows: vec![FlowDef {
                name: "F".into(),
                on_model: "M".into(),
                graphs: vec![],
                metrics: vec![MetricDef {
                    name: "m".into(),
                    expr: ExprAst::Call {
                        name: "sum_nodes".into(),
                        args: vec![], // missing required args
                    },
                }],
                exports: vec![],
                metric_exports: vec![],
                metric_imports: vec![],
            }],
        };

        let result = validate_program(&ast);
        assert!(result.is_err());
    }

    #[test]
    fn validate_belief_model_rejects_non_positive_gaussian_prior_precision() {
        let src = r#"
schema S { node N { x: Real } edge E {} }
belief_model M on S {
  node N { x ~ Gaussian(prior_mean=0.0, prior_precision=0.0) }
  edge E { exist ~ Bernoulli(prior=0.5, pseudo_count=2.0) }
}
"#;
        let ast = crate::parser::parse_program(src).expect("parse");
        let err = validate_program(&ast).expect_err("should reject improper Gaussian prior");
        assert!(err.to_string().contains("prior_precision"));
    }

    #[test]
    fn validate_belief_model_rejects_boundary_bernoulli_prior() {
        let src = r#"
schema S { node N { x: Real } edge E {} }
belief_model M on S {
  edge E { exist ~ Bernoulli(prior=1.0, pseudo_count=2.0) }
}
"#;
        let ast = crate::parser::parse_program(src).expect("parse");
        let err = validate_program(&ast).expect_err("should reject boundary Bernoulli prior");
        assert!(err.to_string().contains("must be in (0, 1)"));
    }

    #[test]
    fn validate_belief_model_rejects_unknown_bernoulli_parameter() {
        let src = r#"
schema S { node N { x: Real } edge E {} }
belief_model M on S {
  edge E { exist ~ Bernoulli(prior=0.5, mystery=2.0) }
}
"#;
        let ast = crate::parser::parse_program(src).expect("parse");
        let err = validate_program(&ast).expect_err("should reject unknown Bernoulli parameter");
        assert!(err.to_string().contains("unknown parameter"));
    }

    #[test]
    fn validate_belief_model_rejects_non_gaussian_node_attribute_posterior() {
        let src = r#"
schema S { node N { x: Real } edge E {} }
belief_model M on S {
  node N { x ~ Bernoulli(prior=0.5, pseudo_count=2.0) }
  edge E { exist ~ Bernoulli(prior=0.5, pseudo_count=2.0) }
}
"#;
        let ast = crate::parser::parse_program(src).expect("parse");
        let err = validate_program(&ast).expect_err("should reject non-Gaussian node posterior");
        assert!(err.to_string().contains("must use GaussianPosterior"));
    }

    #[test]
    fn validate_program_with_source_reports_context_and_range() {
        let src = r#"
schema S { node N { x: Real } edge E {} }
belief_model M on S {}
rule R on M {
  pattern (A:N)-[e:E]->(B:N)
  where prob(A) >= 0.5
}
"#;
        let ast = crate::parser::parse_program(src).expect("parse");
        let err = validate_program_with_source(&ast, src).expect_err("should fail");
        let diag = err
            .validation_diagnostic()
            .expect("expected rich validation diagnostic");
        assert!(format!("{}", diag).contains("rule 'R' where clause"));
        assert!(diag.range.is_some());
    }

    #[test]
    fn validate_rule_action_expr_unknown_var_fails() {
        let rule = RuleDef {
            name: "R".into(),
            on_model: "M".into(),
            mode: Some("for_each".into()),
            patterns: vec![PatternItem {
                src: NodePattern {
                    var: "A".into(),
                    label: "N".into(),
                },
                edge: EdgePattern {
                    var: "e".into(),
                    ty: "E".into(),
                },
                dst: NodePattern {
                    var: "B".into(),
                    label: "N".into(),
                },
            }],
            where_expr: None,
            actions: vec![ActionStmt::SetExpectation {
                node_var: "A".into(),
                attr: "x".into(),
                expr: ExprAst::Var("unknown_local".into()),
            }],
        };
        let err = validate_rule(&rule).expect_err("should fail");
        assert!(err.to_string().contains("unknown variable"));
    }
}
