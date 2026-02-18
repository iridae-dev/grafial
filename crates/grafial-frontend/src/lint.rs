//! Statistical guardrail linting and lint-suppression pragmas.
//!
//! Phase 7 introduces warning/info diagnostics (non-fatal lints) for
//! statistically risky patterns plus scoped suppression pragmas:
//! `// grafial-lint: ignore(<code>)`.

use std::collections::{HashMap, HashSet};

use pest::Parser;

use crate::ast::*;
use crate::errors::{SourcePosition, SourceRange};
use crate::parser::{BayGraphParser, Rule};

/// Stable lint code: repeated non-Bayesian nudges can collapse variance.
pub const LINT_STAT_VARIANCE_COLLAPSE: &str = "stat_variance_collapse";
/// Stable lint code: prior strength can dominate incoming evidence.
pub const LINT_STAT_PRIOR_DOMINANCE: &str = "stat_prior_dominance";
/// Stable lint code: precision/pseudo-count appears numerically extreme.
pub const LINT_STAT_PRECISION_OUTLIER: &str = "stat_precision_outlier";
/// Stable lint code: evidence appears in strong conflict with prior.
pub const LINT_STAT_PRIOR_DATA_CONFLICT: &str = "stat_prior_data_conflict";
/// Stable lint code: expression or update parameters look numerically unstable.
pub const LINT_STAT_NUMERICAL_INSTABILITY: &str = "stat_numerical_instability";
/// Stable lint code: repeated threshold checks suggest multiple testing risk.
pub const LINT_STAT_MULTIPLE_TESTING: &str = "stat_multiple_testing";
/// Stable lint code: update graph suggests circular/feedback update risk.
pub const LINT_STAT_CIRCULAR_UPDATE: &str = "stat_circular_update";
/// Stable lint code: explanatory delete diagnostic.
pub const LINT_STAT_DELETE_EXPLANATION: &str = "stat_delete_explanation";
/// Stable lint code: explanatory suppress diagnostic.
pub const LINT_STAT_SUPPRESS_EXPLANATION: &str = "stat_suppress_explanation";

/// Lint severity used by frontend statistical lints.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum LintSeverity {
    Warning,
    Information,
}

/// Statistical or explanatory lint surfaced by frontend/LSP.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct StatisticalLint {
    pub code: &'static str,
    pub message: String,
    pub range: SourceRange,
    pub severity: LintSeverity,
}

/// Parsed lint-suppression pragma scope.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct LintSuppression {
    start_line: u32,
    end_line: u32,
    codes: Vec<String>,
}

impl LintSuppression {
    fn suppresses(&self, code: &str, line: u32) -> bool {
        if line < self.start_line || line > self.end_line {
            return false;
        }
        self.codes.iter().any(|c| c == "all" || c == code)
    }
}

#[derive(Debug, Default, Clone)]
struct RuleSourceEntry {
    decl_range: Option<SourceRange>,
    where_expr: Option<SourceRange>,
    action_exprs: Vec<SourceRange>,
}

#[derive(Debug, Default, Clone)]
struct FlowSourceEntry {
    decl_range: Option<SourceRange>,
    metric_exprs: Vec<SourceRange>,
}

#[derive(Debug, Default, Clone)]
struct LintSourceMap {
    belief_models: Vec<SourceRange>,
    evidences: Vec<SourceRange>,
    rules: Vec<RuleSourceEntry>,
    flows: Vec<FlowSourceEntry>,
    declarations: Vec<SourceRange>,
}

fn fallback_range() -> SourceRange {
    SourceRange {
        start: SourcePosition { line: 1, column: 1 },
        end: SourcePosition { line: 1, column: 1 },
    }
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

fn build_source_map(source: &str) -> Option<LintSourceMap> {
    let mut map = LintSourceMap::default();
    let mut pairs = BayGraphParser::parse(Rule::program, source).ok()?;
    let program_pair = pairs.next()?;
    for item in program_pair.into_inner() {
        match item.as_rule() {
            Rule::decl => {
                for decl in item.into_inner() {
                    collect_decl_source(decl, &mut map);
                }
            }
            Rule::schema_decl
            | Rule::belief_model_decl
            | Rule::evidence_decl
            | Rule::rule_decl
            | Rule::flow_decl => collect_decl_source(item, &mut map),
            _ => {}
        }
    }
    Some(map)
}

fn collect_decl_source(decl: pest::iterators::Pair<Rule>, map: &mut LintSourceMap) {
    let decl_range = as_source_range(decl.as_span());
    map.declarations.push(decl_range);
    match decl.as_rule() {
        Rule::belief_model_decl => map.belief_models.push(decl_range),
        Rule::evidence_decl => map.evidences.push(decl_range),
        Rule::rule_decl => map.rules.push(collect_rule_source(decl, decl_range)),
        Rule::flow_decl => map.flows.push(collect_flow_source(decl, decl_range)),
        _ => {}
    }
}

fn collect_rule_source(
    rule_decl: pest::iterators::Pair<Rule>,
    decl_range: SourceRange,
) -> RuleSourceEntry {
    let mut entry = RuleSourceEntry {
        decl_range: Some(decl_range),
        ..RuleSourceEntry::default()
    };
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
                                collect_action_spans(part, &mut entry.action_exprs)
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
                                collect_action_spans(part, &mut entry.action_exprs)
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

fn collect_flow_source(
    flow_decl: pest::iterators::Pair<Rule>,
    decl_range: SourceRange,
) -> FlowSourceEntry {
    let mut entry = FlowSourceEntry {
        decl_range: Some(decl_range),
        ..FlowSourceEntry::default()
    };
    for p in flow_decl.into_inner() {
        if p.as_rule() != Rule::flow_body {
            continue;
        }
        for b in p.into_inner() {
            if b.as_rule() == Rule::metric_stmt {
                if let Some(range) = find_metric_expr_span(b) {
                    entry.metric_exprs.push(range);
                }
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

fn parse_ignore_codes(comment: &str) -> Option<Vec<String>> {
    let marker = "grafial-lint:";
    let marker_idx = comment.find(marker)?;
    let rest = comment[marker_idx + marker.len()..].trim_start();
    let rest = rest.strip_prefix("ignore(")?;
    let end_idx = rest.find(')')?;
    let inside = &rest[..end_idx];
    let mut codes = Vec::new();
    for code in inside.split(',') {
        let normalized = code.trim().to_ascii_lowercase();
        if normalized.is_empty() {
            continue;
        }
        if normalized == "*" {
            codes.push("all".to_string());
        } else {
            codes.push(normalized);
        }
    }
    if codes.is_empty() {
        None
    } else {
        Some(codes)
    }
}

fn collect_lint_suppressions_with_ranges(
    source: &str,
    declaration_ranges: &[SourceRange],
) -> Vec<LintSuppression> {
    let mut sorted_ranges = declaration_ranges.to_vec();
    sorted_ranges.sort_by_key(|range| (range.start.line, range.start.column));

    let mut suppressions = Vec::new();

    for (line_idx, line_text) in source.lines().enumerate() {
        let Some(comment_idx) = line_text.find("//") else {
            continue;
        };
        let comment = &line_text[comment_idx + 2..];
        let Some(codes) = parse_ignore_codes(comment) else {
            continue;
        };
        let line_no = (line_idx + 1) as u32;

        let end_line = if let Some(in_decl) = sorted_ranges
            .iter()
            .find(|range| line_no >= range.start.line && line_no <= range.end.line)
        {
            in_decl.end.line
        } else if let Some(next_decl) = sorted_ranges
            .iter()
            .find(|range| range.start.line > line_no)
        {
            next_decl.end.line
        } else {
            u32::MAX
        };

        suppressions.push(LintSuppression {
            start_line: line_no,
            end_line,
            codes,
        });
    }

    suppressions
}

/// Collect scoped lint suppressions from `// grafial-lint: ignore(<code>)` pragmas.
pub fn collect_lint_suppressions(source: &str) -> Vec<LintSuppression> {
    let decl_ranges = build_source_map(source)
        .map(|map| map.declarations)
        .unwrap_or_default();
    collect_lint_suppressions_with_ranges(source, &decl_ranges)
}

/// Returns true when the given lint code/range is suppressed by a pragma.
pub fn lint_is_suppressed(
    suppressions: &[LintSuppression],
    code: &str,
    range: SourceRange,
) -> bool {
    let normalized = code.to_ascii_lowercase();
    suppressions
        .iter()
        .any(|suppression| suppression.suppresses(&normalized, range.start.line))
}

fn push_lint(
    out: &mut Vec<StatisticalLint>,
    code: &'static str,
    message: impl Into<String>,
    range: SourceRange,
    severity: LintSeverity,
) {
    out.push(StatisticalLint {
        code,
        message: message.into(),
        range,
        severity,
    });
}

fn param_value(params: &[(String, f64)], name: &str) -> Option<f64> {
    params
        .iter()
        .find_map(|(param_name, value)| (param_name == name).then_some(*value))
}

fn format_obs_count(n: u64) -> String {
    if n >= 1_000_000_000 {
        format!("{:.1}B", (n as f64) / 1_000_000_000.0)
    } else if n >= 1_000_000 {
        format!("{:.1}M", (n as f64) / 1_000_000.0)
    } else if n >= 1_000 {
        format!("{:.1}K", (n as f64) / 1_000.0)
    } else {
        n.to_string()
    }
}

fn count_prob_calls(expr: &ExprAst) -> usize {
    let mut count = 0usize;
    walk_expr(expr, &mut |node| {
        if let ExprAst::Call { name, .. } = node {
            if name == "prob" || name == "prob_correlated" || name == "credible" {
                count += 1;
            }
        }
    });
    count
}

fn expr_has_tiny_divisor(expr: &ExprAst) -> bool {
    let mut found = false;
    walk_expr(expr, &mut |node| {
        if let ExprAst::Binary {
            op: BinaryOp::Div,
            right,
            ..
        } = node
        {
            if let ExprAst::Number(n) = &**right {
                if n.abs() < 1e-9 {
                    found = true;
                }
            }
        }
    });
    found
}

fn collect_node_refs(expr: &ExprAst, node_vars: &HashSet<String>, out: &mut HashSet<String>) {
    walk_expr(expr, &mut |node| match node {
        ExprAst::Var(name) if node_vars.contains(name) => {
            out.insert(name.clone());
        }
        ExprAst::Field { target, .. } => {
            if let ExprAst::Var(name) = &**target {
                if node_vars.contains(name) {
                    out.insert(name.clone());
                }
            }
        }
        _ => {}
    });
}

fn walk_expr<F>(expr: &ExprAst, f: &mut F)
where
    F: FnMut(&ExprAst),
{
    f(expr);
    match expr {
        ExprAst::Unary { expr, .. } => walk_expr(expr, f),
        ExprAst::Binary { left, right, .. } => {
            walk_expr(left, f);
            walk_expr(right, f);
        }
        ExprAst::Field { target, .. } => walk_expr(target, f),
        ExprAst::Call { args, .. } => {
            for arg in args {
                match arg {
                    CallArg::Positional(expr) => walk_expr(expr, f),
                    CallArg::Named { value, .. } => walk_expr(value, f),
                }
            }
        }
        ExprAst::Exists {
            where_expr,
            pattern,
            ..
        } => {
            // Pattern fields are identifiers, not nested expressions.
            let _ = pattern;
            if let Some(where_expr) = where_expr {
                walk_expr(where_expr, f);
            }
        }
        ExprAst::Number(_) | ExprAst::Bool(_) | ExprAst::Var(_) => {}
    }
}

fn emit_prior_and_precision_lints(
    ast: &ProgramAst,
    source_map: Option<&LintSourceMap>,
    out: &mut Vec<StatisticalLint>,
) {
    for (idx, model) in ast.belief_models.iter().enumerate() {
        let model_range = source_map
            .and_then(|map| map.belief_models.get(idx).copied())
            .unwrap_or_else(fallback_range);

        for node_decl in &model.nodes {
            for (attr_name, posterior) in &node_decl.attrs {
                if let PosteriorType::Gaussian { params } = posterior {
                    let tau = param_value(params, "prior_precision").unwrap_or(1.0);
                    if tau > 100.0 {
                        push_lint(
                            out,
                            LINT_STAT_PRIOR_DOMINANCE,
                            format!(
                                "Gaussian prior for {}.{} in belief_model '{}' has high prior_precision={} and may dominate data",
                                node_decl.node_type, attr_name, model.name, tau
                            ),
                            model_range,
                            LintSeverity::Warning,
                        );
                    }
                    if !(1e-6..=1e6).contains(&tau) {
                        push_lint(
                            out,
                            LINT_STAT_PRECISION_OUTLIER,
                            format!(
                                "Gaussian prior_precision={} for {}.{} in belief_model '{}' is an outlier and may cause unstable inference",
                                tau, node_decl.node_type, attr_name, model.name
                            ),
                            model_range,
                            LintSeverity::Warning,
                        );
                    }
                }
            }
        }

        for edge_decl in &model.edges {
            match &edge_decl.exist {
                PosteriorType::Bernoulli { params } => {
                    let pseudo = param_value(params, "pseudo_count").unwrap_or(1.0);
                    if pseudo > 1000.0 {
                        push_lint(
                            out,
                            LINT_STAT_PRIOR_DOMINANCE,
                            format!(
                                "Bernoulli pseudo_count={} for edge '{}' in belief_model '{}' is strong and may dominate evidence",
                                pseudo, edge_decl.edge_type, model.name
                            ),
                            model_range,
                            LintSeverity::Warning,
                        );
                    }
                    if !(1e-6..=1e9).contains(&pseudo) {
                        push_lint(
                            out,
                            LINT_STAT_PRECISION_OUTLIER,
                            format!(
                                "Bernoulli pseudo_count={} for edge '{}' in belief_model '{}' is an outlier",
                                pseudo, edge_decl.edge_type, model.name
                            ),
                            model_range,
                            LintSeverity::Warning,
                        );
                    }
                }
                PosteriorType::Categorical { prior, .. } => match prior {
                    CategoricalPrior::Uniform { pseudo_count } => {
                        if *pseudo_count > 1000.0 {
                            push_lint(
                                out,
                                LINT_STAT_PRIOR_DOMINANCE,
                                format!(
                                    "Categorical pseudo_count={} for edge '{}' in belief_model '{}' is strong and may dominate evidence",
                                    pseudo_count, edge_decl.edge_type, model.name
                                ),
                                model_range,
                                LintSeverity::Warning,
                            );
                        }
                        if !(1e-6..=1e9).contains(pseudo_count) {
                            push_lint(
                                out,
                                LINT_STAT_PRECISION_OUTLIER,
                                format!(
                                    "Categorical pseudo_count={} for edge '{}' in belief_model '{}' is an outlier",
                                    pseudo_count, edge_decl.edge_type, model.name
                                ),
                                model_range,
                                LintSeverity::Warning,
                            );
                        }
                    }
                    CategoricalPrior::Explicit { concentrations } => {
                        let total: f64 = concentrations.iter().sum();
                        if total > (1000.0 * concentrations.len() as f64) {
                            push_lint(
                                out,
                                LINT_STAT_PRIOR_DOMINANCE,
                                format!(
                                    "Categorical explicit prior total={} for edge '{}' in belief_model '{}' is strong and may dominate evidence",
                                    total, edge_decl.edge_type, model.name
                                ),
                                model_range,
                                LintSeverity::Warning,
                            );
                        }
                    }
                },
                PosteriorType::Gaussian { .. } => {}
            }
        }
    }
}

fn emit_prior_data_conflict_lints(
    ast: &ProgramAst,
    source_map: Option<&LintSourceMap>,
    out: &mut Vec<StatisticalLint>,
) {
    let model_by_name: HashMap<&str, &BeliefModel> = ast
        .belief_models
        .iter()
        .map(|model| (model.name.as_str(), model))
        .collect();

    for (idx, evidence) in ast.evidences.iter().enumerate() {
        let evidence_range = source_map
            .and_then(|map| map.evidences.get(idx).copied())
            .unwrap_or_else(fallback_range);
        let Some(model) = model_by_name.get(evidence.on_model.as_str()) else {
            continue;
        };

        let mut gaussian_priors: HashMap<(String, String), (f64, f64)> = HashMap::new();
        for node_decl in &model.nodes {
            for (attr_name, posterior) in &node_decl.attrs {
                if let PosteriorType::Gaussian { params } = posterior {
                    let prior_mean = param_value(params, "prior_mean").unwrap_or(0.0);
                    let prior_precision = param_value(params, "prior_precision").unwrap_or(1.0);
                    gaussian_priors.insert(
                        (node_decl.node_type.clone(), attr_name.clone()),
                        (prior_mean, prior_precision),
                    );
                }
            }
        }

        let mut edge_priors: HashMap<String, (f64, f64)> = HashMap::new();
        for edge_decl in &model.edges {
            if let PosteriorType::Bernoulli { params } = &edge_decl.exist {
                let prior = param_value(params, "prior").unwrap_or(0.5);
                let pseudo_count = param_value(params, "pseudo_count").unwrap_or(1.0);
                edge_priors.insert(edge_decl.edge_type.clone(), (prior, pseudo_count));
            }
        }

        for obs in &evidence.observations {
            match obs {
                ObserveStmt::Attribute {
                    node,
                    attr,
                    value,
                    precision,
                } => {
                    let key = (node.0.clone(), attr.clone());
                    if let Some((prior_mean, prior_precision)) = gaussian_priors.get(&key).copied()
                    {
                        let z = (value - prior_mean).abs() * prior_precision.max(1e-12).sqrt();
                        if prior_precision >= 1.0 && z >= 6.0 {
                            push_lint(
                                out,
                                LINT_STAT_PRIOR_DATA_CONFLICT,
                                format!(
                                    "Observation {}.{}={} in evidence '{}' strongly conflicts with prior mean={} (z≈{:.2})",
                                    node.0, attr, value, evidence.name, prior_mean, z
                                ),
                                evidence_range,
                                LintSeverity::Warning,
                            );
                        }
                    }
                    if let Some(obs_precision) = precision {
                        if !obs_precision.is_finite()
                            || *obs_precision <= 0.0
                            || *obs_precision < 1e-9
                            || *obs_precision > 1e9
                        {
                            push_lint(
                                out,
                                LINT_STAT_NUMERICAL_INSTABILITY,
                                format!(
                                    "Observation precision={} for {}.{} in evidence '{}' may cause numerical instability",
                                    obs_precision, node.0, attr, evidence.name
                                ),
                                evidence_range,
                                LintSeverity::Warning,
                            );
                        }
                    }
                }
                ObserveStmt::Edge {
                    edge_type, mode, ..
                } => {
                    let Some((prior, pseudo_count)) = edge_priors.get(edge_type).copied() else {
                        continue;
                    };
                    if pseudo_count >= 100.0 {
                        let conflict = matches!(mode, EvidenceMode::Present) && prior <= 0.05
                            || matches!(mode, EvidenceMode::Absent) && prior >= 0.95;
                        if conflict {
                            push_lint(
                                out,
                                LINT_STAT_PRIOR_DATA_CONFLICT,
                                format!(
                                    "Evidence '{}' has {:?} observation for edge '{}' against strong Bernoulli prior={} (pseudo_count={})",
                                    evidence.name, mode, edge_type, prior, pseudo_count
                                ),
                                evidence_range,
                                LintSeverity::Warning,
                            );
                        }
                    }
                }
            }
        }
    }
}

fn emit_rule_pattern_lints(
    ast: &ProgramAst,
    source_map: Option<&LintSourceMap>,
    out: &mut Vec<StatisticalLint>,
) {
    for (idx, rule) in ast.rules.iter().enumerate() {
        let source_entry = source_map.and_then(|map| map.rules.get(idx));
        let rule_range = source_entry
            .and_then(|entry| entry.decl_range)
            .unwrap_or_else(fallback_range);
        let where_range = source_entry
            .and_then(|entry| entry.where_expr)
            .unwrap_or(rule_range);

        if let Some(where_expr) = &rule.where_expr {
            let prob_calls = count_prob_calls(where_expr);
            if prob_calls >= 3 {
                push_lint(
                    out,
                    LINT_STAT_MULTIPLE_TESTING,
                    format!(
                        "Rule '{}' where clause performs {} probability checks (prob/prob_correlated/credible); consider multiple-testing control",
                        rule.name, prob_calls
                    ),
                    where_range,
                    LintSeverity::Warning,
                );
            }
            if expr_has_tiny_divisor(where_expr) {
                push_lint(
                    out,
                    LINT_STAT_NUMERICAL_INSTABILITY,
                    format!(
                        "Rule '{}' where clause divides by a very small literal, which may be numerically unstable",
                        rule.name
                    ),
                    where_range,
                    LintSeverity::Warning,
                );
            }
        }

        let node_vars: HashSet<String> = rule
            .patterns
            .iter()
            .flat_map(|pattern| [pattern.src.var.clone(), pattern.dst.var.clone()])
            .collect();

        let mut nudge_counts: HashMap<(String, String), usize> = HashMap::new();
        let mut update_refs: HashMap<String, HashSet<String>> = HashMap::new();

        for (action_idx, action) in rule.actions.iter().enumerate() {
            let action_range = source_entry
                .and_then(|entry| entry.action_exprs.get(action_idx).copied())
                .unwrap_or(rule_range);

            match action {
                ActionStmt::SetExpectation {
                    node_var,
                    attr,
                    expr,
                    ..
                }
                | ActionStmt::NonBayesianNudge {
                    node_var,
                    attr,
                    expr,
                    ..
                } => {
                    *nudge_counts
                        .entry((node_var.clone(), attr.clone()))
                        .or_insert(0) += 1;

                    let mut refs = HashSet::new();
                    collect_node_refs(expr, &node_vars, &mut refs);
                    refs.remove(node_var);
                    if !refs.is_empty() {
                        update_refs
                            .entry(node_var.clone())
                            .or_default()
                            .extend(refs);
                    }

                    if expr_has_tiny_divisor(expr) {
                        push_lint(
                            out,
                            LINT_STAT_NUMERICAL_INSTABILITY,
                            format!(
                                "Rule '{}' action on {}.{} divides by a very small literal",
                                rule.name, node_var, attr
                            ),
                            action_range,
                            LintSeverity::Warning,
                        );
                    }
                }
                ActionStmt::SoftUpdate {
                    node_var,
                    attr,
                    expr,
                    precision,
                    count,
                } => {
                    let mut refs = HashSet::new();
                    collect_node_refs(expr, &node_vars, &mut refs);
                    refs.remove(node_var);
                    if !refs.is_empty() {
                        update_refs
                            .entry(node_var.clone())
                            .or_default()
                            .extend(refs);
                    }
                    if expr_has_tiny_divisor(expr) {
                        push_lint(
                            out,
                            LINT_STAT_NUMERICAL_INSTABILITY,
                            format!(
                                "Rule '{}' soft_update on {}.{} divides by a very small literal",
                                rule.name, node_var, attr
                            ),
                            action_range,
                            LintSeverity::Warning,
                        );
                    }
                    if let Some(precision) = precision {
                        if !precision.is_finite()
                            || *precision <= 0.0
                            || *precision < 1e-9
                            || *precision > 1e9
                        {
                            push_lint(
                                out,
                                LINT_STAT_NUMERICAL_INSTABILITY,
                                format!(
                                    "Rule '{}' soft_update precision={} on {}.{} may be numerically unstable",
                                    rule.name, precision, node_var, attr
                                ),
                                action_range,
                                LintSeverity::Warning,
                            );
                        }
                    }
                    if let Some(count) = count {
                        if !count.is_finite() || *count <= 0.0 || *count > 1e6 {
                            push_lint(
                                out,
                                LINT_STAT_NUMERICAL_INSTABILITY,
                                format!(
                                    "Rule '{}' soft_update count={} on {}.{} may be numerically unstable",
                                    rule.name, count, node_var, attr
                                ),
                                action_range,
                                LintSeverity::Warning,
                            );
                        }
                    }
                }
                ActionStmt::DeleteEdge {
                    edge_var,
                    confidence,
                } => {
                    let beta: f64 = match confidence
                        .as_deref()
                        .map(|value| value.to_ascii_lowercase())
                    {
                        Some(value) if value == "low" => 1_000.0,
                        Some(value) if value == "high" => 1_000_000_000.0,
                        _ => 1_000_000.0,
                    };
                    let needed = (beta - 1.0).max(0.0).ceil() as u64;
                    push_lint(
                        out,
                        LINT_STAT_DELETE_EXPLANATION,
                        format!(
                            "delete {} sets Beta(alpha=1, beta={:.0}); approximately {} present observations are needed to lift mean existence above 0.5",
                            edge_var,
                            beta,
                            format_obs_count(needed)
                        ),
                        action_range,
                        LintSeverity::Information,
                    );
                }
                ActionStmt::SuppressEdge { edge_var, weight } => {
                    let beta = weight.unwrap_or(10.0).max(0.01);
                    let needed = (beta - 1.0).max(0.0).ceil() as u64;
                    push_lint(
                        out,
                        LINT_STAT_SUPPRESS_EXPLANATION,
                        format!(
                            "suppress {} sets Beta(alpha=1, beta={:.2}); approximately {} present observations are needed to lift mean existence above 0.5",
                            edge_var,
                            beta,
                            format_obs_count(needed)
                        ),
                        action_range,
                        LintSeverity::Information,
                    );
                }
                ActionStmt::Let { expr, .. } => {
                    if expr_has_tiny_divisor(expr) {
                        push_lint(
                            out,
                            LINT_STAT_NUMERICAL_INSTABILITY,
                            format!(
                                "Rule '{}' let-expression divides by a very small literal",
                                rule.name
                            ),
                            action_range,
                            LintSeverity::Warning,
                        );
                    }
                }
                ActionStmt::ForceAbsent { .. } => {}
            }
        }

        if let Some(((node_var, attr), count)) = nudge_counts
            .iter()
            .find(|(_, action_count)| **action_count >= 2)
        {
            push_lint(
                out,
                LINT_STAT_VARIANCE_COLLAPSE,
                format!(
                    "Rule '{}' applies {} non-Bayesian nudges to {}.{}; repeated manual shifts can collapse calibrated uncertainty",
                    rule.name, count, node_var, attr
                ),
                rule_range,
                LintSeverity::Warning,
            );
        }

        let mut warned_circular = false;
        for (left, left_refs) in &update_refs {
            for right in left_refs {
                if left == right {
                    continue;
                }
                if let Some(right_refs) = update_refs.get(right) {
                    if right_refs.contains(left) {
                        warned_circular = true;
                        break;
                    }
                }
            }
            if warned_circular {
                break;
            }
        }
        if warned_circular {
            push_lint(
                out,
                LINT_STAT_CIRCULAR_UPDATE,
                format!(
                    "Rule '{}' appears to contain circular cross-node updates; consider damping, thresholds, or sequencing",
                    rule.name
                ),
                rule_range,
                LintSeverity::Warning,
            );
        }
    }
}

fn emit_flow_lints(
    ast: &ProgramAst,
    source_map: Option<&LintSourceMap>,
    out: &mut Vec<StatisticalLint>,
) {
    for (flow_idx, flow) in ast.flows.iter().enumerate() {
        let source_entry = source_map.and_then(|map| map.flows.get(flow_idx));
        let flow_range = source_entry
            .and_then(|entry| entry.decl_range)
            .unwrap_or_else(fallback_range);
        for (metric_idx, metric) in flow.metrics.iter().enumerate() {
            let metric_range = source_entry
                .and_then(|entry| entry.metric_exprs.get(metric_idx).copied())
                .unwrap_or(flow_range);
            if expr_has_tiny_divisor(&metric.expr) {
                push_lint(
                    out,
                    LINT_STAT_NUMERICAL_INSTABILITY,
                    format!(
                        "Flow '{}' metric '{}' divides by a very small literal",
                        flow.name, metric.name
                    ),
                    metric_range,
                    LintSeverity::Warning,
                );
            }
        }
    }
}

/// Produces Phase 7 statistical/explanatory lints for an AST + source pair.
///
/// This pass is warning/info-only and does not fail compilation.
pub fn lint_statistical_guardrails(ast: &ProgramAst, source: &str) -> Vec<StatisticalLint> {
    let source_map = build_source_map(source);
    let mut lints = Vec::new();

    emit_prior_and_precision_lints(ast, source_map.as_ref(), &mut lints);
    emit_prior_data_conflict_lints(ast, source_map.as_ref(), &mut lints);
    emit_rule_pattern_lints(ast, source_map.as_ref(), &mut lints);
    emit_flow_lints(ast, source_map.as_ref(), &mut lints);

    let declaration_ranges = source_map
        .as_ref()
        .map(|map| map.declarations.as_slice())
        .unwrap_or(&[]);
    let suppressions = collect_lint_suppressions_with_ranges(source, declaration_ranges);

    lints
        .into_iter()
        .filter(|lint| !lint_is_suppressed(&suppressions, lint.code, lint.range))
        .collect()
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::parse_program;

    fn lint_codes(source: &str) -> Vec<&'static str> {
        let ast = parse_program(source).expect("program should parse");
        lint_statistical_guardrails(&ast, source)
            .into_iter()
            .map(|lint| lint.code)
            .collect()
    }

    #[test]
    fn emits_variance_collapse_and_circular_update_lints() {
        let source = r#"
schema S { node N { x: Real } edge E { } }
belief_model M on S {
  node N { x ~ Gaussian(mean=0.0, precision=1.0) }
  edge E { exist ~ Bernoulli(prior=0.5, weight=2.0) }
}
evidence Ev on M { N { "a" { x: 0.0 }, "b" { x: 1.0 } } E(N -> N) { "a" -> "b" } }
rule R on M {
  pattern
    (A:N)-[ab:E]->(B:N)
  where prob(ab) > 0.1 and prob(ab) > 0.2 and prob(ab) > 0.3
  action {
    non_bayesian_nudge A.x to E[B.x] variance=preserve
    non_bayesian_nudge B.x to E[A.x] variance=preserve
    non_bayesian_nudge A.x to E[A.x] + 0.1 variance=decrease(factor=2.0)
    delete ab confidence=high
    suppress ab weight=20
  }
}
flow F on M { graph g = from_evidence Ev }
"#;
        let codes = lint_codes(source);
        assert!(codes.contains(&LINT_STAT_MULTIPLE_TESTING));
        assert!(codes.contains(&LINT_STAT_VARIANCE_COLLAPSE));
        assert!(codes.contains(&LINT_STAT_CIRCULAR_UPDATE));
        assert!(codes.contains(&LINT_STAT_DELETE_EXPLANATION));
        assert!(codes.contains(&LINT_STAT_SUPPRESS_EXPLANATION));
    }

    #[test]
    fn emits_prior_dominance_and_conflict_lints() {
        let source = r#"
schema S { node N { x: Real } edge E { } }
belief_model M on S {
  node N { x ~ Gaussian(mean=0.0, precision=500.0) }
  edge E { exist ~ Bernoulli(prior=0.99, weight=5000.0) }
}
evidence Ev on M {
  N { "a" { x: 100.0 (precision=10000000000.0) } }
}
flow F on M { graph g = from_evidence Ev }
"#;
        let codes = lint_codes(source);
        assert!(codes.contains(&LINT_STAT_PRIOR_DOMINANCE));
        assert!(codes.contains(&LINT_STAT_PRIOR_DATA_CONFLICT));
        assert!(codes.contains(&LINT_STAT_NUMERICAL_INSTABILITY));
    }

    #[test]
    fn lint_suppression_ignores_next_declaration_scope() {
        let source = r#"
schema S { node N { x: Real } edge E { } }
// grafial-lint: ignore(stat_prior_dominance)
belief_model M on S {
  node N { x ~ Gaussian(mean=0.0, precision=1000.0) }
  edge E { exist ~ Bernoulli(prior=0.5, weight=2.0) }
}
flow F on M { graph g = from_evidence Ev }
"#;
        let ast = parse_program(source).expect("program should parse");
        let lints = lint_statistical_guardrails(&ast, source);
        assert!(!lints
            .iter()
            .any(|lint| lint.code == LINT_STAT_PRIOR_DOMINANCE));
    }

    #[test]
    fn lint_is_suppressed_matches_code_and_scope_line() {
        let suppressions = vec![LintSuppression {
            start_line: 2,
            end_line: 10,
            codes: vec![LINT_STAT_MULTIPLE_TESTING.to_string()],
        }];
        let range = SourceRange {
            start: SourcePosition { line: 4, column: 1 },
            end: SourcePosition {
                line: 4,
                column: 10,
            },
        };
        assert!(lint_is_suppressed(
            &suppressions,
            LINT_STAT_MULTIPLE_TESTING,
            range
        ));
        assert!(!lint_is_suppressed(
            &suppressions,
            LINT_STAT_CIRCULAR_UPDATE,
            range
        ));
    }
}
