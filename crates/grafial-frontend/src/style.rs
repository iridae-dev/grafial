//! Canonical style linting and formatting helpers.
//!
//! Phase 5 tooling uses these helpers to surface modernization diagnostics
//! and quick fixes for compatibility syntax forms.

use crate::errors::{SourcePosition, SourceRange};

/// Canonical style lint with an auto-fix replacement.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct CanonicalStyleLint {
    /// Stable lint code identifier.
    pub code: &'static str,
    /// Human-readable lint message.
    pub message: String,
    /// Source range for the lint.
    pub range: SourceRange,
    /// Replacement text for quick fix / formatting.
    pub replacement: String,
}

const LINT_CANONICAL_INLINE_ARGS: &str = "canonical_inline_args";
const LINT_CANONICAL_SET_EXPECTATION: &str = "canonical_set_expectation";
const LINT_CANONICAL_FORCE_ABSENT: &str = "canonical_force_absent";

#[derive(Debug, Clone, PartialEq, Eq)]
struct LineRewrite {
    code: &'static str,
    message: &'static str,
    replacement: String,
}

/// Returns canonical-style lints for compatibility syntax forms.
pub fn lint_canonical_style(source: &str) -> Vec<CanonicalStyleLint> {
    source
        .lines()
        .enumerate()
        .filter_map(|(line_idx, line)| {
            let rewrite = rewrite_line(line)?;
            if rewrite.replacement == line {
                return None;
            }
            let range = SourceRange {
                start: SourcePosition {
                    line: (line_idx + 1) as u32,
                    column: 1,
                },
                end: SourcePosition {
                    line: (line_idx + 1) as u32,
                    column: (line.chars().count() as u32) + 1,
                },
            };
            Some(CanonicalStyleLint {
                code: rewrite.code,
                message: rewrite.message.into(),
                range,
                replacement: rewrite.replacement,
            })
        })
        .collect()
}

/// Rewrites source to canonical inline-argument style where applicable.
pub fn format_canonical_style(source: &str) -> String {
    let mut out = String::with_capacity(source.len());
    for segment in source.split_inclusive('\n') {
        let (line, newline) = if let Some(stripped) = segment.strip_suffix('\n') {
            (stripped, "\n")
        } else {
            (segment, "")
        };
        match rewrite_line(line) {
            Some(rewrite) => out.push_str(&rewrite.replacement),
            None => out.push_str(line),
        }
        out.push_str(newline);
    }
    out
}

fn rewrite_line(line: &str) -> Option<LineRewrite> {
    let (code_part, comment_part) = split_line_comment(line);

    let mut rewritten = code_part.to_string();
    let mut lint_code = "";
    let mut lint_message = "";
    let mut changed = false;

    if let Some(next) = rewrite_set_expectation_legacy(&rewritten) {
        rewritten = next;
        lint_code = LINT_CANONICAL_SET_EXPECTATION;
        lint_message =
            "Use non_bayesian_nudge ... variance=preserve instead of legacy set_expectation";
        changed = true;
    }
    if let Some(next) = rewrite_force_absent_legacy(&rewritten) {
        rewritten = next;
        if lint_code.is_empty() {
            lint_code = LINT_CANONICAL_FORCE_ABSENT;
            lint_message = "Use delete ... confidence=high instead of legacy force_absent";
        }
        changed = true;
    }
    if let Some(next) = rewrite_soft_update_parenthesized(&rewritten) {
        rewritten = next;
        if lint_code.is_empty() {
            lint_code = LINT_CANONICAL_INLINE_ARGS;
            lint_message =
                "Use canonical inline args instead of parenthesized compatibility syntax";
        }
        changed = true;
    }
    if let Some(next) = rewrite_keyword_parenthesized(&rewritten, "delete", &["confidence="]) {
        rewritten = next;
        if lint_code.is_empty() {
            lint_code = LINT_CANONICAL_INLINE_ARGS;
            lint_message =
                "Use canonical inline args instead of parenthesized compatibility syntax";
        }
        changed = true;
    }
    if let Some(next) = rewrite_keyword_parenthesized(&rewritten, "suppress", &["weight="]) {
        rewritten = next;
        if lint_code.is_empty() {
            lint_code = LINT_CANONICAL_INLINE_ARGS;
            lint_message =
                "Use canonical inline args instead of parenthesized compatibility syntax";
        }
        changed = true;
    }

    if !changed {
        return None;
    }

    let mut out = rewritten.trim_end().to_string();
    if let Some(comment) = comment_part {
        if !out.is_empty() {
            out.push(' ');
        }
        out.push_str(comment.trim_start());
    }
    Some(LineRewrite {
        code: lint_code,
        message: lint_message,
        replacement: out,
    })
}

fn split_line_comment(line: &str) -> (&str, Option<&str>) {
    if let Some(idx) = line.find("//") {
        (&line[..idx], Some(&line[idx..]))
    } else {
        (line, None)
    }
}

fn rewrite_set_expectation_legacy(code: &str) -> Option<String> {
    let trimmed = code.trim_start();
    let leading = &code[..code.len().saturating_sub(trimmed.len())];
    let keyword = "set_expectation";
    if !trimmed.starts_with(keyword) {
        return None;
    }
    let post_keyword = trimmed.chars().nth(keyword.len());
    if !matches!(post_keyword, Some(c) if c.is_whitespace()) {
        return None;
    }

    let rhs = trimmed[keyword.len()..].trim_start();
    let (target, expr) = rhs.split_once('=')?;
    let target = target.trim();
    if !is_node_attr(target) {
        return None;
    }

    let (expr, had_semicolon) = split_trailing_semicolon(expr.trim());
    if expr.is_empty() {
        return None;
    }

    let mut rewritten = format!(
        "{}non_bayesian_nudge {} to {} variance=preserve",
        leading, target, expr
    );
    if had_semicolon {
        rewritten.push(';');
    }
    if rewritten == code.trim_end() {
        None
    } else {
        Some(rewritten)
    }
}

fn rewrite_force_absent_legacy(code: &str) -> Option<String> {
    let trimmed = code.trim_start();
    let leading = &code[..code.len().saturating_sub(trimmed.len())];
    let keyword = "force_absent";
    if !trimmed.starts_with(keyword) {
        return None;
    }
    let post_keyword = trimmed.chars().nth(keyword.len());
    if !matches!(post_keyword, Some(c) if c.is_whitespace()) {
        return None;
    }

    let rhs = trimmed[keyword.len()..].trim_start();
    let (edge_var, had_semicolon) = split_trailing_semicolon(rhs);
    let edge_var = edge_var.trim();
    if !is_ident(edge_var) {
        return None;
    }

    let mut rewritten = format!("{}delete {} confidence=high", leading, edge_var);
    if had_semicolon {
        rewritten.push(';');
    }
    if rewritten == code.trim_end() {
        None
    } else {
        Some(rewritten)
    }
}

fn split_trailing_semicolon(input: &str) -> (&str, bool) {
    let trimmed = input.trim_end();
    if let Some(stripped) = trimmed.strip_suffix(';') {
        (stripped.trim_end(), true)
    } else {
        (trimmed, false)
    }
}

fn is_ident(input: &str) -> bool {
    let mut chars = input.chars();
    match chars.next() {
        Some(c) if c.is_ascii_alphabetic() || c == '_' => {}
        _ => return false,
    }
    chars.all(|c| c.is_ascii_alphanumeric() || c == '_')
}

fn is_node_attr(input: &str) -> bool {
    let (lhs, rhs) = match input.split_once('.') {
        Some(parts) => parts,
        None => return false,
    };
    is_ident(lhs) && is_ident(rhs)
}

fn rewrite_soft_update_parenthesized(code: &str) -> Option<String> {
    let op_idx = code.find("~=")?;
    let after_op = &code[op_idx + 2..];
    let open_rel = after_op.find('(')?;
    let open_idx = op_idx + 2 + open_rel;
    let close_rel = code[open_idx + 1..].find(')')?;
    let close_idx = open_idx + 1 + close_rel;
    let args = &code[open_idx + 1..close_idx];
    if !(args.contains("precision=") || args.contains("count=")) {
        return None;
    }
    flatten_parenthesized_args(code, open_idx, close_idx)
}

fn rewrite_keyword_parenthesized(
    code: &str,
    keyword: &str,
    required_keys: &[&str],
) -> Option<String> {
    let trimmed = code.trim_start();
    if !trimmed.starts_with(keyword) {
        return None;
    }
    let post_keyword = trimmed.chars().nth(keyword.chars().count());
    if !matches!(post_keyword, Some(c) if c.is_whitespace()) {
        return None;
    }

    let open_idx = code.find('(')?;
    let close_rel = code[open_idx + 1..].find(')')?;
    let close_idx = open_idx + 1 + close_rel;
    let args = &code[open_idx + 1..close_idx];
    if !required_keys.iter().any(|k| args.contains(k)) {
        return None;
    }
    flatten_parenthesized_args(code, open_idx, close_idx)
}

fn flatten_parenthesized_args(code: &str, open_idx: usize, close_idx: usize) -> Option<String> {
    let prefix = code[..open_idx].trim_end();
    let args_raw = &code[open_idx + 1..close_idx];
    let args = normalize_named_args(args_raw);
    if args.is_empty() {
        return None;
    }
    let suffix = code[close_idx + 1..].trim_start();

    let mut rewritten = String::new();
    rewritten.push_str(prefix);
    rewritten.push(' ');
    rewritten.push_str(&args);
    if !suffix.is_empty() {
        rewritten.push(' ');
        rewritten.push_str(suffix);
    }
    if rewritten == code.trim_end() {
        None
    } else {
        Some(rewritten)
    }
}

fn normalize_named_args(raw: &str) -> String {
    raw.split(',')
        .map(str::trim)
        .filter(|part| !part.is_empty())
        .collect::<Vec<_>>()
        .join(" ")
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn lint_flags_parenthesized_soft_update() {
        let src = r#"soft_update A.score ~= 1.0 (precision=0.2, count=3)"#;
        let lints = lint_canonical_style(src);
        assert_eq!(lints.len(), 1);
        assert_eq!(
            lints[0].replacement,
            "soft_update A.score ~= 1.0 precision=0.2 count=3"
        );
        assert_eq!(lints[0].code, LINT_CANONICAL_INLINE_ARGS);
    }

    #[test]
    fn lint_flags_parenthesized_delete_and_suppress() {
        let src = "delete e (confidence=high)\nsuppress e (weight=10)";
        let lints = lint_canonical_style(src);
        assert_eq!(lints.len(), 2);
        assert_eq!(lints[0].replacement, "delete e confidence=high");
        assert_eq!(lints[1].replacement, "suppress e weight=10");
        assert_eq!(lints[0].code, LINT_CANONICAL_INLINE_ARGS);
        assert_eq!(lints[1].code, LINT_CANONICAL_INLINE_ARGS);
    }

    #[test]
    fn lint_flags_legacy_set_expectation() {
        let src = "set_expectation A.score = E[A.score] + 0.1";
        let lints = lint_canonical_style(src);
        assert_eq!(lints.len(), 1);
        assert_eq!(lints[0].code, LINT_CANONICAL_SET_EXPECTATION);
        assert_eq!(
            lints[0].replacement,
            "non_bayesian_nudge A.score to E[A.score] + 0.1 variance=preserve"
        );
    }

    #[test]
    fn lint_flags_legacy_force_absent() {
        let src = "force_absent e";
        let lints = lint_canonical_style(src);
        assert_eq!(lints.len(), 1);
        assert_eq!(lints[0].code, LINT_CANONICAL_FORCE_ABSENT);
        assert_eq!(lints[0].replacement, "delete e confidence=high");
    }

    #[test]
    fn format_rewrites_and_preserves_comments() {
        let src = "  delete e (confidence=high) // keep\n";
        let out = format_canonical_style(src);
        assert_eq!(out, "  delete e confidence=high // keep\n");
    }

    #[test]
    fn format_rewrites_legacy_forms() {
        let src = "  set_expectation A.x = 10 // keep\n  force_absent e\n";
        let out = format_canonical_style(src);
        assert_eq!(
            out,
            "  non_bayesian_nudge A.x to 10 variance=preserve // keep\n  delete e confidence=high\n"
        );
    }

    #[test]
    fn format_leaves_canonical_source_unchanged() {
        let src = "soft_update A.score ~= 1.0 precision=0.2 count=3\n";
        let out = format_canonical_style(src);
        assert_eq!(out, src);
    }
}
