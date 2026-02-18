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

/// Returns canonical-style lints for compatibility syntax forms.
pub fn lint_canonical_style(source: &str) -> Vec<CanonicalStyleLint> {
    source
        .lines()
        .enumerate()
        .filter_map(|(line_idx, line)| {
            let rewrite = canonicalize_line(line)?;
            if rewrite == line {
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
                code: LINT_CANONICAL_INLINE_ARGS,
                message: "Use canonical inline args instead of parenthesized compatibility syntax"
                    .into(),
                range,
                replacement: rewrite,
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
        match canonicalize_line(line) {
            Some(rewrite) => out.push_str(&rewrite),
            None => out.push_str(line),
        }
        out.push_str(newline);
    }
    out
}

fn canonicalize_line(line: &str) -> Option<String> {
    let (code_part, comment_part) = split_line_comment(line);

    let mut rewritten = code_part.to_string();
    let mut changed = false;

    if let Some(next) = rewrite_soft_update_parenthesized(&rewritten) {
        rewritten = next;
        changed = true;
    }
    if let Some(next) = rewrite_keyword_parenthesized(&rewritten, "delete", &["confidence="]) {
        rewritten = next;
        changed = true;
    }
    if let Some(next) = rewrite_keyword_parenthesized(&rewritten, "suppress", &["weight="]) {
        rewritten = next;
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
    Some(out)
}

fn split_line_comment(line: &str) -> (&str, Option<&str>) {
    if let Some(idx) = line.find("//") {
        (&line[..idx], Some(&line[idx..]))
    } else {
        (line, None)
    }
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
    }

    #[test]
    fn lint_flags_parenthesized_delete_and_suppress() {
        let src = "delete e (confidence=high)\nsuppress e (weight=10)";
        let lints = lint_canonical_style(src);
        assert_eq!(lints.len(), 2);
        assert_eq!(lints[0].replacement, "delete e confidence=high");
        assert_eq!(lints[1].replacement, "suppress e weight=10");
    }

    #[test]
    fn format_rewrites_and_preserves_comments() {
        let src = "  delete e (confidence=high) // keep\n";
        let out = format_canonical_style(src);
        assert_eq!(out, "  delete e confidence=high // keep\n");
    }

    #[test]
    fn format_leaves_canonical_source_unchanged() {
        let src = "soft_update A.score ~= 1.0 precision=0.2 count=3\n";
        let out = format_canonical_style(src);
        assert_eq!(out, src);
    }
}
