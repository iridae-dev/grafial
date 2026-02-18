use grafial_frontend::parser::{BayGraphParser, Rule};
use grafial_frontend::{
    collect_lint_suppressions, lint_canonical_style, lint_is_suppressed,
    lint_statistical_guardrails, parse_program, validate_program_with_source, CanonicalStyleLint,
    FrontendError, LintSeverity, LintSuppression, SourceRange, StatisticalLint,
};
use pest::Parser;
use serde_json::json;
use std::collections::HashMap;
use tokio::sync::RwLock;
use tower_lsp::lsp_types as lsp;
use tower_lsp::{Client, LanguageServer, LspService, Server};

struct Backend {
    client: Client,
    docs: RwLock<HashMap<lsp::Url, String>>, // in-memory document store for hover
}

#[tower_lsp::async_trait]
impl LanguageServer for Backend {
    async fn initialize(
        &self,
        _: lsp::InitializeParams,
    ) -> tower_lsp::jsonrpc::Result<lsp::InitializeResult> {
        let server_capabilities = lsp::ServerCapabilities {
            text_document_sync: Some(lsp::TextDocumentSyncCapability::Kind(
                lsp::TextDocumentSyncKind::FULL,
            )),
            hover_provider: Some(lsp::HoverProviderCapability::Simple(true)),
            code_action_provider: Some(lsp::CodeActionProviderCapability::Simple(true)),
            ..Default::default()
        };

        Ok(lsp::InitializeResult {
            capabilities: server_capabilities,
            server_info: Some(lsp::ServerInfo {
                name: "grafial-lsp".into(),
                version: Some(env!("CARGO_PKG_VERSION").into()),
            }),
        })
    }

    async fn initialized(&self, _: lsp::InitializedParams) {
        self.client
            .log_message(lsp::MessageType::INFO, "grafial LSP initialized")
            .await;
    }

    async fn shutdown(&self) -> tower_lsp::jsonrpc::Result<()> {
        Ok(())
    }

    async fn did_open(&self, params: lsp::DidOpenTextDocumentParams) {
        let uri = params.text_document.uri;
        let text = params.text_document.text;
        {
            let mut docs = self.docs.write().await;
            docs.insert(uri.clone(), text.clone());
        }
        self.publish_diagnostics(uri, text).await;
    }

    async fn did_change(&self, params: lsp::DidChangeTextDocumentParams) {
        // Collect full text from the last change (assuming incremental sync)
        if let Some(change) = params.content_changes.last() {
            let text = change.text.clone();
            let uri = params.text_document.uri;
            {
                let mut docs = self.docs.write().await;
                docs.insert(uri.clone(), text.clone());
            }
            self.publish_diagnostics(uri, text).await;
        }
    }

    async fn did_save(&self, params: lsp::DidSaveTextDocumentParams) {
        if let Some(text) = params.text {
            // Only if client sends full text on save
            let uri = params.text_document.uri;
            {
                let mut docs = self.docs.write().await;
                docs.insert(uri.clone(), text.clone());
            }
            self.publish_diagnostics(uri, text).await;
        }
    }

    async fn hover(
        &self,
        params: lsp::HoverParams,
    ) -> tower_lsp::jsonrpc::Result<Option<lsp::Hover>> {
        let lsp::HoverParams {
            text_document_position_params,
            ..
        } = params;
        let uri = text_document_position_params.text_document.uri;
        let position = text_document_position_params.position;

        let text_opt = { self.docs.read().await.get(&uri).cloned() };
        let Some(text) = text_opt else {
            return Ok(None);
        };

        if let Some((token, range)) = word_at_position(&text, position) {
            if let Some(contents) = builtin_hover(&token) {
                return Ok(Some(lsp::Hover {
                    contents,
                    range: Some(range),
                }));
            }

            if let Some((contents, r)) = ast_symbol_hover(&text, &token, range) {
                return Ok(Some(lsp::Hover {
                    contents,
                    range: Some(r),
                }));
            }
        }

        Ok(None)
    }

    async fn code_action(
        &self,
        params: lsp::CodeActionParams,
    ) -> tower_lsp::jsonrpc::Result<Option<lsp::CodeActionResponse>> {
        let uri = params.text_document.uri;
        let text = { self.docs.read().await.get(&uri).cloned() };
        let Some(text) = text else {
            return Ok(None);
        };

        let mut actions = Vec::new();
        for diag in &params.context.diagnostics {
            if let Some(action) = quick_fix_from_diagnostic_data(&uri, diag) {
                actions.push(action);
            }
            if let Some(action) = uncertainty_wrapper_quick_fix(&uri, &text, diag) {
                actions.push(action);
            }
        }

        if actions.is_empty() {
            Ok(None)
        } else {
            Ok(Some(actions))
        }
    }
}

impl Backend {
    fn new(client: Client) -> Self {
        Self {
            client,
            docs: RwLock::new(HashMap::new()),
        }
    }

    async fn publish_diagnostics(&self, uri: lsp::Url, text: String) {
        let mut diagnostics: Vec<lsp::Diagnostic> = Vec::new();
        let mut syntax_ok = false;
        let mut parsed_ast = None;
        let suppressions = collect_lint_suppressions(&text);

        // Phase 1: syntax diagnostics via Pest parse
        match BayGraphParser::parse(Rule::program, &text) {
            Ok(_) => {
                syntax_ok = true;
                // Parsed ok; now build AST and validate
                match parse_program(&text) {
                    Ok(ast) => {
                        if let Err(err) = validate_program_with_source(&ast, &text) {
                            diagnostics.push(self.validation_error_to_diag(err));
                        }
                        parsed_ast = Some(ast);
                    }
                    Err(err) => {
                        // Unexpected: surface as a generic diagnostic at start
                        diagnostics.push(generic_diag_at_start(err.to_string()));
                    }
                }
            }
            Err(e) => {
                diagnostics.push(pest_error_to_diag(&e, &text));
            }
        }

        diagnostics.extend(canonical_style_diagnostics(&text, &suppressions));
        if syntax_ok {
            if let Some(ast) = parsed_ast.as_ref() {
                diagnostics.extend(statistical_guardrail_diagnostics(ast, &text));
            }
        }

        let _ = self
            .client
            .publish_diagnostics(uri, diagnostics, None)
            .await;
    }

    fn validation_error_to_diag(&self, err: FrontendError) -> lsp::Diagnostic {
        if let Some(diag) = err.validation_diagnostic() {
            let range = diag.range.map(to_lsp_range).unwrap_or_else(|| lsp::Range {
                start: lsp::Position {
                    line: 0,
                    character: 0,
                },
                end: lsp::Position {
                    line: 0,
                    character: 0,
                },
            });
            return lsp::Diagnostic {
                range,
                severity: Some(lsp::DiagnosticSeverity::ERROR),
                code: validation_diag_code(&diag.message)
                    .map(|code| lsp::NumberOrString::String(code.into())),
                code_description: None,
                source: Some("grafial".into()),
                message: diag.to_string(),
                related_information: None,
                tags: None,
                data: None,
            };
        }
        generic_diag_at_start(err.to_string())
    }
}

fn validation_diag_code(message: &str) -> Option<&'static str> {
    if message.contains("Bare field access in rule expression is not allowed") {
        return Some("uncertainty_wrapper_rule");
    }
    if message.contains("bare field access not supported in metric") {
        return Some("uncertainty_wrapper_metric");
    }
    None
}

fn to_lsp_range(range: SourceRange) -> lsp::Range {
    let start_line = range.start.line.saturating_sub(1);
    let start_col = range.start.column.saturating_sub(1);
    let end_line = range.end.line.saturating_sub(1);
    let end_col = range.end.column.saturating_sub(1);
    lsp::Range {
        start: lsp::Position {
            line: start_line,
            character: start_col,
        },
        end: lsp::Position {
            line: end_line,
            character: end_col,
        },
    }
}

fn generic_diag_at_start(message: String) -> lsp::Diagnostic {
    lsp::Diagnostic {
        range: lsp::Range {
            start: lsp::Position {
                line: 0,
                character: 0,
            },
            end: lsp::Position {
                line: 0,
                character: 0,
            },
        },
        severity: Some(lsp::DiagnosticSeverity::ERROR),
        code: None,
        code_description: None,
        source: Some("grafial".into()),
        message,
        related_information: None,
        tags: None,
        data: None,
    }
}

fn pest_error_to_diag(e: &pest::error::Error<Rule>, input: &str) -> lsp::Diagnostic {
    use pest::error::InputLocation;
    use pest::Position;

    // Get the span from the error location and convert to line/column
    let (start, end) = match &e.location {
        InputLocation::Pos(byte_offset) => {
            // Convert byte offset to Position and then to line/column
            if let Some(pos) = Position::new(input, *byte_offset) {
                let (line, col) = pos.line_col();
                let p = lsp::Position {
                    line: (line - 1) as u32,
                    character: (col - 1) as u32,
                };
                (p, p)
            } else {
                // Fallback to start of document if position is invalid
                let p = lsp::Position {
                    line: 0,
                    character: 0,
                };
                (p, p)
            }
        }
        InputLocation::Span((start_offset, end_offset)) => {
            // Convert byte offsets to Positions and then to line/column
            let start_pos = Position::new(input, *start_offset)
                .map(|p| {
                    let (line, col) = p.line_col();
                    lsp::Position {
                        line: (line - 1) as u32,
                        character: (col - 1) as u32,
                    }
                })
                .unwrap_or_else(|| lsp::Position {
                    line: 0,
                    character: 0,
                });

            let end_pos = Position::new(input, *end_offset)
                .map(|p| {
                    let (line, col) = p.line_col();
                    lsp::Position {
                        line: (line - 1) as u32,
                        character: (col - 1) as u32,
                    }
                })
                .unwrap_or_else(|| lsp::Position {
                    line: 0,
                    character: 0,
                });

            (start_pos, end_pos)
        }
    };

    lsp::Diagnostic {
        range: lsp::Range { start, end },
        severity: Some(lsp::DiagnosticSeverity::ERROR),
        code: None,
        code_description: None,
        source: Some("grafial".into()),
        message: e.to_string(),
        related_information: None,
        tags: None,
        data: None,
    }
}

fn canonical_style_diagnostics(
    text: &str,
    suppressions: &[LintSuppression],
) -> Vec<lsp::Diagnostic> {
    lint_canonical_style(text)
        .into_iter()
        .filter(|lint| !lint_is_suppressed(suppressions, lint.code, lint.range))
        .map(style_lint_to_diag)
        .collect()
}

fn style_lint_to_diag(lint: CanonicalStyleLint) -> lsp::Diagnostic {
    lsp::Diagnostic {
        range: to_lsp_range(lint.range),
        severity: Some(lsp::DiagnosticSeverity::WARNING),
        code: Some(lsp::NumberOrString::String(lint.code.to_string())),
        code_description: None,
        source: Some("grafial".into()),
        message: lint.message,
        related_information: None,
        tags: None,
        data: Some(json!({
            "quickfix": {
                "title": "Rewrite to canonical syntax",
                "replacement": lint.replacement,
            }
        })),
    }
}

fn statistical_guardrail_diagnostics(
    ast: &grafial_frontend::ProgramAst,
    text: &str,
) -> Vec<lsp::Diagnostic> {
    lint_statistical_guardrails(ast, text)
        .into_iter()
        .map(stat_lint_to_diag)
        .collect()
}

fn stat_lint_to_diag(lint: StatisticalLint) -> lsp::Diagnostic {
    let severity = match lint.severity {
        LintSeverity::Warning => lsp::DiagnosticSeverity::WARNING,
        LintSeverity::Information => lsp::DiagnosticSeverity::INFORMATION,
    };
    lsp::Diagnostic {
        range: to_lsp_range(lint.range),
        severity: Some(severity),
        code: Some(lsp::NumberOrString::String(lint.code.to_string())),
        code_description: None,
        source: Some("grafial".into()),
        message: lint.message,
        related_information: None,
        tags: None,
        data: None,
    }
}

fn quick_fix_from_diagnostic_data(
    uri: &lsp::Url,
    diag: &lsp::Diagnostic,
) -> Option<lsp::CodeActionOrCommand> {
    let quickfix = diag.data.as_ref()?.get("quickfix")?;
    let title = quickfix.get("title")?.as_str()?.to_string();
    let replacement = quickfix.get("replacement")?.as_str()?.to_string();
    Some(make_single_edit_action(
        uri,
        title,
        diag.range,
        replacement,
        diag.clone(),
        true,
    ))
}

fn uncertainty_wrapper_quick_fix(
    uri: &lsp::Url,
    text: &str,
    diag: &lsp::Diagnostic,
) -> Option<lsp::CodeActionOrCommand> {
    if !looks_like_uncertainty_wrapper_diag(&diag.message) {
        return None;
    }
    let (range, replacement) = first_field_wrapper_edit(text, diag.range)?;
    Some(make_single_edit_action(
        uri,
        "Wrap with E[...]".into(),
        range,
        replacement,
        diag.clone(),
        false,
    ))
}

fn looks_like_uncertainty_wrapper_diag(message: &str) -> bool {
    message.contains("Bare field access in rule expression is not allowed")
        || message.contains("bare field access not supported in metric")
}

fn make_single_edit_action(
    uri: &lsp::Url,
    title: String,
    range: lsp::Range,
    replacement: String,
    diag: lsp::Diagnostic,
    preferred: bool,
) -> lsp::CodeActionOrCommand {
    let mut changes = HashMap::new();
    changes.insert(
        uri.clone(),
        vec![lsp::TextEdit {
            range,
            new_text: replacement,
        }],
    );
    let edit = lsp::WorkspaceEdit {
        changes: Some(changes),
        document_changes: None,
        change_annotations: None,
    };
    lsp::CodeActionOrCommand::CodeAction(lsp::CodeAction {
        title,
        kind: Some(lsp::CodeActionKind::QUICKFIX),
        diagnostics: Some(vec![diag]),
        edit: Some(edit),
        command: None,
        is_preferred: Some(preferred),
        disabled: None,
        data: None,
    })
}

fn first_field_wrapper_edit(text: &str, range: lsp::Range) -> Option<(lsp::Range, String)> {
    let start = offset_at_position(text, range.start)?;
    let end = offset_at_position(text, range.end)?;
    if start >= end {
        return None;
    }
    let snippet = &text[start..end];
    let (field_start, field_end) = find_first_field_token(snippet)?;
    let global_start = start + field_start;
    let global_end = start + field_end;
    let edit_range = lsp::Range {
        start: offset_to_position(text, global_start)?,
        end: offset_to_position(text, global_end)?,
    };
    let field = &snippet[field_start..field_end];
    Some((edit_range, format!("E[{}]", field)))
}

fn find_first_field_token(snippet: &str) -> Option<(usize, usize)> {
    let bytes = snippet.as_bytes();
    let mut i = 0usize;
    while i < bytes.len() {
        if is_ident_start(bytes[i]) {
            let start = i;
            i += 1;
            while i < bytes.len() && is_ident_continue(bytes[i]) {
                i += 1;
            }
            if i < bytes.len() && bytes[i] == b'.' {
                i += 1;
                if i < bytes.len() && is_ident_start(bytes[i]) {
                    i += 1;
                    while i < bytes.len() && is_ident_continue(bytes[i]) {
                        i += 1;
                    }
                    if start >= 2 && &snippet[start - 2..start] == "E[" {
                        continue;
                    }
                    return Some((start, i));
                }
            }
            continue;
        }
        i += 1;
    }
    None
}

fn is_ident_start(b: u8) -> bool {
    (b as char).is_ascii_alphabetic() || b == b'_'
}

fn is_ident_continue(b: u8) -> bool {
    (b as char).is_ascii_alphanumeric() || b == b'_'
}

fn offset_at_position(text: &str, pos: lsp::Position) -> Option<usize> {
    let target_line = pos.line as usize;
    let target_char = pos.character as usize;
    let mut line = 0usize;
    let mut ch = 0usize;

    for (idx, c) in text.char_indices() {
        if line == target_line && ch == target_char {
            return Some(idx);
        }
        if c == '\n' {
            line += 1;
            ch = 0;
        } else {
            ch += 1;
        }
    }
    if line == target_line && ch == target_char {
        Some(text.len())
    } else {
        None
    }
}

fn offset_to_position(text: &str, offset: usize) -> Option<lsp::Position> {
    if offset > text.len() || !text.is_char_boundary(offset) {
        return None;
    }
    let mut line = 0usize;
    let mut ch = 0usize;
    for (idx, c) in text.char_indices() {
        if idx == offset {
            return Some(lsp::Position {
                line: line as u32,
                character: ch as u32,
            });
        }
        if c == '\n' {
            line += 1;
            ch = 0;
        } else {
            ch += 1;
        }
    }
    if offset == text.len() {
        Some(lsp::Position {
            line: line as u32,
            character: ch as u32,
        })
    } else {
        None
    }
}

#[tokio::main]
async fn main() {
    let (stdin, stdout) = (tokio::io::stdin(), tokio::io::stdout());
    let (service, socket) = LspService::new(Backend::new);
    Server::new(stdin, stdout, socket).serve(service).await;
}

fn word_at_position(text: &str, pos: lsp::Position) -> Option<(String, lsp::Range)> {
    let line_idx = pos.line as usize;
    let char_idx = pos.character as usize;
    let line = text.lines().nth(line_idx)?;
    if line.is_empty() {
        return None;
    }

    // Operate on char indices (assumes ASCII for identifiers)
    let chars: Vec<char> = line.chars().collect();
    let mut start = char_idx.min(chars.len());
    let mut end = start;
    let is_word = |c: char| c.is_ascii_alphanumeric() || c == '_' || c == '[' || c == ']';

    while start > 0 && is_word(chars[start - 1]) {
        start -= 1;
    }
    while end < chars.len() && is_word(chars[end]) {
        end += 1;
    }

    let token: String = chars[start..end].iter().collect();
    if token.is_empty() {
        return None;
    }

    Some((
        token,
        lsp::Range {
            start: lsp::Position {
                line: pos.line,
                character: start as u32,
            },
            end: lsp::Position {
                line: pos.line,
                character: end as u32,
            },
        },
    ))
}

fn builtin_hover(token: &str) -> Option<lsp::HoverContents> {
    let t = token.trim_matches(|c: char| c == '[' || c == ']');
    let docs = match t {
        "prob" => Some(("prob(target) -> Real", "Probability of a boolean/edge event.\n\nExamples:\n- prob(E) where E is an edge variable\n- prob(A.attr > 0) for boolean expressions")),
        "prob_correlated" => Some(("prob_correlated(A.attr > B.attr, rho=...) -> Real", "Gaussian comparison probability with explicit correlation.\n\nUse `prob(...)` for independence semantics and `prob_correlated(...)` when covariance should be modeled.")),
        "E" => Some(("E[expr] -> Real", "Expectation (mean) of an expression under the current posterior.\n\nExamples:\n- E[A.score]\n- E[entropy(C)]")),
        "winner" => Some(("winner(C) -> String", "Maximum a posteriori category label for categorical posterior C.")),
        "entropy" => Some(("entropy(X) -> Real", "Shannon entropy of a discrete distribution X (in nats).")),
        "degree" => Some(("degree(node) -> Int", "Node degree. Variants: in_degree(node), out_degree(node).")),
        // Posterior types
        "GaussianPosterior" => Some(("GaussianPosterior(params)", "Posterior for continuous attributes. Params include prior_mean, prior_var, noise.")),
        "BernoulliPosterior" => Some(("BernoulliPosterior(params)", "Beta-Bernoulli posterior for independent edges. Params include prior and pseudo_count.")),
        "CategoricalPosterior" => Some(("CategoricalPosterior(group_by, prior, categories)", "Dirichlet-Categorical posterior for competing edges. group_by in {source,destination}. prior: uniform{pseudo_count} or explicit[α...].")),
        // Keywords
        "schema" => Some(("schema Name { ... }", "Defines node and edge types for a graph schema.")),
        "belief_model" => Some(("belief_model Name on Schema { ... }", "Associates posterior models with a schema (nodes, edges).")),
        "rule" => Some(("rule Name on Model { ... }", "Pattern-action rule executed on a belief model.")),
        "flow" => Some(("flow Name on Model { ... }", "Pipeline of transformations and metrics.")),
        _ => None,
    }?;

    let (sig, body) = docs;
    Some(lsp::HoverContents::Markup(lsp::MarkupContent {
        kind: lsp::MarkupKind::Markdown,
        value: format!("**{}**\n\n{}", sig, body),
    }))
}

fn ast_symbol_hover(
    text: &str,
    token: &str,
    range: lsp::Range,
) -> Option<(lsp::HoverContents, lsp::Range)> {
    let ast = parse_program(text).ok()?;

    // Schema names
    if let Some(schema) = ast.schemas.iter().find(|s| s.name == token) {
        let nodes: Vec<_> = schema.nodes.iter().map(|n| n.name.as_str()).collect();
        let edges: Vec<_> = schema.edges.iter().map(|e| e.name.as_str()).collect();
        let value = format!(
            "**schema {}**\n\n- nodes: {}\n- edges: {}",
            schema.name,
            nodes.join(", "),
            edges.join(", ")
        );
        return Some((md(value), range));
    }

    // Belief model names
    if let Some(model) = ast.belief_models.iter().find(|m| m.name == token) {
        let value = format!(
            "**belief_model {} on {}**\n\n- node beliefs: {}\n- edge beliefs: {}",
            model.name,
            model.on_schema,
            model.nodes.len(),
            model.edges.len()
        );
        return Some((md(value), range));
    }

    // Rule names
    if let Some(rule) = ast.rules.iter().find(|r| r.name == token) {
        let patterns: Vec<_> = rule
            .patterns
            .iter()
            .map(|p| {
                format!(
                    "({}:{})-[{}:{}]->({}:{})",
                    p.src.var, p.src.label, p.edge.var, p.edge.ty, p.dst.var, p.dst.label
                )
            })
            .collect();
        let value = if patterns.is_empty() {
            format!("**rule {} on {}**", rule.name, rule.on_model)
        } else {
            format!(
                "**rule {} on {}**\n\npatterns:\n- {}",
                rule.name,
                rule.on_model,
                patterns.join("\n- ")
            )
        };
        return Some((md(value), range));
    }

    // Flow names
    if let Some(flow) = ast.flows.iter().find(|f| f.name == token) {
        let value = format!(
            "**flow {} on {}**\n\n- graphs: {}\n- metrics: {}\n- exports: {}",
            flow.name,
            flow.on_model,
            flow.graphs.len(),
            flow.metrics.len(),
            flow.exports.len()
        );
        return Some((md(value), range));
    }

    None
}

fn md(value: String) -> lsp::HoverContents {
    lsp::HoverContents::Markup(lsp::MarkupContent {
        kind: lsp::MarkupKind::Markdown,
        value,
    })
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn style_lint_diagnostic_carries_quick_fix_payload() {
        let src = "delete e (confidence=high)\n";
        let diagnostics = canonical_style_diagnostics(src, &[]);
        assert_eq!(diagnostics.len(), 1);
        assert_eq!(
            diagnostics[0].code,
            Some(lsp::NumberOrString::String("canonical_inline_args".into()))
        );
        let replacement = diagnostics[0]
            .data
            .as_ref()
            .and_then(|data| data.get("quickfix"))
            .and_then(|fix| fix.get("replacement"))
            .and_then(|value| value.as_str());
        assert_eq!(replacement, Some("delete e confidence=high"));
    }

    #[test]
    fn legacy_style_lint_diagnostic_carries_quick_fix_payload() {
        let src = "set_expectation A.score = 0.5\n";
        let diagnostics = canonical_style_diagnostics(src, &[]);
        assert_eq!(diagnostics.len(), 1);
        assert_eq!(
            diagnostics[0].code,
            Some(lsp::NumberOrString::String(
                "canonical_set_expectation".into()
            ))
        );
        let replacement = diagnostics[0]
            .data
            .as_ref()
            .and_then(|data| data.get("quickfix"))
            .and_then(|fix| fix.get("replacement"))
            .and_then(|value| value.as_str());
        assert_eq!(
            replacement,
            Some("non_bayesian_nudge A.score to 0.5 variance=preserve")
        );
    }

    #[test]
    fn finds_field_expression_for_uncertainty_wrapper_fix() {
        let src = "rule R on M {\n  where A.score > 0.5\n}\n";
        let range = lsp::Range {
            start: lsp::Position {
                line: 1,
                character: 8,
            },
            end: lsp::Position {
                line: 1,
                character: 21,
            },
        };
        let (edit_range, replacement) = first_field_wrapper_edit(src, range).expect("quick fix");
        assert_eq!(replacement, "E[A.score]");
        let start = offset_at_position(src, edit_range.start).expect("start offset");
        let end = offset_at_position(src, edit_range.end).expect("end offset");
        assert_eq!(&src[start..end], "A.score");
    }

    #[test]
    fn style_lint_is_suppressed_by_pragma() {
        let src = "// grafial-lint: ignore(canonical_inline_args)\ndelete e (confidence=high)\n";
        let suppressions = collect_lint_suppressions(src);
        let diagnostics = canonical_style_diagnostics(src, &suppressions);
        assert!(diagnostics.is_empty());
    }

    #[test]
    fn statistical_lints_surface_stable_codes() {
        let src = r#"
schema S { node N { x: Real } edge E { } }
belief_model M on S {
  node N { x ~ Gaussian(mean=0.0, precision=500.0) }
  edge E { exist ~ Bernoulli(prior=0.5, weight=2.0) }
}
evidence Ev on M { N { "a" { x: 0.0 } } E(N -> N) { "a" -> "a" } }
rule R on M {
  pattern (A:N)-[ab:E]->(B:N)
  where prob(ab) > 0.1 and prob(ab) > 0.2 and prob(ab) > 0.3
  action { delete ab confidence=low }
}
flow F on M { graph g = from_evidence Ev }
"#;
        let ast = parse_program(src).expect("parse");
        let diagnostics = statistical_guardrail_diagnostics(&ast, src);
        let codes: Vec<String> = diagnostics
            .iter()
            .filter_map(|diag| match &diag.code {
                Some(lsp::NumberOrString::String(code)) => Some(code.clone()),
                _ => None,
            })
            .collect();
        assert!(codes.contains(&"stat_prior_dominance".into()));
        assert!(codes.contains(&"stat_multiple_testing".into()));
        assert!(codes.contains(&"stat_delete_explanation".into()));
    }
}
