use grafial_frontend::parser::{BayGraphParser, Rule};
use grafial_frontend::{parse_program, validate_program, FrontendError};
use pest::Parser;
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
    async fn initialize(&self, _: lsp::InitializeParams) -> tower_lsp::jsonrpc::Result<lsp::InitializeResult> {
        let server_capabilities = lsp::ServerCapabilities {
            text_document_sync: Some(lsp::TextDocumentSyncCapability::Kind(
                lsp::TextDocumentSyncKind::FULL,
            )),
            hover_provider: Some(lsp::HoverProviderCapability::Simple(true)),
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
        let _ = self.client.log_message(lsp::MessageType::INFO, "grafial LSP initialized");
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
        if let Some(text) = params.text { // Only if client sends full text on save
            let uri = params.text_document.uri;
            {
                let mut docs = self.docs.write().await;
                docs.insert(uri.clone(), text.clone());
            }
            self.publish_diagnostics(uri, text).await;
        }
    }

    async fn hover(&self, params: lsp::HoverParams) -> tower_lsp::jsonrpc::Result<Option<lsp::Hover>> {
        let lsp::HoverParams { text_document_position_params, .. } = params;
        let uri = text_document_position_params.text_document.uri;
        let position = text_document_position_params.position;

        let text_opt = { self.docs.read().await.get(&uri).cloned() };
        let Some(text) = text_opt else { return Ok(None) };

        if let Some((token, range)) = word_at_position(&text, position) {
            if let Some(contents) = builtin_hover(&token) {
                return Ok(Some(lsp::Hover { contents, range: Some(range) }));
            }

            if let Some((contents, r)) = ast_symbol_hover(&text, &token, range) {
                return Ok(Some(lsp::Hover { contents, range: Some(r) }));
            }
        }

        Ok(None)
    }
}

impl Backend {
    fn new(client: Client) -> Self {
        Self { client, docs: RwLock::new(HashMap::new()) }
    }

    async fn publish_diagnostics(&self, uri: lsp::Url, text: String) {
        let mut diagnostics: Vec<lsp::Diagnostic> = Vec::new();

        // Phase 1: syntax diagnostics via Pest parse
        match BayGraphParser::parse(Rule::program, &text) {
            Ok(_) => {
                // Parsed ok; now build AST and validate
                match parse_program(&text) {
                    Ok(ast) => {
                        if let Err(err) = validate_program(&ast) {
                            diagnostics.push(self.validation_error_to_diag(err));
                        }
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

        let _ = self.client.publish_diagnostics(uri, diagnostics, None).await;
    }

    fn validation_error_to_diag(&self, err: FrontendError) -> lsp::Diagnostic {
        // Currently validation errors have no position info; attach to document start.
        generic_diag_at_start(err.to_string())
    }
}

fn generic_diag_at_start(message: String) -> lsp::Diagnostic {
    lsp::Diagnostic {
        range: lsp::Range {
            start: lsp::Position { line: 0, character: 0 },
            end: lsp::Position { line: 0, character: 0 },
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
                let p = lsp::Position { line: 0, character: 0 };
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
                .unwrap_or_else(|| lsp::Position { line: 0, character: 0 });
            
            let end_pos = Position::new(input, *end_offset)
                .map(|p| {
                    let (line, col) = p.line_col();
                    lsp::Position {
                        line: (line - 1) as u32,
                        character: (col - 1) as u32,
                    }
                })
                .unwrap_or_else(|| lsp::Position { line: 0, character: 0 });
            
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

#[tokio::main]
async fn main() {
    let (stdin, stdout) = (tokio::io::stdin(), tokio::io::stdout());
    let (service, socket) = LspService::new(|client| Backend::new(client));
    Server::new(stdin, stdout, socket).serve(service).await;
}

fn word_at_position(text: &str, pos: lsp::Position) -> Option<(String, lsp::Range)> {
    let line_idx = pos.line as usize;
    let char_idx = pos.character as usize;
    let line = text.lines().nth(line_idx)?;
    if line.is_empty() { return None; }

    // Operate on char indices (assumes ASCII for identifiers)
    let chars: Vec<char> = line.chars().collect();
    let mut start = char_idx.min(chars.len());
    let mut end = start;
    let is_word = |c: char| c.is_ascii_alphanumeric() || c == '_' || c == '[' || c == ']';

    while start > 0 && is_word(chars[start - 1]) { start -= 1; }
    while end < chars.len() && is_word(chars[end]) { end += 1; }

    let token: String = chars[start..end].iter().collect();
    if token.is_empty() { return None; }

    Some((token, lsp::Range {
        start: lsp::Position { line: pos.line, character: start as u32 },
        end: lsp::Position { line: pos.line, character: end as u32 },
    }))
}

fn builtin_hover(token: &str) -> Option<lsp::HoverContents> {
    let t = token.trim_matches(|c: char| c == '[' || c == ']');
    let docs = match t {
        "prob" => Some(("prob(target) -> Real", "Probability of a boolean/edge event.\n\nExamples:\n- prob(E) where E is an edge variable\n- prob(A.attr > 0) for boolean expressions")),
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

fn ast_symbol_hover(text: &str, token: &str, range: lsp::Range) -> Option<(lsp::HoverContents, lsp::Range)> {
    let ast = parse_program(text).ok()?;

    // Schema names
    if let Some(schema) = ast.schemas.iter().find(|s| s.name == token) {
        let nodes: Vec<_> = schema.nodes.iter().map(|n| n.name.as_str()).collect();
        let edges: Vec<_> = schema.edges.iter().map(|e| e.name.as_str()).collect();
        let value = format!("**schema {}**\n\n- nodes: {}\n- edges: {}", schema.name, nodes.join(", "), edges.join(", "));
        return Some((md(value), range));
    }

    // Belief model names
    if let Some(model) = ast.belief_models.iter().find(|m| m.name == token) {
        let value = format!("**belief_model {} on {}**\n\n- node beliefs: {}\n- edge beliefs: {}", 
            model.name, model.on_schema, model.nodes.len(), model.edges.len());
        return Some((md(value), range));
    }

    // Rule names
    if let Some(rule) = ast.rules.iter().find(|r| r.name == token) {
        let patterns: Vec<_> = rule
            .patterns
            .iter()
            .map(|p| format!("({}:{})-[{}:{}]->({}:{})", p.src.var, p.src.label, p.edge.var, p.edge.ty, p.dst.var, p.dst.label))
            .collect();
        let value = if patterns.is_empty() {
            format!("**rule {} on {}**", rule.name, rule.on_model)
        } else {
            format!("**rule {} on {}**\n\npatterns:\n- {}", rule.name, rule.on_model, patterns.join("\n- "))
        };
        return Some((md(value), range));
    }

    // Flow names
    if let Some(flow) = ast.flows.iter().find(|f| f.name == token) {
        let value = format!("**flow {} on {}**\n\n- graphs: {}\n- metrics: {}\n- exports: {}", 
            flow.name, flow.on_model, flow.graphs.len(), flow.metrics.len(), flow.exports.len());
        return Some((md(value), range));
    }

    None
}

fn md(value: String) -> lsp::HoverContents {
    lsp::HoverContents::Markup(lsp::MarkupContent { kind: lsp::MarkupKind::Markdown, value })
}
