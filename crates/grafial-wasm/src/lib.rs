//! WebAssembly bindings for Grafial.
//!
//! Exposes a JSON-string API designed for browser tooling — in particular a
//! visual, graph-based composer for Grafial programs:
//!
//! - [`check`] — parse/validate a program and collect style + statistical lints
//! - [`format_canonical`] — rewrite a program to canonical style
//! - [`program_structure`] — a JSON description of every declaration and the
//!   inter-flow dependency surface, suitable for rendering a program as a graph
//! - [`list_flows`] — flow names in program order
//! - [`run_flow`] — execute a flow (running prerequisite flows automatically)
//!   and return metrics, audits, and full belief-graph structure as JSON
//!
//! All functions take Grafial source as input; results are JSON strings so the
//! JavaScript side needs nothing beyond `JSON.parse`. The heavy lifting lives
//! in target-independent `api::*` functions (unit-tested natively); the
//! `#[wasm_bindgen]` wrappers only adapt errors.

use wasm_bindgen::prelude::*;

pub mod api {
    use grafial_core::engine::flow_exec::FlowResult;
    use grafial_core::BeliefGraph;
    use grafial_frontend::ast::{
        CategoricalPrior, FlowDef, GraphExpr, ModelSelectionCriterion, PosteriorType, ProgramAst,
        Transform,
    };
    use grafial_frontend::{
        collect_lint_suppressions, lint_canonical_style, lint_is_suppressed,
        lint_statistical_guardrails, LintSeverity, SourceRange,
    };
    use serde_json::{json, Value};

    fn parse(source: &str) -> Result<ProgramAst, String> {
        grafial_core::parse_and_validate(source).map_err(|e| e.to_string())
    }

    fn range_json(range: SourceRange) -> Value {
        json!({
            "start": { "line": range.start.line, "column": range.start.column },
            "end": { "line": range.end.line, "column": range.end.column },
        })
    }

    /// Parse + validate + lint. Always returns Ok: parse/validation failures
    /// are reported inside the JSON (`valid: false`), not as an error.
    pub fn check(source: &str) -> Value {
        let suppressions = collect_lint_suppressions(source);
        let style_lints: Vec<Value> = lint_canonical_style(source)
            .into_iter()
            .filter(|l| !lint_is_suppressed(&suppressions, l.code, l.range))
            .map(|l| {
                json!({
                    "code": l.code,
                    "message": l.message,
                    "range": range_json(l.range),
                    "replacement": l.replacement,
                })
            })
            .collect();

        match parse(source) {
            Ok(ast) => {
                let statistical_lints: Vec<Value> = lint_statistical_guardrails(&ast, source)
                    .into_iter()
                    .filter(|l| !lint_is_suppressed(&suppressions, l.code, l.range))
                    .map(|l| {
                        json!({
                            "code": l.code,
                            "message": l.message,
                            "range": range_json(l.range),
                            "severity": match l.severity {
                                LintSeverity::Warning => "warning",
                                LintSeverity::Information => "information",
                            },
                        })
                    })
                    .collect();
                json!({
                    "valid": true,
                    "error": null,
                    "style_lints": style_lints,
                    "statistical_lints": statistical_lints,
                })
            }
            Err(message) => json!({
                "valid": false,
                "error": message,
                "style_lints": style_lints,
                "statistical_lints": [],
            }),
        }
    }

    pub fn list_flows(source: &str) -> Result<Value, String> {
        let ast = parse(source)?;
        Ok(Value::Array(
            ast.flows
                .iter()
                .map(|f| Value::String(f.name.clone()))
                .collect(),
        ))
    }

    fn params_json(params: &[(String, f64)]) -> Value {
        Value::Object(
            params
                .iter()
                .map(|(name, value)| (name.clone(), json!(value)))
                .collect(),
        )
    }

    fn posterior_json(posterior: &PosteriorType) -> Value {
        match posterior {
            PosteriorType::Gaussian { params } => json!({
                "family": "gaussian",
                "params": params_json(params),
            }),
            PosteriorType::Bernoulli { params } => json!({
                "family": "bernoulli",
                "params": params_json(params),
            }),
            PosteriorType::Categorical {
                group_by, prior, ..
            } => json!({
                "family": "categorical",
                "group_by": group_by,
                "prior": match prior {
                    CategoricalPrior::Uniform { pseudo_count } => json!({
                        "kind": "uniform",
                        "pseudo_count": pseudo_count,
                    }),
                    CategoricalPrior::Explicit { concentrations } => json!({
                        "kind": "explicit",
                        "concentrations": concentrations,
                    }),
                },
            }),
        }
    }

    fn graph_expr_json(expr: &GraphExpr) -> Value {
        match expr {
            GraphExpr::FromEvidence { evidence } => json!({
                "kind": "from_evidence",
                "evidence": evidence,
            }),
            GraphExpr::FromGraph { alias } => json!({
                "kind": "from_graph",
                "alias": alias,
            }),
            GraphExpr::Pipeline { start, transforms } => json!({
                "kind": "pipeline",
                "start": start,
                "transforms": transforms.iter().map(|t| match t {
                    Transform::ApplyRule { rule } => json!({ "kind": "apply_rule", "rule": rule }),
                    Transform::ApplyRuleset { rules } => json!({ "kind": "apply_ruleset", "rules": rules }),
                    Transform::Snapshot { name } => json!({ "kind": "snapshot", "name": name }),
                    Transform::InferBeliefs => json!({ "kind": "infer_beliefs" }),
                    Transform::PruneEdges { edge_type, .. } => json!({
                        "kind": "prune_edges",
                        "edge_type": edge_type,
                    }),
                }).collect::<Vec<_>>(),
            }),
            GraphExpr::SelectModel {
                candidates,
                criterion,
            } => json!({
                "kind": "select_model",
                "candidates": candidates,
                "criterion": match criterion {
                    ModelSelectionCriterion::EdgeAic => "edge_aic",
                    ModelSelectionCriterion::EdgeBic => "edge_bic",
                },
            }),
        }
    }

    fn flow_json(flow: &FlowDef) -> Value {
        json!({
            "name": flow.name,
            "on_model": flow.on_model,
            "graphs": flow.graphs.iter().map(|g| json!({
                "name": g.name,
                "expr": graph_expr_json(&g.expr),
            })).collect::<Vec<_>>(),
            "metrics": flow.metrics.iter().map(|m| m.name.clone()).collect::<Vec<_>>(),
            "exports": flow.exports.iter().map(|e| json!({
                "graph": e.graph,
                "alias": e.alias,
            })).collect::<Vec<_>>(),
            "metric_exports": flow.metric_exports.iter().map(|e| json!({
                "metric": e.metric,
                "alias": e.alias,
            })).collect::<Vec<_>>(),
            "metric_imports": flow.metric_imports.iter().map(|i| json!({
                "source_alias": i.source_alias,
                "local_name": i.local_name,
            })).collect::<Vec<_>>(),
            "needs_prior": grafial_core::flow_needs_prior(flow),
        })
    }

    /// Structural description of every declaration in the program, plus the
    /// relationships a visual composer needs to draw it as a graph:
    /// schema <- belief_model <- evidence/rule/flow, pipeline dataflow within
    /// flows, and cross-flow export/import edges.
    pub fn program_structure(source: &str) -> Result<Value, String> {
        let ast = parse(source)?;

        Ok(json!({
            "schemas": ast.schemas.iter().map(|s| json!({
                "name": s.name,
                "nodes": s.nodes.iter().map(|n| json!({
                    "name": n.name,
                    "attrs": n.attrs.iter().map(|a| json!({
                        "name": a.name,
                        "type": a.ty,
                    })).collect::<Vec<_>>(),
                })).collect::<Vec<_>>(),
                "edges": s.edges.iter().map(|e| e.name.clone()).collect::<Vec<_>>(),
            })).collect::<Vec<_>>(),
            "belief_models": ast.belief_models.iter().map(|m| json!({
                "name": m.name,
                "on_schema": m.on_schema,
                "nodes": m.nodes.iter().map(|n| json!({
                    "node_type": n.node_type,
                    "attrs": n.attrs.iter().map(|(name, posterior)| json!({
                        "name": name,
                        "posterior": posterior_json(posterior),
                    })).collect::<Vec<_>>(),
                })).collect::<Vec<_>>(),
                "edges": m.edges.iter().map(|e| json!({
                    "edge_type": e.edge_type,
                    "exist": posterior_json(&e.exist),
                    "has_weight": e.weight.is_some(),
                })).collect::<Vec<_>>(),
            })).collect::<Vec<_>>(),
            "evidences": ast.evidences.iter().map(|e| json!({
                "name": e.name,
                "on_model": e.on_model,
                "observation_count": e.observations.len(),
            })).collect::<Vec<_>>(),
            "rules": ast.rules.iter().map(|r| json!({
                "name": r.name,
                "on_model": r.on_model,
                "mode": r.mode,
                "patterns": r.patterns.iter().map(|p| json!({
                    "src": { "var": p.src.var, "label": p.src.label },
                    "edge": { "var": p.edge.var, "type": p.edge.ty },
                    "dst": { "var": p.dst.var, "label": p.dst.label },
                })).collect::<Vec<_>>(),
                "has_where": r.where_expr.is_some(),
                "action_count": r.actions.len(),
            })).collect::<Vec<_>>(),
            "flows": ast.flows.iter().map(flow_json).collect::<Vec<_>>(),
        }))
    }

    /// Serializes a belief graph's full structure: every node with posterior
    /// attribute means/variances, every edge with its existence probability.
    fn graph_json(graph: &mut BeliefGraph) -> Result<Value, String> {
        graph.ensure_owned();

        let nodes: Vec<Value> = graph
            .nodes()
            .iter()
            .map(|n| {
                let mut attrs = serde_json::Map::new();
                let mut names: Vec<&String> = n.attrs.keys().collect();
                names.sort();
                for name in names {
                    let g = &n.attrs[name];
                    attrs.insert(
                        name.clone(),
                        json!({ "mean": g.mean, "variance": g.variance() }),
                    );
                }
                json!({
                    "id": n.id.0,
                    "label": n.label.as_ref(),
                    "attrs": attrs,
                })
            })
            .collect();

        let mut edges = Vec::with_capacity(graph.edges().len());
        for e in graph.edges() {
            let prob = graph.prob_mean(e.id).map_err(|err| err.to_string())?;
            edges.push(json!({
                "id": e.id.0,
                "src": e.src.0,
                "dst": e.dst.0,
                "type": e.ty.as_ref(),
                "prob": prob,
            }));
        }

        Ok(json!({ "nodes": nodes, "edges": edges }))
    }

    fn graph_map_json(
        graphs: &mut std::collections::HashMap<String, BeliefGraph>,
    ) -> Result<Value, String> {
        let mut out = serde_json::Map::new();
        let mut names: Vec<String> = graphs.keys().cloned().collect();
        names.sort();
        for name in names {
            let graph = graphs.get_mut(&name).expect("key from same map");
            out.insert(name, graph_json(graph)?);
        }
        Ok(Value::Object(out))
    }

    fn flow_result_json(flow_name: &str, mut result: FlowResult) -> Result<Value, String> {
        Ok(json!({
            "flow": flow_name,
            "graphs": graph_map_json(&mut result.graphs)?,
            "exports": graph_map_json(&mut result.exports)?,
            "snapshots": graph_map_json(&mut result.snapshots)?,
            "metrics": result.metrics,
            "metric_exports": result.metric_exports,
            "intervention_audit": result.intervention_audit.iter().map(|e| json!({
                "flow": e.flow,
                "graph": e.graph,
                "transform": e.transform,
                "rule": e.rule,
                "mode": e.mode,
                "matched_bindings": e.matched_bindings,
                "actions_executed": e.actions_executed,
            })).collect::<Vec<_>>(),
            "inference_diagnostics": result.inference_diagnostics.iter().map(|e| json!({
                "flow": e.flow,
                "graph": e.graph,
                "transform": e.transform,
                "algorithm": e.algorithm,
                "iterations_run": e.iterations_run,
                "max_iterations": e.max_iterations,
                "converged": e.converged,
                "final_max_message_delta": e.final_max_message_delta,
                "variable_count": e.variable_count,
                "connected_variable_count": e.connected_variable_count,
            })).collect::<Vec<_>>(),
        }))
    }

    /// Executes a flow, running prerequisite flows in program order first when
    /// the target imports graphs or metrics from prior flows (same semantics
    /// as the CLI).
    pub fn run_flow(source: &str, flow_name: &str) -> Result<Value, String> {
        let ast = parse(source)?;
        let result =
            grafial_core::run_flow_with_dependencies(&ast, flow_name).map_err(|e| e.to_string())?;
        flow_result_json(flow_name, result)
    }
}

fn to_js(value: &serde_json::Value) -> Result<String, JsError> {
    serde_json::to_string(value).map_err(|e| JsError::new(&e.to_string()))
}

/// Crate version (mirrors the workspace version).
#[wasm_bindgen]
pub fn version() -> String {
    env!("CARGO_PKG_VERSION").to_string()
}

/// Parse + validate + lint a program. Returns JSON:
/// `{ valid, error, style_lints: [...], statistical_lints: [...] }`.
#[wasm_bindgen]
pub fn check(source: &str) -> Result<String, JsError> {
    to_js(&api::check(source))
}

/// Rewrites source to canonical style (identity if already canonical).
#[wasm_bindgen]
pub fn format_canonical(source: &str) -> String {
    grafial_frontend::format_canonical_style(source)
}

/// Flow names in program order, as a JSON array of strings.
#[wasm_bindgen]
pub fn list_flows(source: &str) -> Result<String, JsError> {
    to_js(&api::list_flows(source).map_err(|e| JsError::new(&e))?)
}

/// Structural JSON description of the program for visual composition.
#[wasm_bindgen]
pub fn program_structure(source: &str) -> Result<String, JsError> {
    to_js(&api::program_structure(source).map_err(|e| JsError::new(&e))?)
}

/// Executes a flow (with automatic prerequisite chaining) and returns the
/// full result — metrics, audit events, and belief-graph structure — as JSON.
#[wasm_bindgen]
pub fn run_flow(source: &str, flow_name: &str) -> Result<String, JsError> {
    to_js(&api::run_flow(source, flow_name).map_err(|e| JsError::new(&e))?)
}

#[cfg(test)]
mod tests {
    use super::api;

    const PROGRAM: &str = r#"
schema Social {
  node Person { score: Real }
  edge REL { }
}

belief_model SocialBeliefs on Social {
  node Person { score ~ Gaussian(mean=0.0, precision=1.0) }
  edge REL { exist ~ Bernoulli(prior=0.5, weight=2.0) }
}

evidence Ev on SocialBeliefs {
  Person { "Alice" { score: 1.0 }, "Bob" { score: 3.0 } }
  REL(Person -> Person) { "Alice" -> "Bob" }
}

flow First on SocialBeliefs {
  graph base = from_evidence Ev
  metric total = nodes(Person) |> sum(by=E[node.score])
  export base as "first_graph"
  export_metric total as "first_total"
}

flow Second on SocialBeliefs {
  import_metric first_total as prior_total
  graph g = from_graph "first_graph"
  metric doubled = prior_total * 2.0
  export g as "second_graph"
}
"#;

    #[test]
    fn check_reports_valid_program() {
        let value = api::check(PROGRAM);
        assert_eq!(value["valid"], true);
        assert!(value["error"].is_null());
    }

    #[test]
    fn check_reports_parse_failure_inline() {
        let value = api::check("flow Broken on Nothing {");
        assert_eq!(value["valid"], false);
        assert!(value["error"].is_string());
    }

    #[test]
    fn list_flows_in_program_order() {
        let value = api::list_flows(PROGRAM).expect("list_flows");
        let names: Vec<&str> = value
            .as_array()
            .unwrap()
            .iter()
            .map(|v| v.as_str().unwrap())
            .collect();
        assert_eq!(names, vec!["First", "Second"]);
    }

    #[test]
    fn program_structure_describes_declarations_and_dependencies() {
        let value = api::program_structure(PROGRAM).expect("program_structure");

        assert_eq!(value["schemas"][0]["name"], "Social");
        assert_eq!(value["schemas"][0]["nodes"][0]["attrs"][0]["name"], "score");
        assert_eq!(
            value["belief_models"][0]["edges"][0]["exist"]["family"],
            "bernoulli"
        );
        assert_eq!(value["evidences"][0]["on_model"], "SocialBeliefs");

        // Inter-flow dependency surface for the visual composer.
        let second = &value["flows"][1];
        assert_eq!(second["name"], "Second");
        assert_eq!(second["needs_prior"], true);
        assert_eq!(second["graphs"][0]["expr"]["kind"], "from_graph");
        assert_eq!(second["graphs"][0]["expr"]["alias"], "first_graph");
        assert_eq!(second["metric_imports"][0]["source_alias"], "first_total");

        let first = &value["flows"][0];
        assert_eq!(first["needs_prior"], false);
        assert_eq!(first["exports"][0]["alias"], "first_graph");
    }

    #[test]
    fn run_flow_chains_prerequisites_and_serializes_graphs() {
        let value = api::run_flow(PROGRAM, "Second").expect("run_flow");

        // total = 1/2 + 3/2 = 2.0 (Gaussian posterior means); doubled = 4.0
        let doubled = value["metrics"]["doubled"].as_f64().unwrap();
        assert!((doubled - 4.0).abs() < 1e-9, "doubled = {}", doubled);

        // Imported graph is fully serialized: 2 nodes with posteriors, 1 edge
        // with existence probability 2/3 (Beta(1,1) + one present observation).
        let graph = &value["exports"]["second_graph"];
        assert_eq!(graph["nodes"].as_array().unwrap().len(), 2);
        let edge = &graph["edges"][0];
        let prob = edge["prob"].as_f64().unwrap();
        assert!((prob - 2.0 / 3.0).abs() < 1e-9, "prob = {}", prob);
        let mean = graph["nodes"][0]["attrs"]["score"]["mean"]
            .as_f64()
            .unwrap();
        assert!((mean - 0.5).abs() < 1e-9, "mean = {}", mean);
    }

    #[test]
    fn run_flow_unknown_flow_is_an_error() {
        assert!(api::run_flow(PROGRAM, "Nope").is_err());
    }
}
