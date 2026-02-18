//! # Grafial Parser
//!
//! This module implements the parser for the Grafial DSL using the Pest parser generator.
//!
//! ## Overview
//!
//! The parser transforms source text into a typed Abstract Syntax Tree (AST) without
//! performing semantic validation. It handles:
//!
//! - Schema definitions (node and edge types)
//! - Belief model declarations
//! - Evidence specifications
//! - Rule definitions with patterns and actions
//! - Flow definitions with graph transformations
//!
//! ## Error Handling
//!
//! All builder functions return `Result<T, FrontendError>` to provide proper error
//! messages for malformed input. Numbers are parsed at parse time to avoid
//! repeated parsing during evaluation.
//!
//! ## Grammar
//!
//! The grammar is defined in `grammar/grafial.pest` using Pest's PEG syntax.

use crate::ast::*;
use crate::errors::FrontendError;
use pest::Parser;
use pest_derive::Parser;

#[derive(Parser)]
#[grammar = "../grammar.pest"]
pub struct BayGraphParser;

/// Parses Grafial DSL source into an Abstract Syntax Tree.
///
/// This is a pure syntactic parser that does not perform semantic validation.
/// Use [`crate::frontend::validate::validate_program`] to validate the resulting AST.
///
/// # Arguments
///
/// * `source` - The complete Grafial DSL source code
///
/// # Returns
///
/// * `Ok(ProgramAst)` - Successfully parsed program
/// * `Err(FrontendError::ParseError)` - Syntax error with location information
///
/// # Example
///
/// ```rust,ignore
/// use grafial::frontend::parser::parse_program;
///
/// let source = "schema S { node N {} edge E {} }";
/// let ast = parse_program(source)?;
/// assert_eq!(ast.schemas.len(), 1);
/// ```
pub fn parse_program(source: &str) -> Result<ProgramAst, FrontendError> {
    let mut schemas = Vec::new();
    let mut belief_models = Vec::new();
    let mut evidences = Vec::new();
    let mut rules = Vec::new();
    let mut flows = Vec::new();

    let mut pairs = BayGraphParser::parse(Rule::program, source)
        .map_err(|e| FrontendError::ParseError(e.to_string()))?;

    if let Some(program_pair) = pairs.next() {
        debug_assert_eq!(program_pair.as_rule(), Rule::program);
        for inner in program_pair.into_inner() {
            match inner.as_rule() {
                Rule::decl => {
                    for d in inner.into_inner() {
                        match d.as_rule() {
                            Rule::schema_decl => schemas.push(build_schema(d)?),
                            Rule::belief_model_decl => {
                                belief_models.push(build_belief_model(d, source)?)
                            }
                            Rule::evidence_decl => evidences.push(build_evidence(d, source)?),
                            Rule::rule_decl => rules.push(build_rule(d, source)?),
                            Rule::flow_decl => flows.push(build_flow(d, source)?),
                            _ => {}
                        }
                    }
                }
                Rule::schema_decl => schemas.push(build_schema(inner)?),
                Rule::belief_model_decl => belief_models.push(build_belief_model(inner, source)?),
                Rule::evidence_decl => evidences.push(build_evidence(inner, source)?),
                Rule::rule_decl => rules.push(build_rule(inner, source)?),
                Rule::flow_decl => flows.push(build_flow(inner, source)?),
                _ => {}
            }
        }
    }

    Ok(ProgramAst {
        schemas,
        belief_models,
        evidences,
        rules,
        flows,
    })
}

fn build_schema(pair: pest::iterators::Pair<Rule>) -> Result<Schema, FrontendError> {
    let mut name = String::new();
    let mut nodes = Vec::new();
    let mut edges = Vec::new();
    for p in pair.into_inner() {
        match p.as_rule() {
            Rule::ident if name.is_empty() => name = p.as_str().to_string(),
            Rule::block_schema => {
                for b in p.into_inner() {
                    match b.as_rule() {
                        Rule::node_decl => nodes.push(build_node(b)?),
                        Rule::edge_decl => edges.push(build_edge(b)?),
                        _ => {}
                    }
                }
            }
            _ => {}
        }
    }
    Ok(Schema { name, nodes, edges })
}

fn build_node(pair: pest::iterators::Pair<Rule>) -> Result<NodeDef, FrontendError> {
    let mut name = String::new();
    let mut attrs = Vec::new();
    for p in pair.into_inner() {
        match p.as_rule() {
            Rule::ident if name.is_empty() => name = p.as_str().to_string(),
            Rule::attr_decl => {
                let mut ai = p.into_inner();
                let an = ai
                    .next()
                    .ok_or_else(|| FrontendError::ParseError("Missing attribute name".to_string()))?
                    .as_str()
                    .to_string();
                let ty = ai
                    .next()
                    .ok_or_else(|| FrontendError::ParseError("Missing attribute type".to_string()))?
                    .as_str()
                    .to_string();
                attrs.push(AttrDef { name: an, ty });
            }
            _ => {}
        }
    }
    Ok(NodeDef { name, attrs })
}

fn build_edge(pair: pest::iterators::Pair<Rule>) -> Result<EdgeDef, FrontendError> {
    let name = pair
        .into_inner()
        .find(|p| p.as_rule() == Rule::ident)
        .ok_or_else(|| FrontendError::ParseError("Missing edge name".to_string()))?
        .as_str()
        .to_string();
    Ok(EdgeDef { name })
}

/// Extracts the exact source text for a parse tree node.
///
/// This is used to preserve the original source for belief model and evidence bodies,
/// which are stored as-is for future processing.
fn block_src(pair: &pest::iterators::Pair<Rule>, source: &str) -> String {
    let span = pair.as_span();
    source[span.start()..span.end()].to_string()
}

/// Helper to extract a string literal without quotes
fn unquote_string(s: &str) -> String {
    s.trim_matches('"').to_string()
}

/// Helper to extract the first ident from an iterator
fn extract_ident(
    iter: &mut pest::iterators::Pairs<Rule>,
    error_msg: &str,
) -> Result<String, FrontendError> {
    iter.find(|p| p.as_rule() == Rule::ident)
        .map(|p| p.as_str().to_string())
        .ok_or_else(|| FrontendError::ParseError(error_msg.to_string()))
}

/// Helper to extract a string from an iterator
fn extract_string(
    iter: &mut pest::iterators::Pairs<Rule>,
    error_msg: &str,
) -> Result<String, FrontendError> {
    iter.find(|p| p.as_rule() == Rule::string)
        .map(|p| unquote_string(p.as_str()))
        .ok_or_else(|| FrontendError::ParseError(error_msg.to_string()))
}

fn build_belief_model(
    pair: pest::iterators::Pair<Rule>,
    source: &str,
) -> Result<BeliefModel, FrontendError> {
    let mut name = String::new();
    let mut on_schema = String::new();
    let mut nodes = Vec::new();
    let mut edges = Vec::new();
    let mut body_src = String::new();

    for p in pair.into_inner() {
        match p.as_rule() {
            Rule::ident => {
                if name.is_empty() {
                    name = p.as_str().to_string();
                } else if on_schema.is_empty() {
                    on_schema = p.as_str().to_string();
                }
            }
            Rule::belief_model_body => {
                body_src = block_src(&p, source);
                for b in p.into_inner() {
                    match b.as_rule() {
                        Rule::node_belief_decl => nodes.push(build_node_belief(b)?),
                        Rule::edge_belief_decl => edges.push(build_edge_belief(b)?),
                        _ => {}
                    }
                }
            }
            _ => {}
        }
    }
    Ok(BeliefModel {
        name,
        on_schema,
        nodes,
        edges,
        body_src,
    })
}

fn build_node_belief(pair: pest::iterators::Pair<Rule>) -> Result<NodeBeliefDecl, FrontendError> {
    let mut node_type = String::new();
    let mut attrs = Vec::new();

    for p in pair.into_inner() {
        match p.as_rule() {
            Rule::ident if node_type.is_empty() => node_type = p.as_str().to_string(),
            Rule::attr_belief_decl => {
                // Collect all pairs to inspect them
                let pairs: Vec<_> = p.into_inner().collect();
                // First ident is attribute name
                let attr_name = pairs
                    .iter()
                    .find(|p| p.as_rule() == Rule::ident)
                    .ok_or_else(|| FrontendError::ParseError("Missing attribute name".to_string()))?
                    .as_str()
                    .to_string();
                // Find posterior_type (Pest wraps it)
                let posterior_pair = pairs
                    .iter()
                    .find(|e| {
                        let rule = e.as_rule();
                        rule == Rule::posterior_type
                            || rule == Rule::gaussian_posterior
                            || rule == Rule::bernoulli_posterior
                            || rule == Rule::categorical_posterior
                    })
                    .ok_or_else(|| {
                        FrontendError::ParseError(format!(
                            "Missing posterior type. Found rules: {:?}",
                            pairs.iter().map(|p| p.as_rule()).collect::<Vec<_>>()
                        ))
                    })?;
                let posterior = build_posterior_type(posterior_pair.clone())?;
                attrs.push((attr_name, posterior));
            }
            _ => {}
        }
    }
    Ok(NodeBeliefDecl { node_type, attrs })
}

fn build_edge_belief(pair: pest::iterators::Pair<Rule>) -> Result<EdgeBeliefDecl, FrontendError> {
    let mut edge_type = String::new();
    let mut exist: Option<PosteriorType> = None;

    for p in pair.into_inner() {
        match p.as_rule() {
            Rule::ident if edge_type.is_empty() => edge_type = p.as_str().to_string(),
            Rule::exist_belief_decl => {
                let mut ei = p.into_inner();
                // Skip "exist" ~ "~", find the posterior type (may be posterior_type wrapper or specific rule)
                let posterior_pair = ei.find(|e| {
                    matches!(
                        e.as_rule(),
                        Rule::posterior_type
                            | Rule::gaussian_posterior
                            | Rule::bernoulli_posterior
                            | Rule::categorical_posterior
                    )
                });
                if let Some(pp) = posterior_pair {
                    exist = Some(build_posterior_type(pp)?);
                }
            }
            _ => {}
        }
    }

    let exist_posterior = exist
        .ok_or_else(|| FrontendError::ParseError("Missing exist posterior for edge".to_string()))?;
    Ok(EdgeBeliefDecl {
        edge_type,
        exist: exist_posterior,
    })
}

fn build_posterior_type(pair: pest::iterators::Pair<Rule>) -> Result<PosteriorType, FrontendError> {
    let rule = pair.as_rule();
    // Handle case where pair is posterior_type wrapper (get inner rule)
    let actual_pair = if rule == Rule::posterior_type {
        pair.into_inner()
            .next()
            .ok_or_else(|| FrontendError::ParseError("Empty posterior_type".to_string()))?
    } else {
        pair
    };
    match actual_pair.as_rule() {
        Rule::gaussian_posterior => {
            let mut params = Vec::new();
            for p in actual_pair.into_inner() {
                if let Rule::gaussian_param = p.as_rule() {
                    let mut gp = p.into_inner();
                    let raw_name = gp.next().unwrap().as_str().to_string();
                    let value =
                        gp.next().unwrap().as_str().parse::<f64>().map_err(|e| {
                            FrontendError::ParseError(format!("Invalid number: {}", e))
                        })?;
                    // Normalize parameter names: accept short forms
                    let name = match raw_name.as_str() {
                        "mean" => "prior_mean".to_string(),
                        "precision" => "prior_precision".to_string(),
                        other => other.to_string(),
                    };
                    params.push((name, value));
                }
            }
            Ok(PosteriorType::Gaussian { params })
        }
        Rule::bernoulli_posterior => {
            let mut params = Vec::new();
            for p in actual_pair.into_inner() {
                if let Rule::bernoulli_param = p.as_rule() {
                    let mut bp = p.into_inner();
                    let raw_name = bp.next().unwrap().as_str().to_string();
                    let value =
                        bp.next().unwrap().as_str().parse::<f64>().map_err(|e| {
                            FrontendError::ParseError(format!("Invalid number: {}", e))
                        })?;
                    let name = match raw_name.as_str() {
                        "weight" => "pseudo_count".to_string(),
                        other => other.to_string(),
                    };
                    params.push((name, value));
                }
            }
            Ok(PosteriorType::Bernoulli { params })
        }
        Rule::categorical_posterior => {
            let mut group_by: Option<String> = None;
            let mut prior: Option<CategoricalPrior> = None;
            let mut categories: Option<Vec<String>> = None;

            for p in actual_pair.into_inner() {
                if p.as_rule() == Rule::categorical_param {
                    // Parse categorical_param - similar to gaussian_param/bernoulli_param
                    // Structure: ident ~ "=" ~ categorical_param_value
                    let mut cp = p.into_inner();
                    let param_name_pair = cp.next().ok_or_else(|| {
                        FrontendError::ParseError(
                            "Missing parameter name in categorical_param".to_string(),
                        )
                    })?;
                    let param_name = param_name_pair.as_str();
                    // Next item is the value (may be wrapped in categorical_param_value)
                    let raw_value = cp.next().ok_or_else(|| {
                        FrontendError::ParseError(
                            "Missing parameter value in categorical_param".to_string(),
                        )
                    })?;
                    // Unwrap possible wrapper; some alternatives like the literal "uniform"
                    // may result in an empty inner, in which case use the raw text.
                    let value_inner_iter = if raw_value.as_rule() == Rule::categorical_param_value {
                        Some(raw_value.clone().into_inner())
                    } else {
                        None
                    };
                    let value_pair_opt = value_inner_iter.and_then(|mut it| it.next());
                    let value_raw_str = if value_pair_opt.is_none() {
                        Some(raw_value.as_str())
                    } else {
                        None
                    };

                    match param_name {
                        "group_by" => {
                            // Accept string or bare identifier for group_by
                            if let Some(vp) = value_pair_opt.clone() {
                                match vp.as_rule() {
                                    Rule::string => {
                                        let value_str = vp.as_str().trim_matches('"');
                                        group_by = Some(value_str.to_string());
                                    }
                                    Rule::ident => {
                                        group_by = Some(vp.as_str().to_string());
                                    }
                                    _ => {
                                        return Err(FrontendError::ParseError(
                                            "group_by parameter must be a string or identifier"
                                                .to_string(),
                                        ));
                                    }
                                }
                            } else if let Some(raw) = value_raw_str {
                                // Literal form (e.g., uniform) is invalid for group_by
                                return Err(FrontendError::ParseError(format!(
                                    "Invalid group_by value: {}",
                                    raw
                                )));
                            } else {
                                return Err(FrontendError::ParseError(
                                    "group_by parameter requires a value".to_string(),
                                ));
                            }
                        }
                        "prior" => match value_pair_opt.clone().map(|p| p.as_rule()) {
                            Some(Rule::prior_array) => {
                                let mut concentrations = Vec::new();
                                for n in value_pair_opt.unwrap().into_inner() {
                                    if let Rule::number = n.as_rule() {
                                        let val = n.as_str().parse::<f64>().map_err(|e| {
                                            FrontendError::ParseError(format!(
                                                "Invalid number: {}",
                                                e
                                            ))
                                        })?;
                                        concentrations.push(val);
                                    }
                                }
                                prior = Some(CategoricalPrior::Explicit { concentrations });
                            }
                            // Allow unquoted uniform token; actual pseudo_count comes separately
                            Some(Rule::ident) => {
                                if value_pair_opt.as_ref().unwrap().as_str() != "uniform" {
                                    return Err(FrontendError::ParseError(
                                            format!(
                                                "Unknown prior identifier: {} (expected 'uniform' or array)",
                                                value_pair_opt.as_ref().unwrap().as_str()
                                            ),
                                        ));
                                }
                                // Defer setting prior until pseudo_count
                            }
                            // If grammar wraps uniform differently, accept string too
                            Some(Rule::string) => {
                                let s = value_pair_opt.unwrap().as_str().trim_matches('"');
                                if s != "uniform" {
                                    return Err(FrontendError::ParseError(format!(
                                        "Unknown prior string: {} (expected 'uniform')",
                                        s
                                    )));
                                }
                            }
                            _ => {
                                // Don't set prior here; pseudo_count will determine uniform
                                // Handle the case where the wrapper had empty inner but raw text is present
                                if let Some(s) = value_raw_str {
                                    if s == "uniform" {
                                        // ok, defer
                                    } else {
                                        return Err(FrontendError::ParseError(format!(
                                            "Unknown prior: {}",
                                            s
                                        )));
                                    }
                                }
                            }
                        },
                        "pseudo_count" => {
                            if let Some(vp) = value_pair_opt.clone() {
                                if let Rule::number = vp.as_rule() {
                                    let value = vp.as_str().parse::<f64>().map_err(|e| {
                                        FrontendError::ParseError(format!("Invalid number: {}", e))
                                    })?;
                                    prior = Some(CategoricalPrior::Uniform {
                                        pseudo_count: value,
                                    });
                                } else {
                                    return Err(FrontendError::ParseError(
                                        "pseudo_count parameter must be a number".to_string(),
                                    ));
                                }
                            } else {
                                return Err(FrontendError::ParseError(
                                    "pseudo_count parameter must be a number".to_string(),
                                ));
                            }
                        }
                        "categories" => {
                            if let Some(vp) = value_pair_opt.clone() {
                                if let Rule::categorical_categories_array = vp.as_rule() {
                                    let mut cats = Vec::new();
                                    for s in vp.into_inner() {
                                        if let Rule::string = s.as_rule() {
                                            let val = s.as_str().trim_matches('"').to_string();
                                            cats.push(val);
                                        }
                                    }
                                    categories = Some(cats);
                                } else {
                                    return Err(FrontendError::ParseError(
                                        "categories parameter must be an array of strings"
                                            .to_string(),
                                    ));
                                }
                            } else {
                                return Err(FrontendError::ParseError(
                                    "categories parameter must be an array of strings".to_string(),
                                ));
                            }
                        }
                        _ => {
                            return Err(FrontendError::ParseError(format!(
                                "Unknown categorical parameter: {}",
                                param_name
                            )));
                        }
                    }
                }
            }

            let group_by_val = group_by.ok_or_else(|| {
                FrontendError::ParseError("Missing group_by parameter".to_string())
            })?;
            let prior_val = prior
                .ok_or_else(|| FrontendError::ParseError("Missing prior parameter".to_string()))?;
            Ok(PosteriorType::Categorical {
                group_by: group_by_val,
                prior: prior_val,
                categories,
            })
        }
        _ => Err(FrontendError::ParseError(
            "Invalid posterior type".to_string(),
        )),
    }
}

fn build_evidence(
    pair: pest::iterators::Pair<Rule>,
    source: &str,
) -> Result<EvidenceDef, FrontendError> {
    let mut name = String::new();
    let mut on_model = String::new();
    let mut observations = Vec::new();
    let mut body_src = String::new();

    for p in pair.into_inner() {
        match p.as_rule() {
            Rule::ident => {
                if name.is_empty() {
                    name = p.as_str().to_string();
                } else if on_model.is_empty() {
                    on_model = p.as_str().to_string();
                }
            }
            Rule::evidence_body => {
                body_src = block_src(&p, source);
                for b in p.into_inner() {
                    match b.as_rule() {
                        Rule::observe_stmt => observations.push(build_observe_stmt(b)?),
                        Rule::node_group => build_node_group_into(b, &mut observations)?,
                        Rule::edge_group => build_edge_group_into(b, &mut observations)?,
                        Rule::choose_stmt | Rule::unchoose_stmt => {
                            let is_choose = b.as_rule() == Rule::choose_stmt;
                            // Extract edge_observe inside and map to chosen/unchosen
                            let mut edge_pair_opt = None;
                            for inner in b.clone().into_inner() {
                                if inner.as_rule() == Rule::edge_observe {
                                    edge_pair_opt = Some(inner);
                                    break;
                                }
                            }
                            let edge_pair = edge_pair_opt.ok_or_else(|| {
                                FrontendError::ParseError(
                                    "Missing edge target in choose/unchoose".to_string(),
                                )
                            })?;
                            let mut edge_type = String::new();
                            let mut src: Option<(String, String)> = None;
                            let mut dst: Option<(String, String)> = None;
                            for e in edge_pair.into_inner() {
                                match e.as_rule() {
                                    Rule::ident if edge_type.is_empty() => {
                                        edge_type = e.as_str().to_string()
                                    }
                                    Rule::node_ref => {
                                        if src.is_none() {
                                            src = Some(build_node_ref(e)?);
                                        } else if dst.is_none() {
                                            dst = Some(build_node_ref(e)?);
                                        }
                                    }
                                    _ => {}
                                }
                            }
                            let src_val = src.ok_or_else(|| {
                                FrontendError::ParseError(
                                    "Missing source node in choose/unchoose".to_string(),
                                )
                            })?;
                            let dst_val = dst.ok_or_else(|| {
                                FrontendError::ParseError(
                                    "Missing destination node in choose/unchoose".to_string(),
                                )
                            })?;
                            let mode = if is_choose {
                                EvidenceMode::Chosen
                            } else {
                                EvidenceMode::Unchosen
                            };
                            observations.push(ObserveStmt::Edge {
                                edge_type,
                                src: src_val,
                                dst: dst_val,
                                mode,
                            });
                        }
                        _ => {}
                    }
                }
            }
            _ => {}
        }
    }
    Ok(EvidenceDef {
        name,
        on_model,
        observations,
        body_src,
    })
}

/// Expands a node_group into attribute observation statements
fn build_node_group_into(
    pair: pest::iterators::Pair<Rule>,
    out: &mut Vec<ObserveStmt>,
) -> Result<(), FrontendError> {
    let mut it = pair.into_inner();
    let node_type = extract_ident(&mut it, "Missing node type in node group")?;
    for entry in it {
        if entry.as_rule() != Rule::node_entry {
            continue;
        }
        let mut ent_it = entry.into_inner();
        let label_tok = ent_it.next().ok_or_else(|| {
            FrontendError::ParseError("Missing node label in node entry".to_string())
        })?;
        let label = match label_tok.as_rule() {
            Rule::node_label => {
                let mut li = label_tok.into_inner();
                let t = li
                    .next()
                    .ok_or_else(|| FrontendError::ParseError("Empty node_label".to_string()))?;
                match t.as_rule() {
                    Rule::string => unquote_string(t.as_str()),
                    Rule::ident | Rule::number => t.as_str().to_string(),
                    r => {
                        return Err(FrontendError::ParseError(format!(
                            "Invalid node label token: {:?}",
                            r
                        )))
                    }
                }
            }
            Rule::string => unquote_string(label_tok.as_str()),
            Rule::ident | Rule::number => label_tok.as_str().to_string(),
            r => {
                return Err(FrontendError::ParseError(format!(
                    "Invalid node label token: {:?}",
                    r
                )))
            }
        };
        // Remaining are node_attr_kv items
        for kv in ent_it {
            if kv.as_rule() != Rule::node_attr_kv {
                continue;
            }
            let mut kvit = kv.into_inner();
            let attr = kvit
                .next()
                .ok_or_else(|| {
                    FrontendError::ParseError("Missing attr name in node entry".to_string())
                })?
                .as_str()
                .to_string();
            let val = kvit
                .next()
                .ok_or_else(|| {
                    FrontendError::ParseError("Missing attr value in node entry".to_string())
                })?
                .as_str()
                .parse::<f64>()
                .map_err(|e| FrontendError::ParseError(format!("Invalid number: {}", e)))?;
            // Optional precision_annot
            let mut precision: Option<f64> = None;
            if let Some(rem) = kvit.next() {
                if rem.as_rule() == Rule::precision_annot {
                    let mut pit = rem.into_inner();
                    let pname = pit
                        .next()
                        .ok_or_else(|| {
                            FrontendError::ParseError("Missing precision name".to_string())
                        })?
                        .as_str();
                    let pval = pit
                        .next()
                        .ok_or_else(|| {
                            FrontendError::ParseError("Missing precision value".to_string())
                        })?
                        .as_str()
                        .parse::<f64>()
                        .map_err(|e| FrontendError::ParseError(format!("Invalid number: {}", e)))?;
                    if pname != "precision" {
                        return Err(FrontendError::ParseError(
                            "Only precision=... supported in node entry".to_string(),
                        ));
                    }
                    precision = Some(pval);
                }
            }
            out.push(ObserveStmt::Attribute {
                node: (node_type.clone(), label.clone()),
                attr,
                value: val,
                precision,
            });
        }
    }
    Ok(())
}

/// Expands an edge_group into edge observation statements
fn build_edge_group_into(
    pair: pest::iterators::Pair<Rule>,
    out: &mut Vec<ObserveStmt>,
) -> Result<(), FrontendError> {
    let mut it = pair.into_inner();
    let edge_type = extract_ident(&mut it, "Missing edge type in edge group")?;
    // Expect src type "->" dst type
    let src_type = extract_ident(&mut it, "Missing source node type in edge group")?;
    let dst_type = extract_ident(&mut it, "Missing destination node type in edge group")?;
    // The remaining is a block: iterate entries
    for entry in it {
        if entry.as_rule() != Rule::edge_entry {
            continue;
        }
        let mut ent = entry.into_inner();
        // node_label ~ ("->" | "-/>") ~ node_label
        let left = ent
            .next()
            .ok_or_else(|| FrontendError::ParseError("Missing left label".to_string()))?;
        let arrow = ent
            .next()
            .ok_or_else(|| FrontendError::ParseError("Missing arrow".to_string()))?;
        let right = ent
            .next()
            .ok_or_else(|| FrontendError::ParseError("Missing right label".to_string()))?;
        let src_label = match left.as_rule() {
            Rule::node_label => {
                let mut li = left.into_inner();
                let t = li
                    .next()
                    .ok_or_else(|| FrontendError::ParseError("Empty node_label".to_string()))?;
                match t.as_rule() {
                    Rule::string => unquote_string(t.as_str()),
                    Rule::ident | Rule::number => t.as_str().to_string(),
                    r => {
                        return Err(FrontendError::ParseError(format!(
                            "Invalid label token: {:?}",
                            r
                        )))
                    }
                }
            }
            Rule::string => unquote_string(left.as_str()),
            Rule::ident | Rule::number => left.as_str().to_string(),
            r => {
                return Err(FrontendError::ParseError(format!(
                    "Invalid label token: {:?}",
                    r
                )))
            }
        };
        let dst_label = match right.as_rule() {
            Rule::node_label => {
                let mut li = right.into_inner();
                let t = li
                    .next()
                    .ok_or_else(|| FrontendError::ParseError("Empty node_label".to_string()))?;
                match t.as_rule() {
                    Rule::string => unquote_string(t.as_str()),
                    Rule::ident | Rule::number => t.as_str().to_string(),
                    r => {
                        return Err(FrontendError::ParseError(format!(
                            "Invalid label token: {:?}",
                            r
                        )))
                    }
                }
            }
            Rule::string => unquote_string(right.as_str()),
            Rule::ident | Rule::number => right.as_str().to_string(),
            r => {
                return Err(FrontendError::ParseError(format!(
                    "Invalid label token: {:?}",
                    r
                )))
            }
        };
        let arrow_text = arrow.as_str();
        let mode = if arrow_text == "->" {
            EvidenceMode::Present
        } else {
            EvidenceMode::Absent
        };
        out.push(ObserveStmt::Edge {
            edge_type: edge_type.clone(),
            src: (src_type.clone(), src_label),
            dst: (dst_type.clone(), dst_label),
            mode,
        });
    }
    Ok(())
}

fn build_observe_stmt(pair: pest::iterators::Pair<Rule>) -> Result<ObserveStmt, FrontendError> {
    // Handle observe_stmt parent rule - get the inner rule
    let actual_pair = match pair.as_rule() {
        Rule::observe_stmt => {
            // Get the inner rule (observe_edge_stmt or observe_attr_stmt)
            pair.into_inner()
                .next()
                .ok_or_else(|| FrontendError::ParseError("Empty observe_stmt".to_string()))?
        }
        Rule::observe_edge_stmt | Rule::observe_attr_stmt => pair,
        _ => {
            return Err(FrontendError::ParseError(format!(
                "Invalid observe statement rule: {:?}",
                pair.as_rule()
            )))
        }
    };

    match actual_pair.as_rule() {
        Rule::observe_edge_stmt => {
            let mut target_pair = None;
            let mut mode_pair = None;

            for p in actual_pair.into_inner() {
                match p.as_rule() {
                    Rule::edge_observe => target_pair = Some(p),
                    Rule::evidence_mode => mode_pair = Some(p),
                    _ => {}
                }
            }

            let target = target_pair.ok_or_else(|| {
                FrontendError::ParseError("Missing edge observe target".to_string())
            })?;
            let mut edge_type = String::new();
            let mut src: Option<(String, String)> = None;
            let mut dst: Option<(String, String)> = None;

            for e in target.into_inner() {
                match e.as_rule() {
                    Rule::ident if edge_type.is_empty() => edge_type = e.as_str().to_string(),
                    Rule::node_ref => {
                        if src.is_none() {
                            src = Some(build_node_ref(e)?);
                        } else if dst.is_none() {
                            dst = Some(build_node_ref(e)?);
                        }
                    }
                    _ => {}
                }
            }

            let src_val =
                src.ok_or_else(|| FrontendError::ParseError("Missing source node".to_string()))?;
            let dst_val = dst
                .ok_or_else(|| FrontendError::ParseError("Missing destination node".to_string()))?;
            let mode =
                build_evidence_mode(mode_pair.ok_or_else(|| {
                    FrontendError::ParseError("Missing evidence mode".to_string())
                })?)?;

            Ok(ObserveStmt::Edge {
                edge_type,
                src: src_val,
                dst: dst_val,
                mode,
            })
        }
        Rule::observe_attr_stmt => {
            let mut target_pair = None;

            for p in actual_pair.into_inner() {
                if p.as_rule() == Rule::attr_observe {
                    target_pair = Some(p);
                }
            }

            let target = target_pair.ok_or_else(|| {
                FrontendError::ParseError("Missing attribute observe target".to_string())
            })?;
            let mut node: Option<(String, String)> = None;
            let mut attr = String::new();
            let mut value: Option<f64> = None;
            let mut precision: Option<f64> = None;

            for a in target.into_inner() {
                match a.as_rule() {
                    Rule::node_ref => node = Some(build_node_ref(a)?),
                    Rule::ident if attr.is_empty() => attr = a.as_str().to_string(),
                    Rule::number => {
                        value = Some(a.as_str().parse::<f64>().map_err(|e| {
                            FrontendError::ParseError(format!("Invalid number: {}", e))
                        })?);
                    }
                    Rule::precision_annot => {
                        // Only accept precision=number
                        let mut pit = a.into_inner();
                        let pname = pit
                            .next()
                            .ok_or_else(|| {
                                FrontendError::ParseError("Missing precision name".to_string())
                            })?
                            .as_str()
                            .to_string();
                        let pval = pit
                            .next()
                            .ok_or_else(|| {
                                FrontendError::ParseError("Missing precision value".to_string())
                            })?
                            .as_str()
                            .parse::<f64>()
                            .map_err(|e| {
                                FrontendError::ParseError(format!("Invalid number: {}", e))
                            })?;
                        if pname != "precision" {
                            return Err(FrontendError::ParseError(
                                "Only precision=... is supported in attribute observation"
                                    .to_string(),
                            ));
                        }
                        precision = Some(pval);
                    }
                    _ => {}
                }
            }

            let node_val = node
                .ok_or_else(|| FrontendError::ParseError("Missing node reference".to_string()))?;
            let value_val = value
                .ok_or_else(|| FrontendError::ParseError("Missing attribute value".to_string()))?;

            Ok(ObserveStmt::Attribute {
                node: node_val,
                attr,
                value: value_val,
                precision,
            })
        }
        _ => Err(FrontendError::ParseError(format!(
            "Invalid observe statement rule: {:?}",
            actual_pair.as_rule()
        ))),
    }
}

fn build_node_ref(pair: pest::iterators::Pair<Rule>) -> Result<(String, String), FrontendError> {
    let mut it = pair.into_inner();
    let node_type = extract_ident(&mut it, "Missing node type in node reference")?;
    let rest = it
        .next()
        .ok_or_else(|| FrontendError::ParseError("Missing node label".to_string()))?;
    match rest.as_rule() {
        Rule::node_ref_bracket => {
            // ["label"]
            let mut inner = rest.into_inner();
            let label = extract_string(&mut inner, "Missing label in node reference")?;
            Ok((node_type, label))
        }
        Rule::node_ref_call => {
            // (string | ident | number)
            let mut inner = rest.into_inner();
            let tok = inner
                .next()
                .ok_or_else(|| FrontendError::ParseError("Missing label in node()".to_string()))?;
            let label = match tok.as_rule() {
                Rule::string => unquote_string(tok.as_str()),
                Rule::ident | Rule::number => tok.as_str().to_string(),
                r => {
                    return Err(FrontendError::ParseError(format!(
                        "Invalid node label token: {:?}",
                        r
                    )))
                }
            };
            Ok((node_type, label))
        }
        _ => Err(FrontendError::ParseError(
            "Invalid node reference syntax".to_string(),
        )),
    }
}

fn build_evidence_mode(pair: pest::iterators::Pair<Rule>) -> Result<EvidenceMode, FrontendError> {
    match pair.as_str() {
        "present" => Ok(EvidenceMode::Present),
        "absent" => Ok(EvidenceMode::Absent),
        "chosen" => Ok(EvidenceMode::Chosen),
        "unchosen" => Ok(EvidenceMode::Unchosen),
        "forced_choice" => Ok(EvidenceMode::ForcedChoice),
        _ => Err(FrontendError::ParseError(format!(
            "Invalid evidence mode: {}",
            pair.as_str()
        ))),
    }
}

fn build_rule(pair: pest::iterators::Pair<Rule>, _source: &str) -> Result<RuleDef, FrontendError> {
    let mut name = String::new();
    let mut on_model = String::new();
    let mut patterns = Vec::new();
    let mut where_expr = None;
    let mut actions: Vec<ActionStmt> = Vec::new();
    let mut mode = None;

    for p in pair.into_inner() {
        match p.as_rule() {
            Rule::ident => {
                if name.is_empty() {
                    name = p.as_str().to_string();
                } else if on_model.is_empty() {
                    on_model = p.as_str().to_string();
                }
            }
            Rule::rule_body => {
                for b in p.into_inner() {
                    match b.as_rule() {
                        Rule::sugar_rule => {
                            // sugar_rule = pattern_item (KW_where expr)? => action_block
                            let mut it = b.into_inner();
                            // First inner is pattern_item
                            let pat_pair = it.next().ok_or_else(|| {
                                FrontendError::ParseError(
                                    "Missing pattern in sugar rule".to_string(),
                                )
                            })?;
                            patterns.push(build_pattern_item(pat_pair)?);
                            // Next may be KW_where expr, then action_block
                            let mut pending_expr: Option<ExprAst> = None;
                            let mut actions_block: Option<pest::iterators::Pair<Rule>> = None;
                            for nxt in it {
                                match nxt.as_rule() {
                                    Rule::expr => pending_expr = Some(build_expr(nxt)),
                                    Rule::action_block => actions_block = Some(nxt),
                                    _ => {}
                                }
                            }
                            if where_expr.is_some() && pending_expr.is_some() {
                                return Err(FrontendError::ParseError(
                                    "Multiple where clauses not allowed in sugar rule".to_string(),
                                ));
                            }
                            if pending_expr.is_some() {
                                where_expr = pending_expr;
                            }
                            if let Some(ab) = actions_block {
                                collect_actions_from_block(ab, &mut actions)?;
                            } else {
                                return Err(FrontendError::ParseError(
                                    "Missing action block in sugar rule".to_string(),
                                ));
                            }
                        }
                        Rule::for_sugar => {
                            // for (Var:Label) [where expr]? => { actions }
                            let mut var: Option<String> = None;
                            let mut label: Option<String> = None;
                            // Optional where expr and action block
                            let mut pending_expr: Option<ExprAst> = None;
                            let mut actions_block: Option<pest::iterators::Pair<Rule>> = None;
                            for nxt in b.into_inner() {
                                match nxt.as_rule() {
                                    Rule::ident => {
                                        if var.is_none() {
                                            var = Some(nxt.as_str().to_string());
                                        } else if label.is_none() {
                                            // Fallback for cases where label token is emitted as ident.
                                            label = Some(nxt.as_str().to_string());
                                        }
                                    }
                                    Rule::label => {
                                        if label.is_none() {
                                            label = Some(nxt.as_str().to_string());
                                        }
                                    }
                                    Rule::expr => pending_expr = Some(build_expr(nxt)),
                                    Rule::action_block => actions_block = Some(nxt),
                                    _ => {}
                                }
                            }
                            let var = var.ok_or_else(|| {
                                FrontendError::ParseError("Missing var in for()".to_string())
                            })?;
                            let label = label.ok_or_else(|| {
                                FrontendError::ParseError("Missing label in for()".to_string())
                            })?;
                            if where_expr.is_some() && pending_expr.is_some() {
                                return Err(FrontendError::ParseError(
                                    "Multiple where clauses not allowed in rule".to_string(),
                                ));
                            }
                            if pending_expr.is_some() {
                                where_expr = pending_expr;
                            }
                            if let Some(ab) = actions_block {
                                collect_actions_from_block(ab, &mut actions)?;
                            } else {
                                return Err(FrontendError::ParseError(
                                    "Missing action block in for()".to_string(),
                                ));
                            }
                            // Insert a placeholder self-loop pattern that engine can special-case later
                            patterns.push(PatternItem {
                                src: NodePattern {
                                    var: var.clone(),
                                    label: label.clone(),
                                },
                                edge: EdgePattern {
                                    var: "__for_dummy".into(),
                                    ty: "__FOR_NODE__".into(),
                                },
                                dst: NodePattern { var, label },
                            });
                        }
                        Rule::pattern_clause => {
                            // pattern_clause = KW_pattern ~ pattern_list
                            // Iterate through inner to find pattern_list (skip KW_pattern token)
                            for pl in b.into_inner() {
                                if pl.as_rule() == Rule::pattern_list {
                                    // pattern_list = pattern_item ~ ("," ~ pattern_item)*
                                    // Collect all pattern_items from the pattern_list
                                    let pattern_items: Vec<_> = pl
                                        .into_inner()
                                        .filter(|item| item.as_rule() == Rule::pattern_item)
                                        .collect();
                                    for item in pattern_items {
                                        patterns.push(build_pattern_item(item)?);
                                    }
                                    break; // Only one pattern_list per pattern_clause
                                }
                            }
                        }
                        Rule::where_clause => {
                            let mut expr_opt: Option<ExprAst> = None;
                            for piece in b.into_inner() {
                                match piece.as_rule() {
                                    Rule::expr => expr_opt = Some(build_expr(piece)),
                                    Rule::action_block => {
                                        collect_actions_from_block(piece, &mut actions)?
                                    }
                                    _ => {}
                                }
                            }
                            let expr = expr_opt.ok_or_else(|| {
                                FrontendError::ParseError("Missing where expression".to_string())
                            })?;
                            where_expr = Some(expr);
                        }
                        Rule::action_clause => {
                            for a in b.into_inner() {
                                if a.as_rule() == Rule::action_block {
                                    collect_actions_from_block(a, &mut actions)?;
                                }
                            }
                        }
                        Rule::mode_clause => {
                            let m = b
                                .into_inner()
                                .last()
                                .ok_or_else(|| {
                                    FrontendError::ParseError("Missing mode value".to_string())
                                })?
                                .as_str()
                                .to_string();
                            mode = Some(m);
                        }
                        _ => {}
                    }
                }
            }
            _ => {}
        }
    }

    Ok(RuleDef {
        name,
        on_model,
        patterns,
        where_expr,
        actions,
        mode,
    })
}

fn collect_actions_from_block(
    action_block: pest::iterators::Pair<Rule>,
    out: &mut Vec<ActionStmt>,
) -> Result<(), FrontendError> {
    for stmt in action_block.into_inner() {
        match stmt.as_rule() {
            Rule::action_stmt
            | Rule::let_stmt
            | Rule::nbnudge_stmt
            | Rule::soft_update_stmt
            | Rule::delete_stmt
            | Rule::suppress_stmt => out.push(build_action_stmt(stmt)?),
            _ => {}
        }
    }
    Ok(())
}

fn build_pattern_item(pair: pest::iterators::Pair<Rule>) -> Result<PatternItem, FrontendError> {
    let mut it = pair.into_inner();
    let src_var = it
        .next()
        .ok_or_else(|| FrontendError::ParseError("Missing source variable in pattern".to_string()))?
        .as_str()
        .to_string();
    let src_label = it
        .next()
        .ok_or_else(|| FrontendError::ParseError("Missing source label in pattern".to_string()))?
        .as_str()
        .to_string();
    let edge_var = it
        .next()
        .ok_or_else(|| FrontendError::ParseError("Missing edge variable in pattern".to_string()))?
        .as_str()
        .to_string();
    let edge_ty = it
        .next()
        .ok_or_else(|| FrontendError::ParseError("Missing edge type in pattern".to_string()))?
        .as_str()
        .to_string();
    let dst_var = it
        .next()
        .ok_or_else(|| {
            FrontendError::ParseError("Missing destination variable in pattern".to_string())
        })?
        .as_str()
        .to_string();
    let dst_label = it
        .next()
        .ok_or_else(|| {
            FrontendError::ParseError("Missing destination label in pattern".to_string())
        })?
        .as_str()
        .to_string();
    Ok(PatternItem {
        src: NodePattern {
            var: src_var,
            label: src_label,
        },
        edge: EdgePattern {
            var: edge_var,
            ty: edge_ty,
        },
        dst: NodePattern {
            var: dst_var,
            label: dst_label,
        },
    })
}

fn build_flow(pair: pest::iterators::Pair<Rule>, _source: &str) -> Result<FlowDef, FrontendError> {
    let mut name = String::new();
    let mut on_model = String::new();
    let mut graphs = Vec::new();
    let mut metrics = Vec::new();
    let mut exports = Vec::new();
    let mut metric_exports = Vec::new();
    let mut metric_imports = Vec::new();

    for p in pair.into_inner() {
        match p.as_rule() {
            Rule::ident => {
                if name.is_empty() {
                    name = p.as_str().to_string();
                } else if on_model.is_empty() {
                    on_model = p.as_str().to_string();
                }
            }
            Rule::flow_body => {
                for b in p.into_inner() {
                    match b.as_rule() {
                        Rule::graph_stmt => graphs.push(build_graph_stmt(b)?),
                        Rule::metric_stmt => metrics.push(build_metric_stmt(b)?),
                        Rule::export_stmt => exports.push(build_export_stmt(b)?),
                        Rule::metric_export_stmt => {
                            metric_exports.push(build_metric_export_stmt(b)?)
                        }
                        Rule::metric_import_stmt => {
                            metric_imports.push(build_metric_import_stmt(b)?)
                        }
                        _ => {}
                    }
                }
            }
            _ => {}
        }
    }
    Ok(FlowDef {
        name,
        on_model,
        graphs,
        metrics,
        exports,
        metric_exports,
        metric_imports,
    })
}

fn build_action_stmt(pair: pest::iterators::Pair<Rule>) -> Result<ActionStmt, FrontendError> {
    let inner = if pair.as_rule() == Rule::action_stmt {
        pair.into_inner()
            .next()
            .ok_or_else(|| FrontendError::ParseError("Empty action statement".to_string()))?
    } else {
        pair
    };
    match inner.as_rule() {
        Rule::let_stmt => {
            let mut name: Option<String> = None;
            let mut expr: Option<ExprAst> = None;
            for p in inner.into_inner() {
                match p.as_rule() {
                    Rule::ident if name.is_none() => name = Some(p.as_str().to_string()),
                    Rule::expr => expr = Some(build_expr(p)),
                    _ => {}
                }
            }
            Ok(ActionStmt::Let {
                name: name.ok_or_else(|| {
                    FrontendError::ParseError("Missing variable name in let statement".to_string())
                })?,
                expr: expr.ok_or_else(|| {
                    FrontendError::ParseError("Missing expression in let statement".to_string())
                })?,
            })
        }
        Rule::nbnudge_stmt => {
            // non_bayesian_nudge node.attr to expr [variance=...]
            let mut node_var: Option<String> = None;
            let mut attr: Option<String> = None;
            let mut expr: Option<ExprAst> = None;
            // Optional variance_clause
            let mut variance: Option<VarianceSpec> = None;
            for p in inner.into_inner() {
                match p.as_rule() {
                    Rule::node_attr => {
                        let mut na_it = p.into_inner();
                        node_var = Some(
                            na_it
                                .next()
                                .ok_or_else(|| {
                                    FrontendError::ParseError(
                                        "Missing node variable in non_bayesian_nudge".to_string(),
                                    )
                                })?
                                .as_str()
                                .to_string(),
                        );
                        attr = Some(
                            na_it
                                .next()
                                .ok_or_else(|| {
                                    FrontendError::ParseError(
                                        "Missing attribute in non_bayesian_nudge".to_string(),
                                    )
                                })?
                                .as_str()
                                .to_string(),
                        );
                    }
                    Rule::expr => expr = Some(build_expr(p)),
                    Rule::variance_clause => {
                        let mut vc = p.into_inner();
                        let kind = vc.next().ok_or_else(|| {
                            FrontendError::ParseError("Missing variance kind".to_string())
                        })?;
                        match kind.as_rule() {
                            Rule::KW_preserve => variance = Some(VarianceSpec::Preserve),
                            Rule::increase_clause => {
                                let mut factor: Option<f64> = None;
                                for ip in kind.into_inner() {
                                    if ip.as_str() == "factor" {
                                        // next should be '=' then number in grammar, but inner gives only number
                                    } else if ip.as_rule() == Rule::number {
                                        factor = Some(ip.as_str().parse::<f64>().map_err(|e| {
                                            FrontendError::ParseError(e.to_string())
                                        })?);
                                    }
                                }
                                variance = Some(VarianceSpec::Increase { factor });
                            }
                            Rule::decrease_clause => {
                                let mut factor: Option<f64> = None;
                                for ip in kind.into_inner() {
                                    if ip.as_rule() == Rule::number {
                                        factor = Some(ip.as_str().parse::<f64>().map_err(|e| {
                                            FrontendError::ParseError(e.to_string())
                                        })?);
                                    }
                                }
                                variance = Some(VarianceSpec::Decrease { factor });
                            }
                            _ => {}
                        }
                    }
                    _ => {}
                }
            }
            Ok(ActionStmt::NonBayesianNudge {
                node_var: node_var.ok_or_else(|| {
                    FrontendError::ParseError(
                        "Missing node attribute in non_bayesian_nudge".to_string(),
                    )
                })?,
                attr: attr.ok_or_else(|| {
                    FrontendError::ParseError(
                        "Missing node attribute in non_bayesian_nudge".to_string(),
                    )
                })?,
                expr: expr.ok_or_else(|| {
                    FrontendError::ParseError(
                        "Missing expression in non_bayesian_nudge".to_string(),
                    )
                })?,
                variance,
            })
        }
        Rule::soft_update_stmt => {
            // node.attr ~= expr [precision=..., count=...] with optional parenthesized form
            let mut node_var: Option<String> = None;
            let mut attr: Option<String> = None;
            let mut expr: Option<ExprAst> = None;
            let mut precision: Option<f64> = None;
            let mut count: Option<f64> = None;

            let mut parse_soft_arg =
                |arg_pair: pest::iterators::Pair<Rule>| -> Result<(), FrontendError> {
                    let compact: String = arg_pair
                        .as_str()
                        .chars()
                        .filter(|c| !c.is_whitespace())
                        .collect();
                    if let Some(v) = compact.strip_prefix("precision=") {
                        precision = Some(v.parse::<f64>().map_err(|e| {
                            FrontendError::ParseError(format!(
                                "Invalid precision value '{}': {}",
                                v, e
                            ))
                        })?);
                    } else if let Some(v) = compact.strip_prefix("count=") {
                        count = Some(v.parse::<f64>().map_err(|e| {
                            FrontendError::ParseError(format!("Invalid count value '{}': {}", v, e))
                        })?);
                    } else {
                        return Err(FrontendError::ParseError(format!(
                            "Unknown soft update argument '{}'",
                            arg_pair.as_str()
                        )));
                    }
                    Ok(())
                };

            for p in inner.into_inner() {
                match p.as_rule() {
                    Rule::node_attr => {
                        let mut na_it = p.into_inner();
                        node_var = Some(
                            na_it
                                .next()
                                .ok_or_else(|| {
                                    FrontendError::ParseError(
                                        "Missing node var in soft update".to_string(),
                                    )
                                })?
                                .as_str()
                                .to_string(),
                        );
                        attr = Some(
                            na_it
                                .next()
                                .ok_or_else(|| {
                                    FrontendError::ParseError(
                                        "Missing attr in soft update".to_string(),
                                    )
                                })?
                                .as_str()
                                .to_string(),
                        );
                    }
                    Rule::expr => expr = Some(build_expr(p)),
                    Rule::soft_args => {
                        for a in p.into_inner() {
                            if a.as_rule() == Rule::soft_arg {
                                parse_soft_arg(a)?;
                            }
                        }
                    }
                    Rule::soft_arg => parse_soft_arg(p)?,
                    _ => {}
                }
            }
            Ok(ActionStmt::SoftUpdate {
                node_var: node_var.ok_or_else(|| {
                    FrontendError::ParseError("Missing node attr in soft update".to_string())
                })?,
                attr: attr.ok_or_else(|| {
                    FrontendError::ParseError("Missing node attr in soft update".to_string())
                })?,
                expr: expr.ok_or_else(|| {
                    FrontendError::ParseError("Missing expression in soft update".to_string())
                })?,
                precision,
                count,
            })
        }
        Rule::delete_stmt => {
            let mut edge_var: Option<String> = None;
            let mut confidence: Option<String> = None;

            for p in inner.into_inner() {
                match p.as_rule() {
                    Rule::ident => {
                        if edge_var.is_none() {
                            edge_var = Some(p.as_str().to_string());
                        } else if confidence.is_none() {
                            // Fallback for flattened parse trees.
                            confidence = Some(p.as_str().to_string());
                        }
                    }
                    Rule::delete_args => {
                        for a in p.into_inner() {
                            if a.as_rule() == Rule::delete_arg {
                                let value = a
                                    .into_inner()
                                    .find(|x| x.as_rule() == Rule::ident)
                                    .ok_or_else(|| {
                                        FrontendError::ParseError(
                                            "Missing confidence value in delete()".to_string(),
                                        )
                                    })?;
                                confidence = Some(value.as_str().to_string());
                            }
                        }
                    }
                    Rule::delete_arg => {
                        let value = p
                            .into_inner()
                            .find(|x| x.as_rule() == Rule::ident)
                            .ok_or_else(|| {
                                FrontendError::ParseError(
                                    "Missing confidence value in delete".to_string(),
                                )
                            })?;
                        confidence = Some(value.as_str().to_string());
                    }
                    _ => {}
                }
            }
            Ok(ActionStmt::DeleteEdge {
                edge_var: edge_var.ok_or_else(|| {
                    FrontendError::ParseError("Missing edge var in delete".to_string())
                })?,
                confidence,
            })
        }
        Rule::suppress_stmt => {
            let mut edge_var: Option<String> = None;
            let mut weight: Option<f64> = None;

            for p in inner.into_inner() {
                match p.as_rule() {
                    Rule::ident => {
                        if edge_var.is_none() {
                            edge_var = Some(p.as_str().to_string());
                        }
                    }
                    Rule::suppress_args => {
                        for a in p.into_inner() {
                            if a.as_rule() == Rule::suppress_arg {
                                let value = a
                                    .into_inner()
                                    .find(|x| x.as_rule() == Rule::number)
                                    .ok_or_else(|| {
                                        FrontendError::ParseError(
                                            "Missing weight value in suppress()".to_string(),
                                        )
                                    })?;
                                weight = Some(
                                    value
                                        .as_str()
                                        .parse::<f64>()
                                        .map_err(|e| FrontendError::ParseError(e.to_string()))?,
                                );
                            }
                        }
                    }
                    Rule::suppress_arg => {
                        let value = p
                            .into_inner()
                            .find(|x| x.as_rule() == Rule::number)
                            .ok_or_else(|| {
                                FrontendError::ParseError(
                                    "Missing weight value in suppress".to_string(),
                                )
                            })?;
                        weight = Some(
                            value
                                .as_str()
                                .parse::<f64>()
                                .map_err(|e| FrontendError::ParseError(e.to_string()))?,
                        );
                    }
                    Rule::number => {
                        if weight.is_none() {
                            weight = Some(
                                p.as_str()
                                    .parse::<f64>()
                                    .map_err(|e| FrontendError::ParseError(e.to_string()))?,
                            );
                        }
                    }
                    _ => {}
                }
            }
            Ok(ActionStmt::SuppressEdge {
                edge_var: edge_var.ok_or_else(|| {
                    FrontendError::ParseError("Missing edge var in suppress".to_string())
                })?,
                weight,
            })
        }
        _ => Err(FrontendError::ParseError(format!(
            "Unknown action statement: {:?}",
            inner.as_rule()
        ))),
    }
}

fn build_graph_stmt(pair: pest::iterators::Pair<Rule>) -> Result<GraphDef, FrontendError> {
    let mut name = String::new();
    let mut expr_opt: Option<GraphExpr> = None;

    for p in pair.into_inner() {
        match p.as_rule() {
            Rule::ident if name.is_empty() => {
                name = p.as_str().to_string();
            }
            Rule::from_evidence_expr => {
                let mut inner = p.into_inner();
                let ev = extract_ident(&mut inner, "Missing evidence name in from_evidence")?;
                expr_opt = Some(GraphExpr::FromEvidence { evidence: ev });
            }
            Rule::from_graph_expr => {
                let mut inner = p.into_inner();
                let alias = extract_string(&mut inner, "Missing graph alias in from_graph")?;
                expr_opt = Some(GraphExpr::FromGraph { alias });
            }
            Rule::pipeline_expr => {
                expr_opt = Some(build_pipeline_expr(p)?);
            }
            Rule::graph_expr => {
                if let Some(inner) = p.into_inner().next() {
                    match inner.as_rule() {
                        Rule::from_evidence_expr => {
                            let mut ii = inner.into_inner();
                            let ev =
                                extract_ident(&mut ii, "Missing evidence name in from_evidence")?;
                            expr_opt = Some(GraphExpr::FromEvidence { evidence: ev });
                        }
                        Rule::from_graph_expr => {
                            let mut ii = inner.into_inner();
                            let alias =
                                extract_string(&mut ii, "Missing graph alias in from_graph")?;
                            expr_opt = Some(GraphExpr::FromGraph { alias });
                        }
                        Rule::pipeline_expr => {
                            expr_opt = Some(build_pipeline_expr(inner)?);
                        }
                        _ => {}
                    }
                }
            }
            _ => {}
        }
    }

    expr_opt
        .ok_or_else(|| FrontendError::ParseError("Missing graph expression".to_string()))
        .map(|expr| GraphDef { name, expr })
}

/// Build a pipeline expression from a Pest pair.
fn build_pipeline_expr(pair: pest::iterators::Pair<Rule>) -> Result<GraphExpr, FrontendError> {
    let mut pit = pair.into_inner();
    let start = pit
        .next()
        .ok_or_else(|| FrontendError::ParseError("Missing start graph in pipeline".to_string()))?
        .as_str()
        .to_string();
    let mut transforms = Vec::new();
    for t in pit {
        match t.as_rule() {
            Rule::pipe_op => {}
            Rule::apply_rule_tr
            | Rule::apply_ruleset_tr
            | Rule::snapshot_tr
            | Rule::prune_edges_tr => {
                transforms.push(build_transform(t)?);
            }
            _ => {}
        }
    }
    Ok(GraphExpr::Pipeline { start, transforms })
}

fn build_transform(pair: pest::iterators::Pair<Rule>) -> Result<Transform, FrontendError> {
    match pair.as_rule() {
        Rule::apply_rule_tr => {
            let name = pair
                .into_inner()
                .find(|p| p.as_rule() == Rule::ident)
                .ok_or_else(|| {
                    FrontendError::ParseError("Missing rule name in apply_rule".to_string())
                })?
                .as_str()
                .to_string();
            Ok(Transform::ApplyRule { rule: name })
        }
        Rule::apply_ruleset_tr => {
            let mut rules = Vec::new();
            for p in pair.into_inner() {
                if p.as_rule() == Rule::ident {
                    rules.push(p.as_str().to_string());
                }
            }
            if rules.is_empty() {
                return Err(FrontendError::ParseError(
                    "apply_ruleset must have at least one rule".to_string(),
                ));
            }
            Ok(Transform::ApplyRuleset { rules })
        }
        Rule::snapshot_tr => {
            let name = pair
                .into_inner()
                .find(|p| p.as_rule() == Rule::string)
                .ok_or_else(|| {
                    FrontendError::ParseError("Missing snapshot name in snapshot".to_string())
                })?
                .as_str();
            // Remove quotes from string
            let name = name.trim_matches('"').to_string();
            Ok(Transform::Snapshot { name })
        }
        Rule::prune_edges_tr => {
            let mut it = pair.into_inner();
            let edge_type = it
                .find(|p| p.as_rule() == Rule::ident)
                .ok_or_else(|| {
                    FrontendError::ParseError("Missing edge type in prune_edges".to_string())
                })?
                .as_str()
                .to_string();
            let pred_pair = it.find(|p| p.as_rule() == Rule::expr).ok_or_else(|| {
                FrontendError::ParseError("Missing predicate in prune_edges".to_string())
            })?;
            let predicate = build_expr(pred_pair);
            Ok(Transform::PruneEdges {
                edge_type,
                predicate,
            })
        }
        _ => Err(FrontendError::ParseError(format!(
            "Unknown transform: {:?}",
            pair.as_rule()
        ))),
    }
}

fn build_metric_stmt(pair: pest::iterators::Pair<Rule>) -> Result<MetricDef, FrontendError> {
    let mut it = pair.into_inner();
    let name = it
        .find(|p| p.as_rule() == Rule::ident)
        .ok_or_else(|| FrontendError::ParseError("Missing metric name".to_string()))?
        .as_str()
        .to_string();
    // Accept either expr or metric_builder_expr
    let mut expr_opt: Option<ExprAst> = None;
    for p in it {
        match p.as_rule() {
            Rule::expr => expr_opt = Some(build_expr(p)),
            Rule::metric_builder_expr => expr_opt = Some(build_metric_builder_expr(p)?),
            _ => {}
        }
    }
    let expr = expr_opt
        .ok_or_else(|| FrontendError::ParseError("Missing metric expression".to_string()))?;
    Ok(MetricDef { name, expr })
}

/// Desugars a metric builder pipeline into existing metric calls.
/// nodes(Label) |> where(expr)? |> sum(by=expr) | count() | avg(by=expr)
fn build_metric_builder_expr(pair: pest::iterators::Pair<Rule>) -> Result<ExprAst, FrontendError> {
    let mut label: Option<String> = None;
    let mut where_expr: Option<ExprAst> = None;
    let mut order_by: Option<ExprAst> = None;
    enum Agg {
        Sum(Box<ExprAst>),
        Count,
        Avg(Box<ExprAst>),
        Fold {
            init: Box<ExprAst>,
            step: Box<ExprAst>,
            order_by: Option<Box<ExprAst>>,
        },
    }
    let mut agg: Option<Agg> = None;

    // Traverse builder_start and steps
    let mut it = pair.into_inner();
    // First should be builder_start
    if let Some(start) = it.next() {
        if start.as_rule() != Rule::builder_start {
            return Err(FrontendError::ParseError(
                "Invalid metric builder start".to_string(),
            ));
        }
        let mut si = start.into_inner();
        // KW_nodes '(' label ')'
        // extract label
        label = Some(
            si.find(|p| p.as_rule() == Rule::label)
                .ok_or_else(|| FrontendError::ParseError("Missing label in nodes()".to_string()))?
                .as_str()
                .to_string(),
        );
    }
    // Remaining: pipe_op ~ builder_step ...
    for step in it {
        if step.as_rule() == Rule::pipe_op {
            continue;
        }
        // Unwrap builder_step alternation if needed
        let real = if step.as_rule() == Rule::builder_step {
            step.into_inner()
                .next()
                .ok_or_else(|| FrontendError::ParseError("Empty builder_step".to_string()))?
        } else {
            step
        };
        match real.as_rule() {
            Rule::builder_where => {
                let expr_pair = real
                    .into_inner()
                    .find(|p| p.as_rule() == Rule::expr)
                    .ok_or_else(|| {
                        FrontendError::ParseError("Missing expr in where()".to_string())
                    })?;
                where_expr = Some(build_expr(expr_pair));
            }
            Rule::builder_order => {
                let expr_pair = real
                    .into_inner()
                    .find(|p| p.as_rule() == Rule::expr)
                    .ok_or_else(|| {
                        FrontendError::ParseError("Missing expr in order_by()".to_string())
                    })?;
                order_by = Some(build_expr(expr_pair));
            }
            Rule::builder_sum => {
                let expr_pair = real
                    .into_inner()
                    .find(|p| p.as_rule() == Rule::expr)
                    .ok_or_else(|| {
                        FrontendError::ParseError("Missing by=expr in sum()".to_string())
                    })?;
                agg = Some(Agg::Sum(Box::new(build_expr(expr_pair))));
            }
            Rule::builder_count => {
                agg = Some(Agg::Count);
            }
            Rule::builder_avg => {
                let expr_pair = real
                    .into_inner()
                    .find(|p| p.as_rule() == Rule::expr)
                    .ok_or_else(|| {
                        FrontendError::ParseError("Missing by=expr in avg()".to_string())
                    })?;
                agg = Some(Agg::Avg(Box::new(build_expr(expr_pair))));
            }
            Rule::builder_fold => {
                // fold(init=expr, step=expr, order_by=expr)
                let mut init_opt: Option<ExprAst> = None;
                let mut step_opt: Option<ExprAst> = None;
                let mut ord_opt: Option<ExprAst> = None;
                let mut parse_fold_arg =
                    |arg: pest::iterators::Pair<Rule>| -> Result<(), FrontendError> {
                        let exprp = arg
                            .clone()
                            .into_inner()
                            .find(|p| p.as_rule() == Rule::expr)
                            .ok_or_else(|| {
                                FrontendError::ParseError("Missing fold arg expr".to_string())
                            })?;
                        let e = build_expr(exprp);
                        let compact: String = arg
                            .as_str()
                            .chars()
                            .filter(|c| !c.is_whitespace())
                            .collect();
                        if compact.starts_with("init=") {
                            init_opt = Some(e);
                        } else if compact.starts_with("step=") {
                            step_opt = Some(e);
                        } else if compact.starts_with("order_by=") {
                            ord_opt = Some(e);
                        }
                        Ok(())
                    };

                for a in real.into_inner() {
                    match a.as_rule() {
                        Rule::fold_args => {
                            for fa in a.into_inner() {
                                if fa.as_rule() == Rule::fold_arg {
                                    parse_fold_arg(fa)?;
                                }
                            }
                        }
                        Rule::fold_arg => parse_fold_arg(a)?,
                        _ => {}
                    }
                }
                let init = init_opt.ok_or_else(|| {
                    FrontendError::ParseError("Missing init in fold()".to_string())
                })?;
                let step_e = step_opt.ok_or_else(|| {
                    FrontendError::ParseError("Missing step in fold()".to_string())
                })?;
                agg = Some(Agg::Fold {
                    init: Box::new(init),
                    step: Box::new(step_e),
                    order_by: ord_opt.or(order_by.clone()).map(Box::new),
                });
            }
            _ => {}
        }
    }

    if order_by.is_some() && !matches!(agg, Some(Agg::Fold { .. })) {
        return Err(FrontendError::ParseError(
            "order_by() is only valid with fold() in metric builder pipelines".to_string(),
        ));
    }

    let label =
        label.ok_or_else(|| FrontendError::ParseError("Missing nodes() label".to_string()))?;
    // Build underlying calls
    let label_arg = CallArg::Named {
        name: "label".into(),
        value: ExprAst::Var(label),
    };
    let where_arg = where_expr.as_ref().map(|w| CallArg::Named {
        name: "where".into(),
        value: w.clone(),
    });
    match agg.ok_or_else(|| {
        FrontendError::ParseError("Missing terminal aggregate in metric builder".to_string())
    })? {
        Agg::Sum(by) => {
            let mut args = vec![label_arg];
            if let Some(w) = where_arg {
                args.push(w);
            }
            args.push(CallArg::Named {
                name: "contrib".into(),
                value: *by,
            });
            Ok(ExprAst::Call {
                name: "sum_nodes".into(),
                args,
            })
        }
        Agg::Count => {
            let mut args = vec![label_arg];
            if let Some(w) = where_arg {
                args.push(w);
            }
            Ok(ExprAst::Call {
                name: "count_nodes".into(),
                args,
            })
        }
        Agg::Avg(by) => {
            // avg = sum_nodes(... by)/count_nodes(...)
            let mut sum_args = vec![label_arg.clone()];
            if let Some(w) = where_arg.clone() {
                sum_args.push(w);
            }
            sum_args.push(CallArg::Named {
                name: "contrib".into(),
                value: *by,
            });
            let sum_call = ExprAst::Call {
                name: "sum_nodes".into(),
                args: sum_args,
            };
            let mut cnt_args = vec![label_arg];
            if let Some(w) = where_arg {
                cnt_args.push(w);
            }
            let cnt_call = ExprAst::Call {
                name: "count_nodes".into(),
                args: cnt_args,
            };
            Ok(ExprAst::Binary {
                op: BinaryOp::Div,
                left: Box::new(sum_call),
                right: Box::new(cnt_call),
            })
        }
        Agg::Fold {
            init,
            step,
            order_by: ord,
        } => {
            let mut args = vec![label_arg];
            if let Some(w) = where_arg {
                args.push(w);
            }
            if let Some(ob) = ord {
                args.push(CallArg::Named {
                    name: "order_by".into(),
                    value: *ob,
                });
            }
            args.push(CallArg::Named {
                name: "init".into(),
                value: *init,
            });
            args.push(CallArg::Named {
                name: "step".into(),
                value: *step,
            });
            Ok(ExprAst::Call {
                name: "fold_nodes".into(),
                args,
            })
        }
    }
}

fn build_export_stmt(pair: pest::iterators::Pair<Rule>) -> Result<ExportDef, FrontendError> {
    let mut it = pair.into_inner();
    let graph = extract_ident(&mut it, "Missing graph name in export")?;
    let alias = extract_string(&mut it, "Missing alias string in export")?;
    Ok(ExportDef { graph, alias })
}

fn build_metric_export_stmt(
    pair: pest::iterators::Pair<Rule>,
) -> Result<MetricExportDef, FrontendError> {
    let mut it = pair.into_inner();
    let metric = extract_ident(&mut it, "Missing metric name in export_metric")?;
    let alias = extract_string(&mut it, "Missing alias string in export_metric")?;
    Ok(MetricExportDef { metric, alias })
}

fn build_metric_import_stmt(
    pair: pest::iterators::Pair<Rule>,
) -> Result<MetricImportDef, FrontendError> {
    let mut it = pair.into_inner();
    // The first ident is the source alias, the second is the local name
    let source_alias = it
        .find(|p| p.as_rule() == Rule::ident)
        .ok_or_else(|| {
            FrontendError::ParseError("Missing source alias in import_metric".to_string())
        })?
        .as_str()
        .to_string();
    // find the next ident after KW_as; easiest is to take the last ident
    let local_name = it
        .filter(|p| p.as_rule() == Rule::ident)
        .last()
        .ok_or_else(|| {
            FrontendError::ParseError("Missing local name in import_metric".to_string())
        })?
        .as_str()
        .to_string();
    Ok(MetricImportDef {
        source_alias,
        local_name,
    })
}

fn build_expr(pair: pest::iterators::Pair<Rule>) -> ExprAst {
    // Build expression; on error, fall back to Number(0.0) to avoid panics during parsing
    build_expr_result(pair).unwrap_or(ExprAst::Number(0.0))
}

fn build_expr_result(pair: pest::iterators::Pair<Rule>) -> Result<ExprAst, FrontendError> {
    match pair.as_rule() {
        Rule::expr => Ok(build_expr(pair.into_inner().next().unwrap())),
        Rule::or_expr | Rule::and_expr | Rule::cmp_expr | Rule::add_expr | Rule::mul_expr => {
            let mut it = pair.into_inner();
            let mut node = build_expr(it.next().unwrap());
            while let Some(op_or_rhs) = it.next() {
                match op_or_rhs.as_rule() {
                    Rule::op_add | Rule::op_mul | Rule::op_cmp | Rule::op_and | Rule::op_or => {
                        let rhs = build_expr(it.next().unwrap());
                        let op = match op_or_rhs.as_str() {
                            "+" => BinaryOp::Add,
                            "-" => BinaryOp::Sub,
                            "*" => BinaryOp::Mul,
                            "/" => BinaryOp::Div,
                            "==" => BinaryOp::Eq,
                            "!=" => BinaryOp::Ne,
                            "<" => BinaryOp::Lt,
                            "<=" => BinaryOp::Le,
                            ">" => BinaryOp::Gt,
                            ">=" => BinaryOp::Ge,
                            "and" => BinaryOp::And,
                            "or" => BinaryOp::Or,
                            _ => {
                                // Grammar should only produce valid operators
                                unreachable!(
                                    "unexpected binary operator: {:?}",
                                    op_or_rhs.as_rule()
                                )
                            }
                        };
                        node = ExprAst::Binary {
                            op,
                            left: Box::new(node),
                            right: Box::new(rhs),
                        };
                    }
                    _ => {
                        // Should not happen
                    }
                }
            }
            Ok(node)
        }
        Rule::unary_expr => {
            let mut it = pair.into_inner();
            let first = it.next().unwrap();
            match first.as_rule() {
                Rule::op_unary | Rule::KW_not | Rule::kw_not_tok => {
                    let op = match first.as_str() {
                        "-" => UnaryOp::Neg,
                        "not" => UnaryOp::Not,
                        _ => {
                            // Grammar should only produce valid unary operators
                            unreachable!("unexpected unary operator: '{}'", first.as_str())
                        }
                    };
                    let expr = build_expr(it.next().unwrap());
                    Ok(ExprAst::Unary {
                        op,
                        expr: Box::new(expr),
                    })
                }
                _ => Ok(build_expr(first)),
            }
        }
        Rule::postfix => {
            let mut it = pair.into_inner();
            let mut node = build_expr(it.next().unwrap());
            for next in it {
                match next.as_rule() {
                    Rule::field_access => {
                        let field = next.into_inner().next().unwrap().as_str().to_string();
                        node = ExprAst::Field {
                            target: Box::new(node),
                            field,
                        };
                    }
                    Rule::call_suffix => {
                        // Convert current node to a Call, with name from identifier
                        let cit = next.into_inner();
                        let mut args = Vec::new();
                        for p in cit {
                            match p.as_rule() {
                                Rule::arg => {
                                    let inner = p.into_inner().next().unwrap();
                                    if inner.as_rule() == Rule::named_arg {
                                        let mut ni = inner.into_inner();
                                        let name = ni.next().unwrap().as_str().to_string();
                                        let value = build_expr(ni.next().unwrap());
                                        args.push(CallArg::Named { name, value });
                                    } else {
                                        let val = build_expr(inner);
                                        args.push(CallArg::Positional(val));
                                    }
                                }
                                Rule::named_arg => {
                                    let mut ni = p.into_inner();
                                    let name = ni.next().unwrap().as_str().to_string();
                                    let value = build_expr(ni.next().unwrap());
                                    args.push(CallArg::Named { name, value });
                                }
                                Rule::expr
                                | Rule::ident
                                | Rule::number
                                | Rule::boolean
                                | Rule::paren_expr
                                | Rule::e_bracket
                                | Rule::postfix => {
                                    let val = build_expr(p);
                                    args.push(CallArg::Positional(val));
                                }
                                _ => { /* skip commas/others */ }
                            }
                        }
                        match node {
                            ExprAst::Var(name) => node = ExprAst::Call { name, args },
                            _ => {
                                node = ExprAst::Call {
                                    name: String::new(),
                                    args,
                                }
                            }
                        }
                    }
                    _ => {}
                }
            }
            Ok(node)
        }
        Rule::primary => build_expr_result(pair.into_inner().next().unwrap()),
        Rule::exists_expr => {
            let mut it = pair.into_inner();
            // Skip KW_exists, get pattern_item
            let pattern_pair = it
                .find(|p| p.as_rule() == Rule::pattern_item)
                .ok_or_else(|| {
                    FrontendError::ParseError("Missing pattern in exists expression".to_string())
                })?;
            let pattern = build_pattern_item(pattern_pair)?;
            // Get optional where clause
            let where_expr = it
                .find(|p| p.as_rule() == Rule::expr)
                .map(|p| Box::new(build_expr(p)));
            Ok(ExprAst::Exists {
                pattern,
                where_expr,
                negated: false,
            })
        }
        Rule::not_exists_expr => {
            let mut it = pair.into_inner();
            // Skip KW_not and KW_exists, get pattern_item
            let pattern_pair = it
                .find(|p| p.as_rule() == Rule::pattern_item)
                .ok_or_else(|| {
                    FrontendError::ParseError(
                        "Missing pattern in not exists expression".to_string(),
                    )
                })?;
            let pattern = build_pattern_item(pattern_pair)?;
            // Get optional where clause
            let where_expr = it
                .find(|p| p.as_rule() == Rule::expr)
                .map(|p| Box::new(build_expr(p)));
            Ok(ExprAst::Exists {
                pattern,
                where_expr,
                negated: true,
            })
        }
        Rule::paren_expr => Ok(build_expr(pair.into_inner().next().unwrap())),
        Rule::e_bracket => {
            // E[Var.field] → Call("E", [Field(Var, field)])
            let mut it = pair.into_inner();
            let var = it.next().unwrap().as_str().to_string();
            let field = it.next().unwrap().as_str().to_string();
            let inner = ExprAst::Field {
                target: Box::new(ExprAst::Var(var)),
                field,
            };
            Ok(ExprAst::Call {
                name: "E".into(),
                args: vec![CallArg::Positional(inner)],
            })
        }
        Rule::ident => Ok(ExprAst::Var(pair.as_str().to_string())),
        Rule::number => {
            let num_str = pair.as_str();
            let value = num_str.parse::<f64>().unwrap_or_else(|_| {
                eprintln!("Warning: Failed to parse number '{}', using 0.0", num_str);
                0.0
            });
            Ok(ExprAst::Number(value))
        }
        Rule::boolean => Ok(match pair.as_str() {
            "true" => ExprAst::Bool(true),
            _ => ExprAst::Bool(false),
        }),
        Rule::kw_true_tok | Rule::KW_true => Ok(ExprAst::Bool(true)),
        Rule::kw_false_tok | Rule::KW_false => Ok(ExprAst::Bool(false)),
        Rule::metric_builder_expr => build_metric_builder_expr(pair),
        _ => {
            // This should never happen if the grammar is correct
            // If it does, it indicates a grammar/Pest parser mismatch
            unreachable!("unexpected expression rule: {:?}", pair.as_rule())
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn parse_minimal_schema() {
        let src = "schema S { node N {} edge E {} }";
        let result = parse_program(src);

        assert!(result.is_ok());
        let ast = result.unwrap();
        assert_eq!(ast.schemas.len(), 1);
        assert_eq!(ast.schemas[0].name, "S");
        assert_eq!(ast.schemas[0].nodes.len(), 1);
        assert_eq!(ast.schemas[0].edges.len(), 1);
    }

    #[test]
    fn parse_schema_with_attributes() {
        let src = r#"
            schema TestSchema {
                node Person {
                    age: Real
                    score: Real
                }
                edge Knows {}
            }
        "#;

        let result = parse_program(src).unwrap();
        let schema = &result.schemas[0];

        assert_eq!(schema.name, "TestSchema");
        assert_eq!(schema.nodes[0].name, "Person");
        assert_eq!(schema.nodes[0].attrs.len(), 2);
        assert_eq!(schema.nodes[0].attrs[0].name, "age");
        assert_eq!(schema.nodes[0].attrs[0].ty, "Real");
        assert_eq!(schema.nodes[0].attrs[1].name, "score");
    }

    #[test]
    fn parse_belief_model() {
        let src = r#"
            schema S { node N {} edge E {} }
            belief_model M on S {}
        "#;

        let result = parse_program(src).unwrap();
        assert_eq!(result.belief_models.len(), 1);
        assert_eq!(result.belief_models[0].name, "M");
        assert_eq!(result.belief_models[0].on_schema, "S");
    }

    #[test]
    fn parse_rule_with_pattern() {
        let src = r#"
            schema S { node N {} edge E {} }
            belief_model M on S {}
            rule R on M {
                pattern (A:N)-[e:E]->(B:N)
            }
        "#;

        let result = parse_program(src).unwrap();
        assert_eq!(result.rules.len(), 1);

        let rule = &result.rules[0];
        assert_eq!(rule.name, "R");
        assert_eq!(rule.patterns.len(), 1);

        let pattern = &rule.patterns[0];
        assert_eq!(pattern.src.var, "A");
        assert_eq!(pattern.src.label, "N");
        assert_eq!(pattern.edge.var, "e");
        assert_eq!(pattern.edge.ty, "E");
        assert_eq!(pattern.dst.var, "B");
        assert_eq!(pattern.dst.label, "N");
    }

    #[test]
    fn parse_rule_with_where_clause() {
        let src = r#"
            schema S { node N {} edge E {} }
            belief_model M on S {}
            rule R on M {
                pattern (A:N)-[e:E]->(B:N)
                where prob(e) >= 0.5
                action {}
            }
        "#;

        let result = parse_program(src).unwrap();
        let rule = &result.rules[0];

        assert!(rule.where_expr.is_some());
    }

    #[test]
    fn parse_rule_with_actions() {
        let src = r#"
            schema S { node N { x: Real } edge E {} }
            belief_model M on S {}
            rule R on M {
                pattern (A:N)-[e:E]->(B:N)
                action {
                    non_bayesian_nudge A.x to 10 variance=preserve
                    delete e confidence=high
                }
            }
        "#;

        let result = parse_program(src).unwrap();
        let rule = &result.rules[0];

        assert_eq!(rule.name, "R");
        assert_eq!(rule.patterns.len(), 1);
        assert_eq!(rule.actions.len(), 2);
        assert!(matches!(
            rule.actions[0],
            ActionStmt::NonBayesianNudge { .. }
        ));
        assert!(matches!(rule.actions[1], ActionStmt::DeleteEdge { .. }));
    }

    #[test]
    fn parse_rule_rejects_legacy_action_keywords() {
        let src = r#"
            schema S { node N { x: Real } edge E {} }
            belief_model M on S {}
            rule R on M {
                pattern (A:N)-[e:E]->(B:N)
                action {
                    set_expectation A.x = 10
                    force_absent e
                }
            }
        "#;
        let result = parse_program(src);
        assert!(result.is_err(), "legacy action keywords should be rejected");
    }

    #[test]
    fn parse_flow_with_graph_and_metric() {
        let src = r#"
            schema S { node N {} edge E {} }
            belief_model M on S {}
            evidence Ev on M {}
            flow F on M {
                graph g = from_evidence Ev
                metric m = count_nodes(label=N)
            }
        "#;

        let result = parse_program(src).unwrap();
        assert_eq!(result.flows.len(), 1);

        let flow = &result.flows[0];
        assert_eq!(flow.name, "F");
        assert_eq!(flow.graphs.len(), 1);
        assert_eq!(flow.metrics.len(), 1);
        assert_eq!(flow.metrics[0].name, "m");
    }

    #[test]
    fn parse_flow_with_metric_export_and_import() {
        let src = r#"
            schema S { node N {} edge E {} }
            belief_model M on S {}
            evidence Ev on M {}
            flow Producer on M {
                graph g = from_evidence Ev
                metric base = 100.0
                export_metric base as "scenario_budget"
            }
            flow Consumer on M {
                import_metric scenario_budget as budget
                graph g = from_evidence Ev
                metric fin = fold_nodes(label=N, init=budget, step=value)
            }
        "#;

        let result = parse_program(src).unwrap();
        assert_eq!(result.flows.len(), 2);
        let prod = &result.flows[0];
        assert_eq!(prod.metric_exports.len(), 1);
        assert_eq!(prod.metric_exports[0].alias, "scenario_budget");
        let cons = &result.flows[1];
        assert_eq!(cons.metric_imports.len(), 1);
        assert_eq!(cons.metric_imports[0].local_name, "budget");
    }

    #[test]
    fn parse_multiple_patterns_in_rule() {
        let src = r#"
            schema S { node N {} edge E {} }
            belief_model M on S {}
            rule R on M {
                pattern (A:N)-[e1:E]->(B:N)
                pattern (B:N)-[e2:E]->(C:N)
            }
        "#;

        let result = parse_program(src).unwrap();
        let rule = &result.rules[0];
        assert_eq!(rule.patterns.len(), 2);
    }

    #[test]
    fn parse_expression_with_arithmetic() {
        let src = r#"
            schema S { node N { x: Real } edge E {} }
            belief_model M on S {}
            rule R on M {
                pattern (A:N)-[e:E]->(B:N)
                where E[A.x] + 5 >= 10
            }
        "#;

        let result = parse_program(src).unwrap();
        let rule = &result.rules[0];

        assert!(rule.where_expr.is_some());
    }

    #[test]
    fn parse_expression_with_function_calls() {
        let src = r#"
            schema S { node N {} edge E {} }
            belief_model M on S {}
            rule R on M {
                pattern (A:N)-[e:E]->(B:N)
                where prob(e) >= 0.5 and degree(A, min_prob=0.3) > 1
            }
        "#;

        let result = parse_program(src).unwrap();
        let rule = &result.rules[0];

        assert!(rule.where_expr.is_some());
    }

    #[test]
    fn parse_let_statement_in_actions() {
        let src = r#"
            schema S { node N { x: Real } edge E {} }
            belief_model M on S {}
            rule R on M {
                pattern (A:N)-[e:E]->(B:N)
                action {
                    let temp = E[A.x] / 2
                    non_bayesian_nudge B.x to temp variance=preserve
                }
            }
        "#;

        let result = parse_program(src).unwrap();
        let rule = &result.rules[0];

        // Just verify parsing succeeded
        assert_eq!(rule.name, "R");
        assert_eq!(rule.patterns.len(), 1);
    }

    #[test]
    fn parse_empty_program() {
        let src = "";
        let result = parse_program(src);

        // Empty program should parse successfully with empty AST
        if let Ok(ast) = result {
            assert_eq!(ast.schemas.len(), 0);
            assert_eq!(ast.rules.len(), 0);
        }
    }

    #[test]
    fn parse_program_with_all_components() {
        let src = r#"
            schema S {
                node N { x: Real }
                edge E {}
            }
            belief_model M on S {}
            evidence Ev on M {}
            rule R on M {
                pattern (A:N)-[e:E]->(B:N)
            }
            flow F on M {
                graph g = from_evidence Ev
            }
        "#;

        let result = parse_program(src).unwrap();

        assert_eq!(result.schemas.len(), 1);
        assert_eq!(result.belief_models.len(), 1);
        assert_eq!(result.evidences.len(), 1);
        assert_eq!(result.rules.len(), 1);
        assert_eq!(result.flows.len(), 1);
    }

    #[test]
    fn parse_invalid_syntax_returns_error() {
        let src = "this is not valid syntax !!@#";
        let result = parse_program(src);

        assert!(result.is_err());
    }

    #[test]
    fn parse_unclosed_brace_returns_error() {
        let src = "schema S { node N { }";
        let result = parse_program(src);

        assert!(result.is_err());
    }

    #[test]
    fn parse_numbers_and_booleans() {
        let src = r#"
            schema S { node N { x: Real } edge E {} }
            belief_model M on S {}
            rule R on M {
                pattern (A:N)-[e:E]->(B:N)
                where 42 >= 10 and true
            }
        "#;

        let result = parse_program(src).unwrap();
        let rule = &result.rules[0];

        assert!(rule.where_expr.is_some());
    }

    #[test]
    fn parse_nested_expressions() {
        let src = r#"
            schema S { node N { x: Real } edge E {} }
            belief_model M on S {}
            rule R on M {
                pattern (A:N)-[e:E]->(B:N)
                where ((E[A.x] + 1) * 2) >= 10
            }
        "#;

        let result = parse_program(src).unwrap();
        let rule = &result.rules[0];

        assert!(rule.where_expr.is_some());
    }

    #[test]
    fn parse_graph_pipeline() {
        let src = r#"
            schema S { node N {} edge E {} }
            belief_model M on S {}
            rule R on M { pattern (A:N)-[e:E]->(B:N) }
            flow F on M {
                graph base = from_evidence Ev
                graph filtered = base |> apply_rule R |> prune_edges E where prob(edge) < 0.1
            }
        "#;

        let result = parse_program(src).unwrap();
        let flow = &result.flows[0];

        assert_eq!(flow.graphs.len(), 2);

        // Check that the second graph has a pipeline
        match &flow.graphs[1].expr {
            GraphExpr::Pipeline { .. } => {
                // Success
            }
            _ => panic!("Expected pipeline"),
        }
    }

    #[test]
    fn parse_metric_builder_sum_and_count() {
        let src = r#"
            schema S { node Person { value: Real } edge E {} }
            belief_model M on S {}
            flow F on M {
                metric total = nodes(Person) |> where(E[node.value] > 1.0) |> sum(by=E[node.value])
                metric cnt = nodes(Person) |> count()
            }
        "#;
        let ast = parse_program(src).unwrap();
        let flow = &ast.flows[0];
        assert_eq!(flow.metrics.len(), 2);
        // Sum should desugar to sum_nodes(label=Person, where=..., contrib=...)
        if let ExprAst::Call { name, args } = &flow.metrics[0].expr {
            assert_eq!(name, "sum_nodes");
            // Expect named label and contrib
            let mut has_label = false;
            let mut has_contrib = false;
            for a in args {
                match a {
                    CallArg::Named { name, .. } if name == "label" => has_label = true,
                    CallArg::Named { name, .. } if name == "contrib" => has_contrib = true,
                    _ => {}
                }
            }
            assert!(has_label && has_contrib);
        } else {
            panic!("expected sum_nodes call");
        }
        // Count should desugar to count_nodes(label=Person)
        if let ExprAst::Call { name, .. } = &flow.metrics[1].expr {
            assert_eq!(name, "count_nodes");
        } else {
            panic!("expected count_nodes call");
        }
    }

    #[test]
    fn parse_rule_sugar_and_for_sugar() {
        let src = r#"
            schema S { node Person { x: Real } edge REL {} }
            belief_model M on S {}
            rule R1 on M {
                (A:Person)-[e:REL]->(B:Person) => { let v = 1 }
            }
            rule R2 on M {
                for (A:Person) => { let z = 0 }
            }
        "#;
        let ast = parse_program(src).unwrap();
        assert_eq!(ast.rules.len(), 2);
        let r1 = &ast.rules[0];
        assert_eq!(r1.patterns.len(), 1);
        assert_eq!(r1.actions.len(), 1);
        let r2 = &ast.rules[1];
        assert_eq!(r2.patterns.len(), 1);
        assert_eq!(r2.actions.len(), 1);
        // Placeholder edge type for for() sugar
        assert_eq!(r2.patterns[0].edge.ty, "__FOR_NODE__");
        assert_eq!(r2.patterns[0].src.var, "A");
        assert_eq!(r2.patterns[0].src.label, "Person");
    }

    #[test]
    fn parse_soft_update_inline_delete_suppress_inline_and_parenthesized() {
        let src = r#"
            schema S { node Person { score: Real } edge REL {} }
            belief_model M on S {}
            rule Inline on M {
                for (P:Person) where E[P.score] < 1.0 => {
                    P.score ~= 1.0 precision=0.2 count=3
                    delete __for_dummy confidence=high
                    suppress __for_dummy weight=10
                }
            }
            rule Paren on M {
                pattern (A:Person)-[ab:REL]->(B:Person)
                action {
                    A.score ~= 0.5 (precision=0.1, count=2)
                    delete ab (confidence=low)
                    suppress ab (weight=5)
                }
            }
        "#;
        let ast = parse_program(src).unwrap();
        assert_eq!(ast.rules.len(), 2);

        let inline = &ast.rules[0];
        assert_eq!(inline.actions.len(), 3);
        assert!(matches!(
            inline.actions[0],
            ActionStmt::SoftUpdate {
                precision: Some(_),
                count: Some(_),
                ..
            }
        ));
        assert!(matches!(
            inline.actions[1],
            ActionStmt::DeleteEdge {
                confidence: Some(_),
                ..
            }
        ));
        assert!(matches!(
            inline.actions[2],
            ActionStmt::SuppressEdge {
                weight: Some(_),
                ..
            }
        ));

        let paren = &ast.rules[1];
        assert_eq!(paren.actions.len(), 3);
        assert!(matches!(
            paren.actions[0],
            ActionStmt::SoftUpdate {
                precision: Some(_),
                count: Some(_),
                ..
            }
        ));
        assert!(matches!(
            paren.actions[1],
            ActionStmt::DeleteEdge {
                confidence: Some(_),
                ..
            }
        ));
        assert!(matches!(
            paren.actions[2],
            ActionStmt::SuppressEdge {
                weight: Some(_),
                ..
            }
        ));
    }

    #[test]
    fn parse_metric_builder_inside_expression() {
        let src = r#"
            schema S { node Person { value: Real } edge E {} }
            belief_model M on S {}
            flow F on M {
                metric weighted_avg = (nodes(Person) |> sum(by=E[node.value])) / (nodes(Person) |> count())
            }
        "#;
        let ast = parse_program(src).unwrap();
        let flow = &ast.flows[0];
        assert_eq!(flow.metrics.len(), 1);
        assert!(matches!(
            flow.metrics[0].expr,
            ExprAst::Binary {
                op: BinaryOp::Div,
                ..
            }
        ));
    }

    #[test]
    fn parse_grouped_evidence_and_choose() {
        let src = r#"
            schema S { node N { a: Real } edge E {} }
            belief_model M on S {}
            evidence Ev on M {
                N { "A" { a: 1.0 (precision=10.0) }, "B" { a: 2.0 } }
                E(N -> N) { "A" -> "B"; "B" -/> "A" }
                choose edge E(N["A"], N["B"]) ; unchoose edge E(N["B"], N["A"]) ;
            }
        "#;
        let ast = parse_program(src).unwrap();
        assert_eq!(ast.evidences.len(), 1);
        let obs = &ast.evidences[0].observations;
        assert!(obs.len() >= 4);
    }

    #[test]
    fn parse_evidence_declaration() {
        let src = r#"
            schema S { node N {} edge E {} }
            belief_model M on S {}
            evidence Ev on M {}
        "#;

        let result = parse_program(src).unwrap();
        assert_eq!(result.evidences.len(), 1);
        assert_eq!(result.evidences[0].name, "Ev");
    }

    #[test]
    fn parse_rule_for_each_mode() {
        let src = r#"
            schema S { node N {} edge E {} }
            belief_model M on S {}
            rule R on M {
                pattern (A:N)-[e:E]->(B:N)
                action {}
                mode: for_each
            }
        "#;

        let result = parse_program(src).unwrap();
        let rule = &result.rules[0];

        assert_eq!(rule.mode, Some("for_each".to_string()));
    }

    #[test]
    fn parse_categorical_posterior_with_uniform_prior_and_pseudo_count() {
        let src = r#"
            schema PacketRouting { node Router { latency: Real } edge ROUTES_TO {} }
            belief_model RoutingBeliefs on PacketRouting {
              node Router { latency ~ GaussianPosterior(prior_mean=50.0, prior_precision=0.1) }
              edge ROUTES_TO {
                exist ~ CategoricalPosterior(group_by="source", prior=uniform, pseudo_count=1.0)
              }
            }
        "#;

        let ast = parse_program(src).unwrap();
        assert_eq!(ast.belief_models.len(), 1);
        let bm = &ast.belief_models[0];
        assert_eq!(bm.edges.len(), 1);
        let exist = &bm.edges[0].exist;
        match exist {
            PosteriorType::Categorical {
                group_by,
                prior,
                categories,
            } => {
                assert_eq!(group_by, "source");
                match prior {
                    CategoricalPrior::Uniform { pseudo_count } => {
                        assert!((*pseudo_count - 1.0).abs() < 1e-9)
                    }
                    _ => panic!("expected Uniform prior"),
                }
                assert!(categories.is_none());
            }
            _ => panic!("expected Categorical posterior"),
        }
    }

    #[test]
    fn parse_categorical_posterior_with_explicit_prior_and_categories() {
        let src = r#"
            schema PacketRouting { node Router { latency: Real } edge ROUTES_TO {} }
            belief_model RoutingBeliefs on PacketRouting {
              node Router { latency ~ GaussianPosterior(prior_mean=50.0, prior_precision=0.1) }
              edge ROUTES_TO {
                exist ~ CategoricalPosterior(group_by="source", prior=[1.0, 2.0, 3.0], categories=["R2","R3","R6"])
              }
            }
        "#;

        let ast = parse_program(src).unwrap();
        let bm = &ast.belief_models[0];
        let exist = &bm.edges[0].exist;
        match exist {
            PosteriorType::Categorical {
                group_by,
                prior,
                categories,
            } => {
                assert_eq!(group_by, "source");
                match prior {
                    CategoricalPrior::Explicit { concentrations } => {
                        assert_eq!(concentrations, &vec![1.0, 2.0, 3.0]);
                    }
                    _ => panic!("expected Explicit prior"),
                }
                assert_eq!(
                    categories.as_ref().unwrap(),
                    &vec!["R2".to_string(), "R3".to_string(), "R6".to_string()]
                );
            }
            _ => panic!("expected Categorical posterior"),
        }
    }

    #[test]
    fn parse_categorical_group_by_identifier() {
        let src = r#"
            schema PacketRouting { node Router { latency: Real } edge ROUTES_TO {} }
            belief_model RoutingBeliefs on PacketRouting {
              node Router { latency ~ Gaussian(mean=50.0, precision=0.1) }
              edge ROUTES_TO {
                exist ~ Categorical(group_by=source, prior=uniform, pseudo_count=1.0)
              }
            }
        "#;

        let ast = parse_program(src).unwrap();
        let bm = &ast.belief_models[0];
        match &bm.edges[0].exist {
            PosteriorType::Categorical { group_by, .. } => assert_eq!(group_by, "source"),
            _ => panic!("expected Categorical posterior"),
        }
    }
}
