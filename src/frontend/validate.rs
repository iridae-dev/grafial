//! # Semantic Validation
//!
//! This module performs semantic validation on the parsed AST, checking:
//!
//! - **Rule validation**: Ensures `prob()` is only used on edge variables,
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
//! - `E[node.attr]` requires `node` to be a node variable from pattern
//! - `degree(node)` requires `node` to be a node variable
//! - Metric functions require specific named arguments (label, contrib, etc.)
//!
//! Validation is separate from parsing to provide clear, actionable error messages.

use std::collections::HashSet;

use crate::engine::errors::ExecError;
use crate::frontend::ast::*;

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
/// * `Err(ExecError::ValidationError)` - Semantic error with description
///
/// # Example
///
/// ```rust,ignore
/// use grafial::frontend::{parser, validate};
///
/// let ast = parser::parse_program(source)?;
/// validate::validate_program(&ast)?; // Ensure semantic correctness
/// ```
pub fn validate_program(ast: &ProgramAst) -> Result<(), ExecError> {
    // Validate belief models (competing edge groups, posterior parameters)
    for model in &ast.belief_models {
        validate_belief_model(model, &ast.schemas)?;
    }
    
    // Validate evidence (evidence mode matches posterior type)
    for evidence in &ast.evidences {
        validate_evidence(evidence, &ast.belief_models)?;
    }
    
    for rule in &ast.rules {
        validate_rule(rule)?;
    }
    for flow in &ast.flows {
        validate_flow(flow)?;
    }
    for flow in &ast.flows {
        for m in &flow.metrics {
            validate_metric(&m.expr)?;
        }
    }
    Ok(())
}

fn validate_belief_model(model: &BeliefModel, schemas: &[Schema]) -> Result<(), ExecError> {
    // Find the schema this model operates on
    let schema = schemas
        .iter()
        .find(|s| s.name == model.on_schema)
        .ok_or_else(|| ExecError::ValidationError(
            format!("Belief model '{}' references unknown schema '{}'", model.name, model.on_schema)
        ))?;
    
    // Track edge types to ensure uniqueness (independent vs competing)
    let mut edge_types_seen = std::collections::HashMap::new();
    
    // Validate edge declarations
    for edge_decl in &model.edges {
        // Check if edge type exists in schema
        if !schema.edges.iter().any(|e| e.name == edge_decl.edge_type) {
            return Err(ExecError::ValidationError(
                format!("Edge type '{}' not found in schema '{}'", edge_decl.edge_type, model.on_schema)
            ));
        }
        
        // Validate posterior type
        match &edge_decl.exist {
            PosteriorType::Categorical { group_by, prior, categories: _ } => {
                // Check group_by is valid
                if group_by != "source" && group_by != "destination" {
                    return Err(ExecError::ValidationError(
                        format!("CategoricalPosterior 'group_by' must be 'source' or 'destination', got '{}'", group_by)
                    ));
                }
                
                // Check prior specification
                match prior {
                    CategoricalPrior::Uniform { pseudo_count } => {
                        if *pseudo_count <= 0.0 {
                            return Err(ExecError::ValidationError(
                                format!("CategoricalPosterior 'pseudo_count' must be > 0, got {}", pseudo_count)
                            ));
                        }
                    }
                    CategoricalPrior::Explicit { concentrations } => {
                        if concentrations.is_empty() {
                            return Err(ExecError::ValidationError(
                                "CategoricalPosterior explicit prior array cannot be empty".into()
                            ));
                        }
                        for (i, &alpha) in concentrations.iter().enumerate() {
                            if alpha <= 0.0 {
                                return Err(ExecError::ValidationError(
                                    format!("CategoricalPosterior prior array element {} must be > 0, got {}", i, alpha)
                                ));
                            }
                        }
                    }
                }
                
                // Note: categories validation against schema would require knowing all possible destinations
                // This is deferred to runtime when edges are actually created
            }
            PosteriorType::Bernoulli { params } => {
                // Validate Bernoulli parameters if needed
                let prior = params.iter().find(|(name, _)| name == "prior");
                let pseudo_count = params.iter().find(|(name, _)| name == "pseudo_count");
                
                if let Some((_, val)) = prior {
                    if !(0.0..=1.0).contains(val) {
                        return Err(ExecError::ValidationError(
                            format!("BernoulliPosterior 'prior' must be in [0, 1], got {}", val)
                        ));
                    }
                }
                
                if let Some((_, val)) = pseudo_count {
                    if *val <= 0.0 {
                        return Err(ExecError::ValidationError(
                            format!("BernoulliPosterior 'pseudo_count' must be > 0, got {}", val)
                        ));
                    }
                }
            }
            PosteriorType::Gaussian { .. } => {
                // Gaussian validation would go here if needed
            }
        }
        
        // Check for duplicate edge type declarations
        if let Some(prev_type) = edge_types_seen.get(&edge_decl.edge_type) {
            return Err(ExecError::ValidationError(
                format!(
                    "Edge type '{}' declared multiple times in belief model '{}' (previous: {:?})",
                    edge_decl.edge_type, model.name, prev_type
                )
            ));
        }
        edge_types_seen.insert(edge_decl.edge_type.clone(), &edge_decl.exist);
    }
    
    Ok(())
}

fn validate_evidence(evidence: &EvidenceDef, belief_models: &[BeliefModel]) -> Result<(), ExecError> {
    // Find the belief model this evidence applies to
    let model = belief_models
        .iter()
        .find(|m| m.name == evidence.on_model)
        .ok_or_else(|| ExecError::ValidationError(
            format!("Evidence '{}' references unknown belief model '{}'", evidence.name, evidence.on_model)
        ))?;
    
    // Validate each observation statement
    for obs in &evidence.observations {
        match obs {
            ObserveStmt::Edge { edge_type, mode, .. } => {
                // Find the edge declaration in the belief model
                let edge_decl = model.edges.iter().find(|e| e.edge_type == *edge_type);
                
                if let Some(edge_decl) = edge_decl {
                    // Check evidence mode matches posterior type
                    match (&edge_decl.exist, mode) {
                        (PosteriorType::Categorical { .. }, EvidenceMode::Present | EvidenceMode::Absent) => {
                            return Err(ExecError::ValidationError(
                                format!(
                                    "Edge '{}' has competing posterior; use 'chosen', 'unchosen', or 'forced_choice', not '{:?}'",
                                    edge_type, mode
                                )
                            ));
                        }
                        (PosteriorType::Bernoulli { .. } | PosteriorType::Gaussian { .. }, 
                         EvidenceMode::Chosen | EvidenceMode::Unchosen | EvidenceMode::ForcedChoice) => {
                            return Err(ExecError::ValidationError(
                                format!(
                                    "Edge '{}' is independent; use 'present' or 'absent', not '{:?}'",
                                    edge_type, mode
                                )
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

fn validate_rule(rule: &RuleDef) -> Result<(), ExecError> {
    let mut node_vars = HashSet::new();
    let mut edge_vars = HashSet::new();
    for p in &rule.patterns {
        node_vars.insert(p.src.var.clone());
        node_vars.insert(p.dst.var.clone());
        edge_vars.insert(p.edge.var.clone());
    }
    if let Some(expr) = &rule.where_expr {
        validate_expr_in_rule(expr, &node_vars, &edge_vars)?;
    }
    Ok(())
}

fn validate_flow(flow: &FlowDef) -> Result<(), ExecError> {
    for g in &flow.graphs {
        if let GraphExpr::Pipeline { transforms, .. } = &g.expr {
            for t in transforms {
                match t {
                    Transform::ApplyRule { .. } => {}
                    Transform::ApplyRuleset { .. } => {}
                    Transform::Snapshot { .. } => {}
                    Transform::PruneEdges { predicate, .. } => {
                        validate_prune_predicate(predicate)?;
                    }
                }
            }
        }
    }
    Ok(())
}

fn validate_metric(expr: &ExprAst) -> Result<(), ExecError> {
    // Only shape checks for known metric calls; others pass through.
    if let ExprAst::Call { name, args } = expr {
        match name.as_str() {
            "sum_nodes" => {
                let (pos, named) = split_args(args);
                let has_label = named.contains("label") || pos.len() >= 1;
                let has_contrib = named.contains("contrib") || pos.len() >= 3; // label, where, contrib
                if !has_label {
                    return Err(ExecError::ValidationError("sum_nodes: missing 'label' argument".into()));
                }
                if !has_contrib {
                    return Err(ExecError::ValidationError("sum_nodes: missing 'contrib' argument".into()));
                }
            }
            "fold_nodes" => {
                let (pos, named) = split_args(args);
                let has_label = named.contains("label") || pos.len() >= 1;
                let has_init = named.contains("init") || pos.len() >= 4;
                let has_step = named.contains("step") || pos.len() >= 5;
                if !has_label {
                    return Err(ExecError::ValidationError("fold_nodes: missing 'label' argument".into()));
                }
                if !has_init {
                    return Err(ExecError::ValidationError("fold_nodes: missing 'init' argument".into()));
                }
                if !has_step {
                    return Err(ExecError::ValidationError("fold_nodes: missing 'step' argument".into()));
                }
            }
            "count_nodes" => {
                let (pos, named) = split_args(args);
                let has_label = named.contains("label") || pos.len() >= 1;
                if !has_label {
                    return Err(ExecError::ValidationError("count_nodes: missing 'label' argument".into()));
                }
            }
            "avg_degree" => {
                let (pos, named) = split_args(args);
                let has_label = pos.len() >= 1 || named.contains("label");
                let has_edge_type = pos.len() >= 2 || named.contains("edge_type");
                if !has_label {
                    return Err(ExecError::ValidationError("avg_degree: missing 'label' argument".into()));
                }
                if !has_edge_type {
                    return Err(ExecError::ValidationError("avg_degree: missing 'edge_type' argument".into()));
                }
            }
            _ => {}
        }
    }
    Ok(())
}

fn split_args(args: &Vec<CallArg>) -> (Vec<&ExprAst>, ArgNames<'_>) {
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
    fn contains(&self, name: &str) -> bool { self.map.contains_key(name) }
}

fn validate_prune_predicate(expr: &ExprAst) -> Result<(), ExecError> {
    // Ensure any prob(...) refers to 'edge'
    walk_expr(expr, &mut |e| {
        if let ExprAst::Call { name, args } = e {
            if name == "prob" {
                let (pos, named) = split_args(args);
                if !named.map.is_empty() || pos.len() != 1 {
                    return Err(ExecError::ValidationError("prob(): expected single positional argument".into()));
                }
                match pos[0] {
                    ExprAst::Var(v) if v == "edge" => Ok(()),
                    _ => Err(ExecError::ValidationError("prob(): argument must be 'edge' in prune_edges predicate".into())),
                }
            } else { Ok(()) }
        } else { Ok(()) }
    })
}

fn validate_expr_in_rule(expr: &ExprAst, node_vars: &HashSet<String>, edge_vars: &HashSet<String>) -> Result<(), ExecError> {
    walk_expr(expr, &mut |e| match e {
        ExprAst::Call { name, args } if name == "prob" => {
            let (pos, named) = split_args(args);
            if !named.map.is_empty() || pos.len() != 1 {
                return Err(ExecError::ValidationError("prob(): expected single positional argument".into()));
            }
            match pos[0] {
                ExprAst::Var(v) if edge_vars.contains(v) => Ok(()),
                _ => Err(ExecError::ValidationError("prob(): argument must be an edge variable".into())),
            }
        }
        ExprAst::Call { name, args } if name == "E" => {
            let (pos, named) = split_args(args);
            if !named.map.is_empty() || pos.len() != 1 {
                return Err(ExecError::ValidationError("E[]: expected single positional argument".into()));
            }
            match pos[0] {
                ExprAst::Field { target, field: _ } => match &**target {
                    ExprAst::Var(v) if node_vars.contains(v) => Ok(()),
                    _ => Err(ExecError::ValidationError("E[]: must be E[NodeVar.attr]".into())),
                },
                _ => Err(ExecError::ValidationError("E[]: must be a field access expression".into())),
            }
        }
        _ => Ok(()),
    })
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
/// * `Err(ExecError)` - Visitor function returned an error
fn walk_expr<F>(expr: &ExprAst, f: &mut F) -> Result<(), ExecError>
where
    F: FnMut(&ExprAst) -> Result<(), ExecError>,
{
    f(expr)?;
    match expr {
        ExprAst::Unary { expr, .. } => walk_expr(expr, f)?,
        ExprAst::Binary { left, right, .. } => { walk_expr(left, f)?; walk_expr(right, f)?; }
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
                CallArg::Named { name: "label".into(), value: ExprAst::Var("Person".into()) },
                CallArg::Named { name: "contrib".into(), value: ExprAst::Number(1.0) },
            ],
        };

        assert!(validate_metric(&expr).is_ok());
    }

    #[test]
    fn validate_sum_nodes_missing_label_fails() {
        let expr = ExprAst::Call {
            name: "sum_nodes".into(),
            args: vec![
                CallArg::Named { name: "contrib".into(), value: ExprAst::Number(1.0) },
            ],
        };

        let result = validate_metric(&expr);
        assert!(result.is_err());
        assert!(result.unwrap_err().to_string().contains("label"));
    }

    #[test]
    fn validate_sum_nodes_missing_contrib_fails() {
        let expr = ExprAst::Call {
            name: "sum_nodes".into(),
            args: vec![
                CallArg::Named { name: "label".into(), value: ExprAst::Var("Person".into()) },
            ],
        };

        let result = validate_metric(&expr);
        assert!(result.is_err());
        assert!(result.unwrap_err().to_string().contains("contrib"));
    }

    #[test]
    fn validate_count_nodes_with_label() {
        let expr = ExprAst::Call {
            name: "count_nodes".into(),
            args: vec![
                CallArg::Named { name: "label".into(), value: ExprAst::Var("Person".into()) },
            ],
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
                CallArg::Named { name: "label".into(), value: ExprAst::Var("Person".into()) },
                CallArg::Named { name: "edge_type".into(), value: ExprAst::Var("KNOWS".into()) },
            ],
        };

        assert!(validate_metric(&expr).is_ok());
    }

    #[test]
    fn validate_avg_degree_missing_label_fails() {
        let expr = ExprAst::Call {
            name: "avg_degree".into(),
            args: vec![
                CallArg::Named { name: "edge_type".into(), value: ExprAst::Var("KNOWS".into()) },
            ],
        };

        let result = validate_metric(&expr);
        assert!(result.is_err());
        assert!(result.unwrap_err().to_string().contains("label"));
    }

    #[test]
    fn validate_avg_degree_missing_edge_type_fails() {
        let expr = ExprAst::Call {
            name: "avg_degree".into(),
            args: vec![
                CallArg::Named { name: "label".into(), value: ExprAst::Var("Person".into()) },
            ],
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
                CallArg::Named { name: "label".into(), value: ExprAst::Var("Person".into()) },
                CallArg::Named { name: "init".into(), value: ExprAst::Number(0.0) },
                CallArg::Named { name: "step".into(), value: ExprAst::Var("acc".into()) },
            ],
        };

        assert!(validate_metric(&expr).is_ok());
    }

    #[test]
    fn validate_fold_nodes_missing_init_fails() {
        let expr = ExprAst::Call {
            name: "fold_nodes".into(),
            args: vec![
                CallArg::Named { name: "label".into(), value: ExprAst::Var("Person".into()) },
                CallArg::Named { name: "step".into(), value: ExprAst::Var("acc".into()) },
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
                src: NodePattern { var: "A".into(), label: "N".into() },
                edge: EdgePattern { var: "e".into(), ty: "E".into() },
                dst: NodePattern { var: "B".into(), label: "N".into() },
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
                src: NodePattern { var: "A".into(), label: "N".into() },
                edge: EdgePattern { var: "e".into(), ty: "E".into() },
                dst: NodePattern { var: "B".into(), label: "N".into() },
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
    fn validate_rule_with_e_on_valid_field() {
        let rule = RuleDef {
            name: "R".into(),
            on_model: "M".into(),
            mode: Some("for_each".into()),
            patterns: vec![PatternItem {
                src: NodePattern { var: "A".into(), label: "N".into() },
                edge: EdgePattern { var: "e".into(), ty: "E".into() },
                dst: NodePattern { var: "B".into(), label: "N".into() },
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
    fn validate_rule_with_e_on_invalid_var_fails() {
        let rule = RuleDef {
            name: "R".into(),
            on_model: "M".into(),
            mode: Some("for_each".into()),
            patterns: vec![PatternItem {
                src: NodePattern { var: "A".into(), label: "N".into() },
                edge: EdgePattern { var: "e".into(), ty: "E".into() },
                dst: NodePattern { var: "B".into(), label: "N".into() },
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
                src: NodePattern { var: "A".into(), label: "N".into() },
                edge: EdgePattern { var: "e".into(), ty: "E".into() },
                dst: NodePattern { var: "B".into(), label: "N".into() },
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
        assert!(result.unwrap_err().to_string().contains("single positional"));
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
        }).unwrap();

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
                    return Err(ExecError::ValidationError("found bad var".into()));
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
                    src: NodePattern { var: "A".into(), label: "N".into() },
                    edge: EdgePattern { var: "e".into(), ty: "E".into() },
                    dst: NodePattern { var: "B".into(), label: "N".into() },
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
}
