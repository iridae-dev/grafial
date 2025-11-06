//! # Rule Execution Engine
//!
//! This module implements the core rule execution logic for Grafial, including:
//!
//! - **Pattern matching**: Finding subgraph matches in the belief graph
//! - **Expression evaluation**: Computing values for where clauses and action expressions
//! - **Action execution**: Applying graph updates (set_expectation, force_absent, etc.)
//!
//! ## Rule Execution Model
//!
//! Rules follow a pattern-match-execute model:
//! 1. Match patterns in the graph to find variable bindings
//! 2. Evaluate the optional `where` clause with bound variables
//! 3. Execute actions for each successful match (for_each mode) or first match
//!
//! Rules operate on immutable graphs: mutations are applied to a working copy (delta),
//! and the original graph remains unchanged. The working copy is committed at the end
//! of rule execution to produce the output graph.
//!
//! ## Expression Language
//!
//! Supports:
//! - Arithmetic: `+`, `-`, `*`, `/`
//! - Comparisons: `<`, `<=`, `>`, `>=`, `==`, `!=`
//! - Logical: `and`, `or`, `not`
//! - Special functions: `prob(edge)`, `degree(node)`, `E[node.attr]`
//!
//! ## Performance
//!
//! Uses O(1) hash lookups for nodes/edges and eager evaluation with
//! local variable binding for efficient execution.
//!
//! ## Determinism
//!
//! Pattern matching iterates edges in stable order (sorted by EdgeId) to ensure
//! deterministic rule execution. This guarantees that the same input graph and rule
//! always produce the same output, regardless of hash table iteration order.

use std::collections::HashMap;
use std::sync::Arc;

use crate::engine::errors::ExecError;
use crate::engine::expr_eval::{eval_binary_op, eval_expr_core, eval_unary_op, ExprContext};
use crate::engine::graph::{BeliefGraph, EdgeId, NodeId};
use crate::engine::query_plan::{QueryPlan, QueryPlanCache};
use crate::frontend::ast::{ActionStmt, CallArg, ExprAst, PatternItem, RuleDef};

// Phase 7: Fixpoint iteration configuration
/// Maximum iterations for fixpoint rules to prevent infinite loops.
///
/// If a fixpoint rule doesn't converge within this many iterations, execution
/// terminates with an error to prevent infinite loops.
const MAX_FIXPOINT_ITERATIONS: usize = 1000;

/// Convergence tolerance for fixpoint rules
/// If changes are smaller than this threshold, consider converged
const FIXPOINT_TOLERANCE: f64 = 1e-6;

/// Variable bindings for a single pattern match.
///
/// Maps pattern variables (from rule patterns) to concrete graph elements.
/// For example, pattern `(A:Person)-[e:KNOWS]->(B:Person)` creates bindings
/// for variables A, B (nodes) and e (edge).
#[derive(Debug, Clone)]
pub struct MatchBindings {
    /// Maps node variable names to node IDs
    pub node_vars: HashMap<String, NodeId>,
    /// Maps edge variable names to edge IDs
    pub edge_vars: HashMap<String, EdgeId>,
}

impl Default for MatchBindings {
    fn default() -> Self {
        Self {
            node_vars: HashMap::new(),
            edge_vars: HashMap::new(),
        }
    }
}

impl MatchBindings {
    /// Creates bindings with pre-allocated capacity based on expected pattern count.
    ///
    /// Estimates: 2 nodes per pattern (src + dst), 1 edge per pattern.
    /// This avoids multiple reallocations during pattern matching.
    pub fn with_capacity(patterns: &[PatternItem]) -> Self {
        let estimated_nodes = patterns.len() * 2;  // src + dst per pattern
        let estimated_edges = patterns.len();
        Self {
            node_vars: HashMap::with_capacity(estimated_nodes),
            edge_vars: HashMap::with_capacity(estimated_edges),
        }
    }
}

/// Local variable storage for rule action execution.
///
/// Tracks variables created by `let` statements within action blocks.
struct Locals(HashMap<String, f64>);

impl Locals {
    fn new() -> Self {
        Self(HashMap::new())
    }

    fn get(&self, name: &str) -> Option<f64> {
        self.0.get(name).copied()
    }

    fn set(&mut self, name: String, value: f64) {
        self.0.insert(name, value);
    }
}

/// Expression evaluation context for rule execution.
///
/// Provides variable resolution from locals, globals, and pattern bindings,
/// and implements rule-specific functions (prob, degree, E).
struct RuleExprContext<'a> {
    bindings: &'a MatchBindings,
    locals: &'a Locals,
    globals: &'a HashMap<String, f64>,
}

impl<'a> ExprContext for RuleExprContext<'a> {
    fn resolve_var(&self, name: &str) -> Option<f64> {
        // Check locals first, then globals, then node/edge variables
        self.locals.get(name)
            .or_else(|| self.globals.get(name).copied())
            .or_else(|| {
                // Node variables: convert NodeId to f64 for comparisons
                self.bindings.node_vars.get(name).map(|nid| nid.0 as f64)
            })
            .or_else(|| {
                // Edge variables: convert EdgeId to f64 for comparisons
                self.bindings.edge_vars.get(name).map(|eid| eid.0 as f64)
            })
    }

    fn eval_function(
        &self,
        name: &str,
        pos_args: &[ExprAst],
        all_args: &[CallArg],
        graph: &BeliefGraph,
    ) -> Result<f64, ExecError> {
        match name {
            // E[Var.attr] is represented as Call("E", [Field(Var(..), attr)]) by the parser
            "E" => {
                if !all_args.is_empty() && matches!(all_args[0], CallArg::Named { .. }) {
                    return Err(ExecError::Internal("E[] does not accept named arguments".into()));
                }
                if pos_args.len() != 1 {
                    return Err(ExecError::Internal("E[] expects one positional argument".into()));
                }
                match &pos_args[0] {
                    ExprAst::Field { target, field } => match &**target {
                        ExprAst::Var(vn) => {
                            let nid = *self
                                .bindings
                                .node_vars
                                .get(vn)
                                .ok_or_else(|| ExecError::Internal(format!("unknown node var '{}'", vn)))?;
                            graph.expectation(nid, field)
                        }
                        _ => Err(ExecError::Internal("E[] requires NodeVar.attr".into())),
                    },
                    _ => Err(ExecError::Internal("E[] requires a field expression".into())),
                }
            }
            "prob" => {
                if !all_args.is_empty() && matches!(all_args[0], CallArg::Named { .. }) {
                    return Err(ExecError::Internal("prob() does not accept named arguments".into()));
                }
                if pos_args.len() != 1 {
                    return Err(ExecError::Internal("prob() expects one positional argument".into()));
                }
                match &pos_args[0] {
                    ExprAst::Var(v) => {
                        let eid = *self
                            .bindings
                            .edge_vars
                            .get(v)
                            .ok_or_else(|| ExecError::Internal(format!("unknown edge var '{}'", v)))?;
                        let result = graph.prob_mean(eid)?;
                        eprintln!("  >> prob({}) = {:.6} (EdgeId {:?})", v, result, eid);
                        Ok(result)
                    }
                    _ => Err(ExecError::Internal("prob(): argument must be an edge variable".into())),
                }
            }
            "degree" => {
                if pos_args.len() < 1 {
                    return Err(ExecError::Internal("degree(): missing node argument".into()));
                }
                let mut min_prob = 0.0;
                for a in all_args {
                    if let CallArg::Named { name, value } = a {
                        if name == "min_prob" {
                            min_prob = eval_expr_core(value, graph, self)?;
                        }
                    }
                }
                match &pos_args[0] {
                    ExprAst::Var(v) => {
                        let nid = *self
                            .bindings
                            .node_vars
                            .get(v)
                            .ok_or_else(|| ExecError::Internal(format!("unknown node var '{}'", v)))?;
                        Ok(graph.degree_outgoing(nid, min_prob) as f64)
                    }
                    _ => Err(ExecError::Internal("degree(): first argument must be a node variable".into())),
                }
            }
            "winner" => {
                if pos_args.len() < 2 {
                    return Err(ExecError::Internal("winner(): requires node and edge_type arguments".into()));
                }
                let mut epsilon = 0.01;
                for a in all_args {
                    if let CallArg::Named { name, value } = a {
                        if name == "epsilon" {
                            epsilon = eval_expr_core(value, graph, self)?;
                        }
                    }
                }
                let node_var = match &pos_args[0] {
                    ExprAst::Var(v) => v,
                    _ => return Err(ExecError::Internal("winner(): first argument must be a node variable".into())),
                };
                let edge_type = match &pos_args[1] {
                    ExprAst::Var(v) => v.clone(),
                    ExprAst::Number(_) => return Err(ExecError::Internal("winner(): edge_type must be a string identifier".into())),
                    _ => return Err(ExecError::Internal("winner(): edge_type must be an identifier".into())),
                };
                
                let nid = *self
                    .bindings
                    .node_vars
                    .get(node_var)
                    .ok_or_else(|| ExecError::Internal(format!("unknown node var '{}'", node_var)))?;
                
                // Return winner destination node ID as f64, or -1.0 if None (to represent null)
                // This allows expressions like winner(A, ROUTES_TO) == B where B is a node variable
                match graph.winner(nid, &edge_type, epsilon) {
                    Some(winner_node_id) => Ok(winner_node_id.0 as f64),
                    None => Ok(-1.0), // Represent None as -1.0
                }
            }
            "entropy" => {
                if pos_args.len() < 2 {
                    return Err(ExecError::Internal("entropy(): requires node and edge_type arguments".into()));
                }
                let node_var = match &pos_args[0] {
                    ExprAst::Var(v) => v,
                    _ => return Err(ExecError::Internal("entropy(): first argument must be a node variable".into())),
                };
                let edge_type = match &pos_args[1] {
                    ExprAst::Var(v) => v.clone(),
                    _ => return Err(ExecError::Internal("entropy(): edge_type must be an identifier".into())),
                };
                
                let nid = *self
                    .bindings
                    .node_vars
                    .get(node_var)
                    .ok_or_else(|| ExecError::Internal(format!("unknown node var '{}'", node_var)))?;
                
                graph.entropy(nid, &edge_type)
            }
            other => Err(ExecError::Internal(format!("unknown function '{}'", other))),
        }
    }

    fn eval_field(&self, target: &ExprAst, _field: &str, _graph: &BeliefGraph) -> Result<f64, ExecError> {
        Err(ExecError::Internal(format!(
            "bare field access not supported; use E[Node.attr]: {:?}",
            target
        )))
    }
}

impl<'a> RuleExprContext<'a> {
    /// Evaluates an exists subquery expression.
    ///
    /// Searches for a single edge matching the pattern with compatible variable bindings.
    /// Returns 1.0 if found (or 0.0 if negated), 0.0 if not found (or 1.0 if negated).
    ///
    /// Variables in the exists pattern can reference parent rule variables - if a variable
    /// is already bound, it must match the same graph element. This enables correlated
    /// subqueries like `exists (A)-[e:KNOWS]->(B) where prob(e) > 0.5`.
    pub fn eval_exists(
        &self,
        pattern: &crate::frontend::ast::PatternItem,
        where_expr: &Option<Box<ExprAst>>,
        negated: bool,
        graph: &BeliefGraph,
    ) -> Result<f64, ExecError> {
        // Need to ensure graph has deltas applied so find_candidate_edges can see all edges
        // But we can't mutate graph, so we need find_candidate_edges to handle deltas
        // Actually, find_candidate_edges uses graph.edges() which only returns base edges
        // So we need to make find_candidate_edges delta-aware, or ensure graph has deltas applied
        let candidates = find_candidate_edges(graph, &pattern.edge.ty);

        for edge in candidates {
            if !matches_node_labels(graph, edge, pattern)? {
                continue;
            }

            // Build subquery bindings, validating compatibility with parent bindings
            // Subquery has single pattern, so estimate 2 nodes + 1 edge
            let mut subquery_bindings = MatchBindings {
                node_vars: HashMap::with_capacity(2),
                edge_vars: HashMap::with_capacity(1),
            };
            
            // Source node: must match parent binding if already bound
            if let Some(&parent_node) = self.bindings.node_vars.get(&pattern.src.var) {
                if parent_node != edge.src {
                    continue;
                }
            }
            subquery_bindings.node_vars.insert(pattern.src.var.clone(), edge.src);

            // Destination node: must match parent binding if already bound
            if let Some(&parent_node) = self.bindings.node_vars.get(&pattern.dst.var) {
                if parent_node != edge.dst {
                    continue;
                }
            }
            subquery_bindings.node_vars.insert(pattern.dst.var.clone(), edge.dst);

            // Edge variable: must match parent binding if already bound
            if let Some(&parent_edge) = self.bindings.edge_vars.get(&pattern.edge.var) {
                if parent_edge != edge.id {
                    continue;
                }
            }
            subquery_bindings.edge_vars.insert(pattern.edge.var.clone(), edge.id);

            // Merge with parent bindings for where clause evaluation (subquery can reference both)
            let mut merged_bindings = self.bindings.clone();
            merged_bindings.node_vars.extend(subquery_bindings.node_vars);
            merged_bindings.edge_vars.extend(subquery_bindings.edge_vars);

            // Evaluate where clause if present
            if let Some(w) = where_expr {
                let subquery_ctx = RuleExprContext {
                    bindings: &merged_bindings,
                    locals: self.locals,
                    globals: self.globals,
                };
                let v = eval_where_with_exists(w, graph, &subquery_ctx)?;
                if v == 0.0 {
                    continue;
                }
            }

            return Ok(if negated { 0.0 } else { 1.0 });
        }

        Ok(if negated { 1.0 } else { 0.0 })
    }
}

/// Evaluates a where clause with bound variables, returning true if it passes.
///
/// Exists subqueries require special handling (pattern matching), so we use
/// a separate evaluation path rather than the standard expression evaluator.
fn evaluate_where_clause(
    where_expr: &Option<ExprAst>,
    bindings: &MatchBindings,
    globals: &HashMap<String, f64>,
    graph: &BeliefGraph,
) -> Result<bool, ExecError> {
    let Some(expr) = where_expr else {
        return Ok(true);
    };

    let where_locals = Locals::new();
    let where_ctx = RuleExprContext {
        bindings,
        locals: &where_locals,
        globals,
    };

    // DEBUG: Print edge probabilities for diagnosis
    eprintln!("=== WHERE CLAUSE EVALUATION ===");
    eprintln!("Edge bindings: {:?}", bindings.edge_vars);
    for (var_name, edge_id) in &bindings.edge_vars {
        if let Ok(prob) = graph.prob_mean(*edge_id) {
            eprintln!("  Edge '{}' ({:?}): prob = {:.6}", var_name, edge_id, prob);
        }
    }

    let v = eval_where_with_exists(expr, graph, &where_ctx)?;
    eprintln!("Where clause result: {} (evaluated to: {})", v, v != 0.0);
    eprintln!("===============================");

    Ok(v != 0.0)
}

/// Evaluates a where clause expression, handling exists subqueries.
///
/// Exists expressions require pattern matching, so they're handled specially
/// rather than going through the standard expression evaluator.
fn eval_where_with_exists(
    expr: &ExprAst,
    graph: &BeliefGraph,
    ctx: &RuleExprContext,
) -> Result<f64, ExecError> {
    eprintln!("  [eval_where_with_exists] Evaluating: {:?}", expr);
    let result = match expr {
        ExprAst::Exists { pattern, where_expr, negated } => {
            ctx.eval_exists(pattern, where_expr, *negated, graph)
        }
        ExprAst::Binary { op, left, right } => {
            eprintln!("    Binary op: {:?}", op);
            let l = eval_where_with_exists(left, graph, ctx)?;
            eprintln!("    Left result: {}", l);
            let r = eval_where_with_exists(right, graph, ctx)?;
            eprintln!("    Right result: {}", r);
            eval_binary_op(*op, l, r)
        }
        ExprAst::Unary { op, expr } => {
            let v = eval_where_with_exists(expr, graph, ctx)?;
            Ok(eval_unary_op(*op, v))
        }
        _ => {
            // For all other expression types, use the standard evaluator
            eval_expr_core(expr, graph, ctx)
        }
    };
    eprintln!("  [eval_where_with_exists] Result: {:?}", result);
    result
}

/// Executes action statements against a belief graph.
///
/// Actions are executed in order, with local variables from `let` statements
/// available to subsequent actions.
///
/// # Action Semantics
///
/// - `Let`: Evaluates an expression and stores it in a local variable
/// - `SetExpectation`: Updates a node attribute's expected value (soft update)
///   - Adjusts mean without changing precision (variance)
/// - `ForceAbsent`: Forces an edge to be absent with high certainty
///   - Sets Beta parameters to α=1, β=1e6 (mean ≈ 0.000001)
pub fn execute_actions(
    graph: &mut BeliefGraph,
    actions: &[ActionStmt],
    bindings: &MatchBindings,
    globals: &HashMap<String, f64>,
) -> Result<(), ExecError> {
    let mut locals = Locals::new();

    for a in actions {
        // Create context for this action (reborrow locals)
        let ctx = RuleExprContext {
            bindings,
            locals: &locals,
            globals,
        };

        match a {
            ActionStmt::Let { name, expr } => {
                let v = eval_expr_core(expr, graph, &ctx)?;
                locals.set(name.clone(), v);
            }
            ActionStmt::SetExpectation { node_var, attr, expr } => {
                let nid = *bindings
                    .node_vars
                    .get(node_var)
                    .ok_or_else(|| ExecError::Internal(format!("unknown node var '{}'", node_var)))?;
                let v = eval_expr_core(expr, graph, &ctx)?;
                graph.set_expectation(nid, attr, v)?;
            }
            ActionStmt::ForceAbsent { edge_var } => {
                let eid = *bindings
                    .edge_vars
                    .get(edge_var)
                    .ok_or_else(|| ExecError::Internal(format!("unknown edge var '{}'", edge_var)))?;
                graph.force_absent(eid)?;
            }
        }
    }
    Ok(())
}

/// Computes the maximum absolute change between two graphs.
///
/// Used for fixpoint convergence detection. Compares:
/// - Node attribute means (Gaussian posteriors)
/// - Edge existence probabilities (Beta posteriors)
///
/// # Arguments
///
/// * `old` - The previous graph state
/// * `new` - The new graph state
///
/// # Returns
///
/// The maximum absolute difference across all attributes and probabilities.
/// Returns 0.0 if graphs are identical or if either graph is empty.
///
/// Used to determine if a fixpoint rule has converged. Compares posterior means
/// and probabilities between iterations to detect when changes fall below the
/// convergence threshold.
fn compute_max_change(old: &BeliefGraph, new: &BeliefGraph) -> f64 {
    let mut max_delta: f64 = 0.0;

    // Compare node attributes
    for (old_node, new_node) in old.nodes().iter().zip(new.nodes().iter()) {
        if old_node.id != new_node.id {
            // Graph structure changed, return large delta
            return f64::INFINITY;
        }
        for (attr_name, old_attr) in &old_node.attrs {
            if let Some(new_attr) = new_node.attrs.get(attr_name) {
                let delta = (old_attr.mean - new_attr.mean).abs();
                max_delta = max_delta.max(delta);
            }
        }
    }

    // Compare edge probabilities
    for (old_edge, new_edge) in old.edges().iter().zip(new.edges().iter()) {
        if old_edge.id != new_edge.id {
            // Graph structure changed, return large delta
            return f64::INFINITY;
        }
        let old_prob = old_edge.exist.mean_probability(old.competing_groups()).unwrap_or(0.0);
        let new_prob = new_edge.exist.mean_probability(new.competing_groups()).unwrap_or(0.0);
        let delta = (old_prob - new_prob).abs();
        max_delta = max_delta.max(delta);
    }

    max_delta
}

/// Executes a rule in "for_each" mode over all matching patterns.
///
/// Finds all graph patterns matching the rule, evaluates the optional `where` clause,
/// and applies actions to a working copy for each successful match.
///
/// # Current Limitations
///
/// - Supports exactly one pattern item (no multi-pattern rules yet)
///
/// Execution model:
/// 1. Find all pattern matches (deterministic order by EdgeId)
/// 2. Filter matches by where clause
/// 3. Apply actions to working copy (immutable input preserved)
/// 4. Return new graph with changes applied
pub fn run_rule_for_each(input: &BeliefGraph, rule: &RuleDef) -> Result<BeliefGraph, ExecError> {
    run_rule_for_each_with_globals(input, rule, &HashMap::new())
}

/// Executes a rule with access to global scalar variables (e.g., imported metrics).
///
/// Supports both single-pattern and multi-pattern rules. For multi-pattern rules,
/// finds all combinations of edges that match all patterns with shared variables.
/// Uses deterministic iteration over sorted EdgeIds for reproducibility.
pub fn run_rule_for_each_with_globals(
    input: &BeliefGraph,
    rule: &RuleDef,
    globals: &HashMap<String, f64>,
) -> Result<BeliefGraph, ExecError> {
    if rule.patterns.is_empty() {
        return Err(ExecError::ValidationError("rule must have at least one pattern".into()));
    }

    // Single-pattern path: optimized for the common case (no join needed)
    if rule.patterns.len() == 1 {
        let pat = &rule.patterns[0];
        // Use input graph for candidate finding (we want to iterate over original edges)
        let candidates = find_candidate_edges(input, &pat.edge.ty);

        // First pass: collect matches (read-only, no clone needed)
        let mut matches = Vec::new();
        for edge in candidates {
            if !matches_node_labels(input, edge, pat)? {
                continue;
            }

            let mut bindings = MatchBindings::with_capacity(&rule.patterns);
            bindings.node_vars.insert(pat.src.var.clone(), edge.src);
            bindings.node_vars.insert(pat.dst.var.clone(), edge.dst);
            bindings.edge_vars.insert(pat.edge.var.clone(), edge.id);

            // Evaluate where clause against input graph (original state)
            // This ensures where clause evaluation is consistent and doesn't depend on action order
            if !evaluate_where_clause(&rule.where_expr, &bindings, globals, input)? {
                continue;
            }

            matches.push(bindings);
        }

        // If no matches, return input graph (no clone needed)
        if matches.is_empty() {
            return Ok(input.clone());
        }

        // Second pass: clone only if we have matches to process
        // Clone and ensure deltas are applied so find_candidate_edges can see all edges
        let mut work = input.clone();
        work.ensure_owned();

        // Apply actions for all matches
        for bindings in matches {
            execute_actions(&mut work, &rule.actions, &bindings, globals)?;
        }

        return Ok(work);
    }

    // Multi-pattern matching: use query plan to optimize join order
    let mut plan_cache = QueryPlanCache::new();
    let plan = plan_cache.get_or_create(&rule.patterns, input);
    let ordered_patterns: Vec<PatternItem> = plan.ordered_patterns
        .iter()
        .map(|&idx| rule.patterns[idx].clone())
        .collect();
    
    // First pass: collect all matches (read-only, no clone needed)
    let mut matches = Vec::new();
    let initial_bindings = MatchBindings::with_capacity(&rule.patterns);
    find_multi_pattern_matches(input, &ordered_patterns, 0, &initial_bindings, &mut matches)?;

    // Deterministic ordering: sort by EdgeId tuples for reproducibility
    // Use unstable sort (faster) since EdgeId ordering is stable and deterministic
    matches.sort_unstable_by(|a, b| {
        // Collect and sort edge IDs for comparison
        let mut a_edges: Vec<_> = a.edge_vars.values().cloned().collect();
        let mut b_edges: Vec<_> = b.edge_vars.values().cloned().collect();
        a_edges.sort_unstable();
        b_edges.sort_unstable();
        a_edges.cmp(&b_edges)
    });

    // Filter matches by where clause (still read-only)
    eprintln!("\n=== MULTI-PATTERN RULE: {} ===", rule.name);
    eprintln!("Found {} potential matches before where clause", matches.len());

    let mut filtered_matches = Vec::new();
    for (idx, bindings) in matches.iter().enumerate() {
        eprintln!("\nMatch {}:", idx);
        eprintln!("  Nodes: {:?}", bindings.node_vars);
        eprintln!("  Edges: {:?}", bindings.edge_vars);

        if evaluate_where_clause(&rule.where_expr, bindings, globals, input)? {
            eprintln!("  -> PASSED where clause");
            filtered_matches.push(bindings.clone());
        } else {
            eprintln!("  -> FAILED where clause");
        }
    }

    // If no matches, return input graph (no clone needed)
    if filtered_matches.is_empty() {
        eprintln!("No matches passed where clause - rule will not fire");
        return Ok(input.clone());
    }

    eprintln!("\n{} matches passed where clause - applying actions", filtered_matches.len());

    // Second pass: clone only if we have matches to process
    let mut work = input.clone();
    work.ensure_owned();

    // Apply actions to working copy
    for (idx, bindings) in filtered_matches.iter().enumerate() {
        eprintln!("\nApplying actions for match {}:", idx);
        for (var, eid) in &bindings.edge_vars {
            if let Ok(prob_before) = work.prob_mean(*eid) {
                eprintln!("  Edge '{}' ({:?}) BEFORE: prob = {:.6}", var, eid, prob_before);
            }
        }

        execute_actions(&mut work, &rule.actions, bindings, globals)?;

        for (var, eid) in &bindings.edge_vars {
            if let Ok(prob_after) = work.prob_mean(*eid) {
                eprintln!("  Edge '{}' ({:?}) AFTER: prob = {:.6}", var, eid, prob_after);
            }
        }
    }

    Ok(work)
}

/// Finds candidate edges matching a pattern's edge type, sorted deterministically.
///
/// Deterministic ordering by EdgeId ensures reproducible rule execution across runs.
fn find_candidate_edges<'a>(
    graph: &'a BeliefGraph,
    edge_type: &str,
) -> Vec<&'a crate::engine::graph::EdgeData> {
    // Collect edges from both base and delta
    let mut candidates: Vec<_> = Vec::new();
    
    // First, collect all edge IDs we've seen (to avoid duplicates)
    let mut seen_ids = std::collections::HashSet::new();
    
    // Arc<str> implements PartialEq with &str, so we can compare directly using .as_ref()
    // Add edges from base graph
    for edge in graph.edges() {
        if edge.ty.as_ref() == edge_type {
            seen_ids.insert(edge.id);
            candidates.push(edge);
        }
    }
    
    // Add edges from delta (checking for duplicates)
    for change in graph.delta() {
        if let crate::engine::graph::GraphDelta::EdgeChange { id, edge } = change {
            if edge.ty.as_ref() == edge_type && !seen_ids.contains(id) {
                seen_ids.insert(*id);
                candidates.push(edge);
            }
        }
    }
    
    // Use unstable sort (faster) since EdgeId ordering is stable and deterministic
    candidates.sort_unstable_by_key(|e| e.id);
    candidates
}

/// Checks if an edge matches a pattern's node labels.
fn matches_node_labels(
    graph: &BeliefGraph,
    edge: &crate::engine::graph::EdgeData,
    pattern: &PatternItem,
) -> Result<bool, ExecError> {
    let src = graph.node(edge.src)
        .ok_or_else(|| ExecError::Internal("missing src node".into()))?;
    let dst = graph.node(edge.dst)
        .ok_or_else(|| ExecError::Internal("missing dst node".into()))?;
    Ok(src.label.as_ref() == pattern.src.label.as_str() && dst.label.as_ref() == pattern.dst.label.as_str())
}

/// Attempts to extend bindings with a pattern match, validating variable compatibility.
///
/// Returns `Some(new_bindings)` if the edge matches and all variables are compatible,
/// `None` if variables conflict (e.g., shared variable bound to different node).
///
/// Shared variables across patterns must bind to the same graph element - this
/// enforces the join semantics in multi-pattern rules.
fn try_extend_bindings(
    current_bindings: &MatchBindings,
    pattern: &PatternItem,
    edge: &crate::engine::graph::EdgeData,
) -> Option<MatchBindings> {
    let mut new_bindings = current_bindings.clone();

    // Validate source node variable compatibility
    if let Some(&existing) = current_bindings.node_vars.get(&pattern.src.var) {
        if existing != edge.src {
            return None;
        }
    }
    new_bindings.node_vars.insert(pattern.src.var.clone(), edge.src);

    // Validate destination node variable compatibility
    if let Some(&existing) = current_bindings.node_vars.get(&pattern.dst.var) {
        if existing != edge.dst {
            return None;
        }
    }
    new_bindings.node_vars.insert(pattern.dst.var.clone(), edge.dst);

    // Edge variables must be unique per pattern (no conflicts possible in well-formed rules)
    if current_bindings.edge_vars.contains_key(&pattern.edge.var) {
        return None;
    }
    new_bindings.edge_vars.insert(pattern.edge.var.clone(), edge.id);

    Some(new_bindings)
}

/// Recursively finds all matches for multi-pattern rules.
///
/// Builds up bindings pattern-by-pattern, ensuring shared variables (from previous
/// patterns) match consistently. This implements the join semantics for multi-pattern rules.
fn find_multi_pattern_matches(
    graph: &BeliefGraph,
    patterns: &[PatternItem],
    pattern_idx: usize,
    current_bindings: &MatchBindings,
    matches: &mut Vec<MatchBindings>,
) -> Result<(), ExecError> {
    if pattern_idx >= patterns.len() {
        matches.push(current_bindings.clone());
        return Ok(());
    }

    let pat = &patterns[pattern_idx];
    let candidates = find_candidate_edges(graph, &pat.edge.ty);

    for edge in candidates {
        if !matches_node_labels(graph, edge, pat)? {
            continue;
        }

        let Some(new_bindings) = try_extend_bindings(current_bindings, pat, edge) else {
            continue;
        };

        find_multi_pattern_matches(graph, patterns, pattern_idx + 1, &new_bindings, matches)?;
    }

    Ok(())
}

/// Executes a rule in "fixpoint" mode until convergence or iteration limit.
///
/// Applies the rule repeatedly until convergence (changes < FIXPOINT_TOLERANCE)
/// or MAX_FIXPOINT_ITERATIONS is reached.
///
/// Fixpoint iteration strategy:
/// - Run rule, compare output to input
/// - If max change < FIXPOINT_TOLERANCE, converged
/// - Otherwise, use output as new input and repeat
/// - Terminates after MAX_FIXPOINT_ITERATIONS to prevent infinite loops
pub fn run_rule_fixpoint(
    input: &BeliefGraph,
    rule: &RuleDef,
    globals: &HashMap<String, f64>,
) -> Result<BeliefGraph, ExecError> {
    let mut current = input.clone();

    for _iteration in 0..MAX_FIXPOINT_ITERATIONS {
        let next = run_rule_for_each_with_globals(&current, rule, globals)?;
        let max_change = compute_max_change(&current, &next);

        // Check for convergence
        if max_change < FIXPOINT_TOLERANCE {
            #[cfg(feature = "tracing")]
            tracing::debug!(
                "Fixpoint converged after {} iterations (max_change = {:.2e})",
                _iteration + 1,
                max_change
            );
            return Ok(next);
        }

        current = next;
    }

    // Iteration limit reached without convergence
    Err(ExecError::Execution(format!(
        "Fixpoint rule '{}' did not converge after {} iterations",
        rule.name, MAX_FIXPOINT_ITERATIONS
    )))
}

/// Public API for running rules with automatic mode detection.
///
/// Dispatches to `run_rule_for_each_with_globals` or `run_rule_fixpoint` based on mode.
pub fn run_rule(
    input: &BeliefGraph,
    rule: &RuleDef,
    globals: &HashMap<String, f64>,
) -> Result<BeliefGraph, ExecError> {
    match rule.mode.as_deref() {
        Some("fixpoint") => run_rule_fixpoint(input, rule, globals),
        Some("for_each") | None => run_rule_for_each_with_globals(input, rule, globals),
        Some(other) => Err(ExecError::ValidationError(format!(
            "unknown rule mode: '{}' (expected 'for_each' or 'fixpoint')",
            other
        ))),
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::engine::graph::{BeliefGraph, BetaPosterior, EdgeData, GaussianPosterior, NodeData};
    use crate::frontend::ast::{BinaryOp, CallArg, EdgePattern, NodePattern};

    // Helper to create a simple test graph
    fn create_test_graph() -> BeliefGraph {
        let mut g = BeliefGraph::default();
        g.insert_node(NodeData {
            id: NodeId(1),
            label: Arc::from("Person"),
            attrs: HashMap::from([("x".into(), GaussianPosterior { mean: 10.0, precision: 1.0 })]),
        });
        g.insert_node(NodeData {
            id: NodeId(2),
            label: Arc::from("Person"),
            attrs: HashMap::from([("x".into(), GaussianPosterior { mean: 5.0, precision: 1.0 })]),
        });
        g.insert_edge(BeliefGraph::test_edge_with_beta(
            EdgeId(1), NodeId(1), NodeId(2), "KNOWS".into(),
            BetaPosterior { alpha: 5.0, beta: 5.0 },
        ));
        // Apply delta to ensure items are in base for iteration
        g.ensure_owned();
        g
    }

    // ============================================================================
    // eval_expr Tests (now using shared evaluator)
    // ============================================================================

    #[test]
    fn eval_expr_number() {
        let g = BeliefGraph::default();
        let bindings = MatchBindings::default();
        let locals = Locals::new();
        let ctx = RuleExprContext {
            bindings: &bindings,
            locals: &locals,
            globals: &HashMap::new(),
        };

        let result = eval_expr_core(&ExprAst::Number(42.5), &g, &ctx).unwrap();
        assert!((result - 42.5).abs() < 1e-9);
    }

    #[test]
    fn eval_expr_bool() {
        let g = BeliefGraph::default();
        let bindings = MatchBindings::default();
        let locals = Locals::new();
        let ctx = RuleExprContext {
            bindings: &bindings,
            locals: &locals,
            globals: &HashMap::new(),
        };

        let result = eval_expr_core(&ExprAst::Bool(true), &g, &ctx).unwrap();
        assert_eq!(result, 1.0);
        let result = eval_expr_core(&ExprAst::Bool(false), &g, &ctx).unwrap();
        assert_eq!(result, 0.0);
    }

    #[test]
    fn eval_expr_var() {
        let g = BeliefGraph::default();
        let mut locals = Locals::new();
        locals.set("x".into(), 10.0);
        let bindings = MatchBindings::default();
        let ctx = RuleExprContext {
            bindings: &bindings,
            locals: &locals,
            globals: &HashMap::new(),
        };

        let result = eval_expr_core(&ExprAst::Var("x".into()), &g, &ctx).unwrap();
        assert_eq!(result, 10.0);
    }

    #[test]
    fn eval_expr_binary_arithmetic() {
        let g = BeliefGraph::default();
        let bindings = MatchBindings::default();
        let locals = Locals::new();
        let ctx = RuleExprContext {
            bindings: &bindings,
            locals: &locals,
            globals: &HashMap::new(),
        };

        use crate::frontend::ast::BinaryOp;
        let expr = ExprAst::Binary {
            op: BinaryOp::Add,
            left: Box::new(ExprAst::Number(3.0)),
            right: Box::new(ExprAst::Number(4.0)),
        };
        let result = eval_expr_core(&expr, &g, &ctx).unwrap();
        assert_eq!(result, 7.0);

        let expr = ExprAst::Binary {
            op: BinaryOp::Mul,
            left: Box::new(ExprAst::Number(3.0)),
            right: Box::new(ExprAst::Number(4.0)),
        };
        let result = eval_expr_core(&expr, &g, &ctx).unwrap();
        assert_eq!(result, 12.0);
    }

    #[test]
    fn eval_expr_binary_comparison() {
        let g = BeliefGraph::default();
        let bindings = MatchBindings::default();
        let locals = Locals::new();
        let ctx = RuleExprContext {
            bindings: &bindings,
            locals: &locals,
            globals: &HashMap::new(),
        };

        use crate::frontend::ast::BinaryOp;
        let expr = ExprAst::Binary {
            op: BinaryOp::Eq,
            left: Box::new(ExprAst::Number(5.0)),
            right: Box::new(ExprAst::Number(5.0)),
        };
        assert_eq!(eval_expr_core(&expr, &g, &ctx).unwrap(), 1.0);

        let expr = ExprAst::Binary {
            op: BinaryOp::Lt,
            left: Box::new(ExprAst::Number(3.0)),
            right: Box::new(ExprAst::Number(5.0)),
        };
        assert_eq!(eval_expr_core(&expr, &g, &ctx).unwrap(), 1.0);
    }

    #[test]
    fn eval_expr_unary() {
        let g = BeliefGraph::default();
        let bindings = MatchBindings::default();
        let locals = Locals::new();
        let ctx = RuleExprContext {
            bindings: &bindings,
            locals: &locals,
            globals: &HashMap::new(),
        };

        use crate::frontend::ast::UnaryOp;
        let expr = ExprAst::Unary {
            op: UnaryOp::Neg,
            expr: Box::new(ExprAst::Number(5.0)),
        };
        assert_eq!(eval_expr_core(&expr, &g, &ctx).unwrap(), -5.0);

        let expr = ExprAst::Unary {
            op: UnaryOp::Not,
            expr: Box::new(ExprAst::Number(0.0)),
        };
        assert_eq!(eval_expr_core(&expr, &g, &ctx).unwrap(), 1.0);
    }

    #[test]
    fn eval_expr_logical_ops() {
        let g = BeliefGraph::default();
        let bindings = MatchBindings::default();
        let locals = Locals::new();
        let ctx = RuleExprContext {
            bindings: &bindings,
            locals: &locals,
            globals: &HashMap::new(),
        };

        use crate::frontend::ast::BinaryOp;
        let expr = ExprAst::Binary {
            op: BinaryOp::And,
            left: Box::new(ExprAst::Number(1.0)),
            right: Box::new(ExprAst::Number(1.0)),
        };
        assert_eq!(eval_expr_core(&expr, &g, &ctx).unwrap(), 1.0);

        let expr = ExprAst::Binary {
            op: BinaryOp::And,
            left: Box::new(ExprAst::Number(1.0)),
            right: Box::new(ExprAst::Number(0.0)),
        };
        assert_eq!(eval_expr_core(&expr, &g, &ctx).unwrap(), 0.0);

        let expr = ExprAst::Binary {
            op: BinaryOp::Or,
            left: Box::new(ExprAst::Number(0.0)),
            right: Box::new(ExprAst::Number(1.0)),
        };
        assert_eq!(eval_expr_core(&expr, &g, &ctx).unwrap(), 1.0);
    }

    #[test]
    fn eval_expr_globals() {
        let g = BeliefGraph::default();
        let bindings = MatchBindings::default();
        let locals = Locals::new();
        let mut globals = HashMap::new();
        globals.insert("global_var".into(), 99.0);
        let ctx = RuleExprContext {
            bindings: &bindings,
            locals: &locals,
            globals: &globals,
        };

        let result = eval_expr_core(&ExprAst::Var("global_var".into()), &g, &ctx).unwrap();
        assert_eq!(result, 99.0);
    }

    // ============================================================================
    // execute_actions Tests
    // ============================================================================

    #[test]
    fn actions_set_expectation_and_force_absent() {
        let mut g = create_test_graph();
        let mut bindings = MatchBindings::default();
        bindings.node_vars.insert("A".into(), NodeId(1));
        bindings.edge_vars.insert("e".into(), EdgeId(1));

        let actions = vec![
            ActionStmt::Let {
                name: "v_ab".into(),
                expr: ExprAst::Binary {
                    op: crate::frontend::ast::BinaryOp::Div,
                    left: Box::new(ExprAst::Call {
                        name: "E".into(),
                        args: vec![CallArg::Positional(ExprAst::Field {
                            target: Box::new(ExprAst::Var("A".into())),
                            field: "x".into(),
                        })],
                    }),
                    right: Box::new(ExprAst::Number(2.0)),
                },
            },
            ActionStmt::SetExpectation {
                node_var: "A".into(),
                attr: "x".into(),
                expr: ExprAst::Var("v_ab".into()),
            },
            ActionStmt::ForceAbsent { edge_var: "e".into() },
        ];

        execute_actions(&mut g, &actions, &bindings, &HashMap::new()).unwrap();

        let prob = g.prob_mean(EdgeId(1)).unwrap();
        assert!(prob < 0.01, "edge should be forced absent");
    }

    #[test]
    fn run_rule_for_each_multi_pattern_chain() {
        // Test: (A)-[ab]->(B), (B)-[bc]->(C) - find chains of two edges
        let mut g = BeliefGraph::default();
        
        // Create nodes
        let n1 = g.add_node("Person".into(), HashMap::new());
        let n2 = g.add_node("Person".into(), HashMap::new());
        let n3 = g.add_node("Person".into(), HashMap::new());
        
        // Create edges: n1 -> n2 -> n3 (chain)
        let e1 = g.add_edge(n1, n2, "REL".into(), BetaPosterior { alpha: 9.0, beta: 1.0 });
        let e2 = g.add_edge(n2, n3, "REL".into(), BetaPosterior { alpha: 8.0, beta: 2.0 });
        
        // Create another edge that doesn't form a chain
        let _e3 = g.add_edge(n1, n3, "REL".into(), BetaPosterior { alpha: 1.0, beta: 9.0 });
        
        // Apply delta to ensure all items are in base for iteration
        g.ensure_owned();
        
        let rule = RuleDef {
            name: "ChainRule".into(),
            on_model: "M".into(),
            patterns: vec![
                PatternItem {
                    src: NodePattern { var: "A".into(), label: "Person".into() },
                    edge: EdgePattern { var: "ab".into(), ty: "REL".into() },
                    dst: NodePattern { var: "B".into(), label: "Person".into() },
                },
                PatternItem {
                    src: NodePattern { var: "B".into(), label: "Person".into() },
                    edge: EdgePattern { var: "bc".into(), ty: "REL".into() },
                    dst: NodePattern { var: "C".into(), label: "Person".into() },
                },
            ],
            where_expr: None,
            actions: vec![ActionStmt::ForceAbsent { edge_var: "bc".into() }],
            mode: Some("for_each".into()),
        };
        
        let result = run_rule_for_each_with_globals(&g, &rule, &HashMap::new()).unwrap();
        
        // Should have matched the chain (e1 -> e2) and forced e2 absent
        assert!(result.prob_mean(e2).unwrap() < 1e-5);
        // e1 should still be present
        assert!(result.prob_mean(e1).unwrap() > 0.5);
    }

    #[test]
    fn run_rule_for_each_multi_pattern_with_where_clause() {
        // Test multi-pattern with where clause filtering
        let mut g = BeliefGraph::default();
        
        let n1 = g.add_node("Person".into(), HashMap::new());
        let n2 = g.add_node("Person".into(), HashMap::new());
        let n3 = g.add_node("Person".into(), HashMap::new());
        
        // High probability chain
        let _e1 = g.add_edge(n1, n2, "REL".to_string(), BetaPosterior { alpha: 9.0, beta: 1.0 });
        let e2 = g.add_edge(n2, n3, "REL".to_string(), BetaPosterior { alpha: 8.0, beta: 2.0 });
        
        // Low probability chain (should not match)
        let n4 = g.add_node("Person".into(), HashMap::new());
        let _e3 = g.add_edge(n1, n4, "REL".to_string(), BetaPosterior { alpha: 1.0, beta: 9.0 });
        let e4 = g.add_edge(n4, n3, "REL".to_string(), BetaPosterior { alpha: 2.0, beta: 8.0 });
        
        // Apply delta to ensure all items are in base for iteration
        g.ensure_owned();
        
        let rule = RuleDef {
            name: "HighProbChain".into(),
            on_model: "M".into(),
            patterns: vec![
                PatternItem {
                    src: NodePattern { var: "A".into(), label: "Person".into() },
                    edge: EdgePattern { var: "ab".into(), ty: "REL".into() },
                    dst: NodePattern { var: "B".into(), label: "Person".into() },
                },
                PatternItem {
                    src: NodePattern { var: "B".into(), label: "Person".into() },
                    edge: EdgePattern { var: "bc".into(), ty: "REL".into() },
                    dst: NodePattern { var: "C".into(), label: "Person".into() },
                },
            ],
            where_expr: Some(ExprAst::Binary {
                op: BinaryOp::And,
                left: Box::new(ExprAst::Binary {
                    op: BinaryOp::Ge,
                    left: Box::new(ExprAst::Call {
                        name: "prob".into(),
                        args: vec![CallArg::Positional(ExprAst::Var("ab".into()))],
                    }),
                    right: Box::new(ExprAst::Number(0.8)),
                }),
                right: Box::new(ExprAst::Binary {
                    op: BinaryOp::Ge,
                    left: Box::new(ExprAst::Call {
                        name: "prob".into(),
                        args: vec![CallArg::Positional(ExprAst::Var("bc".into()))],
                    }),
                    right: Box::new(ExprAst::Number(0.7)),
                }),
            }),
            actions: vec![ActionStmt::ForceAbsent { edge_var: "bc".into() }],
            mode: Some("for_each".into()),
        };
        
        let result = run_rule_for_each_with_globals(&g, &rule, &HashMap::new()).unwrap();
        
        // High probability chain should match and e2 should be forced absent
        assert!(result.prob_mean(e2).unwrap() < 1e-5);
        // Low probability chain should not match, e4 should remain
        assert!(result.prob_mean(e4).unwrap() > 0.1);
    }

    #[test]
    fn run_rule_for_each_multi_pattern_no_matches() {
        // Test multi-pattern that finds no matches
        let mut g = BeliefGraph::default();
        
        let n1 = g.add_node("Person".into(), HashMap::new());
        let n2 = g.add_node("Person".into(), HashMap::new());
        let n3 = g.add_node("Person".into(), HashMap::new());
        
        // Create edges but no chain
        let _e1 = g.add_edge(n1, n2, "REL".into(), BetaPosterior { alpha: 9.0, beta: 1.0 });
        let _e2 = g.add_edge(n3, n2, "REL".into(), BetaPosterior { alpha: 8.0, beta: 2.0 }); // Wrong direction
        
        // Apply delta to ensure all items are in base for iteration
        g.ensure_owned();
        
        let rule = RuleDef {
            name: "NoMatch".into(),
            on_model: "M".into(),
            patterns: vec![
                PatternItem {
                    src: NodePattern { var: "A".into(), label: "Person".into() },
                    edge: EdgePattern { var: "ab".into(), ty: "REL".into() },
                    dst: NodePattern { var: "B".into(), label: "Person".into() },
                },
                PatternItem {
                    src: NodePattern { var: "B".into(), label: "Person".into() },
                    edge: EdgePattern { var: "bc".into(), ty: "REL".into() },
                    dst: NodePattern { var: "C".into(), label: "Person".into() },
                },
            ],
            where_expr: None,
            actions: vec![ActionStmt::ForceAbsent { edge_var: "bc".into() }],
            mode: Some("for_each".into()),
        };
        
        let result = run_rule_for_each_with_globals(&g, &rule, &HashMap::new()).unwrap();
        
        // Graph should be unchanged (no matches)
        assert_eq!(result.edges().len(), g.edges().len());
    }

    #[test]
    fn run_rule_for_each_single_pattern() {
        let g = create_test_graph();
        let rule = RuleDef {
            name: "TestRule".into(),
            on_model: "M".into(),
            patterns: vec![PatternItem {
                src: NodePattern { var: "A".into(), label: "Person".into() },
                edge: EdgePattern { var: "e".into(), ty: "KNOWS".into() },
                dst: NodePattern { var: "B".into(), label: "Person".into() },
            }],
            where_expr: Some(ExprAst::Call {
                name: "prob".into(),
                args: vec![CallArg::Positional(ExprAst::Var("e".into()))],
            }),
            actions: vec![ActionStmt::ForceAbsent { edge_var: "e".into() }],
            mode: Some("for_each".into()),
        };

        let result = run_rule_for_each(&g, &rule).unwrap();
        let prob = result.prob_mean(EdgeId(1)).unwrap();
        assert!(prob < 0.01, "edge should be forced absent");
    }
}
