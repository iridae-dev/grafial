//! Flow execution engine.
//!
//! Executes flows: sequences of graph transformations that produce named graphs and metrics.
//! Graphs are immutable between transforms (each transform produces a new graph), enabling
//! safe parallel execution and snapshotting.

use std::collections::{HashMap, HashSet};

use crate::engine::errors::ExecError;
use crate::engine::evidence::build_graph_from_evidence_ir;
use crate::engine::expr_eval::{eval_expr_core, ExprContext};
use crate::engine::graph::{BeliefGraph, EdgeId};
use crate::engine::rule_exec::run_rule_for_each_with_globals_audit;
use crate::metrics::{eval_metric_expr, MetricContext, MetricRegistry};
use grafial_frontend::ast::RuleDef;
use grafial_frontend::{CallArg, ExprAst, ProgramAst};
use grafial_ir::{
    EvidenceIR, ExprIR, FlowIR, GraphExprIR, MetricImportDefIR, ProgramIR, RuleIR, TransformIR,
};

/// Graph builder trait for abstracting evidence-to-graph construction.
///
/// Allows `run_flow_internal` to work with both production evidence building and test-only
/// custom builders, eliminating duplication between `run_flow` and `run_flow_with_builder`.
trait GraphBuilder {
    fn build_graph(
        &self,
        evidence: &EvidenceIR,
        program: &ProgramIR,
    ) -> Result<BeliefGraph, ExecError>;
}

/// Production graph builder using standard evidence building.
struct StandardGraphBuilder;

impl GraphBuilder for StandardGraphBuilder {
    fn build_graph(
        &self,
        evidence: &EvidenceIR,
        program: &ProgramIR,
    ) -> Result<BeliefGraph, ExecError> {
        build_graph_from_evidence_ir(evidence, program)
    }
}

/// Custom graph builder for testing.
struct CustomGraphBuilder<'a> {
    builder:
        &'a (dyn Fn(&grafial_frontend::ast::EvidenceDef) -> Result<BeliefGraph, ExecError> + 'a),
}

impl<'a> GraphBuilder for CustomGraphBuilder<'a> {
    fn build_graph(
        &self,
        evidence: &EvidenceIR,
        _program: &ProgramIR,
    ) -> Result<BeliefGraph, ExecError> {
        let evidence_ast = evidence.to_ast();
        (self.builder)(&evidence_ast)
    }
}

/// The result of running a flow: named graphs and exported aliases.
#[derive(Debug, Clone, Default)]
pub struct FlowResult {
    /// All graphs defined in the flow by variable name
    pub graphs: HashMap<String, BeliefGraph>,
    /// Exported graphs by alias string
    pub exports: HashMap<String, BeliefGraph>,
    /// Computed metrics (scalars) by metric variable name
    pub metrics: HashMap<String, f64>,
    /// Exported metrics by alias string
    pub metric_exports: HashMap<String, f64>,
    /// Named graph snapshots saved during pipeline execution
    pub snapshots: HashMap<String, BeliefGraph>,
    /// Runtime intervention audit events emitted by rule transforms.
    pub intervention_audit: Vec<InterventionAuditEvent>,
}

/// Runtime trace event for a rule-based intervention during flow execution.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct InterventionAuditEvent {
    /// Flow name where the rule was executed.
    pub flow: String,
    /// Graph variable currently being transformed.
    pub graph: String,
    /// Transform descriptor (`apply_rule#i`, `apply_ruleset#i[j]`).
    pub transform: String,
    /// Rule name.
    pub rule: String,
    /// Rule execution mode.
    pub mode: String,
    /// Number of bindings that executed actions.
    pub matched_bindings: usize,
    /// Total action statements executed.
    pub actions_executed: usize,
}

/// IR execution backend boundary.
///
/// Phase 10 introduces this trait so runtime execution can swap interpreter/JIT
/// backends without changing frontend or IR lowering entrypoints.
pub trait IrExecutionBackend {
    /// Stable backend identifier for diagnostics/logging.
    fn backend_name(&self) -> &'static str;

    /// Execute a flow from IR.
    fn run_flow_ir(
        &self,
        program: &ProgramIR,
        flow_name: &str,
        prior: Option<&FlowResult>,
    ) -> Result<FlowResult, ExecError>;
}

/// Default interpreter backend (current production behavior).
#[derive(Debug, Clone, Copy, Default)]
pub struct InterpreterExecutionBackend;

impl IrExecutionBackend for InterpreterExecutionBackend {
    fn backend_name(&self) -> &'static str {
        "interpreter"
    }

    fn run_flow_ir(
        &self,
        program: &ProgramIR,
        flow_name: &str,
        prior: Option<&FlowResult>,
    ) -> Result<FlowResult, ExecError> {
        run_flow_ir_interpreter(program, flow_name, prior)
    }
}

/// Runs a named flow from a parsed and validated program.
///
/// Each transform produces a new graph (immutability), enabling safe snapshotting and
/// parallel execution. Metrics are evaluated after all graph transformations complete.
pub fn run_flow(
    program: &ProgramAst,
    flow_name: &str,
    prior: Option<&FlowResult>,
) -> Result<FlowResult, ExecError> {
    let lowered = ProgramIR::from(program);
    run_flow_ir(&lowered, flow_name, prior)
}

/// Runs a named flow from lowered IR.
pub fn run_flow_ir(
    program: &ProgramIR,
    flow_name: &str,
    prior: Option<&FlowResult>,
) -> Result<FlowResult, ExecError> {
    let backend = InterpreterExecutionBackend;
    run_flow_ir_with_backend(program, flow_name, prior, &backend)
}

/// Runs a named flow from IR using an explicit execution backend.
pub fn run_flow_ir_with_backend(
    program: &ProgramIR,
    flow_name: &str,
    prior: Option<&FlowResult>,
    backend: &dyn IrExecutionBackend,
) -> Result<FlowResult, ExecError> {
    backend.run_flow_ir(program, flow_name, prior)
}

fn run_flow_ir_interpreter(
    program: &ProgramIR,
    flow_name: &str,
    prior: Option<&FlowResult>,
) -> Result<FlowResult, ExecError> {
    let optimized_program = program.optimized();
    let builder = StandardGraphBuilder;
    run_flow_internal(&optimized_program, flow_name, prior, &builder)
}

/// Test-only helper that allows custom evidence builders for testing.
///
/// Production code should use `run_flow()`. This exists to support tests that need
/// custom graph construction without going through the standard evidence building path.
pub fn run_flow_with_builder<'a>(
    program: &'a ProgramAst,
    flow_name: &str,
    evidence_builder: &'a (dyn Fn(&grafial_frontend::ast::EvidenceDef) -> Result<BeliefGraph, ExecError>
             + 'a),
    prior: Option<&FlowResult>,
) -> Result<FlowResult, ExecError> {
    let lowered = ProgramIR::from(program).optimized();
    let builder = CustomGraphBuilder {
        builder: evidence_builder,
    };
    run_flow_internal(&lowered, flow_name, prior, &builder)
}

/// Internal flow execution logic shared between `run_flow` and `run_flow_with_builder`.
fn run_flow_internal<B: GraphBuilder>(
    program: &ProgramIR,
    flow_name: &str,
    prior: Option<&FlowResult>,
    graph_builder: &B,
) -> Result<FlowResult, ExecError> {
    let flow = find_flow(program, flow_name)?;
    let graph_plan = build_graph_execution_plan(flow)?;

    let evidence_by_name = build_evidence_index(&program.evidences);
    let rule_defs_by_name = build_rule_defs_index(&program.rules, flow);

    let mut result = initialize_flow_result(prior);
    let rule_globals = build_rule_globals(flow, prior);

    // Evaluate graph definitions according to a deterministic dependency plan.
    for graph_idx in graph_plan {
        let graph_def = &flow.graphs[graph_idx];
        let graph = match &graph_def.expr {
            GraphExprIR::Pipeline {
                start_graph,
                transforms,
            } => {
                let mut current = result
                    .graphs
                    .get(start_graph)
                    .ok_or_else(|| {
                        ExecError::Internal(format!("unknown start graph '{}'", start_graph))
                    })?
                    .clone();
                for (transform_idx, transform) in transforms.iter().enumerate() {
                    current = apply_transform(
                        transform,
                        &current,
                        &rule_defs_by_name,
                        &rule_globals,
                        flow_name,
                        &graph_def.name,
                        transform_idx,
                        &mut result,
                    )?;
                }
                current
            }
            _ => eval_graph_expr(
                &graph_def.expr,
                &evidence_by_name,
                prior,
                graph_builder,
                program,
            )?,
        };
        result.graphs.insert(graph_def.name.clone(), graph);
    }

    evaluate_metrics(flow, prior, &mut result)?;
    handle_exports(flow, &mut result)?;

    Ok(result)
}

/// Find a flow by name in the program
fn find_flow<'a>(program: &'a ProgramIR, flow_name: &str) -> Result<&'a FlowIR, ExecError> {
    program
        .flows
        .iter()
        .find(|f| f.name == flow_name)
        .ok_or_else(|| ExecError::Internal(format!("unknown flow '{}'", flow_name)))
}

/// Build an index of evidences by name for O(1) lookup
fn build_evidence_index(evidences: &[EvidenceIR]) -> HashMap<&str, &EvidenceIR> {
    evidences.iter().map(|e| (e.name.as_str(), e)).collect()
}

/// Build an index of AST rule definitions referenced by this flow.
///
/// This acts as safe dead-rule elimination at execution time: unreferenced rules are not
/// converted or indexed.
fn build_rule_defs_index(rules: &[RuleIR], flow: &FlowIR) -> HashMap<String, RuleDef> {
    let referenced_rules = collect_referenced_rules(flow);
    if referenced_rules.is_empty() {
        return HashMap::new();
    }

    let rule_by_name: HashMap<&str, &RuleIR> = rules
        .iter()
        .map(|rule| (rule.name.as_str(), rule))
        .collect();
    let mut referenced: Vec<_> = referenced_rules.into_iter().collect();
    referenced.sort_unstable();

    let mut out = HashMap::with_capacity(referenced.len());
    for rule_name in referenced {
        if let Some(rule) = rule_by_name.get(rule_name.as_str()) {
            out.insert(rule_name, rule.to_ast());
        }
    }
    out
}

fn collect_referenced_rules(flow: &FlowIR) -> HashSet<String> {
    let mut referenced = HashSet::new();
    for graph in &flow.graphs {
        if let GraphExprIR::Pipeline { transforms, .. } = &graph.expr {
            for transform in transforms {
                match transform {
                    TransformIR::ApplyRule { rule, .. } => {
                        referenced.insert(rule.clone());
                    }
                    TransformIR::ApplyRuleset { rules } => {
                        referenced.extend(rules.iter().cloned());
                    }
                    TransformIR::Snapshot { .. } | TransformIR::PruneEdges { .. } => {}
                }
            }
        }
    }
    referenced
}

/// Build a deterministic graph execution plan for a flow.
///
/// The plan is dependency-driven:
/// - non-pipeline graphs are always ready
/// - pipeline graphs are ready when their start graph has already been produced
///
/// If no progress can be made, the flow has unresolved or cyclic dependencies.
fn build_graph_execution_plan(flow: &FlowIR) -> Result<Vec<usize>, ExecError> {
    let mut seen_names = HashSet::new();
    for graph in &flow.graphs {
        if !seen_names.insert(graph.name.as_str()) {
            return Err(ExecError::Internal(format!(
                "duplicate graph name '{}' in flow '{}'",
                graph.name, flow.name
            )));
        }
    }

    let mut pending: Vec<usize> = (0..flow.graphs.len()).collect();
    let mut produced = HashSet::new();
    let mut plan = Vec::with_capacity(flow.graphs.len());

    while !pending.is_empty() {
        let mut progressed = false;
        let mut next_pending = Vec::new();

        for graph_idx in pending {
            let graph = &flow.graphs[graph_idx];
            let ready = match &graph.expr {
                GraphExprIR::Pipeline { start_graph, .. } => {
                    produced.contains(start_graph.as_str())
                }
                GraphExprIR::FromEvidence(_) | GraphExprIR::FromGraph(_) => true,
            };

            if ready {
                plan.push(graph_idx);
                produced.insert(graph.name.as_str());
                progressed = true;
            } else {
                next_pending.push(graph_idx);
            }
        }

        if !progressed {
            let mut unresolved: Vec<_> = next_pending
                .iter()
                .map(|idx| flow.graphs[*idx].name.clone())
                .collect();
            unresolved.sort_unstable();
            return Err(ExecError::Internal(format!(
                "unable to resolve graph execution order in flow '{}'; unresolved: {}",
                flow.name,
                unresolved.join(", ")
            )));
        }
        pending = next_pending;
    }

    Ok(plan)
}

/// Initialize flow result with prior metrics if available
fn initialize_flow_result(prior: Option<&FlowResult>) -> FlowResult {
    let mut result = FlowResult::default();
    if let Some(p) = prior {
        result.metric_exports.extend(p.metric_exports.clone());
    }
    result
}

/// Build rule globals from imported metrics.
fn build_rule_globals(flow: &FlowIR, prior: Option<&FlowResult>) -> HashMap<String, f64> {
    import_metric_bindings(&flow.metric_imports, prior)
}

/// Evaluate a graph expression to produce a BeliefGraph.
fn eval_graph_expr<B: GraphBuilder>(
    expr: &GraphExprIR,
    evidence_by_name: &HashMap<&str, &EvidenceIR>,
    prior: Option<&FlowResult>,
    graph_builder: &B,
    program: &ProgramIR,
) -> Result<BeliefGraph, ExecError> {
    match expr {
        GraphExprIR::FromEvidence(evidence) => {
            let ev = evidence_by_name
                .get(evidence.as_str())
                .ok_or_else(|| ExecError::Internal(format!("unknown evidence '{}'", evidence)))?;
            graph_builder.build_graph(ev, program)
        }
        GraphExprIR::FromGraph(alias) => lookup_graph_from_prior(alias, prior),
        GraphExprIR::Pipeline { .. } => {
            // Pipelines require the start graph to already exist, so they're handled
            // separately in run_flow_internal after initial graphs are created
            Err(ExecError::Internal(
                "pipeline evaluation should be handled separately".into(),
            ))
        }
    }
}

/// Look up a graph from prior flow's exports or snapshots.
fn lookup_graph_from_prior(
    alias: &str,
    prior: Option<&FlowResult>,
) -> Result<BeliefGraph, ExecError> {
    prior
        .and_then(|p| {
            // First try exports, then snapshots
            p.exports.get(alias).or_else(|| p.snapshots.get(alias))
        })
        .cloned()
        .ok_or_else(|| {
            ExecError::Internal(format!(
                "graph '{}' not found in prior flow exports or snapshots",
                alias
            ))
        })
}

/// Applies a single transform to a graph, returning a new graph.
fn apply_transform(
    transform: &TransformIR,
    graph: &BeliefGraph,
    rules_by_name: &HashMap<String, RuleDef>,
    rule_globals: &HashMap<String, f64>,
    flow_name: &str,
    graph_name: &str,
    transform_idx: usize,
    result: &mut FlowResult,
) -> Result<BeliefGraph, ExecError> {
    match transform {
        TransformIR::ApplyRule { rule, .. } => {
            let r = rules_by_name
                .get(rule)
                .ok_or_else(|| ExecError::Internal(format!("unknown rule '{}'", rule)))?;
            let (next, audit) = run_rule_for_each_with_globals_audit(graph, r, rule_globals)?;
            result.intervention_audit.push(InterventionAuditEvent {
                flow: flow_name.to_string(),
                graph: graph_name.to_string(),
                transform: format!("apply_rule#{}", transform_idx),
                rule: audit.rule_name,
                mode: audit.mode,
                matched_bindings: audit.matched_bindings,
                actions_executed: audit.actions_executed,
            });
            Ok(next)
        }
        TransformIR::ApplyRuleset { rules } => {
            // Sequential application: each rule receives the previous rule's output
            let mut current = graph.clone();
            for (rule_idx, rule_name) in rules.iter().enumerate() {
                let r = rules_by_name.get(rule_name).ok_or_else(|| {
                    ExecError::Internal(format!("unknown rule '{}' in ruleset", rule_name))
                })?;
                let (next, audit) =
                    run_rule_for_each_with_globals_audit(&current, r, rule_globals)?;
                result.intervention_audit.push(InterventionAuditEvent {
                    flow: flow_name.to_string(),
                    graph: graph_name.to_string(),
                    transform: format!("apply_ruleset#{}[{}]", transform_idx, rule_idx),
                    rule: audit.rule_name,
                    mode: audit.mode,
                    matched_bindings: audit.matched_bindings,
                    actions_executed: audit.actions_executed,
                });
                current = next;
            }
            Ok(current)
        }
        TransformIR::Snapshot { name } => {
            // Ensure deltas are applied before snapshotting
            let mut snapshot_graph = graph.clone();
            snapshot_graph.ensure_owned();
            result.snapshots.insert(name.clone(), snapshot_graph);
            Ok(graph.clone())
        }
        TransformIR::PruneEdges {
            edge_type,
            predicate,
        } => {
            let predicate_ast = predicate.to_ast();
            prune_edges(graph, edge_type, &predicate_ast)
        }
    }
}

/// Evaluates metrics against the last defined graph.
///
/// Metrics are evaluated in dependency order (earlier metrics are available to later ones).
/// Imported metrics from prior flows are available to all metric expressions.
fn evaluate_metrics(
    flow: &FlowIR,
    prior: Option<&FlowResult>,
    result: &mut FlowResult,
) -> Result<(), ExecError> {
    if flow.metrics.is_empty() && flow.metric_exports.is_empty() && flow.metric_imports.is_empty() {
        return Ok(());
    }

    let last_graph_name = flow
        .graphs
        .last()
        .ok_or_else(|| ExecError::Internal("no graphs defined for metric evaluation".into()))?
        .name
        .as_str();
    let target_graph = result
        .graphs
        .get(last_graph_name)
        .ok_or_else(|| ExecError::Internal("metric target graph missing".into()))?;

    let registry = MetricRegistry::with_builtins();
    let mut ctx = MetricContext {
        metrics: import_metric_bindings(&flow.metric_imports, prior),
    };

    // Earlier metrics in this flow are available to later ones
    for (k, v) in &result.metrics {
        ctx.metrics.insert(k.clone(), *v);
    }

    let live_metrics = compute_live_metrics(flow);

    // Evaluate each metric in order
    for m in &flow.metrics {
        if !live_metrics.contains(&m.name) {
            if let Some(v) = constant_metric_value(&m.expr) {
                // Safe dead-metric elimination: dead constants need no runtime evaluation.
                result.metrics.insert(m.name.clone(), v);
                ctx.metrics.insert(m.name.clone(), v);
                continue;
            }
        }

        let metric_expr = m.expr.to_ast();
        let v = eval_metric_expr(&metric_expr, target_graph, &registry, &ctx)?;
        result.metrics.insert(m.name.clone(), v);
        ctx.metrics.insert(m.name.clone(), v);
    }

    // Handle metric exports for this flow
    for mex in &flow.metric_exports {
        let val = result.metrics.get(&mex.metric).copied().ok_or_else(|| {
            ExecError::Internal(format!("unknown metric '{}' in export_metric", mex.metric))
        })?;
        result.metric_exports.insert(mex.alias.clone(), val);
    }

    Ok(())
}

/// Computes metrics required for observable flow outputs (`export_metric`) plus dependencies.
fn compute_live_metrics(flow: &FlowIR) -> HashSet<String> {
    let metric_names: HashSet<String> = flow
        .metrics
        .iter()
        .map(|metric| metric.name.clone())
        .collect();
    let mut live: HashSet<String> = flow
        .metric_exports
        .iter()
        .filter_map(|metric_export| {
            if metric_names.contains(&metric_export.metric) {
                Some(metric_export.metric.clone())
            } else {
                None
            }
        })
        .collect();

    // Metric variables are validated to reference earlier metrics only, so a reverse scan
    // is sufficient for transitive dependency closure.
    for metric in flow.metrics.iter().rev() {
        if live.contains(&metric.name) {
            collect_metric_dependencies(&metric.expr, &metric_names, &mut live);
        }
    }

    live
}

fn collect_metric_dependencies(
    expr: &ExprIR,
    metric_names: &HashSet<String>,
    live: &mut HashSet<String>,
) {
    match expr {
        ExprIR::Var(name) => {
            if metric_names.contains(name) {
                live.insert(name.clone());
            }
        }
        ExprIR::Field { target, .. } => {
            collect_metric_dependencies(target, metric_names, live);
        }
        ExprIR::Call { args, .. } => {
            for arg in args {
                match arg {
                    grafial_ir::CallArgIR::Positional(expr) => {
                        collect_metric_dependencies(expr, metric_names, live);
                    }
                    grafial_ir::CallArgIR::Named { value, .. } => {
                        collect_metric_dependencies(value, metric_names, live);
                    }
                }
            }
        }
        ExprIR::Unary { expr, .. } => {
            collect_metric_dependencies(expr, metric_names, live);
        }
        ExprIR::Binary { left, right, .. } => {
            collect_metric_dependencies(left, metric_names, live);
            collect_metric_dependencies(right, metric_names, live);
        }
        ExprIR::Exists { where_expr, .. } => {
            if let Some(expr) = where_expr {
                collect_metric_dependencies(expr, metric_names, live);
            }
        }
        ExprIR::Number(_) | ExprIR::Bool(_) => {}
    }
}

fn constant_metric_value(expr: &ExprIR) -> Option<f64> {
    match expr {
        ExprIR::Number(v) => Some(*v),
        ExprIR::Bool(value) => Some(if *value { 1.0 } else { 0.0 }),
        _ => None,
    }
}

/// Collect imported metric bindings from prior flow exports.
fn import_metric_bindings(
    imports: &[MetricImportDefIR],
    prior: Option<&FlowResult>,
) -> HashMap<String, f64> {
    let mut bindings = HashMap::new();
    if let Some(p) = prior {
        for imp in imports {
            if let Some(v) = p.metric_exports.get(&imp.source_alias) {
                bindings.insert(imp.local_name.clone(), *v);
            }
        }
    }
    bindings
}

/// Handle graph exports by alias.
fn handle_exports(flow: &FlowIR, result: &mut FlowResult) -> Result<(), ExecError> {
    for ex in &flow.exports {
        let g = result
            .graphs
            .get(&ex.graph)
            .ok_or_else(|| ExecError::Internal(format!("unknown graph '{}' in export", ex.graph)))?
            .clone();
        result.exports.insert(ex.alias.clone(), g);
    }
    Ok(())
}

fn prune_edges(
    input: &BeliefGraph,
    edge_type: &str,
    predicate: &ExprAst,
) -> Result<BeliefGraph, ExecError> {
    let (mut keep, mut candidates) = classify_edges(input, edge_type);
    candidates.sort();

    for eid in candidates {
        let should_keep = eval_prune_predicate(predicate, input, eid)? == 0.0;
        if should_keep {
            keep.push(eid);
        }
    }
    keep.sort();

    rebuild_pruned_graph(input, &keep)
}

/// Classify edges into those to keep (wrong type) and candidates for pruning (matching type)
fn classify_edges(input: &BeliefGraph, edge_type: &str) -> (Vec<EdgeId>, Vec<EdgeId>) {
    let mut keep = Vec::new();
    let mut candidates = Vec::new();
    let mut seen_edge_ids = std::collections::HashSet::new();

    for edge in input.edges() {
        seen_edge_ids.insert(edge.id);
        if edge.ty.as_ref() == edge_type {
            candidates.push(edge.id);
        } else {
            keep.push(edge.id);
        }
    }

    process_delta_changes(
        input.delta(),
        edge_type,
        &mut seen_edge_ids,
        &mut keep,
        &mut candidates,
    );

    (keep, candidates)
}

/// Process delta changes to update edge classifications
fn process_delta_changes(
    delta: &[crate::engine::graph::GraphDelta],
    edge_type: &str,
    seen_edge_ids: &mut std::collections::HashSet<EdgeId>,
    keep: &mut Vec<EdgeId>,
    candidates: &mut Vec<EdgeId>,
) {
    use crate::engine::graph::GraphDelta;

    for change in delta {
        match change {
            GraphDelta::EdgeChange { id, edge } => {
                if seen_edge_ids.insert(*id) {
                    // New edge from delta
                    if edge.ty.as_ref() == edge_type {
                        candidates.push(*id);
                    } else {
                        keep.push(*id);
                    }
                } else {
                    // Modified edge - reclassify
                    keep.retain(|&eid| eid != *id);
                    candidates.retain(|&eid| eid != *id);
                    if edge.ty.as_ref() == edge_type {
                        candidates.push(*id);
                    } else {
                        keep.push(*id);
                    }
                }
            }
            GraphDelta::EdgeRemoved { id } => {
                keep.retain(|&eid| eid != *id);
                candidates.retain(|&eid| eid != *id);
            }
            _ => {} // Node changes don't affect edge classification
        }
    }
}

/// Rebuild graph with only the specified edges
fn rebuild_pruned_graph(
    input: &BeliefGraph,
    keep_edges: &[EdgeId],
) -> Result<BeliefGraph, ExecError> {
    let mut input_mut = input.clone();
    input_mut.ensure_owned();
    input_mut.rebuild_with_edges(input_mut.nodes(), keep_edges)
}

/// Expression evaluation context for prune predicates.
///
/// Only allows `prob(edge)` function calls and prohibits variables/fields.
struct PruneExprContext {
    edge: EdgeId,
}

impl ExprContext for PruneExprContext {
    fn resolve_var(&self, _name: &str) -> Option<f64> {
        None // No variables allowed in prune predicates
    }

    fn eval_function(
        &self,
        name: &str,
        pos_args: &[ExprAst],
        all_args: &[CallArg],
        graph: &BeliefGraph,
    ) -> Result<f64, ExecError> {
        match name {
            "prob" => {
                if !all_args.is_empty() && matches!(all_args[0], CallArg::Named { .. }) {
                    return Err(ExecError::ValidationError(
                        "prob() does not accept named arguments".into(),
                    ));
                }
                if pos_args.len() != 1 {
                    return Err(ExecError::ValidationError(
                        "prob(): expected single positional argument".into(),
                    ));
                }
                match &pos_args[0] {
                    ExprAst::Var(v) if v == "edge" => graph.prob_mean(self.edge),
                    _ => Err(ExecError::ValidationError(
                        "prob(): argument must be 'edge' in prune predicate".into(),
                    )),
                }
            }
            _ => Err(ExecError::ValidationError(format!(
                "unsupported function '{}' in prune predicate",
                name
            ))),
        }
    }

    fn eval_field(
        &self,
        _target: &ExprAst,
        _field: &str,
        _graph: &BeliefGraph,
    ) -> Result<f64, ExecError> {
        Err(ExecError::ValidationError(
            "field access not allowed in prune predicate".into(),
        ))
    }
}

fn eval_prune_predicate(
    expr: &ExprAst,
    graph: &BeliefGraph,
    edge: EdgeId,
) -> Result<f64, ExecError> {
    let ctx = PruneExprContext { edge };
    eval_expr_core(expr, graph, &ctx)
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::engine::graph::{BetaPosterior, EdgeId, GaussianPosterior, NodeData, NodeId};
    use grafial_frontend::ast::*;

    fn build_simple_graph() -> BeliefGraph {
        let mut g = BeliefGraph::default();
        g.insert_node(NodeData {
            id: NodeId(1),
            label: "Person".into(),
            attrs: HashMap::from([(
                "value".into(),
                GaussianPosterior {
                    mean: 10.0,
                    precision: 1.0,
                },
            )]),
        });
        g.insert_node(NodeData {
            id: NodeId(2),
            label: "Person".into(),
            attrs: HashMap::from([(
                "value".into(),
                GaussianPosterior {
                    mean: 20.0,
                    precision: 1.0,
                },
            )]),
        });
        g.insert_edge(BeliefGraph::test_edge_with_beta(
            EdgeId(1),
            NodeId(1),
            NodeId(2),
            "REL".into(),
            BetaPosterior {
                alpha: 8.0,
                beta: 2.0,
            },
        ));
        g.insert_edge(BeliefGraph::test_edge_with_beta(
            EdgeId(2),
            NodeId(2),
            NodeId(1),
            "REL".into(),
            BetaPosterior {
                alpha: 1.0,
                beta: 9.0,
            },
        ));
        g
    }

    fn simple_evidence_builder(_: &EvidenceDef) -> Result<BeliefGraph, ExecError> {
        let mut graph = build_simple_graph();
        graph.ensure_owned();
        Ok(graph)
    }

    #[test]
    fn eval_prune_predicate_number_literal() {
        let g = build_simple_graph();
        let result = eval_prune_predicate(&ExprAst::Number(42.5), &g, EdgeId(1)).unwrap();
        assert_eq!(result, 42.5);
    }

    #[test]
    fn eval_prune_predicate_bool_true() {
        let g = build_simple_graph();
        let result = eval_prune_predicate(&ExprAst::Bool(true), &g, EdgeId(1)).unwrap();
        assert_eq!(result, 1.0);
    }

    #[test]
    fn eval_prune_predicate_bool_false() {
        let g = build_simple_graph();
        let result = eval_prune_predicate(&ExprAst::Bool(false), &g, EdgeId(1)).unwrap();
        assert_eq!(result, 0.0);
    }

    #[test]
    fn eval_prune_predicate_prob_edge() {
        let g = build_simple_graph();
        let expr = ExprAst::Call {
            name: "prob".into(),
            args: vec![CallArg::Positional(ExprAst::Var("edge".into()))],
        };
        let result = eval_prune_predicate(&expr, &g, EdgeId(1)).unwrap();
        assert!((result - 0.8).abs() < 0.01); // alpha=8, beta=2 -> 8/10=0.8
    }

    #[test]
    fn eval_prune_predicate_prob_requires_edge_var() {
        let g = build_simple_graph();
        let expr = ExprAst::Call {
            name: "prob".into(),
            args: vec![CallArg::Positional(ExprAst::Var("other".into()))],
        };
        let result = eval_prune_predicate(&expr, &g, EdgeId(1));
        assert!(result.is_err());
    }

    #[test]
    fn eval_prune_predicate_bare_var_fails() {
        let g = build_simple_graph();
        let result = eval_prune_predicate(&ExprAst::Var("x".into()), &g, EdgeId(1));
        assert!(result.is_err());
    }

    #[test]
    fn eval_prune_predicate_field_access_fails() {
        let g = build_simple_graph();
        let expr = ExprAst::Field {
            target: Box::new(ExprAst::Var("edge".into())),
            field: "prob".into(),
        };
        let result = eval_prune_predicate(&expr, &g, EdgeId(1));
        assert!(result.is_err());
    }

    #[test]
    fn eval_prune_predicate_unary_neg() {
        let g = build_simple_graph();
        let expr = ExprAst::Unary {
            op: UnaryOp::Neg,
            expr: Box::new(ExprAst::Number(5.0)),
        };
        let result = eval_prune_predicate(&expr, &g, EdgeId(1)).unwrap();
        assert_eq!(result, -5.0);
    }

    #[test]
    fn eval_prune_predicate_unary_not() {
        let g = build_simple_graph();
        let expr = ExprAst::Unary {
            op: UnaryOp::Not,
            expr: Box::new(ExprAst::Number(0.0)),
        };
        let result = eval_prune_predicate(&expr, &g, EdgeId(1)).unwrap();
        assert_eq!(result, 1.0);
    }

    #[test]
    fn eval_prune_predicate_binary_comparison() {
        let g = build_simple_graph();
        let expr = ExprAst::Binary {
            op: BinaryOp::Lt,
            left: Box::new(ExprAst::Number(0.5)),
            right: Box::new(ExprAst::Number(1.0)),
        };
        let result = eval_prune_predicate(&expr, &g, EdgeId(1)).unwrap();
        assert_eq!(result, 1.0);
    }

    #[test]
    fn prune_edges_removes_matching_edges() {
        let g = build_simple_graph();
        // Prune REL edges where prob(edge) < 0.5
        let predicate = ExprAst::Binary {
            op: BinaryOp::Lt,
            left: Box::new(ExprAst::Call {
                name: "prob".into(),
                args: vec![CallArg::Positional(ExprAst::Var("edge".into()))],
            }),
            right: Box::new(ExprAst::Number(0.5)),
        };
        let result = prune_edges(&g, "REL", &predicate).unwrap();
        // Edge 2 has prob < 0.5, so should be removed
        assert_eq!(result.edges().len(), 1);
        assert_eq!(result.edges()[0].id, EdgeId(1));
    }

    #[test]
    fn prune_edges_keeps_non_matching_type() {
        let mut g = build_simple_graph();
        g.insert_edge(BeliefGraph::test_edge_with_beta(
            EdgeId(3),
            NodeId(1),
            NodeId(2),
            "OTHER".into(),
            BetaPosterior {
                alpha: 1.0,
                beta: 1.0,
            },
        ));
        let predicate = ExprAst::Bool(true);
        let result = prune_edges(&g, "REL", &predicate).unwrap();
        // Should keep OTHER edge
        assert_eq!(result.edges().len(), 1);
        assert_eq!(result.edges()[0].ty.as_ref(), "OTHER");
    }

    #[test]
    fn prune_edges_with_constant_false_keeps_all() {
        let g = build_simple_graph();
        let predicate = ExprAst::Bool(false);
        let result = prune_edges(&g, "REL", &predicate).unwrap();
        assert_eq!(result.edges().len(), 2);
    }

    #[test]
    fn prune_edges_with_constant_true_removes_all_of_type() {
        let g = build_simple_graph();
        let predicate = ExprAst::Bool(true);
        let result = prune_edges(&g, "REL", &predicate).unwrap();
        assert_eq!(result.edges().len(), 0);
    }

    #[test]
    fn graph_execution_plan_handles_out_of_order_pipeline_dependencies() {
        let flow = FlowIR {
            name: "F".into(),
            on_model: "M".into(),
            graphs: vec![
                grafial_ir::GraphDefIR {
                    name: "g3".into(),
                    expr: GraphExprIR::Pipeline {
                        start_graph: "g2".into(),
                        transforms: vec![],
                    },
                },
                grafial_ir::GraphDefIR {
                    name: "g2".into(),
                    expr: GraphExprIR::Pipeline {
                        start_graph: "g1".into(),
                        transforms: vec![],
                    },
                },
                grafial_ir::GraphDefIR {
                    name: "g1".into(),
                    expr: GraphExprIR::FromEvidence("Ev".into()),
                },
            ],
            metrics: vec![],
            exports: vec![],
            metric_exports: vec![],
            metric_imports: vec![],
        };

        let plan = build_graph_execution_plan(&flow).expect("plan");
        assert_eq!(plan, vec![2, 1, 0]);
    }

    #[test]
    fn run_flow_supports_out_of_order_pipeline_chains() {
        let program = ProgramAst {
            schemas: vec![],
            belief_models: vec![],
            evidences: vec![EvidenceDef {
                name: "Ev".into(),
                on_model: "M".into(),
                observations: vec![],
                body_src: "".into(),
            }],
            rules: vec![],
            flows: vec![FlowDef {
                name: "Demo".into(),
                on_model: "M".into(),
                graphs: vec![
                    GraphDef {
                        name: "g3".into(),
                        expr: GraphExpr::Pipeline {
                            start: "g2".into(),
                            transforms: vec![],
                        },
                    },
                    GraphDef {
                        name: "g2".into(),
                        expr: GraphExpr::Pipeline {
                            start: "g1".into(),
                            transforms: vec![],
                        },
                    },
                    GraphDef {
                        name: "g1".into(),
                        expr: GraphExpr::FromEvidence {
                            evidence: "Ev".into(),
                        },
                    },
                ],
                metrics: vec![],
                exports: vec![],
                metric_exports: vec![],
                metric_imports: vec![],
            }],
        };

        let result =
            run_flow_with_builder(&program, "Demo", &simple_evidence_builder, None).expect("flow");
        assert!(result.graphs.contains_key("g1"));
        assert!(result.graphs.contains_key("g2"));
        assert!(result.graphs.contains_key("g3"));
        assert_eq!(
            result.graphs.get("g1").unwrap().edges().len(),
            result.graphs.get("g3").unwrap().edges().len()
        );
    }

    #[test]
    fn compute_live_metrics_tracks_export_dependencies() {
        let flow = FlowIR {
            name: "F".into(),
            on_model: "M".into(),
            graphs: vec![],
            metrics: vec![
                grafial_ir::MetricDefIR {
                    name: "m1".into(),
                    expr: ExprIR::Number(1.0),
                },
                grafial_ir::MetricDefIR {
                    name: "m2".into(),
                    expr: ExprIR::Binary {
                        op: grafial_ir::BinaryOpIR::Add,
                        left: Box::new(ExprIR::Var("m1".into())),
                        right: Box::new(ExprIR::Number(1.0)),
                    },
                },
                grafial_ir::MetricDefIR {
                    name: "m3".into(),
                    expr: ExprIR::Number(999.0),
                },
            ],
            exports: vec![],
            metric_exports: vec![grafial_ir::MetricExportDefIR {
                metric: "m2".into(),
                alias: "out".into(),
            }],
            metric_imports: vec![],
        };

        let live = compute_live_metrics(&flow);
        assert!(live.contains("m1"));
        assert!(live.contains("m2"));
        assert!(!live.contains("m3"));
    }

    #[derive(Debug, Clone, Copy, Default)]
    struct StubBackend;

    impl IrExecutionBackend for StubBackend {
        fn backend_name(&self) -> &'static str {
            "stub"
        }

        fn run_flow_ir(
            &self,
            _program: &ProgramIR,
            flow_name: &str,
            _prior: Option<&FlowResult>,
        ) -> Result<FlowResult, ExecError> {
            let mut result = FlowResult::default();
            result
                .metrics
                .insert("backend_marker".into(), flow_name.len() as f64);
            Ok(result)
        }
    }

    #[test]
    fn run_flow_ir_with_backend_dispatches_to_backend() {
        let program = ProgramIR {
            schemas: vec![],
            belief_models: vec![],
            evidences: vec![],
            rules: vec![],
            flows: vec![],
        };
        let backend = StubBackend;
        let result =
            run_flow_ir_with_backend(&program, "Demo", None, &backend).expect("backend run");
        assert_eq!(backend.backend_name(), "stub");
        assert_eq!(result.metrics.get("backend_marker"), Some(&4.0));
    }
}
