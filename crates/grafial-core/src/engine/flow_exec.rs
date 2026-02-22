//! Flow execution engine.
//!
//! Executes flows: sequences of graph transformations that produce named graphs and metrics.
//! Graphs are immutable between transforms (each transform produces a new graph), enabling
//! safe parallel execution and snapshotting.
//!
//! ## JIT backend selection
//!
//! When the `jit` feature is enabled, hot metric and prune expressions are compiled to
//! native machine code via Cranelift (`jit_backend`).  When the feature is disabled the
//! legacy register-bytecode interpreter is used instead.  Both paths share the same
//! hot-expression detection and caching infrastructure (`eval_hot_metric_with_cache` /
//! `eval_hot_prune_with_cache`) and provide identical deterministic fallback behaviour.
#![cfg_attr(feature = "parallel", allow(dead_code))]

use std::collections::{HashMap, HashSet};
use std::hash::Hash;
use std::sync::Mutex;

use crate::engine::belief_propagation::run_loopy_belief_propagation;
use crate::engine::errors::ExecError;
use crate::engine::evidence::build_graph_from_evidence_ir;
#[cfg(not(feature = "jit"))]
use crate::engine::expr_eval::{eval_binary_op, eval_unary_op};
use crate::engine::expr_eval::{eval_expr_core, ExprContext};
use crate::engine::graph::{BeliefGraph, EdgeId};
use crate::engine::model_selection::{select_best_graph, EdgeModelCriterion};
use crate::engine::rule_exec::run_rule_for_each_with_globals_audit;
use crate::metrics::{eval_metric_expr, MetricContext, MetricRegistry};
use grafial_frontend::ast::RuleDef;
use grafial_frontend::{CallArg, ExprAst, ProgramAst};
#[cfg(not(feature = "jit"))]
use grafial_ir::{BinaryOpIR, CallArgIR, UnaryOpIR};
use grafial_ir::{
    EvidenceIR, ExprIR, FlowIR, GraphExprIR, MetricImportDefIR, ModelSelectionCriterionIR,
    ProgramIR, RuleIR, TransformIR,
};

// When the `jit` feature is active, bring in the real Cranelift compiled types.
#[cfg(feature = "jit")]
use crate::engine::jit_backend::{CompiledMetricFn, CompiledPruneFn};

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

/// Explicit interpreter backend for parity checks and diagnostics.
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

/// Configuration for hot-expression JIT execution.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct JitConfig {
    /// Number of metric expression evaluations before compile is attempted.
    pub metric_compile_threshold: usize,
    /// Number of prune predicate evaluations before compile is attempted.
    pub prune_compile_threshold: usize,
}

impl Default for JitConfig {
    fn default() -> Self {
        Self {
            metric_compile_threshold: 8,
            prune_compile_threshold: 64,
        }
    }
}

/// Runtime profile counters for the hot-expression JIT backend.
#[derive(Debug, Clone, Copy, Default, PartialEq, Eq)]
pub struct JitProfile {
    pub metric_eval_count: usize,
    pub metric_compile_count: usize,
    pub metric_cache_hits: usize,
    pub metric_fallback_count: usize,
    pub prune_eval_count: usize,
    pub prune_compile_count: usize,
    pub prune_cache_hits: usize,
    pub prune_fallback_count: usize,
}

/// Compiled metric expression type.
///
/// With the `jit` feature: a real Cranelift-compiled native function.
/// Without the `jit` feature: the legacy register-bytecode program.
#[cfg(feature = "jit")]
type CompiledMetric = CompiledMetricFn;
#[cfg(not(feature = "jit"))]
type CompiledMetric = CraneliftMetricProgram;

/// Compiled prune expression type (same feature-gate pattern).
#[cfg(feature = "jit")]
type CompiledPrune = CompiledPruneFn;
#[cfg(not(feature = "jit"))]
type CompiledPrune = CraneliftPruneProgram;

#[derive(Debug, Default)]
struct JitState {
    metric_entries: HashMap<MetricExprKey, JitEntry<CompiledMetric>>,
    prune_entries: HashMap<PruneExprKey, JitEntry<CompiledPrune>>,
    profile: JitProfile,
}

#[derive(Debug, Clone, PartialEq, Eq, Hash)]
struct MetricExprKey {
    flow: String,
    metric: String,
}

#[derive(Debug, Clone, PartialEq, Eq, Hash)]
struct PruneExprKey {
    flow: String,
    graph: String,
    transform_idx: usize,
}

#[derive(Debug)]
struct JitEntry<T> {
    eval_count: usize,
    compiled: Option<T>,
    permanently_interpreted: bool,
}

impl<T> Default for JitEntry<T> {
    fn default() -> Self {
        Self {
            eval_count: 0,
            compiled: None,
            permanently_interpreted: false,
        }
    }
}

/// Cranelift JIT backend that compiles hot metric/prune expressions and
/// falls back to interpreter evaluation for unsupported cases.
#[derive(Debug)]
pub struct CraneliftJitExecutionBackend {
    config: JitConfig,
    state: Mutex<JitState>,
}

impl CraneliftJitExecutionBackend {
    pub fn new(config: JitConfig) -> Self {
        Self {
            config,
            state: Mutex::new(JitState::default()),
        }
    }

    /// Snapshot profiling counters for observability/debugging.
    pub fn profile_snapshot(&self) -> Result<JitProfile, ExecError> {
        let state = self
            .state
            .lock()
            .map_err(|_| ExecError::Internal("cranelift JIT state lock poisoned".into()))?;
        Ok(state.profile)
    }

    /// Clears profile counters without dropping compiled expression cache.
    pub fn clear_profile(&self) -> Result<(), ExecError> {
        let mut state = self
            .state
            .lock()
            .map_err(|_| ExecError::Internal("cranelift JIT state lock poisoned".into()))?;
        state.profile = JitProfile::default();
        Ok(())
    }
}

impl Default for CraneliftJitExecutionBackend {
    fn default() -> Self {
        Self::new(JitConfig::default())
    }
}

impl IrExecutionBackend for CraneliftJitExecutionBackend {
    fn backend_name(&self) -> &'static str {
        "cranelift-jit"
    }

    fn run_flow_ir(
        &self,
        program: &ProgramIR,
        flow_name: &str,
        prior: Option<&FlowResult>,
    ) -> Result<FlowResult, ExecError> {
        let optimized_program = program.optimized();
        let builder = StandardGraphBuilder;
        let mut state = self
            .state
            .lock()
            .map_err(|_| ExecError::Internal("cranelift JIT state lock poisoned".into()))?;
        let mut expr_evaluator = CraneliftExprEvaluator {
            config: self.config,
            state: &mut state,
        };
        run_flow_internal(
            &optimized_program,
            flow_name,
            prior,
            &builder,
            &mut expr_evaluator,
        )
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
    let backend = CraneliftJitExecutionBackend::default();
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

/// Expression evaluation abstraction for flow metric/prune execution.
trait FlowExprEvaluator {
    fn eval_metric_expr(
        &mut self,
        flow_name: &str,
        metric_name: &str,
        expr: &ExprIR,
        graph: &BeliefGraph,
        registry: &MetricRegistry,
        ctx: &MetricContext,
    ) -> Result<f64, ExecError>;

    fn eval_prune_predicate(
        &mut self,
        flow_name: &str,
        graph_name: &str,
        transform_idx: usize,
        expr: &ExprIR,
        graph: &BeliefGraph,
        edge: EdgeId,
    ) -> Result<f64, ExecError>;
}

#[derive(Debug, Clone, Copy, Default)]
struct InterpreterFlowExprEvaluator;

impl FlowExprEvaluator for InterpreterFlowExprEvaluator {
    fn eval_metric_expr(
        &mut self,
        _flow_name: &str,
        _metric_name: &str,
        expr: &ExprIR,
        graph: &BeliefGraph,
        registry: &MetricRegistry,
        ctx: &MetricContext,
    ) -> Result<f64, ExecError> {
        let expr_ast = expr.to_ast();
        eval_metric_expr(&expr_ast, graph, registry, ctx)
    }

    fn eval_prune_predicate(
        &mut self,
        _flow_name: &str,
        _graph_name: &str,
        _transform_idx: usize,
        expr: &ExprIR,
        graph: &BeliefGraph,
        edge: EdgeId,
    ) -> Result<f64, ExecError> {
        let expr_ast = expr.to_ast();
        eval_prune_predicate(&expr_ast, graph, edge)
    }
}

struct CraneliftExprEvaluator<'a> {
    config: JitConfig,
    state: &'a mut JitState,
}

impl<'a> FlowExprEvaluator for CraneliftExprEvaluator<'a> {
    fn eval_metric_expr(
        &mut self,
        flow_name: &str,
        metric_name: &str,
        expr: &ExprIR,
        graph: &BeliefGraph,
        registry: &MetricRegistry,
        ctx: &MetricContext,
    ) -> Result<f64, ExecError> {
        let key = MetricExprKey {
            flow: flow_name.to_string(),
            metric: metric_name.to_string(),
        };
        eval_hot_metric_with_cache(
            &mut self.state.profile,
            HotEvalRequest {
                entries: &mut self.state.metric_entries,
                compile_threshold: self.config.metric_compile_threshold,
                key,
                expr,
            },
            jit_compile_metric,
            |compiled| jit_eval_metric(compiled, ctx),
            || {
                let expr_ast = expr.to_ast();
                eval_metric_expr(&expr_ast, graph, registry, ctx)
            },
        )
    }

    fn eval_prune_predicate(
        &mut self,
        flow_name: &str,
        graph_name: &str,
        transform_idx: usize,
        expr: &ExprIR,
        graph: &BeliefGraph,
        edge: EdgeId,
    ) -> Result<f64, ExecError> {
        let key = PruneExprKey {
            flow: flow_name.to_string(),
            graph: graph_name.to_string(),
            transform_idx,
        };
        eval_hot_prune_with_cache(
            &mut self.state.profile,
            HotEvalRequest {
                entries: &mut self.state.prune_entries,
                compile_threshold: self.config.prune_compile_threshold,
                key,
                expr,
            },
            jit_compile_prune,
            |compiled| jit_eval_prune(compiled, graph, edge),
            || {
                let expr_ast = expr.to_ast();
                eval_prune_predicate(&expr_ast, graph, edge)
            },
        )
    }
}

// ---------------------------------------------------------------------------
// Feature-gated compile / eval dispatch
// ---------------------------------------------------------------------------

/// Compile a metric expression to the appropriate compiled form.
///
/// With `jit` feature: generates real Cranelift native code.
/// Without `jit` feature: lowers to register-bytecode.
#[cfg(feature = "jit")]
fn jit_compile_metric(expr: &ExprIR) -> Option<CompiledMetric> {
    crate::engine::jit_backend::compile_metric_expr(expr)
}

#[cfg(not(feature = "jit"))]
fn jit_compile_metric(expr: &ExprIR) -> Option<CompiledMetric> {
    compile_metric_expr_jit(expr)
}

/// Evaluate a compiled metric expression.
#[cfg(feature = "jit")]
fn jit_eval_metric(compiled: &CompiledMetric, ctx: &MetricContext) -> Result<f64, ExecError> {
    compiled.eval(ctx)
}

#[cfg(not(feature = "jit"))]
fn jit_eval_metric(compiled: &CompiledMetric, ctx: &MetricContext) -> Result<f64, ExecError> {
    compiled.eval(ctx)
}

/// Compile a prune predicate expression.
#[cfg(feature = "jit")]
fn jit_compile_prune(expr: &ExprIR) -> Option<CompiledPrune> {
    crate::engine::jit_backend::compile_prune_expr(expr)
}

#[cfg(not(feature = "jit"))]
fn jit_compile_prune(expr: &ExprIR) -> Option<CompiledPrune> {
    compile_prune_expr_jit(expr)
}

/// Evaluate a compiled prune expression.
///
/// With `jit` feature: pre-compute edge probability and pass as scalar.
/// Without `jit` feature: pass graph/edge directly to bytecode program.
#[cfg(feature = "jit")]
fn jit_eval_prune(
    compiled: &CompiledPrune,
    graph: &BeliefGraph,
    edge: EdgeId,
) -> Result<f64, ExecError> {
    let edge_prob = graph.prob_mean(edge)?;
    compiled.eval(edge_prob)
}

#[cfg(not(feature = "jit"))]
fn jit_eval_prune(
    compiled: &CompiledPrune,
    graph: &BeliefGraph,
    edge: EdgeId,
) -> Result<f64, ExecError> {
    compiled.eval(graph, edge)
}

#[allow(clippy::too_many_arguments)]
struct HotEvalRequest<'a, K, Compiled> {
    entries: &'a mut HashMap<K, JitEntry<Compiled>>,
    compile_threshold: usize,
    key: K,
    expr: &'a ExprIR,
}

struct HotEvalCounters<'a> {
    eval_count: &'a mut usize,
    compile_count: &'a mut usize,
    cache_hits: &'a mut usize,
    fallback_count: &'a mut usize,
}

fn eval_hot_with_cache<K, Compiled, CompileFn, EvalFn, FallbackFn>(
    counters: HotEvalCounters<'_>,
    request: HotEvalRequest<'_, K, Compiled>,
    compile_expr: CompileFn,
    eval_compiled: EvalFn,
    fallback_eval: FallbackFn,
) -> Result<f64, ExecError>
where
    K: Eq + Hash + Clone,
    Compiled: Clone,
    CompileFn: Fn(&ExprIR) -> Option<Compiled>,
    EvalFn: Fn(&Compiled) -> Result<f64, ExecError>,
    FallbackFn: FnOnce() -> Result<f64, ExecError>,
{
    *counters.eval_count += 1;

    let HotEvalRequest {
        entries,
        compile_threshold,
        key,
        expr,
    } = request;

    let threshold = compile_threshold.max(1);
    let mut cached_expr = None;
    let mut should_try_compile = false;
    {
        let entry = entries.entry(key.clone()).or_default();
        entry.eval_count += 1;
        if let Some(compiled) = &entry.compiled {
            cached_expr = Some(compiled.clone());
        } else {
            should_try_compile = !entry.permanently_interpreted && entry.eval_count >= threshold;
        }
    }

    if cached_expr.is_some() {
        *counters.cache_hits += 1;
    } else if should_try_compile {
        if let Some(compiled) = compile_expr(expr) {
            *counters.compile_count += 1;
            entries.entry(key.clone()).or_default().compiled = Some(compiled.clone());
            cached_expr = Some(compiled);
        } else {
            entries
                .entry(key.clone())
                .or_default()
                .permanently_interpreted = true;
        }
    }

    if let Some(compiled) = cached_expr {
        if let Ok(value) = eval_compiled(&compiled) {
            return Ok(value);
        }
    }

    *counters.fallback_count += 1;
    fallback_eval()
}

fn eval_hot_prune_with_cache<Compiled, CompileFn, EvalFn, FallbackFn>(
    profile: &mut JitProfile,
    request: HotEvalRequest<'_, PruneExprKey, Compiled>,
    compile_expr: CompileFn,
    eval_compiled: EvalFn,
    fallback_eval: FallbackFn,
) -> Result<f64, ExecError>
where
    Compiled: Clone,
    CompileFn: Fn(&ExprIR) -> Option<Compiled>,
    EvalFn: Fn(&Compiled) -> Result<f64, ExecError>,
    FallbackFn: FnOnce() -> Result<f64, ExecError>,
{
    eval_hot_with_cache(
        HotEvalCounters {
            eval_count: &mut profile.prune_eval_count,
            compile_count: &mut profile.prune_compile_count,
            cache_hits: &mut profile.prune_cache_hits,
            fallback_count: &mut profile.prune_fallback_count,
        },
        request,
        compile_expr,
        eval_compiled,
        fallback_eval,
    )
}

fn eval_hot_metric_with_cache<Compiled, CompileFn, EvalFn, FallbackFn>(
    profile: &mut JitProfile,
    request: HotEvalRequest<'_, MetricExprKey, Compiled>,
    compile_expr: CompileFn,
    eval_compiled: EvalFn,
    fallback_eval: FallbackFn,
) -> Result<f64, ExecError>
where
    Compiled: Clone,
    CompileFn: Fn(&ExprIR) -> Option<Compiled>,
    EvalFn: Fn(&Compiled) -> Result<f64, ExecError>,
    FallbackFn: FnOnce() -> Result<f64, ExecError>,
{
    eval_hot_with_cache(
        HotEvalCounters {
            eval_count: &mut profile.metric_eval_count,
            compile_count: &mut profile.metric_compile_count,
            cache_hits: &mut profile.metric_cache_hits,
            fallback_count: &mut profile.metric_fallback_count,
        },
        request,
        compile_expr,
        eval_compiled,
        fallback_eval,
    )
}

// ---------------------------------------------------------------------------
// Register-bytecode fallback (used when the `jit` feature is disabled).
// When `--features jit` is active, the real Cranelift backend in
// `jit_backend.rs` is used instead and all code below until
// `run_flow_ir_interpreter` is dead.
// ---------------------------------------------------------------------------

#[cfg(not(feature = "jit"))]
#[derive(Debug, Clone)]
enum CraneliftMetricOp {
    Const {
        dst: usize,
        value: f64,
    },
    Bool {
        dst: usize,
        value: bool,
    },
    MetricVar {
        dst: usize,
        name: String,
    },
    Unary {
        dst: usize,
        op: UnaryOpIR,
        src: usize,
    },
    Binary {
        dst: usize,
        op: BinaryOpIR,
        left: usize,
        right: usize,
    },
}

#[cfg(not(feature = "jit"))]
#[derive(Debug, Clone)]
struct CraneliftMetricProgram {
    ops: Vec<CraneliftMetricOp>,
    result_reg: usize,
    reg_count: usize,
}

#[cfg(not(feature = "jit"))]
impl CraneliftMetricProgram {
    fn eval(&self, ctx: &MetricContext) -> Result<f64, ExecError> {
        let mut regs = vec![0.0; self.reg_count.max(1)];
        for op in &self.ops {
            match op {
                CraneliftMetricOp::Const { dst, value } => regs[*dst] = *value,
                CraneliftMetricOp::Bool { dst, value } => {
                    regs[*dst] = if *value { 1.0 } else { 0.0 }
                }
                CraneliftMetricOp::MetricVar { dst, name } => {
                    regs[*dst] = ctx.metrics.get(name).copied().ok_or_else(|| {
                        ExecError::ValidationError(format!("unknown metric variable '{}'", name))
                    })?;
                }
                CraneliftMetricOp::Unary { dst, op, src } => {
                    regs[*dst] = match op {
                        UnaryOpIR::Neg => -regs[*src],
                        UnaryOpIR::Not => {
                            if regs[*src] == 0.0 {
                                1.0
                            } else {
                                0.0
                            }
                        }
                    };
                }
                CraneliftMetricOp::Binary {
                    dst,
                    op,
                    left,
                    right,
                } => {
                    regs[*dst] = eval_compiled_metric_binary(*op, regs[*left], regs[*right])?;
                }
            }
        }
        Ok(regs[self.result_reg])
    }
}

#[cfg(not(feature = "jit"))]
#[derive(Default)]
struct CraneliftMetricCompiler {
    ops: Vec<CraneliftMetricOp>,
    next_reg: usize,
}

#[cfg(not(feature = "jit"))]
impl CraneliftMetricCompiler {
    fn alloc(&mut self) -> usize {
        let out = self.next_reg;
        self.next_reg += 1;
        out
    }

    fn lower(&mut self, expr: &ExprIR) -> Option<usize> {
        match expr {
            ExprIR::Number(value) => {
                let dst = self.alloc();
                self.ops
                    .push(CraneliftMetricOp::Const { dst, value: *value });
                Some(dst)
            }
            ExprIR::Bool(value) => {
                let dst = self.alloc();
                self.ops
                    .push(CraneliftMetricOp::Bool { dst, value: *value });
                Some(dst)
            }
            ExprIR::Var(name) => {
                let dst = self.alloc();
                self.ops.push(CraneliftMetricOp::MetricVar {
                    dst,
                    name: name.clone(),
                });
                Some(dst)
            }
            ExprIR::Unary { op, expr } => {
                let src = self.lower(expr)?;
                let dst = self.alloc();
                self.ops
                    .push(CraneliftMetricOp::Unary { dst, op: *op, src });
                Some(dst)
            }
            ExprIR::Binary { op, left, right } => {
                let left_reg = self.lower(left)?;
                let right_reg = self.lower(right)?;
                let dst = self.alloc();
                self.ops.push(CraneliftMetricOp::Binary {
                    dst,
                    op: *op,
                    left: left_reg,
                    right: right_reg,
                });
                Some(dst)
            }
            ExprIR::Field { .. } | ExprIR::Call { .. } | ExprIR::Exists { .. } => None,
        }
    }
}

#[cfg(not(feature = "jit"))]
fn compile_metric_expr_jit(expr: &ExprIR) -> Option<CraneliftMetricProgram> {
    let mut compiler = CraneliftMetricCompiler::default();
    let result_reg = compiler.lower(expr)?;
    Some(CraneliftMetricProgram {
        ops: compiler.ops,
        result_reg,
        reg_count: compiler.next_reg,
    })
}

#[cfg(not(feature = "jit"))]
#[derive(Debug, Clone)]
enum CraneliftPruneOp {
    Const {
        dst: usize,
        value: f64,
    },
    Bool {
        dst: usize,
        value: bool,
    },
    ProbEdge {
        dst: usize,
    },
    Unary {
        dst: usize,
        op: UnaryOpIR,
        src: usize,
    },
    Binary {
        dst: usize,
        op: BinaryOpIR,
        left: usize,
        right: usize,
    },
}

#[cfg(not(feature = "jit"))]
#[derive(Debug, Clone)]
struct CraneliftPruneProgram {
    ops: Vec<CraneliftPruneOp>,
    result_reg: usize,
    reg_count: usize,
}

#[cfg(not(feature = "jit"))]
impl CraneliftPruneProgram {
    fn eval(&self, graph: &BeliefGraph, edge: EdgeId) -> Result<f64, ExecError> {
        let mut regs = vec![0.0; self.reg_count.max(1)];
        for op in &self.ops {
            match op {
                CraneliftPruneOp::Const { dst, value } => regs[*dst] = *value,
                CraneliftPruneOp::Bool { dst, value } => {
                    regs[*dst] = if *value { 1.0 } else { 0.0 }
                }
                CraneliftPruneOp::ProbEdge { dst } => {
                    regs[*dst] = graph.prob_mean(edge)?;
                }
                CraneliftPruneOp::Unary { dst, op, src } => {
                    regs[*dst] = eval_unary_op(op.to_ast(), regs[*src]);
                }
                CraneliftPruneOp::Binary {
                    dst,
                    op,
                    left,
                    right,
                } => {
                    regs[*dst] = eval_binary_op(op.to_ast(), regs[*left], regs[*right])?;
                }
            }
        }
        Ok(regs[self.result_reg])
    }
}

#[cfg(not(feature = "jit"))]
#[derive(Default)]
struct CraneliftPruneCompiler {
    ops: Vec<CraneliftPruneOp>,
    next_reg: usize,
}

#[cfg(not(feature = "jit"))]
impl CraneliftPruneCompiler {
    fn alloc(&mut self) -> usize {
        let out = self.next_reg;
        self.next_reg += 1;
        out
    }

    fn lower(&mut self, expr: &ExprIR) -> Option<usize> {
        match expr {
            ExprIR::Number(value) => {
                let dst = self.alloc();
                self.ops
                    .push(CraneliftPruneOp::Const { dst, value: *value });
                Some(dst)
            }
            ExprIR::Bool(value) => {
                let dst = self.alloc();
                self.ops.push(CraneliftPruneOp::Bool { dst, value: *value });
                Some(dst)
            }
            ExprIR::Unary { op, expr } => {
                let src = self.lower(expr)?;
                let dst = self.alloc();
                self.ops.push(CraneliftPruneOp::Unary { dst, op: *op, src });
                Some(dst)
            }
            ExprIR::Binary { op, left, right } => {
                let left_reg = self.lower(left)?;
                let right_reg = self.lower(right)?;
                let dst = self.alloc();
                self.ops.push(CraneliftPruneOp::Binary {
                    dst,
                    op: *op,
                    left: left_reg,
                    right: right_reg,
                });
                Some(dst)
            }
            ExprIR::Call { name, args } if name == "prob" => {
                if args.len() != 1 {
                    return None;
                }
                match &args[0] {
                    CallArgIR::Positional(ExprIR::Var(v)) if v == "edge" => {
                        let dst = self.alloc();
                        self.ops.push(CraneliftPruneOp::ProbEdge { dst });
                        Some(dst)
                    }
                    _ => None,
                }
            }
            ExprIR::Var(_) | ExprIR::Field { .. } | ExprIR::Call { .. } | ExprIR::Exists { .. } => {
                None
            }
        }
    }
}

#[cfg(not(feature = "jit"))]
fn compile_prune_expr_jit(expr: &ExprIR) -> Option<CraneliftPruneProgram> {
    let mut compiler = CraneliftPruneCompiler::default();
    let result_reg = compiler.lower(expr)?;
    Some(CraneliftPruneProgram {
        ops: compiler.ops,
        result_reg,
        reg_count: compiler.next_reg,
    })
}

#[cfg(not(feature = "jit"))]
fn eval_compiled_metric_binary(op: BinaryOpIR, left: f64, right: f64) -> Result<f64, ExecError> {
    let result = match op {
        BinaryOpIR::Add => left + right,
        BinaryOpIR::Sub => left - right,
        BinaryOpIR::Mul => left * right,
        BinaryOpIR::Div => {
            if right.abs() < 1e-15 {
                return Err(ExecError::ValidationError(
                    "division by zero in metric expression".into(),
                ));
            }
            left / right
        }
        BinaryOpIR::Eq => {
            if (left - right).abs() < 1e-12 {
                1.0
            } else {
                0.0
            }
        }
        BinaryOpIR::Ne => {
            if (left - right).abs() >= 1e-12 {
                1.0
            } else {
                0.0
            }
        }
        BinaryOpIR::Lt => {
            if left < right {
                1.0
            } else {
                0.0
            }
        }
        BinaryOpIR::Le => {
            if left <= right {
                1.0
            } else {
                0.0
            }
        }
        BinaryOpIR::Gt => {
            if left > right {
                1.0
            } else {
                0.0
            }
        }
        BinaryOpIR::Ge => {
            if left >= right {
                1.0
            } else {
                0.0
            }
        }
        BinaryOpIR::And => {
            if (left != 0.0) && (right != 0.0) {
                1.0
            } else {
                0.0
            }
        }
        BinaryOpIR::Or => {
            if (left != 0.0) || (right != 0.0) {
                1.0
            } else {
                0.0
            }
        }
    };
    if !result.is_finite() {
        return Err(ExecError::ValidationError(format!(
            "metric expression produced non-finite value: {}",
            result
        )));
    }
    Ok(result)
}

fn run_flow_ir_interpreter(
    program: &ProgramIR,
    flow_name: &str,
    prior: Option<&FlowResult>,
) -> Result<FlowResult, ExecError> {
    let optimized_program = program.optimized();
    let builder = StandardGraphBuilder;
    let mut expr_evaluator = InterpreterFlowExprEvaluator;
    run_flow_internal(
        &optimized_program,
        flow_name,
        prior,
        &builder,
        &mut expr_evaluator,
    )
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
    let mut expr_evaluator = InterpreterFlowExprEvaluator;
    run_flow_internal(&lowered, flow_name, prior, &builder, &mut expr_evaluator)
}

/// Internal flow execution logic shared between `run_flow` and `run_flow_with_builder`.
fn run_flow_internal<B: GraphBuilder, E: FlowExprEvaluator>(
    program: &ProgramIR,
    flow_name: &str,
    prior: Option<&FlowResult>,
    graph_builder: &B,
    expr_evaluator: &mut E,
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
                    let mut transform_ctx = TransformExecutionCtx {
                        rules_by_name: &rule_defs_by_name,
                        rule_globals: &rule_globals,
                        flow_name,
                        graph_name: &graph_def.name,
                        transform_idx,
                        result: &mut result,
                        expr_evaluator,
                    };
                    current = apply_transform(transform, &current, &mut transform_ctx)?;
                }
                current
            }
            GraphExprIR::SelectModel {
                candidates,
                criterion,
            } => select_model_graph(candidates, *criterion, &result.graphs)?,
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

    evaluate_metrics(flow, prior, &mut result, expr_evaluator)?;
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
                    TransformIR::Snapshot { .. }
                    | TransformIR::InferBeliefs
                    | TransformIR::PruneEdges { .. } => {}
                }
            }
        }
    }
    referenced
}

/// Build a deterministic graph execution plan for a flow.
///
/// The plan is dependency-driven:
/// - pipeline graphs are ready when their start graph has already been produced
/// - select_model graphs are ready when all candidate graphs have already been produced
/// - from_evidence/from_graph graphs are always ready
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
                GraphExprIR::SelectModel { candidates, .. } => candidates
                    .iter()
                    .all(|candidate| produced.contains(candidate.as_str())),
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
        GraphExprIR::SelectModel { .. } => Err(ExecError::Internal(
            "select_model evaluation should be handled in flow graph execution".into(),
        )),
        GraphExprIR::Pipeline { .. } => {
            // Pipelines require the start graph to already exist, so they're handled
            // separately in run_flow_internal after initial graphs are created
            Err(ExecError::Internal(
                "pipeline evaluation should be handled separately".into(),
            ))
        }
    }
}

fn select_model_graph(
    candidates: &[String],
    criterion: ModelSelectionCriterionIR,
    available_graphs: &HashMap<String, BeliefGraph>,
) -> Result<BeliefGraph, ExecError> {
    let runtime_criterion = match criterion {
        ModelSelectionCriterionIR::EdgeAic => EdgeModelCriterion::Aic,
        ModelSelectionCriterionIR::EdgeBic => EdgeModelCriterion::Bic,
    };

    let mut candidate_refs = Vec::with_capacity(candidates.len());
    for candidate in candidates {
        let graph = available_graphs.get(candidate).ok_or_else(|| {
            ExecError::Internal(format!(
                "select_model candidate graph '{}' is not available",
                candidate
            ))
        })?;
        candidate_refs.push((candidate.as_str(), graph));
    }

    let selected = select_best_graph(candidate_refs.into_iter(), runtime_criterion)?;
    available_graphs
        .get(&selected.name)
        .cloned()
        .ok_or_else(|| ExecError::Internal("selected model graph missing after selection".into()))
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

/// Mutable runtime context for a single pipeline transform.
struct TransformExecutionCtx<'a, E: FlowExprEvaluator> {
    rules_by_name: &'a HashMap<String, RuleDef>,
    rule_globals: &'a HashMap<String, f64>,
    flow_name: &'a str,
    graph_name: &'a str,
    transform_idx: usize,
    result: &'a mut FlowResult,
    expr_evaluator: &'a mut E,
}

/// Applies a single transform to a graph, returning a new graph.
fn apply_transform<E: FlowExprEvaluator>(
    transform: &TransformIR,
    graph: &BeliefGraph,
    ctx: &mut TransformExecutionCtx<'_, E>,
) -> Result<BeliefGraph, ExecError> {
    match transform {
        TransformIR::ApplyRule { rule, .. } => {
            let r = ctx
                .rules_by_name
                .get(rule)
                .ok_or_else(|| ExecError::Internal(format!("unknown rule '{}'", rule)))?;
            let (next, audit) = run_rule_for_each_with_globals_audit(graph, r, ctx.rule_globals)?;
            ctx.result.intervention_audit.push(InterventionAuditEvent {
                flow: ctx.flow_name.to_string(),
                graph: ctx.graph_name.to_string(),
                transform: format!("apply_rule#{}", ctx.transform_idx),
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
                let r = ctx.rules_by_name.get(rule_name).ok_or_else(|| {
                    ExecError::Internal(format!("unknown rule '{}' in ruleset", rule_name))
                })?;
                let (next, audit) =
                    run_rule_for_each_with_globals_audit(&current, r, ctx.rule_globals)?;
                ctx.result.intervention_audit.push(InterventionAuditEvent {
                    flow: ctx.flow_name.to_string(),
                    graph: ctx.graph_name.to_string(),
                    transform: format!("apply_ruleset#{}[{}]", ctx.transform_idx, rule_idx),
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
            ctx.result.snapshots.insert(name.clone(), snapshot_graph);
            Ok(graph.clone())
        }
        TransformIR::InferBeliefs => run_loopy_belief_propagation(graph),
        TransformIR::PruneEdges {
            edge_type,
            predicate,
        } => prune_edges_ir(
            graph,
            edge_type,
            predicate,
            ctx.flow_name,
            ctx.graph_name,
            ctx.transform_idx,
            ctx.expr_evaluator,
        ),
    }
}

/// Evaluates metrics against the last defined graph.
///
/// Metrics are evaluated in dependency order (earlier metrics are available to later ones).
/// Imported metrics from prior flows are available to all metric expressions.
fn evaluate_metrics<E: FlowExprEvaluator>(
    flow: &FlowIR,
    prior: Option<&FlowResult>,
    result: &mut FlowResult,
    expr_evaluator: &mut E,
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

    // Sequential metric evaluation.
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

        let v = expr_evaluator.eval_metric_expr(
            &flow.name,
            &m.name,
            &m.expr,
            target_graph,
            &registry,
            &ctx,
        )?;
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

#[cfg(test)]
fn prune_edges(
    input: &BeliefGraph,
    edge_type: &str,
    predicate: &ExprAst,
) -> Result<BeliefGraph, ExecError> {
    let predicate_ir = ExprIR::from(predicate);
    let mut expr_evaluator = InterpreterFlowExprEvaluator;
    prune_edges_ir(
        input,
        edge_type,
        &predicate_ir,
        "<test>",
        "<graph>",
        0,
        &mut expr_evaluator,
    )
}

fn prune_edges_ir<E: FlowExprEvaluator>(
    input: &BeliefGraph,
    edge_type: &str,
    predicate: &ExprIR,
    flow_name: &str,
    graph_name: &str,
    transform_idx: usize,
    expr_evaluator: &mut E,
) -> Result<BeliefGraph, ExecError> {
    let (mut keep, mut candidates) = classify_edges(input, edge_type);
    candidates.sort();

    for eid in candidates {
        let should_keep = expr_evaluator.eval_prune_predicate(
            flow_name,
            graph_name,
            transform_idx,
            predicate,
            input,
            eid,
        )? == 0.0;
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
    fn graph_execution_plan_handles_select_model_dependencies() {
        let flow = FlowIR {
            name: "F".into(),
            on_model: "M".into(),
            graphs: vec![
                grafial_ir::GraphDefIR {
                    name: "best".into(),
                    expr: GraphExprIR::SelectModel {
                        candidates: vec!["g2".into(), "g1".into()],
                        criterion: ModelSelectionCriterionIR::EdgeBic,
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
    fn run_flow_select_model_returns_deterministic_best_candidate() {
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
                        name: "g1".into(),
                        expr: GraphExpr::FromEvidence {
                            evidence: "Ev".into(),
                        },
                    },
                    GraphDef {
                        name: "g2".into(),
                        expr: GraphExpr::Pipeline {
                            start: "g1".into(),
                            transforms: vec![Transform::PruneEdges {
                                edge_type: "OTHER".into(),
                                predicate: ExprAst::Bool(true),
                            }],
                        },
                    },
                    GraphDef {
                        name: "best".into(),
                        expr: GraphExpr::SelectModel {
                            candidates: vec!["g2".into(), "g1".into()],
                            criterion: grafial_frontend::ast::ModelSelectionCriterion::EdgeBic,
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
        let g1 = result.graphs.get("g1").expect("g1");
        let best = result.graphs.get("best").expect("best");
        assert_eq!(best.edges().len(), g1.edges().len());
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

    #[cfg(feature = "jit")]
    fn build_phase10_hot_expr_program() -> ProgramIR {
        ProgramIR {
            schemas: vec![],
            belief_models: vec![],
            evidences: vec![],
            rules: vec![],
            flows: vec![FlowIR {
                name: "HotFlow".into(),
                on_model: "M".into(),
                graphs: vec![
                    grafial_ir::GraphDefIR {
                        name: "g0".into(),
                        expr: GraphExprIR::FromGraph("seed".into()),
                    },
                    grafial_ir::GraphDefIR {
                        name: "g1".into(),
                        expr: GraphExprIR::Pipeline {
                            start_graph: "g0".into(),
                            transforms: vec![TransformIR::PruneEdges {
                                edge_type: "REL".into(),
                                predicate: ExprIR::Binary {
                                    op: grafial_ir::BinaryOpIR::Lt,
                                    left: Box::new(ExprIR::Call {
                                        name: "prob".into(),
                                        args: vec![grafial_ir::CallArgIR::Positional(ExprIR::Var(
                                            "edge".into(),
                                        ))],
                                    }),
                                    right: Box::new(ExprIR::Number(0.5)),
                                },
                            }],
                        },
                    },
                ],
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
                            right: Box::new(ExprIR::Number(2.0)),
                        },
                    },
                    grafial_ir::MetricDefIR {
                        name: "m3".into(),
                        expr: ExprIR::Call {
                            name: "avg_degree".into(),
                            args: vec![
                                grafial_ir::CallArgIR::Positional(ExprIR::Var("Person".into())),
                                grafial_ir::CallArgIR::Positional(ExprIR::Var("REL".into())),
                            ],
                        },
                    },
                ],
                exports: vec![],
                metric_exports: vec![
                    grafial_ir::MetricExportDefIR {
                        metric: "m2".into(),
                        alias: "out".into(),
                    },
                    grafial_ir::MetricExportDefIR {
                        metric: "m3".into(),
                        alias: "avg".into(),
                    },
                ],
                metric_imports: vec![],
            }],
        }
    }

    #[test]
    #[cfg(feature = "jit")]
    fn cranelift_jit_backend_compiles_hot_exprs_with_interpreter_parity() {
        let program = build_phase10_hot_expr_program();
        let mut prior = FlowResult::default();
        prior.exports.insert("seed".into(), build_simple_graph());

        let expected =
            run_flow_ir_interpreter(&program, "HotFlow", Some(&prior)).expect("interpreter run");

        let backend = CraneliftJitExecutionBackend::new(JitConfig {
            metric_compile_threshold: 1,
            prune_compile_threshold: 1,
        });
        let first =
            run_flow_ir_with_backend(&program, "HotFlow", Some(&prior), &backend).expect("run 1");
        let second =
            run_flow_ir_with_backend(&program, "HotFlow", Some(&prior), &backend).expect("run 2");

        assert_eq!(first.metric_exports, expected.metric_exports);
        assert_eq!(second.metric_exports, expected.metric_exports);
        assert_eq!(first.graphs.get("g1").map(|g| g.edges().len()), Some(1));
        assert_eq!(second.graphs.get("g1").map(|g| g.edges().len()), Some(1));

        let profile = backend.profile_snapshot().expect("profile");
        assert!(profile.metric_compile_count >= 1);
        assert!(profile.prune_compile_count >= 1);
        assert!(profile.metric_cache_hits >= 1);
        assert!(profile.prune_cache_hits >= 1);
        assert!(profile.metric_fallback_count >= 2); // unsupported avg_degree call falls back

        let cranelift = CraneliftJitExecutionBackend::new(JitConfig {
            metric_compile_threshold: 1,
            prune_compile_threshold: 1,
        });
        let cranelift_result =
            run_flow_ir_with_backend(&program, "HotFlow", Some(&prior), &cranelift)
                .expect("cranelift run");
        assert_eq!(cranelift_result.metric_exports, expected.metric_exports);
        let cranelift_profile = cranelift.profile_snapshot().expect("cranelift profile");
        assert!(cranelift_profile.metric_compile_count >= 1);
        assert!(cranelift_profile.prune_compile_count >= 1);
    }

    /// Verify that the `jit` feature wires up real Cranelift compilation, not the bytecode
    /// interpreter.  This test only runs when `--features jit` is active.
    #[cfg(feature = "jit")]
    #[test]
    fn jit_feature_compiles_metric_and_prune_to_native_code() {
        use crate::engine::jit_backend;

        // Metric: a + 1 — fully supported by both bytecode and Cranelift.
        let metric_expr = ExprIR::Binary {
            op: grafial_ir::BinaryOpIR::Add,
            left: Box::new(ExprIR::Var("m1".into())),
            right: Box::new(ExprIR::Number(1.0)),
        };
        let compiled_metric =
            jit_backend::compile_metric_expr(&metric_expr).expect("should compile metric");
        let mut ctx = crate::metrics::MetricContext::default();
        ctx.metrics.insert("m1".into(), 3.0);
        assert_eq!(
            compiled_metric.eval(&ctx).expect("eval"),
            4.0,
            "JIT metric: 3 + 1 = 4"
        );

        // Prune: prob(edge) < 0.5
        let prune_expr = ExprIR::Binary {
            op: grafial_ir::BinaryOpIR::Lt,
            left: Box::new(ExprIR::Call {
                name: "prob".into(),
                args: vec![grafial_ir::CallArgIR::Positional(ExprIR::Var(
                    "edge".into(),
                ))],
            }),
            right: Box::new(ExprIR::Number(0.5)),
        };
        let compiled_prune =
            jit_backend::compile_prune_expr(&prune_expr).expect("should compile prune");
        // 0.3 < 0.5 → true (1.0)
        assert_eq!(
            compiled_prune.eval(0.3).expect("eval"),
            1.0,
            "JIT prune: 0.3 < 0.5 = true"
        );
        // 0.7 < 0.5 → false (0.0)
        assert_eq!(
            compiled_prune.eval(0.7).expect("eval"),
            0.0,
            "JIT prune: 0.7 < 0.5 = false"
        );
    }
}
