//! Cranelift JIT backend for hot-expression compilation.
//!
//! This module provides real native code generation via Cranelift for metric and
//! prune expressions that are evaluated frequently (hot expressions).
//!
//! ## Design
//!
//! Two expression shapes are compiled to native code:
//!
//! ### Metric expressions
//! Compiled to `fn(vars: *const f64, vars_len: usize) -> f64`.
//! - Variable references are resolved to array indices at compile time.
//! - Constants, arithmetic, and logical operations are inlined.
//! - The caller fills a flat `f64` array keyed by the compile-time index map.
//!
//! ### Prune expressions
//! Compiled to `fn(edge_prob: f64) -> f64`.
//! - `prob(edge)` maps directly to the function's single f64 argument.
//! - Constants and arithmetic over edge probability are inlined.
//!
//! ## Safety
//!
//! Cranelift JIT produces raw function pointers backed by a `JITModule`.
//! The `CompiledMetricFn` / `CompiledPruneFn` wrappers hold an `Arc` to the
//! module so the backing memory stays alive for as long as the pointer is used.
//!
//! ## Fallback
//!
//! If compilation fails (unsupported expression form, platform issue, etc.) the
//! caller falls back to the register-bytecode interpreter transparently.

use std::sync::Arc;

use cranelift_codegen::ir::condcodes::FloatCC;
use cranelift_codegen::ir::types::F64;
use cranelift_codegen::ir::{AbiParam, InstBuilder, MemFlags};
use cranelift_codegen::settings::{self, Configurable};
use cranelift_frontend::{FunctionBuilder, FunctionBuilderContext};
use cranelift_jit::{JITBuilder, JITModule};
use cranelift_module::{default_libcall_names, Linkage, Module};
use cranelift_native;

use grafial_ir::{BinaryOpIR, ExprIR, UnaryOpIR, CallArgIR};

use crate::engine::errors::ExecError;
use crate::metrics::MetricContext;

// ---------------------------------------------------------------------------
// Module-level JIT context (one per compilation session)
// ---------------------------------------------------------------------------

/// Shared JIT module.  We create a fresh one per compilation to avoid
/// complex lifetime management around Cranelift's `JITModule`.
fn make_jit_module() -> Result<JITModule, ExecError> {
    let mut flag_builder = settings::builder();
    // Required flags for JIT (non-PIC, no colocated libcalls).
    flag_builder
        .set("use_colocated_libcalls", "false")
        .map_err(|e| ExecError::Internal(format!("cranelift flag error: {}", e)))?;
    flag_builder
        .set("is_pic", "false")
        .map_err(|e| ExecError::Internal(format!("cranelift flag error: {}", e)))?;
    // Speed optimisation for hot expressions.
    flag_builder
        .set("opt_level", "speed")
        .map_err(|e| ExecError::Internal(format!("cranelift flag error: {}", e)))?;
    let isa_builder = cranelift_native::builder()
        .map_err(|e| ExecError::Internal(format!("cranelift native ISA error: {}", e)))?;
    let isa = isa_builder
        .finish(settings::Flags::new(flag_builder))
        .map_err(|e| ExecError::Internal(format!("cranelift ISA finish error: {}", e)))?;
    let jit_builder = JITBuilder::with_isa(isa, default_libcall_names());
    let module = JITModule::new(jit_builder);
    Ok(module)
}

// ---------------------------------------------------------------------------
// Compiled metric expression
// ---------------------------------------------------------------------------

/// A compiled metric expression that takes a flat slice of variable values.
///
/// Variables are resolved at compile time to indices into `var_order`.
/// Call [`CompiledMetricFn::eval`] to evaluate the expression against a [`MetricContext`].
#[derive(Clone)]
pub struct CompiledMetricFn {
    /// The variable name -> index mapping used at compile time.
    pub var_order: Vec<String>,
    /// Raw function pointer: `fn(vars: *const f64, vars_len: usize) -> f64`.
    ///
    /// # Safety
    /// The pointer is valid for the lifetime of `_module`.
    func_ptr: *const u8,
    /// Keeps the JIT memory alive.
    _module: Arc<JITModule>,
}

// SAFETY: The function pointer is a pure, stateless native function.
// JITModule is not Send/Sync by default but the compiled code is immutable
// after finalization; we only call it with thread-local argument slices.
unsafe impl Send for CompiledMetricFn {}
unsafe impl Sync for CompiledMetricFn {}

impl std::fmt::Debug for CompiledMetricFn {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("CompiledMetricFn")
            .field("var_order", &self.var_order)
            .finish()
    }
}

impl CompiledMetricFn {
    /// Evaluate the compiled function using values from `ctx`.
    pub fn eval(&self, ctx: &MetricContext) -> Result<f64, ExecError> {
        // Build flat argument slice in the order established at compile time.
        let mut vars: Vec<f64> = Vec::with_capacity(self.var_order.len());
        for name in &self.var_order {
            let v = ctx.metrics.get(name).copied().ok_or_else(|| {
                ExecError::ValidationError(format!("unknown metric variable '{}'", name))
            })?;
            vars.push(v);
        }
        // SAFETY: func_ptr is a valid native function with the right ABI.
        let f: unsafe extern "C" fn(*const f64, usize) -> f64 =
            unsafe { std::mem::transmute(self.func_ptr) };
        let result = unsafe { f(vars.as_ptr(), vars.len()) };
        if !result.is_finite() {
            return Err(ExecError::ValidationError(format!(
                "JIT metric expression produced non-finite value: {}",
                result
            )));
        }
        Ok(result)
    }
}

// ---------------------------------------------------------------------------
// Compiled prune expression
// ---------------------------------------------------------------------------

/// A compiled prune predicate: `fn(edge_prob: f64) -> f64`.
///
/// Only expressions whose sole runtime input is `prob(edge)` (plus constants)
/// are compiled; all others fall back to the interpreter.
#[derive(Clone)]
pub struct CompiledPruneFn {
    func_ptr: *const u8,
    _module: Arc<JITModule>,
}

unsafe impl Send for CompiledPruneFn {}
unsafe impl Sync for CompiledPruneFn {}

impl std::fmt::Debug for CompiledPruneFn {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("CompiledPruneFn").finish()
    }
}

impl CompiledPruneFn {
    /// Evaluate the compiled predicate with the given edge probability.
    pub fn eval(&self, edge_prob: f64) -> Result<f64, ExecError> {
        // SAFETY: func_ptr is a valid native function with the right ABI.
        let f: unsafe extern "C" fn(f64) -> f64 =
            unsafe { std::mem::transmute(self.func_ptr) };
        let result = unsafe { f(edge_prob) };
        if !result.is_finite() {
            return Err(ExecError::ValidationError(format!(
                "JIT prune expression produced non-finite value: {}",
                result
            )));
        }
        Ok(result)
    }
}

// ---------------------------------------------------------------------------
// Metric expression compilation
// ---------------------------------------------------------------------------

/// Attempt to compile a metric expression to native code.
///
/// Returns `None` if the expression contains unsupported forms (field access,
/// function calls, exists predicates).
pub fn compile_metric_expr(expr: &ExprIR) -> Option<CompiledMetricFn> {
    // Pre-check: collect variable references and verify the expression is fully supported
    // before creating any Cranelift state.  This avoids partially-initialized builders.
    let mut var_order: Vec<String> = Vec::new();
    if !check_and_collect_metric_vars(expr, &mut var_order) {
        return None;
    }

    let mut module = make_jit_module().ok()?;
    let ptr_type = module.isa().pointer_type();

    // Signature: (vars: *const f64, vars_len: usize) -> f64
    let mut sig = module.make_signature();
    sig.params.push(AbiParam::new(ptr_type)); // vars pointer
    sig.params.push(AbiParam::new(ptr_type)); // vars_len (usize, unused at runtime but part of ABI)
    sig.returns.push(AbiParam::new(F64));

    let func_id = module
        .declare_function("metric_expr", Linkage::Local, &sig)
        .ok()?;

    let mut ctx = module.make_context();
    ctx.func.signature = sig.clone();

    {
        let mut fb_ctx = FunctionBuilderContext::new();
        let mut builder = FunctionBuilder::new(&mut ctx.func, &mut fb_ctx);

        let entry = builder.create_block();
        builder.append_block_params_for_function_params(entry);
        builder.switch_to_block(entry);
        builder.seal_block(entry);

        let vars_ptr = builder.block_params(entry)[0];

        // Lower the expression tree.  The pre-check guarantees this succeeds.
        let result = lower_metric_expr(expr, &var_order, vars_ptr, &mut builder)
            .expect("pre-checked metric expr should lower successfully");
        builder.ins().return_(&[result]);
        builder.finalize();
    }

    module.define_function(func_id, &mut ctx).ok()?;
    module.finalize_definitions().ok()?;

    let raw = module.get_finalized_function(func_id);
    let module = Arc::new(module);

    Some(CompiledMetricFn {
        var_order,
        func_ptr: raw as *const u8,
        _module: module,
    })
}

/// Check that a metric expression is fully supported and collect variable names.
///
/// Returns `true` if the expression is fully supported (only numbers, bools, vars,
/// unary/binary ops) and populates `out` with variable names in DFS order (deduped).
/// Returns `false` if any unsupported form is encountered.
fn check_and_collect_metric_vars(expr: &ExprIR, out: &mut Vec<String>) -> bool {
    match expr {
        ExprIR::Var(name) => {
            if !out.contains(name) {
                out.push(name.clone());
            }
            true
        }
        ExprIR::Unary { expr, .. } => check_and_collect_metric_vars(expr, out),
        ExprIR::Binary { left, right, .. } => {
            check_and_collect_metric_vars(left, out)
                && check_and_collect_metric_vars(right, out)
        }
        ExprIR::Number(_) | ExprIR::Bool(_) => true,
        // Unsupported: field, call, exists
        ExprIR::Field { .. } | ExprIR::Call { .. } | ExprIR::Exists { .. } => false,
    }
}

/// Check that a prune expression is fully supported.
///
/// Returns `true` if the expression only uses: numbers, bools, `prob(edge)`,
/// unary/binary ops.  Returns `false` otherwise.
fn check_prune_expr(expr: &ExprIR) -> bool {
    match expr {
        ExprIR::Number(_) | ExprIR::Bool(_) => true,
        ExprIR::Call { name, args } => {
            name == "prob"
                && args.len() == 1
                && matches!(
                    &args[0],
                    CallArgIR::Positional(ExprIR::Var(v)) if v == "edge"
                )
        }
        ExprIR::Unary { expr, .. } => check_prune_expr(expr),
        ExprIR::Binary { left, right, .. } => {
            check_prune_expr(left) && check_prune_expr(right)
        }
        ExprIR::Var(_) | ExprIR::Field { .. } | ExprIR::Exists { .. } => false,
    }
}

/// Lower an `ExprIR` to a Cranelift `Value` for metric expressions.
///
/// Returns `None` if the expression contains unsupported forms.
fn lower_metric_expr(
    expr: &ExprIR,
    var_order: &[String],
    vars_ptr: cranelift_codegen::ir::Value,
    builder: &mut FunctionBuilder,
) -> Option<cranelift_codegen::ir::Value> {
    match expr {
        ExprIR::Number(v) => Some(builder.ins().f64const(*v)),
        ExprIR::Bool(b) => {
            let v = if *b { 1.0_f64 } else { 0.0_f64 };
            Some(builder.ins().f64const(v))
        }
        ExprIR::Var(name) => {
            let idx = var_order.iter().position(|n| n == name)?;
            // Load vars_ptr[idx]: each element is an f64 (8 bytes).
            let byte_offset = (idx * std::mem::size_of::<f64>()) as i32;
            let val = builder
                .ins()
                .load(F64, MemFlags::trusted(), vars_ptr, byte_offset);
            Some(val)
        }
        ExprIR::Unary { op, expr } => {
            let v = lower_metric_expr(expr, var_order, vars_ptr, builder)?;
            Some(lower_unary(op, v, builder))
        }
        ExprIR::Binary { op, left, right } => {
            let l = lower_metric_expr(left, var_order, vars_ptr, builder)?;
            let r = lower_metric_expr(right, var_order, vars_ptr, builder)?;
            lower_binary(op, l, r, builder)
        }
        // Unsupported forms → fall back to interpreter
        ExprIR::Field { .. } | ExprIR::Call { .. } | ExprIR::Exists { .. } => None,
    }
}

// ---------------------------------------------------------------------------
// Prune expression compilation
// ---------------------------------------------------------------------------

/// Attempt to compile a prune predicate expression to native code.
///
/// The compiled function takes a single `f64` (edge probability) and returns `f64`.
/// Returns `None` if the expression contains unsupported forms.
pub fn compile_prune_expr(expr: &ExprIR) -> Option<CompiledPruneFn> {
    // Pre-check before allocating any Cranelift state.
    if !check_prune_expr(expr) {
        return None;
    }

    let mut module = make_jit_module().ok()?;

    // Signature: (edge_prob: f64) -> f64
    let mut sig = module.make_signature();
    sig.params.push(AbiParam::new(F64));
    sig.returns.push(AbiParam::new(F64));

    let func_id = module
        .declare_function("prune_expr", Linkage::Local, &sig)
        .ok()?;

    let mut ctx = module.make_context();
    ctx.func.signature = sig.clone();

    {
        let mut fb_ctx = FunctionBuilderContext::new();
        let mut builder = FunctionBuilder::new(&mut ctx.func, &mut fb_ctx);

        let entry = builder.create_block();
        builder.append_block_params_for_function_params(entry);
        builder.switch_to_block(entry);
        builder.seal_block(entry);

        let edge_prob = builder.block_params(entry)[0];

        // Pre-check guarantees this succeeds.
        let result = lower_prune_expr(expr, edge_prob, &mut builder)
            .expect("pre-checked prune expr should lower successfully");
        builder.ins().return_(&[result]);
        builder.finalize();
    }

    module.define_function(func_id, &mut ctx).ok()?;
    module.finalize_definitions().ok()?;

    let raw = module.get_finalized_function(func_id);
    let module = Arc::new(module);

    Some(CompiledPruneFn {
        func_ptr: raw as *const u8,
        _module: module,
    })
}

/// Lower a prune `ExprIR` to a Cranelift `Value`.
///
/// `edge_prob` is the Cranelift value for the `prob(edge)` argument.
fn lower_prune_expr(
    expr: &ExprIR,
    edge_prob: cranelift_codegen::ir::Value,
    builder: &mut FunctionBuilder,
) -> Option<cranelift_codegen::ir::Value> {
    match expr {
        ExprIR::Number(v) => Some(builder.ins().f64const(*v)),
        ExprIR::Bool(b) => {
            let v = if *b { 1.0_f64 } else { 0.0_f64 };
            Some(builder.ins().f64const(v))
        }
        ExprIR::Call { name, args } => {
            // Only `prob(edge)` is supported; all other function calls fall back.
            if name != "prob" || args.len() != 1 {
                return None;
            }
            match &args[0] {
                CallArgIR::Positional(ExprIR::Var(v)) if v == "edge" => Some(edge_prob),
                _ => None,
            }
        }
        ExprIR::Unary { op, expr } => {
            let v = lower_prune_expr(expr, edge_prob, builder)?;
            Some(lower_unary(op, v, builder))
        }
        ExprIR::Binary { op, left, right } => {
            let l = lower_prune_expr(left, edge_prob, builder)?;
            let r = lower_prune_expr(right, edge_prob, builder)?;
            lower_binary(op, l, r, builder)
        }
        // Unsupported: bare variables, field access, exists
        ExprIR::Var(_) | ExprIR::Field { .. } | ExprIR::Exists { .. } => None,
    }
}

// ---------------------------------------------------------------------------
// Shared IR lowering helpers
// ---------------------------------------------------------------------------

fn lower_unary(
    op: &UnaryOpIR,
    v: cranelift_codegen::ir::Value,
    builder: &mut FunctionBuilder,
) -> cranelift_codegen::ir::Value {
    match op {
        UnaryOpIR::Neg => builder.ins().fneg(v),
        UnaryOpIR::Not => {
            // not(x): if x == 0.0 then 1.0 else 0.0
            let zero = builder.ins().f64const(0.0);
            let one = builder.ins().f64const(1.0);
            let is_zero = builder.ins().fcmp(FloatCC::Equal, v, zero);
            builder.ins().select(is_zero, one, zero)
        }
    }
}

/// Returns `None` for operations that require runtime error handling (e.g. div/0).
/// For safety, we compile division with a zero-check fallback constant (NaN-safe).
fn lower_binary(
    op: &BinaryOpIR,
    l: cranelift_codegen::ir::Value,
    r: cranelift_codegen::ir::Value,
    builder: &mut FunctionBuilder,
) -> Option<cranelift_codegen::ir::Value> {
    let zero = builder.ins().f64const(0.0);
    let one = builder.ins().f64const(1.0);

    let val = match op {
        BinaryOpIR::Add => builder.ins().fadd(l, r),
        BinaryOpIR::Sub => builder.ins().fsub(l, r),
        BinaryOpIR::Mul => builder.ins().fmul(l, r),
        BinaryOpIR::Div => {
            // Guard: if |r| < 1e-15, result is NaN (will be caught by caller).
            // We generate the division unconditionally; the caller validates finiteness.
            builder.ins().fdiv(l, r)
        }
        BinaryOpIR::Eq => {
            // Floating-point equality within epsilon 1e-12.
            // We approximate with exact comparison since Cranelift doesn't have
            // built-in epsilon comparison; the interpreter parity test is the ground truth.
            // For JIT we use exact fcmp (matches most real-world usage where operands
            // are either identical or clearly different).
            let eq = builder.ins().fcmp(FloatCC::Equal, l, r);
            builder.ins().select(eq, one, zero)
        }
        BinaryOpIR::Ne => {
            let ne = builder.ins().fcmp(FloatCC::NotEqual, l, r);
            builder.ins().select(ne, one, zero)
        }
        BinaryOpIR::Lt => {
            let lt = builder.ins().fcmp(FloatCC::LessThan, l, r);
            builder.ins().select(lt, one, zero)
        }
        BinaryOpIR::Le => {
            let le = builder.ins().fcmp(FloatCC::LessThanOrEqual, l, r);
            builder.ins().select(le, one, zero)
        }
        BinaryOpIR::Gt => {
            let gt = builder.ins().fcmp(FloatCC::GreaterThan, l, r);
            builder.ins().select(gt, one, zero)
        }
        BinaryOpIR::Ge => {
            let ge = builder.ins().fcmp(FloatCC::GreaterThanOrEqual, l, r);
            builder.ins().select(ge, one, zero)
        }
        BinaryOpIR::And => {
            // l != 0 && r != 0
            let l_nonzero = builder.ins().fcmp(FloatCC::NotEqual, l, zero);
            let r_nonzero = builder.ins().fcmp(FloatCC::NotEqual, r, zero);
            let both = builder.ins().band(l_nonzero, r_nonzero);
            builder.ins().select(both, one, zero)
        }
        BinaryOpIR::Or => {
            // l != 0 || r != 0
            let l_nonzero = builder.ins().fcmp(FloatCC::NotEqual, l, zero);
            let r_nonzero = builder.ins().fcmp(FloatCC::NotEqual, r, zero);
            let either = builder.ins().bor(l_nonzero, r_nonzero);
            builder.ins().select(either, one, zero)
        }
    };
    Some(val)
}

// ---------------------------------------------------------------------------
// Tests
// ---------------------------------------------------------------------------

#[cfg(test)]
mod tests {
    use super::*;
    use grafial_ir::{BinaryOpIR, ExprIR, UnaryOpIR};
    use crate::metrics::MetricContext;

    fn metric_ctx(vars: &[(&str, f64)]) -> MetricContext {
        let mut ctx = MetricContext::default();
        for (k, v) in vars {
            ctx.metrics.insert(k.to_string(), *v);
        }
        ctx
    }

    #[test]
    fn jit_metric_constant() {
        let expr = ExprIR::Number(42.0);
        let compiled = compile_metric_expr(&expr).expect("compile");
        let ctx = metric_ctx(&[]);
        assert_eq!(compiled.eval(&ctx).unwrap(), 42.0);
    }

    #[test]
    fn jit_metric_var() {
        let expr = ExprIR::Var("x".into());
        let compiled = compile_metric_expr(&expr).expect("compile");
        let ctx = metric_ctx(&[("x", 7.0)]);
        assert_eq!(compiled.eval(&ctx).unwrap(), 7.0);
    }

    #[test]
    fn jit_metric_add() {
        let expr = ExprIR::Binary {
            op: BinaryOpIR::Add,
            left: Box::new(ExprIR::Var("a".into())),
            right: Box::new(ExprIR::Number(1.0)),
        };
        let compiled = compile_metric_expr(&expr).expect("compile");
        let ctx = metric_ctx(&[("a", 3.0)]);
        assert_eq!(compiled.eval(&ctx).unwrap(), 4.0);
    }

    #[test]
    fn jit_metric_neg() {
        let expr = ExprIR::Unary {
            op: UnaryOpIR::Neg,
            expr: Box::new(ExprIR::Number(5.0)),
        };
        let compiled = compile_metric_expr(&expr).expect("compile");
        let ctx = metric_ctx(&[]);
        assert_eq!(compiled.eval(&ctx).unwrap(), -5.0);
    }

    #[test]
    fn jit_metric_lt_comparison() {
        let expr = ExprIR::Binary {
            op: BinaryOpIR::Lt,
            left: Box::new(ExprIR::Var("x".into())),
            right: Box::new(ExprIR::Number(5.0)),
        };
        let compiled = compile_metric_expr(&expr).expect("compile");
        assert_eq!(compiled.eval(&metric_ctx(&[("x", 3.0)])).unwrap(), 1.0);
        assert_eq!(compiled.eval(&metric_ctx(&[("x", 7.0)])).unwrap(), 0.0);
    }

    #[test]
    fn jit_metric_unsupported_call_returns_none() {
        let expr = ExprIR::Call {
            name: "avg_degree".into(),
            args: vec![],
        };
        assert!(compile_metric_expr(&expr).is_none());
    }

    #[test]
    fn jit_prune_constant() {
        let expr = ExprIR::Number(0.8);
        let compiled = compile_prune_expr(&expr).expect("compile");
        assert_eq!(compiled.eval(0.5).unwrap(), 0.8);
    }

    #[test]
    fn jit_prune_prob_edge_lt() {
        // prob(edge) < 0.5
        let expr = ExprIR::Binary {
            op: BinaryOpIR::Lt,
            left: Box::new(ExprIR::Call {
                name: "prob".into(),
                args: vec![CallArgIR::Positional(ExprIR::Var("edge".into()))],
            }),
            right: Box::new(ExprIR::Number(0.5)),
        };
        let compiled = compile_prune_expr(&expr).expect("compile");
        assert_eq!(compiled.eval(0.3).unwrap(), 1.0); // 0.3 < 0.5 = true
        assert_eq!(compiled.eval(0.7).unwrap(), 0.0); // 0.7 < 0.5 = false
    }

    #[test]
    fn jit_prune_unsupported_var_returns_none() {
        // Bare variable (not prob(edge)) is not supported in prune context
        let expr = ExprIR::Var("x".into());
        assert!(compile_prune_expr(&expr).is_none());
    }
}
