//! Native kernel compilation for rule predicates and actions.
//!
//! This module provides Cranelift-based JIT compilation for performance-critical
//! rule evaluation paths. Rule predicates (where clauses) and simple actions are
//! compiled to native code, while complex graph traversal remains in the interpreter.
//!
//! ## Architecture
//!
//! - **Predicate kernels**: Compile where clause expressions to native functions
//! - **Action kernels**: Compile simple graph updates (set_expectation, observe) to native code
//! - **Hybrid execution**: Pattern matching and graph traversal stay interpreted
//! - **Cache management**: Hot rules are compiled and cached, cold rules stay interpreted
//!
//! ## Feature gating
//!
//! Rule kernel compilation is behind the `jit` feature flag. When disabled,
//! all rules execute through the interpreter.

#[cfg(feature = "jit")]
use cranelift_codegen::ir::condcodes::FloatCC;
#[cfg(feature = "jit")]
use cranelift_codegen::ir::{AbiParam, InstBuilder, MemFlags, Value, types};
#[cfg(feature = "jit")]
use cranelift_codegen::settings::{self, Configurable};
#[cfg(feature = "jit")]
use cranelift_frontend::{FunctionBuilder, FunctionBuilderContext};
#[cfg(feature = "jit")]
use cranelift_jit::{JITBuilder, JITModule};
#[cfg(feature = "jit")]
use cranelift_module::{default_libcall_names, Linkage, Module};
#[cfg(feature = "jit")]
use cranelift_native;
use grafial_frontend::ast::{BinaryOp, ExprAst, UnaryOp};
use std::collections::HashMap;
use std::sync::Arc;

use crate::engine::errors::ExecError;

/// Threshold for compiling a rule predicate to native code.
/// Rules executed fewer times than this stay interpreted.
const JIT_COMPILATION_THRESHOLD: usize = 10;

/// Function signature for compiled rule predicates.
/// Takes pointers to binding arrays and returns boolean (0 or 1).
///
/// Signature: fn(node_bindings: *const f64, edge_bindings: *const f64, globals: *const f64) -> u8
pub type CompiledPredicateFn = unsafe extern "C" fn(*const f64, *const f64, *const f64) -> u8;

/// Function signature for compiled action kernels.
/// Takes binding pointers and graph reference, returns error code.
///
/// Signature: fn(graph: *mut (), node_bindings: *const f64, edge_bindings: *const f64, globals: *const f64) -> i32
pub type CompiledActionFn =
    unsafe extern "C" fn(*mut (), *const f64, *const f64, *const f64) -> i32;

/// Compiled rule kernel with predicate and action functions.
#[derive(Clone)]
pub struct CompiledRuleKernel {
    /// Optional compiled predicate (where clause)
    pub predicate: Option<CompiledPredicateFn>,
    /// Optional compiled actions
    pub actions: Vec<CompiledActionFn>,
    /// Variable name to index mappings for marshalling
    pub node_var_indices: HashMap<String, usize>,
    pub edge_var_indices: HashMap<String, usize>,
    pub global_var_indices: HashMap<String, usize>,
}

/// Rule compilation cache and statistics.
pub struct RuleKernelCache {
    /// Cached compiled kernels by rule name
    kernels: HashMap<String, Arc<CompiledRuleKernel>>,
    /// Execution counts for adaptive compilation
    exec_counts: HashMap<String, usize>,
    #[cfg(feature = "jit")]
    jit_module: Option<JITModule>,
}

impl RuleKernelCache {
    #[cfg(feature = "jit")]
    fn init_jit_module() -> Result<JITModule, ExecError> {
        let mut flag_builder = settings::builder();
        flag_builder.set("use_colocated_libcalls", "false")
            .map_err(|e| ExecError::Internal(format!("Failed to set cranelift flag: {}", e)))?;
        flag_builder.set("is_pic", "false")
            .map_err(|e| ExecError::Internal(format!("Failed to set cranelift flag: {}", e)))?;

        let isa_builder = cranelift_native::builder()
            .map_err(|msg| ExecError::Internal(format!("Host machine not supported: {}", msg)))?;

        let isa = isa_builder.finish(settings::Flags::new(flag_builder))
            .map_err(|e| ExecError::Internal(format!("Failed to create ISA: {}", e)))?;

        let builder = JITBuilder::with_isa(isa, default_libcall_names());
        Ok(JITModule::new(builder))
    }

    pub fn new() -> Self {
        #[cfg(feature = "jit")]
        let jit_module = {
            match Self::init_jit_module() {
                Ok(module) => Some(module),
                Err(e) => {
                    eprintln!("Failed to initialize JIT module: {}", e);
                    None
                }
            }
        };

        Self {
            kernels: HashMap::new(),
            exec_counts: HashMap::new(),
            #[cfg(feature = "jit")]
            jit_module,
        }
    }

    /// Check if a rule should be compiled based on execution frequency.
    pub fn should_compile(&mut self, rule_name: &str) -> bool {
        let count = self.exec_counts.entry(rule_name.to_string()).or_insert(0);
        *count += 1;
        *count == JIT_COMPILATION_THRESHOLD
    }

    /// Get a compiled kernel if available.
    pub fn get(&self, rule_name: &str) -> Option<Arc<CompiledRuleKernel>> {
        self.kernels.get(rule_name).cloned()
    }

    /// Compile and cache a rule's predicate and actions.
    #[cfg(feature = "jit")]
    pub fn compile_rule(
        &mut self,
        rule_name: &str,
        predicate: Option<&ExprAst>,
        _actions: &[grafial_frontend::ast::ActionStmt],
    ) -> Result<Arc<CompiledRuleKernel>, ExecError> {
        let module = self.jit_module.as_mut().ok_or_else(|| {
            ExecError::Internal("JIT module not initialized".to_string())
        })?;

        // Extract variable names from predicate
        let mut node_vars = Vec::new();
        let mut edge_vars = Vec::new();
        let mut global_vars = Vec::new();
        if let Some(pred) = predicate {
            extract_variables(pred, &mut node_vars, &mut edge_vars, &mut global_vars);
        }

        // Create index mappings
        let node_var_indices: HashMap<String, usize> = node_vars
            .iter()
            .enumerate()
            .map(|(i, name)| (name.clone(), i))
            .collect();
        let edge_var_indices: HashMap<String, usize> = edge_vars
            .iter()
            .enumerate()
            .map(|(i, name)| (name.clone(), i))
            .collect();
        let global_var_indices: HashMap<String, usize> = global_vars
            .iter()
            .enumerate()
            .map(|(i, name)| (name.clone(), i))
            .collect();

        // Compile predicate if present
        let compiled_predicate = if let Some(pred) = predicate {
            Some(compile_predicate_kernel(
                module,
                pred,
                &node_var_indices,
                &edge_var_indices,
                &global_var_indices,
            )?)
        } else {
            None
        };

        // TODO: Compile actions (more complex due to graph mutations)
        let compiled_actions = Vec::new();

        let kernel = Arc::new(CompiledRuleKernel {
            predicate: compiled_predicate,
            actions: compiled_actions,
            node_var_indices,
            edge_var_indices,
            global_var_indices,
        });

        self.kernels.insert(rule_name.to_string(), kernel.clone());
        Ok(kernel)
    }

    #[cfg(not(feature = "jit"))]
    pub fn compile_rule(
        &mut self,
        _rule_name: &str,
        _predicate: Option<&ExprAst>,
        _actions: &[grafial_frontend::ast::ActionStmt],
    ) -> Result<Arc<CompiledRuleKernel>, ExecError> {
        Err(ExecError::Internal(
            "JIT compilation not available without 'jit' feature".to_string(),
        ))
    }
}

/// Extract variable names from an expression for binding marshalling.
fn extract_variables(
    expr: &ExprAst,
    node_vars: &mut Vec<String>,
    edge_vars: &mut Vec<String>,
    global_vars: &mut Vec<String>,
) {
    match expr {
        ExprAst::Var(name) => {
            // Heuristic: uppercase = node var, lowercase = edge/global var
            if name.chars().next().map_or(false, |c| c.is_uppercase()) {
                if !node_vars.contains(name) {
                    node_vars.push(name.clone());
                }
            } else if !edge_vars.contains(name) && !global_vars.contains(name) {
                // We don't know if it's edge or global yet, assume edge for now
                edge_vars.push(name.clone());
            }
        }
        ExprAst::Binary { left, right, .. } => {
            extract_variables(left, node_vars, edge_vars, global_vars);
            extract_variables(right, node_vars, edge_vars, global_vars);
        }
        ExprAst::Unary { expr, .. } => {
            extract_variables(expr, node_vars, edge_vars, global_vars);
        }
        ExprAst::Field { target, .. } => {
            extract_variables(target, node_vars, edge_vars, global_vars);
        }
        ExprAst::Call { args, .. } => {
            for arg in args {
                if let grafial_frontend::ast::CallArg::Positional(expr) = arg {
                    extract_variables(expr, node_vars, edge_vars, global_vars);
                }
            }
        }
        _ => {}
    }
}

/// Compile a predicate expression to native code.
#[cfg(feature = "jit")]
fn compile_predicate_kernel(
    module: &mut JITModule,
    expr: &ExprAst,
    node_var_indices: &HashMap<String, usize>,
    edge_var_indices: &HashMap<String, usize>,
    global_var_indices: &HashMap<String, usize>,
) -> Result<CompiledPredicateFn, ExecError> {
    let mut ctx = module.make_context();
    let mut func_ctx = FunctionBuilderContext::new();

    // Define function signature: (node_bindings, edge_bindings, globals) -> u8
    ctx.func.signature.params.push(AbiParam::new(types::I64)); // node_bindings ptr
    ctx.func.signature.params.push(AbiParam::new(types::I64)); // edge_bindings ptr
    ctx.func.signature.params.push(AbiParam::new(types::I64)); // globals ptr
    ctx.func.signature.returns.push(AbiParam::new(types::I8)); // boolean result

    let mut builder = FunctionBuilder::new(&mut ctx.func, &mut func_ctx);
    let entry_block = builder.create_block();
    builder.append_block_params_for_function_params(entry_block);
    builder.switch_to_block(entry_block);
    builder.seal_block(entry_block);

    let node_ptr = builder.block_params(entry_block)[0];
    let edge_ptr = builder.block_params(entry_block)[1];
    let global_ptr = builder.block_params(entry_block)[2];

    // Compile the expression
    let result = compile_expr(
        &mut builder,
        expr,
        node_ptr,
        edge_ptr,
        global_ptr,
        node_var_indices,
        edge_var_indices,
        global_var_indices,
    )?;

    // Convert to boolean (0 or 1)
    let zero = builder.ins().f64const(0.0);
    let cmp = builder.ins().fcmp(FloatCC::NotEqual, result, zero);
    // Use select to convert boolean to integer (0 or 1)
    let one = builder.ins().iconst(types::I8, 1);
    let zero_i8 = builder.ins().iconst(types::I8, 0);
    let bool_result = builder.ins().select(cmp, one, zero_i8);

    builder.ins().return_(&[bool_result]);
    builder.finalize();

    // Compile to machine code
    // Use a simple counter for function names
    use std::sync::atomic::{AtomicU64, Ordering};
    static FUNCTION_COUNTER: AtomicU64 = AtomicU64::new(0);
    let counter = FUNCTION_COUNTER.fetch_add(1, Ordering::Relaxed);
    let func_name = format!("rule_predicate_{}", counter);

    let id = module
        .declare_function(
            &func_name,
            Linkage::Local,
            &ctx.func.signature,
        )
        .map_err(|e| ExecError::Internal(format!("Failed to declare function: {}", e)))?;

    // Enable verification for debugging
    #[cfg(debug_assertions)]
    {
        use cranelift_codegen::verify_function;
        verify_function(&ctx.func, module.isa())
            .map_err(|e| ExecError::Internal(format!("Function verification failed: {}", e)))?;
    }

    module
        .define_function(id, &mut ctx)
        .map_err(|e| ExecError::Internal(format!("Failed to define function: {}", e)))?;

    module.clear_context(&mut ctx);
    module.finalize_definitions().unwrap();

    let code_ptr = module.get_finalized_function(id);
    Ok(unsafe { std::mem::transmute(code_ptr) })
}

/// Compile an expression to Cranelift IR.
#[cfg(feature = "jit")]
fn compile_expr(
    builder: &mut FunctionBuilder<'_>,
    expr: &ExprAst,
    node_ptr: Value,
    edge_ptr: Value,
    global_ptr: Value,
    node_var_indices: &HashMap<String, usize>,
    edge_var_indices: &HashMap<String, usize>,
    global_var_indices: &HashMap<String, usize>,
) -> Result<Value, ExecError> {
    match expr {
        ExprAst::Number(n) => Ok(builder.ins().f64const(*n)),

        ExprAst::Var(name) => {
            // Load variable from appropriate binding array
            if let Some(&idx) = node_var_indices.get(name) {
                let offset = (idx * 8) as i32; // f64 is 8 bytes
                let addr = builder.ins().iadd_imm(node_ptr, offset as i64);
                Ok(builder.ins().load(types::F64, MemFlags::new(), addr, 0))
            } else if let Some(&idx) = edge_var_indices.get(name) {
                let offset = (idx * 8) as i32;
                let addr = builder.ins().iadd_imm(edge_ptr, offset as i64);
                Ok(builder.ins().load(types::F64, MemFlags::new(), addr, 0))
            } else if let Some(&idx) = global_var_indices.get(name) {
                let offset = (idx * 8) as i32;
                let addr = builder.ins().iadd_imm(global_ptr, offset as i64);
                Ok(builder.ins().load(types::F64, MemFlags::new(), addr, 0))
            } else {
                Err(ExecError::Internal(format!(
                    "Variable '{}' not found in bindings",
                    name
                )))
            }
        }

        ExprAst::Binary { op, left, right } => {
            let lhs = compile_expr(
                builder,
                left,
                node_ptr,
                edge_ptr,
                global_ptr,
                node_var_indices,
                edge_var_indices,
                global_var_indices,
            )?;
            let rhs = compile_expr(
                builder,
                right,
                node_ptr,
                edge_ptr,
                global_ptr,
                node_var_indices,
                edge_var_indices,
                global_var_indices,
            )?;

            let result = match op {
                BinaryOp::Add => builder.ins().fadd(lhs, rhs),
                BinaryOp::Sub => builder.ins().fsub(lhs, rhs),
                BinaryOp::Mul => builder.ins().fmul(lhs, rhs),
                BinaryOp::Div => builder.ins().fdiv(lhs, rhs),
                BinaryOp::Lt => {
                    let cmp = builder.ins().fcmp(FloatCC::LessThan, lhs, rhs);
                    let extended = builder.ins().uextend(types::I32, cmp);
                    builder.ins().fcvt_from_uint(types::F64, extended)
                }
                BinaryOp::Le => {
                    let cmp = builder.ins().fcmp(FloatCC::LessThanOrEqual, lhs, rhs);
                    let extended = builder.ins().uextend(types::I32, cmp);
                    builder.ins().fcvt_from_uint(types::F64, extended)
                }
                BinaryOp::Gt => {
                    let cmp = builder.ins().fcmp(FloatCC::GreaterThan, lhs, rhs);
                    let extended = builder.ins().uextend(types::I32, cmp);
                    builder.ins().fcvt_from_uint(types::F64, extended)
                }
                BinaryOp::Ge => {
                    let cmp = builder.ins().fcmp(FloatCC::GreaterThanOrEqual, lhs, rhs);
                    let extended = builder.ins().uextend(types::I32, cmp);
                    builder.ins().fcvt_from_uint(types::F64, extended)
                }
                BinaryOp::Eq => {
                    let cmp = builder.ins().fcmp(FloatCC::Equal, lhs, rhs);
                    let extended = builder.ins().uextend(types::I32, cmp);
                    builder.ins().fcvt_from_uint(types::F64, extended)
                }
                BinaryOp::Ne => {
                    let cmp = builder.ins().fcmp(FloatCC::NotEqual, lhs, rhs);
                    let extended = builder.ins().uextend(types::I32, cmp);
                    builder.ins().fcvt_from_uint(types::F64, extended)
                }
                BinaryOp::And => {
                    // Logical AND: both non-zero
                    let zero = builder.ins().f64const(0.0);
                    let lhs_true = builder.ins().fcmp(FloatCC::NotEqual, lhs, zero);
                    let rhs_true = builder.ins().fcmp(FloatCC::NotEqual, rhs, zero);
                    let both = builder.ins().band(lhs_true, rhs_true);
                    let extended = builder.ins().uextend(types::I32, both);
                    builder.ins().fcvt_from_uint(types::F64, extended)
                }
                BinaryOp::Or => {
                    // Logical OR: either non-zero
                    let zero = builder.ins().f64const(0.0);
                    let lhs_true = builder.ins().fcmp(FloatCC::NotEqual, lhs, zero);
                    let rhs_true = builder.ins().fcmp(FloatCC::NotEqual, rhs, zero);
                    let either = builder.ins().bor(lhs_true, rhs_true);
                    let extended = builder.ins().uextend(types::I32, either);
                    builder.ins().fcvt_from_uint(types::F64, extended)
                }
            };
            Ok(result)
        }

        ExprAst::Unary { op, expr } => {
            let val = compile_expr(
                builder,
                expr,
                node_ptr,
                edge_ptr,
                global_ptr,
                node_var_indices,
                edge_var_indices,
                global_var_indices,
            )?;

            let result = match op {
                UnaryOp::Neg => builder.ins().fneg(val),
                UnaryOp::Not => {
                    let zero = builder.ins().f64const(0.0);
                    let is_zero = builder.ins().fcmp(FloatCC::Equal, val, zero);
                    let extended = builder.ins().uextend(types::I32, is_zero);
                    builder.ins().fcvt_from_uint(types::F64, extended)
                }
            };
            Ok(result)
        }

        // Complex expressions (Field, Call, etc.) would require graph access
        // For now, we only compile simple arithmetic/logical expressions
        _ => Err(ExecError::Internal(
            "Complex expressions not yet supported in JIT compilation".to_string(),
        )),
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use grafial_frontend::ast::ExprAst;

    #[test]
    fn test_variable_extraction() {
        let expr = ExprAst::Binary {
            op: BinaryOp::Add,
            left: Box::new(ExprAst::Var("NodeA".to_string())),
            right: Box::new(ExprAst::Var("edge_weight".to_string())),
        };

        let mut node_vars = Vec::new();
        let mut edge_vars = Vec::new();
        let mut global_vars = Vec::new();

        extract_variables(&expr, &mut node_vars, &mut edge_vars, &mut global_vars);

        assert_eq!(node_vars, vec!["NodeA"]);
        assert_eq!(edge_vars, vec!["edge_weight"]);
        assert!(global_vars.is_empty());
    }

    #[test]
    #[cfg(feature = "jit")]
    fn test_simple_predicate_compilation() {
        let mut cache = RuleKernelCache::new();

        // Test 1: Simple numeric comparison (no variables) - should always compile
        let predicate1 = ExprAst::Binary {
            op: BinaryOp::Gt,
            left: Box::new(ExprAst::Number(10.0)),
            right: Box::new(ExprAst::Number(5.0)),
        };

        let result = cache.compile_rule("test_rule_1", Some(&predicate1), &[]);
        assert!(result.is_ok(), "Compilation of numeric predicate failed: {:?}", result.err());

        let kernel = result.unwrap();
        assert!(kernel.predicate.is_some());

        // Test 2: Complex expression (should be compilable)
        let predicate2 = ExprAst::Binary {
            op: BinaryOp::And,
            left: Box::new(ExprAst::Binary {
                op: BinaryOp::Gt,
                left: Box::new(ExprAst::Number(10.0)),
                right: Box::new(ExprAst::Number(5.0)),
            }),
            right: Box::new(ExprAst::Binary {
                op: BinaryOp::Lt,
                left: Box::new(ExprAst::Number(3.0)),
                right: Box::new(ExprAst::Number(8.0)),
            }),
        };

        let result2 = cache.compile_rule("test_rule_2", Some(&predicate2), &[]);
        assert!(result2.is_ok(), "Compilation of complex predicate failed: {:?}", result2.err());
    }
}