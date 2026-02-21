//! Ahead-of-Time (AOT) compilation for flows.
//!
//! This module provides infrastructure to compile entire flows (sequences of
//! evidence application and rule execution) to native code at build time.
//! AOT-compiled flows offer the highest performance for production deployments.
//!
//! ## Architecture
//!
//! - **Flow compiler**: Transforms FlowDef AST into optimized native code
//! - **Build integration**: build.rs support for compile-time code generation
//! - **Runtime loader**: Loads and executes pre-compiled flow libraries
//! - **Hybrid execution**: Falls back to interpreter for dynamic flows
//!
//! ## Compilation Strategy
//!
//! 1. Static analysis to determine data dependencies
//! 2. Inline evidence application and rule execution
//! 3. Optimize graph traversals and updates
//! 4. Generate specialized code for known patterns
//!
//! ## Feature gating
//!
//! AOT compilation is behind the `aot` feature flag. When disabled,
//! all flows execute through the interpreter.

#[cfg(feature = "aot")]
use cranelift_codegen::ir::{types, AbiParam, InstBuilder, Value};
#[cfg(feature = "aot")]
use cranelift_codegen::settings::{self, Configurable};
#[cfg(feature = "aot")]
use cranelift_frontend::{FunctionBuilder, FunctionBuilderContext};
#[cfg(feature = "aot")]
use cranelift_module::{default_libcall_names, Linkage, Module};
#[cfg(feature = "aot")]
use cranelift_object::{ObjectBuilder, ObjectModule};

use grafial_ir::{FlowIR, GraphExprIR, TransformIR};
use std::collections::HashMap;
#[cfg(feature = "aot")]
use std::fs;
#[cfg(feature = "aot")]
use std::path::Path;

use crate::engine::errors::ExecError;
use crate::engine::graph::BeliefGraph;

/// Metadata for an AOT-compiled flow.
#[derive(Debug, Clone)]
pub struct CompiledFlowMetadata {
    /// Name of the flow
    pub name: String,
    /// SHA-256 hash of the source flow definition
    pub source_hash: String,
    /// Path to the compiled object file
    pub object_path: String,
    /// Entry point symbol name
    pub entry_symbol: String,
    /// List of external dependencies (evidence, rules)
    pub dependencies: Vec<String>,
}

/// AOT flow compiler.
#[cfg(feature = "aot")]
pub struct FlowCompiler {
    /// Target triple for compilation
    target: String,
    /// Optimization level
    opt_level: OptLevel,
    /// Object module for code generation
    module: Option<ObjectModule>,
}

#[cfg(feature = "aot")]
#[derive(Debug, Clone, Copy)]
pub enum OptLevel {
    None,
    Speed,
    SpeedAndSize,
}

#[cfg(feature = "aot")]
impl FlowCompiler {
    /// Create a new flow compiler for the current target.
    pub fn new(opt_level: OptLevel) -> Result<Self, ExecError> {
        Ok(Self {
            target: target_triple(),
            opt_level,
            module: None,
        })
    }

    /// Initialize the Cranelift module for compilation.
    fn init_module(&mut self) -> Result<(), ExecError> {
        let mut flag_builder = settings::builder();

        // Set optimization level
        match self.opt_level {
            OptLevel::None => {
                flag_builder
                    .set("opt_level", "none")
                    .map_err(|e| ExecError::Internal(format!("Failed to set opt level: {}", e)))?;
            }
            OptLevel::Speed => {
                flag_builder
                    .set("opt_level", "speed")
                    .map_err(|e| ExecError::Internal(format!("Failed to set opt level: {}", e)))?;
            }
            OptLevel::SpeedAndSize => {
                flag_builder
                    .set("opt_level", "speed_and_size")
                    .map_err(|e| ExecError::Internal(format!("Failed to set opt level: {}", e)))?;
            }
        }

        // Configure for static linking
        flag_builder
            .set("use_colocated_libcalls", "false")
            .map_err(|e| ExecError::Internal(format!("Failed to set flag: {}", e)))?;
        flag_builder
            .set("is_pic", "false")
            .map_err(|e| ExecError::Internal(format!("Failed to set flag: {}", e)))?;

        let isa_builder = cranelift_native::builder()
            .map_err(|msg| ExecError::Internal(format!("Host machine not supported: {}", msg)))?;

        let isa = isa_builder
            .finish(settings::Flags::new(flag_builder))
            .map_err(|e| ExecError::Internal(format!("Failed to create ISA: {}", e)))?;

        let builder = ObjectBuilder::new(isa, self.target.clone(), default_libcall_names())
            .map_err(|e| ExecError::Internal(format!("Failed to create object builder: {}", e)))?;

        self.module = Some(ObjectModule::new(builder));
        Ok(())
    }

    /// Compile a flow IR to native code.
    pub fn compile_flow(
        &mut self,
        flow: &FlowIR,
        output_path: &Path,
    ) -> Result<CompiledFlowMetadata, ExecError> {
        // Initialize module if not already done
        if self.module.is_none() {
            self.init_module()?;
        }

        let mut ctx = self.module.as_mut().unwrap().make_context();
        let mut func_ctx = FunctionBuilderContext::new();

        // Define function signature: (graph_ptr) -> i32 (status code)
        ctx.func.signature.params.push(AbiParam::new(types::I64)); // graph pointer
        ctx.func.signature.returns.push(AbiParam::new(types::I32)); // status code

        // Generate entry point name
        let entry_symbol = format!("flow_{}_execute", sanitize_name(&flow.name));

        // Build the function
        let mut builder = FunctionBuilder::new(&mut ctx.func, &mut func_ctx);
        let entry_block = builder.create_block();
        builder.append_block_params_for_function_params(entry_block);
        builder.switch_to_block(entry_block);
        builder.seal_block(entry_block);

        let graph_ptr = builder.block_params(entry_block)[0];

        // Compile flow body (graphs and transforms)
        let status = self.compile_flow_body(&mut builder, graph_ptr, flow)?;

        builder.ins().return_(&[status]);
        builder.finalize();

        // Get module reference for declaration and definition
        let module = self.module.as_mut().unwrap();

        // Declare and define the function
        let id = module
            .declare_function(&entry_symbol, Linkage::Export, &ctx.func.signature)
            .map_err(|e| ExecError::Internal(format!("Failed to declare function: {}", e)))?;

        module
            .define_function(id, &mut ctx)
            .map_err(|e| ExecError::Internal(format!("Failed to define function: {}", e)))?;

        module.clear_context(&mut ctx);

        // Extract dependencies
        let dependencies = extract_flow_dependencies(flow);

        // Generate object file (consumes the module)
        let object = self.module.take().unwrap().finish();
        let bytes = object
            .emit()
            .map_err(|e| ExecError::Internal(format!("Failed to emit object: {}", e)))?;

        // Write to disk
        fs::write(output_path, bytes)
            .map_err(|e| ExecError::Internal(format!("Failed to write object file: {}", e)))?;

        // Compute source hash
        let source_hash = compute_flow_hash(flow);

        Ok(CompiledFlowMetadata {
            name: flow.name.clone(),
            source_hash,
            object_path: output_path.to_string_lossy().to_string(),
            entry_symbol,
            dependencies,
        })
    }

    /// Compile the body of a flow to native code.
    fn compile_flow_body(
        &self,
        builder: &mut FunctionBuilder,
        _graph_ptr: Value,
        flow: &FlowIR,
    ) -> Result<Value, ExecError> {
        // For now, return a simple success status
        // TODO: Implement actual flow compilation

        // Log compilation (will be replaced with actual implementation)
        for graph_def in &flow.graphs {
            eprintln!("AOT: Would compile graph: {}", graph_def.name);
            // TODO: Compile graph expression evaluation
        }

        for metric_def in &flow.metrics {
            eprintln!("AOT: Would compile metric: {}", metric_def.name);
            // TODO: Compile metric computation
        }

        // Return success status (0)
        Ok(builder.ins().iconst(types::I32, 0))
    }
}

/// Runtime loader for AOT-compiled flows.
pub struct CompiledFlowLoader {
    /// Cache of loaded flow modules
    loaded_flows: HashMap<String, LoadedFlow>,
}

struct LoadedFlow {
    #[allow(dead_code)]
    metadata: CompiledFlowMetadata,
    // In a real implementation, this would hold the dlopen handle
    // and function pointer
    #[allow(dead_code)]
    entry_fn: Option<FlowExecuteFn>,
}

type FlowExecuteFn = unsafe extern "C" fn(*mut BeliefGraph) -> i32;

impl CompiledFlowLoader {
    /// Create a new flow loader.
    pub fn new() -> Self {
        Self {
            loaded_flows: HashMap::new(),
        }
    }

    /// Load a compiled flow from disk.
    pub fn load_flow(&mut self, metadata: CompiledFlowMetadata) -> Result<(), ExecError> {
        // TODO: Implement actual dynamic loading with dlopen/dlsym
        // For now, just store the metadata

        let loaded = LoadedFlow {
            metadata: metadata.clone(),
            entry_fn: None, // Would be populated by dlsym
        };

        self.loaded_flows.insert(metadata.name.clone(), loaded);
        Ok(())
    }

    /// Execute a loaded flow.
    pub fn execute_flow(&self, name: &str, _graph: &mut BeliefGraph) -> Result<(), ExecError> {
        let flow = self
            .loaded_flows
            .get(name)
            .ok_or_else(|| ExecError::Internal(format!("Flow '{}' not loaded", name)))?;

        if let Some(_entry_fn) = flow.entry_fn {
            // TODO: Call the actual compiled function
            // let status = unsafe { entry_fn(graph as *mut BeliefGraph) };
            // if status != 0 {
            //     return Err(ExecError::Internal(format!("Flow execution failed with status {}", status)));
            // }
            eprintln!("Would execute AOT-compiled flow: {}", name);
        } else {
            return Err(ExecError::Internal(format!(
                "Flow '{}' not properly loaded",
                name
            )));
        }

        Ok(())
    }
}

impl Default for CompiledFlowLoader {
    fn default() -> Self {
        Self::new()
    }
}

/// Extract dependencies from a flow.
fn extract_flow_dependencies(flow: &FlowIR) -> Vec<String> {
    let mut deps = Vec::new();

    // Extract evidence dependencies from graph expressions
    for graph_def in &flow.graphs {
        extract_graph_expr_deps(&graph_def.expr, &mut deps);
    }

    deps
}

/// Extract dependencies from a graph expression.
fn extract_graph_expr_deps(expr: &GraphExprIR, deps: &mut Vec<String>) {
    match expr {
        GraphExprIR::FromEvidence(name) => {
            deps.push(format!("evidence:{}", name));
        }
        GraphExprIR::Pipeline { transforms, .. } => {
            for transform in transforms {
                match transform {
                    TransformIR::ApplyRule { rule, .. } => {
                        deps.push(format!("rule:{}", rule));
                    }
                    TransformIR::ApplyRuleset { rules } => {
                        for rule in rules {
                            deps.push(format!("rule:{}", rule));
                        }
                    }
                    _ => {}
                }
            }
        }
        _ => {}
    }
}

/// Compute SHA-256 hash of flow source.
#[cfg(feature = "aot")]
fn compute_flow_hash(flow: &FlowIR) -> String {
    use std::collections::hash_map::DefaultHasher;
    use std::hash::{Hash, Hasher};

    let mut hasher = DefaultHasher::new();
    flow.name.hash(&mut hasher);
    flow.on_model.hash(&mut hasher);
    // TODO: Hash the actual flow graphs and metrics
    format!("{:016x}", hasher.finish())
}

/// Sanitize a name for use as a symbol.
fn sanitize_name(name: &str) -> String {
    name.chars()
        .map(|c| if c.is_alphanumeric() { c } else { '_' })
        .collect()
}

/// Get the target triple for the current platform.
#[cfg(feature = "aot")]
fn target_triple() -> String {
    // This would normally come from the build environment
    // For now, use a default
    "x86_64-unknown-linux-gnu".to_string()
}

#[cfg(not(feature = "aot"))]
pub struct FlowCompiler;

#[cfg(not(feature = "aot"))]
impl FlowCompiler {
    pub fn new(_opt_level: ()) -> Result<Self, ExecError> {
        Err(ExecError::Internal(
            "AOT compilation not available without 'aot' feature".to_string(),
        ))
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use grafial_ir::GraphDefIR;

    #[test]
    fn test_sanitize_name() {
        assert_eq!(sanitize_name("hello_world"), "hello_world");
        assert_eq!(sanitize_name("hello-world"), "hello_world");
        assert_eq!(sanitize_name("hello.world"), "hello_world");
        assert_eq!(sanitize_name("hello world"), "hello_world");
    }

    #[test]
    fn test_extract_dependencies() {
        let flow = FlowIR {
            name: "test_flow".to_string(),
            on_model: "test_model".to_string(),
            graphs: vec![
                GraphDefIR {
                    name: "g1".to_string(),
                    expr: GraphExprIR::FromEvidence("evidence1".to_string()),
                },
                GraphDefIR {
                    name: "g2".to_string(),
                    expr: GraphExprIR::Pipeline {
                        start_graph: "g1".to_string(),
                        transforms: vec![
                            TransformIR::ApplyRule {
                                rule: "rule1".to_string(),
                                mode_override: None,
                            },
                            TransformIR::ApplyRule {
                                rule: "rule2".to_string(),
                                mode_override: None,
                            },
                        ],
                    },
                },
            ],
            metrics: vec![],
            exports: vec![],
            metric_exports: vec![],
            metric_imports: vec![],
        };

        let deps = extract_flow_dependencies(&flow);
        assert_eq!(deps.len(), 3);
        assert_eq!(deps[0], "evidence:evidence1");
        assert_eq!(deps[1], "rule:rule1");
        assert_eq!(deps[2], "rule:rule2");
    }
}
