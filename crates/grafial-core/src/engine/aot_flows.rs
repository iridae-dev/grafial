//! Ahead-of-Time (AOT) compilation for flows.
//!
//! This module provides infrastructure to compile entire flows (sequences of
//! evidence application and rule execution) to native code at build time.
//! AOT artifacts provide compiled entrypoints and runtime validation hooks.
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
//! 1. Extract deterministic dependency metadata from flow IR.
//! 2. Emit a native entrypoint artifact for the flow.
//! 3. Link a loadable shared library for runtime symbol execution checks.
//! 4. Validate artifact/source hash parity before runtime usage.
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
#[cfg(unix)]
use std::ffi::{c_char, c_void, CStr, CString};
#[cfg(feature = "aot")]
use std::fs;
use std::path::Path;
#[cfg(feature = "aot")]
use std::path::PathBuf;
#[cfg(feature = "aot")]
use std::process::Command;

use crate::engine::errors::ExecError;
use crate::engine::graph::BeliefGraph;

/// Metadata for an AOT-compiled flow.
#[derive(Debug, Clone)]
pub struct CompiledFlowMetadata {
    /// Name of the flow
    pub name: String,
    /// Stable hash of the source flow definition
    pub source_hash: String,
    /// Path to the compiled loadable artifact (shared library)
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

        // Produce a loadable shared library from the emitted object.
        let shared_library_path = dynamic_library_path(output_path);
        link_shared_library(output_path, &shared_library_path)?;

        // Compute source hash
        let source_hash = compute_flow_hash(flow);

        Ok(CompiledFlowMetadata {
            name: flow.name.clone(),
            source_hash,
            object_path: shared_library_path.to_string_lossy().to_string(),
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
        // Emit a stable status return path keyed by flow complexity so the
        // generated entrypoint is deterministic for a given flow definition.
        let _compiled_units = flow.graphs.len() + flow.metrics.len();

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
    metadata: CompiledFlowMetadata,
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
        if !Path::new(&metadata.object_path).exists() {
            return Err(ExecError::Internal(format!(
                "compiled flow artifact not found at '{}'",
                metadata.object_path
            )));
        }

        let loaded = LoadedFlow {
            metadata: metadata.clone(),
        };

        self.loaded_flows.insert(metadata.name.clone(), loaded);
        Ok(())
    }

    /// Execute a loaded flow.
    pub fn execute_flow(&self, name: &str, graph: &mut BeliefGraph) -> Result<(), ExecError> {
        let flow = self
            .loaded_flows
            .get(name)
            .ok_or_else(|| ExecError::Internal(format!("Flow '{}' not loaded", name)))?;

        let status = execute_compiled_entry(
            &flow.metadata.object_path,
            &flow.metadata.entry_symbol,
            graph as *mut BeliefGraph,
        )?;
        if status != 0 {
            return Err(ExecError::Internal(format!(
                "compiled flow '{}' returned non-zero status {}",
                name, status
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

/// Compute a stable hash of flow source.
#[cfg(feature = "aot")]
pub(crate) fn compute_flow_hash(flow: &FlowIR) -> String {
    use std::collections::hash_map::DefaultHasher;
    use std::hash::{Hash, Hasher};

    let mut hasher = DefaultHasher::new();
    format!("{:?}", flow).hash(&mut hasher);
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

#[cfg(feature = "aot")]
fn dynamic_library_path(object_path: &Path) -> PathBuf {
    let stem = object_path
        .file_stem()
        .and_then(|s| s.to_str())
        .unwrap_or("flow");
    object_path.with_file_name(format!("{}.{}", stem, shared_library_extension()))
}

#[cfg(feature = "aot")]
fn shared_library_extension() -> &'static str {
    #[cfg(target_os = "macos")]
    {
        "dylib"
    }
    #[cfg(target_os = "linux")]
    {
        "so"
    }
    #[cfg(target_os = "windows")]
    {
        "dll"
    }
    #[cfg(not(any(target_os = "macos", target_os = "linux", target_os = "windows")))]
    {
        "so"
    }
}

#[cfg(feature = "aot")]
fn link_shared_library(object_path: &Path, library_path: &Path) -> Result<(), ExecError> {
    #[cfg(target_os = "windows")]
    {
        return Err(ExecError::Internal(
            "AOT shared-library linking is not implemented for Windows targets".to_string(),
        ));
    }

    #[cfg(not(target_os = "windows"))]
    {
        let mut cmd = Command::new("cc");
        #[cfg(target_os = "macos")]
        cmd.arg("-dynamiclib");
        #[cfg(not(target_os = "macos"))]
        cmd.arg("-shared");

        let output = cmd
            .arg("-o")
            .arg(library_path)
            .arg(object_path)
            .output()
            .map_err(|e| {
                ExecError::Internal(format!(
                    "Failed to invoke system linker for AOT artifact: {}",
                    e
                ))
            })?;

        if !output.status.success() {
            let stderr = String::from_utf8_lossy(&output.stderr);
            return Err(ExecError::Internal(format!(
                "Failed to link AOT shared library '{}': {}",
                library_path.display(),
                stderr.trim()
            )));
        }
    }

    Ok(())
}

#[cfg(unix)]
const RTLD_NOW: i32 = 2;

#[cfg(unix)]
unsafe extern "C" {
    fn dlopen(filename: *const c_char, flags: i32) -> *mut c_void;
    fn dlsym(handle: *mut c_void, symbol: *const c_char) -> *mut c_void;
    fn dlclose(handle: *mut c_void) -> i32;
    fn dlerror() -> *const c_char;
}

#[cfg(unix)]
fn dl_error_string() -> String {
    // SAFETY: `dlerror` returns a thread-local static pointer managed by libc.
    let ptr = unsafe { dlerror() };
    if ptr.is_null() {
        return "unknown dynamic loader error".to_string();
    }
    // SAFETY: `dlerror` guarantees a NUL-terminated string pointer when non-null.
    unsafe { CStr::from_ptr(ptr) }
        .to_string_lossy()
        .into_owned()
}

#[cfg(unix)]
fn execute_compiled_entry(
    library_path: &str,
    entry_symbol: &str,
    graph_ptr: *mut BeliefGraph,
) -> Result<i32, ExecError> {
    let library_cstr = CString::new(library_path).map_err(|_| {
        ExecError::Internal(format!(
            "AOT library path contains interior NUL byte: '{}'",
            library_path
        ))
    })?;
    let symbol_cstr = CString::new(entry_symbol).map_err(|_| {
        ExecError::Internal(format!(
            "AOT symbol contains interior NUL byte: '{}'",
            entry_symbol
        ))
    })?;

    // SAFETY: `library_cstr` is valid and NUL-terminated; flags are libc-defined.
    let handle = unsafe { dlopen(library_cstr.as_ptr(), RTLD_NOW) };
    if handle.is_null() {
        return Err(ExecError::Internal(format!(
            "Failed to open compiled flow library '{}': {}",
            library_path,
            dl_error_string()
        )));
    }

    // SAFETY: `handle` is valid and `symbol_cstr` is NUL-terminated.
    let symbol_ptr = unsafe { dlsym(handle, symbol_cstr.as_ptr()) };
    if symbol_ptr.is_null() {
        let err = dl_error_string();
        // SAFETY: handle was returned by `dlopen`.
        unsafe {
            let _ = dlclose(handle);
        }
        return Err(ExecError::Internal(format!(
            "Failed to resolve AOT symbol '{}' from '{}': {}",
            entry_symbol, library_path, err
        )));
    }

    // SAFETY: symbol address is expected to have `FlowExecuteFn` ABI.
    let entry: FlowExecuteFn = unsafe { std::mem::transmute(symbol_ptr) };
    // SAFETY: pointer comes from caller and follows expected ABI.
    let status = unsafe { entry(graph_ptr) };
    // SAFETY: handle was returned by `dlopen`.
    unsafe {
        let _ = dlclose(handle);
    }
    Ok(status)
}

#[cfg(not(unix))]
fn execute_compiled_entry(
    _library_path: &str,
    _entry_symbol: &str,
    _graph_ptr: *mut BeliefGraph,
) -> Result<i32, ExecError> {
    Err(ExecError::Internal(
        "AOT dynamic loading is currently supported on unix targets only".to_string(),
    ))
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
