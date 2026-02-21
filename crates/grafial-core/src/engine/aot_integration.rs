//! Integration layer for AOT-compiled flows with the runtime.
//!
//! This module provides the runtime integration for executing AOT-compiled
//! flows alongside interpreted flows, with automatic fallback when compiled
//! versions are not available.

use std::collections::HashMap;
use std::sync::RwLock;

#[cfg(feature = "aot")]
use crate::engine::aot_flows::FlowCompiler;
use crate::engine::aot_flows::{CompiledFlowLoader, CompiledFlowMetadata};
use crate::engine::errors::ExecError;
use crate::engine::flow_exec;
use grafial_ir::{FlowIR, ProgramIR};

/// Global registry of AOT-compiled flows.
pub struct AotFlowRegistry {
    /// Loader for compiled flows
    loader: RwLock<CompiledFlowLoader>,
    /// Metadata for available compiled flows
    metadata: RwLock<HashMap<String, CompiledFlowMetadata>>,
}

impl AotFlowRegistry {
    /// Create a new AOT flow registry.
    pub fn new() -> Self {
        Self {
            loader: RwLock::new(CompiledFlowLoader::new()),
            metadata: RwLock::new(HashMap::new()),
        }
    }

    /// Initialize the registry with pre-compiled flows.
    ///
    /// This is typically called at program startup to load flows
    /// that were compiled at build time.
    pub fn init_from_manifest(&self) -> Result<(), ExecError> {
        // Check for compiled flows directory from build script
        if let Ok(compiled_dir) = std::env::var("GRAFIAL_COMPILED_FLOWS_DIR") {
            self.load_compiled_flows(&compiled_dir)?;
        }
        Ok(())
    }

    /// Load compiled flows from a directory.
    fn load_compiled_flows(&self, dir: &str) -> Result<(), ExecError> {
        use std::fs;
        use std::path::Path;

        let manifest_path = Path::new(dir).parent().unwrap().join("flow_manifest.txt");

        if !manifest_path.exists() {
            return Ok(());
        }

        let manifest = fs::read_to_string(&manifest_path)
            .map_err(|e| ExecError::Internal(format!("Failed to read manifest: {}", e)))?;

        let mut loader = self.loader.write().unwrap();
        let mut metadata_map = self.metadata.write().unwrap();

        for line in manifest.lines() {
            let parts: Vec<&str> = line.split(':').collect();
            if parts.len() != 2 {
                continue;
            }

            let name = parts[0].to_string();
            let object_path = parts[1].to_string();

            // Create metadata (in real implementation, would read from object file)
            let metadata = CompiledFlowMetadata {
                name: name.clone(),
                source_hash: "placeholder".to_string(),
                object_path,
                entry_symbol: format!("flow_{}_execute", sanitize_name(&name)),
                dependencies: Vec::new(),
            };

            loader.load_flow(metadata.clone())?;
            metadata_map.insert(name, metadata);
        }

        Ok(())
    }

    /// Check if a flow has an AOT-compiled version available.
    pub fn has_compiled(&self, name: &str) -> bool {
        self.metadata.read().unwrap().contains_key(name)
    }

    /// Execute a flow, using the compiled version if available.
    pub fn execute_flow(
        &self,
        flow: &FlowIR,
        program: &ProgramIR,
    ) -> Result<flow_exec::FlowResult, ExecError> {
        // Check if we have a compiled version
        if self.has_compiled(&flow.name) {
            // Verify the compiled version matches the source
            if self.verify_flow_hash(flow) {
                // TODO: Execute the compiled version
                eprintln!("Would execute compiled flow '{}'", flow.name);
                // For now, fall through to interpreted execution
            } else {
                eprintln!(
                    "Warning: Compiled flow '{}' is out of date, falling back to interpreter",
                    flow.name
                );
            }
        }

        // Fall back to interpreted execution
        flow_exec::run_flow_ir(program, &flow.name, None)
    }

    /// Verify that a compiled flow matches its source.
    fn verify_flow_hash(&self, _flow: &FlowIR) -> bool {
        // TODO: Implement actual hash verification
        // For now, always consider compiled versions valid
        true
    }
}

impl Default for AotFlowRegistry {
    fn default() -> Self {
        Self::new()
    }
}

lazy_static::lazy_static! {
    /// Global AOT flow registry instance.
    pub static ref AOT_REGISTRY: AotFlowRegistry = {
        let registry = AotFlowRegistry::new();
        // Initialize with pre-compiled flows
        if let Err(e) = registry.init_from_manifest() {
            eprintln!("Warning: Failed to load AOT-compiled flows: {}", e);
        }
        registry
    };
}

/// Execute a flow with AOT optimization if available.
///
/// This is the main entry point for flow execution that automatically
/// uses compiled versions when available.
pub fn execute_flow_optimized(
    flow: &FlowIR,
    program: &ProgramIR,
) -> Result<flow_exec::FlowResult, ExecError> {
    AOT_REGISTRY.execute_flow(flow, program)
}

/// Compile a flow to native code for future execution.
///
/// This can be used at runtime to JIT-compile frequently used flows.
#[cfg(feature = "aot")]
pub fn compile_flow_runtime(
    flow: &FlowIR,
    opt_level: crate::engine::aot_flows::OptLevel,
) -> Result<CompiledFlowMetadata, ExecError> {
    let mut compiler = FlowCompiler::new(opt_level)?;

    // Generate output path in temp directory
    let temp_dir = std::env::temp_dir();
    let output_path = temp_dir.join(format!("flow_{}.o", sanitize_name(&flow.name)));

    let metadata = compiler.compile_flow(flow, &output_path)?;

    // Register with the global registry
    let mut loader = AOT_REGISTRY.loader.write().unwrap();
    loader.load_flow(metadata.clone())?;

    let mut metadata_map = AOT_REGISTRY.metadata.write().unwrap();
    metadata_map.insert(flow.name.clone(), metadata.clone());

    Ok(metadata)
}

fn sanitize_name(name: &str) -> String {
    name.chars()
        .map(|c| if c.is_alphanumeric() { c } else { '_' })
        .collect()
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_registry_initialization() {
        let registry = AotFlowRegistry::new();
        assert!(!registry.has_compiled("test_flow"));
    }

    #[test]
    fn test_sanitize_name() {
        assert_eq!(sanitize_name("my-flow"), "my_flow");
        assert_eq!(sanitize_name("my.flow"), "my_flow");
        assert_eq!(sanitize_name("my flow"), "my_flow");
    }
}
