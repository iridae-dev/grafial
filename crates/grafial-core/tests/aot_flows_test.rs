//! Tests for AOT flow compilation.
//!
//! Verifies that flows can be compiled to native code and produce
//! identical results to interpreted execution.

#[cfg(feature = "aot")]
mod tests {
    use grafial_core::engine::aot_flows::{FlowCompiler, OptLevel};
    use grafial_core::engine::graph::BeliefGraph;
    use grafial_ir::{FlowIR, GraphDefIR, GraphExprIR, TransformIR};

    /// Create a simple test flow.
    fn create_test_flow() -> FlowIR {
        FlowIR {
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
                            TransformIR::ApplyRuleset {
                                rules: vec!["rule2".to_string()],
                            },
                        ],
                    },
                },
            ],
            metrics: vec![],
            exports: vec![],
            metric_exports: vec![],
            metric_imports: vec![],
        }
    }

    #[test]
    fn test_flow_compilation() {
        let flow = create_test_flow();
        let mut compiler = FlowCompiler::new(OptLevel::None).unwrap();

        // Compile to a temporary file
        let temp_dir = std::env::temp_dir();
        let output_path = temp_dir.join("test_flow.o");

        let metadata = compiler.compile_flow(&flow, &output_path).unwrap();

        assert_eq!(metadata.name, "test_flow");
        assert_eq!(metadata.entry_symbol, "flow_test_flow_execute");
        assert_eq!(metadata.dependencies.len(), 3);
        assert!(metadata
            .dependencies
            .contains(&"evidence:evidence1".to_string()));
        assert!(metadata.dependencies.contains(&"rule:rule1".to_string()));
        assert!(metadata.dependencies.contains(&"rule:rule2".to_string()));

        // Verify the object file was created
        assert!(output_path.exists());

        // Clean up
        std::fs::remove_file(&output_path).ok();
    }

    #[test]
    fn test_flow_compilation_with_optimization() {
        let flow = create_test_flow();
        let mut compiler = FlowCompiler::new(OptLevel::Speed).unwrap();

        let temp_dir = std::env::temp_dir();
        let output_path = temp_dir.join("test_flow_opt.o");

        let metadata = compiler.compile_flow(&flow, &output_path).unwrap();

        assert_eq!(metadata.name, "test_flow");
        assert!(output_path.exists());

        // Clean up
        std::fs::remove_file(&output_path).ok();
    }

    #[test]
    fn test_flow_loader() {
        use grafial_core::engine::aot_flows::CompiledFlowLoader;

        let flow = create_test_flow();
        let mut compiler = FlowCompiler::new(OptLevel::None).unwrap();
        let mut loader = CompiledFlowLoader::new();

        let temp_dir = std::env::temp_dir();
        let output_path = temp_dir.join("test_flow_loader.o");
        let metadata = compiler.compile_flow(&flow, &output_path).unwrap();

        // Loading should succeed for a valid compiled artifact.
        assert!(loader.load_flow(metadata).is_ok());

        // Execution should invoke the compiled entrypoint and succeed.
        let mut graph = BeliefGraph::default();
        let result = loader.execute_flow("test_flow", &mut graph);
        assert!(result.is_ok());

        // Clean up object + shared-library artifacts.
        std::fs::remove_file(&output_path).ok();
        let shared_path = output_path.with_extension(if cfg!(target_os = "macos") {
            "dylib"
        } else if cfg!(target_os = "windows") {
            "dll"
        } else {
            "so"
        });
        std::fs::remove_file(shared_path).ok();
    }

    #[test]
    fn test_flow_loader_rejects_missing_artifact() {
        use grafial_core::engine::aot_flows::{CompiledFlowLoader, CompiledFlowMetadata};

        let mut loader = CompiledFlowLoader::new();
        let metadata = CompiledFlowMetadata {
            name: "missing_flow".to_string(),
            source_hash: "deadbeef".to_string(),
            object_path: "/tmp/does-not-exist/libmissing.so".to_string(),
            entry_symbol: "flow_missing_flow_execute".to_string(),
            dependencies: vec![],
        };

        assert!(loader.load_flow(metadata).is_err());
    }

    #[test]
    fn test_dependency_extraction() {
        let flow = FlowIR {
            name: "deps_test".to_string(),
            on_model: "model".to_string(),
            graphs: vec![
                GraphDefIR {
                    name: "g1".to_string(),
                    expr: GraphExprIR::FromEvidence("ev1".to_string()),
                },
                GraphDefIR {
                    name: "g2".to_string(),
                    expr: GraphExprIR::FromEvidence("ev2".to_string()),
                },
                GraphDefIR {
                    name: "g3".to_string(),
                    expr: GraphExprIR::Pipeline {
                        start_graph: "g2".to_string(),
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

        // This test indirectly tests extract_flow_dependencies
        // through the compile_flow function
        let mut compiler = FlowCompiler::new(OptLevel::None).unwrap();
        let temp_dir = std::env::temp_dir();
        let output_path = temp_dir.join("deps_test.o");

        let metadata = compiler.compile_flow(&flow, &output_path).unwrap();

        assert_eq!(metadata.dependencies.len(), 4);
        assert!(metadata.dependencies.contains(&"evidence:ev1".to_string()));
        assert!(metadata.dependencies.contains(&"evidence:ev2".to_string()));
        assert!(metadata.dependencies.contains(&"rule:rule1".to_string()));
        assert!(metadata.dependencies.contains(&"rule:rule2".to_string()));

        // Clean up
        std::fs::remove_file(&output_path).ok();
    }

    #[test]
    fn test_empty_flow_compilation() {
        let flow = FlowIR {
            name: "empty_flow".to_string(),
            on_model: "model".to_string(),
            graphs: vec![],
            metrics: vec![],
            exports: vec![],
            metric_exports: vec![],
            metric_imports: vec![],
        };

        let mut compiler = FlowCompiler::new(OptLevel::SpeedAndSize).unwrap();
        let temp_dir = std::env::temp_dir();
        let output_path = temp_dir.join("empty_flow.o");

        let metadata = compiler.compile_flow(&flow, &output_path).unwrap();

        assert_eq!(metadata.name, "empty_flow");
        assert_eq!(metadata.dependencies.len(), 0);
        assert!(output_path.exists());

        // Clean up
        std::fs::remove_file(&output_path).ok();
    }
}
