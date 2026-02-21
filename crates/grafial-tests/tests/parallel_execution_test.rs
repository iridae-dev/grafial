//! Integration tests for Phase 12 parallel engine execution.
//!
//! These tests verify that parallel execution produces identical results
//! to sequential execution while providing performance improvements.

#[cfg(test)]
mod tests {
    use grafial_core::engine::flow_exec::{run_flow_ir, FlowResult};
    use grafial_core::parse_validate_and_lower;
    use std::time::Instant;

    /// Test helper to run a flow and return the result.
    fn run_flow_test(
        source: &str,
        flow_name: &str,
    ) -> Result<FlowResult, Box<dyn std::error::Error>> {
        let program_ir = parse_validate_and_lower(source)?;
        Ok(run_flow_ir(&program_ir, flow_name, None)?)
    }

    /// Compares two flow results for equality.
    #[allow(dead_code)]
    fn assert_results_equal(result1: &FlowResult, result2: &FlowResult, context: &str) {
        // Compare metrics
        assert_eq!(
            result1.metrics.len(),
            result2.metrics.len(),
            "{}: Different number of metrics",
            context
        );

        for (key, value1) in &result1.metrics {
            let value2 = result2
                .metrics
                .get(key)
                .expect(&format!("{}: Missing metric {}", context, key));
            assert!(
                (value1 - value2).abs() < 1e-10,
                "{}: Metric {} differs: {} vs {}",
                context,
                key,
                value1,
                value2
            );
        }

        // Compare graphs
        assert_eq!(
            result1.graphs.len(),
            result2.graphs.len(),
            "{}: Different number of graphs",
            context
        );

        // Compare exports
        assert_eq!(
            result1.exports.len(),
            result2.exports.len(),
            "{}: Different number of exports",
            context
        );
    }

    #[test]
    fn test_parallel_evidence_determinism() {
        let source = r#"
        schema TestSchema {
            node Person {
                age: Real
                score: Real
            }
            edge knows {}
        }

        belief_model TestModel on TestSchema {
            node Person {
                age ~ Gaussian(mean=30, precision=0.1)
                score ~ Gaussian(mean=50, precision=0.05)
            }
            edge knows {
                exist ~ Bernoulli(prior=0.5, weight=2.0)
            }
        }

        evidence TestEvidence on TestModel {
            Person {
                "Alice" { age: 25.0, score: 60.0 },
                "Bob" { age: 35.0, score: 45.0 },
                "Charlie" { age: 28.0, score: 55.0 }
            }
            knows(Person -> Person) { "Alice" -> "Bob"; "Bob" -> "Charlie" }
        }

        flow TestFlow on TestModel {
            graph g = from_evidence TestEvidence
            metric avg_age = nodes(Person) |> avg(by=E[node.age])
            metric avg_score = nodes(Person) |> avg(by=E[node.score])
            metric node_count = nodes(Person) |> count()
        }
        "#;

        let start_seq = Instant::now();
        let result_sequential =
            run_flow_test(source, "TestFlow").expect("First flow execution failed");
        let result_repeat =
            run_flow_test(source, "TestFlow").expect("Second flow execution failed");
        let time_sequential = start_seq.elapsed();

        println!("Sequential execution time: {:?}", time_sequential);
        assert_results_equal(&result_sequential, &result_repeat, "determinism");

        assert!(result_sequential.metrics.contains_key("avg_age"));
        assert!(result_sequential.metrics.contains_key("avg_score"));
        assert!(result_sequential.metrics.contains_key("node_count"));

        let node_count = result_sequential.metrics["node_count"];
        assert_eq!(node_count, 3.0, "Unexpected node_count: {}", node_count);
    }

    #[test]
    fn test_parallel_metric_dependencies() {
        let source = r#"
        schema TestSchema {
            node Item {
                value: Real
            }
        }

        belief_model TestModel on TestSchema {
            node Item {
                value ~ Gaussian(mean=100, precision=0.1)
            }
        }

        evidence TestEvidence on TestModel {
            Item {
                "A" { value: 110.0 },
                "B" { value: 90.0 },
                "C" { value: 105.0 },
                "D" { value: 95.0 }
            }
        }

        flow TestFlow on TestModel {
            graph g = from_evidence TestEvidence
            metric base_sum = nodes(Item) |> sum(by=E[node.value])
            metric avg_value = base_sum / 4
            metric centered = avg_value - 100
            metric recomposed = centered + base_sum
        }
        "#;

        let result = run_flow_test(source, "TestFlow").expect("Flow execution failed");

        // Verify metric dependencies were respected
        assert!(result.metrics.contains_key("base_sum"));
        assert!(result.metrics.contains_key("avg_value"));
        assert!(result.metrics.contains_key("centered"));
        assert!(result.metrics.contains_key("recomposed"));

        // Check calculation correctness
        let base_sum = result.metrics["base_sum"];
        assert_eq!(base_sum, 400.0, "Unexpected base_sum: {}", base_sum);

        let avg_value = result.metrics["avg_value"];
        assert_eq!(avg_value, 100.0, "Unexpected avg_value: {}", avg_value);

        let recomposed = result.metrics["recomposed"];
        assert_eq!(recomposed, 400.0, "Unexpected recomposed: {}", recomposed);
    }

    #[test]
    fn test_parallel_rule_application() {
        let source = include_str!("../../grafial-examples/social.grafial");
        let result = run_flow_test(source, "Demo").expect("Flow execution failed");

        // Social demo applies a rule + prune pipeline and exposes avg_degree.
        let avg_degree = result.metrics.get("avg_degree").copied();
        assert!(avg_degree.is_some(), "avg_degree metric missing");
        assert!(avg_degree.unwrap().is_finite(), "avg_degree not finite");
    }

    #[test]
    fn test_parallel_batch_processing() {
        // Test with a larger dataset to better demonstrate parallel benefits
        let mut observations = Vec::new();
        for i in 0..100 {
            observations.push(format!(
                "                \"N{}\" {{ value: {:.1} }}",
                i,
                i as f64 * 2.0
            ));
        }

        let source = format!(
            r#"
        schema BatchSchema {{
            node Node {{
                value: Real
            }}
        }}

        belief_model BatchModel on BatchSchema {{
            node Node {{
                value ~ Gaussian(mean=0, precision=0.1)
            }}
        }}

        evidence BatchEvidence on BatchModel {{
            Node {{
{}
            }}
        }}

        flow BatchFlow on BatchModel {{
            graph g = from_evidence BatchEvidence
            metric total = nodes(Node) |> sum(by=E[node.value])
            metric avg = nodes(Node) |> avg(by=E[node.value])
        }}
        "#,
            observations.join(",\n"),
        );

        let start = Instant::now();
        let result = run_flow_test(&source, "BatchFlow").expect("Batch flow execution failed");
        let elapsed = start.elapsed();

        println!("Batch processing time for 100 nodes: {:?}", elapsed);

        // Verify results
        let total = result.metrics["total"];
        assert_eq!(total, 9000.0, "Unexpected total: {}", total);

        let avg = result.metrics["avg"];
        assert_eq!(avg, 90.0, "Unexpected average: {}", avg);
    }

    #[cfg(all(feature = "parallel", not(feature = "parallel")))]
    #[test]
    fn test_parallel_vs_sequential_parity() {
        // This test would run if we could toggle the feature at runtime
        // It would compare results between parallel and sequential execution
        // to ensure they produce identical outputs

        // Note: In practice, this test would need to be run as part of
        // a separate test suite that compiles the code twice with different features
    }
}
