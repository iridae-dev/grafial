//! Debug program to diagnose the social.grafial bug
//! Run with: cargo run --example debug_social

use grafial::parse_and_validate;
use grafial::engine::flow_exec::run_flow;

fn main() {
    let src = std::fs::read_to_string("examples/social.grafial").expect("read example");
    let program = parse_and_validate(&src).expect("parse and validate");

    println!("Executing flow: {}", program.flows[0].name);
    println!("Rule: {}", program.rules[0].name);
    println!("Patterns: {} pattern(s)", program.rules[0].patterns.len());
    println!();

    let result = run_flow(&program, &program.flows[0].name, None);

    match result {
        Ok(flow_result) => {
            println!("\n=== FLOW EXECUTION COMPLETE ===");

            // Check each graph
            for graph_def in &program.flows[0].graphs {
                if let Some(graph) = flow_result.graphs.get(&graph_def.name) {
                    println!("\nGraph '{}': {} nodes, {} edges",
                        graph_def.name,
                        graph.nodes().len(),
                        graph.edges().len()
                    );
                    for edge in graph.edges() {
                        if let Ok(prob) = graph.prob_mean(edge.id) {
                            println!("  Edge {:?} ({}): {:.6}",
                                edge.id, edge.ty, prob);
                        }
                    }
                }
            }

            println!("\nMetrics:");
            for (name, value) in &flow_result.metrics {
                println!("  {} = {}", name, value);
            }
        }
        Err(e) => {
            eprintln!("Flow execution failed: {:?}", e);
            std::process::exit(1);
        }
    }
}
