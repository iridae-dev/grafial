//! Grafial CLI - Command-line interface for the Grafial Bayesian Belief Graph Language
//!
//! Usage:
//!   grafial <file>                    # Validate a .grafial file
//!   grafial <file> --flow <name>      # Execute a specific flow
//!   grafial <file> --flow <name> -o json  # Output results as JSON

use clap::Parser;
use grafial_core::parse_and_validate;
use grafial_core::run_flow;
use std::collections::HashMap;
use std::process;

#[derive(Parser)]
#[command(name = "grafial")]
#[command(version)]
#[command(about = "Grafial - Bayesian Belief Graph Language CLI")]
#[command(long_about = "Compile and execute Grafial programs for probabilistic graph reasoning")]
struct Cli {
    /// Input .grafial file
    #[arg(value_name = "FILE")]
    file: String,

    /// Flow name to execute (optional - just validate if not provided)
    #[arg(short, long, value_name = "NAME")]
    flow: Option<String>,

    /// Output format: summary, json, or debug
    #[arg(short, long, default_value = "summary", value_name = "FORMAT")]
    output: String,

    /// List all flows in the program instead of executing
    #[arg(short, long)]
    list_flows: bool,
}

fn main() {
    let cli = Cli::parse();

    // Read and parse
    let source = match std::fs::read_to_string(&cli.file) {
        Ok(s) => s,
        Err(e) => {
            eprintln!("Error reading file '{}': {}", cli.file, e);
            process::exit(1);
        }
    };

    let program = match parse_and_validate(&source) {
        Ok(p) => p,
        Err(e) => {
            eprintln!("Parse/validation error: {}", e);
            process::exit(1);
        }
    };

    // List flows if requested
    if cli.list_flows {
        if program.flows.is_empty() {
            println!("No flows defined in this program");
        } else {
            println!("Flows in '{}':", cli.file);
            for flow in &program.flows {
                println!("  - {}", flow.name);
            }
        }
        return;
    }

    // Execute flow if specified
    if let Some(flow_name) = &cli.flow {
        match run_flow(&program, flow_name, None) {
            Ok(result) => match cli.output.as_str() {
                "json" => match serde_json::to_string_pretty(&format_flow_result(&result)) {
                    Ok(json) => println!("{}", json),
                    Err(e) => {
                        eprintln!("Error serializing to JSON: {}", e);
                        process::exit(1);
                    }
                },
                "debug" => {
                    println!("{:#?}", result);
                }
                "summary" | _ => {
                    print_summary(flow_name, &result);
                }
            },
            Err(e) => {
                eprintln!("Error executing flow '{}': {}", flow_name, e);
                process::exit(1);
            }
        }
    } else {
        // Just validate
        println!("✓ Program validated successfully");
        if !program.flows.is_empty() {
            println!("\nAvailable flows:");
            for flow in &program.flows {
                println!("  - {}", flow.name);
            }
            println!("\nRun with --flow <name> to execute a flow");
        }
    }
}

fn print_summary(flow_name: &str, result: &grafial_core::engine::flow_exec::FlowResult) {
    println!("✓ Flow '{}' executed successfully\n", flow_name);

    if !result.graphs.is_empty() {
        println!("Graphs ({}):", result.graphs.len());
        for (name, graph) in &result.graphs {
            println!(
                "  {}: {} nodes, {} edges",
                name,
                graph.nodes().len(),
                graph.edges().len()
            );
        }
    }

    if !result.exports.is_empty() {
        println!("\nExports ({}):", result.exports.len());
        for (alias, graph) in &result.exports {
            println!(
                "  {}: {} nodes, {} edges",
                alias,
                graph.nodes().len(),
                graph.edges().len()
            );
        }
    }

    if !result.metrics.is_empty() {
        println!("\nMetrics ({}):", result.metrics.len());
        for (name, value) in &result.metrics {
            println!("  {} = {:.6}", name, value);
        }
    }

    if !result.metric_exports.is_empty() {
        println!("\nMetric Exports ({}):", result.metric_exports.len());
        for (alias, value) in &result.metric_exports {
            println!("  {} = {:.6}", alias, value);
        }
    }

    if !result.snapshots.is_empty() {
        println!("\nSnapshots ({}):", result.snapshots.len());
        for (name, graph) in &result.snapshots {
            println!(
                "  {}: {} nodes, {} edges",
                name,
                graph.nodes().len(),
                graph.edges().len()
            );
        }
    }
}

/// Format FlowResult for JSON serialization
fn format_flow_result(
    result: &grafial_core::engine::flow_exec::FlowResult,
) -> HashMap<String, serde_json::Value> {
    use serde_json::json;
    let mut output = HashMap::new();

    // Format graphs (just counts for now - full serialization would be complex)
    let graphs: HashMap<String, serde_json::Value> = result
        .graphs
        .iter()
        .map(|(name, graph)| {
            (
                name.clone(),
                json!({
                    "nodes": graph.nodes().len(),
                    "edges": graph.edges().len(),
                }),
            )
        })
        .collect();
    output.insert("graphs".to_string(), json!(graphs));

    // Format exports
    let exports: HashMap<String, serde_json::Value> = result
        .exports
        .iter()
        .map(|(alias, graph)| {
            (
                alias.clone(),
                json!({
                    "nodes": graph.nodes().len(),
                    "edges": graph.edges().len(),
                }),
            )
        })
        .collect();
    output.insert("exports".to_string(), json!(exports));

    // Format metrics
    output.insert("metrics".to_string(), json!(result.metrics));
    output.insert("metric_exports".to_string(), json!(result.metric_exports));

    // Format snapshots
    let snapshots: HashMap<String, serde_json::Value> = result
        .snapshots
        .iter()
        .map(|(name, graph)| {
            (
                name.clone(),
                json!({
                    "nodes": graph.nodes().len(),
                    "edges": graph.edges().len(),
                }),
            )
        })
        .collect();
    output.insert("snapshots".to_string(), json!(snapshots));

    output
}
