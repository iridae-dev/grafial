//! Grafial CLI - Command-line interface for the Grafial Bayesian Belief Graph Language
//!
//! Usage:
//!   grafial <file>                    # Validate a .grafial file
//!   grafial <file> --flow <name>      # Execute a specific flow
//!   grafial <file> --flow <name> -o json  # Output results as JSON

use clap::Parser;
use grafial_core::{parse_and_validate, run_flow};
use grafial_frontend::{
    collect_lint_suppressions, format_canonical_style, lint_canonical_style, lint_is_suppressed,
    CanonicalStyleLint,
};
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

    /// Report canonical-style compatibility forms
    #[arg(long)]
    lint_style: bool,

    /// Rewrite source file to canonical style in-place
    #[arg(long)]
    fix_style: bool,
}

fn main() {
    let cli = Cli::parse();

    // Read and parse
    let mut source = match std::fs::read_to_string(&cli.file) {
        Ok(s) => s,
        Err(e) => {
            eprintln!("Error reading file '{}': {}", cli.file, e);
            process::exit(1);
        }
    };

    if cli.fix_style {
        let formatted = format_canonical_style(&source);
        if formatted != source {
            if let Err(e) = std::fs::write(&cli.file, &formatted) {
                eprintln!("Error writing file '{}': {}", cli.file, e);
                process::exit(1);
            }
            println!("✓ Applied canonical style fixes to '{}'", cli.file);
            source = formatted;
        } else {
            println!("✓ Source already in canonical style");
        }
    }

    if cli.lint_style {
        let suppressions = collect_lint_suppressions(&source);
        let lints: Vec<_> = lint_canonical_style(&source)
            .into_iter()
            .filter(|lint| !lint_is_suppressed(&suppressions, lint.code, lint.range))
            .collect();
        print_style_lints(&lints);
    }

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
                "summary" => {
                    print_summary(flow_name, &result);
                }
                _ => {
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

fn print_style_lints(lints: &[CanonicalStyleLint]) {
    if lints.is_empty() {
        println!("✓ Canonical style: no compatibility forms found");
        return;
    }

    println!("Canonical style warnings ({}):", lints.len());
    for lint in lints {
        println!(
            "  {}:{} [{}] {}",
            lint.range.start.line, lint.range.start.column, lint.code, lint.message
        );
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

    if !result.intervention_audit.is_empty() {
        println!(
            "\nIntervention Audit Events ({}):",
            result.intervention_audit.len()
        );
        for event in &result.intervention_audit {
            println!(
                "  [{}:{}] rule={} matched={} actions={}",
                event.graph,
                event.transform,
                event.rule,
                event.matched_bindings,
                event.actions_executed
            );
        }
    }

    if !result.inference_diagnostics.is_empty() {
        println!(
            "\nInference Diagnostics ({}):",
            result.inference_diagnostics.len()
        );
        for event in &result.inference_diagnostics {
            println!(
                "  [{}:{}] alg={} converged={} iterations={}/{} final_delta={:.3e} vars={} connected={}",
                event.graph,
                event.transform,
                event.algorithm,
                event.converged,
                event.iterations_run,
                event.max_iterations,
                event.final_max_message_delta,
                event.variable_count,
                event.connected_variable_count
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

    let intervention_audit: Vec<serde_json::Value> = result
        .intervention_audit
        .iter()
        .map(|event| {
            json!({
                "flow": event.flow,
                "graph": event.graph,
                "transform": event.transform,
                "rule": event.rule,
                "mode": event.mode,
                "matched_bindings": event.matched_bindings,
                "actions_executed": event.actions_executed,
            })
        })
        .collect();
    output.insert("intervention_audit".to_string(), json!(intervention_audit));

    let inference_diagnostics: Vec<serde_json::Value> = result
        .inference_diagnostics
        .iter()
        .map(|event| {
            json!({
                "flow": event.flow,
                "graph": event.graph,
                "transform": event.transform,
                "algorithm": event.algorithm,
                "iterations_run": event.iterations_run,
                "max_iterations": event.max_iterations,
                "converged": event.converged,
                "final_max_message_delta": event.final_max_message_delta,
                "variable_count": event.variable_count,
                "connected_variable_count": event.connected_variable_count,
            })
        })
        .collect();
    output.insert(
        "inference_diagnostics".to_string(),
        json!(inference_diagnostics),
    );

    output
}
