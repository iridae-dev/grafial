//! Phase 6 release-gate checks.
//!
//! Verifies that all shipped examples parse, validate, and execute for every declared flow.

use grafial_core::{
    parse_and_validate, parse_validate_and_lower, run_flow, run_flow_ir, ExecError,
};
use grafial_frontend::lint_canonical_style;
use std::fs;
use std::path::{Path, PathBuf};

fn examples_dir() -> PathBuf {
    // tests run from the grafial-tests crate; examples live at ../grafial-examples
    let mut p = PathBuf::from(env!("CARGO_MANIFEST_DIR"));
    p.push("../grafial-examples");
    p
}

fn read_file(path: &Path) -> Result<String, ExecError> {
    fs::read_to_string(path)
        .map_err(|e| ExecError::Internal(format!("failed to read {}: {}", path.display(), e)))
}

fn example_files() -> Result<Vec<PathBuf>, ExecError> {
    let dir = examples_dir();
    let mut files = Vec::new();
    for entry in fs::read_dir(&dir)
        .map_err(|e| ExecError::Internal(format!("read_dir {}: {}", dir.display(), e)))?
    {
        let entry = entry.map_err(|e| ExecError::Internal(format!("read_dir entry: {}", e)))?;
        let path = entry.path();
        if path.extension().and_then(|s| s.to_str()) == Some("grafial") {
            files.push(path);
        }
    }
    files.sort();
    Ok(files)
}

#[test]
fn phase6_release_gate_examples_parse_validate_and_execute_all_flows() -> Result<(), ExecError> {
    let files = example_files()?;
    assert!(
        !files.is_empty(),
        "release gate misconfigured: no example .grafial files found"
    );

    let mut executed_flows = 0usize;

    for file in files {
        let src = read_file(&file)?;
        let style_lints = lint_canonical_style(&src);
        assert!(
            style_lints.is_empty(),
            "example {} must use canonical syntax only (found {} style lints)",
            file.display(),
            style_lints.len()
        );
        let ast = parse_and_validate(&src)?;
        let ir = parse_validate_and_lower(&src)?;

        assert!(
            !ast.flows.is_empty(),
            "example {} must define at least one flow",
            file.display()
        );

        let mut ast_prior = None;
        let mut ir_prior = None;

        for flow in &ast.flows {
            let flow_name = flow.name.as_str();
            let ast_result = run_flow(&ast, flow_name, ast_prior.as_ref()).map_err(|e| {
                ExecError::Internal(format!(
                    "{} flow {} AST execution failed: {}",
                    file.display(),
                    flow_name,
                    e
                ))
            })?;
            let ir_result = run_flow_ir(&ir, flow_name, ir_prior.as_ref()).map_err(|e| {
                ExecError::Internal(format!(
                    "{} flow {} IR execution failed: {}",
                    file.display(),
                    flow_name,
                    e
                ))
            })?;

            let mut ast_metric_keys: Vec<_> = ast_result.metrics.keys().cloned().collect();
            ast_metric_keys.sort();
            let mut ir_metric_keys: Vec<_> = ir_result.metrics.keys().cloned().collect();
            ir_metric_keys.sort();
            assert_eq!(
                ast_metric_keys,
                ir_metric_keys,
                "metric surface mismatch for {} flow {}",
                file.display(),
                flow_name
            );
            for metric in ast_metric_keys {
                let ast_value = ast_result.metrics[&metric];
                let ir_value = ir_result.metrics[&metric];
                assert!(
                    (ast_value - ir_value).abs() < 1e-12,
                    "metric mismatch for {} flow {} metric {}: ast={}, ir={}",
                    file.display(),
                    flow_name,
                    metric,
                    ast_value,
                    ir_value
                );
            }

            let mut ast_export_keys: Vec<_> = ast_result.exports.keys().cloned().collect();
            ast_export_keys.sort();
            let mut ir_export_keys: Vec<_> = ir_result.exports.keys().cloned().collect();
            ir_export_keys.sort();
            assert_eq!(
                ast_export_keys,
                ir_export_keys,
                "export surface mismatch for {} flow {}",
                file.display(),
                flow_name
            );

            ast_prior = Some(ast_result.clone());
            ir_prior = Some(ir_result.clone());
            executed_flows += 1;
        }
    }

    assert!(executed_flows > 0, "release gate executed zero flows");
    Ok(())
}
