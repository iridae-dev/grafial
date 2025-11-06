use grafial_core::{parse_and_validate, run_flow, ExecError};
use std::fs;
use std::path::PathBuf;

fn examples_dir() -> PathBuf {
    // tests run from the grafial-tests crate; examples live at ../grafial-examples
    let mut p = PathBuf::from(env!("CARGO_MANIFEST_DIR"));
    p.push("../grafial-examples");
    p
}

fn read_example(name: &str) -> Result<String, ExecError> {
    let mut p = examples_dir();
    p.push(name);
    fs::read_to_string(&p)
        .map_err(|e| ExecError::Internal(format!("failed to read {}: {}", p.display(), e)))
}

#[test]
fn parse_all_examples() -> Result<(), ExecError> {
    let dir = examples_dir();
    let mut found = 0usize;
    for entry in fs::read_dir(&dir)
        .map_err(|e| ExecError::Internal(format!("read_dir {}: {}", dir.display(), e)))?
    {
        let entry = entry.map_err(|e| ExecError::Internal(format!("read_dir entry: {}", e)))?;
        let path = entry.path();
        if path.extension().and_then(|s| s.to_str()) == Some("grafial") {
            found += 1;
            let src = fs::read_to_string(&path).map_err(|e| {
                ExecError::Internal(format!("failed to read {}: {}", path.display(), e))
            })?;
            // Parse + validate end-to-end
            let _ast = parse_and_validate(&src)?;
        }
    }
    assert!(found > 0, "no grafial example files found");
    Ok(())
}

#[test]
fn example_minimal_behaves() -> Result<(), ExecError> {
    // Expect: parses, flow runs, metric ~ 1.0, export exists
    let src = read_example("minimal.grafial")?;
    let program = parse_and_validate(&src)?;
    let result = run_flow(&program, "MinimalFlow", None)?;

    // Metric: total should be close to 1.0 (weak prior, one observed value = 1.0)
    let total = *result
        .metrics
        .get("total")
        .ok_or_else(|| ExecError::Internal("MinimalFlow: missing metric 'total'".into()))?;
    assert!(total.is_finite());
    assert!(total > 0.8 && total < 1.2, "unexpected total: {}", total);

    // Export exists
    assert!(result.exports.contains_key("output"), "expected export 'output'");
    Ok(())
}

#[test]
fn example_social_behaves() -> Result<(), ExecError> {
    // Expect: after pruning only Alice->Bob remains; avg_degree ≈ 1/3
    let src = read_example("social.grafial")?;
    let program = parse_and_validate(&src)?;
    let result = run_flow(&program, "Demo", None)?;

    let avg_deg = *result
        .metrics
        .get("avg_degree")
        .ok_or_else(|| ExecError::Internal("Demo: missing metric 'avg_degree'".into()))?;
    assert!(avg_deg.is_finite());
    // With min_prob=0.8 and evidence-implied probabilities ~0.67,
    // no edges count toward degree, so avg_degree should be 0.0
    assert!(avg_deg.abs() < 1e-9, "unexpected avg_degree: {}", avg_deg);

    assert!(result.exports.contains_key("demo"), "expected export 'demo'");
    Ok(())
}

#[test]
fn example_ab_testing_behaves() -> Result<(), ExecError> {
    // Expect: B > A slightly; mean_A around 0.10-0.11; good_variants == 0
    let src = read_example("ab_testing.grafial")?;
    let program = parse_and_validate(&src)?;
    let result = run_flow(&program, "ABTestAnalysis", None)?;

    let mean_a = *result
        .metrics
        .get("mean_A")
        .ok_or_else(|| ExecError::Internal("ABTestAnalysis: missing metric 'mean_A'".into()))?;
    assert!(mean_a.is_finite());
    assert!(
        (0.09..=0.12).contains(&mean_a),
        "mean_A out of expected range: {}",
        mean_a
    );

    let good_variants = *result
        .metrics
        .get("good_variants")
        .ok_or_else(|| ExecError::Internal("ABTestAnalysis: missing 'good_variants'".into()))?;
    assert_eq!(good_variants, 0.0, "expected no good_variants, got {}", good_variants);

    assert!(result.exports.contains_key("winner"), "expected export 'winner'");
    Ok(())
}

#[test]
fn example_competing_choices_behaves() -> Result<(), ExecError> {
    // Expect: competing choices produce non-zero entropy; export exists
    let src = read_example("competing_choices.grafial")?;
    let program = parse_and_validate(&src)?;
    let result = run_flow(&program, "RoutingPipeline", None)?;

    let avg_entropy = *result
        .metrics
        .get("avg_entropy")
        .ok_or_else(|| ExecError::Internal("RoutingPipeline: missing 'avg_entropy'".into()))?;
    assert!(avg_entropy.is_finite());
    assert!(avg_entropy >= 0.0, "entropy must be non-negative: {}", avg_entropy);

    assert!(
        result.exports.contains_key("final_routing"),
        "expected export 'final_routing'"
    );
    Ok(())
}

#[test]
fn example_transitive_closure_sanity() -> Result<(), ExecError> {
    // Sanity: flow runs; metrics finite; export exists
    let src = read_example("transitive_closure.grafial")?;
    let program = parse_and_validate(&src)?;
    let result = run_flow(&program, "ReachabilityAnalysis", None)?;

    let avg_reachability = *result
        .metrics
        .get("avg_reachability")
        .ok_or_else(|| ExecError::Internal("ReachabilityAnalysis: missing 'avg_reachability'".into()))?;
    assert!(avg_reachability.is_finite());
    assert!(avg_reachability >= 0.0 && avg_reachability <= 1.0);

    let reachable_count = *result
        .metrics
        .get("reachable_count")
        .ok_or_else(|| ExecError::Internal("ReachabilityAnalysis: missing 'reachable_count'".into()))?;
    assert!(reachable_count >= 1.0);

    assert!(
        result.exports.contains_key("reachable"),
        "expected export 'reachable'"
    );
    Ok(())
}
