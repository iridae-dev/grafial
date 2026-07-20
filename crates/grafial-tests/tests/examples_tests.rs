use grafial_core::{
    parse_and_validate, parse_validate_and_lower, run_flow, run_flow_ir, ExecError,
};
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
fn example_minimal_ir_entrypoint_matches_ast_wrapper() -> Result<(), ExecError> {
    let src = read_example("minimal.grafial")?;

    let ast_program = parse_and_validate(&src)?;
    let ir_program = parse_validate_and_lower(&src)?;

    let ast_result = run_flow(&ast_program, "MinimalFlow", None)?;
    let ir_result = run_flow_ir(&ir_program, "MinimalFlow", None)?;

    let ast_total = *ast_result
        .metrics
        .get("total")
        .ok_or_else(|| ExecError::Internal("MinimalFlow: missing metric 'total'".into()))?;
    let ir_total = *ir_result
        .metrics
        .get("total")
        .ok_or_else(|| ExecError::Internal("MinimalFlow: missing metric 'total'".into()))?;

    assert!(
        (ast_total - ir_total).abs() < 1e-12,
        "IR/AST metric mismatch: ast={}, ir={}",
        ast_total,
        ir_total
    );

    assert_eq!(ast_result.exports.len(), ir_result.exports.len());
    assert!(
        ir_result.exports.contains_key("output"),
        "expected export 'output'"
    );
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
    assert!(
        result.exports.contains_key("output"),
        "expected export 'output'"
    );
    Ok(())
}

#[test]
fn example_social_behaves() -> Result<(), ExecError> {
    // Expect: TransferAndDisconnect fires (edges at P=0.8 clear the 0.75 bar),
    // Bob->Carol is deleted then pruned, and only Alice->Bob remains:
    // avg_degree = 1/3.
    let src = read_example("social.grafial")?;
    let program = parse_and_validate(&src)?;
    let result = run_flow(&program, "Demo", None)?;

    let transfer_audit = result
        .intervention_audit
        .iter()
        .find(|e| e.rule == "TransferAndDisconnect")
        .expect("missing TransferAndDisconnect audit event");
    assert_eq!(
        transfer_audit.matched_bindings, 1,
        "TransferAndDisconnect should fire exactly once"
    );

    let avg_deg = *result
        .metrics
        .get("avg_degree")
        .ok_or_else(|| ExecError::Internal("Demo: missing metric 'avg_degree'".into()))?;
    assert!(
        (avg_deg - 1.0 / 3.0).abs() < 1e-9,
        "unexpected avg_degree: {}",
        avg_deg
    );

    let cleaned = result
        .exports
        .get("demo")
        .ok_or_else(|| ExecError::Internal("expected export 'demo'".into()))?;
    assert_eq!(cleaned.edges().len(), 1, "only Alice->Bob should survive");
    Ok(())
}

#[test]
fn example_ab_testing_behaves() -> Result<(), ExecError> {
    // Expect: with sample-size-scaled observation precision, B's lift is
    // practically significant (0.0273 > 0.02), so DetermineWinner fires and
    // exactly one variant (B) clears the 12% bar.
    let src = read_example("ab_testing.grafial")?;
    let program = parse_and_validate(&src)?;
    let result = run_flow(&program, "ABTestAnalysis", None)?;

    let winner_audit = result
        .intervention_audit
        .iter()
        .find(|e| e.rule == "DetermineWinner")
        .expect("missing DetermineWinner audit event");
    assert_eq!(
        winner_audit.matched_bindings, 1,
        "DetermineWinner should fire exactly once"
    );

    let avg_conversion = *result.metrics.get("avg_conversion").ok_or_else(|| {
        ExecError::Internal("ABTestAnalysis: missing metric 'avg_conversion'".into())
    })?;
    // (0.1182 + 0.1455 * 1.01) / 2 ≈ 0.1326
    assert!(
        (0.125..=0.14).contains(&avg_conversion),
        "avg_conversion out of expected range: {}",
        avg_conversion
    );

    let good_variants = *result
        .metrics
        .get("good_variants")
        .ok_or_else(|| ExecError::Internal("ABTestAnalysis: missing 'good_variants'".into()))?;
    assert_eq!(
        good_variants, 1.0,
        "expected exactly one good variant (B), got {}",
        good_variants
    );

    assert!(
        result.exports.contains_key("winner"),
        "expected export 'winner'"
    );
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
    // Only R1 carries entropy: H([7/15, 4/15, 4/15]) ≈ 1.06 nats over 6 routers.
    assert!(
        (0.15..=0.2).contains(&avg_entropy),
        "avg_entropy out of expected range: {}",
        avg_entropy
    );

    // Both showcased rules must actually fire.
    for rule in ["OptimizeLatency", "HandleUncertainRoute"] {
        let audit = result
            .intervention_audit
            .iter()
            .find(|e| e.rule == rule)
            .unwrap_or_else(|| panic!("missing audit event for {}", rule));
        assert_eq!(audit.matched_bindings, 1, "{} should fire once", rule);
    }

    assert!(
        result.exports.contains_key("final_routing"),
        "expected export 'final_routing'"
    );
    Ok(())
}

#[test]
fn example_transitive_closure_behaves() -> Result<(), ExecError> {
    // Expect genuine transitive propagation: two apply_rule hops reach every
    // server (S1: 0.909, S2/S4: 0.818, S3/S5: 0.736 -> all above 0.5).
    let src = read_example("transitive_closure.grafial")?;
    let program = parse_and_validate(&src)?;
    let result = run_flow(&program, "ReachabilityAnalysis", None)?;

    let avg_reachability = *result.metrics.get("avg_reachability").ok_or_else(|| {
        ExecError::Internal("ReachabilityAnalysis: missing 'avg_reachability'".into())
    })?;
    // (0.909 + 0.818 + 0.736 + 0.818 + 0.736) / 5 ≈ 0.803
    assert!(
        (0.79..=0.82).contains(&avg_reachability),
        "avg_reachability out of expected range: {}",
        avg_reachability
    );

    let reachable_count = *result.metrics.get("reachable_count").ok_or_else(|| {
        ExecError::Internal("ReachabilityAnalysis: missing 'reachable_count'".into())
    })?;
    assert_eq!(
        reachable_count, 5.0,
        "all 5 servers should be transitively reachable"
    );

    // Propagation must actually happen on both hops.
    let hop_matches: usize = result
        .intervention_audit
        .iter()
        .filter(|e| e.rule == "PropagateReachability")
        .map(|e| e.matched_bindings)
        .sum();
    assert!(
        hop_matches >= 4,
        "expected reachability propagation across hops, got {} total matches",
        hop_matches
    );

    assert!(
        result.exports.contains_key("reachable"),
        "expected export 'reachable'"
    );
    Ok(())
}
