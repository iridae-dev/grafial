# Examples Index

All examples live in [`crates/grafial-examples/`](../crates/grafial-examples/) and are used as parser/runtime
fixtures (Phase 6 release gate) and Composer dropdown entries.

| Example | Problem | Concepts | Flows | Expected outputs | Commands |
|---|---|---|---|---|---|
| [minimal.grafial](../crates/grafial-examples/minimal.grafial) | Tiny Gaussian node + Bernoulli edge | schema, belief_model, evidence, metric builder | `MinimalFlow` | `total` ∈ (0.8, 1.2); export `output` | `grafial … --flow MinimalFlow` · Composer → minimal |
| [social.grafial](../crates/grafial-examples/social.grafial) | Social reach / prune after transfer | rules, pipelines, `avg_degree`, prune | `Demo` | `TransferAndDisconnect` fires once; `avg_degree = 1/3`; export `demo` has 1 edge | `--flow Demo` |
| [ab_testing.grafial](../crates/grafial-examples/ab_testing.grafial) | A/B conversion lift | soft updates, winner rule, metrics | `ABTestAnalysis` | `DetermineWinner` fires once; `avg_conversion` ∈ [0.125, 0.14]; `good_variants = 1`; export `winner` | `--flow ABTestAnalysis` |
| [advanced_metrics.grafial](../crates/grafial-examples/advanced_metrics.grafial) | Metric API showcase | count/sum/fold/avg_degree, composition | `MetricsShowcase` | All metrics finite; no runtime errors | `--flow MetricsShowcase` |
| [common_mistakes.grafial](../crates/grafial-examples/common_mistakes.grafial) | Educational anti-patterns | lint/validation teaching | `CorrectConnectivityCheck`, `RiskyDivision`, `SafeDivision`, `ExplicitSequencing` | Prefer safe flows; comments document rejected patterns | read + `--lint-style` |
| [competing_choices.grafial](../crates/grafial-examples/competing_choices.grafial) | Exclusive routing | Categorical / Dirichlet, entropy metrics | `RoutingPipeline` | Competing means sum to 1; entropy/deterministic metrics finite | `--flow RoutingPipeline` |
| [pipeline_composition.grafial](../crates/grafial-examples/pipeline_composition.grafial) | Multi-stage pipelines | `export_metric` / `import_metric`, graph import | `CleaningStage` → `EnrichmentStage` → `QualityAnalysis` | `cleaning_stats` / `enrichment_stats` pass between flows; `pipeline_quality` finite | run stages in order (CLI deps resolve) |
| [prior_sensitivity.grafial](../crates/grafial-examples/prior_sensitivity.grafial) | Prior strength vs data | Gaussian prior precision ladders | `WeakPriorFlow`, `ModeratePriorFlow`, `StrongPriorFlow` | Stronger priors move `posterior_mean` less toward the observation | compare three `--flow` runs |
| [probabilistic_pattern_matching.grafial](../crates/grafial-examples/probabilistic_pattern_matching.grafial) | Probabilistic `where` | `prob` / `credible`, influence metrics | `SocialAnalysis` | High-prob edges drive rule firings; `avg_influence` / `avg_strong_connections` finite | `--flow SocialAnalysis` |
| [soft_vs_hard_updates.grafial](../crates/grafial-examples/soft_vs_hard_updates.grafial) | Soft `~=` vs hard evidence | soft updates, capacity/allocation | `AllocationFlow` | `avg_allocation` / `overload_risk` finite after soft updates | `--flow AllocationFlow` |
| [transitive_closure.grafial](../crates/grafial-examples/transitive_closure.grafial) | Multi-hop reachability | repeated `apply_rule`, reachability attrs | `ReachabilityAnalysis` | `avg_reachability` rises across hops; `reachable_count` > 0 | `--flow ReachabilityAnalysis` |
| [uncertainty_propagation.grafial](../crates/grafial-examples/uncertainty_propagation.grafial) | Fraud risk under uncertainty | prune, `infer_beliefs`, diagnostics | `FraudAnalysis` | `inference_diagnostics` present; `avg_risk` / `suspicious_count` finite | `--flow FraudAnalysis` · JSON for diagnostics |

## Commands

```bash
# List flows
grafial crates/grafial-examples/minimal.grafial --list-flows

# Run a flow (human or JSON)
grafial crates/grafial-examples/minimal.grafial --flow MinimalFlow
grafial crates/grafial-examples/minimal.grafial --flow MinimalFlow --output json

# Python
python - <<'PY'
import pathlib, grafial
src = pathlib.Path("crates/grafial-examples/minimal.grafial").read_text()
ctx = grafial.run_flow(grafial.compile(src), "MinimalFlow")
print(ctx.metrics)
PY

# Composer
# Hosted: https://grafial.iridae.com/  → Examples → minimal → Run
# Local:  ./scripts/serve_composer.sh
```

Also see the [language guide](LANGUAGE_GUIDE.md) and [probabilistic semantics](PROBABILISTIC_SEMANTICS.md).
