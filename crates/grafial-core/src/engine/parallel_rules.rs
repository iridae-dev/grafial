//! Rule-application orchestration behind the `parallel` feature.
//!
//! This module now runs real rule execution (via `rule_exec`) and keeps a stable
//! API/telemetry surface for future non-overlapping parallel scheduling.

use std::collections::HashMap;

use crate::engine::errors::ExecError;
use crate::engine::graph::BeliefGraph;
use crate::engine::rule_exec::run_rule_for_each_with_globals_audit;
use grafial_ir::RuleIR;

/// Result of rule-application execution.
#[derive(Debug)]
pub struct ParallelRuleResult {
    /// Number of rules that had at least one matched binding.
    pub rules_applied: usize,
    /// Total number of matched bindings across all rules.
    pub matches_found: usize,
    /// Statistics about execution.
    pub stats: ParallelRuleStats,
}

/// Statistics about rule-application execution.
#[derive(Debug, Default)]
pub struct ParallelRuleStats {
    /// Number of execution batches (currently deterministic sequential batches).
    pub parallel_batches: usize,
    /// Maximum parallelism achieved in execution batches.
    pub max_parallelism: usize,
    /// Number of conflicts detected (0 in deterministic sequential mode).
    pub conflicts_detected: usize,
}

/// Apply rules using engine rule semantics.
///
/// The API is retained under the `parallel` feature and currently executes with
/// deterministic ruleset semantics (each rule receives previous output).
pub fn apply_rules_parallel(
    graph: &mut BeliefGraph,
    rules: &[RuleIR],
) -> Result<ParallelRuleResult, ExecError> {
    let mut matched_rules = 0;
    let mut total_matches = 0;
    let mut batch_count = 0;

    let globals: HashMap<String, f64> = HashMap::new();
    let mut current = graph.clone();

    for rule_ir in rules {
        let rule = rule_ir.to_ast();
        let (next, audit) = run_rule_for_each_with_globals_audit(&current, &rule, &globals)?;
        if audit.matched_bindings > 0 {
            matched_rules += 1;
            batch_count += 1;
        }
        total_matches += audit.matched_bindings;
        current = next;
    }

    *graph = current;

    Ok(ParallelRuleResult {
        rules_applied: matched_rules,
        matches_found: total_matches,
        stats: ParallelRuleStats {
            parallel_batches: batch_count,
            max_parallelism: if rules.is_empty() { 0 } else { 1 },
            conflicts_detected: 0,
        },
    })
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::engine::graph::{BetaPosterior, GaussianPosterior};
    use grafial_frontend::{parse_program, validate_program};
    use std::collections::HashMap;

    fn make_test_rule_ir() -> RuleIR {
        let src = r#"
        schema S {
            node Person { score: Real }
            edge REL {}
        }

        belief_model M on S {
            node Person { score ~ Gaussian(mean=0.0, precision=1.0) }
            edge REL { exist ~ Bernoulli(prior=0.6, weight=2.0) }
        }

        rule Bump on M {
            (A:Person)-[ab:REL]->(B:Person) => {
                B.score ~= 5.0 precision=1.0 count=1
            }
        }
        "#;

        let ast = parse_program(src).expect("parse test rule source");
        validate_program(&ast).expect("validate test rule source");
        RuleIR::from(&ast.rules[0])
    }

    #[test]
    fn apply_rules_parallel_applies_real_rule_logic() {
        let mut graph = BeliefGraph::default();

        let attrs_a = HashMap::from([(
            "score".to_string(),
            GaussianPosterior {
                mean: 1.0,
                precision: 1.0,
            },
        )]);
        let attrs_b = HashMap::from([(
            "score".to_string(),
            GaussianPosterior {
                mean: 0.0,
                precision: 1.0,
            },
        )]);

        let a = graph.add_node("Person".to_string(), attrs_a);
        let b = graph.add_node("Person".to_string(), attrs_b);
        graph.add_edge(
            a,
            b,
            "REL".to_string(),
            BetaPosterior {
                alpha: 6.0,
                beta: 1.0,
            },
        );

        let before = graph.expectation(b, "score").expect("pre-rule expectation");
        let rule = make_test_rule_ir();

        let stats = apply_rules_parallel(&mut graph, &[rule]).expect("apply rules");

        let after = graph
            .expectation(b, "score")
            .expect("post-rule expectation");

        assert!(after > before, "rule should increase B.score expectation");
        assert_eq!(stats.rules_applied, 1);
        assert_eq!(stats.matches_found, 1);
        assert_eq!(stats.stats.max_parallelism, 1);
    }

    #[test]
    fn apply_rules_parallel_empty_rules_is_noop() {
        let mut graph = BeliefGraph::default();
        let stats = apply_rules_parallel(&mut graph, &[]).expect("empty rules should succeed");

        assert_eq!(stats.rules_applied, 0);
        assert_eq!(stats.matches_found, 0);
        assert_eq!(stats.stats.parallel_batches, 0);
        assert_eq!(stats.stats.max_parallelism, 0);
    }
}
