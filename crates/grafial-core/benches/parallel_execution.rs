//! Benchmarks comparing parallel vs sequential execution performance.
//!
//! Run with: cargo bench --features parallel --bench parallel_execution

use criterion::{black_box, criterion_group, criterion_main, BenchmarkId, Criterion};
use grafial_core::engine::flow_exec::run_flow_ir;
use grafial_core::parse_validate_and_lower;
use std::time::Duration;

/// Generate a test program with N nodes and observations.
fn generate_test_program(num_nodes: usize) -> String {
    let mut observations = Vec::new();
    for i in 0..num_nodes {
        observations.push(format!(
            "            Person[\"P{}\"].score = {}\n            Person[\"P{}\"].age = {}",
            i,
            (i as f64) * 1.5,
            i,
            20.0 + (i as f64) * 0.5
        ));
    }

    let mut edges = Vec::new();
    // Create edges in a pattern that allows parallel processing
    for i in 0..num_nodes / 2 {
        for j in (num_nodes / 2)..num_nodes.min(i + num_nodes / 2 + 3) {
            edges.push(format!(
                "            knows(Person[\"P{}\"], Person[\"P{}\"]) = present",
                i, j
            ));
        }
    }

    format!(
        r#"
        schema BenchSchema {{
            node Person {{
                name: String
                age: Float
                score: Float
                status: String
            }}
            edge knows: Person -> Person
        }}

        model BenchModel on BenchSchema {{
            Person.age ~ Gaussian(mean=30, precision=0.1)
            Person.score ~ Gaussian(mean=50, precision=0.05)
            knows ~ Beta(alpha=2, beta=5)
        }}

        evidence BenchEvidence on BenchModel {{
{}
{}
        }}

        rule HighScorer on BenchModel {{
            Person[p]
            where p.score > 75
            then {{
                p.status = "high"
            }}
        }}

        rule LowScorer on BenchModel {{
            Person[p]
            where p.score < 25
            then {{
                p.status = "low"
            }}
        }}

        flow BenchFlow {{
            graph g = BenchEvidence
                |> apply_ruleset(HighScorer, LowScorer)

            // Multiple metrics with dependencies
            metric total_score = sum([n.score for n in g.nodes(Person)])
            metric avg_score = total_score / {}
            metric total_age = sum([n.age for n in g.nodes(Person)])
            metric avg_age = total_age / {}

            // Independent metrics that can be parallel
            metric max_score = max([n.score for n in g.nodes(Person)])
            metric min_score = min([n.score for n in g.nodes(Person)])
            metric max_age = max([n.age for n in g.nodes(Person)])
            metric min_age = min([n.age for n in g.nodes(Person)])

            metric edge_count = count([e for e in g.edges(knows)])
            metric high_scorers = count([n for n in g.nodes(Person) where n.status == "high"])
            metric low_scorers = count([n for n in g.nodes(Person) where n.status == "low"])
        }}
        "#,
        observations.join("\n"),
        edges.join("\n"),
        num_nodes,
        num_nodes
    )
}

fn benchmark_evidence_processing(c: &mut Criterion) {
    let mut group = c.benchmark_group("evidence_processing");
    group.measurement_time(Duration::from_secs(10));

    for size in [10, 50, 100, 200] {
        let program = generate_test_program(size);
        let program_ir = parse_validate_and_lower(&program).unwrap();

        group.bench_with_input(BenchmarkId::new("nodes", size), &program_ir, |b, prog| {
            b.iter(|| {
                let _ = run_flow_ir(black_box(prog), "BenchFlow", None);
            });
        });
    }

    group.finish();
}

fn benchmark_metric_evaluation(c: &mut Criterion) {
    let mut group = c.benchmark_group("metric_evaluation");

    // Create a program with many interdependent metrics
    let program = r#"
        schema MetricSchema {
            node Value {
                x: Float
                y: Float
            }
        }

        model MetricModel on MetricSchema {
            Value.x ~ Gaussian(mean=0, precision=1)
            Value.y ~ Gaussian(mean=0, precision=1)
        }

        evidence MetricEvidence on MetricModel {
            Value["V0"].x = 1
            Value["V0"].y = 2
            Value["V1"].x = 3
            Value["V1"].y = 4
            Value["V2"].x = 5
            Value["V2"].y = 6
            Value["V3"].x = 7
            Value["V3"].y = 8
            Value["V4"].x = 9
            Value["V4"].y = 10
        }

        flow MetricFlow {
            graph g = MetricEvidence

            // Create a complex dependency graph
            metric m1 = sum([n.x for n in g.nodes(Value)])
            metric m2 = sum([n.y for n in g.nodes(Value)])
            metric m3 = m1 + m2
            metric m4 = m1 * 2
            metric m5 = m2 * 3
            metric m6 = m3 + m4
            metric m7 = m3 + m5
            metric m8 = m6 + m7
            metric m9 = sqrt(m8)
            metric m10 = m9 * m1

            // Independent metrics that can be evaluated in parallel
            metric max_x = max([n.x for n in g.nodes(Value)])
            metric max_y = max([n.y for n in g.nodes(Value)])
            metric min_x = min([n.x for n in g.nodes(Value)])
            metric min_y = min([n.y for n in g.nodes(Value)])
            metric range_x = max_x - min_x
            metric range_y = max_y - min_y
        }
    "#;

    let program_ir = parse_validate_and_lower(program).unwrap();

    group.bench_function("complex_metrics", |b| {
        b.iter(|| {
            let _ = run_flow_ir(black_box(&program_ir), "MetricFlow", None);
        });
    });

    group.finish();
}

fn benchmark_rule_application(c: &mut Criterion) {
    let mut group = c.benchmark_group("rule_application");

    for num_rules in [5, 10, 20] {
        let mut rules = Vec::new();
        let mut rule_names = Vec::new();

        for i in 0..num_rules {
            let threshold = (i as f64) * 5.0;
            rules.push(format!(
                r#"
                rule Rule{} on RuleModel {{
                    Item[i]
                    where i.value > {}
                    then {{
                        i.category = "cat{}"
                    }}
                }}
                "#,
                i, threshold, i
            ));
            rule_names.push(format!("Rule{}", i));
        }

        let program = format!(
            r#"
            schema RuleSchema {{
                node Item {{
                    value: Float
                    category: String
                }}
            }}

            model RuleModel on RuleSchema {{
                Item.value ~ Gaussian(mean=50, precision=0.1)
            }}

            evidence RuleEvidence on RuleModel {{
                Item["I0"].value = 10
                Item["I1"].value = 20
                Item["I2"].value = 30
                Item["I3"].value = 40
                Item["I4"].value = 50
                Item["I5"].value = 60
                Item["I6"].value = 70
                Item["I7"].value = 80
                Item["I8"].value = 90
                Item["I9"].value = 100
            }}

            {}

            flow RuleFlow {{
                graph g = RuleEvidence
                    |> apply_ruleset({})

                metric categorized = count([n for n in g.nodes(Item) where n.category != ""])
            }}
            "#,
            rules.join("\n"),
            rule_names.join(", ")
        );

        let program_ir = parse_validate_and_lower(&program).unwrap();

        group.bench_with_input(
            BenchmarkId::new("rules", num_rules),
            &program_ir,
            |b, prog| {
                b.iter(|| {
                    let _ = run_flow_ir(black_box(prog), "RuleFlow", None);
                });
            },
        );
    }

    group.finish();
}

fn benchmark_large_graph(c: &mut Criterion) {
    let mut group = c.benchmark_group("large_graph");
    group.sample_size(10); // Reduce sample size for large graphs

    // Create a large graph with many nodes and edges
    let program = generate_test_program(500);
    let program_ir = parse_validate_and_lower(&program).unwrap();

    group.bench_function("500_nodes", |b| {
        b.iter(|| {
            let _ = run_flow_ir(black_box(&program_ir), "BenchFlow", None);
        });
    });

    group.finish();
}

criterion_group!(
    benches,
    benchmark_evidence_processing,
    benchmark_metric_evaluation,
    benchmark_rule_application,
    benchmark_large_graph
);
criterion_main!(benches);
