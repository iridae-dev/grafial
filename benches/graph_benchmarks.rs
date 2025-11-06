//! # Grafial Performance Benchmarks
//!
//! Phase 7 benchmarks for scale testing (10k-100k nodes).
//! Tests key operations:
//! - Evidence application
//! - Rule evaluation
//! - Metric computation
//! - Adjacency queries
//!

use criterion::{black_box, criterion_group, criterion_main, BenchmarkId, Criterion, Throughput};
use std::collections::HashMap;

use grafial::engine::graph::{BeliefGraph, BetaPosterior, EdgeData, GaussianPosterior, NodeData, NodeId, EdgeId};
use grafial::engine::rule_exec::run_rule_for_each;
use grafial::frontend::ast::{ActionStmt, EdgePattern, ExprAst, NodePattern, PatternItem, RuleDef, CallArg};

/// Creates a synthetic belief graph for benchmarking.
///
/// Generates a graph with:
/// - `num_nodes` nodes with a single numeric attribute
/// - Random edges with specified density
/// - Deterministic structure for reproducibility
fn create_synthetic_graph(num_nodes: usize, edge_density: f64) -> BeliefGraph {
    let mut graph = BeliefGraph::default();

    // Create nodes
    for i in 0..num_nodes {
        let mut attrs = HashMap::new();
        attrs.insert(
            "value".to_string(),
            GaussianPosterior {
                mean: (i as f64) * 0.1,
                precision: 1.0,
            },
        );
        graph.insert_node(NodeData {
            id: NodeId(i as u32),
            label: "Person".to_string(),
            attrs,
        });
    }

    // Create edges with specified density
    let num_edges = (num_nodes as f64 * num_nodes as f64 * edge_density) as usize;
    for i in 0..num_edges {
        let src = NodeId((i % num_nodes) as u32);
        let dst = NodeId(((i * 7) % num_nodes) as u32); // Prime multiplier for distribution
        graph.insert_edge(EdgeData {
            id: EdgeId(i as u32),
            src,
            dst,
            ty: "KNOWS".to_string(),
            exist: BetaPosterior {
                alpha: 2.0 + (i % 10) as f64 * 0.1,
                beta: 2.0 + ((i * 3) % 10) as f64 * 0.1,
            },
        });
    }

    graph
}

/// Benchmarks evidence application (Bayesian updates) at scale.
fn bench_evidence_application(c: &mut Criterion) {
    let mut group = c.benchmark_group("evidence_application");

    for size in [100, 1000, 10_000].iter() {
        group.throughput(Throughput::Elements(*size as u64));
        group.bench_with_input(BenchmarkId::from_parameter(size), size, |b, &size| {
            let mut graph = create_synthetic_graph(size, 0.01);
            b.iter(|| {
                // Simulate evidence application on random nodes
                for i in 0..10 {
                    let node_id = NodeId((i * 7) % size as u32);
                    graph.observe_attr(node_id, "value", 5.0, 1.0).unwrap();
                }
            });
        });
    }

    group.finish();
}

/// Benchmarks rule evaluation performance at scale.
fn bench_rule_evaluation(c: &mut Criterion) {
    let mut group = c.benchmark_group("rule_evaluation");

    for size in [100, 1000, 10_000].iter() {
        group.throughput(Throughput::Elements(*size as u64));
        group.bench_with_input(BenchmarkId::from_parameter(size), size, |b, &size| {
            let graph = create_synthetic_graph(size, 0.01);

            // Simple rule: filter edges by probability
            let rule = RuleDef {
                name: "TestRule".to_string(),
                on_model: "TestModel".to_string(),
                patterns: vec![PatternItem {
                    src: NodePattern {
                        var: "A".to_string(),
                        label: "Person".to_string(),
                    },
                    edge: EdgePattern {
                        var: "e".to_string(),
                        ty: "KNOWS".to_string(),
                    },
                    dst: NodePattern {
                        var: "B".to_string(),
                        label: "Person".to_string(),
                    },
                }],
                where_expr: Some(ExprAst::Binary {
                    op: grafial::frontend::ast::BinaryOp::Gt,
                    left: Box::new(ExprAst::Call {
                        name: "prob".to_string(),
                        args: vec![CallArg::Positional(ExprAst::Var("e".to_string()))],
                    }),
                    right: Box::new(ExprAst::Number(0.5)),
                }),
                actions: vec![ActionStmt::ForceAbsent {
                    edge_var: "e".to_string(),
                }],
                mode: Some("for_each".to_string()),
            };

            b.iter(|| {
                let result = run_rule_for_each(black_box(&graph), black_box(&rule));
                black_box(result).unwrap();
            });
        });
    }

    group.finish();
}

/// Benchmarks adjacency index build and query performance.
fn bench_adjacency_queries(c: &mut Criterion) {
    let mut group = c.benchmark_group("adjacency_queries");

    for size in [100, 1000, 10_000, 100_000].iter() {
        group.throughput(Throughput::Elements(*size as u64));
        group.bench_with_input(BenchmarkId::from_parameter(size), size, |b, &size| {
            let mut graph = create_synthetic_graph(size, 0.001); // Lower density for large graphs
            graph.build_adjacency();

            b.iter(|| {
                // Query adjacency for multiple nodes
                for i in 0..100 {
                    let node_id = NodeId((i * 7) % size as u32);
                    let edges = graph.get_outgoing_edges(node_id, "KNOWS");
                    black_box(edges);
                }
            });
        });
    }

    group.finish();
}

/// Benchmarks degree computation at scale.
fn bench_degree_computation(c: &mut Criterion) {
    let mut group = c.benchmark_group("degree_computation");

    for size in [100, 1000, 10_000].iter() {
        group.throughput(Throughput::Elements(*size as u64));
        group.bench_with_input(BenchmarkId::from_parameter(size), size, |b, &size| {
            let graph = create_synthetic_graph(size, 0.01);

            b.iter(|| {
                // Compute degree for multiple nodes
                let mut total_degree = 0;
                for i in 0..100 {
                    let node_id = NodeId((i * 7) % size as u32);
                    total_degree += graph.degree_outgoing(node_id, 0.5);
                }
                black_box(total_degree);
            });
        });
    }

    group.finish();
}

/// Benchmarks graph cloning (important for immutable semantics).
fn bench_graph_clone(c: &mut Criterion) {
    let mut group = c.benchmark_group("graph_clone");

    for size in [100, 1000, 10_000].iter() {
        group.throughput(Throughput::Elements(*size as u64));
        group.bench_with_input(BenchmarkId::from_parameter(size), size, |b, &size| {
            let graph = create_synthetic_graph(size, 0.01);

            b.iter(|| {
                let cloned = black_box(&graph).clone();
                black_box(cloned);
            });
        });
    }

    group.finish();
}

criterion_group!(
    benches,
    bench_evidence_application,
    bench_rule_evaluation,
    bench_adjacency_queries,
    bench_degree_computation,
    bench_graph_clone,
);
criterion_main!(benches);
