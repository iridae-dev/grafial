//! Benchmarks comparing scalar vs vectorized evidence ingestion.
//!
//! Run with: cargo bench --bench vectorized_evidence --features vectorized

use criterion::{black_box, criterion_group, criterion_main, BenchmarkId, Criterion};
use grafial_core::engine::graph::{BeliefGraph, BetaPosterior, GaussianPosterior, NodeId};
#[cfg(feature = "vectorized")]
use grafial_core::engine::vectorized::{beta_batch_update, gaussian_batch_update};
use std::collections::HashMap;

/// Helper to create a test graph with nodes and attributes
fn setup_test_graph(num_nodes: usize, attrs_per_node: usize) -> BeliefGraph {
    let mut graph = BeliefGraph::default();

    for i in 0..num_nodes {
        let mut attrs = HashMap::new();
        for j in 0..attrs_per_node {
            attrs.insert(
                format!("attr_{}", j),
                GaussianPosterior {
                    mean: 0.0,
                    precision: 1.0,
                },
            );
        }
        graph.add_node(format!("node_{}", i), attrs);
    }

    graph
}

/// Benchmark scalar Gaussian updates (sequential)
fn bench_gaussian_scalar(c: &mut Criterion) {
    let mut group = c.benchmark_group("gaussian_updates");

    for num_obs in [10, 100, 1000] {
        group.bench_with_input(BenchmarkId::new("scalar", num_obs), &num_obs, |b, &n| {
            b.iter(|| {
                let mut posterior = GaussianPosterior {
                    mean: 0.0,
                    precision: 1.0,
                };

                for i in 0..n {
                    let value = i as f64;
                    let tau_obs = 1.0;

                    // Simulate scalar update (from graph.rs observe_attr logic)
                    let tau_old = posterior.precision;
                    let tau_new = tau_old + tau_obs;
                    let mu_new = (tau_old * posterior.mean + tau_obs * value) / tau_new;

                    posterior = GaussianPosterior {
                        mean: mu_new,
                        precision: tau_new,
                    };
                }

                black_box(posterior)
            });
        });
    }

    group.finish();
}

/// Benchmark vectorized Gaussian updates (batch)
#[cfg(feature = "vectorized")]
fn bench_gaussian_vectorized(c: &mut Criterion) {
    let mut group = c.benchmark_group("gaussian_updates");

    for num_obs in [10, 100, 1000] {
        group.bench_with_input(
            BenchmarkId::new("vectorized", num_obs),
            &num_obs,
            |b, &n| {
                b.iter(|| {
                    let prior = GaussianPosterior {
                        mean: 0.0,
                        precision: 1.0,
                    };

                    let observations: Vec<(f64, f64)> = (0..n).map(|i| (i as f64, 1.0)).collect();

                    let result = gaussian_batch_update(&prior, &observations).unwrap();
                    black_box(result)
                });
            },
        );
    }

    group.finish();
}

/// Benchmark scalar Beta updates (sequential)
fn bench_beta_scalar(c: &mut Criterion) {
    let mut group = c.benchmark_group("beta_updates");

    for num_obs in [10, 100, 1000] {
        group.bench_with_input(BenchmarkId::new("scalar", num_obs), &num_obs, |b, &n| {
            b.iter(|| {
                let mut posterior = BetaPosterior {
                    alpha: 1.0,
                    beta: 1.0,
                };

                for i in 0..n {
                    let present = i % 2 == 0;

                    // Simulate scalar update (from graph.rs observe_edge logic)
                    if present {
                        posterior.alpha += 1.0;
                    } else {
                        posterior.beta += 1.0;
                    }
                }

                black_box(posterior)
            });
        });
    }

    group.finish();
}

/// Benchmark vectorized Beta updates (batch)
#[cfg(feature = "vectorized")]
fn bench_beta_vectorized(c: &mut Criterion) {
    let mut group = c.benchmark_group("beta_updates");

    for num_obs in [10, 100, 1000] {
        group.bench_with_input(
            BenchmarkId::new("vectorized", num_obs),
            &num_obs,
            |b, &n| {
                b.iter(|| {
                    let prior = BetaPosterior {
                        alpha: 1.0,
                        beta: 1.0,
                    };

                    let observations: Vec<bool> = (0..n).map(|i| i % 2 == 0).collect();

                    let result = beta_batch_update(&prior, &observations).unwrap();
                    black_box(result)
                });
            },
        );
    }

    group.finish();
}

/// Benchmark full evidence ingestion pipeline
fn bench_evidence_pipeline(c: &mut Criterion) {
    let mut group = c.benchmark_group("evidence_pipeline");

    // Test with different scales of evidence
    for (num_nodes, observations_per_node) in [(10, 100), (100, 10), (50, 50)] {
        let test_name = format!("{}nodes_{}obs", num_nodes, observations_per_node);

        group.bench_function(&test_name, |b| {
            b.iter(|| {
                let mut graph = setup_test_graph(num_nodes, 3);

                // Simulate evidence ingestion
                for node_idx in 0..num_nodes {
                    let node_id = NodeId(node_idx as u32);

                    // Apply multiple observations to each attribute
                    for attr_idx in 0..3 {
                        let attr_name = format!("attr_{}", attr_idx);

                        #[cfg(feature = "vectorized")]
                        {
                            // Vectorized path: batch all observations
                            let observations: Vec<(f64, f64)> = (0..observations_per_node)
                                .map(|i| (i as f64, 1.0))
                                .collect();

                            graph
                                .observe_attr_batch(node_id, &attr_name, &observations)
                                .unwrap();
                        }

                        #[cfg(not(feature = "vectorized"))]
                        {
                            // Scalar path: apply observations one by one
                            for i in 0..observations_per_node {
                                graph
                                    .observe_attr(node_id, &attr_name, i as f64, 1.0)
                                    .unwrap();
                            }
                        }
                    }
                }

                black_box(graph)
            });
        });
    }

    group.finish();
}

// Select benchmarks based on features
#[cfg(feature = "vectorized")]
criterion_group!(
    benches,
    bench_gaussian_scalar,
    bench_gaussian_vectorized,
    bench_beta_scalar,
    bench_beta_vectorized,
    bench_evidence_pipeline
);

#[cfg(not(feature = "vectorized"))]
criterion_group!(
    benches,
    bench_gaussian_scalar,
    bench_beta_scalar,
    bench_evidence_pipeline
);

criterion_main!(benches);
