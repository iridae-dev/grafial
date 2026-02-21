//! Benchmarks for Dirichlet probability numeric kernel backends.
//!
//! Run with:
//! - `cargo bench --bench numeric_kernels`
//! - `cargo bench --bench numeric_kernels --features simd-kernels`
//! - `cargo bench --bench numeric_kernels --features gpu-kernels`
//! - `cargo bench --bench numeric_kernels --features simd-kernels,gpu-kernels`

use criterion::{black_box, criterion_group, criterion_main, BenchmarkId, Criterion};
use grafial_core::engine::numeric_kernels::{
    dirichlet_mean_probabilities_with_backend, KernelBackend,
};

fn make_concentrations(len: usize, seed: u64) -> Vec<f64> {
    let mut state = seed;
    let mut out = Vec::with_capacity(len);
    for _ in 0..len {
        state = state
            .wrapping_mul(6364136223846793005)
            .wrapping_add(1442695040888963407);
        let unit = ((state >> 11) as f64) / ((u64::MAX >> 11) as f64);
        out.push(0.001 + unit * 1000.0);
    }
    out
}

fn bench_dirichlet_backends(c: &mut Criterion) {
    let mut group = c.benchmark_group("dirichlet_mean_probabilities");
    for (idx, size) in [16_usize, 64, 256, 1024, 4096].iter().enumerate() {
        let concentrations = make_concentrations(*size, idx as u64 + 1);

        group.bench_with_input(
            BenchmarkId::new("scalar", size),
            &concentrations,
            |b, data| {
                b.iter(|| {
                    black_box(dirichlet_mean_probabilities_with_backend(
                        black_box(data),
                        black_box(0.01),
                        KernelBackend::Scalar,
                    ))
                });
            },
        );

        group.bench_with_input(
            BenchmarkId::new("auto", size),
            &concentrations,
            |b, data| {
                b.iter(|| {
                    black_box(dirichlet_mean_probabilities_with_backend(
                        black_box(data),
                        black_box(0.01),
                        KernelBackend::Auto,
                    ))
                });
            },
        );

        #[cfg(feature = "simd-kernels")]
        group.bench_with_input(
            BenchmarkId::new("simd_preferred", size),
            &concentrations,
            |b, data| {
                b.iter(|| {
                    black_box(dirichlet_mean_probabilities_with_backend(
                        black_box(data),
                        black_box(0.01),
                        KernelBackend::SimdPreferred,
                    ))
                });
            },
        );

        #[cfg(feature = "gpu-kernels")]
        group.bench_with_input(
            BenchmarkId::new("gpu_preferred", size),
            &concentrations,
            |b, data| {
                b.iter(|| {
                    black_box(dirichlet_mean_probabilities_with_backend(
                        black_box(data),
                        black_box(0.01),
                        KernelBackend::GpuPreferred,
                    ))
                });
            },
        );
    }
    group.finish();
}

criterion_group!(benches, bench_dirichlet_backends);
criterion_main!(benches);
