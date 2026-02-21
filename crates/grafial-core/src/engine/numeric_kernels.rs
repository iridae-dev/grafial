//! Numeric kernels for probability computations.
//!
//! This module contains optimized and scalar-reference kernels for probability
//! vectors. Optimized paths are always feature-gated and threshold-gated.

/// Minimum vector length before attempting SIMD path for Dirichlet means.
///
/// This threshold is intentionally conservative and should be revisited with
/// benchmark data before any default enablement.
pub const DIRICHLET_SIMD_MIN_LEN: usize = 64;

/// Equivalence epsilon for optimized-vs-reference numerical checks.
pub const KERNEL_EQUIVALENCE_EPSILON: f64 = 1e-12;

/// Backend selector for numeric kernels.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum KernelBackend {
    /// Always use scalar reference implementation.
    Scalar,
    /// Use optimized implementation when available, otherwise scalar.
    Auto,
    /// Prefer optimized implementation even below thresholds, fallback to scalar.
    #[cfg(feature = "simd-kernels")]
    SimdPreferred,
}

/// Public entrypoint for Dirichlet mean probability vector kernel.
///
/// Computes `p_k = max(alpha_k, min_param) / sum_j max(alpha_j, min_param)`.
pub fn dirichlet_mean_probabilities(concentrations: &[f64], min_param: f64) -> Vec<f64> {
    dirichlet_mean_probabilities_with_backend(concentrations, min_param, KernelBackend::Auto)
}

/// Dirichlet mean probability vector with explicit backend selection.
pub fn dirichlet_mean_probabilities_with_backend(
    concentrations: &[f64],
    min_param: f64,
    backend: KernelBackend,
) -> Vec<f64> {
    if concentrations.is_empty() {
        return Vec::new();
    }

    match backend {
        KernelBackend::Scalar => dirichlet_mean_probabilities_scalar(concentrations, min_param),
        KernelBackend::Auto => {
            #[cfg(feature = "simd-kernels")]
            {
                if concentrations.len() >= DIRICHLET_SIMD_MIN_LEN {
                    if let Some(simd_probs) =
                        dirichlet_mean_probabilities_simd(concentrations, min_param)
                    {
                        return simd_probs;
                    }
                }
            }
            dirichlet_mean_probabilities_scalar(concentrations, min_param)
        }
        #[cfg(feature = "simd-kernels")]
        KernelBackend::SimdPreferred => {
            if let Some(simd_probs) = dirichlet_mean_probabilities_simd(concentrations, min_param) {
                return simd_probs;
            }
            dirichlet_mean_probabilities_scalar(concentrations, min_param)
        }
    }
}

/// Scalar reference implementation for Dirichlet mean probabilities.
pub fn dirichlet_mean_probabilities_scalar(concentrations: &[f64], min_param: f64) -> Vec<f64> {
    let sum_alpha: f64 = concentrations.iter().map(|&a| a.max(min_param)).sum();
    concentrations
        .iter()
        .map(|&a| a.max(min_param) / sum_alpha)
        .collect()
}

#[cfg(all(feature = "simd-kernels", target_arch = "x86_64"))]
fn dirichlet_mean_probabilities_simd(concentrations: &[f64], min_param: f64) -> Option<Vec<f64>> {
    if std::arch::is_x86_feature_detected!("avx2") {
        // SAFETY: AVX2 support is checked at runtime before calling.
        return Some(unsafe { dirichlet_mean_probabilities_avx2(concentrations, min_param) });
    }
    None
}

#[cfg(all(feature = "simd-kernels", not(target_arch = "x86_64")))]
fn dirichlet_mean_probabilities_simd(_: &[f64], _: f64) -> Option<Vec<f64>> {
    None
}

#[cfg(all(feature = "simd-kernels", target_arch = "x86_64"))]
#[target_feature(enable = "avx2")]
unsafe fn dirichlet_mean_probabilities_avx2(concentrations: &[f64], min_param: f64) -> Vec<f64> {
    use std::arch::x86_64::{
        _mm256_add_pd, _mm256_loadu_pd, _mm256_max_pd, _mm256_mul_pd, _mm256_set1_pd,
        _mm256_setzero_pd, _mm256_storeu_pd,
    };

    let len = concentrations.len();
    let mut i = 0usize;
    let min_vec = _mm256_set1_pd(min_param);
    let mut sum_vec = _mm256_setzero_pd();

    while i + 4 <= len {
        // SAFETY: loadu/storeu support unaligned access; bounds guarded by loop condition.
        let values = unsafe { _mm256_loadu_pd(concentrations.as_ptr().add(i)) };
        let clipped = _mm256_max_pd(values, min_vec);
        sum_vec = _mm256_add_pd(sum_vec, clipped);
        i += 4;
    }

    let mut sum_lanes = [0.0_f64; 4];
    // SAFETY: output buffer has exactly 4 f64 lanes.
    unsafe { _mm256_storeu_pd(sum_lanes.as_mut_ptr(), sum_vec) };
    let mut sum_alpha = sum_lanes.iter().sum::<f64>();
    for &a in &concentrations[i..] {
        sum_alpha += a.max(min_param);
    }

    let inv_sum = 1.0 / sum_alpha;
    let inv_sum_vec = _mm256_set1_pd(inv_sum);
    let mut probs = vec![0.0_f64; len];
    i = 0;
    while i + 4 <= len {
        // SAFETY: loadu/storeu support unaligned access; bounds guarded by loop condition.
        let values = unsafe { _mm256_loadu_pd(concentrations.as_ptr().add(i)) };
        let clipped = _mm256_max_pd(values, min_vec);
        let lane_probs = _mm256_mul_pd(clipped, inv_sum_vec);
        // SAFETY: destination range [i..i+4] is in-bounds by loop condition.
        unsafe { _mm256_storeu_pd(probs.as_mut_ptr().add(i), lane_probs) };
        i += 4;
    }
    for j in i..len {
        probs[j] = concentrations[j].max(min_param) * inv_sum;
    }
    probs
}

#[cfg(test)]
mod tests {
    use super::*;

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

    fn assert_close_vec(lhs: &[f64], rhs: &[f64], eps: f64) {
        assert_eq!(lhs.len(), rhs.len());
        for (i, (a, b)) in lhs.iter().zip(rhs.iter()).enumerate() {
            let diff = (a - b).abs();
            assert!(
                diff <= eps,
                "difference at index {} too large: |{} - {}| = {} (eps={})",
                i,
                a,
                b,
                diff,
                eps
            );
        }
    }

    #[test]
    fn dirichlet_scalar_normalizes() {
        let concentrations = vec![1.0, 2.0, 3.0, 4.0];
        let probs =
            dirichlet_mean_probabilities_with_backend(&concentrations, 0.01, KernelBackend::Scalar);
        let sum: f64 = probs.iter().sum();
        assert!((sum - 1.0).abs() <= KERNEL_EQUIVALENCE_EPSILON);
    }

    #[test]
    fn dirichlet_auto_matches_scalar_reference() {
        let sizes = [1, 2, 3, 4, 7, 8, 15, 16, 31, 32, 63, 64, 65, 128, 257];
        for (idx, size) in sizes.iter().enumerate() {
            let concentrations = make_concentrations(*size, 1000 + idx as u64);
            let scalar = dirichlet_mean_probabilities_with_backend(
                &concentrations,
                0.01,
                KernelBackend::Scalar,
            );
            let auto = dirichlet_mean_probabilities_with_backend(
                &concentrations,
                0.01,
                KernelBackend::Auto,
            );
            assert_close_vec(&scalar, &auto, KERNEL_EQUIVALENCE_EPSILON);
        }
    }

    #[test]
    fn dirichlet_auto_is_deterministic_across_repeated_runs() {
        let concentrations = make_concentrations(256, 42);
        let first =
            dirichlet_mean_probabilities_with_backend(&concentrations, 0.01, KernelBackend::Auto);
        for _ in 0..25 {
            let next = dirichlet_mean_probabilities_with_backend(
                &concentrations,
                0.01,
                KernelBackend::Auto,
            );
            assert_close_vec(&first, &next, 0.0);
        }
    }

    #[cfg(feature = "simd-kernels")]
    #[test]
    fn dirichlet_simd_backend_matches_scalar_reference() {
        let concentrations = make_concentrations(512, 1234);
        let scalar =
            dirichlet_mean_probabilities_with_backend(&concentrations, 0.01, KernelBackend::Scalar);
        let simd = dirichlet_mean_probabilities_with_backend(
            &concentrations,
            0.01,
            KernelBackend::SimdPreferred,
        );
        assert_close_vec(&scalar, &simd, KERNEL_EQUIVALENCE_EPSILON);
    }
}
