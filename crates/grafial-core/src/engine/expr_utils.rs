//! Utility functions for expression argument parsing.
//!
//! Provides reusable functions for extracting positional and named arguments
//! from CallArg slices, eliminating duplication across different evaluation contexts.

use grafial_frontend::ast::{CallArg, ExprAst};

/// Splits call arguments into positional and named arguments.
///
/// This is a utility function used by multiple expression evaluators to
/// parse function call arguments consistently.
///
/// # Returns
///
/// A tuple of:
/// - `Vec<&ExprAst>`: Positional arguments in order
/// - `Vec<(&str, &ExprAst)>`: Named arguments as (name, value) pairs
pub fn split_args(args: &[CallArg]) -> (Vec<&ExprAst>, Vec<(&str, &ExprAst)>) {
    let mut pos = Vec::new();
    let mut named = Vec::new();
    for a in args {
        match a {
            CallArg::Positional(e) => pos.push(e),
            CallArg::Named { name, value } => named.push((name.as_str(), value)),
        }
    }
    (pos, named)
}

/// Approximate inverse CDF (quantile) of the standard normal distribution.
///
/// Uses a rational approximation by Peter John Acklam (2003), with domain
/// split into central and tail regions. Accuracy is ~1e-9 for double precision.
pub fn inv_norm_cdf(p: f64) -> f64 {
    // Coefficients in rational approximations
    const A: [f64; 6] = [
        -3.969683028665376e+01,
        2.209460984245205e+02,
        -2.759285104469687e+02,
        1.383_577_518_672_69e+02,
        -3.066479806614716e+01,
        2.506628277459239e+00,
    ];
    const B: [f64; 5] = [
        -5.447609879822406e+01,
        1.615858368580409e+02,
        -1.556989798598866e+02,
        6.680131188771972e+01,
        -1.328068155288572e+01,
    ];
    const C: [f64; 6] = [
        -7.784894002430293e-03,
        -3.223964580411365e-01,
        -2.400758277161838e+00,
        -2.549732539343734e+00,
        4.374664141464968e+00,
        2.938163982698783e+00,
    ];
    const D: [f64; 4] = [
        7.784695709041462e-03,
        3.224671290700398e-01,
        2.445134137142996e+00,
        3.754408661907416e+00,
    ];

    // Define break-points
    const P_LOW: f64 = 0.02425; // lower region 0..P_LOW
    const P_HIGH: f64 = 1.0 - P_LOW; // upper region P_HIGH..1

    if p <= 0.0 {
        return f64::NEG_INFINITY;
    }
    if p >= 1.0 {
        return f64::INFINITY;
    }

    if p < P_LOW {
        // Rational approximation for lower tail:
        // x = (((((c1 q + c2) q + c3) q + c4) q + c5) q + c6)
        //     / ((((d1 q + d2) q + d3) q + d4) q + 1)
        let q = (-2.0 * p.ln()).sqrt();
        let num = ((((C[0] * q + C[1]) * q + C[2]) * q + C[3]) * q + C[4]) * q + C[5];
        let den = (((D[0] * q + D[1]) * q + D[2]) * q + D[3]) * q + 1.0;
        num / den
    } else if p <= P_HIGH {
        // Rational approximation for central region
        let q = p - 0.5;
        let r = q * q;
        (((((A[0] * r + A[1]) * r + A[2]) * r + A[3]) * r + A[4]) * r + A[5]) * q
            / (((((B[0] * r + B[1]) * r + B[2]) * r + B[3]) * r + B[4]) * r + 1.0)
    } else {
        // Rational approximation for upper tail (mirror of the lower tail)
        let q = (-2.0 * (1.0 - p).ln()).sqrt();
        let num = ((((C[0] * q + C[1]) * q + C[2]) * q + C[3]) * q + C[4]) * q + C[5];
        let den = (((D[0] * q + D[1]) * q + D[2]) * q + D[3]) * q + 1.0;
        -(num / den)
    }
}

/// Approximate standard normal CDF Φ(z) using rational approximation.
///
/// Accuracy is sufficient for statistical thresholding and CI bounds.
pub fn norm_cdf(z: f64) -> f64 {
    // Abramowitz and Stegun formula 7.1.26 based approximation
    // Handles symmetry for negative z for better accuracy
    let x = z;
    let abs_x = x.abs();
    let t = 1.0 / (1.0 + 0.231_641_9 * abs_x);
    let poly = (((((1.330_274_429 * t - 1.821_255_978) * t) + 1.781_477_937) * t - 0.356_563_782)
        * t
        + 0.319_381_530)
        * t;
    let phi = (-(abs_x * abs_x) / 2.0).exp() / (2.0 * std::f64::consts::PI).sqrt();
    let cdf_approx = 1.0 - phi * poly;
    if x >= 0.0 {
        cdf_approx
    } else {
        1.0 - cdf_approx
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn split_args_positional_only() {
        let args = vec![
            CallArg::Positional(ExprAst::Number(1.0)),
            CallArg::Positional(ExprAst::Number(2.0)),
        ];
        let (pos, named) = split_args(&args);
        assert_eq!(pos.len(), 2);
        assert_eq!(named.len(), 0);
    }

    #[test]
    fn split_args_mixed() {
        let args = vec![
            CallArg::Positional(ExprAst::Number(1.0)),
            CallArg::Named {
                name: "x".into(),
                value: ExprAst::Number(2.0),
            },
            CallArg::Positional(ExprAst::Number(3.0)),
        ];
        let (pos, named) = split_args(&args);
        assert_eq!(pos.len(), 2);
        assert_eq!(named.len(), 1);
        assert_eq!(named[0].0, "x");
    }

    #[test]
    fn inv_norm_cdf_matches_known_quantiles() {
        // Reference values of Φ⁻¹(p); Acklam's approximation is accurate to ~1.15e-9,
        // so 1e-6 comfortably covers all three branches including both tails.
        let cases = [
            (0.001, -3.090_232_306_167_813),
            (0.01, -2.326_347_874_040_841),
            (0.025, -1.959_963_984_540_054),
            (0.5, 0.0),
            (0.975, 1.959_963_984_540_054),
            (0.99, 2.326_347_874_040_841),
            (0.999, 3.090_232_306_167_813),
        ];
        for (p, expected) in cases {
            let got = inv_norm_cdf(p);
            assert!(
                (got - expected).abs() < 1e-6,
                "inv_norm_cdf({}) = {}, expected {}",
                p,
                got,
                expected
            );
        }
    }

    #[test]
    fn inv_norm_cdf_round_trips_norm_cdf_in_tails() {
        // norm_cdf is a coarser approximation (~1e-7), so allow 1e-5.
        for p in [0.005, 0.02, 0.05, 0.5, 0.95, 0.98, 0.995] {
            let z = inv_norm_cdf(p);
            let back = norm_cdf(z);
            assert!(
                (back - p).abs() < 1e-5,
                "round trip failed: p={}, z={}, back={}",
                p,
                z,
                back
            );
        }
    }
}
