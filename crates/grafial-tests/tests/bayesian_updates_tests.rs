//! Analytical tests for core Bayesian posterior updates.
//!
//! These tests validate Gaussian, Beta, and Dirichlet updates against
//! known closed-form conjugate formulas.

use grafial_core::engine::graph::{BetaPosterior, DirichletPosterior, GaussianPosterior};

fn assert_close(actual: f64, expected: f64, tol: f64, label: &str) {
    assert!(
        (actual - expected).abs() <= tol,
        "{} mismatch: expected {:.15}, got {:.15}, diff={:.3e}",
        label,
        expected,
        actual,
        (actual - expected).abs()
    );
}

#[test]
fn gaussian_single_update_matches_closed_form() {
    // Prior: N(mu0=10, tau0=2), observation x=13 with tau_obs=3
    // Posterior:
    //   tau_n = tau0 + tau_obs = 5
    //   mu_n  = (tau0*mu0 + tau_obs*x) / tau_n = 11.8
    let mut posterior = GaussianPosterior {
        mean: 10.0,
        precision: 2.0,
    };
    posterior.update(13.0, 3.0);

    assert_close(posterior.precision, 5.0, 1e-12, "gaussian precision");
    assert_close(posterior.mean, 11.8, 1e-12, "gaussian mean");
}

#[test]
fn gaussian_multiple_updates_match_closed_form() {
    // Prior: N(mu0=5, tau0=1.5)
    // Observations: (7,2), (2,0.5), (10,1)
    // tau_n = 1.5 + 2 + 0.5 + 1 = 5
    // mu_n = (1.5*5 + 2*7 + 0.5*2 + 1*10) / 5 = 6.5
    let mut posterior = GaussianPosterior {
        mean: 5.0,
        precision: 1.5,
    };
    for (x, tau_obs) in [(7.0, 2.0), (2.0, 0.5), (10.0, 1.0)] {
        posterior.update(x, tau_obs);
    }

    assert_close(posterior.precision, 5.0, 1e-12, "gaussian precision");
    assert_close(posterior.mean, 6.5, 1e-12, "gaussian mean");
}

#[test]
fn beta_updates_match_beta_bernoulli_closed_form() {
    // Prior Beta(alpha=2, beta=3), observations: T, F, T, T, F, F
    // alpha_n = 2 + 3 = 5
    // beta_n  = 3 + 3 = 6
    let mut posterior = BetaPosterior {
        alpha: 2.0,
        beta: 3.0,
    };
    for observed_present in [true, false, true, true, false, false] {
        posterior.observe(observed_present);
    }

    assert_close(posterior.alpha, 5.0, 1e-12, "beta alpha");
    assert_close(posterior.beta, 6.0, 1e-12, "beta beta");
}

#[test]
fn beta_mean_and_variance_match_closed_form() {
    // Beta(5,6):
    // E[p] = 5 / 11
    // Var[p] = 5*6 / (11^2 * 12)
    let posterior = BetaPosterior {
        alpha: 5.0,
        beta: 6.0,
    };
    let expected_mean = 5.0 / 11.0;
    let expected_variance = (5.0 * 6.0) / (11.0 * 11.0 * 12.0);

    assert_close(
        posterior.mean_probability(),
        expected_mean,
        1e-12,
        "beta mean",
    );
    assert_close(
        posterior.variance(),
        expected_variance,
        1e-12,
        "beta variance",
    );
}

#[test]
fn dirichlet_chosen_updates_match_closed_form_counts() {
    // Prior alpha = [1,2,3]
    // Observed categories: [2,0,2,1,2,2]
    // Counts: [1,1,4]
    // Posterior alpha = [2,3,7]
    let mut posterior = DirichletPosterior::new(vec![1.0, 2.0, 3.0]);
    for category in [2usize, 0, 2, 1, 2, 2] {
        posterior.observe_chosen(category);
    }

    assert_eq!(posterior.concentrations, vec![2.0, 3.0, 7.0]);
}

#[test]
fn dirichlet_mean_probabilities_match_closed_form() {
    // For alpha = [2,3,7], E[pi_k] = alpha_k / sum(alpha) = alpha_k / 12
    let posterior = DirichletPosterior::new(vec![2.0, 3.0, 7.0]);
    let means = posterior.mean_probabilities();

    assert_close(means[0], 2.0 / 12.0, 1e-12, "dirichlet mean[0]");
    assert_close(means[1], 3.0 / 12.0, 1e-12, "dirichlet mean[1]");
    assert_close(means[2], 7.0 / 12.0, 1e-12, "dirichlet mean[2]");
    assert_close(means.iter().sum::<f64>(), 1.0, 1e-12, "dirichlet mean sum");
}

#[test]
fn dirichlet_binary_unchosen_is_equivalent_to_other_category_chosen() {
    let mut posterior = DirichletPosterior::new(vec![4.0, 1.0]);

    // In K=2, "category 0 unchosen" => category 1 chosen.
    posterior.observe_unchosen(0).unwrap();
    assert_eq!(posterior.concentrations, vec![4.0, 2.0]);
}
