//! Tests for `DELTR`.
//!
//! The headline is [`analytic_gradient_matches_finite_differences`]: the analytic
//! gradient — including the softmax-through-a-group-mean term that is the easy
//! part to get wrong — is checked against central finite differences of the loss.
//! That is the independent ground truth. [`training_reduces_exposure_disparity`]
//! is the ablation: the disparate-exposure penalty must actually shrink the
//! exposure gap, and the utility cost is reported.

use crate::fairness_ranking::deltr::{DeltrConfig, DeltrModel, DeltrSample};

/// A fixed (non-random) training query with **discriminatory labels**: the
/// protected group (docs 0-2) carries systematically lower relevances than the
/// unprotected group (docs 3-5), which is exactly the situation `DELTR` exists
/// for — a scorer trained to fit these labels will under-expose the protected
/// group, and the disparate-exposure penalty is what pushes back.
///
/// Feature 0 is a per-document quality signal (lower for the protected group);
/// feature 1 is the group indicator (`1.0` for unprotected). Both channels let a
/// linear model separate the groups, so plain `ListNet` over-exposes the
/// unprotected one — the one-sided penalty is then active and, at the test
/// weights used below, safely away from its kink at gap = 0.
fn biased_sample() -> DeltrSample {
    let features = vec![
        0.5, 0.0, // doc 0, protected
        0.4, 0.0, // doc 1, protected
        0.3, 0.0, // doc 2, protected
        0.9, 1.0, // doc 3, unprotected
        0.8, 1.0, // doc 4, unprotected
        0.7, 1.0, // doc 5, unprotected
    ];
    let relevances = vec![0.5, 0.4, 0.3, 0.9, 0.8, 0.7];
    let protected = vec![true, true, true, false, false, false];
    DeltrSample::new(features, relevances, protected, 2).expect("valid sample")
}

// ── (DELTR gradient) analytic == finite differences ──────────────────────────

#[test]
fn analytic_gradient_matches_finite_differences() {
    let samples = vec![biased_sample()];
    let config = DeltrConfig {
        gamma: 2.0, // a large penalty weight, to exercise the exposure gradient
        l2_regularization: 0.05,
        ..DeltrConfig::default()
    };
    let mut model = DeltrModel::new(2, config).expect("valid");

    // Test weights that put the unprotected group ahead (positive second weight),
    // so the exposure gap is strictly positive and the penalty is differentiable
    // here.
    model.set_weights(vec![0.3, 1.2]).expect("ok");

    // The gap must be positive, or this test would silently exercise the
    // penalty-off branch and prove nothing about its gradient.
    assert!(
        model.exposure_gap(&samples[0]).expect("ok") > 0.05,
        "the fixture must have the penalty active"
    );

    let analytic = model.gradient(&samples).expect("ok");

    // Central finite differences: (L(w + e) - L(w - e)) / 2e per coordinate.
    let epsilon = 1e-6;
    let base = model.weights().to_vec();
    for j in 0..base.len() {
        let mut plus = base.clone();
        plus[j] += epsilon;
        model.set_weights(plus).expect("ok");
        let loss_plus = model.loss(&samples).expect("ok").total;

        let mut minus = base.clone();
        minus[j] -= epsilon;
        model.set_weights(minus).expect("ok");
        let loss_minus = model.loss(&samples).expect("ok").total;

        let numerical = (loss_plus - loss_minus) / (2.0 * epsilon);
        assert!(
            (analytic[j] - numerical).abs() < 1e-5,
            "gradient[{j}]: analytic {} vs finite-difference {numerical}",
            analytic[j]
        );
        model.set_weights(base.clone()).expect("ok");
    }
}

#[test]
fn listnet_gradient_matches_finite_differences_with_gamma_zero() {
    // With gamma = 0 the model is plain ListNet; the gradient reduces to
    // X^T (softmax(s) - softmax(y)). Check that path independently.
    let samples = vec![biased_sample()];
    let config = DeltrConfig {
        gamma: 0.0,
        l2_regularization: 0.0,
        ..DeltrConfig::default()
    };
    let mut model = DeltrModel::new(2, config).expect("valid");
    model.set_weights(vec![-0.4, 0.7]).expect("ok");

    let analytic = model.gradient(&samples).expect("ok");
    let epsilon = 1e-6;
    let base = model.weights().to_vec();
    for j in 0..base.len() {
        let mut plus = base.clone();
        plus[j] += epsilon;
        model.set_weights(plus).expect("ok");
        let loss_plus = model.loss(&samples).expect("ok").total;
        let mut minus = base.clone();
        minus[j] -= epsilon;
        model.set_weights(minus).expect("ok");
        let loss_minus = model.loss(&samples).expect("ok").total;
        let numerical = (loss_plus - loss_minus) / (2.0 * epsilon);
        assert!((analytic[j] - numerical).abs() < 1e-6);
        model.set_weights(base.clone()).expect("ok");
    }
}

// ── (DELTR ablation) the penalty shrinks the exposure gap ────────────────────

#[test]
fn training_reduces_exposure_disparity() {
    // Train two models on the same biased query: one plain ListNet (gamma = 0),
    // one with a strong disparate-exposure penalty. The penalized model must end
    // with a smaller surrogate exposure gap. The listwise-loss difference is the
    // utility price and is reported.
    let samples = vec![biased_sample()];

    let mut plain = DeltrModel::new(
        2,
        DeltrConfig {
            gamma: 0.0,
            learning_rate: 0.5,
            iterations: 2000,
            seed: 1,
            ..DeltrConfig::default()
        },
    )
    .expect("valid");
    plain.train(&samples).expect("trains");

    let mut fair = DeltrModel::new(
        2,
        DeltrConfig {
            gamma: 20.0,
            learning_rate: 0.5,
            iterations: 2000,
            seed: 1,
            ..DeltrConfig::default()
        },
    )
    .expect("valid");
    fair.train(&samples).expect("trains");

    let plain_gap = plain.exposure_gap(&samples[0]).expect("ok");
    let fair_gap = fair.exposure_gap(&samples[0]).expect("ok");
    let plain_loss = plain.loss(&samples).expect("ok").listwise;
    let fair_loss = fair.loss(&samples).expect("ok").listwise;

    // The measured numbers, for the record. The plain model over-exposes the
    // unprotected group; the penalized one does not.
    println!(
        "DELTR ablation: plain exposure gap = {plain_gap:.6}, fair exposure gap = {fair_gap:.6}, \
         listwise loss plain = {plain_loss:.6}, fair = {fair_loss:.6}"
    );

    assert!(
        fair_gap < plain_gap - 0.02,
        "the penalty must strictly shrink the exposure gap: plain {plain_gap}, fair {fair_gap}"
    );
    // The fair model has driven the one-sided gap to essentially zero.
    assert!(
        fair_gap < 0.02,
        "fair gap should be near zero, was {fair_gap}"
    );
    // Utility is not destroyed: the listwise loss rises, but only modestly.
    assert!(
        fair_loss >= plain_loss - 1e-9,
        "penalizing exposure cannot *improve* the listwise fit"
    );
    assert!(
        fair_loss - plain_loss < 0.5,
        "utility cost {} is larger than expected",
        fair_loss - plain_loss
    );
}

#[test]
fn training_loss_decreases_and_converges() {
    // A basic sanity check that gradient descent is descending: the total loss at
    // the end is below the loss at the start, and the gradient norm shrinks.
    let samples = vec![biased_sample()];
    let mut model = DeltrModel::new(
        2,
        DeltrConfig {
            gamma: 1.0,
            learning_rate: 0.3,
            iterations: 1000,
            seed: 7,
            ..DeltrConfig::default()
        },
    )
    .expect("valid");
    let history = model.train(&samples).expect("trains").to_vec();
    assert!(history.len() >= 2);
    assert!(
        history.last().unwrap().total <= history.first().unwrap().total + 1e-9,
        "loss should not increase over training"
    );
    assert!(history.last().unwrap().gradient_norm <= history.first().unwrap().gradient_norm + 1e-9);
}

// ── determinism and guards ───────────────────────────────────────────────────

#[test]
fn training_is_reproducible_from_the_seed() {
    let samples = vec![biased_sample()];
    let config = DeltrConfig {
        seed: 42,
        iterations: 100,
        ..DeltrConfig::default()
    };
    let mut a = DeltrModel::new(2, config.clone()).expect("valid");
    let mut b = DeltrModel::new(2, config).expect("valid");
    a.train(&samples).expect("ok");
    b.train(&samples).expect("ok");
    assert_eq!(a.weights(), b.weights());
}

#[test]
fn invalid_configuration_is_rejected() {
    assert!(DeltrModel::new(0, DeltrConfig::default()).is_err());
    assert!(
        DeltrModel::new(
            2,
            DeltrConfig {
                learning_rate: -0.1,
                ..DeltrConfig::default()
            }
        )
        .is_err()
    );
    assert!(
        DeltrModel::new(
            2,
            DeltrConfig {
                gamma: -1.0,
                ..DeltrConfig::default()
            }
        )
        .is_err()
    );
    // A NaN feature is rejected at sample construction.
    assert!(DeltrSample::new(vec![f64::NAN, 0.0], vec![1.0], vec![true], 2).is_err());
}
