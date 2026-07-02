//! The pairwise `RankNet`-style training loop — the algorithmic core of
//! `learning_to_rank`.
//!
//! # The objective
//!
//! Given preference pairs `(x_a, x_b, y)` where `y` is the target probability
//! that `x_a` outranks `x_b`, the model scores each document linearly,
//! `s = w · x`, and models the probability that `a` beats `b` with the logistic
//! function of the score gap:
//!
//! ```text
//! z      = w · (x_a - x_b)
//! P(a≻b) = sigmoid(z)
//! ```
//!
//! Training minimises the mean pairwise binary cross-entropy plus an optional
//! `L2` penalty:
//!
//! ```text
//! L(w) = (1/N) Σ_i [ softplus(z_i) - y_i · z_i ]  +  (λ/2) · ‖w‖²
//! ```
//!
//! This is exactly logistic regression on the difference vectors
//! `Δx_i = x_a - x_b`, so `L` is **convex** in `w`. Full-batch gradient descent
//! with a sufficiently small learning rate therefore decreases the loss
//! monotonically — the property the convergence tests check. The gradient is
//!
//! ```text
//! ∇L(w) = (1/N) Σ_i (sigmoid(z_i) - y_i) · Δx_i  +  λ · w
//! ```
//!
//! Everything is deterministic: the same training set and configuration always
//! produce byte-identical weights. Any weight initialisation randomness is a
//! `splitmix64` stream seeded from [`LtrConfig::seed`].

use super::types::{LtrConfig, LtrModel, LtrResult, LtrTrainingPair, LtrTrainingSet};

// ── numerically stable primitives ────────────────────────────────────────────

/// Numerically stable logistic sigmoid, `1 / (1 + e^{-x})`.
///
/// The two-branch form keeps the exponent argument non-positive, so neither
/// large positive nor large negative `x` overflows: the result is always a
/// finite value in `[0.0, 1.0]`.
#[must_use]
pub(crate) fn sigmoid(x: f64) -> f64 {
    if x >= 0.0 {
        let e = (-x).exp();
        1.0 / (1.0 + e)
    } else {
        let e = x.exp();
        e / (1.0 + e)
    }
}

/// Numerically stable softplus, `ln(1 + e^x)`.
///
/// Uses the identity `softplus(x) = max(x, 0) + ln(1 + e^{-|x|})`, which keeps
/// the exponent non-positive and so never overflows.
#[must_use]
pub(crate) fn softplus(x: f64) -> f64 {
    x.max(0.0) + (-x.abs()).exp().ln_1p()
}

/// One step of the `splitmix64` generator: maps a 64-bit state to a well-mixed
/// 64-bit output.
#[inline]
fn splitmix64(state: u64) -> u64 {
    let mut z = state.wrapping_add(0x9E37_79B9_7F4A_7C15);
    z = (z ^ (z >> 30)).wrapping_mul(0xBF58_476D_1CE4_E5B9);
    z = (z ^ (z >> 27)).wrapping_mul(0x94D0_49BB_1331_11EB);
    z ^ (z >> 31)
}

/// Deterministically initialise `dim` weights.
///
/// With `scale == 0.0` every weight starts at zero (a valid initialisation for
/// this convex objective). Otherwise weight `i` is drawn uniformly from
/// `(-scale, scale)` using a `splitmix64` stream seeded from `seed` and `i`, so
/// the initialisation is reproducible for a fixed seed.
fn init_weights(dim: usize, seed: u64, scale: f64) -> Vec<f64> {
    if scale == 0.0 {
        return vec![0.0; dim];
    }
    (0..dim)
        .map(|i| {
            let bits = splitmix64(seed.wrapping_add(splitmix64(i as u64)));
            // Top 53 bits -> uniform f64 in [0, 1).
            #[allow(clippy::cast_precision_loss)]
            let unit = (bits >> 11) as f64 / ((1u64 << 53) as f64);
            (unit * 2.0 - 1.0) * scale
        })
        .collect()
}

// ── loss and gradient ────────────────────────────────────────────────────────

/// Compute the mean pairwise cross-entropy loss and its gradient at `weights`,
/// including the `L2` penalty controlled by `l2`.
fn loss_and_gradient(weights: &[f64], pairs: &[LtrTrainingPair], l2: f64) -> (f64, Vec<f64>) {
    let dim = weights.len();
    let mut gradient = vec![0.0f64; dim];
    let mut loss = 0.0f64;
    #[allow(clippy::cast_precision_loss)]
    let n = pairs.len() as f64;

    for pair in pairs {
        let diff = pair.difference();
        // z = w · Δx over the common prefix.
        let z: f64 = weights.iter().zip(diff.iter()).map(|(w, d)| w * d).sum();
        // Stable BCE: softplus(z) - y·z == y·softplus(-z) + (1-y)·softplus(z).
        loss += softplus(z) - pair.label * z;
        let residual = sigmoid(z) - pair.label;
        for (g, d) in gradient.iter_mut().zip(diff.iter()) {
            *g += residual * d;
        }
    }

    if n > 0.0 {
        loss /= n;
        for g in &mut gradient {
            *g /= n;
        }
    }

    if l2 > 0.0 {
        let mut penalty = 0.0f64;
        for (g, w) in gradient.iter_mut().zip(weights.iter()) {
            *g += l2 * w;
            penalty += w * w;
        }
        loss += 0.5 * l2 * penalty;
    }

    (loss, gradient)
}

// ── the training loop ────────────────────────────────────────────────────────

/// Fit an [`LtrModel`] to `training_set` under `config` via full-batch gradient
/// descent on the convex pairwise logistic objective.
///
/// The loop records the loss at the *current* weights at the start of each
/// epoch, then takes a gradient step. It stops early — with
/// [`LtrModel::converged`] set to `true` — once the absolute change in
/// consecutive epoch losses falls below [`LtrConfig::tolerance`], and otherwise
/// runs the full [`LtrConfig::epochs`] budget.
///
/// # Errors
///
/// Returns an [`super::types::LtrError`] when `config` is invalid (see
/// [`LtrConfig::validate`]) or when `training_set` is empty or contains a pair
/// whose vectors do not match [`LtrConfig::feature_dim`] or whose label is out
/// of range (see [`LtrTrainingSet::validate`]).
pub(crate) fn train_model(
    training_set: &LtrTrainingSet,
    config: &LtrConfig,
) -> LtrResult<LtrModel> {
    config.validate()?;
    training_set.validate(config.feature_dim)?;

    let mut weights = init_weights(config.feature_dim, config.seed, config.weight_init_scale);
    let mut loss_history: Vec<f64> = Vec::with_capacity(config.epochs);
    let mut previous_loss = f64::INFINITY;
    let mut converged = false;
    let mut epochs_run = 0usize;

    for epoch in 0..config.epochs {
        let (loss, gradient) =
            loss_and_gradient(&weights, &training_set.pairs, config.l2_regularization);
        loss_history.push(loss);
        epochs_run = epoch + 1;

        if (previous_loss - loss).abs() < config.tolerance {
            converged = true;
            break;
        }
        previous_loss = loss;

        for (w, g) in weights.iter_mut().zip(gradient.iter()) {
            *w -= config.learning_rate * g;
        }
    }

    // The loss at the final weights (equals the last recorded loss when the run
    // stopped early before an update).
    let final_loss = loss_and_gradient(&weights, &training_set.pairs, config.l2_regularization).0;

    Ok(LtrModel {
        weights,
        feature_dim: config.feature_dim,
        epochs_run,
        final_loss,
        converged,
        loss_history,
    })
}
