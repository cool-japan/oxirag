#![allow(
    clippy::float_cmp,
    clippy::similar_names,
    clippy::cast_precision_loss,
    clippy::cast_possible_truncation,
    clippy::cast_sign_loss,
    clippy::cast_possible_wrap,
    clippy::many_single_char_names,
    clippy::too_many_lines,
    clippy::unreadable_literal,
    clippy::suboptimal_flops,
    clippy::needless_range_loop,
    clippy::manual_midpoint,
    clippy::doc_markdown
)]
//! Tests for group exposure fairness.
//!
//! The suite is organized so that every headline claim of the module is checked
//! against an **independent** ground truth rather than against the module's own
//! belief about itself:
//!
//! * [`binomial`] — the exact binomial `CDF` is checked against **exact rational
//!   arithmetic** (`u128` coefficients, exact powers of a simple rational `p`),
//!   and is asserted to *diverge* from the normal approximation on small `n`.
//!   That divergence test is what proves `FA*IR` was not silently built on
//!   `normal_cdf`.
//! * [`assignment`] — the Hungarian solver's optimum is checked against
//!   **brute-force** enumeration of every injection, for `n <= 8`, on seeded
//!   random cost matrices.
//! * [`fair`] — the `m-table` is checked against an `m_alpha(k)` **worked out by
//!   hand** in the test comments, and the multiple-test correction is checked to
//!   attain its nominal family-wise error rate against a Monte-Carlo estimate.
//! * [`deltr`] — the analytic gradient is checked against **central finite
//!   differences** of the loss, and the disparate-exposure penalty is ablated.
//! * [`ranker`] — the ablation: exposure disparity must strictly fall from the
//!   unfair baseline to the fair ranking, and the `nDCG` cost is reported.
//! * [`amortized`] — the amortization must actually amortize: the cross-group
//!   attention-per-relevance gap at `T = 50` must be a fraction of the gap a
//!   single fair ranking (or the greedy relevance baseline) leaves at `T = 1`.

mod amortized;
mod assignment;
mod binomial;
mod deltr;
mod exposure;
mod fair;
mod ranker;

use crate::fairness_ranking::rng::FairnessRng;

/// A reference **normal approximation** to `P(Bin(n, p) <= k)`, with the standard
/// continuity correction: `Phi((k + 0.5 - n p) / sqrt(n p (1 - p)))`.
///
/// This exists only so the binomial tests can assert their *exact* `CDF` is
/// **not** equal to it on small `n`. It is the thing the module must not have
/// silently become. `Phi` is the same `erf`-based normal `CDF` the crate ships in
/// `watermarking::stats`, reproduced here so the test does not depend on that
/// module's feature flag.
pub fn normal_approx_binomial_cdf(k: usize, n: usize, p: f64) -> f64 {
    let mean = (n as f64) * p;
    let variance = (n as f64) * p * (1.0 - p);
    if variance <= 0.0 {
        return if (k as f64) >= mean { 1.0 } else { 0.0 };
    }
    let z = ((k as f64) + 0.5 - mean) / variance.sqrt();
    0.5 * (1.0 + erf(z / std::f64::consts::SQRT_2))
}

/// Abramowitz & Stegun 7.1.26 `erf`, reproduced from `watermarking::stats` so the
/// normal-approximation reference is self-contained.
fn erf(x: f64) -> f64 {
    const A1: f64 = 0.254_829_592;
    const A2: f64 = -0.284_496_736;
    const A3: f64 = 1.421_413_741;
    const A4: f64 = -1.453_152_027;
    const A5: f64 = 1.061_405_429;
    const P: f64 = 0.327_591_1;

    let sign = if x < 0.0 { -1.0 } else { 1.0 };
    let x = x.abs();
    let t = 1.0 / P.mul_add(x, 1.0);
    let poly = ((((A5 * t + A4) * t + A3) * t + A2) * t + A1) * t;
    let y = 1.0 - poly * (-x * x).exp();
    sign * y
}

/// A seeded matrix of costs in `[low, high)`, row-major `rows x cols`.
pub fn random_matrix(
    rng: &mut FairnessRng,
    rows: usize,
    cols: usize,
    low: f64,
    high: f64,
) -> Vec<f64> {
    (0..rows * cols)
        .map(|_| rng.next_range(low, high))
        .collect()
}
