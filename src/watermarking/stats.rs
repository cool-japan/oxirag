//! Detection statistics: the z-test for green-token over-representation, and
//! a hand-rolled normal CDF (via an `erf` rational approximation) to convert
//! a z-score into a p-value.
//!
//! No dependency is used for either — this crate's convention is to
//! hand-roll small numerical primitives (see e.g. `ab_eval`'s FNV-1a bootstrap)
//! rather than pull in a statistics crate for a handful of closed-form
//! formulas.

/// Abramowitz & Stegun formula 7.1.26: a rational approximation of the
/// error function `erf(x) = (2/sqrt(pi)) * integral(0..x, exp(-t^2), dt)`,
/// accurate to within `1.5e-7` (absolute error) for all real `x`.
///
/// This is the standard closed-form approximation used when a full
/// `libm`-quality `erf` is unavailable; `1.5e-7` is far tighter than any
/// tolerance a z-score/p-value decision needs (z-scores here are computed
/// from token counts, which cannot resolve differences anywhere near that
/// fine).
#[must_use]
pub fn erf(x: f64) -> f64 {
    // Constants for Abramowitz & Stegun 7.1.26.
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

/// The standard normal cumulative distribution function
/// `Phi(x) = P(Z <= x)` for `Z ~ N(0, 1)`, computed from [`erf`] via
/// `Phi(x) = 0.5 * (1 + erf(x / sqrt(2)))`.
///
/// Inherits [`erf`]'s `~1.5e-7` absolute-error bound.
#[must_use]
pub fn normal_cdf(x: f64) -> f64 {
    0.5 * (1.0 + erf(x / std::f64::consts::SQRT_2))
}

/// Compute the detection z-score for `green_count` green tokens out of
/// `total_scored` scored tokens, under the null hypothesis that each scored
/// token is independently green with probability `gamma`:
///
/// ```text
/// z = (green_count - gamma * total_scored) / sqrt(total_scored * gamma * (1 - gamma))
/// ```
///
/// # Degenerate case
///
/// When `gamma` is exactly `0.0` or `1.0`, the null distribution has zero
/// variance: every token is deterministically red (`gamma = 0`) or green
/// (`gamma = 1`) by construction, with or without a watermark, so the
/// observed `green_count` carries no information either way. This function
/// returns `0.0` in that case rather than dividing by zero. Note that
/// [`z_score_and_p_value`] does **not** derive its p-value by feeding this
/// `0.0` through [`p_value_from_z`] (that would yield the ordinary z=0
/// p-value of `0.5`, misrepresenting "no information" as "unremarkable") —
/// it detects the same degenerate condition itself and reports the maximal
/// p-value of `1.0` directly.
///
/// Callers guarantee `total_scored >= 1`; the crate-level caller
/// ([`crate::watermarking::WatermarkDetector::detect`]) already rejects a
/// token sequence too short to score anything before reaching here.
#[must_use]
pub fn z_score(green_count: usize, total_scored: usize, gamma: f64) -> f64 {
    #[allow(clippy::cast_precision_loss)]
    let t = total_scored as f64;
    #[allow(clippy::cast_precision_loss)]
    let g = green_count as f64;

    let variance = t * gamma * (1.0 - gamma);
    if variance <= 0.0 {
        return 0.0;
    }
    let expected = gamma * t;
    (g - expected) / variance.sqrt()
}

/// Convert a z-score into the one-sided upper-tail p-value `1 - Phi(z)`
/// (the probability, under the null hypothesis, of a green count at least
/// this large arising by chance), clamped to `[0.0, 1.0]` to absorb the
/// tiny floating-point overshoot [`normal_cdf`]'s `erf` approximation can
/// produce at the extremes.
#[must_use]
pub fn p_value_from_z(z: f64) -> f64 {
    (1.0 - normal_cdf(z)).clamp(0.0, 1.0)
}

/// Compute both the z-score and its p-value in one call — the pair
/// [`crate::watermarking::WatermarkDetector::detect`] actually needs.
///
/// # Degenerate case
///
/// This does **not** simply feed [`z_score`]'s output through
/// [`p_value_from_z`]: doing so would map the degenerate `z = 0.0` (see
/// [`z_score`]'s docs) through the *ordinary* z=0 p-value of `0.5` (a
/// coin-flip), which would misrepresent a case that actually carries **zero**
/// information as merely "unremarkable". Instead, when `gamma` is `0.0` or
/// `1.0` this returns `(0.0, 1.0)` directly — the maximal, least-significant
/// p-value — since the observed count could not have been anything else
/// regardless of watermarking.
#[must_use]
pub fn z_score_and_p_value(green_count: usize, total_scored: usize, gamma: f64) -> (f64, f64) {
    #[allow(clippy::cast_precision_loss)]
    let t = total_scored as f64;
    let variance = t * gamma * (1.0 - gamma);
    if variance <= 0.0 {
        return (0.0, 1.0);
    }
    let z = z_score(green_count, total_scored, gamma);
    let p = p_value_from_z(z);
    (z, p)
}
