//! **Exact** binomial statistics: the probability mass function, the cumulative
//! distribution function, and the quantile (inverse `CDF`) that the `FA*IR`
//! `m-table` is built from.
//!
//! # Why this file exists, and why it may not be replaced by a normal approximation
//!
//! `FA*IR` (Zehlike et al., `CIKM` 2017) requires, at every prefix `k` of a
//! ranking, that the number of protected candidates in the top-`k` is not
//! *significantly* below what a `Bin(k, p)` null model would produce. The
//! decision boundary is the `alpha`-quantile of that binomial law — an integer,
//! read off an exact discrete distribution.
//!
//! It is tempting to reach for the normal approximation that already lives
//! elsewhere in this crate (`watermarking::stats::normal_cdf`, built on an `erf`
//! rational approximation). **It would be wrong here, and not by a little.**
//! The prefixes that matter most in a ranking are the *short* ones — `k = 1`,
//! `2`, `3`, the top of the page — and that is exactly the regime where the
//! `De Moivre`–`Laplace` approximation is worst. For `Bin(5, 0.1)`, the exact
//! `P(X <= 0)` is `0.9^5 = 0.59049`; the uncorrected normal approximation gives
//! `0.2280` and the continuity-corrected one gives `0.5000`. A ranked-group
//! fairness test built on either would accept and reject *different rankings*
//! than `FA*IR` does. A module that claimed to implement `FA*IR` on top of a
//! normal `CDF` would be a fabrication that compiles, runs, and returns
//! plausible numbers. This module's tests assert the divergence explicitly, so
//! the substitution cannot be made silently later.
//!
//! # The three numerical paths, and why each exists
//!
//! **Exact integer coefficients.** `C(n, k)` is built with the multiplicative
//! recurrence `C(n, i+1) = C(n, i) * (n - i) / (i + 1)` in `u128`. Every
//! intermediate is an exact integer (each division is exact, because
//! `C(n, i+1)` *is* an integer), so the only error is the single correctly
//! rounded `u128 -> f64` conversion at the end: at most half an ulp. This works
//! up to the point where `C(n, n/2)` overflows `u128`, which is around
//! `n = 130` — comfortably past any ranking anyone will ever audit — and
//! [`exact_binomial_coefficient`] reports the overflow as `None` rather than
//! wrapping.
//!
//! **Log space.** Past that point [`log_binomial_coefficient`] falls back to
//! differences of [`log_gamma`] (Lanczos, `g = 7`), and the mass function is
//! evaluated as `exp(log C(n, k) + k*ln(p) + (n - k)*ln(1 - p))`. `ln(1 - p)` is
//! computed with `ln_1p` so that a small `p` does not lose its leading digits to
//! the subtraction.
//!
//! **The mode-anchored table.** [`binomial_pmf_table`] tabulates the whole mass
//! function in `O(n)` by starting at the **mode** `m* = floor((n + 1) p)` — the
//! single largest term, which is bounded below by roughly `1 / (n + 1)` and
//! therefore cannot underflow — and walking *outward* with the exact ratio
//! recurrences
//!
//! ```text
//! pmf(m + 1) = pmf(m) * ((n - m) / (m + 1)) * (p / (1 - p))
//! pmf(m - 1) = pmf(m) * (m / (n - m + 1)) * ((1 - p) / p)
//! ```
//!
//! Anchoring at the mode is what makes this stable. Both recurrences are
//! *contracting* away from the mode (the ratios are below one there), so
//! relative error accumulates additively — a few ulps per step — instead of
//! being amplified, and the tail terms simply underflow to zero, which is the
//! right answer for a term that is genuinely below `1e-308`. Starting the
//! recurrence at `pmf(0) = (1 - p)^n` instead — the obvious thing to do — is the
//! classic way to get this wrong: for `n = 2000, p = 0.5` that seed *is* zero in
//! `f64`, and the entire table comes out zero.
//!
//! [`binomial_cdf`] is then a Kahan-compensated sum of the table's first `k + 1`
//! entries. Every term is non-negative, so there is no cancellation and the
//! relative error of the sum is a few ulps regardless of `n`.

/// Lanczos parameter `g` for [`log_gamma`], paired with the nine coefficients in
/// [`LANCZOS_COEFFICIENTS`]. This `(g = 7, n = 9)` pair is the standard choice
/// and delivers about 15 significant digits over the whole positive half-line.
const LANCZOS_G: f64 = 7.0;

/// The nine Lanczos coefficients for `g = 7`.
const LANCZOS_COEFFICIENTS: [f64; 9] = [
    0.999_999_999_999_809_9,
    676.520_368_121_885_1,
    -1_259.139_216_722_402_8,
    771.323_428_777_653_1,
    -176.615_029_162_140_6,
    12.507_343_278_686_905,
    -0.138_571_095_265_720_12,
    9.984_369_578_019_572e-6,
    1.505_632_735_149_311_6e-7,
];

/// `ln(sqrt(2 * pi))`, the constant term of the Lanczos formula.
const LN_SQRT_TWO_PI: f64 = 0.918_938_533_204_672_8;

/// The natural logarithm of the gamma function, `ln(Gamma(x))`, for `x > 0`.
///
/// Uses the Lanczos approximation with `g = 7` and nine coefficients, plus the
/// reflection formula `Gamma(x) Gamma(1 - x) = pi / sin(pi x)` below `x = 0.5`
/// where the series is least accurate.
///
/// # Domain
///
/// `Gamma` has poles at the non-positive integers and takes negative values on
/// parts of the negative axis, so its real logarithm does not exist there:
/// `x <= 0` returns `NaN`, loudly, rather than a plausible-looking number. Every
/// call from inside this module passes `x >= 1`.
#[must_use]
pub fn log_gamma(x: f64) -> f64 {
    if !x.is_finite() || x <= 0.0 {
        return f64::NAN;
    }
    if x < 0.5 {
        // Reflection: ln G(x) = ln(pi / sin(pi x)) - ln G(1 - x).
        let sin_term = (std::f64::consts::PI * x).sin();
        return (std::f64::consts::PI / sin_term).ln() - log_gamma(1.0 - x);
    }

    let z = x - 1.0;
    let mut series = LANCZOS_COEFFICIENTS[0];
    for (i, coefficient) in LANCZOS_COEFFICIENTS.iter().enumerate().skip(1) {
        #[allow(clippy::cast_precision_loss)] // i <= 8.
        let denominator = z + i as f64;
        series += coefficient / denominator;
    }
    let t = z + LANCZOS_G + 0.5;
    LN_SQRT_TWO_PI + (z + 0.5) * t.ln() - t + series.ln()
}

/// The binomial coefficient `C(n, k)` as an **exact integer**, or `None` if it
/// overflows `u128`.
///
/// Built with the multiplicative recurrence
/// `C(n, i + 1) = C(n, i) * (n - i) / (i + 1)`, taking `k <- min(k, n - k)` first
/// so the loop is as short as possible and the intermediates as small as
/// possible. Each division is *exact*: `C(n, i + 1)` is an integer by
/// construction, so no remainder is ever discarded, and the result is the true
/// coefficient rather than a rounded one.
///
/// `None` (rather than a wrapped value) is returned on overflow, which happens
/// around `n = 130`; callers fall back to [`log_binomial_coefficient`].
#[must_use]
pub fn exact_binomial_coefficient(n: usize, k: usize) -> Option<u128> {
    if k > n {
        return Some(0);
    }
    let k = k.min(n - k);
    let mut result: u128 = 1;
    for i in 0..k {
        result = result.checked_mul((n - i) as u128)?;
        // Exact: after the multiply, `result` is divisible by `i + 1`, because
        // `C(n, i) * (n - i) / (i + 1)` is the integer `C(n, i + 1)`.
        result /= (i + 1) as u128;
    }
    Some(result)
}

/// `ln C(n, k)`, exactly when the coefficient fits in a `u128` and via
/// [`log_gamma`] otherwise.
///
/// Returns `f64::NEG_INFINITY` for `k > n` (the coefficient is zero).
#[must_use]
#[allow(clippy::cast_precision_loss)] // See the module docs: one correctly rounded conversion.
pub fn log_binomial_coefficient(n: usize, k: usize) -> f64 {
    if k > n {
        return f64::NEG_INFINITY;
    }
    if let Some(exact) = exact_binomial_coefficient(n, k) {
        // `exact >= 1` here, so the logarithm is finite. The `u128 -> f64`
        // conversion is correctly rounded (half an ulp), which is strictly
        // better than anything the gamma path can offer.
        return (exact as f64).ln();
    }
    log_gamma((n as f64) + 1.0) - log_gamma((k as f64) + 1.0) - log_gamma((n - k) as f64 + 1.0)
}

/// `ln P(Bin(n, p) = k)`.
///
/// Returns `f64::NEG_INFINITY` for a mass of exactly zero (`k > n`, or a
/// degenerate `p` that puts no mass on `k`) and `NaN` for a `p` outside
/// `[0, 1]`.
#[must_use]
#[allow(clippy::cast_precision_loss)] // Trial counts are ranking lengths.
pub fn log_binomial_pmf(successes: usize, trials: usize, p: f64) -> f64 {
    if !p.is_finite() || !(0.0..=1.0).contains(&p) {
        return f64::NAN;
    }
    if successes > trials {
        return f64::NEG_INFINITY;
    }
    // Degenerate parameters are point masses, and must not be routed through
    // `ln(0)`.
    if p <= 0.0 {
        return if successes == 0 {
            0.0
        } else {
            f64::NEG_INFINITY
        };
    }
    if p >= 1.0 {
        return if successes == trials {
            0.0
        } else {
            f64::NEG_INFINITY
        };
    }

    let failures = trials - successes;
    // `(-p).ln_1p()` is `ln(1 - p)` evaluated without the catastrophic
    // cancellation that `(1.0 - p).ln()` suffers for a small `p`.
    log_binomial_coefficient(trials, successes)
        + (successes as f64) * p.ln()
        + (failures as f64) * (-p).ln_1p()
}

/// `P(Bin(n, p) = k)`, the binomial probability mass function.
///
/// Evaluated as the direct product `C(n, k) * p^k * (1 - p)^(n - k)` whenever the
/// exact integer coefficient is available *and* the product neither underflows
/// nor overflows — that path costs three roundings and is the most accurate
/// available — and via `exp` of [`log_binomial_pmf`] otherwise.
///
/// Degenerate parameters are handled as the point masses they are: `p = 0` puts
/// all mass on `0`, `p = 1` puts all mass on `n`. A `p` outside `[0, 1]`, or a
/// non-finite one, yields `NaN`.
#[must_use]
#[allow(clippy::cast_precision_loss)] // Exact `u128` coefficient, one rounding.
pub fn binomial_pmf(successes: usize, trials: usize, p: f64) -> f64 {
    if !p.is_finite() || !(0.0..=1.0).contains(&p) {
        return f64::NAN;
    }
    if successes > trials {
        return 0.0;
    }
    if p <= 0.0 {
        return if successes == 0 { 1.0 } else { 0.0 };
    }
    if p >= 1.0 {
        return if successes == trials { 1.0 } else { 0.0 };
    }

    let failures = trials - successes;
    if let Some(exact) = exact_binomial_coefficient(trials, successes) {
        // The exact-coefficient path is only reachable for `trials` in the low
        // hundreds, so both exponents fit an `i32` many times over.
        if let (Ok(successes_i32), Ok(failures_i32)) =
            (i32::try_from(successes), i32::try_from(failures))
        {
            let direct = (exact as f64) * p.powi(successes_i32) * (1.0 - p).powi(failures_i32);
            // Reject the direct product only when it has *lost* information:
            // an underflow to zero or into the subnormals, or an overflow. In
            // every one of those cases the log path still has the answer.
            if direct.is_finite() && direct >= f64::MIN_POSITIVE {
                return direct;
            }
        }
    }
    log_binomial_pmf(successes, trials, p).exp()
}

/// The complete mass function of `Bin(n, p)` as a table of length `n + 1`,
/// computed in `O(n)` by the **mode-anchored** ratio recurrence.
///
/// See the [module documentation](self) for why the recurrence is seeded at the
/// mode rather than at `pmf(0)`: the mode is the largest term and cannot
/// underflow, and walking outward from it is a contracting recurrence whose
/// error accumulates additively instead of being amplified.
///
/// A `p` outside `[0, 1]` yields a table of `NaN`.
#[must_use]
#[allow(
    clippy::cast_precision_loss,
    clippy::cast_possible_truncation,
    clippy::cast_sign_loss
)]
pub fn binomial_pmf_table(trials: usize, p: f64) -> Vec<f64> {
    let mut table = vec![0.0; trials + 1];
    if !p.is_finite() || !(0.0..=1.0).contains(&p) {
        table.fill(f64::NAN);
        return table;
    }
    if p <= 0.0 {
        table[0] = 1.0;
        return table;
    }
    if p >= 1.0 {
        table[trials] = 1.0;
        return table;
    }

    // The mode of `Bin(n, p)` is `floor((n + 1) p)`, clamped into `[0, n]`.
    // `(n + 1) p` is finite and non-negative here, so the truncation is exact in
    // intent and the clamp makes it safe.
    let mode_real = ((trials as f64) + 1.0) * p;
    let mode = (mode_real.floor().max(0.0) as usize).min(trials);

    let seed = binomial_pmf(mode, trials, p);
    if !seed.is_finite() || seed <= 0.0 {
        // Unreachable for a genuine binomial — the mode carries at least about
        // `1 / (n + 1)` of the mass, so it cannot underflow — but a zero seed
        // would propagate through the recurrence and silently return an
        // all-zero table, which is exactly the failure mode this module refuses
        // to have. Fall back to the per-point evaluator instead.
        for (k, slot) in table.iter_mut().enumerate() {
            *slot = binomial_pmf(k, trials, p);
        }
        return table;
    }
    table[mode] = seed;

    let odds = p / (1.0 - p);
    // Upward: pmf(m + 1) = pmf(m) * ((n - m) / (m + 1)) * odds.
    for m in mode..trials {
        let ratio = ((trials - m) as f64) / ((m + 1) as f64) * odds;
        table[m + 1] = table[m] * ratio;
    }
    // Downward: pmf(m - 1) = pmf(m) * (m / (n - m + 1)) / odds.
    for m in (1..=mode).rev() {
        let ratio = (m as f64) / ((trials - m + 1) as f64) / odds;
        table[m - 1] = table[m] * ratio;
    }
    table
}

/// `P(Bin(n, p) <= k)`, the **exact** binomial cumulative distribution function.
///
/// A Kahan-compensated sum of the first `k + 1` entries of
/// [`binomial_pmf_table`]. Every term is non-negative, so the sum has no
/// cancellation and its relative error is a few ulps whatever `n` is; the result
/// is clamped into `[0, 1]` to absorb the last ulp of a saturated tail.
///
/// This is **not** a normal approximation, and must never become one — see the
/// [module documentation](self).
#[must_use]
pub fn binomial_cdf(successes: usize, trials: usize, p: f64) -> f64 {
    if !p.is_finite() || !(0.0..=1.0).contains(&p) {
        return f64::NAN;
    }
    if successes >= trials {
        return 1.0;
    }
    let table = binomial_pmf_table(trials, p);

    // Kahan compensated summation: `compensation` carries the low-order bits
    // that `total` cannot hold, so the sum is accurate to a few ulps rather than
    // to `O(n)` ulps.
    let mut total = 0.0f64;
    let mut compensation = 0.0f64;
    for &mass in table.iter().take(successes + 1) {
        let adjusted = mass - compensation;
        let next = total + adjusted;
        compensation = (next - total) - adjusted;
        total = next;
    }
    total.clamp(0.0, 1.0)
}

/// The binomial quantile: the **smallest** `m` in `0..=n` with
/// `P(Bin(n, p) <= m) >= q`.
///
/// This is the inverse `CDF` (the "`ppf`") of a discrete law, and it is the exact
/// quantity `FA*IR`'s `m-table` is made of:
///
/// ```text
/// m_alpha(k) = min { m : P(Bin(k, p) <= m) >= alpha }
/// ```
///
/// Note the `<=`. The equivalent strict form is
/// `m_alpha(k) = max { m : P(Bin(k, p) < m) <= alpha }` — the two agree, and both
/// give `m_alpha(1) = 0` for any `alpha <= 1 - p`, i.e. the top-1 slot is *not*
/// forced to be protected. A `min`/strict-`<` mixture
/// (`min { m : P(X < m) >= alpha }`) is off by one and would force it; see this
/// module's tests, which pin `m_alpha(1) = 0`.
///
/// `q` is clamped into `[0, 1]`. A non-finite `q` yields `trials`, since no `m`
/// can satisfy an unsatisfiable condition.
#[must_use]
pub fn binomial_quantile(q: f64, trials: usize, p: f64) -> usize {
    if !q.is_finite() {
        return trials;
    }
    let q = q.clamp(0.0, 1.0);
    if q <= 0.0 {
        // `P(X <= 0) >= 0` holds for every distribution.
        return 0;
    }
    let table = binomial_pmf_table(trials, p);

    let mut total = 0.0f64;
    let mut compensation = 0.0f64;
    for (m, &mass) in table.iter().enumerate() {
        let adjusted = mass - compensation;
        let next = total + adjusted;
        compensation = (next - total) - adjusted;
        total = next;
        if total.min(1.0) >= q {
            return m;
        }
    }
    trials
}
