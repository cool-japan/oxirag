//! Tests for the exact binomial statistics.
//!
//! The headline is [`exact_cdf_matches_exact_rational_arithmetic`] and its
//! companion [`exact_cdf_is_not_the_normal_approximation`]: together they prove
//! the `CDF` is the exact binomial law and *not* the normal approximation the
//! crate ships elsewhere. The independent ground truth is a `u128`
//! Pascal's-triangle computation — structurally different from the module's
//! multiplicative recurrence — combined with exact integer powers of a rational
//! `p`.

use super::normal_approx_binomial_cdf;
use crate::fairness_ranking::binomial::{
    binomial_cdf, binomial_pmf, binomial_pmf_table, binomial_quantile, exact_binomial_coefficient,
    log_binomial_pmf, log_gamma,
};

/// `C(n, k)` by **Pascal's triangle** — an independent oracle for the module's
/// multiplicative recurrence. `u128` throughout; the tests stay in the range
/// where it does not overflow.
fn pascal(n: usize, k: usize) -> u128 {
    if k > n {
        return 0;
    }
    let mut row = vec![0u128; n + 1];
    row[0] = 1;
    for i in 1..=n {
        for j in (1..=i).rev() {
            row[j] += row[j - 1];
        }
    }
    row[k]
}

/// Exact `P(Bin(n, num/den) <= k)` as a ratio of two `u128` integers, for a
/// rational success probability `num/den`. Returns the exact value rounded once
/// to `f64`.
///
/// `P(X <= k) = [ sum_{i=0}^{k} C(n, i) num^i (den - num)^(n - i) ] / den^n`.
fn exact_rational_cdf(k: usize, n: usize, num: u128, den: u128) -> f64 {
    let complement = den - num;
    let mut numerator: u128 = 0;
    for i in 0..=k.min(n) {
        let coefficient = pascal(n, i);
        let success_power = num.pow(i as u32);
        let failure_power = complement.pow((n - i) as u32);
        numerator += coefficient * success_power * failure_power;
    }
    let denominator = den.pow(n as u32);
    (numerator as f64) / (denominator as f64)
}

// ── (a) exact CDF vs exact rational arithmetic ───────────────────────────────

#[test]
fn exact_cdf_matches_exact_rational_arithmetic() {
    // p = 1/2 and p = 1/4, every n up to 20, every prefix k. The reference is
    // exact integer arithmetic; the tolerance is for the module's f64 summation
    // only, and 1e-12 is generous next to the few-ulp error a Kahan sum of
    // non-negative terms actually incurs.
    for &(num, den) in &[(1u128, 2u128), (1u128, 4u128), (3u128, 4u128)] {
        let p = (num as f64) / (den as f64);
        for n in 1..=20usize {
            for k in 0..=n {
                let exact = exact_rational_cdf(k, n, num, den);
                let computed = binomial_cdf(k, n, p);
                assert!(
                    (exact - computed).abs() < 1e-12,
                    "CDF mismatch at n={n} k={k} p={p}: exact={exact} computed={computed}"
                );
            }
        }
    }
}

#[test]
fn exact_pmf_matches_exact_rational_arithmetic() {
    // P(X = k) = C(n, k) num^k (den - num)^(n-k) / den^n, exactly.
    for &(num, den) in &[(1u128, 2u128), (1u128, 4u128)] {
        let p = (num as f64) / (den as f64);
        let complement = den - num;
        for n in 1..=20usize {
            let denominator = den.pow(n as u32);
            for k in 0..=n {
                let numerator = pascal(n, k) * num.pow(k as u32) * complement.pow((n - k) as u32);
                let exact = (numerator as f64) / (denominator as f64);
                let computed = binomial_pmf(k, n, p);
                assert!(
                    (exact - computed).abs() < 1e-13,
                    "PMF mismatch at n={n} k={k} p={p}: exact={exact} computed={computed}"
                );
            }
        }
    }
}

// ── (a, second half) the CDF is NOT the normal approximation ─────────────────

#[test]
fn exact_cdf_is_not_the_normal_approximation() {
    // This is the test that proves the module did not silently substitute
    // `normal_cdf` for the exact binomial CDF. Small n is exactly where the
    // De Moivre–Laplace approximation is worst.
    //
    // Bin(5, 0.1), P(X <= 0) = 0.9^5 = 0.59049 exactly. The continuity-corrected
    // normal approximation gives Phi(0) = 0.5. They differ by ~0.09 — far more
    // than any tolerance the exact value is claimed to meet.
    let exact = 0.9_f64.powi(5);
    let computed = binomial_cdf(0, 5, 0.1);
    let normal = normal_approx_binomial_cdf(0, 5, 0.1);

    assert!(
        (computed - exact).abs() < 1e-13,
        "exact CDF should equal 0.9^5 = {exact}, got {computed}"
    );
    assert!(
        (normal - 0.5).abs() < 1e-3,
        "the continuity-corrected normal approximation should be ~0.5, got {normal}"
    );
    assert!(
        (computed - normal).abs() > 0.05,
        "the exact CDF ({computed}) must diverge from the normal approximation ({normal}); \
         if they agree the implementation is the wrong one"
    );

    // A second small-n divergent case: Bin(3, 0.25), P(X <= 0) = 0.75^3 = 0.421875.
    // The continuity-corrected normal gives Phi((0.5 - 0.75)/0.75) = Phi(-0.333)
    // ~= 0.370 — off by more than 0.05. (Mid-distribution cases at larger n, e.g.
    // Bin(10, 0.5) at k = 3, genuinely *do* agree with the normal to ~1e-3; the
    // divergence this test relies on lives at small n and extreme p, which is
    // exactly the ranking-prefix regime FA*IR cares about.)
    let exact_tail = 0.75_f64.powi(3);
    let computed_tail = binomial_cdf(0, 3, 0.25);
    let normal_tail = normal_approx_binomial_cdf(0, 3, 0.25);
    assert!((computed_tail - exact_tail).abs() < 1e-13);
    assert!(
        (computed_tail - normal_tail).abs() > 0.03,
        "exact {computed_tail} vs normal {normal_tail} should diverge at small n"
    );
}

// ── the quantile, and its off-by-one trap ────────────────────────────────────

#[test]
fn quantile_is_the_correct_inverse_cdf() {
    // m_alpha(k) = min { m : P(Bin(k, p) <= m) >= alpha }.
    //
    // For p = 0.5, alpha = 0.1: the top-1 slot must NOT be forced protected.
    // P(Bin(1, 0.5) <= 0) = 0.5 >= 0.1, so the smallest m clearing 0.1 is 0.
    // The off-by-one form min{ m : P(X < m) >= alpha } would give 1 here and
    // wrongly force the top slot.
    assert_eq!(binomial_quantile(0.1, 1, 0.5), 0);

    // P(Bin(2, 0.5) <= m): m=0 -> 0.25, already >= 0.1, so m_alpha(2) = 0.
    assert_eq!(binomial_quantile(0.1, 2, 0.5), 0);

    // A stricter alpha forces more. P(Bin(4, 0.5) <= 0) = 1/16 = 0.0625 < 0.3;
    // P(<= 1) = 5/16 = 0.3125 >= 0.3. So m_alpha = 1.
    assert_eq!(binomial_quantile(0.3, 4, 0.5), 1);

    // The definitional consistency: for every m returned, the CDF clears q and
    // the CDF one below does not.
    for n in 1..=30usize {
        for &p in &[0.2, 0.5, 0.7] {
            for &q in &[0.05, 0.1, 0.25, 0.5, 0.9] {
                let m = binomial_quantile(q, n, p);
                assert!(binomial_cdf(m, n, p) >= q - 1e-12);
                if m > 0 {
                    assert!(binomial_cdf(m - 1, n, p) < q + 1e-12);
                }
            }
        }
    }
}

// ── the mode-anchored table is stable where the naive seed is not ────────────

#[test]
fn pmf_table_is_stable_for_large_n() {
    // For n = 2000, p = 0.5, the naive seed pmf(0) = 0.5^2000 underflows f64 to
    // exactly zero, so a table built forward from it would be identically zero.
    // The mode-anchored table must carry the real mass, summing to one and
    // peaking near the mean.
    let n = 2000;
    let p = 0.5;
    let table = binomial_pmf_table(n, p);
    let total: f64 = table.iter().sum();
    assert!(
        (total - 1.0).abs() < 1e-9,
        "large-n pmf table must still sum to 1, summed to {total}"
    );
    // The mode is at n/2 = 1000, and its mass must be positive (not underflowed).
    assert!(table[1000] > 0.0);
    // A full CDF over the whole support is 1.
    assert!((binomial_cdf(n, n, p) - 1.0).abs() < 1e-12);
    // The median is at the mean by symmetry.
    assert!(binomial_cdf(1000, n, p) >= 0.5);
    assert!(binomial_cdf(999, n, p) < 0.5);
}

#[test]
fn pmf_table_matches_per_point_pmf() {
    for n in [1usize, 5, 13, 40] {
        for &p in &[0.1, 0.37, 0.5, 0.83] {
            let table = binomial_pmf_table(n, p);
            for k in 0..=n {
                assert!(
                    (table[k] - binomial_pmf(k, n, p)).abs() < 1e-14,
                    "table[{k}] vs pmf at n={n} p={p}"
                );
            }
        }
    }
}

// ── degenerate parameters are point masses, not NaNs ─────────────────────────

#[test]
fn degenerate_parameters_are_point_masses() {
    // p = 0: all mass on 0.
    assert_eq!(binomial_pmf(0, 7, 0.0), 1.0);
    assert_eq!(binomial_pmf(3, 7, 0.0), 0.0);
    assert_eq!(binomial_cdf(0, 7, 0.0), 1.0);
    // p = 1: all mass on n.
    assert_eq!(binomial_pmf(7, 7, 1.0), 1.0);
    assert_eq!(binomial_pmf(3, 7, 1.0), 0.0);
    assert_eq!(binomial_cdf(6, 7, 1.0), 0.0);
    assert_eq!(binomial_cdf(7, 7, 1.0), 1.0);
    // The log pmf agrees.
    assert_eq!(log_binomial_pmf(0, 7, 0.0), 0.0);
    assert_eq!(log_binomial_pmf(3, 7, 0.0), f64::NEG_INFINITY);
    // Out-of-range p is a loud NaN, not a plausible number.
    assert!(binomial_pmf(3, 7, 1.5).is_nan());
    assert!(binomial_cdf(3, 7, -0.1).is_nan());
}

// ── the building blocks ──────────────────────────────────────────────────────

#[test]
fn exact_coefficient_matches_pascal() {
    for n in 0..=60usize {
        for k in 0..=n {
            assert_eq!(exact_binomial_coefficient(n, k), Some(pascal(n, k)));
        }
    }
    // Overflow is reported as None, not wrapped.
    assert_eq!(exact_binomial_coefficient(200, 100), None);
    assert_eq!(exact_binomial_coefficient(5, 9), Some(0));
}

#[test]
fn log_gamma_matches_known_factorials() {
    // ln Gamma(n+1) = ln(n!).
    let mut factorial = 1.0f64;
    for n in 1..=15u64 {
        factorial *= n as f64;
        let expected = factorial.ln();
        let computed = log_gamma((n as f64) + 1.0);
        assert!(
            (expected - computed).abs() < 1e-9,
            "ln Gamma({}) = ln({n}!) = {expected}, got {computed}",
            n + 1
        );
    }
    // ln Gamma(1/2) = ln(sqrt(pi)).
    let expected = std::f64::consts::PI.sqrt().ln();
    assert!((log_gamma(0.5) - expected).abs() < 1e-10);
    // Out of domain: loud NaN.
    assert!(log_gamma(-1.0).is_nan());
    assert!(log_gamma(0.0).is_nan());
}
