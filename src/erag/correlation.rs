//! Pure-Rust rank correlation: Kendall's tau-b and Spearman's rho.
//!
//! Both are implemented from scratch (no external crate) — perfectly
//! reasonable at the small case-batch sizes
//! [`ERagEvaluator::evaluate_batch`](super::evaluator::ERagEvaluator::evaluate_batch)
//! deals with. Kendall's tau-b uses the tie-corrected `O(n^2)` pairwise
//! formulation; Spearman's rho ranks each sequence (with tie-averaged
//! fractional ranks) and takes the Pearson correlation of the two rank
//! sequences.

/// Kendall's tau-b rank correlation between `xs` and `ys`.
///
/// For every pair of indices `(i, j)` with `i < j`, the pair is:
///
/// * **concordant** if `xs[i] - xs[j]` and `ys[i] - ys[j]` have the same sign;
/// * **discordant** if they have opposite (non-zero) signs;
/// * **tied** in `x`, `y`, or both, if either difference is exactly zero.
///
/// The tau-b statistic corrects for ties in either sequence:
///
/// ```text
/// tau_b = (C - D) / sqrt((n0 - n1) * (n0 - n2))
/// ```
///
/// where `C`/`D` are the concordant/discordant pair counts, `n0` is the total
/// pair count, `n1` is the number of pairs tied in `x` (regardless of `y`),
/// and `n2` is the number of pairs tied in `y` (regardless of `x`).
///
/// Returns `0.0` if `xs` and `ys` differ in length, if either has fewer than
/// two elements, or if the tau-b denominator is zero (e.g. every value in `x`
/// or every value in `y` is identical, so no order exists to correlate).
#[allow(
    clippy::float_cmp,
    clippy::cast_precision_loss,
    clippy::cast_possible_truncation,
    clippy::similar_names
)]
#[must_use]
pub fn kendall_tau(xs: &[f32], ys: &[f32]) -> f32 {
    let n = xs.len();
    if n < 2 || ys.len() != n {
        return 0.0;
    }

    let mut concordant: i64 = 0;
    let mut discordant: i64 = 0;
    let mut tied_x_only: i64 = 0;
    let mut tied_y_only: i64 = 0;
    let mut tied_both: i64 = 0;

    for i in 0..n {
        for j in (i + 1)..n {
            // Exact equality is intentional here: these are ranks/scores from
            // a deterministic pipeline, and a "tie" must mean bit-identical
            // values, not merely close ones.
            let dx = xs[i] - xs[j];
            let dy = ys[i] - ys[j];
            let x_tied = dx == 0.0;
            let y_tied = dy == 0.0;
            match (x_tied, y_tied) {
                (true, true) => tied_both += 1,
                (true, false) => tied_x_only += 1,
                (false, true) => tied_y_only += 1,
                (false, false) => {
                    if (dx > 0.0) == (dy > 0.0) {
                        concordant += 1;
                    } else {
                        discordant += 1;
                    }
                }
            }
        }
    }

    let n1 = tied_x_only + tied_both;
    let n2 = tied_y_only + tied_both;
    let n0 = concordant + discordant + tied_x_only + tied_y_only + tied_both;

    let denom = (((n0 - n1) * (n0 - n2)) as f64).sqrt();
    if denom <= 0.0 {
        0.0
    } else {
        ((concordant - discordant) as f64 / denom) as f32
    }
}

/// Spearman's rho rank correlation between `xs` and `ys`.
///
/// Computed as the Pearson correlation of the two sequences' fractional ranks
/// (ties receive the average of the ranks they span, the standard
/// tie-handling convention for Spearman's rho).
///
/// Returns `0.0` if `xs` and `ys` differ in length, if either has fewer than
/// two elements, or if either sequence's rank vector has zero variance (every
/// value identical).
#[must_use]
pub fn spearman_rho(xs: &[f32], ys: &[f32]) -> f32 {
    if xs.len() < 2 || ys.len() != xs.len() {
        return 0.0;
    }
    let rank_x = fractional_ranks(xs);
    let rank_y = fractional_ranks(ys);
    pearson(&rank_x, &rank_y)
}

/// Fractional (tie-averaged) ranks of `values`, 1-based.
///
/// Values are sorted ascending; a run of exactly-equal values is assigned the
/// average of the 1-based positions the run spans (the standard convention
/// for Spearman's rho).
#[allow(clippy::float_cmp)]
fn fractional_ranks(values: &[f32]) -> Vec<f32> {
    let n = values.len();
    let mut order: Vec<usize> = (0..n).collect();
    order.sort_by(|&a, &b| {
        values[a]
            .partial_cmp(&values[b])
            .unwrap_or(std::cmp::Ordering::Equal)
    });

    let mut ranks = vec![0.0_f32; n];
    let mut i = 0;
    while i < n {
        let mut j = i;
        // Exact equality is intentional: a "tie" for rank-averaging purposes
        // means bit-identical deterministic scores, not merely close ones.
        while j + 1 < n && values[order[j + 1]] == values[order[i]] {
            j += 1;
        }
        #[allow(clippy::cast_precision_loss)]
        let average_rank = (i + j) as f32 / 2.0 + 1.0;
        for slot in order.iter().take(j + 1).skip(i) {
            ranks[*slot] = average_rank;
        }
        i = j + 1;
    }
    ranks
}

/// Pearson product-moment correlation coefficient between `xs` and `ys`.
///
/// Callers guarantee `xs.len() == ys.len()`. Returns `0.0` if either sequence
/// has zero variance (the correlation is undefined, and `0.0` — "no linear
/// relationship detectable" — is the safe, finite fallback). By the
/// Cauchy-Schwarz inequality the ratio is mathematically confined to
/// `[-1.0, 1.0]`; the result is clamped to that range to absorb the tiny `f32`
/// rounding error that can otherwise push a near-perfect (dis)agreement
/// infinitesimally outside it.
#[allow(clippy::cast_precision_loss)]
fn pearson(xs: &[f32], ys: &[f32]) -> f32 {
    let n = xs.len();
    if n == 0 {
        return 0.0;
    }
    let nf = n as f32;
    let mean_x = xs.iter().sum::<f32>() / nf;
    let mean_y = ys.iter().sum::<f32>() / nf;

    let mut covariance = 0.0_f32;
    let mut variance_x = 0.0_f32;
    let mut variance_y = 0.0_f32;
    for i in 0..n {
        let dx = xs[i] - mean_x;
        let dy = ys[i] - mean_y;
        covariance += dx * dy;
        variance_x += dx * dx;
        variance_y += dy * dy;
    }

    if variance_x <= 0.0 || variance_y <= 0.0 {
        0.0
    } else {
        (covariance / (variance_x.sqrt() * variance_y.sqrt())).clamp(-1.0, 1.0)
    }
}
