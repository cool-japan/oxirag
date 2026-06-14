//! The paired A/B bootstrap evaluator.

use crate::ab_eval::types::{AbConfig, AbError, AbResult, AbWinner};

/// FNV-1a offset basis for 64-bit hashing.
const FNV_OFFSET: u64 = 0xcbf2_9ce4_8422_2325;
/// FNV-1a prime for 64-bit hashing.
const FNV_PRIME: u64 = 0x0000_0100_0000_01b3;

/// Mix one byte into a running FNV-1a hash state.
const fn fnv1a_byte(state: u64, byte: u8) -> u64 {
    (state ^ byte as u64).wrapping_mul(FNV_PRIME)
}

/// Hash the eight bytes of a `u64` (little-endian) into a running FNV-1a state.
const fn fnv1a_u64(mut state: u64, value: u64) -> u64 {
    let bytes = value.to_le_bytes();
    let mut i = 0;
    while i < 8 {
        state = fnv1a_byte(state, bytes[i]);
        i += 1;
    }
    state
}

// ── AbEvaluator ───────────────────────────────────────────────────────────────

/// Evaluator that compares two RAG systems from their paired per-query scores.
///
/// Given two equal-length score lists (the systems' scores on the *same* queries
/// in the same order), [`compare`](Self::compare) reports the per-system means,
/// the mean paired difference, win/loss/tie counts, and a deterministic paired
/// bootstrap confidence interval and two-sided p-value for the mean difference.
///
/// The bootstrap uses **no randomness**: resample indices are produced by an
/// [FNV-1a](https://en.wikipedia.org/wiki/Fowler%E2%80%93Noll%E2%80%93Vo_hash_function)
/// hash of the `(sample, position)` pair, so the whole [`AbResult`] is a pure
/// function of the inputs and the [`AbConfig`].
#[derive(Debug, Clone, Default)]
pub struct AbEvaluator {
    /// Configuration controlling the bootstrap and tie handling.
    pub config: AbConfig,
}

impl AbEvaluator {
    /// Create a new evaluator with the given configuration.
    #[must_use]
    pub fn new(config: AbConfig) -> Self {
        Self { config }
    }

    /// Deterministic bootstrap resample index for one `(sample, position)` pair.
    ///
    /// Mixes both `sample` and `position` into an FNV-1a hash and reduces the
    /// result modulo `n`, giving a reproducible index in `0..n`. Callers
    /// guarantee `n > 0`.
    #[allow(clippy::unused_self, clippy::cast_possible_truncation)]
    fn resample_index(&self, sample: usize, position: usize, n: usize) -> usize {
        let mut state = FNV_OFFSET;
        state = fnv1a_u64(state, sample as u64);
        state = fnv1a_u64(state, position as u64);
        (state % n as u64) as usize
    }

    /// Compare two systems from their paired per-query scores.
    ///
    /// `scores_a` and `scores_b` must be the two systems' scores on the same
    /// queries, in the same order. The paired differences `diffs[i] = a[i] - b[i]`
    /// drive every statistic:
    ///
    /// * means and `mean_diff` are exact;
    /// * `wins_a` counts `diff > tie_margin`, `wins_b` counts `diff < -tie_margin`,
    ///   and everything else is a tie;
    /// * a paired bootstrap of `config.bootstrap_samples` resamples of the
    ///   differences yields the confidence interval (empirical quantiles at
    ///   `(1 - confidence) / 2` and `(1 + confidence) / 2`) and the two-sided
    ///   p-value `2 * min(fraction <= 0, fraction >= 0)` clamped to `[0, 1]`;
    /// * `winner` is `Some(A)` when `ci_low > 0`, `Some(B)` when `ci_high < 0`,
    ///   else `None`.
    ///
    /// # Errors
    ///
    /// Returns [`AbError::LengthMismatch`] if the two lists differ in length, or
    /// [`AbError::EmptyScores`] if they are empty.
    #[allow(clippy::cast_precision_loss)]
    pub fn compare(&self, scores_a: &[f32], scores_b: &[f32]) -> Result<AbResult, AbError> {
        if scores_a.len() != scores_b.len() {
            return Err(AbError::LengthMismatch {
                a: scores_a.len(),
                b: scores_b.len(),
            });
        }
        if scores_a.is_empty() {
            return Err(AbError::EmptyScores);
        }

        let n = scores_a.len();
        let nf = n as f32;

        // Paired per-query differences and the win/loss/tie tallies.
        let diffs: Vec<f32> = scores_a
            .iter()
            .zip(scores_b.iter())
            .map(|(&a, &b)| a - b)
            .collect();

        let margin = self.config.tie_margin;
        let mut wins_a = 0usize;
        let mut wins_b = 0usize;
        for &d in &diffs {
            if d > margin {
                wins_a += 1;
            } else if d < -margin {
                wins_b += 1;
            }
        }
        let ties = n - wins_a - wins_b;

        let sum_a: f32 = scores_a.iter().copied().sum();
        let sum_b: f32 = scores_b.iter().copied().sum();
        let mean_a = sum_a / nf;
        let mean_b = sum_b / nf;
        let sum_diff: f32 = diffs.iter().copied().sum();
        let mean_diff = sum_diff / nf;

        // Deterministic paired bootstrap: resample the differences by FNV indices.
        let samples = self.config.bootstrap_samples;
        let mut resample_means: Vec<f32> = Vec::with_capacity(samples);
        for s in 0..samples {
            let mut acc = 0.0f32;
            for i in 0..n {
                let idx = self.resample_index(s, i, n);
                acc += diffs[idx];
            }
            resample_means.push(acc / nf);
        }

        let (ci_low, ci_high, p_value) = if resample_means.is_empty() {
            // No bootstrap requested: fall back to the point estimate with no
            // interval width and a neutral p-value.
            (mean_diff, mean_diff, 1.0)
        } else {
            // Two-sided p-value from the sign balance of the resample means.
            let le_zero = resample_means.iter().filter(|&&m| m <= 0.0).count() as f32;
            let ge_zero = resample_means.iter().filter(|&&m| m >= 0.0).count() as f32;
            let total = resample_means.len() as f32;
            let p = (2.0 * (le_zero.min(ge_zero) / total)).clamp(0.0, 1.0);

            // Empirical confidence-interval quantiles over the sorted means.
            let mut sorted = resample_means.clone();
            sorted.sort_by(|a, b| a.partial_cmp(b).unwrap_or(std::cmp::Ordering::Equal));
            let conf = self.config.confidence;
            let low_q = (1.0 - conf) / 2.0;
            let high_q = f32::midpoint(1.0, conf);
            let lo = empirical_quantile(&sorted, low_q);
            let hi = empirical_quantile(&sorted, high_q);
            (lo, hi, p)
        };

        let winner = if ci_low > 0.0 {
            Some(AbWinner::A)
        } else if ci_high < 0.0 {
            Some(AbWinner::B)
        } else {
            None
        };

        Ok(AbResult {
            mean_a,
            mean_b,
            mean_diff,
            wins_a,
            wins_b,
            ties,
            ci_low,
            ci_high,
            p_value,
            winner,
        })
    }
}

/// Empirical quantile of an already-sorted, non-empty slice via
/// nearest-rank indexing.
///
/// `q` is clamped to `[0.0, 1.0]`; the index is `round(q * (len - 1))`.
#[allow(
    clippy::cast_precision_loss,
    clippy::cast_possible_truncation,
    clippy::cast_sign_loss
)]
fn empirical_quantile(sorted: &[f32], q: f32) -> f32 {
    let len = sorted.len();
    if len == 1 {
        return sorted[0];
    }
    let q = q.clamp(0.0, 1.0);
    let pos = q * (len - 1) as f32;
    let idx = pos.round() as usize;
    sorted[idx.min(len - 1)]
}
