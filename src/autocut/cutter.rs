//! The [`AutoCutter`] — relevance-gap result truncation.
use crate::autocut::types::{AutoCutConfig, AutoCutError, AutoCutReport, AutoCutStrategy};
use crate::types::SearchResult;
// ── AutoCutter ────────────────────────────────────────────────────────────────
/// Dynamically truncates a descending result set at relevance discontinuities.
///
/// Instead of returning a fixed `top_k`, the cutter inspects the gaps between
/// consecutive descending scores and decides where the "interesting" results
/// stop. See [`AutoCutStrategy`] for the available heuristics.
#[derive(Debug, Clone)]
pub struct AutoCutter {
    /// Configuration controlling the cut behaviour.
    pub config: AutoCutConfig,
}
impl AutoCutter {
    /// Create a new cutter with the given configuration.
    #[must_use]
    pub fn new(config: AutoCutConfig) -> Self {
        Self { config }
    }
    /// Create a new cutter after validating the configuration.
    ///
    /// # Errors
    ///
    /// Returns [`AutoCutError::InvalidKeepRange`] when the configured keep range
    /// is invalid (see [`AutoCutConfig::validate`]).
    pub fn try_new(config: AutoCutConfig) -> Result<Self, AutoCutError> {
        config.validate()?;
        Ok(Self { config })
    }
    /// Compute how many leading results to keep from a score slice.
    ///
    /// The input is copied and sorted descending before analysis, so the caller
    /// need not pre-sort. The returned count is always clamped to
    /// `[min_keep, max_keep]` (with the floor never exceeding the available
    /// count, and `max_keep == 0` meaning no upper bound).
    #[must_use]
    pub fn keep_count(&self, scores: &[f32]) -> usize {
        let n = scores.len();
        if n == 0 {
            return 0;
        }
        let sorted = sorted_desc(scores);
        let raw = self.raw_keep(&sorted);
        self.clamp(raw, n)
    }
    /// Truncate `results` at the dynamically chosen cut point.
    ///
    /// Results are sorted descending by score, the leading `keep_count` are
    /// retained, and the kept items are re-ranked `0..`.
    #[must_use]
    pub fn cut(&self, results: &[SearchResult]) -> Vec<SearchResult> {
        let (kept, _) = self.cut_with_report(results);
        kept
    }
    /// Truncate `results` and also return an [`AutoCutReport`].
    ///
    /// The report captures the kept/total counts, the score of the first dropped
    /// result (if any), and the largest gap observed.
    #[must_use]
    pub fn cut_with_report(&self, results: &[SearchResult]) -> (Vec<SearchResult>, AutoCutReport) {
        let total = results.len();
        if total == 0 {
            return (
                Vec::new(),
                AutoCutReport {
                    kept: 0,
                    total: 0,
                    cut_score: None,
                    largest_gap: 0.0,
                },
            );
        }
        let mut sorted = results.to_vec();
        sorted.sort_by(|a, b| {
            b.score
                .partial_cmp(&a.score)
                .unwrap_or(std::cmp::Ordering::Equal)
        });
        let scores: Vec<f32> = sorted.iter().map(|r| r.score).collect();
        let raw = self.raw_keep(&scores);
        let kept_n = self.clamp(raw, total);
        let largest_gap = largest_gap(&scores);
        let cut_score = sorted.get(kept_n).map(|r| r.score);
        sorted.truncate(kept_n);
        for (i, r) in sorted.iter_mut().enumerate() {
            r.rank = i;
        }
        (
            sorted,
            AutoCutReport {
                kept: kept_n,
                total,
                cut_score,
                largest_gap,
            },
        )
    }
    /// Apply the configured strategy to a descending score slice, returning the
    /// pre-clamp keep count.
    fn raw_keep(&self, sorted: &[f32]) -> usize {
        let n = sorted.len();
        if n <= 1 {
            return n;
        }
        match self.config.strategy {
            AutoCutStrategy::Jumps(k) => keep_jumps(sorted, self.config.sensitivity, k),
            AutoCutStrategy::RelativeThreshold(r) => keep_relative(sorted, r),
            AutoCutStrategy::StdDev(s) => keep_std_dev(sorted, s),
            AutoCutStrategy::Knee => keep_knee(sorted),
        }
    }
    /// Clamp `raw` into the configured `[min_keep, max_keep]` window, bounded by
    /// the available count `n`.
    fn clamp(&self, raw: usize, n: usize) -> usize {
        let mut keep = raw;
        let floor = self.config.min_keep.min(n);
        if keep < floor {
            keep = floor;
        }
        if self.config.max_keep > 0 && keep > self.config.max_keep {
            keep = self.config.max_keep;
        }
        keep.min(n)
    }
}
impl Default for AutoCutter {
    fn default() -> Self {
        Self::new(AutoCutConfig::default())
    }
}
// ── free helpers ──────────────────────────────────────────────────────────────
/// Return a copy of `scores` sorted descending.
fn sorted_desc(scores: &[f32]) -> Vec<f32> {
    let mut v = scores.to_vec();
    v.sort_by(|a, b| b.partial_cmp(a).unwrap_or(std::cmp::Ordering::Equal));
    v
}
/// Gaps between consecutive descending scores (always `>= 0.0`).
fn gaps(sorted: &[f32]) -> Vec<f32> {
    sorted.windows(2).map(|w| (w[0] - w[1]).max(0.0)).collect()
}
/// Largest gap in the descending sequence (`0.0` for fewer than two items).
fn largest_gap(sorted: &[f32]) -> f32 {
    gaps(sorted).into_iter().fold(0.0_f32, f32::max)
}
/// Arithmetic mean of `values` (`0.0` for an empty slice).
#[allow(clippy::cast_precision_loss)]
fn mean(values: &[f32]) -> f32 {
    if values.is_empty() {
        return 0.0;
    }
    values.iter().sum::<f32>() / values.len() as f32
}
/// Cut *after* the `k`-th gap exceeding `mean_gap * sensitivity`.
fn keep_jumps(sorted: &[f32], sensitivity: f32, k: usize) -> usize {
    let n = sorted.len();
    if k == 0 {
        return n;
    }
    let g = gaps(sorted);
    let threshold = mean(&g) * sensitivity;
    let mut seen = 0usize;
    for (i, &gap) in g.iter().enumerate() {
        if gap > threshold {
            seen += 1;
            if seen == k {
                // Keep everything up to and including index `i` (gap `i` is
                // between items `i` and `i + 1`).
                return i + 1;
            }
        }
    }
    n
}
/// Keep every score `>= r * top_score`.
fn keep_relative(sorted: &[f32], r: f32) -> usize {
    let top = sorted[0];
    let cutoff = r * top;
    sorted.iter().take_while(|&&s| s >= cutoff).count()
}
/// Cut at the first gap exceeding `mean_gap + s * stddev_gap`.
#[allow(clippy::cast_precision_loss)]
fn keep_std_dev(sorted: &[f32], s: f32) -> usize {
    let n = sorted.len();
    let g = gaps(sorted);
    if g.is_empty() {
        return n;
    }
    let m = mean(&g);
    let variance = g.iter().map(|&x| (x - m) * (x - m)).sum::<f32>() / g.len() as f32;
    let std = variance.max(0.0).sqrt();
    let threshold = m + s * std;
    for (i, &gap) in g.iter().enumerate() {
        if gap > threshold {
            return i + 1;
        }
    }
    n
}
/// Cut at the single largest gap (keep up to and including the item before it).
fn keep_knee(sorted: &[f32]) -> usize {
    let n = sorted.len();
    let g = gaps(sorted);
    if g.is_empty() {
        return n;
    }
    let mut best_idx = 0usize;
    let mut best_gap = g[0];
    for (i, &gap) in g.iter().enumerate() {
        if gap > best_gap {
            best_gap = gap;
            best_idx = i;
        }
    }
    if best_gap <= 0.0 {
        // No discontinuity at all (e.g. all-equal scores) — keep everything.
        return n;
    }
    best_idx + 1
}
