//! Types for the `autocut` module.
use thiserror::Error;
// ── AutoCutStrategy ───────────────────────────────────────────────────────────
/// Strategy used to decide how many leading results to keep.
///
/// All strategies operate on the descending-sorted score sequence and look for
/// discontinuities ("jumps") in the scores to choose a cut point dynamically,
/// rather than relying on a fixed `top_k`.
#[derive(Debug, Clone, Copy, PartialEq)]
pub enum AutoCutStrategy {
    /// Cut *after* the `k`-th significant gap.
    ///
    /// A gap between two consecutive descending scores is "significant" when it
    /// exceeds `mean_gap * sensitivity` (see [`AutoCutConfig::sensitivity`]).
    /// This mirrors Weaviate's `autocut = k` semantics.
    Jumps(usize),
    /// Keep every result whose `score >= r * top_score`.
    RelativeThreshold(f32),
    /// Cut at the first gap exceeding `mean_gap + s * stddev_gap`.
    StdDev(f32),
    /// Cut at the single largest gap (kneedle-lite).
    ///
    /// Keep everything up to and including the item just before the biggest
    /// discontinuity in the descending sequence.
    Knee,
}
impl Default for AutoCutStrategy {
    fn default() -> Self {
        Self::Jumps(1)
    }
}
impl AutoCutStrategy {
    /// Human-readable label for this strategy.
    #[must_use]
    pub fn as_str(&self) -> &str {
        match self {
            Self::Jumps(_) => "jumps",
            Self::RelativeThreshold(_) => "relative_threshold",
            Self::StdDev(_) => "std_dev",
            Self::Knee => "knee",
        }
    }
}
// ── AutoCutConfig ─────────────────────────────────────────────────────────────
/// Configuration for [`AutoCutter`](crate::autocut::cutter::AutoCutter).
#[derive(Debug, Clone, Copy)]
pub struct AutoCutConfig {
    /// Cut strategy. Defaults to [`AutoCutStrategy::Jumps`]`(1)`.
    pub strategy: AutoCutStrategy,
    /// Multiplier applied to the mean gap when detecting significant jumps.
    ///
    /// Only consulted by [`AutoCutStrategy::Jumps`]. Defaults to `1.0`.
    pub sensitivity: f32,
    /// Lower bound on the number of results to keep. Defaults to `1`.
    pub min_keep: usize,
    /// Upper bound on the number of results to keep.
    ///
    /// `0` means no upper bound. Defaults to `0`.
    pub max_keep: usize,
}
impl Default for AutoCutConfig {
    fn default() -> Self {
        Self {
            strategy: AutoCutStrategy::Jumps(1),
            sensitivity: 1.0,
            min_keep: 1,
            max_keep: 0,
        }
    }
}
impl AutoCutConfig {
    /// Create a new config with default values.
    #[must_use]
    pub fn new() -> Self {
        Self::default()
    }
    /// Set the cut strategy.
    #[must_use]
    pub fn with_strategy(mut self, v: AutoCutStrategy) -> Self {
        self.strategy = v;
        self
    }
    /// Set the jump-detection sensitivity multiplier.
    #[must_use]
    pub fn with_sensitivity(mut self, v: f32) -> Self {
        self.sensitivity = v;
        self
    }
    /// Set the minimum number of results to keep.
    #[must_use]
    pub fn with_min_keep(mut self, v: usize) -> Self {
        self.min_keep = v;
        self
    }
    /// Set the maximum number of results to keep (`0` = no upper bound).
    #[must_use]
    pub fn with_max_keep(mut self, v: usize) -> Self {
        self.max_keep = v;
        self
    }
    /// Validate the configuration.
    ///
    /// # Errors
    ///
    /// Returns [`AutoCutError::InvalidKeepRange`] when `max_keep` is non-zero and
    /// `min_keep` exceeds it.
    pub fn validate(&self) -> Result<(), AutoCutError> {
        if self.max_keep > 0 && self.min_keep > self.max_keep {
            return Err(AutoCutError::InvalidKeepRange {
                min: self.min_keep,
                max: self.max_keep,
            });
        }
        Ok(())
    }
}
// ── AutoCutReport ─────────────────────────────────────────────────────────────
/// Describes the truncation that was applied.
#[derive(Debug, Clone, Copy)]
pub struct AutoCutReport {
    /// Number of results kept after the cut.
    pub kept: usize,
    /// Total number of results before the cut.
    pub total: usize,
    /// Score of the first *dropped* result, if any were dropped.
    pub cut_score: Option<f32>,
    /// Largest gap observed in the descending score sequence.
    pub largest_gap: f32,
}
// ── AutoCutError ──────────────────────────────────────────────────────────────
/// Errors from the `autocut` module.
#[derive(Debug, Error)]
pub enum AutoCutError {
    /// `min_keep` exceeds a positive `max_keep`.
    #[error("min_keep ({min}) exceeds max_keep ({max})")]
    InvalidKeepRange {
        /// The configured minimum keep count.
        min: usize,
        /// The configured maximum keep count.
        max: usize,
    },
}
