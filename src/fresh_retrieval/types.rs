//! Types for the `fresh_retrieval` module.

use serde::{Deserialize, Serialize};
use thiserror::Error;

// ── TimeSensitivity ───────────────────────────────────────────────────────────

/// How quickly the correct answer to a query is expected to change over time.
///
/// Produced by [`FreshnessAnalyzer::classify`](crate::fresh_retrieval::FreshnessAnalyzer::classify)
/// from temporal markers, comparatives, and fast-changing-domain cues found in
/// the query text.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Default, Serialize, Deserialize)]
pub enum TimeSensitivity {
    /// The answer is essentially fixed (e.g. *"Who wrote Hamlet?"*).
    #[default]
    Static,
    /// The answer drifts gradually over months or years
    /// (e.g. *"What is the population of France?"*).
    SlowChanging,
    /// The answer changes rapidly — within days, hours, or minutes
    /// (e.g. *"What is the current stock price?"*).
    FastChanging,
}

impl TimeSensitivity {
    /// Human-readable lowercase label.
    #[must_use]
    pub fn label(self) -> &'static str {
        match self {
            Self::Static => "static",
            Self::SlowChanging => "slow_changing",
            Self::FastChanging => "fast_changing",
        }
    }

    /// Baseline freshness demand in `[0,1]` implied purely by this category,
    /// before per-signal reinforcement is applied.
    ///
    /// `Static` ⇒ `0.0`, `SlowChanging` ⇒ `0.4`, `FastChanging` ⇒ `0.8`.
    #[must_use]
    pub fn base_demand(self) -> f32 {
        match self {
            Self::Static => 0.0,
            Self::SlowChanging => 0.4,
            Self::FastChanging => 0.8,
        }
    }
}

// ── FreshConfig ───────────────────────────────────────────────────────────────

/// Configuration for the freshness analyzer.
///
/// Controls the freshness decay curve, the staleness cutoff, and how strongly a
/// time-sensitive query lets freshness override pure relevance during
/// [`FreshnessAnalyzer::rerank`](crate::fresh_retrieval::FreshnessAnalyzer::rerank).
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct FreshConfig {
    /// Half-life, in days, of the freshness score.
    ///
    /// A document of age `half_life_days` scores `0.5`; the score halves again
    /// every further `half_life_days`. Defaults to `180.0`.
    pub half_life_days: f32,
    /// Age, in days, beyond which a *fast-changing* answer is considered stale.
    ///
    /// Used by
    /// [`FreshnessAnalyzer::is_stale`](crate::fresh_retrieval::FreshnessAnalyzer::is_stale).
    /// Defaults to `30.0`.
    pub staleness_threshold_days: f64,
    /// Maximum weight freshness may carry during re-ranking, in `[0,1]`.
    ///
    /// The effective blend weight is
    /// `freshness_demand * freshness_weight`, so a static query (demand `0`)
    /// leaves scores unchanged. Defaults to `0.5`.
    pub freshness_weight: f32,
}

impl Default for FreshConfig {
    fn default() -> Self {
        Self {
            half_life_days: 180.0,
            staleness_threshold_days: 30.0,
            freshness_weight: 0.5,
        }
    }
}

impl FreshConfig {
    /// Create a new configuration with the default values.
    #[must_use]
    pub fn new() -> Self {
        Self::default()
    }

    /// Set the freshness half-life in days.
    #[must_use]
    pub fn with_half_life_days(mut self, v: f32) -> Self {
        self.half_life_days = v;
        self
    }

    /// Set the staleness threshold in days.
    #[must_use]
    pub fn with_staleness_threshold_days(mut self, v: f64) -> Self {
        self.staleness_threshold_days = v;
        self
    }

    /// Set the maximum freshness blend weight.
    #[must_use]
    pub fn with_freshness_weight(mut self, v: f32) -> Self {
        self.freshness_weight = v;
        self
    }
}

// ── FreshnessAssessment ───────────────────────────────────────────────────────

/// Outcome of classifying a query's time-sensitivity.
///
/// Returned by
/// [`FreshnessAnalyzer::classify`](crate::fresh_retrieval::FreshnessAnalyzer::classify).
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct FreshnessAssessment {
    /// The detected time-sensitivity category.
    pub sensitivity: TimeSensitivity,
    /// How strongly the query demands fresh information, in `[0,1]`.
    ///
    /// `0.0` means freshness is irrelevant (a static fact); `1.0` means the
    /// answer is highly volatile and the freshest source should dominate.
    pub freshness_demand: f32,
    /// The individual cues that fired during classification.
    ///
    /// Each entry names the matched signal (e.g. `"marker:latest"`,
    /// `"year:2024"`, `"comparative:newest"`, `"domain:price"`), in a stable,
    /// deterministic order. Empty when the query is fully static.
    pub signals: Vec<String>,
}

impl FreshnessAssessment {
    /// Whether any time-sensitivity cue fired.
    #[must_use]
    pub fn is_time_sensitive(&self) -> bool {
        self.sensitivity != TimeSensitivity::Static
    }
}

// ── FreshError ────────────────────────────────────────────────────────────────

/// Errors from the `fresh_retrieval` module.
#[derive(Debug, Error, PartialEq, Eq)]
pub enum FreshError {
    /// The supplied query was empty (or whitespace only).
    #[error("query must not be empty")]
    EmptyQuery,
    /// The `ages_days` slice did not align one-to-one with the results slice.
    #[error("ages length {ages} != results length {results}")]
    LengthMismatch {
        /// Number of supplied ages.
        ages: usize,
        /// Number of supplied results.
        results: usize,
    },
}
