//! Types for the `temporal` module.

use serde::{Deserialize, Serialize};
use thiserror::Error;

// ── DecayFunction ─────────────────────────────────────────────────────────────

/// Temporal decay function applied to search result scores.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub enum DecayFunction {
    /// Exponential decay: `exp(-age_days * ln(2) / half_life_days)`.
    Exponential {
        /// Half-life in days — score halves every this many days.
        half_life_days: f64,
    },
    /// Linear decay: `max(0, 1 - age_days / max_age_days)`.
    Linear {
        /// Age at which the score reaches zero.
        max_age_days: f64,
    },
    /// Gaussian decay: `exp(-0.5 * (age_days / sigma_days)^2)`.
    Gaussian {
        /// Standard deviation in days.
        sigma_days: f64,
    },
    /// No decay — score is unchanged regardless of age.
    None,
}

impl DecayFunction {
    /// Compute the decay multiplier `[0.0, 1.0]` for `age_days`.
    ///
    /// Age is always clamped to `[0.0, ∞)` (future documents = age 0 → decay 1).
    #[must_use]
    pub fn apply(&self, age_days: f64) -> f64 {
        let age = age_days.max(0.0);
        match self {
            Self::Exponential { half_life_days } => {
                if *half_life_days <= 0.0 {
                    return 1.0;
                }
                (-(age * std::f64::consts::LN_2) / half_life_days).exp()
            }
            Self::Linear { max_age_days } => {
                if *max_age_days <= 0.0 {
                    return 1.0;
                }
                (1.0 - age / max_age_days).max(0.0)
            }
            Self::Gaussian { sigma_days } => {
                if *sigma_days <= 0.0 {
                    return 1.0;
                }
                let z = age / sigma_days;
                (-0.5 * z * z).exp()
            }
            Self::None => 1.0,
        }
    }

    /// Human-readable label.
    #[must_use]
    pub fn label(&self) -> &'static str {
        match self {
            Self::Exponential { .. } => "exponential",
            Self::Linear { .. } => "linear",
            Self::Gaussian { .. } => "gaussian",
            Self::None => "none",
        }
    }
}

impl Default for DecayFunction {
    fn default() -> Self {
        Self::Exponential {
            half_life_days: 30.0,
        }
    }
}

// ── TemporalScore ─────────────────────────────────────────────────────────────

/// Temporal scoring breakdown for a single result.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct TemporalScore {
    /// Original retrieval score.
    pub original: f32,
    /// Temporally-adjusted score.
    pub decayed: f32,
    /// Document age in fractional days at time of re-ranking.
    pub age_days: f64,
}

// ── TemporalConfig ────────────────────────────────────────────────────────────

/// Configuration for the temporal re-ranker.
#[derive(Debug, Clone)]
pub struct TemporalConfig {
    /// Decay function to apply.
    pub decay: DecayFunction,
    /// Blending weight `w` in `[0.0, 1.0]`.
    ///
    /// Final score = `(1 - w) * original + w * original * decay(age)`.
    /// Defaults to `0.5`.
    pub weight: f32,
    /// Whether to prefer `updated_at` over `created_at` metadata.
    ///
    /// Defaults to `true`.
    pub use_updated: bool,
}

impl Default for TemporalConfig {
    fn default() -> Self {
        Self {
            decay: DecayFunction::Exponential {
                half_life_days: 30.0,
            },
            weight: 0.5,
            use_updated: true,
        }
    }
}

impl TemporalConfig {
    /// Set the decay function.
    #[must_use]
    pub fn with_decay(mut self, v: DecayFunction) -> Self {
        self.decay = v;
        self
    }

    /// Set the blending weight.
    #[must_use]
    pub fn with_weight(mut self, v: f32) -> Self {
        self.weight = v;
        self
    }

    /// Set whether to use `updated_at`.
    #[must_use]
    pub fn with_use_updated(mut self, v: bool) -> Self {
        self.use_updated = v;
        self
    }
}

// ── TemporalError ─────────────────────────────────────────────────────────────

/// Errors from the `temporal` module.
#[derive(Debug, Error)]
pub enum TemporalError {
    /// Empty result list provided to re-ranker.
    #[error("Result list must not be empty")]
    EmptyResults,
}
