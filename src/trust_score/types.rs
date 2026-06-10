//! Types for the `trust_score` module.

use thiserror::Error;

// ── TrustComponents ───────────────────────────────────────────────────────────

/// Weighted components that combine into an overall trust score.
///
/// By default each component is set to `0.25` so that they sum to `1.0`.
#[derive(Debug, Clone, Copy)]
pub struct TrustComponents {
    /// Weight/score for answer grounding (from hallucination detection).
    pub grounding: f32,
    /// Weight/score for cross-claim consistency.
    pub consistency: f32,
    /// Weight/score for source coverage quality.
    pub source_quality: f32,
    /// Weight/score for query-answer completeness.
    pub completeness: f32,
}

impl Default for TrustComponents {
    fn default() -> Self {
        Self {
            grounding: 0.25,
            consistency: 0.25,
            source_quality: 0.25,
            completeness: 0.25,
        }
    }
}

impl TrustComponents {
    /// Sum all four components.
    #[must_use]
    pub fn sum(&self) -> f32 {
        self.grounding + self.consistency + self.source_quality + self.completeness
    }

    /// Return `true` when the four weights sum to approximately `1.0` (±0.01).
    #[must_use]
    pub fn is_valid(&self) -> bool {
        (self.sum() - 1.0).abs() < 0.01
    }

    /// Return a copy normalised so that all four values sum to `1.0`.
    ///
    /// Falls back to the uniform default when the sum is zero.
    #[must_use]
    pub fn normalize(&self) -> Self {
        let s = self.sum();
        if s.abs() < f32::EPSILON {
            return Self::default();
        }
        Self {
            grounding: self.grounding / s,
            consistency: self.consistency / s,
            source_quality: self.source_quality / s,
            completeness: self.completeness / s,
        }
    }

    /// Builder: set the `grounding` component.
    #[must_use]
    pub fn with_grounding(mut self, grounding: f32) -> Self {
        self.grounding = grounding;
        self
    }

    /// Builder: set the `consistency` component.
    #[must_use]
    pub fn with_consistency(mut self, consistency: f32) -> Self {
        self.consistency = consistency;
        self
    }

    /// Builder: set the `source_quality` component.
    #[must_use]
    pub fn with_source_quality(mut self, source_quality: f32) -> Self {
        self.source_quality = source_quality;
        self
    }

    /// Builder: set the `completeness` component.
    #[must_use]
    pub fn with_completeness(mut self, completeness: f32) -> Self {
        self.completeness = completeness;
        self
    }
}

// ── TrustScore ────────────────────────────────────────────────────────────────

/// The computed trust score for a RAG-generated answer.
#[derive(Debug, Clone, Copy)]
pub struct TrustScore {
    /// Overall trust score in `[0.0, 1.0]`.
    pub overall: f32,
    /// Confidence in the overall score (proxy: minimum of component scores).
    pub confidence: f32,
    /// The raw per-component scores used to derive `overall`.
    pub components: TrustComponents,
}

impl TrustScore {
    /// Construct a `TrustScore` from its parts.
    #[must_use]
    pub fn new(overall: f32, confidence: f32, components: TrustComponents) -> Self {
        Self {
            overall,
            confidence,
            components,
        }
    }

    /// Return `true` when the overall score meets or exceeds `threshold`.
    #[must_use]
    pub fn is_trustworthy(&self, threshold: f32) -> bool {
        self.overall >= threshold
    }

    /// Human-readable trust label.
    ///
    /// Returns `"high"` (≥0.7), `"medium"` (≥0.4), or `"low"` (<0.4).
    #[must_use]
    pub fn label(&self) -> &str {
        if self.overall >= 0.7 {
            "high"
        } else if self.overall >= 0.4 {
            "medium"
        } else {
            "low"
        }
    }
}

// ── TrustConfig ───────────────────────────────────────────────────────────────

/// Configuration for `TrustScorer`.
#[derive(Debug, Clone)]
pub struct TrustConfig {
    /// Per-component weights (will be normalised before use).
    pub weights: TrustComponents,
    /// Minimum `overall` score considered trustworthy. Default: `0.6`.
    pub min_trustworthy: f32,
}

impl Default for TrustConfig {
    fn default() -> Self {
        Self {
            weights: TrustComponents::default(),
            min_trustworthy: 0.6,
        }
    }
}

impl TrustConfig {
    /// Create a new `TrustConfig`.
    #[must_use]
    pub fn new(weights: TrustComponents, min_trustworthy: f32) -> Self {
        Self {
            weights,
            min_trustworthy,
        }
    }

    /// Set the component weights.
    #[must_use]
    pub fn with_weights(mut self, weights: TrustComponents) -> Self {
        self.weights = weights;
        self
    }

    /// Set the minimum trustworthy threshold.
    #[must_use]
    pub fn with_min_trustworthy(mut self, min_trustworthy: f32) -> Self {
        self.min_trustworthy = min_trustworthy;
        self
    }
}

// ── TrustScorer ───────────────────────────────────────────────────────────────

/// Re-exported from `scorer` — see [`crate::trust_score::scorer`].
pub use crate::trust_score::scorer::TrustScorer;

// ── TrustError ────────────────────────────────────────────────────────────────

/// Errors produced by [`TrustScorer`].
#[derive(Debug, Error)]
pub enum TrustError {
    /// The inputs were insufficient to compute a meaningful trust score.
    #[error("Insufficient data to compute trust score")]
    InsufficientData,
}
