//! Faithfulness computation for attributed answers.
//!
//! [`FaithfulnessChecker`] measures the fraction of answer sentences that are
//! grounded by at least one citation above the configured threshold.

use super::types::CitedSpan;

// ── FaithfulnessChecker ───────────────────────────────────────────────────────

/// Computes grounding-based faithfulness metrics over a set of [`CitedSpan`]s.
pub struct FaithfulnessChecker;

impl FaithfulnessChecker {
    /// Construct a new [`FaithfulnessChecker`].
    #[must_use]
    pub fn new() -> Self {
        Self
    }

    /// Return `true` when `span.grounding_score >= threshold`.
    #[must_use]
    pub fn check_span(&self, span: &CitedSpan, threshold: f32) -> bool {
        span.grounding_score >= threshold
    }

    /// Compute the fraction of grounded spans over all spans.
    ///
    /// A span is "grounded" when its `grounding_score >= grounding_threshold`.
    /// Returns `0.0` when `spans` is empty (no division-by-zero risk).
    #[must_use]
    #[allow(clippy::cast_precision_loss)]
    pub fn faithfulness(&self, spans: &[CitedSpan], grounding_threshold: f32) -> f32 {
        let n = spans.len();
        if n == 0 {
            return 0.0;
        }
        let grounded = spans
            .iter()
            .filter(|s| self.check_span(s, grounding_threshold))
            .count();
        grounded as f32 / n as f32
    }

    /// Return references to all spans whose `grounding_score` is below
    /// `threshold`.
    #[must_use]
    pub fn ungrounded_spans<'a>(
        &self,
        spans: &'a [CitedSpan],
        threshold: f32,
    ) -> Vec<&'a CitedSpan> {
        spans
            .iter()
            .filter(|s| s.grounding_score < threshold)
            .collect()
    }
}

impl Default for FaithfulnessChecker {
    fn default() -> Self {
        Self::new()
    }
}
