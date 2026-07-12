//! [`WatermarkDetector`] — recovering the watermark signal from text alone.
//!
//! Detection needs only the token sequence and the shared
//! [`crate::watermarking::WatermarkConfig`] — never the model, never the
//! logits that produced the text. For every scoreable position it
//! recomputes the green list from that position's own predecessor tokens
//! (via [`crate::watermarking::WatermarkHasher`], the exact same
//! computation [`crate::watermarking::WatermarkGenerator`] used at
//! generation time) and checks whether the actual token that appeared there
//! is on it. Tallying green hits across the sequence and comparing the
//! count to its null-hypothesis expectation is the entire detector; see
//! [`crate::watermarking::stats`] for the z-score/p-value math itself.

use super::hasher::WatermarkHasher;
use super::stats::z_score_and_p_value;
use super::types::{
    WatermarkConfig, WatermarkDetection, WatermarkError, WatermarkResult, WatermarkTokenId,
};

/// Recovers the watermark detection statistics from a token sequence.
#[derive(Debug, Clone, PartialEq)]
pub struct WatermarkDetector {
    config: WatermarkConfig,
    hasher: WatermarkHasher,
}

impl WatermarkDetector {
    /// Build a detector for `config`.
    ///
    /// # Errors
    ///
    /// Propagates [`WatermarkConfig::validate`]'s error if `config` is
    /// invalid.
    pub fn new(config: WatermarkConfig) -> WatermarkResult<Self> {
        let hasher = WatermarkHasher::new(&config)?;
        Ok(Self { config, hasher })
    }

    /// The configuration this detector was built with.
    #[must_use]
    pub fn config(&self) -> &WatermarkConfig {
        &self.config
    }

    /// Score `tokens` for the watermark signal.
    ///
    /// For every position `i >= context_width`, recomputes the green list
    /// from `tokens[i - context_width .. i]` and checks whether
    /// `tokens[i]` is on it. The leading `context_width` tokens have no
    /// full predecessor context and are excluded from scoring entirely —
    /// this is not an approximation of the true count, it is the same
    /// exclusion the original Kirchenbauer et al. (2023) detector applies,
    /// since there is no principled green list to test those positions
    /// against.
    ///
    /// The resulting `green_count` out of `total_scored` feeds
    /// [`crate::watermarking::stats::z_score_and_p_value`] to produce the
    /// z-score and p-value, and `is_watermarked` is
    /// `z_score >= config.z_threshold`.
    ///
    /// # Errors
    ///
    /// Returns [`WatermarkError::EmptyText`] if `tokens` is empty, or
    /// [`WatermarkError::InsufficientContext`] if `tokens` has
    /// `context_width` or fewer entries (so nothing can be scored — this is
    /// distinct from [`WatermarkError::EmptyText`] so callers can tell "no
    /// text" apart from "text too short to score" honestly). Returns
    /// [`WatermarkError::TokenOutOfRange`] if any token id in `tokens` is
    /// `>= config.vocab_size`.
    pub fn detect(&self, tokens: &[WatermarkTokenId]) -> WatermarkResult<WatermarkDetection> {
        if tokens.is_empty() {
            return Err(WatermarkError::EmptyText);
        }
        let h = self.config.context_width;
        if tokens.len() <= h {
            return Err(WatermarkError::InsufficientContext {
                required: h + 1,
                actual: tokens.len(),
            });
        }

        let mut green_count = 0usize;
        let mut total_scored = 0usize;
        for i in h..tokens.len() {
            let token_id = tokens[i];
            if token_id as usize >= self.config.vocab_size {
                return Err(WatermarkError::TokenOutOfRange {
                    token_id,
                    vocab_size: self.config.vocab_size,
                });
            }
            let context = &tokens[i - h..i];
            let mask = self.hasher.green_mask(context)?;
            if mask[token_id as usize] {
                green_count += 1;
            }
            total_scored += 1;
        }

        let (z_score, p_value) = z_score_and_p_value(green_count, total_scored, self.config.gamma);
        let is_watermarked = z_score >= self.config.z_threshold;

        Ok(WatermarkDetection {
            green_count,
            total_scored,
            z_score,
            p_value,
            is_watermarked,
        })
    }
}
