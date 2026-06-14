//! Query-difficulty predictor: scores a query and assigns a [`DifficultyBand`].
//!
//! The [`DifficultyPredictor`] is a pure, deterministic function of its
//! [`DifficultyConfig`].  It holds no mutable state and is safe to share
//! across threads.
//!
//! ## Algorithm
//!
//! 1. Tokenise the lowercased query by splitting on every non-alphanumeric
//!    character and discarding single-character tokens.
//! 2. Detect the five difficulty signals (see [`DifficultySignals`]).
//! 3. Compute the weighted sum and clamp it to `[0.0, 1.0]`.
//! 4. Map the score to a [`DifficultyBand`] via fixed thresholds.

use super::types::{
    AMBIGUITY_MARKERS, DifficultyBand, DifficultyConfig, DifficultyError, DifficultyScore,
    DifficultySignals, MULTI_HOP_INDICATORS, NEGATION_WORDS, STOP_WORDS,
};

// ── Internal helpers ──────────────────────────────────────────────────────────

/// Tokenise `text` using the canonical `OxiRAG` tokeniser:
/// split on non-alphanumeric characters, discard tokens shorter than 2 chars,
/// lowercase every surviving token.
fn tokenise(text: &str) -> Vec<String> {
    text.split(|c: char| !c.is_alphanumeric())
        .filter(|t| t.len() >= 2)
        .map(str::to_lowercase)
        .collect()
}

/// Returns `true` when `phrase` appears inside `text` at a whole-word (or
/// whole-phrase) boundary.
///
/// A valid match requires that the character immediately before the match
/// position (if any) is not an ASCII letter, and the character immediately
/// after the end of the match (if any) is also not an ASCII letter.  This
/// prevents short words like `"or"` from matching inside `"horror"` and
/// similarly prevents `"no"` from matching inside `"notable"`.
fn phrase_in_text(text: &str, phrase: &str) -> bool {
    let text_bytes = text.as_bytes();
    let phrase_len = phrase.len();
    let text_len = text_bytes.len();

    if phrase_len == 0 || phrase_len > text_len {
        return false;
    }

    let mut search_start = 0usize;
    while search_start + phrase_len <= text_len {
        match text[search_start..].find(phrase) {
            None => return false,
            Some(rel_pos) => {
                let pos = search_start + rel_pos;
                let before_ok = pos == 0 || !text_bytes[pos - 1].is_ascii_alphabetic();
                let after_pos = pos + phrase_len;
                let after_ok =
                    after_pos >= text_len || !text_bytes[after_pos].is_ascii_alphabetic();
                if before_ok && after_ok {
                    return true;
                }
                // This occurrence failed boundary check; keep searching.
                search_start = pos + 1;
            }
        }
    }
    false
}

/// Normalise raw token count to `[0.0, 1.0]`.
///
/// Uses `token_count / 15` so that a 15-token query already reaches the
/// maximum.  Shorter queries score proportionally lower.
#[inline]
#[allow(clippy::cast_precision_loss)]
fn length_score(token_count: usize) -> f32 {
    (token_count as f32 / 15.0_f32).min(1.0_f32)
}

/// Map a score in `[0.0, 1.0]` to a [`DifficultyBand`].
///
/// | Range | Band |
/// |-------|------|
/// | `[0.0, 0.3)` | [`Easy`](DifficultyBand::Easy) |
/// | `[0.3, 0.55)` | [`Medium`](DifficultyBand::Medium) |
/// | `[0.55, 0.75)` | [`Hard`](DifficultyBand::Hard) |
/// | `[0.75, 1.0]` | [`Ambiguous`](DifficultyBand::Ambiguous) |
fn score_to_band(score: f32) -> DifficultyBand {
    if score < 0.3 {
        DifficultyBand::Easy
    } else if score < 0.55 {
        DifficultyBand::Medium
    } else if score < 0.75 {
        DifficultyBand::Hard
    } else {
        DifficultyBand::Ambiguous
    }
}

// ── DifficultyPredictor ───────────────────────────────────────────────────────

/// Estimates the difficulty of answering a natural-language query.
///
/// Constructs a scalar difficulty score from five complementary signals:
/// negation presence, multi-hop indicators, ambiguity markers, query length,
/// and rare-word ratio.  The score is mapped to a [`DifficultyBand`] and
/// returned as a [`DifficultyScore`] that also exposes the decomposed
/// [`DifficultySignals`] for explainability.
///
/// `DifficultyPredictor` holds no mutable state; it is safe to use from
/// multiple threads simultaneously.
#[derive(Debug, Clone)]
pub struct DifficultyPredictor {
    /// The configuration governing signal weights.
    pub config: DifficultyConfig,
}

impl DifficultyPredictor {
    /// Creates a new predictor with the supplied configuration.
    #[must_use]
    pub fn new(config: DifficultyConfig) -> Self {
        Self { config }
    }

    /// Predicts the difficulty of `query` and returns a [`DifficultyScore`].
    ///
    /// # Errors
    ///
    /// - [`DifficultyError::EmptyQuery`] — `query` is empty or contains only
    ///   whitespace after trimming.
    /// - [`DifficultyError::PredictionFailed`] — score computation produced a
    ///   non-finite value (e.g. NaN or Inf due to extreme weight values).
    pub fn predict(&self, query: &str) -> Result<DifficultyScore, DifficultyError> {
        let trimmed = query.trim();
        if trimmed.is_empty() {
            return Err(DifficultyError::EmptyQuery);
        }

        let query_lower = trimmed.to_lowercase();

        // ── 1. Tokenise ──────────────────────────────────────────────────────
        let tokens = tokenise(trimmed);
        let token_count = tokens.len();

        // ── 2. Detect signals ────────────────────────────────────────────────
        let has_negation = NEGATION_WORDS
            .iter()
            .any(|w| phrase_in_text(&query_lower, w));

        let multi_hop_indicator = MULTI_HOP_INDICATORS
            .iter()
            .any(|p| phrase_in_text(&query_lower, p));

        let ambiguity_marker = AMBIGUITY_MARKERS
            .iter()
            .any(|m| phrase_in_text(&query_lower, m));

        // ── 3. Rare-word ratio ────────────────────────────────────────────────
        #[allow(clippy::cast_precision_loss)]
        let rare_word_ratio = if token_count == 0 {
            0.0_f32
        } else {
            let rare_count = tokens
                .iter()
                .filter(|t| !STOP_WORDS.contains(&t.as_str()))
                .count();
            rare_count as f32 / token_count as f32
        };

        let signals = DifficultySignals {
            has_negation,
            multi_hop_indicator,
            ambiguity_marker,
            token_count,
            rare_word_ratio,
        };

        // ── 4. Aggregate score ────────────────────────────────────────────────
        let bool_to_f32 = |b: bool| -> f32 { if b { 1.0_f32 } else { 0.0_f32 } };
        let raw_score = self.config.negation_weight * bool_to_f32(has_negation)
            + self.config.multi_hop_weight * bool_to_f32(multi_hop_indicator)
            + self.config.ambiguity_weight * bool_to_f32(ambiguity_marker)
            + self.config.length_weight * length_score(token_count)
            + self.config.rare_word_weight * rare_word_ratio;

        if !raw_score.is_finite() {
            return Err(DifficultyError::PredictionFailed(
                "score is not finite — check for NaN or Inf weights in DifficultyConfig"
                    .to_string(),
            ));
        }

        let score = raw_score.clamp(0.0_f32, 1.0_f32);
        let band = score_to_band(score);

        Ok(DifficultyScore {
            query: trimmed.to_string(),
            score,
            band,
            signals,
        })
    }
}
