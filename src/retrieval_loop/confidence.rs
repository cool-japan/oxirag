//! Confidence estimation for the FLARE retrieval loop.
//!
//! FLARE requires a way to assign a pseudo-confidence score to each generated
//! token so that low-confidence spans can be identified and used as retrieval
//! queries.  Because we operate without a live LLM and its token probabilities,
//! we use a *context-frequency heuristic*:
//!
//! - Tokens that appear more often in the current retrieval context are assumed
//!   to be "supported" (high confidence).
//! - Tokens absent from the context receive a base confidence derived from
//!   log-smoothing, which is low but non-zero.
//!
//! This mirrors the spirit of FLARE's use of log-probabilities from a language
//! model while remaining fully dependency-free.

use super::types::{SentenceConfidence, TokenConfidence};

// ── ConfidenceEstimator ───────────────────────────────────────────────────────

/// Stateless helper that computes confidence scores for tokens and sentences
/// relative to an in-context corpus.
///
/// All methods are pure functions with no mutable state.
pub struct ConfidenceEstimator;

impl ConfidenceEstimator {
    /// Estimate the pseudo-confidence of a single `token` given the current
    /// `context`.
    ///
    /// # Algorithm
    ///
    /// 1. Normalise `token` to lowercase and strip surrounding punctuation.
    /// 2. Count occurrences of the normalised token in `context` (case-insensitive).
    /// 3. Apply log-smoothing:
    ///    ```text
    ///    confidence = ln(1 + occurrences) / ln(1 + context_word_count)
    ///    ```
    /// 4. Clamp the result to `[0.0, 1.0]`.
    ///
    /// When `context` is empty the denominator becomes `ln(1)  = 0`, which
    /// would be a division by zero.  In that case `0.0` is returned.
    #[must_use]
    pub fn estimate_token_confidence(token: &str, context: &str) -> f32 {
        let normalised = token
            .to_lowercase()
            .trim_matches(|c: char| !c.is_alphanumeric())
            .to_string();

        if normalised.is_empty() {
            return 0.0;
        }

        let context_words: Vec<&str> = context.split_whitespace().collect();
        let context_word_count = context_words.len();

        if context_word_count == 0 {
            return 0.0;
        }

        // Count occurrences — compare normalised token against lowercase context words.
        let occurrences = context_words
            .iter()
            .filter(|&&w| {
                let w_norm = w
                    .to_lowercase()
                    .trim_matches(|c: char| !c.is_alphanumeric())
                    .to_string();
                w_norm == normalised
            })
            .count();

        #[allow(clippy::cast_precision_loss)]
        let occ_f = occurrences as f32;
        #[allow(clippy::cast_precision_loss)]
        let ctx_f = context_word_count as f32;
        let confidence = (1.0 + occ_f).ln() / (1.0 + ctx_f).ln();

        confidence.clamp(0.0, 1.0)
    }

    /// Estimate confidence for every token in `sentence` against `context`,
    /// returning a [`SentenceConfidence`] with the per-token scores and the
    /// sentence-level average.
    ///
    /// Tokenisation splits on one or more characters that are not
    /// alphanumeric or apostrophe (`[^a-zA-Z0-9']+`), discarding empty
    /// fragments.
    #[must_use]
    pub fn estimate_sentence_confidence(sentence: &str, context: &str) -> SentenceConfidence {
        // Use a simple manual split to avoid requiring regex at this layer.
        let raw_tokens: Vec<&str> = sentence
            .split(|c: char| !c.is_alphanumeric() && c != '\'')
            .filter(|s| !s.is_empty())
            .collect();

        let tokens: Vec<TokenConfidence> = raw_tokens
            .into_iter()
            .map(|t| {
                let confidence = Self::estimate_token_confidence(t, context);
                TokenConfidence {
                    token: t.to_lowercase(),
                    confidence,
                }
            })
            .collect();

        SentenceConfidence::new(sentence.to_string(), tokens)
    }

    /// Split `text` into individual sentences.
    ///
    /// Sentence boundaries are detected at `. `, `? `, `! `, and `\n\n`.
    /// Each resulting sentence is trimmed and empty fragments are discarded.
    ///
    /// The trailing punctuation (`.`, `?`, `!`) is retained on the preceding
    /// sentence fragment.
    #[must_use]
    pub fn split_into_sentences(text: &str) -> Vec<String> {
        // We split on the *space* that follows a sentence-ending punctuation mark
        // so that the punctuation stays with its sentence.  We also split on
        // paragraph breaks (`\n\n`).
        //
        // Strategy: scan character-by-character and emit a sentence whenever we
        // see one of the boundary sequences.
        let mut sentences: Vec<String> = Vec::new();
        let mut current = String::new();
        let chars: Vec<char> = text.chars().collect();
        let len = chars.len();
        let mut i = 0;

        while i < len {
            let c = chars[i];

            // Paragraph break → flush.
            if c == '\n' && i + 1 < len && chars[i + 1] == '\n' {
                let trimmed = current.trim().to_string();
                if !trimmed.is_empty() {
                    sentences.push(trimmed);
                }
                current.clear();
                i += 2; // skip the second '\n'
                continue;
            }

            // Check for `. `, `? `, `! ` — keep the punctuation, flush on the space.
            if matches!(c, '.' | '?' | '!') && i + 1 < len && chars[i + 1] == ' ' {
                current.push(c);
                let trimmed = current.trim().to_string();
                if !trimmed.is_empty() {
                    sentences.push(trimmed);
                }
                current.clear();
                i += 2; // skip the punctuation + space
                continue;
            }

            current.push(c);
            i += 1;
        }

        // Flush any trailing content.
        let trimmed = current.trim().to_string();
        if !trimmed.is_empty() {
            sentences.push(trimmed);
        }

        sentences
    }

    /// Identify sentences in `text` that are below `threshold` in average
    /// confidence, according to the current `context`.
    ///
    /// Returns the [`SentenceConfidence`] records for all uncertain sentences
    /// in document order.
    #[must_use]
    pub fn identify_uncertain_sentences(
        text: &str,
        context: &str,
        threshold: f32,
    ) -> Vec<SentenceConfidence> {
        Self::split_into_sentences(text)
            .into_iter()
            .map(|s| Self::estimate_sentence_confidence(&s, context))
            .filter(|sc| sc.is_uncertain(threshold))
            .collect()
    }
}

// ── Helper: tokeniser (used in tests) ────────────────────────────────────────

/// Tokenise `text` using the canonical `[^a-zA-Z0-9']+` split described in the
/// module specification.  Returns only non-empty fragments.
///
/// Splits on any character that is not alphanumeric or an apostrophe.
#[must_use]
pub fn tokenise(text: &str) -> Vec<String> {
    text.split(|c: char| !c.is_alphanumeric() && c != '\'')
        .filter(|s| !s.is_empty())
        .map(std::string::ToString::to_string)
        .collect()
}
