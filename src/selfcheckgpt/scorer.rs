//! [`SelfCheckScorer`]: the top-level entry point that splits the main
//! response into sentences and dispatches to the configured variant.
//!
//! This file also hosts the small set of text-processing helpers
//! (tokenisation, sentence splitting, Jaccard overlap) shared by
//! `ngram`, `nli_lite`, and `qa_lite`.

use std::collections::HashSet;

use super::ngram::ngram_inconsistency_scores;
use super::nli_lite::nli_lite_inconsistency_scores;
use super::qa_lite::qa_lite_inconsistency_scores;
use super::types::{
    SelfCheckConfig, SelfCheckError, SelfCheckScore, SelfCheckVariant, SentenceCheck,
};

// ── shared text helpers ───────────────────────────────────────────────────────

/// Tokenise `text` into a lowercase vector of alphanumeric tokens.
///
/// Splits on every non-alphanumeric character and drops empty fragments.
/// This is the canonical tokeniser shared by all three scoring variants so
/// that "similar" tokens are computed identically everywhere in this module.
pub(crate) fn tokenize(text: &str) -> Vec<String> {
    text.split(|c: char| !c.is_alphanumeric())
        .filter(|t| !t.is_empty())
        .map(str::to_lowercase)
        .collect()
}

/// Split `text` into individual sentences.
///
/// A sentence boundary is a `.`, `!`, or `?` immediately followed by a space
/// or newline (or end of input). Empty fragments are dropped and each
/// sentence is trimmed of leading/trailing whitespace.
pub(crate) fn split_sentences(text: &str) -> Vec<String> {
    let mut sentences: Vec<String> = Vec::new();
    let mut current = String::new();

    let chars: Vec<char> = text.chars().collect();
    let n = chars.len();
    let mut i = 0;

    while i < n {
        let c = chars[i];
        current.push(c);

        if matches!(c, '.' | '!' | '?') {
            let at_newline = i + 1 < n && chars[i + 1] == '\n';
            let at_space = i + 1 < n && chars[i + 1] == ' ';
            if at_newline || at_space {
                let trimmed = current.trim().to_string();
                if !trimmed.is_empty() {
                    sentences.push(trimmed);
                }
                current.clear();
                i += 2;
                continue;
            }
        }
        i += 1;
    }

    let trimmed = current.trim().to_string();
    if !trimmed.is_empty() {
        sentences.push(trimmed);
    }

    sentences
}

/// Jaccard similarity between two token slices, treated as sets.
///
/// Returns `0.0` when both slices are empty.
pub(crate) fn jaccard(a_tokens: &[String], b_tokens: &[String]) -> f32 {
    if a_tokens.is_empty() && b_tokens.is_empty() {
        return 0.0;
    }
    let a_set: HashSet<&String> = a_tokens.iter().collect();
    let b_set: HashSet<&String> = b_tokens.iter().collect();
    let intersection = a_set.intersection(&b_set).count();
    let union = a_set.union(&b_set).count();
    if union == 0 {
        return 0.0;
    }
    #[allow(clippy::cast_precision_loss)]
    let score = intersection as f32 / union as f32;
    score
}

// ── SelfCheckScorer ────────────────────────────────────────────────────────────

/// Zero-resource hallucination detector via sampling consistency
/// (`SelfCheckGPT`; Manakul et al., EMNLP 2023).
///
/// Splits the main response into sentences and, for each sentence, computes
/// an inconsistency score against `K` independently sampled responses to the
/// same prompt, using the heuristic selected by
/// [`SelfCheckConfig::variant`]. `SelfCheckScorer` holds no mutable state and
/// is safe to share across threads.
#[derive(Debug, Clone)]
pub struct SelfCheckScorer {
    /// Configuration controlling the variant, thresholds, and sample bounds.
    pub config: SelfCheckConfig,
}

impl SelfCheckScorer {
    /// Constructs a new scorer with the supplied configuration.
    #[must_use]
    pub fn new(config: SelfCheckConfig) -> Self {
        Self { config }
    }

    /// Scores `main_response` for hallucination risk using `samples` as the
    /// `K` stochastically sampled responses to the same prompt.
    ///
    /// # Errors
    ///
    /// - [`SelfCheckError::InvalidConfig`] — the scorer's configuration fails
    ///   [`SelfCheckConfig::validate`].
    /// - [`SelfCheckError::EmptyResponse`] — `main_response` is empty or
    ///   contains only whitespace.
    /// - [`SelfCheckError::InsufficientSamples`] — `samples.len()` is less
    ///   than [`SelfCheckConfig::min_samples`].
    pub fn score(
        &self,
        main_response: &str,
        samples: &[String],
    ) -> Result<SelfCheckScore, SelfCheckError> {
        self.config.validate()?;

        if main_response.trim().is_empty() {
            return Err(SelfCheckError::EmptyResponse);
        }

        if samples.len() < self.config.min_samples {
            return Err(SelfCheckError::InsufficientSamples {
                got: samples.len(),
                need: self.config.min_samples,
            });
        }

        let sentences = {
            let split = split_sentences(main_response);
            if split.is_empty() {
                vec![main_response.trim().to_string()]
            } else {
                split
            }
        };

        let raw_scores: Vec<f32> = match self.config.variant {
            SelfCheckVariant::NGram => {
                ngram_inconsistency_scores(&sentences, samples, self.config.ngram_size)
            }
            SelfCheckVariant::NliLite => nli_lite_inconsistency_scores(&sentences, samples),
            SelfCheckVariant::QaLite => qa_lite_inconsistency_scores(&sentences, samples),
        };

        debug_assert_eq!(raw_scores.len(), sentences.len());

        let threshold = self.config.hallucination_threshold;
        let sentence_checks: Vec<SentenceCheck> = sentences
            .into_iter()
            .zip(raw_scores)
            .map(|(sentence, raw_score)| {
                let inconsistency_score = raw_score.clamp(0.0, 1.0);
                let is_hallucination = inconsistency_score >= threshold;
                SentenceCheck {
                    sentence,
                    inconsistency_score,
                    is_hallucination,
                }
            })
            .collect();

        #[allow(clippy::cast_precision_loss)]
        let overall_score = if sentence_checks.is_empty() {
            0.0_f32
        } else {
            sentence_checks
                .iter()
                .map(|c| c.inconsistency_score)
                .sum::<f32>()
                / sentence_checks.len() as f32
        };

        Ok(SelfCheckScore {
            sentence_checks,
            overall_score,
            variant_used: self.config.variant,
        })
    }
}

impl Default for SelfCheckScorer {
    fn default() -> Self {
        Self::new(SelfCheckConfig::default())
    }
}
