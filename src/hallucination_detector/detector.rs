//! Core hallucination-detection logic: tokenisation, Jaccard scoring, claim splitting.

use std::collections::HashSet;

use super::types::{ClaimSupport, HallucinationConfig, HallucinationError, HallucinationReport};

// ── helpers ───────────────────────────────────────────────────────────────────

/// Split `text` on whitespace, lowercase, and strip non-alphanumeric characters.
fn tokenize(text: &str) -> Vec<String> {
    text.split_whitespace()
        .map(|w| {
            w.to_lowercase()
                .chars()
                .filter(|c| c.is_alphanumeric())
                .collect::<String>()
        })
        .filter(|s| !s.is_empty())
        .collect()
}

/// Jaccard similarity between two token slices.
///
/// Returns `0.0` when both slices are empty.
fn jaccard(a_tokens: &[String], b_tokens: &[String]) -> f32 {
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

/// Split `text` into individual claims by sentence boundaries.
///
/// Splits on `". "`, `"! "`, `"? "`, and `".\n"`.  Empty strings are dropped.
fn split_claims(text: &str) -> Vec<String> {
    let mut sentences: Vec<String> = Vec::new();
    let mut current = String::new();

    let chars: Vec<char> = text.chars().collect();
    let n = chars.len();
    let mut i = 0;

    while i < n {
        let c = chars[i];
        current.push(c);

        if matches!(c, '.' | '!' | '?') {
            // Check for ".\n" or ". " / "! " / "? "
            let at_newline = c == '.' && i + 1 < n && chars[i + 1] == '\n';
            let at_space = i + 1 < n && chars[i + 1] == ' ';
            if at_newline || at_space {
                let trimmed = current.trim().to_string();
                if !trimmed.is_empty() {
                    sentences.push(trimmed);
                }
                current.clear();
                i += 2; // skip the '\n' or ' '
                continue;
            }
        }
        i += 1;
    }

    // Flush remainder
    let trimmed = current.trim().to_string();
    if !trimmed.is_empty() {
        sentences.push(trimmed);
    }

    sentences.into_iter().filter(|s| !s.is_empty()).collect()
}

/// Score a single claim against one document's content.
///
/// When `sentence_level` is `true` the document is split into sentences and the
/// maximum Jaccard score across all sentence-pairs is returned; otherwise the
/// claim is compared against the full document.
fn score_claim_against_doc(claim: &str, doc_content: &str, sentence_level: bool) -> f32 {
    let claim_tokens = tokenize(claim);
    if sentence_level {
        let doc_sentences = split_claims(doc_content);
        if doc_sentences.is_empty() {
            // Fall back to whole-doc comparison
            let doc_tokens = tokenize(doc_content);
            return jaccard(&claim_tokens, &doc_tokens);
        }
        doc_sentences
            .iter()
            .map(|s| jaccard(&claim_tokens, &tokenize(s)))
            .fold(0.0_f32, f32::max)
    } else {
        let doc_tokens = tokenize(doc_content);
        jaccard(&claim_tokens, &doc_tokens)
    }
}

// ── HallucinationDetector ─────────────────────────────────────────────────────

/// Lexical claim-support scorer for detecting hallucinations in RAG answers.
///
/// Uses Jaccard similarity between claim tokens and source-document tokens to
/// determine whether each claim is grounded in the provided sources.
#[derive(Debug, Clone)]
pub struct HallucinationDetector {
    /// Configuration controlling scoring thresholds and splitting behaviour.
    pub config: HallucinationConfig,
}

impl HallucinationDetector {
    /// Construct a detector with the given configuration.
    #[must_use]
    pub fn new(config: HallucinationConfig) -> Self {
        Self { config }
    }

    /// Detect hallucinations in `answer` relative to the provided `sources`.
    ///
    /// # Errors
    ///
    /// Returns [`HallucinationError::EmptyAnswer`] when `answer` is blank.
    /// Returns [`HallucinationError::EmptySources`] when `sources` is empty.
    pub fn detect(
        &self,
        answer: &str,
        sources: &[crate::types::Document],
    ) -> Result<HallucinationReport, HallucinationError> {
        if answer.trim().is_empty() {
            return Err(HallucinationError::EmptyAnswer);
        }
        if sources.is_empty() {
            return Err(HallucinationError::EmptySources);
        }

        let claims: Vec<String> = if self.config.sentence_level {
            let extracted = split_claims(answer);
            if extracted.is_empty() {
                vec![answer.to_string()]
            } else {
                extracted
            }
        } else {
            vec![answer.to_string()]
        };

        let min_support = self.config.min_support;
        let sentence_level = self.config.sentence_level;

        let claim_supports: Vec<ClaimSupport> = claims
            .into_iter()
            .map(|claim| {
                // Score claim against every source
                let scores: Vec<f32> = sources
                    .iter()
                    .map(|src| score_claim_against_doc(&claim, &src.content, sentence_level))
                    .collect();

                let max_score = scores.iter().copied().fold(0.0_f32, f32::max);

                // Collect contributing source descriptions (score >= min_support * 0.8)
                let contribution_threshold = min_support * 0.8;
                let supporting_sources: Vec<String> = sources
                    .iter()
                    .zip(scores.iter())
                    .filter(|&(_, &s)| s >= contribution_threshold)
                    .map(|(src, _)| src.id.as_str().to_string())
                    .collect();

                let is_hallucination = max_score < min_support;

                ClaimSupport::new(claim, max_score, supporting_sources, is_hallucination)
            })
            .collect();

        Ok(HallucinationReport::new(claim_supports))
    }
}

impl Default for HallucinationDetector {
    fn default() -> Self {
        Self::new(HallucinationConfig::default())
    }
}
