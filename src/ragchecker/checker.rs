//! Claim-level `RAGChecker` diagnostics engine.
//!
//! Implements the lexical, deterministic core of `RAGChecker` (Ru et al. 2024):
//! decompose the response and the ground-truth answer into atomic claims, test
//! each claim for lexical *entailment* (token-overlap ratio at or above a
//! threshold) against the retrieved context and the ground truth, and aggregate
//! the per-claim outcomes into six retriever / generator diagnostics.

use std::collections::HashSet;

use crate::types::Document;

use super::types::{RagCheckResult, RagCheckerConfig, RagCheckerError, RagCheckerMetrics};

// ── Tokeniser ─────────────────────────────────────────────────────────────────

/// Tokenise `text` into lowercase alphanumeric tokens of length >= 2.
///
/// The string is split on every non-alphanumeric character; fragments shorter
/// than two characters are discarded and the survivors are lowercased.
fn tokenize(text: &str) -> Vec<String> {
    text.split(|c: char| !c.is_alphanumeric())
        .filter(|t| t.len() >= 2)
        .map(str::to_lowercase)
        .collect()
}

/// Tokenise `text` into a deduplicated set of lowercase alphanumeric tokens.
fn token_set(text: &str) -> HashSet<String> {
    tokenize(text).into_iter().collect()
}

// ── Sentence / clause splitting ───────────────────────────────────────────────

/// Clause-boundary markers (matched after sentence splitting).
const CLAUSE_BOUNDARIES: &[&str] = &[" which ", " but ", " and ", " who ", " that ", "; ", ", "];

/// Split `text` into trimmed, non-empty sentences on `.`, `!`, and `?`.
fn split_sentences(text: &str) -> Vec<String> {
    text.split(['.', '!', '?'])
        .map(str::trim)
        .filter(|s| !s.is_empty())
        .map(str::to_string)
        .collect()
}

/// Split a single sentence into trimmed, non-empty clause fragments.
fn split_clauses(sentence: &str) -> Vec<String> {
    let mut fragments: Vec<String> = vec![sentence.to_string()];
    for boundary in CLAUSE_BOUNDARIES {
        let mut next: Vec<String> = Vec::new();
        for fragment in &fragments {
            for piece in fragment.split(boundary) {
                next.push(piece.to_string());
            }
        }
        fragments = next;
    }
    fragments
        .into_iter()
        .map(|f| f.trim().to_string())
        .filter(|f| !f.is_empty())
        .collect()
}

/// Decompose `text` into atomic claims (sentences split into clauses).
///
/// Fragments that contain no usable tokens are dropped so that empty or purely
/// punctuation clauses never become claims.
fn decompose_claims(text: &str) -> Vec<String> {
    let mut claims: Vec<String> = Vec::new();
    for sentence in split_sentences(text) {
        for clause in split_clauses(&sentence) {
            if tokenize(&clause).is_empty() {
                continue;
            }
            claims.push(clause);
        }
    }
    claims
}

/// Compute the fraction `entailed / total`, returning `0.0` for an empty input.
#[allow(clippy::cast_precision_loss)]
fn fraction(entailed: usize, total: usize) -> f32 {
    if total == 0 {
        0.0
    } else {
        entailed as f32 / total as f32
    }
}

// ── RagChecker ────────────────────────────────────────────────────────────────

/// Claim-level RAG diagnostics engine.
///
/// Decomposes the response and the ground-truth answer into atomic claims and
/// scores each claim's lexical entailment against the retrieved context and the
/// ground truth to produce the six [`RagCheckerMetrics`].
#[derive(Debug, Clone, Copy, Default)]
pub struct RagChecker {
    /// The active configuration.
    pub config: RagCheckerConfig,
}

impl RagChecker {
    /// Create a new checker with the given configuration.
    #[must_use]
    pub fn new(config: RagCheckerConfig) -> Self {
        Self { config }
    }

    /// Test whether `text` lexically entails `claim`.
    ///
    /// Returns `true` when the fraction of the claim's distinct tokens that also
    /// appear in `text` is at least the configured entailment threshold. A claim
    /// with no usable tokens is never entailed.
    #[must_use]
    #[allow(clippy::cast_precision_loss)]
    pub fn entailed(&self, claim: &str, text: &str) -> bool {
        let claim_tokens = token_set(claim);
        if claim_tokens.is_empty() {
            return false;
        }
        let text_tokens = token_set(text);
        let matched = claim_tokens
            .iter()
            .filter(|t| text_tokens.contains(*t))
            .count();
        let ratio = matched as f32 / claim_tokens.len() as f32;
        ratio >= self.config.entailment_threshold
    }

    /// Test whether any passage in `retrieved` entails `claim`.
    fn entailed_by_context(self, claim: &str, retrieved: &[Document]) -> bool {
        retrieved
            .iter()
            .any(|doc| self.entailed(claim, &doc.content))
    }

    /// Run the full claim-level diagnostic over a single RAG sample.
    ///
    /// Decomposes `response` and `ground_truth` into atomic claims and computes
    /// all six metrics. Context entailment treats a claim as supported when it is
    /// entailed by *any* retrieved passage.
    ///
    /// # Errors
    ///
    /// Returns [`RagCheckerError::EmptyResponse`] when `response` is blank, or
    /// [`RagCheckerError::EmptyGroundTruth`] when `ground_truth` is blank.
    pub fn check(
        &self,
        response: &str,
        ground_truth: &str,
        retrieved: &[Document],
    ) -> Result<RagCheckResult, RagCheckerError> {
        if response.trim().is_empty() {
            return Err(RagCheckerError::EmptyResponse);
        }
        if ground_truth.trim().is_empty() {
            return Err(RagCheckerError::EmptyGroundTruth);
        }

        let response_claims = decompose_claims(response);
        let gt_claims = decompose_claims(ground_truth);

        // ── Retriever diagnostics ──────────────────────────────────────────
        // claim_recall: fraction of GT claims entailed by the retrieved context.
        let recalled = gt_claims
            .iter()
            .filter(|claim| self.entailed_by_context(claim, retrieved))
            .count();
        let claim_recall = fraction(recalled, gt_claims.len());

        // context_precision: fraction of retrieved passages entailing >= 1 GT claim.
        let useful_passages = retrieved
            .iter()
            .filter(|doc| {
                gt_claims
                    .iter()
                    .any(|claim| self.entailed(claim, &doc.content))
            })
            .count();
        let context_precision = fraction(useful_passages, retrieved.len());

        // ── Generator diagnostics ──────────────────────────────────────────
        let mut faithful = 0usize;
        let mut hallucinated = 0usize;
        let mut correct = 0usize;
        let mut noisy = 0usize;
        for claim in &response_claims {
            let in_context = self.entailed_by_context(claim, retrieved);
            let in_gt = self.entailed(claim, ground_truth);
            if in_context {
                faithful += 1;
            }
            if in_gt {
                correct += 1;
            }
            if !in_context && !in_gt {
                hallucinated += 1;
            }
            if in_context && !in_gt {
                noisy += 1;
            }
        }
        let total_response = response_claims.len();
        let faithfulness = fraction(faithful, total_response);
        let hallucination_rate = fraction(hallucinated, total_response);
        let correctness = fraction(correct, total_response);
        let noise_sensitivity = fraction(noisy, total_response);

        let metrics = RagCheckerMetrics {
            claim_recall,
            context_precision,
            faithfulness,
            hallucination_rate,
            correctness,
            noise_sensitivity,
        };

        Ok(RagCheckResult {
            metrics,
            response_claims,
            gt_claims,
        })
    }
}
