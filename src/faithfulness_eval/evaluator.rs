//! The faithfulness evaluator: claim decomposition and entailment scoring.

use std::collections::HashSet;

use crate::faithfulness_eval::types::{
    ClaimEntailment, FaithfulnessConfig, FaithfulnessError, FaithfulnessScore,
};

/// Minimum number of word-tokens (each of length ≥ 2) a sentence fragment must
/// contain to be kept as an atomic claim.
const MIN_CLAIM_TOKEN_COUNT: usize = 2;

// ── FaithfulnessEvaluator ─────────────────────────────────────────────────────

/// Evaluates the faithfulness of a generated answer with respect to a set of
/// retrieved context passages.
///
/// The algorithm follows the RAGAS-style faithfulness metric:
///
/// 1. **Claim decomposition** — the answer is split into atomic claim fragments
///    at punctuation boundaries (configured via
///    [`FaithfulnessConfig::claim_split_tokens`]).  Fragments with fewer than
///    two vocabulary tokens are discarded.
///
/// 2. **Entailment scoring** — for each claim, every context passage is scored
///    by computing the *claim coverage*: the fraction of the claim's unique
///    lowercase alphanumeric tokens (length ≥ 2) that appear anywhere in the
///    passage.  The maximum score across all passages is the claim's entailment
///    score.  The passage achieving that maximum is reported as the
///    `supporting_passage`.
///
/// 3. **Aggregation** — the aggregate faithfulness score is
///    `entailed_claims / total_claims`, where a claim is entailed when its
///    entailment score ≥ [`FaithfulnessConfig::min_entailment_score`].
///
/// Everything is deterministic and pure Rust (`std` + `thiserror`): no ML, no
/// randomness, no numeric dependencies.
#[derive(Debug, Clone)]
pub struct FaithfulnessEvaluator {
    /// Configuration for this evaluator.
    pub config: FaithfulnessConfig,
}

impl FaithfulnessEvaluator {
    /// Create a new evaluator with the given configuration.
    #[must_use]
    pub fn new(config: FaithfulnessConfig) -> Self {
        Self { config }
    }

    // ── internal helpers ──────────────────────────────────────────────────────

    /// Tokenise a text string into the set of unique lowercase alphanumeric
    /// tokens of length ≥ 2.
    ///
    /// Splitting is performed on any non-alphanumeric character, which handles
    /// spaces, punctuation, hyphens, apostrophes, etc.
    fn tokenize(text: &str) -> HashSet<String> {
        text.split(|c: char| !c.is_alphanumeric())
            .filter(|t| t.len() >= 2)
            .map(str::to_lowercase)
            .collect()
    }

    // ── public API ────────────────────────────────────────────────────────────

    /// Split an answer string into atomic claim fragments.
    ///
    /// Each character found in any of the
    /// [`FaithfulnessConfig::claim_split_tokens`] strings acts as a boundary.
    /// Resulting fragments are trimmed of surrounding whitespace and then
    /// filtered: only fragments containing at least two vocabulary tokens
    /// (alphanumeric runs of length ≥ 2) are kept.
    #[must_use]
    pub fn split_into_claims(&self, answer: &str) -> Vec<String> {
        // Build the flat set of split characters from all configured tokens.
        let split_chars: HashSet<char> = self
            .config
            .claim_split_tokens
            .iter()
            .flat_map(|t| t.chars())
            .collect();

        answer
            .split(|c: char| split_chars.contains(&c))
            .map(str::trim)
            .filter(|fragment| {
                fragment
                    .split(|c: char| !c.is_alphanumeric())
                    .filter(|t| t.len() >= 2)
                    .count()
                    >= MIN_CLAIM_TOKEN_COUNT
            })
            .map(str::to_string)
            .collect()
    }

    /// Compute the entailment score for a single claim against a set of context
    /// passages.
    ///
    /// The heuristic is **claim coverage**: for each passage, compute
    /// `|claim_tokens ∩ passage_tokens| / |claim_tokens|` (the fraction of the
    /// claim's unique vocabulary tokens that appear in the passage).  The
    /// maximum score over all passages is returned together with the passage
    /// that produced it — the *supporting passage*.
    ///
    /// Returns `(0.0, None)` when the claim has no vocabulary tokens or no
    /// passage achieves any token overlap.
    ///
    /// # Arguments
    ///
    /// * `claim` — the atomic claim fragment to evaluate.
    /// * `context_passages` — slices of context passage strings.
    #[must_use]
    #[allow(clippy::cast_precision_loss)]
    pub fn entailment_score(
        &self,
        claim: &str,
        context_passages: &[&str],
    ) -> (f32, Option<String>) {
        let claim_tokens = Self::tokenize(claim);
        if claim_tokens.is_empty() {
            return (0.0, None);
        }
        let claim_size = claim_tokens.len() as f32;

        let mut best_score = 0.0_f32;
        let mut best_passage: Option<&str> = None;

        for &passage in context_passages {
            let passage_tokens = Self::tokenize(passage);
            let intersection_count = claim_tokens
                .iter()
                .filter(|t| passage_tokens.contains(*t))
                .count();

            // Claim coverage: what fraction of the claim's tokens appear in
            // this passage?  Higher means the passage better supports the claim.
            let score = intersection_count as f32 / claim_size;

            if score > best_score {
                best_score = score;
                best_passage = Some(passage);
            }
        }

        // Only report a supporting passage when there is genuine token overlap.
        if best_score > 0.0 {
            (best_score, best_passage.map(str::to_string))
        } else {
            (0.0, None)
        }
    }

    /// Evaluate the faithfulness of `answer` with respect to the provided
    /// context passages.
    ///
    /// Decomposes the answer into atomic claims via [`split_into_claims`], scores
    /// each claim with [`entailment_score`], and aggregates the results into a
    /// [`FaithfulnessScore`].
    ///
    /// # Errors
    ///
    /// * [`FaithfulnessError::EmptyAnswer`] — if `answer` is empty or
    ///   contains only whitespace.
    /// * [`FaithfulnessError::EmptyContext`] — if `context` is empty.
    ///
    /// [`split_into_claims`]: Self::split_into_claims
    /// [`entailment_score`]: Self::entailment_score
    #[allow(clippy::cast_precision_loss)]
    pub fn evaluate(
        &self,
        answer: &str,
        context: &[String],
    ) -> Result<FaithfulnessScore, FaithfulnessError> {
        if answer.trim().is_empty() {
            return Err(FaithfulnessError::EmptyAnswer);
        }
        if context.is_empty() {
            return Err(FaithfulnessError::EmptyContext);
        }

        let claims = self.split_into_claims(answer);
        let passages: Vec<&str> = context.iter().map(String::as_str).collect();

        let claim_entailments: Vec<ClaimEntailment> = claims
            .iter()
            .map(|claim| {
                let (score, passage) = self.entailment_score(claim, &passages);
                let is_entailed = score >= self.config.min_entailment_score;
                ClaimEntailment {
                    claim: claim.clone(),
                    entailment_score: score,
                    is_entailed,
                    supporting_passage: passage,
                }
            })
            .collect();

        let total_claims = claim_entailments.len();
        let entailed_count = claim_entailments.iter().filter(|e| e.is_entailed).count();
        let faithfulness = if total_claims == 0 {
            0.0_f32
        } else {
            entailed_count as f32 / total_claims as f32
        };

        Ok(FaithfulnessScore {
            answer: answer.to_owned(),
            claims: claim_entailments,
            faithfulness,
            entailed_count,
            total_claims,
        })
    }
}
