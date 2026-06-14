//! Core citation-grounding heuristic engine.
//!
//! [`CitationVerifier`] implements a lightweight, model-free NLI heuristic that
//! combines two orthogonal signals:
//!
//! 1. **Token-overlap ratio** — the fraction of the claim's distinct
//!    alphanumeric tokens (length ≥ 2, lowercased) that also appear in the
//!    source passage.
//!
//! 2. **N-gram substring match** — `1.0` when at least one 3-token n-gram from
//!    the claim appears verbatim in the token-normalised source; `0.0` otherwise.
//!
//! The two signals are combined linearly with the weights in [`CitationConfig`]
//! and the result is clamped to `[0.0, 1.0]`.

use std::collections::HashSet;

use super::types::{
    CitationCheck, CitationConfig, CitationError, CitationReport, VerifiedCitation,
};

// ── Minimum n-gram size ────────────────────────────────────────────────────────

/// Minimum contiguous token window size for the substring-match signal.
const MIN_NGRAM_SIZE: usize = 3;

// ── Internal tokeniser ─────────────────────────────────────────────────────────

/// Tokenise `text` into lowercase alphanumeric tokens of length ≥ 2.
///
/// The string is split on every non-alphanumeric character; fragments shorter
/// than two characters are discarded and the survivors are lowercased. This is
/// the canonical tokeniser used throughout the `citation_verification` module.
fn tokenize(text: &str) -> Vec<String> {
    text.split(|c: char| !c.is_alphanumeric())
        .filter(|t| t.len() >= 2)
        .map(str::to_lowercase)
        .collect()
}

/// Tokenise `text` into a deduplicated set of lowercase alphanumeric tokens.
fn token_set(tokens: &[String]) -> HashSet<&str> {
    tokens.iter().map(String::as_str).collect()
}

// ── Internal scoring helpers ───────────────────────────────────────────────────

/// Compute the fraction of `claim_tokens` whose lowercase forms appear in
/// `source_set`. Returns `0.0` when `claim_tokens` is empty.
#[allow(clippy::cast_precision_loss)]
fn compute_token_overlap_ratio(claim_tokens: &[String], source_set: &HashSet<&str>) -> f32 {
    if claim_tokens.is_empty() {
        return 0.0;
    }
    let matched = claim_tokens
        .iter()
        .filter(|t| source_set.contains(t.as_str()))
        .count();
    matched as f32 / claim_tokens.len() as f32
}

/// Extract all distinct 3-token windows from `claim_tokens` whose space-joined
/// form appears as a contiguous substring of the space-joined `source_tokens`.
///
/// Returns a sorted, deduplicated list of matching span strings.
fn find_ngram_matches(claim_tokens: &[String], source_tokens: &[String]) -> Vec<String> {
    if claim_tokens.len() < MIN_NGRAM_SIZE {
        return Vec::new();
    }
    let normalized_source = source_tokens.join(" ");
    let mut matched: Vec<String> = claim_tokens
        .windows(MIN_NGRAM_SIZE)
        .map(|window| window.join(" "))
        .filter(|ngram| normalized_source.contains(ngram.as_str()))
        .collect();
    matched.sort_unstable();
    matched.dedup();
    matched
}

// ── CitationVerifier ───────────────────────────────────────────────────────────

/// Heuristic citation-grounding verifier.
///
/// Given a claim and its cited source passage, [`CitationVerifier`] applies a
/// two-signal lexical heuristic (token overlap + n-gram substring match) to
/// produce a grounding score in `[0.0, 1.0]`.
///
/// For batch evaluation use [`CitationVerifier::verify_batch`], which returns a
/// [`CitationReport`] with per-citation results and summary statistics.
#[derive(Debug, Clone, Copy)]
pub struct CitationVerifier {
    /// The active configuration.
    pub config: CitationConfig,
}

impl CitationVerifier {
    /// Create a new verifier with the given configuration.
    #[must_use]
    pub fn new(config: CitationConfig) -> Self {
        Self { config }
    }

    /// Verify a single citation and return a detailed [`VerifiedCitation`].
    ///
    /// # Algorithm
    ///
    /// 1. Tokenise both `claim` and `source_passage` with the canonical
    ///    tokeniser (`split(non-alphanumeric)`, `filter(len ≥ 2)`, `lowercase`).
    /// 2. Compute the **token-overlap ratio**: `|claim_tokens ∩ source_tokens| /
    ///    |claim_tokens|`.
    /// 3. Collect all 3-token windows from the claim whose space-joined form
    ///    appears as a substring of the space-joined source tokens.
    /// 4. `ngram_score = 1.0` when any window matched; `0.0` otherwise.
    /// 5. `grounding_score` = clamp(`token_overlap_weight` × `overlap_ratio`
    ///    + `substring_weight` × `ngram_score`, 0.0, 1.0).
    /// 6. `is_grounded = grounding_score ≥ min_overlap_ratio`.
    ///
    /// # Errors
    ///
    /// Returns [`CitationError::EmptyClaim`] when the claim is blank, or
    /// [`CitationError::EmptySource`] when the source passage is blank.
    pub fn verify_one(&self, check: CitationCheck) -> Result<VerifiedCitation, CitationError> {
        if check.claim.trim().is_empty() {
            return Err(CitationError::EmptyClaim);
        }
        if check.source_passage.trim().is_empty() {
            return Err(CitationError::EmptySource);
        }

        let claim_tokens = tokenize(&check.claim);
        let source_tokens = tokenize(&check.source_passage);
        let source_set = token_set(&source_tokens);

        let overlap_ratio = compute_token_overlap_ratio(&claim_tokens, &source_set);
        let matched_spans = find_ngram_matches(&claim_tokens, &source_tokens);
        let ngram_score = if matched_spans.is_empty() {
            0.0_f32
        } else {
            1.0_f32
        };

        let grounding_score = (self.config.token_overlap_weight * overlap_ratio
            + self.config.substring_weight * ngram_score)
            .clamp(0.0, 1.0);

        let is_grounded = grounding_score >= self.config.min_overlap_ratio;

        Ok(VerifiedCitation {
            check,
            grounding_score,
            is_grounded,
            matched_spans,
        })
    }

    /// Verify a batch of citations and return an aggregate [`CitationReport`].
    ///
    /// Citations that fail validation (e.g., empty claim or source) are silently
    /// skipped and do not contribute to the report's `total_count` or
    /// `overall_score`. The `overall_score` is the arithmetic mean of grounding
    /// scores across all *successfully* verified citations, or `0.0` when none
    /// were valid.
    #[must_use]
    pub fn verify_batch(&self, checks: Vec<CitationCheck>) -> CitationReport {
        let mut verified: Vec<VerifiedCitation> = Vec::with_capacity(checks.len());
        for check in checks {
            if let Ok(result) = self.verify_one(check) {
                verified.push(result);
            }
        }

        let total_count = verified.len();
        let grounded_count = verified.iter().filter(|v| v.is_grounded).count();

        #[allow(clippy::cast_precision_loss)]
        let overall_score = if total_count == 0 {
            0.0
        } else {
            verified.iter().map(|v| v.grounding_score).sum::<f32>() / total_count as f32
        };

        CitationReport {
            verified,
            overall_score,
            grounded_count,
            total_count,
        }
    }
}
