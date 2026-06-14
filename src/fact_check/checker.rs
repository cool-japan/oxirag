//! Core fact-verification logic: tokenisation, sentence splitting, evidence
//! retrieval, contradiction detection, and verdict assignment.

use std::collections::HashSet;

use super::types::{Evidence, FactCheckConfig, FactCheckError, FactCheckResult, Verdict};
use crate::types::Document;

// ── helpers ─────────────────────────────────────────────────────────────────────

/// Tokenize `text`: split on non-alphanumeric boundaries, lowercase, keep tokens
/// of length `>= 2`.
fn tokenize(text: &str) -> Vec<String> {
    text.split(|c: char| !c.is_alphanumeric())
        .filter(|s| !s.is_empty())
        .map(str::to_lowercase)
        .filter(|s| s.chars().count() >= 2)
        .collect()
}

/// Split `text` into sentences on `'.'`, `'!'`, and `'?'`.
///
/// Trimmed, non-empty fragments are returned. A trailing fragment without a
/// terminator is also included.
fn split_sentences(text: &str) -> Vec<String> {
    let mut sentences: Vec<String> = Vec::new();
    let mut current = String::new();

    for c in text.chars() {
        current.push(c);
        if matches!(c, '.' | '!' | '?') {
            let trimmed = current.trim();
            if !trimmed.is_empty() {
                sentences.push(trimmed.to_string());
            }
            current.clear();
        }
    }

    let trimmed = current.trim();
    if !trimmed.is_empty() {
        sentences.push(trimmed.to_string());
    }

    sentences
}

/// Token-overlap score between two token slices.
///
/// Computed as `intersection / claim_size`, i.e. the fraction of the claim's
/// distinct tokens that appear in the evidence. Returns `0.0` when the claim has
/// no tokens.
fn overlap(claim_tokens: &[String], evidence_tokens: &[String]) -> f32 {
    let claim_set: HashSet<&String> = claim_tokens.iter().collect();
    if claim_set.is_empty() {
        return 0.0;
    }
    let evidence_set: HashSet<&String> = evidence_tokens.iter().collect();
    let intersection = claim_set.intersection(&evidence_set).count();
    #[allow(clippy::cast_precision_loss)]
    let score = intersection as f32 / claim_set.len() as f32;
    score
}

/// Negation markers. Whole-token markers and a substring marker (`n't`).
const NEGATION_TOKENS: &[&str] = &[
    "not", "no", "never", "without", "fails", "fail", "cannot", "false",
];

/// Return `true` when `text` exhibits a negation signal.
///
/// Detected via the whole-token markers in [`NEGATION_TOKENS`] or the contracted
/// `n't` suffix (e.g. `isn't`, `wasn't`, `doesn't`).
fn has_negation(text: &str) -> bool {
    let lower = text.to_lowercase();
    if lower.contains("n't") {
        return true;
    }
    let tokens = tokenize(&lower);
    tokens.iter().any(|t| NEGATION_TOKENS.contains(&t.as_str()))
}

/// Extract distinct numeric tokens from `text`.
///
/// Pure digit runs are parsed; surrounding punctuation (commas, currency,
/// percent signs) is stripped. Values are returned as `f64`.
fn extract_numbers(text: &str) -> Vec<f64> {
    text.split_whitespace()
        .filter_map(|token| {
            let cleaned: String = token
                .chars()
                .filter(|c| c.is_ascii_digit() || *c == '.')
                .collect();
            // Reject fragments that are just "." or empty.
            if cleaned.is_empty() || cleaned.chars().all(|c| c == '.') {
                return None;
            }
            cleaned.parse::<f64>().ok()
        })
        .collect()
}

/// Distinct content tokens shared between two token slices.
fn shared_tokens(a: &[String], b: &[String]) -> Vec<String> {
    let a_set: HashSet<&String> = a.iter().collect();
    let b_set: HashSet<&String> = b.iter().collect();
    a_set.intersection(&b_set).map(|s| (*s).clone()).collect()
}

/// Determine whether `evidence` contradicts `claim`.
///
/// Two contradiction signals are checked, both gated on the claim and evidence
/// sharing subject vocabulary (at least one shared content token):
///
/// * **Negation mismatch** — exactly one of the two carries a negation marker.
/// * **Number mismatch** — both cite numbers, but they share no numeric value.
fn detect_contradiction(claim_tokens: &[String], claim: &str, evidence: &str) -> bool {
    let evidence_tokens = tokenize(evidence);
    let shared = shared_tokens(claim_tokens, &evidence_tokens);
    if shared.is_empty() {
        // No shared subject: cannot be a meaningful contradiction.
        return false;
    }

    // ── Negation mismatch ──────────────────────────────────────────────────
    let claim_neg = has_negation(claim);
    let evidence_neg = has_negation(evidence);
    if claim_neg ^ evidence_neg {
        return true;
    }

    // ── Number mismatch ────────────────────────────────────────────────────
    let claim_nums = extract_numbers(claim);
    let evidence_nums = extract_numbers(evidence);
    if !claim_nums.is_empty() && !evidence_nums.is_empty() {
        let shares_value = claim_nums.iter().any(|cn| {
            evidence_nums
                .iter()
                .any(|en| (cn - en).abs() < f64::EPSILON)
        });
        if !shares_value {
            return true;
        }
    }

    false
}

// ── FactChecker ───────────────────────────────────────────────────────────────

/// FEVER-style fact verifier over a corpus of evidence documents.
///
/// Given a claim, the checker splits each document into sentences, scores them by
/// token overlap with the claim, flags contradictions (negation or numeric
/// mismatch), and assigns a three-way [`Verdict`] with a confidence and
/// rationale.
#[derive(Debug, Clone)]
pub struct FactChecker {
    /// Configuration controlling retrieval depth and verdict thresholds.
    pub config: FactCheckConfig,
}

impl FactChecker {
    /// Construct a fact checker with the given configuration.
    #[must_use]
    pub fn new(config: FactCheckConfig) -> Self {
        Self { config }
    }

    /// Retrieve the most relevant evidence sentences for `claim` from `docs`.
    ///
    /// Each document is split into sentences; every sentence is scored by token
    /// overlap with the claim and flagged for contradiction. Sentences are sorted
    /// by descending score (ties broken by descending contradiction flag, then by
    /// sentence text for determinism) and the top `evidence_top_k` are returned.
    #[must_use]
    pub fn retrieve_evidence(&self, claim: &str, docs: &[Document]) -> Vec<Evidence> {
        let claim_tokens = tokenize(claim);

        let mut candidates: Vec<Evidence> = Vec::new();
        for doc in docs {
            for sentence in split_sentences(&doc.content) {
                let sentence_tokens = tokenize(&sentence);
                let score = overlap(&claim_tokens, &sentence_tokens);
                let contradicts = detect_contradiction(&claim_tokens, claim, &sentence);
                candidates.push(Evidence::new(doc.id.clone(), sentence, score, contradicts));
            }
        }

        // Deterministic ordering: score desc, then contradicts desc, then text asc.
        candidates.sort_by(|a, b| {
            b.score
                .partial_cmp(&a.score)
                .unwrap_or(std::cmp::Ordering::Equal)
                .then_with(|| b.contradicts.cmp(&a.contradicts))
                .then_with(|| a.sentence.cmp(&b.sentence))
        });

        candidates.truncate(self.config.evidence_top_k);
        candidates
    }

    /// Verify `claim` against `docs`, producing a [`FactCheckResult`].
    ///
    /// The verdict is derived from the retrieved evidence:
    ///
    /// * If any strong-overlap evidence (score `>= support_threshold`)
    ///   contradicts the claim → [`Verdict::Refutes`].
    /// * Else if the top evidence score `>= support_threshold` (and does not
    ///   contradict) → [`Verdict::Supports`].
    /// * Else if the top evidence score `< nei_threshold` → [`Verdict::NotEnoughInfo`].
    /// * Otherwise (weak-but-present overlap) → [`Verdict::NotEnoughInfo`].
    ///
    /// Confidence is derived from the margin of the deciding score relative to
    /// the relevant threshold, clamped to `[0.0, 1.0]`.
    ///
    /// # Errors
    ///
    /// Returns [`FactCheckError::EmptyClaim`] when `claim` is blank.
    /// Returns [`FactCheckError::EmptyCorpus`] when `docs` is empty.
    pub fn verify(
        &self,
        claim: &str,
        docs: &[Document],
    ) -> Result<FactCheckResult, FactCheckError> {
        if claim.trim().is_empty() {
            return Err(FactCheckError::EmptyClaim);
        }
        if docs.is_empty() {
            return Err(FactCheckError::EmptyCorpus);
        }

        let evidence = self.retrieve_evidence(claim, docs);

        // No sentences at all in the corpus → not enough information.
        let Some(top) = evidence.first() else {
            return Ok(FactCheckResult::new(
                Verdict::NotEnoughInfo,
                0.0,
                evidence,
                "No usable evidence sentences were found in the corpus.",
            ));
        };
        let top_score = top.score;

        let support_threshold = self.config.support_threshold;
        let nei_threshold = self.config.nei_threshold;

        // A refutation requires a strong-overlap contradicting sentence.
        let refuting = evidence
            .iter()
            .find(|e| e.contradicts && e.score >= support_threshold);

        if let Some(refuter) = refuting {
            let margin = (refuter.score - support_threshold).clamp(0.0, 1.0);
            let confidence = (0.5 + margin).clamp(0.0, 1.0);
            let rationale = format!(
                "Refuted: evidence with overlap {:.2} contradicts the claim (negation or numeric mismatch): \"{}\".",
                refuter.score, refuter.sentence
            );
            return Ok(FactCheckResult::new(
                Verdict::Refutes,
                confidence,
                evidence,
                rationale,
            ));
        }

        if top_score >= support_threshold && !top.contradicts {
            let margin = (top_score - support_threshold).clamp(0.0, 1.0);
            let confidence = (0.5 + margin).clamp(0.0, 1.0);
            let rationale = format!(
                "Supported: top evidence overlap {top_score:.2} meets the support threshold {support_threshold:.2} with no contradiction signal."
            );
            return Ok(FactCheckResult::new(
                Verdict::Supports,
                confidence,
                evidence,
                rationale,
            ));
        }

        if top_score < nei_threshold {
            let confidence = (nei_threshold - top_score).clamp(0.0, 1.0);
            let rationale = format!(
                "Not enough info: best evidence overlap {top_score:.2} is below the relevance floor {nei_threshold:.2}."
            );
            return Ok(FactCheckResult::new(
                Verdict::NotEnoughInfo,
                confidence,
                evidence,
                rationale,
            ));
        }

        // Weak but present overlap, short of the support threshold.
        let span = (support_threshold - nei_threshold).max(f32::EPSILON);
        let confidence = (1.0 - (top_score - nei_threshold) / span).clamp(0.0, 1.0);
        let rationale = format!(
            "Not enough info: best evidence overlap {top_score:.2} is below the support threshold {support_threshold:.2} and shows no decisive contradiction."
        );
        Ok(FactCheckResult::new(
            Verdict::NotEnoughInfo,
            confidence,
            evidence,
            rationale,
        ))
    }
}

impl Default for FactChecker {
    fn default() -> Self {
        Self::new(FactCheckConfig::default())
    }
}
