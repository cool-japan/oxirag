//! Corpus-poisoning / adversarial-passage detection over a retrieved result set.

use std::collections::HashSet;

use crate::types::SearchResult;

use super::types::{PoisonAssessment, PoisonConfig, PoisonError};

// ── Tokenisation ────────────────────────────────────────────────────────────────

/// Ordered token list of `text` (lowercased alphanumeric runs of length >= 2).
///
/// Preserves repetition and order, which the stuffing and diversity signals both
/// depend on — duplicate tokens are *not* collapsed.
fn tokens(text: &str) -> Vec<String> {
    text.split(|c: char| !c.is_alphanumeric())
        .filter(|t| t.len() >= 2)
        .map(str::to_lowercase)
        .collect()
}

/// Lowercase distinct token set of `text` (alphanumeric runs of length >= 2).
fn token_set(text: &str) -> HashSet<String> {
    text.split(|c: char| !c.is_alphanumeric())
        .filter(|t| t.len() >= 2)
        .map(str::to_lowercase)
        .collect()
}

/// Jaccard similarity between two token sets, in `[0, 1]`.
///
/// Two empty sets are perfectly similar (`1.0`); a non-empty set against an
/// empty one is fully dissimilar (`0.0`).
#[allow(clippy::cast_precision_loss)]
fn jaccard(a: &HashSet<String>, b: &HashSet<String>) -> f32 {
    if a.is_empty() && b.is_empty() {
        return 1.0;
    }
    let inter = a.intersection(b).count() as f32;
    let union = a.union(b).count() as f32;
    if union == 0.0 { 0.0 } else { inter / union }
}

// ── PoisoningDetector ───────────────────────────────────────────────────────────

/// Detects adversarial passages crafted to be retrieved and to mislead
/// generation (corpus poisoning).
///
/// Unlike a noise / distractor filter — which removes passages that are merely
/// *irrelevant* to the query — this detector targets passages that are
/// deliberately *engineered*. Three orthogonal signals are combined:
///
/// * **stuffing** — query-term density inside the passage. Adversarial passages
///   often pack the query's own terms to win retrieval, producing an abnormally
///   high density relative to genuine prose.
/// * **diversity** — the type/token ratio of the passage. Crafted text is
///   frequently padded by repeating a handful of tokens, collapsing the ratio
///   far below that of natural language. Here a *low* score is the suspicious
///   one.
/// * **anomaly** — disagreement with the corpus consensus, measured as
///   `1 - mean lexical similarity` to the other retrieved passages. A passage
///   that contradicts or stands apart from what the rest of the set is "about"
///   scores high.
///
/// The three are blended into a single `risk`, and a passage is flagged when the
/// risk crosses [`PoisonConfig::risk_threshold`] **or** when either hard signal
/// trips on its own (stuffing too high, or diversity too low).
///
/// The detector is **pure compute**: deterministic tokenisation and set
/// arithmetic, no I/O, no randomness, no model.
#[derive(Debug, Clone, Default)]
pub struct PoisoningDetector {
    /// Detection configuration.
    config: PoisonConfig,
}

impl PoisoningDetector {
    /// Create a new detector with the given configuration.
    #[must_use]
    pub fn new(config: PoisonConfig) -> Self {
        Self { config }
    }

    /// Access the detector's configuration.
    #[must_use]
    pub fn config(&self) -> &PoisonConfig {
        &self.config
    }

    /// Query-term density in `text`: the fraction of `text` tokens that are also
    /// query tokens.
    ///
    /// This is a *token-weighted* density rather than a set-overlap ratio, so a
    /// passage that repeats the query terms over and over scores far higher than
    /// one that mentions them once — exactly the keyword-stuffing pattern. The
    /// result lies in `[0, 1]`; an empty query or empty text yields `0.0`.
    #[must_use]
    #[allow(clippy::cast_precision_loss)]
    pub fn stuffing_score(&self, query: &str, text: &str) -> f32 {
        let query_terms = token_set(query);
        if query_terms.is_empty() {
            return 0.0;
        }
        let text_tokens = tokens(text);
        if text_tokens.is_empty() {
            return 0.0;
        }
        let hits = text_tokens
            .iter()
            .filter(|t| query_terms.contains(*t))
            .count() as f32;
        (hits / text_tokens.len() as f32).clamp(0.0, 1.0)
    }

    /// Lexical diversity of `text`: its type/token ratio.
    ///
    /// Defined as `distinct tokens / total tokens`, in `(0, 1]`. Natural prose
    /// sits well above the configured floor; text padded by repeating a few
    /// tokens collapses toward `0`. Empty text yields `1.0` (vacuously diverse,
    /// so it never trips the low-diversity rule on its own).
    #[must_use]
    #[allow(clippy::cast_precision_loss)]
    pub fn diversity_score(&self, text: &str) -> f32 {
        let all = tokens(text);
        let total = all.len();
        if total == 0 {
            return 1.0;
        }
        let distinct = all.into_iter().collect::<HashSet<String>>().len();
        (distinct as f32 / total as f32).clamp(0.0, 1.0)
    }

    /// Anomaly of `text` versus the surrounding corpus: `1 - mean similarity` to
    /// the other passages.
    ///
    /// Similarity is Jaccard over distinct token sets. A passage that contradicts
    /// or simply stands apart from the corpus consensus shares few tokens with
    /// the others and so scores high; a passage that blends in scores low. The
    /// passage's own identical copy inside `corpus_texts` is skipped so that a
    /// passage is never compared against itself. With no other passages to
    /// compare against, anomaly is `0.0` (nothing to disagree with). The result
    /// lies in `[0, 1]`.
    #[must_use]
    #[allow(clippy::cast_precision_loss)]
    pub fn anomaly_score(&self, text: &str, corpus_texts: &[&str]) -> f32 {
        let target = token_set(text);
        let mut sum = 0.0f32;
        let mut count = 0usize;
        for other in corpus_texts {
            // Skip the passage's own entry (exact text match) so a passage is
            // never treated as its own neighbour.
            if *other == text {
                continue;
            }
            sum += jaccard(&target, &token_set(other));
            count += 1;
        }
        if count == 0 {
            return 0.0;
        }
        let mean_similarity = sum / count as f32;
        (1.0 - mean_similarity).clamp(0.0, 1.0)
    }

    /// Assess a single passage `text` against the `query` and the surrounding
    /// `corpus_texts`.
    ///
    /// Computes the three signal scores, blends them into a poison `risk`, and
    /// decides whether the passage is poisoned. The blend weights stuffing and
    /// anomaly directly and folds in a *diversity deficit* (`1 - diversity`) so
    /// that low diversity raises risk:
    ///
    /// `risk = 0.4 * stuffing + 0.3 * (1 - diversity) + 0.3 * anomaly`.
    ///
    /// The passage is flagged (`is_poisoned`) when
    /// `risk >= risk_threshold` **or** `stuffing >= stuffing_threshold` **or**
    /// `diversity <= diversity_threshold`.
    #[must_use]
    pub fn assess(&self, query: &str, text: &str, corpus_texts: &[&str]) -> PoisonAssessment {
        let stuffing = self.stuffing_score(query, text);
        let diversity = self.diversity_score(text);
        let anomaly = self.anomaly_score(text, corpus_texts);

        let risk = (0.4 * stuffing + 0.3 * (1.0 - diversity) + 0.3 * anomaly).clamp(0.0, 1.0);
        let is_poisoned = risk >= self.config.risk_threshold
            || stuffing >= self.config.stuffing_threshold
            || diversity <= self.config.diversity_threshold;

        PoisonAssessment {
            index: 0,
            stuffing_score: stuffing,
            diversity_score: diversity,
            anomaly_score: anomaly,
            risk,
            is_poisoned,
        }
    }

    /// Scan a full retrieved result set, returning one [`PoisonAssessment`] per
    /// result in input order.
    ///
    /// Each passage is assessed against every *other* passage's content as its
    /// corpus context, so the anomaly signal reflects the set's own consensus.
    /// An empty `results` slice yields an empty vector. The `query` is *not*
    /// validated here; use [`PoisoningDetector::filter`] for empty-query
    /// rejection, or check it yourself.
    #[must_use]
    pub fn scan(&self, query: &str, results: &[SearchResult]) -> Vec<PoisonAssessment> {
        if results.is_empty() {
            return Vec::new();
        }
        let corpus: Vec<&str> = results
            .iter()
            .map(|r| r.document.content.as_str())
            .collect();
        results
            .iter()
            .enumerate()
            .map(|(index, result)| {
                let mut assessment = self.assess(query, &result.document.content, &corpus);
                assessment.index = index;
                assessment
            })
            .collect()
    }

    /// Filter `results`, dropping passages flagged as poisoned and re-ranking the
    /// survivors `0..`.
    ///
    /// Survivors keep their original relative order and are re-ranked from `0`.
    /// An empty input yields an empty output.
    ///
    /// # Errors
    ///
    /// Returns [`PoisonError::EmptyQuery`] when `query` is empty or whitespace
    /// only and `results` is non-empty. An empty `results` slice short-circuits
    /// to an empty vector without validating the query.
    pub fn filter(
        &self,
        query: &str,
        results: &[SearchResult],
    ) -> Result<Vec<SearchResult>, PoisonError> {
        if results.is_empty() {
            return Ok(Vec::new());
        }
        if query.trim().is_empty() {
            return Err(PoisonError::EmptyQuery);
        }
        let assessments = self.scan(query, results);
        let mut kept: Vec<SearchResult> = Vec::new();
        for (idx, result) in results.iter().enumerate() {
            if !assessments[idx].is_poisoned {
                let mut keep = result.clone();
                keep.rank = kept.len();
                kept.push(keep);
            }
        }
        Ok(kept)
    }
}
