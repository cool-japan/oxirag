//! Post-deletion leakage re-audit: probing a store, after deletion, to
//! verify that content actually became unreachable rather than just
//! assuming a `delete()` call was sufficient.
//!
//! This is the module's central claim made checkable: [`UnlearningAuditor`]
//! extracts distinctive probe terms from the content that was supposed to
//! be forgotten, scans every surviving document for verbatim or
//! near-verbatim traces of it, and reports a [`UnlearningLeakageVerdict`]
//! that [`crate::knowledge_unlearning::UnlearningEngine`] cannot silently
//! override.

use std::collections::HashMap;

use super::dedup::{UnlearningNearDuplicateDetector, jaccard, tokenize};
use super::engine::UnlearnableStore;
use super::types::{UnlearningAuditEvidence, UnlearningConfig, UnlearningLeakageVerdict};

/// A small, hand-rolled English stopword list used only to keep probe
/// n-grams from being dominated by function words. Not exhaustive by
/// design — the goal is to bias term *selection*, not to build a general
/// NLP stopword filter.
const STOPWORDS: &[&str] = &[
    "a", "an", "and", "are", "as", "at", "be", "been", "being", "but", "by", "can", "could", "did",
    "do", "does", "for", "from", "had", "has", "have", "if", "in", "into", "is", "it", "its",
    "may", "might", "no", "not", "of", "on", "onto", "or", "over", "should", "so", "than", "that",
    "the", "then", "these", "this", "those", "to", "under", "was", "were", "will", "with", "would",
];

/// `true` when `word` is in [`STOPWORDS`].
fn is_stopword(word: &str) -> bool {
    STOPWORDS.contains(&word)
}

/// The average character length of `words`, used as a cheap, deterministic
/// proxy for "distinctiveness" in the absence of real corpus term-frequency
/// statistics: longer words are, on average, rarer and more specific than
/// short function words.
#[allow(clippy::cast_precision_loss)]
fn average_word_length(words: &[String]) -> f64 {
    if words.is_empty() {
        return 0.0;
    }
    let total: usize = words.iter().map(|w| w.chars().count()).sum();
    total as f64 / words.len() as f64
}

/// Extract up to `top_n` distinctive `ngram_size`-word probe terms from
/// `content`.
///
/// N-grams composed entirely of [`STOPWORDS`] are discarded; the survivors
/// are ranked by [`average_word_length`] (longer, more specific spans
/// first) and deduplicated, keeping each distinct n-gram's best-seen score.
/// Ties are broken lexicographically so extraction is fully deterministic.
pub(crate) fn extract_probe_terms(content: &str, ngram_size: usize, top_n: usize) -> Vec<String> {
    let tokens = tokenize(content);
    if tokens.is_empty() {
        return Vec::new();
    }
    let size = ngram_size.max(1);

    let windows: Vec<&[String]> = if tokens.len() < size {
        vec![tokens.as_slice()]
    } else {
        tokens.windows(size).collect()
    };

    let mut best: HashMap<String, f64> = HashMap::new();
    let mut order: Vec<String> = Vec::new();
    for window in windows {
        if window.iter().all(|w| is_stopword(w)) {
            continue;
        }
        let gram = window.join(" ");
        let score = average_word_length(window);
        match best.get_mut(&gram) {
            Some(existing) if *existing >= score => {}
            Some(existing) => *existing = score,
            None => {
                best.insert(gram.clone(), score);
                order.push(gram);
            }
        }
    }

    let mut ranked: Vec<(String, f64)> = order
        .into_iter()
        .map(|gram| {
            let score = best[&gram];
            (gram, score)
        })
        .collect();
    ranked.sort_by(|a, b| b.1.total_cmp(&a.1).then_with(|| a.0.cmp(&b.0)));
    ranked
        .into_iter()
        .take(top_n.max(1))
        .map(|(gram, _)| gram)
        .collect()
}

// ── UnlearningAuditor ─────────────────────────────────────────────────────────

/// Probes a store for residual leakage of content that was supposed to have
/// been deleted.
///
/// An audit does not know or care *how* the content was deleted (one
/// document, several near-duplicates, a full cascade) — it only asks: given
/// the text that was supposed to disappear, can I still find it, verbatim
/// or near-verbatim, anywhere in what remains?
#[derive(Debug, Clone)]
pub struct UnlearningAuditor {
    /// The configuration this auditor scores with.
    pub config: UnlearningConfig,
}

impl UnlearningAuditor {
    /// Create a new auditor with the given configuration.
    #[must_use]
    pub fn new(config: UnlearningConfig) -> Self {
        Self { config }
    }

    /// The distinctive probe terms this auditor would extract from
    /// `deleted_content`, exposed for transparency and testing.
    #[must_use]
    pub fn probe_terms(&self, deleted_content: &str) -> Vec<String> {
        extract_probe_terms(
            deleted_content,
            self.config.probe_ngram_size,
            self.config.probe_top_n,
        )
    }

    /// Probe `store` for residual leakage of `deleted_content`.
    ///
    /// For every document currently in `store`, this computes:
    ///
    /// - a **verbatim hit ratio**: the fraction of
    ///   [`Self::probe_terms`] found as a case-insensitive substring of the
    ///   document's content;
    /// - a **similarity**: the `MinHash`-estimated `Jaccard` similarity
    ///   between the document and `deleted_content` as a whole.
    ///
    /// combined into a per-document `leakage_score` via
    /// `verbatim_weight * verbatim_ratio + similarity_weight * similarity`.
    /// Documents scoring at or above [`UnlearningConfig::leakage_threshold`]
    /// are reported as evidence, sorted by descending score (ties broken by
    /// ascending document id) so the worst offender is always first.
    #[must_use]
    pub fn audit<S: UnlearnableStore>(
        &self,
        store: &S,
        deleted_content: &str,
    ) -> UnlearningLeakageVerdict {
        let probes = self.probe_terms(deleted_content);
        let detector =
            UnlearningNearDuplicateDetector::new(self.config.shingle_size, self.config.num_perm);
        let deleted_signature = detector.signature(deleted_content);

        let mut evidence: Vec<UnlearningAuditEvidence> = Vec::new();
        for doc in store.iter_documents() {
            let content_lower = doc.content.to_lowercase();
            let matched_terms: Vec<String> = probes
                .iter()
                .filter(|term| !term.is_empty() && content_lower.contains(term.as_str()))
                .cloned()
                .collect();

            let doc_signature = detector.signature(&doc.content);
            let similarity = jaccard(&deleted_signature, &doc_signature);

            #[allow(clippy::cast_precision_loss)]
            let verbatim_ratio = if probes.is_empty() {
                0.0
            } else {
                matched_terms.len() as f64 / probes.len() as f64
            };

            let leakage_score = self.config.verbatim_weight * verbatim_ratio
                + self.config.similarity_weight * similarity;

            // `leakage_score > 0.0` is required in addition to the
            // configured threshold so that `leakage_threshold == 0.0`
            // ("nothing tolerated") flags every document with *any*
            // detected signal, rather than degenerating into flagging
            // every surviving document unconditionally (since a bare
            // `>= 0.0` is trivially true for a score of exactly zero).
            if leakage_score > 0.0 && leakage_score >= self.config.leakage_threshold {
                evidence.push(UnlearningAuditEvidence {
                    document_id: doc.id,
                    matched_terms,
                    similarity,
                    leakage_score,
                });
            }
        }

        if evidence.is_empty() {
            return UnlearningLeakageVerdict::Clean;
        }

        evidence.sort_by(|a, b| {
            b.leakage_score
                .total_cmp(&a.leakage_score)
                .then_with(|| a.document_id.as_str().cmp(b.document_id.as_str()))
        });
        let top_leakage_score = evidence[0].leakage_score;
        let surviving_doc_ids = evidence.iter().map(|e| e.document_id.clone()).collect();

        UnlearningLeakageVerdict::ResidualLeakage {
            evidence,
            surviving_doc_ids,
            score: top_leakage_score,
        }
    }
}
