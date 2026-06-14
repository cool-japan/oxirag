//! Per-document credibility scoring and result re-ranking.
//!
//! [`CredibilityScorer`] blends three deterministic signals into a single
//! credibility score:
//!
//! 1. **`PageRank`** over the citation graph (passed in per document).
//! 2. **Recency** — exponential decay of the document's age.
//! 3. **Authority** — trusted-source membership plus author reputation.
//!
//! It can also [re-rank](CredibilityScorer::rerank) a result set by blending
//! the original relevance score with the computed credibility.
//!
//! All inputs are explicit; in particular **age is supplied as `age_days`** so
//! that scoring never reads the wall clock and stays fully reproducible.

use std::collections::HashMap;

use super::types::{AuthoritySignals, CredibilityConfig, CredibilityScore};
use crate::types::{Document, DocumentId, SearchResult};

/// Metadata key consulted for author attribution.
const AUTHOR_KEY: &str = "author";

/// Blends citation `PageRank`, recency, and metadata authority into a
/// per-document credibility score.
#[derive(Debug, Clone)]
pub struct CredibilityScorer {
    /// Blend configuration (weights, damping, half-life, …).
    pub config: CredibilityConfig,
    /// Out-of-band authority knowledge (trusted sources, author reputation).
    pub signals: AuthoritySignals,
}

impl CredibilityScorer {
    /// Create a scorer with the given configuration and empty authority signals.
    #[must_use]
    pub fn new(config: CredibilityConfig) -> Self {
        Self {
            config,
            signals: AuthoritySignals::new(),
        }
    }

    /// Create a scorer with both configuration and authority signals.
    #[must_use]
    pub fn with_signals(config: CredibilityConfig, signals: AuthoritySignals) -> Self {
        Self { config, signals }
    }

    /// Recency score from an age in days.
    ///
    /// Exponential decay with the configured half-life:
    /// `0.5 ^ (age_days / half_life_days)`.
    ///
    /// * `age_days = 0` → `1.0`
    /// * `age_days = half_life_days` → `0.5`
    ///
    /// Negative ages (future-dated documents) are clamped to `0`, and the result
    /// is clamped to `[0.0, 1.0]`.  A non-positive half-life degenerates to a
    /// step function: `1.0` at age `0`, else `0.0`.
    #[must_use]
    #[allow(clippy::cast_possible_truncation)]
    pub fn recency_score(&self, age_days: f64) -> f32 {
        let age = age_days.max(0.0);
        let half_life = f64::from(self.config.half_life_days);
        if half_life <= 0.0 {
            return if age == 0.0 { 1.0 } else { 0.0 };
        }
        let decay = 0.5_f64.powf(age / half_life);
        (decay as f32).clamp(0.0, 1.0)
    }

    /// Authority score for a document.
    ///
    /// Combines two contributions, each in `[0.0, 1.0]`, by taking their
    /// maximum so a single strong signal already yields high authority:
    ///
    /// * `1.0` when [`Document::source`] is in the trusted-source set, else `0.0`.
    /// * The author reputation looked up from the `"author"` metadata key
    ///   (clamped to `[0.0, 1.0]`), or `0.0` when absent.
    ///
    /// The result is clamped to `[0.0, 1.0]`.
    #[must_use]
    pub fn authority_score(&self, doc: &Document) -> f32 {
        let trusted = doc
            .source
            .as_deref()
            .is_some_and(|s| self.signals.is_trusted(s));
        let trust_component: f32 = if trusted { 1.0 } else { 0.0 };

        let author_component = doc
            .metadata
            .get(AUTHOR_KEY)
            .and_then(|name| self.signals.reputation_of(name))
            .map_or(0.0, |rep| rep.clamp(0.0, 1.0));

        trust_component.max(author_component).clamp(0.0, 1.0)
    }

    /// Compute the full [`CredibilityScore`] for a document.
    ///
    /// `pagerank` is the document's `PageRank` value (already in `[0.0, 1.0]` when
    /// it comes from [`crate::source_credibility::SourceGraph::pagerank`]); it is
    /// clamped defensively.  `age_days` drives the recency component.  The
    /// `total` is the weighted blend with weights normalised to sum to `1.0`.
    #[must_use]
    pub fn score(&self, doc: &Document, pagerank: f32, age_days: f64) -> CredibilityScore {
        let pr = pagerank.clamp(0.0, 1.0);
        let recency = self.recency_score(age_days);
        let authority = self.authority_score(doc);

        let (w_pr, w_rec, w_auth) = self.config.normalized_weights();
        let total = (w_pr * pr + w_rec * recency + w_auth * authority).clamp(0.0, 1.0);

        CredibilityScore::new(total, pr, recency, authority)
    }

    /// Re-rank `results` by blending relevance with credibility.
    ///
    /// For each result the blended score is
    /// `relevance_weight * relevance + (1 - relevance_weight) * credibility.total`,
    /// where credibility is computed from the result document's `PageRank` (looked
    /// up in `graph`) and its age (looked up in `ages`, defaulting to `0.0` —
    /// i.e. maximally recent — when missing).
    ///
    /// `relevance_weight` is clamped to `[0.0, 1.0]`.  The returned vector is
    /// sorted by descending blended score, `rank` fields are re-assigned
    /// `0..n`, and ties break deterministically by document id so the ordering
    /// is reproducible.
    #[must_use]
    pub fn rerank(
        &self,
        results: &[SearchResult],
        graph: &super::graph::SourceGraph,
        ages: &HashMap<DocumentId, f64>,
        relevance_weight: f32,
    ) -> Vec<SearchResult> {
        let rel_w = relevance_weight.clamp(0.0, 1.0);
        let cred_w = 1.0 - rel_w;

        let pageranks = graph.pagerank(self.config.damping, self.config.iterations);

        let mut scored: Vec<(f32, SearchResult)> = results
            .iter()
            .map(|res| {
                let id = &res.document.id;
                let pr = pageranks.get(id).copied().unwrap_or(0.0);
                let age = ages.get(id).copied().unwrap_or(0.0);
                let cred = self.score(&res.document, pr, age);
                let blended = rel_w * res.score + cred_w * cred.total;
                (blended, res.clone())
            })
            .collect();

        scored.sort_by(|(sa, ra), (sb, rb)| {
            sb.partial_cmp(sa)
                .unwrap_or(std::cmp::Ordering::Equal)
                .then_with(|| ra.document.id.as_str().cmp(rb.document.id.as_str()))
        });

        scored
            .into_iter()
            .enumerate()
            .map(|(rank, (blended, mut res))| {
                res.score = blended;
                res.rank = rank;
                res
            })
            .collect()
    }
}

impl Default for CredibilityScorer {
    fn default() -> Self {
        Self::new(CredibilityConfig::default())
    }
}
