//! Types for the `cross_encoder` module.
use thiserror::Error;
// ── InteractionFeatures ───────────────────────────────────────────────────────
/// Per-(query,doc) interaction feature vector for cross-encoder scoring.
#[derive(Debug, Clone, Default)]
pub struct InteractionFeatures {
    /// Fraction of query tokens that appear verbatim in the document.
    pub exact_match_ratio: f32,
    /// Jaccard coefficient over query and document token sets.
    pub term_overlap: f32,
    /// IDF-weighted overlap score.
    pub idf_weighted_overlap: f32,
    /// Fraction of query tokens covered by the document.
    pub query_coverage: f32,
    /// Fraction of document tokens covered by the query.
    pub doc_coverage: f32,
    /// Fraction of ordered query bigrams found in the document.
    pub ordered_bigram_match: f32,
    /// Ratio `min(|q|,|d|)/max(|q|,|d|)` — penalises extreme length mismatches.
    pub length_ratio: f32,
}
// ── FeatureWeights ────────────────────────────────────────────────────────────
/// Weights applied to each [`InteractionFeatures`] dimension.
#[derive(Debug, Clone)]
pub struct FeatureWeights {
    /// Weight for [`InteractionFeatures::exact_match_ratio`].
    pub exact_match: f32,
    /// Weight for [`InteractionFeatures::term_overlap`].
    pub term_overlap: f32,
    /// Weight for [`InteractionFeatures::idf_weighted_overlap`].
    pub idf_weighted: f32,
    /// Weight for [`InteractionFeatures::query_coverage`].
    pub query_coverage: f32,
    /// Weight for [`InteractionFeatures::doc_coverage`].
    pub doc_coverage: f32,
    /// Weight for [`InteractionFeatures::ordered_bigram_match`].
    pub bigram_match: f32,
    /// Weight for [`InteractionFeatures::length_ratio`].
    pub length_ratio: f32,
}
impl Default for FeatureWeights {
    fn default() -> Self {
        Self {
            exact_match: 0.25,
            term_overlap: 0.20,
            idf_weighted: 0.25,
            query_coverage: 0.10,
            doc_coverage: 0.05,
            bigram_match: 0.10,
            length_ratio: 0.05,
        }
    }
}
impl FeatureWeights {
    /// Normalise all weights so they sum to 1.0.
    #[must_use]
    pub fn normalize(mut self) -> Self {
        let s = self.exact_match
            + self.term_overlap
            + self.idf_weighted
            + self.query_coverage
            + self.doc_coverage
            + self.bigram_match
            + self.length_ratio;
        if s > 0.0 {
            self.exact_match /= s;
            self.term_overlap /= s;
            self.idf_weighted /= s;
            self.query_coverage /= s;
            self.doc_coverage /= s;
            self.bigram_match /= s;
            self.length_ratio /= s;
        }
        self
    }
    /// Set the exact-match weight.
    #[must_use]
    pub fn with_exact_match(mut self, v: f32) -> Self {
        self.exact_match = v;
        self
    }
    /// Set the IDF-weighted overlap weight.
    #[must_use]
    pub fn with_idf_weighted(mut self, v: f32) -> Self {
        self.idf_weighted = v;
        self
    }
}
// ── CrossEncoderConfig ────────────────────────────────────────────────────────
/// Configuration for `CrossEncoderReranker`.
#[derive(Debug, Clone)]
pub struct CrossEncoderConfig {
    /// Maximum number of results to return. `0` means return all.
    pub top_n: usize,
    /// Blending coefficient: `fused = blend_alpha*orig + (1-blend_alpha)*cross`.
    ///
    /// Defaults to `0.4`.
    pub blend_alpha: f32,
    /// Minimum fused score a candidate must achieve to be included.
    ///
    /// Defaults to `0.0` (keep all).
    pub score_threshold: f32,
    /// Feature weights used by `LexicalCrossEncoder`.
    pub weights: FeatureWeights,
}
impl Default for CrossEncoderConfig {
    fn default() -> Self {
        Self {
            top_n: 0,
            blend_alpha: 0.4,
            score_threshold: 0.0,
            weights: FeatureWeights::default(),
        }
    }
}
impl CrossEncoderConfig {
    /// Set how many results to return (0 = all).
    #[must_use]
    pub fn with_top_n(mut self, v: usize) -> Self {
        self.top_n = v;
        self
    }
    /// Set the blending coefficient.
    #[must_use]
    pub fn with_blend_alpha(mut self, v: f32) -> Self {
        self.blend_alpha = v;
        self
    }
    /// Set the minimum fused-score threshold.
    #[must_use]
    pub fn with_score_threshold(mut self, v: f32) -> Self {
        self.score_threshold = v;
        self
    }
    /// Set custom feature weights.
    #[must_use]
    pub fn with_weights(mut self, v: FeatureWeights) -> Self {
        self.weights = v;
        self
    }
}
// ── RerankedResult ────────────────────────────────────────────────────────────
/// A single reranked search result.
#[derive(Debug, Clone)]
pub struct RerankedResult {
    /// The underlying document.
    pub document: crate::types::Document,
    /// Score from the bi-encoder retrieval step.
    pub original_score: f32,
    /// Score produced by the cross-encoder.
    pub cross_score: f32,
    /// Blended final score.
    pub fused_score: f32,
    /// Rank before reranking (0-indexed).
    pub original_rank: usize,
    /// Rank after reranking (0-indexed).
    pub new_rank: usize,
}
// ── CrossEncoderError ─────────────────────────────────────────────────────────
/// Errors from the `cross_encoder` module.
#[derive(Debug, Error)]
pub enum CrossEncoderError {
    /// The query string was empty.
    #[error("Query must not be empty")]
    EmptyQuery,
    /// No candidate results were supplied.
    #[error("Candidate list must not be empty")]
    EmptyCandidates,
}
// ── CrossEncoderScorer ────────────────────────────────────────────────────────

/// Synchronous cross-encoder scorer for a (query, document) pair.
pub trait CrossEncoderScorer {
    /// Score the relevance of `doc` to `query`.
    ///
    /// Returns a value in [0.0, 1.0] (higher = more relevant).
    fn score(&self, query: &str, doc: &crate::types::Document) -> f32;
}
