//! Retrieval diversity & coverage metrics over a ranked result list.
//!
//! Standard IR ranking metrics (precision, nDCG, MRR — see the
//! `retrieval_eval` module) reward *relevance*. They are blind
//! to redundancy: a list of ten near-identical relevant documents scores as well
//! as ten complementary ones. This module measures the orthogonal quality —
//! **diversity and subtopic coverage** — over a ranked list whose results carry
//! per-result subtopic labels.
//!
//! Three metrics are provided, all in `[0.0, 1.0]`:
//!
//! - **Intra-List Diversity (ILD)** — the mean pairwise dissimilarity
//!   (`1 - cosine`) of the top-`k` results' FNV-1a pseudo-embeddings. High ILD
//!   means the surface forms of the results differ.
//! - **Subtopic Recall (S-recall@k)** — the fraction of all subtopics covered by
//!   the union of the top-`k` results' labels. This is *coverage*: did the list
//!   reach every facet of the information need?
//! - **α-nDCG@k** — a novelty-discounted nDCG. A result's gain for a subtopic is
//!   multiplied by `(1 - alpha)^c`, where `c` counts how many times that subtopic
//!   was already seen above it, so re-covering a subtopic pays diminishing
//!   returns. It is normalised by the ideal greedy ordering, and `alpha = 1.0`
//!   removes the discount entirely.
//!
//! # Relationship to the `diversity_rank` module
//!
//! That module *selects* a diverse subset of results (greedy DPP MAP inference);
//! this module *measures* the diversity and coverage of a list you already have.
//! To avoid a prelude collision its config and error are named
//! [`RetrievalDiversityConfig`] and [`RetrievalDiversityError`] rather than
//! `DiversityConfig` / `DiversityRankError`.
//!
//! # Determinism
//!
//! Everything is pure, allocation-light, and deterministic: embeddings come from
//! the FNV-1a scheme shared across `OxiRAG`, and the α-nDCG ideal ordering is a
//! deterministic greedy with lowest-index tie-breaking.
//!
//! # Quick start
//!
//! ```rust
//! # #[cfg(feature = "retrieval-diversity")]
//! # {
//! use oxirag::retrieval_diversity::{DiversityScorer, RetrievalDiversityConfig};
//!
//! let scorer = DiversityScorer::new(RetrievalDiversityConfig::default());
//!
//! // Three results, each tagged with the subtopics it covers.
//! let texts = ["rust async tokio", "python pandas frame", "rust async tokio"];
//! let subtopics = vec![vec![0_usize], vec![1], vec![0]];
//!
//! let metrics = scorer.compute(&texts, &subtopics, 2, 2).unwrap();
//! assert!(metrics.ild >= 0.0 && metrics.ild <= 1.0);
//! assert!((metrics.s_recall - 1.0).abs() < 1e-6); // both subtopics covered in top-2
//! # }
//! ```

pub mod metrics;
#[cfg(test)]
mod tests;
pub mod types;

pub use metrics::{alpha_ndcg, cosine, embed, intra_list_diversity, subtopic_recall};
pub use types::{DiversityMetrics, RetrievalDiversityConfig, RetrievalDiversityError};

// ── DiversityScorer ───────────────────────────────────────────────────────────

/// Computes diversity and coverage metrics over a ranked result list.
///
/// Wraps a [`RetrievalDiversityConfig`] and exposes the three metrics
/// individually as well as a [`compute`](Self::compute) method that validates
/// its inputs and returns all three at once.
#[derive(Debug, Clone, Default)]
pub struct DiversityScorer {
    /// Diversity-scoring configuration.
    pub config: RetrievalDiversityConfig,
}

impl DiversityScorer {
    /// Create a new scorer with the given configuration.
    #[must_use]
    pub fn new(config: RetrievalDiversityConfig) -> Self {
        Self { config }
    }

    /// Intra-List Diversity over the top-`k` of `texts`.
    ///
    /// Each text is embedded with the deterministic [`embed`] function at the
    /// configured [`RetrievalDiversityConfig::dim`]; ILD is then the mean
    /// pairwise dissimilarity (`1 - cosine`) over all pairs of the top-`k`
    /// embeddings. Identical texts score `0.0`; orthogonal (disjoint-vocabulary)
    /// texts score near `1.0`. Fewer than two texts score `0.0`.
    #[must_use]
    pub fn intra_list_diversity(&self, texts: &[&str], k: usize) -> f32 {
        let embeddings: Vec<Vec<f32>> = texts.iter().map(|t| embed(t, self.config.dim)).collect();
        intra_list_diversity(&embeddings, k)
    }

    /// Subtopic recall over the top-`k` results.
    ///
    /// Returns the fraction of `total_subtopics` distinct subtopics covered by
    /// the union of the first `k` entries of `result_subtopics`. Labels at or
    /// above `total_subtopics` are ignored. Returns `0.0` when `total_subtopics`
    /// is `0`.
    #[must_use]
    pub fn subtopic_recall(
        &self,
        result_subtopics: &[Vec<usize>],
        total_subtopics: usize,
        k: usize,
    ) -> f32 {
        subtopic_recall(result_subtopics, total_subtopics, k)
    }

    /// α-nDCG over the top-`k` results.
    ///
    /// Uses the configured [`RetrievalDiversityConfig::alpha`] as the novelty
    /// discount. The number of distinct subtopics is inferred from the labels
    /// (one plus the maximum). The score is normalised by the ideal greedy
    /// ordering and lies in `[0.0, 1.0]`.
    #[must_use]
    pub fn alpha_ndcg(&self, result_subtopics: &[Vec<usize>], k: usize) -> f32 {
        alpha_ndcg(result_subtopics, self.config.alpha, k)
    }

    /// Compute all three diversity metrics at once.
    ///
    /// `texts` and `result_subtopics` describe the same ranked list and must
    /// have equal length: `texts[i]` is the content of the result whose subtopic
    /// labels are `result_subtopics[i]`.
    ///
    /// # Errors
    ///
    /// Returns [`RetrievalDiversityError::EmptyResults`] when either slice is
    /// empty, and [`RetrievalDiversityError::LengthMismatch`] when the two slices
    /// differ in length.
    pub fn compute(
        &self,
        texts: &[&str],
        result_subtopics: &[Vec<usize>],
        total_subtopics: usize,
        k: usize,
    ) -> Result<DiversityMetrics, RetrievalDiversityError> {
        if texts.is_empty() || result_subtopics.is_empty() {
            return Err(RetrievalDiversityError::EmptyResults);
        }
        if texts.len() != result_subtopics.len() {
            return Err(RetrievalDiversityError::LengthMismatch);
        }
        Ok(DiversityMetrics {
            ild: self.intra_list_diversity(texts, k),
            s_recall: self.subtopic_recall(result_subtopics, total_subtopics, k),
            alpha_ndcg: self.alpha_ndcg(result_subtopics, k),
        })
    }
}
