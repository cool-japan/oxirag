//! MMR: Maximal Marginal Relevance reranking.
//!
//! Maximal Marginal Relevance (Carbonell & Goldstein 1998) iteratively builds
//! a result set that balances *relevance* to the query against *diversity*
//! among the selected documents.
//!
//! At each step the algorithm selects the candidate `d` not yet chosen that
//! maximises:
//!
//! ```text
//! MMR(d) = λ · sim(d, q) − (1−λ) · max_{d' ∈ S} sim(d, d')
//! ```
//!
//! where `q` is the query embedding, `S` is the set of already-selected
//! documents, `sim` is cosine similarity, and `λ ∈ [0, 1]` controls the
//! diversity–relevance trade-off.
//!
//! - `λ = 1.0` → pure relevance (same order as the original ranking).
//! - `λ = 0.0` → pure diversity (maximally spread documents).

use std::collections::HashMap;

use crate::types::SearchResult;

// ── Configuration ─────────────────────────────────────────────────────────────

/// Configuration for [`MmrReranker`].
#[derive(Debug, Clone)]
pub struct MmrConfig {
    /// Diversity–relevance trade-off parameter (0.0 = pure diversity, 1.0 =
    /// pure relevance).
    ///
    /// Defaults to `0.5`.
    pub lambda: f32,

    /// Number of results to return.
    ///
    /// Defaults to `5`.
    pub top_k: usize,
}

impl Default for MmrConfig {
    fn default() -> Self {
        Self {
            lambda: 0.5,
            top_k: 5,
        }
    }
}

impl MmrConfig {
    /// Set the diversity–relevance trade-off `λ`.
    #[must_use]
    pub fn with_lambda(mut self, lambda: f32) -> Self {
        self.lambda = lambda;
        self
    }

    /// Set the number of results to return.
    #[must_use]
    pub fn with_top_k(mut self, top_k: usize) -> Self {
        self.top_k = top_k;
        self
    }
}

// ── MmrReranker ───────────────────────────────────────────────────────────────

/// Reranker that implements Maximal Marginal Relevance.
///
/// # Example
///
/// ```rust
/// # #[cfg(feature = "advanced-retrieval")]
/// # {
/// use std::collections::HashMap;
/// use oxirag::advanced_retrieval::{MmrReranker, MmrConfig};
/// use oxirag::types::{Document, SearchResult};
///
/// let reranker = MmrReranker::new(MmrConfig::default().with_top_k(2));
///
/// let results = vec![
///     SearchResult::new(Document::new("doc A").with_id("a"), 0.9, 0),
///     SearchResult::new(Document::new("doc B").with_id("b"), 0.8, 1),
/// ];
///
/// let q_emb = vec![1.0_f32, 0.0];
/// let doc_embeddings: HashMap<String, Vec<f32>> = [
///     ("a".to_string(), vec![1.0_f32, 0.0]),
///     ("b".to_string(), vec![0.0_f32, 1.0]),
/// ]
/// .into_iter()
/// .collect();
///
/// let reranked = reranker.rerank(&results, &q_emb, &doc_embeddings);
/// assert_eq!(reranked.len(), 2);
/// # }
/// ```
pub struct MmrReranker {
    config: MmrConfig,
}

impl MmrReranker {
    /// Create a new [`MmrReranker`] with the given configuration.
    #[must_use]
    pub fn new(config: MmrConfig) -> Self {
        Self { config }
    }

    /// Cosine similarity between two vectors.
    ///
    /// Returns `0.0` for mismatched lengths or zero-magnitude inputs.
    #[must_use]
    fn cosine_similarity(a: &[f32], b: &[f32]) -> f32 {
        if a.len() != b.len() {
            return 0.0;
        }
        let dot: f32 = a.iter().zip(b.iter()).map(|(x, y)| x * y).sum();
        let norm_a: f32 = a.iter().map(|x| x * x).sum::<f32>().sqrt();
        let norm_b: f32 = b.iter().map(|x| x * x).sum::<f32>().sqrt();

        if norm_a == 0.0 || norm_b == 0.0 {
            0.0
        } else {
            (dot / (norm_a * norm_b)).clamp(-1.0, 1.0)
        }
    }

    /// Rerank `candidates` using the MMR algorithm.
    ///
    /// # Arguments
    ///
    /// * `candidates` — Initial ranked list from a prior retrieval step.
    /// * `query_embedding` — Dense embedding of the original query.
    /// * `doc_embeddings` — Map from document-ID string to its embedding.
    ///   Documents whose IDs are absent from this map are treated as having
    ///   no embedding; their relevance score is taken from their original
    ///   `score` field and their redundancy score is `0.0`.
    ///
    /// # Returns
    ///
    /// At most `top_k` results, re-ranked for relevance and diversity.
    #[must_use]
    pub fn rerank(
        &self,
        candidates: &[SearchResult],
        query_embedding: &[f32],
        doc_embeddings: &HashMap<String, Vec<f32>>,
    ) -> Vec<SearchResult> {
        if candidates.is_empty() {
            return Vec::new();
        }

        let lambda = self.config.lambda;
        let target_k = self.config.top_k.min(candidates.len());

        // Pre-compute relevance scores: sim(doc, query).
        let relevance: Vec<f32> = candidates
            .iter()
            .map(|sr| {
                match doc_embeddings.get(sr.document.id.as_str()) {
                    Some(emb) => Self::cosine_similarity(emb, query_embedding),
                    None => {
                        // Fall back to the original score when no embedding is available.
                        sr.score
                    }
                }
            })
            .collect();

        // Boolean mask for candidates not yet selected.
        let n = candidates.len();
        let mut selected = vec![false; n];
        let mut selected_indices: Vec<usize> = Vec::with_capacity(target_k);

        for _ in 0..target_k {
            let mut best_idx: Option<usize> = None;
            let mut best_score = f32::NEG_INFINITY;

            for i in 0..n {
                if selected[i] {
                    continue;
                }

                let rel = relevance[i];

                // Max similarity to any already-selected document.
                let max_redundancy = if selected_indices.is_empty() {
                    0.0_f32
                } else {
                    selected_indices
                        .iter()
                        .map(|&j| {
                            let emb_i = doc_embeddings.get(candidates[i].document.id.as_str());
                            let emb_j = doc_embeddings.get(candidates[j].document.id.as_str());
                            match (emb_i, emb_j) {
                                (Some(ei), Some(ej)) => Self::cosine_similarity(ei, ej),
                                _ => 0.0,
                            }
                        })
                        .fold(f32::NEG_INFINITY, f32::max)
                        .max(0.0)
                };

                let mmr_score = lambda * rel - (1.0 - lambda) * max_redundancy;

                if best_idx.is_none() || mmr_score > best_score {
                    best_score = mmr_score;
                    best_idx = Some(i);
                }
            }

            match best_idx {
                Some(idx) => {
                    selected[idx] = true;
                    selected_indices.push(idx);
                }
                None => break, // no more unselected candidates
            }
        }

        // Build output with updated ranks and scores.
        selected_indices
            .into_iter()
            .enumerate()
            .map(|(new_rank, orig_idx)| {
                let mut sr = candidates[orig_idx].clone();
                sr.rank = new_rank;
                sr
            })
            .collect()
    }
}
