//! Cross-encoder reranker orchestrator.
use crate::cross_encoder::scorer::LexicalCrossEncoder;
use crate::cross_encoder::types::{CrossEncoderConfig, CrossEncoderError, RerankedResult};
use crate::types::SearchResult;

// ── CrossEncoderReranker ──────────────────────────────────────────────────────

/// Reranks retrieval results using a cross-encoder scorer.
///
/// The underlying [`LexicalCrossEncoder`] is passed *per call* so the struct
/// remains free of lifetime parameters.
#[derive(Debug, Clone)]
pub struct CrossEncoderReranker {
    /// Configuration for this reranker.
    pub config: CrossEncoderConfig,
}

impl CrossEncoderReranker {
    /// Create a new reranker with the given config.
    #[must_use]
    pub fn new(config: CrossEncoderConfig) -> Self {
        Self { config }
    }

    /// Rerank `results` for `query`, blending original and cross-encoder scores.
    ///
    /// # Errors
    ///
    /// Returns [`CrossEncoderError::EmptyQuery`] if `query` is empty.
    /// Returns [`CrossEncoderError::EmptyCandidates`] if `results` is empty.
    pub fn rerank(
        &self,
        query: &str,
        results: &[SearchResult],
    ) -> Result<Vec<RerankedResult>, CrossEncoderError> {
        if query.trim().is_empty() {
            return Err(CrossEncoderError::EmptyQuery);
        }
        if results.is_empty() {
            return Err(CrossEncoderError::EmptyCandidates);
        }

        let encoder = LexicalCrossEncoder::new(self.config.weights.clone());
        let scored = encoder.score_all(query, results)?;

        let alpha = self.config.blend_alpha;
        let mut reranked: Vec<RerankedResult> = scored
            .into_iter()
            .filter_map(|(orig_idx, orig_score, cross_score)| {
                let fused = alpha * orig_score + (1.0 - alpha) * cross_score;
                if fused < self.config.score_threshold {
                    return None;
                }
                let doc = results[orig_idx].document.clone();
                Some(RerankedResult {
                    document: doc,
                    original_score: orig_score,
                    cross_score,
                    fused_score: fused,
                    original_rank: orig_idx,
                    new_rank: 0,
                })
            })
            .collect();

        reranked.sort_by(|a, b| {
            b.fused_score
                .partial_cmp(&a.fused_score)
                .unwrap_or(std::cmp::Ordering::Equal)
        });
        for (i, r) in reranked.iter_mut().enumerate() {
            r.new_rank = i;
        }

        if self.config.top_n > 0 {
            reranked.truncate(self.config.top_n);
        }
        Ok(reranked)
    }
}

impl Default for CrossEncoderReranker {
    fn default() -> Self {
        Self::new(CrossEncoderConfig::default())
    }
}
