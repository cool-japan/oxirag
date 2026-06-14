//! Noise / distractor detection and filtering over a retrieved result set.

use std::collections::HashSet;

use crate::types::SearchResult;

use super::types::{NoiseConfig, NoiseFilterError, NoiseReport, PassageAssessment};

// ── Deterministic lexical pseudo-embedding ──────────────────────────────────────

/// Compute a deterministic lexical pseudo-embedding for `text`.
///
/// Algorithm: tokenise → hash each token to a bucket with FNV-1a → accumulate
/// per-bucket counts → L2-normalise to the requested `dim`.
fn embed(text: &str, dim: usize) -> Vec<f32> {
    if dim == 0 {
        return Vec::new();
    }
    let mut buckets = vec![0.0f32; dim];
    for token in text.split(|c: char| !c.is_alphanumeric()) {
        if token.len() < 2 {
            continue;
        }
        // Deterministic hash: FNV-1a over the lowercased token bytes.
        let mut h: u64 = 14_695_981_039_346_656_037;
        for b in token.to_lowercase().as_bytes() {
            h ^= u64::from(*b);
            h = h.wrapping_mul(1_099_511_628_211);
        }
        #[allow(clippy::cast_possible_truncation)]
        let idx = (h as usize) % dim;
        buckets[idx] += 1.0;
    }
    let norm: f32 = buckets.iter().map(|x| x * x).sum::<f32>().sqrt();
    if norm > 1e-10 {
        for x in &mut buckets {
            *x /= norm;
        }
    }
    buckets
}

/// Cosine similarity between two L2-normalised, equal-length vectors.
///
/// Clamped to `[0, 1]`: negative dot products (rare with non-negative bucket
/// counts) are treated as zero similarity.
fn cosine(a: &[f32], b: &[f32]) -> f32 {
    if a.len() != b.len() || a.is_empty() {
        return 0.0;
    }
    let dot: f32 = a.iter().zip(b.iter()).map(|(x, y)| x * y).sum();
    dot.clamp(0.0, 1.0)
}

/// Lowercase token set of `text` (alphanumeric runs of length >= 2).
fn token_set(text: &str) -> HashSet<String> {
    text.split(|c: char| !c.is_alphanumeric())
        .filter(|t| t.len() >= 2)
        .map(str::to_lowercase)
        .collect()
}

/// Lexical overlap of the query within a passage.
///
/// Fraction of distinct query tokens that also appear in the passage — a coarse
/// recall-style overlap in `[0, 1]`. Superficial keyword matches inflate this
/// term, which is precisely what the consensus term and the semantic relevance
/// term are meant to counterbalance.
#[allow(clippy::cast_precision_loss)]
fn lexical_overlap(query_tokens: &HashSet<String>, passage_tokens: &HashSet<String>) -> f32 {
    if query_tokens.is_empty() {
        return 0.0;
    }
    let hits = query_tokens.intersection(passage_tokens).count() as f32;
    hits / query_tokens.len() as f32
}

/// Element-wise mean (centroid) of a non-empty set of equal-length vectors,
/// re-normalised to unit length so that [`cosine`] against it stays in range.
fn centroid(vectors: &[Vec<f32>], dim: usize) -> Vec<f32> {
    if vectors.is_empty() || dim == 0 {
        return vec![0.0f32; dim];
    }
    let mut acc = vec![0.0f32; dim];
    for v in vectors {
        if v.len() != dim {
            continue;
        }
        for (slot, value) in acc.iter_mut().zip(v.iter()) {
            *slot += *value;
        }
    }
    let norm: f32 = acc.iter().map(|x| x * x).sum::<f32>().sqrt();
    if norm > 1e-10 {
        for x in &mut acc {
            *x /= norm;
        }
    }
    acc
}

// ── NoiseFilter ─────────────────────────────────────────────────────────────────

/// Detects and removes irrelevant or distracting passages from a retrieved set.
///
/// A *distractor* is a passage with superficial lexical overlap to the query but
/// low semantic relevance, **or** a topic-drift outlier whose content diverges
/// from the consensus (centroid) of the retrieved set. Each passage is scored on
/// two axes:
///
/// * **relevance** — cosine similarity between the query and the passage,
///   blended with the query's lexical overlap inside the passage;
/// * **consensus** — cosine similarity between the passage and the centroid of
///   *all* retrieved passages.
///
/// The two are combined as
/// `(1 - consensus_weight) * relevance + consensus_weight * consensus`, and any
/// passage scoring below [`NoiseConfig::relevance_threshold`] is flagged. The
/// filter always retains at least [`NoiseConfig::keep_min`] of the
/// highest-scoring passages, even when every passage is flagged.
///
/// The filter is **pure compute**: deterministic FNV-1a embeddings, no I/O, no
/// randomness, no model.
#[derive(Debug, Clone, Default)]
pub struct NoiseFilter {
    /// Scoring configuration.
    config: NoiseConfig,
}

impl NoiseFilter {
    /// Create a new filter with the given configuration.
    #[must_use]
    pub fn new(config: NoiseConfig) -> Self {
        Self { config }
    }

    /// Access the filter's configuration.
    #[must_use]
    pub fn config(&self) -> &NoiseConfig {
        &self.config
    }

    /// Score every passage in `results` for relevance and consensus alignment.
    ///
    /// Returns one [`PassageAssessment`] per input result, in input order. An
    /// empty `results` slice yields an empty vector. The `query` is *not*
    /// validated here; use [`NoiseFilter::filter_checked`] for empty-query
    /// rejection.
    #[must_use]
    pub fn assess(&self, query: &str, results: &[SearchResult]) -> Vec<PassageAssessment> {
        if results.is_empty() {
            return Vec::new();
        }
        let dim = self.config.dim;
        let q_emb = embed(query, dim);
        let q_tokens = token_set(query);

        // Per-passage embeddings and the set centroid.
        let embeddings: Vec<Vec<f32>> = results
            .iter()
            .map(|r| embed(&r.document.content, dim))
            .collect();
        let center = centroid(&embeddings, dim);

        let cw = self.config.consensus_weight;
        results
            .iter()
            .enumerate()
            .map(|(index, result)| {
                let p_emb = &embeddings[index];
                // Semantic relevance blended with lexical overlap (mean of the two).
                let semantic = cosine(&q_emb, p_emb);
                let p_tokens = token_set(&result.document.content);
                let lexical = lexical_overlap(&q_tokens, &p_tokens);
                let relevance = ((semantic + lexical) * 0.5).clamp(0.0, 1.0);
                // Consensus alignment with the retrieved-set centroid.
                let consensus = cosine(p_emb, &center);
                let combined = ((1.0 - cw) * relevance + cw * consensus).clamp(0.0, 1.0);
                let is_noise = combined < self.config.relevance_threshold;
                PassageAssessment {
                    index,
                    relevance,
                    consensus,
                    combined,
                    is_noise,
                }
            })
            .collect()
    }

    /// Filter `results`, dropping passages flagged as noise and re-ranking the
    /// survivors `0..`.
    ///
    /// At least [`NoiseConfig::keep_min`] of the highest-`combined` passages are
    /// always retained, even when every passage is flagged. An empty input
    /// yields an empty output. The `query` is not validated; use
    /// [`NoiseFilter::filter_checked`] for empty-query rejection.
    #[must_use]
    pub fn filter(&self, query: &str, results: &[SearchResult]) -> Vec<SearchResult> {
        self.filter_with_report(query, results).0
    }

    /// Filter `results` and additionally return a [`NoiseReport`] describing the
    /// pass.
    ///
    /// The report's `assessments` are in input order; the returned results are
    /// the kept passages, re-ranked `0..`. `kept + removed` always equals the
    /// number of input results.
    #[must_use]
    pub fn filter_with_report(
        &self,
        query: &str,
        results: &[SearchResult],
    ) -> (Vec<SearchResult>, NoiseReport) {
        let assessments = self.assess(query, results);
        if results.is_empty() {
            return (
                Vec::new(),
                NoiseReport {
                    kept: 0,
                    removed: 0,
                    assessments,
                },
            );
        }

        // Indices that pass the threshold (not flagged as noise).
        let mut keep_flags: Vec<bool> = assessments.iter().map(|a| !a.is_noise).collect();

        // Enforce the keep_min floor: if too few survive, promote the
        // highest-combined passages (deterministic tie-break by index) until the
        // floor is met. keep_min is clamped to the available count.
        let floor = self.config.keep_min.min(results.len());
        let surviving = keep_flags.iter().filter(|k| **k).count();
        if surviving < floor {
            let mut order: Vec<usize> = (0..results.len()).collect();
            order.sort_by(|&a, &b| {
                assessments[b]
                    .combined
                    .partial_cmp(&assessments[a].combined)
                    .unwrap_or(std::cmp::Ordering::Equal)
                    .then_with(|| a.cmp(&b))
            });
            for &idx in &order {
                if keep_flags.iter().filter(|k| **k).count() >= floor {
                    break;
                }
                keep_flags[idx] = true;
            }
        }

        // Materialise the kept results in input order, re-ranking 0..
        let mut kept_results: Vec<SearchResult> = Vec::new();
        for (idx, result) in results.iter().enumerate() {
            if keep_flags[idx] {
                let mut keep = result.clone();
                keep.rank = kept_results.len();
                kept_results.push(keep);
            }
        }

        let kept = kept_results.len();
        let removed = results.len() - kept;
        (
            kept_results,
            NoiseReport {
                kept,
                removed,
                assessments,
            },
        )
    }

    /// Filter `results` after validating that `query` is non-empty.
    ///
    /// Behaves like [`NoiseFilter::filter`] on success.
    ///
    /// # Errors
    ///
    /// Returns [`NoiseFilterError::EmptyQuery`] when `query` is empty or
    /// whitespace only.
    pub fn filter_checked(
        &self,
        query: &str,
        results: &[SearchResult],
    ) -> Result<Vec<SearchResult>, NoiseFilterError> {
        if query.trim().is_empty() {
            return Err(NoiseFilterError::EmptyQuery);
        }
        Ok(self.filter(query, results))
    }
}
