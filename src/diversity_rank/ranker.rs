//! Greedy MAP inference for a determinantal point process (DPP).

use crate::diversity_rank::types::{
    DiversityConfig, DiversityRankError, DiversitySelection, SimilarityKind,
};
use crate::types::SearchResult;

// ── Deterministic lexical pseudo-embedding ────────────────────────────────────

/// Compute a deterministic lexical pseudo-embedding for `text`.
///
/// Algorithm: tokenise on non-alphanumeric boundaries → hash each token of
/// length >= 2 to a bucket with FNV-1a → accumulate per-bucket counts →
/// L2-normalise to the requested `dim`. This mirrors the embedding scheme used
/// elsewhere in `OxiRAG` so cosine similarities are comparable across modules.
#[must_use]
pub fn embed(text: &str, dim: usize) -> Vec<f32> {
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

// ── DiversityRanker ───────────────────────────────────────────────────────────

/// A determinantal-point-process-inspired diverse subset selector.
///
/// # How it differs from MMR
///
/// Maximal Marginal Relevance (see
/// [`MmrReranker`](crate::advanced_retrieval::MmrReranker)) uses a *linear*
/// trade-off and only looks at the single most-similar selected item:
///
/// ```text
/// MMR(i) = λ · quality_i − (1 − λ) · max_{s ∈ S} sim(i, s)
/// ```
///
/// This ranker instead approximates the **log-volume** that a DPP assigns to a
/// set. The marginal gain of adding item `i` to the selected set `S` is a
/// *product* over every already-selected item:
///
/// ```text
/// gain_i = (quality_weight · quality_i)^2 · Π_{s ∈ S} (1 − sim(i, s))
/// ```
///
/// Squaring the quality mirrors the diagonal (quality²) of a DPP kernel, and
/// the product over `(1 − sim)` mirrors the off-diagonal volume shrinkage: an
/// item that is redundant with *any* selected item is penalised, and being
/// redundant with *several* compounds the penalty. This yields genuine global
/// diversity rather than the single-nearest-neighbour penalty of MMR.
///
/// # Approximation honesty
///
/// This is the standard greedy MAP heuristic for DPPs, not exact MAP inference
/// (which is NP-hard). The `(1 − sim)` product is a tractable surrogate for the
/// true determinant of the selected sub-kernel; it agrees with the determinant
/// in the limiting cases (orthogonal items → product of qualities², identical
/// items → zero volume) but is not the determinant in general. Embeddings are
/// the deterministic FNV-1a lexical pseudo-embeddings produced by [`embed`], so
/// "similarity" is lexical overlap, not semantic similarity.
///
/// # Example
///
/// ```rust
/// # #[cfg(feature = "diversity-rank")]
/// # {
/// use oxirag::diversity_rank::{DiversityRanker, DiversityConfig};
/// use oxirag::types::{Document, SearchResult};
///
/// let ranker = DiversityRanker::new(DiversityConfig::default().with_k(2));
/// let results = vec![
///     SearchResult::new(Document::new("rust async runtime tokio").with_id("a"), 0.9, 0),
///     SearchResult::new(Document::new("rust async runtime tokio").with_id("b"), 0.8, 1),
///     SearchResult::new(Document::new("python data science pandas").with_id("c"), 0.7, 2),
/// ];
/// let ranked = ranker.select(&results);
/// assert_eq!(ranked.len(), 2);
/// # }
/// ```
#[derive(Debug, Clone)]
pub struct DiversityRanker {
    /// Selection configuration.
    config: DiversityConfig,
}

impl DiversityRanker {
    /// Create a new ranker with the given configuration.
    #[must_use]
    pub fn new(config: DiversityConfig) -> Self {
        Self { config }
    }

    /// Access the ranker's configuration.
    #[must_use]
    pub fn config(&self) -> &DiversityConfig {
        &self.config
    }

    /// Cosine similarity between two equal-length vectors.
    ///
    /// Returns `0.0` for mismatched or empty lengths and for zero-magnitude
    /// inputs. The result is clamped to `[-1.0, 1.0]`.
    #[must_use]
    pub fn cosine(a: &[f32], b: &[f32]) -> f32 {
        if a.len() != b.len() || a.is_empty() {
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

    /// Similarity between two embeddings under the configured kind.
    fn similarity(&self, a: &[f32], b: &[f32]) -> f32 {
        match self.config.similarity {
            SimilarityKind::Cosine => Self::cosine(a, b),
        }
    }

    /// Select a diverse, high-quality subset of `results`.
    ///
    /// Each `result.document.content` is embedded with the deterministic
    /// [`embed`] function. Quality is taken from `result.score`; if every score
    /// is equal (e.g. all zero), a rank-based fallback is used so that the
    /// original ordering still influences quality. The greedy DPP then selects
    /// up to [`DiversityConfig::k`] items, which are returned as fresh
    /// [`SearchResult`]s re-ranked from `0`.
    ///
    /// Returns an empty vector when `results` is empty.
    #[must_use]
    pub fn select(&self, results: &[SearchResult]) -> Vec<SearchResult> {
        if results.is_empty() {
            return Vec::new();
        }

        let embeddings: Vec<Vec<f32>> = results
            .iter()
            .map(|r| embed(&r.document.content, self.config.dim))
            .collect();
        let qualities = Self::qualities_from_results(results);

        // `select_with_embeddings` only errors on empty/length-mismatch, neither
        // of which can occur here, so an empty fallback is unreachable in
        // practice.
        let selected = self
            .select_with_embeddings(&qualities, &embeddings, self.config.k)
            .unwrap_or_default();

        selected
            .into_iter()
            .enumerate()
            .map(|(new_rank, orig_idx)| {
                let mut sr = results[orig_idx].clone();
                sr.rank = new_rank;
                sr
            })
            .collect()
    }

    /// Derive per-candidate quality values from search results.
    ///
    /// When all scores are equal the scores carry no discriminating signal, so
    /// a descending rank-based fallback (`1 / (rank + 1)`) is used instead.
    ///
    /// The exact `==` comparison is deliberate: it detects whether every score
    /// is bit-identical (e.g. all `0.0`), which is precisely when the scores
    /// provide no ordering signal.
    #[allow(clippy::cast_precision_loss, clippy::float_cmp)]
    fn qualities_from_results(results: &[SearchResult]) -> Vec<f32> {
        let first = results[0].score;
        let all_equal = results.iter().all(|r| r.score == first);
        if all_equal {
            results
                .iter()
                .enumerate()
                .map(|(i, _)| 1.0 / (i as f32 + 1.0))
                .collect()
        } else {
            results.iter().map(|r| r.score).collect()
        }
    }

    /// Greedy DPP MAP inference over explicit qualities and embeddings.
    ///
    /// The first pick is the highest-quality item (ties broken by lowest
    /// index). Each subsequent pick maximises the volume-style product gain
    ///
    /// ```text
    /// gain_i = (quality_weight · quality_i)^2 · Π_{s ∈ selected} (1 − sim(i, s))
    /// ```
    ///
    /// Selection continues until `min(k, n)` items have been chosen. Once no
    /// remaining item has a positive marginal gain (all remaining items are
    /// fully redundant with the selection, or have zero quality), the diversity
    /// term is exhausted and the remaining slots are filled by descending
    /// quality (ties broken by lowest index) so the result always reaches
    /// `min(k, n)` items.
    ///
    /// # Errors
    ///
    /// Returns [`DiversityRankError::EmptyCandidates`] when `qualities` is empty
    /// and [`DiversityRankError::LengthMismatch`] when `qualities` and
    /// `embeddings` differ in length.
    pub fn select_with_embeddings(
        &self,
        qualities: &[f32],
        embeddings: &[Vec<f32>],
        k: usize,
    ) -> Result<Vec<usize>, DiversityRankError> {
        if qualities.is_empty() {
            return Err(DiversityRankError::EmptyCandidates);
        }
        if qualities.len() != embeddings.len() {
            return Err(DiversityRankError::LengthMismatch);
        }

        let n = qualities.len();
        let target = k.min(n);
        if target == 0 {
            return Ok(Vec::new());
        }

        let weight = self.config.quality_weight;
        // Weighted, non-negative quality squared — the DPP kernel diagonal.
        let weighted_sq: Vec<f32> = qualities
            .iter()
            .map(|&q| {
                let v = weight * q.max(0.0);
                v * v
            })
            .collect();

        let mut selected: Vec<usize> = Vec::with_capacity(target);
        let mut taken = vec![false; n];

        // First pick: highest quality (ties → lowest index). Scanning in
        // increasing-index order with a strict `>` keeps the lowest index on a
        // tie and needs no panic-on-empty unwrap (`n >= 1` is guaranteed above).
        let mut first = 0usize;
        for i in 1..n {
            if weighted_sq[i] > weighted_sq[first] {
                first = i;
            }
        }
        selected.push(first);
        taken[first] = true;

        // Subsequent picks: maximise the product gain.
        while selected.len() < target {
            let mut best_idx: Option<usize> = None;
            let mut best_gain = f32::NEG_INFINITY;

            for i in 0..n {
                if taken[i] {
                    continue;
                }
                // Π over the already-selected set of (1 − sim), clamped so a
                // similarity above 1 (impossible after clamping) can never make
                // the volume negative.
                let mut volume = 1.0f32;
                for &s in &selected {
                    let sim = self.similarity(&embeddings[i], &embeddings[s]);
                    volume *= (1.0 - sim).max(0.0);
                }
                let gain = weighted_sq[i] * volume;

                // Strict `>` keeps the lowest index on ties, since candidates
                // are scanned in increasing-index order.
                if best_idx.is_none() || gain > best_gain {
                    best_gain = gain;
                    best_idx = Some(i);
                }
            }

            let Some(idx) = best_idx else {
                break; // no candidates left
            };

            // When the best marginal gain is non-positive, every remaining
            // candidate is fully redundant with the selection (or has zero
            // quality), so the diversity term contributes nothing. Rather than
            // truncate below `k`, fall back to filling the slot with the
            // highest-quality remaining candidate (ties → lowest index). This
            // keeps the result size at `min(k, n)` while still preferring
            // diversity whenever any positive-gain candidate exists.
            let chosen = if best_gain > 0.0 {
                idx
            } else {
                let mut fallback = None::<usize>;
                for i in 0..n {
                    if taken[i] {
                        continue;
                    }
                    match fallback {
                        Some(f) if weighted_sq[i] > weighted_sq[f] => fallback = Some(i),
                        Some(_) => {}
                        None => fallback = Some(i),
                    }
                }
                match fallback {
                    Some(f) => f,
                    None => break,
                }
            };

            selected.push(chosen);
            taken[chosen] = true;
        }

        Ok(selected)
    }

    /// Compute the mean pairwise dissimilarity of `indices` under `embeddings`.
    ///
    /// Dissimilarity of a pair is `1 - similarity`. A set of fewer than two
    /// items has no pairs and scores `0.0`.
    #[must_use]
    #[allow(clippy::cast_precision_loss)]
    fn diversity_score(&self, indices: &[usize], embeddings: &[Vec<f32>]) -> f32 {
        if indices.len() < 2 {
            return 0.0;
        }
        let mut total = 0.0f32;
        let mut pairs = 0u32;
        for (a_pos, &a) in indices.iter().enumerate() {
            for &b in &indices[a_pos + 1..] {
                let sim = self.similarity(&embeddings[a], &embeddings[b]);
                total += 1.0 - sim;
                pairs += 1;
            }
        }
        if pairs == 0 {
            0.0
        } else {
            total / pairs as f32
        }
    }

    /// Select a diverse subset and report its diversity score.
    ///
    /// Convenience wrapper combining [`select_with_embeddings`] with the
    /// set's diversity score, returning a [`DiversitySelection`].
    ///
    /// [`select_with_embeddings`]: Self::select_with_embeddings
    ///
    /// # Errors
    ///
    /// Propagates the errors of [`select_with_embeddings`](Self::select_with_embeddings).
    pub fn select_detailed(
        &self,
        qualities: &[f32],
        embeddings: &[Vec<f32>],
        k: usize,
    ) -> Result<DiversitySelection, DiversityRankError> {
        let selected = self.select_with_embeddings(qualities, embeddings, k)?;
        let score = self.diversity_score(&selected, embeddings);
        Ok(DiversitySelection::new(selected, score))
    }
}
