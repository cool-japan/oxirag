//! [`SetwiseReranker`] — k-way `Setwise` comparisons (Zhuang et al. 2024,
//! "Setwise: A Setwise Approach for Effective and Highly-Efficient Zero-shot
//! Ranking with Large Language Models") driving heapsort- and
//! bubblesort-shaped comparison schedules. See the
//! [module documentation](crate::setwise_rerank) for how this differs from
//! `pairwise_rerank` and `listwise_rerank`.

use std::cmp::Ordering;
use std::collections::HashSet;

use super::types::{
    SetwiseCandidate, SetwiseComparison, SetwiseConfig, SetwiseError, SetwiseRankEntry,
    SetwiseRanking, SetwiseResult, SetwiseSortStrategy,
};

// ── deterministic lexical pseudo-embedding (house style; FNV-1a) ───────────
//
// Deliberately self-contained (mirrors `diversity_rank::embed` /
// `retrieval_diversity::embed`) rather than imported from a sibling module:
// each module in this crate owns its own small deterministic-embedding
// toolkit instead of sharing private helpers across module boundaries.

/// Compute a deterministic lexical pseudo-embedding for `text`.
///
/// Algorithm: tokenise on non-alphanumeric boundaries → hash each token of
/// length >= 2 to a bucket with FNV-1a → accumulate per-bucket counts →
/// L2-normalise to the requested `dim`. Returns an empty vector when `dim`
/// is `0`.
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

/// Cosine similarity between two equal-length vectors.
///
/// Returns `0.0` for mismatched or empty lengths and for zero-magnitude
/// inputs. The result is clamped to `[-1.0, 1.0]`.
fn cosine(a: &[f32], b: &[f32]) -> f32 {
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

/// Tokenise `text` into a lowercase set of alphanumeric fragments of length
/// >= 2, for Jaccard lexical-overlap scoring.
fn tokenize(text: &str) -> HashSet<String> {
    text.split(|c: char| !c.is_alphanumeric())
        .filter(|t| t.len() >= 2)
        .map(str::to_lowercase)
        .collect()
}

/// Jaccard overlap between a pre-tokenised query token set and `candidate`'s
/// own tokens: `|intersection| / |union|`, or `0.0` when either side is
/// empty.
fn jaccard_overlap(query_tokens: &HashSet<String>, candidate: &str) -> f32 {
    let candidate_tokens = tokenize(candidate);
    if query_tokens.is_empty() || candidate_tokens.is_empty() {
        return 0.0;
    }
    let intersection = query_tokens.intersection(&candidate_tokens).count();
    let union = query_tokens.union(&candidate_tokens).count();
    #[allow(clippy::cast_precision_loss)]
    if union == 0 {
        0.0
    } else {
        intersection as f32 / union as f32
    }
}

// ── ScoringContext ───────────────────────────────────────────────────────────

/// Bundles the per-`rerank`-call scoring inputs so the sort-strategy
/// functions don't need to thread four separate arguments through every
/// recursive helper (which would otherwise trip `clippy::too_many_arguments`).
struct ScoringContext<'a> {
    /// Pre-tokenised query, used for the lexical component.
    query_tokens: &'a HashSet<String>,
    /// Pre-computed query pseudo-embedding, used for the semantic component.
    query_embedding: &'a [f32],
    /// The full candidate slice being reranked.
    candidates: &'a [SetwiseCandidate],
    /// The active configuration (scoring weights, embedding dimension).
    config: &'a SetwiseConfig,
}

impl ScoringContext<'_> {
    /// Blended relevance score for the candidate at original-slice index
    /// `idx`: `semantic_weight * cosine(query, candidate) + lexical_weight *
    /// jaccard(query, candidate)`.
    fn relevance_score(&self, idx: usize) -> f32 {
        let content = &self.candidates[idx].content;
        let semantic = cosine(
            self.query_embedding,
            &embed(content, self.config.embedding_dim),
        );
        let lexical = jaccard_overlap(self.query_tokens, content);
        self.config.semantic_weight * semantic + self.config.lexical_weight * lexical
    }

    /// The k-way `SetwiseComparison` primitive: score every candidate named
    /// in `indices` and return them ranked best-to-worst in one call.
    ///
    /// Ties are broken by ascending original index, so the primitive is
    /// deterministic and stable regardless of how many times it is invoked.
    fn compare_set(&self, indices: &[usize]) -> SetwiseComparison {
        let scores: Vec<f32> = indices
            .iter()
            .map(|&idx| self.relevance_score(idx))
            .collect();
        let mut local_order: Vec<usize> = (0..indices.len()).collect();
        local_order.sort_by(|&a, &b| {
            scores[b]
                .partial_cmp(&scores[a])
                .unwrap_or(Ordering::Equal)
                .then(indices[a].cmp(&indices[b]))
        });
        let ranked = local_order.iter().map(|&pos| indices[pos]).collect();
        SetwiseComparison {
            indices: indices.to_vec(),
            ranked,
            scores,
        }
    }
}

// ── k-ary heapsort schedule ──────────────────────────────────────────────────

/// Sift the element at heap position `start` down into a `branching`-ary
/// max-heap of size `heap_size`, using one [`SetwiseComparison`] per step
/// (the node together with its up-to-`branching` children).
fn sift_down(
    ctx: &ScoringContext,
    heap: &mut [usize],
    start: usize,
    heap_size: usize,
    branching: usize,
    comparison_count: &mut usize,
) {
    let mut pos = start;
    loop {
        let first_child = pos * branching + 1;
        if first_child >= heap_size {
            break;
        }
        let last_child = (first_child + branching).min(heap_size);

        let mut set_positions: Vec<usize> = Vec::with_capacity(1 + last_child - first_child);
        set_positions.push(pos);
        set_positions.extend(first_child..last_child);

        let set_indices: Vec<usize> = set_positions.iter().map(|&p| heap[p]).collect();
        let comparison = ctx.compare_set(&set_indices);
        *comparison_count += 1;

        let Some(best_original) = comparison.best() else {
            break;
        };
        let Some(&best_pos) = set_positions.iter().find(|&&p| heap[p] == best_original) else {
            break;
        };

        if best_pos == pos {
            // The node already dominates its whole child set: the heap
            // property holds locally and sifting can stop.
            break;
        }
        heap.swap(pos, best_pos);
        pos = best_pos;
    }
}

/// Build a `branching`-ary max-heap over all `n` candidates, then extract
/// the maximum `top_m` times, surfacing the top-`top_m` candidates
/// (best-first) in `indices`-order. Returns the ordering and the total
/// number of [`SetwiseComparison`] calls made.
fn heapsort_top_m(
    ctx: &ScoringContext,
    top_m: usize,
    comparison_size: usize,
) -> (Vec<usize>, usize) {
    let n = ctx.candidates.len();
    let branching = comparison_size.saturating_sub(1).max(1);
    let mut heap: Vec<usize> = (0..n).collect();
    let mut comparison_count = 0usize;

    // Bottom-up build: sift down from the last internal node to the root.
    if n >= 2 {
        let last_parent = (n - 2) / branching;
        for start in (0..=last_parent).rev() {
            sift_down(ctx, &mut heap, start, n, branching, &mut comparison_count);
        }
    }

    // Repeated extract-max: swap the root (current max) to the boundary of
    // the shrinking heap, then restore the heap property in the remainder.
    let mut heap_end = n;
    let mut result: Vec<usize> = Vec::with_capacity(top_m);
    for _ in 0..top_m {
        if heap_end == 0 {
            break;
        }
        result.push(heap[0]);
        heap_end -= 1;
        heap.swap(0, heap_end);
        if heap_end > 1 {
            sift_down(
                ctx,
                &mut heap[..heap_end],
                0,
                heap_end,
                branching,
                &mut comparison_count,
            );
        }
    }

    (result, comparison_count)
}

// ── k-window bubblesort schedule ─────────────────────────────────────────────

/// Run right-to-left sweeps of `comparison_size`-wide adjacent windows over
/// `order`, each sweep bubbling its pass's best remaining candidate to the
/// front, for enough passes to guarantee the top-`top_m` prefix is correct.
/// Returns the (at-least-`top_m`-correct-prefix) ordering and the total
/// number of [`SetwiseComparison`] calls made.
///
/// # Correctness sketch
///
/// Pass `p` (0-indexed) sweeps window-start `i` from `n - k` down to `p`.
/// By induction on `i` (decreasing), after window `i` is processed,
/// position `i` holds `max(order[i..n])`: the window `[i, i+k)` contains
/// position `i+1`, which (by the outer induction on completed sweep steps)
/// already holds `max(order[i+1..n])` and therefore dominates every other
/// position in the window, so the window's true best is either position `i`
/// or `i+1` — exactly what one `SetwiseComparison` call over the window
/// finds and swaps to the front. After pass `p` completes, position `p`
/// holds the `(p+1)`-th largest element overall, so `top_m` passes correctly
/// settle positions `0..top_m`. This is the direct k-way generalisation of
/// the classic proof that `m` passes of adjacent-swap bubble sort correctly
/// place the top-`m` extrema.
fn bubblesort_top_m(
    ctx: &ScoringContext,
    top_m: usize,
    comparison_size: usize,
) -> (Vec<usize>, usize) {
    let n = ctx.candidates.len();
    let k = comparison_size.min(n.max(1));
    let mut order: Vec<usize> = (0..n).collect();
    let mut comparison_count = 0usize;
    let passes = top_m.min(n);

    let mut pass = 0usize;
    while pass < passes {
        let remaining = n - pass;
        if remaining <= k {
            // The entire unsettled suffix fits in one set: a single
            // SetwiseComparison call fully (and exactly) resolves its order,
            // which covers every remaining pass at once.
            if remaining >= 2 {
                let window_indices: Vec<usize> = order[pass..n].to_vec();
                let comparison = ctx.compare_set(&window_indices);
                comparison_count += 1;
                for (offset, &original_idx) in comparison.ranked.iter().enumerate() {
                    order[pass + offset] = original_idx;
                }
            }
            break;
        }

        // `remaining > k` here guarantees `pass <= n - k - 1 < n - k`, so this
        // inclusive descending range is always non-empty and never
        // underflows.
        for i in (pass..=(n - k)).rev() {
            let end = i + k;
            let window_indices: Vec<usize> = order[i..end].to_vec();
            let comparison = ctx.compare_set(&window_indices);
            comparison_count += 1;
            if let Some(best_original) = comparison.best()
                && let Some(rel_pos) = window_indices.iter().position(|&x| x == best_original)
            {
                order.swap(i, i + rel_pos);
            }
        }
        pass += 1;
    }

    (order, comparison_count)
}

// ── SetwiseReranker ──────────────────────────────────────────────────────────

/// Reranks a candidate corpus using k-way `Setwise` set comparisons
/// (Zhuang et al. 2024) as the sort primitive.
///
/// Configure the comparison size `k`
/// ([`SetwiseConfig::comparison_size`]) and the schedule
/// ([`SetwiseConfig::strategy`]: [`SetwiseSortStrategy::Heapsort`] or
/// [`SetwiseSortStrategy::Bubblesort`]), then call
/// [`rerank`](Self::rerank) to obtain the top-`top_m` candidates plus the
/// total number of [`SetwiseComparison`] calls used to find them.
///
/// # Example
///
/// ```
/// use oxirag::setwise_rerank::{SetwiseCandidate, SetwiseConfig, SetwiseReranker};
///
/// let config = SetwiseConfig::new().with_comparison_size(3);
/// let reranker = SetwiseReranker::new(config);
///
/// let candidates = vec![
///     SetwiseCandidate::new("d1", "rust memory ownership systems programming"),
///     SetwiseCandidate::new("d2", "banana smoothie recipe"),
///     SetwiseCandidate::new("d3", "rust borrow checker compiler"),
/// ];
///
/// let result = reranker.rerank("rust memory systems", &candidates, 2).unwrap();
/// assert_eq!(result.ranking.len(), 2);
/// ```
#[derive(Debug, Clone, Default)]
pub struct SetwiseReranker {
    /// Configuration that controls comparison size, strategy, and scoring.
    pub config: SetwiseConfig,
}

impl SetwiseReranker {
    /// Create a new reranker with the given configuration.
    #[must_use]
    pub fn new(config: SetwiseConfig) -> Self {
        Self { config }
    }

    /// Relevance score of `candidate` against `query`, blending pseudo-
    /// embedding cosine similarity (weight
    /// [`SetwiseConfig::semantic_weight`]) with lexical Jaccard overlap
    /// (weight [`SetwiseConfig::lexical_weight`]).
    ///
    /// Exposed so callers (and tests) can independently recompute the exact
    /// score any [`SetwiseComparison`] call would have used — e.g. to
    /// brute-force a full sort as an oracle to check a reranking against.
    #[must_use]
    pub fn score(&self, query: &str, candidate: &str) -> f32 {
        let query_tokens = tokenize(query);
        let query_embedding = embed(query, self.config.embedding_dim);
        let semantic = cosine(
            &query_embedding,
            &embed(candidate, self.config.embedding_dim),
        );
        let lexical = jaccard_overlap(&query_tokens, candidate);
        self.config.semantic_weight * semantic + self.config.lexical_weight * lexical
    }

    /// Perform a single k-way [`SetwiseComparison`]: rank the candidates
    /// named by `indices` (positions into `candidates`) against `query` in
    /// one call.
    ///
    /// This is the primitive both [`SetwiseSortStrategy`] schedules are
    /// built from: every heapsort sift-down step and every bubblesort
    /// window step is exactly one call shaped like this one, regardless of
    /// how many items are in the set.
    #[must_use]
    pub fn compare_set(
        &self,
        query: &str,
        candidates: &[SetwiseCandidate],
        indices: &[usize],
    ) -> SetwiseComparison {
        let query_tokens = tokenize(query);
        let query_embedding = embed(query, self.config.embedding_dim);
        let ctx = ScoringContext {
            query_tokens: &query_tokens,
            query_embedding: &query_embedding,
            candidates,
            config: &self.config,
        };
        ctx.compare_set(indices)
    }

    /// Rerank `candidates` for `query`, returning the top-`top_m` results
    /// together with the total [`SetwiseComparison`] call count.
    ///
    /// The configured [`SetwiseConfig::strategy`] determines the comparison
    /// schedule:
    ///
    /// - [`SetwiseSortStrategy::Heapsort`] builds a
    ///   `(comparison_size - 1)`-ary max-heap and extracts the maximum
    ///   `top_m` times.
    /// - [`SetwiseSortStrategy::Bubblesort`] runs `top_m` right-to-left
    ///   sweeps of `comparison_size`-wide adjacent windows.
    ///
    /// Both schedules use the same underlying [`Self::compare_set`]
    /// primitive and, on the same input, agree on the resulting top-`top_m`
    /// ordering.
    ///
    /// # Errors
    ///
    /// - [`SetwiseError::EmptyCandidates`] if `candidates` is empty.
    /// - [`SetwiseError::InvalidComparisonSize`] if
    ///   [`SetwiseConfig::comparison_size`] is less than `2`.
    /// - [`SetwiseError::InvalidTopM`] if `top_m` is `0` or exceeds
    ///   `candidates.len()`.
    pub fn rerank(
        &self,
        query: &str,
        candidates: &[SetwiseCandidate],
        top_m: usize,
    ) -> Result<SetwiseResult, SetwiseError> {
        if candidates.is_empty() {
            return Err(SetwiseError::EmptyCandidates);
        }
        let configured_k = self.config.comparison_size;
        if configured_k < 2 {
            return Err(SetwiseError::InvalidComparisonSize(configured_k));
        }
        let n = candidates.len();
        if top_m == 0 || top_m > n {
            return Err(SetwiseError::InvalidTopM {
                top_m,
                candidate_count: n,
            });
        }

        let comparison_size = configured_k.min(n);
        let query_tokens = tokenize(query);
        let query_embedding = embed(query, self.config.embedding_dim);
        let ctx = ScoringContext {
            query_tokens: &query_tokens,
            query_embedding: &query_embedding,
            candidates,
            config: &self.config,
        };

        let (order, comparison_count) = match self.config.strategy {
            SetwiseSortStrategy::Heapsort => heapsort_top_m(&ctx, top_m, comparison_size),
            SetwiseSortStrategy::Bubblesort => bubblesort_top_m(&ctx, top_m, comparison_size),
        };

        let entries: Vec<SetwiseRankEntry> = order
            .into_iter()
            .take(top_m)
            .enumerate()
            .map(|(new_rank, original_index)| SetwiseRankEntry {
                candidate: candidates[original_index].clone(),
                score: ctx.relevance_score(original_index),
                original_index,
                new_rank,
            })
            .collect();

        Ok(SetwiseResult {
            ranking: SetwiseRanking { entries },
            comparison_count,
            strategy: self.config.strategy,
            comparison_size,
        })
    }

    /// Rerank `candidates` for `query` using
    /// [`SetwiseConfig::top_m`](SetwiseConfig::top_m) as the cutoff.
    ///
    /// Convenience wrapper around [`Self::rerank`].
    ///
    /// # Errors
    ///
    /// See [`Self::rerank`].
    pub fn rerank_default(
        &self,
        query: &str,
        candidates: &[SetwiseCandidate],
    ) -> Result<SetwiseResult, SetwiseError> {
        self.rerank(query, candidates, self.config.top_m)
    }
}
