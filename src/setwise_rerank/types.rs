//! Types for the `setwise_rerank` module: configuration, the k-way
//! comparison primitive's data, candidate/ranking records, and errors.

use thiserror::Error;

// ── SetwiseSortStrategy ──────────────────────────────────────────────────────

/// Which classical sort algorithm supplies the *shape* of the comparison
/// schedule built on top of the k-way [`SetwiseComparison`] primitive.
///
/// Both strategies replace the usual "compare two items" step of a textbook
/// sort with "rank a *set* of up to `comparison_size` items in one call" —
/// see the [module documentation](crate::setwise_rerank) for the full
/// rationale and how this differs from `pairwise_rerank` and
/// `listwise_rerank`.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default)]
pub enum SetwiseSortStrategy {
    /// A k-ary max-heap: each sift-down step ranks a node together with its
    /// (up to `comparison_size - 1`) children in a single
    /// [`SetwiseComparison`] call, promoting the best of that set into the
    /// node's slot. Repeated extract-max operations surface the top-`top_m`
    /// candidates in roughly `O(top_m · log n)` set comparisons.
    #[default]
    Heapsort,
    /// A sliding-window generalisation of bubble sort: each step ranks a
    /// window of `comparison_size` adjacent candidates in a single
    /// [`SetwiseComparison`] call and swaps that window's best entry to the
    /// front. `top_m` right-to-left sweeps guarantee the top-`top_m`
    /// candidates end up correctly ordered, exactly mirroring how `top_m`
    /// passes of classic adjacent-swap bubble sort guarantee the top-`top_m`
    /// extrema are correctly placed.
    Bubblesort,
}

impl SetwiseSortStrategy {
    /// Stable, lowercase name of the strategy.
    #[must_use]
    pub fn as_str(&self) -> &'static str {
        match self {
            Self::Heapsort => "heapsort",
            Self::Bubblesort => "bubblesort",
        }
    }
}

// ── SetwiseConfig ────────────────────────────────────────────────────────────

/// Configuration for [`crate::setwise_rerank::SetwiseReranker`].
///
/// # Examples
///
/// ```
/// use oxirag::setwise_rerank::{SetwiseConfig, SetwiseSortStrategy};
///
/// let cfg = SetwiseConfig::new()
///     .with_comparison_size(4)
///     .with_strategy(SetwiseSortStrategy::Bubblesort)
///     .with_top_m(5);
/// assert_eq!(cfg.comparison_size, 4);
/// assert_eq!(cfg.strategy, SetwiseSortStrategy::Bubblesort);
/// assert_eq!(cfg.top_m, 5);
/// ```
#[derive(Debug, Clone, PartialEq)]
pub struct SetwiseConfig {
    /// The size `k` of the candidate set ranked by a single
    /// [`SetwiseComparison`] call. Must be at least `2` — a "set" of size
    /// `1` carries no ordering information, so
    /// [`SetwiseReranker::rerank`](crate::setwise_rerank::SetwiseReranker::rerank)
    /// rejects `comparison_size < 2` with
    /// [`SetwiseError::InvalidComparisonSize`]. Defaults to `4`.
    pub comparison_size: usize,
    /// Which sort strategy drives the comparison schedule. Defaults to
    /// [`SetwiseSortStrategy::Heapsort`].
    pub strategy: SetwiseSortStrategy,
    /// Default number of top candidates to surface, used by
    /// [`SetwiseReranker::rerank_default`](crate::setwise_rerank::SetwiseReranker::rerank_default).
    /// Defaults to `10`.
    pub top_m: usize,
    /// Blend weight applied to the pseudo-embedding cosine-similarity
    /// ("semantic") component of the relevance score. Defaults to `0.6`.
    pub semantic_weight: f32,
    /// Blend weight applied to the lexical (token Jaccard overlap) component
    /// of the relevance score. Defaults to `0.4`.
    pub lexical_weight: f32,
    /// Dimensionality of the deterministic FNV-1a lexical pseudo-embeddings
    /// used for the semantic component. Defaults to `128`.
    pub embedding_dim: usize,
}

impl Default for SetwiseConfig {
    fn default() -> Self {
        Self {
            comparison_size: 4,
            strategy: SetwiseSortStrategy::Heapsort,
            top_m: 10,
            semantic_weight: 0.6,
            lexical_weight: 0.4,
            embedding_dim: 128,
        }
    }
}

impl SetwiseConfig {
    /// Create a new configuration with default values.
    #[must_use]
    pub fn new() -> Self {
        Self::default()
    }

    /// Set the k-way comparison size.
    #[must_use]
    pub fn with_comparison_size(mut self, comparison_size: usize) -> Self {
        self.comparison_size = comparison_size;
        self
    }

    /// Set the sort strategy.
    #[must_use]
    pub fn with_strategy(mut self, strategy: SetwiseSortStrategy) -> Self {
        self.strategy = strategy;
        self
    }

    /// Set the default number of top candidates to surface.
    #[must_use]
    pub fn with_top_m(mut self, top_m: usize) -> Self {
        self.top_m = top_m;
        self
    }

    /// Set the semantic (cosine-similarity) blend weight.
    #[must_use]
    pub fn with_semantic_weight(mut self, semantic_weight: f32) -> Self {
        self.semantic_weight = semantic_weight;
        self
    }

    /// Set the lexical (Jaccard-overlap) blend weight.
    #[must_use]
    pub fn with_lexical_weight(mut self, lexical_weight: f32) -> Self {
        self.lexical_weight = lexical_weight;
        self
    }

    /// Set the pseudo-embedding dimensionality.
    #[must_use]
    pub fn with_embedding_dim(mut self, embedding_dim: usize) -> Self {
        self.embedding_dim = embedding_dim;
        self
    }
}

// ── SetwiseCandidate ─────────────────────────────────────────────────────────

/// A single candidate document to be ranked by [`crate::setwise_rerank::SetwiseReranker`].
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct SetwiseCandidate {
    /// Caller-supplied identifier for this candidate.
    pub id: String,
    /// Full text content of the candidate, scored against the query.
    pub content: String,
}

impl SetwiseCandidate {
    /// Create a new candidate from an id and its text content.
    #[must_use]
    pub fn new(id: impl Into<String>, content: impl Into<String>) -> Self {
        Self {
            id: id.into(),
            content: content.into(),
        }
    }
}

// ── SetwiseComparison ────────────────────────────────────────────────────────

/// One invocation of the k-way `Setwise` comparison primitive.
///
/// Given a query and a *set* of candidate indices (positions into the
/// original candidate slice), a single call ranks every member of that set
/// in one shot — the single operation that both [`SetwiseSortStrategy`]
/// schedules are built from. This mirrors the real Setwise algorithm
/// (Zhuang et al. 2024), where one LLM call is prompted with the query and
/// all `k` candidates together and returns either the single best candidate
/// or a full ordering of the set; here, since there is no live LLM, that
/// call is answered deterministically by a lexical/pseudo-embedding scorer
/// (see [`crate::setwise_rerank::SetwiseReranker::score`]).
#[derive(Debug, Clone, PartialEq)]
pub struct SetwiseComparison {
    /// Original-slice indices of the candidates that were compared in this
    /// call, in the order they were presented to the comparison.
    pub indices: Vec<usize>,
    /// `indices`, reordered best-to-worst by this one comparison call. A
    /// permutation of `indices`.
    pub ranked: Vec<usize>,
    /// Relevance score computed for each entry of `indices`, aligned by
    /// position (i.e. `scores[i]` is the score of `indices[i]`, *not*
    /// `ranked[i]`).
    pub scores: Vec<f32>,
}

impl SetwiseComparison {
    /// The best (highest-relevance) original-slice index from this
    /// comparison, i.e. `ranked[0]`.
    #[must_use]
    pub fn best(&self) -> Option<usize> {
        self.ranked.first().copied()
    }

    /// Number of candidates that were part of this comparison's set.
    #[must_use]
    pub fn len(&self) -> usize {
        self.indices.len()
    }

    /// `true` when this comparison's set was empty.
    #[must_use]
    pub fn is_empty(&self) -> bool {
        self.indices.is_empty()
    }
}

// ── SetwiseRankEntry / SetwiseRanking ────────────────────────────────────────

/// A single entry in a [`SetwiseRanking`]: a candidate together with its
/// relevance score and its position before and after reranking.
#[derive(Debug, Clone, PartialEq)]
pub struct SetwiseRankEntry {
    /// The ranked candidate.
    pub candidate: SetwiseCandidate,
    /// Relevance score assigned by
    /// [`SetwiseReranker::score`](crate::setwise_rerank::SetwiseReranker::score).
    pub score: f32,
    /// This candidate's index in the original input slice.
    pub original_index: usize,
    /// This candidate's 0-based rank after reranking (`0` = most relevant).
    pub new_rank: usize,
}

/// The final, best-first ordering produced by
/// [`crate::setwise_rerank::SetwiseReranker`], truncated to the requested
/// `top_m`.
#[derive(Debug, Clone, PartialEq, Default)]
pub struct SetwiseRanking {
    /// Ranked entries, best (`new_rank == 0`) first.
    pub entries: Vec<SetwiseRankEntry>,
}

impl SetwiseRanking {
    /// Number of entries in this ranking.
    #[must_use]
    pub fn len(&self) -> usize {
        self.entries.len()
    }

    /// `true` when this ranking has no entries.
    #[must_use]
    pub fn is_empty(&self) -> bool {
        self.entries.is_empty()
    }

    /// Candidate ids in ranked (best-first) order.
    #[must_use]
    pub fn ids(&self) -> Vec<&str> {
        self.entries
            .iter()
            .map(|e| e.candidate.id.as_str())
            .collect()
    }
}

// ── SetwiseResult ────────────────────────────────────────────────────────────

/// The complete output of a
/// [`SetwiseReranker::rerank`](crate::setwise_rerank::SetwiseReranker::rerank)
/// call: the reranked [`SetwiseRanking`] together with bookkeeping that
/// demonstrates the efficiency of the k-way comparison schedule.
#[derive(Debug, Clone, PartialEq)]
pub struct SetwiseResult {
    /// The reranked, best-first top-`top_m` candidates.
    pub ranking: SetwiseRanking,
    /// Total number of [`SetwiseComparison`] calls made to produce this
    /// result. For `n` candidates this is expected to be far below the
    /// `n · (n − 1) / 2` calls a full pairwise round-robin tournament would
    /// require.
    pub comparison_count: usize,
    /// The sort strategy that produced this result.
    pub strategy: SetwiseSortStrategy,
    /// The effective comparison-set size actually used (the configured
    /// `comparison_size`, clamped to the candidate count).
    pub comparison_size: usize,
}

// ── SetwiseError ─────────────────────────────────────────────────────────────

/// Errors from the `setwise_rerank` module.
#[derive(Debug, Error, Clone, PartialEq, Eq)]
pub enum SetwiseError {
    /// The candidate slice was empty; nothing to rank.
    #[error("candidates must not be empty")]
    EmptyCandidates,
    /// [`SetwiseConfig::comparison_size`] was less than `2`.
    #[error("comparison_size must be at least 2, got {0}")]
    InvalidComparisonSize(usize),
    /// `top_m` was `0`, or greater than the number of supplied candidates.
    #[error(
        "top_m must be at least 1 and at most the candidate count ({candidate_count}), got {top_m}"
    )]
    InvalidTopM {
        /// The requested `top_m`.
        top_m: usize,
        /// The number of candidates that were supplied.
        candidate_count: usize,
    },
}
