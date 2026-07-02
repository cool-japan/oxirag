//! Types, configuration, and errors for the `skr` module.
//!
//! SKR — Self-Knowledge guided Retrieval augmentation (Wang et al. 2023, "Self-
//! Knowledge Guided Retrieval Augmentation for Large Language Models") — is a
//! **binary retrieve-or-not gate** driven by a memory of the model's own past
//! performance:
//!
//! - [`SkrConfig`] — kNN neighbourhood size, decision threshold, embedding
//!   dimension, pool capacity, and similarity-weighting toggle.
//! - [`SkrExemplar`] — one self-knowledge datapoint: a question paired with
//!   whether it was previously answerable *without* retrieval.
//! - [`SkrNeighbor`] — one scored nearest-neighbour match returned by a gate
//!   decision.
//! - [`SkrDecision`] — the full result of a gate decision: the chosen
//!   [`SkrRetrievalChoice`], the similarity-weighted "known" score that drove
//!   it, and the neighbours consulted.
//! - [`SkrRetrievalChoice`] — `Skip` (answer directly) or `Retrieve` (augment
//!   with retrieval).
//! - [`SkrError`] / [`SkrResult`] — the module's error type and result alias.

use thiserror::Error;

// ── SkrRetrievalChoice ───────────────────────────────────────────────────────

/// Whether a [`crate::skr::SkrGate`] decision recommends skipping retrieval
/// or performing it.
///
/// | Choice | Meaning |
/// |--------|---------|
/// | [`Skip`](SkrRetrievalChoice::Skip) | The self-knowledge pool indicates the question is answerable from parametric knowledge alone; retrieval is skipped. |
/// | [`Retrieve`](SkrRetrievalChoice::Retrieve) | The self-knowledge pool indicates the question needs retrieval augmentation (or there is not enough self-knowledge to say otherwise). |
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Default)]
pub enum SkrRetrievalChoice {
    /// Skip retrieval; answer directly from parametric knowledge.
    Skip,
    /// Perform retrieval augmentation before answering.
    #[default]
    Retrieve,
}

impl SkrRetrievalChoice {
    /// Return a stable lowercase string representation of the choice.
    #[must_use]
    pub fn as_str(self) -> &'static str {
        match self {
            Self::Skip => "skip",
            Self::Retrieve => "retrieve",
        }
    }

    /// Return `true` for [`SkrRetrievalChoice::Retrieve`].
    #[must_use]
    pub fn should_retrieve(self) -> bool {
        matches!(self, Self::Retrieve)
    }
}

impl std::fmt::Display for SkrRetrievalChoice {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.write_str(self.as_str())
    }
}

// ── SkrExemplar ───────────────────────────────────────────────────────────────

/// A single self-knowledge datapoint stored in a
/// [`SelfKnowledgePool`](crate::skr::SelfKnowledgePool).
///
/// Pairs a `question` with a label recording whether the model previously
/// answered it correctly **without** retrieval augmentation
/// (`answerable_without_retrieval = true`, a "positive"/"I know this"
/// exemplar) or needed retrieval to answer it
/// (`answerable_without_retrieval = false`, a "negative"/"I don't know this"
/// exemplar).
#[derive(Debug, Clone, PartialEq)]
pub struct SkrExemplar {
    /// The question text this exemplar records self-knowledge about.
    pub question: String,
    /// `true` when the model previously answered `question` correctly without
    /// retrieval; `false` when retrieval was needed.
    pub answerable_without_retrieval: bool,
    /// The deterministic pseudo-embedding of `question`.
    ///
    /// Left empty by default; [`SelfKnowledgePool::add_exemplar`
    /// ](crate::skr::SelfKnowledgePool::add_exemplar) computes and fills it in
    /// at insertion time (once) whenever its length does not already match
    /// the pool's configured [`SkrConfig::embedding_dim`] — mirroring the
    /// `PromptExemplar` pattern used by `uprise_retrieval`, so callers with an
    /// expensive/precise embedder can pre-populate this field via
    /// [`SkrExemplar::with_embedding`] to skip the pseudo-embedding fallback.
    pub embedding: Vec<f32>,
}

impl SkrExemplar {
    /// Create a new exemplar with no precomputed embedding.
    #[must_use]
    pub fn new(question: impl Into<String>, answerable_without_retrieval: bool) -> Self {
        Self {
            question: question.into(),
            answerable_without_retrieval,
            embedding: Vec::new(),
        }
    }

    /// Attach a precomputed embedding (builder).
    #[must_use]
    pub fn with_embedding(mut self, embedding: Vec<f32>) -> Self {
        self.embedding = embedding;
        self
    }
}

// ── SkrNeighbor ───────────────────────────────────────────────────────────────

/// One scored nearest-neighbour exemplar consulted by a
/// [`crate::skr::SkrGate`] decision.
///
/// A snapshot of the matched [`SkrExemplar`]'s label plus its similarity to
/// the query question, ordered best-first when returned as part of a
/// [`SkrDecision`].
#[derive(Debug, Clone, PartialEq)]
pub struct SkrNeighbor {
    /// The matched exemplar's question text.
    pub question: String,
    /// The matched exemplar's self-knowledge label.
    pub answerable_without_retrieval: bool,
    /// Cosine similarity in `[-1.0, 1.0]` between the query embedding and the
    /// exemplar embedding (in practice `[0.0, 1.0]`, since the FNV-1a
    /// bucket-histogram pseudo-embeddings used by default are non-negative).
    pub similarity: f32,
}

// ── SkrDecision ───────────────────────────────────────────────────────────────

/// The full result of a [`crate::skr::SkrGate::decide`] call.
#[derive(Debug, Clone, PartialEq)]
pub struct SkrDecision {
    /// The resolved retrieve-or-not choice.
    pub choice: SkrRetrievalChoice,
    /// The similarity-weighted (or, when disabled, plain-majority) "known"
    /// score in `[0.0, 1.0]` that drove `choice`: the fraction of
    /// neighbour-vote weight cast by exemplars labelled
    /// `answerable_without_retrieval = true`. Reported as `0.0` when the pool
    /// was empty and `choice` came from [`SkrConfig::default_choice`] instead
    /// of a genuine vote.
    pub known_score: f32,
    /// The nearest-neighbour exemplars consulted, best-match first. Empty
    /// exactly when the pool was empty and `choice` fell back to
    /// [`SkrConfig::default_choice`].
    pub neighbors: Vec<SkrNeighbor>,
    /// A human-readable explanation of the decision.
    pub reason: String,
}

impl SkrDecision {
    /// Return `true` when [`SkrDecision::choice`] is
    /// [`SkrRetrievalChoice::Retrieve`].
    #[must_use]
    pub fn should_retrieve(&self) -> bool {
        self.choice.should_retrieve()
    }

    /// Return `true` when this decision was produced by the empty-pool
    /// fallback rather than a genuine kNN vote (i.e. [`SkrDecision::neighbors`]
    /// is empty).
    #[must_use]
    pub fn is_fallback(&self) -> bool {
        self.neighbors.is_empty()
    }

    /// Return the single closest neighbour consulted, or `None` when
    /// [`SkrDecision::neighbors`] is empty.
    #[must_use]
    pub fn closest_neighbor(&self) -> Option<&SkrNeighbor> {
        self.neighbors.first()
    }
}

// ── SkrConfig ─────────────────────────────────────────────────────────────────

/// Configuration for [`crate::skr::SkrGate`] and
/// [`crate::skr::SelfKnowledgePool`].
///
/// # Fields at a glance
///
/// | Field | Meaning | Default |
/// |-------|---------|---------|
/// | [`k`](Self::k) | number of nearest exemplars consulted per decision | `5` |
/// | [`decision_threshold`](Self::decision_threshold) | minimum weighted "known" score to choose [`Skip`](SkrRetrievalChoice::Skip) | `0.5` |
/// | [`default_choice`](Self::default_choice) | fallback choice when the pool is empty | [`Retrieve`](SkrRetrievalChoice::Retrieve) |
/// | [`embedding_dim`](Self::embedding_dim) | dimensionality of the FNV-1a pseudo-embeddings | `128` |
/// | [`pool_cap`](Self::pool_cap) | maximum pool size before oldest-first eviction; `0` = unbounded | `500` |
/// | [`weight_by_similarity`](Self::weight_by_similarity) | weight each neighbour's vote by its similarity rather than counting it equally | `true` |
#[derive(Debug, Clone, PartialEq)]
pub struct SkrConfig {
    /// Number of nearest exemplars consulted for each gate decision. Must be
    /// greater than zero; see [`SkrConfig::validate`].
    pub k: usize,
    /// Minimum similarity-weighted "known" score (in `[0.0, 1.0]`) required to
    /// choose [`SkrRetrievalChoice::Skip`]. Scores below this choose
    /// [`SkrRetrievalChoice::Retrieve`].
    pub decision_threshold: f32,
    /// The choice returned when the self-knowledge pool is empty (no
    /// exemplars to vote with).
    pub default_choice: SkrRetrievalChoice,
    /// Dimensionality of the deterministic FNV-1a bucket-histogram
    /// pseudo-embeddings used to compare questions. Must be greater than
    /// zero; see [`SkrConfig::validate`].
    pub embedding_dim: usize,
    /// Maximum number of exemplars the pool retains. Once exceeded, the
    /// oldest exemplars are evicted first (FIFO) to make room for new ones.
    /// `0` means unbounded (no eviction ever occurs).
    pub pool_cap: usize,
    /// When `true`, each neighbour's vote is weighted by its cosine
    /// similarity to the query (closer exemplars count for more). When
    /// `false`, every one of the `k` neighbours casts an equal vote (a plain
    /// majority).
    pub weight_by_similarity: bool,
}

impl Default for SkrConfig {
    fn default() -> Self {
        Self {
            k: 5,
            decision_threshold: 0.5,
            default_choice: SkrRetrievalChoice::Retrieve,
            embedding_dim: 128,
            pool_cap: 500,
            weight_by_similarity: true,
        }
    }
}

impl SkrConfig {
    /// Create a new configuration with default values.
    #[must_use]
    pub fn new() -> Self {
        Self::default()
    }

    /// Set the neighbourhood size `k` (builder).
    #[must_use]
    pub fn with_k(mut self, k: usize) -> Self {
        self.k = k;
        self
    }

    /// Set the decision threshold (builder).
    #[must_use]
    pub fn with_decision_threshold(mut self, decision_threshold: f32) -> Self {
        self.decision_threshold = decision_threshold;
        self
    }

    /// Set the empty-pool fallback choice (builder).
    #[must_use]
    pub fn with_default_choice(mut self, default_choice: SkrRetrievalChoice) -> Self {
        self.default_choice = default_choice;
        self
    }

    /// Set the pseudo-embedding dimensionality (builder).
    #[must_use]
    pub fn with_embedding_dim(mut self, embedding_dim: usize) -> Self {
        self.embedding_dim = embedding_dim;
        self
    }

    /// Set the pool capacity; `0` means unbounded (builder).
    #[must_use]
    pub fn with_pool_cap(mut self, pool_cap: usize) -> Self {
        self.pool_cap = pool_cap;
        self
    }

    /// Set whether neighbour votes are weighted by similarity (builder).
    #[must_use]
    pub fn with_weight_by_similarity(mut self, weight_by_similarity: bool) -> Self {
        self.weight_by_similarity = weight_by_similarity;
        self
    }

    /// Validate this configuration.
    ///
    /// # Errors
    ///
    /// - [`SkrError::InvalidK`] if [`k`](Self::k) is `0`.
    /// - [`SkrError::InvalidEmbeddingDim`] if [`embedding_dim`](Self::embedding_dim)
    ///   is `0`.
    /// - [`SkrError::InvalidThreshold`] if [`decision_threshold`
    ///   ](Self::decision_threshold) is not finite or lies outside
    ///   `[0.0, 1.0]`.
    pub fn validate(&self) -> SkrResult<()> {
        if self.k == 0 {
            return Err(SkrError::InvalidK);
        }
        if self.embedding_dim == 0 {
            return Err(SkrError::InvalidEmbeddingDim);
        }
        if !self.decision_threshold.is_finite() || !(0.0..=1.0).contains(&self.decision_threshold) {
            return Err(SkrError::InvalidThreshold(self.decision_threshold));
        }
        Ok(())
    }
}

// ── SkrError / SkrResult ─────────────────────────────────────────────────────

/// Errors produced by the `skr` module.
#[derive(Debug, Error, Clone, PartialEq)]
pub enum SkrError {
    /// A question string was empty or contained only whitespace.
    #[error("question must not be empty")]
    EmptyQuestion,
    /// [`SkrConfig::k`] was `0`; a gate needs at least one neighbour to vote
    /// with.
    #[error("k (neighbor count) must be greater than zero")]
    InvalidK,
    /// [`SkrConfig::embedding_dim`] was `0`.
    #[error("embedding dimension must be greater than zero")]
    InvalidEmbeddingDim,
    /// [`SkrConfig::decision_threshold`] was not finite or fell outside
    /// `[0.0, 1.0]`.
    #[error("decision threshold must be finite and within [0.0, 1.0], got {0}")]
    InvalidThreshold(f32),
}

/// Convenient result alias for the `skr` module.
pub type SkrResult<T> = Result<T, SkrError>;
