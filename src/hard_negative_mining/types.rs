//! Types, configuration, and error definitions for the `hard_negative_mining`
//! module.
//!
//! `hard_negative_mining` implements ANCE-style (Xiong et al. 2020,
//! "Approximate Nearest Neighbor Negative Contrastive Learning for Dense Text
//! Retrieval") **asynchronous** hard-negative mining. The central data
//! structures are:
//!
//! - [`EmbeddingVersion`] — a deterministic pseudo-embedding *model version*.
//!   Two different versions produce two different (but each internally
//!   reproducible) vectors for the same text, standing in for "the embedding
//!   model was re-trained since the last mining round".
//! - [`HardNegativePositivePair`] — a labelled `(query, positive document)`
//!   training pair.
//! - [`HardNegativeDocument`] — one `(doc_id, text)` corpus entry.
//! - [`HardNegativeSample`] — one mined hard negative: a document that the
//!   *current* embedding version ranks highly for a query even though it is
//!   not that query's labelled positive.
//! - [`HardNegativeQueryResult`] — the per-query mining outcome (the positive's
//!   own rank plus the mined negatives).
//! - [`MiningRound`] — the full result of mining every query once under a
//!   single [`EmbeddingVersion`].
//! - [`HardNegativeStaleness`] / [`HardNegativeQueryOverlap`] — how much the
//!   mined negative sets *changed* between two rounds (the "asynchronous"
//!   drift signal).

use crate::types::DocumentId;
use thiserror::Error;

/// Convenience alias for a `Result` whose error is always a
/// [`HardNegativeError`].
pub type HardNegativeResult<T> = core::result::Result<T, HardNegativeError>;

// ── EmbeddingVersion ──────────────────────────────────────────────────────────

/// A deterministic pseudo-embedding *model version*.
///
/// The mining algorithm never trains a real model. Instead it embeds text with
/// a deterministic `FNV-1a`/`splitmix64` hash function parameterised by this
/// version, so that:
///
/// - the *same* `(text, version)` pair always yields the *same* vector
///   (reproducibility), and
/// - a *different* [`EmbeddingVersion::seed`] yields a *different* vector for
///   the same text — simulating "the embedding model changed since the last
///   mining round" without needing an actual trainable model.
///
/// Two axes are exposed:
///
/// - [`EmbeddingVersion::seed`] selects the pseudo-random *direction* of this
///   version's model-specific component. Different seeds point in unrelated
///   directions.
/// - [`EmbeddingVersion::drift`] (in `[0.0, 1.0]`) selects the *magnitude* of
///   that component relative to the shared, version-independent content
///   embedding. At `drift == 0.0` the embedding is the pure content histogram
///   (the *canonical* embedding, identical for every seed); as `drift`
///   increases the version-specific noise increasingly reorders rankings, so a
///   large `drift` change scrambles the mined negatives (high staleness) while
///   a small one barely perturbs them (low staleness).
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct EmbeddingVersion {
    /// Selects the pseudo-random direction of this version's model-specific
    /// component. Two versions with different seeds diverge in unrelated
    /// directions; two versions with the same seed and the same `drift` are
    /// identical.
    pub seed: u64,
    /// Magnitude, in `[0.0, 1.0]`, of the version-specific component relative
    /// to the shared content embedding. `0.0` is the canonical, seed-agnostic
    /// content embedding; `1.0` is pure version-specific noise. Values outside
    /// `[0.0, 1.0]` are clamped when embedding.
    pub drift: f32,
}

impl EmbeddingVersion {
    /// Create a version from an explicit `seed` and `drift`.
    #[must_use]
    pub fn new(seed: u64, drift: f32) -> Self {
        Self { seed, drift }
    }

    /// The canonical, seed-agnostic embedding version (`drift == 0.0`).
    ///
    /// Every seed collapses to the same pure content embedding at this drift,
    /// so [`EmbeddingVersion::canonical`] is the natural reference point
    /// against which drifted versions are compared.
    #[must_use]
    pub fn canonical() -> Self {
        Self {
            seed: 0,
            drift: 0.0,
        }
    }

    /// Return a copy of this version with `drift` replaced.
    #[must_use]
    pub fn with_drift(mut self, drift: f32) -> Self {
        self.drift = drift;
        self
    }

    /// Return a copy of this version with `seed` replaced.
    #[must_use]
    pub fn with_seed(mut self, seed: u64) -> Self {
        self.seed = seed;
        self
    }

    /// The effective drift used when embedding: [`EmbeddingVersion::drift`]
    /// clamped into `[0.0, 1.0]`.
    #[must_use]
    pub fn effective_drift(&self) -> f32 {
        self.drift.clamp(0.0, 1.0)
    }
}

impl Default for EmbeddingVersion {
    fn default() -> Self {
        Self::canonical()
    }
}

// ── HardNegativeDocument ──────────────────────────────────────────────────────

/// One corpus entry: a `(doc_id, text)` pair the miner ranks and mines from.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct HardNegativeDocument {
    /// The document's unique identifier.
    pub id: DocumentId,
    /// The document's text, embedded to rank it against queries.
    pub text: String,
}

impl HardNegativeDocument {
    /// Create a corpus document from an id and its text.
    #[must_use]
    pub fn new(id: impl Into<DocumentId>, text: impl Into<String>) -> Self {
        Self {
            id: id.into(),
            text: text.into(),
        }
    }
}

// ── HardNegativePositivePair ──────────────────────────────────────────────────

/// A labelled positive training pair: a `query` and the id of the single
/// document known to be its correct (positive) answer.
///
/// Hard-negative mining never emits `positive_id` as a negative for `query`
/// (subject to [`HardNegativeConfig::exclude_positive`]); every *other*
/// highly-ranked document is a candidate hard negative.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct HardNegativePositivePair {
    /// The query text.
    pub query: String,
    /// The id of the labelled positive document for `query`. Must reference a
    /// document present in the corpus.
    pub positive_id: DocumentId,
}

impl HardNegativePositivePair {
    /// Create a labelled positive pair.
    #[must_use]
    pub fn new(query: impl Into<String>, positive_id: impl Into<DocumentId>) -> Self {
        Self {
            query: query.into(),
            positive_id: positive_id.into(),
        }
    }
}

// ── HardNegativeSample ────────────────────────────────────────────────────────

/// One mined hard negative for a query.
///
/// A *hard* negative is a document the current [`EmbeddingVersion`] ranks
/// highly for the query — it looks plausible to the model — yet is not the
/// labelled positive. These are the informative negatives for contrastive
/// training, as opposed to *easy* negatives that the model already places far
/// down the ranking.
#[derive(Debug, Clone, PartialEq)]
pub struct HardNegativeSample {
    /// The id of the mined negative document.
    pub doc_id: DocumentId,
    /// The document's 1-based rank in the query's full corpus ranking (rank
    /// `1` is the most similar document). Always within
    /// [`HardNegativeConfig::hard_rank_cutoff`].
    pub rank: usize,
    /// The cosine similarity between the query and this document under the
    /// round's embedding version.
    pub similarity: f32,
}

// ── HardNegativeQueryResult ───────────────────────────────────────────────────

/// The mining outcome for a single query within a [`MiningRound`].
#[derive(Debug, Clone, PartialEq)]
pub struct HardNegativeQueryResult {
    /// The query these negatives were mined for.
    pub query: String,
    /// The id of the query's labelled positive document.
    pub positive_id: DocumentId,
    /// The 1-based rank of the labelled positive in the full corpus ranking
    /// for this query — the core retrieval-quality diagnostic (rank `1` means
    /// the model already retrieves the positive first).
    pub positive_rank: usize,
    /// The mined hard negatives, best-ranked first, never longer than
    /// [`HardNegativeConfig::negatives_per_query`].
    pub negatives: Vec<HardNegativeSample>,
}

impl HardNegativeQueryResult {
    /// The ids of the mined negatives, best-ranked first.
    #[must_use]
    pub fn negative_ids(&self) -> Vec<DocumentId> {
        self.negatives.iter().map(|s| s.doc_id.clone()).collect()
    }

    /// `true` when `doc_id` was mined as a negative for this query.
    #[must_use]
    pub fn contains_negative(&self, doc_id: &DocumentId) -> bool {
        self.negatives.iter().any(|s| &s.doc_id == doc_id)
    }
}

// ── MiningRound ───────────────────────────────────────────────────────────────

/// The complete result of mining every labelled query exactly once under one
/// [`EmbeddingVersion`].
#[derive(Debug, Clone, PartialEq)]
pub struct MiningRound {
    /// Zero-based index of this round within a miner's history.
    pub round_index: usize,
    /// The embedding version used to rank the corpus for this round.
    pub version: EmbeddingVersion,
    /// The per-query mining results, in the order the positive pairs were
    /// supplied.
    pub results: Vec<HardNegativeQueryResult>,
    /// The mean, over all queries, of the labelled positive's 1-based rank —
    /// the round's aggregate retrieval-quality diagnostic. Lower is better.
    /// `0.0` when there are no queries.
    pub mean_positive_rank: f32,
}

impl MiningRound {
    /// The total number of hard negatives mined across every query this round.
    #[must_use]
    pub fn total_negatives(&self) -> usize {
        self.results.iter().map(|r| r.negatives.len()).sum()
    }

    /// The per-query result for `query`, if present.
    #[must_use]
    pub fn query_result(&self, query: &str) -> Option<&HardNegativeQueryResult> {
        self.results.iter().find(|r| r.query == query)
    }

    /// The mined negative ids for `query`, if the query is present.
    #[must_use]
    pub fn negative_ids(&self, query: &str) -> Option<Vec<DocumentId>> {
        self.query_result(query)
            .map(HardNegativeQueryResult::negative_ids)
    }
}

// ── HardNegativeQueryOverlap ──────────────────────────────────────────────────

/// How much one query's mined negative set changed between two rounds.
///
/// The Jaccard overlap is `|prev ∩ cur| / |prev ∪ cur|`: `1.0` when the two
/// rounds mined exactly the same negatives (fully fresh), `0.0` when they share
/// none (fully stale). Two empty sets are defined to overlap perfectly
/// (`1.0`).
#[derive(Debug, Clone, PartialEq)]
pub struct HardNegativeQueryOverlap {
    /// The query whose negative sets are being compared.
    pub query: String,
    /// Jaccard overlap of the previous and current negative id sets, in
    /// `[0.0, 1.0]`.
    pub jaccard: f32,
    /// Number of negatives present in *both* rounds (`|prev ∩ cur|`).
    pub retained: usize,
    /// Number of negatives newly introduced this round (`|cur \ prev|`).
    pub added: usize,
    /// Number of negatives dropped since the previous round (`|prev \ cur|`).
    pub removed: usize,
}

// ── HardNegativeStaleness ─────────────────────────────────────────────────────

/// The drift between two [`MiningRound`]s — the signal that makes this miner
/// *asynchronous*.
///
/// As the embedding version evolves, previously-mined negatives go stale: what
/// the old model ranked highly, the new model may not. This report measures
/// exactly that, per query and in aggregate, rather than silently re-mining.
#[derive(Debug, Clone, PartialEq)]
pub struct HardNegativeStaleness {
    /// Index of the earlier round being compared against.
    pub previous_round_index: usize,
    /// Index of the later round.
    pub current_round_index: usize,
    /// The earlier round's embedding version.
    pub previous_version: EmbeddingVersion,
    /// The later round's embedding version.
    pub current_version: EmbeddingVersion,
    /// Per-query overlap between the two rounds' mined negative sets.
    pub per_query: Vec<HardNegativeQueryOverlap>,
    /// Mean of [`HardNegativeQueryOverlap::jaccard`] across all queries, in
    /// `[0.0, 1.0]`. `1.0` means every query mined an identical negative set
    /// (no drift); lower values mean the sets diverged. `1.0` when there are no
    /// queries.
    pub mean_jaccard: f32,
    /// `current.mean_positive_rank - previous.mean_positive_rank`: how the
    /// aggregate positive rank moved between the two versions (negative means
    /// the positive is now ranked *better*).
    pub mean_positive_rank_delta: f32,
}

impl HardNegativeStaleness {
    /// The staleness score `1.0 - mean_jaccard`: `0.0` when nothing changed,
    /// `1.0` when the negative sets fully turned over.
    #[must_use]
    pub fn staleness(&self) -> f32 {
        1.0 - self.mean_jaccard
    }

    /// `true` when every query's mined negative set is byte-for-byte unchanged
    /// (`mean_jaccard == 1.0`).
    #[must_use]
    pub fn is_unchanged(&self) -> bool {
        (self.mean_jaccard - 1.0).abs() <= f32::EPSILON
    }
}

// ── HardNegativeConfig ────────────────────────────────────────────────────────

/// Configuration for [`crate::hard_negative_mining::HardNegativeMiner`].
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct HardNegativeConfig {
    /// The per-query negatives budget `n`: at most this many hard negatives are
    /// mined for each query. Default `5`.
    pub negatives_per_query: usize,
    /// The top-K cutoff: a document must rank within the top
    /// `hard_rank_cutoff` positions of a query's corpus ranking (1-based) to
    /// qualify as a *hard* negative. Documents ranked below this are *easy*
    /// negatives and are never mined, however much budget remains. Default
    /// `10`.
    pub hard_rank_cutoff: usize,
    /// The pseudo-embedding dimensionality. Larger values reduce hash-bucket
    /// collisions between distinct tokens. Default `64`.
    pub embedding_dim: usize,
    /// When `true` (the default), the labelled positive is guaranteed never to
    /// be mined as a negative for its own query, even if the current version
    /// ranks it highly. When `false`, the positive may appear among the
    /// negatives if it falls within the cutoff — useful only for diagnostics.
    pub exclude_positive: bool,
}

impl Default for HardNegativeConfig {
    fn default() -> Self {
        Self {
            negatives_per_query: 5,
            hard_rank_cutoff: 10,
            embedding_dim: 64,
            exclude_positive: true,
        }
    }
}

impl HardNegativeConfig {
    /// Create a configuration with default values.
    #[must_use]
    pub fn new() -> Self {
        Self::default()
    }

    /// Set the per-query negatives budget `n`.
    #[must_use]
    pub fn with_negatives_per_query(mut self, value: usize) -> Self {
        self.negatives_per_query = value;
        self
    }

    /// Set the top-K cutoff within which a negative must rank to count as hard.
    #[must_use]
    pub fn with_hard_rank_cutoff(mut self, value: usize) -> Self {
        self.hard_rank_cutoff = value;
        self
    }

    /// Set the pseudo-embedding dimensionality.
    #[must_use]
    pub fn with_embedding_dim(mut self, value: usize) -> Self {
        self.embedding_dim = value;
        self
    }

    /// Set whether the labelled positive is excluded from its own negatives.
    #[must_use]
    pub fn with_exclude_positive(mut self, value: bool) -> Self {
        self.exclude_positive = value;
        self
    }

    /// Validate the configuration.
    ///
    /// # Errors
    ///
    /// - [`HardNegativeError::ZeroNegativesPerQuery`] if `negatives_per_query`
    ///   is `0`.
    /// - [`HardNegativeError::ZeroHardRankCutoff`] if `hard_rank_cutoff` is
    ///   `0`.
    /// - [`HardNegativeError::ZeroEmbeddingDim`] if `embedding_dim` is `0`.
    pub fn validate(&self) -> HardNegativeResult<()> {
        if self.negatives_per_query == 0 {
            return Err(HardNegativeError::ZeroNegativesPerQuery);
        }
        if self.hard_rank_cutoff == 0 {
            return Err(HardNegativeError::ZeroHardRankCutoff);
        }
        if self.embedding_dim == 0 {
            return Err(HardNegativeError::ZeroEmbeddingDim);
        }
        Ok(())
    }
}

// ── HardNegativeError ─────────────────────────────────────────────────────────

/// Errors produced by the `hard_negative_mining` module.
#[derive(Debug, Error, PartialEq, Eq)]
pub enum HardNegativeError {
    /// The corpus contained no documents.
    #[error("corpus is empty")]
    EmptyCorpus,
    /// No labelled positive pairs were supplied.
    #[error("no labelled positive pairs were provided")]
    EmptyPositives,
    /// A positive pair had an empty (or whitespace-only) query.
    #[error("a positive pair has an empty query")]
    EmptyQuery,
    /// `negatives_per_query` was `0`.
    #[error("negatives_per_query must be greater than zero")]
    ZeroNegativesPerQuery,
    /// `hard_rank_cutoff` was `0`.
    #[error("hard_rank_cutoff must be greater than zero")]
    ZeroHardRankCutoff,
    /// `embedding_dim` was `0`.
    #[error("embedding_dim must be greater than zero")]
    ZeroEmbeddingDim,
    /// The corpus contained two documents sharing the same id.
    #[error("corpus contains a duplicate document id: {0}")]
    DuplicateDocumentId(DocumentId),
    /// A positive pair referenced a document id absent from the corpus.
    #[error("positive pair references unknown document id: {0}")]
    UnknownPositiveDocument(DocumentId),
    /// [`crate::hard_negative_mining::HardNegativeMiner::refresh`] was called
    /// with no prior round to compare against.
    #[error("no prior mining round is available to compute staleness against")]
    NoPriorRound,
}
