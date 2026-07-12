//! Configuration, results, statistics and errors for dynamic index pruning.
//!
//! Nothing in this file performs retrieval. It defines the tunable knobs of
//! the BM25 scorer and the block layout ([`PruningConfig`]), the choice of
//! traversal algorithm ([`PruningStrategy`]), the shape of a ranked answer
//! ([`PruningHit`] / [`PruningSearchResult`]), the work-accounting record
//! that *proves* pruning actually happened ([`PruningStats`]), and the error
//! surface ([`DynamicPruningError`]).

use thiserror::Error;

// ── PruningStrategy ──────────────────────────────────────────────────────────

/// Which posting-list traversal algorithm [`crate::dynamic_pruning::DynamicPruningIndex::search`]
/// should use.
///
/// **All four variants are exact.** They are required to return the *identical*
/// ranked top-k — the same document ids, in the same order, with the same
/// scores — for every query. They differ only in how much work they are able
/// to *prove unnecessary* and therefore skip. [`PruningStrategy::Exhaustive`]
/// skips nothing and is the ground truth against which the other three are
/// tested.
///
/// See the [module documentation](crate::dynamic_pruning) for the upper-bound
/// arguments that make the skipping safe.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Default)]
pub enum PruningStrategy {
    /// Document-at-a-time full scan: every posting of every query term is
    /// read and scored, every matching document is fully evaluated.
    ///
    /// This is the ground-truth baseline. `postings_skipped` is always `0`.
    Exhaustive,
    /// Weak-AND (Broder, Carmel, Herscovici, Soffer & Zien, 2003).
    ///
    /// Cursors are kept sorted by their current document id. Per-term *global*
    /// upper bounds are accumulated in that order until the running sum can
    /// reach the current k-th best score θ; the term at which that happens is
    /// the *pivot*, and its current document is the first document that could
    /// possibly enter the top-k. Everything below the pivot document is
    /// skipped without being scored.
    Wand,
    /// Block-Max WAND (Ding & Suel, 2011).
    ///
    /// As [`PruningStrategy::Wand`], but the pivot candidate is re-checked
    /// against per-*block* upper bounds, which are far tighter than the global
    /// per-term bounds. When the block-level bound fails, an entire block of
    /// postings is jumped over rather than a single document.
    #[default]
    BlockMaxWand,
    /// `MaxScore` (Turtle & Flood, 1995).
    ///
    /// Query terms are sorted by their global upper bound and split into a
    /// *non-essential* prefix (whose upper bounds sum to less than θ, so a
    /// document appearing only in those lists cannot possibly place) and an
    /// *essential* suffix. Only the essential lists are iterated to generate
    /// candidates; the non-essential lists are probed by random access, from
    /// the largest upper bound downwards, abandoning a candidate as soon as
    /// its best remaining potential falls below θ.
    MaxScore,
}

impl PruningStrategy {
    /// Every strategy, in a stable order. Handy for exhaustively testing that
    /// they all agree.
    #[must_use]
    pub const fn all() -> [PruningStrategy; 4] {
        [
            PruningStrategy::Exhaustive,
            PruningStrategy::Wand,
            PruningStrategy::BlockMaxWand,
            PruningStrategy::MaxScore,
        ]
    }

    /// A short, stable, lower-case name for the strategy.
    #[must_use]
    pub const fn name(self) -> &'static str {
        match self {
            PruningStrategy::Exhaustive => "exhaustive",
            PruningStrategy::Wand => "wand",
            PruningStrategy::BlockMaxWand => "block_max_wand",
            PruningStrategy::MaxScore => "max_score",
        }
    }

    /// Whether this strategy is allowed to skip postings.
    ///
    /// `false` only for [`PruningStrategy::Exhaustive`].
    #[must_use]
    pub const fn prunes(self) -> bool {
        !matches!(self, PruningStrategy::Exhaustive)
    }
}

// ── PruningConfig ────────────────────────────────────────────────────────────

/// Tunable parameters of a [`crate::dynamic_pruning::DynamicPruningIndex`].
///
/// The BM25 parameters and the block size are **structural**: they are baked
/// into the per-posting scores, the per-term upper bounds and the block
/// metadata when the index is built, so changing them requires a rebuild.
/// `top_k` and `strategy` are merely the defaults used by the convenience
/// [`search`](crate::dynamic_pruning::DynamicPruningIndex::search) entry
/// point and can be overridden per query.
#[derive(Debug, Clone, PartialEq)]
pub struct PruningConfig {
    /// Number of results to return (the "k" of top-k). Must be at least `1`.
    pub top_k: usize,
    /// Number of consecutive postings per block for the block-max metadata.
    /// Must be at least `1`; typical values are 64 or 128.
    ///
    /// Smaller blocks give tighter (more selective) upper bounds at the cost
    /// of more metadata; larger blocks are cheaper but bound more loosely.
    pub block_size: usize,
    /// BM25 term-frequency saturation parameter `k1`. Must be finite and
    /// non-negative; the usual value is `1.2`.
    pub bm25_k1: f64,
    /// BM25 length-normalisation parameter `b`. Must lie in `[0, 1]`; the
    /// usual value is `0.75`.
    pub bm25_b: f64,
    /// Traversal algorithm used by
    /// [`search`](crate::dynamic_pruning::DynamicPruningIndex::search).
    pub strategy: PruningStrategy,
}

impl Default for PruningConfig {
    fn default() -> Self {
        Self {
            top_k: 10,
            block_size: 64,
            bm25_k1: 1.2,
            bm25_b: 0.75,
            strategy: PruningStrategy::BlockMaxWand,
        }
    }
}

impl PruningConfig {
    /// Validate the configuration.
    ///
    /// # Errors
    ///
    /// Returns [`DynamicPruningError::InvalidTopK`] if `top_k` is zero,
    /// [`DynamicPruningError::InvalidBlockSize`] if `block_size` is zero, and
    /// [`DynamicPruningError::InvalidBm25Parameter`] if `bm25_k1` is negative
    /// or non-finite, or if `bm25_b` is outside `[0, 1]` (or non-finite).
    pub fn validate(&self) -> Result<(), DynamicPruningError> {
        if self.top_k == 0 {
            return Err(DynamicPruningError::InvalidTopK);
        }
        if self.block_size == 0 {
            return Err(DynamicPruningError::InvalidBlockSize);
        }
        if !self.bm25_k1.is_finite() || self.bm25_k1 < 0.0 {
            return Err(DynamicPruningError::InvalidBm25Parameter {
                name: "bm25_k1",
                value: self.bm25_k1,
                reason: "must be finite and non-negative",
            });
        }
        if !self.bm25_b.is_finite() || !(0.0..=1.0).contains(&self.bm25_b) {
            return Err(DynamicPruningError::InvalidBm25Parameter {
                name: "bm25_b",
                value: self.bm25_b,
                reason: "must be finite and within [0, 1]",
            });
        }
        Ok(())
    }
}

// ── PruningHit ───────────────────────────────────────────────────────────────

/// One entry of a ranked answer.
///
/// Hits are ordered by descending [`score`](PruningHit::score); documents whose
/// scores are *bitwise* equal are ordered by ascending
/// [`ordinal`](PruningHit::ordinal) — i.e. the document added to the index
/// first wins the tie. That rule is a strict total order (ordinals are unique),
/// so the top-k is uniquely determined and every strategy must reproduce it
/// exactly.
#[derive(Debug, Clone, PartialEq)]
pub struct PruningHit {
    /// The caller-supplied document identifier.
    pub document_id: String,
    /// The document's BM25 score for the query.
    pub score: f64,
    /// The document's insertion ordinal, i.e. its internal, dense, ascending
    /// document id. This is the value used to break score ties.
    pub ordinal: u32,
}

// ── PruningStats ─────────────────────────────────────────────────────────────

/// Work-accounting for a single [`search`](crate::dynamic_pruning::DynamicPruningIndex::search)
/// call.
///
/// These counters are the *evidence* that a pruning strategy actually pruned.
/// A strategy that returned the right answer while touching every posting
/// would be a correct-but-useless implementation, and these numbers are what
/// distinguish the two.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default)]
pub struct PruningStats {
    /// The strategy that produced these numbers.
    pub strategy: PruningStrategy,
    /// The total number of postings held in the posting lists of the query's
    /// (de-duplicated) terms. This is the amount of work
    /// [`PruningStrategy::Exhaustive`] must do, and therefore the denominator
    /// against which the other strategies are measured.
    pub total_postings: u64,
    /// The number of postings whose score was actually read and accumulated
    /// into a document score.
    ///
    /// For [`PruningStrategy::Exhaustive`] this always equals
    /// [`total_postings`](PruningStats::total_postings).
    pub postings_scored: u64,
    /// `total_postings - postings_scored`: postings that were provably unable
    /// to affect the top-k and were never scored — whether they were jumped
    /// over by a cursor skip, bypassed by a whole-block skip, or simply left
    /// unread in the tail of a list when the traversal terminated early.
    ///
    /// Always `0` for [`PruningStrategy::Exhaustive`].
    pub postings_skipped: u64,
    /// The number of times a *block-level* upper bound proved that no document
    /// in the current block configuration could enter the top-k, allowing the
    /// traversal to jump past at least one whole block without reading any of
    /// its postings.
    ///
    /// Non-zero only for [`PruningStrategy::BlockMaxWand`] — it is precisely
    /// the extra pruning power that block-max metadata buys over plain
    /// [`PruningStrategy::Wand`].
    pub blocks_skipped: u64,
    /// The number of documents that survived every upper-bound check and were
    /// therefore fully scored (i.e. offered to the top-k heap, or abandoned
    /// only part-way through a `MaxScore` refinement).
    pub full_evaluations: u64,
}

impl PruningStats {
    /// The fraction of the query's postings that were skipped, in `[0, 1]`.
    ///
    /// Returns `0.0` when the query touched no postings at all.
    #[must_use]
    pub fn skip_ratio(&self) -> f64 {
        if self.total_postings == 0 {
            return 0.0;
        }
        #[allow(clippy::cast_precision_loss)]
        let ratio = self.postings_skipped as f64 / self.total_postings as f64;
        ratio
    }
}

// ── PruningSearchResult ──────────────────────────────────────────────────────

/// The full outcome of a search: the ranked hits plus the work it took.
#[derive(Debug, Clone, PartialEq)]
pub struct PruningSearchResult {
    /// The ranked top-k, best first.
    pub hits: Vec<PruningHit>,
    /// How much work the traversal did.
    pub stats: PruningStats,
}

impl PruningSearchResult {
    /// The document ids of the hits, best first.
    #[must_use]
    pub fn document_ids(&self) -> Vec<&str> {
        self.hits
            .iter()
            .map(|hit| hit.document_id.as_str())
            .collect()
    }

    /// The number of hits returned.
    #[must_use]
    pub fn len(&self) -> usize {
        self.hits.len()
    }

    /// Whether the search returned no hits at all.
    #[must_use]
    pub fn is_empty(&self) -> bool {
        self.hits.is_empty()
    }
}

// ── DynamicPruningError ──────────────────────────────────────────────────────

/// Errors produced by the `dynamic_pruning` module.
#[derive(Debug, Error, Clone, PartialEq)]
pub enum DynamicPruningError {
    /// [`PruningConfig::top_k`] (or an explicit per-query `top_k`) was zero.
    #[error("top_k must be at least 1")]
    InvalidTopK,
    /// [`PruningConfig::block_size`] was zero.
    #[error("block_size must be at least 1")]
    InvalidBlockSize,
    /// A BM25 parameter was outside its permitted range.
    #[error("invalid BM25 parameter {name} = {value}: {reason}")]
    InvalidBm25Parameter {
        /// The offending parameter's field name.
        name: &'static str,
        /// The value that was supplied.
        value: f64,
        /// Why it was rejected.
        reason: &'static str,
    },
    /// Two documents were added under the same identifier.
    #[error("duplicate document id: {document_id}")]
    DuplicateDocumentId {
        /// The identifier that was added twice.
        document_id: String,
    },
    /// A document was added with no terms at all. Such a document could never
    /// match any query, and a zero length would distort the average document
    /// length that BM25 normalises against, so it is rejected outright.
    #[error("document {document_id} has no terms")]
    EmptyDocument {
        /// The offending document's identifier.
        document_id: String,
    },
    /// The index already holds `u32::MAX` documents and cannot assign another
    /// ordinal (`u32::MAX` is reserved as the cursor-exhausted sentinel).
    #[error("index is full: at most {max} documents are supported")]
    IndexFull {
        /// The maximum supported document count.
        max: u32,
    },
    /// A search was issued with no query terms.
    #[error("query must contain at least one term")]
    EmptyQuery,
    /// A search was issued against an index that has been mutated since the
    /// last call to [`build`](crate::dynamic_pruning::DynamicPruningIndex::build).
    ///
    /// Per-term and per-block upper bounds depend on corpus-wide statistics
    /// (the document count, the average document length, and every term's
    /// document frequency), so they cannot be finalised until the corpus is.
    #[error("index has not been built since it was last modified; call build() first")]
    IndexNotBuilt,
}

/// Convenience alias for this module's fallible return type.
pub type DynamicPruningResult<T> = Result<T, DynamicPruningError>;
