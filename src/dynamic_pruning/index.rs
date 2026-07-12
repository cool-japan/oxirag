//! The inverted index: posting lists, block-max metadata, BM25 scoring, and
//! the per-term / per-block upper bounds that make dynamic pruning *safe*.
//!
//! # Why this module owns its own inverted index
//!
//! Dynamic pruning is not a re-ranking trick that can be bolted onto an
//! existing scorer; it is a property of the *access path*. To skip a document
//! you must be able to (a) reach the next candidate without reading the
//! postings in between, and (b) bound, from cheap metadata alone, the score of
//! everything you are about to jump over. Neither is possible over a forward
//! index or a `Vec<(DocumentId, SparseVector)>`, which is why this module
//! builds a genuine term → posting-list structure with sorted document ids,
//! fixed-size blocks, and precomputed score bounds.
//!
//! # The bound hierarchy
//!
//! Every posting `(d, tf)` in term `t`'s list carries a *precomputed* BM25
//! contribution `s_t(d)` — an `f64` produced by [`bm25_term_score`]. Two
//! upper bounds are then derived from those very same `f64` values:
//!
//! * **Global term bound** `U_t = max_{d ∈ L_t} s_t(d)`
//!   ([`PruningPostingList::max_score`]).
//! * **Block bound** `U_{t,β} = max_{d ∈ β} s_t(d)` for each block `β` of
//!   consecutive postings ([`PruningBlock::max_score`]).
//!
//! Because each bound is a *maximum over exactly the stored values* rather
//! than an analytic over-estimate, `s_t(d) ≤ U_{t,β(d)} ≤ U_t` holds
//! **exactly in `f64`**, with no rounding slack anywhere. That is the
//! foundation the whole family of algorithms rests on: any subset `S` of the
//! query's terms bounds a document score by `Σ_{t ∈ S} U_t ≥ Σ_{t ∈ S} s_t(d)`
//! in exact arithmetic, and the only error left is the rounding of the *sums*
//! themselves — which the traversal absorbs with an explicit, strictly
//! conservative tolerance (`cursor::can_reach`).

use std::collections::HashMap;

use crate::dynamic_pruning::cursor::TermCursor;
use crate::dynamic_pruning::maxscore::search_max_score;
use crate::dynamic_pruning::types::{
    DynamicPruningError, DynamicPruningResult, PruningConfig, PruningHit, PruningSearchResult,
    PruningStats, PruningStrategy,
};
use crate::dynamic_pruning::wand::{search_block_max_wand, search_exhaustive, search_wand};

/// The document ordinal used to mean "this cursor is exhausted".
///
/// It is deliberately larger than any real ordinal, so an exhausted cursor
/// sorts to the end of a doc-id-ordered cursor array and never becomes a
/// pivot. [`DynamicPruningIndex::add_document`] refuses to mint this ordinal.
pub(crate) const EXHAUSTED: u32 = u32::MAX;

// ── BM25 ─────────────────────────────────────────────────────────────────────

/// The BM25 inverse document frequency of a term, in the "probabilistic
/// IDF with a `1 +` guard" form used by Lucene:
///
/// ```text
/// idf(t) = ln(1 + (N - df + 0.5) / (df + 0.5))
/// ```
///
/// The `1 +` keeps the value strictly positive even for a term that occurs in
/// every document, which matters here: a negative per-term contribution would
/// break the monotonicity that the upper bounds rely on (adding a term could
/// then *lower* a document's score, and `Σ U_t` would no longer bound `Σ s_t`
/// over a *subset* of terms).
#[must_use]
pub fn bm25_idf(document_count: usize, document_frequency: usize) -> f64 {
    #[allow(clippy::cast_precision_loss)]
    let n = document_count as f64;
    #[allow(clippy::cast_precision_loss)]
    let df = document_frequency as f64;
    (1.0 + (n - df + 0.5) / (df + 0.5)).ln()
}

/// A single term's BM25 contribution to a single document:
///
/// ```text
/// s_t(d) = idf(t) · tf · (k1 + 1) / ( tf + k1 · (1 - b + b · |d| / avgdl) )
/// ```
///
/// Strictly increasing in `tf` and (for `b > 0`) strictly decreasing in the
/// document length `|d|`, hence always non-negative given a non-negative
/// `idf`.
#[must_use]
pub fn bm25_term_score(
    idf: f64,
    term_frequency: u32,
    document_length: u32,
    average_document_length: f64,
    k1: f64,
    b: f64,
) -> f64 {
    let tf = f64::from(term_frequency);
    let length_ratio = if average_document_length > 0.0 {
        f64::from(document_length) / average_document_length
    } else {
        1.0
    };
    let denominator = tf + k1 * (1.0 - b + b * length_ratio);
    if denominator <= 0.0 {
        return 0.0;
    }
    idf * (tf * (k1 + 1.0)) / denominator
}

// ── PruningPosting ───────────────────────────────────────────────────────────

/// One entry of a [`PruningPostingList`]: a document, how often the term
/// occurs in it, and the term's precomputed BM25 contribution to it.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct PruningPosting {
    /// The document's dense insertion ordinal. Postings within a list are
    /// strictly ascending in this field.
    pub document_ordinal: u32,
    /// How many times the term occurs in that document. Always at least `1`.
    pub term_frequency: u32,
    /// `s_t(d)`: the term's BM25 contribution to that document. Only
    /// meaningful after [`DynamicPruningIndex::build`]; zero before.
    pub score: f64,
}

// ── PruningBlock ─────────────────────────────────────────────────────────────

/// Block-max metadata for one contiguous run of postings.
///
/// A block covers postings `[start, end)` of its list. Because the list is
/// sorted by document ordinal, the blocks partition the ordinal axis into
/// disjoint, ascending, half-open intervals: block `β` owns every document
/// ordinal in `(max_document_ordinal(β - 1), max_document_ordinal(β)]`.
///
/// That is exactly what makes a *shallow* advance sound: the first block whose
/// `max_document_ordinal` is `≥ d` is the **only** block that can possibly
/// contain `d`, so its `max_score` bounds the term's contribution to `d` — and
/// it does so without reading a single posting.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct PruningBlock {
    first_document_ordinal: u32,
    max_document_ordinal: u32,
    max_score: f64,
    start: usize,
    end: usize,
}

impl PruningBlock {
    /// The smallest document ordinal in the block.
    #[must_use]
    pub const fn first_document_ordinal(&self) -> u32 {
        self.first_document_ordinal
    }

    /// The largest document ordinal in the block. Together with the previous
    /// block's value this delimits the half-open ordinal interval the block
    /// owns.
    #[must_use]
    pub const fn max_document_ordinal(&self) -> u32 {
        self.max_document_ordinal
    }

    /// `max_{d ∈ β} s_t(d)`: the largest BM25 contribution any posting in this
    /// block makes. An exact maximum over the stored `f64` scores, never an
    /// over-estimate.
    #[must_use]
    pub const fn max_score(&self) -> f64 {
        self.max_score
    }

    /// Index of the block's first posting within the list.
    #[must_use]
    pub const fn start(&self) -> usize {
        self.start
    }

    /// One past the index of the block's last posting within the list.
    #[must_use]
    pub const fn end(&self) -> usize {
        self.end
    }

    /// How many postings the block holds.
    #[must_use]
    pub const fn len(&self) -> usize {
        self.end - self.start
    }

    /// Whether the block holds no postings. Never true for blocks produced by
    /// [`DynamicPruningIndex::build`].
    #[must_use]
    pub const fn is_empty(&self) -> bool {
        self.start == self.end
    }
}

// ── PruningPostingList ───────────────────────────────────────────────────────

/// A single term's posting list: document ordinals in strictly ascending
/// order, their term frequencies, their precomputed BM25 contributions, the
/// block-max metadata over them, and the global per-term upper bound.
///
/// Stored as parallel arrays rather than a `Vec<PruningPosting>` so that a
/// binary search over document ordinals — the inner loop of every cursor
/// skip — touches only the ordinals.
#[derive(Debug, Clone, PartialEq)]
pub struct PruningPostingList {
    document_ordinals: Vec<u32>,
    term_frequencies: Vec<u32>,
    scores: Vec<f64>,
    blocks: Vec<PruningBlock>,
    max_score: f64,
    idf: f64,
    block_size: usize,
}

impl PruningPostingList {
    fn new(block_size: usize) -> Self {
        Self {
            document_ordinals: Vec::new(),
            term_frequencies: Vec::new(),
            scores: Vec::new(),
            blocks: Vec::new(),
            max_score: 0.0,
            idf: 0.0,
            block_size,
        }
    }

    /// Number of postings in the list, i.e. the term's document frequency.
    #[must_use]
    pub fn len(&self) -> usize {
        self.document_ordinals.len()
    }

    /// Whether the list holds no postings.
    #[must_use]
    pub fn is_empty(&self) -> bool {
        self.document_ordinals.is_empty()
    }

    /// `U_t = max_{d ∈ L_t} s_t(d)`: the largest BM25 contribution this term
    /// can make to *any* document in the corpus. Zero before
    /// [`DynamicPruningIndex::build`].
    #[must_use]
    pub const fn max_score(&self) -> f64 {
        self.max_score
    }

    /// The term's BM25 inverse document frequency. Zero before
    /// [`DynamicPruningIndex::build`].
    #[must_use]
    pub const fn idf(&self) -> f64 {
        self.idf
    }

    /// The block-max metadata, one entry per `block_size` consecutive
    /// postings (the last block may be shorter). Empty before
    /// [`DynamicPruningIndex::build`].
    #[must_use]
    pub fn blocks(&self) -> &[PruningBlock] {
        &self.blocks
    }

    /// The list's document ordinals, strictly ascending.
    #[must_use]
    pub fn document_ordinals(&self) -> &[u32] {
        &self.document_ordinals
    }

    /// The list's precomputed per-posting BM25 contributions, parallel to
    /// [`document_ordinals`](PruningPostingList::document_ordinals).
    #[must_use]
    pub fn scores(&self) -> &[f64] {
        &self.scores
    }

    /// The posting at index `position`, if any.
    #[must_use]
    pub fn posting(&self, position: usize) -> Option<PruningPosting> {
        Some(PruningPosting {
            document_ordinal: *self.document_ordinals.get(position)?,
            term_frequency: self.term_frequencies[position],
            score: self.scores[position],
        })
    }

    /// Iterate the postings in ascending document order.
    pub fn iter(&self) -> impl Iterator<Item = PruningPosting> + '_ {
        (0..self.len()).filter_map(|position| self.posting(position))
    }

    /// The index of the block that owns posting `position`.
    pub(crate) const fn block_of(&self, position: usize) -> usize {
        position / self.block_size
    }

    /// Recompute every derived quantity (`idf`, per-posting scores, block
    /// metadata, global upper bound) from corpus-wide statistics.
    fn finalize(
        &mut self,
        document_count: usize,
        document_lengths: &[u32],
        average_document_length: f64,
        config: &PruningConfig,
    ) {
        self.idf = bm25_idf(document_count, self.len());
        self.scores.clear();
        self.scores.reserve(self.len());
        for (&ordinal, &term_frequency) in self
            .document_ordinals
            .iter()
            .zip(self.term_frequencies.iter())
        {
            let length = document_lengths[ordinal as usize];
            self.scores.push(bm25_term_score(
                self.idf,
                term_frequency,
                length,
                average_document_length,
                config.bm25_k1,
                config.bm25_b,
            ));
        }

        self.block_size = config.block_size;
        self.blocks.clear();
        let mut start = 0usize;
        while start < self.len() {
            let end = (start + self.block_size).min(self.len());
            // The list is ascending, so the block's largest ordinal is simply
            // its last one -- no scan required.
            let max_document_ordinal = self.document_ordinals[end - 1];
            let first_document_ordinal = self.document_ordinals[start];
            let max_score = self.scores[start..end]
                .iter()
                .copied()
                .fold(f64::NEG_INFINITY, f64::max);
            self.blocks.push(PruningBlock {
                first_document_ordinal,
                max_document_ordinal,
                max_score,
                start,
                end,
            });
            start = end;
        }

        // The global bound is the max over the *block* bounds, which is the
        // same value as the max over the postings but costs less to compute.
        self.max_score = self
            .blocks
            .iter()
            .map(PruningBlock::max_score)
            .fold(f64::NEG_INFINITY, f64::max);
        if !self.max_score.is_finite() {
            self.max_score = 0.0;
        }
    }
}

// ── DynamicPruningIndex ──────────────────────────────────────────────────────

/// A BM25 inverted index with block-max metadata, searchable with any of the
/// four [`PruningStrategy`] traversals.
///
/// # Lifecycle
///
/// 1. [`DynamicPruningIndex::new`] with a validated [`PruningConfig`].
/// 2. [`DynamicPruningIndex::add_document`] once per document. Each call
///    assigns the next dense *ordinal*, which is both the document's key in
///    every posting list and its tie-break rank.
/// 3. [`DynamicPruningIndex::build`] to finalise. Upper bounds cannot be
///    computed incrementally: `idf` depends on the corpus-wide document count
///    and the term's document frequency, and the length normaliser depends on
///    the corpus-wide average document length, so *every* posting's score (and
///    therefore every bound) shifts whenever a document is added.
/// 4. [`DynamicPruningIndex::search`]. Searching a dirty index is an
///    [`DynamicPruningError::IndexNotBuilt`] error rather than a silently
///    stale answer.
#[derive(Debug, Clone)]
pub struct DynamicPruningIndex {
    config: PruningConfig,
    lists: HashMap<String, PruningPostingList>,
    document_ids: Vec<String>,
    document_lengths: Vec<u32>,
    ordinal_of: HashMap<String, u32>,
    total_length: u64,
    average_document_length: f64,
    built: bool,
}

impl DynamicPruningIndex {
    /// Create an empty index.
    ///
    /// # Errors
    ///
    /// Propagates [`PruningConfig::validate`].
    pub fn new(config: PruningConfig) -> DynamicPruningResult<Self> {
        config.validate()?;
        Ok(Self {
            config,
            lists: HashMap::new(),
            document_ids: Vec::new(),
            document_lengths: Vec::new(),
            ordinal_of: HashMap::new(),
            total_length: 0,
            average_document_length: 0.0,
            built: false,
        })
    }

    /// Create an empty index with the default [`PruningConfig`].
    ///
    /// # Errors
    ///
    /// Cannot actually fail — the default configuration is always valid — but
    /// returns a `Result` to mirror [`DynamicPruningIndex::new`].
    pub fn with_defaults() -> DynamicPruningResult<Self> {
        Self::new(PruningConfig::default())
    }

    /// The configuration this index was built with.
    #[must_use]
    pub const fn config(&self) -> &PruningConfig {
        &self.config
    }

    /// How many documents the index holds.
    #[must_use]
    pub fn document_count(&self) -> usize {
        self.document_ids.len()
    }

    /// How many distinct terms the index holds.
    #[must_use]
    pub fn term_count(&self) -> usize {
        self.lists.len()
    }

    /// The corpus-wide average document length (in tokens). Zero for an empty
    /// index, and stale until [`DynamicPruningIndex::build`].
    #[must_use]
    pub const fn average_document_length(&self) -> f64 {
        self.average_document_length
    }

    /// Whether the index has been built since it was last modified.
    #[must_use]
    pub const fn is_built(&self) -> bool {
        self.built
    }

    /// The posting list of `term`, if the term occurs in the corpus.
    #[must_use]
    pub fn posting_list(&self, term: &str) -> Option<&PruningPostingList> {
        self.lists.get(term)
    }

    /// The document id that was assigned `ordinal`, if any.
    #[must_use]
    pub fn document_id(&self, ordinal: u32) -> Option<&str> {
        self.document_ids.get(ordinal as usize).map(String::as_str)
    }

    /// The ordinal assigned to `document_id`, if it is in the index.
    #[must_use]
    pub fn ordinal_of(&self, document_id: &str) -> Option<u32> {
        self.ordinal_of.get(document_id).copied()
    }

    /// The length, in tokens, of the document with ordinal `ordinal`.
    #[must_use]
    pub fn document_length(&self, ordinal: u32) -> Option<u32> {
        self.document_lengths.get(ordinal as usize).copied()
    }

    /// Add a document, given its identifier and its already-tokenised terms.
    ///
    /// Term frequencies are counted from `terms`; duplicates are folded into a
    /// single posting. The document's length is `terms.len()` — i.e. tokens,
    /// not distinct terms — which is what BM25's length normaliser expects.
    ///
    /// Marks the index dirty: [`DynamicPruningIndex::build`] must be called
    /// again before the next search.
    ///
    /// # Errors
    ///
    /// * [`DynamicPruningError::EmptyDocument`] if `terms` is empty.
    /// * [`DynamicPruningError::DuplicateDocumentId`] if the id is already in
    ///   the index.
    /// * [`DynamicPruningError::IndexFull`] if the index already holds
    ///   `u32::MAX` documents.
    pub fn add_document<I, T>(&mut self, document_id: &str, terms: I) -> DynamicPruningResult<u32>
    where
        I: IntoIterator<Item = T>,
        T: AsRef<str>,
    {
        if self.ordinal_of.contains_key(document_id) {
            return Err(DynamicPruningError::DuplicateDocumentId {
                document_id: document_id.to_string(),
            });
        }
        if self.document_ids.len() >= EXHAUSTED as usize {
            return Err(DynamicPruningError::IndexFull { max: EXHAUSTED - 1 });
        }

        let mut frequencies: HashMap<String, u32> = HashMap::new();
        let mut length: u32 = 0;
        for term in terms {
            let term = term.as_ref();
            if term.is_empty() {
                continue;
            }
            length = length.saturating_add(1);
            *frequencies.entry(term.to_string()).or_insert(0) += 1;
        }
        if length == 0 {
            return Err(DynamicPruningError::EmptyDocument {
                document_id: document_id.to_string(),
            });
        }

        #[allow(clippy::cast_possible_truncation)]
        let ordinal = self.document_ids.len() as u32;

        // Postings are appended in ascending ordinal order by construction:
        // ordinals are handed out densely and monotonically, and a document is
        // only ever added once. No sort is ever needed.
        let block_size = self.config.block_size;
        for (term, term_frequency) in frequencies {
            let list = self
                .lists
                .entry(term)
                .or_insert_with(|| PruningPostingList::new(block_size));
            list.document_ordinals.push(ordinal);
            list.term_frequencies.push(term_frequency);
        }

        self.document_ids.push(document_id.to_string());
        self.document_lengths.push(length);
        self.ordinal_of.insert(document_id.to_string(), ordinal);
        self.total_length += u64::from(length);
        self.built = false;
        Ok(ordinal)
    }

    /// Finalise the index: recompute every `idf`, every per-posting BM25
    /// contribution, all block-max metadata and every global per-term upper
    /// bound from the current corpus.
    ///
    /// Idempotent, and safe to interleave with further
    /// [`add_document`](DynamicPruningIndex::add_document) calls (each of which
    /// simply marks the index dirty again).
    pub fn build(&mut self) {
        #[allow(clippy::cast_precision_loss)]
        let average = if self.document_ids.is_empty() {
            0.0
        } else {
            self.total_length as f64 / self.document_ids.len() as f64
        };
        self.average_document_length = average;

        let document_count = self.document_ids.len();
        for list in self.lists.values_mut() {
            list.finalize(
                document_count,
                &self.document_lengths,
                average,
                &self.config,
            );
        }
        self.built = true;
    }

    /// Search with the configured [`PruningConfig::strategy`] and
    /// [`PruningConfig::top_k`].
    ///
    /// # Errors
    ///
    /// See [`DynamicPruningIndex::search_top_k`].
    pub fn search<T: AsRef<str>>(&self, query: &[T]) -> DynamicPruningResult<PruningSearchResult> {
        self.search_top_k(query, self.config.strategy, self.config.top_k)
    }

    /// Search with an explicit strategy and the configured
    /// [`PruningConfig::top_k`].
    ///
    /// # Errors
    ///
    /// See [`DynamicPruningIndex::search_top_k`].
    pub fn search_with<T: AsRef<str>>(
        &self,
        query: &[T],
        strategy: PruningStrategy,
    ) -> DynamicPruningResult<PruningSearchResult> {
        self.search_top_k(query, strategy, self.config.top_k)
    }

    /// Search with an explicit strategy and an explicit `top_k`.
    ///
    /// Duplicate query terms are folded into a single cursor carrying a query
    /// term frequency `qtf`, and that term's contribution to a document is
    /// `qtf · s_t(d)`. Its upper bounds are scaled by the same `qtf`, so every
    /// bound argument survives unchanged.
    ///
    /// Query terms that do not occur in the corpus are dropped: they have an
    /// empty posting list and therefore contribute nothing to any document and
    /// nothing to any upper bound.
    ///
    /// # Errors
    ///
    /// * [`DynamicPruningError::EmptyQuery`] if `query` has no non-empty terms.
    /// * [`DynamicPruningError::InvalidTopK`] if `top_k` is zero.
    /// * [`DynamicPruningError::IndexNotBuilt`] if the index has been mutated
    ///   since the last [`DynamicPruningIndex::build`].
    pub fn search_top_k<T: AsRef<str>>(
        &self,
        query: &[T],
        strategy: PruningStrategy,
        top_k: usize,
    ) -> DynamicPruningResult<PruningSearchResult> {
        if top_k == 0 {
            return Err(DynamicPruningError::InvalidTopK);
        }
        if !self.built {
            return Err(DynamicPruningError::IndexNotBuilt);
        }

        // Fold duplicate query terms, preserving first-occurrence order. That
        // order defines the *canonical summation order* used by every strategy
        // when it fully scores a document, which is what makes their f64
        // scores bit-for-bit identical (see `cursor::ScoreAccumulator`).
        let mut term_order: Vec<(&str, f64)> = Vec::new();
        let mut seen: HashMap<&str, usize> = HashMap::new();
        for term in query {
            let term = term.as_ref();
            if term.is_empty() {
                continue;
            }
            if let Some(&position) = seen.get(term) {
                term_order[position].1 += 1.0;
            } else {
                seen.insert(term, term_order.len());
                term_order.push((term, 1.0));
            }
        }
        if term_order.is_empty() {
            return Err(DynamicPruningError::EmptyQuery);
        }

        let mut cursors: Vec<TermCursor<'_>> = Vec::new();
        let mut total_postings: u64 = 0;
        for (term_index, (term, query_term_frequency)) in term_order.iter().enumerate() {
            let Some(list) = self.lists.get(*term) else {
                continue;
            };
            if list.is_empty() {
                continue;
            }
            total_postings += list.len() as u64;
            cursors.push(TermCursor::new(list, term_index, *query_term_frequency));
        }

        let mut stats = PruningStats {
            strategy,
            total_postings,
            ..PruningStats::default()
        };

        let ranked = if cursors.is_empty() {
            Vec::new()
        } else {
            let term_count = term_order.len();
            match strategy {
                PruningStrategy::Exhaustive => {
                    search_exhaustive(&mut cursors, term_count, top_k, &mut stats)
                }
                PruningStrategy::Wand => search_wand(&mut cursors, term_count, top_k, &mut stats),
                PruningStrategy::BlockMaxWand => {
                    search_block_max_wand(&mut cursors, term_count, top_k, &mut stats)
                }
                PruningStrategy::MaxScore => {
                    search_max_score(&mut cursors, term_count, top_k, &mut stats)
                }
            }
        };

        stats.postings_skipped = stats.total_postings.saturating_sub(stats.postings_scored);

        let hits = ranked
            .into_iter()
            .map(|entry| PruningHit {
                document_id: self
                    .document_ids
                    .get(entry.document_ordinal as usize)
                    .cloned()
                    .unwrap_or_default(),
                score: entry.score,
                ordinal: entry.document_ordinal,
            })
            .collect();

        Ok(PruningSearchResult { hits, stats })
    }
}
