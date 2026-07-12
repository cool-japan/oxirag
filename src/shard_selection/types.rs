//! Types and data structures for the `shard_selection` module: lightweight
//! per-shard resource descriptions, CORI scoring configuration, ranked
//! results, and errors.
//!
//! Nothing in this file touches an actual document, embedding, or query
//! result — every type here describes either a shard's *summary statistics*
//! ([`ShardDescriptor`], [`ShardTermStats`]), the tunable constants of the
//! CORI belief formula ([`ShardSelectionConfig`]), or the ranked output of a
//! selection run ([`ShardSelectionScore`]).

use std::collections::HashMap;

use thiserror::Error;

// ── ShardTermStats ───────────────────────────────────────────────────────────

/// A shard's published term-level "resource description": for every term the
/// shard chooses to summarize, how many of its documents contain that term
/// at least once, and how many times the term occurs in total across the
/// shard's documents.
///
/// This is the lightweight, document-free summary a real distributed-IR
/// participant would gossip or publish about itself — enough for a broker to
/// judge relevance without ever seeing (or paying the network/compute cost
/// of querying) the shard's actual documents.
///
/// Each entry is stored as `(document_frequency, collection_term_frequency)`:
///
/// - `document_frequency` — the number of documents *within this shard*
///   that contain the term at least once (bounded above by the shard's
///   `doc_count`).
/// - `collection_term_frequency` — the total number of occurrences of the
///   term summed across all of the shard's documents (i.e. term frequency
///   summed over the whole shard, not capped at one per document).
///
/// Terms not present as a key are treated as having a document frequency of
/// `0` — i.e. this shard has no evidence of containing them.
#[derive(Debug, Clone, Default, PartialEq)]
pub struct ShardTermStats {
    entries: HashMap<String, (u64, u64)>,
}

impl ShardTermStats {
    /// Create an empty term-statistics map.
    #[must_use]
    pub fn new() -> Self {
        Self {
            entries: HashMap::new(),
        }
    }

    /// Record (or overwrite) the statistics for `term` and return `self` for
    /// chaining.
    #[must_use]
    pub fn with_term(
        mut self,
        term: impl Into<String>,
        document_frequency: u64,
        collection_term_frequency: u64,
    ) -> Self {
        self.entries
            .insert(term.into(), (document_frequency, collection_term_frequency));
        self
    }

    /// The number of documents within this shard that contain `term` at
    /// least once, or `0` if the term was never recorded for this shard.
    #[must_use]
    pub fn document_frequency(&self, term: &str) -> u64 {
        self.entries.get(term).map_or(0, |(df, _)| *df)
    }

    /// The total number of occurrences of `term` across this shard's
    /// documents, or `0` if the term was never recorded for this shard.
    #[must_use]
    pub fn collection_term_frequency(&self, term: &str) -> u64 {
        self.entries.get(term).map_or(0, |(_, ctf)| *ctf)
    }

    /// Return `true` when `term` has a recorded entry for this shard (even
    /// if its recorded document frequency happens to be `0`).
    #[must_use]
    pub fn contains_term(&self, term: &str) -> bool {
        self.entries.contains_key(term)
    }

    /// The number of distinct terms this shard has published statistics for.
    #[must_use]
    pub fn len(&self) -> usize {
        self.entries.len()
    }

    /// Return `true` when no term statistics have been recorded.
    #[must_use]
    pub fn is_empty(&self) -> bool {
        self.entries.is_empty()
    }

    /// Iterate over every term this shard has published statistics for, in
    /// unspecified order.
    pub fn terms(&self) -> impl Iterator<Item = &str> + '_ {
        self.entries.keys().map(String::as_str)
    }
}

// ── ShardDescriptor ──────────────────────────────────────────────────────────

/// A shard's lightweight, document-free "resource description": its id,
/// coarse size statistics, and published [`ShardTermStats`].
///
/// This is the *only* per-shard input [`crate::shard_selection::ShardSelector`]
/// consumes. It deliberately excludes anything that would require touching
/// the shard's actual documents — no embeddings, no passages, no scores.
#[derive(Debug, Clone, PartialEq)]
pub struct ShardDescriptor {
    /// The shard's unique identifier within the candidate set being
    /// considered by a single [`crate::shard_selection::ShardSelector::select`]
    /// call.
    pub shard_id: String,
    /// The number of documents indexed in this shard.
    pub doc_count: u64,
    /// The average document length (in whatever unit term frequencies were
    /// counted in — typically tokens) across this shard's documents.
    pub avg_doc_length: f64,
    /// The shard's published per-term resource description.
    pub term_stats: ShardTermStats,
}

impl ShardDescriptor {
    /// Create a new descriptor with empty term statistics.
    #[must_use]
    pub fn new(shard_id: impl Into<String>, doc_count: u64, avg_doc_length: f64) -> Self {
        Self {
            shard_id: shard_id.into(),
            doc_count,
            avg_doc_length,
            term_stats: ShardTermStats::new(),
        }
    }

    /// Replace this descriptor's term statistics wholesale.
    #[must_use]
    pub fn with_term_stats(mut self, term_stats: ShardTermStats) -> Self {
        self.term_stats = term_stats;
        self
    }

    /// Record a single term's statistics and return `self` for chaining —
    /// shorthand for `with_term_stats` when building descriptors term by
    /// term.
    #[must_use]
    pub fn with_term(
        mut self,
        term: impl Into<String>,
        document_frequency: u64,
        collection_term_frequency: u64,
    ) -> Self {
        self.term_stats =
            self.term_stats
                .with_term(term, document_frequency, collection_term_frequency);
        self
    }

    /// This shard's approximate total collection word count (`cw` in the
    /// CORI literature): `doc_count × avg_doc_length`.
    #[must_use]
    pub fn collection_word_count(&self) -> f64 {
        #[allow(clippy::cast_precision_loss)]
        let doc_count = self.doc_count as f64;
        doc_count * self.avg_doc_length
    }
}

// ── ShardSelectionScore ──────────────────────────────────────────────────────

/// A single shard's rank position in a [`crate::shard_selection::ShardSelector::select`]
/// result: its id, its CORI belief score, and how many query-term
/// occurrences it had non-zero evidence for.
///
/// This carries only a score and an id — never a document, a passage, or a
/// merged/calibrated relevance value. Score calibration and cross-shard
/// result merging happen later, in other modules, only *after* the shards
/// named here have actually been queried.
#[derive(Debug, Clone, PartialEq)]
pub struct ShardSelectionScore {
    /// The scored shard's id, copied from its [`ShardDescriptor::shard_id`].
    pub shard_id: String,
    /// The shard's overall CORI belief score for the query: the arithmetic
    /// mean of the per-query-term-occurrence belief values (see
    /// `ShardSelectionEngine::term_belief`).
    /// Always within `[belief_floor, belief_floor + belief_scale]` (the
    /// default configuration bounds this to `[0.4, 1.0]`).
    pub score: f64,
    /// The number of query-term *occurrences* (positions in the query-term
    /// list, duplicates counted individually) for which this shard reported
    /// a non-zero document frequency. `0` means every query term was either
    /// absent from this shard or the query itself was empty.
    pub matched_term_count: usize,
}

impl ShardSelectionScore {
    /// Construct a score record directly. Exposed mainly so callers and
    /// tests can build expected values without going through a full
    /// [`crate::shard_selection::ShardSelector::select`] run.
    #[must_use]
    pub fn new(shard_id: impl Into<String>, score: f64, matched_term_count: usize) -> Self {
        Self {
            shard_id: shard_id.into(),
            score,
            matched_term_count,
        }
    }
}

// ── ShardSelectionConfig ─────────────────────────────────────────────────────

/// Tunable constants for the CORI resource-selection formula, plus a
/// documented default `top_k`.
///
/// # The CORI formula
///
/// For a query term with document frequency `df` in a shard whose
/// collection word count is `cw`, against a candidate set whose average
/// collection word count is `avgcw`:
///
/// ```text
/// T = df / (df + frequency_saturation_constant + length_normalization_constant * cw / avgcw)
/// I = ln((num_shards + 0.5) / shard_frequency) / ln(num_shards + 1.0)
/// belief = belief_floor + belief_scale * T * I
/// ```
///
/// `frequency_saturation_constant` and `length_normalization_constant` are
/// the classic CORI tuning constants (`50` and `150` respectively in Callan,
/// Lu & Croft's original formulation); `belief_floor` and `belief_scale`
/// (`0.4` / `0.6`) set the floor every term contributes even with zero
/// evidence, and the range added on top of it.
#[derive(Debug, Clone, PartialEq)]
pub struct ShardSelectionConfig {
    /// The additive document-frequency saturation constant in the `T`
    /// component (`50` in the classic CORI formulation). Larger values make
    /// `T` less sensitive to `df`.
    pub frequency_saturation_constant: f64,
    /// The multiplier on the shard-size ratio `cw / avgcw` in the `T`
    /// component (`150` in the classic CORI formulation). Larger values
    /// penalize above-average-size shards more heavily.
    pub length_normalization_constant: f64,
    /// The baseline belief every query term contributes, even when it is
    /// entirely absent from a shard (`T = 0`). Defaults to `0.4`.
    pub belief_floor: f64,
    /// The scale applied to `T * I` on top of `belief_floor`. Defaults to
    /// `0.6`, so `belief_floor + belief_scale` bounds a term's maximum
    /// possible contribution at `1.0` with the default floor.
    pub belief_scale: f64,
    /// A documented, sensible default for the `top_k` argument of
    /// [`crate::shard_selection::ShardSelector::select`]. Note that `select`
    /// always requires `top_k` to be passed explicitly — this field is never
    /// read internally by `select`; it exists purely so callers have a
    /// discoverable, well-reasoned starting point (e.g.
    /// `selector.select(&shards, &terms, selector.config().default_top_k)`).
    pub default_top_k: usize,
}

impl Default for ShardSelectionConfig {
    fn default() -> Self {
        Self {
            frequency_saturation_constant: 50.0,
            length_normalization_constant: 150.0,
            belief_floor: 0.4,
            belief_scale: 0.6,
            default_top_k: 10,
        }
    }
}

impl ShardSelectionConfig {
    /// Create a configuration with the classic CORI default constants.
    #[must_use]
    pub fn new() -> Self {
        Self::default()
    }

    /// Override the document-frequency saturation constant (`50` by
    /// default).
    #[must_use]
    pub fn with_frequency_saturation_constant(mut self, value: f64) -> Self {
        self.frequency_saturation_constant = value;
        self
    }

    /// Override the shard-size length-normalization constant (`150` by
    /// default).
    #[must_use]
    pub fn with_length_normalization_constant(mut self, value: f64) -> Self {
        self.length_normalization_constant = value;
        self
    }

    /// Override the baseline belief floor (`0.4` by default).
    #[must_use]
    pub fn with_belief_floor(mut self, value: f64) -> Self {
        self.belief_floor = value;
        self
    }

    /// Override the belief scale (`0.6` by default).
    #[must_use]
    pub fn with_belief_scale(mut self, value: f64) -> Self {
        self.belief_scale = value;
        self
    }

    /// Override the documented default `top_k`.
    #[must_use]
    pub fn with_default_top_k(mut self, value: usize) -> Self {
        self.default_top_k = value;
        self
    }
}

// ── ShardSelectionError ──────────────────────────────────────────────────────

/// Errors produced by the `shard_selection` module.
#[derive(Debug, Error)]
pub enum ShardSelectionError {
    /// [`crate::shard_selection::ShardSelector::select`] was called with no
    /// candidate shards at all.
    #[error("shard set must not be empty")]
    EmptyShardSet,
    /// [`crate::shard_selection::ShardSelector::select`] was called with no
    /// query terms.
    #[error("query terms must not be empty")]
    EmptyQueryTerms,
    /// Two [`ShardDescriptor`]s in the same candidate set declared the same
    /// `shard_id`, making the candidate set ambiguous.
    #[error("duplicate shard id: {shard_id}")]
    DuplicateShardId {
        /// The `shard_id` that appeared more than once.
        shard_id: String,
    },
    /// A [`ShardDescriptor`] carried a value that cannot be used in the CORI
    /// formula (currently: a negative or non-finite `avg_doc_length`).
    #[error("invalid shard descriptor for shard {shard_id}: {reason}")]
    InvalidShardDescriptor {
        /// The offending shard's id.
        shard_id: String,
        /// A human-readable explanation of what was invalid.
        reason: String,
    },
}

/// Convenience alias for this module's fallible return type.
pub type ShardSelectionResult<T> = Result<T, ShardSelectionError>;
