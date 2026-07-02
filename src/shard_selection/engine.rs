//! [`ShardSelectionEngine`] and [`ShardSelector`] — CORI-style resource
//! selection: scoring and ranking shards from their lightweight resource
//! descriptions ([`ShardDescriptor`]), entirely *before* any shard is
//! actually queried.
//!
//! [`ShardSelectionEngine`] owns the CORI math, decomposed into small,
//! independently-testable steps (the `T` document-frequency-and-size belief
//! component, the `I` resource-selection IDF analogue, and their combination
//! into a per-term belief). [`ShardSelector`] is the thin, ergonomic façade
//! most callers should reach for: it wraps an engine and exposes exactly one
//! method, [`ShardSelector::select`].

use std::collections::{HashMap, HashSet};

use super::types::{
    ShardDescriptor, ShardSelectionConfig, ShardSelectionError, ShardSelectionResult,
    ShardSelectionScore,
};

// ── ShardSelectionEngine ─────────────────────────────────────────────────────

/// Computes CORI belief scores for candidate shards and ranks them.
///
/// The granular `T` / `I` / per-term-belief computations
/// (`term_frequency_belief`, `inverse_shard_frequency`, `term_belief`) and
/// the per-shard/per-query-term orchestration (`average_collection_word_count`,
/// `shard_frequency`, `score_shard`) are crate-internal: they exist so this
/// module's own tests can verify each piece of the CORI formula against
/// hand-calculated values, but they are not part of the module's public
/// surface. [`ShardSelectionEngine::select`] (and the [`ShardSelector`]
/// façade that wraps it) is the full public API.
#[derive(Debug, Clone, Default)]
pub struct ShardSelectionEngine {
    /// The CORI formula constants and default `top_k` this engine scores
    /// with.
    pub config: ShardSelectionConfig,
}

impl ShardSelectionEngine {
    /// Create a new engine with the given configuration.
    #[must_use]
    pub fn new(config: ShardSelectionConfig) -> Self {
        Self { config }
    }

    /// The CORI `T` component: a term's document-frequency-and-shard-size
    /// belief within a single shard.
    ///
    /// ```text
    /// T = df / (df + frequency_saturation_constant + length_normalization_constant * cw / avgcw)
    /// ```
    ///
    /// `document_frequency` is `df`, `shard_collection_word_count` is this
    /// shard's `cw`, and `average_collection_word_count` is `avgcw` — the
    /// mean `cw` across the full candidate set, which normalizes away raw
    /// shard-size differences so a term's `df` is judged relative to how
    /// large the shard *typically* is in this candidate set.
    ///
    /// A shard reporting `document_frequency == 0` for the term (i.e. the
    /// term is absent from this shard, or was never published) yields
    /// `T = 0` unconditionally — this is what floors
    /// [`ShardSelectionEngine::term_belief`] at `belief_floor` for absent
    /// terms.
    ///
    /// Division-by-zero is avoided defensively: when
    /// `average_collection_word_count` is not positive (a degenerate
    /// candidate set where every shard reports zero size), the size ratio is
    /// treated as `0` rather than propagating a `NaN`.
    #[must_use]
    pub(crate) fn term_frequency_belief(
        &self,
        document_frequency: u64,
        shard_collection_word_count: f64,
        average_collection_word_count: f64,
    ) -> f64 {
        #[allow(clippy::cast_precision_loss)]
        let document_frequency = document_frequency as f64;
        let size_ratio = if average_collection_word_count > 0.0 {
            shard_collection_word_count / average_collection_word_count
        } else {
            0.0
        };
        let denominator = document_frequency
            + self.config.frequency_saturation_constant
            + self.config.length_normalization_constant * size_ratio;
        if denominator > 0.0 {
            document_frequency / denominator
        } else {
            0.0
        }
    }

    /// The CORI `I` component: the resource-selection analogue of IDF,
    /// measuring how discriminating a term is *across shards* (not across
    /// documents).
    ///
    /// ```text
    /// I = ln((num_candidate_shards + 0.5) / shard_frequency) / ln(num_candidate_shards + 1.0)
    /// ```
    ///
    /// `shard_frequency` is the number of shards, out of the full
    /// `num_candidate_shards`-shard candidate set, that contain the term at
    /// all. A term present in only a few shards is more discriminating for
    /// *deciding which shards to query* than a term present in nearly every
    /// shard, so rarer-across-shards terms yield a larger `I`.
    ///
    /// Returns `0.0` when `shard_frequency == 0` (the term is absent from
    /// every candidate shard — there is nothing to discriminate) or
    /// `num_candidate_shards == 0`, both handled explicitly to avoid a
    /// `ln(0)` or `0.0 / 0.0`.
    #[must_use]
    #[allow(clippy::unused_self)]
    pub(crate) fn inverse_shard_frequency(
        &self,
        shard_frequency: usize,
        num_candidate_shards: usize,
    ) -> f64 {
        if shard_frequency == 0 || num_candidate_shards == 0 {
            return 0.0;
        }
        #[allow(clippy::cast_precision_loss)]
        let num_candidate_shards = num_candidate_shards as f64;
        #[allow(clippy::cast_precision_loss)]
        let shard_frequency = shard_frequency as f64;
        let numerator = (num_candidate_shards + 0.5) / shard_frequency;
        let denominator = (num_candidate_shards + 1.0).ln();
        numerator.ln() / denominator
    }

    /// Combine a term's `T` and `I` components into its per-term belief:
    ///
    /// ```text
    /// belief = belief_floor + belief_scale * T * I
    /// ```
    ///
    /// Because `T = 0` whenever a term is absent from a shard (see
    /// [`ShardSelectionEngine::term_frequency_belief`]), an absent term
    /// always contributes exactly `belief_floor` here — regardless of how
    /// large `I` is — which is this module's chosen, deterministic rule for
    /// "a term the shard has no evidence of still contributes the baseline
    /// belief, never zero and never a penalty."
    #[must_use]
    pub(crate) fn term_belief(
        &self,
        term_frequency_belief: f64,
        inverse_shard_frequency: f64,
    ) -> f64 {
        self.config.belief_floor
            + self.config.belief_scale * term_frequency_belief * inverse_shard_frequency
    }

    /// The mean collection word count (`avgcw`) across every shard in
    /// `shards` — the shard-size normalizer shared by every `T` computation
    /// within one [`ShardSelectionEngine::select`] call.
    #[must_use]
    #[allow(clippy::unused_self)]
    pub(crate) fn average_collection_word_count(&self, shards: &[ShardDescriptor]) -> f64 {
        if shards.is_empty() {
            return 0.0;
        }
        let total: f64 = shards
            .iter()
            .map(ShardDescriptor::collection_word_count)
            .sum();
        #[allow(clippy::cast_precision_loss)]
        let shard_count = shards.len() as f64;
        total / shard_count
    }

    /// The number of shards, out of `shards`, that report a non-zero
    /// document frequency for `term`.
    #[must_use]
    #[allow(clippy::unused_self)]
    pub(crate) fn shard_frequency(&self, shards: &[ShardDescriptor], term: &str) -> usize {
        shards
            .iter()
            .filter(|shard| shard.term_stats.document_frequency(term) > 0)
            .count()
    }

    /// Score a single `shard` against every occurrence in `query_terms`.
    ///
    /// The shard's score is the arithmetic mean of the per-occurrence CORI
    /// beliefs (see [`ShardSelectionEngine::term_belief`]) — duplicated
    /// query terms are not deduplicated, so a repeated term's belief is
    /// counted once per occurrence, weighting the average toward terms the
    /// caller emphasized by repetition. `shard_frequency_by_term` must
    /// already contain an entry for every term in `query_terms` (computed
    /// once per candidate set by the caller, since it is the same for every
    /// shard).
    ///
    /// Returns a `0.0` score with `matched_term_count == 0` for an empty
    /// `query_terms` (nothing to average), so this method never divides by
    /// zero even when called outside of [`ShardSelectionEngine::select`]'s
    /// own empty-query-terms guard.
    #[must_use]
    pub(crate) fn score_shard(
        &self,
        shard: &ShardDescriptor,
        query_terms: &[String],
        shard_frequency_by_term: &HashMap<&str, usize>,
        num_candidate_shards: usize,
        average_collection_word_count: f64,
    ) -> ShardSelectionScore {
        if query_terms.is_empty() {
            return ShardSelectionScore::new(shard.shard_id.clone(), 0.0, 0);
        }

        let shard_collection_word_count = shard.collection_word_count();
        let mut belief_sum = 0.0_f64;
        let mut matched_term_count = 0_usize;

        for term in query_terms {
            let document_frequency = shard.term_stats.document_frequency(term);
            if document_frequency > 0 {
                matched_term_count += 1;
            }
            let shard_frequency = shard_frequency_by_term
                .get(term.as_str())
                .copied()
                .unwrap_or(0);

            let term_frequency_belief = self.term_frequency_belief(
                document_frequency,
                shard_collection_word_count,
                average_collection_word_count,
            );
            let inverse_shard_frequency =
                self.inverse_shard_frequency(shard_frequency, num_candidate_shards);
            belief_sum += self.term_belief(term_frequency_belief, inverse_shard_frequency);
        }

        #[allow(clippy::cast_precision_loss)]
        let query_term_count = query_terms.len() as f64;

        ShardSelectionScore::new(
            shard.shard_id.clone(),
            belief_sum / query_term_count,
            matched_term_count,
        )
    }

    /// Rank `shards` for `query_terms` by CORI belief score and return the
    /// top `top_k`, best-first.
    ///
    /// This is the entire pre-query shard-selection step: the only inputs
    /// are lightweight [`ShardDescriptor`] resource-description summaries
    /// and the query's terms, and the only output is a ranked list of shard
    /// ids with scores. No shard is queried, no document is touched, and no
    /// score calibration or cross-shard result merging happens here — that
    /// is the job of `ensemble_retriever`, `rank_fusion`, or `collections`,
    /// *after* the caller has used this ranking to decide which of the
    /// returned shard ids are worth querying at all.
    ///
    /// Ties (equal scores) are broken by ascending `shard_id`, so ranking is
    /// deterministic and stable across repeated calls with the same inputs,
    /// regardless of input order.
    ///
    /// `top_k == 0` returns an empty (but `Ok`) result — it is treated as a
    /// valid degenerate request ("rank, but keep none"), not an error.
    ///
    /// # Errors
    ///
    /// - [`ShardSelectionError::EmptyShardSet`] if `shards` is empty.
    /// - [`ShardSelectionError::EmptyQueryTerms`] if `query_terms` is empty.
    /// - [`ShardSelectionError::DuplicateShardId`] if two descriptors in
    ///   `shards` share a `shard_id`.
    /// - [`ShardSelectionError::InvalidShardDescriptor`] if a descriptor's
    ///   `avg_doc_length` is negative or non-finite.
    pub fn select(
        &self,
        shards: &[ShardDescriptor],
        query_terms: &[String],
        top_k: usize,
    ) -> ShardSelectionResult<Vec<ShardSelectionScore>> {
        if shards.is_empty() {
            return Err(ShardSelectionError::EmptyShardSet);
        }
        if query_terms.is_empty() {
            return Err(ShardSelectionError::EmptyQueryTerms);
        }

        let mut seen_shard_ids: HashSet<&str> = HashSet::with_capacity(shards.len());
        for shard in shards {
            if !seen_shard_ids.insert(shard.shard_id.as_str()) {
                return Err(ShardSelectionError::DuplicateShardId {
                    shard_id: shard.shard_id.clone(),
                });
            }
            if !shard.avg_doc_length.is_finite() || shard.avg_doc_length < 0.0 {
                return Err(ShardSelectionError::InvalidShardDescriptor {
                    shard_id: shard.shard_id.clone(),
                    reason: format!(
                        "avg_doc_length must be finite and non-negative, got {}",
                        shard.avg_doc_length
                    ),
                });
            }
        }

        if top_k == 0 {
            return Ok(Vec::new());
        }

        let num_candidate_shards = shards.len();
        let average_collection_word_count = self.average_collection_word_count(shards);

        // Shard frequency (the `I` component's denominator) only depends on
        // the term and the candidate set, not on any individual shard being
        // scored — compute it once per unique query term rather than once
        // per (shard, term occurrence) pair.
        let mut shard_frequency_by_term: HashMap<&str, usize> =
            HashMap::with_capacity(query_terms.len());
        for term in query_terms {
            let term_str = term.as_str();
            shard_frequency_by_term
                .entry(term_str)
                .or_insert_with(|| self.shard_frequency(shards, term_str));
        }

        let mut scores: Vec<ShardSelectionScore> = shards
            .iter()
            .map(|shard| {
                self.score_shard(
                    shard,
                    query_terms,
                    &shard_frequency_by_term,
                    num_candidate_shards,
                    average_collection_word_count,
                )
            })
            .collect();

        scores.sort_by(|a, b| {
            b.score
                .total_cmp(&a.score)
                .then_with(|| a.shard_id.cmp(&b.shard_id))
        });
        scores.truncate(top_k);

        Ok(scores)
    }
}

// ── ShardSelector ─────────────────────────────────────────────────────────────

/// The ergonomic entry point for pre-query shard selection: wraps a
/// [`ShardSelectionEngine`] and exposes exactly one method,
/// [`ShardSelector::select`].
///
/// ```rust
/// # #[cfg(feature = "shard-selection")]
/// # {
/// use oxirag::shard_selection::{ShardDescriptor, ShardSelectionConfig, ShardSelector};
///
/// let shards = vec![ShardDescriptor::new("shard-a", 100, 200.0).with_term("rust", 40, 90)];
/// let selector = ShardSelector::new(ShardSelectionConfig::default());
/// let ranked = selector
///     .select(&shards, &["rust".to_string()], 5)
///     .expect("non-empty shards and query terms");
/// assert_eq!(ranked.len(), 1);
/// # }
/// ```
#[derive(Debug, Clone, Default)]
pub struct ShardSelector {
    engine: ShardSelectionEngine,
}

impl ShardSelector {
    /// Create a new selector with the given configuration.
    #[must_use]
    pub fn new(config: ShardSelectionConfig) -> Self {
        Self {
            engine: ShardSelectionEngine::new(config),
        }
    }

    /// The configuration this selector scores with.
    #[must_use]
    pub fn config(&self) -> &ShardSelectionConfig {
        &self.engine.config
    }

    /// Rank `shards` for `query_terms` by CORI belief score and return the
    /// top `top_k`, best-first. See
    /// [`ShardSelectionEngine::select`] for the full contract.
    ///
    /// # Errors
    ///
    /// See [`ShardSelectionEngine::select`].
    pub fn select(
        &self,
        shards: &[ShardDescriptor],
        query_terms: &[String],
        top_k: usize,
    ) -> ShardSelectionResult<Vec<ShardSelectionScore>> {
        self.engine.select(shards, query_terms, top_k)
    }
}
