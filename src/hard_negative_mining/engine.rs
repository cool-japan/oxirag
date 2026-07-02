//! [`HardNegativeMiner`] — the asynchronous-refresh orchestrator.
//!
//! The miner binds a fixed labelled set (positive pairs) and corpus, then mines
//! successive [`MiningRound`]s under evolving [`EmbeddingVersion`]s. Each
//! [`HardNegativeMiner::refresh`] re-ranks the whole corpus under a *new*
//! version and measures how far the mined negative sets drifted from the
//! previous round — the staleness signal that makes the mining "asynchronous".
//! Round and staleness history both accumulate.

use std::collections::{HashMap, HashSet};

use crate::types::DocumentId;

use super::miner::{mine_round, validate_inputs};
use super::types::{
    EmbeddingVersion, HardNegativeConfig, HardNegativeDocument, HardNegativeError,
    HardNegativePositivePair, HardNegativeQueryOverlap, HardNegativeQueryResult,
    HardNegativeResult, HardNegativeStaleness, MiningRound,
};

/// Drives ANCE-style asynchronous hard-negative mining over a fixed labelled
/// set and corpus.
///
/// Construct with [`HardNegativeMiner::new`] (which validates the inputs once),
/// call [`HardNegativeMiner::mine`] to produce the first round under some
/// [`EmbeddingVersion`], then [`HardNegativeMiner::refresh`] each time the
/// embedding model "changes" (a new version) to re-mine and measure the drift.
///
/// The corpus and positives are stored so that `refresh` re-mines exactly the
/// same labelled set under the new version — the definition of asynchronous
/// re-mining.
#[derive(Debug, Clone)]
pub struct HardNegativeMiner {
    config: HardNegativeConfig,
    positives: Vec<HardNegativePositivePair>,
    corpus: Vec<HardNegativeDocument>,
    history: Vec<MiningRound>,
    staleness_log: Vec<HardNegativeStaleness>,
}

impl HardNegativeMiner {
    /// Create a miner bound to `config`, `positives`, and `corpus`.
    ///
    /// The inputs are validated immediately so later `mine`/`refresh` calls
    /// cannot fail on malformed inputs.
    ///
    /// # Errors
    ///
    /// See [`crate::hard_negative_mining::miner`]-level validation via
    /// [`HardNegativeError`]: empty corpus/positives, duplicate corpus ids,
    /// empty queries, unknown positive ids, and the zero-valued config guards.
    pub fn new(
        config: HardNegativeConfig,
        positives: Vec<HardNegativePositivePair>,
        corpus: Vec<HardNegativeDocument>,
    ) -> HardNegativeResult<Self> {
        validate_inputs(&config, &positives, &corpus)?;
        Ok(Self {
            config,
            positives,
            corpus,
            history: Vec::new(),
            staleness_log: Vec::new(),
        })
    }

    /// The miner's configuration.
    #[must_use]
    pub fn config(&self) -> &HardNegativeConfig {
        &self.config
    }

    /// The bound corpus.
    #[must_use]
    pub fn corpus(&self) -> &[HardNegativeDocument] {
        &self.corpus
    }

    /// The bound labelled positive pairs.
    #[must_use]
    pub fn positives(&self) -> &[HardNegativePositivePair] {
        &self.positives
    }

    /// Every mined round so far, oldest first.
    #[must_use]
    pub fn history(&self) -> &[MiningRound] {
        &self.history
    }

    /// The number of rounds mined so far.
    #[must_use]
    pub fn round_count(&self) -> usize {
        self.history.len()
    }

    /// The most recent mined round, if any.
    #[must_use]
    pub fn latest_round(&self) -> Option<&MiningRound> {
        self.history.last()
    }

    /// Every staleness report produced by [`HardNegativeMiner::refresh`],
    /// oldest first.
    #[must_use]
    pub fn staleness_log(&self) -> &[HardNegativeStaleness] {
        &self.staleness_log
    }

    /// The most recent staleness report, if any refresh has occurred.
    #[must_use]
    pub fn latest_staleness(&self) -> Option<&HardNegativeStaleness> {
        self.staleness_log.last()
    }

    /// Mine one round under `version`, append it to the history, and return it.
    ///
    /// This is the entry point for the *first* round; subsequent rounds usually
    /// come from [`HardNegativeMiner::refresh`] (which additionally reports
    /// drift), though calling `mine` repeatedly is also valid and simply
    /// records rounds without a staleness comparison.
    ///
    /// # Errors
    ///
    /// Returns a [`HardNegativeError`] only in the pathological case that the
    /// bound inputs no longer validate; with a miner built by
    /// [`HardNegativeMiner::new`] this cannot happen.
    pub fn mine(&mut self, version: EmbeddingVersion) -> HardNegativeResult<MiningRound> {
        let round = mine_round(
            &self.config,
            &self.positives,
            &self.corpus,
            version,
            self.history.len(),
        )?;
        self.history.push(round.clone());
        Ok(round)
    }

    /// Re-mine the bound labelled set under a *new* `version`, measure how far
    /// the mined negatives drifted from the previous round, append both the
    /// round and its [`HardNegativeStaleness`] report, and return the new
    /// round.
    ///
    /// The staleness report — retrievable afterwards via
    /// [`HardNegativeMiner::latest_staleness`] — is the asynchronous signal:
    /// negatives mined under the old version go stale as the embedding function
    /// drifts, and this quantifies that per query (Jaccard overlap of the mined
    /// negative sets) and in aggregate.
    ///
    /// # Errors
    ///
    /// - [`HardNegativeError::NoPriorRound`] if no round has been mined yet
    ///   (there is nothing to compare against).
    /// - Any validation error, in the pathological case noted on
    ///   [`HardNegativeMiner::mine`].
    pub fn refresh(&mut self, new_version: EmbeddingVersion) -> HardNegativeResult<MiningRound> {
        let previous = self
            .history
            .last()
            .cloned()
            .ok_or(HardNegativeError::NoPriorRound)?;

        let round = mine_round(
            &self.config,
            &self.positives,
            &self.corpus,
            new_version,
            self.history.len(),
        )?;

        let staleness = compute_staleness(&previous, &round);
        self.history.push(round.clone());
        self.staleness_log.push(staleness);
        Ok(round)
    }

    /// Mine one round under `version` without recording it in any history — a
    /// stateless single-round primitive over the *supplied* inputs.
    ///
    /// Equivalent to what [`HardNegativeMiner::mine`] runs internally, exposed
    /// for callers who want a one-shot mining pass with no refresh cycle.
    ///
    /// # Errors
    ///
    /// See [`HardNegativeMiner::new`].
    pub fn mine_round(
        config: &HardNegativeConfig,
        positives: &[HardNegativePositivePair],
        corpus: &[HardNegativeDocument],
        version: EmbeddingVersion,
        round_index: usize,
    ) -> HardNegativeResult<MiningRound> {
        mine_round(config, positives, corpus, version, round_index)
    }

    /// Compute the [`HardNegativeStaleness`] between two arbitrary rounds
    /// without mining — the drift of `current`'s mined negatives relative to
    /// `previous`'s, per query and in aggregate.
    #[must_use]
    pub fn staleness(previous: &MiningRound, current: &MiningRound) -> HardNegativeStaleness {
        compute_staleness(previous, current)
    }
}

// ── Staleness computation ─────────────────────────────────────────────────────

/// The set of mined negative ids for a query result.
fn negative_id_set(result: &HardNegativeQueryResult) -> HashSet<DocumentId> {
    result.negatives.iter().map(|s| s.doc_id.clone()).collect()
}

/// Jaccard overlap `|a ∩ b| / |a ∪ b|` of two id sets; two empty sets overlap
/// perfectly (`1.0`).
#[allow(clippy::cast_precision_loss)]
fn jaccard(a: &HashSet<DocumentId>, b: &HashSet<DocumentId>) -> (f32, usize, usize, usize) {
    let intersection = a.intersection(b).count();
    let union = a.union(b).count();
    let added = b.difference(a).count();
    let removed = a.difference(b).count();
    let overlap = if union == 0 {
        1.0
    } else {
        intersection as f32 / union as f32
    };
    (overlap, intersection, added, removed)
}

/// Compute per-query and aggregate staleness of `current` relative to
/// `previous`.
///
/// Queries are matched by text: each query in `current` is compared against the
/// same query in `previous` (or the empty set if `previous` never mined it).
fn compute_staleness(previous: &MiningRound, current: &MiningRound) -> HardNegativeStaleness {
    let previous_sets: HashMap<&str, HashSet<DocumentId>> = previous
        .results
        .iter()
        .map(|r| (r.query.as_str(), negative_id_set(r)))
        .collect();

    let empty: HashSet<DocumentId> = HashSet::new();
    let mut per_query = Vec::with_capacity(current.results.len());
    let mut jaccard_sum: f64 = 0.0;

    for result in &current.results {
        let current_set = negative_id_set(result);
        let previous_set = previous_sets.get(result.query.as_str()).unwrap_or(&empty);
        let (overlap, retained, added, removed) = jaccard(previous_set, &current_set);
        jaccard_sum += f64::from(overlap);
        per_query.push(HardNegativeQueryOverlap {
            query: result.query.clone(),
            jaccard: overlap,
            retained,
            added,
            removed,
        });
    }

    #[allow(clippy::cast_precision_loss, clippy::cast_possible_truncation)]
    let mean_jaccard = if per_query.is_empty() {
        1.0
    } else {
        (jaccard_sum / per_query.len() as f64) as f32
    };

    HardNegativeStaleness {
        previous_round_index: previous.round_index,
        current_round_index: current.round_index,
        previous_version: previous.version,
        current_version: current.version,
        per_query,
        mean_jaccard,
        mean_positive_rank_delta: current.mean_positive_rank - previous.mean_positive_rank,
    }
}
