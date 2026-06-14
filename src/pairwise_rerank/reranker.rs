//! Pairwise tournament reranker (monoT5 / duoT5 inspired).
use std::fmt;

use crate::pairwise_rerank::types::{
    PairwiseComparer, PairwiseConfig, PairwiseError, PairwiseHit, PairwiseScoredPair,
};

// ── PairwiseReranker ──────────────────────────────────────────────────────────

/// Reranks a document corpus via a pairwise tournament bracket.
///
/// Inspired by the monoT5 / duoT5 family of rerankers, this struct runs
/// `tournament_rounds` full round-robin passes over all O(n²) document pairs.
/// Each pair is scored by a [`PairwiseComparer`]; the winning document receives
/// one point. After all rounds the documents are sorted by total win count
/// (descending) and the top-`k` results are returned as [`PairwiseHit`]s.
///
/// # Example
///
/// ```
/// use oxirag::pairwise_rerank::{MockPairwiseComparer, PairwiseConfig, PairwiseReranker};
///
/// let config = PairwiseConfig::default();
/// let comparer = Box::new(MockPairwiseComparer::new(false));
/// let reranker = PairwiseReranker::new(config, comparer);
///
/// let docs = vec![
///     ("d1".to_string(), "rust memory ownership".to_string()),
///     ("d2".to_string(), "banana fruit smoothie".to_string()),
/// ];
/// let hits = reranker.rerank("rust memory", &docs, 2).unwrap();
/// assert_eq!(hits[0].id, "d1");
/// ```
pub struct PairwiseReranker {
    /// Configuration that controls tournament behaviour.
    pub config: PairwiseConfig,
    /// The comparison oracle used for every pairwise matchup.
    pub comparer: Box<dyn PairwiseComparer>,
}

impl fmt::Debug for PairwiseReranker {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.debug_struct("PairwiseReranker")
            .field("config", &self.config)
            .field("comparer", &self.comparer)
            .finish()
    }
}

impl PairwiseReranker {
    /// Create a new reranker with the given `config` and `comparer`.
    #[must_use]
    pub fn new(config: PairwiseConfig, comparer: Box<dyn PairwiseComparer>) -> Self {
        Self { config, comparer }
    }

    /// Run one full round-robin pass, accumulating wins into `wins`.
    ///
    /// Every ordered pair `(i, j)` where `i < j` is compared exactly once.
    /// The winning document's slot in `wins` is incremented.
    ///
    /// Returns the [`PairwiseScoredPair`] record for every comparison made in
    /// this round, which can be inspected for testing or debugging.
    fn run_round(
        &self,
        query: &str,
        docs: &[(String, String)],
        wins: &mut [usize],
    ) -> Vec<PairwiseScoredPair> {
        let n = docs.len();
        let mut pairs = Vec::with_capacity(n * (n.saturating_sub(1)) / 2);
        for i in 0..n {
            for j in (i + 1)..n {
                let winner_idx = self.comparer.compare(query, &docs[i].1, &docs[j].1);
                if winner_idx == 0 {
                    wins[i] += 1;
                } else {
                    wins[j] += 1;
                }
                pairs.push(PairwiseScoredPair {
                    doc_a_idx: i,
                    doc_b_idx: j,
                    winner_idx,
                });
            }
        }
        pairs
    }

    /// Run the complete tournament and return the pairs from the final round.
    ///
    /// This convenience method exposes the per-pair decisions from the last
    /// tournament round, which can be useful for explainability or unit-testing
    /// the comparer integration.
    ///
    /// # Errors
    ///
    /// - [`PairwiseError::EmptyCorpus`] if `docs` is empty.
    pub fn tournament_pairs(
        &self,
        query: &str,
        docs: &[(String, String)],
    ) -> Result<Vec<PairwiseScoredPair>, PairwiseError> {
        if docs.is_empty() {
            return Err(PairwiseError::EmptyCorpus);
        }
        let n = docs.len();
        let mut wins = vec![0usize; n];
        let rounds = self.config.effective_rounds();
        let mut last_pairs = Vec::new();
        for _ in 0..rounds {
            last_pairs = self.run_round(query, docs, &mut wins);
        }
        Ok(last_pairs)
    }

    /// Rerank `docs` for `query`, returning the top-`k` results.
    ///
    /// `docs` is a slice of `(id, content)` pairs. The method runs
    /// [`config.tournament_rounds`](PairwiseConfig::tournament_rounds) full
    /// round-robin passes (at least one). Each pass compares every pair
    /// `(i, j)` with `i < j` via the configured [`PairwiseComparer`].
    /// Win totals are accumulated, then documents are sorted by wins
    /// descending (ties broken by original index, ascending) and the top-`k`
    /// are returned.
    ///
    /// The `score` field on each [`PairwiseHit`] is normalised to `[0.0, 1.0]`
    /// as `wins / (effective_rounds × (n − 1))`.
    ///
    /// # Errors
    ///
    /// - [`PairwiseError::EmptyCorpus`] if `docs` is empty.
    /// - [`PairwiseError::InvalidK`] if `k` is zero.
    pub fn rerank(
        &self,
        query: &str,
        docs: &[(String, String)],
        k: usize,
    ) -> Result<Vec<PairwiseHit>, PairwiseError> {
        if docs.is_empty() {
            return Err(PairwiseError::EmptyCorpus);
        }
        if k == 0 {
            return Err(PairwiseError::InvalidK(k));
        }

        let n = docs.len();
        let mut wins = vec![0usize; n];
        let rounds = self.config.effective_rounds();

        for _ in 0..rounds {
            self.run_round(query, docs, &mut wins);
        }

        // Maximum possible wins for a single doc: one win per opponent per round.
        let max_wins = rounds * (n - 1);

        // Sort by wins descending; tiebreak by original index ascending for
        // determinism.
        let mut indexed: Vec<(usize, usize)> = wins.iter().copied().enumerate().collect();
        indexed.sort_by(|a, b| b.1.cmp(&a.1).then(a.0.cmp(&b.0)));

        let take = k.min(n);
        let result = indexed
            .into_iter()
            .take(take)
            .map(|(i, w)| {
                #[allow(clippy::cast_precision_loss)]
                let score = if max_wins == 0 {
                    0.0_f32
                } else {
                    w as f32 / max_wins as f32
                };
                PairwiseHit {
                    id: docs[i].0.clone(),
                    content: docs[i].1.clone(),
                    score,
                    wins: w,
                }
            })
            .collect();

        Ok(result)
    }
}
