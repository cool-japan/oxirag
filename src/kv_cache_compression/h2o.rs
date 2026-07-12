//! **H2O — the Heavy-Hitter Oracle** (Zhang et al., *H2O: Heavy-Hitter Oracle
//! for Efficient Generative Inference of Large Language Models*, `NeurIPS` 2023).
//!
//! # The policy
//!
//! Given a budget `B`, a recent window `R`, and the accumulated attention
//! statistics, H2O retains
//!
//! ```text
//! keep = { the last R slots }  ∪  { the top (B - R) slots, by saliency, among the rest }
//! ```
//!
//! and evicts everything else. The retained set is exactly `B` slots whenever
//! the cache is over budget.
//!
//! # Why the accumulated attention column sum is the right saliency signal
//!
//! Fix a query `q` and a cached token `t`. The attention output for `q` is
//!
//! ```text
//! o_q = Σ_t A[q, t] · v_t          with  Σ_t A[q, t] = 1,  A[q, t] >= 0
//! ```
//!
//! — a *convex combination* of the value vectors. Evict `t` and the softmax is
//! recomputed over the survivors, which is the same as deleting `t`'s term and
//! renormalizing the rest by `1 / (1 - A[q, t])`. A short calculation bounds the
//! damage:
//!
//! ```text
//! ‖o_q - o_q^evicted‖  =  (A[q, t] / (1 - A[q, t])) · ‖v_t - o_q^evicted‖
//!                       <=  (A[q, t] / (1 - A[q, t])) · diam(V)
//! ```
//!
//! The perturbation to *this* query's output is thus controlled by `A[q, t]`
//! alone (the value geometry `diam(V)` being a property of the model, not of the
//! eviction choice). Summing the bound over the query steps that have actually
//! been issued makes the total damage a monotone function of `Σ_q A[q, t]` — the
//! **column sum of the attention matrix**. So the greedy way to spend a fixed
//! budget is: keep the tokens with the largest column sums, evict the smallest.
//! That is H2O, and it is why the signal is a *sum of realized softmax weights*
//! rather than a heuristic such as key norm, embedding similarity, or position.
//!
//! (H2O's own theoretical result is stronger — it shows this greedy rule is
//! near-optimal for the submodular objective of retained attention mass — but
//! the bound above is the operational reason it works, and it is the quantity
//! this module's tests measure directly.)
//!
//! # The early-token bias
//!
//! The raw column sum is biased toward old tokens for a purely structural
//! reason: under a causal mask, the token at position `t` is visible to every
//! query at position `>= t`, so early tokens are summed over more terms. See
//! [`KvScoreNormalization`](super::types::KvScoreNormalization) for the analysis
//! and for the correction this module applies by default (dividing by the number
//! of rows a token actually competed in, which makes the score scale-free in
//! cache age).
//!
//! Note that the correction and the recent window *compose*: the mean has its
//! own opposite-facing bias — a brand-new token has competed in a single row, so
//! one lucky row inflates its mean — but heavy hitters are only ever drawn from
//! **outside** the recent window, so those noisy, freshly-arrived means are
//! never the ones being ranked. Each mechanism covers the other's blind spot.

use std::collections::BTreeSet;

use super::attention::KvAttentionStats;
use super::types::{
    KvCompressionConfig, KvCompressionError, KvEvictionPolicy, KvResult, ordered_selection,
    top_k_by_score,
};

/// Choose the slots H2O retains, given the current cache length and the
/// accumulated attention statistics.
///
/// Returns a strictly-increasing list of slot indices whose length is at most
/// `config.budget` (and exactly `config.budget` when the cache is over budget).
///
/// # Errors
///
/// - [`KvCompressionError::StatsLengthMismatch`] if the statistics do not cover
///   exactly `seq_len` slots.
/// - [`KvCompressionError::NoAttentionHistory`] if no query step has been
///   recorded. Ranking by an all-zero score vector would keep whichever tokens
///   happen to win the index tie-break — a decision dressed up as a
///   heavy-hitter decision while being nothing of the kind — so this is an
///   error rather than a silent fallback.
pub fn plan_h2o(
    seq_len: usize,
    config: &KvCompressionConfig,
    stats: &KvAttentionStats,
) -> KvResult<Vec<usize>> {
    if stats.num_tokens() != seq_len {
        return Err(KvCompressionError::StatsLengthMismatch {
            stats_tokens: stats.num_tokens(),
            cache_tokens: seq_len,
        });
    }

    // Already within budget: nothing is evicted, and the cache must come out
    // bit-for-bit unchanged.
    if seq_len <= config.budget {
        return Ok((0..seq_len).collect());
    }

    if !stats.has_history() {
        return Err(KvCompressionError::NoAttentionHistory {
            policy: KvEvictionPolicy::H2O,
        });
    }

    // `seq_len > budget >= recent_window` here (the floor is enforced by
    // `KvCompressionConfig::validate`), so the recent window fits strictly
    // inside the cache and leaves a non-empty heavy-hitter candidate pool.
    let recent = config.recent_window.min(seq_len);
    let recent_start = seq_len - recent;
    let heavy_budget = config.budget.saturating_sub(recent);

    let saliency = stats.saliency(config.normalization);
    let candidates: Vec<usize> = (0..recent_start).collect();
    let heavy_hitters = top_k_by_score(&candidates, &saliency, heavy_budget);

    let mut retained: BTreeSet<usize> = heavy_hitters.into_iter().collect();
    retained.extend(recent_start..seq_len);
    Ok(ordered_selection(&retained))
}
