//! **`StreamingLLM` — attention sinks** (Xiao et al., *Efficient Streaming
//! Language Models with Attention Sinks*, ICLR 2024).
//!
//! # The policy
//!
//! Given a budget `B` and a sink count `S`, `StreamingLLM` retains
//!
//! ```text
//! keep = { the first S slots }  ∪  { the last (B - S) slots }
//! ```
//!
//! and evicts the entire middle. No attention score is consulted: the policy is
//! purely positional, which is exactly what makes it cheap enough to run on
//! every decoding step of an unbounded stream.
//!
//! Note that the *whole* budget is used: any budget beyond the configured
//! `sink_tokens + recent_window` floor is spent widening the recent window,
//! since that is the only region where a larger cache can still buy relevance.
//!
//! # Why the first few tokens must be kept even though they say nothing
//!
//! This is the counterintuitive, load-bearing claim of the paper, and this
//! module's tests *demonstrate* it rather than assert it.
//!
//! Softmax has no abstain option. Its weights are forced to sum to one:
//!
//! ```text
//! Σ_t A[q, t] = 1        for every query q
//! ```
//!
//! so a query with nothing relevant in its context cannot simply attend weakly
//! to everything — the normalization pushes the mass back up. A trained model's
//! only escape is to learn a **dumping ground**: a few keys, reliably present in
//! every context, whose logits are high for *every* query regardless of content.
//! The first tokens of the sequence are the only positions with that property
//! (they are the only ones every query can see, in every window, at every step),
//! so that is where the model puts them. Empirically these sink keys absorb a
//! large fraction of every row's attention mass while their *values* contribute
//! almost nothing semantic — the model is using them as a place to throw away
//! probability, not as a place to read information from.
//!
//! Now evict them. The mass they were absorbing does not vanish with them; the
//! softmax simply renormalizes over the survivors. If the sinks held a fraction
//! `m` of the row's mass, every surviving token's weight is multiplied by
//!
//! ```text
//! 1 / (1 - m)
//! ```
//!
//! In this module's sink fixture the measured sink mass is `m = 0.8575`, so
//! every surviving, largely-irrelevant token has its weight multiplied by
//! `1 / (1 - 0.8575) ≈ 7`. The output, which was a convex combination dominated
//! by a semantically empty sink value, becomes a convex combination dominated by
//! whatever was left, and it moves a long way. Evicting `S` *middle* tokens with
//! the same slot count but a negligible mass share does nothing of the kind:
//! their weights were near zero, the renormalizing factor is `1 / (1 - ε) ≈ 1`,
//! and the output barely moves.
//!
//! The asymmetry is therefore **not** about the sinks being informative. It is
//! about softmax being a normalized measure, and about what happens to the rest
//! of the measure when you delete the atom that was holding most of it.
//!
//! The test `streaming_llm_sink_removal_is_catastrophic_middle_removal_is_not`
//! builds a cache with a real sink structure, computes real attention, and
//! measures both deviations:
//!
//! | What is removed | Relative L2 deviation of the attention output |
//! |---|---|
//! | the 3 sink tokens | **5.363** |
//! | 3 middle tokens (same count) | **0.00021** |
//!
//! — a **25,000x** asymmetry for deleting the same number of tokens. A companion
//! test, `streaming_llm_beats_its_own_sinkless_ablation_at_the_same_budget`,
//! turns this into a controlled policy comparison: at an identical budget of 16
//! tokens, `StreamingLLM` (sinks kept) deviates by **0.0031** while
//! [`KvEvictionPolicy::RecencyLru`](super::types::KvEvictionPolicy::RecencyLru)
//! — which *is* `StreamingLLM` with `sink_tokens = 0`, the exact ablation —
//! deviates by **5.485**. Both keep the content the query needs; the sinks are
//! the only variable.

use std::collections::BTreeSet;

use super::types::{KvCompressionConfig, KvResult, ordered_selection};

/// Choose the slots `StreamingLLM` retains: the leading attention sinks plus a
/// trailing sliding window, with the middle evicted.
///
/// Returns a strictly-increasing list of slot indices whose length is at most
/// `config.budget` (and exactly `config.budget` when the cache is over budget).
///
/// # Errors
///
/// Never fails once the configuration has passed
/// [`KvCompressionConfig::validate`] (which is what guarantees
/// `budget >= sink_tokens + recent_window`, so the sinks and the window cannot
/// collide). The `Result` is part of the signature so that every policy in this
/// module composes uniformly.
pub fn plan_streaming_llm(seq_len: usize, config: &KvCompressionConfig) -> KvResult<Vec<usize>> {
    if seq_len <= config.budget {
        return Ok((0..seq_len).collect());
    }

    // `seq_len > budget >= sink_tokens + recent_window`, so:
    //   * the sink block [0, sinks) is strictly inside the cache, and
    //   * the window [seq_len - recent, seq_len) starts strictly after it,
    // hence the two blocks are disjoint and together are exactly `budget` slots.
    let sinks = config.sink_tokens.min(seq_len);
    let recent = config.budget.saturating_sub(sinks).min(seq_len - sinks);
    let recent_start = seq_len - recent;

    let mut retained: BTreeSet<usize> = (0..sinks).collect();
    retained.extend(recent_start..seq_len);
    Ok(ordered_selection(&retained))
}
