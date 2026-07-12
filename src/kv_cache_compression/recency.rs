//! The **recency baseline** — the honest straw man.
//!
//! Keep the last `budget` tokens; evict everything older. No attention score is
//! consulted, no sink is pinned, no vote is counted. This is what a KV cache
//! looks like when it is managed the way an LRU manages web pages: *recency is
//! the only signal*.
//!
//! It is here to be beaten, and to make the beating measurable. Every claim this
//! module makes about H2O, `StreamingLLM` and `SnapKV` is a *comparative* claim —
//! "these policies preserve the attention output under a tight budget" is
//! meaningless without a policy that does not. The module's headline test builds
//! a workload whose attention mass genuinely sits on a handful of *old* tokens,
//! compresses with each policy to the same budget, and measures the deviation of
//! the resulting attention output from the uncompressed one. This policy is the
//! one that throws the salient tokens away, and the gap between its deviation and
//! H2O's is the entire justification for the rest of the module.
//!
//! It is a straw man but not a *dishonest* one: when the salient tokens really are
//! the recent ones (a short, local dependency), this policy is optimal and costs
//! nothing to run. The tests say so explicitly rather than pretending the
//! baseline is always bad.
//!
//! Relation to [`KvEvictionPolicy::StreamingLlm`](super::types::KvEvictionPolicy::StreamingLlm):
//! this policy is exactly `StreamingLLM` with `sink_tokens = 0`. That is not a
//! coincidence — it is the *ablation* of `StreamingLLM`, and comparing the two at
//! an identical budget isolates the contribution of the attention sinks to a
//! single variable.

use super::types::{KvCompressionConfig, KvResult};

/// Choose the slots the recency baseline retains: the last `budget` of them.
///
/// Returns a strictly-increasing list of slot indices whose length is at most
/// `config.budget` (and exactly `config.budget` when the cache is over budget).
///
/// # Errors
///
/// Never fails for a configuration that has passed
/// [`KvCompressionConfig::validate`]. The `Result` is part of the signature so
/// that every policy in this module composes uniformly.
pub fn plan_recency_lru(seq_len: usize, config: &KvCompressionConfig) -> KvResult<Vec<usize>> {
    if seq_len <= config.budget {
        return Ok((0..seq_len).collect());
    }
    Ok((seq_len - config.budget..seq_len).collect())
}
