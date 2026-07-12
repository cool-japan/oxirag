//! [`KvCacheCompressor`] — the façade that ties an eviction policy, a
//! [`KvCacheTensor`], and its [`KvAttentionStats`] together.
//!
//! The compressor owns one invariant and enforces it on every successful return
//! path, for every policy and every budget:
//!
//! ```text
//! cache.seq_len() <= config.budget
//! ```
//!
//! It also owns the *other* half of eviction, the half that is easy to forget:
//! when slots are removed from the cache, the accumulated attention statistics
//! must be **re-indexed onto the survivors in the same pass**. Statistics and
//! cache are gathered by the identical `keep` vector inside
//! [`KvCacheCompressor::compress`], so it is not possible to end up with a
//! score vector that silently refers to tokens that are no longer there.

use super::attention::{KvAttentionOutput, KvAttentionStats};
use super::h2o::plan_h2o;
use super::recency::plan_recency_lru;
use super::snapkv::plan_snap_kv;
use super::streaming::plan_streaming_llm;
use super::types::{
    KvCacheTensor, KvCompressionConfig, KvCompressionError, KvCompressionReport, KvEvictionPolicy,
    KvResult,
};

/// Applies an eviction policy to a [`KvCacheTensor`] and its
/// [`KvAttentionStats`].
///
/// # Lifecycle
///
/// ```text
/// append tokens ──▶ attend (real softmax) ──▶ observe (fold the weight matrix
///                                              into the stats)
///                          │
///                          └──▶ compress (evict; cache and stats are re-indexed
///                                         together)
/// ```
#[derive(Debug, Clone, PartialEq)]
pub struct KvCacheCompressor {
    config: KvCompressionConfig,
}

impl KvCacheCompressor {
    /// Build a compressor, validating the configuration up front.
    ///
    /// # Errors
    ///
    /// Whatever [`KvCompressionConfig::validate`] rejects: a zero budget, an
    /// even pooling kernel, an out-of-range decay factor, or — most importantly
    /// — a budget below the policy's mandatory floor
    /// ([`KvCompressionError::BudgetBelowFloor`]).
    pub fn new(config: KvCompressionConfig) -> KvResult<Self> {
        config.validate()?;
        Ok(Self { config })
    }

    /// The configuration this compressor was built with.
    #[must_use]
    pub const fn config(&self) -> &KvCompressionConfig {
        &self.config
    }

    /// Fold one attention call's weight matrix into the running statistics,
    /// applying the configured forgetting factor.
    ///
    /// # Errors
    ///
    /// Whatever [`KvAttentionStats::accumulate`] rejects — chiefly a
    /// [`KvCompressionError::StatsLengthMismatch`] when the attention was
    /// computed over a cache of a different length than the statistics track.
    pub fn observe(
        &self,
        stats: &mut KvAttentionStats,
        attention: &KvAttentionOutput,
    ) -> KvResult<()> {
        stats.accumulate(attention, self.config.score_decay)
    }

    /// Decide which slots survive, without touching the cache.
    ///
    /// Exposed separately from [`Self::compress`] so a caller (or a test) can
    /// inspect a policy's decision before acting on it.
    ///
    /// # Errors
    ///
    /// - [`KvCompressionError::StatsLengthMismatch`] if the statistics do not
    ///   cover the cache's slots.
    /// - [`KvCompressionError::NoAttentionHistory`] if a score-driven policy is
    ///   asked to decide with an empty attention history.
    pub fn plan(&self, cache: &KvCacheTensor, stats: &KvAttentionStats) -> KvResult<Vec<usize>> {
        let seq_len = cache.seq_len();
        let keep = match self.config.policy {
            KvEvictionPolicy::H2O => plan_h2o(seq_len, &self.config, stats)?,
            KvEvictionPolicy::StreamingLlm => plan_streaming_llm(seq_len, &self.config)?,
            KvEvictionPolicy::SnapKv => plan_snap_kv(seq_len, &self.config, stats)?,
            KvEvictionPolicy::RecencyLru => plan_recency_lru(seq_len, &self.config)?,
        };

        // The budget is a hard cap, not an aspiration. Every policy is written
        // to respect it, and this check is the belt to that pair of braces: a
        // future policy that got the arithmetic wrong fails loudly here rather
        // than quietly handing back an over-budget cache.
        if keep.len() > self.config.budget {
            return Err(KvCompressionError::InvalidConfig {
                reason: format!(
                    "policy {:?} selected {} token(s), which exceeds the budget of {}",
                    self.config.policy,
                    keep.len(),
                    self.config.budget
                ),
            });
        }
        Ok(keep)
    }

    /// Evict from `cache` (and re-index `stats` onto the survivors) according to
    /// the configured policy.
    ///
    /// If the cache already fits the budget this is a no-op: **not one byte of
    /// the cache is touched**, so a subsequent attention call is bit-for-bit
    /// identical to the uncompressed one. The returned report says
    /// [`KvCompressionReport::is_noop`].
    ///
    /// # Errors
    ///
    /// - [`KvCompressionError::EmptyCache`] if there is nothing to compress.
    /// - [`KvCompressionError::StatsLengthMismatch`] if the statistics were
    ///   gathered for a different number of slots than the cache holds (which
    ///   would mean the scores refer to the wrong tokens).
    /// - [`KvCompressionError::NoAttentionHistory`] if a score-driven policy is
    ///   asked to decide with an empty attention history.
    pub fn compress(
        &self,
        cache: &mut KvCacheTensor,
        stats: &mut KvAttentionStats,
    ) -> KvResult<KvCompressionReport> {
        if cache.is_empty() {
            return Err(KvCompressionError::EmptyCache {
                operation: "compress",
            });
        }
        if stats.num_tokens() != cache.seq_len() {
            return Err(KvCompressionError::StatsLengthMismatch {
                stats_tokens: stats.num_tokens(),
                cache_tokens: cache.seq_len(),
            });
        }

        let tokens_before = cache.seq_len();
        let bytes_before = cache.memory_bytes();
        let positions_before = cache.positions().to_vec();

        let keep = self.plan(cache, stats)?;

        let retained_positions: Vec<usize> =
            keep.iter().map(|&index| positions_before[index]).collect();
        let mut kept_flags = vec![false; tokens_before];
        for &index in &keep {
            kept_flags[index] = true;
        }
        let evicted_positions: Vec<usize> = positions_before
            .iter()
            .enumerate()
            .filter_map(|(index, &position)| (!kept_flags[index]).then_some(position))
            .collect();

        // Cache and statistics are gathered by the *same* `keep` vector, in the
        // same pass, so slot `i` of the compressed cache and slot `i` of the
        // compressed statistics always describe the same token.
        cache.retain_tokens(&keep)?;
        stats.retain_tokens(&keep)?;

        let tokens_after = cache.seq_len();
        let bytes_after = cache.memory_bytes();

        Ok(KvCompressionReport {
            policy: self.config.policy,
            tokens_before,
            tokens_after,
            tokens_evicted: tokens_before - tokens_after,
            budget: self.config.budget,
            within_budget: tokens_after <= self.config.budget,
            retained_positions,
            evicted_positions,
            bytes_before,
            bytes_after,
        })
    }
}
