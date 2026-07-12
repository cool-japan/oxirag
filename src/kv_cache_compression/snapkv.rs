//! **`SnapKV`** (Li et al., *`SnapKV`: LLM Knows What You are Looking for Before
//! Generation*, `NeurIPS` 2024).
//!
//! # The policy
//!
//! Given a budget `B` and an observation window `W`, `SnapKV` retains
//!
//! ```text
//! prefix      = slots [0, seq_len - W)
//! observation = slots [seq_len - W, seq_len)          ← always kept
//! votes[t]    = Σ_{q ∈ last W query steps} A[q, t]    ← only these queries vote
//! pooled      = pool(votes over the prefix)           ← max/mean pool, width `pooling_kernel`
//! keep        = top-(B - W) of `pooled`  ∪  observation
//! ```
//!
//! # What makes this different from H2O
//!
//! Not the arithmetic — both rank tokens by realized attention mass. The
//! difference is **whose votes count**.
//!
//! H2O's score is a sum over *every* query step in history. In a long prompt
//! that is a democracy of the whole context: a token that was heavily attended
//! during some early, now-irrelevant part of the prompt keeps its score forever
//! (modulo the optional forgetting factor). `SnapKV`'s observation is that the
//! queries at the very *end* of the prompt — the ones immediately preceding the
//! tokens about to be generated — are dramatically better predictors of what the
//! generation will need, because they are the queries whose attention pattern
//! the first generated token's query will most resemble. So `SnapKV` throws the
//! older ballots away entirely and lets only the last `W` query positions decide.
//!
//! The paper's title is the point: the model *already knows*, in the attention of
//! the final prompt tokens, what it is going to want to look at. `SnapKV` reads
//! that off before generating a single token, which is why it is a **prefill-time,
//! one-shot** compression rather than a per-step eviction rule like H2O.
//!
//! # Why pool before the top-`k`
//!
//! Attention over a long prefix is spiky. A query may put nearly all of its mass
//! on one token of an informative phrase and almost none on the token beside it —
//! not because the neighbour is uninformative, but because attention is a
//! *competition*, and within a phrase one token usually wins it. A raw top-`k`
//! over such a vector keeps a scatter of isolated tokens torn out of their
//! context, and the surviving cache reads like a bag of disconnected words.
//!
//! Max-pooling the score vector with a width-`k` window first smears every peak
//! across its neighbourhood, so the tokens *around* a strong peak inherit a high
//! pooled score and get selected with it. The retained set clusters into
//! contiguous spans covering the informative regions. Crucially the pooling only
//! affects the *ranking*: the tokens that end up in the cache are the original,
//! unmodified tokens at the selected indices. Nothing is blurred or merged.
//!
//! [`KvSnapPooling::Mean`] is offered as the
//! smoother alternative, and it trades differently: a lone peak surrounded by
//! zeros has its score divided by the kernel width, so mean-pooling prefers
//! *dense regions of moderate interest* over *single points of high interest*.
//! [`KvSnapPooling::None`] reproduces the
//! unpooled ablation.

use std::collections::BTreeSet;

use super::attention::KvAttentionStats;
use super::types::{
    KvCompressionConfig, KvCompressionError, KvEvictionPolicy, KvResult, KvSnapPooling,
    ordered_selection, top_k_by_score,
};

/// Pool a score vector along the token axis with a **centred** window of
/// `kernel` tokens.
///
/// `kernel` must be odd (enforced by [`KvCompressionConfig::validate`]) so the
/// window has a well-defined centre; an even kernel would bias every selection
/// half a token in one direction.
///
/// At the edges the window is *clamped* to the valid range, and
/// [`KvSnapPooling::Mean`] divides by the clamped window's actual width rather
/// than by `kernel`. Dividing by `kernel` would attenuate the first and last
/// `kernel / 2` tokens by up to 2x purely because they have fewer neighbours —
/// a boundary artefact that would systematically under-select the beginning of
/// the prefix, which is precisely where a prompt's instructions live.
#[must_use]
pub fn pool_scores(scores: &[f64], kernel: usize, pooling: KvSnapPooling) -> Vec<f64> {
    if pooling == KvSnapPooling::None || kernel <= 1 || scores.is_empty() {
        return scores.to_vec();
    }
    let radius = kernel / 2;
    let length = scores.len();
    (0..length)
        .map(|centre| {
            let start = centre.saturating_sub(radius);
            let end = (centre + radius + 1).min(length);
            let window = &scores[start..end];
            match pooling {
                KvSnapPooling::Max => window.iter().copied().fold(f64::NEG_INFINITY, f64::max),
                KvSnapPooling::Mean => {
                    let width = window.len();
                    if width == 0 {
                        0.0
                    } else {
                        #[allow(clippy::cast_precision_loss)]
                        let denominator = width as f64;
                        window.iter().sum::<f64>() / denominator
                    }
                }
                KvSnapPooling::None => scores[centre],
            }
        })
        .collect()
}

/// Choose the slots `SnapKV` retains: the observation window, plus the top-`k`
/// prefix tokens as voted for by the observation window's own attention.
///
/// Returns a strictly-increasing list of slot indices whose length is at most
/// `config.budget` (and exactly `config.budget` when the cache is over budget).
///
/// # Errors
///
/// - [`KvCompressionError::StatsLengthMismatch`] if the statistics do not cover
///   exactly `seq_len` slots.
/// - [`KvCompressionError::NoAttentionHistory`] if no query step has been
///   recorded, so the ballot box is empty. An all-zero vote vector would make
///   the top-`k` fall through to its index tie-break and quietly keep the
///   *earliest* prefix tokens while claiming to have consulted the observation
///   window — so this is an error, not a fallback.
pub fn plan_snap_kv(
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

    if seq_len <= config.budget {
        return Ok((0..seq_len).collect());
    }

    if !stats.has_history() || stats.ballot_count() == 0 {
        return Err(KvCompressionError::NoAttentionHistory {
            policy: KvEvictionPolicy::SnapKv,
        });
    }

    // `seq_len > budget >= observation_window >= 1` (the floor and the
    // non-zero-window rule are enforced by `KvCompressionConfig::validate`).
    let observation = config.observation_window.min(seq_len);
    let prefix_len = seq_len - observation;
    let prefix_budget = config.budget.saturating_sub(observation);

    let votes = stats.observation_votes(config.observation_window);
    let prefix_votes = &votes[..prefix_len];
    let pooled = pool_scores(prefix_votes, config.pooling_kernel, config.pooling);

    let candidates: Vec<usize> = (0..prefix_len).collect();
    let chosen = top_k_by_score(&candidates, &pooled, prefix_budget);

    let mut retained: BTreeSet<usize> = chosen.into_iter().collect();
    retained.extend(prefix_len..seq_len);
    Ok(ordered_selection(&retained))
}
