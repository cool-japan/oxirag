//! Core types for the `kv_cache_compression` module: the token-axis KV cache
//! tensor, the eviction-policy enum, the compression configuration, the
//! per-run report, and the error type.
//!
//! The single most important type here is [`KvCacheTensor`]. Unlike the
//! prompt-blob caches elsewhere in this crate, it is a *real* attention
//! key/value cache with an explicit **token axis** that eviction can operate
//! along: keys and values are stored per layer as a dense `[token][head][dim]`
//! block, and [`KvCacheTensor::retain_tokens`] physically removes token slots
//! from every layer and every head at once.
//!
//! Every token slot also carries its **absolute stream position**
//! ([`KvCacheTensor::positions`]). Positions are assigned at append time from a
//! monotonically increasing counter that is *never* rewound by eviction, so a
//! surviving token keeps the position it had in the original, uncompressed
//! sequence. This is what makes causal masking still correct after a
//! compression pass has punched holes in the middle of the cache: a query at
//! absolute position `p` attends to a surviving key iff that key's *original*
//! position is `<= p`, regardless of which slot it now occupies.

#![allow(
    // Cache/token/byte counts are `usize` and are routinely converted to
    // `f64` for ratios (memory-saved fraction, mean attention mass). The
    // sequence lengths involved (at most a few million tokens) are exactly
    // representable in `f64`, so the pedantic precision-loss warning is not
    // meaningful here.
    clippy::cast_precision_loss
)]

use std::collections::BTreeSet;

use thiserror::Error;

/// Result alias for every fallible operation in this module.
pub type KvResult<T> = Result<T, KvCompressionError>;

// ── KvCompressionError ───────────────────────────────────────────────────────

/// Everything that can go wrong while building, attending over, or compressing
/// a [`KvCacheTensor`].
///
/// Note in particular [`KvCompressionError::BudgetBelowFloor`]: a policy that
/// *mandates* keeping certain tokens (`StreamingLLM`'s attention sinks, H2O's
/// recent window, `SnapKV`'s observation window) has a hard floor on how small
/// its budget can be. Asking for a budget below that floor is a configuration
/// bug, and this module reports it as an error rather than silently truncating
/// the mandatory region — a silent truncation would quietly turn `StreamingLLM`
/// into a plain sliding window and destroy exactly the property the policy
/// exists to preserve.
#[derive(Debug, Clone, PartialEq, Eq, Error)]
pub enum KvCompressionError {
    /// A [`KvCompressionConfig`] field was structurally invalid (zero budget,
    /// even pooling kernel, out-of-range decay factor, ...).
    #[error("invalid kv-compression config: {reason}")]
    InvalidConfig {
        /// Human-readable explanation of what was invalid.
        reason: String,
    },
    /// The requested token budget is smaller than the number of tokens the
    /// selected policy is *required* to keep.
    #[error(
        "budget {budget} is below the {policy:?} policy's mandatory floor of {floor} token(s): {reason}"
    )]
    BudgetBelowFloor {
        /// The policy whose floor was violated.
        policy: KvEvictionPolicy,
        /// The configured budget.
        budget: usize,
        /// The minimum number of tokens this policy must retain.
        floor: usize,
        /// Which mandatory regions make up the floor.
        reason: String,
    },
    /// An operation that needs at least one cached token was performed on an
    /// empty cache (attention has no keys to attend to; compression has
    /// nothing to compress).
    #[error("kv cache is empty: {operation} requires at least one cached token")]
    EmptyCache {
        /// The operation that was attempted.
        operation: &'static str,
    },
    /// A layer index was out of range for the cache's `num_layers`.
    #[error("layer index {layer} is out of range for a cache with {num_layers} layer(s)")]
    LayerOutOfRange {
        /// The offending layer index.
        layer: usize,
        /// The cache's layer count.
        num_layers: usize,
    },
    /// A supplied buffer did not have the length the cache's geometry demands.
    #[error("{what}: expected {expected} element(s), got {actual}")]
    ShapeMismatch {
        /// Which buffer was wrong (`"key buffer"`, `"query buffer"`, ...).
        what: &'static str,
        /// The length the cache geometry requires.
        expected: usize,
        /// The length that was actually supplied.
        actual: usize,
    },
    /// A supplied buffer contained a `NaN` or an infinity.
    ///
    /// This module guarantees finite attention outputs for finite inputs (see
    /// the module docs), so non-finite inputs are rejected at the boundary
    /// rather than being allowed to poison a softmax.
    #[error("{what} contains a non-finite value at index {index}")]
    NonFiniteInput {
        /// Which buffer contained the bad value.
        what: &'static str,
        /// The offset of the first offending element.
        index: usize,
    },
    /// A token index (or a retained-token index) was out of range, unsorted,
    /// or duplicated.
    #[error("invalid token selection: {reason}")]
    InvalidTokenSelection {
        /// Human-readable explanation.
        reason: String,
    },
    /// The attention statistics were gathered for a different number of token
    /// slots than the cache currently holds.
    #[error(
        "attention stats track {stats_tokens} token slot(s) but the cache holds {cache_tokens}"
    )]
    StatsLengthMismatch {
        /// Number of slots the stats track.
        stats_tokens: usize,
        /// Number of slots the cache holds.
        cache_tokens: usize,
    },
    /// A score-driven policy ([`KvEvictionPolicy::H2O`],
    /// [`KvEvictionPolicy::SnapKv`]) was asked to evict with no recorded
    /// attention history to score with.
    ///
    /// Silently falling back to "keep the earliest tokens" (which is what a
    /// scoreless top-k with index tie-breaking would do) would be a
    /// fabrication: it would look like a heavy-hitter decision while being
    /// nothing of the kind.
    #[error(
        "the {policy:?} policy requires accumulated attention history, but no query step has been recorded"
    )]
    NoAttentionHistory {
        /// The policy that needs history.
        policy: KvEvictionPolicy,
    },
}

// ── KvEvictionPolicy ─────────────────────────────────────────────────────────

/// Which tokens a compression pass keeps when the cache exceeds its budget.
///
/// All four policies obey the same hard contract: after
/// [`crate::kv_cache_compression::KvCacheCompressor::compress`] returns `Ok`,
/// the cache holds **at most `budget`** token slots. They differ only in
/// *which* tokens they consider worth the budget.
#[derive(
    Debug, Clone, Copy, PartialEq, Eq, Hash, Default, serde::Serialize, serde::Deserialize,
)]
pub enum KvEvictionPolicy {
    /// **H2O — Heavy-Hitter Oracle** (Zhang et al., 2023).
    ///
    /// Keeps the union of a *recent window* (the last
    /// [`KvCompressionConfig::recent_window`] slots) and the top-`k` **heavy
    /// hitters**: the tokens that have accumulated the most attention mass
    /// across the query steps observed so far, where
    /// `k = budget - recent_window`.
    ///
    /// The saliency signal is the **column sum of the attention matrix**. For
    /// an attention matrix `A` with `A[q, t]` the weight query `q` placed on
    /// cached token `t`, token `t`'s accumulated score is `Σ_q A[q, t]`. This
    /// is exactly the total probability mass the model has *spent* on `t`, and
    /// because every row of `A` sums to one, the column sums partition the
    /// model's entire attention budget across the cached tokens. A token with
    /// a large column sum is one the model keeps coming back to; a token with
    /// a near-zero column sum has contributed nothing to any output computed so
    /// far, and evicting it perturbs those outputs by (at most) its weight
    /// times the spread of the value vectors.
    ///
    /// See [`KvScoreNormalization`] for the well-known *early-token bias* of the
    /// raw column sum, and what this module does about it.
    #[default]
    H2O,
    /// **`StreamingLLM` — attention sinks** (Xiao et al., 2023).
    ///
    /// Keeps the first [`KvCompressionConfig::sink_tokens`] tokens of the
    /// stream (the *attention sinks*) plus a sliding window of the last
    /// [`KvCompressionConfig::recent_window`] tokens, and evicts **everything
    /// in between**.
    ///
    /// The counterintuitive claim — the one this module's tests demonstrate
    /// rather than assume — is that the first handful of tokens must be kept
    /// *even though they are usually semantically worthless* (they are often
    /// just a BOS token). Softmax has no "attend to nothing" option: its
    /// weights are forced to sum to one, so a query with nothing relevant to
    /// look at must still dump its probability mass somewhere. Trained models
    /// exploit this by learning a few high-norm early keys that act as a
    /// dumping ground. Those sinks therefore absorb a large fraction of every
    /// row's mass. Evict them and that mass does not disappear — it is
    /// *renormalized onto the remaining tokens*, multiplying the weight of
    /// every surviving (largely irrelevant) token by `1 / (1 - sink_mass)` and
    /// corrupting the attention output far more than the sinks' own tiny value
    /// contribution ever did.
    StreamingLlm,
    /// **`SnapKV`** (Li et al., 2024).
    ///
    /// Uses the last [`KvCompressionConfig::observation_window`] query
    /// positions as an *observation window* that **votes** on which prefix
    /// tokens matter: it aggregates only those queries' attention over the
    /// prefix, optionally pools the resulting score vector along the token axis
    /// (see [`KvSnapPooling`]) so that selections form contiguous spans rather
    /// than isolated tokens, keeps the top-`k` prefix tokens, and always keeps
    /// the observation window itself.
    ///
    /// The difference from H2O is *whose* votes count. H2O scores a token by
    /// the attention it received from *every* query step in history — including
    /// steps issued long before the current question was asked. `SnapKV`'s
    /// insight is that the queries at the very end of the prompt (the ones
    /// closest to what is about to be generated) are far better predictors of
    /// what the *upcoming* generation will need, so it throws the older votes
    /// away instead of averaging them in.
    SnapKv,
    /// **Recency baseline** — keep the last `budget` tokens, evict everything
    /// older. No attention score is consulted at all.
    ///
    /// This is the honest straw man. It is what a cache does when it treats KV
    /// entries the way an LRU treats web pages: recency is the only signal.
    /// It exists so the score-driven policies have something to be measured
    /// *against* — and the module's headline test measures exactly that,
    /// showing that when the tokens carrying the attention mass are old, this
    /// policy throws them away and its attention output deviates by orders of
    /// magnitude more than H2O's or `SnapKV`'s.
    RecencyLru,
}

impl KvEvictionPolicy {
    /// Whether this policy needs accumulated attention statistics to make its
    /// decision.
    ///
    /// [`KvEvictionPolicy::StreamingLlm`] and [`KvEvictionPolicy::RecencyLru`]
    /// are purely positional and need none.
    #[must_use]
    pub const fn requires_attention_history(self) -> bool {
        matches!(self, Self::H2O | Self::SnapKv)
    }
}

// ── KvScoreNormalization ─────────────────────────────────────────────────────

/// How an accumulated attention column sum is turned into the saliency score a
/// heavy-hitter policy ranks by.
///
/// # The early-token bias, and why it is real
///
/// Under a causal mask, a token at position `t` is visible to every query at
/// position `>= t`. In a sequence of length `n` the *first* token therefore
/// competes in `n` softmax rows, while the *last* token competes in exactly
/// one. The raw column sum `Σ_q A[q, t]` — H2O's original signal — is a sum
/// over a number of terms that shrinks linearly with `t`. Even if every token
/// were equally interesting (say every row were uniform, `A[q, t] = 1/(q+1)`),
/// the raw column sums would come out as the harmonic partial sums
/// `Σ_{q>=t} 1/(q+1)`, which are strictly *decreasing* in `t`. The raw signal
/// thus systematically prefers old tokens over new ones for reasons that have
/// nothing to do with saliency and everything to do with the shape of the
/// causal mask.
#[derive(
    Debug, Clone, Copy, PartialEq, Eq, Hash, Default, serde::Serialize, serde::Deserialize,
)]
pub enum KvScoreNormalization {
    /// The raw, unnormalized column sum `Σ_q A[q, t]` — H2O exactly as
    /// published.
    ///
    /// Retained because it is the reference behaviour and because the bias is
    /// not always harmful: in a long prompt followed by short generation, the
    /// early tokens really do get more chances to prove themselves, and a
    /// policy that keeps them is not obviously wrong. Choose this when you want
    /// to reproduce the paper.
    Cumulative,
    /// The **mean attention mass per softmax row the token actually competed
    /// in**: `(Σ_q A[q, t]) / |{q : q >= t}|`.
    ///
    /// This is the default, and it is the fix for the bias described above.
    /// Dividing by the number of rows a token was *visible* in makes the score
    /// answer the scale-free question "when the model was allowed to look at
    /// this token, how much did it actually look?", which is invariant to how
    /// long the token has been in the cache.
    ///
    /// The mean has a bias of its own, in the opposite direction: the newest
    /// token has competed in a single row, so a single lucky row makes its mean
    /// look enormous. That bias is *structurally harmless here*, because every
    /// policy that consumes this score already keeps a recent window
    /// unconditionally and draws its heavy hitters only from the tokens
    /// *outside* it — so the recent, high-variance means are never the ones
    /// being ranked. This is why the two corrections compose rather than fight.
    #[default]
    MeanPerVisibleStep,
}

// ── KvSnapPooling ────────────────────────────────────────────────────────────

/// The pooling applied to `SnapKV`'s per-token vote vector before its top-`k`.
///
/// Pooling is what makes `SnapKV` select *spans* instead of *specks*. Attention
/// over a long prefix is spiky: a query may put all its mass on one token of an
/// informative phrase and almost none on the token beside it. Taking a raw
/// top-`k` of such a vector keeps a scatter of isolated tokens ripped out of
/// their context. Running a max-pool of width `w` first smears each peak across
/// its `w`-neighbourhood, so a token adjacent to a strong peak inherits a high
/// pooled score and is selected too — the retained set clusters into contiguous
/// spans around the informative regions.
///
/// Note the pooled score is used **only for ranking**. The tokens that get kept
/// are the original tokens at the selected indices; nothing is blurred, merged,
/// or synthesized.
#[derive(
    Debug, Clone, Copy, PartialEq, Eq, Hash, Default, serde::Serialize, serde::Deserialize,
)]
pub enum KvSnapPooling {
    /// No pooling: rank by the raw vote vector. Selects isolated peaks.
    None,
    /// Max-pool over a centred window of [`KvCompressionConfig::pooling_kernel`]
    /// tokens (the pooling used by the `SnapKV` paper).
    #[default]
    Max,
    /// Mean-pool over a centred window of
    /// [`KvCompressionConfig::pooling_kernel`] tokens.
    ///
    /// Smoother than max-pooling, and correspondingly harsher on genuinely
    /// isolated heavy hitters: a lone peak surrounded by zeros has its score
    /// divided by the kernel width, so a mean-pooled `SnapKV` prefers *dense
    /// regions of moderate interest* over *single points of high interest*.
    Mean,
}

// ── KvCompressionConfig ──────────────────────────────────────────────────────

/// Everything that parameterizes a compression pass.
///
/// # Budget floors
///
/// Each policy mandates a region it must keep; see [`Self::policy_floor`]. A
/// `budget` below that floor is rejected by [`Self::validate`] with
/// [`KvCompressionError::BudgetBelowFloor`].
#[derive(Debug, Clone, PartialEq, serde::Serialize, serde::Deserialize)]
pub struct KvCompressionConfig {
    /// Which tokens to keep. See [`KvEvictionPolicy`].
    pub policy: KvEvictionPolicy,
    /// The maximum number of token slots the cache may hold after a
    /// compression pass. This is a hard cap, asserted on every return path.
    pub budget: usize,
    /// Number of leading tokens [`KvEvictionPolicy::StreamingLlm`] pins as
    /// attention sinks.
    pub sink_tokens: usize,
    /// Size of the trailing window that [`KvEvictionPolicy::H2O`] and
    /// [`KvEvictionPolicy::StreamingLlm`] keep unconditionally.
    pub recent_window: usize,
    /// Number of trailing query positions whose attention
    /// [`KvEvictionPolicy::SnapKv`] uses to vote on the prefix (and, in the
    /// canonical prefill setting, the number of trailing tokens it pins).
    pub observation_window: usize,
    /// Pooling applied to `SnapKV`'s vote vector. See [`KvSnapPooling`].
    pub pooling: KvSnapPooling,
    /// Width of the pooling window, in tokens. Must be odd and non-zero so the
    /// window can be centred on the token being scored.
    pub pooling_kernel: usize,
    /// How accumulated attention becomes a saliency score. See
    /// [`KvScoreNormalization`].
    pub normalization: KvScoreNormalization,
    /// Optional per-query-step exponential forgetting factor in `(0, 1]`,
    /// applied to the accumulated scores before each new query step is folded
    /// in.
    ///
    /// `None` (the default) means no forgetting: a token's score is its total
    /// history. `Some(d)` makes a token's score an exponentially-weighted
    /// moving statistic with the mass from `s` steps ago down-weighted by
    /// `d^s`, so a token that was a heavy hitter a thousand steps ago but has
    /// been ignored since decays out of the heavy-hitter set instead of
    /// squatting in it forever. This is the "forgetting factor" refinement of
    /// H2O; `Some(1.0)` is exactly equivalent to `None`.
    pub score_decay: Option<f64>,
}

impl Default for KvCompressionConfig {
    fn default() -> Self {
        Self {
            policy: KvEvictionPolicy::H2O,
            budget: 128,
            sink_tokens: 4,
            recent_window: 32,
            observation_window: 16,
            pooling: KvSnapPooling::Max,
            pooling_kernel: 7,
            normalization: KvScoreNormalization::MeanPerVisibleStep,
            score_decay: None,
        }
    }
}

impl KvCompressionConfig {
    /// A config for `policy` with the given `budget`, all other fields left at
    /// their defaults.
    #[must_use]
    pub fn new(policy: KvEvictionPolicy, budget: usize) -> Self {
        Self {
            policy,
            budget,
            ..Self::default()
        }
    }

    /// Set the number of pinned attention-sink tokens.
    #[must_use]
    pub const fn with_sink_tokens(mut self, sink_tokens: usize) -> Self {
        self.sink_tokens = sink_tokens;
        self
    }

    /// Set the size of the unconditionally-kept trailing window.
    #[must_use]
    pub const fn with_recent_window(mut self, recent_window: usize) -> Self {
        self.recent_window = recent_window;
        self
    }

    /// Set the size of `SnapKV`'s observation window.
    #[must_use]
    pub const fn with_observation_window(mut self, observation_window: usize) -> Self {
        self.observation_window = observation_window;
        self
    }

    /// Set `SnapKV`'s pooling mode and kernel width (the width must be odd).
    #[must_use]
    pub const fn with_pooling(mut self, pooling: KvSnapPooling, pooling_kernel: usize) -> Self {
        self.pooling = pooling;
        self.pooling_kernel = pooling_kernel;
        self
    }

    /// Set how accumulated attention is normalized into a saliency score.
    #[must_use]
    pub const fn with_normalization(mut self, normalization: KvScoreNormalization) -> Self {
        self.normalization = normalization;
        self
    }

    /// Set the per-step exponential forgetting factor (`None` = remember
    /// everything).
    #[must_use]
    pub const fn with_score_decay(mut self, score_decay: Option<f64>) -> Self {
        self.score_decay = score_decay;
        self
    }

    /// The number of tokens the configured policy is *required* to keep,
    /// regardless of the budget.
    ///
    /// - [`KvEvictionPolicy::H2O`] must keep its recent window (heavy hitters
    ///   are chosen from *outside* it, so with a budget below the window there
    ///   is no room for a single heavy hitter and the policy degenerates into
    ///   a truncated sliding window).
    /// - [`KvEvictionPolicy::StreamingLlm`] must keep its sinks *and* its
    ///   recent window — dropping either half is not `StreamingLLM`.
    /// - [`KvEvictionPolicy::SnapKv`] must keep its observation window (the
    ///   window is both the voter and a pinned region).
    /// - [`KvEvictionPolicy::RecencyLru`] pins nothing; its floor is `1`
    ///   because a zero-token cache cannot be attended over.
    #[must_use]
    pub const fn policy_floor(&self) -> usize {
        match self.policy {
            KvEvictionPolicy::H2O => self.recent_window,
            KvEvictionPolicy::StreamingLlm => self.sink_tokens + self.recent_window,
            KvEvictionPolicy::SnapKv => self.observation_window,
            KvEvictionPolicy::RecencyLru => 1,
        }
    }

    /// A human-readable spelling of what makes up [`Self::policy_floor`], used
    /// in the [`KvCompressionError::BudgetBelowFloor`] message.
    fn floor_reason(&self) -> String {
        match self.policy {
            KvEvictionPolicy::H2O => format!("recent_window = {}", self.recent_window),
            KvEvictionPolicy::StreamingLlm => format!(
                "sink_tokens = {} + recent_window = {}",
                self.sink_tokens, self.recent_window
            ),
            KvEvictionPolicy::SnapKv => {
                format!("observation_window = {}", self.observation_window)
            }
            KvEvictionPolicy::RecencyLru => "a cache must retain at least one token".to_string(),
        }
    }

    /// Check the configuration for structural validity and budget-floor
    /// adherence.
    ///
    /// # Errors
    ///
    /// - [`KvCompressionError::InvalidConfig`] if the budget is zero, the
    ///   pooling kernel is zero or even (a centred window needs an odd width),
    ///   or `score_decay` is outside `(0, 1]` / non-finite.
    /// - [`KvCompressionError::BudgetBelowFloor`] if `budget < policy_floor()`.
    pub fn validate(&self) -> KvResult<()> {
        if self.budget == 0 {
            return Err(KvCompressionError::InvalidConfig {
                reason: "budget must be at least 1 token".to_string(),
            });
        }
        if self.pooling_kernel == 0 || self.pooling_kernel.is_multiple_of(2) {
            return Err(KvCompressionError::InvalidConfig {
                reason: format!(
                    "pooling_kernel must be odd and non-zero so it can be centred on a token, got {}",
                    self.pooling_kernel
                ),
            });
        }
        if let Some(decay) = self.score_decay
            && (!decay.is_finite() || decay <= 0.0 || decay > 1.0)
        {
            return Err(KvCompressionError::InvalidConfig {
                reason: format!("score_decay must lie in (0, 1], got {decay}"),
            });
        }
        // SnapKV's observation window is both its electorate and its pinned
        // region. With zero voters the vote vector is all-zero and the top-k
        // would silently fall through to its index tie-break, keeping the
        // *earliest* prefix tokens while claiming to have consulted the model's
        // own attention. That is a fabrication, so it is a configuration error.
        if self.policy == KvEvictionPolicy::SnapKv && self.observation_window == 0 {
            return Err(KvCompressionError::InvalidConfig {
                reason: "SnapKv requires observation_window >= 1: it is the set of queries that \
                         vote on the prefix, and an empty electorate cannot vote"
                    .to_string(),
            });
        }
        let floor = self.policy_floor();
        if self.budget < floor {
            return Err(KvCompressionError::BudgetBelowFloor {
                policy: self.policy,
                budget: self.budget,
                floor,
                reason: self.floor_reason(),
            });
        }
        Ok(())
    }
}

// ── KvCacheTensor ────────────────────────────────────────────────────────────

/// One layer's dense key/value storage: `[token][head][dim]`, token-major so a
/// whole token's `num_heads * head_dim` key (or value) block is contiguous and
/// appending a token is a plain `extend_from_slice`.
#[derive(Debug, Clone, PartialEq)]
struct KvLayerStore {
    keys: Vec<f32>,
    values: Vec<f32>,
}

/// A transformer attention KV cache with an explicit, evictable **token axis**.
///
/// Logical shape: `[num_layers, seq_len, num_heads, head_dim]` for each of the
/// keys and the values. The token axis (`seq_len`) is the one this module's
/// eviction policies cut along, and cutting it removes the token from *every*
/// layer and *every* head simultaneously — see the "Eviction granularity" note
/// in the module docs for why that uniformity is a deliberate (and
/// conservative) choice.
///
/// # Positions survive eviction
///
/// Each slot carries the token's **absolute position in the stream** (see the
/// file-level docs). Positions come from an internal counter that eviction
/// never rewinds, so after a compression pass the position vector is a
/// *strictly increasing but gappy* sequence like `[0, 1, 2, 3, 57, 91, 92, 93]`
/// — exactly the picture of "sinks, a couple of heavy hitters, and the recent
/// window" that these policies are supposed to produce.
#[derive(Debug, Clone, PartialEq)]
pub struct KvCacheTensor {
    num_layers: usize,
    num_heads: usize,
    head_dim: usize,
    /// Absolute stream position of each occupied slot; strictly increasing.
    positions: Vec<usize>,
    /// The next position to hand out. Monotonic; never rewound by eviction.
    next_position: usize,
    layers: Vec<KvLayerStore>,
}

impl KvCacheTensor {
    /// Create an empty cache with the given geometry.
    ///
    /// # Errors
    ///
    /// [`KvCompressionError::InvalidConfig`] if any dimension is zero.
    pub fn new(num_layers: usize, num_heads: usize, head_dim: usize) -> KvResult<Self> {
        if num_layers == 0 || num_heads == 0 || head_dim == 0 {
            return Err(KvCompressionError::InvalidConfig {
                reason: format!(
                    "cache geometry must be non-degenerate, got num_layers = {num_layers}, num_heads = {num_heads}, head_dim = {head_dim}"
                ),
            });
        }
        Ok(Self {
            num_layers,
            num_heads,
            head_dim,
            positions: Vec::new(),
            next_position: 0,
            layers: (0..num_layers)
                .map(|_| KvLayerStore {
                    keys: Vec::new(),
                    values: Vec::new(),
                })
                .collect(),
        })
    }

    /// Number of transformer layers.
    #[must_use]
    pub const fn num_layers(&self) -> usize {
        self.num_layers
    }

    /// Number of attention heads per layer.
    #[must_use]
    pub const fn num_heads(&self) -> usize {
        self.num_heads
    }

    /// Dimension of each attention head (the `d` in `1 / sqrt(d)`).
    #[must_use]
    pub const fn head_dim(&self) -> usize {
        self.head_dim
    }

    /// Number of token slots currently held.
    #[must_use]
    pub fn seq_len(&self) -> usize {
        self.positions.len()
    }

    /// Whether the cache holds no tokens.
    #[must_use]
    pub fn is_empty(&self) -> bool {
        self.positions.is_empty()
    }

    /// The absolute stream position of each occupied slot, in slot order
    /// (strictly increasing, possibly gappy after eviction).
    #[must_use]
    pub fn positions(&self) -> &[usize] {
        &self.positions
    }

    /// The position that the next appended token will receive. Unaffected by
    /// eviction.
    #[must_use]
    pub const fn next_position(&self) -> usize {
        self.next_position
    }

    /// The number of `f32` elements in one token's per-layer key (or value)
    /// block: `num_heads * head_dim`.
    #[must_use]
    pub const fn per_layer_token_stride(&self) -> usize {
        self.num_heads * self.head_dim
    }

    /// The number of `f32` elements one token contributes to
    /// [`Self::append_token`]'s `keys` (or `values`) argument:
    /// `num_layers * num_heads * head_dim`.
    #[must_use]
    pub const fn token_buffer_len(&self) -> usize {
        self.num_layers * self.num_heads * self.head_dim
    }

    /// Append one token to the cache, assigning it the next absolute stream
    /// position.
    ///
    /// `keys` and `values` are each laid out `[layer][head][dim]` and must be
    /// exactly [`Self::token_buffer_len`] elements long.
    ///
    /// # Errors
    ///
    /// - [`KvCompressionError::ShapeMismatch`] if either buffer has the wrong
    ///   length.
    /// - [`KvCompressionError::NonFiniteInput`] if either buffer contains a
    ///   `NaN` or an infinity. Finiteness is enforced *here*, at the boundary,
    ///   which is what lets the attention kernel guarantee a finite output (see
    ///   the module docs).
    pub fn append_token(&mut self, keys: &[f32], values: &[f32]) -> KvResult<usize> {
        let expected = self.token_buffer_len();
        if keys.len() != expected {
            return Err(KvCompressionError::ShapeMismatch {
                what: "key buffer",
                expected,
                actual: keys.len(),
            });
        }
        if values.len() != expected {
            return Err(KvCompressionError::ShapeMismatch {
                what: "value buffer",
                expected,
                actual: values.len(),
            });
        }
        check_finite(keys, "key buffer")?;
        check_finite(values, "value buffer")?;

        let stride = self.per_layer_token_stride();
        for (layer_idx, layer) in self.layers.iter_mut().enumerate() {
            let start = layer_idx * stride;
            let end = start + stride;
            layer.keys.extend_from_slice(&keys[start..end]);
            layer.values.extend_from_slice(&values[start..end]);
        }

        let position = self.next_position;
        self.positions.push(position);
        self.next_position += 1;
        Ok(position)
    }

    /// The key vector of `head` for the token in slot `token` of `layer`, or
    /// `None` if any index is out of range.
    #[must_use]
    pub fn key(&self, layer: usize, token: usize, head: usize) -> Option<&[f32]> {
        self.head_slice(layer, token, head, true)
    }

    /// The value vector of `head` for the token in slot `token` of `layer`, or
    /// `None` if any index is out of range.
    #[must_use]
    pub fn value(&self, layer: usize, token: usize, head: usize) -> Option<&[f32]> {
        self.head_slice(layer, token, head, false)
    }

    fn head_slice(
        &self,
        layer: usize,
        token: usize,
        head: usize,
        want_key: bool,
    ) -> Option<&[f32]> {
        if layer >= self.num_layers || token >= self.seq_len() || head >= self.num_heads {
            return None;
        }
        let store = self.layers.get(layer)?;
        let buffer = if want_key { &store.keys } else { &store.values };
        let start = token * self.per_layer_token_stride() + head * self.head_dim;
        buffer.get(start..start + self.head_dim)
    }

    /// The whole `[head][dim]` key block of one slot in one layer.
    ///
    /// Crate-internal fast path used by the attention kernel, which has already
    /// validated `layer` and iterates `token` over `0..seq_len`.
    pub(super) fn layer_key_block(&self, layer: usize, token: usize) -> &[f32] {
        let stride = self.per_layer_token_stride();
        let start = token * stride;
        &self.layers[layer].keys[start..start + stride]
    }

    /// The whole `[head][dim]` value block of one slot in one layer. See
    /// [`Self::layer_key_block`].
    pub(super) fn layer_value_block(&self, layer: usize, token: usize) -> &[f32] {
        let stride = self.per_layer_token_stride();
        let start = token * stride;
        &self.layers[layer].values[start..start + stride]
    }

    /// Physically drop every slot not listed in `keep`, in every layer and
    /// every head at once.
    ///
    /// `keep` must be strictly increasing and in range. The surviving slots
    /// stay in stream order and keep their absolute positions.
    ///
    /// # Errors
    ///
    /// [`KvCompressionError::InvalidTokenSelection`] if `keep` is unsorted, has
    /// duplicates, or names a slot that does not exist.
    pub fn retain_tokens(&mut self, keep: &[usize]) -> KvResult<()> {
        let seq_len = self.seq_len();
        let mut previous: Option<usize> = None;
        for &index in keep {
            if index >= seq_len {
                return Err(KvCompressionError::InvalidTokenSelection {
                    reason: format!(
                        "slot {index} is out of range for a cache of {seq_len} token(s)"
                    ),
                });
            }
            if let Some(prev) = previous
                && index <= prev
            {
                return Err(KvCompressionError::InvalidTokenSelection {
                    reason: format!(
                        "retained slots must be strictly increasing, got {index} after {prev}"
                    ),
                });
            }
            previous = Some(index);
        }

        // Nothing to do: `keep` already names every slot, in order.
        if keep.len() == seq_len {
            return Ok(());
        }

        let stride = self.per_layer_token_stride();
        for layer in &mut self.layers {
            let mut new_keys = Vec::with_capacity(keep.len() * stride);
            let mut new_values = Vec::with_capacity(keep.len() * stride);
            for &index in keep {
                let start = index * stride;
                new_keys.extend_from_slice(&layer.keys[start..start + stride]);
                new_values.extend_from_slice(&layer.values[start..start + stride]);
            }
            layer.keys = new_keys;
            layer.values = new_values;
        }
        self.positions = keep.iter().map(|&index| self.positions[index]).collect();
        Ok(())
    }

    /// Bytes of `f32` payload currently held (keys **and** values, all layers,
    /// all heads).
    #[must_use]
    pub fn memory_bytes(&self) -> usize {
        2 * self.num_layers
            * self.seq_len()
            * self.per_layer_token_stride()
            * std::mem::size_of::<f32>()
    }

    /// Bytes one additional token costs across the whole cache.
    #[must_use]
    pub fn bytes_per_token(&self) -> usize {
        2 * self.num_layers * self.per_layer_token_stride() * std::mem::size_of::<f32>()
    }

    /// Validate that `layer` names an existing layer.
    ///
    /// # Errors
    ///
    /// [`KvCompressionError::LayerOutOfRange`] if it does not.
    pub(super) fn check_layer(&self, layer: usize) -> KvResult<()> {
        if layer >= self.num_layers {
            return Err(KvCompressionError::LayerOutOfRange {
                layer,
                num_layers: self.num_layers,
            });
        }
        Ok(())
    }
}

/// Reject `NaN`/infinity anywhere in `buffer`.
///
/// # Errors
///
/// [`KvCompressionError::NonFiniteInput`] naming the offset of the first
/// offending element.
pub(super) fn check_finite(buffer: &[f32], what: &'static str) -> KvResult<()> {
    for (index, value) in buffer.iter().enumerate() {
        if !value.is_finite() {
            return Err(KvCompressionError::NonFiniteInput { what, index });
        }
    }
    Ok(())
}

// ── KvCompressionReport ──────────────────────────────────────────────────────

/// What a compression pass did.
///
/// [`Self::within_budget`] is the module's central invariant made observable:
/// it is `true` on every successful pass, for every policy and every budget,
/// and the compressor asserts it before returning.
#[derive(Debug, Clone, PartialEq, serde::Serialize, serde::Deserialize)]
pub struct KvCompressionReport {
    /// The policy that made the decision.
    pub policy: KvEvictionPolicy,
    /// Token slots held before the pass.
    pub tokens_before: usize,
    /// Token slots held after the pass.
    pub tokens_after: usize,
    /// `tokens_before - tokens_after`.
    pub tokens_evicted: usize,
    /// The budget the pass was run under.
    pub budget: usize,
    /// `tokens_after <= budget`. Always `true` on a successful pass.
    pub within_budget: bool,
    /// Absolute stream positions of the surviving tokens, ascending. Reading
    /// this is the direct way to see *which* tokens a policy considered worth
    /// keeping.
    pub retained_positions: Vec<usize>,
    /// Absolute stream positions of the evicted tokens, ascending.
    pub evicted_positions: Vec<usize>,
    /// Payload bytes before the pass.
    pub bytes_before: usize,
    /// Payload bytes after the pass.
    pub bytes_after: usize,
}

impl KvCompressionReport {
    /// Fraction of the cache's bytes reclaimed, in `[0, 1]`. `0.0` when the
    /// cache was already within budget (nothing was evicted), `0.0` for an
    /// already-empty cache.
    #[must_use]
    pub fn memory_saved_ratio(&self) -> f64 {
        if self.bytes_before == 0 {
            return 0.0;
        }
        (self.bytes_before - self.bytes_after) as f64 / self.bytes_before as f64
    }

    /// Fraction of token slots evicted, in `[0, 1]`.
    #[must_use]
    pub fn eviction_ratio(&self) -> f64 {
        if self.tokens_before == 0 {
            return 0.0;
        }
        self.tokens_evicted as f64 / self.tokens_before as f64
    }

    /// Whether the pass left the cache untouched (the cache already fitted the
    /// budget).
    #[must_use]
    pub const fn is_noop(&self) -> bool {
        self.tokens_evicted == 0
    }
}

// ── Selection helpers shared by the policies ─────────────────────────────────

/// Turn a set of retained slot indices into the strictly-increasing vector the
/// rest of the module (and [`KvCacheTensor::retain_tokens`]) expects.
pub(super) fn ordered_selection(selected: &BTreeSet<usize>) -> Vec<usize> {
    selected.iter().copied().collect()
}

/// Rank `candidates` by `score` descending, break ties by ascending slot index,
/// and take the best `k`.
///
/// Float ordering is [`f64::total_cmp`] — a total order, so the sort is stable
/// and well-defined even in the presence of `-0.0` (and would be even for
/// `NaN`, which cannot occur here because scores are sums of softmax weights).
pub(super) fn top_k_by_score(candidates: &[usize], score: &[f64], k: usize) -> Vec<usize> {
    let mut ranked: Vec<usize> = candidates.to_vec();
    ranked.sort_by(|&left, &right| {
        let left_score = score.get(left).copied().unwrap_or(0.0);
        let right_score = score.get(right).copied().unwrap_or(0.0);
        right_score
            .total_cmp(&left_score)
            .then_with(|| left.cmp(&right))
    });
    ranked.truncate(k);
    ranked
}
