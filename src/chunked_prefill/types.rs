//! Budgets, chunk plans, geometry and errors.
//!
//! The three load-bearing types here are [`TokenBudget`] (the `B` of the
//! stall-free bound), [`PrefillChunk`] (one token-budgeted slice of a prompt)
//! and [`PrefillPlan`] (a prompt's complete decomposition into chunks). A plan
//! is pure arithmetic — it knows nothing about tensors — which is what lets the
//! cost identities below be *proved* rather than measured.

use serde::{Deserialize, Serialize};
use thiserror::Error;

use crate::kv_cache_compression::KvCompressionError;

// ── Errors ───────────────────────────────────────────────────────────────────

/// Everything chunked prefill can refuse to do.
///
/// Two of these deserve a word, because they exist to make a *silent* wrong
/// answer impossible:
///
/// * [`Self::MaskDivergence`] fires when the piecewise mask this module derives
///   from segment geometry disagrees, in even one cell, with the causal mask the
///   attention kernel derives from absolute stream positions. Those two masks
///   are computed from *different information*, so their agreement is a real
///   check and not a tautology — and it is the check that catches an off-by-one
///   at a chunk boundary before it becomes a subtly wrong tensor.
/// * [`Self::DecodeCapExceeded`] fires when more sequences are decoding than the
///   budget reserves room for. This module does **not** perform admission
///   control (that is `request_scheduling`'s job); it refuses, loudly, rather
///   than quietly dropping a decode from an iteration — which is precisely the
///   stall it exists to prevent.
#[derive(Debug, Clone, PartialEq, Eq, Error)]
pub enum ChunkedPrefillError {
    /// A configuration value is outside its permitted range.
    #[error("invalid chunked-prefill configuration: {reason}")]
    InvalidConfig {
        /// What was wrong.
        reason: String,
    },

    /// A prompt of zero tokens has no prefill to chunk.
    #[error("a prompt must contain at least one token")]
    EmptyPrompt,

    /// The chunk lengths do not add up to the prompt length.
    #[error("chunk lengths sum to {actual} but the prompt has {expected} token(s)")]
    ChunkLengthMismatch {
        /// The prompt's token count.
        expected: usize,
        /// What the chunk lengths summed to.
        actual: usize,
    },

    /// A chunk carries no tokens. Every chunk must make progress, or the
    /// prefill would never terminate.
    #[error("chunk {ordinal} carries no tokens; every chunk must make progress")]
    EmptyChunk {
        /// The offending chunk's ordinal within the plan.
        ordinal: usize,
    },

    /// A chunk is larger than the token budget allows.
    #[error("chunk {ordinal} carries {tokens} token(s), over the {budget}-token iteration budget")]
    ChunkOverBudget {
        /// The offending chunk's ordinal within the plan.
        ordinal: usize,
        /// The chunk's token count.
        tokens: usize,
        /// The budget it broke.
        budget: usize,
    },

    /// A prompt token index does not exist.
    #[error("token {token} is out of range for a prompt of {prompt_tokens} token(s)")]
    TokenOutOfRange {
        /// The offending index.
        token: usize,
        /// The prompt's token count.
        prompt_tokens: usize,
    },

    /// A layer index does not exist.
    #[error("layer {layer} is out of range for a model with {num_layers} layer(s)")]
    LayerOutOfRange {
        /// The offending index.
        layer: usize,
        /// The model's layer count.
        num_layers: usize,
    },

    /// A tensor buffer has the wrong length.
    #[error("{what} holds {actual} element(s), expected {expected}")]
    ShapeMismatch {
        /// Which buffer.
        what: &'static str,
        /// How many elements it should have held.
        expected: usize,
        /// How many it actually held.
        actual: usize,
    },

    /// The model's geometry and the cache's geometry disagree.
    #[error(
        "the model is [{model_layers} layer(s), {model_heads} head(s), head_dim {model_head_dim}] \
         but the cache is [{cache_layers} layer(s), {cache_heads} head(s), head_dim {cache_head_dim}]"
    )]
    GeometryMismatch {
        /// Layers the model has.
        model_layers: usize,
        /// Heads the model has.
        model_heads: usize,
        /// Head dimension the model has.
        model_head_dim: usize,
        /// Layers the cache has.
        cache_layers: usize,
        /// Heads the cache has.
        cache_heads: usize,
        /// Head dimension the cache has.
        cache_head_dim: usize,
    },

    /// A tensor buffer contains a `NaN` or an infinity.
    #[error("{what} holds a non-finite value at offset {index}")]
    NonFiniteInput {
        /// Which buffer.
        what: &'static str,
        /// Offset of the first offending element.
        index: usize,
    },

    /// The plan and the model disagree about how long the prompt is.
    #[error("the plan covers {plan_tokens} prompt token(s) but the model holds {model_tokens}")]
    PromptLengthMismatch {
        /// Tokens the plan covers.
        plan_tokens: usize,
        /// Tokens the model holds.
        model_tokens: usize,
    },

    /// The piecewise mask disagrees with the kernel's absolute-position mask.
    ///
    /// This is the module's central invariant, and it is checked on every chunk.
    #[error(
        "piecewise mask disagrees with the kernel at (query {query}, key {key}): \
         segment geometry says visible = {geometry}, the absolute-position mask says {kernel}"
    )]
    MaskDivergence {
        /// The query row the two masks disagreed on.
        query: usize,
        /// The key column the two masks disagreed on.
        key: usize,
        /// What the segment geometry claimed.
        geometry: bool,
        /// What the kernel's absolute-position mask claimed.
        kernel: bool,
    },

    /// The piecewise mask and the kernel's mask are not even the same shape.
    #[error(
        "piecewise mask is [{expected_queries} query(s) x {expected_keys} key(s)] \
         but the kernel produced [{actual_queries} x {actual_keys}]"
    )]
    MaskShapeDivergence {
        /// Query rows the geometry expected.
        expected_queries: usize,
        /// Key columns the geometry expected.
        expected_keys: usize,
        /// Query rows the kernel produced.
        actual_queries: usize,
        /// Key columns the kernel produced.
        actual_keys: usize,
    },

    /// A chunk's tokens did not land on the absolute stream positions the
    /// bookkeeping expected.
    #[error(
        "chunk token {offset} landed on absolute position {actual}, but the chunk's \
         positions must be consecutive from {expected}"
    )]
    PositionDivergence {
        /// Offset of the token within its chunk.
        offset: usize,
        /// The position the bookkeeping expected.
        expected: usize,
        /// The position the cache actually assigned.
        actual: usize,
    },

    /// More sequences are decoding than the token budget admits per iteration.
    #[error(
        "{running} sequence(s) are decoding but the token budget admits only {cap} decode \
         token(s) per iteration; admission control is `request_scheduling`'s job, and \
         silently dropping a decode is the very stall this module prevents"
    )]
    DecodeCapExceeded {
        /// How many sequences wanted to decode.
        running: usize,
        /// How many the budget admits.
        cap: usize,
    },

    /// The schedule failed to make progress. Unreachable — every iteration is
    /// proved to consume at least one token — and kept as a tripwire.
    #[error("the schedule did not terminate within its {limit}-iteration bound")]
    ScheduleDidNotTerminate {
        /// The bound that was exceeded.
        limit: usize,
    },

    /// The underlying KV cache rejected an operation.
    #[error(transparent)]
    Cache(#[from] KvCompressionError),
}

/// The result type every fallible operation in this module returns.
pub type ChunkedPrefillResult<T> = Result<T, ChunkedPrefillError>;

// ── Widening ─────────────────────────────────────────────────────────────────

/// Widen a count to `u64`.
///
/// Cell and slot-visit counters are quadratic in the prompt length, so a
/// 32-bit `usize` (`wasm32`) overflows them at prompt lengths a serving system
/// reaches routinely: a 100 000-token prompt has 5 x 10^9 mask cells, well past
/// `u32::MAX`. `usize` is at most 64 bits on every supported target, so the
/// widening is lossless.
#[allow(
    // Lossless on every target this crate builds for; see above.
    clippy::cast_possible_truncation,
    clippy::cast_lossless
)]
pub(super) const fn widen(value: usize) -> u64 {
    value as u64
}

// ── TokenBudget ──────────────────────────────────────────────────────────────

/// The per-iteration token budget `B`, plus the decode admission cap that keeps
/// room inside it for a prefill chunk.
///
/// `B` is the number of tokens a single scheduling iteration is allowed to
/// process — prefill tokens and decode tokens together. It is the *only* knob
/// in this module, and it buys exactly one thing:
///
/// > **no iteration ever costs more than `B` tokens, whatever the prompt
/// > length.**
///
/// The decode cap is what makes that promise keepable. Decodes have priority (a
/// decode that is not scheduled *is* a stall), so if the decode batch could grow
/// to `B` the prefill chunk would be squeezed to zero and the prefill would
/// never finish. Requiring `max_decode_slots < total` reserves
/// [`Self::prefill_reserve`] `>= 1` tokens for the chunk in *every* iteration,
/// which is what makes prefill progress — and therefore termination — a theorem
/// rather than a hope.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub struct TokenBudget {
    total: usize,
    max_decode_slots: usize,
}

impl TokenBudget {
    /// A budget of `total` tokens per iteration, of which at most
    /// `max_decode_slots` may be spent on decodes.
    ///
    /// # Errors
    ///
    /// [`ChunkedPrefillError::InvalidConfig`] if `total` is zero, or if
    /// `max_decode_slots >= total` — a budget with no prefill reserve cannot
    /// guarantee prefill progress, so it is rejected rather than silently
    /// deadlocking.
    pub fn new(total: usize, max_decode_slots: usize) -> ChunkedPrefillResult<Self> {
        if total == 0 {
            return Err(ChunkedPrefillError::InvalidConfig {
                reason: "the iteration token budget must be at least 1".to_string(),
            });
        }
        if max_decode_slots >= total {
            return Err(ChunkedPrefillError::InvalidConfig {
                reason: format!(
                    "the decode cap ({max_decode_slots}) must be strictly below the token budget \
                     ({total}), or no tokens are left for the prefill chunk and prefill never \
                     finishes"
                ),
            });
        }
        Ok(Self {
            total,
            max_decode_slots,
        })
    }

    /// `B` — the maximum number of tokens one scheduling iteration may process.
    #[must_use]
    pub const fn total(&self) -> usize {
        self.total
    }

    /// The maximum number of decode tokens one iteration admits.
    #[must_use]
    pub const fn max_decode_slots(&self) -> usize {
        self.max_decode_slots
    }

    /// `total - max_decode_slots` — the tokens *guaranteed* to be available to
    /// the prefill chunk in every iteration. Always `>= 1`.
    #[must_use]
    pub const fn prefill_reserve(&self) -> usize {
        self.total - self.max_decode_slots
    }

    /// The largest prefill chunk this budget permits: the whole budget, in an
    /// iteration with no decodes to co-schedule.
    #[must_use]
    pub const fn max_chunk_tokens(&self) -> usize {
        self.total
    }
}

impl Default for TokenBudget {
    /// A 512-token iteration budget with room for 128 decodes — the middle of
    /// the range the `Sarathi-Serve` paper evaluates, leaving a 384-token
    /// prefill reserve.
    fn default() -> Self {
        Self {
            total: 512,
            max_decode_slots: 128,
        }
    }
}

// ── PrefillGeometry ──────────────────────────────────────────────────────────

/// The `[num_layers, num_heads, head_dim]` shape shared by a model and the KV
/// cache it writes into.
///
/// The token axis is deliberately absent: it is the axis chunking cuts along,
/// and it is the one thing that differs between a chunk, a prompt and a cache.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub struct PrefillGeometry {
    num_layers: usize,
    num_heads: usize,
    head_dim: usize,
}

impl PrefillGeometry {
    /// A geometry with the given shape.
    ///
    /// # Errors
    ///
    /// [`ChunkedPrefillError::InvalidConfig`] if any dimension is zero.
    pub fn new(num_layers: usize, num_heads: usize, head_dim: usize) -> ChunkedPrefillResult<Self> {
        if num_layers == 0 || num_heads == 0 || head_dim == 0 {
            return Err(ChunkedPrefillError::InvalidConfig {
                reason: format!(
                    "the model geometry must be non-degenerate, got num_layers = {num_layers}, \
                     num_heads = {num_heads}, head_dim = {head_dim}"
                ),
            });
        }
        Ok(Self {
            num_layers,
            num_heads,
            head_dim,
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

    /// Dimension of each attention head.
    #[must_use]
    pub const fn head_dim(&self) -> usize {
        self.head_dim
    }

    /// `num_heads * head_dim` — one token's key (or value) block within **one**
    /// layer, and the length of one token's per-layer query vector.
    #[must_use]
    pub const fn layer_stride(&self) -> usize {
        self.num_heads * self.head_dim
    }

    /// `num_layers * num_heads * head_dim` — one token's key (or value) buffer
    /// across **all** layers, which is exactly what
    /// [`KvCacheTensor::append_token`](crate::kv_cache_compression::KvCacheTensor::append_token)
    /// consumes.
    #[must_use]
    pub const fn token_stride(&self) -> usize {
        self.num_layers * self.layer_stride()
    }
}

// ── PrefillChunk ─────────────────────────────────────────────────────────────

/// One token-budgeted slice of a prompt's prefill.
///
/// A chunk is described entirely by *where it starts in the prompt* and *how
/// many prompt tokens it carries*. It says nothing about where its keys and
/// values will live — that is `continuous_batching`'s concern — and nothing
/// about which request it belongs to — that is `request_scheduling`'s.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub struct PrefillChunk {
    ordinal: usize,
    start: usize,
    tokens: usize,
}

impl PrefillChunk {
    /// A chunk at its three coordinates.
    ///
    /// Crate-internal: the public paths are [`PrefillPlan::uniform`],
    /// [`PrefillPlan::explicit`] and [`PrefillPlan::one_shot`], all of which
    /// validate that the chunks tile the prompt. The scheduler already knows a
    /// chunk's coordinates by construction and does not need re-deriving.
    pub(super) const fn at(ordinal: usize, start: usize, tokens: usize) -> Self {
        Self {
            ordinal,
            start,
            tokens,
        }
    }

    /// The chunk's 0-based ordinal within its plan.
    #[must_use]
    pub const fn ordinal(&self) -> usize {
        self.ordinal
    }

    /// Offset of the chunk's first token **within the prompt** (not within the
    /// KV cache, and not an absolute stream position — see
    /// [`crate::chunked_prefill::PrefillChunkReport::stream_positions`] for
    /// those, which differ whenever the cache already holds a shared prefix or
    /// has been compressed).
    #[must_use]
    pub const fn start(&self) -> usize {
        self.start
    }

    /// How many prompt tokens the chunk carries. Always `>= 1`.
    #[must_use]
    pub const fn token_count(&self) -> usize {
        self.tokens
    }

    /// One past the chunk's last prompt token.
    #[must_use]
    pub const fn end(&self) -> usize {
        self.start + self.tokens
    }
}

// ── PrefillPlan ──────────────────────────────────────────────────────────────

/// A prompt's complete decomposition into chunks: a *composition* of the prompt
/// length into positive parts.
///
/// # The two cost identities
///
/// A plan is pure arithmetic, so the cost of executing it can be *derived*
/// rather than benchmarked — and the two derivations say opposite things, which
/// is the whole tension of chunked prefill.
///
/// **1. Chunking is free in arithmetic.** The number of `Q·K` dot products is
/// the number of visible mask cells, and a query at absolute position `p` sees
/// exactly the keys at positions `<= p` — a fact about the *causal mask*, which
/// no chunk boundary can change. So for a prompt of `P` tokens over a cache
/// already holding `c` committed keys,
///
/// ```text
/// attention_cells = Σ_{j=0}^{P-1} (c + j + 1) = P·c + P(P+1)/2
/// ```
///
/// **independently of how the prompt is chunked.** Every composition of `P`
/// costs exactly the same number of dot products. (See
/// [`Self::attention_cells`].)
///
/// **2. Chunking is not free in memory traffic.** Each chunk invocation must
/// have the *whole* committed cache resident, so it "visits" `c + start + len`
/// KV slots; summed over `m` uniform chunks of size `B` covering `P = m·B`
/// tokens with `c = 0`, that is
///
/// ```text
/// kv_slot_visits = Σ_{i=1}^{m} i·B = P·(m + 1) / 2      →  amplification (m+1)/2
/// ```
///
/// against `P` for a one-shot prefill. Eight chunks means **4.5x** the KV-cache
/// traffic. (See [`Self::kv_slot_visits`] and [`Self::kv_read_amplification`].)
///
/// That is the trade the token budget buys: halving `B` halves the worst-case
/// time-between-tokens and *raises* KV traffic by roughly the same factor. The
/// arithmetic is identical either way, so a `FLOP` count alone would say
/// chunking is free — and it is not.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct PrefillPlan {
    prompt_tokens: usize,
    chunks: Vec<PrefillChunk>,
}

impl PrefillPlan {
    /// Chunks of exactly `chunk_tokens` tokens, with the tail taking whatever
    /// remains. This is what a token budget produces when no decodes are
    /// competing for it.
    ///
    /// # Errors
    ///
    /// - [`ChunkedPrefillError::EmptyPrompt`] if `prompt_tokens` is zero.
    /// - [`ChunkedPrefillError::InvalidConfig`] if `chunk_tokens` is zero (a
    ///   chunk that carries nothing never finishes the prompt).
    pub fn uniform(prompt_tokens: usize, chunk_tokens: usize) -> ChunkedPrefillResult<Self> {
        if prompt_tokens == 0 {
            return Err(ChunkedPrefillError::EmptyPrompt);
        }
        if chunk_tokens == 0 {
            return Err(ChunkedPrefillError::InvalidConfig {
                reason: "a chunk must carry at least one token".to_string(),
            });
        }
        let mut chunks = Vec::new();
        let mut start = 0usize;
        while start < prompt_tokens {
            let tokens = chunk_tokens.min(prompt_tokens - start);
            chunks.push(PrefillChunk {
                ordinal: chunks.len(),
                start,
                tokens,
            });
            start += tokens;
        }
        Ok(Self {
            prompt_tokens,
            chunks,
        })
    }

    /// Exactly these chunk lengths, in order.
    ///
    /// This is the constructor the scheduler's residual chunk sizes flow
    /// through, and it is the one the exhaustive equivalence test drives with
    /// every composition of the prompt length.
    ///
    /// # Errors
    ///
    /// - [`ChunkedPrefillError::EmptyPrompt`] if `chunk_lengths` is empty.
    /// - [`ChunkedPrefillError::EmptyChunk`] if any length is zero.
    pub fn explicit(chunk_lengths: &[usize]) -> ChunkedPrefillResult<Self> {
        if chunk_lengths.is_empty() {
            return Err(ChunkedPrefillError::EmptyPrompt);
        }
        let mut chunks = Vec::with_capacity(chunk_lengths.len());
        let mut start = 0usize;
        for (ordinal, &tokens) in chunk_lengths.iter().enumerate() {
            if tokens == 0 {
                return Err(ChunkedPrefillError::EmptyChunk { ordinal });
            }
            chunks.push(PrefillChunk {
                ordinal,
                start,
                tokens,
            });
            start += tokens;
        }
        Ok(Self {
            prompt_tokens: start,
            chunks,
        })
    }

    /// The whole prompt in one chunk: the unchunked baseline this module exists
    /// to beat. Executing it reproduces a classical prefill exactly, because a
    /// one-chunk plan *is* a classical prefill.
    ///
    /// # Errors
    ///
    /// [`ChunkedPrefillError::EmptyPrompt`] if `prompt_tokens` is zero.
    pub fn one_shot(prompt_tokens: usize) -> ChunkedPrefillResult<Self> {
        Self::explicit(&[prompt_tokens]).map_err(|error| match error {
            ChunkedPrefillError::EmptyChunk { .. } => ChunkedPrefillError::EmptyPrompt,
            other => other,
        })
    }

    /// The prompt's token count.
    #[must_use]
    pub const fn prompt_tokens(&self) -> usize {
        self.prompt_tokens
    }

    /// The chunks, in execution order.
    #[must_use]
    pub fn chunks(&self) -> &[PrefillChunk] {
        &self.chunks
    }

    /// How many chunks the prompt is split into. Always `>= 1`.
    #[must_use]
    pub fn num_chunks(&self) -> usize {
        self.chunks.len()
    }

    /// The largest chunk in the plan.
    #[must_use]
    pub fn max_chunk_tokens(&self) -> usize {
        self.chunks
            .iter()
            .map(PrefillChunk::token_count)
            .max()
            .unwrap_or(0)
    }

    /// The smallest chunk in the plan. Small tail chunks are the price of a
    /// prompt length that is not a multiple of the budget.
    #[must_use]
    pub fn min_chunk_tokens(&self) -> usize {
        self.chunks
            .iter()
            .map(PrefillChunk::token_count)
            .min()
            .unwrap_or(0)
    }

    /// The chunk lengths, in order.
    #[must_use]
    pub fn chunk_lengths(&self) -> Vec<usize> {
        self.chunks.iter().map(PrefillChunk::token_count).collect()
    }

    /// The number of `Q·K` dot products executing this plan performs, over a
    /// cache that already holds `prefix_tokens` committed keys.
    ///
    /// This is the *exact* count of visible mask cells, and by identity 1 in the
    /// type docs it is `P·prefix + P(P+1)/2` — **the same for every plan over
    /// the same prompt**, one-shot included. Chunking moves work between
    /// iterations; it does not create any.
    #[must_use]
    pub fn attention_cells(&self, prefix_tokens: usize) -> u64 {
        self.chunks
            .iter()
            .map(|chunk| {
                let committed = widen(prefix_tokens) + widen(chunk.start());
                let tokens = widen(chunk.token_count());
                tokens * committed + tokens * (tokens + 1) / 2
            })
            .sum()
    }

    /// The number of KV-cache slots this plan requires to be resident, summed
    /// over chunk invocations: `Σ_i (prefix + start_i + len_i)`.
    ///
    /// This is the cost identity that does *not* cancel — the honest price of
    /// chunking. It counts slots a kernel invocation must be able to read, so it
    /// is an upper bound on the extra KV traffic: a fused kernel amortizes its
    /// reads *within* one invocation, but nothing amortizes them *across*
    /// invocations, which is exactly what chunking creates.
    #[must_use]
    pub fn kv_slot_visits(&self, prefix_tokens: usize) -> u64 {
        self.chunks
            .iter()
            .map(|chunk| widen(prefix_tokens) + widen(chunk.end()))
            .sum()
    }

    /// [`Self::kv_slot_visits`] divided by what a one-shot prefill would visit
    /// (`prefix + P`). `1.0` for a one-shot plan; `(m + 1) / 2` for `m` uniform
    /// chunks over an empty cache.
    #[must_use]
    #[allow(
        // Slot counts are far below 2^53, where f64 stops representing integers
        // exactly, so this ratio is exact for any prompt a serving system will
        // ever see.
        clippy::cast_precision_loss
    )]
    pub fn kv_read_amplification(&self, prefix_tokens: usize) -> f64 {
        let one_shot = widen(prefix_tokens) + widen(self.prompt_tokens);
        if one_shot == 0 {
            return 0.0;
        }
        self.kv_slot_visits(prefix_tokens) as f64 / one_shot as f64
    }

    /// Check that no chunk exceeds the iteration token budget.
    ///
    /// # Errors
    ///
    /// [`ChunkedPrefillError::ChunkOverBudget`] naming the first chunk that
    /// does.
    pub fn verify_budget(&self, budget: &TokenBudget) -> ChunkedPrefillResult<()> {
        for chunk in &self.chunks {
            if chunk.token_count() > budget.total() {
                return Err(ChunkedPrefillError::ChunkOverBudget {
                    ordinal: chunk.ordinal(),
                    tokens: chunk.token_count(),
                    budget: budget.total(),
                });
            }
        }
        Ok(())
    }
}

// ── ChunkedPrefillConfig ─────────────────────────────────────────────────────

/// How [`ChunkedPrefill`](crate::chunked_prefill::ChunkedPrefill) executes a
/// plan.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub struct ChunkedPrefillConfig {
    budget: TokenBudget,
    audit_mask: bool,
    retain_attention: bool,
}

impl ChunkedPrefillConfig {
    /// A configuration with the given budget, mask auditing on and attention
    /// retention off.
    #[must_use]
    pub const fn new(budget: TokenBudget) -> Self {
        Self {
            budget,
            audit_mask: true,
            retain_attention: false,
        }
    }

    /// The iteration token budget. No chunk may exceed
    /// [`TokenBudget::total`].
    #[must_use]
    pub const fn budget(&self) -> TokenBudget {
        self.budget
    }

    /// Whether every chunk's piecewise mask is cross-checked against the
    /// kernel's absolute-position mask. **On by default.**
    ///
    /// The check is `O(queries · keys)` booleans against an
    /// `O(queries · keys · heads · head_dim)` attention, so it is free in
    /// practice — and it is the one thing standing between a boundary off-by-one
    /// and a silently wrong tensor. Turn it off only if you have profiled it and
    /// found it to matter.
    #[must_use]
    pub const fn audit_mask(&self) -> bool {
        self.audit_mask
    }

    /// Whether each chunk keeps its full
    /// [`KvAttentionOutput`](crate::kv_cache_compression::KvAttentionOutput)
    /// (weights, mask and all). **Off by default**, because the weight matrix is
    /// quadratic in the context length and a serving system does not want it.
    #[must_use]
    pub const fn retain_attention(&self) -> bool {
        self.retain_attention
    }

    /// Set [`Self::audit_mask`].
    #[must_use]
    pub const fn with_mask_audit(mut self, audit_mask: bool) -> Self {
        self.audit_mask = audit_mask;
        self
    }

    /// Set [`Self::retain_attention`].
    #[must_use]
    pub const fn with_retained_attention(mut self, retain_attention: bool) -> Self {
        self.retain_attention = retain_attention;
        self
    }
}

impl Default for ChunkedPrefillConfig {
    fn default() -> Self {
        Self::new(TokenBudget::default())
    }
}
