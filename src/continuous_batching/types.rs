//! Core types for the `continuous_batching` module: the block identifier, the
//! engine configuration, the preemption modes, the per-sequence lifecycle
//! states, the per-step record, and the error type.
//!
//! The geometry lives on [`ContinuousBatchConfig`] rather than on the allocator
//! because *every* structure here — blocks, block tables, the paged attention
//! kernel — needs the same four numbers (`num_layers`, `num_heads`, `head_dim`,
//! `block_size`), and a single owner is the only way to make "this block table
//! belongs to this allocator" checkable rather than assumed.

#![allow(
    // Block/token/step counts are `usize` and are routinely divided into `f64`
    // ratios (occupancy, saving factors). The counts involved are far below
    // 2^53, so the pedantic precision-loss warning does not apply.
    clippy::cast_precision_loss
)]

use thiserror::Error;

/// Result alias for every fallible operation in this module.
pub type ContinuousBatchResult<T> = Result<T, ContinuousBatchError>;

/// Reject `NaN`/infinity anywhere in `buffer`.
///
/// The mirror of `kv_cache_compression`'s boundary check, and it exists for the
/// same reason: the attention kernel's guarantee that a finite input cannot
/// produce a non-finite output is a *theorem about finite inputs*, and a paged
/// cache that let a `NaN` in through the side door would be the hole in it.
pub(super) fn check_finite(buffer: &[f32], what: &'static str) -> ContinuousBatchResult<()> {
    for (index, value) in buffer.iter().enumerate() {
        if !value.is_finite() {
            return Err(ContinuousBatchError::NonFiniteInput { what, index });
        }
    }
    Ok(())
}

// ── KvBlockId ────────────────────────────────────────────────────────────────

/// A physical block's index into [`KvBlockAllocator`]'s pool.
///
/// This is a *physical* identifier and it is deliberately **not** a sequence's
/// logical block index: the whole point of a block table is that logical block
/// `k` of a sequence may live in any physical block, may be shared with another
/// sequence, and may change identity underneath the sequence when a
/// copy-on-write forces a copy.
///
/// [`KvBlockAllocator`]: crate::continuous_batching::KvBlockAllocator
#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Hash)]
pub struct KvBlockId(pub usize);

impl KvBlockId {
    /// The raw pool index.
    #[must_use]
    pub const fn index(self) -> usize {
        self.0
    }
}

impl std::fmt::Display for KvBlockId {
    fn fmt(&self, formatter: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(formatter, "block#{}", self.0)
    }
}

// ── BatchSequenceId ──────────────────────────────────────────────────────────

/// A sequence's identifier within a [`ContinuousBatchEngine`].
///
/// [`ContinuousBatchEngine`]: crate::continuous_batching::ContinuousBatchEngine
#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Hash)]
pub struct BatchSequenceId(pub usize);

impl BatchSequenceId {
    /// The raw index.
    #[must_use]
    pub const fn index(self) -> usize {
        self.0
    }
}

impl std::fmt::Display for BatchSequenceId {
    fn fmt(&self, formatter: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(formatter, "seq#{}", self.0)
    }
}

// ── BatchPreemptionMode ──────────────────────────────────────────────────────

/// How the engine reclaims a running sequence's blocks when the pool runs dry.
///
/// Both modes free **all** of the victim's blocks and both are lossless — the
/// difference is *where the bytes go* and therefore *what the sequence pays to
/// come back*:
///
/// | Mode | Cost to evict | Cost to resume | Extra memory while evicted |
/// |---|---|---|---|
/// | [`Recompute`] | free the blocks | re-drive the model over every live token | none |
/// | [`Swap`] | copy the blocks out of the pool | copy them back in | the swapped bytes, off-pool |
///
/// [`Recompute`] is the right default for short sequences (the re-prefill is
/// cheap and the evicted sequence occupies nothing at all while it waits), and
/// [`Swap`] is the right choice when the sequence is long enough that
/// recomputing it costs more than moving its bytes.
///
/// Both are exact: [`Recompute`] reproduces the KV **bit-for-bit** (the
/// producer is a deterministic function of the token prefix and the absolute
/// position — see [`BatchKvProducer`]), and [`Swap`] reproduces it bit-for-bit
/// trivially, because it is a byte copy.
///
/// [`Recompute`]: BatchPreemptionMode::Recompute
/// [`Swap`]: BatchPreemptionMode::Swap
/// [`BatchKvProducer`]: crate::continuous_batching::BatchKvProducer
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Default)]
pub enum BatchPreemptionMode {
    /// Drop the blocks and rebuild the KV from the token stream on resume.
    #[default]
    Recompute,
    /// Copy the blocks into an off-pool swap slab and copy them back on resume.
    Swap,
}

impl std::fmt::Display for BatchPreemptionMode {
    fn fmt(&self, formatter: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        let name = match self {
            Self::Recompute => "recompute",
            Self::Swap => "swap",
        };
        formatter.write_str(name)
    }
}

// ── BatchSequenceState ───────────────────────────────────────────────────────

/// Where a sequence sits in the engine's lifecycle.
///
/// The legal transitions are exactly:
///
/// ```text
///                    admit                 finish
///   Waiting  ─────────────────▶  Running  ─────────▶  Finished
///      ▲                          │   │
///      │  preempt(Recompute)      │   │  preempt(Swap)
///      └──────────────────────────┘   ▼
///                                  Swapped
///                                     │  admit (swap-in)
///                                     └──────────▶  Running
/// ```
///
/// A `Finished` sequence holds no blocks. A `Waiting` sequence holds no blocks
/// (a recompute-preempted one has had them all freed). A `Swapped` sequence
/// holds no *pool* blocks — its bytes live in an off-pool slab.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum BatchSequenceState {
    /// Submitted (or recompute-preempted) and queued for admission. Holds no
    /// blocks.
    Waiting,
    /// Admitted and decoding. Owns (or shares) the blocks of its block table.
    Running,
    /// Swap-preempted: its blocks were copied out of the pool and freed. Holds
    /// no pool blocks.
    Swapped,
    /// Reached its target length. Holds no blocks.
    Finished,
}

impl std::fmt::Display for BatchSequenceState {
    fn fmt(&self, formatter: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        let name = match self {
            Self::Waiting => "waiting",
            Self::Running => "running",
            Self::Swapped => "swapped",
            Self::Finished => "finished",
        };
        formatter.write_str(name)
    }
}

// ── ContinuousBatchConfig ────────────────────────────────────────────────────

/// The geometry of the KV blocks and the policy of the iteration-level loop.
///
/// # The one number that matters
///
/// `block_size` is the granularity of *everything* here. It sets the internal
/// fragmentation (a sequence wastes, on average, `block_size / 2` slots in its
/// partially filled tail block), the granularity at which a shared prompt
/// prefix can be reused (only **whole, full** blocks are shared), and the
/// amount of data a copy-on-write copies. Small blocks waste less and share
/// more finely; large blocks amortise the per-block bookkeeping. Real systems
/// sit at 16 or 32; this module's tests use 4 so that the block boundaries are
/// visible in a 20-token fixture.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct ContinuousBatchConfig {
    /// Transformer layers in the model whose KV this pool holds.
    pub num_layers: usize,
    /// Attention heads per layer.
    pub num_heads: usize,
    /// Dimension of each attention head.
    pub head_dim: usize,
    /// Token slots per block. See the type-level note.
    pub block_size: usize,
    /// Blocks in the pool. This is the memory budget, and it is the *only*
    /// thing that forces preemption.
    pub total_blocks: usize,
    /// Maximum sequences decoding concurrently. A second, independent bound on
    /// the batch: the pool may have room for more, but the model's compute may
    /// not.
    pub max_running_sequences: usize,
    /// Blocks held back from admission so that a freshly admitted sequence has
    /// somewhere to decode into for a few steps before it forces a preemption.
    ///
    /// Admission requires `free_blocks >= prompt_blocks + watermark_blocks`;
    /// decoding a *running* sequence is never blocked by the watermark (a
    /// sequence that has already been admitted must be able to make progress,
    /// or the engine would livelock).
    pub watermark_blocks: usize,
    /// How a victim's blocks are reclaimed. See [`BatchPreemptionMode`].
    pub preemption_mode: BatchPreemptionMode,
    /// Share the **full** blocks of a common prompt prefix between sequences,
    /// copy-on-write.
    ///
    /// This is exact, not heuristic: a block is reused only when the adopting
    /// sequence's token ids for that block range compare **equal, element by
    /// element**, to the block's own, and only when the block's absolute
    /// positions line up. The hash is a lookup key, never a proof (see
    /// [`KvBlockAllocator::lookup_prefix_block`]).
    ///
    /// [`KvBlockAllocator::lookup_prefix_block`]:
    ///     crate::continuous_batching::KvBlockAllocator::lookup_prefix_block
    pub enable_prefix_sharing: bool,
}

impl ContinuousBatchConfig {
    /// A config with the given geometry and pool size, `Recompute` preemption,
    /// prefix sharing on, no watermark, and an unbounded running set.
    #[must_use]
    pub const fn new(
        num_layers: usize,
        num_heads: usize,
        head_dim: usize,
        block_size: usize,
        total_blocks: usize,
    ) -> Self {
        Self {
            num_layers,
            num_heads,
            head_dim,
            block_size,
            total_blocks,
            max_running_sequences: usize::MAX,
            watermark_blocks: 0,
            preemption_mode: BatchPreemptionMode::Recompute,
            enable_prefix_sharing: true,
        }
    }

    /// Bound the number of concurrently decoding sequences.
    #[must_use]
    pub const fn with_max_running_sequences(mut self, max_running_sequences: usize) -> Self {
        self.max_running_sequences = max_running_sequences;
        self
    }

    /// Hold `watermark_blocks` back from admission.
    #[must_use]
    pub const fn with_watermark_blocks(mut self, watermark_blocks: usize) -> Self {
        self.watermark_blocks = watermark_blocks;
        self
    }

    /// Choose how preemption reclaims a victim's blocks.
    #[must_use]
    pub const fn with_preemption_mode(mut self, preemption_mode: BatchPreemptionMode) -> Self {
        self.preemption_mode = preemption_mode;
        self
    }

    /// Turn copy-on-write prefix sharing on or off.
    #[must_use]
    pub const fn with_prefix_sharing(mut self, enable_prefix_sharing: bool) -> Self {
        self.enable_prefix_sharing = enable_prefix_sharing;
        self
    }

    /// The number of `f32` elements one token contributes to a key (or value)
    /// buffer: `num_layers * num_heads * head_dim`. Identical to
    /// [`KvCacheTensor::token_buffer_len`], because the buffers are the same
    /// buffers.
    ///
    /// [`KvCacheTensor::token_buffer_len`]:
    ///     crate::kv_cache_compression::KvCacheTensor::token_buffer_len
    #[must_use]
    pub const fn token_buffer_len(&self) -> usize {
        self.num_layers * self.num_heads * self.head_dim
    }

    /// The number of `f32` elements one token contributes to *one layer* of a
    /// key (or value) buffer: `num_heads * head_dim`.
    #[must_use]
    pub const fn per_layer_token_stride(&self) -> usize {
        self.num_heads * self.head_dim
    }

    /// Bytes of `f32` payload one block occupies (keys **and** values).
    #[must_use]
    pub const fn block_bytes(&self) -> usize {
        2 * self.block_size * self.token_buffer_len() * std::mem::size_of::<f32>()
    }

    /// The number of blocks `num_tokens` tokens occupy: `ceil(n / block_size)`.
    #[must_use]
    pub const fn blocks_for_tokens(&self, num_tokens: usize) -> usize {
        num_tokens.div_ceil(self.block_size)
    }

    /// Reject a structurally impossible configuration.
    ///
    /// # Errors
    ///
    /// [`ContinuousBatchError::InvalidConfig`] if any of `num_layers`,
    /// `num_heads`, `head_dim`, `block_size`, `total_blocks` or
    /// `max_running_sequences` is zero, or if `watermark_blocks >=
    /// total_blocks` (which would make admission impossible).
    pub fn validate(&self) -> ContinuousBatchResult<()> {
        for (name, value) in [
            ("num_layers", self.num_layers),
            ("num_heads", self.num_heads),
            ("head_dim", self.head_dim),
            ("block_size", self.block_size),
            ("total_blocks", self.total_blocks),
            ("max_running_sequences", self.max_running_sequences),
        ] {
            if value == 0 {
                return Err(ContinuousBatchError::InvalidConfig {
                    reason: format!("{name} must be non-zero"),
                });
            }
        }
        if self.watermark_blocks >= self.total_blocks {
            return Err(ContinuousBatchError::InvalidConfig {
                reason: format!(
                    "watermark_blocks ({}) must leave at least one admissible block out of total_blocks ({})",
                    self.watermark_blocks, self.total_blocks
                ),
            });
        }
        Ok(())
    }
}

// ── BatchPreemption ──────────────────────────────────────────────────────────

/// A single preemption, as it happened.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct BatchPreemption {
    /// The victim.
    pub sequence: BatchSequenceId,
    /// How its blocks were reclaimed.
    pub mode: BatchPreemptionMode,
    /// Live token slots the victim held when it was evicted. Under
    /// [`BatchPreemptionMode::Recompute`] this is exactly the number of tokens
    /// the producer must be re-driven over to bring it back — the *price* of
    /// the eviction, in producer calls.
    pub tokens_dropped: usize,
    /// Blocks the victim's table referenced.
    pub blocks_released: usize,
    /// Blocks that actually returned to the free list. This is **smaller** than
    /// `blocks_released` exactly when some of the victim's blocks were still
    /// referenced by another sequence — the price of copy-on-write sharing is
    /// that evicting a sharer does not free a shared block.
    pub blocks_freed: usize,
}

// ── BatchStep ────────────────────────────────────────────────────────────────

/// Everything one iteration of the batch loop did.
///
/// This is the module's measurement surface. A step is an *iteration*, not a
/// request: sequences enter and leave it, and the numbers below are what makes
/// "continuous batching retires a finished sequence immediately instead of
/// waiting for the longest one in the batch" a checkable claim rather than a
/// slogan.
#[derive(Debug, Clone, PartialEq, Eq, Default)]
pub struct BatchStep {
    /// The logical tick. Starts at 1 for the first step; there is no wall clock
    /// anywhere in this module.
    pub step: u64,
    /// Sequences that decoded one token this step, in admission order.
    pub decoded: Vec<BatchSequenceId>,
    /// Sequences admitted this step (prefilled, or swapped back in).
    pub admitted: Vec<BatchSequenceId>,
    /// Sequences evicted this step, in the order they were chosen.
    pub preempted: Vec<BatchPreemption>,
    /// Sequences that reached their target length this step and released their
    /// blocks.
    pub finished: Vec<BatchSequenceId>,
    /// Tokens decoded this step: exactly one per running sequence, which is the
    /// defining property of an iteration-level loop.
    pub tokens_decoded: usize,
    /// Tokens whose KV was **computed** during this step's prefills.
    pub prefill_tokens: usize,
    /// Tokens whose KV was **adopted from a shared block** during this step's
    /// prefills, and therefore never computed. This is the prefix-sharing
    /// saving, in tokens.
    pub prefix_cached_tokens: usize,
    /// Tokens re-materialised this step to bring recompute-preempted sequences
    /// back. This is the price of [`BatchPreemptionMode::Recompute`], in
    /// producer calls.
    pub recomputed_tokens: usize,
    /// Blocks held by live sequences after the step.
    pub blocks_in_use: usize,
    /// Blocks in the free list after the step. Always
    /// `total_blocks - blocks_in_use`.
    pub free_blocks: usize,
    /// Sequences running after the step.
    pub running: usize,
    /// Sequences waiting after the step.
    pub waiting: usize,
    /// Sequences swapped out after the step.
    pub swapped: usize,
}

impl BatchStep {
    /// Whether this step evicted anybody.
    #[must_use]
    pub fn preempted_anything(&self) -> bool {
        !self.preempted.is_empty()
    }

    /// Whether any sequence made progress this step.
    #[must_use]
    pub fn made_progress(&self) -> bool {
        self.tokens_decoded > 0 || !self.admitted.is_empty()
    }
}

// ── ContinuousBatchStats ─────────────────────────────────────────────────────

/// Cumulative counters over an engine's whole life.
#[derive(Debug, Clone, PartialEq, Eq, Default)]
pub struct ContinuousBatchStats {
    /// Iterations run.
    pub steps: u64,
    /// Tokens decoded (one per running sequence per step).
    pub tokens_decoded: u64,
    /// Prompt tokens whose KV was computed by the producer.
    pub prefill_tokens: u64,
    /// Prompt tokens whose KV came from a shared block and was never computed.
    pub prefix_cached_tokens: u64,
    /// Tokens re-materialised by [`BatchPreemptionMode::Recompute`] resumes.
    pub recomputed_tokens: u64,
    /// Copy-on-write block copies performed.
    pub cow_copies: u64,
    /// Recompute preemptions.
    pub preemptions_recompute: u64,
    /// Swap preemptions.
    pub preemptions_swap: u64,
    /// Blocks copied out of the pool by swap preemptions.
    pub swapped_out_blocks: u64,
    /// Blocks copied back into the pool by swap-in resumes.
    pub swapped_in_blocks: u64,
    /// The high-water mark of pool occupancy, in blocks.
    pub peak_blocks_in_use: usize,
}

impl ContinuousBatchStats {
    /// Producer calls the engine made: every token whose KV it computed, for a
    /// prefill, a decode, or a recompute resume.
    ///
    /// Prefix-shared tokens are **not** here — that is the point of them.
    #[must_use]
    pub const fn producer_calls(&self) -> u64 {
        self.prefill_tokens + self.tokens_decoded + self.recomputed_tokens
    }

    /// Fraction of prompt tokens served from a shared block rather than
    /// computed, in `[0, 1]`. Zero when no prompt tokens have been seen.
    #[must_use]
    pub fn prefix_hit_rate(&self) -> f64 {
        let prompt_tokens = self.prefill_tokens + self.prefix_cached_tokens;
        if prompt_tokens == 0 {
            return 0.0;
        }
        self.prefix_cached_tokens as f64 / prompt_tokens as f64
    }
}

// ── ContinuousBatchError ─────────────────────────────────────────────────────

/// Everything that can go wrong allocating, sharing, preempting or attending
/// over a paged KV cache.
#[derive(Debug, Clone, PartialEq, Eq, Error)]
pub enum ContinuousBatchError {
    /// A [`ContinuousBatchConfig`] field was structurally invalid.
    #[error("invalid continuous-batching config: {reason}")]
    InvalidConfig {
        /// Human-readable explanation of what was invalid.
        reason: String,
    },
    /// The block pool is exhausted and nothing further can be preempted.
    ///
    /// This is a genuine out-of-memory: the pool is too small to hold even the
    /// sequences that cannot be evicted. It is *not* the ordinary "we are full,
    /// so evict somebody" path, which the engine handles by preempting.
    #[error(
        "block pool exhausted: {requested} block(s) requested, {free} free of {total}; nothing left to preempt"
    )]
    OutOfBlocks {
        /// Blocks the caller needed.
        requested: usize,
        /// Blocks actually available.
        free: usize,
        /// The pool size.
        total: usize,
    },
    /// A block id does not name a block in this pool.
    #[error("{block} is not a block of a pool with {total} block(s)")]
    UnknownBlock {
        /// The offending id.
        block: KvBlockId,
        /// The pool size.
        total: usize,
    },
    /// A block was released, read or shared while it was in the free list.
    #[error("{block} is not allocated: {operation} requires a live block")]
    BlockNotAllocated {
        /// The offending id.
        block: KvBlockId,
        /// The operation that was attempted.
        operation: &'static str,
    },
    /// A sequence id does not name a sequence of this engine.
    #[error("{sequence} is not a sequence of an engine with {total} sequence(s)")]
    UnknownSequence {
        /// The offending id.
        sequence: BatchSequenceId,
        /// The number of submitted sequences.
        total: usize,
    },
    /// A sequence was asked to do something its state does not allow.
    #[error("{sequence} is {state}: {operation} is not legal in that state")]
    InvalidState {
        /// The offending sequence.
        sequence: BatchSequenceId,
        /// The state it was actually in.
        state: BatchSequenceState,
        /// The operation that was attempted.
        operation: &'static str,
    },
    /// A supplied buffer did not have the length the geometry demands.
    #[error("{what}: expected {expected} element(s), got {actual}")]
    ShapeMismatch {
        /// Which buffer.
        what: &'static str,
        /// The length the geometry demands.
        expected: usize,
        /// The length supplied.
        actual: usize,
    },
    /// A buffer contained a `NaN` or an infinity.
    ///
    /// Rejected at the boundary, exactly as [`KvCacheTensor::append_token`]
    /// does, because the attention kernel's finiteness guarantee is a theorem
    /// about *finite inputs* and this module must not be the hole in it.
    ///
    /// [`KvCacheTensor::append_token`]:
    ///     crate::kv_cache_compression::KvCacheTensor::append_token
    #[error("{what} contains a non-finite value at index {index}")]
    NonFiniteInput {
        /// Which buffer.
        what: &'static str,
        /// The offset of the first offending element.
        index: usize,
    },
    /// A layer index was out of range for the configured `num_layers`.
    #[error("layer index {layer} is out of range for a model with {num_layers} layer(s)")]
    LayerOutOfRange {
        /// The offending layer index.
        layer: usize,
        /// The configured layer count.
        num_layers: usize,
    },
    /// An operation that needs at least one token was performed on an empty
    /// block table.
    #[error("block table is empty: {operation} requires at least one token")]
    EmptyTable {
        /// The operation that was attempted.
        operation: &'static str,
    },
    /// A token, position or query selection was malformed.
    #[error("invalid selection: {reason}")]
    InvalidSelection {
        /// What was wrong with it.
        reason: String,
    },
    /// A block was offered for prefix sharing but its contents do not match the
    /// adopting sequence's.
    ///
    /// Reaching this means the exact token-by-token verification that guards
    /// every share rejected a candidate — which is the guard working. It is
    /// surfaced rather than silently skipped because a *hash hit* with a
    /// *content miss* is either a 64-bit collision or a bookkeeping bug, and
    /// neither should pass quietly.
    #[error("prefix block does not match the adopting sequence: {reason}")]
    PrefixMismatch {
        /// What failed to line up.
        reason: String,
    },
    /// A swap slab did not fit the pool it was being restored into.
    #[error("swap slab does not fit this pool: {reason}")]
    SwapMismatch {
        /// What failed to line up.
        reason: String,
    },
    /// A structural invariant of the pool does not hold.
    ///
    /// Raised only by [`KvBlockAllocator::verify_invariants`], which walks the
    /// whole pool and checks conservation (`free + allocated == total`), the
    /// free list, and the prefix index against the blocks themselves. Every
    /// variant of this error is a **bug in this module**, never bad input, and
    /// it is a distinct variant precisely so that it cannot be mistaken for one.
    ///
    /// [`KvBlockAllocator::verify_invariants`]:
    ///     crate::continuous_batching::KvBlockAllocator::verify_invariants
    #[error("continuous-batching invariant violated: {reason}")]
    InvariantViolated {
        /// The first statement that did not hold.
        reason: String,
    },
    /// The `kv_cache_compression` tensor this module materialises into rejected
    /// the data.
    ///
    /// Only reachable from [`KvBlockTable::materialize`], which is the one place
    /// this module hands bytes to the upstream cache.
    ///
    /// [`KvBlockTable::materialize`]:
    ///     crate::continuous_batching::KvBlockTable::materialize
    #[error(transparent)]
    KvCache(#[from] crate::kv_cache_compression::KvCompressionError),
}
