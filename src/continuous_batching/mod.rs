//! `PagedAttention`-style KV block allocation and iteration-level batching: a
//! block-structured view over the KV cache, with copy-on-write sharing and
//! preemption.
//!
//! # The problem, in one number
//!
//! A serving engine does not know how long a sequence will be until it ends. The
//! obvious thing to do — reserve a contiguous run of KV memory at the maximum
//! possible length — wastes almost all of it: a request that could have produced
//! 2048 tokens and produced 40 has 2008 slots of reserved, untouchable, empty
//! cache. Published measurements of that design put the *useful* fraction of KV
//! memory at **20–40%**. The rest is internal fragmentation, external
//! fragmentation, and reservation for output that never arrived.
//!
//! `PagedAttention`'s answer is the one operating systems reached for the same
//! reason: stop requiring the memory to be contiguous. Chop the KV cache into
//! fixed-size **blocks**, give each sequence a **block table** mapping logical
//! token `t` to whichever physical block currently holds it, and grow that table
//! one block at a time as the sequence actually grows. The waste collapses from
//! "the maximum length minus the actual length" to "at most `block_size - 1`
//! slots in the tail block", **per sequence** — a bound that does not depend on
//! the maximum length at all.
//!
//! And once the KV is a *table of blocks* rather than a slab of bytes, two things
//! that were impossible become nearly free:
//!
//! - **Copy-on-write sharing.** Two sequences whose prompts begin the same way
//!   can point at the *same blocks*. Forking a sequence copies a list of block
//!   ids, not a megabyte of keys. The first token that diverges copies exactly
//!   one block — the shared, partially filled tail — and never the history.
//! - **Preemption.** A sequence's blocks can be taken away and given back, by
//!   recomputing them from the tokens or by swapping the bytes out, because
//!   nothing anywhere depends on *which* physical blocks a sequence had.
//!
//! # Distinct from the other paged things in this crate
//!
//! Distinct from `prefix_cache::paging`, which allocates a fresh, immutable run
//! of pages per cached fingerprint and **never shares a page between entries**:
//! this module's blocks are copy-on-write-shared across live sequences, grow one
//! token per decode step, and are reclaimed by preemption. Distinct from
//! `memory_paging`, which pages **conversational text**, not KV bytes.
//!
//! The distinction is worth one more sentence, because the names are so close.
//! `prefix_cache::paging` is a *store*: it chops a finished KV blob into pages so
//! it can be evicted and defragmented at a granularity finer than the whole blob.
//! Its pages belong to one entry, forever, and its `PagedCache::put` allocates
//! fresh pages for every entry it stores. This module is an *allocator for live
//! sequences*: its blocks are written into one token at a time by a decode loop,
//! shared between sequences that are running *right now*, and taken back from a
//! sequence that is still running when somebody else needs them more.
//!
//! # What is actually computed here
//!
//! Everything, and against an oracle this module did not write.
//!
//! [`paged_attention`] computes `softmax(QKᵀ/√d + M)·V` by gathering every key
//! and every value **through the block table** — `blocks[t / block_size]` at slot
//! `t % block_size` — over blocks deliberately scattered across a fragmented
//! pool. [`kv_cache_compression::scaled_dot_product_attention`] computes the same
//! thing over a contiguous [`KvCacheTensor`]. The module's headline test asserts
//! that the two agree **bit for bit** — not to a tolerance, but on the exact `f32`
//! output buffer and the exact `f64` weight buffer:
//!
//! | Fixture | Blocks | Physical block order | Result |
//! |---|---|---|---|
//! | 3 layers, 3 heads, `head_dim` 8, 23 tokens, `block_size` 4 | 6 | scattered, non-monotone | **bit-identical**, 4416 `f32` + 12144 `f64` |
//! | the same, after an `H2O` pass evicted 9 tokens (gappy positions) | 4 | scattered | **bit-identical** |
//!
//! A tolerance would have let a stride bug, an off-by-one in a block offset, or a
//! mask built from slot indices instead of absolute positions hide in the last
//! bits. Bit-equality lets none of them hide anywhere. (It is a legitimate demand
//! rather than a lucky one because the paged kernel performs the *same* IEEE-754
//! operations in the *same* order as the reference — see [`attention`].)
//!
//! # The three measured claims
//!
//! **1. Copy-on-write sharing is real, and it is exact.** Eight sequences sharing
//! a 16-token prompt prefix and adding 6 private tokens each occupy
//! `4 + 8 * 2 = 20` blocks, not `8 * 6 = 48`. That is the exact predicted count —
//! `floor(P/B) + N * (ceil(L/B) - floor(P/B))` — asserted as an integer, not a
//! trend, together with a **2.4x** saving and a reference count of exactly 8 on
//! each shared block. After a fork and a write by the child, the parent's KV is
//! **bit-identical** to what it was before the fork.
//!
//! **2. Preemption is lossless.** A sequence evicted mid-decode and brought back
//! reproduces its KV **bit for bit** — by recompute (re-driving the producer over
//! its token stream) and by swap (moving the bytes off-pool and back into
//! *different physical blocks*). Both are asserted against a `KvCacheTensor`
//! snapshot taken before the eviction.
//!
//! **3. Iteration-level batching retires a finished sequence immediately.** Eight
//! requests of wildly unequal lengths (10 to 80 tokens), through a pool that fits
//! about half of them at once: a static batch — one cohort, dissolved when its
//! longest member finishes — spends **58.9%** of its sequence-slots on sequences
//! that had already finished. This loop spends **0%**, because a sequence that
//! finishes at step 40 releases its blocks at step 40 and the next request is
//! admitted at step 41.
//!
//! # The allocator's conscience
//!
//! [`KvBlockAllocator::verify_invariants`] walks the entire pool and checks seven
//! statements — conservation (`free + allocated == total`), that the free list is
//! *exactly* the set of unreferenced blocks with no duplicates, and that the
//! prefix index and the blocks cannot disagree about who is cached. The
//! randomized churn test calls it after **every one** of several thousand
//! alloc/free/fork/copy-on-write/preempt operations drawn from a seeded stream,
//! and debug builds call it after every mutation of every test in the suite.
//!
//! # Scope: this module owns memory, not time
//!
//! Victims are chosen last-admitted-first; admission is FIFO. There is
//! deliberately no priority, no deadline and no fairness weighting here — *where
//! the bytes live* is this module's problem, and *whose turn it is* is
//! `request_scheduling`'s. Nor does this module chunk a prefill: a prompt is
//! materialised in one shot at admission, and splitting that across steps is
//! `chunked_prefill`'s.
//!
//! # Quick start
//!
//! ```rust
//! # #[cfg(feature = "continuous-batching")]
//! # {
//! use oxirag::continuous_batching::{
//!     BatchKvProducer, ContinuousBatchConfig, ContinuousBatchEngine, ContinuousBatchResult,
//! };
//!
//! // A deterministic stand-in for a model's forward pass. The KV of a token is a
//! // reproducible function of the tokens before it and its position -- which is
//! // exactly, and only, what `BatchKvProducer` asks for, and exactly what makes
//! // prefix sharing and recompute-preemption sound.
//! struct HashKv;
//! impl BatchKvProducer for HashKv {
//!     fn produce_kv(
//!         &self,
//!         context: &[u32],
//!         position: usize,
//!         keys: &mut [f32],
//!         values: &mut [f32],
//!     ) -> ContinuousBatchResult<()> {
//!         let mut seed = 0xcbf2_9ce4_8422_2325_u64 ^ (position as u64);
//!         for &token in context {
//!             seed = (seed ^ u64::from(token)).wrapping_mul(0x0000_0100_0000_01b3);
//!         }
//!         for (offset, buffer) in [keys, values].into_iter().enumerate() {
//!             for (index, slot) in buffer.iter_mut().enumerate() {
//!                 let mixed = seed
//!                     .wrapping_add((offset * 4096 + index) as u64)
//!                     .wrapping_mul(0x9e37_79b9_7f4a_7c15);
//!                 *slot = f32::from((mixed >> 48) as u16) / 32_768.0 - 1.0;
//!             }
//!         }
//!         Ok(())
//!     }
//! }
//!
//! // 2 layers, 2 heads, head_dim 4; 4-token blocks; a pool of 24 of them.
//! let config = ContinuousBatchConfig::new(2, 2, 4, 4, 24);
//! let mut engine = ContinuousBatchEngine::new(config).expect("valid config");
//!
//! // Two requests behind the same 16-token system prompt.
//! let system_prompt: Vec<u32> = (0..16).collect();
//! for suffix in [100_u32, 200] {
//!     let mut tokens = system_prompt.clone();
//!     tokens.extend([suffix, suffix + 1, suffix + 2, suffix + 3]);
//!     // 18-token prompt, then 2 tokens decoded one per iteration.
//!     engine.submit(tokens, 18).expect("well-formed request");
//! }
//!
//! engine.run_until_idle(&HashKv, 64).expect("the loop makes progress");
//!
//! // The four blocks of the shared prefix were computed once, by the first
//! // request, and adopted by the second for free: 16 of its 18 prompt tokens
//! // cost nothing at all, and not one key was copied.
//! let stats = engine.stats();
//! assert_eq!(stats.prefix_cached_tokens, 16);
//! assert_eq!(stats.prefill_tokens, 20); // 18 for the first, 2 for the second
//! assert!(engine.sequences().iter().all(|sequence| sequence.is_complete()));
//! # }
//! ```
//!
//! [`kv_cache_compression::scaled_dot_product_attention`]:
//!     crate::kv_cache_compression::scaled_dot_product_attention
//! [`KvCacheTensor`]: crate::kv_cache_compression::KvCacheTensor

pub mod allocator;
pub mod attention;
pub mod block;
pub mod engine;
pub mod table;
pub mod types;

#[cfg(all(test, not(target_arch = "wasm32")))]
mod tests;

pub use allocator::{KvBlockAllocator, KvBlockAllocatorStats, PREFIX_HASH_SEED, fold_prefix_hash};
pub use attention::{KvBlockAttentionOutput, paged_attention};
pub use block::KvBlock;
pub use engine::{BatchKvProducer, BatchSequence, ContinuousBatchEngine};
pub use table::{KvBlockSwapSlab, KvBlockTable};
pub use types::{
    BatchPreemption, BatchPreemptionMode, BatchSequenceId, BatchSequenceState, BatchStep,
    ContinuousBatchConfig, ContinuousBatchError, ContinuousBatchResult, ContinuousBatchStats,
    KvBlockId,
};
