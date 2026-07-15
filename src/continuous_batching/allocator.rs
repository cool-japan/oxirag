//! The block pool: a free list, a reference count per block, copy-on-write, and
//! an **exact** content-addressed index for prompt-prefix sharing.
//!
//! # The invariant everything else stands on
//!
//! ```text
//! free_blocks + allocated_blocks == total_blocks        (always, after every operation)
//! ```
//!
//! where `allocated_blocks` is *by definition* the number of blocks with
//! `ref_count > 0`. [`KvBlockAllocator::verify_invariants`] checks that, and six
//! stronger statements besides, by walking the whole pool; the churn test calls
//! it after **every** operation of a seeded random alloc/free/fork/preempt
//! workload. Conservation is not a comment here — it is an assertion.
//!
//! # Reference counts are the whole design
//!
//! A block is not owned by a sequence. It is *referenced* by zero or more block
//! tables, and the count is the only thing that decides what may happen to it:
//!
//! | `ref_count` | Meaning | May be written in place? | Is it in the free list? |
//! |---|---|---|---|
//! | `0` | free (bytes retained, still cacheable) | — | yes |
//! | `1` | privately held by one sequence | **yes** | no |
//! | `> 1` | shared, copy-on-write | **no** — copy first | no |
//!
//! The `> 1` row is copy-on-write, and it is the entire reason a fork is O(1)
//! and a shared 4000-token system prompt costs one copy of its blocks instead of
//! one *per request*.
//!
//! # A free block is still a cache entry
//!
//! When the last sequence holding a block releases it, the block returns to the
//! free list **with its bytes intact** and stays in the prefix index. A later
//! request whose prompt begins with the same tokens can therefore still hit it —
//! this is *cross-request* prefix reuse, and it is why the index survives the
//! sequences that populated it. The bytes are destroyed exactly once: when the
//! block is popped off the free list and re-allocated, at which point its cache
//! entry is dropped in the same breath (`KvBlock::reset_for_allocation`).
//!
//! A cache hit on a free block *revives* it — the block leaves the free list
//! without passing through [`KvBlockAllocator::allocate_block`]. That path is
//! where a naive implementation double-hands-out a block, so it is called out
//! here and asserted in [`KvBlockAllocator::verify_invariants`]: the free list is
//! exactly the set of blocks with `ref_count == 0`, with no duplicates, at all
//! times.
//!
//! # The hash is a key, not a proof
//!
//! Blocks are indexed by a 64-bit rolling hash of the token ids of the whole
//! prefix up to and including the block. A 64-bit hash *can* collide, and a
//! collision here would graft one sequence's keys onto another's — a silent,
//! catastrophic wrong answer. So the hash is used only to *find* a candidate,
//! and the adoption then compares the block's token ids and absolute positions
//! **element by element** against the adopting sequence's before taking the
//! reference ([`KvBlockTable::adopt_prefix_block`]). Sharing is exact, and a
//! collision costs a rejected candidate, never a wrong answer.
//!
//! [`KvBlockTable::adopt_prefix_block`]:
//!     crate::continuous_batching::KvBlockTable::adopt_prefix_block

use std::collections::{HashMap, VecDeque};

use super::block::KvBlock;
use super::types::{ContinuousBatchConfig, ContinuousBatchError, ContinuousBatchResult, KvBlockId};

// ── KvBlockAllocatorStats ────────────────────────────────────────────────────

/// A snapshot of the pool.
///
/// The pair that matters is (`allocated_blocks`, `total_references`):
/// `total_references` is how many logical blocks the live block tables point at
/// — i.e. how many blocks a **non-sharing** allocator would have had to hand out
/// — and `allocated_blocks` is how many physical blocks that actually cost.
/// Their ratio is the copy-on-write saving, measured rather than asserted.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default)]
pub struct KvBlockAllocatorStats {
    /// The pool size.
    pub total_blocks: usize,
    /// Blocks with `ref_count > 0`.
    pub allocated_blocks: usize,
    /// Blocks with `ref_count == 0`. Always `total_blocks - allocated_blocks`.
    pub free_blocks: usize,
    /// Blocks with `ref_count > 1` — the ones that are copy-on-write shared.
    pub shared_blocks: usize,
    /// The sum of every block's `ref_count`: the number of *logical* blocks the
    /// live block tables reference.
    pub total_references: usize,
    /// Blocks currently published in the prefix index (including free ones,
    /// which are still valid cache entries).
    pub cached_blocks: usize,
    /// Blocks handed out by [`KvBlockAllocator::allocate_block`] over the pool's
    /// life.
    pub allocations: u64,
    /// Blocks returned to the free list over the pool's life.
    pub frees: u64,
    /// Copy-on-write copies performed.
    pub cow_copies: u64,
    /// Prefix-index lookups that found (and verified) a block.
    pub prefix_hits: u64,
    /// Prefix-index lookups that found nothing.
    pub prefix_misses: u64,
}

impl KvBlockAllocatorStats {
    /// Physical blocks saved by copy-on-write sharing:
    /// `total_references - allocated_blocks`.
    #[must_use]
    pub const fn blocks_saved_by_sharing(&self) -> usize {
        self.total_references.saturating_sub(self.allocated_blocks)
    }

    /// `total_references / allocated_blocks` — the factor by which sharing
    /// stretched the pool. `1.0` when nothing is shared.
    #[must_use]
    #[allow(clippy::cast_precision_loss)]
    pub fn sharing_factor(&self) -> f64 {
        if self.allocated_blocks == 0 {
            return 1.0;
        }
        self.total_references as f64 / self.allocated_blocks as f64
    }

    /// Fraction of the pool in use, in `[0, 1]`.
    #[must_use]
    #[allow(clippy::cast_precision_loss)]
    pub fn occupancy(&self) -> f64 {
        if self.total_blocks == 0 {
            return 0.0;
        }
        self.allocated_blocks as f64 / self.total_blocks as f64
    }
}

// ── KvBlockAllocator ─────────────────────────────────────────────────────────

/// The fixed pool of KV blocks, its free list, and its prefix index.
///
/// See the module documentation for the invariants. The short version: blocks
/// are never created or destroyed after construction, only referenced and
/// unreferenced; `free + allocated == total` always; and a block with
/// `ref_count > 1` is immutable until somebody copies it.
#[derive(Debug, Clone)]
pub struct KvBlockAllocator {
    config: ContinuousBatchConfig,
    blocks: Vec<KvBlock>,
    /// Exactly the blocks with `ref_count == 0`, no duplicates. Popped from the
    /// front and pushed at the back, so a block that has been free longest is
    /// reused first — which is what makes a *recently* freed block's cached
    /// prefix survive long enough to be hit again.
    free_list: VecDeque<KvBlockId>,
    /// `content_hash -> block`. At most one block carries a given hash, and that
    /// block's `content_hash` is exactly that hash: the map and the blocks are
    /// two views of one fact, and `verify_invariants` checks both directions.
    prefix_index: HashMap<u64, KvBlockId>,
    allocated_blocks: usize,
    allocations: u64,
    frees: u64,
    cow_copies: u64,
    prefix_hits: u64,
    prefix_misses: u64,
}

impl KvBlockAllocator {
    /// Build a pool of `config.total_blocks` empty blocks.
    ///
    /// # Errors
    ///
    /// Whatever [`ContinuousBatchConfig::validate`] rejects.
    pub fn new(config: ContinuousBatchConfig) -> ContinuousBatchResult<Self> {
        config.validate()?;
        let token_buffer_len = config.token_buffer_len();
        let per_layer_token_stride = config.per_layer_token_stride();
        let blocks: Vec<KvBlock> = (0..config.total_blocks)
            .map(|index| {
                KvBlock::new(
                    KvBlockId(index),
                    config.block_size,
                    token_buffer_len,
                    per_layer_token_stride,
                )
            })
            .collect();
        let free_list: VecDeque<KvBlockId> = (0..config.total_blocks).map(KvBlockId).collect();
        Ok(Self {
            config,
            blocks,
            free_list,
            prefix_index: HashMap::new(),
            allocated_blocks: 0,
            allocations: 0,
            frees: 0,
            cow_copies: 0,
            prefix_hits: 0,
            prefix_misses: 0,
        })
    }

    /// The geometry and policy this pool was built with.
    #[must_use]
    pub const fn config(&self) -> &ContinuousBatchConfig {
        &self.config
    }

    /// The pool size.
    #[must_use]
    pub const fn total_blocks(&self) -> usize {
        self.config.total_blocks
    }

    /// Blocks with `ref_count > 0`.
    #[must_use]
    pub const fn allocated_blocks(&self) -> usize {
        self.allocated_blocks
    }

    /// Blocks with `ref_count == 0`.
    #[must_use]
    pub const fn free_blocks(&self) -> usize {
        self.config.total_blocks - self.allocated_blocks
    }

    /// Read a block, or `None` if the id is out of range.
    #[must_use]
    pub fn block(&self, id: KvBlockId) -> Option<&KvBlock> {
        self.blocks.get(id.index())
    }

    /// A block's reference count, or `0` if the id is out of range.
    #[must_use]
    pub fn ref_count(&self, id: KvBlockId) -> usize {
        self.blocks.get(id.index()).map_or(0, KvBlock::ref_count)
    }

    /// Take a live block off the free list.
    ///
    /// The block is scrubbed and its prefix-cache entry (if any) is dropped: the
    /// bytes of the previous tenant do not survive re-allocation, and a cache
    /// entry never outlives the bytes it describes.
    ///
    /// # Errors
    ///
    /// [`ContinuousBatchError::OutOfBlocks`] when the free list is empty. This
    /// is the pressure signal the engine turns into a preemption; it is not, by
    /// itself, a fatal condition.
    pub fn allocate_block(&mut self) -> ContinuousBatchResult<KvBlockId> {
        let Some(id) = self.free_list.pop_front() else {
            return Err(ContinuousBatchError::OutOfBlocks {
                requested: 1,
                free: 0,
                total: self.config.total_blocks,
            });
        };

        // Everything in the free list has `ref_count == 0` (revived blocks are
        // removed from it, never tombstoned), so this pop is unconditionally a
        // free block.
        debug_assert_eq!(self.blocks[id.index()].ref_count(), 0);

        // The block may still be a prefix-cache entry. Re-allocating destroys
        // the bytes, so the entry goes with them — and only if the index still
        // names *this* block, which it always does (at most one block carries a
        // given hash), but the check is what makes that invariant unable to rot.
        if let Some(hash) = self.blocks[id.index()].content_hash()
            && self.prefix_index.get(&hash) == Some(&id)
        {
            self.prefix_index.remove(&hash);
        }

        self.blocks[id.index()].reset_for_allocation();
        self.allocated_blocks += 1;
        self.allocations += 1;
        self.debug_verify();
        Ok(id)
    }

    /// Take `count` blocks, or none at all.
    ///
    /// Admission is all-or-nothing: a half-prefilled sequence holding blocks it
    /// cannot finish with is strictly worse than one that never started, so the
    /// blocks are rolled back on failure.
    ///
    /// # Errors
    ///
    /// [`ContinuousBatchError::OutOfBlocks`] if fewer than `count` are free.
    pub fn allocate_blocks(&mut self, count: usize) -> ContinuousBatchResult<Vec<KvBlockId>> {
        if self.free_blocks() < count {
            return Err(ContinuousBatchError::OutOfBlocks {
                requested: count,
                free: self.free_blocks(),
                total: self.config.total_blocks,
            });
        }
        let mut taken = Vec::with_capacity(count);
        for _ in 0..count {
            match self.allocate_block() {
                Ok(id) => taken.push(id),
                Err(error) => {
                    for id in taken {
                        self.release_block(id)?;
                    }
                    return Err(error);
                }
            }
        }
        Ok(taken)
    }

    /// Drop one reference to a block, returning it to the free list if that was
    /// the last one.
    ///
    /// The bytes are **not** scrubbed here: a freed block stays a valid
    /// prefix-cache entry until it is re-allocated. See the module docs.
    ///
    /// # Errors
    ///
    /// - [`ContinuousBatchError::UnknownBlock`] if the id is out of range.
    /// - [`ContinuousBatchError::BlockNotAllocated`] if the block is already
    ///   free. A double free is a bookkeeping bug and is refused, not absorbed.
    pub fn release_block(&mut self, id: KvBlockId) -> ContinuousBatchResult<()> {
        let total = self.config.total_blocks;
        let block = self
            .blocks
            .get_mut(id.index())
            .ok_or(ContinuousBatchError::UnknownBlock { block: id, total })?;
        if block.ref_count() == 0 {
            return Err(ContinuousBatchError::BlockNotAllocated {
                block: id,
                operation: "release_block",
            });
        }
        if block.release() {
            self.free_list.push_back(id);
            self.allocated_blocks -= 1;
            self.frees += 1;
        }
        self.debug_verify();
        Ok(())
    }

    /// Take an additional reference to a block: the O(1) half of a fork, and the
    /// whole of a prefix-cache hit.
    ///
    /// A block with `ref_count == 0` is **revived**: it is removed from the free
    /// list and rejoins the allocated set with its bytes intact. That is how a
    /// cached prefix survives the sequence that produced it.
    ///
    /// # Errors
    ///
    /// [`ContinuousBatchError::UnknownBlock`] if the id is out of range.
    pub fn add_ref(&mut self, id: KvBlockId) -> ContinuousBatchResult<()> {
        let total = self.config.total_blocks;
        let block = self
            .blocks
            .get_mut(id.index())
            .ok_or(ContinuousBatchError::UnknownBlock { block: id, total })?;
        let was_free = block.ref_count() == 0;
        block.add_ref();
        if was_free {
            // Revival. The block must leave the free list *now*: leaving it there
            // as a tombstone would let `allocate_block` hand out a block that a
            // sequence is already reading.
            self.free_list.retain(|&free| free != id);
            self.allocated_blocks += 1;
        }
        self.debug_verify();
        Ok(())
    }

    /// Copy a shared block so the caller can write into its own copy.
    ///
    /// This **is** copy-on-write. It allocates a fresh block, copies the shared
    /// block's live slots (bytes, absolute positions and token ids) into it,
    /// drops the caller's reference to the shared original, and returns the copy
    /// — which the caller then points its block table at.
    ///
    /// The copy is deliberately not published to the prefix index: it exists
    /// because it is about to diverge, and a block that is about to change is not
    /// a cache entry.
    ///
    /// # Errors
    ///
    /// - [`ContinuousBatchError::UnknownBlock`] if the id is out of range.
    /// - [`ContinuousBatchError::BlockNotAllocated`] if the block is free.
    /// - [`ContinuousBatchError::OutOfBlocks`] if the pool has no room for the
    ///   copy. A copy-on-write **can** fail under memory pressure, and pretending
    ///   otherwise would be the bug: the engine reserves the block *before* it
    ///   commits to the write.
    pub fn copy_on_write(&mut self, id: KvBlockId) -> ContinuousBatchResult<KvBlockId> {
        let total = self.config.total_blocks;
        let source = self
            .blocks
            .get(id.index())
            .ok_or(ContinuousBatchError::UnknownBlock { block: id, total })?;
        if source.ref_count() == 0 {
            return Err(ContinuousBatchError::BlockNotAllocated {
                block: id,
                operation: "copy_on_write",
            });
        }

        // The copy itself. `source` has `ref_count >= 1`, so it is not in the
        // free list and `allocate_block` below cannot possibly hand back the
        // very block being copied.
        let keys = source.raw_keys().to_vec();
        let values = source.raw_values().to_vec();
        let positions = source.positions().to_vec();
        let tokens = source.tokens().to_vec();
        let num_filled = source.num_filled();

        let copy = self.allocate_block()?;
        self.blocks[copy.index()]
            .copy_contents_from(&keys, &values, &positions, &tokens, num_filled);

        // Only now, once the copy is safely made, does the caller stop referring
        // to the original.
        self.release_block(id)?;
        self.cow_copies += 1;
        self.debug_verify();
        Ok(copy)
    }

    /// Publish a **full** block under a prefix-cache key.
    ///
    /// Ignored (deliberately, and reported as `false`) when the block is not
    /// full, or when the hash is already taken by another block: at most one
    /// block may carry a given hash, which is what makes the index and the blocks
    /// impossible to disagree.
    ///
    /// # Errors
    ///
    /// - [`ContinuousBatchError::UnknownBlock`] if the id is out of range.
    /// - [`ContinuousBatchError::BlockNotAllocated`] if the block is free.
    pub fn publish_prefix_block(
        &mut self,
        id: KvBlockId,
        hash: u64,
    ) -> ContinuousBatchResult<bool> {
        let total = self.config.total_blocks;
        let block = self
            .blocks
            .get_mut(id.index())
            .ok_or(ContinuousBatchError::UnknownBlock { block: id, total })?;
        if block.ref_count() == 0 {
            return Err(ContinuousBatchError::BlockNotAllocated {
                block: id,
                operation: "publish_prefix_block",
            });
        }
        if !block.is_full() || block.content_hash().is_some() {
            return Ok(false);
        }
        if self.prefix_index.contains_key(&hash) {
            // Somebody else got there first — two sequences prefilled the same
            // prefix in the same step and both missed. Theirs stays; ours simply
            // is not a cache entry. Correct, and it keeps the one-block-per-hash
            // invariant free of special cases.
            return Ok(false);
        }
        self.blocks[id.index()].set_content_hash(Some(hash));
        self.prefix_index.insert(hash, id);
        self.debug_verify();
        Ok(true)
    }

    /// Look a prefix block up by its rolling hash.
    ///
    /// A hit here is a *candidate*, not a decision: the caller must still verify
    /// the block's token ids and positions against its own before adopting it
    /// (see the module docs on why the hash is not a proof). The returned block
    /// may be free — a free block is still a valid cache entry — and adopting it
    /// revives it.
    pub fn lookup_prefix_block(&mut self, hash: u64) -> Option<KvBlockId> {
        if let Some(id) = self.prefix_index.get(&hash).copied() {
            self.prefix_hits += 1;
            Some(id)
        } else {
            self.prefix_misses += 1;
            None
        }
    }

    /// A snapshot of the pool, including the sharing factor.
    #[must_use]
    pub fn stats(&self) -> KvBlockAllocatorStats {
        let mut shared_blocks = 0;
        let mut total_references = 0;
        for block in &self.blocks {
            if block.ref_count() > 1 {
                shared_blocks += 1;
            }
            total_references += block.ref_count();
        }
        KvBlockAllocatorStats {
            total_blocks: self.config.total_blocks,
            allocated_blocks: self.allocated_blocks,
            free_blocks: self.free_blocks(),
            shared_blocks,
            total_references,
            cached_blocks: self.prefix_index.len(),
            allocations: self.allocations,
            frees: self.frees,
            cow_copies: self.cow_copies,
            prefix_hits: self.prefix_hits,
            prefix_misses: self.prefix_misses,
        }
    }

    /// Copy-on-write copies performed over the pool's life.
    #[must_use]
    pub const fn cow_copies(&self) -> u64 {
        self.cow_copies
    }

    /// Walk the entire pool and check every structural invariant.
    ///
    /// This is the module's conscience. It is not a smoke test — it is the
    /// complete list of things that must be true of the pool between any two
    /// operations, and the randomized churn test calls it after *every* one of
    /// several thousand alloc/free/fork/copy-on-write/preempt operations:
    ///
    /// 1. `allocated_blocks` really is the number of blocks with `ref_count > 0`.
    /// 2. `free + allocated == total`.
    /// 3. The free list contains exactly the blocks with `ref_count == 0` — no
    ///    duplicates, no stragglers, no tombstones.
    /// 4. Every index entry `hash -> block` names a block whose own
    ///    `content_hash` is that same hash (the map cannot point at a block that
    ///    has forgotten it).
    /// 5. Every block carrying a `content_hash` is the block the index maps that
    ///    hash to (a block cannot claim a cache entry it does not hold).
    /// 6. Every cached block is **full** — a partially filled block is still
    ///    being written and must never be shared.
    /// 7. Every block's live slot count, position count and token count agree.
    ///
    /// # Errors
    ///
    /// [`ContinuousBatchError::InvariantViolated`] naming the first statement
    /// that does not hold.
    pub fn verify_invariants(&self) -> ContinuousBatchResult<()> {
        let violated = |reason: String| ContinuousBatchError::InvariantViolated { reason };

        let live = self.blocks.iter().filter(|b| b.ref_count() > 0).count();
        if live != self.allocated_blocks {
            return Err(violated(format!(
                "allocated_blocks is {} but {live} block(s) have a non-zero ref_count",
                self.allocated_blocks
            )));
        }
        if self.free_blocks() + self.allocated_blocks != self.config.total_blocks {
            return Err(violated(format!(
                "conservation broken: {} free + {} allocated != {} total",
                self.free_blocks(),
                self.allocated_blocks,
                self.config.total_blocks
            )));
        }

        // The free list must be exactly the free set, as a set *and* as a
        // multiset: a duplicate entry would let one block be handed to two
        // sequences.
        let mut seen = vec![false; self.blocks.len()];
        for &id in &self.free_list {
            let Some(block) = self.blocks.get(id.index()) else {
                return Err(violated(format!("free list holds out-of-range {id}")));
            };
            if block.ref_count() != 0 {
                return Err(violated(format!(
                    "free list holds {id}, which has ref_count {}",
                    block.ref_count()
                )));
            }
            if seen[id.index()] {
                return Err(violated(format!("free list holds {id} twice")));
            }
            seen[id.index()] = true;
        }
        for block in &self.blocks {
            if block.ref_count() == 0 && !seen[block.id().index()] {
                return Err(violated(format!(
                    "{} has ref_count 0 but is not in the free list",
                    block.id()
                )));
            }
        }

        for (&hash, &id) in &self.prefix_index {
            let Some(block) = self.blocks.get(id.index()) else {
                return Err(violated(format!("prefix index maps {hash:#x} to {id}")));
            };
            if block.content_hash() != Some(hash) {
                return Err(violated(format!(
                    "prefix index maps {hash:#x} to {id}, whose content_hash is {:?}",
                    block.content_hash()
                )));
            }
            if !block.is_full() {
                return Err(violated(format!(
                    "{id} is cached but only {}/{} slot(s) are filled",
                    block.num_filled(),
                    block.block_size()
                )));
            }
        }
        for block in &self.blocks {
            if let Some(hash) = block.content_hash()
                && self.prefix_index.get(&hash) != Some(&block.id())
            {
                return Err(violated(format!(
                    "{} claims content_hash {hash:#x}, which the index does not map to it",
                    block.id()
                )));
            }
            if block.positions().len() != block.num_filled()
                || block.tokens().len() != block.num_filled()
            {
                return Err(violated(format!(
                    "{} has {} filled slot(s) but {} position(s) and {} token(s)",
                    block.id(),
                    block.num_filled(),
                    block.positions().len(),
                    block.tokens().len()
                )));
            }
        }

        Ok(())
    }

    /// The invariant check, in debug builds only.
    ///
    /// Release builds pay nothing; debug builds (and therefore the test suite,
    /// *every* test, not only the churn one) re-verify the whole pool after each
    /// mutation.
    fn debug_verify(&self) {
        debug_assert!(
            self.verify_invariants().is_ok(),
            "allocator invariant violated: {:?}",
            self.verify_invariants().err()
        );
    }

    /// Mutable access to a block, for the block table's write paths.
    ///
    /// Crate-internal: everything outside this module reaches blocks through a
    /// [`KvBlockTable`], which is what enforces "you may only write into a block
    /// you hold the sole reference to".
    ///
    /// [`KvBlockTable`]: crate::continuous_batching::KvBlockTable
    pub(super) fn block_mut(&mut self, id: KvBlockId) -> ContinuousBatchResult<&mut KvBlock> {
        let total = self.config.total_blocks;
        self.blocks
            .get_mut(id.index())
            .ok_or(ContinuousBatchError::UnknownBlock { block: id, total })
    }

    /// Immutable access with a proper error rather than an `Option`.
    pub(super) fn block_ref(&self, id: KvBlockId) -> ContinuousBatchResult<&KvBlock> {
        let total = self.config.total_blocks;
        self.blocks
            .get(id.index())
            .ok_or(ContinuousBatchError::UnknownBlock { block: id, total })
    }
}

// ── Prefix hashing ───────────────────────────────────────────────────────────

/// The seed of the rolling prefix hash: `FNV-1a`'s 64-bit offset basis.
pub const PREFIX_HASH_SEED: u64 = 0xcbf2_9ce4_8422_2325;

/// The 64-bit `FNV-1a` prime.
const PREFIX_HASH_PRIME: u64 = 0x0000_0100_0000_01b3;

/// Fold one block's token ids into the running prefix hash.
///
/// The hash is **rolling**: block `k`'s key is a function of *every* token from
/// the start of the sequence up to the end of block `k`, not just of block `k`'s
/// own tokens. That is what makes it a *prefix* key — two sequences whose third
/// blocks happen to contain the same four tokens, but whose first two blocks
/// differ, must not collide onto the same cache entry, and with a rolling hash
/// they do not (except by the ordinary 64-bit collision chance, which the
/// element-by-element verification at adoption time then catches).
#[must_use]
pub fn fold_prefix_hash(previous: u64, tokens: &[u32]) -> u64 {
    let mut hash = previous;
    for &token in tokens {
        for byte in token.to_le_bytes() {
            hash ^= u64::from(byte);
            hash = hash.wrapping_mul(PREFIX_HASH_PRIME);
        }
    }
    hash
}
