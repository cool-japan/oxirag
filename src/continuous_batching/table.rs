//! The per-sequence block table: logical token `t` lives in physical block
//! `blocks[t / block_size]`, at slot `t % block_size`.
//!
//! This is the indirection that the whole module exists for. A sequence's KV is
//! *not* a contiguous run of memory that must be reserved up front at its
//! maximum possible length; it is a **list of block ids**, extended one block at
//! a time as the sequence actually grows, whose blocks may be shared with other
//! sequences, may live anywhere in the pool, and may be swapped for copies
//! underneath the sequence when somebody writes.
//!
//! # The two rules of writing
//!
//! 1. **Only the tail block is ever written**, and only at slot `num_filled`.
//!    Every other block of a live sequence is full and finished.
//! 2. **A shared block is never written.** If the tail's `ref_count > 1`, the
//!    write copies it first ([`KvBlockAllocator::copy_on_write`]) and re-points
//!    the table's last entry at the copy. The other sharers keep the original,
//!    bit-for-bit.
//!
//! Together these give the copy-on-write property that makes forking a sequence
//! O(number of blocks) *pointer* work instead of O(bytes): a fork touches no KV
//! bytes at all, and the first divergent token costs exactly one block copy —
//! never the whole history.
//!
//! Note the consequence of rule 1 that makes prefix sharing cheap: a **full**
//! shared block can never need a copy, because nothing will ever be written into
//! it again. A 4000-token shared system prompt is 4000/`block_size` blocks that
//! are copied *zero* times, no matter how many sequences attend over them.
//!
//! # Positions are absolute, and they are the mask
//!
//! Slot `t` carries the token's **absolute stream position**, exactly as
//! [`KvCacheTensor`] does, taken from a monotone counter that compaction never
//! rewinds. So after [`KvBlockTable::compact_to_positions`] has dropped tokens
//! from the middle of a sequence, the survivors still know where they were, the
//! causal mask built from them is still the mask of the *uncompressed* sequence,
//! and [`super::attention::paged_attention`] is still exactly right.
//!
//! [`KvCacheTensor`]: crate::kv_cache_compression::KvCacheTensor

use crate::kv_cache_compression::KvCacheTensor;

use super::allocator::KvBlockAllocator;
use super::types::{
    ContinuousBatchConfig, ContinuousBatchError, ContinuousBatchResult, KvBlockId, check_finite,
};

// ── KvBlockSwapSlab ──────────────────────────────────────────────────────────

/// One swap-preempted sequence's KV, copied out of the pool.
///
/// A slab holds the bytes, the absolute positions and the token ids of every
/// block the sequence had, plus the table's monotone position counter — i.e.
/// everything needed to rebuild the sequence **bit-for-bit**, which
/// [`KvBlockTable::swap_in`] does.
///
/// # What a swap costs, honestly
///
/// Swapping *out* copies the blocks; swapping *in* copies them back into
/// whatever blocks happen to be free, which will generally be **different
/// physical blocks**. Two consequences, both real:
///
/// - The restored sequence's block ids differ from its original ones. Nothing
///   may depend on a block id's stability, and nothing here does.
/// - A block that was **shared** when it was swapped out comes back
///   **private**. The sharing is not restored, because the slab is a copy of the
///   bytes, not a reference to the original block. The KV is identical; the
///   memory is not deduplicated any more. This is a real cost of swapping a
///   sharer, and it is recorded here rather than papered over.
#[derive(Debug, Clone, PartialEq)]
pub struct KvBlockSwapSlab {
    block_size: usize,
    token_buffer_len: usize,
    next_position: usize,
    num_tokens: usize,
    blocks: Vec<SwapEntry>,
}

/// One swapped-out block's contents.
#[derive(Debug, Clone, PartialEq)]
struct SwapEntry {
    keys: Vec<f32>,
    values: Vec<f32>,
    positions: Vec<usize>,
    tokens: Vec<u32>,
    num_filled: usize,
}

impl KvBlockSwapSlab {
    /// Blocks held in the slab.
    #[must_use]
    pub fn num_blocks(&self) -> usize {
        self.blocks.len()
    }

    /// Live token slots held in the slab.
    #[must_use]
    pub const fn num_tokens(&self) -> usize {
        self.num_tokens
    }

    /// Bytes of `f32` KV payload the slab occupies off-pool (keys **and**
    /// values). This is the memory a swap preemption *moves* rather than frees.
    #[must_use]
    pub fn bytes(&self) -> usize {
        2 * self.blocks.len() * self.block_size * self.token_buffer_len * size_of::<f32>()
    }
}

// ── KvBlockTable ─────────────────────────────────────────────────────────────

/// One sequence's map from logical token index to physical block.
///
/// See the module docs. The table owns no bytes — it owns *references*, and the
/// bytes live in the [`KvBlockAllocator`] that handed them out. Every read
/// therefore takes the allocator too, which is not an inconvenience but the
/// point: it is impossible to read a block this table does not hold.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct KvBlockTable {
    /// Logical block index -> physical block. Every entry but the last is full.
    blocks: Vec<KvBlockId>,
    num_tokens: usize,
    block_size: usize,
    token_buffer_len: usize,
    /// The position the next appended token will receive. Monotone; compaction
    /// never rewinds it, exactly as in [`KvCacheTensor`].
    next_position: usize,
}

impl KvBlockTable {
    /// An empty table for a sequence in the given pool geometry. Allocates
    /// nothing: a sequence costs no memory until it has a token.
    #[must_use]
    pub const fn new(config: &ContinuousBatchConfig) -> Self {
        Self {
            blocks: Vec::new(),
            num_tokens: 0,
            block_size: config.block_size,
            token_buffer_len: config.token_buffer_len(),
            next_position: 0,
        }
    }

    /// An empty table whose monotone position counter is preset to
    /// `next_position`, ready to be refilled at explicit absolute positions by
    /// [`Self::restore_token`].
    ///
    /// This is what a **recompute** resume rebuilds into. A sequence that had
    /// decoded 40 tokens and then dropped some of them to an eviction policy must
    /// come back holding *those* positions, with the counter still at 40 — not as
    /// a fresh sequence that happens to have the same tokens.
    #[must_use]
    pub const fn for_restore(config: &ContinuousBatchConfig, next_position: usize) -> Self {
        Self {
            blocks: Vec::new(),
            num_tokens: 0,
            block_size: config.block_size,
            token_buffer_len: config.token_buffer_len(),
            next_position,
        }
    }

    /// Live token slots.
    #[must_use]
    pub const fn len(&self) -> usize {
        self.num_tokens
    }

    /// Whether the sequence holds no tokens.
    #[must_use]
    pub const fn is_empty(&self) -> bool {
        self.num_tokens == 0
    }

    /// Physical blocks this table references. Always
    /// `ceil(len / block_size)`.
    #[must_use]
    pub fn num_blocks(&self) -> usize {
        self.blocks.len()
    }

    /// The physical blocks, in logical order.
    #[must_use]
    pub fn block_ids(&self) -> &[KvBlockId] {
        &self.blocks
    }

    /// The position the next appended token will receive.
    #[must_use]
    pub const fn next_position(&self) -> usize {
        self.next_position
    }

    /// Slots wasted in the partially filled tail block — the *internal
    /// fragmentation* of this sequence, which is the price paid for never having
    /// to reserve a contiguous run at the maximum length.
    ///
    /// It is bounded by `block_size - 1`, **per sequence**, and by nothing else.
    /// That bound is the entire memory argument for paged attention.
    #[must_use]
    pub fn wasted_slots(&self) -> usize {
        self.blocks.len() * self.block_size - self.num_tokens
    }

    // ── Reading ──────────────────────────────────────────────────────────────

    /// The `head_dim`-long key vector of one `(layer, token, head)`, read
    /// **through the block table**, or `None` if any index is out of range.
    #[must_use]
    pub fn key<'pool>(
        &self,
        allocator: &'pool KvBlockAllocator,
        layer: usize,
        token: usize,
        head: usize,
    ) -> Option<&'pool [f32]> {
        let config = allocator.config();
        if layer >= config.num_layers || head >= config.num_heads {
            return None;
        }
        let (id, slot) = self.locate(token).ok()?;
        allocator.block(id)?.key(layer, slot, head, config.head_dim)
    }

    /// The `head_dim`-long value vector of one `(layer, token, head)`. See
    /// [`Self::key`].
    #[must_use]
    pub fn value<'pool>(
        &self,
        allocator: &'pool KvBlockAllocator,
        layer: usize,
        token: usize,
        head: usize,
    ) -> Option<&'pool [f32]> {
        let config = allocator.config();
        if layer >= config.num_layers || head >= config.num_heads {
            return None;
        }
        let (id, slot) = self.locate(token).ok()?;
        allocator
            .block(id)?
            .value(layer, slot, head, config.head_dim)
    }

    /// The absolute stream position of one token slot.
    #[must_use]
    pub fn position_at(&self, allocator: &KvBlockAllocator, token: usize) -> Option<usize> {
        let (id, slot) = self.locate(token).ok()?;
        allocator.block(id)?.positions().get(slot).copied()
    }

    /// The token id of one token slot.
    #[must_use]
    pub fn token_at(&self, allocator: &KvBlockAllocator, token: usize) -> Option<u32> {
        let (id, slot) = self.locate(token).ok()?;
        allocator.block(id)?.tokens().get(slot).copied()
    }

    /// The absolute stream position of every live slot, in slot order (strictly
    /// increasing, gappy after a compaction).
    ///
    /// This is the vector the causal mask is built from.
    ///
    /// # Errors
    ///
    /// [`ContinuousBatchError::UnknownBlock`] if the table references a block
    /// outside the pool — i.e. if it was built by a different allocator.
    pub fn positions(&self, allocator: &KvBlockAllocator) -> ContinuousBatchResult<Vec<usize>> {
        let mut positions = Vec::with_capacity(self.num_tokens);
        for &id in &self.blocks {
            positions.extend_from_slice(allocator.block_ref(id)?.positions());
        }
        debug_assert_eq!(positions.len(), self.num_tokens);
        Ok(positions)
    }

    /// The token id of every live slot, in slot order.
    ///
    /// # Errors
    ///
    /// [`ContinuousBatchError::UnknownBlock`] if the table references a block
    /// outside the pool.
    pub fn tokens(&self, allocator: &KvBlockAllocator) -> ContinuousBatchResult<Vec<u32>> {
        let mut tokens = Vec::with_capacity(self.num_tokens);
        for &id in &self.blocks {
            tokens.extend_from_slice(allocator.block_ref(id)?.tokens());
        }
        Ok(tokens)
    }

    /// Locate a logical token: `(physical block, slot within it)`.
    fn locate(&self, token: usize) -> ContinuousBatchResult<(KvBlockId, usize)> {
        if token >= self.num_tokens {
            return Err(ContinuousBatchError::InvalidSelection {
                reason: format!(
                    "token {token} is out of range for a sequence of {} token(s)",
                    self.num_tokens
                ),
            });
        }
        let logical = token / self.block_size;
        let slot = token % self.block_size;
        let id = self.blocks.get(logical).copied().ok_or_else(|| {
            ContinuousBatchError::InvalidSelection {
                reason: format!(
                    "logical block {logical} is missing from a table of {} block(s)",
                    self.blocks.len()
                ),
            }
        })?;
        Ok((id, slot))
    }

    /// The whole `[layer][head][dim]` key and value buffers of one live slot.
    fn slot_buffers<'pool>(
        &self,
        allocator: &'pool KvBlockAllocator,
        token: usize,
    ) -> ContinuousBatchResult<(&'pool [f32], &'pool [f32])> {
        let (id, slot) = self.locate(token)?;
        let block = allocator.block_ref(id)?;
        Ok((block.slot_keys(slot), block.slot_values(slot)))
    }

    // ── Writing ──────────────────────────────────────────────────────────────

    /// Append one token's KV, assigning it the next absolute stream position.
    ///
    /// This is the decode step's write, and it is the only way a sequence grows:
    /// **one token**. It extends the table by a block only when the tail block is
    /// full, and it copies the tail block first if the tail is shared (see the
    /// module docs).
    ///
    /// `keys` and `values` are each laid out `[layer][head][dim]` and must be
    /// exactly [`ContinuousBatchConfig::token_buffer_len`] elements long — the
    /// same buffers [`KvCacheTensor::append_token`] takes, deliberately.
    ///
    /// # Errors
    ///
    /// - [`ContinuousBatchError::ShapeMismatch`] if either buffer is the wrong
    ///   length.
    /// - [`ContinuousBatchError::NonFiniteInput`] if either buffer holds a `NaN`
    ///   or an infinity. Enforced *here*, at the boundary, because the attention
    ///   kernel's finiteness guarantee is a theorem about finite inputs.
    /// - [`ContinuousBatchError::OutOfBlocks`] if a new block (or a
    ///   copy-on-write copy) is needed and the pool is empty.
    ///
    /// [`KvCacheTensor::append_token`]:
    ///     crate::kv_cache_compression::KvCacheTensor::append_token
    pub fn append_token(
        &mut self,
        allocator: &mut KvBlockAllocator,
        token: u32,
        keys: &[f32],
        values: &[f32],
    ) -> ContinuousBatchResult<usize> {
        self.check_buffer("key buffer", keys)?;
        self.check_buffer("value buffer", values)?;
        check_finite(keys, "key buffer")?;
        check_finite(values, "value buffer")?;

        let position = self.next_position;
        self.push_slot(allocator, token, position, keys, values)?;
        self.next_position += 1;
        Ok(position)
    }

    /// Append one token at an **explicit** absolute position, without moving the
    /// monotone counter.
    ///
    /// This is the recompute resume's write. It is deliberately separate from
    /// [`Self::append_token`]: appending assigns the *next* position, whereas a
    /// resume must give each token back the position it already had, which may be
    /// gappy and which certainly is not `next_position`.
    ///
    /// # Errors
    ///
    /// - [`ContinuousBatchError::ShapeMismatch`] / [`ContinuousBatchError::NonFiniteInput`]
    ///   exactly as [`Self::append_token`].
    /// - [`ContinuousBatchError::InvalidSelection`] if `position` is not strictly
    ///   greater than the last live position, or is at or beyond
    ///   [`Self::next_position`] — a restored token cannot land in the future.
    /// - [`ContinuousBatchError::OutOfBlocks`] if the pool is empty.
    pub fn restore_token(
        &mut self,
        allocator: &mut KvBlockAllocator,
        token: u32,
        position: usize,
        keys: &[f32],
        values: &[f32],
    ) -> ContinuousBatchResult<()> {
        self.check_buffer("key buffer", keys)?;
        self.check_buffer("value buffer", values)?;
        check_finite(keys, "key buffer")?;
        check_finite(values, "value buffer")?;

        if position >= self.next_position {
            return Err(ContinuousBatchError::InvalidSelection {
                reason: format!(
                    "restored position {position} is at or beyond the sequence's next position {}",
                    self.next_position
                ),
            });
        }
        if self.num_tokens > 0 {
            let last = self.position_at(allocator, self.num_tokens - 1);
            if let Some(previous) = last
                && position <= previous
            {
                return Err(ContinuousBatchError::InvalidSelection {
                    reason: format!(
                        "restored positions must be strictly increasing, got {position} after {previous}"
                    ),
                });
            }
        }

        self.push_slot(allocator, token, position, keys, values)
    }

    /// Write one token into the tail, copying the tail first if it is shared and
    /// extending by a block if it is full. Does **not** touch `next_position`,
    /// so compaction can replay a token at its original absolute position.
    fn push_slot(
        &mut self,
        allocator: &mut KvBlockAllocator,
        token: u32,
        position: usize,
        keys: &[f32],
        values: &[f32],
    ) -> ContinuousBatchResult<()> {
        let target = match self.blocks.last().copied() {
            // There is a tail with room in it.
            Some(tail) if !allocator.block_ref(tail)?.is_full() => {
                if allocator.block_ref(tail)?.is_shared() {
                    // Copy-on-write. Somebody else is reading this block; we may
                    // not write into it. Note that this can fail with
                    // `OutOfBlocks` — a copy-on-write needs a block, and under
                    // pressure there may not be one.
                    let copy = allocator.copy_on_write(tail)?;
                    if let Some(entry) = self.blocks.last_mut() {
                        *entry = copy;
                    }
                    copy
                } else {
                    tail
                }
            }
            // No blocks at all, or the tail is full: grow by one block. A *full*
            // shared block needs no copy — nothing will ever be written into it.
            _ => {
                let fresh = allocator.allocate_block()?;
                self.blocks.push(fresh);
                fresh
            }
        };

        allocator
            .block_mut(target)?
            .push_slot(token, position, keys, values);
        self.num_tokens += 1;
        Ok(())
    }

    /// Adopt a **full** block from another sequence as this sequence's next
    /// logical block, without copying a byte and without computing a single key.
    ///
    /// This is prompt-prefix sharing. It is exact: the block is accepted only if
    /// its token ids compare **equal, element by element**, to `expected_tokens`,
    /// *and* its absolute positions are exactly the ones this sequence would have
    /// assigned. A hash hit is a candidate; this is the proof.
    ///
    /// # Errors
    ///
    /// - [`ContinuousBatchError::PrefixMismatch`] if the block is not full, if
    ///   this table is not currently block-aligned (a prefix block can only be
    ///   adopted at a block boundary), if the token ids differ, or if the
    ///   positions do not line up.
    /// - [`ContinuousBatchError::UnknownBlock`] if the id is out of range.
    pub fn adopt_prefix_block(
        &mut self,
        allocator: &mut KvBlockAllocator,
        id: KvBlockId,
        expected_tokens: &[u32],
    ) -> ContinuousBatchResult<()> {
        if !self.num_tokens.is_multiple_of(self.block_size) {
            return Err(ContinuousBatchError::PrefixMismatch {
                reason: format!(
                    "table holds {} token(s), which is not a multiple of the {}-token block size; \
                     a prefix block can only be adopted at a block boundary",
                    self.num_tokens, self.block_size
                ),
            });
        }

        let block = allocator.block_ref(id)?;
        if !block.is_full() {
            return Err(ContinuousBatchError::PrefixMismatch {
                reason: format!(
                    "{id} holds {}/{} slot(s); only a full block may be shared",
                    block.num_filled(),
                    block.block_size()
                ),
            });
        }
        if block.tokens() != expected_tokens {
            return Err(ContinuousBatchError::PrefixMismatch {
                reason: format!(
                    "{id} holds tokens {:?} but the adopting sequence expects {expected_tokens:?}",
                    block.tokens()
                ),
            });
        }
        let expected_positions: Vec<usize> =
            (self.next_position..self.next_position + self.block_size).collect();
        if block.positions() != expected_positions.as_slice() {
            return Err(ContinuousBatchError::PrefixMismatch {
                reason: format!(
                    "{id} holds positions {:?} but the adopting sequence expects {expected_positions:?}",
                    block.positions()
                ),
            });
        }

        allocator.add_ref(id)?;
        self.blocks.push(id);
        self.num_tokens += self.block_size;
        self.next_position += self.block_size;
        Ok(())
    }

    /// Fork this sequence: the child shares **every** block, copy-on-write, and
    /// not one byte of KV is copied.
    ///
    /// The first token either side appends after the fork copies exactly one
    /// block — the shared, partially filled tail — and no more. Everything before
    /// it stays shared forever, because a full block is never written again.
    ///
    /// # Errors
    ///
    /// [`ContinuousBatchError::UnknownBlock`] if this table references a block
    /// outside `allocator`'s pool. No reference is taken in that case: the fork
    /// is validated before it is committed, so a failure leaves the pool exactly
    /// as it was.
    pub fn fork(&self, allocator: &mut KvBlockAllocator) -> ContinuousBatchResult<Self> {
        // Validate first, then commit: a half-forked table would leak references.
        for &id in &self.blocks {
            allocator.block_ref(id)?;
        }
        for &id in &self.blocks {
            allocator.add_ref(id)?;
        }
        Ok(Self {
            blocks: self.blocks.clone(),
            num_tokens: self.num_tokens,
            block_size: self.block_size,
            token_buffer_len: self.token_buffer_len,
            next_position: self.next_position,
        })
    }

    /// Drop every reference this table holds and empty it, returning the number
    /// of blocks that actually returned to the free list.
    ///
    /// That number is **smaller** than [`Self::num_blocks`] exactly when some of
    /// the blocks were still shared with another sequence — releasing a sharer
    /// frees nothing. This is what makes preemption's yield unpredictable, and
    /// why the engine re-checks the free count after every eviction instead of
    /// assuming.
    ///
    /// `next_position` is **not** rewound: a released sequence that is later
    /// recomputed must give its tokens back the positions they had.
    ///
    /// # Errors
    ///
    /// - [`ContinuousBatchError::UnknownBlock`] if a block id is out of range.
    /// - [`ContinuousBatchError::BlockNotAllocated`] on a double free.
    pub fn release(&mut self, allocator: &mut KvBlockAllocator) -> ContinuousBatchResult<usize> {
        let before = allocator.free_blocks();
        for id in std::mem::take(&mut self.blocks) {
            allocator.release_block(id)?;
        }
        self.num_tokens = 0;
        Ok(allocator.free_blocks() - before)
    }

    // ── Compaction ───────────────────────────────────────────────────────────

    /// Physically drop every token whose absolute position is not in `keep`, and
    /// re-pack the survivors into fresh blocks.
    ///
    /// This is the paged counterpart of [`KvCacheTensor::retain_tokens`], and it
    /// is what lets a `kv_cache_compression` eviction plan (H2O, `SnapKV`,
    /// `StreamingLLM`) be applied to a *paged* sequence: materialise, compress,
    /// then compact the block table to the positions the policy retained.
    ///
    /// The survivors keep their **absolute positions**, so the sequence's
    /// position vector becomes strictly increasing but gappy, and every mask
    /// built from it afterwards is still the mask of the uncompressed sequence.
    /// `next_position` is not rewound.
    ///
    /// The survivors are re-packed into blocks taken fresh from the pool, so the
    /// compaction **un-shares**: a sequence that compacts stops sharing its
    /// prefix blocks with anybody. That is a real cost (it is why one would
    /// compact a sequence rather than a whole prefix), and it is why the old
    /// blocks are released *before* the new ones are taken — a compaction can
    /// never fail for want of memory, because it strictly shrinks.
    ///
    /// # Errors
    ///
    /// - [`ContinuousBatchError::InvalidSelection`] if `keep` is not strictly
    ///   increasing, or names a position this sequence does not hold.
    /// - [`ContinuousBatchError::OutOfBlocks`] cannot occur in practice (see
    ///   above) but is propagated rather than assumed away.
    ///
    /// [`KvCacheTensor::retain_tokens`]:
    ///     crate::kv_cache_compression::KvCacheTensor::retain_tokens
    pub fn compact_to_positions(
        &mut self,
        allocator: &mut KvBlockAllocator,
        keep: &[usize],
    ) -> ContinuousBatchResult<()> {
        let positions = self.positions(allocator)?;

        // Resolve every kept position to its slot, checking monotonicity and
        // membership as we go. Both vectors are strictly increasing, so one pass
        // with a cursor is enough.
        let mut slots = Vec::with_capacity(keep.len());
        let mut cursor = 0usize;
        let mut previous: Option<usize> = None;
        for &position in keep {
            if let Some(prior) = previous
                && position <= prior
            {
                return Err(ContinuousBatchError::InvalidSelection {
                    reason: format!(
                        "retained positions must be strictly increasing, got {position} after {prior}"
                    ),
                });
            }
            previous = Some(position);
            while cursor < positions.len() && positions[cursor] < position {
                cursor += 1;
            }
            if positions.get(cursor) != Some(&position) {
                return Err(ContinuousBatchError::InvalidSelection {
                    reason: format!("position {position} is not live in this sequence"),
                });
            }
            slots.push(cursor);
        }

        // Copy the survivors out before anything is released.
        let mut survivors = Vec::with_capacity(slots.len());
        for &slot in &slots {
            let (id, offset) = self.locate(slot)?;
            let block = allocator.block_ref(id)?;
            let token = block.tokens().get(offset).copied().ok_or_else(|| {
                ContinuousBatchError::InvariantViolated {
                    reason: format!("{id} has no token id at slot {offset}"),
                }
            })?;
            let position = block.positions().get(offset).copied().ok_or_else(|| {
                ContinuousBatchError::InvariantViolated {
                    reason: format!("{id} has no position at slot {offset}"),
                }
            })?;
            survivors.push((
                token,
                position,
                block.slot_keys(offset).to_vec(),
                block.slot_values(offset).to_vec(),
            ));
        }

        // Release *first*: the rebuild can then never fail for want of blocks,
        // because it needs no more than we just gave back.
        for id in std::mem::take(&mut self.blocks) {
            allocator.release_block(id)?;
        }
        self.num_tokens = 0;

        for (token, position, keys, values) in survivors {
            self.push_slot(allocator, token, position, &keys, &values)?;
        }
        Ok(())
    }

    // ── Materialisation ──────────────────────────────────────────────────────

    /// Gather this sequence's scattered blocks into a contiguous
    /// [`KvCacheTensor`] — the exact tensor a non-paged cache would have held.
    ///
    /// This is the bridge to `kv_cache_compression`: it is how a paged sequence
    /// is handed to [`scaled_dot_product_attention`], to a [`KvCacheCompressor`],
    /// or to anything else that wants its KV dense. It is also the module's
    /// **oracle**: the headline test asserts that attention computed over the
    /// blocks equals attention computed over this tensor, bit for bit.
    ///
    /// # The gappy case costs a dense pass
    ///
    /// [`KvCacheTensor`] assigns positions from its own monotone counter and
    /// exposes no way to set them, so the only faithful way to reconstruct a
    /// *compacted* (gappy) sequence is to lay the whole position range down
    /// densely — zeros where the sequence no longer holds a token — and then
    /// [`KvCacheTensor::retain_tokens`] the live slots, which preserves their
    /// positions and the counter. That is exactly how the upstream module
    /// produces a gappy cache itself, so the result is bit-identical to one; it
    /// simply costs `O(next_position)` transiently rather than `O(len)`. A dense
    /// sequence — every live sequence that has not been compacted — takes the
    /// straight path and pays nothing extra.
    ///
    /// # Errors
    ///
    /// - [`ContinuousBatchError::UnknownBlock`] if the table references a block
    ///   outside the pool.
    /// - [`ContinuousBatchError::KvCache`] if the upstream cache rejects the
    ///   geometry or the data (it cannot: the buffers came out of it).
    ///
    /// [`scaled_dot_product_attention`]:
    ///     crate::kv_cache_compression::scaled_dot_product_attention
    /// [`KvCacheCompressor`]: crate::kv_cache_compression::KvCacheCompressor
    /// [`KvCacheTensor::retain_tokens`]:
    ///     crate::kv_cache_compression::KvCacheTensor::retain_tokens
    pub fn materialize(
        &self,
        allocator: &KvBlockAllocator,
    ) -> ContinuousBatchResult<KvCacheTensor> {
        let config = allocator.config();
        let mut tensor = KvCacheTensor::new(config.num_layers, config.num_heads, config.head_dim)?;
        if self.num_tokens == 0 {
            return Ok(tensor);
        }

        let positions = self.positions(allocator)?;
        let is_dense = positions.len() == self.next_position
            && positions
                .iter()
                .enumerate()
                .all(|(slot, &position)| slot == position);

        if is_dense {
            for token in 0..self.num_tokens {
                let (keys, values) = self.slot_buffers(allocator, token)?;
                tensor.append_token(keys, values)?;
            }
            return Ok(tensor);
        }

        let zeros = vec![0.0f32; self.token_buffer_len];
        let mut live = 0usize;
        for position in 0..self.next_position {
            if positions.get(live) == Some(&position) {
                let (keys, values) = self.slot_buffers(allocator, live)?;
                tensor.append_token(keys, values)?;
                live += 1;
            } else {
                tensor.append_token(&zeros, &zeros)?;
            }
        }
        // In the dense build, the slot index of the token at position `p` is
        // exactly `p`, so the live positions *are* the slots to retain.
        tensor.retain_tokens(&positions)?;
        Ok(tensor)
    }

    // ── Swapping ─────────────────────────────────────────────────────────────

    /// Copy this sequence's blocks out of the pool and release them.
    ///
    /// The table is left empty, holding no blocks; the slab holds everything
    /// needed to rebuild it bit-for-bit. See [`KvBlockSwapSlab`] for what the
    /// round trip does and does not preserve.
    ///
    /// # Errors
    ///
    /// - [`ContinuousBatchError::UnknownBlock`] if a block id is out of range.
    /// - [`ContinuousBatchError::BlockNotAllocated`] on a double free.
    pub fn swap_out(
        &mut self,
        allocator: &mut KvBlockAllocator,
    ) -> ContinuousBatchResult<KvBlockSwapSlab> {
        let mut entries = Vec::with_capacity(self.blocks.len());
        for &id in &self.blocks {
            let block = allocator.block_ref(id)?;
            entries.push(SwapEntry {
                keys: block.raw_keys().to_vec(),
                values: block.raw_values().to_vec(),
                positions: block.positions().to_vec(),
                tokens: block.tokens().to_vec(),
                num_filled: block.num_filled(),
            });
        }
        let slab = KvBlockSwapSlab {
            block_size: self.block_size,
            token_buffer_len: self.token_buffer_len,
            next_position: self.next_position,
            num_tokens: self.num_tokens,
            blocks: entries,
        };
        for id in std::mem::take(&mut self.blocks) {
            allocator.release_block(id)?;
        }
        self.num_tokens = 0;
        Ok(slab)
    }

    /// Rebuild a swapped-out sequence, bit-for-bit, in whatever blocks are free.
    ///
    /// All-or-nothing: either every block is restored or none is and the pool is
    /// untouched.
    ///
    /// # Errors
    ///
    /// - [`ContinuousBatchError::SwapMismatch`] if the slab's geometry is not
    ///   this pool's.
    /// - [`ContinuousBatchError::OutOfBlocks`] if the pool cannot hold it.
    pub fn swap_in(
        allocator: &mut KvBlockAllocator,
        slab: &KvBlockSwapSlab,
    ) -> ContinuousBatchResult<Self> {
        let config = allocator.config().clone();
        if slab.block_size != config.block_size
            || slab.token_buffer_len != config.token_buffer_len()
        {
            return Err(ContinuousBatchError::SwapMismatch {
                reason: format!(
                    "slab is {}-slot x {} f32/token, pool is {}-slot x {} f32/token",
                    slab.block_size,
                    slab.token_buffer_len,
                    config.block_size,
                    config.token_buffer_len()
                ),
            });
        }

        let ids = allocator.allocate_blocks(slab.blocks.len())?;
        for (&id, entry) in ids.iter().zip(slab.blocks.iter()) {
            allocator.block_mut(id)?.copy_contents_from(
                &entry.keys,
                &entry.values,
                &entry.positions,
                &entry.tokens,
                entry.num_filled,
            );
        }
        Ok(Self {
            blocks: ids,
            num_tokens: slab.num_tokens,
            block_size: slab.block_size,
            token_buffer_len: slab.token_buffer_len,
            next_position: slab.next_position,
        })
    }

    // ── Internals ────────────────────────────────────────────────────────────

    fn check_buffer(&self, what: &'static str, buffer: &[f32]) -> ContinuousBatchResult<()> {
        if buffer.len() != self.token_buffer_len {
            return Err(ContinuousBatchError::ShapeMismatch {
                what,
                expected: self.token_buffer_len,
                actual: buffer.len(),
            });
        }
        Ok(())
    }
}
