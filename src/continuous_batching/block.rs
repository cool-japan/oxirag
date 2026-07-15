//! The fixed-size KV block: `block_size` token slots of key/value bytes, a
//! reference count, and the bookkeeping that makes copy-on-write sharing exact.
//!
//! # Layout
//!
//! A block stores `block_size` **token slots**. One slot's keys are
//! `num_layers * num_heads * head_dim` contiguous `f32` — the *same* layout as
//! the buffer [`KvCacheTensor::append_token`] takes, deliberately, so that a
//! slot can be copied into a contiguous cache (or out of one) without a
//! transpose:
//!
//! ```text
//! keys[slot * token_buffer_len + layer * (num_heads * head_dim) + head * head_dim + d]
//! ```
//!
//! Slots `0..num_filled` are live. Slots at or beyond `num_filled` are **never
//! read** — the block table's length, not the block's capacity, bounds every
//! access path in this module.
//!
//! # Why a block owns its positions and its token ids
//!
//! Both are load-bearing, and neither is decoration:
//!
//! - **Positions** are the absolute stream positions of the slots, exactly as in
//!   [`KvCacheTensor::positions`]. They are what the causal mask is built from,
//!   so a shared block carries the mask information with it, and a block adopted
//!   by the wrong sequence at the wrong offset is *detectable* (its positions
//!   will not line up) rather than silently producing a wrong mask.
//! - **Token ids** are what makes prefix sharing **exact**. The content hash is a
//!   lookup key; the token ids are the proof. A hash hit whose token ids do not
//!   compare equal is rejected, so a 64-bit collision cannot silently graft one
//!   sequence's keys onto another's.
//!
//! [`KvCacheTensor::append_token`]:
//!     crate::kv_cache_compression::KvCacheTensor::append_token
//! [`KvCacheTensor::positions`]:
//!     crate::kv_cache_compression::KvCacheTensor::positions

use super::types::KvBlockId;

// ── KvBlock ──────────────────────────────────────────────────────────────────

/// One fixed-size block of the KV pool.
///
/// A block is created once, when the pool is built, and then recycled forever.
/// It is never reallocated and never resized: [`KvBlockAllocator`] hands out
/// *references* to blocks, and the block's `ref_count` is the number of block
/// tables currently pointing at it.
///
/// `ref_count == 0` means the block is in the free list. A free block **keeps
/// its bytes**, which is what lets it still serve as a prefix-cache hit for a
/// later request; the bytes are destroyed (and the cache entry dropped) only
/// when the block is actually re-allocated.
///
/// [`KvBlockAllocator`]: crate::continuous_batching::KvBlockAllocator
#[derive(Debug, Clone, PartialEq)]
pub struct KvBlock {
    id: KvBlockId,
    block_size: usize,
    token_buffer_len: usize,
    per_layer_token_stride: usize,
    keys: Vec<f32>,
    values: Vec<f32>,
    positions: Vec<usize>,
    tokens: Vec<u32>,
    num_filled: usize,
    ref_count: usize,
    content_hash: Option<u64>,
}

impl KvBlock {
    /// Build an empty, unreferenced block. Called only by the pool constructor.
    pub(super) fn new(
        id: KvBlockId,
        block_size: usize,
        token_buffer_len: usize,
        per_layer_token_stride: usize,
    ) -> Self {
        Self {
            id,
            block_size,
            token_buffer_len,
            per_layer_token_stride,
            keys: vec![0.0; block_size * token_buffer_len],
            values: vec![0.0; block_size * token_buffer_len],
            positions: Vec::with_capacity(block_size),
            tokens: Vec::with_capacity(block_size),
            num_filled: 0,
            ref_count: 0,
            content_hash: None,
        }
    }

    /// This block's physical id.
    #[must_use]
    pub const fn id(&self) -> KvBlockId {
        self.id
    }

    /// Token slots this block can hold.
    #[must_use]
    pub const fn block_size(&self) -> usize {
        self.block_size
    }

    /// Token slots currently live.
    #[must_use]
    pub const fn num_filled(&self) -> usize {
        self.num_filled
    }

    /// Whether every slot is live. **Only a full block is ever shared for a
    /// prompt prefix**, and only a *non-full* block ever needs a copy-on-write:
    /// nothing more will ever be written into a full one.
    #[must_use]
    pub const fn is_full(&self) -> bool {
        self.num_filled == self.block_size
    }

    /// Whether no slot is live.
    #[must_use]
    pub const fn is_empty(&self) -> bool {
        self.num_filled == 0
    }

    /// How many block tables currently point at this block.
    ///
    /// `0` means it is in the free list. `> 1` means it is shared, and a write
    /// to it must copy first.
    #[must_use]
    pub const fn ref_count(&self) -> usize {
        self.ref_count
    }

    /// Whether this block is shared, and therefore immutable in place.
    #[must_use]
    pub const fn is_shared(&self) -> bool {
        self.ref_count > 1
    }

    /// The absolute stream positions of the live slots, in slot order.
    #[must_use]
    pub fn positions(&self) -> &[usize] {
        &self.positions
    }

    /// The token ids of the live slots, in slot order.
    #[must_use]
    pub fn tokens(&self) -> &[u32] {
        &self.tokens
    }

    /// The prefix-cache key this block is published under, if any.
    #[must_use]
    pub const fn content_hash(&self) -> Option<u64> {
        self.content_hash
    }

    /// The `head_dim`-long key vector of one `(layer, slot, head)`, or `None` if
    /// any index is out of range — including a slot at or beyond `num_filled`,
    /// which holds no token.
    #[must_use]
    pub fn key(&self, layer: usize, slot: usize, head: usize, head_dim: usize) -> Option<&[f32]> {
        self.head_slice(&self.keys, layer, slot, head, head_dim)
    }

    /// The `head_dim`-long value vector of one `(layer, slot, head)`. See
    /// [`Self::key`].
    #[must_use]
    pub fn value(&self, layer: usize, slot: usize, head: usize, head_dim: usize) -> Option<&[f32]> {
        self.head_slice(&self.values, layer, slot, head, head_dim)
    }

    fn head_slice<'a>(
        &self,
        buffer: &'a [f32],
        layer: usize,
        slot: usize,
        head: usize,
        head_dim: usize,
    ) -> Option<&'a [f32]> {
        if slot >= self.num_filled {
            return None;
        }
        let start =
            slot * self.token_buffer_len + layer * self.per_layer_token_stride + head * head_dim;
        buffer.get(start..start + head_dim)
    }

    /// The whole `[layer][head][dim]` key buffer of one live slot.
    pub(super) fn slot_keys(&self, slot: usize) -> &[f32] {
        let start = slot * self.token_buffer_len;
        &self.keys[start..start + self.token_buffer_len]
    }

    /// The whole `[layer][head][dim]` value buffer of one live slot.
    pub(super) fn slot_values(&self, slot: usize) -> &[f32] {
        let start = slot * self.token_buffer_len;
        &self.values[start..start + self.token_buffer_len]
    }

    /// Write one token into the next free slot.
    ///
    /// The caller (the block table) has already established that this block is
    /// **not shared** — a shared block is copied first — and that it is not
    /// full. Both are re-checked here with a debug assertion rather than a
    /// silent overwrite, because a write through a shared block is precisely the
    /// bug copy-on-write exists to prevent, and it must not be able to hide.
    pub(super) fn push_slot(&mut self, token: u32, position: usize, keys: &[f32], values: &[f32]) {
        debug_assert!(!self.is_full(), "push_slot into a full block");
        debug_assert!(
            self.ref_count == 1,
            "push_slot through a block with ref_count {} — copy-on-write was skipped",
            self.ref_count
        );
        debug_assert_eq!(keys.len(), self.token_buffer_len);
        debug_assert_eq!(values.len(), self.token_buffer_len);

        let start = self.num_filled * self.token_buffer_len;
        self.keys[start..start + self.token_buffer_len].copy_from_slice(keys);
        self.values[start..start + self.token_buffer_len].copy_from_slice(values);
        self.positions.push(position);
        self.tokens.push(token);
        self.num_filled += 1;
    }

    /// Take a reference. Used by `fork`, by prefix sharing, and by nothing else.
    pub(super) fn add_ref(&mut self) {
        self.ref_count += 1;
    }

    /// Drop a reference, returning `true` when the block became free.
    pub(super) fn release(&mut self) -> bool {
        debug_assert!(self.ref_count > 0, "release of an unreferenced block");
        self.ref_count -= 1;
        self.ref_count == 0
    }

    /// Reset for reuse: scrub the bytes and clear every trace of the previous
    /// tenant.
    ///
    /// The scrub is not hygiene theatre. The pool is shared by unrelated
    /// sequences, and handing block bytes from one sequence to another would be
    /// a *cross-request KV leak* — the paged-attention equivalent of handing out
    /// an uninitialised page. It costs one `memset` per allocation and it makes
    /// "a sequence can only ever read bytes it wrote (or explicitly adopted)"
    /// true by construction rather than by argument.
    pub(super) fn reset_for_allocation(&mut self) {
        self.keys.fill(0.0);
        self.values.fill(0.0);
        self.positions.clear();
        self.tokens.clear();
        self.num_filled = 0;
        self.ref_count = 1;
        self.content_hash = None;
    }

    /// Overwrite this (freshly allocated, unshared) block with another's
    /// contents. This *is* the copy in copy-on-write.
    ///
    /// The copy is **not** published to the prefix cache: it exists precisely
    /// because it is about to be mutated, and a block that is about to change is
    /// not a cache entry.
    pub(super) fn copy_contents_from(
        &mut self,
        keys: &[f32],
        values: &[f32],
        positions: &[usize],
        tokens: &[u32],
        num_filled: usize,
    ) {
        self.keys.copy_from_slice(keys);
        self.values.copy_from_slice(values);
        self.positions.clear();
        self.positions.extend_from_slice(positions);
        self.tokens.clear();
        self.tokens.extend_from_slice(tokens);
        self.num_filled = num_filled;
        self.content_hash = None;
    }

    /// The raw `[slot][layer][head][dim]` key buffer — the whole block,
    /// including slots beyond `num_filled`. Used by the copy-on-write copy and
    /// by the swap slab, which both move the block wholesale.
    pub(super) fn raw_keys(&self) -> &[f32] {
        &self.keys
    }

    /// The raw `[slot][layer][head][dim]` value buffer. See `Self::raw_keys`.
    pub(super) fn raw_values(&self) -> &[f32] {
        &self.values
    }

    /// Publish this block under a prefix-cache key. Only ever called for a full
    /// block.
    pub(super) fn set_content_hash(&mut self, hash: Option<u64>) {
        self.content_hash = hash;
    }
}
