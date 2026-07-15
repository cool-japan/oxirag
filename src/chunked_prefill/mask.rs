//! The piecewise causal mask of one prefill chunk.
//!
//! # The mask a chunk actually needs
//!
//! When a chunk of `n` prompt tokens is prefilled against a cache that already
//! holds `c` committed keys, the chunk's own keys are appended first and the
//! attention runs over all `c + n` of them. The resulting mask is **not** a
//! triangle, and it is **not** a rectangle. It is two segments glued together:
//!
//! ```text
//!            committed keys (c)          this chunk's keys (n)
//!         ┌─────────────────────────┬───────────────────────────┐
//!  q = 0  │ 1  1  1  1  1  1  1  1  │ 1  0  0  0  0             │
//!  q = 1  │ 1  1  1  1  1  1  1  1  │ 1  1  0  0  0             │
//!  q = 2  │ 1  1  1  1  1  1  1  1  │ 1  1  1  0  0             │
//!  q = 3  │ 1  1  1  1  1  1  1  1  │ 1  1  1  1  0             │
//!  q = 4  │ 1  1  1  1  1  1  1  1  │ 1  1  1  1  1             │
//!         └─────────────────────────┴───────────────────────────┘
//!            fully visible: no mask     inclusive lower triangle
//! ```
//!
//! * **The prefix segment is unmasked.** Every committed key was appended before
//!   this chunk existed, so its absolute stream position is strictly below every
//!   query's. There is nothing to mask.
//! * **The diagonal segment is an *inclusive* lower triangle.** Query `j` may see
//!   chunk key `i` iff `i <= j`. Inclusive: a token attends to *itself*.
//!
//! [`PrefillMaskGeometry`] derives that mask from **`(c, n)` alone** — two
//! integers, no positions, no comparisons against the cache. That is exactly
//! what a real fused kernel is handed (`context_len` and `query_len`), and it is
//! exactly where real implementations ship bugs.
//!
//! # Why deriving it from geometry alone is *sound*
//!
//! The prefix segment's "no mask" rule rests on one lemma:
//!
//! > **Every slot already in the cache carries an absolute position strictly
//! > below every position this chunk is about to be assigned.**
//!
//! It holds because positions come from a counter that
//! [`KvCacheTensor`](crate::kv_cache_compression::KvCacheTensor) only ever
//! advances, and that eviction never rewinds. So it survives things that look
//! like they should break it:
//!
//! * a **shared prefix** already in the cache (the chunk's positions simply start
//!   higher), and
//! * a **compressed** cache, whose surviving positions are gappy — `[0, 1, 57,
//!   91]` — but still monotone, and still all below the chunk's.
//!
//! The geometry never looks at the positions, so gaps cannot confuse it. A mask
//! built the "obvious" way — by comparing slot *indices* — silently breaks under
//! both.
//!
//! # Why it is checked anyway
//!
//! [`PrefillMaskGeometry::audit`] compares this mask, cell by cell, against the
//! mask the attention kernel built from **absolute stream positions**. The two
//! are computed from *different information* — segment lengths versus positions
//! — so their agreement is a real check, not a tautology, and it is **two-sided**:
//!
//! | Bug | Which side goes wrong | Caught because |
//! |---|---|---|
//! | wrong `committed_keys` (e.g. read *after* appending the chunk) | the geometry | the prefix segment swallows the chunk's own keys — a query sees its own future |
//! | strict triangle (`i < j`) | the geometry | the diagonal loses its self-attention |
//! | chunk-local query positions (`0..n` instead of `s..s+n`) | the kernel's mask | the prefix segment stops being fully visible |
//! | query positions shifted by one | the kernel's mask | the triangle stops being inclusive at exactly one cell per row |
//!
//! Both masks would have to be wrong *in the same way* to slip through, and they
//! cannot be, because they do not share a line of code.

use crate::kv_cache_compression::KvAttentionOutput;

use super::types::{ChunkedPrefillError, ChunkedPrefillResult, widen};

/// The piecewise causal mask of one prefill chunk, derived from segment geometry
/// alone.
///
/// See the [module docs](self) for the picture and the soundness argument.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct PrefillMaskGeometry {
    committed_keys: usize,
    chunk_tokens: usize,
}

impl PrefillMaskGeometry {
    /// The mask of a chunk of `chunk_tokens` queries running against a cache
    /// that held `committed_keys` slots **before the chunk's own keys were
    /// appended**.
    ///
    /// That "before" is the whole game. Read the cache length *after* appending
    /// and the prefix segment swallows the chunk's own keys, every query sees
    /// every key, and the mask stops being causal — the single most common
    /// chunked-prefill bug there is, and one that produces a perfectly
    /// finite-looking tensor.
    #[must_use]
    pub const fn new(committed_keys: usize, chunk_tokens: usize) -> Self {
        Self {
            committed_keys,
            chunk_tokens,
        }
    }

    /// Slots the cache held before this chunk's keys were appended.
    #[must_use]
    pub const fn committed_keys(&self) -> usize {
        self.committed_keys
    }

    /// Prompt tokens in this chunk.
    #[must_use]
    pub const fn chunk_tokens(&self) -> usize {
        self.chunk_tokens
    }

    /// Query rows in the mask: one per token in the chunk.
    #[must_use]
    pub const fn num_queries(&self) -> usize {
        self.chunk_tokens
    }

    /// Key columns in the mask: the committed prefix **plus** the chunk's own
    /// keys, which are appended before the attention runs.
    #[must_use]
    pub const fn num_keys(&self) -> usize {
        self.committed_keys + self.chunk_tokens
    }

    /// Whether query `query` of this chunk may attend to key column `key`.
    ///
    /// Out-of-range indices are not visible, which keeps the predicate total.
    #[must_use]
    pub const fn is_visible(&self, query: usize, key: usize) -> bool {
        if query >= self.chunk_tokens || key >= self.num_keys() {
            return false;
        }
        if key < self.committed_keys {
            // The prefix segment. Committed before this chunk existed, so its
            // position is strictly below every query's: nothing to mask.
            return true;
        }
        // The diagonal segment. Inclusive, so a token attends to itself.
        key - self.committed_keys <= query
    }

    /// The number of visible cells — which is exactly the number of `Q·K` dot
    /// products one head of this chunk performs.
    ///
    /// Closed form: `n·c + n(n+1)/2`, the `n · c` unmasked prefix cells plus the
    /// triangular number of the diagonal segment.
    #[must_use]
    pub const fn cells(&self) -> u64 {
        let tokens = widen(self.chunk_tokens);
        tokens * widen(self.committed_keys) + tokens * (tokens + 1) / 2
    }

    /// The number of KV slots this chunk's attention must have resident:
    /// `committed_keys + chunk_tokens`.
    ///
    /// Unlike [`Self::cells`], summing this over a plan's chunks does **not**
    /// give the one-shot figure — it is the term that makes chunking cost
    /// something. See [`PrefillPlan::kv_slot_visits`](super::PrefillPlan::kv_slot_visits).
    #[must_use]
    pub const fn kv_slot_visits(&self) -> u64 {
        widen(self.num_keys())
    }

    /// The mask as a dense `[query][key]` bitmap, row-major.
    #[must_use]
    pub fn to_bitmap(&self) -> Vec<bool> {
        let mut bitmap = vec![false; self.num_queries() * self.num_keys()];
        for query in 0..self.num_queries() {
            let row = &mut bitmap[query * self.num_keys()..(query + 1) * self.num_keys()];
            for (key, cell) in row.iter_mut().enumerate() {
                *cell = self.is_visible(query, key);
            }
        }
        bitmap
    }

    /// Cross-check this mask, cell by cell, against the causal mask the
    /// attention kernel derived from absolute stream positions.
    ///
    /// This is the module's central invariant. It is checked on every chunk (see
    /// [`ChunkedPrefillConfig::audit_mask`](super::ChunkedPrefillConfig::audit_mask))
    /// and it is two-sided: it catches a bug in the segment bookkeeping *and* a
    /// bug in the positions handed to the kernel. See the [module docs](self).
    ///
    /// # Errors
    ///
    /// - [`ChunkedPrefillError::MaskShapeDivergence`] if the kernel attended over
    ///   a different number of queries or keys than the geometry expected —
    ///   which is what "I forgot to append the chunk's own keys" looks like.
    /// - [`ChunkedPrefillError::MaskDivergence`] naming the first cell the two
    ///   masks disagree on.
    pub fn audit(&self, attention: &KvAttentionOutput) -> ChunkedPrefillResult<()> {
        if attention.num_queries() != self.num_queries() || attention.num_keys() != self.num_keys()
        {
            return Err(ChunkedPrefillError::MaskShapeDivergence {
                expected_queries: self.num_queries(),
                expected_keys: self.num_keys(),
                actual_queries: attention.num_queries(),
                actual_keys: attention.num_keys(),
            });
        }
        for query in 0..self.num_queries() {
            for key in 0..self.num_keys() {
                let geometry = self.is_visible(query, key);
                let kernel = attention.is_visible(query, key);
                if geometry != kernel {
                    return Err(ChunkedPrefillError::MaskDivergence {
                        query,
                        key,
                        geometry,
                        kernel,
                    });
                }
            }
        }
        Ok(())
    }
}
