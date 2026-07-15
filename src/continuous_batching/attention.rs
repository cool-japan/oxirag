//! Scaled dot-product attention read **through a block table**, over blocks that
//! are scattered anywhere in the pool and shared with other sequences.
//!
//! ```text
//! A = softmax( Q Kᵀ / sqrt(d_head)  +  M )     M[q, t] = 0 if pos(t) <= pos(q), else -inf
//! O = A V
//! ```
//!
//! This is the same mathematics as
//! [`kv_cache_compression::scaled_dot_product_attention`], because it is the same
//! attention: paging changes *where the bytes live*, and must change **nothing
//! else**. That is not an aspiration here, it is the module's headline test:
//!
//! > For any sequence, [`paged_attention`] over the block table and
//! > [`scaled_dot_product_attention`] over the equivalent contiguous
//! > [`KvCacheTensor`] produce **bit-identical** outputs and **bit-identical**
//! > softmax weights.
//!
//! # Why bit-identical, and not "close enough"
//!
//! Floating-point addition is not associative, so two kernels that compute "the
//! same" sum in different orders will differ in the last bits — and a test with a
//! tolerance is a test that a stride bug can hide inside. So this kernel does not
//! merely compute the same *value* as the reference; it performs the **same
//! operations in the same order**:
//!
//! - dot products accumulated in `f64`, ascending over `d`;
//! - the row maximum taken with a strict `>`, ascending over `t`;
//! - the max-subtracted softmax, denominator accumulated ascending over `t`;
//! - the value sum accumulated in `f64`, ascending over `t`, skipping exactly the
//!   zero weights, narrowed to `f32` once at the end.
//!
//! Every arithmetic operation is IEEE-754 deterministic and `exp` is called on
//! identical bit patterns, so the two kernels' results are equal *bit for bit* —
//! and a tolerance-free `assert_eq!` on the raw buffers is therefore a legitimate
//! assertion, not a lucky one. A single wrong stride, a single off-by-one in the
//! block offset, a mask built from slot indices instead of absolute positions:
//! none of them can survive it.
//!
//! # The one thing paging really does change
//!
//! Nothing about the arithmetic — but everything about the *gather*. The
//! reference kernel walks one contiguous `[token][head][dim]` run per layer. This
//! one resolves every single token through `blocks[t / block_size]` and reads it
//! at `t % block_size`, from a block that may sit anywhere in the pool and may be
//! shared with three other sequences. The tests deliberately fragment the pool
//! first, so the blocks of the sequence under test are *not* in ascending
//! physical order and are *not* adjacent — because a block table that happens to
//! be the identity map tests nothing at all.
//!
//! [`kv_cache_compression::scaled_dot_product_attention`]:
//!     crate::kv_cache_compression::scaled_dot_product_attention
//! [`scaled_dot_product_attention`]:
//!     crate::kv_cache_compression::scaled_dot_product_attention
//! [`KvCacheTensor`]: crate::kv_cache_compression::KvCacheTensor

#![allow(
    // `head_dim` and token counts are `usize` and are cast to `f64` for the
    // `1/sqrt(d)` scale. The values involved are far below 2^53.
    clippy::cast_precision_loss
)]

use super::allocator::KvBlockAllocator;
use super::table::KvBlockTable;
use super::types::{ContinuousBatchError, ContinuousBatchResult, check_finite};

// ── KvBlockAttentionOutput ───────────────────────────────────────────────────

/// The result of one paged attention call: the output vectors, the softmax
/// weights that produced them, and the causal mask.
///
/// The weights are retained, and in `f64`, for the same reason the reference
/// kernel retains them: they *are* the eviction signal that
/// `kv_cache_compression`'s policies rank by, and they are what the equivalence
/// test compares. An output-only kernel could be wrong in the weights and right
/// in the output for a degenerate fixture; this one cannot.
#[derive(Debug, Clone, PartialEq)]
pub struct KvBlockAttentionOutput {
    num_heads: usize,
    num_queries: usize,
    num_keys: usize,
    head_dim: usize,
    /// `[head][query][dim]`, `f32` like the blocks it was read from.
    output: Vec<f32>,
    /// `[head][query][key]` softmax weights, in `f64`.
    weights: Vec<f64>,
    /// `[query][key]` causal visibility. Head-independent.
    mask: Vec<bool>,
    /// Absolute stream position of each query, in query order.
    query_positions: Vec<usize>,
    /// Absolute stream position of each attended key slot, in slot order. Gappy
    /// after a compaction — which is exactly when a mask bug would appear.
    key_positions: Vec<usize>,
}

impl KvBlockAttentionOutput {
    /// Number of attention heads.
    #[must_use]
    pub const fn num_heads(&self) -> usize {
        self.num_heads
    }

    /// Number of query positions.
    #[must_use]
    pub const fn num_queries(&self) -> usize {
        self.num_queries
    }

    /// Number of cached key/value slots attended over.
    #[must_use]
    pub const fn num_keys(&self) -> usize {
        self.num_keys
    }

    /// Head dimension.
    #[must_use]
    pub const fn head_dim(&self) -> usize {
        self.head_dim
    }

    /// Absolute stream positions of the queries, in query order.
    #[must_use]
    pub fn query_positions(&self) -> &[usize] {
        &self.query_positions
    }

    /// Absolute stream positions of the attended key slots, in slot order.
    #[must_use]
    pub fn key_positions(&self) -> &[usize] {
        &self.key_positions
    }

    /// The `head_dim`-long output vector for one `(head, query)` pair.
    #[must_use]
    pub fn output_row(&self, head: usize, query: usize) -> Option<&[f32]> {
        if head >= self.num_heads || query >= self.num_queries {
            return None;
        }
        let start = (head * self.num_queries + query) * self.head_dim;
        self.output.get(start..start + self.head_dim)
    }

    /// The `num_keys`-long softmax weight row for one `(head, query)` pair.
    ///
    /// Sums to `1` whenever the query can see at least one key, and is all-zero
    /// when the causal mask hides every key from it.
    #[must_use]
    pub fn weight_row(&self, head: usize, query: usize) -> Option<&[f64]> {
        if head >= self.num_heads || query >= self.num_queries {
            return None;
        }
        let start = (head * self.num_queries + query) * self.num_keys;
        self.weights.get(start..start + self.num_keys)
    }

    /// Whether the causal mask lets `query` see the key in slot `key`.
    #[must_use]
    pub fn is_visible(&self, query: usize, key: usize) -> bool {
        if query >= self.num_queries || key >= self.num_keys {
            return false;
        }
        self.mask[query * self.num_keys + key]
    }

    /// The whole output buffer, `[head][query][dim]`. This is the buffer the
    /// equivalence test compares, bit for bit, against the reference kernel's.
    #[must_use]
    pub fn output(&self) -> &[f32] {
        &self.output
    }

    /// The whole weight buffer, `[head][query][key]`.
    #[must_use]
    pub fn weights(&self) -> &[f64] {
        &self.weights
    }
}

// ── The paged kernel ─────────────────────────────────────────────────────────

/// A read-only view of one layer of one sequence's KV, through its block table.
///
/// Every key and value the kernel touches goes through here, and therefore
/// through `blocks[t / block_size][t % block_size]`. There is no fast path that
/// bypasses the indirection, because a fast path that bypassed it would be the
/// thing the tests are trying to check.
struct BlockGather<'pool> {
    table: &'pool KvBlockTable,
    allocator: &'pool KvBlockAllocator,
    layer: usize,
}

impl<'pool> BlockGather<'pool> {
    fn key(&self, token: usize, head: usize) -> ContinuousBatchResult<&'pool [f32]> {
        self.table
            .key(self.allocator, self.layer, token, head)
            .ok_or_else(|| ContinuousBatchError::InvariantViolated {
                reason: format!(
                    "no key at (layer {}, token {token}, head {head}) in a table of {} token(s)",
                    self.layer,
                    self.table.len()
                ),
            })
    }

    fn value(&self, token: usize, head: usize) -> ContinuousBatchResult<&'pool [f32]> {
        self.table
            .value(self.allocator, self.layer, token, head)
            .ok_or_else(|| ContinuousBatchError::InvariantViolated {
                reason: format!(
                    "no value at (layer {}, token {token}, head {head}) in a table of {} token(s)",
                    self.layer,
                    self.table.len()
                ),
            })
    }
}

/// Compute `softmax(Q Kᵀ / sqrt(d) + causal_mask) · V` for one layer of a
/// sequence, reading its keys and values **through its block table**.
///
/// `queries` is laid out `[head][query][dim]` and must be exactly
/// `num_heads * query_positions.len() * head_dim` elements long — the same
/// buffer [`scaled_dot_product_attention`] takes. `query_positions` gives each
/// query's **absolute stream position**, and a query at absolute position `p`
/// attends to slot `t` iff `positions(t) <= p`. Slot indices never enter the
/// mask, which is what keeps it correct after a compaction has punched holes in
/// the sequence.
///
/// The result is **bit-identical** to the reference kernel's over the equivalent
/// contiguous [`KvCacheTensor`]. See the module docs for why that is a theorem
/// and not a coincidence.
///
/// # Errors
///
/// - [`ContinuousBatchError::EmptyTable`] if the sequence holds no tokens (a
///   softmax over zero keys does not exist).
/// - [`ContinuousBatchError::LayerOutOfRange`] if `layer` does not exist.
/// - [`ContinuousBatchError::InvalidSelection`] if `query_positions` is empty or
///   not strictly increasing.
/// - [`ContinuousBatchError::ShapeMismatch`] if `queries` has the wrong length.
/// - [`ContinuousBatchError::NonFiniteInput`] if `queries` holds a `NaN` or an
///   infinity.
/// - [`ContinuousBatchError::UnknownBlock`] if the table was built by a different
///   allocator.
///
/// [`scaled_dot_product_attention`]:
///     crate::kv_cache_compression::scaled_dot_product_attention
/// [`KvCacheTensor`]: crate::kv_cache_compression::KvCacheTensor
pub fn paged_attention(
    table: &KvBlockTable,
    allocator: &KvBlockAllocator,
    layer: usize,
    queries: &[f32],
    query_positions: &[usize],
) -> ContinuousBatchResult<KvBlockAttentionOutput> {
    let config = allocator.config();
    let num_heads = config.num_heads;
    let head_dim = config.head_dim;
    let num_keys = table.len();
    let num_queries = query_positions.len();

    if num_keys == 0 {
        return Err(ContinuousBatchError::EmptyTable {
            operation: "paged_attention",
        });
    }
    if layer >= config.num_layers {
        return Err(ContinuousBatchError::LayerOutOfRange {
            layer,
            num_layers: config.num_layers,
        });
    }
    if query_positions.is_empty() {
        return Err(ContinuousBatchError::InvalidSelection {
            reason: "at least one query position is required".to_string(),
        });
    }
    for window in query_positions.windows(2) {
        if window[1] <= window[0] {
            return Err(ContinuousBatchError::InvalidSelection {
                reason: format!(
                    "query positions must be strictly increasing, got {} after {}",
                    window[1], window[0]
                ),
            });
        }
    }
    let expected = num_heads * num_queries * head_dim;
    if queries.len() != expected {
        return Err(ContinuousBatchError::ShapeMismatch {
            what: "query buffer",
            expected,
            actual: queries.len(),
        });
    }
    check_finite(queries, "query buffer")?;

    let key_positions = table.positions(allocator)?;
    let mask = build_causal_mask(&key_positions, query_positions);

    // The scaling that keeps the logits' variance ~1 as `head_dim` grows. Taken
    // in `f64`, exactly as the reference kernel does — a `f32` reciprocal square
    // root here would diverge in the last bits and the equivalence would become
    // approximate.
    let scale = 1.0 / (head_dim as f64).sqrt();

    let gather = BlockGather {
        table,
        allocator,
        layer,
    };

    let mut output = vec![0.0f32; num_heads * num_queries * head_dim];
    let mut weights = vec![0.0f64; num_heads * num_queries * num_keys];
    let mut logits = vec![0.0f64; num_keys];

    for head in 0..num_heads {
        for query in 0..num_queries {
            let query_start = (head * num_queries + query) * head_dim;
            let query_vector = &queries[query_start..query_start + head_dim];
            let mask_row = &mask[query * num_keys..(query + 1) * num_keys];

            let row_max = fill_logits(&gather, head, query_vector, mask_row, scale, &mut logits)?;

            let weight_start = (head * num_queries + query) * num_keys;
            let weight_row = &mut weights[weight_start..weight_start + num_keys];

            // Degenerate row: the causal mask hides every key, so every logit is
            // `-inf` and so is their maximum. A softmax over the empty set does
            // not exist; the row and its output stay all-zero rather than
            // inventing a distribution out of nothing. (The reference kernel
            // defines the same thing, and the equivalence test drives a query at
            // a position before every key to check that both agree.)
            if !row_max.is_finite() {
                continue;
            }

            stable_softmax_into(&logits, row_max, weight_row);

            let output_start = (head * num_queries + query) * head_dim;
            let output_row = &mut output[output_start..output_start + head_dim];
            weighted_value_sum(&gather, head, head_dim, weight_row, output_row)?;
        }
    }

    Ok(KvBlockAttentionOutput {
        num_heads,
        num_queries,
        num_keys,
        head_dim,
        output,
        weights,
        mask,
        query_positions: query_positions.to_vec(),
        key_positions,
    })
}

/// `mask[q * num_keys + t] = (key_positions[t] <= query_positions[q])`.
///
/// Driven by **absolute stream positions**, never by slot indices, so it stays
/// the causal mask of the uncompressed sequence even after a compaction has
/// removed slots from the middle.
fn build_causal_mask(key_positions: &[usize], query_positions: &[usize]) -> Vec<bool> {
    let num_keys = key_positions.len();
    let mut mask = vec![false; query_positions.len() * num_keys];
    for (query, &query_position) in query_positions.iter().enumerate() {
        let row = &mut mask[query * num_keys..(query + 1) * num_keys];
        for (cell, &key_position) in row.iter_mut().zip(key_positions.iter()) {
            *cell = key_position <= query_position;
        }
    }
    mask
}

/// Fill `logits` with `q · k_t / sqrt(d)` for every visible `t` (and `-inf` for
/// every masked one), returning the row maximum.
///
/// Each `k_t` is fetched through the block table. Dot products accumulate in
/// `f64`, ascending over `d`, so that the sum is bit-identical to the reference
/// kernel's over the same values. The maximum is `-inf` exactly when the row has
/// no visible key.
fn fill_logits(
    gather: &BlockGather<'_>,
    head: usize,
    query_vector: &[f32],
    mask_row: &[bool],
    scale: f64,
    logits: &mut [f64],
) -> ContinuousBatchResult<f64> {
    let mut row_max = f64::NEG_INFINITY;
    for ((token, logit), &visible) in logits.iter_mut().enumerate().zip(mask_row.iter()) {
        if !visible {
            *logit = f64::NEG_INFINITY;
            continue;
        }
        let key_vector = gather.key(token, head)?;
        let mut dot = 0.0f64;
        for (&query_element, &key_element) in query_vector.iter().zip(key_vector.iter()) {
            dot += f64::from(query_element) * f64::from(key_element);
        }
        let scaled = dot * scale;
        *logit = scaled;
        if scaled > row_max {
            row_max = scaled;
        }
    }
    Ok(row_max)
}

/// The max-subtracted softmax: `weight_row[t] = exp(l_t - max) / Σ exp(l - max)`.
///
/// Masked logits are `-inf`, and `exp(-inf - max) = 0` exactly for any finite
/// `max`, so masked entries need no branch and contribute an exact `0.0` to the
/// denominator. The element attaining the maximum contributes `exp(0) = 1`, so
/// the denominator is always `>= 1` and the division cannot blow up.
fn stable_softmax_into(logits: &[f64], row_max: f64, weight_row: &mut [f64]) {
    let mut denominator = 0.0f64;
    for (cell, &logit) in weight_row.iter_mut().zip(logits.iter()) {
        let unnormalized = (logit - row_max).exp();
        *cell = unnormalized;
        denominator += unnormalized;
    }
    for cell in weight_row.iter_mut() {
        *cell /= denominator;
    }
}

/// `output_row = Σ_t weight_row[t] · v_t`, with each `v_t` fetched through the
/// block table.
///
/// Accumulated in `f64` ascending over `t`, skipping exactly the zero weights,
/// and narrowed to `f32` once at the end. Because the weights are non-negative
/// and sum to one, every element is a convex combination of value elements and
/// the narrowing is a rounding, never an overflow: paging cannot manufacture
/// magnitude any more than attention can.
fn weighted_value_sum(
    gather: &BlockGather<'_>,
    head: usize,
    head_dim: usize,
    weight_row: &[f64],
    output_row: &mut [f32],
) -> ContinuousBatchResult<()> {
    let mut accumulator = vec![0.0f64; head_dim];
    for (token, &weight) in weight_row.iter().enumerate() {
        if weight <= 0.0 {
            continue;
        }
        let value_vector = gather.value(token, head)?;
        for (slot, &value) in accumulator.iter_mut().zip(value_vector.iter()) {
            *slot += weight * f64::from(value);
        }
    }
    for (slot, &accumulated) in output_row.iter_mut().zip(accumulator.iter()) {
        *slot = narrow_to_f32(accumulated);
    }
    Ok(())
}

/// Narrow an `f64` attention accumulator back to the blocks' `f32` dtype.
///
/// Bounded by construction: the accumulator is a convex combination of `f32`
/// value elements, so it lies within the `f32` range and this is a rounding.
#[allow(clippy::cast_possible_truncation)]
const fn narrow_to_f32(value: f64) -> f32 {
    value as f32
}
