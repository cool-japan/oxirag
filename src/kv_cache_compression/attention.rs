//! Scaled dot-product attention over a [`KvCacheTensor`], and the accumulation
//! of the per-token attention statistics the eviction policies rank by.
//!
//! Nothing here is simulated. [`scaled_dot_product_attention`] computes
//!
//! ```text
//! A = softmax( Q Kᵀ / sqrt(d_head)  +  M )        M[q, t] = 0 if pos(t) <= pos(q), else -inf
//! O = A V
//! ```
//!
//! for real `Q`, and the real `K`/`V` held in the cache, and the eviction
//! policies then rank tokens by the columns of that real `A`. This ordering
//! matters: an eviction policy driven by a *fabricated* score would be
//! untestable — it could not be wrong. Because the score here is a genuine
//! softmax column sum, the claim "H2O keeps the tokens that carry the attention
//! mass, and a recency baseline does not" is a claim about a measurable
//! quantity, and the module's tests measure it.
//!
//! # Numerical stability
//!
//! The softmax is the max-subtracted form, `exp(l_t - max_t l_t) / Σ exp(...)`.
//! Two properties follow, and together they are a *proof* that a finite input
//! cannot produce a non-finite output:
//!
//! 1. **No overflow in the logits.** Dot products are accumulated in `f64`
//!    even though the cache is `f32`. The largest finite `f32` is ≈ `3.4e38`,
//!    so the most extreme possible logit is bounded by
//!    `d_head · (3.4e38)² ≈ d_head · 1.2e77`, which is comfortably finite in
//!    `f64` (max ≈ `1.8e308`) for any head dimension below ~`1e230`. Finite
//!    inputs are enforced at the boundary — [`KvCacheTensor::append_token`] and
//!    this function both reject `NaN`/infinity — so the logits are always
//!    finite.
//! 2. **No overflow or underflow-to-zero in the exponentials.** After
//!    subtracting the row max, every exponent is `<= 0`, so every
//!    `exp(l - max) ∈ (0, 1]`, and the element attaining the max contributes
//!    exactly `exp(0) = 1`. The denominator is therefore **always `>= 1`** and
//!    the division can never blow up. (This is the whole point of the
//!    subtraction: the naive form overflows to `inf/inf = NaN` for logits above
//!    ~`709`, which extreme keys reach easily.)
//!
//! Masked entries need no branch: their logit is `-inf`, and
//! `exp(-inf - max) = 0` exactly, for any finite `max`. The single degenerate
//! case is a query row with *no* visible key at all (every logit `-inf`, so
//! `max = -inf` and `-inf - -inf = NaN`). That case is detected explicitly and
//! defined to produce an all-zero weight row and an all-zero output row — the
//! only defensible answer, since a softmax over the empty set does not exist.
//!
//! The consequence is worth stating plainly: every output element is a **convex
//! combination of value vectors**, so `|O| <= max |V|`. Attention here cannot
//! manufacture magnitude.

#![allow(
    // Token/step counts are `usize`/`u64` and are divided into `f64`
    // attention masses to form means. The counts involved are far below
    // 2^53, so the pedantic precision-loss warning does not apply.
    clippy::cast_precision_loss
)]

use std::collections::BTreeMap;

use super::types::{
    KvCacheTensor, KvCompressionConfig, KvCompressionError, KvResult, KvScoreNormalization,
    check_finite,
};

// ── KvAttentionOutput ────────────────────────────────────────────────────────

/// The full result of one attention call: the output vectors **and** the
/// attention matrix that produced them.
///
/// The weight matrix is retained (rather than being an internal detail) because
/// it *is* the eviction signal — [`KvAttentionStats::accumulate`] consumes it,
/// and the tests assert directly on it (rows sum to one, the causal mask is
/// respected, hand-computed weights match).
#[derive(Debug, Clone, PartialEq)]
pub struct KvAttentionOutput {
    num_heads: usize,
    num_queries: usize,
    num_keys: usize,
    head_dim: usize,
    /// `[head][query][dim]`, `f32` like the cache it was read from.
    output: Vec<f32>,
    /// `[head][query][key]` softmax weights, kept in `f64` so that the score
    /// accumulation that consumes them does not compound `f32` rounding across
    /// thousands of query steps.
    weights: Vec<f64>,
    /// `[query][key]` causal visibility. Head-independent, so it is stored once.
    mask: Vec<bool>,
    /// Absolute stream position of each query, in query order.
    query_positions: Vec<usize>,
}

impl KvAttentionOutput {
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

    /// The `head_dim`-long output vector for one `(head, query)` pair, or
    /// `None` if either index is out of range.
    #[must_use]
    pub fn output_row(&self, head: usize, query: usize) -> Option<&[f32]> {
        if head >= self.num_heads || query >= self.num_queries {
            return None;
        }
        let start = (head * self.num_queries + query) * self.head_dim;
        self.output.get(start..start + self.head_dim)
    }

    /// The `num_keys`-long softmax weight row for one `(head, query)` pair, or
    /// `None` if either index is out of range.
    ///
    /// The row sums to `1` whenever the query can see at least one key, and is
    /// all-zero when the causal mask hides every key from it.
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

    /// The whole output buffer, `[head][query][dim]`.
    #[must_use]
    pub fn output(&self) -> &[f32] {
        &self.output
    }

    /// The whole weight buffer, `[head][query][key]`.
    #[must_use]
    pub fn weights(&self) -> &[f64] {
        &self.weights
    }

    /// The number of `(head, query)` softmax rows in which slot `key` competed
    /// (i.e. was causally visible).
    #[must_use]
    pub fn visible_row_count(&self, key: usize) -> usize {
        if key >= self.num_keys {
            return 0;
        }
        let visible_queries = (0..self.num_queries)
            .filter(|&query| self.mask[query * self.num_keys + key])
            .count();
        visible_queries * self.num_heads
    }
}

// ── The attention kernel ─────────────────────────────────────────────────────

/// Compute `softmax(Q Kᵀ / sqrt(d) + causal_mask) · V` for one layer of a
/// [`KvCacheTensor`].
///
/// `queries` is laid out `[head][query][dim]` and must be exactly
/// `num_heads * query_positions.len() * head_dim` elements long.
/// `query_positions` gives each query's **absolute stream position**, and is
/// what makes the causal mask survive compression: a query at absolute position
/// `p` may attend to slot `t` iff `cache.positions()[t] <= p`. After eviction
/// has punched holes in the cache, the surviving slots still carry their
/// original positions, so the mask is still the mask of the *uncompressed*
/// sequence restricted to the surviving columns — never an accidental
/// re-indexing of it.
///
/// # Errors
///
/// - [`KvCompressionError::EmptyCache`] if the cache holds no tokens (a softmax
///   over zero keys does not exist).
/// - [`KvCompressionError::LayerOutOfRange`] if `layer` does not exist.
/// - [`KvCompressionError::InvalidTokenSelection`] if `query_positions` is
///   empty or not strictly increasing (the score accumulation folds query steps
///   in the order given, and a decayed accumulation is only meaningful in
///   chronological order).
/// - [`KvCompressionError::ShapeMismatch`] if `queries` has the wrong length.
/// - [`KvCompressionError::NonFiniteInput`] if `queries` contains a `NaN` or an
///   infinity.
pub fn scaled_dot_product_attention(
    cache: &KvCacheTensor,
    layer: usize,
    queries: &[f32],
    query_positions: &[usize],
) -> KvResult<KvAttentionOutput> {
    validate_attention_inputs(cache, layer, queries, query_positions)?;

    let num_heads = cache.num_heads();
    let head_dim = cache.head_dim();
    let num_keys = cache.seq_len();
    let num_queries = query_positions.len();

    let mask = build_causal_mask(cache.positions(), query_positions);

    // 1 / sqrt(d_head): the scaling that keeps the logits' variance ~1 when Q
    // and K have unit-variance entries, without which the softmax saturates
    // into a one-hot as d_head grows.
    let scale = 1.0 / (head_dim as f64).sqrt();

    let mut output = vec![0.0f32; num_heads * num_queries * head_dim];
    let mut weights = vec![0.0f64; num_heads * num_queries * num_keys];
    let mut logits = vec![0.0f64; num_keys];

    for head in 0..num_heads {
        for query in 0..num_queries {
            let query_start = (head * num_queries + query) * head_dim;
            let query_vector = &queries[query_start..query_start + head_dim];
            let mask_row = &mask[query * num_keys..(query + 1) * num_keys];

            let row_max = fill_logits(
                cache,
                layer,
                head,
                query_vector,
                mask_row,
                scale,
                &mut logits,
            );

            let weight_start = (head * num_queries + query) * num_keys;
            let weight_row = &mut weights[weight_start..weight_start + num_keys];

            // Degenerate row: the causal mask hides every key, so every logit is
            // `-inf` and `row_max` is `-inf` too. A softmax over the empty set is
            // undefined; the row (and its output) is defined to be all-zero,
            // rather than inventing a distribution out of nothing.
            if !row_max.is_finite() {
                continue;
            }

            stable_softmax_into(&logits, row_max, weight_row);

            let output_start = (head * num_queries + query) * head_dim;
            let output_row = &mut output[output_start..output_start + head_dim];
            weighted_value_sum(cache, layer, head, head_dim, weight_row, output_row);
        }
    }

    Ok(KvAttentionOutput {
        num_heads,
        num_queries,
        num_keys,
        head_dim,
        output,
        weights,
        mask,
        query_positions: query_positions.to_vec(),
    })
}

/// Reject every input the attention kernel cannot honour.
fn validate_attention_inputs(
    cache: &KvCacheTensor,
    layer: usize,
    queries: &[f32],
    query_positions: &[usize],
) -> KvResult<()> {
    if cache.is_empty() {
        return Err(KvCompressionError::EmptyCache {
            operation: "scaled_dot_product_attention",
        });
    }
    cache.check_layer(layer)?;
    if query_positions.is_empty() {
        return Err(KvCompressionError::InvalidTokenSelection {
            reason: "at least one query position is required".to_string(),
        });
    }
    for window in query_positions.windows(2) {
        if window[1] <= window[0] {
            return Err(KvCompressionError::InvalidTokenSelection {
                reason: format!(
                    "query positions must be strictly increasing, got {} after {}",
                    window[1], window[0]
                ),
            });
        }
    }
    let expected = cache.num_heads() * query_positions.len() * cache.head_dim();
    if queries.len() != expected {
        return Err(KvCompressionError::ShapeMismatch {
            what: "query buffer",
            expected,
            actual: queries.len(),
        });
    }
    check_finite(queries, "query buffer")
}

/// `mask[q * num_keys + t] = (key_positions[t] <= query_positions[q])`.
///
/// The mask is driven by **absolute stream positions**, not by slot indices, so
/// it remains the causal mask of the *uncompressed* sequence even after eviction
/// has removed slots from the middle of the cache. Head-independent, so it is
/// built once per attention call.
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
/// Dot products are accumulated in `f64` so that even the most extreme finite
/// `f32` inputs cannot overflow the logit (see the module docs). The returned
/// maximum is `-inf` exactly when the row has no visible key at all.
fn fill_logits(
    cache: &KvCacheTensor,
    layer: usize,
    head: usize,
    query_vector: &[f32],
    mask_row: &[bool],
    scale: f64,
    logits: &mut [f64],
) -> f64 {
    let head_dim = cache.head_dim();
    let mut row_max = f64::NEG_INFINITY;
    for ((token, logit), &visible) in logits.iter_mut().enumerate().zip(mask_row.iter()) {
        if !visible {
            *logit = f64::NEG_INFINITY;
            continue;
        }
        let key_block = cache.layer_key_block(layer, token);
        let key_vector = &key_block[head * head_dim..(head + 1) * head_dim];
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
    row_max
}

/// The max-subtracted softmax: `weight_row[t] = exp(l_t - max) / Σ exp(l - max)`.
///
/// Masked logits are `-inf`, and `exp(-inf - max) = 0` exactly for any finite
/// `max`, so masked entries need no branch. The element attaining the maximum
/// contributes `exp(0) = 1`, so the denominator is always `>= 1` and the division
/// can never blow up. `row_max` must be finite (the caller checks).
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

/// `output_row = Σ_t weight_row[t] · v_t` — a convex combination of the value
/// vectors, accumulated in `f64` and narrowed back to the cache's `f32` on the
/// way out.
///
/// Because the weights are non-negative and sum to one, every element of the
/// result is bounded by the largest value element, so the narrowing cannot
/// overflow: attention here cannot manufacture magnitude.
fn weighted_value_sum(
    cache: &KvCacheTensor,
    layer: usize,
    head: usize,
    head_dim: usize,
    weight_row: &[f64],
    output_row: &mut [f32],
) {
    let mut accumulator = vec![0.0f64; head_dim];
    for (token, &weight) in weight_row.iter().enumerate() {
        if weight <= 0.0 {
            continue;
        }
        let value_block = cache.layer_value_block(layer, token);
        let value_vector = &value_block[head * head_dim..(head + 1) * head_dim];
        for (slot, &value) in accumulator.iter_mut().zip(value_vector.iter()) {
            *slot += weight * f64::from(value);
        }
    }
    for (slot, &accumulated) in output_row.iter_mut().zip(accumulator.iter()) {
        *slot = narrow_to_f32(accumulated);
    }
}

/// Narrow an `f64` attention accumulator back to the cache's `f32` dtype.
///
/// The truncation is intentional and bounded: the accumulator is a convex
/// combination of `f32` value elements, so it lies within the `f32` range by
/// construction and this is a rounding, never an overflow.
#[allow(clippy::cast_possible_truncation)]
const fn narrow_to_f32(value: f64) -> f32 {
    value as f32
}

// ── KvAttentionStats ─────────────────────────────────────────────────────────

/// The per-token attention history that the score-driven eviction policies rank
/// by.
///
/// # What is accumulated, and why that is the right signal
///
/// For each cached slot `t` this tracks two running quantities:
///
/// - `accumulated[t] = Σ_q mean_h A[h, q, t]` — the **attention mass spent on
///   `t`**, the column sum of the attention matrix. Because every softmax row
///   sums to one, the column sums partition the model's *entire* attention
///   budget across the cached tokens. This is exactly H2O's heavy-hitter
///   signal.
/// - `visible[t]` — the number of softmax rows `t` actually competed in (it was
///   causally visible in). Under a causal mask this is a strictly decreasing
///   function of position, and it is the denominator that
///   [`KvScoreNormalization::MeanPerVisibleStep`] uses to cancel the early-token
///   bias documented on that enum.
///
/// Heads are aggregated by **mean**, not sum. For ranking this is a no-op (it
/// divides every token's score by the same constant `num_heads`), but it keeps
/// the accumulated score on the interpretable scale of "probability mass per
/// softmax row", so `accumulated[t] / visible[t] ∈ [0, 1]` reads directly as
/// *"the average fraction of its attention a query spends on `t` when it can
/// see `t`"*.
///
/// # Eviction granularity
///
/// Scores are pooled across heads (and, if you accumulate more than one layer,
/// across layers), and eviction removes a token from every head and every layer
/// at once. The published policies are typically per-head — a token may be a
/// heavy hitter for head 3 and dead weight for head 5 — which would require a
/// *ragged* cache with a different surviving token set per head. That is a
/// strictly larger design (and a strictly larger memory saving); pooling is the
/// conservative choice, since a token that any head loves keeps a high pooled
/// mean and survives. The ragged variant is noted as follow-up work rather than
/// silently pretended.
#[derive(Debug, Clone, PartialEq)]
pub struct KvAttentionStats {
    /// Column sums of the attention matrix, one per cached slot.
    accumulated: Vec<f64>,
    /// Number of softmax rows each slot competed in (fractional once a decay
    /// factor is in play, since the count is decayed in lockstep with the mass
    /// so that their ratio stays a proper weighted mean).
    visible: Vec<f64>,
    /// The last `observation_capacity` query rows, keyed by absolute query
    /// position: `position -> per-slot attention (head-mean)`. This is
    /// `SnapKV`'s ballot box.
    observation: BTreeMap<usize, Vec<f64>>,
    observation_capacity: usize,
    /// Total query rows folded in, ever. Undecayed, so it is an honest "has
    /// anything been observed at all?" flag.
    steps_recorded: u64,
}

impl KvAttentionStats {
    /// Fresh statistics for a cache of `num_tokens` slots, retaining the last
    /// `observation_capacity` query rows for `SnapKV`.
    #[must_use]
    pub fn new(num_tokens: usize, observation_capacity: usize) -> Self {
        Self {
            accumulated: vec![0.0; num_tokens],
            visible: vec![0.0; num_tokens],
            observation: BTreeMap::new(),
            observation_capacity,
            steps_recorded: 0,
        }
    }

    /// Fresh statistics sized for `cache`, retaining exactly as many query rows
    /// as `config`'s observation window needs.
    #[must_use]
    pub fn for_cache(cache: &KvCacheTensor, config: &KvCompressionConfig) -> Self {
        Self::new(cache.seq_len(), config.observation_window)
    }

    /// Number of slots tracked. Must equal the cache's `seq_len`.
    #[must_use]
    pub fn num_tokens(&self) -> usize {
        self.accumulated.len()
    }

    /// Whether any query step has ever been folded in.
    #[must_use]
    pub const fn has_history(&self) -> bool {
        self.steps_recorded > 0
    }

    /// Total query rows folded in, ever.
    #[must_use]
    pub const fn steps_recorded(&self) -> u64 {
        self.steps_recorded
    }

    /// The raw accumulated attention mass per slot (H2O's unnormalized column
    /// sum).
    #[must_use]
    pub fn accumulated(&self) -> &[f64] {
        &self.accumulated
    }

    /// The number of softmax rows each slot competed in.
    #[must_use]
    pub fn visible_steps(&self) -> &[f64] {
        &self.visible
    }

    /// Extend the tracked slots by `count` freshly-appended tokens (zero mass,
    /// zero visible rows).
    pub fn on_append(&mut self, count: usize) {
        self.accumulated.resize(self.accumulated.len() + count, 0.0);
        self.visible.resize(self.visible.len() + count, 0.0);
    }

    /// Fold one attention call's weight matrix into the running statistics.
    ///
    /// Query rows are folded in the order they appear in `attention`, which
    /// [`scaled_dot_product_attention`] guarantees is chronological. When
    /// `decay` is `Some(d)`, the running mass *and* the running visible-row
    /// count are both multiplied by `d` before each new row is added, so their
    /// ratio remains a proper exponentially-weighted mean rather than drifting.
    ///
    /// Accumulating several layers for the same query steps applies the decay
    /// once per *folded row*, i.e. once per `(layer, query)` pair. If you want
    /// decay to mean "once per query step", accumulate a single representative
    /// layer (which is what the published policies do — they read the scores off
    /// one layer, or off each layer independently).
    ///
    /// # Errors
    ///
    /// - [`KvCompressionError::StatsLengthMismatch`] if `attention` was computed
    ///   over a different number of slots than these statistics track.
    /// - [`KvCompressionError::InvalidConfig`] if `decay` is outside `(0, 1]`.
    pub fn accumulate(
        &mut self,
        attention: &KvAttentionOutput,
        decay: Option<f64>,
    ) -> KvResult<()> {
        if attention.num_keys() != self.num_tokens() {
            return Err(KvCompressionError::StatsLengthMismatch {
                stats_tokens: self.num_tokens(),
                cache_tokens: attention.num_keys(),
            });
        }
        if let Some(factor) = decay
            && (!factor.is_finite() || factor <= 0.0 || factor > 1.0)
        {
            return Err(KvCompressionError::InvalidConfig {
                reason: format!("score_decay must lie in (0, 1], got {factor}"),
            });
        }

        let num_keys = attention.num_keys();
        let num_heads = attention.num_heads();
        let head_scale = 1.0 / num_heads as f64;

        for query in 0..attention.num_queries() {
            if let Some(factor) = decay {
                for mass in &mut self.accumulated {
                    *mass *= factor;
                }
                for count in &mut self.visible {
                    *count *= factor;
                }
            }

            // Mean over heads of this query's weight row.
            let mut row = vec![0.0f64; num_keys];
            for head in 0..num_heads {
                let weight_row = attention.weight_row(head, query).ok_or_else(|| {
                    KvCompressionError::InvalidTokenSelection {
                        reason: format!(
                            "attention output is missing row (head {head}, query {query})"
                        ),
                    }
                })?;
                for (slot, &weight) in row.iter_mut().zip(weight_row.iter()) {
                    *slot += weight * head_scale;
                }
            }

            for (token, &mass) in row.iter().enumerate() {
                if attention.is_visible(query, token) {
                    self.accumulated[token] += mass;
                    self.visible[token] += 1.0;
                }
            }

            // Record the ballot for SnapKV. Re-accumulating the same query
            // position (e.g. from a second layer) adds into the existing ballot
            // rather than replacing it.
            let position = attention.query_positions()[query];
            match self.observation.get_mut(&position) {
                Some(existing) => {
                    if existing.len() < num_keys {
                        existing.resize(num_keys, 0.0);
                    }
                    for (slot, &weight) in existing.iter_mut().zip(row.iter()) {
                        *slot += weight;
                    }
                }
                None => {
                    self.observation.insert(position, row);
                }
            }
            self.trim_observation();

            self.steps_recorded += 1;
        }
        Ok(())
    }

    /// Drop the oldest ballots beyond the observation capacity.
    fn trim_observation(&mut self) {
        while self.observation.len() > self.observation_capacity {
            let Some(&oldest) = self.observation.keys().next() else {
                break;
            };
            self.observation.remove(&oldest);
        }
    }

    /// The saliency score per slot, under the given normalization. This is the
    /// vector the heavy-hitter policies take their top-`k` from.
    ///
    /// A slot that has never been visible to any query scores `0.0` under either
    /// normalization (rather than `0/0`).
    #[must_use]
    pub fn saliency(&self, normalization: KvScoreNormalization) -> Vec<f64> {
        match normalization {
            KvScoreNormalization::Cumulative => self.accumulated.clone(),
            KvScoreNormalization::MeanPerVisibleStep => self
                .accumulated
                .iter()
                .zip(self.visible.iter())
                .map(|(&mass, &count)| if count > 0.0 { mass / count } else { 0.0 })
                .collect(),
        }
    }

    /// `SnapKV`'s vote vector: the attention the **last `window` query
    /// positions** placed on each slot, summed.
    ///
    /// Returns a vector of length [`Self::num_tokens`]. Ballots recorded before
    /// slots were appended are shorter than the current slot count; their
    /// missing tail counts as zero mass, which is correct — those queries could
    /// not have attended to tokens that did not yet exist.
    #[must_use]
    pub fn observation_votes(&self, window: usize) -> Vec<f64> {
        let mut votes = vec![0.0f64; self.num_tokens()];
        for ballot in self.observation.values().rev().take(window) {
            for (slot, &weight) in votes.iter_mut().zip(ballot.iter()) {
                *slot += weight;
            }
        }
        votes
    }

    /// The number of ballots currently held (at most the observation capacity).
    #[must_use]
    pub fn ballot_count(&self) -> usize {
        self.observation.len()
    }

    /// Re-index the statistics onto the surviving slots after an eviction.
    ///
    /// `keep` is the same strictly-increasing slot list handed to
    /// [`KvCacheTensor::retain_tokens`]; the statistics are gathered by it so
    /// that slot `i` of the compressed cache keeps the history of slot
    /// `keep[i]` of the old one. Without this the accumulated scores would
    /// silently refer to the wrong tokens after the very first compression pass.
    ///
    /// # Errors
    ///
    /// [`KvCompressionError::InvalidTokenSelection`] if `keep` is unsorted, has
    /// duplicates, or names a slot that does not exist.
    pub fn retain_tokens(&mut self, keep: &[usize]) -> KvResult<()> {
        let num_tokens = self.num_tokens();
        let mut previous: Option<usize> = None;
        for &index in keep {
            if index >= num_tokens {
                return Err(KvCompressionError::InvalidTokenSelection {
                    reason: format!(
                        "slot {index} is out of range for statistics over {num_tokens} token(s)"
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

        self.accumulated = keep.iter().map(|&index| self.accumulated[index]).collect();
        self.visible = keep.iter().map(|&index| self.visible[index]).collect();
        for ballot in self.observation.values_mut() {
            *ballot = keep
                .iter()
                .map(|&index| ballot.get(index).copied().unwrap_or(0.0))
                .collect();
        }
        Ok(())
    }
}
