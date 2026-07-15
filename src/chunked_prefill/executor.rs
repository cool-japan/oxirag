//! Executing a [`PrefillPlan`] against a real KV cache with a real attention
//! kernel.
//!
//! # The loop, and the three places it can go wrong
//!
//! For each chunk, in order:
//!
//! ```text
//! 1.  committed  ←  cache.seq_len()          ── BEFORE appending. The mask boundary.
//! 2.  append the chunk's keys and values, token by token, in prompt order
//! 3.  positions  ←  cache.positions()[committed ..]   ── absolute, from the cache itself
//! 4.  for each layer:  O_chunk = attention(cache, layer, Q_chunk, positions)
//! 5.  scatter O_chunk into rows [start .. start+n] of the prompt's output
//! ```
//!
//! Every step is a trap:
//!
//! * **Step 1 before step 2, never after.** `committed` is the width of the
//!   unmasked prefix segment. Read the cache length *after* appending and the
//!   prefix segment swallows the chunk's own keys: every query sees every key,
//!   the mask stops being causal, and the output is a plausible, finite,
//!   completely wrong tensor.
//! * **Step 2 before step 4, never after.** A chunk's queries must attend to the
//!   chunk's *own* keys — a token attends to itself, and to its predecessors
//!   inside the chunk. Attending before appending silently deletes the diagonal.
//! * **Step 3 from the cache, never from the chunk.** The chunk knows its offset
//!   *within the prompt*; the kernel's causal mask is driven by the token's
//!   *absolute stream position*. Those coincide only when the cache starts
//!   empty. Hand the kernel `0..n` and the mask collapses; hand it `s..s+n` when
//!   a shared prefix pushed the stream forward, and it is wrong by the length of
//!   the prefix. The cache is the only thing that knows the truth, so the
//!   positions are read back out of it.
//!
//! Step 4's mask is then cross-checked against [`PrefillMaskGeometry`], which
//! derived the same mask from `(committed, n)` without ever looking at a
//! position. Any of the three traps above makes those two masks disagree.
//!
//! # Why the answer is *bit-for-bit* the one-shot answer
//!
//! Not "within a tolerance" — identical, every bit. For a query at absolute
//! position `p`:
//!
//! 1. **The visible key set is the same.** Chunked or not, the causal mask admits
//!    exactly the slots with position `<= p`, and every one of them is already in
//!    the cache when `p`'s chunk runs (its own chunk's keys were appended in step
//!    2). The chunked cache is a *prefix* of the one-shot cache, so those slots
//!    even sit at the same indices.
//! 2. **The logits are the same numbers.** Each is an `f64` dot product of the
//!    same `f32` vectors accumulated in the same order, so it is not merely close
//!    — it is the same `f64`.
//! 3. **The extra columns contribute exactly nothing.** A one-shot row spans `P`
//!    keys where the chunked row spans `committed + n <= P`. The extra columns are
//!    masked, so their weight is `exp(-inf - max) = 0.0` *exactly*; adding `0.0`
//!    to the `f64` softmax denominator is the identity, and the value sum skips
//!    zero weights outright.
//! 4. **The reductions run in the same order.** Both sum over slots ascending.
//!
//! So the softmax denominator, the weights, the `f64` value accumulator and the
//! `f32` it narrows to are all identical. **The tolerance is `0.0`, and the tests
//! assert equality, not closeness** — a one-`ULP` drift would be a real bug, not
//! floating-point noise.

use crate::kv_cache_compression::{KvAttentionOutput, KvCacheTensor, scaled_dot_product_attention};

use super::mask::PrefillMaskGeometry;
use super::model::{PrefillModel, check_cache_geometry};
use super::types::{
    ChunkedPrefillConfig, ChunkedPrefillError, ChunkedPrefillResult, PrefillChunk, PrefillGeometry,
    PrefillPlan, widen,
};

// ── PrefillOutput ────────────────────────────────────────────────────────────

/// The attention output of a whole prompt, `[layer][head][token][dim]`.
///
/// Assembled chunk by chunk, and — by the argument in the [module docs](self) —
/// bit-for-bit identical to what a one-shot prefill would have produced.
#[derive(Debug, Clone, PartialEq)]
pub struct PrefillOutput {
    geometry: PrefillGeometry,
    prompt_tokens: usize,
    values: Vec<f32>,
}

impl PrefillOutput {
    /// The model geometry these outputs were produced under.
    #[must_use]
    pub const fn geometry(&self) -> PrefillGeometry {
        self.geometry
    }

    /// The prompt's token count.
    #[must_use]
    pub const fn prompt_tokens(&self) -> usize {
        self.prompt_tokens
    }

    /// The `head_dim`-long output vector of one `(layer, head, token)`, or
    /// `None` if any index is out of range.
    #[must_use]
    pub fn row(&self, layer: usize, head: usize, token: usize) -> Option<&[f32]> {
        if layer >= self.geometry.num_layers()
            || head >= self.geometry.num_heads()
            || token >= self.prompt_tokens
        {
            return None;
        }
        let head_dim = self.geometry.head_dim();
        let start =
            ((layer * self.geometry.num_heads() + head) * self.prompt_tokens + token) * head_dim;
        self.values.get(start..start + head_dim)
    }

    /// The whole buffer, `[layer][head][token][dim]`.
    #[must_use]
    pub fn values(&self) -> &[f32] {
        &self.values
    }
}

// ── PrefillChunkReport ───────────────────────────────────────────────────────

/// What one chunk's execution did.
#[derive(Debug, Clone, PartialEq)]
pub struct PrefillChunkReport {
    chunk: PrefillChunk,
    geometry: PrefillMaskGeometry,
    stream_positions: Vec<usize>,
    cache_tokens_after: usize,
    attention: Vec<KvAttentionOutput>,
}

impl PrefillChunkReport {
    /// The chunk that was executed.
    #[must_use]
    pub const fn chunk(&self) -> PrefillChunk {
        self.chunk
    }

    /// The piecewise mask this chunk ran under.
    #[must_use]
    pub const fn mask(&self) -> PrefillMaskGeometry {
        self.geometry
    }

    /// The cache slots that were already committed when this chunk ran — the
    /// width of the unmasked prefix segment.
    ///
    /// This equals the chunk's prompt offset only for a lossless cache that
    /// started empty. A shared prefix raises it; compression lowers it.
    #[must_use]
    pub const fn committed_keys(&self) -> usize {
        self.geometry.committed_keys()
    }

    /// The **absolute stream positions** this chunk's tokens landed on, in
    /// prompt order. Read back out of the cache, not assumed.
    ///
    /// Concatenating these across a plan's chunks reproduces, element for
    /// element, the positions a one-shot prefill would have assigned — which is
    /// exactly what makes the causal mask survive a chunk boundary.
    #[must_use]
    pub fn stream_positions(&self) -> &[usize] {
        &self.stream_positions
    }

    /// Cache slots held after this chunk's keys were appended.
    #[must_use]
    pub const fn cache_tokens_after(&self) -> usize {
        self.cache_tokens_after
    }

    /// This chunk's attention output for `layer`, if
    /// [`ChunkedPrefillConfig::retain_attention`] was set.
    #[must_use]
    pub fn attention(&self, layer: usize) -> Option<&KvAttentionOutput> {
        self.attention.get(layer)
    }

    /// `Q·K` dot products this chunk performed, per head.
    #[must_use]
    pub const fn attention_cells(&self) -> u64 {
        self.geometry.cells()
    }

    /// KV slots this chunk's attention required to be resident.
    #[must_use]
    pub const fn kv_slot_visits(&self) -> u64 {
        self.geometry.kv_slot_visits()
    }
}

// ── PrefillRunStats ──────────────────────────────────────────────────────────

/// The measured cost of one prefill, and what a one-shot prefill would have
/// cost.
///
/// The two pairs of counters are the module's whole economic argument, and they
/// point in opposite directions: see [`PrefillPlan`]'s cost identities.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct PrefillRunStats {
    prompt_tokens: usize,
    prefix_tokens: usize,
    num_chunks: usize,
    min_chunk_tokens: usize,
    max_chunk_tokens: usize,
    attention_cells: u64,
    one_shot_attention_cells: u64,
    kv_slot_visits: u64,
    one_shot_kv_slot_visits: u64,
}

impl PrefillRunStats {
    /// Prompt tokens prefilled.
    #[must_use]
    pub const fn prompt_tokens(&self) -> usize {
        self.prompt_tokens
    }

    /// Cache slots already committed before the prefill began (a shared prefix,
    /// a previous turn, or zero).
    #[must_use]
    pub const fn prefix_tokens(&self) -> usize {
        self.prefix_tokens
    }

    /// Chunks executed.
    #[must_use]
    pub const fn num_chunks(&self) -> usize {
        self.num_chunks
    }

    /// The smallest chunk executed.
    #[must_use]
    pub const fn min_chunk_tokens(&self) -> usize {
        self.min_chunk_tokens
    }

    /// The largest chunk executed. Never above the token budget.
    #[must_use]
    pub const fn max_chunk_tokens(&self) -> usize {
        self.max_chunk_tokens
    }

    /// `Q·K` dot products actually performed, per head, summed over chunks.
    #[must_use]
    pub const fn attention_cells(&self) -> u64 {
        self.attention_cells
    }

    /// What a one-shot prefill would have performed. **Equal to
    /// [`Self::attention_cells`]** on a lossless cache: chunking is free in
    /// arithmetic, and the tests assert that for every composition of the prompt.
    #[must_use]
    pub const fn one_shot_attention_cells(&self) -> u64 {
        self.one_shot_attention_cells
    }

    /// KV slots the chunks required to be resident, summed over invocations.
    #[must_use]
    pub const fn kv_slot_visits(&self) -> u64 {
        self.kv_slot_visits
    }

    /// What a one-shot prefill would have required: `prefix + prompt`, once.
    #[must_use]
    pub const fn one_shot_kv_slot_visits(&self) -> u64 {
        self.one_shot_kv_slot_visits
    }

    /// [`Self::kv_slot_visits`] over [`Self::one_shot_kv_slot_visits`] — the
    /// price of chunking, and the *only* thing it costs. `1.0` for a one-shot
    /// plan; `(m + 1) / 2` for `m` uniform chunks over an empty cache.
    #[must_use]
    #[allow(
        // Slot counts are far below 2^53, so the ratio is exact.
        clippy::cast_precision_loss
    )]
    pub fn kv_read_amplification(&self) -> f64 {
        if self.one_shot_kv_slot_visits == 0 {
            return 0.0;
        }
        self.kv_slot_visits as f64 / self.one_shot_kv_slot_visits as f64
    }
}

// ── PrefillRun ───────────────────────────────────────────────────────────────

/// Everything one chunked prefill produced.
#[derive(Debug, Clone, PartialEq)]
pub struct PrefillRun {
    output: PrefillOutput,
    chunks: Vec<PrefillChunkReport>,
    stats: PrefillRunStats,
}

impl PrefillRun {
    /// The prompt's assembled attention output.
    #[must_use]
    pub const fn output(&self) -> &PrefillOutput {
        &self.output
    }

    /// Per-chunk execution records, in order.
    #[must_use]
    pub fn chunks(&self) -> &[PrefillChunkReport] {
        &self.chunks
    }

    /// The measured cost.
    #[must_use]
    pub const fn stats(&self) -> PrefillRunStats {
        self.stats
    }

    /// Every prompt token's absolute stream position, in prompt order — the
    /// chunks' [`PrefillChunkReport::stream_positions`] concatenated.
    #[must_use]
    pub fn stream_positions(&self) -> Vec<usize> {
        self.chunks
            .iter()
            .flat_map(|report| report.stream_positions().iter().copied())
            .collect()
    }
}

// ── ChunkedPrefill ───────────────────────────────────────────────────────────

/// `Sarathi-Serve` chunked prefill: run a prompt through a KV cache one
/// token-budgeted chunk at a time, and get the one-shot answer back.
///
/// ```
/// use oxirag::chunked_prefill::{
///     ChunkedPrefill, ChunkedPrefillConfig, PrefillGeometry, PrefillTensorModel, TokenBudget,
/// };
/// use oxirag::kv_cache_compression::KvCacheTensor;
/// # fn main() -> Result<(), Box<dyn std::error::Error>> {
/// let geometry = PrefillGeometry::new(2, 2, 4)?;
/// let prompt_tokens = 10;
/// let elements = prompt_tokens * geometry.token_stride();
/// # #[allow(clippy::cast_precision_loss)]
/// let fill = |scale: f32| -> Vec<f32> {
///     (0..elements).map(|i| scale * ((i % 7) as f32 - 3.0)).collect()
/// };
/// let model = PrefillTensorModel::new(geometry, prompt_tokens, fill(0.1), fill(0.2), fill(0.3))?;
///
/// // Four tokens per iteration: the 10-token prompt becomes chunks of 4, 4, 2.
/// let budget = TokenBudget::new(4, 1)?;
/// let engine = ChunkedPrefill::new(ChunkedPrefillConfig::new(budget));
///
/// let mut cache = KvCacheTensor::new(2, 2, 4)?;
/// let run = engine.prefill(&model, &mut cache)?;
///
/// assert_eq!(run.chunks().len(), 3);
/// assert_eq!(run.stats().max_chunk_tokens(), 4);
/// // Absolute positions survive every boundary.
/// assert_eq!(run.stream_positions(), (0..10).collect::<Vec<_>>());
/// // Chunking cost no arithmetic at all...
/// assert_eq!(run.stats().attention_cells(), run.stats().one_shot_attention_cells());
/// // ...and cost exactly this much extra KV traffic.
/// assert_eq!(run.stats().kv_slot_visits(), 4 + 8 + 10);
/// # Ok(())
/// # }
/// ```
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default)]
pub struct ChunkedPrefill {
    config: ChunkedPrefillConfig,
}

impl ChunkedPrefill {
    /// An executor with the given configuration.
    #[must_use]
    pub const fn new(config: ChunkedPrefillConfig) -> Self {
        Self { config }
    }

    /// The configuration.
    #[must_use]
    pub const fn config(&self) -> ChunkedPrefillConfig {
        self.config
    }

    /// Split a prompt into chunks of at most
    /// [`TokenBudget::total`](super::TokenBudget::total) tokens.
    ///
    /// This is the no-decodes plan: the whole budget goes to prefill. In a live
    /// hybrid batch the chunks are smaller, because decodes take their share
    /// first — see [`PrefillScheduler`](super::PrefillScheduler), whose
    /// [`PrefillSequenceTrace::plan`](super::PrefillSequenceTrace::plan) hands
    /// back exactly the chunk sizes it scheduled.
    ///
    /// # Errors
    ///
    /// [`ChunkedPrefillError::EmptyPrompt`] if `prompt_tokens` is zero.
    pub fn plan(&self, prompt_tokens: usize) -> ChunkedPrefillResult<PrefillPlan> {
        PrefillPlan::uniform(prompt_tokens, self.config.budget().total())
    }

    /// Plan and run in one step.
    ///
    /// # Errors
    ///
    /// As [`Self::plan`] and [`Self::run`].
    pub fn prefill<M>(
        &self,
        model: &M,
        cache: &mut KvCacheTensor,
    ) -> ChunkedPrefillResult<PrefillRun>
    where
        M: PrefillModel + ?Sized,
    {
        let plan = self.plan(model.prompt_tokens())?;
        self.run(model, &plan, cache)
    }

    /// Execute `plan` against `cache`, carrying the KV forward across every
    /// chunk boundary.
    ///
    /// `cache` may already hold committed tokens — a shared prefix, or an earlier
    /// turn of a conversation. Those tokens are attended to by every chunk and
    /// are never rewritten, and the prompt's own tokens are appended after them.
    ///
    /// # Errors
    ///
    /// - [`ChunkedPrefillError::PromptLengthMismatch`] if the plan and the model
    ///   disagree about the prompt length.
    /// - [`ChunkedPrefillError::GeometryMismatch`] if the model cannot write into
    ///   this cache.
    /// - [`ChunkedPrefillError::ChunkOverBudget`] if any chunk is larger than the
    ///   token budget. A chunked prefill that breaks its budget has abandoned the
    ///   only thing it was for.
    /// - [`ChunkedPrefillError::MaskDivergence`] or
    ///   [`ChunkedPrefillError::MaskShapeDivergence`] if the piecewise mask and
    ///   the kernel's absolute-position mask disagree — see
    ///   [`PrefillMaskGeometry::audit`].
    /// - [`ChunkedPrefillError::PositionDivergence`] if the cache did not assign
    ///   the chunk consecutive absolute positions.
    /// - [`ChunkedPrefillError::Cache`] if the cache or the attention kernel
    ///   rejects an input.
    pub fn run<M>(
        &self,
        model: &M,
        plan: &PrefillPlan,
        cache: &mut KvCacheTensor,
    ) -> ChunkedPrefillResult<PrefillRun>
    where
        M: PrefillModel + ?Sized,
    {
        let geometry = model.geometry();
        if plan.prompt_tokens() != model.prompt_tokens() {
            return Err(ChunkedPrefillError::PromptLengthMismatch {
                plan_tokens: plan.prompt_tokens(),
                model_tokens: model.prompt_tokens(),
            });
        }
        check_cache_geometry(geometry, cache)?;
        plan.verify_budget(&self.config.budget())?;

        let prompt_tokens = plan.prompt_tokens();
        let prefix_tokens = cache.seq_len();

        let mut output = PrefillOutput {
            geometry,
            prompt_tokens,
            values: vec![
                0.0f32;
                geometry.num_layers()
                    * geometry.num_heads()
                    * prompt_tokens
                    * geometry.head_dim()
            ],
        };
        let mut reports = Vec::with_capacity(plan.num_chunks());

        for &chunk in plan.chunks() {
            reports.push(self.run_chunk(model, chunk, cache, &mut output)?);
        }

        let stats = summarize(plan, prefix_tokens, &reports);
        Ok(PrefillRun {
            output,
            chunks: reports,
            stats,
        })
    }

    /// One chunk: commit its KV, attend over everything committed, scatter the
    /// result. The comments here mirror the numbered steps in the
    /// [module docs](self).
    fn run_chunk<M>(
        &self,
        model: &M,
        chunk: PrefillChunk,
        cache: &mut KvCacheTensor,
        output: &mut PrefillOutput,
    ) -> ChunkedPrefillResult<PrefillChunkReport>
    where
        M: PrefillModel + ?Sized,
    {
        // ── 1. The mask boundary, captured BEFORE the chunk's keys land. ──────
        let committed_keys = cache.seq_len();

        // ── 2. Commit the chunk's KV, in prompt order. It must happen before the
        //       attention: a token attends to itself and to its predecessors
        //       inside the chunk, and those keys have to exist for it to. ──────
        for token in chunk.start()..chunk.end() {
            cache.append_token(model.token_keys(token)?, model.token_values(token)?)?;
        }

        // ── 3. The chunk's ABSOLUTE stream positions, read back out of the cache
        //       rather than assumed from its prompt offset. The two coincide only
        //       when the cache started empty. ────────────────────────────────────
        let stream_positions: Vec<usize> = cache.positions()[committed_keys..].to_vec();
        verify_consecutive(&stream_positions)?;

        // The piecewise mask, from segment geometry alone: no position ever
        // enters this derivation, which is why gappy (compressed) caches cannot
        // confuse it.
        let mask = PrefillMaskGeometry::new(committed_keys, chunk.token_count());

        // ── 4/5. Attend, audit, scatter — once per layer. ─────────────────────
        let geometry = model.geometry();
        let mut retained = Vec::new();
        for layer in 0..geometry.num_layers() {
            let queries = chunk_query_buffer(model, geometry, chunk, layer)?;
            let attention =
                scaled_dot_product_attention(cache, layer, &queries, &stream_positions)?;

            if self.config.audit_mask() {
                mask.audit(&attention)?;
            }

            scatter_output(&attention, geometry, chunk, layer, output);

            if self.config.retain_attention() {
                retained.push(attention);
            }
        }

        Ok(PrefillChunkReport {
            chunk,
            geometry: mask,
            stream_positions,
            cache_tokens_after: cache.seq_len(),
            attention: retained,
        })
    }
}

// ── Helpers ──────────────────────────────────────────────────────────────────

/// Build one layer's `[head][query][dim]` query buffer for a chunk.
///
/// The model stores a token's query as `[head][dim]`; the kernel wants the
/// chunk's queries **head-major**, so the copy is a transpose. The query axis has
/// length `n` (this chunk), not `P` (the prompt) — get the stride wrong and every
/// head reads another head's queries, which is a bug no amount of squinting at the
/// output will reveal.
fn chunk_query_buffer<M>(
    model: &M,
    geometry: PrefillGeometry,
    chunk: PrefillChunk,
    layer: usize,
) -> ChunkedPrefillResult<Vec<f32>>
where
    M: PrefillModel + ?Sized,
{
    let num_heads = geometry.num_heads();
    let head_dim = geometry.head_dim();
    let num_queries = chunk.token_count();

    let mut buffer = vec![0.0f32; num_heads * num_queries * head_dim];
    for offset in 0..num_queries {
        let token_query = model.token_query(layer, chunk.start() + offset)?;
        if token_query.len() != geometry.layer_stride() {
            return Err(ChunkedPrefillError::ShapeMismatch {
                what: "layer query block",
                expected: geometry.layer_stride(),
                actual: token_query.len(),
            });
        }
        for head in 0..num_heads {
            let source = &token_query[head * head_dim..(head + 1) * head_dim];
            let start = (head * num_queries + offset) * head_dim;
            buffer[start..start + head_dim].copy_from_slice(source);
        }
    }
    Ok(buffer)
}

/// Copy one chunk's `[head][query][dim]` attention output into rows
/// `[start .. start + n]` of the prompt's `[layer][head][token][dim]` output.
fn scatter_output(
    attention: &KvAttentionOutput,
    geometry: PrefillGeometry,
    chunk: PrefillChunk,
    layer: usize,
    output: &mut PrefillOutput,
) {
    let head_dim = geometry.head_dim();
    let prompt_tokens = output.prompt_tokens;
    for head in 0..geometry.num_heads() {
        for offset in 0..chunk.token_count() {
            let Some(row) = attention.output_row(head, offset) else {
                continue;
            };
            let token = chunk.start() + offset;
            let start = ((layer * geometry.num_heads() + head) * prompt_tokens + token) * head_dim;
            output.values[start..start + head_dim].copy_from_slice(row);
        }
    }
}

/// A chunk's tokens are appended back to back, so the positions the cache hands
/// them must be consecutive. They are not merely *expected* to be — the causal
/// mask's triangular segment is built on the assumption, so it is checked.
fn verify_consecutive(positions: &[usize]) -> ChunkedPrefillResult<()> {
    let Some(&first) = positions.first() else {
        return Ok(());
    };
    for (offset, &position) in positions.iter().enumerate() {
        let expected = first + offset;
        if position != expected {
            return Err(ChunkedPrefillError::PositionDivergence {
                offset,
                expected,
                actual: position,
            });
        }
    }
    Ok(())
}

/// Fold the per-chunk records into the run's cost summary.
fn summarize(
    plan: &PrefillPlan,
    prefix_tokens: usize,
    reports: &[PrefillChunkReport],
) -> PrefillRunStats {
    let prompt_tokens = plan.prompt_tokens();
    let prompt = widen(prompt_tokens);
    let prefix = widen(prefix_tokens);

    PrefillRunStats {
        prompt_tokens,
        prefix_tokens,
        num_chunks: reports.len(),
        min_chunk_tokens: plan.min_chunk_tokens(),
        max_chunk_tokens: plan.max_chunk_tokens(),
        attention_cells: reports
            .iter()
            .map(PrefillChunkReport::attention_cells)
            .sum(),
        // The closed form of identity 1: what a single P-token chunk over the
        // same prefix would have cost. Equal to the sum above, always.
        one_shot_attention_cells: prompt * prefix + prompt * (prompt + 1) / 2,
        kv_slot_visits: reports.iter().map(PrefillChunkReport::kv_slot_visits).sum(),
        one_shot_kv_slot_visits: prefix + prompt,
    }
}
