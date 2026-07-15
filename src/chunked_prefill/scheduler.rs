//! Hybrid batching: packing one prefill chunk and every runnable decode into a
//! single token budget — and the stall it thereby prevents.
//!
//! # The stall
//!
//! A prefill is one enormous parallel forward pass over the whole prompt; a
//! decode is one token. Batch them the classical way — an iteration is *either*
//! a prefill *or* a batch of decodes, and prefill goes first so new requests get
//! admitted — and a 4096-token prompt arriving mid-flight seizes an entire
//! iteration. Every sequence already generating stops dead until it finishes.
//! That is a **generation stall**, and its length is the prompt's length: a
//! quantity the running sequences have no relationship with and no defence
//! against. This is [`PrefillPacking::Exclusive`], and it is the baseline.
//!
//! # The fix, and the exact bound it buys
//!
//! [`PrefillPacking::StallFree`] — `Sarathi-Serve` — makes an iteration carry
//! **exactly one prefill chunk plus every runnable decode**, inside one token
//! budget `B`:
//!
//! ```text
//! decode_tokens  =  (number of sequences currently decoding)      ≤ D_max
//! chunk_tokens   =  min(prompt tokens left,  B − decode_tokens)   ≥ B − D_max ≥ 1
//! ```
//!
//! Two consequences, and both are *integer theorems*, not measurements:
//!
//! 1. **No iteration exceeds `B` tokens**, because the chunk takes only what the
//!    decodes left behind. The prompt's length has dropped out entirely.
//! 2. **No decoding sequence is ever skipped.** Decodes are taken first and in
//!    full, so a sequence emits a token in *every* iteration from the one its
//!    prefill finished in. Consecutive tokens are therefore exactly one iteration
//!    apart, and by (1) that iteration costs at most `B` tokens.
//!
//! Together: **`max time-between-tokens ≤ cost(B tokens)`, whatever the prompt
//! lengths are.** The stall is not shortened; it is *bounded*, and bounded by a
//! number the operator chooses. [`PrefillStallStats::max_time_between_tokens`]
//! reports the measured figure and [`PrefillStallStats::is_stall_free`] asserts
//! the bound.
//!
//! Requiring `D_max < B` is what makes step 2 keepable: it reserves
//! [`TokenBudget::prefill_reserve`] `>= 1` tokens for the chunk in every
//! iteration, so prefill cannot be starved by a decode batch that has grown to
//! fill the budget. And if more sequences want to decode than `D_max` admits,
//! this module **refuses** ([`ChunkedPrefillError::DecodeCapExceeded`]) rather
//! than dropping one, because dropping one is the stall.
//!
//! # What this is not
//!
//! This is *within-request* work splitting and *within-iteration* budget
//! packing. It is deliberately not a request scheduler:
//!
//! * Prompts are prefilled in **the order given**. There is no priority, no
//!   deadline, no fairness, no starvation control — `request_scheduling` owns all
//!   of that, and it decides the order this module consumes.
//! * There is **no admission control**. If the caller hands over more
//!   concurrently-decoding sequences than the budget's decode cap, that is an
//!   admission bug upstream, and it is reported as an error.
//! * There is **no KV allocator**. Where a chunk's keys physically live, and what
//!   happens when memory runs out, is `continuous_batching`'s concern.
//!
//! # Ticks, not clocks
//!
//! Everything here is counted in **logical iterations** and in **tokens**. No
//! wall-clock time is measured, and none is simulated: an iteration's cost is its
//! token count (the linear-layer work, which is what the token budget exists to
//! cap) and, separately, its exact `Q·K` dot-product count
//! ([`PrefillIteration::attention_cells`], which is quadratic in the context and
//! which the token budget does *not* cap — stating otherwise would be a lie about
//! the model this bound rests on).

use serde::{Deserialize, Serialize};

use super::types::{
    ChunkedPrefillError, ChunkedPrefillResult, PrefillChunk, PrefillPlan, TokenBudget, widen,
};

// ── PrefillPacking ───────────────────────────────────────────────────────────

/// How an iteration's token budget is filled.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default, Serialize, Deserialize)]
pub enum PrefillPacking {
    /// `Sarathi-Serve`: one token-budgeted prefill chunk **plus** every runnable
    /// decode, in every iteration. Decodes are taken first; the chunk gets what
    /// is left, which is never zero.
    #[default]
    StallFree,

    /// The classical baseline this module exists to beat: prefill takes priority
    /// and runs **alone and unchunked**, a whole prompt to an iteration, while
    /// every in-flight decode waits. Its iterations are as long as the longest
    /// prompt, so it honours no token budget at all — which is the finding, not a
    /// defect in the model.
    Exclusive,
}

// ── PrefillSequenceSpec ──────────────────────────────────────────────────────

/// One request's shape: how much prompt to prefill, how many tokens to generate.
///
/// The first generated token is emitted by the prompt's **final prefill chunk**
/// (its logits are that chunk's last row), so it costs no extra budget and lands
/// in the same iteration. A sequence therefore spends `output_tokens - 1`
/// iterations in the decode set — and `output_tokens = 0` (a prefill-only pass:
/// scoring, embedding, a reranker) never enters it at all.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub struct PrefillSequenceSpec {
    prompt_tokens: usize,
    output_tokens: usize,
}

impl PrefillSequenceSpec {
    /// A request with `prompt_tokens` to prefill and `output_tokens` to generate.
    ///
    /// # Errors
    ///
    /// [`ChunkedPrefillError::EmptyPrompt`] if `prompt_tokens` is zero.
    pub fn new(prompt_tokens: usize, output_tokens: usize) -> ChunkedPrefillResult<Self> {
        if prompt_tokens == 0 {
            return Err(ChunkedPrefillError::EmptyPrompt);
        }
        Ok(Self {
            prompt_tokens,
            output_tokens,
        })
    }

    /// Prompt tokens to prefill.
    #[must_use]
    pub const fn prompt_tokens(&self) -> usize {
        self.prompt_tokens
    }

    /// Tokens to generate, the first of which the final prefill chunk emits.
    #[must_use]
    pub const fn output_tokens(&self) -> usize {
        self.output_tokens
    }
}

// ── PrefillIteration ─────────────────────────────────────────────────────────

/// The one prefill chunk an iteration carries, and whose prompt it belongs to.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub struct PrefillIterationChunk {
    sequence: usize,
    chunk: PrefillChunk,
}

impl PrefillIterationChunk {
    /// Index of the sequence whose prompt this chunk belongs to.
    #[must_use]
    pub const fn sequence(&self) -> usize {
        self.sequence
    }

    /// The chunk.
    #[must_use]
    pub const fn chunk(&self) -> PrefillChunk {
        self.chunk
    }
}

/// One scheduling iteration: at most one prefill chunk, plus the decodes that
/// fit around it.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct PrefillIteration {
    tick: usize,
    prefill: Option<PrefillIterationChunk>,
    decodes: Vec<usize>,
    prefill_tokens: usize,
    decode_tokens: usize,
    attention_cells: u64,
}

impl PrefillIteration {
    /// The iteration's logical index. There is no clock here.
    #[must_use]
    pub const fn tick(&self) -> usize {
        self.tick
    }

    /// The prefill chunk this iteration carried, if any.
    #[must_use]
    pub const fn prefill(&self) -> Option<PrefillIterationChunk> {
        self.prefill
    }

    /// The sequences that emitted a decode token in this iteration.
    #[must_use]
    pub fn decodes(&self) -> &[usize] {
        &self.decodes
    }

    /// Prompt tokens processed.
    #[must_use]
    pub const fn prefill_tokens(&self) -> usize {
        self.prefill_tokens
    }

    /// Decode tokens processed — one per decoding sequence.
    #[must_use]
    pub const fn decode_tokens(&self) -> usize {
        self.decode_tokens
    }

    /// The iteration's cost in tokens: what the budget caps, and the unit the
    /// stall-free bound is stated in.
    #[must_use]
    pub const fn total_tokens(&self) -> usize {
        self.prefill_tokens + self.decode_tokens
    }

    /// The iteration's exact `Q·K` dot-product count, per head — a quantity the
    /// token budget does **not** bound, because it is quadratic in the context
    /// length. Reported rather than swept under the rug: the stall-free bound is
    /// a bound on the *linear* term.
    #[must_use]
    pub const fn attention_cells(&self) -> u64 {
        self.attention_cells
    }
}

// ── PrefillSequenceTrace ─────────────────────────────────────────────────────

/// What the schedule did to one sequence.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct PrefillSequenceTrace {
    prompt_tokens: usize,
    output_tokens: usize,
    chunk_lengths: Vec<usize>,
    token_ticks: Vec<usize>,
}

impl PrefillSequenceTrace {
    /// The prompt's token count.
    #[must_use]
    pub const fn prompt_tokens(&self) -> usize {
        self.prompt_tokens
    }

    /// The tokens the sequence was to generate.
    #[must_use]
    pub const fn output_tokens(&self) -> usize {
        self.output_tokens
    }

    /// The chunk sizes the scheduler actually handed this prompt — the residuals
    /// the decodes left behind, iteration by iteration.
    #[must_use]
    pub fn chunk_lengths(&self) -> &[usize] {
        &self.chunk_lengths
    }

    /// The iterations in which this sequence emitted a token, in order. The first
    /// is the one its prefill finished in.
    #[must_use]
    pub fn token_ticks(&self) -> &[usize] {
        &self.token_ticks
    }

    /// The iteration this sequence's first token came out of, or `None` if it
    /// generates nothing.
    #[must_use]
    pub fn first_token_tick(&self) -> Option<usize> {
        self.token_ticks.first().copied()
    }

    /// The largest number of iterations this sequence waited between two
    /// consecutive tokens. Under [`PrefillPacking::StallFree`] this is always
    /// `1`; under [`PrefillPacking::Exclusive`] it grows with every prompt that
    /// jumps ahead of it. `0` if it emitted fewer than two tokens.
    #[must_use]
    pub fn max_token_gap_ticks(&self) -> usize {
        self.token_ticks
            .windows(2)
            .map(|pair| pair[1] - pair[0])
            .max()
            .unwrap_or(0)
    }

    /// This prompt's chunk decomposition, ready to hand to
    /// [`ChunkedPrefill::run`](super::ChunkedPrefill::run).
    ///
    /// This is the seam between the scheduler and the executor: the chunk sizes
    /// the budget produced are exactly the chunk sizes that get executed, and the
    /// tests run this plan end to end and check it against a one-shot prefill.
    ///
    /// # Errors
    ///
    /// [`ChunkedPrefillError::EmptyPrompt`] if the sequence was never scheduled.
    pub fn plan(&self) -> ChunkedPrefillResult<PrefillPlan> {
        PrefillPlan::explicit(&self.chunk_lengths)
    }
}

// ── PrefillStallStats ────────────────────────────────────────────────────────

/// The schedule's measured behaviour — and the numbers the stall-free claim
/// lives or dies by.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub struct PrefillStallStats {
    iterations: usize,
    total_tokens: u64,
    max_iteration_tokens: usize,
    max_iteration_attention_cells: u64,
    max_token_gap_ticks: usize,
    max_time_between_tokens: u64,
    max_time_to_first_token: u64,
    peak_decode_sequences: usize,
}

impl PrefillStallStats {
    /// Iterations the schedule took.
    #[must_use]
    pub const fn iterations(&self) -> usize {
        self.iterations
    }

    /// Tokens processed over the whole schedule. Invariant under the packing:
    /// both packings do exactly the same work, in a different order.
    #[must_use]
    pub const fn total_tokens(&self) -> u64 {
        self.total_tokens
    }

    /// The most tokens any single iteration processed. **Under
    /// [`PrefillPacking::StallFree`] this never exceeds
    /// [`TokenBudget::total`]**; under [`PrefillPacking::Exclusive`] it is as
    /// large as the longest prompt.
    #[must_use]
    pub const fn max_iteration_tokens(&self) -> usize {
        self.max_iteration_tokens
    }

    /// The most `Q·K` dot products any single iteration performed. Grows with the
    /// context length and is *not* bounded by the token budget.
    #[must_use]
    pub const fn max_iteration_attention_cells(&self) -> u64 {
        self.max_iteration_attention_cells
    }

    /// The most **iterations** any sequence waited between two consecutive
    /// tokens. `1` under [`PrefillPacking::StallFree`]: nobody is ever skipped.
    #[must_use]
    pub const fn max_token_gap_ticks(&self) -> usize {
        self.max_token_gap_ticks
    }

    /// The most **tokens** any sequence waited between two consecutive tokens:
    /// the sum of the costs of the iterations it sat through.
    ///
    /// This is the module's headline quantity — time-between-tokens, in the only
    /// unit available without a clock — and the stall-free theorem says it never
    /// exceeds [`TokenBudget::total`].
    #[must_use]
    pub const fn max_time_between_tokens(&self) -> u64 {
        self.max_time_between_tokens
    }

    /// The most tokens any sequence waited for its **first** token, counting from
    /// the start of the schedule. Chunking does not improve this — it is the
    /// throughput side of the trade — and it is reported so that it cannot be
    /// quietly ignored.
    #[must_use]
    pub const fn max_time_to_first_token(&self) -> u64 {
        self.max_time_to_first_token
    }

    /// The largest decode batch any iteration carried. Never above
    /// [`TokenBudget::max_decode_slots`], because exceeding it is an error.
    #[must_use]
    pub const fn peak_decode_sequences(&self) -> usize {
        self.peak_decode_sequences
    }

    /// Whether the schedule kept the stall-free promise under `budget`: no
    /// iteration over budget, no sequence skipped, and no time-between-tokens
    /// above the cost of `B` tokens.
    #[must_use]
    pub fn is_stall_free(&self, budget: TokenBudget) -> bool {
        self.max_iteration_tokens <= budget.total()
            && self.max_token_gap_ticks <= 1
            && self.max_time_between_tokens <= widen(budget.total())
    }
}

// ── PrefillSchedule ──────────────────────────────────────────────────────────

/// A completed schedule: every iteration, every sequence's trace, and the stats.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct PrefillSchedule {
    budget: TokenBudget,
    packing: PrefillPacking,
    iterations: Vec<PrefillIteration>,
    sequences: Vec<PrefillSequenceTrace>,
    stats: PrefillStallStats,
}

impl PrefillSchedule {
    /// The budget this schedule was built under.
    #[must_use]
    pub const fn budget(&self) -> TokenBudget {
        self.budget
    }

    /// The packing this schedule was built under.
    #[must_use]
    pub const fn packing(&self) -> PrefillPacking {
        self.packing
    }

    /// Every iteration, in order.
    #[must_use]
    pub fn iterations(&self) -> &[PrefillIteration] {
        &self.iterations
    }

    /// Every sequence's trace, in the order they were given.
    #[must_use]
    pub fn sequences(&self) -> &[PrefillSequenceTrace] {
        &self.sequences
    }

    /// The measured behaviour.
    #[must_use]
    pub const fn stats(&self) -> PrefillStallStats {
        self.stats
    }
}

// ── PrefillScheduler ─────────────────────────────────────────────────────────

/// Packs a token budget, iteration by iteration.
///
/// ```
/// use oxirag::chunked_prefill::{
///     PrefillPacking, PrefillScheduler, PrefillSequenceSpec, TokenBudget,
/// };
/// # fn main() -> Result<(), Box<dyn std::error::Error>> {
/// // One sequence already generating; a 20-token prompt lands on top of it.
/// let work = [
///     PrefillSequenceSpec::new(1, 6)?,   // short prompt, then 5 decode iterations
///     PrefillSequenceSpec::new(20, 1)?,  // the long prompt that would stall it
/// ];
/// let budget = TokenBudget::new(8, 4)?;
///
/// let stall_free = PrefillScheduler::new(budget, PrefillPacking::StallFree).schedule(&work)?;
/// let exclusive = PrefillScheduler::new(budget, PrefillPacking::Exclusive).schedule(&work)?;
///
/// // The chunked schedule keeps its promise: nothing over 8 tokens, nobody skipped.
/// assert!(stall_free.stats().is_stall_free(budget));
/// assert_eq!(stall_free.stats().max_iteration_tokens(), 8);
/// assert_eq!(stall_free.stats().max_token_gap_ticks(), 1);
/// assert_eq!(stall_free.stats().max_time_between_tokens(), 8);
///
/// // The classical one stalls the decoder for a whole 20-token prefill.
/// assert!(!exclusive.stats().is_stall_free(budget));
/// assert_eq!(exclusive.stats().max_iteration_tokens(), 20);
/// assert_eq!(exclusive.stats().max_time_between_tokens(), 21);
///
/// // Same work, either way -- only its arrangement differs.
/// assert_eq!(stall_free.stats().total_tokens(), exclusive.stats().total_tokens());
/// # Ok(())
/// # }
/// ```
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct PrefillScheduler {
    budget: TokenBudget,
    packing: PrefillPacking,
}

impl PrefillScheduler {
    /// A scheduler with the given budget and packing.
    #[must_use]
    pub const fn new(budget: TokenBudget, packing: PrefillPacking) -> Self {
        Self { budget, packing }
    }

    /// The budget.
    #[must_use]
    pub const fn budget(&self) -> TokenBudget {
        self.budget
    }

    /// The packing.
    #[must_use]
    pub const fn packing(&self) -> PrefillPacking {
        self.packing
    }

    /// Run `sequences` to completion, in the order given, and record every
    /// iteration.
    ///
    /// # Errors
    ///
    /// - [`ChunkedPrefillError::DecodeCapExceeded`] if more sequences want to
    ///   decode in one iteration than [`TokenBudget::max_decode_slots`] admits.
    ///   Admission control is `request_scheduling`'s job; this module will not
    ///   silently drop a decode to paper over its absence.
    /// - [`ChunkedPrefillError::ScheduleDidNotTerminate`] — unreachable, since
    ///   every iteration is proved to consume at least one token of work, and
    ///   kept as a tripwire.
    pub fn schedule(
        &self,
        sequences: &[PrefillSequenceSpec],
    ) -> ChunkedPrefillResult<PrefillSchedule> {
        let mut states: Vec<SequenceState> = sequences
            .iter()
            .map(|spec| SequenceState::new(*spec))
            .collect();
        let mut traces: Vec<PrefillSequenceTrace> = sequences
            .iter()
            .map(|spec| PrefillSequenceTrace {
                prompt_tokens: spec.prompt_tokens(),
                output_tokens: spec.output_tokens(),
                chunk_lengths: Vec::new(),
                token_ticks: Vec::new(),
            })
            .collect();

        // Every iteration consumes at least one token of work (a chunk of >= 1
        // prompt token, or >= 1 decode token), so the schedule cannot run longer
        // than the total work. The `Exclusive` packing additionally alternates
        // between prefill and decode iterations, which cannot double the count
        // but is allowed for anyway.
        let work: usize = sequences
            .iter()
            .map(|spec| spec.prompt_tokens() + spec.output_tokens())
            .sum();
        let limit = 2 * work + sequences.len() + 1;

        let mut iterations: Vec<PrefillIteration> = Vec::new();
        loop {
            if states.iter().all(SequenceState::is_done) {
                break;
            }
            if iterations.len() > limit {
                return Err(ChunkedPrefillError::ScheduleDidNotTerminate { limit });
            }
            let tick = iterations.len();
            let iteration = match self.packing {
                PrefillPacking::StallFree => {
                    self.stall_free_iteration(tick, &mut states, &mut traces)?
                }
                PrefillPacking::Exclusive => {
                    self.exclusive_iteration(tick, &mut states, &mut traces)?
                }
            };
            iterations.push(iteration);
        }

        let summary = summarize(&iterations, &traces);
        Ok(PrefillSchedule {
            budget: self.budget,
            packing: self.packing,
            iterations,
            sequences: traces,
            stats: summary,
        })
    }

    /// One `Sarathi-Serve` iteration: decodes first and in full, then the single
    /// prefill chunk that fits in what they left behind.
    fn stall_free_iteration(
        &self,
        tick: usize,
        states: &mut [SequenceState],
        traces: &mut [PrefillSequenceTrace],
    ) -> ChunkedPrefillResult<PrefillIteration> {
        // Decodes have priority. A decode that is not scheduled *is* the stall
        // this whole module exists to prevent, so they are taken first, in full,
        // and never trimmed to make room.
        let decodes = self.admit_decodes(states)?;
        let decode_tokens = decodes.len();

        // `decode_tokens <= max_decode_slots < total`, so the residual is at
        // least `prefill_reserve >= 1`: the chunk always makes progress, and the
        // prefill always terminates.
        let residual = self.budget.total() - decode_tokens;

        let mut cells = 0u64;
        let prefill = Self::take_chunk(states, traces, residual, tick, &mut cells);
        let prefill_tokens = prefill.map_or(0, |assignment| assignment.chunk().token_count());

        cells += run_decodes(&decodes, states, traces, tick);

        Ok(PrefillIteration {
            tick,
            prefill,
            decodes,
            prefill_tokens,
            decode_tokens,
            attention_cells: cells,
        })
    }

    /// One classical iteration: a whole prompt, alone and unchunked, if any prompt
    /// is waiting; otherwise every decode. This is the stall, reproduced faithfully
    /// so that it can be measured rather than asserted.
    fn exclusive_iteration(
        &self,
        tick: usize,
        states: &mut [SequenceState],
        traces: &mut [PrefillSequenceTrace],
    ) -> ChunkedPrefillResult<PrefillIteration> {
        let mut cells = 0u64;

        // Prefill takes priority, and takes the whole iteration: an unbounded
        // residual means the chunk is the entire prompt.
        if let Some(prefill) = Self::take_chunk(states, traces, usize::MAX, tick, &mut cells) {
            return Ok(PrefillIteration {
                tick,
                prefill: Some(prefill),
                decodes: Vec::new(),
                prefill_tokens: prefill.chunk().token_count(),
                decode_tokens: 0,
                attention_cells: cells,
            });
        }

        let decodes = self.admit_decodes(states)?;
        let decode_tokens = decodes.len();
        cells += run_decodes(&decodes, states, traces, tick);

        Ok(PrefillIteration {
            tick,
            prefill: None,
            decodes,
            prefill_tokens: 0,
            decode_tokens,
            attention_cells: cells,
        })
    }

    /// Every sequence currently in the decode set — or an error, if there are
    /// more of them than the budget reserved room for.
    fn admit_decodes(&self, states: &[SequenceState]) -> ChunkedPrefillResult<Vec<usize>> {
        let decodes: Vec<usize> = states
            .iter()
            .enumerate()
            .filter(|(_, state)| state.is_decoding())
            .map(|(index, _)| index)
            .collect();
        if decodes.len() > self.budget.max_decode_slots() {
            return Err(ChunkedPrefillError::DecodeCapExceeded {
                running: decodes.len(),
                cap: self.budget.max_decode_slots(),
            });
        }
        Ok(decodes)
    }

    /// Give the first sequence with prompt left a chunk of at most `residual`
    /// tokens, advance it, and emit its first generated token if this chunk
    /// finished the prompt.
    ///
    /// An associated function, not a method: which sequence to prefill and how
    /// large a chunk it gets depend only on the sequence states and the residual
    /// the caller already computed from the budget, never on the scheduler.
    fn take_chunk(
        states: &mut [SequenceState],
        traces: &mut [PrefillSequenceTrace],
        residual: usize,
        tick: usize,
        cells: &mut u64,
    ) -> Option<PrefillIterationChunk> {
        if residual == 0 {
            return None;
        }
        let sequence = states.iter().position(SequenceState::needs_prefill)?;
        let state = &mut states[sequence];
        let trace = &mut traces[sequence];

        let start = state.prefilled;
        let tokens = state.prompt_remaining().min(residual);
        let ordinal = trace.chunk_lengths.len();

        // Cells: query `start + j` attends to the `context + j + 1` keys at or
        // below its own position.
        let committed = widen(state.context);
        let width = widen(tokens);
        *cells += width * committed + width * (width + 1) / 2;

        state.prefilled += tokens;
        state.context += tokens;
        trace.chunk_lengths.push(tokens);

        // The chunk that finishes the prompt also produces the first generated
        // token: its last row *is* that token's logits. It costs no budget and
        // waits no iteration.
        if state.prefilled == state.prompt_tokens && state.output_tokens > 0 {
            state.emitted = 1;
            trace.token_ticks.push(tick);
        }

        Some(PrefillIterationChunk {
            sequence,
            chunk: PrefillChunk::at(ordinal, start, tokens),
        })
    }
}

// ── Simulation state ─────────────────────────────────────────────────────────

/// One sequence, mid-flight.
#[derive(Debug, Clone, Copy)]
struct SequenceState {
    prompt_tokens: usize,
    output_tokens: usize,
    /// Prompt tokens prefilled so far.
    prefilled: usize,
    /// Generated tokens emitted so far. The final prefill chunk emits the first.
    emitted: usize,
    /// KV slots this sequence holds: every prefilled prompt token, plus every
    /// generated token that has been fed back in.
    context: usize,
}

impl SequenceState {
    const fn new(spec: PrefillSequenceSpec) -> Self {
        Self {
            prompt_tokens: spec.prompt_tokens(),
            output_tokens: spec.output_tokens(),
            prefilled: 0,
            emitted: 0,
            context: 0,
        }
    }

    const fn prompt_remaining(&self) -> usize {
        self.prompt_tokens - self.prefilled
    }

    const fn needs_prefill(&self) -> bool {
        self.prefilled < self.prompt_tokens
    }

    /// In the decode set: prefill finished, first token already out, more to
    /// come.
    const fn is_decoding(&self) -> bool {
        self.prefilled == self.prompt_tokens
            && self.emitted >= 1
            && self.emitted < self.output_tokens
    }

    const fn is_done(&self) -> bool {
        self.prefilled == self.prompt_tokens && self.emitted >= self.output_tokens
    }
}

// ── Helpers ──────────────────────────────────────────────────────────────────

/// Advance every decoding sequence by one token, and return the `Q·K` cells it
/// cost.
///
/// The decode token's own key lands in the cache before it attends, so it sees
/// `context + 1` keys — itself included.
fn run_decodes(
    decodes: &[usize],
    states: &mut [SequenceState],
    traces: &mut [PrefillSequenceTrace],
    tick: usize,
) -> u64 {
    let mut cells = 0u64;
    for &sequence in decodes {
        let state = &mut states[sequence];
        state.context += 1;
        cells += widen(state.context);
        state.emitted += 1;
        traces[sequence].token_ticks.push(tick);
    }
    cells
}

/// Fold the iterations and traces into the schedule's statistics.
///
/// The two time-between-token figures are computed from a prefix sum over the
/// iterations' token costs, so "the tokens a sequence waited through" is an exact
/// sum, not an estimate.
fn summarize(
    iterations: &[PrefillIteration],
    traces: &[PrefillSequenceTrace],
) -> PrefillStallStats {
    // `cumulative[i]` = tokens processed in iterations `0 ..= i`.
    let mut cumulative: Vec<u64> = Vec::with_capacity(iterations.len());
    let mut running = 0u64;
    for iteration in iterations {
        running += widen(iteration.total_tokens());
        cumulative.push(running);
    }
    let span = |from: usize, to: usize| -> u64 {
        let upper = cumulative.get(to).copied().unwrap_or(0);
        let lower = if from == 0 {
            0
        } else {
            cumulative.get(from - 1).copied().unwrap_or(0)
        };
        upper - lower
    };

    let mut max_time_between_tokens = 0u64;
    let mut max_time_to_first_token = 0u64;
    let mut max_token_gap_ticks = 0usize;
    for trace in traces {
        if let Some(first) = trace.first_token_tick() {
            max_time_to_first_token = max_time_to_first_token.max(span(0, first));
        }
        for pair in trace.token_ticks.windows(2) {
            max_token_gap_ticks = max_token_gap_ticks.max(pair[1] - pair[0]);
            max_time_between_tokens = max_time_between_tokens.max(span(pair[0] + 1, pair[1]));
        }
    }

    PrefillStallStats {
        iterations: iterations.len(),
        total_tokens: running,
        max_iteration_tokens: iterations
            .iter()
            .map(PrefillIteration::total_tokens)
            .max()
            .unwrap_or(0),
        max_iteration_attention_cells: iterations
            .iter()
            .map(PrefillIteration::attention_cells)
            .max()
            .unwrap_or(0),
        max_token_gap_ticks,
        max_time_between_tokens,
        max_time_to_first_token,
        peak_decode_sequences: iterations
            .iter()
            .map(|iteration| iteration.decodes.len())
            .max()
            .unwrap_or(0),
    }
}
