//! The iteration-level batch loop: admit, decode **one token**, preempt, retire —
//! and then do it again.
//!
//! # What "iteration-level" actually means
//!
//! A static batch is formed once and dissolved once. Eight requests go in, and
//! the batch runs until the *longest* of them is done; the seven that finished
//! earlier keep their slots, and their KV, and produce nothing. The batch is only
//! as fast as its slowest member, and the memory is only as free as its longest.
//!
//! An iteration-level batch is re-formed **every decode step**. A sequence that
//! finishes at step 40 releases its blocks at step 40, and a waiting request is
//! admitted into the room it left at step 41 — not when the last member of some
//! cohort finally stops. The batch is not a cohort at all; it is whoever happens
//! to be running this iteration.
//!
//! That is the whole idea, and it is measurable: the module's throughput test runs
//! the same eight requests, with wildly unequal lengths, through the same pool,
//! and compares the slot-occupancy of a static cohort against this loop. See the
//! module-level docs of [`crate::continuous_batching`] for the measured numbers.
//!
//! # The loop, in order, and why that order
//!
//! 1. **Decode** every running sequence by one token, preempting to make room if
//!    the pool cannot cover the step.
//! 2. **Retire** every sequence that just reached its target length, releasing its
//!    blocks *immediately* — not at the end of some cohort.
//! 3. **Admit** waiting and swapped-out sequences into whatever room now exists.
//!
//! Decode comes before admission, and that is a deliberate, load-bearing choice:
//! an already-admitted sequence must be able to make progress, or the engine can
//! livelock (admit A, preempt B to fit it, admit B, preempt A to fit it, forever,
//! decoding nothing). Because admission never preempts a running sequence, and
//! decode only preempts to let *other running sequences* proceed, every step in
//! which anything is running decodes at least one token. Progress is structural,
//! not lucky.
//!
//! # This module owns memory, not time
//!
//! Victims are chosen **last-admitted-first**, which is the policy that preserves
//! the progress of the oldest sequences (the ones closest to finishing, and the
//! ones that would cost the most to recompute). Admission is FIFO. There is
//! deliberately **no** priority, no deadline, no fairness weighting and no
//! preemptive reordering here: *where the bytes live* is this module's problem,
//! and *whose turn it is* belongs to `request_scheduling`. Wiring a policy from
//! there into the admission queue here is a composition, not a rewrite.

use std::collections::VecDeque;

use super::allocator::{KvBlockAllocator, PREFIX_HASH_SEED, fold_prefix_hash};
use super::table::{KvBlockSwapSlab, KvBlockTable};
use super::types::{
    BatchPreemption, BatchPreemptionMode, BatchSequenceId, BatchSequenceState, BatchStep,
    ContinuousBatchConfig, ContinuousBatchError, ContinuousBatchResult, ContinuousBatchStats,
    KvBlockId,
};

// ── BatchKvProducer ──────────────────────────────────────────────────────────

/// The source of the key/value bytes a decode step appends: a model's forward
/// pass, in the only two properties this module actually depends on.
///
/// # The contract, and why each half of it is load-bearing
///
/// > `produce_kv(context, position, ..)` must be **deterministic**: the same
/// > `context` at the same `position` must yield the same bytes, every time.
///
/// `context` is the sequence's token ids **up to and including** the token being
/// produced, so the contract is exactly the one a real transformer already
/// satisfies — a token's keys and values are a function of the tokens before it
/// (through the attention stack) and of its position (through the positional
/// encoding), and of nothing else. Nothing weaker would do, and nothing stronger
/// is needed:
///
/// - **Prefix sharing** is valid *because* of this contract. Two requests that
///   begin with the same tokens have, by determinism, the same KV for those
///   tokens — so one may read the other's blocks. Weaken the contract to "a
///   function of the token id alone" and the sharing would still work but would
///   no longer model anything real; weaken it to "may depend on hidden state" and
///   the sharing becomes *wrong*, silently.
/// - **Preempt-by-recompute** is lossless *because* of this contract. Re-driving
///   the producer over the same contexts at the same positions reproduces the
///   evicted KV bit-for-bit. That is not an approximation the module tolerates;
///   it is an equality the module's tests assert.
///
/// This module never inspects the bytes. It does not know or care what is in
/// them — it only ever moves them, shares them, and (on request) hands them back
/// unchanged. Everything above is about *when it is allowed to skip calling you*.
///
/// # This is not a sampler
///
/// The engine is told each sequence's token stream up front (see
/// [`ContinuousBatchEngine::submit`]) and never chooses a token. In a real server
/// the next token id comes from the model's sampler at each step, and the engine
/// would consume it there; that loop is orthogonal to block management, and
/// pretending to sample here would be pretending to be a model.
pub trait BatchKvProducer {
    /// Fill `keys` and `values` — each exactly
    /// [`ContinuousBatchConfig::token_buffer_len`] elements, laid out
    /// `[layer][head][dim]` — with the KV of `context`'s **last** token, which
    /// sits at absolute stream position `position`.
    ///
    /// # Errors
    ///
    /// Implementations may fail; the engine propagates the error and leaves the
    /// sequence's block table exactly as it was before the step.
    fn produce_kv(
        &self,
        context: &[u32],
        position: usize,
        keys: &mut [f32],
        values: &mut [f32],
    ) -> ContinuousBatchResult<()>;
}

// ── BatchSequence ────────────────────────────────────────────────────────────

/// One request in the engine: its tokens, its lifecycle state, and its block
/// table.
#[derive(Debug, Clone)]
pub struct BatchSequence {
    id: BatchSequenceId,
    token_ids: Vec<u32>,
    prompt_len: usize,
    state: BatchSequenceState,
    table: KvBlockTable,
    swap_slab: Option<KvBlockSwapSlab>,
    /// Absolute positions this sequence held when it was recompute-preempted, to
    /// be re-materialised on resume. Empty for a sequence that has never been
    /// preempted.
    restore_positions: Vec<usize>,
    /// The monotone position counter as of the preemption.
    restore_next_position: usize,
    /// The rolling prefix hash, folded up to `hashed_blocks`.
    rolling_hash: u64,
    /// How many of this sequence's full blocks have been offered to the prefix
    /// index.
    hashed_blocks: usize,
    preemptions: u32,
}

impl BatchSequence {
    /// This sequence's id.
    #[must_use]
    pub const fn id(&self) -> BatchSequenceId {
        self.id
    }

    /// The full token stream: prompt, then the tokens it will decode.
    #[must_use]
    pub fn token_ids(&self) -> &[u32] {
        &self.token_ids
    }

    /// Prompt tokens (materialised in one shot at admission).
    #[must_use]
    pub const fn prompt_len(&self) -> usize {
        self.prompt_len
    }

    /// Tokens in the whole request: the prompt plus everything it will decode.
    #[must_use]
    pub fn target_len(&self) -> usize {
        self.token_ids.len()
    }

    /// How far the sequence has got: the number of tokens whose KV it has ever
    /// materialised.
    ///
    /// This is the sequence's *progress*, and it is deliberately the block
    /// table's monotone position counter rather than its length — a compaction
    /// shortens the table without un-decoding anything, and a sequence that
    /// dropped half its KV to an eviction policy has not gone backwards.
    #[must_use]
    pub const fn progress(&self) -> usize {
        self.table.next_position()
    }

    /// Tokens decoded beyond the prompt.
    #[must_use]
    pub fn generated(&self) -> usize {
        self.progress().saturating_sub(self.prompt_len)
    }

    /// Whether the sequence has reached its target length.
    #[must_use]
    pub fn is_complete(&self) -> bool {
        self.progress() >= self.token_ids.len()
    }

    /// Where the sequence sits in the lifecycle.
    #[must_use]
    pub const fn state(&self) -> BatchSequenceState {
        self.state
    }

    /// The block table. Empty unless the sequence is running.
    #[must_use]
    pub const fn table(&self) -> &KvBlockTable {
        &self.table
    }

    /// Times this sequence has been evicted.
    #[must_use]
    pub const fn preemptions(&self) -> u32 {
        self.preemptions
    }

    /// The swap slab holding this sequence's KV, if it is swapped out.
    #[must_use]
    pub const fn swap_slab(&self) -> Option<&KvBlockSwapSlab> {
        self.swap_slab.as_ref()
    }
}

// ── The prefill plan ─────────────────────────────────────────────────────────

/// What admitting one waiting sequence will cost, worked out **before** a single
/// block is touched.
///
/// Admission is all-or-nothing, so the cost has to be known in advance — and the
/// cost is not simply `ceil(tokens / block_size)`, for two reasons that both
/// bite:
///
/// - a block adopted from the prefix index costs **nothing** if some live
///   sequence already holds it, but costs **a free block** if it is merely
///   cached (adopting revives it out of the free list);
/// - the tail block of the prompt is never shared, so it is always fresh.
#[derive(Debug, Clone, Default)]
struct PrefillPlan {
    /// Full blocks to adopt from the prefix index, in logical order.
    shared: Vec<KvBlockId>,
    /// The first token index whose KV must actually be computed.
    compute_from: usize,
    /// Free blocks this admission will consume: the fresh blocks the computed
    /// tokens need, plus the shared blocks that are currently sitting in the free
    /// list.
    blocks_required: usize,
}

// ── ContinuousBatchEngine ────────────────────────────────────────────────────

/// The iteration-level loop over a paged KV pool.
///
/// See the module docs for the order of the phases and why it is that order.
#[derive(Debug)]
pub struct ContinuousBatchEngine {
    config: ContinuousBatchConfig,
    allocator: KvBlockAllocator,
    sequences: Vec<BatchSequence>,
    waiting: VecDeque<BatchSequenceId>,
    running: Vec<BatchSequenceId>,
    swapped: VecDeque<BatchSequenceId>,
    step_index: u64,
    stats: ContinuousBatchStats,
}

impl ContinuousBatchEngine {
    /// Build an engine over a fresh pool of `config.total_blocks` blocks.
    ///
    /// # Errors
    ///
    /// Whatever [`ContinuousBatchConfig::validate`] rejects.
    pub fn new(config: ContinuousBatchConfig) -> ContinuousBatchResult<Self> {
        let allocator = KvBlockAllocator::new(config.clone())?;
        Ok(Self {
            config,
            allocator,
            sequences: Vec::new(),
            waiting: VecDeque::new(),
            running: Vec::new(),
            swapped: VecDeque::new(),
            step_index: 0,
            stats: ContinuousBatchStats::default(),
        })
    }

    /// Queue a request.
    ///
    /// `token_ids` is the **whole** stream: `token_ids[..prompt_len]` is the
    /// prompt, materialised in one shot when the sequence is admitted, and
    /// `token_ids[prompt_len..]` are the tokens it will decode, one per step. See
    /// [`BatchKvProducer`] on why the engine is told them rather than sampling
    /// them.
    ///
    /// # Errors
    ///
    /// [`ContinuousBatchError::InvalidSelection`] if `prompt_len` is zero or
    /// longer than `token_ids`, or if the prompt cannot fit in the pool at all
    /// (in which case the request could never be admitted, and saying so now is
    /// better than a deadlocked queue).
    pub fn submit(
        &mut self,
        token_ids: Vec<u32>,
        prompt_len: usize,
    ) -> ContinuousBatchResult<BatchSequenceId> {
        if prompt_len == 0 {
            return Err(ContinuousBatchError::InvalidSelection {
                reason: "a sequence must have at least one prompt token".to_string(),
            });
        }
        if prompt_len > token_ids.len() {
            return Err(ContinuousBatchError::InvalidSelection {
                reason: format!(
                    "prompt_len {prompt_len} exceeds the {}-token stream",
                    token_ids.len()
                ),
            });
        }
        let prompt_blocks = self.config.blocks_for_tokens(prompt_len);
        if prompt_blocks > self.config.total_blocks {
            return Err(ContinuousBatchError::InvalidSelection {
                reason: format!(
                    "a {prompt_len}-token prompt needs {prompt_blocks} block(s), more than the pool's {}",
                    self.config.total_blocks
                ),
            });
        }

        let id = BatchSequenceId(self.sequences.len());
        self.sequences.push(BatchSequence {
            id,
            token_ids,
            prompt_len,
            state: BatchSequenceState::Waiting,
            table: KvBlockTable::new(&self.config),
            swap_slab: None,
            restore_positions: Vec::new(),
            restore_next_position: 0,
            rolling_hash: PREFIX_HASH_SEED,
            hashed_blocks: 0,
            preemptions: 0,
        });
        self.waiting.push_back(id);
        Ok(id)
    }

    /// Fork a running sequence: the child shares **every** block of the parent,
    /// copy-on-write, and starts running immediately.
    ///
    /// This is parallel sampling (and beam search): `n` continuations of one
    /// prompt cost one copy of the prompt's KV, not `n`. The child's blocks are
    /// the parent's blocks — no bytes move — and the first token either side
    /// decodes copies exactly one block, the shared partially filled tail.
    ///
    /// `token_ids` must begin with the parent's tokens up to the parent's current
    /// progress: the child inherits that KV, so claiming a different prefix would
    /// be claiming KV that was never computed for it.
    ///
    /// # Errors
    ///
    /// - [`ContinuousBatchError::UnknownSequence`] if `parent` is not a sequence
    ///   of this engine.
    /// - [`ContinuousBatchError::InvalidState`] if the parent is not running.
    /// - [`ContinuousBatchError::PrefixMismatch`] if `token_ids` does not begin
    ///   with the parent's decoded tokens, or is no longer than them.
    pub fn fork_sequence(
        &mut self,
        parent: BatchSequenceId,
        token_ids: Vec<u32>,
    ) -> ContinuousBatchResult<BatchSequenceId> {
        let parent_sequence = self.sequence(parent)?;
        if parent_sequence.state != BatchSequenceState::Running {
            return Err(ContinuousBatchError::InvalidState {
                sequence: parent,
                state: parent_sequence.state,
                operation: "fork_sequence",
            });
        }
        let progress = parent_sequence.progress();
        if token_ids.len() <= progress {
            return Err(ContinuousBatchError::PrefixMismatch {
                reason: format!(
                    "a fork of a {progress}-token sequence must have more than {progress} token(s), got {}",
                    token_ids.len()
                ),
            });
        }
        if token_ids.get(..progress) != parent_sequence.token_ids.get(..progress) {
            return Err(ContinuousBatchError::PrefixMismatch {
                reason: "a fork must begin with the parent's decoded tokens: the child inherits \
                         their KV, and it was computed for those tokens"
                    .to_string(),
            });
        }

        let prompt_len = parent_sequence.prompt_len;
        let rolling_hash = parent_sequence.rolling_hash;
        let hashed_blocks = parent_sequence.hashed_blocks;
        // Clone the parent's table (a list of block ids and a few scalars — no KV
        // bytes) to end the immutable borrow of `self.sequences`, then fork the
        // clone, which takes the copy-on-write references in `self.allocator`.
        let parent_table = parent_sequence.table.clone();
        let table = parent_table.fork(&mut self.allocator)?;

        let id = BatchSequenceId(self.sequences.len());
        self.sequences.push(BatchSequence {
            id,
            token_ids,
            prompt_len,
            state: BatchSequenceState::Running,
            table,
            swap_slab: None,
            restore_positions: Vec::new(),
            restore_next_position: 0,
            rolling_hash,
            hashed_blocks,
            preemptions: 0,
        });
        self.running.push(id);
        Ok(id)
    }

    // ── Accessors ────────────────────────────────────────────────────────────

    /// The configuration.
    #[must_use]
    pub const fn config(&self) -> &ContinuousBatchConfig {
        &self.config
    }

    /// The block pool.
    #[must_use]
    pub const fn allocator(&self) -> &KvBlockAllocator {
        &self.allocator
    }

    /// Cumulative counters.
    #[must_use]
    pub const fn stats(&self) -> &ContinuousBatchStats {
        &self.stats
    }

    /// Every submitted sequence, in submission order.
    #[must_use]
    pub fn sequences(&self) -> &[BatchSequence] {
        &self.sequences
    }

    /// One sequence.
    ///
    /// # Errors
    ///
    /// [`ContinuousBatchError::UnknownSequence`] if the id is not this engine's.
    pub fn sequence(&self, id: BatchSequenceId) -> ContinuousBatchResult<&BatchSequence> {
        self.sequences
            .get(id.index())
            .ok_or(ContinuousBatchError::UnknownSequence {
                sequence: id,
                total: self.sequences.len(),
            })
    }

    /// The sequences decoding right now, in admission order.
    #[must_use]
    pub fn running(&self) -> &[BatchSequenceId] {
        &self.running
    }

    /// Sequences queued for admission.
    #[must_use]
    pub fn waiting_len(&self) -> usize {
        self.waiting.len()
    }

    /// Sequences evicted to the swap slab.
    #[must_use]
    pub fn swapped_len(&self) -> usize {
        self.swapped.len()
    }

    /// Whether every submitted sequence has finished.
    #[must_use]
    pub fn is_idle(&self) -> bool {
        self.running.is_empty() && self.waiting.is_empty() && self.swapped.is_empty()
    }

    // ── The loop ─────────────────────────────────────────────────────────────

    /// Run one iteration: decode, retire, admit.
    ///
    /// # Errors
    ///
    /// - [`ContinuousBatchError::OutOfBlocks`] if the pool cannot cover even a
    ///   single running sequence's next token with nothing left to evict. That is
    ///   a genuine out-of-memory: the pool is smaller than the shortest sequence
    ///   that must run.
    /// - Whatever the [`BatchKvProducer`] returns.
    pub fn step(&mut self, producer: &dyn BatchKvProducer) -> ContinuousBatchResult<BatchStep> {
        self.step_index += 1;
        let mut record = BatchStep {
            step: self.step_index,
            ..BatchStep::default()
        };

        self.decode_phase(producer, &mut record)?;
        self.retire_phase(&mut record)?;
        self.admit_phase(producer, &mut record)?;

        record.blocks_in_use = self.allocator.allocated_blocks();
        record.free_blocks = self.allocator.free_blocks();
        record.running = self.running.len();
        record.waiting = self.waiting.len();
        record.swapped = self.swapped.len();

        self.stats.steps += 1;
        self.stats.tokens_decoded += record.tokens_decoded as u64;
        self.stats.prefill_tokens += record.prefill_tokens as u64;
        self.stats.prefix_cached_tokens += record.prefix_cached_tokens as u64;
        self.stats.recomputed_tokens += record.recomputed_tokens as u64;
        self.stats.cow_copies = self.allocator.cow_copies();
        self.stats.peak_blocks_in_use = self.stats.peak_blocks_in_use.max(record.blocks_in_use);

        Ok(record)
    }

    /// Step until every sequence has finished, or until `max_steps` have run.
    ///
    /// # Errors
    ///
    /// Whatever [`Self::step`] returns, plus
    /// [`ContinuousBatchError::InvariantViolated`] if the loop hits `max_steps`
    /// with work outstanding — which would mean the engine failed to make
    /// progress, and is a bug in *this* module, not bad input.
    pub fn run_until_idle(
        &mut self,
        producer: &dyn BatchKvProducer,
        max_steps: usize,
    ) -> ContinuousBatchResult<Vec<BatchStep>> {
        let mut steps = Vec::new();
        for _ in 0..max_steps {
            if self.is_idle() {
                return Ok(steps);
            }
            steps.push(self.step(producer)?);
        }
        if self.is_idle() {
            return Ok(steps);
        }
        Err(ContinuousBatchError::InvariantViolated {
            reason: format!(
                "still {} running / {} waiting / {} swapped after {max_steps} step(s): the loop is not making progress",
                self.running.len(),
                self.waiting.len(),
                self.swapped.len()
            ),
        })
    }

    // ── Phase 1: decode ──────────────────────────────────────────────────────

    fn decode_phase(
        &mut self,
        producer: &dyn BatchKvProducer,
        record: &mut BatchStep,
    ) -> ContinuousBatchResult<()> {
        // Make room. Each iteration removes one sequence from the running set, so
        // the loop terminates; it exits either because the pool can cover the step
        // or because there is nobody left to evict.
        loop {
            let needed = self.blocks_needed_for_decode()?;
            if self.allocator.free_blocks() >= needed {
                break;
            }
            // Evicting the *only* running sequence would leave nothing to decode
            // and would cost more to bring back than it freed. If a single
            // sequence cannot get one more block out of an entire pool, the pool
            // is structurally too small for it.
            if self.running.len() <= 1 {
                return Err(ContinuousBatchError::OutOfBlocks {
                    requested: needed,
                    free: self.allocator.free_blocks(),
                    total: self.config.total_blocks,
                });
            }
            let Some(&victim) = self.running.last() else {
                return Err(ContinuousBatchError::OutOfBlocks {
                    requested: needed,
                    free: self.allocator.free_blocks(),
                    total: self.config.total_blocks,
                });
            };
            self.preempt(victim, record)?;
        }

        let token_buffer_len = self.config.token_buffer_len();
        let mut keys = vec![0.0f32; token_buffer_len];
        let mut values = vec![0.0f32; token_buffer_len];

        let running = self.running.clone();
        let Self {
            sequences,
            allocator,
            ..
        } = self;
        for id in running {
            let Some(sequence) = sequences.get_mut(id.index()) else {
                continue;
            };
            if sequence.is_complete() {
                // Reached its target in an earlier step's prefill; it retires
                // below rather than decoding a token it does not have.
                continue;
            }
            let position = sequence.progress();
            materialize_token(
                allocator,
                sequence,
                producer,
                position,
                &mut keys,
                &mut values,
            )?;
            publish_filled_blocks(allocator, sequence)?;
            record.decoded.push(id);
            record.tokens_decoded += 1;
        }
        Ok(())
    }

    /// Free blocks this step's decode will consume.
    ///
    /// A running sequence takes a block when its tail block is **full** (it grows
    /// by one) *or* when its tail block is **shared** (a copy-on-write needs
    /// somewhere to put the copy). Missing the second case is the classic way to
    /// run out of memory halfway through a step that was declared affordable.
    fn blocks_needed_for_decode(&self) -> ContinuousBatchResult<usize> {
        let mut needed = 0;
        for &id in &self.running {
            let Some(sequence) = self.sequences.get(id.index()) else {
                continue;
            };
            if sequence.is_complete() {
                continue;
            }
            match sequence.table.block_ids().last() {
                None => needed += 1,
                Some(&tail) => {
                    let block =
                        self.allocator
                            .block(tail)
                            .ok_or(ContinuousBatchError::UnknownBlock {
                                block: tail,
                                total: self.config.total_blocks,
                            })?;
                    if block.is_full() || block.is_shared() {
                        needed += 1;
                    }
                }
            }
        }
        Ok(needed)
    }

    /// Evict one sequence, by whichever mode the config names.
    fn preempt(
        &mut self,
        victim: BatchSequenceId,
        record: &mut BatchStep,
    ) -> ContinuousBatchResult<()> {
        let mode = self.config.preemption_mode;
        self.running.retain(|&id| id != victim);

        let Self {
            sequences,
            allocator,
            waiting,
            swapped,
            stats,
            ..
        } = self;
        let sequence =
            sequences
                .get_mut(victim.index())
                .ok_or(ContinuousBatchError::UnknownSequence {
                    sequence: victim,
                    total: 0,
                })?;

        let tokens_dropped = sequence.table.len();
        let blocks_released = sequence.table.num_blocks();
        let free_before = allocator.free_blocks();

        match mode {
            BatchPreemptionMode::Recompute => {
                sequence.restore_positions = sequence.table.positions(allocator)?;
                sequence.restore_next_position = sequence.table.next_position();
                sequence.table.release(allocator)?;
                // The table's blocks are gone, so the hashes it published belong
                // to blocks it no longer holds. Start the chain again; the resume
                // will re-publish (or, if the old blocks are still cached and
                // un-recycled, simply adopt them straight back).
                sequence.rolling_hash = PREFIX_HASH_SEED;
                sequence.hashed_blocks = 0;
                sequence.state = BatchSequenceState::Waiting;
                // To the *front*: it has already queued once, and sending it to
                // the back would let a fresh request overtake a sequence that is
                // half-finished.
                waiting.push_front(victim);
                stats.preemptions_recompute += 1;
            }
            BatchPreemptionMode::Swap => {
                let slab = sequence.table.swap_out(allocator)?;
                stats.swapped_out_blocks += slab.num_blocks() as u64;
                sequence.swap_slab = Some(slab);
                sequence.rolling_hash = PREFIX_HASH_SEED;
                sequence.hashed_blocks = 0;
                sequence.state = BatchSequenceState::Swapped;
                swapped.push_front(victim);
                stats.preemptions_swap += 1;
            }
        }

        sequence.preemptions += 1;
        let blocks_freed = allocator.free_blocks() - free_before;
        record.preempted.push(BatchPreemption {
            sequence: victim,
            mode,
            tokens_dropped,
            blocks_released,
            blocks_freed,
        });
        Ok(())
    }

    // ── Phase 2: retire ──────────────────────────────────────────────────────

    fn retire_phase(&mut self, record: &mut BatchStep) -> ContinuousBatchResult<()> {
        let mut finished = Vec::new();
        {
            let sequences = &self.sequences;
            self.running.retain(|&id| {
                let done = sequences
                    .get(id.index())
                    .is_some_and(BatchSequence::is_complete);
                if done {
                    finished.push(id);
                }
                !done
            });
        }

        let Self {
            sequences,
            allocator,
            ..
        } = self;
        for &id in &finished {
            let Some(sequence) = sequences.get_mut(id.index()) else {
                continue;
            };
            // Released *now*, at the step the sequence finished — not when the
            // slowest member of some cohort finally stops. This one line is what
            // continuous batching is.
            sequence.table.release(allocator)?;
            sequence.state = BatchSequenceState::Finished;
        }
        record.finished = finished;
        Ok(())
    }

    // ── Phase 3: admit ───────────────────────────────────────────────────────

    fn admit_phase(
        &mut self,
        producer: &dyn BatchKvProducer,
        record: &mut BatchStep,
    ) -> ContinuousBatchResult<()> {
        loop {
            if self.running.len() >= self.config.max_running_sequences {
                return Ok(());
            }

            // Swapped-out sequences go first: their KV is already computed and is
            // sitting off-pool doing nothing for anybody. Bringing one back costs
            // a copy; leaving it there costs the memory *and* the progress.
            if let Some(&id) = self.swapped.front() {
                let blocks = self
                    .sequence(id)?
                    .swap_slab
                    .as_ref()
                    .map_or(0, KvBlockSwapSlab::num_blocks);
                if self.allocator.free_blocks() < blocks + self.config.watermark_blocks {
                    return Ok(());
                }
                self.swapped.pop_front();
                self.admit_swapped(id)?;
                record.admitted.push(id);
                continue;
            }

            let Some(&id) = self.waiting.front() else {
                return Ok(());
            };
            let plan = self.plan_prefill(id)?;
            if self.allocator.free_blocks() < plan.blocks_required + self.config.watermark_blocks {
                // Not enough room. Stop — admission never preempts a running
                // sequence to make space for a waiting one, because that is the
                // livelock.
                return Ok(());
            }
            self.waiting.pop_front();
            self.admit_waiting(id, &plan, producer, record)?;
            record.admitted.push(id);
        }
    }

    /// Work out what admitting `id` will cost, without touching a block.
    fn plan_prefill(&mut self, id: BatchSequenceId) -> ContinuousBatchResult<PrefillPlan> {
        let sequence = self.sequence(id)?;
        let restore_len = if sequence.restore_positions.is_empty() {
            sequence.prompt_len
        } else {
            sequence.restore_positions.len()
        };
        let block_size = self.config.block_size;

        // A gappy sequence (one that was compacted before it was preempted) is
        // not a dense prefix of anything, so no block of it can be shared and it
        // must be rebuilt slot by slot.
        let is_dense = sequence
            .restore_positions
            .iter()
            .enumerate()
            .all(|(slot, &position)| slot == position);
        if !is_dense {
            return Ok(PrefillPlan {
                shared: Vec::new(),
                compute_from: 0,
                blocks_required: restore_len.div_ceil(block_size),
            });
        }

        let mut plan = PrefillPlan {
            shared: Vec::new(),
            compute_from: 0,
            blocks_required: 0,
        };

        if self.config.enable_prefix_sharing {
            let mut rolling = PREFIX_HASH_SEED;
            let mut block_index = 0usize;
            // Only *full* blocks that lie entirely inside the tokens being
            // materialised can be shared. The tail is always fresh.
            while (block_index + 1) * block_size <= restore_len {
                let start = block_index * block_size;
                let end = start + block_size;
                let Some(tokens) = self.sequences[id.index()].token_ids.get(start..end) else {
                    break;
                };
                rolling = fold_prefix_hash(rolling, tokens);
                let Some(candidate) = self.allocator.lookup_prefix_block(rolling) else {
                    break;
                };
                // A hash hit is a candidate, never a decision. Verify the tokens
                // and the positions element by element; a mismatch means a 64-bit
                // collision, and the block is simply not used.
                let Some(block) = self.allocator.block(candidate) else {
                    break;
                };
                let positions_match = block
                    .positions()
                    .iter()
                    .enumerate()
                    .all(|(slot, &position)| position == start + slot);
                if !block.is_full() || block.tokens() != tokens || !positions_match {
                    break;
                }
                if block.ref_count() == 0 {
                    // Cached but free: adopting it takes it back out of the free
                    // list, so it costs a block.
                    plan.blocks_required += 1;
                }
                plan.shared.push(candidate);
                block_index += 1;
            }
            plan.compute_from = block_index * block_size;
        }

        let computed_tokens = restore_len - plan.compute_from;
        plan.blocks_required += computed_tokens.div_ceil(block_size);
        Ok(plan)
    }

    /// Prefill a waiting sequence: adopt its shared prefix blocks, compute the
    /// rest.
    fn admit_waiting(
        &mut self,
        id: BatchSequenceId,
        plan: &PrefillPlan,
        producer: &dyn BatchKvProducer,
        record: &mut BatchStep,
    ) -> ContinuousBatchResult<()> {
        let token_buffer_len = self.config.token_buffer_len();
        let block_size = self.config.block_size;
        let mut keys = vec![0.0f32; token_buffer_len];
        let mut values = vec![0.0f32; token_buffer_len];
        let config = self.config.clone();

        let Self {
            sequences,
            allocator,
            ..
        } = self;
        let sequence =
            sequences
                .get_mut(id.index())
                .ok_or(ContinuousBatchError::UnknownSequence {
                    sequence: id,
                    total: 0,
                })?;

        let is_resume = !sequence.restore_positions.is_empty();
        let restore_len = if is_resume {
            sequence.restore_positions.len()
        } else {
            sequence.prompt_len
        };
        let is_dense = sequence
            .restore_positions
            .iter()
            .enumerate()
            .all(|(slot, &position)| slot == position);

        if is_resume && !is_dense {
            // A compacted sequence: rebuild it slot by slot, at its original
            // absolute positions, with the counter where it left off.
            let positions = std::mem::take(&mut sequence.restore_positions);
            let next_position = sequence.restore_next_position;
            sequence.table = KvBlockTable::for_restore(&config, next_position);
            for &position in &positions {
                let token = sequence.token_ids.get(position).copied().ok_or_else(|| {
                    ContinuousBatchError::InvariantViolated {
                        reason: format!("no token id at position {position} to recompute"),
                    }
                })?;
                let context = sequence.token_ids.get(..=position).ok_or_else(|| {
                    ContinuousBatchError::InvariantViolated {
                        reason: format!("no context for position {position} to recompute"),
                    }
                })?;
                producer.produce_kv(context, position, &mut keys, &mut values)?;
                sequence
                    .table
                    .restore_token(allocator, token, position, &keys, &values)?;
            }
            record.recomputed_tokens += positions.len();
            sequence.state = BatchSequenceState::Running;
            self.running.push(id);
            return Ok(());
        }

        sequence.table = KvBlockTable::new(&config);
        sequence.rolling_hash = PREFIX_HASH_SEED;
        sequence.hashed_blocks = 0;

        // Adopt the shared prefix blocks. Not one key is computed for them.
        for (block_index, &block) in plan.shared.iter().enumerate() {
            let start = block_index * block_size;
            let end = start + block_size;
            let tokens = sequence.token_ids.get(start..end).ok_or_else(|| {
                ContinuousBatchError::InvariantViolated {
                    reason: format!(
                        "prefill plan names a block beyond the token stream at {start}"
                    ),
                }
            })?;
            let expected: Vec<u32> = tokens.to_vec();
            sequence
                .table
                .adopt_prefix_block(allocator, block, &expected)?;
            sequence.rolling_hash = fold_prefix_hash(sequence.rolling_hash, &expected);
            sequence.hashed_blocks += 1;
        }
        let shared_tokens = plan.shared.len() * block_size;

        // Compute the rest.
        for position in plan.compute_from..restore_len {
            materialize_token(
                allocator,
                sequence,
                producer,
                position,
                &mut keys,
                &mut values,
            )?;
        }
        publish_filled_blocks(allocator, sequence)?;

        let computed = restore_len - plan.compute_from;
        if is_resume {
            record.recomputed_tokens += computed;
        } else {
            record.prefill_tokens += computed;
        }
        record.prefix_cached_tokens += shared_tokens;

        sequence.restore_positions.clear();
        sequence.state = BatchSequenceState::Running;
        self.running.push(id);
        Ok(())
    }

    /// Bring a swapped-out sequence back: copy its bytes into whatever blocks are
    /// free now. Not a single key is recomputed.
    fn admit_swapped(&mut self, id: BatchSequenceId) -> ContinuousBatchResult<()> {
        let Self {
            sequences,
            allocator,
            stats,
            ..
        } = self;
        let sequence =
            sequences
                .get_mut(id.index())
                .ok_or(ContinuousBatchError::UnknownSequence {
                    sequence: id,
                    total: 0,
                })?;
        let slab = sequence
            .swap_slab
            .take()
            .ok_or(ContinuousBatchError::InvalidState {
                sequence: id,
                state: sequence.state,
                operation: "admit_swapped",
            })?;
        sequence.table = KvBlockTable::swap_in(allocator, &slab)?;
        stats.swapped_in_blocks += slab.num_blocks() as u64;
        // The restored blocks are private copies with fresh ids, so the sequence's
        // published-block chain starts again from nothing.
        sequence.rolling_hash = PREFIX_HASH_SEED;
        sequence.hashed_blocks = 0;
        sequence.state = BatchSequenceState::Running;
        self.running.push(id);
        Ok(())
    }
}

// ── Free helpers (kept free so the borrows stay disjoint) ────────────────────

/// Compute one token's KV and append it at the sequence's next position.
fn materialize_token(
    allocator: &mut KvBlockAllocator,
    sequence: &mut BatchSequence,
    producer: &dyn BatchKvProducer,
    position: usize,
    keys: &mut [f32],
    values: &mut [f32],
) -> ContinuousBatchResult<()> {
    if position != sequence.table.next_position() {
        return Err(ContinuousBatchError::InvariantViolated {
            reason: format!(
                "asked to materialise position {position} but the table's next position is {}",
                sequence.table.next_position()
            ),
        });
    }
    let token = sequence.token_ids.get(position).copied().ok_or_else(|| {
        ContinuousBatchError::InvalidSelection {
            reason: format!(
                "position {position} is beyond the {}-token stream",
                sequence.token_ids.len()
            ),
        }
    })?;
    let context = sequence.token_ids.get(..=position).ok_or_else(|| {
        ContinuousBatchError::InvalidSelection {
            reason: format!("no context for position {position}"),
        }
    })?;
    producer.produce_kv(context, position, keys, values)?;
    sequence
        .table
        .append_token(allocator, token, keys, values)?;
    Ok(())
}

/// Offer every newly filled block of a dense sequence to the prefix index.
///
/// Only dense sequences (`len == next_position`) are published: a compacted
/// sequence's blocks are not the KV of any contiguous token prefix, so nothing
/// could correctly adopt them.
fn publish_filled_blocks(
    allocator: &mut KvBlockAllocator,
    sequence: &mut BatchSequence,
) -> ContinuousBatchResult<()> {
    if !allocator.config().enable_prefix_sharing {
        return Ok(());
    }
    let block_size = allocator.config().block_size;
    if sequence.table.len() != sequence.table.next_position() {
        return Ok(());
    }
    while (sequence.hashed_blocks + 1) * block_size <= sequence.table.len() {
        let index = sequence.hashed_blocks;
        let start = index * block_size;
        let end = start + block_size;
        let Some(tokens) = sequence.token_ids.get(start..end) else {
            break;
        };
        let Some(&block) = sequence.table.block_ids().get(index) else {
            break;
        };
        sequence.rolling_hash = fold_prefix_hash(sequence.rolling_hash, tokens);
        allocator.publish_prefix_block(block, sequence.rolling_hash)?;
        sequence.hashed_blocks += 1;
    }
    Ok(())
}
