//! The execution backend the scheduler drives — and a deterministic,
//! table-driven implementation of it for tests, doctests, and validation.
//!
//! # Why this module defines its own executor trait
//!
//! Following the house precedent set by `replug` (state your own minimal
//! requirement rather than widen someone else's trait), the scheduler does not
//! reach into a batching backend or a language model. It owns **time and order**,
//! and the only thing it needs from whatever runs the tokens is: *given the ids I
//! selected to serve this tick, advance them one round and tell me what happened*.
//! That is the whole of [`SchedulerExecutor`].
//!
//! Crucially, the scheduler never asks the executor *where the bytes live* — no
//! `KV` blocks, no copy-on-write, no preemption of memory. Those belong to the
//! sibling `continuous_batching` module. The executor here reports only
//! **progress**: a token emitted, work remaining, completion. The scheduler turns
//! that into `TTFT`/`TPOT`/lateness measurements without ever touching the
//! backend's memory.
//!
//! # The fixture
//!
//! [`StaticSchedulerExecutor`] is the `replug` module's `ReplugStaticLanguageModel`
//! analogue: a table saying "request `r` needs `k` output tokens at decode cost
//! `c`". It does no inference; it counts ticks. Because its behaviour is a pure
//! function of that table, every scheduling claim — an earliest-deadline-first
//! schedule's maximum lateness, a flow's served bytes, a request's measured
//! `TTFT` — can be checked against numbers computed by hand or by brute force,
//! with no clock, no threads, and no randomness.

use std::collections::HashMap;

use super::types::{QueuedRequest, RequestId};

// ── SchedulerExecutor ────────────────────────────────────────────────────────

/// The per-request progress the executor reports for one served tick.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct RoundProgress {
    /// The request this progress is for.
    pub id: RequestId,
    /// Whether an output token was emitted on this tick. The first served tick
    /// of a request (its prefill) always emits its first token; thereafter a
    /// token is emitted once every `tick_cost` served ticks.
    pub token_emitted: bool,
    /// The total number of output tokens emitted so far, across all ticks.
    pub tokens_emitted_total: u32,
    /// The number of service ticks still required before completion.
    pub remaining_ticks: u64,
    /// Whether this tick completed the request.
    pub completed: bool,
}

/// The backend the scheduler drives, one service round (tick) at a time.
///
/// The contract is intentionally tiny — one method — and every clause of it is
/// load-bearing:
///
/// - [`run_round`](SchedulerExecutor::run_round) is handed the ids the scheduler
///   selected for the given `tick` and must advance **each** of them by one unit
///   of service, returning one [`RoundProgress`] per selected id, in the same
///   order.
/// - It reports *progress only*. It must not, and cannot through this interface,
///   reveal or manage where a request's state lives; that is the batching layer's
///   concern, not the scheduler's.
/// - It is driven purely by logical ticks. The `tick` argument is advisory (for
///   logging and for executors that model tick-dependent cost); a pure
///   table-driven executor ignores it.
pub trait SchedulerExecutor {
    /// Advance each id in `selected` by one service tick at logical time `tick`,
    /// returning per-id progress in the same order as `selected`.
    ///
    /// Ids that the executor has no profile for, or that have already completed,
    /// are reported with `remaining_ticks == 0` and `completed == true` and no
    /// token, so a caller that double-serves a finished request sees a no-op
    /// rather than a panic.
    fn run_round(&mut self, tick: u64, selected: &[RequestId]) -> Vec<RoundProgress>;
}

// ── StaticSchedulerExecutor ──────────────────────────────────────────────────

/// The per-request service profile and running state inside the fixture.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
struct ExecCell {
    /// Total output tokens the request will emit.
    output_tokens: u32,
    /// Decode cost, in ticks per output token.
    tick_cost: u32,
    /// Total service ticks required: `1 + (output_tokens - 1) * tick_cost`.
    total_ticks: u64,
    /// Ticks of service delivered so far.
    ticks_served: u64,
    /// Output tokens emitted so far.
    tokens_emitted: u32,
}

impl ExecCell {
    fn new(output_tokens: u32, tick_cost: u32) -> Self {
        let output_tokens = output_tokens.max(1);
        let tick_cost = tick_cost.max(1);
        let decode_tokens = u64::from(output_tokens - 1);
        let total_ticks = 1 + decode_tokens * u64::from(tick_cost);
        Self {
            output_tokens,
            tick_cost,
            total_ticks,
            ticks_served: 0,
            tokens_emitted: 0,
        }
    }

    /// Advance one tick; return `(token_emitted, remaining, completed)`.
    fn advance(&mut self) -> (bool, u64, bool) {
        if self.ticks_served >= self.total_ticks {
            return (false, 0, true);
        }
        self.ticks_served += 1;
        // A token is emitted on the prefill tick (`ticks_served == 1`) and then
        // once every `tick_cost` ticks thereafter: at served ticks
        // `1, 1 + c, 1 + 2c, …`, which is exactly the ticks where
        // `(ticks_served - 1) % tick_cost == 0`.
        let token_emitted = (self.ticks_served - 1).is_multiple_of(u64::from(self.tick_cost))
            && self.tokens_emitted < self.output_tokens;
        if token_emitted {
            self.tokens_emitted += 1;
        }
        let completed = self.ticks_served >= self.total_ticks;
        let remaining = self.total_ticks - self.ticks_served;
        (token_emitted, remaining, completed)
    }
}

/// A fully deterministic, table-driven [`SchedulerExecutor`].
///
/// **This is a test fixture, not an inference backend.** It runs no model; it
/// counts ticks against a table you supplied — "request `r` needs `k` output
/// tokens at decode cost `c`" — and emits tokens on exactly the ticks the
/// two-phase decode model prescribes. It exists so that the scheduler's every
/// claim is checkable against hand-computed or brute-forced ground truth, and it
/// is public because that is as useful to a caller wiring up their own
/// [`SchedulerExecutor`] as to this crate's tests.
#[derive(Debug, Clone, Default)]
pub struct StaticSchedulerExecutor {
    cells: HashMap<RequestId, ExecCell>,
}

impl StaticSchedulerExecutor {
    /// An executor with no requests registered.
    #[must_use]
    pub fn new() -> Self {
        Self {
            cells: HashMap::new(),
        }
    }

    /// Register (or overwrite) a request's service profile.
    #[must_use]
    pub fn with_request(mut self, id: RequestId, output_tokens: u32, tick_cost: u32) -> Self {
        self.cells
            .insert(id, ExecCell::new(output_tokens, tick_cost));
        self
    }

    /// Build an executor whose profiles match a slice of requests exactly, so the
    /// scheduler's *declared* service times (from [`QueuedRequest::service_ticks`])
    /// and the executor's *actual* progress agree by construction.
    #[must_use]
    pub fn from_requests(requests: &[QueuedRequest]) -> Self {
        let mut executor = Self::new();
        for request in requests {
            executor.cells.insert(
                request.id,
                ExecCell::new(request.output_tokens, request.tick_cost),
            );
        }
        executor
    }

    /// Whether the executor has a profile for `id`.
    #[must_use]
    pub fn knows(&self, id: RequestId) -> bool {
        self.cells.contains_key(&id)
    }

    /// The total service time the executor will deliver for `id`, if known.
    #[must_use]
    pub fn total_ticks(&self, id: RequestId) -> Option<u64> {
        self.cells.get(&id).map(|cell| cell.total_ticks)
    }
}

impl SchedulerExecutor for StaticSchedulerExecutor {
    fn run_round(&mut self, _tick: u64, selected: &[RequestId]) -> Vec<RoundProgress> {
        selected
            .iter()
            .map(|&id| match self.cells.get_mut(&id) {
                Some(cell) => {
                    let (token_emitted, remaining, completed) = cell.advance();
                    RoundProgress {
                        id,
                        token_emitted,
                        tokens_emitted_total: cell.tokens_emitted,
                        remaining_ticks: remaining,
                        completed,
                    }
                }
                None => RoundProgress {
                    id,
                    token_emitted: false,
                    tokens_emitted_total: 0,
                    remaining_ticks: 0,
                    completed: true,
                },
            })
            .collect()
    }
}
