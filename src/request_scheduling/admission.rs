//! `SLO`-aware admission control and queue-depth backpressure.
//!
//! The scheduler admits a request only if it can *plausibly* meet the request's
//! service-level objective. Two independent gates apply, in this order:
//!
//! 1. **Backpressure.** If the queue is already at
//!    [`RequestSchedulingConfig::max_queue_depth`](
//!    crate::request_scheduling::RequestSchedulingConfig), the arrival is shed —
//!    a bounded queue is what keeps latency bounded, so shedding *earlier* is
//!    strictly kinder than admitting into an already-hopeless backlog.
//! 2. **`SLO` feasibility.** The request's steady-state decode cost must not
//!    already exceed its [`SloTarget::tpot_budget`] (no schedule could fix a
//!    request that is too slow token-for-token), and the estimated
//!    time-to-first-token — the service work that will run ahead of it — must not
//!    exceed its [`SloTarget::ttft_budget`].
//!
//! # Soundness and non-conservatism
//!
//! The `TTFT` estimate is provided by the scheduler and is policy-dependent. Under
//! [`SchedulingPolicy::Fifo`](crate::request_scheduling::SchedulingPolicy) it is
//! the total remaining service work in the system, which is **exactly** the delay
//! a first-come-first-served, work-conserving single server imposes before the
//! new request starts — nothing can jump ahead of a queued request under FIFO. So
//! under FIFO the estimate is an exact upper bound: every admitted request meets
//! its budget (soundness), and a request whose true delay is *at* its budget is
//! admitted rather than rejected out of excess caution (non-conservatism). Both
//! halves are asserted, with measured numbers, in the module's tests.

use super::types::{QueuedRequest, RejectionReason, SchedulingOutcome, SloTarget};

// ── Backpressure ─────────────────────────────────────────────────────────────

/// The queue-depth gate: sheds arrivals once the admitted-but-incomplete backlog
/// reaches capacity.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct Backpressure {
    /// The maximum admitted-but-incomplete depth.
    capacity: usize,
}

impl Backpressure {
    /// A gate with the given capacity (clamped up to `1`; a zero-capacity gate
    /// could never admit anything).
    #[must_use]
    pub fn new(capacity: usize) -> Self {
        Self {
            capacity: capacity.max(1),
        }
    }

    /// The configured capacity.
    #[must_use]
    pub fn capacity(self) -> usize {
        self.capacity
    }

    /// Whether an arrival is admissible at the given current `depth`. Admissible
    /// strictly below capacity; shed at or above it.
    #[must_use]
    pub fn admits(self, depth: usize) -> bool {
        depth < self.capacity
    }

    /// The rejection reason for an arrival at `depth`, if backpressure sheds it.
    #[must_use]
    pub fn rejection(self, depth: usize) -> Option<RejectionReason> {
        if self.admits(depth) {
            None
        } else {
            Some(RejectionReason::Backpressure {
                depth,
                capacity: self.capacity,
            })
        }
    }
}

// ── AdmissionController ──────────────────────────────────────────────────────

/// The composite admission gate: backpressure first, then `TPOT` and `TTFT`
/// feasibility.
///
/// The controller is *stateless* in the load it reasons about — the scheduler
/// passes in the current queue depth and the `TTFT` estimate for the arriving
/// request — so its decision is a pure function of its inputs and trivially
/// testable in isolation, exactly as the module's soundness and backpressure
/// tests do.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct AdmissionController {
    /// The backpressure gate.
    backpressure: Backpressure,
    /// Whether the `TTFT`/`TPOT` checks are applied (backpressure always is).
    slo_enabled: bool,
}

impl AdmissionController {
    /// A controller with the given queue-depth capacity. `slo_enabled` toggles
    /// the `TTFT`/`TPOT` gates; backpressure is always active.
    #[must_use]
    pub fn new(capacity: usize, slo_enabled: bool) -> Self {
        Self {
            backpressure: Backpressure::new(capacity),
            slo_enabled,
        }
    }

    /// The backpressure gate this controller uses.
    #[must_use]
    pub fn backpressure(self) -> Backpressure {
        self.backpressure
    }

    /// Decide whether to admit `request`, given the current admitted-but-
    /// incomplete queue `depth` and the scheduler's `ttft_estimate` (the service
    /// work, in ticks, that will run ahead of this request before its first
    /// token).
    ///
    /// The order is deliberate: backpressure is checked first (an over-full queue
    /// is shed regardless of how tight its `SLO` is), then `TPOT` (an intrinsic
    /// property of the request, independent of load), then `TTFT` (the
    /// load-dependent estimate).
    #[must_use]
    pub fn decide(
        self,
        request: &QueuedRequest,
        depth: usize,
        ttft_estimate: u64,
    ) -> SchedulingOutcome {
        if let Some(reason) = self.backpressure.rejection(depth) {
            return SchedulingOutcome::Rejected(reason);
        }
        let slo_reason = self
            .slo_enabled
            .then(|| slo_rejection(request.slo, request.tick_cost, ttft_estimate))
            .flatten();
        if let Some(reason) = slo_reason {
            return SchedulingOutcome::Rejected(reason);
        }
        SchedulingOutcome::Admitted
    }
}

/// The `SLO` rejection reason for a request whose decode cost is `tick_cost` and
/// whose estimated time-to-first-token is `ttft_estimate`, or `None` if both
/// budgets are met. `TPOT` is checked before `TTFT` so an intrinsically-too-slow
/// request is rejected on its own merits rather than blamed on queueing.
fn slo_rejection(slo: SloTarget, tick_cost: u32, ttft_estimate: u64) -> Option<RejectionReason> {
    let required_tpot = u64::from(tick_cost);
    if required_tpot > slo.tpot_budget {
        return Some(RejectionReason::TpotInfeasible {
            required: required_tpot,
            budget: slo.tpot_budget,
        });
    }
    if ttft_estimate > slo.ttft_budget {
        return Some(RejectionReason::TtftInfeasible {
            estimated: ttft_estimate,
            budget: slo.ttft_budget,
        });
    }
    None
}
