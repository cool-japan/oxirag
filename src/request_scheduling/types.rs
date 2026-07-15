//! Public data types for the `request_scheduling` module: the request model, the
//! configuration, the scheduling-policy enum, admission outcomes, errors, and the
//! measurement report a run produces.
//!
//! Everything here is over **logical ticks** — a `u64` counter owned by the
//! [`crate::request_scheduling::RequestScheduler`], advanced one unit per service
//! round. There is no wall-clock time anywhere: a "deadline at tick 40" and a
//! "`TTFT` budget of 8 ticks" are exact, reproducible quantities, so every
//! scheduling claim this module makes is checkable with no clock, no threads, and
//! no randomness (the precedent is `memory_paging`'s `created_tick` /
//! `last_accessed_tick`).

use std::collections::BTreeMap;
use std::fmt;

use thiserror::Error;

// ── RequestId ────────────────────────────────────────────────────────────────

/// A stable, cheap identifier for a request, unique within one scheduler.
///
/// A `u64` rather than a string: it is the tie-breaker of last resort in every
/// scheduling decision (earliest-deadline-first breaks deadline ties by id,
/// strict priority breaks priority ties by arrival then id, and so on), so it has
/// to carry a **total order** that is cheap to compare and impossible to tie.
#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Hash)]
pub struct RequestId(pub u64);

impl RequestId {
    /// Wrap a raw `u64` as a request id.
    #[must_use]
    pub fn new(value: u64) -> Self {
        Self(value)
    }
}

impl fmt::Display for RequestId {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "r{}", self.0)
    }
}

// ── RequestPriority ──────────────────────────────────────────────────────────

/// An ordinal priority level; **higher is more urgent**.
///
/// This is the axis [`SchedulingPolicy::StrictPriority`] orders by — and it is
/// exactly the axis that *starves* a low-priority request under a sustained
/// stream of high-priority arrivals, which is why the fair-queueing policies
/// order by [`PriorityClass::weight`] instead. The module's starvation test
/// contrasts the two on the same workload.
#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Hash)]
pub struct RequestPriority(pub u8);

impl RequestPriority {
    /// Best-effort background work; yields to everything else.
    pub const BULK: Self = Self(0);
    /// Ordinary offline batch traffic.
    pub const BATCH: Self = Self(64);
    /// The default interactive tier.
    pub const NORMAL: Self = Self(128);
    /// Latency-sensitive interactive traffic.
    pub const INTERACTIVE: Self = Self(192);
    /// Hard real-time; served ahead of all other tiers.
    pub const REALTIME: Self = Self(255);

    /// Wrap a raw level.
    #[must_use]
    pub fn new(level: u8) -> Self {
        Self(level)
    }
}

impl Default for RequestPriority {
    fn default() -> Self {
        Self::NORMAL
    }
}

// ── RequestDeadline ──────────────────────────────────────────────────────────

/// An absolute logical tick by which a request should have *completed*.
///
/// Lateness is `completion_tick - deadline.tick` (negative when a request
/// finishes early), and [`SchedulingPolicy::EarliestDeadlineFirst`] minimises the
/// maximum lateness across a workload — the property the module's headline test
/// pins against brute force.
#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Hash)]
pub struct RequestDeadline {
    /// The absolute logical tick the request is due by.
    pub tick: u64,
}

impl RequestDeadline {
    /// A deadline at an explicit tick.
    #[must_use]
    pub fn at(tick: u64) -> Self {
        Self { tick }
    }

    /// A request with effectively no deadline (`u64::MAX`). Under
    /// earliest-deadline-first these sort last, behind every dated request.
    #[must_use]
    pub fn never() -> Self {
        Self { tick: u64::MAX }
    }
}

impl Default for RequestDeadline {
    fn default() -> Self {
        Self::never()
    }
}

// ── SloTarget ────────────────────────────────────────────────────────────────

/// The per-request service-level objective the admission controller checks
/// against, in logical ticks.
///
/// Modelled on a `vLLM`-style serving stack, whose two headline latencies are
/// **`TTFT`** (time-to-first-token — the prefill latency, from arrival to the
/// first output token) and **`TPOT`** (time-per-output-token — the steady-state
/// decode cadence). A request is admitted only if the controller can plausibly
/// meet *both* budgets given the committed load; see
/// [`crate::request_scheduling::AdmissionController`].
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub struct SloTarget {
    /// Maximum admissible time-to-first-token, in ticks from arrival.
    pub ttft_budget: u64,
    /// Maximum admissible time-per-output-token, in ticks.
    pub tpot_budget: u64,
}

impl SloTarget {
    /// A target with explicit `TTFT` and `TPOT` budgets.
    #[must_use]
    pub fn new(ttft_budget: u64, tpot_budget: u64) -> Self {
        Self {
            ttft_budget,
            tpot_budget,
        }
    }

    /// A target that imposes no constraint (both budgets `u64::MAX`), so
    /// admission turns purely on backpressure.
    #[must_use]
    pub fn unbounded() -> Self {
        Self {
            ttft_budget: u64::MAX,
            tpot_budget: u64::MAX,
        }
    }
}

impl Default for SloTarget {
    fn default() -> Self {
        Self::unbounded()
    }
}

// ── PriorityClass ────────────────────────────────────────────────────────────

/// A **flow**: the traffic class a request belongs to, carrying both the
/// strict-priority level and the fair-queueing weight.
///
/// The two fields answer to two different policies and are deliberately
/// independent. [`SchedulingPolicy::StrictPriority`] orders by
/// [`Self::priority`]; the fair-queueing policies
/// ([`SchedulingPolicy::DeficitRoundRobin`],
/// [`SchedulingPolicy::WeightedFairQueueing`]) order by [`Self::weight`], giving
/// each backlogged flow service proportional to its weight rather than letting a
/// single high-priority flow monopolise the server.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct PriorityClass {
    /// Stable identifier for the flow, unique within one scheduler.
    pub id: u32,
    /// Optional human-readable label, carried through to [`FlowReport`].
    pub label: Option<String>,
    /// Fair-queueing weight, at least `1`. A flow of weight `2w` receives twice
    /// the long-run service of a flow of weight `w` under the fair policies.
    pub weight: u32,
    /// Strict-priority level; higher is served first under
    /// [`SchedulingPolicy::StrictPriority`].
    pub priority: RequestPriority,
}

impl PriorityClass {
    /// A class with the given id and weight, default (`NORMAL`) priority and no
    /// label. The weight is clamped up to `1` (a zero-weight flow would never be
    /// served and is never what a caller means).
    #[must_use]
    pub fn new(id: u32, weight: u32) -> Self {
        Self {
            id,
            label: None,
            weight: weight.max(1),
            priority: RequestPriority::default(),
        }
    }

    /// Set the strict-priority level.
    #[must_use]
    pub fn with_priority(mut self, priority: RequestPriority) -> Self {
        self.priority = priority;
        self
    }

    /// Set the human-readable label.
    #[must_use]
    pub fn with_label(mut self, label: impl Into<String>) -> Self {
        self.label = Some(label.into());
        self
    }
}

// ── SchedulingPolicy ─────────────────────────────────────────────────────────

/// The discipline that decides, at each tick, which admitted request the single
/// server serves.
///
/// The first three order individual requests; the last two are fair-queueing
/// disciplines that arbitrate between *flows* ([`PriorityClass`]).
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default)]
pub enum SchedulingPolicy {
    /// First-in-first-out by arrival tick, ties by id. Non-preemptive: a started
    /// request runs to completion. This is the policy the admission controller's
    /// `TTFT` estimate is *exact* for (nothing can jump ahead of a queued
    /// request), so it anchors the soundness test.
    Fifo,
    /// Earliest-deadline-first (Jackson's rule). Preemptive at tick granularity:
    /// each tick the ready request with the earliest [`RequestDeadline`] runs.
    /// On a single machine this minimises maximum lateness — the module's
    /// headline guarantee.
    #[default]
    EarliestDeadlineFirst,
    /// Strict priority by [`PriorityClass::priority`], preemptive. Included
    /// precisely because it **starves**: under a sustained stream of
    /// high-priority arrivals a low-priority request waits forever. The
    /// starvation test uses it as the ablation the fair policies must beat.
    StrictPriority,
    /// Deficit Round Robin over flows (Shreedhar & Varghese, 1996). Each
    /// backlogged flow accumulates a quantum proportional to its
    /// [`PriorityClass::weight`] and serves whole requests while its deficit
    /// allows. Weight-proportional within one maximum request, and
    /// starvation-free: every flow gets a turn every sweep.
    DeficitRoundRobin,
    /// A self-clocked weighted-fair-queueing discipline: each request is stamped
    /// with a virtual finish time `max(virtual_now, flow_last_finish) +
    /// size / weight`, and requests are served in increasing virtual finish
    /// order. Also weight-proportional and starvation-free.
    WeightedFairQueueing,
}

impl SchedulingPolicy {
    /// Whether the policy re-decides the served request every tick (preemptive)
    /// rather than running a selected request to completion (non-preemptive).
    #[must_use]
    pub fn is_preemptive(self) -> bool {
        matches!(self, Self::EarliestDeadlineFirst | Self::StrictPriority)
    }

    /// Whether the policy arbitrates between flows by weight (the fair-queueing
    /// disciplines) rather than ordering individual requests.
    #[must_use]
    pub fn is_fair_queueing(self) -> bool {
        matches!(self, Self::DeficitRoundRobin | Self::WeightedFairQueueing)
    }
}

// ── RequestSchedulingConfig ──────────────────────────────────────────────────

/// Configuration for a [`crate::request_scheduling::RequestScheduler`].
///
/// Named `RequestSchedulingConfig` rather than a bare `SchedulerConfig` on
/// purpose: the sibling `continuous_batching` module owns `ContinuousBatchConfig`
/// and the crate prelude is flat, so neither may export an ambiguous
/// `SchedulerConfig`.
#[derive(Debug, Clone)]
pub struct RequestSchedulingConfig {
    /// The scheduling discipline. Defaults to
    /// [`SchedulingPolicy::EarliestDeadlineFirst`].
    pub policy: SchedulingPolicy,
    /// The Deficit-Round-Robin base quantum, in ticks of service; a flow of
    /// weight `w` accumulates `w * base_quantum` per sweep. Must be at least the
    /// largest request's service time for the exact one-sweep starvation bound
    /// to hold; defaults to `16`.
    pub base_quantum: u64,
    /// Maximum number of admitted-but-incomplete requests the scheduler will
    /// hold. A request arriving when the queue is at this depth is shed
    /// ([`RejectionReason::Backpressure`]). Defaults to `1024`.
    pub max_queue_depth: usize,
    /// Whether the `TTFT`/`TPOT` admission checks are applied. When `false`,
    /// admission turns purely on backpressure. Defaults to `true`.
    pub admission_enabled: bool,
    /// A hard cap on the number of ticks a single [`run`](
    /// crate::request_scheduling::RequestScheduler::run) will simulate, so an
    /// unexpected non-terminating workload surfaces as
    /// [`SchedulerError::ExceededTickBudget`] rather than a hang. `None` derives
    /// a safe cap from the workload. Defaults to `None`.
    pub max_ticks: Option<u64>,
    /// Seed for the workload generator used by tests; the scheduler itself is
    /// deterministic and does not consume it. Defaults to `0`.
    pub seed: u64,
}

impl Default for RequestSchedulingConfig {
    fn default() -> Self {
        Self {
            policy: SchedulingPolicy::default(),
            base_quantum: 16,
            max_queue_depth: 1024,
            admission_enabled: true,
            max_ticks: None,
            seed: 0,
        }
    }
}

impl RequestSchedulingConfig {
    /// A configuration using the given policy and otherwise-default settings.
    #[must_use]
    pub fn new(policy: SchedulingPolicy) -> Self {
        Self {
            policy,
            ..Self::default()
        }
    }

    /// Set the scheduling policy.
    #[must_use]
    pub fn with_policy(mut self, policy: SchedulingPolicy) -> Self {
        self.policy = policy;
        self
    }

    /// Set the Deficit-Round-Robin base quantum.
    #[must_use]
    pub fn with_base_quantum(mut self, base_quantum: u64) -> Self {
        self.base_quantum = base_quantum;
        self
    }

    /// Set the backpressure queue-depth ceiling.
    #[must_use]
    pub fn with_max_queue_depth(mut self, max_queue_depth: usize) -> Self {
        self.max_queue_depth = max_queue_depth;
        self
    }

    /// Enable or disable the `TTFT`/`TPOT` admission checks.
    #[must_use]
    pub fn with_admission(mut self, enabled: bool) -> Self {
        self.admission_enabled = enabled;
        self
    }

    /// Set the hard tick-budget cap.
    #[must_use]
    pub fn with_max_ticks(mut self, max_ticks: u64) -> Self {
        self.max_ticks = Some(max_ticks);
        self
    }

    /// Set the workload-generator seed.
    #[must_use]
    pub fn with_seed(mut self, seed: u64) -> Self {
        self.seed = seed;
        self
    }

    /// Validate the configuration.
    ///
    /// # Errors
    ///
    /// Returns [`SchedulerError::InvalidConfig`] when `base_quantum` is `0` (a
    /// zero quantum means no flow is ever served) or `max_queue_depth` is `0`
    /// (nothing could ever be admitted).
    pub fn validate(&self) -> SchedulerResult<()> {
        if self.base_quantum == 0 {
            return Err(SchedulerError::InvalidConfig {
                reason: "base_quantum must be at least 1".to_string(),
            });
        }
        if self.max_queue_depth == 0 {
            return Err(SchedulerError::InvalidConfig {
                reason: "max_queue_depth must be at least 1".to_string(),
            });
        }
        Ok(())
    }
}

// ── QueuedRequest ────────────────────────────────────────────────────────────

/// A request submitted for scheduling, with everything the policy and admission
/// controller need to reason about it.
///
/// The **work model** is deliberately a faithful two-phase decode: a request of
/// `output_tokens` tokens takes one tick of prefill (producing its first token)
/// and then `tick_cost` ticks per subsequent decode token, so its total service
/// time is [`service_ticks`](Self::service_ticks) `= 1 + (output_tokens - 1) *
/// tick_cost`. `TPOT` is exactly `tick_cost` when the request runs uninterrupted,
/// and `TTFT` is the delay from arrival to that first prefill tick.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct QueuedRequest {
    /// The request's unique id.
    pub id: RequestId,
    /// The flow this request belongs to.
    pub class: PriorityClass,
    /// The logical tick at which the request arrives (becomes eligible for
    /// admission and service).
    pub arrival_tick: u64,
    /// The tick by which the request should complete.
    pub deadline: RequestDeadline,
    /// The number of output tokens the request will generate; at least `1`.
    pub output_tokens: u32,
    /// The steady-state decode cost, in ticks per output token; at least `1`.
    pub tick_cost: u32,
    /// The service-level objective the admission controller checks.
    pub slo: SloTarget,
}

impl QueuedRequest {
    /// Build a request. `output_tokens` and `tick_cost` are clamped up to `1`
    /// (a zero-length or zero-cost request is never meaningful).
    #[must_use]
    pub fn new(id: RequestId, class: PriorityClass, arrival_tick: u64, output_tokens: u32) -> Self {
        Self {
            id,
            class,
            arrival_tick,
            deadline: RequestDeadline::never(),
            output_tokens: output_tokens.max(1),
            tick_cost: 1,
            slo: SloTarget::unbounded(),
        }
    }

    /// Set the completion deadline.
    #[must_use]
    pub fn with_deadline(mut self, deadline: RequestDeadline) -> Self {
        self.deadline = deadline;
        self
    }

    /// Set the per-output-token decode cost (clamped up to `1`).
    #[must_use]
    pub fn with_tick_cost(mut self, tick_cost: u32) -> Self {
        self.tick_cost = tick_cost.max(1);
        self
    }

    /// Set the service-level objective.
    #[must_use]
    pub fn with_slo(mut self, slo: SloTarget) -> Self {
        self.slo = slo;
        self
    }

    /// The total service time of this request in ticks:
    /// `1 + (output_tokens - 1) * tick_cost`.
    #[must_use]
    pub fn service_ticks(&self) -> u64 {
        let decode_tokens = u64::from(self.output_tokens.saturating_sub(1));
        1 + decode_tokens * u64::from(self.tick_cost)
    }
}

// ── RejectionReason ──────────────────────────────────────────────────────────

/// Why the admission controller rejected a request.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum RejectionReason {
    /// The queue was already at [`RequestSchedulingConfig::max_queue_depth`]:
    /// backpressure shed the arrival.
    Backpressure {
        /// The occupied depth at the moment of arrival.
        depth: usize,
        /// The configured ceiling.
        capacity: usize,
    },
    /// The estimated time-to-first-token exceeded the request's
    /// [`SloTarget::ttft_budget`].
    TtftInfeasible {
        /// The controller's `TTFT` estimate, in ticks.
        estimated: u64,
        /// The request's budget, in ticks.
        budget: u64,
    },
    /// The request's steady-state decode cost already exceeds its
    /// [`SloTarget::tpot_budget`], so no schedule could meet the `TPOT` budget.
    TpotInfeasible {
        /// The request's required time-per-output-token (`tick_cost`).
        required: u64,
        /// The request's budget, in ticks.
        budget: u64,
    },
}

impl fmt::Display for RejectionReason {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::Backpressure { depth, capacity } => {
                write!(
                    f,
                    "backpressure: queue depth {depth} at capacity {capacity}"
                )
            }
            Self::TtftInfeasible { estimated, budget } => {
                write!(
                    f,
                    "TTFT infeasible: estimated {estimated} > budget {budget}"
                )
            }
            Self::TpotInfeasible { required, budget } => {
                write!(f, "TPOT infeasible: required {required} > budget {budget}")
            }
        }
    }
}

// ── SchedulingOutcome ────────────────────────────────────────────────────────

/// The result of offering a request to the scheduler at its arrival tick.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum SchedulingOutcome {
    /// The request was admitted and enqueued for service.
    Admitted,
    /// The request was rejected, with the reason.
    Rejected(RejectionReason),
}

impl SchedulingOutcome {
    /// Whether the request was admitted.
    #[must_use]
    pub fn is_admitted(self) -> bool {
        matches!(self, Self::Admitted)
    }

    /// The rejection reason, if any.
    #[must_use]
    pub fn rejection(self) -> Option<RejectionReason> {
        match self {
            Self::Admitted => None,
            Self::Rejected(reason) => Some(reason),
        }
    }
}

// ── Report types ─────────────────────────────────────────────────────────────

/// One tick of service, recorded in [`SchedulingReport::events`].
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct ServiceEvent {
    /// The tick this service happened on.
    pub tick: u64,
    /// The request that was served.
    pub request: RequestId,
    /// Whether an output token was emitted on this tick.
    pub token_emitted: bool,
    /// Whether this tick completed the request.
    pub completed: bool,
}

/// Per-request measurements distilled from a run.
#[derive(Debug, Clone, PartialEq)]
pub struct RequestReport {
    /// The request's id.
    pub id: RequestId,
    /// The flow it belonged to.
    pub class_id: u32,
    /// Its arrival tick.
    pub arrival_tick: u64,
    /// Whether it was admitted.
    pub admitted: bool,
    /// The rejection reason, if it was rejected.
    pub rejection: Option<RejectionReason>,
    /// The tick its first output token was emitted, if it ran.
    pub first_token_tick: Option<u64>,
    /// The tick it completed, if it completed.
    pub completion_tick: Option<u64>,
    /// The total ticks of service it received.
    pub service_ticks: u64,
    /// Measured time-to-first-token (`first_token_tick - arrival_tick`).
    pub ttft: Option<u64>,
    /// Measured worst-case time-per-output-token: the largest gap, in ticks,
    /// between two consecutive emitted tokens.
    pub tpot_max: Option<u64>,
    /// The request's deadline tick.
    pub deadline_tick: u64,
    /// Measured lateness (`completion_tick - deadline_tick`), negative when
    /// early; `None` if it did not complete.
    pub lateness: Option<i64>,
}

/// Per-flow measurements distilled from a run.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct FlowReport {
    /// The flow's id.
    pub class_id: u32,
    /// The flow's configured weight.
    pub weight: u32,
    /// The flow's Deficit-Round-Robin quantum (`weight * base_quantum`).
    pub quantum: u64,
    /// The total ticks of service the flow received.
    pub served_ticks: u64,
    /// The number of whole requests from the flow that completed.
    pub served_packets: u64,
}

/// A snapshot taken at the end of a Deficit-Round-Robin sweep, so the fairness
/// bound can be checked at a point where the relevant flows are all still
/// backlogged.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct DrrSweepSnapshot {
    /// The sweep index (`0`-based).
    pub sweep_index: u64,
    /// Cumulative service ticks per flow at the end of this sweep.
    pub served_bytes: BTreeMap<u32, u64>,
    /// Each flow's deficit counter at the end of this sweep.
    pub deficits: BTreeMap<u32, u64>,
    /// The flows that were backlogged for the whole of this sweep.
    pub backlogged: Vec<u32>,
}

/// Everything a [`run`](crate::request_scheduling::RequestScheduler::run)
/// measured: the tick-by-tick service log plus per-request and per-flow summaries.
#[derive(Debug, Clone, Default)]
pub struct SchedulingReport {
    /// Every service tick, in order.
    pub events: Vec<ServiceEvent>,
    /// Per-request summaries, keyed and iterated in id order.
    pub per_request: BTreeMap<RequestId, RequestReport>,
    /// Per-flow summaries, keyed and iterated in flow-id order.
    pub per_flow: BTreeMap<u32, FlowReport>,
    /// Deficit-Round-Robin sweep snapshots (empty under other policies).
    pub drr_sweeps: Vec<DrrSweepSnapshot>,
    /// The tick the last request completed (`0` if none did).
    pub makespan: u64,
}

impl SchedulingReport {
    /// The maximum lateness over all *completed* requests, or `None` if none
    /// completed. This is the objective earliest-deadline-first minimises.
    #[must_use]
    pub fn max_lateness(&self) -> Option<i64> {
        self.per_request
            .values()
            .filter_map(|report| report.lateness)
            .max()
    }

    /// The ids of every admitted request, in id order.
    #[must_use]
    pub fn admitted_ids(&self) -> Vec<RequestId> {
        self.per_request
            .values()
            .filter(|report| report.admitted)
            .map(|report| report.id)
            .collect()
    }

    /// The completion tick of a request, if it completed.
    #[must_use]
    pub fn completion_tick(&self, id: RequestId) -> Option<u64> {
        self.per_request
            .get(&id)
            .and_then(|report| report.completion_tick)
    }

    /// The order in which requests *completed*, earliest first.
    #[must_use]
    pub fn completion_order(&self) -> Vec<RequestId> {
        let mut completed: Vec<(u64, RequestId)> = self
            .per_request
            .values()
            .filter_map(|report| report.completion_tick.map(|tick| (tick, report.id)))
            .collect();
        completed.sort_unstable();
        completed.into_iter().map(|(_, id)| id).collect()
    }
}

// ── Errors ───────────────────────────────────────────────────────────────────

/// Errors produced by the `request_scheduling` module.
#[derive(Debug, Error, PartialEq, Eq)]
pub enum SchedulerError {
    /// Two requests with the same [`RequestId`] were submitted to one scheduler.
    #[error("duplicate request id: {id}")]
    DuplicateRequest {
        /// The id that appeared more than once.
        id: RequestId,
    },
    /// A request was submitted whose service profile the executor does not know
    /// how to run.
    #[error("executor has no service profile for request {id}")]
    MissingExecutorSpec {
        /// The id with no profile.
        id: RequestId,
    },
    /// A configuration value was outside its admissible range.
    #[error("invalid scheduler configuration: {reason}")]
    InvalidConfig {
        /// A human-readable explanation.
        reason: String,
    },
    /// A run hit its hard tick-budget cap without draining, which indicates a
    /// malformed workload (or a genuine scheduler bug) rather than a legitimate
    /// long run.
    #[error("run exceeded its tick budget of {cap} ticks before draining")]
    ExceededTickBudget {
        /// The cap that was hit.
        cap: u64,
    },
}

/// Convenience alias for this module's fallible return type.
pub type SchedulerResult<T> = Result<T, SchedulerError>;
