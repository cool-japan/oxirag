//! Request scheduling: the **policy layer above a serving backend** — `SLO`-aware
//! admission control, priority classes and fair queueing, deadline ordering, and
//! backpressure — all decided over **logical ticks**, with no wall clock, no
//! threads, and no randomness anywhere in a scheduling decision.
//!
//! This module answers *which request runs next, and whether a request should be
//! let in at all* — not *how the tokens are produced* and not *where their state
//! lives*. It owns **time and order**. Given a stream of requests, each carrying a
//! priority class, a deadline, a service-level objective, and a declared amount
//! of work, it decides:
//!
//! - **Admission.** Admit a request only if its `TTFT` (time-to-first-token) and
//!   `TPOT` (time-per-output-token) budgets can plausibly be met given the
//!   committed load — and shed it under backpressure when the queue is full. See
//!   [`AdmissionController`] and [`Backpressure`].
//! - **Order.** Serve admitted requests by the configured [`SchedulingPolicy`]:
//!   first-come-first-served, earliest-deadline-first (which minimises maximum
//!   lateness — Jackson's rule), strict priority, Deficit Round Robin, or a
//!   self-clocked weighted fair queue. See [`RequestScheduler`] and [`FairQueue`].
//! - **Fairness and liveness.** Under the fair-queueing policies, service is
//!   proportional to [`PriorityClass::weight`] within one maximum request, and
//!   **no admitted request is ever starved**: every backlogged flow gets a turn
//!   every sweep, regardless of how hard another flow floods the system.
//!
//! # Distinct from `connection_pool`
//!
//! Distinct from `connection_pool`, which caps concurrency with an unfair
//! semaphore over idle connections and knows nothing of a request's priority,
//! deadline, or remaining work: this module orders and admits **requests**
//! against explicit `TTFT`/`TPOT` budgets, and never touches the execution
//! backend's memory.
//!
//! # Scope: where this module ends
//!
//! It never touches bytes, `KV` blocks, or memory. It reports and consumes only
//! **progress** — a token emitted, work remaining, completion — through its own
//! minimal [`SchedulerExecutor`] trait (the house precedent for stating your own
//! requirement rather than widening someone else's is `replug`'s language-model
//! trait). Where a request's state lives, and how within-request work is split,
//! belong to the sibling serving modules, not here.
//!
//! # Why logical ticks
//!
//! Time is a `u64` counter advanced one unit per service round. A "deadline at
//! tick 40" and a "`TTFT` budget of 8 ticks" are therefore *exact*: every claim —
//! an earliest-deadline-first schedule's maximum lateness, a flow's served bytes,
//! whether an admitted request met its budget — is checkable against numbers
//! computed by hand or by brute force, with no `sleep`s and no clock-resolution
//! ties (the precedent is `memory_paging`'s tick-based recency).
//!
//! # Determinism
//!
//! Every tie in every policy is broken by a total order that ends in the
//! [`RequestId`], so a run is a pure function of its workload: the same requests
//! produce a byte-identical schedule, tick for tick. [`SchedulerRng`] exists only
//! to synthesise *workloads* for tests; it never enters a scheduling decision.
//!
//! # Quick start
//!
//! ```
//! # #[cfg(feature = "request-scheduling")]
//! # {
//! use oxirag::request_scheduling::{
//!     PriorityClass, QueuedRequest, RequestDeadline, RequestId, RequestScheduler,
//!     RequestSchedulingConfig, SchedulingPolicy, StaticSchedulerExecutor,
//! };
//!
//! // A single flow, three requests arriving together with different deadlines.
//! let class = PriorityClass::new(0, 1);
//! let requests = vec![
//!     QueuedRequest::new(RequestId::new(0), class.clone(), 0, 4)
//!         .with_deadline(RequestDeadline::at(12)),
//!     QueuedRequest::new(RequestId::new(1), class.clone(), 0, 2)
//!         .with_deadline(RequestDeadline::at(4)),
//!     QueuedRequest::new(RequestId::new(2), class.clone(), 0, 3)
//!         .with_deadline(RequestDeadline::at(9)),
//! ];
//! let executor = StaticSchedulerExecutor::from_requests(&requests);
//! let config = RequestSchedulingConfig::new(SchedulingPolicy::EarliestDeadlineFirst);
//! let mut scheduler = RequestScheduler::new(config, executor).expect("valid config");
//! for request in requests {
//!     scheduler.enqueue(request).expect("distinct ids");
//! }
//!
//! let report = scheduler.run().expect("drains within budget");
//! // Earliest deadline (r1) completes first; nothing finishes late.
//! assert_eq!(report.completion_order().first().copied(), Some(RequestId::new(1)));
//! assert!(report.max_lateness().is_some_and(|lateness| lateness <= 0));
//! # }
//! ```

pub mod admission;
pub mod engine;
pub mod executor;
pub mod fair_queue;
pub mod rng;
pub mod types;

#[cfg(all(test, not(target_arch = "wasm32")))]
mod tests;

pub use admission::{AdmissionController, Backpressure};
pub use engine::RequestScheduler;
pub use executor::{RoundProgress, SchedulerExecutor, StaticSchedulerExecutor};
pub use fair_queue::{FairQueue, QueuedItem, RequestQueue, WeightedFairClock};
pub use rng::SchedulerRng;
pub use types::{
    DrrSweepSnapshot, FlowReport, PriorityClass, QueuedRequest, RejectionReason, RequestDeadline,
    RequestId, RequestPriority, RequestReport, RequestSchedulingConfig, SchedulerError,
    SchedulerResult, SchedulingOutcome, SchedulingPolicy, SchedulingReport, ServiceEvent,
    SloTarget,
};
