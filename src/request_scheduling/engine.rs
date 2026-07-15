//! [`RequestScheduler`] — the policy layer that admits, orders, and serves
//! requests over logical ticks, driving a [`SchedulerExecutor`] one round at a
//! time and distilling the run into a [`SchedulingReport`].
//!
//! The scheduler owns **time and order** and nothing else. It decides *whether*
//! to admit a request (against its `TTFT`/`TPOT` budget and the queue-depth
//! ceiling), *which* admitted request the single server serves each tick (by the
//! configured [`SchedulingPolicy`]), and *when* each request first produces a
//! token and finally completes. It never touches where a request's state lives —
//! that is the batching layer's concern.

use std::collections::{BTreeMap, BTreeSet, HashMap};

use super::admission::AdmissionController;
use super::executor::SchedulerExecutor;
use super::fair_queue::{FairQueue, QueuedItem, WeightedFairClock};
use super::types::{
    FlowReport, PriorityClass, QueuedRequest, RequestId, RequestPriority, RequestReport,
    RequestSchedulingConfig, SchedulerError, SchedulerResult, SchedulingPolicy, SchedulingReport,
    ServiceEvent, SloTarget,
};

/// Immutable per-request facts, captured at [`RequestScheduler::enqueue`] time.
#[derive(Debug, Clone)]
struct StaticInfo {
    flow: u32,
    weight: u32,
    priority: RequestPriority,
    arrival_tick: u64,
    deadline_tick: u64,
    service_ticks: u64,
    tick_cost: u32,
    slo: SloTarget,
}

/// Mutable per-request state, evolving as the request is served.
#[derive(Debug, Clone)]
struct RuntimeState {
    remaining_ticks: u64,
    first_token_tick: Option<u64>,
    last_token_tick: Option<u64>,
    tpot_max: Option<u64>,
    service_ticks_done: u64,
}

/// The policy layer above an execution backend: `SLO`-aware admission, priority
/// classes, fair queueing, deadline ordering, starvation prevention, and
/// backpressure — all over logical ticks.
///
/// Distinct from `connection_pool`, which caps concurrency with an unfair
/// semaphore over idle connections and knows nothing of a request's priority,
/// deadline, or remaining work: this scheduler orders and admits **requests**
/// against explicit `TTFT`/`TPOT` budgets, and never touches the execution
/// backend's memory.
///
/// # Example
///
/// ```
/// # #[cfg(feature = "request-scheduling")]
/// # {
/// use oxirag::request_scheduling::{
///     PriorityClass, QueuedRequest, RequestDeadline, RequestId, RequestScheduler,
///     RequestSchedulingConfig, SchedulingPolicy, StaticSchedulerExecutor,
/// };
///
/// let class = PriorityClass::new(0, 1);
/// // Two requests: r1 is due sooner, so earliest-deadline-first serves it first.
/// let requests = vec![
///     QueuedRequest::new(RequestId::new(0), class.clone(), 0, 3)
///         .with_deadline(RequestDeadline::at(10)),
///     QueuedRequest::new(RequestId::new(1), class.clone(), 0, 3)
///         .with_deadline(RequestDeadline::at(5)),
/// ];
/// let executor = StaticSchedulerExecutor::from_requests(&requests);
/// let config = RequestSchedulingConfig::new(SchedulingPolicy::EarliestDeadlineFirst);
/// let mut scheduler = RequestScheduler::new(config, executor).expect("valid config");
/// for request in requests {
///     scheduler.enqueue(request).expect("distinct ids");
/// }
/// let report = scheduler.run().expect("drains within budget");
///
/// // The earlier-deadline request completes first, and nothing is late.
/// assert_eq!(report.completion_order().first().copied(), Some(RequestId::new(1)));
/// assert!(report.max_lateness().is_some_and(|lateness| lateness <= 0));
/// # }
/// ```
#[derive(Debug)]
pub struct RequestScheduler<E: SchedulerExecutor> {
    config: RequestSchedulingConfig,
    executor: E,
    admission: AdmissionController,
    requests: HashMap<RequestId, StaticInfo>,
    pending: BTreeMap<u64, Vec<RequestId>>,
    classes: BTreeMap<u32, PriorityClass>,
    runtime: HashMap<RequestId, RuntimeState>,
    active_ids: BTreeSet<RequestId>,
    fair: FairQueue,
    wfq: WeightedFairClock,
    current: Option<RequestId>,
    depth: usize,
    events: Vec<ServiceEvent>,
    per_request: BTreeMap<RequestId, RequestReport>,
    per_flow: BTreeMap<u32, FlowReport>,
    makespan: u64,
    total_service: u64,
}

impl<E: SchedulerExecutor> RequestScheduler<E> {
    /// Build a scheduler with the given configuration and execution backend.
    ///
    /// # Errors
    ///
    /// Returns [`SchedulerError::InvalidConfig`] when the configuration fails
    /// [`RequestSchedulingConfig::validate`].
    pub fn new(config: RequestSchedulingConfig, executor: E) -> SchedulerResult<Self> {
        config.validate()?;
        let admission = AdmissionController::new(config.max_queue_depth, config.admission_enabled);
        Ok(Self {
            config,
            executor,
            admission,
            requests: HashMap::new(),
            pending: BTreeMap::new(),
            classes: BTreeMap::new(),
            runtime: HashMap::new(),
            active_ids: BTreeSet::new(),
            fair: FairQueue::new(BTreeMap::new(), 1),
            wfq: WeightedFairClock::new(),
            current: None,
            depth: 0,
            events: Vec::new(),
            per_request: BTreeMap::new(),
            per_flow: BTreeMap::new(),
            makespan: 0,
            total_service: 0,
        })
    }

    /// A shared reference to the execution backend (for post-run inspection).
    pub fn executor(&self) -> &E {
        &self.executor
    }

    /// Register a request with its arrival tick.
    ///
    /// # Errors
    ///
    /// Returns [`SchedulerError::DuplicateRequest`] if a request with the same id
    /// was already enqueued.
    pub fn enqueue(&mut self, request: QueuedRequest) -> SchedulerResult<()> {
        if self.requests.contains_key(&request.id) {
            return Err(SchedulerError::DuplicateRequest { id: request.id });
        }
        let id = request.id;
        let flow = request.class.id;
        let weight = request.class.weight;
        let priority = request.class.priority;
        let arrival_tick = request.arrival_tick;
        let deadline_tick = request.deadline.tick;
        let tick_cost = request.tick_cost;
        let slo = request.slo;
        let service_ticks = request.service_ticks();
        self.total_service += service_ticks;
        self.classes.entry(flow).or_insert(request.class);
        self.requests.insert(
            id,
            StaticInfo {
                flow,
                weight,
                priority,
                arrival_tick,
                deadline_tick,
                service_ticks,
                tick_cost,
                slo,
            },
        );
        self.pending.entry(arrival_tick).or_default().push(id);
        Ok(())
    }

    /// Run the scheduler until every admitted request has completed and no
    /// further arrivals remain, returning the measurement report.
    ///
    /// # Errors
    ///
    /// Returns [`SchedulerError::ExceededTickBudget`] if the run does not drain
    /// within its tick budget (a malformed workload or a scheduler bug), and
    /// [`SchedulerError::InvalidConfig`] if the configuration is invalid.
    pub fn run(&mut self) -> SchedulerResult<SchedulingReport> {
        self.config.validate()?;
        self.init_fair_state();
        for ids in self.pending.values_mut() {
            ids.sort_unstable();
        }
        let max_arrival = self.pending.keys().copied().max().unwrap_or(0);
        let cap = self
            .config
            .max_ticks
            .unwrap_or(max_arrival + self.total_service + 8);

        let mut tick: u64 = 0;
        loop {
            self.admit_arrivals(tick);
            if let Some(id) = self.select() {
                self.serve(id, tick);
            }
            if self.depth == 0 && tick >= max_arrival {
                break;
            }
            tick += 1;
            if tick > cap {
                return Err(SchedulerError::ExceededTickBudget { cap });
            }
        }
        Ok(self.build_report())
    }

    /// Initialise the fair-queueing state from the registered flows.
    fn init_fair_state(&mut self) {
        let quanta: BTreeMap<u32, u64> = self
            .classes
            .values()
            .map(|class| (class.id, u64::from(class.weight) * self.config.base_quantum))
            .collect();
        self.fair = FairQueue::new(quanta, self.config.base_quantum);
        self.wfq = WeightedFairClock::new();
    }

    /// Admit (or reject) every request arriving at `tick`, in id order.
    fn admit_arrivals(&mut self, tick: u64) {
        let Some(ids) = self.pending.get(&tick).cloned() else {
            return;
        };
        for id in ids {
            let Some(info) = self.requests.get(&id).cloned() else {
                continue;
            };
            let ttft_estimate = self.ttft_estimate(&info);
            let request_shim = build_admission_shim(&info);
            let outcome = self
                .admission
                .decide(&request_shim, self.depth, ttft_estimate);
            let admitted = outcome.is_admitted();
            self.per_request.insert(
                id,
                RequestReport {
                    id,
                    class_id: info.flow,
                    arrival_tick: info.arrival_tick,
                    admitted,
                    rejection: outcome.rejection(),
                    first_token_tick: None,
                    completion_tick: None,
                    service_ticks: 0,
                    ttft: None,
                    tpot_max: None,
                    deadline_tick: info.deadline_tick,
                    lateness: None,
                },
            );
            if admitted {
                self.on_admit(id, &info);
            }
        }
    }

    /// Wire an admitted request into the runtime and the policy's ready
    /// structure.
    fn on_admit(&mut self, id: RequestId, info: &StaticInfo) {
        self.depth += 1;
        self.runtime.insert(
            id,
            RuntimeState {
                remaining_ticks: info.service_ticks,
                first_token_tick: None,
                last_token_tick: None,
                tpot_max: None,
                service_ticks_done: 0,
            },
        );
        self.active_ids.insert(id);
        match self.config.policy {
            SchedulingPolicy::DeficitRoundRobin => self.fair.enqueue(QueuedItem {
                id,
                flow: info.flow,
                size: info.service_ticks,
                arrival: info.arrival_tick,
            }),
            SchedulingPolicy::WeightedFairQueueing => {
                self.wfq.enqueue(
                    id,
                    info.flow,
                    info.service_ticks,
                    info.weight,
                    info.arrival_tick,
                );
            }
            _ => {}
        }
    }

    /// The scheduler's estimate of the service work, in ticks, that will run
    /// ahead of a newly-arriving request before its first token.
    fn ttft_estimate(&self, info: &StaticInfo) -> u64 {
        match self.config.policy {
            // FIFO: the new request is last, so every unit of remaining work is
            // ahead of it. This is exact for a work-conserving single server.
            SchedulingPolicy::Fifo => self.active_remaining(|_| true),
            // EDF: requests that will still outrank it by deadline.
            SchedulingPolicy::EarliestDeadlineFirst => {
                let key = (info.deadline_tick, info.priority, info.arrival_tick);
                self.active_remaining(|other| {
                    (other.deadline_tick, other.priority, other.arrival_tick) < key
                })
            }
            // Strict priority: requests at least as urgent that arrived no later.
            SchedulingPolicy::StrictPriority => self.active_remaining(|other| {
                (other.priority, std::cmp::Reverse(other.arrival_tick))
                    > (info.priority, std::cmp::Reverse(info.arrival_tick))
            }),
            // Fair disciplines: the request's own flow backlog (a best-effort
            // estimate; the exact-soundness claim is made only for FIFO).
            SchedulingPolicy::DeficitRoundRobin | SchedulingPolicy::WeightedFairQueueing => {
                self.active_remaining(|other| other.flow == info.flow)
            }
        }
    }

    /// Sum the remaining service ticks of the active requests matching `keep`.
    fn active_remaining(&self, keep: impl Fn(&StaticInfo) -> bool) -> u64 {
        self.active_ids
            .iter()
            .filter_map(|id| {
                let info = self.requests.get(id)?;
                let runtime = self.runtime.get(id)?;
                keep(info).then_some(runtime.remaining_ticks)
            })
            .sum()
    }

    /// The request to serve this tick, or `None` when nothing is runnable.
    fn select(&mut self) -> Option<RequestId> {
        match self.config.policy {
            SchedulingPolicy::EarliestDeadlineFirst => {
                self.select_min_key(|info| (info.deadline_tick, 0, 0))
            }
            SchedulingPolicy::StrictPriority => self.select_min_key(|info| {
                // Highest priority first (invert), then earliest arrival.
                (u64::from(u8::MAX - info.priority.0), info.arrival_tick, 0)
            }),
            SchedulingPolicy::Fifo => self.select_non_preemptive(Self::next_fifo),
            SchedulingPolicy::DeficitRoundRobin => self.select_non_preemptive(Self::next_drr),
            SchedulingPolicy::WeightedFairQueueing => self.select_non_preemptive(Self::next_wfq),
        }
    }

    /// Preemptive selection: the active request minimising a total-order key.
    /// The key's final component is the request id, injected here, so the order
    /// is total and the choice unique — no `max_by_key` last-element ambiguity.
    fn select_min_key(&self, key: impl Fn(&StaticInfo) -> (u64, u64, u64)) -> Option<RequestId> {
        self.active_ids
            .iter()
            .filter_map(|&id| self.requests.get(&id).map(|info| (id, info)))
            .min_by_key(|(id, info)| {
                let (a, b, c) = key(info);
                (a, b, c, id.0)
            })
            .map(|(id, _)| id)
    }

    /// Non-preemptive selection: keep serving the current request until it
    /// completes, then pick the next via `pick`.
    fn select_non_preemptive(
        &mut self,
        pick: impl Fn(&mut Self) -> Option<RequestId>,
    ) -> Option<RequestId> {
        let keep_current = self.current.filter(|current| {
            self.runtime
                .get(current)
                .is_some_and(|r| r.remaining_ticks > 0)
        });
        if let Some(current) = keep_current {
            return Some(current);
        }
        let next = pick(self);
        self.current = next;
        next
    }

    /// FIFO pick: the active request with the earliest arrival, ties by id.
    fn next_fifo(&mut self) -> Option<RequestId> {
        self.active_ids
            .iter()
            .filter_map(|&id| self.requests.get(&id).map(|info| (id, info.arrival_tick)))
            .min_by_key(|(id, arrival)| (*arrival, id.0))
            .map(|(id, _)| id)
    }

    /// Deficit-Round-Robin pick: the next request the fair queue hands out.
    fn next_drr(&mut self) -> Option<RequestId> {
        self.fair.dequeue().map(|item| item.id)
    }

    /// Weighted-fair-queueing pick: the request with the smallest virtual finish.
    fn next_wfq(&mut self) -> Option<RequestId> {
        self.wfq.dequeue().map(|(id, _, _)| id)
    }

    /// Serve one tick of `id`, updating runtime state, the event log, and — on
    /// completion — the per-request and per-flow summaries.
    fn serve(&mut self, id: RequestId, tick: u64) {
        let Some(progress) = self.executor.run_round(tick, &[id]).into_iter().next() else {
            return;
        };
        let flow = self.requests.get(&id).map_or(0, |info| info.flow);
        let mut completed = false;
        if let Some(runtime) = self.runtime.get_mut(&id) {
            runtime.remaining_ticks = progress.remaining_ticks;
            runtime.service_ticks_done += 1;
            if progress.token_emitted {
                if runtime.first_token_tick.is_none() {
                    runtime.first_token_tick = Some(tick);
                }
                if let Some(previous) = runtime.last_token_tick {
                    let gap = tick - previous;
                    runtime.tpot_max = Some(runtime.tpot_max.map_or(gap, |m| m.max(gap)));
                }
                runtime.last_token_tick = Some(tick);
            }
            completed = progress.completed;
        }
        self.events.push(ServiceEvent {
            tick,
            request: id,
            token_emitted: progress.token_emitted,
            completed,
        });
        self.bump_flow_service(flow);
        if completed {
            self.on_complete(id, flow, tick);
        }
    }

    /// Increment a flow's served-tick counter.
    fn bump_flow_service(&mut self, flow: u32) {
        let weight = self.classes.get(&flow).map_or(1, |class| class.weight);
        let quantum = u64::from(weight) * self.config.base_quantum;
        let entry = self.per_flow.entry(flow).or_insert(FlowReport {
            class_id: flow,
            weight,
            quantum,
            served_ticks: 0,
            served_packets: 0,
        });
        entry.served_ticks += 1;
    }

    /// Finalise a request that has just completed.
    fn on_complete(&mut self, id: RequestId, flow: u32, tick: u64) {
        self.depth -= 1;
        self.active_ids.remove(&id);
        self.current = None;
        self.makespan = self.makespan.max(tick);
        if let Some(entry) = self.per_flow.get_mut(&flow) {
            entry.served_packets += 1;
        }
        let runtime = self.runtime.get(&id).cloned();
        if let (Some(report), Some(runtime)) = (self.per_request.get_mut(&id), runtime) {
            report.completion_tick = Some(tick);
            report.service_ticks = runtime.service_ticks_done;
            report.first_token_tick = runtime.first_token_tick;
            report.tpot_max = runtime.tpot_max;
            report.ttft = runtime
                .first_token_tick
                .map(|first| first - report.arrival_tick);
            report.lateness = lateness(tick, report.deadline_tick);
        }
    }

    /// Assemble the final report.
    fn build_report(&mut self) -> SchedulingReport {
        SchedulingReport {
            events: std::mem::take(&mut self.events),
            per_request: self.per_request.clone(),
            per_flow: self.per_flow.clone(),
            drr_sweeps: self.fair.sweeps().to_vec(),
            makespan: self.makespan,
        }
    }
}

/// A minimal request shim carrying only what [`AdmissionController::decide`]
/// reads (the `SLO` and the decode cost), rebuilt from [`StaticInfo`] so the
/// controller stays decoupled from the scheduler's internal state.
fn build_admission_shim(info: &StaticInfo) -> QueuedRequest {
    QueuedRequest {
        id: RequestId::new(0),
        class: PriorityClass::new(info.flow, info.weight).with_priority(info.priority),
        arrival_tick: info.arrival_tick,
        deadline: super::types::RequestDeadline::at(info.deadline_tick),
        output_tokens: 1,
        tick_cost: info.tick_cost,
        slo: info.slo,
    }
}

/// Lateness `completion - deadline`, or `None` for an undated request.
#[allow(clippy::cast_possible_wrap)] // ticks are bounded well below `i64::MAX`.
fn lateness(completion: u64, deadline: u64) -> Option<i64> {
    if deadline == u64::MAX {
        return None;
    }
    Some(completion as i64 - deadline as i64)
}
