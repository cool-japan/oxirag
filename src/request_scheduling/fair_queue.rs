//! Fair-queueing machinery: per-flow FIFO queues ([`RequestQueue`]), a Deficit
//! Round Robin scheduler ([`FairQueue`]), and a self-clocked weighted-fair-
//! queueing virtual clock ([`WeightedFairClock`]).
//!
//! Both disciplines arbitrate between *flows* ([`super::PriorityClass`]) so that
//! service is proportional to weight and no flow can be starved. They are the
//! answer to the failure mode that
//! [`SchedulingPolicy::StrictPriority`](super::SchedulingPolicy) exhibits by
//! design: a low-priority request waiting forever behind an unending stream of
//! higher-priority arrivals.
//!
//! # Deficit Round Robin
//!
//! Each backlogged flow keeps a **deficit counter**. When its turn comes round it
//! gains a **quantum** `weight * base_quantum` of credit and serves whole
//! requests from the head of its queue while the credit covers them; unspent
//! credit carries to its next turn. The two facts that make it fair *and*
//! starvation-free are theorems, not hopes, and this module measures both:
//!
//! - **Bounded deficit.** After any turn of a backlogged flow, its deficit is in
//!   `[0, Max)` where `Max` is the largest request size — because service stops
//!   only when the head request is larger than the remaining credit, and that
//!   head is at most `Max`.
//! - **Weight-proportional service.** A flow backlogged across `m` of its turns
//!   has served bytes `sent_i` with `m * Q_i - Max < sent_i <= m * Q_i` (the
//!   credit granted, minus at most one carried deficit). So two flows backlogged
//!   over the same `m` turns satisfy
//!   `|sent_i / Q_i - sent_j / Q_j| < Max / min(Q_i, Q_j)` — proportional to
//!   weight within one maximum request.
//!
//! # Self-clocked weighted fair queueing
//!
//! [`WeightedFairClock`] stamps each request with a **virtual finish time**
//! `max(virtual_now, flow_last_finish) + size / weight` and serves requests in
//! increasing finish order, `virtual_now` tracking the finish tag of the request
//! in service. This is the self-clocked (`SCFQ`) approximation of the ideal
//! generalized-processor-sharing schedule; it too gives weight-proportional
//! service and cannot starve a flow, because every tag is finite and
//! `virtual_now` never decreases.

use std::collections::{BTreeMap, VecDeque};

use super::types::{DrrSweepSnapshot, RequestId};

// ── RequestQueue ─────────────────────────────────────────────────────────────

/// One queued request as the fair-queueing layer sees it: an id, the flow it
/// belongs to, its service size in ticks, and its arrival tick (the FIFO and
/// tie-break key).
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct QueuedItem {
    /// The request id.
    pub id: RequestId,
    /// The flow (class) id.
    pub flow: u32,
    /// The request's total service time in ticks (its Deficit-Round-Robin
    /// "packet size").
    pub size: u64,
    /// The arrival tick, used to keep each flow's queue in FIFO order.
    pub arrival: u64,
}

/// Per-flow FIFO queues, keyed by flow id.
///
/// The shared substrate under both fair disciplines: requests within a flow are
/// served strictly in arrival order, and the disciplines decide only *which flow*
/// to serve from next.
#[derive(Debug, Clone, Default)]
pub struct RequestQueue {
    queues: BTreeMap<u32, VecDeque<QueuedItem>>,
}

impl RequestQueue {
    /// An empty set of per-flow queues.
    #[must_use]
    pub fn new() -> Self {
        Self {
            queues: BTreeMap::new(),
        }
    }

    /// Append `item` to the tail of its flow's queue.
    pub fn push(&mut self, item: QueuedItem) {
        self.queues.entry(item.flow).or_default().push_back(item);
    }

    /// The item at the head of `flow`'s queue, if any.
    #[must_use]
    pub fn front(&self, flow: u32) -> Option<QueuedItem> {
        self.queues.get(&flow).and_then(|q| q.front().copied())
    }

    /// Remove and return the item at the head of `flow`'s queue.
    pub fn pop_front(&mut self, flow: u32) -> Option<QueuedItem> {
        self.queues.get_mut(&flow).and_then(VecDeque::pop_front)
    }

    /// Whether `flow` currently has any queued request.
    #[must_use]
    pub fn is_backlogged(&self, flow: u32) -> bool {
        self.queues.get(&flow).is_some_and(|q| !q.is_empty())
    }

    /// The number of queued requests in `flow`.
    #[must_use]
    pub fn backlog_len(&self, flow: u32) -> usize {
        self.queues.get(&flow).map_or(0, VecDeque::len)
    }

    /// The total number of queued requests across all flows.
    #[must_use]
    pub fn total_len(&self) -> usize {
        self.queues.values().map(VecDeque::len).sum()
    }

    /// Whether every flow's queue is empty.
    #[must_use]
    pub fn is_empty(&self) -> bool {
        self.queues.values().all(VecDeque::is_empty)
    }

    /// The ids of flows with at least one queued request, in flow-id order.
    #[must_use]
    pub fn backlogged_flows(&self) -> Vec<u32> {
        self.queues
            .iter()
            .filter(|(_, q)| !q.is_empty())
            .map(|(&flow, _)| flow)
            .collect()
    }
}

// ── FairQueue (Deficit Round Robin) ──────────────────────────────────────────

/// A Deficit Round Robin scheduler over flows.
///
/// Feed it requests with [`enqueue`](Self::enqueue); pull the next request to run
/// to completion with [`dequeue`](Self::dequeue). It maintains the deficit counters,
/// the active (backlogged) flow list, per-flow served-byte and served-packet
/// counters, and — for the fairness test — a snapshot at the end of every
/// round-robin sweep.
#[derive(Debug, Clone)]
pub struct FairQueue {
    /// The per-flow FIFO queues.
    queue: RequestQueue,
    /// Per-flow deficit counters.
    deficits: BTreeMap<u32, u64>,
    /// Per-flow quanta (`weight * base_quantum`).
    quanta: BTreeMap<u32, u64>,
    /// The active list: backlogged flows in round-robin order.
    active: VecDeque<u32>,
    /// Cumulative committed service (bytes) per flow.
    served_bytes: BTreeMap<u32, u64>,
    /// Cumulative completed packets per flow.
    served_packets: BTreeMap<u32, u64>,
    /// The flow currently mid-turn (quantum already granted this turn), if any.
    current_flow: Option<u32>,
    /// The quantum handed to a flow that arrives without a pre-registered
    /// quantum; always positive, so every backlogged flow makes progress and
    /// [`dequeue`](Self::dequeue) can never spin forever.
    default_quantum: u64,
    /// The largest request size ever enqueued (the `Max` of the deficit bound).
    max_packet: u64,
    /// The flow that anchors sweep detection (the first ever to take a turn).
    anchor: Option<u32>,
    /// How many turns the anchor flow has completed.
    anchor_turns: u64,
    /// Sweep snapshots, one per completed round-robin sweep.
    sweeps: Vec<DrrSweepSnapshot>,
}

impl FairQueue {
    /// A Deficit Round Robin scheduler whose flow `f` has quantum
    /// `weight(f) * base_quantum`. `default_quantum` (which must be positive) is
    /// handed to any flow that arrives without a pre-registered quantum.
    #[must_use]
    pub fn new(quanta: BTreeMap<u32, u64>, default_quantum: u64) -> Self {
        let deficits = quanta.keys().map(|&flow| (flow, 0)).collect();
        let served_bytes = quanta.keys().map(|&flow| (flow, 0)).collect();
        let served_packets = quanta.keys().map(|&flow| (flow, 0)).collect();
        Self {
            queue: RequestQueue::new(),
            deficits,
            quanta,
            active: VecDeque::new(),
            served_bytes,
            served_packets,
            current_flow: None,
            default_quantum: default_quantum.max(1),
            max_packet: 0,
            anchor: None,
            anchor_turns: 0,
            sweeps: Vec::new(),
        }
    }

    /// The quantum of a flow (`0` for an unknown flow).
    #[must_use]
    pub fn quantum(&self, flow: u32) -> u64 {
        self.quanta.get(&flow).copied().unwrap_or(0)
    }

    /// The largest request size enqueued so far — the `Max` of the deficit bound.
    #[must_use]
    pub fn max_packet(&self) -> u64 {
        self.max_packet
    }

    /// Cumulative committed service (bytes) for `flow`.
    #[must_use]
    pub fn served_bytes(&self, flow: u32) -> u64 {
        self.served_bytes.get(&flow).copied().unwrap_or(0)
    }

    /// The current deficit counter of `flow`.
    #[must_use]
    pub fn deficit(&self, flow: u32) -> u64 {
        self.deficits.get(&flow).copied().unwrap_or(0)
    }

    /// The sweep snapshots recorded so far.
    #[must_use]
    pub fn sweeps(&self) -> &[DrrSweepSnapshot] {
        &self.sweeps
    }

    /// Whether any flow has a queued request.
    #[must_use]
    pub fn is_empty(&self) -> bool {
        self.queue.is_empty()
    }

    /// Enqueue a request into its flow, joining the active list if the flow was
    /// idle. An unknown flow is given a zero quantum (it will never be served),
    /// which never happens through the scheduler because every flow is registered
    /// from the workload up front.
    pub fn enqueue(&mut self, item: QueuedItem) {
        self.max_packet = self.max_packet.max(item.size);
        let was_backlogged = self.queue.is_backlogged(item.flow);
        let flow = item.flow;
        self.queue.push(item);
        self.quanta.entry(flow).or_insert(self.default_quantum);
        self.deficits.entry(flow).or_insert(0);
        self.served_bytes.entry(flow).or_insert(0);
        self.served_packets.entry(flow).or_insert(0);
        // A flow that was idle (and is not the one currently mid-turn) rejoins
        // the round-robin at the tail with a fresh, zero deficit.
        if !was_backlogged && self.current_flow != Some(flow) && !self.active.contains(&flow) {
            self.deficits.insert(flow, 0);
            self.active.push_back(flow);
        }
    }

    /// The next request to run to completion, or `None` when nothing is
    /// backlogged. Advances the Deficit-Round-Robin state: grants quanta on turn
    /// boundaries, spends deficit on served requests, and records a sweep snapshot
    /// each time the anchor flow comes round again.
    pub fn dequeue(&mut self) -> Option<QueuedItem> {
        loop {
            if let Some(flow) = self.current_flow {
                // Mid-turn: try to serve another request from this flow while its
                // head fits the remaining credit.
                let servable = self
                    .queue
                    .front(flow)
                    .filter(|item| item.size <= self.deficit(flow));
                if let Some(item) = servable {
                    let served = self.queue.pop_front(flow)?;
                    *self.deficits.entry(flow).or_insert(0) -= item.size;
                    *self.served_bytes.entry(flow).or_insert(0) += item.size;
                    *self.served_packets.entry(flow).or_insert(0) += 1;
                    if !self.queue.is_backlogged(flow) {
                        // Flow drained: leave the active list, reset deficit, end
                        // the turn.
                        self.remove_active(flow);
                        self.deficits.insert(flow, 0);
                        self.current_flow = None;
                    }
                    return Some(served);
                }
                // Head too big for the remaining credit, or flow drained: the turn
                // ends. A still-backlogged flow keeps its deficit and goes to the
                // tail; a drained one resets to zero.
                if self.queue.is_backlogged(flow) {
                    self.rotate_active(flow);
                } else {
                    self.remove_active(flow);
                    self.deficits.insert(flow, 0);
                }
                self.current_flow = None;
            }

            // Start a fresh turn on the flow at the head of the active list.
            let flow = *self.active.front()?;
            self.begin_turn(flow);
            self.current_flow = Some(flow);
        }
    }

    /// Grant a flow its quantum at the start of a turn, updating sweep accounting.
    fn begin_turn(&mut self, flow: u32) {
        if self.anchor.is_none() {
            self.anchor = Some(flow);
        }
        if self.anchor == Some(flow) {
            // The anchor is starting a new turn. If it has already completed at
            // least one turn, a full sweep has just finished; snapshot it.
            if self.anchor_turns >= 1 {
                self.record_sweep(self.anchor_turns - 1);
            }
            self.anchor_turns += 1;
        }
        let quantum = self.quantum(flow);
        *self.deficits.entry(flow).or_insert(0) += quantum;
    }

    /// Snapshot cumulative served bytes and deficits at a sweep boundary.
    fn record_sweep(&mut self, sweep_index: u64) {
        self.sweeps.push(DrrSweepSnapshot {
            sweep_index,
            served_bytes: self.served_bytes.clone(),
            deficits: self.deficits.clone(),
            backlogged: self.queue.backlogged_flows(),
        });
    }

    /// Move `flow` from the head of the active list to the tail.
    fn rotate_active(&mut self, flow: u32) {
        if self.active.front() == Some(&flow) {
            self.active.pop_front();
            self.active.push_back(flow);
        }
    }

    /// Remove `flow` from the active list wherever it sits.
    fn remove_active(&mut self, flow: u32) {
        self.active.retain(|&f| f != flow);
    }

    /// The completed-packet count for `flow`.
    #[must_use]
    pub fn served_packets(&self, flow: u32) -> u64 {
        self.served_packets.get(&flow).copied().unwrap_or(0)
    }
}

// ── WeightedFairClock (self-clocked WFQ) ─────────────────────────────────────

/// A ready request stamped with its virtual finish time.
#[derive(Debug, Clone, Copy, PartialEq)]
struct FinishTagged {
    /// The virtual finish time (`max(virtual_now, flow_last) + size / weight`).
    finish: f64,
    /// The arrival tick — first tie-break.
    arrival: u64,
    /// The request id — final, total tie-break.
    id: RequestId,
    /// The flow id.
    flow: u32,
    /// The request's service size in ticks.
    size: u64,
}

/// A self-clocked weighted-fair-queueing virtual clock.
///
/// Feed it ready requests with [`enqueue`](Self::enqueue); pull the one with the
/// smallest virtual finish time with [`dequeue`](Self::dequeue). `virtual_now` tracks
/// the finish tag of the request last put into service, which is what makes the
/// clock *self*-clocked — it needs no simulation of the ideal fluid schedule.
#[derive(Debug, Clone, Default)]
pub struct WeightedFairClock {
    /// The virtual time, equal to the finish tag of the request in service.
    virtual_now: f64,
    /// Per-flow last-assigned finish tag.
    flow_last_finish: BTreeMap<u32, f64>,
    /// Ready, tagged requests awaiting service.
    pending: Vec<FinishTagged>,
}

impl WeightedFairClock {
    /// A fresh virtual clock at virtual time zero.
    #[must_use]
    pub fn new() -> Self {
        Self::default()
    }

    /// Stamp a request with its virtual finish time and enqueue it.
    ///
    /// The finish tag is `max(virtual_now, flow_last_finish) + size / weight`; the
    /// `max` is what prevents a flow that has been idle from accumulating credit
    /// and then bursting ahead of everyone else.
    #[allow(clippy::cast_precision_loss)] // sizes and weights are small integers.
    pub fn enqueue(&mut self, id: RequestId, flow: u32, size: u64, weight: u32, arrival: u64) {
        let weight = f64::from(weight.max(1));
        let start = self
            .virtual_now
            .max(self.flow_last_finish.get(&flow).copied().unwrap_or(0.0));
        let finish = start + (size as f64) / weight;
        self.flow_last_finish.insert(flow, finish);
        self.pending.push(FinishTagged {
            finish,
            arrival,
            id,
            flow,
            size,
        });
    }

    /// The next request to serve — the smallest virtual finish time, ties broken
    /// by arrival then id (a total order, so the choice is unique and
    /// deterministic). Advances `virtual_now` to the served request's finish tag.
    pub fn dequeue(&mut self) -> Option<(RequestId, u32, u64)> {
        let best = self.pending.iter().enumerate().min_by(|(_, a), (_, b)| {
            a.finish
                .total_cmp(&b.finish)
                .then(a.arrival.cmp(&b.arrival))
                .then(a.id.cmp(&b.id))
        })?;
        let index = best.0;
        let tagged = self.pending.swap_remove(index);
        self.virtual_now = tagged.finish;
        Some((tagged.id, tagged.flow, tagged.size))
    }

    /// Whether any request is waiting.
    #[must_use]
    pub fn is_empty(&self) -> bool {
        self.pending.is_empty()
    }

    /// The current virtual time.
    #[must_use]
    pub fn virtual_now(&self) -> f64 {
        self.virtual_now
    }
}
