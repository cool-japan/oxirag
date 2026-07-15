//! Tests for `request_scheduling`.
//!
//! These are **measurements against independent ground truth**, not
//! self-consistency checks. The headline is EDF optimality pinned against a
//! brute force over all `n!` permutations (Jackson's rule); the rest hold the
//! implementation to the exact theorems it claims — the Deficit-Round-Robin
//! one-max-packet fairness bound, exact starvation-freedom under an adversarial
//! flood (with the naive priority queue shown to starve as an ablation), `SLO`
//! admission soundness *and* non-conservatism with the numbers reported, both
//! sides of the backpressure threshold, and tick-for-tick determinism.

#![allow(
    clippy::float_cmp,
    clippy::cast_precision_loss,
    clippy::cast_possible_truncation,
    clippy::cast_sign_loss,
    clippy::cast_possible_wrap,
    clippy::too_many_lines,
    clippy::similar_names,
    clippy::unreadable_literal,
    clippy::many_single_char_names,
    clippy::needless_range_loop,
    clippy::doc_markdown,
    clippy::items_after_statements
)]

use super::admission::{AdmissionController, Backpressure};
use super::engine::RequestScheduler;
use super::executor::{SchedulerExecutor, StaticSchedulerExecutor};
use super::fair_queue::{QueuedItem, RequestQueue};
use super::rng::SchedulerRng;
use super::types::{
    PriorityClass, QueuedRequest, RejectionReason, RequestDeadline, RequestId, RequestPriority,
    RequestSchedulingConfig, SchedulingOutcome, SchedulingPolicy, SchedulingReport, SloTarget,
};

// ── Shared helpers ───────────────────────────────────────────────────────────

/// Build an executor matching the requests and run the workload to completion.
fn run_workload(config: RequestSchedulingConfig, requests: Vec<QueuedRequest>) -> SchedulingReport {
    let executor = StaticSchedulerExecutor::from_requests(&requests);
    let mut scheduler = RequestScheduler::new(config, executor).expect("valid config");
    for request in requests {
        scheduler.enqueue(request).expect("distinct ids");
    }
    scheduler
        .run()
        .expect("workload drains within its tick budget")
}

/// A single-flow request with unit decode cost (so `service_ticks == length`).
fn unit_request(id: u64, flow: u32, arrival: u64, length: u32, deadline: u64) -> QueuedRequest {
    QueuedRequest::new(
        RequestId::new(id),
        PriorityClass::new(flow, 1),
        arrival,
        length,
    )
    .with_deadline(RequestDeadline::at(deadline))
}

/// Apply `f` to every permutation of `items` (Heap-style recursion).
fn for_each_permutation<F: FnMut(&[usize])>(items: &mut Vec<usize>, k: usize, f: &mut F) {
    if k == items.len() {
        f(items);
        return;
    }
    for i in k..items.len() {
        items.swap(k, i);
        for_each_permutation(items, k + 1, f);
        items.swap(k, i);
    }
}

/// The minimum, over all `n!` orderings, of the maximum lateness of a
/// non-preemptive back-to-back schedule of `jobs = (length, deadline)` all
/// released at tick 0.
///
/// This is the brute-force ground truth for Jackson's rule — computed by a
/// method that has nothing to do with the scheduler. Completion of a job is its
/// 0-based *last-service tick* (`start + length - 1`), matching exactly the tick
/// convention the scheduler measures lateness in.
fn brute_force_min_max_lateness(jobs: &[(u64, u64)]) -> i64 {
    let mut indices: Vec<usize> = (0..jobs.len()).collect();
    let mut best = i64::MAX;
    for_each_permutation(&mut indices, 0, &mut |perm| {
        let mut cursor: i64 = 0;
        let mut worst = i64::MIN;
        for &job in perm {
            let (length, deadline) = jobs[job];
            let completion = cursor + length as i64 - 1;
            worst = worst.max(completion - deadline as i64);
            cursor += length as i64;
        }
        best = best.min(worst);
    });
    best
}

// ── (a) THE headline test: EDF optimality vs brute force ─────────────────────

#[test]
fn edf_maximum_lateness_matches_brute_force_over_all_permutations() {
    let mut instances_checked = 0_u32;
    let mut nontrivial = 0_u32; // instances where some ordering is actually late
    for seed in 0..48_u64 {
        let mut rng = SchedulerRng::new(seed);
        let n = 2 + (seed as usize % 7); // n in 2..=8
        // Random lengths, then deadlines spread around the total processing time.
        let lengths: Vec<u64> = (0..n).map(|_| rng.next_range(1, 5)).collect();
        let total: u64 = lengths.iter().sum();
        let jobs: Vec<(u64, u64)> = lengths
            .iter()
            .map(|&length| (length, rng.next_range(1, total + 2)))
            .collect();

        let requests: Vec<QueuedRequest> = jobs
            .iter()
            .enumerate()
            .map(|(i, &(length, deadline))| unit_request(i as u64, 0, 0, length as u32, deadline))
            .collect();
        let config = RequestSchedulingConfig::new(SchedulingPolicy::EarliestDeadlineFirst)
            .with_admission(false)
            .with_max_queue_depth(10_000);
        let report = run_workload(config, requests);

        let edf = report.max_lateness().expect("every request completes");
        let optimal = brute_force_min_max_lateness(&jobs);
        assert_eq!(
            edf, optimal,
            "EDF lateness {edf} != brute-force optimum {optimal} (seed {seed}, jobs {jobs:?})"
        );
        instances_checked += 1;
        if optimal > 0 {
            nontrivial += 1;
        }
    }
    assert!(
        instances_checked >= 40,
        "expected a broad sweep of instances"
    );
    // If nothing was ever late the test could pass with a broken scheduler that
    // always reports a huge negative lateness; insist the sweep exercised the
    // regime where ordering genuinely matters.
    assert!(
        nontrivial >= 5,
        "sweep never produced a late-schedule instance; brute force was untested"
    );
}

// ── (b) DRR fairness bound ───────────────────────────────────────────────────

#[test]
fn drr_service_is_weight_proportional_within_one_max_packet() {
    const MAX_PACKET: u64 = 6;
    const BASE_QUANTUM: u64 = MAX_PACKET; // quantum(f) = weight(f) * MAX_PACKET
    let weights = [1_u32, 2, 3];
    let flows = [0_u32, 1, 2];

    // Three flows, each deeply backlogged with random-size requests, all present
    // at tick 0 so every flow is backlogged from the first sweep.
    let mut rng = SchedulerRng::new(20260712);
    let mut requests = Vec::new();
    for (slot, (&flow, &weight)) in flows.iter().zip(weights.iter()).enumerate() {
        for k in 0..60_u64 {
            let id = (slot as u64) * 1000 + k;
            let size = rng.next_range(1, MAX_PACKET) as u32;
            requests.push(QueuedRequest::new(
                RequestId::new(id),
                PriorityClass::new(flow, weight),
                0,
                size,
            ));
        }
    }

    let config = RequestSchedulingConfig::new(SchedulingPolicy::DeficitRoundRobin)
        .with_admission(false)
        .with_base_quantum(BASE_QUANTUM)
        .with_max_queue_depth(10_000);
    let report = run_workload(config, requests);

    // Pick the latest sweep at which all three flows were still backlogged.
    let snapshot = report
        .drr_sweeps
        .iter()
        .filter(|s| flows.iter().all(|f| s.backlogged.contains(f)))
        .max_by_key(|s| s.sweep_index)
        .expect("at least one sweep with all flows backlogged");
    let m = snapshot.sweep_index + 1; // turns per flow at this sweep
    assert!(m >= 2, "need a few sweeps to exercise the deficit");

    let quantum = |flow: u32| report.per_flow[&flow].quantum;
    let q_min = flows
        .iter()
        .map(|&f| quantum(f))
        .min()
        .expect("flows exist");

    for &flow in &flows {
        let q = quantum(flow);
        let served = snapshot.served_bytes[&flow];
        let deficit = snapshot.deficits[&flow];
        // Deficit is bounded by one max packet (the DRR invariant).
        assert!(
            deficit < MAX_PACKET,
            "deficit {deficit} of flow {flow} not < Max {MAX_PACKET}"
        );
        // Exact accounting: granted - carried = served.
        assert_eq!(
            served,
            m * q - deficit,
            "flow {flow}: served {served} != m*Q - deficit ({m}*{q} - {deficit})"
        );
        // The one-max-packet service bound.
        assert!(
            m * q - MAX_PACKET < served && served <= m * q,
            "flow {flow}: served {served} outside ({}, {}]",
            m * q - MAX_PACKET,
            m * q
        );
    }

    // Pairwise: normalised service is equal across flows within Max / Q_min.
    let bound = MAX_PACKET as f64 / q_min as f64;
    for &a in &flows {
        for &b in &flows {
            let na = snapshot.served_bytes[&a] as f64 / quantum(a) as f64;
            let nb = snapshot.served_bytes[&b] as f64 / quantum(b) as f64;
            assert!(
                (na - nb).abs() < bound + 1e-9,
                "flows {a},{b}: |{na} - {nb}| not < {bound}"
            );
        }
    }
}

// ── (c) Starvation freedom (exact) + naive-priority ablation ─────────────────

/// The victim's last completion tick under `policy`, for `flood` competing
/// requests, plus the flood's total service work.
fn starvation_run(
    policy: SchedulingPolicy,
    flood: u64,
    victim_priority: RequestPriority,
    flood_priority: RequestPriority,
) -> (u64, u64) {
    const P: u64 = 3; // victim depth
    const S: u32 = 4; // uniform packet size
    let victim_class = PriorityClass::new(0, 1).with_priority(victim_priority);
    let flood_class = PriorityClass::new(1, 1).with_priority(flood_priority);

    let mut requests = Vec::new();
    for i in 0..P {
        requests.push(QueuedRequest::new(
            RequestId::new(i),
            victim_class.clone(),
            0,
            S,
        ));
    }
    for j in 0..flood {
        requests.push(QueuedRequest::new(
            RequestId::new(1000 + j),
            flood_class.clone(),
            0,
            S,
        ));
    }

    let config = RequestSchedulingConfig::new(policy)
        .with_admission(false)
        .with_base_quantum(u64::from(S))
        .with_max_queue_depth(100_000);
    let report = run_workload(config, requests);

    let victim_last = (0..P)
        .filter_map(|i| report.completion_tick(RequestId::new(i)))
        .max()
        .expect("victim requests complete");
    (victim_last, flood * u64::from(S))
}

#[test]
fn drr_prevents_the_starvation_a_naive_priority_queue_suffers() {
    const P: u64 = 3;
    const S: u64 = 4;

    // Ablation — STRICT PRIORITY starves the low-priority victim: its wait grows
    // with the size of the high-priority flood and is at least the whole flood.
    let (victim_strict_small, flood_work_small) = starvation_run(
        SchedulingPolicy::StrictPriority,
        10,
        RequestPriority::BULK,
        RequestPriority::REALTIME,
    );
    let (victim_strict_large, flood_work_large) = starvation_run(
        SchedulingPolicy::StrictPriority,
        20,
        RequestPriority::BULK,
        RequestPriority::REALTIME,
    );
    assert!(
        victim_strict_small >= flood_work_small,
        "strict priority should make the victim wait behind the whole flood: {victim_strict_small} < {flood_work_small}"
    );
    assert!(
        victim_strict_large > victim_strict_small,
        "victim wait must grow with the flood under strict priority: {victim_strict_large} !> {victim_strict_small}"
    );
    let _ = flood_work_large;

    // OURS — DEFICIT ROUND ROBIN is starvation-free: the victim completes within
    // a bound that depends only on its own depth, not on the flood at all.
    let (victim_drr_small, _) = starvation_run(
        SchedulingPolicy::DeficitRoundRobin,
        10,
        RequestPriority::NORMAL,
        RequestPriority::NORMAL,
    );
    let (victim_drr_large, _) = starvation_run(
        SchedulingPolicy::DeficitRoundRobin,
        20,
        RequestPriority::NORMAL,
        RequestPriority::NORMAL,
    );
    assert_eq!(
        victim_drr_small, victim_drr_large,
        "DRR victim completion must be independent of flood size: {victim_drr_small} != {victim_drr_large}"
    );
    // Exact bound: 2 flows in round robin, victim depth P, uniform size S ⇒ the
    // victim's last packet completes by tick 2*P*S.
    let bound = 2 * P * S;
    assert!(
        victim_drr_small <= bound,
        "DRR victim completion {victim_drr_small} exceeds the starvation bound {bound}"
    );
    // And it is dramatically sooner than strict priority under the same flood.
    assert!(
        victim_drr_small < victim_strict_small,
        "DRR ({victim_drr_small}) should beat strict priority ({victim_strict_small})"
    );
}

// ── (d) SLO admission: sound and not conservative ────────────────────────────

#[test]
fn slo_admission_is_sound_and_admits_feasible_requests_at_the_boundary() {
    // FIFO makes the TTFT estimate exact: nothing can jump ahead of a queued
    // request, so estimated == actual delay. All arrive at tick 0.
    let class = PriorityClass::new(0, 1);
    let make = |id: u64, len: u32, cost: u32, ttft: u64, tpot: u64| {
        QueuedRequest::new(RequestId::new(id), class.clone(), 0, len)
            .with_tick_cost(cost)
            .with_slo(SloTarget::new(ttft, tpot))
    };

    // r0: len 3, TTFT budget 0 (est 0) -> admit.
    // r1: len 4, TTFT budget 3 == est 3 (boundary) -> admit, not rejected.
    // r2: len 2, TTFT budget 6 < est 7 -> reject (TTFT infeasible).
    // r3: len 2, cost 3 > TPOT budget 2 -> reject (TPOT infeasible).
    // r4: len 2, cost 2 == TPOT budget 2 (boundary), TTFT budget 100 -> admit.
    let requests = vec![
        make(0, 3, 1, 0, 100),
        make(1, 4, 1, 3, 100),
        make(2, 2, 1, 6, 100),
        make(3, 2, 3, 100, 2),
        make(4, 2, 2, 100, 2),
    ];

    let config = RequestSchedulingConfig::new(SchedulingPolicy::Fifo)
        .with_admission(true)
        .with_max_queue_depth(10_000);
    let report = run_workload(config, requests);
    let r = |id: u64| report.per_request[&RequestId::new(id)].clone();

    // The exact rejections, with the numbers the controller computed.
    assert_eq!(
        r(2).rejection,
        Some(RejectionReason::TtftInfeasible {
            estimated: 7,
            budget: 6
        })
    );
    assert_eq!(
        r(3).rejection,
        Some(RejectionReason::TpotInfeasible {
            required: 3,
            budget: 2
        })
    );

    // NON-CONSERVATIVE: the two boundary requests (r1 at TTFT==budget, r4 at
    // TPOT==budget) are admitted, not rejected out of excess caution.
    assert!(r(1).admitted, "boundary TTFT request must be admitted");
    assert!(r(4).admitted, "boundary TPOT request must be admitted");

    // The admitted set is exactly {r0, r1, r4}: nothing infeasible slipped
    // through (over-admission would show up here), nothing feasible was shed.
    assert_eq!(
        report.admitted_ids(),
        vec![RequestId::new(0), RequestId::new(1), RequestId::new(4)]
    );

    // SOUND: every admitted request meets both budgets under actual execution.
    let admitted = [(0_u64, 100_u64, 100_u64), (1, 3, 100), (4, 100, 2)];
    for &(id, ttft_budget, tpot_budget) in &admitted {
        let report_row = r(id);
        assert!(report_row.admitted);
        let ttft = report_row.ttft.expect("admitted request produced a token");
        assert!(
            ttft <= ttft_budget,
            "r{id} measured TTFT {ttft} exceeds budget {ttft_budget}"
        );
        if let Some(tpot) = report_row.tpot_max {
            assert!(
                tpot <= tpot_budget,
                "r{id} measured TPOT {tpot} exceeds budget {tpot_budget}"
            );
        }
    }

    // Report the boundary numbers explicitly: r1 lands its first token exactly at
    // its TTFT budget, and r4's decode cadence lands exactly at its TPOT budget.
    assert_eq!(r(0).ttft, Some(0));
    assert_eq!(r(1).ttft, Some(3)); // == budget 3
    assert_eq!(r(4).ttft, Some(7));
    assert_eq!(r(4).tpot_max, Some(2)); // == budget 2
}

#[test]
fn admission_controller_gates_in_the_documented_order() {
    // Backpressure is checked before SLO: an over-full queue is shed regardless.
    let controller = AdmissionController::new(4, true);
    let tight = QueuedRequest::new(RequestId::new(0), PriorityClass::new(0, 1), 0, 1)
        .with_tick_cost(9)
        .with_slo(SloTarget::new(0, 1));
    // depth == capacity -> backpressure, even though the SLO is also violated.
    assert_eq!(
        controller.decide(&tight, 4, 1_000),
        SchedulingOutcome::Rejected(RejectionReason::Backpressure {
            depth: 4,
            capacity: 4
        })
    );
    // Below capacity, TPOT (intrinsic) is reported before TTFT (load-dependent).
    assert_eq!(
        controller.decide(&tight, 0, 1_000),
        SchedulingOutcome::Rejected(RejectionReason::TpotInfeasible {
            required: 9,
            budget: 1
        })
    );
}

// ── (e) Backpressure: both sides of the threshold ────────────────────────────

#[test]
fn backpressure_sheds_at_capacity_and_admits_below_it() {
    const DEPTH: usize = 5;
    let class = PriorityClass::new(0, 1);
    // DEPTH + 1 long requests all arriving at tick 0. The first DEPTH fill the
    // queue; the next one arrives to a full queue and is shed.
    let requests: Vec<QueuedRequest> = (0..=DEPTH as u64)
        .map(|id| QueuedRequest::new(RequestId::new(id), class.clone(), 0, 50))
        .collect();

    let config = RequestSchedulingConfig::new(SchedulingPolicy::Fifo)
        .with_admission(false) // isolate backpressure from the SLO gates
        .with_max_queue_depth(DEPTH);
    let report = run_workload(config, requests);

    // Below threshold: the request that arrived at depth DEPTH-1 is admitted.
    assert!(
        report.per_request[&RequestId::new(DEPTH as u64 - 1)].admitted,
        "request arriving below capacity must be admitted"
    );
    // At threshold: the request that arrived at depth DEPTH is shed.
    assert_eq!(
        report.per_request[&RequestId::new(DEPTH as u64)].rejection,
        Some(RejectionReason::Backpressure {
            depth: DEPTH,
            capacity: DEPTH
        })
    );
    assert_eq!(report.admitted_ids().len(), DEPTH);
}

// ── (f) Determinism ──────────────────────────────────────────────────────────

/// A pseudo-random multi-flow workload, a pure function of `seed`.
fn random_workload(seed: u64, n: usize) -> Vec<QueuedRequest> {
    let mut rng = SchedulerRng::new(seed);
    let mut requests = Vec::new();
    for i in 0..n as u64 {
        let flow = rng.next_range(0, 2) as u32;
        let weight = rng.next_range(1, 3) as u32;
        let priority = RequestPriority::new(rng.next_range(0, 255) as u8);
        let class = PriorityClass::new(flow, weight).with_priority(priority);
        let arrival = rng.next_range(0, 12);
        let length = rng.next_range(1, 6) as u32;
        let deadline = arrival + rng.next_range(1, 40);
        requests.push(
            QueuedRequest::new(RequestId::new(i), class, arrival, length)
                .with_deadline(RequestDeadline::at(deadline)),
        );
    }
    requests
}

#[test]
fn same_seed_and_workload_produce_an_identical_schedule() {
    for policy in [
        SchedulingPolicy::EarliestDeadlineFirst,
        SchedulingPolicy::DeficitRoundRobin,
        SchedulingPolicy::WeightedFairQueueing,
        SchedulingPolicy::Fifo,
        SchedulingPolicy::StrictPriority,
    ] {
        let config = || {
            RequestSchedulingConfig::new(policy)
                .with_admission(false)
                .with_base_quantum(8)
                .with_max_queue_depth(10_000)
        };
        // Two independent regenerations of the workload from the same seed.
        let report_a = run_workload(config(), random_workload(7, 40));
        let report_b = run_workload(config(), random_workload(7, 40));
        assert_eq!(
            report_a.events, report_b.events,
            "policy {policy:?}: event logs diverged tick-for-tick"
        );
        assert_eq!(report_a.completion_order(), report_b.completion_order());
        assert_eq!(report_a.makespan, report_b.makespan);
    }
}

// ── Weighted fair queueing ───────────────────────────────────────────────────

#[test]
fn wfq_interleaves_service_in_proportion_to_weight_and_never_starves() {
    // Flow A weight 1, flow B weight 3, each with six uniform requests present at
    // tick 0. During the backlogged phase WFQ should serve B three times for
    // every A, and both flows must fully drain (no starvation).
    let class_a = PriorityClass::new(0, 1);
    let class_b = PriorityClass::new(1, 3);
    let mut requests = Vec::new();
    for i in 0..6 {
        requests.push(QueuedRequest::new(RequestId::new(i), class_a.clone(), 0, 3));
    }
    for i in 0..6 {
        requests.push(QueuedRequest::new(
            RequestId::new(100 + i),
            class_b.clone(),
            0,
            3,
        ));
    }

    let config = RequestSchedulingConfig::new(SchedulingPolicy::WeightedFairQueueing)
        .with_admission(false)
        .with_max_queue_depth(10_000);
    let report = run_workload(config, requests);

    // Starvation-free: every request from both flows completes.
    assert_eq!(report.per_flow[&0].served_packets, 6);
    assert_eq!(report.per_flow[&1].served_packets, 6);

    // Proportional interleaving: over the first eight completions (two full
    // virtual-time rounds), the weight-3 flow is served three times as often.
    let order = report.completion_order();
    let flow_of = |id: RequestId| report.per_request[&id].class_id;
    let first_eight = &order[..8];
    let b_count = first_eight.iter().filter(|&&id| flow_of(id) == 1).count();
    let a_count = first_eight.iter().filter(|&&id| flow_of(id) == 0).count();
    assert_eq!(
        (b_count, a_count),
        (6, 2),
        "expected a 3:1 interleave over the first eight completions, got B={b_count} A={a_count}"
    );
}

// ── Unit tests for the moving parts ──────────────────────────────────────────

#[test]
fn static_executor_emits_tokens_on_the_two_phase_schedule() {
    // 3 output tokens at decode cost 2 ⇒ total 1 + 2*2 = 5 ticks; tokens at served
    // ticks 1, 3, 5.
    let id = RequestId::new(0);
    let mut executor = StaticSchedulerExecutor::new().with_request(id, 3, 2);
    assert_eq!(executor.total_ticks(id), Some(5));
    let mut token_ticks = Vec::new();
    for tick in 0..5 {
        let progress = executor.run_round(tick, &[id]);
        let step = progress[0];
        if step.token_emitted {
            token_ticks.push(tick);
        }
        assert_eq!(step.completed, tick == 4);
    }
    assert_eq!(token_ticks, vec![0, 2, 4]); // 1st, 3rd, 5th served tick
    // Serving a finished request is a no-op, not a panic.
    let after = executor.run_round(99, &[id]);
    assert!(after[0].completed && !after[0].token_emitted);
}

#[test]
fn service_ticks_matches_the_two_phase_formula() {
    let request =
        QueuedRequest::new(RequestId::new(0), PriorityClass::new(0, 1), 0, 4).with_tick_cost(3);
    // 1 + (4 - 1) * 3 = 10
    assert_eq!(request.service_ticks(), 10);
    let single = QueuedRequest::new(RequestId::new(1), PriorityClass::new(0, 1), 0, 1);
    assert_eq!(single.service_ticks(), 1);
}

#[test]
fn config_validation_rejects_degenerate_settings() {
    assert!(RequestSchedulingConfig::default().validate().is_ok());
    assert!(
        RequestSchedulingConfig::default()
            .with_base_quantum(0)
            .validate()
            .is_err()
    );
    assert!(
        RequestSchedulingConfig::default()
            .with_max_queue_depth(0)
            .validate()
            .is_err()
    );
}

#[test]
fn duplicate_request_ids_are_rejected() {
    let config = RequestSchedulingConfig::default();
    let executor = StaticSchedulerExecutor::new();
    let mut scheduler = RequestScheduler::new(config, executor).expect("valid config");
    let class = PriorityClass::new(0, 1);
    scheduler
        .enqueue(QueuedRequest::new(RequestId::new(0), class.clone(), 0, 1))
        .expect("first enqueue");
    assert!(
        scheduler
            .enqueue(QueuedRequest::new(RequestId::new(0), class, 1, 1))
            .is_err()
    );
}

#[test]
fn request_queue_is_fifo_within_a_flow() {
    let mut queue = RequestQueue::new();
    for id in 0..3 {
        queue.push(QueuedItem {
            id: RequestId::new(id),
            flow: 7,
            size: 1,
            arrival: id,
        });
    }
    assert_eq!(queue.backlog_len(7), 3);
    assert_eq!(
        queue.pop_front(7).map(|item| item.id),
        Some(RequestId::new(0))
    );
    assert_eq!(
        queue.pop_front(7).map(|item| item.id),
        Some(RequestId::new(1))
    );
    assert!(queue.is_backlogged(7));
    assert_eq!(queue.backlogged_flows(), vec![7]);
}

#[test]
fn backpressure_admits_strictly_below_capacity() {
    let gate = Backpressure::new(3);
    assert!(gate.admits(0));
    assert!(gate.admits(2));
    assert!(!gate.admits(3));
    assert!(gate.rejection(2).is_none());
    assert_eq!(
        gate.rejection(3),
        Some(RejectionReason::Backpressure {
            depth: 3,
            capacity: 3
        })
    );
}

#[test]
fn scheduler_rng_range_is_reproducible_and_in_bounds() {
    let mut a = SchedulerRng::new(99);
    let mut b = SchedulerRng::new(99);
    for _ in 0..1000 {
        let low = 5;
        let high = 17;
        let x = a.next_range(low, high);
        assert_eq!(x, b.next_range(low, high));
        assert!((low..=high).contains(&x));
    }
    assert_eq!(SchedulerRng::new(3).next_range(10, 10), 10); // degenerate span
}
