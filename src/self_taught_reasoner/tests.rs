#![allow(
    clippy::float_cmp,
    clippy::similar_names,
    clippy::cast_precision_loss,
    clippy::too_many_lines,
    clippy::unreadable_literal
)]
//! Tests for the self-taught reasoner (`STaR`).
//!
//! The organising principle is **measurement, not self-consistency**: the
//! learning tests assert the actual per-round accuracy numbers a genuinely
//! improving fixture produces, computed by hand in the test, so a passing
//! assertion is an independent check on the loop rather than a restatement of it.
//!
//! Sections: config & errors → text primitives (equivalence, the cheat detector,
//! both sides) → `StarRng` → `RationaleSet` → **(a)** bootstrapping improves
//! forward accuracy → **(b)** the rationalization ablation → **(c)** cheat
//! rejection in the loop → **(d)** correctness-filter soundness → **(e)** fixed-
//! point termination → **(f)** determinism.

use std::collections::{BTreeSet, HashMap};

use super::engine::SelfTaughtReasoner;
use super::model::{RationalizationStyle, ReasoningModel, StaticReasoningModel};
use super::rng::{StarRng, mix_seed};
use super::text::{answers_equivalent, is_cheating_rationale, normalize_answer};
use super::types::{
    RationaleSet, RationaleSource, StarConfig, StarError, StarGeneration, StarProblem,
    StarRationale,
};

// ── Helpers ────────────────────────────────────────────────────────────────────

const EPS: f64 = 1e-9;

fn approx(a: f64, b: f64) -> bool {
    (a - b).abs() < EPS
}

/// Build the full domain `x in 0..m` for one class of `f(x) = (x + offset) mod m`.
fn class_problems(class: &str, m: u64, offset: u64) -> Vec<StarProblem> {
    (0..m)
        .map(|x| {
            StarProblem::new(
                format!("{}{x}", class.to_lowercase()),
                format!("[{class}] x={x}"),
                ((x + offset) % m).to_string(),
            )
        })
        .collect()
}

/// A fully-scripted model: forward and backward outputs are looked up by problem
/// id, and fine-tuning is a no-op (it only counts calls). Used where a test needs
/// to drive the engine's filters with exact, hand-written generations rather than
/// through the arithmetic fixture's learning dynamics.
struct ScriptedModel {
    forward: HashMap<String, StarGeneration>,
    backward: HashMap<String, StarGeneration>,
    updates: usize,
}

impl ScriptedModel {
    fn new() -> Self {
        Self {
            forward: HashMap::new(),
            backward: HashMap::new(),
            updates: 0,
        }
    }

    fn on_forward(mut self, id: &str, rationale: &str, answer: &str) -> Self {
        self.forward
            .insert(id.to_string(), StarGeneration::new(rationale, answer));
        self
    }

    fn on_rationalize(mut self, id: &str, rationale: &str, answer: &str) -> Self {
        self.backward
            .insert(id.to_string(), StarGeneration::new(rationale, answer));
        self
    }
}

impl ReasoningModel for ScriptedModel {
    fn generate(&self, problem: &StarProblem) -> StarGeneration {
        self.forward
            .get(&problem.id)
            .cloned()
            .unwrap_or_else(|| StarGeneration::new("no idea", "?"))
    }

    fn rationalize(&self, problem: &StarProblem, _hint: &str) -> StarGeneration {
        self.backward
            .get(&problem.id)
            .cloned()
            .unwrap_or_else(|| StarGeneration::new("no idea", "?"))
    }

    fn update_from(&mut self, _accumulated: &[StarRationale]) {
        self.updates += 1;
    }
}

// ── Config & errors ─────────────────────────────────────────────────────────────

#[test]
fn test_config_defaults() {
    let config = StarConfig::default();
    assert_eq!(config.max_rounds, 32);
    assert_eq!(config.equivalence_threshold, 0.8);
    assert!(config.use_rationalization);
    assert!(!config.shuffle_each_round);
    assert_eq!(config.seed, 0);
}

#[test]
fn test_config_builders() {
    let config = StarConfig::new()
        .with_max_rounds(7)
        .with_equivalence_threshold(0.5)
        .with_rationalization(false)
        .with_shuffle_each_round(true)
        .with_seed(99);
    assert_eq!(config.max_rounds, 7);
    assert_eq!(config.equivalence_threshold, 0.5);
    assert!(!config.use_rationalization);
    assert!(config.shuffle_each_round);
    assert_eq!(config.seed, 99);
}

#[test]
fn test_config_validate_rejects_zero_rounds() {
    let err = StarConfig::new().with_max_rounds(0).validate().unwrap_err();
    assert!(matches!(err, StarError::InvalidConfig { .. }));
}

#[test]
fn test_config_validate_rejects_bad_threshold() {
    assert!(
        StarConfig::new()
            .with_equivalence_threshold(1.5)
            .validate()
            .is_err()
    );
    assert!(
        StarConfig::new()
            .with_equivalence_threshold(-0.1)
            .validate()
            .is_err()
    );
    assert!(
        StarConfig::new()
            .with_equivalence_threshold(f32::NAN)
            .validate()
            .is_err()
    );
}

#[test]
fn test_run_rejects_empty_problems() {
    let reasoner = SelfTaughtReasoner::new(StarConfig::default());
    let mut model = StaticReasoningModel::new(13, 2);
    let err = reasoner.run(&[], &mut model).unwrap_err();
    assert_eq!(err, StarError::NoProblems);
}

#[test]
fn test_run_rejects_duplicate_problem_ids() {
    let reasoner = SelfTaughtReasoner::new(StarConfig::default());
    let mut model = StaticReasoningModel::new(13, 2);
    let problems = vec![
        StarProblem::new("dup", "[A] x=1", "5"),
        StarProblem::new("dup", "[A] x=2", "6"),
    ];
    let err = reasoner.run(&problems, &mut model).unwrap_err();
    assert_eq!(
        err,
        StarError::DuplicateProblemId {
            id: "dup".to_string()
        }
    );
}

// ── Text primitives: answer equivalence ─────────────────────────────────────────

#[test]
fn test_normalize_answer_integers() {
    assert_eq!(normalize_answer("07"), "7");
    assert_eq!(normalize_answer("+3."), "3");
    assert_eq!(normalize_answer("-0"), "0");
    assert_eq!(normalize_answer(" 42 "), "42");
}

#[test]
fn test_normalize_answer_text() {
    // Surrounding punctuation/whitespace is stripped and internal whitespace
    // runs collapse, but internal punctuation (the comma) is kept — matching the
    // answer-normalization semantics used elsewhere in the crate.
    assert_eq!(normalize_answer("  Hello,  World! "), "hello, world");
}

#[test]
fn test_answers_equivalent() {
    assert!(answers_equivalent("10", "10", 0.8));
    assert!(answers_equivalent("10", "10.", 0.8));
    assert!(answers_equivalent("07", "7", 0.8));
    assert!(!answers_equivalent("10", "11", 0.8));
    // Multi-token: same bag of words, different order.
    assert!(answers_equivalent("cat dog", "dog cat", 0.8));
    assert!(!answers_equivalent("cat", "dog", 0.8));
}

// ── Text primitives: the cheat detector (test c, direct — both sides) ────────────

#[test]
fn test_cheat_detector_rejects_bare_echo() {
    // A rationalization that only restates the hint.
    assert!(is_cheating_rationale(
        "the answer is 2 because the answer is 2",
        "[A] x=7",
        "2",
    ));
}

#[test]
fn test_cheat_detector_rejects_filler_only_echo() {
    // Different filler, still no derivation.
    assert!(is_cheating_rationale(
        "so obviously the answer must be 2, thus 2",
        "[A] x=7",
        "2",
    ));
}

#[test]
fn test_cheat_detector_rejects_problem_restatement_as_answer() {
    // Restating the problem's own value as the answer, with no computed
    // intermediate, is also a cheat: no work was done.
    assert!(is_cheating_rationale(
        "the answer is 7 because x is 7",
        "[A] x=7",
        "7",
    ));
}

#[test]
fn test_cheat_detector_keeps_genuine_derivation() {
    // A real derivation: grounds in the problem (x=7) and computes intermediates
    // (8, 15) distinct from the answer.
    assert!(!is_cheating_rationale(
        "class A, input x=7: offset 8, so 7 + 8 = 15, and 15 mod 13 = 2",
        "[A] x=7",
        "2",
    ));
}

#[test]
fn test_cheat_detector_keeps_genuine_derivation_that_mentions_answer() {
    // The genuine-that-happens-to-mention-the-answer side of the line: the
    // rationale states the answer at the end, but shows real working first.
    assert!(!is_cheating_rationale(
        "7 plus 8 equals 15, and 15 mod 13 is 2, therefore the answer is 2",
        "[A] x=7",
        "2",
    ));
}

// ── StarRng ─────────────────────────────────────────────────────────────────────

#[test]
fn test_rng_reproducible() {
    let mut a = StarRng::new(42);
    let mut b = StarRng::new(42);
    for _ in 0..64 {
        assert_eq!(a.next_u64(), b.next_u64());
    }
}

#[test]
fn test_rng_shuffle_is_seed_deterministic() {
    let mut first: Vec<usize> = (0..20).collect();
    let mut second: Vec<usize> = (0..20).collect();
    StarRng::new(7).shuffle(&mut first);
    StarRng::new(7).shuffle(&mut second);
    assert_eq!(first, second);
    // A shuffle is (with overwhelming probability) a real permutation.
    let mut sorted = first.clone();
    sorted.sort_unstable();
    assert_eq!(sorted, (0..20).collect::<Vec<_>>());
}

#[test]
fn test_rng_usize_below_bounds() {
    assert_eq!(StarRng::new(1).next_usize_below(0), None);
    let mut rng = StarRng::new(123);
    for _ in 0..1000 {
        let draw = rng.next_usize_below(10).unwrap();
        assert!(draw < 10);
    }
}

#[test]
fn test_mix_seed_distinct_salts_differ() {
    let a = mix_seed(5, 0);
    let b = mix_seed(5, 1);
    let c = mix_seed(5, 2);
    assert_ne!(a, b);
    assert_ne!(b, c);
    assert_ne!(a, c);
}

// ── RationaleSet ────────────────────────────────────────────────────────────────

fn forward_rationale(id: &str, answer: &str) -> StarRationale {
    StarRationale {
        problem_id: id.to_string(),
        statement: format!("[A] x={id}"),
        rationale: "r".to_string(),
        answer: answer.to_string(),
        source: RationaleSource::Forward,
    }
}

#[test]
fn test_rationale_set_dedups_by_problem_id() {
    let mut set = RationaleSet::new();
    assert!(set.is_empty());
    assert!(set.insert(forward_rationale("p1", "10")));
    assert!(!set.insert(forward_rationale("p1", "999"))); // same id: rejected
    assert!(set.insert(forward_rationale("p2", "20")));
    assert_eq!(set.len(), 2);
    assert!(set.contains("p1"));
    // First-seen wins: the second p1 answer never replaced the first.
    assert_eq!(set.get("p1").unwrap().answer, "10");
    assert_eq!(
        set.problem_ids(),
        BTreeSet::from(["p1".to_string(), "p2".to_string()])
    );
}

#[test]
fn test_rationale_set_count_by_source() {
    let mut set = RationaleSet::new();
    set.insert(forward_rationale("p1", "10"));
    set.insert(StarRationale {
        problem_id: "p2".to_string(),
        statement: "[A] x=2".to_string(),
        rationale: "r".to_string(),
        answer: "20".to_string(),
        source: RationaleSource::Rationalized,
    });
    assert_eq!(set.count_by_source(RationaleSource::Forward), 1);
    assert_eq!(set.count_by_source(RationaleSource::Rationalized), 1);
}

// ── (a) Bootstrapping improves forward accuracy ─────────────────────────────────

/// The learnable task: `f(x) = (x + 4) mod 13`, one class, seeded at x = 6, no
/// rationalization. The forward pass must bootstrap on its own — induce the
/// offset from the seed, then generalize outward one radius per round.
#[test]
fn test_a_bootstrapping_strictly_improves_forward_accuracy() {
    let m = 13;
    let offset = 4;
    let problems = class_problems("A", m, offset);
    let mut model =
        StaticReasoningModel::new(m, 2).with_seed("A", 6, ((6 + offset) % m).to_string());

    let reasoner = SelfTaughtReasoner::new(
        StarConfig::new()
            .with_rationalization(false)
            .with_max_rounds(32),
    );
    let outcome = reasoner.run(&problems, &mut model).unwrap();

    let curve = outcome.accuracy_curve();
    eprintln!("(a) forward-accuracy curve: {curve:?}");
    eprintln!(
        "(a) accumulated sizes: {:?}",
        outcome
            .rounds
            .iter()
            .map(|r| r.accumulated_size)
            .collect::<Vec<_>>()
    );

    // Hand-computed: seed {6}; radius 2 spreads {6}->{4..8}->{2..10}->{0..12}.
    assert_eq!(curve.len(), 5);
    assert!(approx(curve[0], 1.0 / 13.0));
    assert!(approx(curve[1], 5.0 / 13.0));
    assert!(approx(curve[2], 9.0 / 13.0));
    assert_eq!(curve[3], 1.0);
    assert_eq!(curve[4], 1.0);

    // Strictly increasing until the ceiling, then flat — never decreasing.
    assert!(curve[0] < curve[1] && curve[1] < curve[2] && curve[2] < curve[3]);
    for pair in curve.windows(2) {
        assert!(pair[1] >= pair[0]);
    }

    // Converges to the stated ceiling with full coverage, all forward.
    assert!(outcome.converged);
    assert_eq!(outcome.final_coverage(), 1.0);
    assert_eq!(outcome.final_forward_accuracy(), 1.0);
    assert_eq!(
        outcome.final_set.count_by_source(RationaleSource::Forward),
        13
    );
    assert_eq!(
        outcome
            .final_set
            .count_by_source(RationaleSource::Rationalized),
        0
    );

    // The accumulated set grew as predicted.
    let sizes: Vec<usize> = outcome.rounds.iter().map(|r| r.accumulated_size).collect();
    assert_eq!(sizes, vec![1, 5, 9, 13, 13]);
}

/// A larger generalization radius reaches the ceiling in fewer rounds — evidence
/// that the multi-round curve is produced by real bounded extrapolation, not a
/// hard-coded schedule.
#[test]
fn test_a_larger_radius_converges_faster() {
    let m = 13;
    let offset = 4;
    let problems = class_problems("A", m, offset);

    let mut fast =
        StaticReasoningModel::new(m, 6).with_seed("A", 6, ((6 + offset) % m).to_string());
    let reasoner = SelfTaughtReasoner::new(
        StarConfig::new()
            .with_rationalization(false)
            .with_max_rounds(32),
    );
    let outcome = reasoner.run(&problems, &mut fast).unwrap();

    // Radius 6 covers [0,12] from the seed at 6 in a single generalization round.
    assert!(outcome.converged);
    assert_eq!(outcome.final_coverage(), 1.0);
    assert!(outcome.num_rounds() < 5);
}

// ── (b) The rationalization ablation ─────────────────────────────────────────────

/// Build the two-class task: class A is forward-learnable (seeded), class B is
/// the hard class (a different offset, never seeded, outside the forward
/// inducer's reach until an example arrives). Returns `(problems, class_b_ids)`.
fn ablation_task() -> (Vec<StarProblem>, BTreeSet<String>) {
    let m = 13;
    let mut problems = class_problems("A", m, 4);
    problems.extend(class_problems("B", m, 7));
    let class_b_ids: BTreeSet<String> = (0..m).map(|x| format!("b{x}")).collect();
    (problems, class_b_ids)
}

#[test]
fn test_b_rationalization_ablation() {
    let (problems, class_b_ids) = ablation_task();
    let m = 13;
    let seed_answer = ((6 + 4) % m).to_string();

    // WITHOUT rationalization: forward can only ever learn class A.
    let mut model_without = StaticReasoningModel::new(m, 2).with_seed("A", 6, seed_answer.clone());
    let without = SelfTaughtReasoner::new(
        StarConfig::new()
            .with_rationalization(false)
            .with_max_rounds(64),
    )
    .run(&problems, &mut model_without)
    .unwrap();

    // WITH rationalization: class B is rescued by backward rationalization.
    let mut model_with = StaticReasoningModel::new(m, 2).with_seed("A", 6, seed_answer);
    let with = SelfTaughtReasoner::new(
        StarConfig::new()
            .with_rationalization(true)
            .with_max_rounds(64),
    )
    .run(&problems, &mut model_with)
    .unwrap();

    eprintln!(
        "(b) without-rationalization: coverage {:.3}, curve {:?}",
        without.final_coverage(),
        without.accuracy_curve()
    );
    eprintln!(
        "(b) with-rationalization:    coverage {:.3}, curve {:?}",
        with.final_coverage(),
        with.accuracy_curve()
    );

    // The ablation's core result: with-rationalization reaches strictly higher
    // final coverage.
    assert!(with.final_coverage() > without.final_coverage());
    assert_eq!(with.final_coverage(), 1.0); // all 26 solved
    assert_eq!(without.final_coverage(), 0.5); // only the 13 class-A problems

    // And the gap is concentrated on *exactly* the hard problems: the set of
    // problems solved with rationalization but not without is precisely class B.
    let solved_with = with.solved_problem_ids();
    let solved_without = without.solved_problem_ids();
    let gap: BTreeSet<String> = solved_with.difference(&solved_without).cloned().collect();
    assert_eq!(gap, class_b_ids);

    // Without rationalization, every class-B problem stays unsolved forever;
    // class A is fully solved by both runs.
    for id in &class_b_ids {
        assert!(!solved_without.contains(id));
        assert!(solved_with.contains(id));
    }
    for x in 0..13 {
        let a_id = format!("a{x}");
        assert!(solved_without.contains(&a_id));
        assert!(solved_with.contains(&a_id));
    }
}

// ── (c) Cheat rejection inside the loop ──────────────────────────────────────────

#[test]
fn test_c_cheating_rationalizations_never_enter_the_set() {
    let m = 13;
    // Class B only, no seed: the forward pass solves nothing, so every problem
    // reaches the rationalization pass.
    let problems = class_problems("B", m, 7);

    // A model whose rationalizations are cheats.
    let mut cheater =
        StaticReasoningModel::new(m, 2).with_rationalization_style(RationalizationStyle::Cheat);
    let outcome = SelfTaughtReasoner::new(StarConfig::new().with_max_rounds(8))
        .run(&problems, &mut cheater)
        .unwrap();

    // Nothing was learned: every rationalization was rejected as a cheat.
    assert!(outcome.final_set.is_empty());
    assert_eq!(outcome.final_coverage(), 0.0);
    assert!(outcome.converged);
    let round0 = &outcome.rounds[0];
    assert_eq!(round0.rationalized_solved, 0);
    assert_eq!(round0.cheats_rejected, 13);

    // The same task with genuine rationalizations solves everything — proving it
    // was the cheating, not the task, that blocked the cheater.
    let mut honest =
        StaticReasoningModel::new(m, 2).with_rationalization_style(RationalizationStyle::Genuine);
    let honest_outcome = SelfTaughtReasoner::new(StarConfig::new().with_max_rounds(8))
        .run(&problems, &mut honest)
        .unwrap();
    assert_eq!(honest_outcome.final_coverage(), 1.0);
    assert_eq!(
        honest_outcome
            .final_set
            .count_by_source(RationaleSource::Rationalized),
        13
    );
}

// ── (d) Correctness-filter soundness ─────────────────────────────────────────────

#[test]
fn test_d_scripted_filter_admits_only_correct_non_cheating_triples() {
    // Four problems exercising every filter path.
    let problems = vec![
        StarProblem::new("p1", "compute 10", "10"), // forward-correct
        StarProblem::new("p2", "sum of 8 and 12", "20"), // forward-wrong, rationalized-genuine
        StarProblem::new("p3", "compute 30", "30"), // forward-wrong, rationalized-cheat
        StarProblem::new("p4", "compute 40", "40"), // forward-wrong, rationalized-misses-gold
    ];

    let model = ScriptedModel::new()
        .on_forward("p1", "10 is given directly", "10")
        .on_forward("p2", "guessing", "99")
        .on_forward("p3", "guessing", "99")
        .on_forward("p4", "guessing", "99")
        .on_rationalize("p2", "8 plus 12 gives 20", "20") // genuine, correct
        .on_rationalize("p3", "the answer is 30 because the answer is 30", "30") // cheat
        .on_rationalize("p4", "8 plus 8 is 16", "42"); // wrong answer (misses 40)

    let mut model = model;
    let outcome = SelfTaughtReasoner::new(StarConfig::new().with_max_rounds(4))
        .run(&problems, &mut model)
        .unwrap();

    // Only p1 (forward) and p2 (genuine rationalization) entered the set.
    assert_eq!(outcome.final_set.len(), 2);
    assert!(outcome.final_set.contains("p1"));
    assert!(outcome.final_set.contains("p2"));
    assert!(!outcome.final_set.contains("p3")); // cheat rejected
    assert!(!outcome.final_set.contains("p4")); // wrong answer rejected
    assert_eq!(
        outcome.final_set.get("p1").unwrap().source,
        RationaleSource::Forward
    );
    assert_eq!(
        outcome.final_set.get("p2").unwrap().source,
        RationaleSource::Rationalized
    );

    // Round 0 accounting: p2/p3/p4 wrong forward + p4 wrong rationalization = 4
    // incorrect; p3 = 1 cheat.
    let round0 = &outcome.rounds[0];
    assert_eq!(round0.incorrect_rejected, 4);
    assert_eq!(round0.cheats_rejected, 1);

    // Soundness invariant: every kept triple's answer equals its gold answer.
    let gold: HashMap<&str, &str> = problems
        .iter()
        .map(|p| (p.id.as_str(), p.gold_answer.as_str()))
        .collect();
    for entry in outcome.final_set.rationales() {
        assert!(answers_equivalent(
            &entry.answer,
            gold[entry.problem_id.as_str()],
            0.8
        ));
    }
}

#[test]
fn test_d_no_incorrect_arithmetic_answer_ever_accumulates() {
    let m = 13;
    let offset = 4;
    let problems = class_problems("A", m, offset);
    let mut model =
        StaticReasoningModel::new(m, 2).with_seed("A", 6, ((6 + offset) % m).to_string());

    let outcome = SelfTaughtReasoner::new(
        StarConfig::new()
            .with_rationalization(false)
            .with_max_rounds(32),
    )
    .run(&problems, &mut model)
    .unwrap();

    // Round 0: the model can only solve the seed; its guesses for the other 12
    // problems are all wrong and must all be filtered out.
    assert_eq!(outcome.rounds[0].incorrect_rejected, 12);
    assert_eq!(outcome.rounds[0].accumulated_size, 1);

    // Across the whole run, every accumulated triple is correct — no wrong guess
    // ever slipped through the filter.
    let gold: HashMap<String, String> = problems
        .iter()
        .map(|p| (p.id.clone(), p.gold_answer.clone()))
        .collect();
    for entry in outcome.final_set.rationales() {
        assert_eq!(&entry.answer, &gold[&entry.problem_id]);
    }
}

// ── (e) Fixed-point termination ──────────────────────────────────────────────────

#[test]
fn test_e_converges_to_a_fixed_point_without_oscillation() {
    let m = 13;
    let offset = 4;
    let problems = class_problems("A", m, offset);
    let mut model =
        StaticReasoningModel::new(m, 2).with_seed("A", 6, ((6 + offset) % m).to_string());

    let outcome = SelfTaughtReasoner::new(
        StarConfig::new()
            .with_rationalization(false)
            .with_max_rounds(100),
    )
    .run(&problems, &mut model)
    .unwrap();

    // It converged, well within the `problems + 1` bound, without exhausting the
    // round cap.
    assert!(outcome.converged);
    assert!(outcome.num_rounds() <= problems.len() + 1);
    assert!(outcome.num_rounds() < 100);

    // The accumulated set is monotone non-decreasing and the forward accuracy
    // never oscillates (never decreases) — the two properties that together rule
    // out a non-terminating loop.
    let sizes: Vec<usize> = outcome.rounds.iter().map(|r| r.accumulated_size).collect();
    for pair in sizes.windows(2) {
        assert!(pair[1] >= pair[0]);
    }
    let curve = outcome.accuracy_curve();
    for pair in curve.windows(2) {
        assert!(pair[1] >= pair[0]);
    }

    // The terminating round is the one that added nothing new.
    assert_eq!(outcome.rounds.last().unwrap().newly_accumulated, 0);
}

#[test]
fn test_e_unlearnable_task_terminates_immediately() {
    // No seed, no rationalization: nothing is ever solvable. The loop must not
    // spin — it should converge on round 0 with an empty set.
    let problems = class_problems("A", 13, 4);
    let mut model = StaticReasoningModel::new(13, 2);
    let outcome = SelfTaughtReasoner::new(
        StarConfig::new()
            .with_rationalization(false)
            .with_max_rounds(50),
    )
    .run(&problems, &mut model)
    .unwrap();
    assert!(outcome.converged);
    assert_eq!(outcome.num_rounds(), 1);
    assert!(outcome.final_set.is_empty());
}

// ── (f) Determinism ──────────────────────────────────────────────────────────────

#[test]
fn test_f_same_seed_is_bit_for_bit_identical() {
    let m = 13;
    let offset = 4;
    let problems = class_problems("A", m, offset);
    let seed_answer = ((6 + offset) % m).to_string();

    let config = StarConfig::new()
        .with_shuffle_each_round(true)
        .with_seed(42)
        .with_rationalization(false)
        .with_max_rounds(32);

    let mut model_a = StaticReasoningModel::new(m, 2).with_seed("A", 6, seed_answer.clone());
    let mut model_b = StaticReasoningModel::new(m, 2).with_seed("A", 6, seed_answer.clone());
    let out_a = SelfTaughtReasoner::new(config.clone())
        .run(&problems, &mut model_a)
        .unwrap();
    let out_b = SelfTaughtReasoner::new(config)
        .run(&problems, &mut model_b)
        .unwrap();

    // Same seed ⇒ identical rounds and identical final set, bit-for-bit.
    assert_eq!(out_a, out_b);
}

#[test]
fn test_f_fixed_point_is_seed_invariant() {
    let m = 13;
    let offset = 4;
    let problems = class_problems("A", m, offset);
    let seed_answer = ((6 + offset) % m).to_string();

    let run = |seed: u64| {
        let mut model = StaticReasoningModel::new(m, 2).with_seed("A", 6, seed_answer.clone());
        SelfTaughtReasoner::new(
            StarConfig::new()
                .with_shuffle_each_round(true)
                .with_seed(seed)
                .with_rationalization(false)
                .with_max_rounds(32),
        )
        .run(&problems, &mut model)
        .unwrap()
    };

    let out_42 = run(42);
    let out_7 = run(7);

    // The processing order differs, so the runs are not bit-identical...
    assert_ne!(out_42.final_set.rationales(), out_7.final_set.rationales());
    // ...but the fixed point they reach is the same set of solved problems and
    // the same coverage: order does not change *what* gets learned.
    assert_eq!(out_42.solved_problem_ids(), out_7.solved_problem_ids());
    assert_eq!(out_42.final_coverage(), out_7.final_coverage());
    assert_eq!(out_42.final_coverage(), 1.0);
}
