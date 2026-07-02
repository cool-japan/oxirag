#![allow(
    clippy::float_cmp,
    clippy::cast_precision_loss,
    clippy::cast_possible_truncation,
    clippy::cast_sign_loss,
    clippy::similar_names,
    clippy::too_many_lines,
    clippy::items_after_statements,
    clippy::unreadable_literal,
    clippy::many_single_char_names,
    clippy::doc_markdown,
    clippy::uninlined_format_args
)]
//! Tests for the `belief_revision` module.
//!
//! The suite is organized around proving the *math is right*, not merely that
//! the code compiles:
//!
//! * [`logsumexp`](super::bayes::logsumexp) is checked directly against
//!   hand-computed values and on extreme inputs (very large negatives,
//!   near-equal values, a single dominant value, `+/-inf`) — this is the
//!   numerically delicate primitive the whole module rests on.
//! * [`hand_computed_bayes_two_hypotheses`] pins the posterior of a tiny
//!   two-hypothesis example against the closed-form Bayes-rule answer — the
//!   ground-truth correctness check for the entire update.
//! * [`numerical_stability_under_many_updates`] and
//!   [`naive_linear_space_underflows_where_log_space_survives`] together prove
//!   the log-space representation was actually necessary and is correctly
//!   implemented.
//! * [`epsilon_floor_prevents_permanent_impossibility`] proves the likelihood
//!   floor rescues a hypothesis from the `log(0) = -inf` trap.

use super::bayes::{
    LikelihoodRatio, bayes_update_log, entropy_nats, log_likelihood_floored, log_probs_to_linear,
    logsumexp, normalize_log_probs,
};
use super::engine::BeliefRevisionEngine;
use super::types::{BeliefHypothesis, BeliefRevisionConfig, BeliefRevisionError, EvidenceUpdate};

// ── constants (hand-computed reference values) ───────────────────────────────

const LN2: f64 = std::f64::consts::LN_2;
const LN3: f64 = 1.098_612_288_668_109_8;
const LN4: f64 = 1.386_294_361_119_890_6;
/// `ln(1e-6)`, the log of the default likelihood floor.
const LN_FLOOR: f64 = -13.815_510_557_964_274;

// ── helpers ──────────────────────────────────────────────────────────────────

#[track_caller]
fn assert_close(actual: f64, expected: f64, tol: f64) {
    assert!(
        (actual - expected).abs() <= tol,
        "expected {actual} ~= {expected} (tolerance {tol}, diff {})",
        (actual - expected).abs()
    );
}

fn default_engine() -> BeliefRevisionEngine {
    BeliefRevisionEngine::new(BeliefRevisionConfig::default())
}

fn two_hyps() -> Vec<BeliefHypothesis> {
    vec![
        BeliefHypothesis::new("h0", "Hypothesis zero."),
        BeliefHypothesis::new("h1", "Hypothesis one."),
    ]
}

fn three_hyps() -> Vec<BeliefHypothesis> {
    vec![
        BeliefHypothesis::new("h0", "Hypothesis zero."),
        BeliefHypothesis::new("h1", "Hypothesis one."),
        BeliefHypothesis::new("h2", "Hypothesis two."),
    ]
}

fn four_hyps() -> Vec<BeliefHypothesis> {
    vec![
        BeliefHypothesis::new("h0", "Hypothesis zero."),
        BeliefHypothesis::new("h1", "Hypothesis one."),
        BeliefHypothesis::new("h2", "Hypothesis two."),
        BeliefHypothesis::new("h3", "Hypothesis three."),
    ]
}

/// A two-hypothesis evidence update with the given likelihoods for `h0`/`h1`.
fn ev2(description: &str, l0: f64, l1: f64) -> EvidenceUpdate {
    EvidenceUpdate::new(description)
        .with_likelihood("h0", l0)
        .with_likelihood("h1", l1)
}

fn posterior_sum(state: &super::types::BeliefState) -> f64 {
    state.posterior().iter().map(|(_, p)| p).sum()
}

fn prob(state: &super::types::BeliefState, id: &str) -> f64 {
    state.probability_of(id).expect("hypothesis present")
}

// ═════════════════════════════════════════════════════════════════════════════
// logsumexp — direct, hand-computed, and stability tests
// ═════════════════════════════════════════════════════════════════════════════

#[test]
fn logsumexp_empty_is_neg_infinity() {
    assert!(logsumexp(&[]).is_infinite());
    assert!(logsumexp(&[]).is_sign_negative());
}

#[test]
fn logsumexp_single_value_is_identity() {
    assert_close(logsumexp(&[0.0]), 0.0, 1e-15);
    assert_close(logsumexp(&[5.0]), 5.0, 1e-15);
    assert_close(logsumexp(&[-7.25]), -7.25, 1e-15);
}

#[test]
fn logsumexp_two_zeros_is_ln2() {
    assert_close(logsumexp(&[0.0, 0.0]), LN2, 1e-15);
}

#[test]
fn logsumexp_three_ones_is_one_plus_ln3() {
    assert_close(logsumexp(&[1.0, 1.0, 1.0]), 1.0 + LN3, 1e-15);
}

#[test]
fn logsumexp_large_values_do_not_overflow() {
    // Naive exp(1000) is +inf; the stable form must return 1000 + ln(2).
    let result = logsumexp(&[1000.0, 1000.0]);
    assert!(result.is_finite());
    assert_close(result, 1000.0 + LN2, 1e-9);
}

#[test]
fn logsumexp_very_negative_values_are_stable() {
    // Naive exp(-1000) underflows to 0 whose ln is -inf; the stable form must
    // return -1000 + ln(2).
    let result = logsumexp(&[-1000.0, -1000.0]);
    assert!(result.is_finite());
    assert_close(result, -1000.0 + LN2, 1e-12);
}

#[test]
fn logsumexp_single_dominant_value() {
    // The -1000 term contributes exp(-1000) ~= 0, so the result is ~= 0.
    assert_close(logsumexp(&[0.0, -1000.0]), 0.0, 1e-12);
    // Symmetrically for a large dominant value.
    assert_close(logsumexp(&[50.0, -50.0]), 50.0, 1e-12);
}

#[test]
fn logsumexp_near_equal_values() {
    let result = logsumexp(&[100.0, 100.0 + 1e-10]);
    assert_close(result, 100.0 + LN2, 1e-9);
    assert!(result >= 100.0 + 1e-10);
}

#[test]
fn logsumexp_all_neg_infinity_is_neg_infinity() {
    let result = logsumexp(&[f64::NEG_INFINITY, f64::NEG_INFINITY]);
    assert!(result.is_infinite() && result.is_sign_negative());
}

#[test]
fn logsumexp_ignores_neg_infinity_terms() {
    // exp(-inf) = 0, so a -inf entry contributes nothing.
    assert_close(logsumexp(&[f64::NEG_INFINITY, 0.0]), 0.0, 1e-15);
    assert_close(logsumexp(&[f64::NEG_INFINITY, 1.0, 1.0]), 1.0 + LN2, 1e-15);
}

#[test]
fn logsumexp_positive_infinity_dominates() {
    let result = logsumexp(&[f64::INFINITY, 0.0]);
    assert!(result.is_infinite() && result.is_sign_positive());
}

#[test]
fn logsumexp_is_at_least_the_max() {
    for values in [
        vec![1.0, 2.0, 3.0],
        vec![-5.0, -1.0, -3.0],
        vec![0.0],
        vec![10.0, 10.0, 10.0, 10.0],
    ] {
        let max = values.iter().copied().fold(f64::NEG_INFINITY, f64::max);
        assert!(logsumexp(&values) >= max - 1e-12);
    }
}

// ═════════════════════════════════════════════════════════════════════════════
// log_likelihood_floored & normalization primitives
// ═════════════════════════════════════════════════════════════════════════════

#[test]
fn floored_zero_likelihood_is_finite() {
    let value = log_likelihood_floored(0.0, 1e-6);
    assert!(value.is_finite());
    assert_close(value, LN_FLOOR, 1e-12);
}

#[test]
fn floored_above_floor_is_plain_log() {
    assert_close(log_likelihood_floored(0.5, 1e-6), (0.5_f64).ln(), 1e-15);
    assert_close(log_likelihood_floored(1.0, 1e-6), 0.0, 1e-15);
}

#[test]
fn floored_with_bad_floor_falls_back_to_min_positive() {
    // A non-positive configured floor must not yield ln(0) = -inf.
    assert!(log_likelihood_floored(0.0, 0.0).is_finite());
    assert!(log_likelihood_floored(0.0, -1.0).is_finite());
    // A value above the fallback floor is unaffected.
    assert_close(log_likelihood_floored(0.5, 0.0), (0.5_f64).ln(), 1e-15);
}

#[test]
fn normalize_makes_logsumexp_zero() {
    let mut log_probs = vec![0.0, 0.0, 0.0];
    normalize_log_probs(&mut log_probs);
    assert_close(logsumexp(&log_probs), 0.0, 1e-12);
    for &lp in &log_probs {
        assert_close(lp, -LN3, 1e-12);
    }
}

#[test]
fn bayes_update_keeps_distribution_normalized() {
    let mut log_probs = vec![(0.5_f64).ln(), (0.5_f64).ln()];
    let log_liks = vec![(0.9_f64).ln(), (0.1_f64).ln()];
    bayes_update_log(&mut log_probs, &log_liks);
    assert_close(logsumexp(&log_probs), 0.0, 1e-12);
    let linear = log_probs_to_linear(&log_probs);
    assert_close(linear[0], 0.9, 1e-12);
    assert_close(linear[1], 0.1, 1e-12);
}

#[test]
fn log_probs_to_linear_sums_to_one() {
    let linear = log_probs_to_linear(&[(0.2_f64).ln(), (0.3_f64).ln(), (0.5_f64).ln()]);
    let total: f64 = linear.iter().sum();
    assert_close(total, 1.0, 1e-12);
}

#[test]
fn entropy_uniform_is_ln_k() {
    let uniform = vec![(0.25_f64).ln(); 4];
    assert_close(entropy_nats(&uniform), LN4, 1e-12);
    let uniform3 = vec![(1.0_f64 / 3.0).ln(); 3];
    assert_close(entropy_nats(&uniform3), LN3, 1e-12);
}

#[test]
fn entropy_certain_is_zero() {
    // Almost all mass on one hypothesis => entropy ~ 0.
    let mut log_probs = vec![0.0, -50.0, -50.0];
    normalize_log_probs(&mut log_probs);
    assert_close(entropy_nats(&log_probs), 0.0, 1e-15);
}

// ═════════════════════════════════════════════════════════════════════════════
// LikelihoodRatio — the log-odds / Bayes-factor primitive
// ═════════════════════════════════════════════════════════════════════════════

#[test]
fn likelihood_ratio_from_likelihoods_math() {
    let ratio = LikelihoodRatio::from_likelihoods("a", "b", 0.9, 0.2, 1e-6);
    // log(0.9 / 0.2) = log(4.5).
    assert_close(ratio.log_ratio, (4.5_f64).ln(), 1e-12);
    assert_close(ratio.ratio(), 4.5, 1e-12);
    assert!(ratio.favors_numerator());
    assert!(!ratio.favors_denominator());
    assert!(!ratio.is_neutral());
    assert_eq!(ratio.numerator_id, "a");
    assert_eq!(ratio.denominator_id, "b");
}

#[test]
fn likelihood_ratio_neutral_when_equal() {
    let ratio = LikelihoodRatio::from_likelihoods("a", "b", 0.5, 0.5, 1e-6);
    assert_close(ratio.log_ratio, 0.0, 1e-15);
    assert_close(ratio.ratio(), 1.0, 1e-15);
    assert!(ratio.is_neutral());
    assert!(!ratio.favors_numerator());
    assert!(!ratio.favors_denominator());
}

#[test]
fn likelihood_ratio_apply_to_log_odds() {
    let ratio = LikelihoodRatio::new("a", "b", (4.5_f64).ln());
    assert_close(ratio.apply_to_log_odds(0.0), (4.5_f64).ln(), 1e-12);
    assert_close(ratio.apply_to_log_odds(1.0), 1.0 + (4.5_f64).ln(), 1e-12);
}

#[test]
fn likelihood_ratio_zero_likelihood_is_finite_and_favors_other_side() {
    // A zero numerator likelihood is floored, so the ratio is large-negative
    // and finite (never -inf), and favors the denominator.
    let ratio = LikelihoodRatio::from_likelihoods("a", "b", 0.0, 1.0, 1e-6);
    assert!(ratio.log_ratio.is_finite());
    assert_close(ratio.log_ratio, LN_FLOOR, 1e-9);
    assert!(ratio.favors_denominator());
}

// ═════════════════════════════════════════════════════════════════════════════
// Engine — initialization
// ═════════════════════════════════════════════════════════════════════════════

#[test]
fn init_uniform_prior_is_equiprobable() {
    let engine = default_engine();
    let state = engine.init(four_hyps(), None).unwrap();
    assert_close(posterior_sum(&state), 1.0, 1e-12);
    for (_, p) in state.posterior() {
        assert_close(p, 0.25, 1e-12);
    }
    assert_eq!(state.len(), 4);
    assert!(!state.is_empty());
    assert_eq!(state.updates_applied(), 0);
}

#[test]
fn init_custom_prior_is_reflected_in_posterior() {
    let engine = default_engine();
    let prior = [0.2, 0.3, 0.5];
    let state = engine.init(three_hyps(), Some(&prior)).unwrap();
    assert_close(prob(&state, "h0"), 0.2, 1e-9);
    assert_close(prob(&state, "h1"), 0.3, 1e-9);
    assert_close(prob(&state, "h2"), 0.5, 1e-9);
    assert_close(posterior_sum(&state), 1.0, 1e-12);
}

#[test]
fn init_empty_hypotheses_is_error() {
    let engine = default_engine();
    let result = engine.init(Vec::new(), None);
    assert_eq!(result.unwrap_err(), BeliefRevisionError::EmptyHypothesisSet);
}

#[test]
fn init_duplicate_ids_is_error() {
    let engine = default_engine();
    let hyps = vec![
        BeliefHypothesis::new("dup", "first"),
        BeliefHypothesis::new("dup", "second"),
    ];
    let result = engine.init(hyps, None);
    assert_eq!(
        result.unwrap_err(),
        BeliefRevisionError::DuplicateHypothesisId {
            id: "dup".to_string()
        }
    );
}

#[test]
fn init_prior_length_mismatch_is_error() {
    let engine = default_engine();
    let prior = [0.5, 0.5];
    let result = engine.init(three_hyps(), Some(&prior));
    assert_eq!(
        result.unwrap_err(),
        BeliefRevisionError::PriorLengthMismatch {
            expected: 3,
            found: 2
        }
    );
}

#[test]
fn init_prior_bad_sum_is_error() {
    let engine = default_engine();
    let prior = [0.5, 0.3]; // sums to 0.8
    let result = engine.init(two_hyps(), Some(&prior));
    match result.unwrap_err() {
        BeliefRevisionError::InvalidPriorSum { sum } => assert_close(sum, 0.8, 1e-12),
        other => panic!("expected InvalidPriorSum, got {other:?}"),
    }
}

#[test]
fn init_prior_negative_entry_is_error() {
    let engine = default_engine();
    let prior = [-0.1, 1.1]; // sums to 1.0 but has a negative entry
    let result = engine.init(two_hyps(), Some(&prior));
    assert_eq!(
        result.unwrap_err(),
        BeliefRevisionError::NegativePrior {
            id: "h0".to_string(),
            value: -0.1
        }
    );
}

#[test]
fn init_prior_non_finite_entry_is_error() {
    let engine = default_engine();
    let prior = [f64::NAN, 1.0];
    let result = engine.init(two_hyps(), Some(&prior));
    assert_eq!(
        result.unwrap_err(),
        BeliefRevisionError::NonFinitePrior {
            id: "h0".to_string()
        }
    );
}

#[test]
fn init_prior_with_zero_entry_is_floored_not_impossible() {
    let engine = default_engine();
    let prior = [1.0, 0.0];
    let state = engine.init(two_hyps(), Some(&prior)).unwrap();
    // The zero entry is floored, so it is tiny but strictly positive, and the
    // distribution still sums to 1.
    assert!(prob(&state, "h1") > 0.0);
    assert!(prob(&state, "h1") < 1e-4);
    assert!(prob(&state, "h0") > 0.99);
    assert_close(posterior_sum(&state), 1.0, 1e-12);
}

#[test]
fn init_rejects_invalid_config() {
    let engine = BeliefRevisionEngine::new(BeliefRevisionConfig::new().with_min_likelihood(2.0));
    let result = engine.init(two_hyps(), None);
    assert!(matches!(
        result.unwrap_err(),
        BeliefRevisionError::InvalidConfig { .. }
    ));
}

// ═════════════════════════════════════════════════════════════════════════════
// Engine — evidence update error paths
// ═════════════════════════════════════════════════════════════════════════════

#[test]
fn update_empty_evidence_is_error() {
    let engine = default_engine();
    let mut state = engine.init(two_hyps(), None).unwrap();
    let evidence = EvidenceUpdate::new("nothing");
    assert_eq!(
        engine.update(&mut state, &evidence).unwrap_err(),
        BeliefRevisionError::EmptyEvidence {
            description: "nothing".to_string()
        }
    );
}

#[test]
fn update_missing_likelihood_strict_is_error() {
    let engine = default_engine();
    let mut state = engine.init(two_hyps(), None).unwrap();
    // Only h0 has a likelihood; strict mode requires h1 too.
    let evidence = EvidenceUpdate::new("partial").with_likelihood("h0", 0.9);
    assert_eq!(
        engine.update(&mut state, &evidence).unwrap_err(),
        BeliefRevisionError::MissingLikelihood {
            id: "h1".to_string()
        }
    );
}

#[test]
fn update_unknown_hypothesis_strict_is_error() {
    let engine = default_engine();
    let mut state = engine.init(two_hyps(), None).unwrap();
    let evidence = ev2("e", 0.5, 0.5).with_likelihood("ghost", 0.5);
    assert_eq!(
        engine.update(&mut state, &evidence).unwrap_err(),
        BeliefRevisionError::UnknownHypothesis {
            id: "ghost".to_string()
        }
    );
}

#[test]
fn update_lenient_missing_defaults_to_neutral() {
    let engine = BeliefRevisionEngine::new(BeliefRevisionConfig::new().with_strict(false));
    let mut state = engine.init(two_hyps(), None).unwrap();
    // h1 missing => neutral 1.0; h0 gets 0.9. Neutral 1.0 > 0.9, so h1 wins.
    let evidence = EvidenceUpdate::new("partial").with_likelihood("h0", 0.9);
    engine.update(&mut state, &evidence).unwrap();
    assert_close(posterior_sum(&state), 1.0, 1e-12);
    assert!(prob(&state, "h1") > prob(&state, "h0"));
}

#[test]
fn update_lenient_ignores_unknown_hypothesis() {
    let engine = BeliefRevisionEngine::new(BeliefRevisionConfig::new().with_strict(false));
    let mut state = engine.init(two_hyps(), None).unwrap();
    let evidence = ev2("e", 0.9, 0.1).with_likelihood("ghost", 0.5);
    engine.update(&mut state, &evidence).unwrap();
    assert_close(posterior_sum(&state), 1.0, 1e-12);
    assert_close(prob(&state, "h0"), 0.9, 1e-12);
}

#[test]
fn update_negative_likelihood_is_error() {
    let engine = default_engine();
    let mut state = engine.init(two_hyps(), None).unwrap();
    let evidence = ev2("e", -0.5, 0.5);
    assert_eq!(
        engine.update(&mut state, &evidence).unwrap_err(),
        BeliefRevisionError::NegativeLikelihood {
            id: "h0".to_string(),
            value: -0.5
        }
    );
}

#[test]
fn update_non_finite_likelihood_is_error() {
    let engine = default_engine();
    let mut state = engine.init(two_hyps(), None).unwrap();
    let evidence = ev2("e", f64::NAN, 0.5);
    assert_eq!(
        engine.update(&mut state, &evidence).unwrap_err(),
        BeliefRevisionError::NonFiniteLikelihood {
            id: "h0".to_string()
        }
    );
}

#[test]
fn update_increments_counter() {
    let engine = default_engine();
    let mut state = engine.init(two_hyps(), None).unwrap();
    for _ in 0..3 {
        engine.update(&mut state, &ev2("e", 0.6, 0.4)).unwrap();
    }
    assert_eq!(state.updates_applied(), 3);
}

#[test]
fn update_sequence_empty_is_noop() {
    let engine = default_engine();
    let mut state = engine.init(three_hyps(), None).unwrap();
    let before = state.posterior();
    engine.update_sequence(&mut state, &[]).unwrap();
    assert_eq!(state.posterior(), before);
    assert_eq!(state.updates_applied(), 0);
}

#[test]
fn update_sequence_applies_all_in_order() {
    let engine = default_engine();
    let mut sequential = engine.init(two_hyps(), None).unwrap();
    let updates = [ev2("a", 0.7, 0.3), ev2("b", 0.8, 0.2), ev2("c", 0.6, 0.4)];
    engine.update_sequence(&mut sequential, &updates).unwrap();

    // Applying them one-by-one yields exactly the same posterior.
    let mut one_by_one = engine.init(two_hyps(), None).unwrap();
    for update in &updates {
        engine.update(&mut one_by_one, update).unwrap();
    }
    assert_eq!(sequential.posterior(), one_by_one.posterior());
    assert_eq!(sequential.updates_applied(), 3);
}

// ═════════════════════════════════════════════════════════════════════════════
// Engine — CORE CORRECTNESS (ground-truth Bayesian math)
// ═════════════════════════════════════════════════════════════════════════════

#[test]
fn hand_computed_bayes_two_hypotheses() {
    // Uniform prior P(h0) = P(h1) = 0.5. Evidence with P(e|h0) = 0.9,
    // P(e|h1) = 0.2. Closed-form Bayes:
    //   P(h0|e) = 0.5*0.9 / (0.5*0.9 + 0.5*0.2) = 0.45 / 0.55 = 0.8181818...
    //   P(h1|e) = 0.5*0.2 / 0.55            = 0.10 / 0.55 = 0.1818181...
    let engine = default_engine();
    let mut state = engine.init(two_hyps(), None).unwrap();
    engine.update(&mut state, &ev2("e", 0.9, 0.2)).unwrap();

    assert_close(prob(&state, "h0"), 0.45 / 0.55, 1e-9);
    assert_close(prob(&state, "h1"), 0.10 / 0.55, 1e-9);
    assert_close(posterior_sum(&state), 1.0, 1e-12);
}

#[test]
fn single_contradictory_evidence_swings_posterior_correctly() {
    // Start uniform (P(h0) = 0.5). Evidence contradicts h0 and supports h1:
    // P(e|h0) = 0.2, P(e|h1) = 0.8. Closed-form:
    //   P(h0|e) = 0.5*0.2 / (0.5*0.2 + 0.5*0.8) = 0.1 / 0.5 = 0.2
    // Direction: P(h0) must DROP from 0.5; magnitude: exactly 0.2.
    let engine = default_engine();
    let mut state = engine.init(two_hyps(), None).unwrap();
    let before = prob(&state, "h0");
    engine
        .update(&mut state, &ev2("contradiction", 0.2, 0.8))
        .unwrap();
    let after = prob(&state, "h0");

    assert!(after < before, "posterior for h0 should drop");
    assert_close(after, 0.2, 1e-9);
    assert_close(prob(&state, "h1"), 0.8, 1e-9);
}

#[test]
fn likelihood_ratio_ties_to_the_full_update() {
    // The pairwise LikelihoodRatio's log-odds shift must equal the actual
    // posterior log-odds shift produced by the full multi-hypothesis update.
    let engine = default_engine();
    let mut state = engine.init(two_hyps(), None).unwrap();

    let evidence = ev2("e", 0.9, 0.2);
    let ratio = engine
        .likelihood_ratio(&state, &evidence, "h0", "h1")
        .unwrap();

    // Prior log-odds (uniform) is 0; the ratio predicts posterior log-odds.
    let predicted = ratio.apply_to_log_odds(0.0);

    engine.update(&mut state, &evidence).unwrap();
    let log_post = state.log_posterior();
    let observed = log_post[0].1 - log_post[1].1;

    assert_close(ratio.log_ratio, (4.5_f64).ln(), 1e-12);
    assert_close(predicted, observed, 1e-12);
}

#[test]
fn convergence_toward_one_is_monotonic() {
    // Repeated evidence all favoring h0 must drive P(h0) monotonically up
    // (allowing only tiny numerical wobble) and P(h1) monotonically down.
    let engine = default_engine();
    let mut state = engine.init(two_hyps(), None).unwrap();

    let mut prev_h0 = prob(&state, "h0");
    let mut prev_h1 = prob(&state, "h1");
    for _ in 0..25 {
        engine
            .update(&mut state, &ev2("supports h0", 0.8, 0.2))
            .unwrap();
        let cur_h0 = prob(&state, "h0");
        let cur_h1 = prob(&state, "h1");
        assert!(cur_h0 >= prev_h0 - 1e-12, "h0 must not decrease");
        assert!(cur_h1 <= prev_h1 + 1e-12, "h1 must not increase");
        assert_close(posterior_sum(&state), 1.0, 1e-9);
        prev_h0 = cur_h0;
        prev_h1 = cur_h1;
    }
    assert!(prob(&state, "h0") > 0.99);
    assert!(prob(&state, "h1") < 0.01);
}

#[test]
fn losing_hypotheses_shrink_toward_zero() {
    // With three hypotheses and evidence always favoring h1, the other two
    // must both shrink toward 0 while h1 approaches 1.
    let engine = default_engine();
    let mut state = engine.init(three_hyps(), None).unwrap();
    let evidence = EvidenceUpdate::new("supports h1")
        .with_likelihood("h0", 0.1)
        .with_likelihood("h1", 0.8)
        .with_likelihood("h2", 0.1);
    for _ in 0..30 {
        engine.update(&mut state, &evidence).unwrap();
    }
    assert!(prob(&state, "h1") > 0.99);
    assert!(prob(&state, "h0") < 0.01);
    assert!(prob(&state, "h2") < 0.01);
    assert_close(posterior_sum(&state), 1.0, 1e-9);
}

#[test]
fn numerical_stability_under_many_updates() {
    // THE key stability test: after 100+ sequential updates the posterior must
    // still be a valid probability distribution (all finite, summing to 1).
    // A naive linear-space implementation would have underflowed by now.
    let engine = default_engine();
    let mut state = engine.init(two_hyps(), None).unwrap();

    let mut prev_winner = prob(&state, "h0");
    for step in 0..150 {
        engine
            .update(&mut state, &ev2("supports h0", 0.9, 0.1))
            .unwrap();
        let total = posterior_sum(&state);
        assert_close(total, 1.0, 1e-9);
        for (_, p) in state.posterior() {
            assert!(p.is_finite(), "step {step}: probability must stay finite");
            assert!(
                (0.0..=1.0).contains(&p),
                "step {step}: probability out of range"
            );
        }
        // Under identical winning evidence the winner never regresses.
        let winner = prob(&state, "h0");
        assert!(winner >= prev_winner - 1e-12);
        prev_winner = winner;
    }
    assert_eq!(state.updates_applied(), 150);
    assert!(prob(&state, "h0") > 0.999);
}

#[test]
fn naive_linear_space_underflows_where_log_space_survives() {
    // Demonstrate WHY log-space is required. In linear space, multiplying a
    // likelihood of 0.1 four hundred times underflows all the way to exactly
    // 0.0 — a naive filter's normalizer would become 0/0 = NaN. The log-space
    // engine processes the same 400 updates and stays a valid distribution.
    let naive_unnormalized = 0.5_f64 * (0.1_f64).powi(400);
    assert_eq!(naive_unnormalized, 0.0, "0.1^400 must underflow to zero");

    let engine = default_engine();
    let mut state = engine.init(two_hyps(), None).unwrap();
    // Equal likelihoods keep the true posterior uniform; the point is that the
    // machinery never underflows or produces NaN.
    for _ in 0..400 {
        engine
            .update(&mut state, &ev2("uninformative", 0.1, 0.1))
            .unwrap();
    }
    assert_close(posterior_sum(&state), 1.0, 1e-9);
    assert_close(prob(&state, "h0"), 0.5, 1e-9);
    assert_close(prob(&state, "h1"), 0.5, 1e-9);
    for (_, p) in state.posterior() {
        assert!(p.is_finite());
    }
}

#[test]
fn epsilon_floor_prevents_permanent_impossibility() {
    // Evidence 1 gives h0 a likelihood of exactly 0. Without the floor this
    // would set log P(h0) = -inf permanently; with the floor h0 survives as a
    // tiny-but-positive probability and can RECOVER under later favorable
    // evidence.
    let engine = default_engine();
    let mut state = engine.init(two_hyps(), None).unwrap();

    // Damning evidence against h0.
    engine
        .update(&mut state, &ev2("h0 impossible", 0.0, 1.0))
        .unwrap();
    let after_bad = prob(&state, "h0");
    assert!(after_bad.is_finite());
    assert!(
        after_bad > 0.0,
        "floored, so strictly positive (not the -inf trap)"
    );
    assert!(after_bad < 1e-4, "but crushed to near-zero");

    // Strongly favorable evidence twice: h0 climbs back out.
    engine
        .update(&mut state, &ev2("h0 supported", 1.0, 0.0))
        .unwrap();
    engine
        .update(&mut state, &ev2("h0 supported", 1.0, 0.0))
        .unwrap();
    let recovered = prob(&state, "h0");
    assert!(recovered > 0.9, "h0 recovered to {recovered}");
    assert_close(posterior_sum(&state), 1.0, 1e-9);
}

#[test]
fn determinism_identical_posterior_across_runs() {
    // The same hypotheses + prior + evidence sequence must yield a
    // bit-identical posterior, twice.
    let engine = default_engine();
    let prior = [0.3, 0.45, 0.25];
    let updates = [
        EvidenceUpdate::new("a")
            .with_likelihood("h0", 0.7)
            .with_likelihood("h1", 0.2)
            .with_likelihood("h2", 0.1),
        EvidenceUpdate::new("b")
            .with_likelihood("h0", 0.1)
            .with_likelihood("h1", 0.6)
            .with_likelihood("h2", 0.3),
    ];

    let run = || {
        let mut state = engine.init(three_hyps(), Some(&prior)).unwrap();
        engine.update_sequence(&mut state, &updates).unwrap();
        state.posterior()
    };

    let first = run();
    let second = run();
    assert_eq!(first, second);
    // And exactly equal element-by-element (bitwise via ==).
    for ((id_a, p_a), (id_b, p_b)) in first.iter().zip(&second) {
        assert_eq!(id_a, id_b);
        assert!(p_a == p_b);
    }
}

// ═════════════════════════════════════════════════════════════════════════════
// Engine — most_likely / posterior / run
// ═════════════════════════════════════════════════════════════════════════════

#[test]
fn most_likely_returns_argmax_hand_built() {
    let engine = default_engine();
    let prior = [0.2, 0.5, 0.3];
    let state = engine.init(three_hyps(), Some(&prior)).unwrap();
    let best = engine.most_likely(&state).unwrap();
    assert_eq!(best.id, "h1");
    assert_eq!(state.most_likely_index(), Some(1));
}

#[test]
fn most_likely_breaks_ties_by_lowest_index() {
    let engine = default_engine();
    let state = engine.init(three_hyps(), None).unwrap();
    // Uniform => all equal => the first hypothesis wins deterministically.
    assert_eq!(engine.most_likely(&state).unwrap().id, "h0");
    assert_eq!(state.most_likely_index(), Some(0));
}

#[test]
fn most_likely_follows_the_evidence() {
    let engine = default_engine();
    let mut state = engine.init(two_hyps(), None).unwrap();
    assert_eq!(engine.most_likely(&state).unwrap().id, "h0"); // tie -> h0

    engine
        .update(&mut state, &ev2("supports h1", 0.1, 0.9))
        .unwrap();
    assert_eq!(engine.most_likely(&state).unwrap().id, "h1");
}

#[test]
fn engine_posterior_matches_state_posterior() {
    let engine = default_engine();
    let mut state = engine.init(three_hyps(), None).unwrap();
    engine
        .update(
            &mut state,
            &EvidenceUpdate::new("e")
                .with_likelihood("h0", 0.5)
                .with_likelihood("h1", 0.3)
                .with_likelihood("h2", 0.2),
        )
        .unwrap();
    assert_eq!(engine.posterior(&state), state.posterior());
}

#[test]
fn run_end_to_end_packages_result() {
    let engine = default_engine();
    let updates = [ev2("a", 0.9, 0.1), ev2("b", 0.8, 0.2)];
    let result = engine.run(two_hyps(), None, &updates).unwrap();

    assert_eq!(result.most_likely.id, "h0");
    assert_eq!(result.updates_applied, 2);
    assert_close(
        result.posterior().iter().map(|(_, p)| p).sum::<f64>(),
        1.0,
        1e-9,
    );
    assert!(result.probability_of("h0").unwrap() > result.probability_of("h1").unwrap());
    assert!(result.entropy >= 0.0 && result.entropy <= LN2 + 1e-12);
    // The embedded state agrees with the summary.
    assert_eq!(result.state.updates_applied(), 2);
}

#[test]
fn run_propagates_init_errors() {
    let engine = default_engine();
    let result = engine.run(Vec::new(), None, &[]);
    assert_eq!(result.unwrap_err(), BeliefRevisionError::EmptyHypothesisSet);
}

#[test]
fn likelihood_ratio_unknown_hypothesis_is_error() {
    let engine = default_engine();
    let state = engine.init(two_hyps(), None).unwrap();
    let evidence = ev2("e", 0.9, 0.1);
    assert_eq!(
        engine
            .likelihood_ratio(&state, &evidence, "h0", "ghost")
            .unwrap_err(),
        BeliefRevisionError::UnknownHypothesis {
            id: "ghost".to_string()
        }
    );
}

// ═════════════════════════════════════════════════════════════════════════════
// Read-out invariants & the text-support heuristic
// ═════════════════════════════════════════════════════════════════════════════

#[test]
fn posterior_is_never_negative() {
    let engine = default_engine();
    let mut state = engine.init(four_hyps(), None).unwrap();
    let updates = [
        EvidenceUpdate::new("a")
            .with_likelihood("h0", 0.0)
            .with_likelihood("h1", 0.5)
            .with_likelihood("h2", 0.9)
            .with_likelihood("h3", 0.1),
        EvidenceUpdate::new("b")
            .with_likelihood("h0", 0.2)
            .with_likelihood("h1", 0.0)
            .with_likelihood("h2", 0.4)
            .with_likelihood("h3", 0.7),
    ];
    engine.update_sequence(&mut state, &updates).unwrap();
    for (_, p) in state.posterior() {
        assert!(p >= 0.0, "probability must be non-negative");
    }
    assert_close(posterior_sum(&state), 1.0, 1e-9);
}

#[test]
fn log_posterior_exponentiates_to_one() {
    let engine = default_engine();
    let mut state = engine.init(three_hyps(), None).unwrap();
    engine
        .update(
            &mut state,
            &EvidenceUpdate::from_likelihoods(
                "e",
                vec![
                    ("h0".to_string(), 0.6),
                    ("h1".to_string(), 0.3),
                    ("h2".to_string(), 0.1),
                ],
            ),
        )
        .unwrap();
    let total: f64 = state.log_posterior().iter().map(|(_, lp)| lp.exp()).sum();
    assert_close(total, 1.0, 1e-12);
    // Every normalized log-probability is <= 0.
    for (_, lp) in state.log_posterior() {
        assert!(lp <= 1e-12);
    }
}

#[test]
fn text_support_likelihoods_are_bounded() {
    let hyps = three_hyps();
    let evidence = EvidenceUpdate::from_text_support("hypothesis zero is correct", &hyps, 1e-6);
    assert_eq!(evidence.likelihoods.len(), 3);
    for (_, likelihood) in &evidence.likelihoods {
        assert!(*likelihood >= 1e-6);
        assert!(*likelihood <= 1.0);
    }
}

#[test]
fn text_support_rewards_overlap() {
    // The evidence text overlaps the "paris" hypothesis description strongly
    // and the "berlin" one not at all.
    let hyps = vec![
        BeliefHypothesis::new("paris", "the capital city is paris in france"),
        BeliefHypothesis::new("berlin", "the capital city is berlin in germany"),
    ];
    let evidence =
        EvidenceUpdate::from_text_support("reports confirm paris france definitively", &hyps, 1e-6);
    let paris = evidence.likelihood_for("paris").unwrap();
    let berlin = evidence.likelihood_for("berlin").unwrap();
    assert!(
        paris > berlin,
        "paris ({paris}) should exceed berlin ({berlin})"
    );

    // Feeding those heuristic likelihoods through the engine favors paris.
    let engine = default_engine();
    let mut state = engine.init(hyps, None).unwrap();
    engine.update(&mut state, &evidence).unwrap();
    assert_eq!(engine.most_likely(&state).unwrap().id, "paris");
}

// ═════════════════════════════════════════════════════════════════════════════
// Config, data-structure, and temp-file round-trip tests
// ═════════════════════════════════════════════════════════════════════════════

#[test]
fn config_default_and_builders() {
    let default = BeliefRevisionConfig::default();
    assert_close(default.min_likelihood, 1e-6, 1e-18);
    assert_close(default.prior_sum_tolerance, 1e-6, 1e-18);
    assert!(default.strict);
    assert!(default.validate().is_ok());

    let custom = BeliefRevisionConfig::new()
        .with_min_likelihood(1e-4)
        .with_prior_sum_tolerance(1e-3)
        .with_strict(false);
    assert_close(custom.min_likelihood, 1e-4, 1e-18);
    assert_close(custom.prior_sum_tolerance, 1e-3, 1e-18);
    assert!(!custom.strict);
    assert!(custom.validate().is_ok());
}

#[test]
fn config_validation_rejects_bad_values() {
    assert!(
        BeliefRevisionConfig::new()
            .with_min_likelihood(0.0)
            .validate()
            .is_err()
    );
    assert!(
        BeliefRevisionConfig::new()
            .with_min_likelihood(1.0)
            .validate()
            .is_err()
    );
    assert!(
        BeliefRevisionConfig::new()
            .with_min_likelihood(f64::NAN)
            .validate()
            .is_err()
    );
    assert!(
        BeliefRevisionConfig::new()
            .with_prior_sum_tolerance(-1.0)
            .validate()
            .is_err()
    );
}

#[test]
fn hypothesis_and_evidence_constructors() {
    let hypothesis = BeliefHypothesis::new("id", "desc");
    assert_eq!(hypothesis.id, "id");
    assert_eq!(hypothesis.description, "desc");

    let evidence = EvidenceUpdate::new("text")
        .with_likelihood("a", 0.7)
        .with_likelihood("b", 0.3);
    assert_eq!(evidence.description, "text");
    assert_eq!(evidence.likelihood_for("a"), Some(0.7));
    assert_eq!(evidence.likelihood_for("b"), Some(0.3));
    assert_eq!(evidence.likelihood_for("missing"), None);
}

#[test]
fn engine_config_accessor() {
    let engine = default_engine();
    assert!(engine.config().strict);
    assert_close(engine.config().min_likelihood, 1e-6, 1e-18);
}

#[test]
fn posterior_round_trips_through_temp_file() {
    // Persist the posterior and read it back, honoring the temp-file policy and
    // exercising the stability of the read-out across a serialization boundary.
    let engine = default_engine();
    let mut state = engine.init(three_hyps(), None).unwrap();
    engine
        .update(
            &mut state,
            &EvidenceUpdate::new("e")
                .with_likelihood("h0", 0.7)
                .with_likelihood("h1", 0.2)
                .with_likelihood("h2", 0.1),
        )
        .unwrap();

    let serialized = state
        .posterior()
        .iter()
        .map(|(id, p)| format!("{id}={p:.17}"))
        .collect::<Vec<_>>()
        .join("\n");

    let path =
        std::env::temp_dir().join(format!("oxirag_belief_revision_{}.txt", std::process::id()));
    std::fs::write(&path, &serialized).unwrap();
    let read_back = std::fs::read_to_string(&path).unwrap();
    let _ = std::fs::remove_file(&path);

    assert_eq!(read_back, serialized);
    let total: f64 = read_back
        .lines()
        .filter_map(|line| line.split('=').nth(1))
        .filter_map(|value| value.parse::<f64>().ok())
        .sum();
    assert_close(total, 1.0, 1e-9);
}
