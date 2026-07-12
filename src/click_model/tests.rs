#![allow(
    clippy::float_cmp,
    clippy::similar_names,
    clippy::cast_precision_loss,
    clippy::cast_possible_truncation,
    clippy::cast_sign_loss,
    clippy::many_single_char_names,
    clippy::too_many_lines,
    clippy::unreadable_literal,
    clippy::manual_midpoint,
    clippy::suboptimal_flops,
    clippy::needless_range_loop,
    clippy::items_after_statements,
    clippy::doc_markdown
)]
//! Tests for `click_model`.
//!
//! The sections mirror the module: log validation, then each of the three click
//! models — their hand-computed EM steps, their identifiability, their edge
//! cases, and above all **parameter recovery from synthetic logs with known
//! ground truth**.
//!
//! The headline tests are:
//!
//! - `pbm_em_recovers_true_parameters` / `dbn_em_recovers_true_parameters` /
//!   `cascade_recovers_true_attractiveness` — generate a log from *known*
//!   `γ`, `α`, `σ`, fit by EM, demand the true values back. A wrong E-step
//!   cannot pass these.
//!
//! The cross-model EM-monotonicity invariant, and the counterfactual half —
//! propensities, IPS / SNIPS / DR, the `DebiasedRanker`, and the end-to-end bias
//! demonstration — live in the sibling `tests_counterfactual` module.

use super::cascade::{CascadeClickModel, cascade_examined_depth};
use super::counterfactual::naive_click_through_rates;
use super::dbn::DbnClickModel;
use super::pbm::PositionBasedModel;
use super::synthetic::{ClickLogSimulator, ClickSimulationModel};
use super::types::{
    ClickLog, ClickModel, ClickModelConfig, ClickModelError, ClickModelFit, ClickSession,
    position_gain,
};

// ── helpers ──────────────────────────────────────────────────────────────────

pub(super) fn ids(names: &[&str]) -> Vec<String> {
    names.iter().map(|name| (*name).to_owned()).collect()
}

pub(super) fn session(query: &str, docs: &[&str], clicks: &[bool]) -> ClickSession {
    ClickSession::new(query, ids(docs), clicks.to_vec())
}

pub(super) fn log_of(sessions: Vec<ClickSession>) -> ClickLog {
    ClickLog::from_sessions(sessions).expect("hand-built log is valid")
}

pub(super) fn close(actual: f64, expected: f64, epsilon: f64) -> bool {
    (actual - expected).abs() < epsilon
}

/// The largest absolute deviation between two equal-length vectors.
pub(super) fn max_abs_error(actual: &[f64], expected: &[f64]) -> f64 {
    actual
        .iter()
        .zip(expected.iter())
        .map(|(left, right)| (left - right).abs())
        .fold(0.0_f64, f64::max)
}

// ═══════════════════════════════════════════════════════════════════════════
// ClickSession / ClickLog validation
// ═══════════════════════════════════════════════════════════════════════════

#[test]
fn session_records_clicks_by_rank() {
    let s = session("q", &["a", "b", "c"], &[false, true, true]);
    assert_eq!(s.len(), 3);
    assert!(!s.is_empty());
    assert_eq!(s.first_click_rank(), Some(1));
    assert_eq!(s.last_click_rank(), Some(2));
    assert_eq!(s.click_count(), 2);
}

#[test]
fn session_with_no_clicks_has_no_click_ranks() {
    let s = session("q", &["a", "b"], &[false, false]);
    assert_eq!(s.first_click_rank(), None);
    assert_eq!(s.last_click_rank(), None);
    assert_eq!(s.click_count(), 0);
}

#[test]
fn empty_session_is_rejected() {
    let s = ClickSession::new("q", Vec::new(), Vec::new());
    assert!(matches!(
        s.validate(3),
        Err(ClickModelError::EmptySession { session_index: 3 })
    ));
    assert!(matches!(
        ClickLog::from_sessions(vec![s]),
        Err(ClickModelError::EmptySession { session_index: 0 })
    ));
}

#[test]
fn click_length_mismatch_is_rejected() {
    let s = ClickSession::new("q", ids(&["a", "b"]), vec![true]);
    assert!(matches!(
        s.validate(0),
        Err(ClickModelError::ClickLengthMismatch {
            doc_count: 2,
            click_count: 1,
            ..
        })
    ));
}

#[test]
fn examination_length_mismatch_is_rejected() {
    let s = session("q", &["a", "b"], &[true, false]).with_examinations(vec![true]);
    assert!(matches!(
        s.validate(0),
        Err(ClickModelError::ExaminationLengthMismatch {
            doc_count: 2,
            examination_count: 1,
            ..
        })
    ));
}

#[test]
fn click_at_an_unexamined_position_is_rejected() {
    // `click ⇒ examined` is an axiom of every model here.
    let s = session("q", &["a", "b"], &[false, true]).with_examinations(vec![true, false]);
    assert!(matches!(
        s.validate(0),
        Err(ClickModelError::ClickWithoutExamination { rank: 1, .. })
    ));
}

#[test]
fn duplicate_document_within_a_session_is_rejected() {
    let s = session("q", &["a", "b", "a"], &[false, false, false]);
    assert!(matches!(
        s.validate(0),
        Err(ClickModelError::DuplicateDocumentInSession { doc_id, .. }) if doc_id == "a"
    ));
}

#[test]
fn click_log_reports_depth_vocabulary_and_counts() {
    let log = log_of(vec![
        session("q1", &["a", "b"], &[true, false]),
        session("q2", &["c", "a", "b"], &[false, false, true]),
    ]);
    assert_eq!(log.len(), 2);
    assert_eq!(log.depth(), 3);
    assert_eq!(log.doc_ids(), ids(&["a", "b", "c"])); // sorted, for determinism
    assert_eq!(log.impression_count(), 5);
    assert_eq!(log.click_count(), 2);
    assert!(!log.is_empty());
}

#[test]
fn click_log_impressions_iterate_session_then_rank() {
    let log = log_of(vec![session("q", &["a", "b"], &[true, false])]);
    let impressions: Vec<_> = log.impressions().collect();
    assert_eq!(impressions.len(), 2);
    assert_eq!(impressions[0].doc_id, "a");
    assert_eq!(impressions[0].rank, 0);
    assert!(impressions[0].clicked);
    assert_eq!(impressions[0].examined, None);
    assert_eq!(impressions[1].doc_id, "b");
    assert_eq!(impressions[1].rank, 1);
    assert!(!impressions[1].clicked);
}

#[test]
fn click_log_push_validates_and_subset_selects() {
    let mut log = ClickLog::new();
    assert!(log.is_empty());
    log.push(session("q", &["a"], &[true])).expect("valid");
    assert!(
        log.push(ClickSession::new("q", Vec::new(), Vec::new()))
            .is_err()
    );
    assert_eq!(log.len(), 1, "a rejected session must not be appended");

    log.push(session("q", &["b"], &[false])).expect("valid");
    let subset = log.subset(&[1, 99]); // out-of-range indices are skipped
    assert_eq!(subset.len(), 1);
    assert_eq!(subset.sessions()[0].ranked_doc_ids[0], "b");
}

#[test]
fn position_gain_is_the_dcg_discount() {
    assert!(close(position_gain(0), 1.0, 1e-12));
    assert!(close(position_gain(1), 1.0 / 3.0_f64.log2(), 1e-12));
    assert!(close(position_gain(2), 0.5, 1e-12));
    for rank in 0..20 {
        assert!(position_gain(rank) > position_gain(rank + 1));
    }
}

#[test]
fn config_validation_rejects_impossible_values() {
    assert!(ClickModelConfig::default().validate().is_ok());
    for bad in [
        ClickModelConfig::default().with_max_iterations(0),
        ClickModelConfig::default().with_tolerance(-1.0),
        ClickModelConfig::default().with_parameter_floor(0.0),
        ClickModelConfig::default().with_parameter_floor(0.6),
        ClickModelConfig::default().with_propensity_clip(0.0),
        ClickModelConfig::default().with_propensity_clip(1.5),
        ClickModelConfig::default().with_initial_attractiveness(0.0),
        ClickModelConfig::default().with_initial_satisfaction(1.0),
        ClickModelConfig::default().with_initial_persistence(1.0),
        ClickModelConfig::default().with_initial_examination_decay(0.0),
    ] {
        assert!(matches!(
            bad.validate(),
            Err(ClickModelError::InvalidConfig { .. })
        ));
    }
}

#[test]
fn config_clamps_parameters_into_the_open_unit_interval() {
    let config = ClickModelConfig::default().with_parameter_floor(1e-3);
    assert_eq!(config.clamp_parameter(0.0), 1e-3);
    assert_eq!(config.clamp_parameter(1.0), 1.0 - 1e-3);
    assert_eq!(config.clamp_parameter(0.5), 0.5);
    assert_eq!(config.clamp_parameter(f64::NAN), 1e-3);
}

#[test]
fn fit_report_detects_monotonicity_violations() {
    let good = ClickModelFit {
        iterations: 2,
        converged: true,
        log_likelihood_history: vec![-10.0, -5.0, -5.0],
        final_log_likelihood: -5.0,
    };
    assert!(good.is_monotone(0.0));
    assert_eq!(good.worst_decrease(), 0.0);
    assert!(close(good.total_improvement(), 5.0, 1e-12));

    let bad = ClickModelFit {
        iterations: 2,
        converged: false,
        log_likelihood_history: vec![-10.0, -5.0, -7.0],
        final_log_likelihood: -7.0,
    };
    assert!(!bad.is_monotone(0.0));
    assert!(close(bad.worst_decrease(), 2.0, 1e-12));
}

// ═══════════════════════════════════════════════════════════════════════════
// PBM: the E-step and M-step, computed by hand
// ═══════════════════════════════════════════════════════════════════════════

#[test]
fn pbm_one_em_step_matches_the_hand_derived_bayes_posteriors() {
    // Two sessions of [a, b]. γ₀ = 1 (anchored), γ₁ = 0.5 (decay 0.5),
    // α_a = α_b = 0.2 at initialisation.
    let log = log_of(vec![
        session("q", &["a", "b"], &[true, false]),
        session("q", &["a", "b"], &[false, true]),
    ]);
    let config = ClickModelConfig::default()
        .with_max_iterations(1)
        .with_tolerance(0.0)
        .with_parameter_floor(1e-9)
        .with_initial_attractiveness(0.2)
        .with_initial_examination_decay(0.5);
    let mut model = PositionBasedModel::new(config);
    let fit = model.fit(&log).expect("fit succeeds");
    assert_eq!(fit.iterations, 1);

    // ── E-step, by hand ─────────────────────────────────────────────────────
    // s0 r0 (a, clicked)      → E = 1,   A = 1
    // s0 r1 (b, not clicked)  → γ=0.5, α=0.2, 1−γα = 0.9
    //                           E = 0.5·0.8/0.9 = 4/9
    //                           A = 0.2·0.5/0.9 = 1/9
    // s1 r0 (a, not clicked)  → γ=1.0, α=0.2, 1−γα = 0.8
    //                           E = 1.0·0.8/0.8 = 1   (the top rank IS examined)
    //                           A = 0.2·0.0/0.8 = 0   (so it must be unattractive)
    // s1 r1 (b, clicked)      → E = 1,   A = 1
    //
    // ── M-step: the mean of the posteriors ──────────────────────────────────
    // γ₀  = 1 (anchored, never updated)
    // γ₁  = (4/9 + 1) / 2   = 0.7222…
    // α_a = (1 + 0) / 2     = 0.5
    // α_b = (1/9 + 1) / 2   = 0.5555…
    assert_eq!(model.examination()[0], 1.0);
    assert!(
        close(model.examination()[1], (4.0 / 9.0 + 1.0) / 2.0, 1e-12),
        "γ₁ = {}",
        model.examination()[1]
    );
    assert!(close(model.attractiveness("a").expect("known"), 0.5, 1e-12));
    assert!(close(
        model.attractiveness("b").expect("known"),
        (1.0 / 9.0 + 1.0) / 2.0,
        1e-12
    ));
}

#[test]
fn pbm_predict_click_prob_is_the_product_of_the_two_latents() {
    let log = log_of(vec![
        session("q", &["a", "b"], &[true, false]),
        session("q", &["b", "a"], &[false, true]),
    ]);
    let mut model = PositionBasedModel::new(ClickModelConfig::default());
    model.fit(&log).expect("fit succeeds");
    for doc in ["a", "b"] {
        for rank in 0..2 {
            let expected =
                model.examination()[rank] * model.attractiveness(doc).expect("known doc");
            assert!(close(
                model.predict_click_prob(doc, rank).expect("known"),
                expected,
                1e-12
            ));
        }
    }
}

// ═══════════════════════════════════════════════════════════════════════════
// PBM: THE headline test — parameter recovery from a synthetic log
// ═══════════════════════════════════════════════════════════════════════════

#[test]
fn pbm_em_recovers_true_parameters() {
    // Ground truth. γ₀ = 1 matches the identifiability anchor, so the true
    // parameters are the ones EM should actually be able to reach.
    let true_examination = vec![1.0, 0.72, 0.51, 0.36, 0.24, 0.15, 0.09, 0.05];
    let true_attractiveness = vec![0.85, 0.7, 0.55, 0.45, 0.35, 0.25, 0.15, 0.08, 0.5, 0.62];
    let doc_ids: Vec<String> = (0..true_attractiveness.len())
        .map(|index| format!("d{index}"))
        .collect();

    let log = ClickLogSimulator::new(doc_ids.clone())
        .with_attractiveness(true_attractiveness.clone())
        .with_examination(true_examination.clone())
        .with_slate_size(true_examination.len())
        .with_seed(0x0C11_C43D)
        .simulate(ClickSimulationModel::PositionBased, 12_000)
        .expect("valid simulation");

    let mut model = PositionBasedModel::new(
        ClickModelConfig::default()
            .with_max_iterations(200)
            .with_tolerance(1e-4),
    );
    let fit = model.fit(&log).expect("fit succeeds");

    // EM's theorem, asserted. The log-likelihood here is ≈ −4e4, so its
    // floating-point rounding error is ≈ 1e-11 even with compensated summation;
    // the relative check is the one that means anything at this scale (see
    // `ClickModelFit::is_monotone_relative`).
    assert!(
        fit.is_monotone_relative(1e-10),
        "log-likelihood decreased by {} (final LL = {})",
        fit.worst_decrease(),
        fit.final_log_likelihood
    );
    assert!(fit.converged, "EM should reach the tolerance");

    // The log is fully identifiable: random slates connect every rank to every
    // document, and the γ₀ = 1 anchor pins the one free scale.
    let identifiability = model.identifiability().expect("fitted");
    assert!(
        identifiability.fully_identified,
        "{}",
        identifiability.summary()
    );

    let fitted_attractiveness: Vec<f64> = doc_ids
        .iter()
        .map(|doc| model.attractiveness(doc).expect("known doc"))
        .collect();
    let examination_error = max_abs_error(model.examination(), &true_examination);
    let attractiveness_error = max_abs_error(&fitted_attractiveness, &true_attractiveness);

    println!("PBM recovery over 12,000 sessions:");
    println!("  true γ     = {true_examination:?}");
    println!("  fitted γ   = {:?}", model.examination());
    println!("  true α     = {true_attractiveness:?}");
    println!("  fitted α   = {fitted_attractiveness:?}");
    println!("  max |Δγ|   = {examination_error:.5}");
    println!("  max |Δα|   = {attractiveness_error:.5}");
    println!("  iterations = {}", fit.iterations);

    // Sampling noise on ~9,600 impressions per document is around 0.005–0.01 here;
    // a tolerance of 0.02 leaves room for it while remaining an order of magnitude
    // tighter than the error any wrong E-step or M-step would produce.
    assert!(
        examination_error < 0.02,
        "examination recovery error {examination_error} is too large"
    );
    assert!(
        attractiveness_error < 0.02,
        "attractiveness recovery error {attractiveness_error} is too large"
    );
}

#[test]
fn pbm_anchor_is_what_makes_the_scale_identifiable() {
    // The same log, fitted with and without the γ₀ = 1 anchor. Both reproduce
    // the observed click rates (the PRODUCTS are identical); only the anchored
    // fit recovers the true SPLIT into examination and attractiveness.
    let log = ClickLogSimulator::new(ids(&["a", "b", "c"]))
        .with_attractiveness(vec![0.8, 0.5, 0.2])
        .with_examination(vec![1.0, 0.5, 0.25])
        .with_seed(31)
        .simulate(ClickSimulationModel::PositionBased, 4_000)
        .expect("valid simulation");

    let mut anchored =
        PositionBasedModel::new(ClickModelConfig::default().with_max_iterations(300));
    anchored.fit(&log).expect("fit succeeds");
    assert_eq!(anchored.examination()[0], 1.0);
    assert!(close(anchored.examination()[1], 0.5, 0.05));
    assert!(close(
        anchored.attractiveness("a").expect("known"),
        0.8,
        0.05
    ));

    let mut unanchored = PositionBasedModel::new(
        ClickModelConfig::default()
            .with_max_iterations(300)
            .with_anchor_top_examination(false),
    );
    unanchored.fit(&log).expect("fit succeeds");
    let report = unanchored.identifiability().expect("fitted");
    assert!(!report.anchored);
    assert!(
        !report.fully_identified,
        "without the anchor, NOTHING is identified"
    );
    assert!(report.summary().contains("global scale"));

    // The unanchored fit does not fail loudly — it silently returns whatever its
    // initialisation implied (here `γ_r = decay^r`, whose γ₀ is already 1). That
    // quiet agreement is exactly the danger the anchor exists to remove: had the
    // initialisation started elsewhere on the ridge, EM would happily have stayed
    // there, and nothing in the likelihood would ever have objected. The next
    // test proves that the likelihood really is blind to where on the ridge you
    // sit.
    assert!(
        close(unanchored.examination()[0], 1.0, 1e-3),
        "unanchored EM simply keeps the scale its initialisation handed it: {}",
        unanchored.examination()[0]
    );

    // And the PRODUCTS — the only thing the likelihood can actually see — agree
    // between the two fits.
    for rank in 0..3 {
        for doc in ["a", "b", "c"] {
            let anchored_product = anchored.predict_click_prob(doc, rank).expect("known");
            let unanchored_product = unanchored.predict_click_prob(doc, rank).expect("known");
            assert!(
                close(anchored_product, unanchored_product, 0.02),
                "rank {rank} doc {doc}: {anchored_product} vs {unanchored_product}"
            );
        }
    }
}

#[test]
fn pbm_likelihood_is_literally_blind_to_the_overall_scale() {
    // The non-identifiability theorem itself, demonstrated rather than asserted:
    // rescale EVERY examination probability by `c` and EVERY attractiveness by
    // `1/c`, and the log-likelihood does not move at all — because it only ever
    // sees the products `γ_r · α_d`. THIS is why `γ₀ = 1` must be imposed from
    // outside: no quantity of data can ever pin `c` down.
    let log = ClickLogSimulator::new(ids(&["a", "b", "c"]))
        .with_attractiveness(vec![0.6, 0.4, 0.2])
        .with_examination(vec![1.0, 0.5, 0.25])
        .with_seed(77)
        .simulate(ClickSimulationModel::PositionBased, 2_000)
        .expect("valid simulation");

    let mut model = PositionBasedModel::new(ClickModelConfig::default().with_max_iterations(300));
    let fit = model.fit(&log).expect("fit succeeds");

    // Recompute the log-likelihood by hand from the module's own formula,
    // `Σ [ c·ln(γα) + (1−c)·ln(1−γα) ]`, under the fitted parameters and under a
    // rescaled copy of them.
    let scale = 0.75_f64;
    let log_likelihood_at = |rescale: f64| -> f64 {
        let mut total = 0.0_f64;
        for impression in log.impressions() {
            let gamma = model.examination()[impression.rank] * rescale;
            let alpha = model.attractiveness(impression.doc_id).expect("known") / rescale;
            let click_prob = gamma * alpha;
            total += if impression.clicked {
                click_prob.ln()
            } else {
                (1.0 - click_prob).ln()
            };
        }
        total
    };

    let unscaled = log_likelihood_at(1.0);
    let rescaled = log_likelihood_at(scale);
    assert!(
        close(unscaled, fit.final_log_likelihood, 1e-6),
        "the hand-computed likelihood must reproduce the model\'s: {unscaled} vs {}",
        fit.final_log_likelihood
    );
    assert!(
        close(rescaled, unscaled, 1e-6),
        "scaling γ by {scale} and α by 1/{scale} must leave the likelihood UNCHANGED: \
         {rescaled} vs {unscaled}"
    );

    // Yet the parameters themselves have moved a long way. The data cannot tell
    // these two models apart; only the anchor can.
    let anchored_alpha = model.attractiveness("a").expect("known");
    let rescaled_alpha = anchored_alpha / scale;
    assert!(
        (rescaled_alpha - anchored_alpha).abs() > 0.1,
        "the rescaled α is a genuinely different parameter: {rescaled_alpha} vs {anchored_alpha}"
    );
}

#[test]
fn pbm_reports_documents_the_log_cannot_identify() {
    // "lonely" only ever appears at rank 2, and rank 2 only ever holds "lonely" —
    // so {rank 2, lonely} is a component of its own, disconnected from the γ₀ = 1
    // anchor, and its scale is undetermined. It must not NaN, and the model must
    // SAY SO.
    let log = log_of(vec![
        session("q", &["a", "b", "lonely"], &[true, false, false]),
        session("q", &["b", "a", "lonely"], &[false, true, true]),
        session("q", &["a", "b", "lonely"], &[false, false, false]),
        session("q", &["b", "a", "lonely"], &[true, false, false]),
    ]);
    let mut model = PositionBasedModel::new(ClickModelConfig::default().with_max_iterations(100));
    model.fit(&log).expect("fit succeeds");

    let report = model.identifiability().expect("fitted");
    assert!(report.anchored);
    assert!(!report.fully_identified);
    assert_eq!(report.unanchored_ranks, vec![2]);
    assert_eq!(report.unanchored_doc_ids, ids(&["lonely"]));
    assert_eq!(report.component_count, 2);
    assert!(report.summary().contains("partially identified"));

    // No NaN, no infinity — every parameter is a usable probability.
    for &gamma in model.examination() {
        assert!(
            gamma.is_finite() && (0.0..=1.0).contains(&gamma),
            "γ = {gamma}"
        );
    }
    let alpha = model.attractiveness("lonely").expect("known");
    assert!(
        alpha.is_finite() && (0.0..=1.0).contains(&alpha),
        "α = {alpha}"
    );
}

#[test]
fn pbm_handles_an_all_zero_click_log_without_nan() {
    let log = log_of(vec![
        session("q", &["a", "b", "c"], &[false, false, false]),
        session("q", &["c", "b", "a"], &[false, false, false]),
    ]);
    let mut model = PositionBasedModel::new(ClickModelConfig::default().with_max_iterations(50));
    let fit = model
        .fit(&log)
        .expect("an all-zero log is legal, if uninformative");

    assert!(fit.is_monotone(1e-9));
    assert!(fit.final_log_likelihood.is_finite());
    // With no clicks anywhere, the likelihood is maximised by driving
    // attractiveness to its floor — the honest conclusion that nothing is
    // attractive. Never NaN.
    for doc in ["a", "b", "c"] {
        let alpha = model.attractiveness(doc).expect("known");
        assert!(alpha.is_finite(), "α = {alpha}");
        assert!(alpha < 1e-3, "with zero clicks, α should collapse: {alpha}");
    }
}

#[test]
fn pbm_rejects_empty_logs_and_unknown_lookups() {
    let mut model = PositionBasedModel::new(ClickModelConfig::default());

    // Before any fit.
    assert!(matches!(
        model.attractiveness("a"),
        Err(ClickModelError::NotFitted { .. })
    ));
    assert!(matches!(
        model.predict_click_prob("a", 0),
        Err(ClickModelError::NotFitted { .. })
    ));
    assert!(matches!(
        model.identifiability(),
        Err(ClickModelError::NotFitted { .. })
    ));

    assert!(matches!(
        model.fit(&ClickLog::new()),
        Err(ClickModelError::EmptyLog)
    ));

    let log = log_of(vec![session("q", &["a", "b"], &[true, false])]);
    model.fit(&log).expect("fit succeeds");
    assert!(matches!(
        model.attractiveness("ghost"),
        Err(ClickModelError::UnknownDocument { doc_id }) if doc_id == "ghost"
    ));
    assert!(matches!(
        model.predict_click_prob("a", 9),
        Err(ClickModelError::UnknownRank { rank: 9, depth: 2 })
    ));
}

// ═══════════════════════════════════════════════════════════════════════════
// Cascade model
// ═══════════════════════════════════════════════════════════════════════════

#[test]
fn cascade_examined_depth_stops_at_the_first_click() {
    assert_eq!(
        cascade_examined_depth(&session("q", &["a", "b", "c"], &[false, true, false])),
        2
    );
    assert_eq!(
        cascade_examined_depth(&session("q", &["a", "b", "c"], &[true, false, false])),
        1
    );
    assert_eq!(
        cascade_examined_depth(&session("q", &["a", "b", "c"], &[false, false, false])),
        3
    );
    // A multi-click session is outside the cascade's support; only the prefix
    // through the FIRST click is used.
    assert_eq!(
        cascade_examined_depth(&session("q", &["a", "b", "c"], &[false, true, true])),
        2
    );
}

#[test]
fn cascade_positions_below_the_first_click_contribute_nothing() {
    // THE cascade invariant, asserted directly: take a log, then append an
    // arbitrary extra document below every first click. Neither the fitted
    // parameters nor the log-likelihood may move by a single ULP.
    let base = log_of(vec![
        session("q", &["a", "b"], &[true, false]),
        session("q", &["b", "a"], &[false, true]),
        session("q", &["a", "b"], &[false, false]),
    ]);
    let extended = log_of(vec![
        session("q", &["a", "b", "ghost"], &[true, false, true]),
        session("q", &["b", "a", "ghost"], &[false, true, true]),
        // The third session has no click, so "ghost" WOULD be examined there —
        // leave it out, to isolate the invariant.
        session("q", &["a", "b"], &[false, false]),
    ]);

    let config = ClickModelConfig::default().with_max_iterations(50);
    let mut base_model = CascadeClickModel::new(config.clone());
    let base_fit = base_model.fit(&base).expect("fit succeeds");
    let mut extended_model = CascadeClickModel::new(config);
    let extended_fit = extended_model.fit(&extended).expect("fit succeeds");

    for doc in ["a", "b"] {
        assert_eq!(
            base_model.attractiveness(doc).expect("known"),
            extended_model.attractiveness(doc).expect("known"),
            "appending documents below the first click moved α_{doc}"
        );
        assert_eq!(
            base_model.examined_impressions(doc).expect("known"),
            extended_model.examined_impressions(doc).expect("known")
        );
    }
    assert_eq!(
        base_fit.final_log_likelihood, extended_fit.final_log_likelihood,
        "positions below the first click must contribute ZERO likelihood"
    );

    // And "ghost" — seen only below a first click — is honestly reported as
    // unidentified, sitting at the prior rather than at a fabricated estimate.
    assert_eq!(extended_model.unidentified_doc_ids(), vec!["ghost"]);
    assert_eq!(
        extended_model.examined_impressions("ghost").expect("known"),
        0
    );
    assert_eq!(
        extended_model.attractiveness("ghost").expect("known"),
        extended_model
            .config()
            .clamp_parameter(extended_model.config().initial_attractiveness)
    );
}

#[test]
fn cascade_attractiveness_is_the_examined_click_rate() {
    let mut sessions = Vec::new();
    for index in 0..100 {
        let top = index % 2 == 0;
        let middle = !top && index % 5 == 1; // 10 of the 50 sessions that get past `top`
        sessions.push(session(
            "q",
            &["top", "middle", "deep"],
            &[top, middle, false],
        ));
    }
    let log = log_of(sessions);
    let mut model = CascadeClickModel::new(ClickModelConfig::default());
    model.fit(&log).expect("fit succeeds");

    assert!(close(
        model.attractiveness("top").expect("known"),
        0.5,
        1e-12
    ));
    // Examined in only the 50 sessions where `top` was not clicked, and clicked
    // in 10 of them. A raw CTR over all 100 impressions would have said 0.1 —
    // half the truth, because it counted 50 impressions nobody ever saw.
    assert!(close(
        model.attractiveness("middle").expect("known"),
        0.2,
        1e-12
    ));
    assert_eq!(model.examined_impressions("middle").expect("known"), 50);
    let ctr = naive_click_through_rates(&log);
    assert!(close(ctr["middle"], 0.1, 1e-12));

    // The exact cascade prediction for a specific ranking.
    let probabilities = model
        .session_click_probabilities(&ids(&["top", "middle", "deep"]))
        .expect("known docs");
    assert!(close(probabilities[0], 0.5, 1e-12));
    assert!(close(probabilities[1], 0.5 * 0.2, 1e-12));
    let examinations = model
        .session_examination_probabilities(&ids(&["top", "middle", "deep"]))
        .expect("known docs");
    assert!(close(examinations[0], 1.0, 1e-12));
    assert!(close(examinations[1], 0.5, 1e-12));
    assert!(close(examinations[2], 0.5 * 0.8, 1e-12));
}

#[test]
fn cascade_recovers_true_attractiveness() {
    let true_attractiveness = vec![0.55, 0.4, 0.28, 0.18, 0.1, 0.05];
    let doc_ids: Vec<String> = (0..true_attractiveness.len())
        .map(|index| format!("d{index}"))
        .collect();
    let log = ClickLogSimulator::new(doc_ids.clone())
        .with_attractiveness(true_attractiveness.clone())
        .with_seed(0x00CA_5CAD)
        .simulate(ClickSimulationModel::Cascade, 12_000)
        .expect("valid simulation");

    let mut model = CascadeClickModel::new(ClickModelConfig::default().with_max_iterations(50));
    let fit = model.fit(&log).expect("fit succeeds");
    assert!(fit.is_monotone_relative(1e-10));

    let fitted: Vec<f64> = doc_ids
        .iter()
        .map(|doc| model.attractiveness(doc).expect("known"))
        .collect();
    let error = max_abs_error(&fitted, &true_attractiveness);
    println!("Cascade recovery over 12,000 sessions:");
    println!("  true α   = {true_attractiveness:?}");
    println!("  fitted α = {fitted:?}");
    println!("  max |Δα| = {error:.5}");
    assert!(error < 0.02, "cascade recovery error {error} is too large");

    // The examined-set rule is what makes this work: naive CTR, which counts
    // never-seen impressions, systematically UNDER-estimates every document.
    let ctr = naive_click_through_rates(&log);
    for (index, &truth) in true_attractiveness.iter().enumerate() {
        let naive = ctr[&format!("d{index}")];
        assert!(
            naive < truth,
            "naive CTR should under-estimate d{index}: {naive} vs {truth}"
        );
    }
}

#[test]
fn cascade_converges_in_a_single_step_because_it_has_no_latent_variables() {
    let log = log_of(vec![
        session("q", &["a", "b"], &[true, false]),
        session("q", &["b", "a"], &[false, true]),
        session("q", &["a", "b"], &[false, false]),
    ]);
    let mut model = CascadeClickModel::new(ClickModelConfig::default().with_max_iterations(50));
    let fit = model.fit(&log).expect("fit succeeds");
    assert!(fit.is_monotone(1e-12));
    // Step 1 jumps to the global MLE; step 2 is a fixed point and trips the
    // tolerance immediately.
    assert_eq!(fit.iterations, 2, "closed-form MLE should settle at once");
    assert!(fit.converged);
    assert!(close(
        fit.log_likelihood_history[1],
        fit.log_likelihood_history[2],
        1e-12
    ));
}

#[test]
fn cascade_rejects_empty_logs_and_unknown_lookups() {
    let mut model = CascadeClickModel::new(ClickModelConfig::default());
    assert!(matches!(
        model.attractiveness("a"),
        Err(ClickModelError::NotFitted { .. })
    ));
    assert!(matches!(
        model.fit(&ClickLog::new()),
        Err(ClickModelError::EmptyLog)
    ));
    model
        .fit(&log_of(vec![session("q", &["a"], &[true])]))
        .expect("fit succeeds");
    assert!(matches!(
        model.session_click_probabilities(&ids(&["ghost"])),
        Err(ClickModelError::UnknownDocument { .. })
    ));
    assert!(matches!(
        model.predict_click_prob("a", 5),
        Err(ClickModelError::UnknownRank { rank: 5, depth: 1 })
    ));
}

// ═══════════════════════════════════════════════════════════════════════════
// DBN
// ═══════════════════════════════════════════════════════════════════════════

#[test]
fn dbn_session_posterior_matches_the_hand_derived_recursions() {
    // Session [a, b, c], clicks [false, true, false] → last click k = 1.
    // α = [0.4, 0.6, 0.2], σ = [0.3, 0.5, 0.7], γ = 0.8.
    //
    //   W[3] = 1
    //   W[2] = (1 − α_c)·(γ·1 + 1 − γ)          = 0.8
    //   D    = σ_b + (1 − σ_b)·(γ·W[2] + 1 − γ)
    //        = 0.5 + 0.5·(0.8·0.8 + 0.2)        = 0.92
    //   P(S_b = 1 | data) = σ_b / D             = 0.5 / 0.92
    //   f[2] = (1 − σ_b)·γ                      = 0.4
    //   P(E_c = 1 | data) = f[2]·W[2] / D       = 0.32 / 0.92
    //   P(A_c = 1 | data) = (1 − P(E_c=1))·α_c
    //   LL   = ln(1 − α_a) + ln(α_b) + 1·ln(γ) + ln(D)
    let attractiveness = vec![0.4, 0.6, 0.2];
    let satisfaction = vec![0.3, 0.5, 0.7];
    let persistence = 0.8_f64;
    let s = session("q", &["a", "b", "c"], &[false, true, false]);

    // The E-step is a pure function of the parameters, so it can be driven
    // directly with the hand-chosen values above.
    let posterior = DbnClickModel::session_posterior(
        &s,
        &[0, 1, 2],
        &attractiveness,
        &satisfaction,
        persistence,
    );

    let w2 = 0.8_f64;
    let denominator = 0.5 + 0.5 * (0.8 * w2 + 0.2);
    assert!(close(denominator, 0.92, 1e-12));

    // Everything at or above the last click is KNOWN, not inferred.
    assert_eq!(posterior.examined[0], 1.0);
    assert_eq!(posterior.examined[1], 1.0);
    assert_eq!(posterior.attractive[0], 0.0); // examined, not clicked ⇒ unattractive
    assert_eq!(posterior.attractive[1], 1.0); // clicked ⇒ attractive
    assert_eq!(posterior.satisfied[0], 0.0); // an unclicked rank draws no S

    assert!(close(posterior.satisfied[1], 0.5 / 0.92, 1e-12));
    assert!(close(posterior.examined[2], 0.32 / 0.92, 1e-12));
    assert!(close(
        posterior.attractive[2],
        (1.0 - 0.32 / 0.92) * 0.2,
        1e-12
    ));

    let expected_ll = (1.0_f64 - 0.4).ln() + 0.6_f64.ln() + 0.8_f64.ln() + denominator.ln();
    assert!(
        close(posterior.log_likelihood, expected_ll, 1e-12),
        "{} vs {expected_ll}",
        posterior.log_likelihood
    );
}

#[test]
fn dbn_backward_recursion_never_underflows() {
    let attractiveness = vec![1.0 - 1e-6; 3];
    // Even with every document maximally attractive and a nearly-certain
    // persistence, the (1 − γ) abandonment branch keeps W bounded below.
    let w = DbnClickModel::backward_no_click(&[0, 1, 2], 1.0 - 1e-6, &attractiveness);
    assert_eq!(w.len(), 4);
    assert_eq!(w[3], 1.0);
    for value in &w {
        assert!(value.is_finite() && *value > 0.0, "W = {value}");
    }
}

#[test]
fn dbn_em_recovers_true_parameters() {
    let true_attractiveness = vec![0.7, 0.55, 0.4, 0.25];
    let true_satisfaction = vec![0.25, 0.5, 0.75, 0.4];
    let true_persistence = 0.85_f64;
    let doc_ids: Vec<String> = (0..true_attractiveness.len())
        .map(|index| format!("d{index}"))
        .collect();

    let log = ClickLogSimulator::new(doc_ids.clone())
        .with_attractiveness(true_attractiveness.clone())
        .with_satisfaction(true_satisfaction.clone())
        .with_persistence(true_persistence)
        .with_slate_size(4)
        .with_seed(0x00DB_1234)
        .simulate(ClickSimulationModel::Dbn, 16_000)
        .expect("valid simulation");

    let mut model = DbnClickModel::new(
        ClickModelConfig::default()
            .with_max_iterations(300)
            .with_tolerance(1e-6),
    );
    let fit = model.fit(&log).expect("fit succeeds");
    assert!(
        fit.is_monotone_relative(1e-10),
        "log-likelihood decreased by {} (final LL = {})",
        fit.worst_decrease(),
        fit.final_log_likelihood
    );

    let fitted_attractiveness: Vec<f64> = doc_ids
        .iter()
        .map(|doc| model.attractiveness(doc).expect("known"))
        .collect();
    let fitted_satisfaction: Vec<f64> = doc_ids
        .iter()
        .map(|doc| model.satisfaction(doc).expect("known"))
        .collect();
    let attractiveness_error = max_abs_error(&fitted_attractiveness, &true_attractiveness);
    let satisfaction_error = max_abs_error(&fitted_satisfaction, &true_satisfaction);
    let persistence_error = (model.persistence() - true_persistence).abs();

    println!("DBN recovery over 16,000 sessions:");
    println!("  true α     = {true_attractiveness:?}");
    println!("  fitted α   = {fitted_attractiveness:?}");
    println!("  true σ     = {true_satisfaction:?}");
    println!("  fitted σ   = {fitted_satisfaction:?}");
    println!(
        "  true γ     = {true_persistence}, fitted γ = {}",
        model.persistence()
    );
    println!("  max |Δα|   = {attractiveness_error:.5}");
    println!("  max |Δσ|   = {satisfaction_error:.5}");
    println!("  |Δγ|       = {persistence_error:.5}");
    println!("  iterations = {}", fit.iterations);

    assert!(
        attractiveness_error < 0.03,
        "attractiveness recovery error {attractiveness_error} is too large"
    );
    // Satisfaction is the noisiest of the three by a wide margin, and for a
    // structural reason worth naming: `σ_d` is informed *only* by clicks that are
    // the LAST click of their session, and even then only when the click was not
    // at the bottom of the page (where `D` collapses to 1 and the click says
    // nothing at all). It therefore sees a fraction of the evidence `α_d` does.
    assert!(
        satisfaction_error < 0.03,
        "satisfaction recovery error {satisfaction_error} is too large"
    );
    assert!(
        persistence_error < 0.03,
        "persistence recovery error {persistence_error} is too large"
    );
}

#[test]
fn dbn_click_at_the_last_rank_carries_no_satisfaction_information() {
    // "tail" is only ever clicked at the LAST position. There is nothing below it
    // to continue to, so whether the user was satisfied is unobservable, and the
    // model's normaliser D collapses to exactly 1 — leaving σ_tail at the prior.
    // This is the sharpest possible check of the D formula.
    let mut sessions = Vec::new();
    for index in 0..200 {
        sessions.push(session(
            "q",
            &["head", "body", "tail"],
            &[index % 3 == 0, index % 4 == 0, index % 2 == 0],
        ));
    }
    let log = log_of(sessions);
    let config = ClickModelConfig::default()
        .with_max_iterations(200)
        .with_initial_satisfaction(0.37);
    let mut model = DbnClickModel::new(config.clone());
    model.fit(&log).expect("fit succeeds");

    let prior = config.clamp_parameter(0.37);
    let sigma_tail = model.satisfaction("tail").expect("known");
    assert!(
        close(sigma_tail, prior, 1e-9),
        "σ_tail must stay at the prior {prior}, got {sigma_tail}"
    );
    // ...and the documents above it, which DO have a rank below them, learn a
    // satisfaction that has moved away from the prior.
    let sigma_head = model.satisfaction("head").expect("known");
    assert!(
        (sigma_head - prior).abs() > 1e-3,
        "σ_head should be informed by the data, got {sigma_head}"
    );
}

#[test]
fn dbn_never_clicked_documents_keep_the_satisfaction_prior() {
    let log = log_of(vec![
        session("q", &["a", "quiet"], &[true, false]),
        session("q", &["quiet", "a"], &[false, true]),
        session("q", &["a", "quiet"], &[false, false]),
    ]);
    let config = ClickModelConfig::default().with_initial_satisfaction(0.42);
    let mut model = DbnClickModel::new(config.clone());
    model.fit(&log).expect("fit succeeds");

    assert_eq!(model.unidentified_satisfaction_doc_ids(), vec!["quiet"]);
    assert!(close(
        model.satisfaction("quiet").expect("known"),
        config.clamp_parameter(0.42),
        1e-12
    ));
}

#[test]
fn dbn_examination_recursion_is_exact_for_a_known_ranking() {
    let log = log_of(vec![
        session("q", &["a", "b", "c"], &[true, false, false]),
        session("q", &["c", "b", "a"], &[false, true, false]),
        session("q", &["b", "a", "c"], &[false, false, true]),
    ]);
    let mut model = DbnClickModel::new(ClickModelConfig::default().with_max_iterations(100));
    model.fit(&log).expect("fit succeeds");

    let ranking = ids(&["a", "b", "c"]);
    let examinations = model
        .session_examination_probabilities(&ranking)
        .expect("known docs");
    let clicks = model
        .session_click_probabilities(&ranking)
        .expect("known docs");

    // e₀ = 1; e_{r+1} = e_r · γ · (1 − α·σ); click_r = e_r · α.
    assert_eq!(examinations[0], 1.0);
    let mut expected = 1.0_f64;
    for (rank, doc) in ranking.iter().enumerate() {
        let alpha = model.attractiveness(doc).expect("known");
        let sigma = model.satisfaction(doc).expect("known");
        assert!(close(examinations[rank], expected, 1e-12));
        assert!(close(clicks[rank], expected * alpha, 1e-12));
        expected *= model.persistence() * (1.0 - alpha * sigma);
    }
    // Examination is strictly non-increasing with depth.
    for rank in 1..examinations.len() {
        assert!(examinations[rank] <= examinations[rank - 1]);
    }
}

#[test]
fn dbn_handles_an_all_zero_click_log_without_nan() {
    let log = log_of(vec![
        session("q", &["a", "b", "c"], &[false, false, false]),
        session("q", &["c", "a", "b"], &[false, false, false]),
    ]);
    let mut model = DbnClickModel::new(ClickModelConfig::default().with_max_iterations(80));
    let fit = model.fit(&log).expect("an all-zero log is legal");
    assert!(fit.is_monotone(1e-9));
    assert!(fit.final_log_likelihood.is_finite());
    assert!(model.persistence().is_finite());
    for doc in ["a", "b", "c"] {
        assert!(model.attractiveness(doc).expect("known").is_finite());
        assert!(model.satisfaction(doc).expect("known").is_finite());
    }
    // Nothing was clicked, so nothing has any satisfaction evidence at all.
    assert_eq!(model.unidentified_satisfaction_doc_ids().len(), 3);
}

#[test]
fn dbn_rejects_empty_logs_and_unknown_lookups() {
    let mut model = DbnClickModel::new(ClickModelConfig::default());
    assert!(matches!(
        model.satisfaction("a"),
        Err(ClickModelError::NotFitted { .. })
    ));
    assert!(matches!(
        model.fit(&ClickLog::new()),
        Err(ClickModelError::EmptyLog)
    ));
    model
        .fit(&log_of(vec![session("q", &["a", "b"], &[true, false])]))
        .expect("fit succeeds");
    assert!(matches!(
        model.attractiveness("ghost"),
        Err(ClickModelError::UnknownDocument { .. })
    ));
    assert!(matches!(
        model.predict_click_prob("a", 7),
        Err(ClickModelError::UnknownRank { rank: 7, .. })
    ));
}
