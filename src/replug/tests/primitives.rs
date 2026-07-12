//! Math primitives, temperature-controlled document weights (λ), KL divergence and entropy.
//!
//! Shared helpers, constants and model fixtures live in the parent `tests`
//! module and are pulled in via `use super::*`.

use super::*;

#[test]
fn log_sum_exp_matches_the_naive_form_on_benign_input() {
    // Hand: log(e^1 + e^2 + e^3) = log(2.718281828 + 7.389056099 + 20.085536923)
    //                            = log(30.19287485) = 3.40760596...
    let value = log_sum_exp(&[1.0, 2.0, 3.0]);
    let naive = (1.0_f64.exp() + 2.0_f64.exp() + 3.0_f64.exp()).ln();
    assert_close(value, naive, "logsumexp vs naive");
    assert_close(value, 3.4076059644443806, "logsumexp hand value");
}

#[test]
fn log_sum_exp_survives_what_the_naive_form_cannot() {
    // The naive `ln(Σ exp(x))` overflows to `inf` here, and `ln(inf) = inf`.
    let huge = [800.0_f64, 801.0, 802.0];
    let naive: f64 = huge.iter().map(|&x| x.exp()).sum::<f64>().ln();
    assert!(
        naive.is_infinite(),
        "the naive form is supposed to overflow"
    );

    let stable = log_sum_exp(&huge);
    assert!(stable.is_finite());
    // Shift-invariance: logsumexp(x + c) == logsumexp(x) + c.
    assert_close(
        stable,
        log_sum_exp(&[0.0, 1.0, 2.0]) + 800.0,
        "shift invariance",
    );
}

#[test]
fn log_sum_exp_treats_neg_infinity_as_zero_mass_and_never_nans() {
    // A `-inf` component contributes exactly zero mass...
    assert_close(
        log_sum_exp(&[0.0, f64::NEG_INFINITY]),
        log_sum_exp(&[0.0]),
        "-inf contributes no mass",
    );
    // ...and an all-`-inf` input must not compute `-inf - -inf = NaN`.
    let all_empty = log_sum_exp(&[f64::NEG_INFINITY, f64::NEG_INFINITY]);
    assert!(all_empty.is_infinite() && all_empty < 0.0);
    assert!(!all_empty.is_nan());
    assert!(log_sum_exp(&[]).is_infinite());
}

#[test]
fn softmax_is_a_distribution_and_log_softmax_is_its_logarithm() {
    let logits = [2.0, -1.0, 0.5, 3.25];
    let probs = softmax(&logits);
    assert_is_distribution(&probs, "softmax");

    let log_probs = log_softmax(&logits);
    for (&p, &lp) in probs.iter().zip(&log_probs) {
        assert_close(lp, p.ln(), "log_softmax == ln(softmax)");
    }

    // Softmax is shift-invariant; that identity is what every stable
    // implementation is built on, so check it holds for ours.
    let shifted = softmax(&[12.0, 9.0, 10.5, 13.25]);
    for (&a, &b) in probs.iter().zip(&shifted) {
        assert_close(a, b, "softmax shift invariance");
    }
}

#[test]
fn softmax_hand_computed_two_way() {
    // Hand: softmax([1, 0]) = [e/(e+1), 1/(e+1)] = [0.7310585786, 0.2689414214]
    let probs = softmax(&[1.0, 0.0]);
    let e = std::f64::consts::E;
    assert_close(probs[0], e / (e + 1.0), "softmax[0]");
    assert_close(probs[1], 1.0 / (e + 1.0), "softmax[1]");
    assert_close(probs[0], 0.7310585786300049, "softmax[0] literal");
    assert_is_distribution(&probs, "softmax 2-way");
}

#[test]
fn arg_max_uses_a_total_order_and_breaks_ties_low() {
    assert_eq!(arg_max(&[0.1, 0.9, 0.3]), Some(1));
    // A tie must resolve to the *lowest* index, deterministically.
    assert_eq!(arg_max(&[0.5, 0.5, 0.5]), Some(0));
    assert_eq!(arg_max(&[]), None);
    // `total_cmp` orders `-inf` correctly rather than producing an `Option::None`
    // from `partial_cmp` (which is where a naive `.unwrap()` would panic).
    assert_eq!(arg_max(&[f64::NEG_INFINITY, -5.0]), Some(1));
}

#[test]
fn lambda_is_a_hand_computable_softmax_over_retrieval_scores() {
    // τ = 0.5, scores [0.9, 0.6, 0.1] => scaled [1.8, 1.2, 0.2]
    //   e^1.8 = 6.049647464, e^1.2 = 3.320116923, e^0.2 = 1.221402758
    //   Z = 10.591167145
    //   λ = [0.571193..., 0.313478..., 0.115328...]
    let engine =
        ReplugEngine::new(ReplugConfig::default().with_temperature(0.5)).expect("valid config");
    let documents = documents_for_weighting();
    let weights = engine
        .document_weights(&documents)
        .expect("weights computable");

    let e18 = 1.8_f64.exp();
    let e12 = 1.2_f64.exp();
    let e02 = 0.2_f64.exp();
    let z = e18 + e12 + e02;
    assert_close(weights[0], e18 / z, "lambda top");
    assert_close(weights[1], e12 / z, "lambda mid");
    assert_close(weights[2], e02 / z, "lambda low");
    assert_is_distribution(&weights, "lambda");

    // Monotone in the score: better-retrieved documents get more of the vote.
    assert!(weights[0] > weights[1] && weights[1] > weights[2]);
}

#[test]
fn lambda_tau_to_zero_collapses_onto_the_top_scored_document() {
    // τ → 0⁺ is the arg-max limit. Note the naive "divide by τ first"
    // implementation would compute 0.9 / 1e-6 = 900_000 and then exp() it —
    // this one shifts first, so nothing ever overflows on the way to the limit.
    let engine =
        ReplugEngine::new(ReplugConfig::default().with_temperature(1e-6)).expect("valid config");
    let weights = engine
        .document_weights(&documents_for_weighting())
        .expect("weights computable");

    assert_is_distribution(&weights, "lambda at tau -> 0");
    assert_close(weights[0], 1.0, "all mass on the top document");
    assert_close(weights[1], 0.0, "no mass on the middle document");
    assert_close(weights[2], 0.0, "no mass on the bottom document");

    // Even at an absurd τ the arithmetic stays finite rather than NaN.
    let extreme =
        ReplugEngine::new(ReplugConfig::default().with_temperature(1e-300)).expect("valid config");
    let extreme_weights = extreme
        .document_weights(&documents_for_weighting())
        .expect("weights computable");
    assert_is_distribution(&extreme_weights, "lambda at tau = 1e-300");
    assert_eq!(extreme_weights[0], 1.0);
}

#[test]
fn lambda_tau_to_infinity_becomes_uniform() {
    let engine =
        ReplugEngine::new(ReplugConfig::default().with_temperature(1e9)).expect("valid config");
    let weights = engine
        .document_weights(&documents_for_weighting())
        .expect("weights computable");

    assert_is_distribution(&weights, "lambda at tau -> inf");
    for (index, &weight) in weights.iter().enumerate() {
        assert!(
            (weight - 1.0 / 3.0).abs() < 1e-9,
            "weight {index} should be uniform, got {weight}"
        );
    }

    // A literally infinite τ is *usable* here rather than a NaN generator,
    // because the shift happens before the division: (s - max)/inf == 0.
    let uniform = temperature_softmax(&[0.9, 0.6, 0.1], f64::INFINITY).expect("tau = inf is fine");
    assert_is_distribution(&uniform, "lambda at tau = inf");
    for &weight in &uniform {
        assert_close(weight, 1.0 / 3.0, "exactly uniform at tau = inf");
    }
}

#[test]
fn lambda_ties_split_the_mass_evenly() {
    let engine =
        ReplugEngine::new(ReplugConfig::default().with_temperature(1e-6)).expect("valid config");
    let documents = vec![
        ReplugDocument::new("a", "A", 0.7),
        ReplugDocument::new("b", "B", 0.7),
        ReplugDocument::new("c", "C", 0.1),
    ];
    let weights = engine.document_weights(&documents).expect("weights");
    assert_is_distribution(&weights, "lambda with tied top scores");
    // Two exactly-tied maxima at τ → 0 must share the mass, not have it
    // arbitrarily assigned to whichever happened to be first.
    assert_close(weights[0], 0.5, "tied a");
    assert_close(weights[1], 0.5, "tied b");
    assert_close(weights[2], 0.0, "loser c");
}

#[test]
fn identical_retrieval_scores_give_exactly_uniform_lambda_at_any_temperature() {
    for temperature in [1e-6, 0.1, 1.0, 1e6] {
        let engine = ReplugEngine::new(ReplugConfig::default().with_temperature(temperature))
            .expect("valid config");
        let documents = vec![
            ReplugDocument::new("a", "A", 0.42),
            ReplugDocument::new("b", "B", 0.42),
            ReplugDocument::new("c", "C", 0.42),
            ReplugDocument::new("d", "D", 0.42),
        ];
        let weights = engine.document_weights(&documents).expect("weights");
        assert_is_distribution(&weights, "lambda with identical scores");
        for &weight in &weights {
            assert_close(weight, 0.25, "identical scores => uniform lambda");
        }
    }
}

#[test]
fn zero_and_negative_temperatures_are_refused_not_approximated() {
    // τ = 0 would evaluate 0/0 = NaN at the maximizing element. Refuse it.
    assert_eq!(
        temperature_log_softmax(&[1.0, 2.0], 0.0),
        Err(ReplugError::InvalidTemperature { temperature: 0.0 })
    );
    assert!(matches!(
        temperature_log_softmax(&[1.0, 2.0], -1.0),
        Err(ReplugError::InvalidTemperature { .. })
    ));
    assert!(matches!(
        temperature_log_softmax(&[1.0, 2.0], f64::NAN),
        Err(ReplugError::InvalidTemperature { .. })
    ));
    assert!(
        ReplugConfig::default()
            .with_temperature(0.0)
            .validate()
            .is_err()
    );
}

#[test]
fn kl_is_non_negative_and_vanishes_exactly_on_equality() {
    let p = log_softmax(&[1.0, 2.0, 0.5, -1.0]);
    let q = log_softmax(&[0.2, -0.3, 1.7, 2.2]);

    // Gibbs: KL >= 0, always.
    let forward = kl_divergence_from_log_probs(&p, &q).expect("kl");
    let reverse = kl_divergence_from_log_probs(&q, &p).expect("kl");
    assert!(forward >= 0.0, "KL(P||Q) = {forward} must be non-negative");
    assert!(reverse >= 0.0, "KL(Q||P) = {reverse} must be non-negative");
    assert!(forward.is_finite() && reverse.is_finite());

    // KL == 0 iff P == Q. Both directions of the "iff":
    //   (=>) identical inputs give exactly zero...
    assert_eq!(
        kl_divergence_from_log_probs(&p, &p).expect("kl"),
        0.0,
        "KL(P||P) must be exactly zero"
    );
    //   (<=) ...and non-identical inputs give strictly more than zero.
    assert!(
        forward > 1e-6,
        "KL between different distributions must be > 0"
    );

    // KL is not symmetric — asserting that keeps anyone from "simplifying" it
    // into a distance.
    assert!(
        (forward - reverse).abs() > 1e-6,
        "KL is not symmetric: {forward} vs {reverse}"
    );
}

#[test]
fn kl_hand_computed() {
    // P = [0.5, 0.5], Q = [0.25, 0.75]
    // KL(P||Q) = 0.5*ln(0.5/0.25) + 0.5*ln(0.5/0.75)
    //          = 0.5*ln(2)       + 0.5*ln(2/3)
    //          = 0.5*0.6931472   + 0.5*(-0.4054651)
    //          = 0.3465736       - 0.2027326       = 0.1438410
    let log_p = [0.5_f64.ln(), 0.5_f64.ln()];
    let log_q = [0.25_f64.ln(), 0.75_f64.ln()];
    let kl = kl_divergence_from_log_probs(&log_p, &log_q).expect("kl");
    assert_close(
        kl,
        0.5 * 2.0_f64.ln() + 0.5 * (2.0_f64 / 3.0).ln(),
        "kl formula",
    );
    assert!((kl - 0.14384104).abs() < 1e-7, "kl literal, got {kl}");
}

#[test]
fn kl_applies_the_zero_log_zero_convention_instead_of_producing_nan() {
    // P has a token whose probability underflows to EXACTLY 0.0 in f64, and Q
    // assigns that same token `-inf` log-probability. The summand is then
    // `0.0 * inf`, which is NaN unless the 0·log0 = 0 convention is applied.
    // This is the exact shape of the bug that kills naive KL implementations.
    let log_p = [0.0_f64, -800.0]; // exp(-800) == 0.0 in f64
    assert_eq!(
        log_p[1].exp(),
        0.0,
        "the premise: this really does underflow"
    );
    let log_q = [0.0_f64, f64::NEG_INFINITY];

    let kl = kl_divergence_from_log_probs(&log_p, &log_q).expect("kl");
    assert!(!kl.is_nan(), "0 * log 0 must be 0, not NaN");
    assert!(kl.is_finite());
    assert_close(kl, 0.0, "the zero-mass token contributes nothing");

    // The genuinely infinite case is reported as infinite, not clamped: P puts
    // real mass where Q has none.
    let log_p = log_softmax(&[0.0, 0.0]);
    let log_q = [0.0_f64, f64::NEG_INFINITY];
    let kl = kl_divergence_from_log_probs(&log_p, &log_q).expect("kl");
    assert!(kl.is_infinite() && kl > 0.0, "should be +inf, got {kl}");
}

#[test]
fn entropy_is_bounded_by_the_uniform_and_vanishes_on_a_point_mass() {
    // Uniform over 4 => ln(4) nats, the maximum.
    let uniform = log_softmax(&[0.0, 0.0, 0.0, 0.0]);
    assert_close(
        entropy_from_log_probs(&uniform),
        4.0_f64.ln(),
        "uniform entropy",
    );

    // A point mass (in floating point: a 1e4 logit gap) => 0 nats.
    let point_mass = log_softmax(&[1e4, 0.0, 0.0, 0.0]);
    let entropy = entropy_from_log_probs(&point_mass);
    assert!(entropy.is_finite() && entropy >= 0.0);
    assert_close(entropy, 0.0, "point-mass entropy");

    let general = entropy_from_log_probs(&log_softmax(&[2.0, 1.0, 0.0, -1.0]));
    assert!(general > 0.0 && general < 4.0_f64.ln());
}
