//! Numerical stability under extreme logits and underflow, plus edge cases and honest errors.
//!
//! Shared helpers, constants and model fixtures live in the parent `tests`
//! module and are pulled in via `use super::*`.

use super::*;

#[test]
fn extreme_logits_do_not_produce_nan_or_inf_anywhere_in_the_ensemble() {
    // ±1e4 logits: the softmax underflows to an exact 0.0 / 1.0 point mass in
    // probability space, while log-space keeps every entry finite at ~∓2e4.
    // Everything downstream — the mixture, the entropy, the KL — must stay
    // finite.
    let model =
        ReplugStaticLanguageModel::new(vec!["a".to_string(), "b".to_string(), "c".to_string()])
            .expect("vocab")
            .with_rule(vec!["EXTREME-HIGH"], vec![1e4, -1e4, 0.0])
            .expect("rule")
            .with_rule(vec!["EXTREME-LOW"], vec![-1e4, 1e4, 0.0])
            .expect("rule");

    let engine =
        ReplugEngine::new(ReplugConfig::default().with_temperature(1.0)).expect("valid config");
    let documents = vec![
        ReplugDocument::new("hi", "EXTREME-HIGH doc", 1e4),
        ReplugDocument::new("lo", "EXTREME-LOW doc", -1e4),
    ];

    let (log_distribution, weights, _) = engine
        .ensemble_next_token(&model, "q", &documents)
        .expect("ensemble survives extreme logits");

    for (index, &lp) in log_distribution.iter().enumerate() {
        assert!(
            lp.is_finite(),
            "ensembled log-prob {index} must be finite, got {lp}"
        );
        assert!(lp <= TOL, "a log-probability cannot exceed 0, got {lp}");
    }
    let probs = exp_all(&log_distribution);
    assert_is_distribution(&probs, "ensemble with +-1e4 logits");

    // λ itself is a point mass here (score gap of 2e4 at τ = 1), and that is
    // *fine* — it must be a clean 1.0/0.0, not a NaN.
    assert_is_distribution(&weights, "lambda with +-1e4 scores");
    assert_eq!(weights[0], 1.0);
    assert_eq!(weights[1], 0.0);

    // The λ = 0 document contributes log λ = -inf. `-inf + finite = -inf`, which
    // logsumexp must treat as zero mass rather than as a NaN — that is the
    // 0·log 0 = 0 convention showing up inside the mixture itself.
    assert!(weights[1].ln().is_infinite());
    assert_close(
        probs[0],
        1.0,
        "the surviving document is a point mass on token 0",
    );

    // Entropy and the pooling divergence stay finite too.
    assert!(entropy_from_log_probs(&log_distribution).is_finite());
    let divergence = engine
        .pooling_divergence(&model, "q", &documents)
        .expect("divergence");
    assert!(
        divergence.is_finite(),
        "pooling divergence must be finite, got {divergence}"
    );
    assert!(divergence >= 0.0);
}

#[test]
fn a_near_zero_probability_token_does_not_break_the_log_space_kl() {
    // A vocabulary containing a token whose probability underflows to exactly
    // 0.0 in f64 (a -800 log-prob). Both the KL and the entropy must survive it.
    let logits = vec![0.0_f32, -800.0, -1.0];
    let promoted = promote_logits(&logits).expect("finite logits");
    let log_probs = log_softmax(&promoted);

    let probs = exp_all(&log_probs);
    assert_eq!(
        probs[1], 0.0,
        "the premise: this token really does underflow"
    );
    assert!(
        log_probs[1].is_finite(),
        "but its LOG-probability is finite"
    );
    assert_is_distribution(&probs, "distribution with an underflowed token");

    // The naive `p.ln()` route would give -inf here; the log-space route does not.
    assert!(probs[1].ln().is_infinite());

    let other = log_softmax(&[0.0, -900.0, -2.0]);
    let kl = kl_divergence_from_log_probs(&log_probs, &other).expect("kl");
    assert!(
        kl.is_finite() && kl >= 0.0,
        "KL must survive the underflow, got {kl}"
    );
    assert!(entropy_from_log_probs(&log_probs).is_finite());

    // And the same distribution used in a real mixture.
    let mixed =
        mixture_log_probs(&[0.5_f64.ln(), 0.5_f64.ln()], &[log_probs, other]).expect("mixture");
    assert!(mixed.iter().all(|value| value.is_finite()));
    assert_is_distribution(&exp_all(&mixed), "mixture with underflowed tokens");
}

#[test]
fn a_point_mass_document_is_handled_without_special_casing() {
    // One document is certain; the other is uniform. The mixture must be
    // λ·point_mass + (1−λ)·uniform, finite and normalized throughout.
    let model =
        ReplugStaticLanguageModel::new((0..4).map(|i| format!("t{i}")).collect::<Vec<String>>())
            .expect("vocab")
            .with_rule(vec!["CERTAIN"], vec![1e4, 0.0, 0.0, 0.0])
            .expect("point mass")
            .with_rule(vec!["UNSURE"], vec![0.0, 0.0, 0.0, 0.0])
            .expect("uniform");

    let engine =
        ReplugEngine::new(ReplugConfig::default().with_temperature(1.0)).expect("valid config");
    let documents = vec![
        ReplugDocument::new("certain", "CERTAIN doc", 1.0),
        ReplugDocument::new("unsure", "UNSURE doc", 1.0),
    ];

    let (log_distribution, weights, _) = engine
        .ensemble_next_token(&model, "q", &documents)
        .expect("ensemble");
    let probs = exp_all(&log_distribution);
    assert_is_distribution(&probs, "point-mass mixture");

    // λ = [0.5, 0.5]. p_certain = [1, 0, 0, 0] (the 1e4 gap underflows the rest
    // to an exact zero); p_unsure = [0.25, 0.25, 0.25, 0.25].
    //   y=0: 0.5*1.00 + 0.5*0.25 = 0.500 + 0.125 = 0.625
    //   y>0: 0.5*0.00 + 0.5*0.25 = 0.000 + 0.125 = 0.125
    assert_close(weights[0], 0.5, "lambda certain");
    assert_close(probs[0], 0.625, "mixture at the point mass");
    for &probability in &probs[1..4] {
        assert_close(probability, 0.125, "mixture away from the point mass");
    }

    // The point mass did not veto the other tokens — the OR-semantics of the
    // arithmetic mixture in action. A geometric pool would have zeroed them.
    let pooled = exp_all(
        &engine
            .log_linear_pool_next_token(&model, "q", &documents)
            .expect("pool"),
    );
    assert!(
        pooled[1] < 1e-6,
        "the geometric pool should have annihilated the non-certain tokens, got {}",
        pooled[1]
    );
    assert!(probs[1] > 0.1, "the mixture kept them alive");
}

#[test]
fn a_non_finite_logit_is_an_error_not_a_silently_repaired_distribution() {
    assert!(matches!(
        promote_logits(&[1.0, f32::NAN, 0.0]),
        Err(ReplugError::NonFiniteLogit { index: 1, .. })
    ));
    assert!(matches!(
        promote_logits(&[f32::INFINITY, 0.0]),
        Err(ReplugError::NonFiniteLogit { index: 0, .. })
    ));
    assert!(matches!(
        promote_logits(&[1.0, f32::NEG_INFINITY]),
        Err(ReplugError::NonFiniteLogit { index: 1, .. })
    ));
    assert_eq!(promote_logits(&[]), Err(ReplugError::EmptyLogits));

    // Through the engine: a model that returns a NaN must fail the call rather
    // than poison the mixture.
    let model = ReplugStaticLanguageModel::new(vec!["a".to_string(), "b".to_string()])
        .expect("vocab")
        .with_rule(vec!["BAD"], vec![f32::NAN, 0.0])
        .expect("rule");
    let engine = ReplugEngine::new(ReplugConfig::default()).expect("valid config");
    assert!(matches!(
        engine.ensemble_next_token(&model, "q", &[ReplugDocument::new("x", "BAD doc", 1.0)]),
        Err(ReplugError::NonFiniteLogit { .. })
    ));
}

#[test]
fn a_non_finite_retrieval_score_is_an_error() {
    let engine = ReplugEngine::new(ReplugConfig::default()).expect("valid config");
    let documents = vec![ReplugDocument::new("x", "X", f64::NAN)];
    assert!(matches!(
        engine.select_documents(&documents),
        Err(ReplugError::NonFiniteLogit { .. })
    ));
}

#[test]
fn a_thousand_document_mixture_over_a_large_vocabulary_stays_normalized() {
    // Accumulated rounding is the quiet failure mode of any mixture: sum 1000
    // weighted distributions over 1024 tokens and a sloppy implementation drifts
    // off 1. The log-space form does not.
    const DOCUMENTS: usize = 1000;
    const VOCAB: usize = 1024;

    let mut rng = ReplugRng::new(0xDEAD_BEEF_CAFE_F00D);
    let log_probs: Vec<Vec<f64>> = (0..DOCUMENTS)
        .map(|_| {
            let logits: Vec<f64> = (0..VOCAB)
                .map(|_| (rng.next_f64() - 0.5) * 40.0) // logits in [-20, 20]
                .collect();
            log_softmax(&logits)
        })
        .collect();

    let scores: Vec<f64> = (0..DOCUMENTS).map(|_| rng.next_f64() * 4.0 - 2.0).collect();
    let log_weights = temperature_log_softmax(&scores, 0.25).expect("weights");

    let mixed = mixture_log_probs(&log_weights, &log_probs).expect("mixture");
    assert!(mixed.iter().all(|value| value.is_finite()));
    assert_is_distribution(&exp_all(&mixed), "1000-document mixture");

    // The entropy of a mixture is at least the weighted average of the component
    // entropies (concavity of H — mixing can only add uncertainty). A sharp,
    // parameter-free check that the mixture really is a *mixture*.
    let weights = exp_all(&log_weights);
    let mean_component_entropy: f64 = weights
        .iter()
        .zip(&log_probs)
        .map(|(&weight, component)| weight * entropy_from_log_probs(component))
        .sum();
    let mixture_entropy = entropy_from_log_probs(&mixed);
    assert!(
        mixture_entropy >= mean_component_entropy - 1e-9,
        "H(Σ λ p) >= Σ λ H(p) must hold: {mixture_entropy} vs {mean_component_entropy}"
    );
}

#[test]
fn zero_documents_is_an_honest_error_not_a_fallback_to_the_bare_lm() {
    let model = consensus_runner_up_model();
    let engine = ReplugEngine::new(ReplugConfig::default()).expect("valid config");

    assert_eq!(engine.select_documents(&[]), Err(ReplugError::NoDocuments));
    assert_eq!(
        engine.document_weights(&[]).unwrap_err(),
        ReplugError::NoDocuments
    );
    assert_eq!(
        engine.ensemble_next_token(&model, "q", &[]).unwrap_err(),
        ReplugError::NoDocuments
    );
    assert_eq!(
        engine.generate(&model, "q", &[]).unwrap_err(),
        ReplugError::NoDocuments
    );
    assert_eq!(
        engine.lsr_signal(&model, "q", &[], &[0]).unwrap_err(),
        ReplugError::NoDocuments
    );
    assert_eq!(
        mixture_log_probs(&[], &[]).unwrap_err(),
        ReplugError::NoDocuments
    );
}

#[test]
fn an_empty_ground_truth_continuation_is_an_error() {
    // Q_LM would be uniform by construction, producing a confident-looking but
    // entirely content-free gradient.
    let model = lsr_model();
    let engine = ReplugEngine::new(ReplugConfig::default()).expect("valid config");
    let documents = misranked_documents();

    assert_eq!(
        engine.lsr_signal(&model, "q", &documents, &[]).unwrap_err(),
        ReplugError::EmptyTarget
    );
    assert_eq!(
        engine
            .ensemble_target_log_likelihood(&model, "q", &documents, &[])
            .unwrap_err(),
        ReplugError::EmptyTarget
    );
    assert_eq!(
        ReplugEngine::sequence_log_likelihood(&model, "ctx", &[]).unwrap_err(),
        ReplugError::EmptyTarget
    );
}

#[test]
fn a_model_that_contradicts_its_own_vocab_size_is_caught() {
    // The fixture refuses to be built with the wrong logit length in the first
    // place...
    let built = ReplugStaticLanguageModel::new(vec!["a".to_string(), "b".to_string()])
        .expect("vocab")
        .with_rule(vec!["X"], vec![1.0, 2.0, 3.0]);
    assert!(matches!(
        built,
        Err(ReplugError::VocabSizeMismatch {
            expected: 2,
            actual: 3,
            ..
        })
    ));

    // ...and the mixture refuses ragged per-document distributions.
    assert!(matches!(
        mixture_log_probs(&[0.0, 0.0], &[vec![-1.0, -1.0], vec![-1.0]]),
        Err(ReplugError::VocabSizeMismatch { .. })
    ));
    assert!(matches!(
        mixture_log_probs(&[0.0], &[vec![-1.0], vec![-1.0]]),
        Err(ReplugError::WeightCountMismatch {
            weights: 1,
            documents: 2
        })
    ));
}

#[test]
fn an_out_of_vocabulary_target_token_is_caught() {
    let model = lsr_model(); // vocab size 2
    let engine = ReplugEngine::new(ReplugConfig::default()).expect("valid config");
    assert!(matches!(
        ReplugEngine::sequence_log_likelihood(&model, "ctx", &[7]),
        Err(ReplugError::TokenOutOfRange {
            token_id: 7,
            vocab_size: 2
        })
    ));
    assert!(matches!(
        engine.ensemble_target_log_likelihood(&model, "q", &misranked_documents(), &[7]),
        Err(ReplugError::TokenOutOfRange { .. })
    ));
    assert!(model.decode(7).is_err());
    assert!(model.encode(&["not-a-token"]).is_err());
}

#[test]
fn invalid_configurations_are_rejected_at_construction() {
    assert!(ReplugEngine::new(ReplugConfig::default().with_temperature(-1.0)).is_err());
    assert!(ReplugEngine::new(ReplugConfig::default().with_lsr_beta(0.0)).is_err());
    assert!(ReplugEngine::new(ReplugConfig::default().with_max_tokens(0)).is_err());
    assert!(ReplugEngine::new(ReplugConfig::default().with_top_k_documents(Some(0))).is_err());
    assert!(
        ReplugEngine::new(
            ReplugConfig::default().with_decoding(ReplugDecoding::Sampling {
                temperature: 0.0,
                seed: 1,
            })
        )
        .is_err()
    );
    assert!(
        ReplugEngine::new(ReplugConfig::default().with_lsr_learning_rate(f64::INFINITY)).is_err()
    );
    assert!(ReplugEngine::new(ReplugConfig::default()).is_ok());
}

#[test]
fn a_probability_of_zero_cannot_be_turned_into_a_logit() {
    // ln(0) = -inf is not a logit. The fixture says so rather than emitting a
    // -inf and letting it propagate.
    assert!(matches!(
        ReplugStaticLanguageModel::logits_from_probabilities(&[0.5, 0.0, 0.5]),
        Err(ReplugError::InvalidConfig { .. })
    ));
    assert!(matches!(
        ReplugStaticLanguageModel::logits_from_probabilities(&[0.5, -0.1, 0.6]),
        Err(ReplugError::InvalidConfig { .. })
    ));
    assert_eq!(
        ReplugStaticLanguageModel::logits_from_probabilities(&[]),
        Err(ReplugError::EmptyLogits)
    );
    // The supported way to express a point mass: a large logit gap.
    let logits =
        ReplugStaticLanguageModel::logits_from_probabilities(&[0.5, 0.5]).expect("valid probs");
    assert_close_f32(f64::from(logits[0]), 0.5_f64.ln(), "ln(0.5) as a logit");
}

#[test]
fn record_distributions_can_be_switched_off_without_changing_the_generation() {
    let model = stateful_model();
    let documents = vec![
        ReplugDocument::new("a", "SOURCE-A body", 2.0),
        ReplugDocument::new("b", "SOURCE-B body", 2.0),
    ];

    let base = ReplugConfig::default()
        .with_temperature(1.0)
        .with_max_tokens(8);
    let recorded = ReplugEngine::new(base.clone()).expect("valid config");

    let mut lean_config = base;
    lean_config.record_distributions = false;
    let lean = ReplugEngine::new(lean_config).expect("valid config");

    let with_distributions = recorded.generate(&model, " Q:", &documents).expect("gen");
    let without = lean.generate(&model, " Q:", &documents).expect("gen");

    // The generation is identical; only the bookkeeping differs.
    assert_eq!(with_distributions.token_ids, without.token_ids);
    assert_eq!(with_distributions.text, without.text);
    assert_close(
        with_distributions.stats.sequence_log_prob,
        without.stats.sequence_log_prob,
        "sequence log-prob is unaffected by recording",
    );

    assert!(with_distributions.steps[0].distribution.is_some());
    assert!(without.steps[0].distribution.is_none());
    assert!(without.steps[0].log_distribution.is_none());
    // The chosen token and its ensembled log-probability survive either way.
    assert!(without.steps[0].token_log_prob.is_finite());
}

#[test]
fn the_fixture_model_selects_the_most_specific_matching_rule() {
    // Determinism of the fixture itself — if rule selection were ambiguous, every
    // hand-computed test above would be resting on sand.
    let model = ReplugStaticLanguageModel::new(vec!["a".to_string(), "b".to_string()])
        .expect("vocab")
        .with_default_logits(vec![0.0, 0.0])
        .expect("default")
        .with_rule(vec!["DOC"], vec![1.0, 0.0])
        .expect("general rule")
        .with_rule(vec!["DOC", "SUFFIX"], vec![0.0, 1.0])
        .expect("specific rule");

    // The general rule fires on its own...
    assert_eq!(
        model.next_token_logits("DOC body").expect("logits"),
        vec![1.0, 0.0]
    );
    // ...and the two-substring rule wins as soon as both are present, regardless
    // of insertion order relative to the general one.
    assert_eq!(
        model.next_token_logits("DOC body SUFFIX").expect("logits"),
        vec![0.0, 1.0]
    );
    // No rule matches => the default.
    assert_eq!(
        model.next_token_logits("nothing here").expect("logits"),
        vec![0.0, 0.0]
    );
    assert_eq!(model.vocab_size(), 2);
    assert_eq!(model.eos_token_id(), None);
    // A rule with no required substrings is refused — that is what the default is.
    assert!(
        model
            .with_rule(Vec::<String>::new(), vec![1.0, 1.0])
            .is_err()
    );
}

#[test]
fn the_seeded_rng_is_deterministic_and_stays_in_range() {
    let mut left = ReplugRng::new(42);
    let mut right = ReplugRng::new(42);
    for _ in 0..1000 {
        let value = left.next_f64();
        assert_eq!(
            value,
            right.next_f64(),
            "the same seed must give the same stream"
        );
        assert!(
            (0.0..1.0).contains(&value),
            "next_f64 must be in [0, 1): {value}"
        );
    }
    // Different seeds must actually diverge.
    let mut other = ReplugRng::new(43);
    assert_ne!(ReplugRng::new(42).next_u64(), other.next_u64());
}

#[test]
fn serde_round_trips_the_public_output_types() {
    let model = stateful_model();
    let engine = ReplugEngine::new(
        ReplugConfig::default()
            .with_temperature(1.0)
            .with_max_tokens(4),
    )
    .expect("valid config");
    let documents = vec![
        ReplugDocument::new("a", "SOURCE-A body", 2.0),
        ReplugDocument::new("b", "SOURCE-B body", 2.0),
    ];
    let output = engine
        .generate(&model, " Q:", &documents)
        .expect("generate");

    let json = serde_json::to_string(&output).expect("serialize");
    let restored: crate::replug::types::ReplugEnsembleOutput =
        serde_json::from_str(&json).expect("deserialize");

    // Everything discrete must round-trip EXACTLY. Any drift here would be a
    // real defect.
    assert_eq!(restored.text, output.text);
    assert_eq!(restored.token_ids, output.token_ids);
    assert_eq!(restored.document_ids, output.document_ids);
    assert_eq!(restored.steps.len(), output.steps.len());
    assert_eq!(restored.stats.lm_calls, output.stats.lm_calls);
    assert_eq!(
        restored.stats.generated_tokens,
        output.stats.generated_tokens
    );
    assert_eq!(restored.stats.vocab_size, output.stats.vocab_size);

    // The floats are compared to within an ULP rather than with `==`. This is
    // NOT slack for our arithmetic — it is honesty about JSON: `serde_json`
    // serializes an `f64` via its shortest round-tripping decimal form, but its
    // *parser* accumulates the significand arithmetically and can land one unit
    // in the last place away from the correctly-rounded value. Asserting bitwise
    // equality here would be asserting a property of `serde_json`'s float parser,
    // not of this module. A 1-ULP tolerance at |x| ~ 1 is ~2.2e-16; 1e-15 covers
    // it with room to spare while still being ~10 orders of magnitude tighter
    // than any semantic error could hide in.
    const ULP_TOL: f64 = 1e-15;
    let close = |left: f64, right: f64| (left - right).abs() <= ULP_TOL;

    for (restored_step, original_step) in restored.steps.iter().zip(&output.steps) {
        assert_eq!(restored_step.step, original_step.step);
        assert_eq!(restored_step.token_id, original_step.token_id);
        assert!(close(
            restored_step.token_log_prob,
            original_step.token_log_prob
        ));
        assert!(close(
            restored_step.entropy_nats,
            original_step.entropy_nats
        ));

        let restored_distribution = restored_step
            .distribution
            .as_ref()
            .expect("distribution recorded");
        let original_distribution = original_step
            .distribution
            .as_ref()
            .expect("distribution recorded");
        assert_eq!(restored_distribution.len(), original_distribution.len());
        for (&left, &right) in restored_distribution.iter().zip(original_distribution) {
            assert!(
                close(left, right),
                "distribution drifted: {left} vs {right}"
            );
        }
        // A round-tripped distribution is still a distribution — the property
        // that actually matters downstream.
        assert_is_distribution(restored_distribution, "round-tripped distribution");
    }

    for (&left, &right) in restored
        .document_weights
        .iter()
        .zip(&output.document_weights)
    {
        assert!(close(left, right));
    }
    assert!(close(
        restored.stats.sequence_log_prob,
        output.stats.sequence_log_prob
    ));

    // The config holds only exactly-representable values, so it DOES round-trip
    // bit-for-bit and is asserted as such.
    let config_json = serde_json::to_string(&engine.config).expect("serialize config");
    let restored_config: ReplugConfig =
        serde_json::from_str(&config_json).expect("deserialize config");
    assert_eq!(restored_config, engine.config);
}
