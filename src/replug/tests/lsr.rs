//! REPLUG-LSR: the LM-supervised retrieval feedback signal, its gradient, and convergence.
//!
//! Shared helpers, constants and model fixtures live in the parent `tests`
//! module and are pulled in via `use super::*`.

use super::*;

#[test]
fn lsr_gradient_raises_the_score_of_the_document_that_helped_the_lm() {
    // τ = 1.0, β = 1.0, η = 1.0 so every number below is hand-checkable.
    //
    // Retrieval scores s = [1.0 (useless), 0.5 (helpful)]:
    //   P_R = softmax([1.0, 0.5]) = [e^0.5/(e^0.5+1), 1/(e^0.5+1)]
    //       = [0.62245933, 0.37754067]        <- the retriever prefers USELESS
    //
    // LM target log-likelihoods for y* = ["yes"]:
    //   log P_LM(yes | USELESS ⊕ q) = ln(0.20) = -1.60943791
    //   log P_LM(yes | HELPFUL ⊕ q) = ln(0.90) = -0.10536052
    //
    //   Q_LM = softmax([-1.60943791, -0.10536052])
    //        shift by the max (-0.10536052):  [e^-1.5040774, e^0] = [0.22222222, 1.0]
    //        normalize by Z = 1.22222222:     [0.18181818, 0.81818182] = [2/11, 9/11]
    //                                          ^-- the LM prefers HELPFUL, strongly
    //
    //   (Exactly [0.2/(0.2+0.9), 0.9/(0.2+0.9)]. With β = 1, softmax(ln x) = x / Σx,
    //    so the β=1 softmax of a LOG-likelihood is just the NORMALIZED LIKELIHOOD.
    //    Worth internalizing: β is the only thing standing between a raw sequence
    //    log-likelihood and a usable preference distribution over documents.)
    //
    //   grad_j = (P_j − Q_j) / τ
    //   grad_useless = 0.62245933 − 0.18181818 = +0.44064115  -> score FALLS
    //   grad_helpful = 0.37754067 − 0.81818182 = −0.44064115  -> score RISES
    //
    //   s' = s − η·grad, η = 1:
    //   useless: 1.0 − 0.44064115 = 0.55935885
    //   helpful: 0.5 + 0.44064115 = 0.94064115
    //
    // The ranking FLIPS. That is the entire point of LSR.
    let model = lsr_model();
    let engine = ReplugEngine::new(
        ReplugConfig::default()
            .with_temperature(1.0)
            .with_lsr_beta(1.0)
            .with_lsr_learning_rate(1.0),
    )
    .expect("valid config");

    let documents = misranked_documents();
    let target = model.encode(&["yes"]).expect("yes is in the vocabulary");

    let signal = engine
        .lsr_signal(&model, "is it?", &documents, &target)
        .expect("lsr signal");

    let useless = &signal.gradients[0];
    let helpful = &signal.gradients[1];
    assert_eq!(useless.document_id, "useless");
    assert_eq!(helpful.document_id, "helpful");

    // The LM's verdict, hand-computed.
    assert_close_f32(
        useless.target_log_likelihood,
        0.20_f64.ln(),
        "log P(y*|useless)",
    );
    assert_close_f32(
        helpful.target_log_likelihood,
        0.90_f64.ln(),
        "log P(y*|helpful)",
    );
    assert!(
        helpful.target_log_likelihood > useless.target_log_likelihood,
        "the premise: HELPFUL really does raise P(y*)"
    );

    // P_R: the retriever prefers the wrong document.
    let e_half = 0.5_f64.exp();
    assert_close(
        useless.retrieval_prob,
        e_half / (e_half + 1.0),
        "P_R(useless)",
    );
    assert_close(helpful.retrieval_prob, 1.0 / (e_half + 1.0), "P_R(helpful)");
    assert!(useless.retrieval_prob > helpful.retrieval_prob);

    // Q_LM: the LM prefers the right one. With β = 1 this is the normalized
    // likelihood, 2/11 and 9/11.
    assert_close_f32(useless.lm_prob, 2.0 / 11.0, "Q_LM(useless)");
    assert_close_f32(helpful.lm_prob, 9.0 / 11.0, "Q_LM(helpful)");
    assert!(helpful.lm_prob > useless.lm_prob);
    assert_close(useless.lm_prob + helpful.lm_prob, 1.0, "Q_LM sums to 1");

    // The gradient: (P − Q) / τ, equal and opposite because both are
    // distributions over the same 2 documents.
    assert_close(
        useless.score_gradient,
        useless.retrieval_prob - useless.lm_prob,
        "grad(useless)",
    );
    assert_close(
        helpful.score_gradient,
        helpful.retrieval_prob - helpful.lm_prob,
        "grad(helpful)",
    );
    assert!(
        useless.score_gradient > 0.0,
        "the over-rated document is pushed down"
    );
    assert!(
        helpful.score_gradient < 0.0,
        "the under-rated document is pushed up"
    );
    assert_close(
        useless.score_gradient + helpful.score_gradient,
        0.0,
        "the gradients of a softmax residual sum to zero",
    );

    // THE CLAIM: one descent step REORDERS the documents toward the LM.
    assert!(
        helpful.updated_score > useless.updated_score,
        "LSR must promote HELPFUL above USELESS: {} vs {}",
        helpful.updated_score,
        useless.updated_score
    );
    assert_close_f32(
        useless.updated_score,
        1.0 - 0.44064115,
        "updated useless score",
    );
    assert_close_f32(
        helpful.updated_score,
        0.5 + 0.44064115,
        "updated helpful score",
    );

    // Before: [useless, helpful]. After: [helpful, useless].
    assert_eq!(
        signal.reranked_document_ids(),
        vec!["helpful".to_string(), "useless".to_string()],
        "the ranking must flip"
    );

    // And the loss itself: KL(Q||P) with Q = [2/11, 9/11], P = [0.62245933, 0.37754067]
    //   = (2/11)·ln((2/11)/0.62245933) + (9/11)·ln((9/11)/0.37754067)
    //   = 0.18181818·ln(0.29208...) + 0.81818182·ln(2.16713...)
    //   = 0.18181818·(-1.23079...) + 0.81818182·(0.77362...)
    //   = -0.22378...            + 0.63296...            = 0.40918...
    let expected_kl = (2.0 / 11.0) * ((2.0 / 11.0) / useless.retrieval_prob).ln()
        + (9.0 / 11.0) * ((9.0 / 11.0) / helpful.retrieval_prob).ln();
    assert_close_f32(signal.kl_loss, expected_kl, "KL loss");
    assert!(signal.kl_loss > 0.0 && signal.kl_loss.is_finite());
}

#[test]
fn repeated_lsr_steps_monotonically_decrease_the_kl_and_converge() {
    // The sharpest test of the derived gradient: Q_LM is fixed (the LM is
    // frozen), so the loss is a convex function of the scores and plain gradient
    // descent must decrease it EVERY step. A sign error, a missing 1/τ, or a
    // gradient taken with respect to the wrong argument would break monotonicity
    // immediately.
    let model = lsr_model();
    let engine = ReplugEngine::new(
        ReplugConfig::default()
            .with_temperature(1.0)
            .with_lsr_beta(1.0)
            .with_lsr_learning_rate(0.5),
    )
    .expect("valid config");
    let documents = misranked_documents();
    let target = model.encode(&["yes"]).expect("target");

    let (adapted, history) = engine
        .lsr_adapt(&model, "is it?", &documents, &target, 40)
        .expect("adapt");

    for window in history.windows(2) {
        assert!(
            window[1].kl_loss <= window[0].kl_loss + 1e-12,
            "KL must not increase: {} -> {}",
            window[0].kl_loss,
            window[1].kl_loss
        );
        assert!(window[0].kl_loss >= 0.0 && window[0].kl_loss.is_finite());
    }

    let first = history.first().expect("at least one step").kl_loss;
    let last = history.last().expect("at least one step").kl_loss;
    assert!(last < first, "KL must actually fall: {first} -> {last}");
    assert!(last < 1e-3, "KL should converge toward 0, ended at {last}");

    // Convergence means P_R has become Q_LM: the retriever now believes what the
    // LM's behaviour implied all along.
    let final_signal = history.last().expect("history");
    for gradient in &final_signal.gradients {
        assert!(
            (gradient.retrieval_prob - gradient.lm_prob).abs() < 1e-2,
            "P_R should have converged to Q_LM for {}: {} vs {}",
            gradient.document_id,
            gradient.retrieval_prob,
            gradient.lm_prob
        );
    }

    // And the adapted documents carry the corrected, reordered scores.
    let helpful = adapted
        .iter()
        .find(|document| document.id == "helpful")
        .expect("helpful survives");
    let useless = adapted
        .iter()
        .find(|document| document.id == "useless")
        .expect("useless survives");
    assert!(
        helpful.score > useless.score,
        "after adaptation the LM's preferred document must rank first: {} vs {}",
        helpful.score,
        useless.score
    );
}

#[test]
fn lsr_reordering_changes_what_the_ensemble_actually_generates() {
    // The end-to-end consequence: because λ IS P_R, a reordered retriever
    // produces a different mixture — LSR does not merely produce a number, it
    // changes the model's output. Here τ is small, so the ensemble is dominated
    // by whichever document ranks first.
    //
    // NOTE the learning rate. The gradient (P − Q)/τ carries a 1/τ factor, and
    // with two documents the update on the score GAP d = s_0 − s_1 is
    //
    //     d  ←  d − (2η/τ) · ( σ(d/τ) − Q_0 )
    //
    // whose effective step in the logit variable u = d/τ is 2η/τ². The logistic
    // derivative is bounded by 1/4, so plain gradient descent is stable only
    // while (2η/τ²)·(1/4) < 2, i.e.  η < 4τ².  At τ = 0.1 that is η < 0.04 —
    // the η = 0.5 that works fine at τ = 1.0 would OSCILLATE here, never
    // converging. η and τ genuinely have to be tuned together, exactly as the
    // module docs warn.
    let model = lsr_model();
    let engine = ReplugEngine::new(
        ReplugConfig::default()
            .with_temperature(0.1)
            .with_lsr_beta(1.0)
            .with_lsr_learning_rate(0.01) // < 4τ² = 0.04
            .with_max_tokens(1),
    )
    .expect("valid config");
    let documents = misranked_documents();
    let target = model.encode(&["yes"]).expect("target");

    // Before LSR: the retriever's favourite is USELESS, which says p(yes) = 0.2.
    let before = engine
        .ensemble_target_log_likelihood(&model, "is it?", &documents, &target)
        .expect("before");

    let (adapted, history) = engine
        .lsr_adapt(&model, "is it?", &documents, &target, 100)
        .expect("adapt");

    // Stable step size => monotone descent here too.
    for window in history.windows(2) {
        assert!(
            window[1].kl_loss <= window[0].kl_loss + 1e-12,
            "KL must not increase at a stable step size: {} -> {}",
            window[0].kl_loss,
            window[1].kl_loss
        );
    }

    let after = engine
        .ensemble_target_log_likelihood(&model, "is it?", &adapted, &target)
        .expect("after");

    // The whole loop closes: LM feedback -> retriever scores -> mixture weights
    // -> a higher probability of the ground truth under the ensemble.
    assert!(
        after > before,
        "LSR must raise the ensemble's P(y*): ln P went {before} -> {after}"
    );
    assert!(before.is_finite() && after.is_finite());

    // Concretely: the greedy ensemble now emits "yes".
    let generated = engine
        .generate(&model, "is it?", &adapted)
        .expect("generate");
    assert_eq!(generated.text, "yes");
}

#[test]
fn lsr_is_a_no_op_when_the_retriever_already_agrees_with_the_lm() {
    // KL == 0 => zero gradient => unchanged scores. The fixed point exists and
    // is where it should be.
    let model = lsr_model();
    let engine = ReplugEngine::new(
        ReplugConfig::default()
            .with_temperature(1.0)
            .with_lsr_beta(1.0)
            .with_lsr_learning_rate(1.0),
    )
    .expect("valid config");

    // Choose scores so that softmax(s) == Q_LM == [2/11, 9/11]:
    //   s = ln(Q) works, since softmax(ln q) = q.
    let documents = vec![
        ReplugDocument::new("useless", "USELESS text", (2.0_f64 / 11.0).ln()),
        ReplugDocument::new("helpful", "HELPFUL text", (9.0_f64 / 11.0).ln()),
    ];
    let target = model.encode(&["yes"]).expect("target");
    let signal = engine
        .lsr_signal(&model, "is it?", &documents, &target)
        .expect("signal");

    // Not exactly zero: Q_LM is reconstructed through the fixture's f32 logits
    // while the scores are exact f64, so P and Q agree only to the f32 budget.
    // The KL, being quadratic in (P - Q) near the fixed point, is smaller still.
    assert_close_f32(signal.kl_loss, 0.0, "KL at the fixed point");
    for gradient in &signal.gradients {
        assert_close_f32(gradient.score_gradient, 0.0, "gradient at the fixed point");
        assert_close_f32(
            gradient.updated_score,
            gradient.original_score,
            "scores unchanged at the fixed point",
        );
    }
}

#[test]
fn lsr_beta_controls_how_sharply_the_lm_preference_is_expressed() {
    let model = lsr_model();
    let documents = misranked_documents();
    let target = model.encode(&["yes"]).expect("target");

    // β → 0: Q_LM collapses onto the single most helpful document.
    let sharp = ReplugEngine::new(
        ReplugConfig::default()
            .with_temperature(1.0)
            .with_lsr_beta(1e-4),
    )
    .expect("valid config");
    let sharp_signal = sharp
        .lsr_signal(&model, "is it?", &documents, &target)
        .expect("signal");
    assert_close(
        sharp_signal.gradients[1].lm_prob,
        1.0,
        "beta -> 0 is arg-max",
    );
    assert_close(
        sharp_signal.gradients[0].lm_prob,
        0.0,
        "beta -> 0 is arg-max",
    );

    // β → ∞: Q_LM becomes uniform, i.e. the LM expresses no preference and the
    // gradient reduces to "flatten the retriever".
    let flat = ReplugEngine::new(
        ReplugConfig::default()
            .with_temperature(1.0)
            .with_lsr_beta(1e9),
    )
    .expect("valid config");
    let flat_signal = flat
        .lsr_signal(&model, "is it?", &documents, &target)
        .expect("signal");
    for gradient in &flat_signal.gradients {
        assert!(
            (gradient.lm_prob - 0.5).abs() < 1e-6,
            "beta -> inf should be uniform, got {}",
            gradient.lm_prob
        );
    }
}

#[test]
fn lsr_length_normalization_divides_by_the_continuation_length() {
    let model = lsr_model();
    let documents = misranked_documents();
    // y* = ["yes", "yes"]: log P = 2·ln(0.9) for HELPFUL, 2·ln(0.2) for USELESS,
    // because the fixture's rules do not depend on the generated prefix here.
    let target = model.encode(&["yes", "yes"]).expect("target");

    let raw =
        ReplugEngine::new(ReplugConfig::default().with_temperature(1.0)).expect("valid config");
    let raw_values = raw
        .document_target_log_likelihoods(&model, "q", &documents, &target)
        .expect("likelihoods");
    assert_close_f32(
        raw_values[0],
        2.0 * 0.20_f64.ln(),
        "raw sequence log-likelihood",
    );
    assert_close_f32(
        raw_values[1],
        2.0 * 0.90_f64.ln(),
        "raw sequence log-likelihood",
    );

    let mut normalized_config = ReplugConfig::default().with_temperature(1.0);
    normalized_config.lsr_length_normalize = true;
    let normalized = ReplugEngine::new(normalized_config).expect("valid config");
    let normalized_values = normalized
        .document_target_log_likelihoods(&model, "q", &documents, &target)
        .expect("likelihoods");
    assert_close_f32(
        normalized_values[0],
        0.20_f64.ln(),
        "mean per-token log-likelihood",
    );
    assert_close_f32(
        normalized_values[1],
        0.90_f64.ln(),
        "mean per-token log-likelihood",
    );

    // The RANKING is invariant to the normalization; only the scale changes.
    // That is exactly what the option is for.
    assert!(raw_values[1] > raw_values[0]);
    assert!(normalized_values[1] > normalized_values[0]);
}

#[test]
fn applying_a_misaligned_lsr_signal_is_refused_rather_than_guessed_at() {
    let model = lsr_model();
    let engine =
        ReplugEngine::new(ReplugConfig::default().with_temperature(1.0)).expect("valid config");
    let documents = misranked_documents();
    let target = model.encode(&["yes"]).expect("target");
    let signal = engine
        .lsr_signal(&model, "q", &documents, &target)
        .expect("signal");

    // Wrong length.
    assert!(matches!(
        ReplugEngine::apply_lsr_update(&documents[..1], &signal),
        Err(ReplugError::WeightCountMismatch { .. })
    ));

    // Right length, wrong documents. An update silently applied to the wrong
    // documents would produce a confident and completely meaningless ranking.
    let mismatched = vec![
        ReplugDocument::new("other", "USELESS text", 1.0),
        ReplugDocument::new("helpful", "HELPFUL text", 0.5),
    ];
    assert!(matches!(
        ReplugEngine::apply_lsr_update(&mismatched, &signal),
        Err(ReplugError::InvalidConfig { .. })
    ));
}
