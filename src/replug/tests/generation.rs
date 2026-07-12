//! Autoregressive multi-step generation: greedy, sampling, EOS, and top-k truncation.
//!
//! Shared helpers, constants and model fixtures live in the parent `tests`
//! module and are pulled in via `use super::*`.

use super::*;

#[test]
fn greedy_generation_ensembles_at_every_step_and_extends_every_document_context() {
    let model = stateful_model();
    let engine = ReplugEngine::new(
        ReplugConfig::default()
            .with_temperature(1.0)
            .with_max_tokens(8),
    )
    .expect("valid config");

    let documents = vec![
        ReplugDocument::new("a", "SOURCE-A body", 2.0),
        ReplugDocument::new("b", "SOURCE-B body", 2.0),
    ];

    let output = engine
        .generate(&model, " Q:", &documents)
        .expect("generation");

    // λ = [0.5, 0.5] from the equal scores.
    // Step 0: 0.5*[.50,.35,.10,.05] + 0.5*[.10,.35,.50,.05] = [.30,.35,.30,.05]
    //         -> argmax = 1 = "G".  (Note: NEITHER document's argmax is "G".)
    // Step 1: both docs now see "G"; mixture = [.05,.05,.80,.10] -> "B".
    // Step 2: both docs see "GB";  mixture = [.02,.02,.06,.90] -> "<eos>", stop.
    assert_eq!(output.text, "GB");
    assert_eq!(output.token_ids, vec![1, 2, 3]);
    assert_eq!(output.stats.generated_tokens, 3);

    // Three steps, two documents => six forward passes. The cost is reported,
    // not hidden.
    assert_eq!(output.stats.lm_calls, 6);
    assert_eq!(output.stats.documents_retained, 2);
    assert_eq!(output.document_ids, vec!["a".to_string(), "b".to_string()]);

    // Step 0's ensembled distribution, hand-computed.
    let step0 = output.steps[0]
        .distribution
        .as_ref()
        .expect("distributions recorded by default");
    assert_is_distribution(step0, "step 0");
    assert_close_f32(step0[0], 0.30, "step0 p(R)");
    assert_close_f32(step0[1], 0.35, "step0 p(G)");
    assert_close_f32(step0[2], 0.30, "step0 p(B)");
    assert_close_f32(step0[3], 0.05, "step0 p(eos)");

    // The generated token really was the ensemble's arg-max, and its recorded
    // log-probability really is the mixture's.
    assert_close_f32(
        output.steps[0].token_log_prob,
        0.35_f64.ln(),
        "step0 log p(G)",
    );
    assert_close_f32(
        output.steps[1].token_log_prob,
        0.80_f64.ln(),
        "step1 log p(B)",
    );

    // Sequence log-probability accumulates the ensembled per-token values:
    // ln(0.35) + ln(0.80) + ln(0.90) = -1.0498221 + -0.2231436 + -0.1053605
    let expected_sequence = 0.35_f64.ln() + 0.80_f64.ln() + 0.90_f64.ln();
    assert_close_f32(
        output.stats.sequence_log_prob,
        expected_sequence,
        "sequence log-prob",
    );
    assert_close_f32(
        output.stats.mean_token_log_prob,
        expected_sequence / 3.0,
        "mean token log-prob",
    );

    // Entropy is a real diagnostic: step 0 (documents disagree) must be more
    // uncertain than step 1 (documents agree).
    assert!(
        output.steps[0].entropy_nats > output.steps[1].entropy_nats,
        "disagreement should raise the ensemble's entropy: {} vs {}",
        output.steps[0].entropy_nats,
        output.steps[1].entropy_nats
    );
    for step in &output.steps {
        assert!(step.entropy_nats.is_finite() && step.entropy_nats >= 0.0);
    }

    // λ entropy: two equally-weighted documents => ln(2) nats.
    assert_close(
        output.stats.weight_entropy_nats,
        2.0_f64.ln(),
        "lambda entropy",
    );
}

#[test]
fn eos_halts_generation_before_max_tokens() {
    let model = stateful_model();
    let engine = ReplugEngine::new(
        ReplugConfig::default()
            .with_temperature(1.0)
            .with_max_tokens(100),
    )
    .expect("valid config");
    let documents = vec![
        ReplugDocument::new("a", "SOURCE-A body", 2.0),
        ReplugDocument::new("b", "SOURCE-B body", 2.0),
    ];
    let output = engine
        .generate(&model, " Q:", &documents)
        .expect("generate");

    assert_eq!(output.token_ids.last(), Some(&3), "must stop on <eos>");
    assert_eq!(output.stats.generated_tokens, 3);
    // The EOS token is recorded but its text is not appended to the output.
    assert_eq!(output.text, "GB");
}

#[test]
fn sampling_is_seeded_reproducible_and_draws_from_the_mixture() {
    let model = stateful_model();
    let documents = vec![
        ReplugDocument::new("a", "SOURCE-A body", 2.0),
        ReplugDocument::new("b", "SOURCE-B body", 2.0),
    ];

    let sampled = |seed: u64| {
        let engine = ReplugEngine::new(
            ReplugConfig::default()
                .with_temperature(1.0)
                .with_max_tokens(4)
                .with_decoding(ReplugDecoding::Sampling {
                    temperature: 1.0,
                    seed,
                }),
        )
        .expect("valid config");
        engine
            .generate(&model, " Q:", &documents)
            .expect("generation")
    };

    // Same seed => bit-identical run. A "sampled" pipeline that cannot be
    // reproduced cannot be debugged.
    assert_eq!(sampled(7).token_ids, sampled(7).token_ids);

    // Every sampled token must be in-vocabulary and carry a finite, non-positive
    // log-probability under the mixture.
    let output = sampled(7);
    for step in &output.steps {
        assert!(step.token_id < 4);
        assert!(step.token_log_prob.is_finite() && step.token_log_prob <= 0.0);
        assert_is_distribution(
            step.distribution.as_ref().expect("recorded"),
            "sampled step distribution",
        );
    }

    // A near-zero sampling temperature degenerates to greedy, which is the
    // defining limit of temperature sampling.
    let engine = ReplugEngine::new(
        ReplugConfig::default()
            .with_temperature(1.0)
            .with_max_tokens(4)
            .with_decoding(ReplugDecoding::Sampling {
                temperature: 1e-6,
                seed: 99,
            }),
    )
    .expect("valid config");
    let cold = engine
        .generate(&model, " Q:", &documents)
        .expect("generate");
    let greedy_engine = ReplugEngine::new(
        ReplugConfig::default()
            .with_temperature(1.0)
            .with_max_tokens(4),
    )
    .expect("valid config");
    let greedy = greedy_engine
        .generate(&model, " Q:", &documents)
        .expect("generate");
    assert_eq!(
        cold.token_ids, greedy.token_ids,
        "sampling at T -> 0 must reproduce greedy decoding"
    );
}

#[test]
fn sampling_empirically_recovers_the_mixture_probabilities() {
    // The strongest available check that the sampler really samples from the
    // ensembled distribution rather than from something adjacent to it: draw a
    // few thousand times and compare the empirical frequencies against the
    // hand-computed mixture [0.30, 0.35, 0.30, 0.05].
    let model = stateful_model();
    let engine =
        ReplugEngine::new(ReplugConfig::default().with_temperature(1.0)).expect("valid config");
    let documents = vec![
        ReplugDocument::new("a", "SOURCE-A body", 2.0),
        ReplugDocument::new("b", "SOURCE-B body", 2.0),
    ];
    let (log_distribution, _, _) = engine
        .ensemble_next_token(&model, " Q:", &documents)
        .expect("ensemble");

    const DRAWS: usize = 200_000;
    let mut rng = ReplugRng::new(0xC0FF_EE12_3456_789A);
    let mut counts = [0_usize; 4];
    for _ in 0..DRAWS {
        let token = rng
            .sample_from_log_probs(&log_distribution)
            .expect("non-empty distribution");
        counts[token] += 1;
    }

    let expected = [0.30, 0.35, 0.30, 0.05];
    for (token, &expected_p) in expected.iter().enumerate() {
        let empirical = counts[token] as f64 / DRAWS as f64;
        // Standard error at p ~ 0.35 over 200k draws is ~0.001; 0.01 is a
        // ~10-sigma band, so this is deterministic in practice (the PRNG is
        // seeded) yet still a real check on the sampler.
        assert!(
            (empirical - expected_p).abs() < 0.01,
            "token {token}: sampled {empirical}, mixture says {expected_p}"
        );
    }
}

#[test]
fn top_k_truncation_happens_before_lambda_is_computed() {
    let model = consensus_runner_up_model();
    let engine = ReplugEngine::new(
        ReplugConfig::default()
            .with_temperature(1.0)
            .with_top_k_documents(Some(2)),
    )
    .expect("valid config");

    let documents = vec![
        ReplugDocument::new("low", "SOURCE-A: ...", 0.1),
        ReplugDocument::new("a", "SOURCE-A: ...", 3.0),
        ReplugDocument::new("b", "SOURCE-B: ...", 3.0),
    ];

    let (_, weights, retained) = engine
        .ensemble_next_token(&model, "q", &documents)
        .expect("ensemble");

    // Only the top 2 by score survive, sorted descending with the id as the
    // deterministic tiebreak.
    assert_eq!(retained.len(), 2);
    assert_eq!(retained[0].id, "a");
    assert_eq!(retained[1].id, "b");

    // And λ is a softmax over the RETAINED scores, so it sums to 1 over exactly
    // the two documents that participate — not to 1 minus whatever the dropped
    // document would have taken.
    assert_is_distribution(&weights, "lambda after truncation");
    assert_close(weights[0], 0.5, "retained a");
    assert_close(weights[1], 0.5, "retained b");
}
