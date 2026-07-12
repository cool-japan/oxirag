//! The REPLUG mixture: hand-computed correctness, arithmetic-vs-geometric pooling, and the ensemble-beats-its-parts value proposition.
//!
//! Shared helpers, constants and model fixtures live in the parent `tests`
//! module and are pulled in via `use super::*`.

use super::*;

#[test]
fn mixture_equals_the_hand_computed_weighted_average() {
    // Two documents, λ = [0.75, 0.25] (written directly, not derived).
    //
    //   p_1 = [0.60, 0.30, 0.10]
    //   p_2 = [0.10, 0.20, 0.70]
    //
    // Hand-computed mixture, term by term:
    //   y=0:  0.75 * 0.60 + 0.25 * 0.10  =  0.4500 + 0.0250  =  0.4750
    //   y=1:  0.75 * 0.30 + 0.25 * 0.20  =  0.2250 + 0.0500  =  0.2750
    //   y=2:  0.75 * 0.10 + 0.25 * 0.70  =  0.0750 + 0.1750  =  0.2500
    //                                                 sum    =  1.0000
    let log_weights = vec![0.75_f64.ln(), 0.25_f64.ln()];
    let log_probs = vec![
        vec![0.60_f64.ln(), 0.30_f64.ln(), 0.10_f64.ln()],
        vec![0.10_f64.ln(), 0.20_f64.ln(), 0.70_f64.ln()],
    ];

    let mixed = exp_all(&mixture_log_probs(&log_weights, &log_probs).expect("mixture"));

    assert_close(mixed[0], 0.75 * 0.60 + 0.25 * 0.10, "mixture y=0");
    assert_close(mixed[1], 0.75 * 0.30 + 0.25 * 0.20, "mixture y=1");
    assert_close(mixed[2], 0.75 * 0.10 + 0.25 * 0.70, "mixture y=2");

    assert_close(mixed[0], 0.4750, "mixture y=0 literal");
    assert_close(mixed[1], 0.2750, "mixture y=1 literal");
    assert_close(mixed[2], 0.2500, "mixture y=2 literal");

    assert_is_distribution(&mixed, "hand-computed mixture");
}

#[test]
fn mixture_through_the_engine_equals_the_hand_computed_weighted_average() {
    // The same check, but end-to-end through the engine: λ derived from real
    // retrieval scores rather than written down.
    //
    // τ = 1.0, scores [1.0, 0.0]:
    //   λ = softmax([1, 0]) = [e/(e+1), 1/(e+1)] = [0.7310585786, 0.2689414214]
    //
    //   p_A = [0.60, 0.30, 0.10]   p_B = [0.10, 0.20, 0.70]
    //
    //   y=0: 0.7310585786*0.60 + 0.2689414214*0.10 = 0.43863515 + 0.02689414 = 0.46552929
    //   y=1: 0.7310585786*0.30 + 0.2689414214*0.20 = 0.21931757 + 0.05378828 = 0.27310586
    //   y=2: 0.7310585786*0.10 + 0.2689414214*0.70 = 0.07310586 + 0.18825899 = 0.26136485
    let model =
        ReplugStaticLanguageModel::new(vec!["x".to_string(), "y".to_string(), "z".to_string()])
            .expect("vocab")
            .with_probability_rule(vec!["SRC-A"], &[0.60, 0.30, 0.10])
            .expect("rule a")
            .with_probability_rule(vec!["SRC-B"], &[0.10, 0.20, 0.70])
            .expect("rule b");

    let documents = vec![
        ReplugDocument::new("a", "SRC-A body", 1.0),
        ReplugDocument::new("b", "SRC-B body", 0.0),
    ];
    let engine =
        ReplugEngine::new(ReplugConfig::default().with_temperature(1.0)).expect("valid config");

    let (log_distribution, weights, retained) = engine
        .ensemble_next_token(&model, "the query", &documents)
        .expect("ensemble");
    let probs = exp_all(&log_distribution);

    assert_eq!(retained.len(), 2);
    let e = std::f64::consts::E;
    let lambda_a = e / (e + 1.0);
    let lambda_b = 1.0 / (e + 1.0);
    assert_close(weights[0], lambda_a, "engine lambda a");
    assert_close(weights[1], lambda_b, "engine lambda b");

    assert_close_f32(
        probs[0],
        lambda_a * 0.60 + lambda_b * 0.10,
        "engine mixture y=0",
    );
    assert_close_f32(
        probs[1],
        lambda_a * 0.30 + lambda_b * 0.20,
        "engine mixture y=1",
    );
    assert_close_f32(
        probs[2],
        lambda_a * 0.10 + lambda_b * 0.70,
        "engine mixture y=2",
    );

    // And the literals, so the test is checkable with a pocket calculator.
    assert!((probs[0] - 0.46552929).abs() < 1e-7, "got {}", probs[0]);
    assert!((probs[1] - 0.27310586).abs() < 1e-7, "got {}", probs[1]);
    assert!((probs[2] - 0.26136485).abs() < 1e-7, "got {}", probs[2]);

    assert_is_distribution(&probs, "engine mixture");
}

#[test]
fn a_single_document_ensemble_reduces_exactly_to_that_documents_distribution() {
    // With k = 1, λ = [1] and the mixture is the identity. Any deviation here
    // would mean the mixture is doing something other than what it claims.
    let distribution = [0.15, 0.05, 0.60, 0.20];
    let model =
        ReplugStaticLanguageModel::new((0..4).map(|i| format!("t{i}")).collect::<Vec<String>>())
            .expect("vocab")
            .with_probability_rule(vec!["ONLY"], &distribution)
            .expect("rule");

    let documents = vec![ReplugDocument::new("only", "ONLY doc", 0.33)];
    let engine = ReplugEngine::new(ReplugConfig::default()).expect("valid config");

    let (log_distribution, weights, _) = engine
        .ensemble_next_token(&model, "q", &documents)
        .expect("ensemble");
    let probs = exp_all(&log_distribution);

    assert_eq!(weights.len(), 1);
    assert_close(weights[0], 1.0, "single-document lambda is 1");
    for (index, (&actual, &expected)) in probs.iter().zip(&distribution).enumerate() {
        assert_close_f32(
            actual,
            expected,
            &format!("single-doc passthrough at {index}"),
        );
    }
    assert_is_distribution(&probs, "single-document ensemble");
}

#[test]
fn duplicate_documents_do_not_change_the_mixture() {
    // Mixing a distribution with itself, at any weights, must return it
    // unchanged: Σ_i λ_i p = p · Σ_i λ_i = p. A cheap but sharp check that the
    // weights really are normalized.
    let p = [0.2_f64, 0.5, 0.3];
    let log_p: Vec<f64> = p.iter().map(|x| x.ln()).collect();
    let log_weights = temperature_log_softmax(&[3.0, 1.0, -2.0], 0.4).expect("weights");
    let mixed = exp_all(
        &mixture_log_probs(&log_weights, &[log_p.clone(), log_p.clone(), log_p]).expect("mixture"),
    );
    for (&actual, &expected) in mixed.iter().zip(&p) {
        assert_close(actual, expected, "self-mixture is the identity");
    }
    assert_is_distribution(&mixed, "self-mixture");
}

#[test]
fn mixing_distributions_is_not_averaging_logits_and_they_can_disagree_on_the_argmax() {
    // This is the bug REPLUG implementations get wrong, so it gets the most
    // explicit test in the file.
    //
    //   p_1 = [0.80, 0.19, 0.01]      p_2 = [0.02, 0.19, 0.79]
    //   λ   = [0.50, 0.50]
    //
    // ARITHMETIC (what REPLUG specifies) — a linear opinion pool, an OR:
    //   y=0: 0.5*0.80 + 0.5*0.02 = 0.400 + 0.010 = 0.410   <-- ARG MAX
    //   y=1: 0.5*0.19 + 0.5*0.19 = 0.095 + 0.095 = 0.190
    //   y=2: 0.5*0.01 + 0.5*0.79 = 0.005 + 0.395 = 0.400
    //
    // GEOMETRIC (what averaging logits computes) — a log opinion pool, an AND.
    // Because softmax(Σ λ_i z_i) ∝ Π_i p_i^{λ_i}, the unnormalized values are:
    //   y=0: sqrt(0.80 * 0.02) = sqrt(0.0160) = 0.12649111
    //   y=1: sqrt(0.19 * 0.19) = 0.19                      <-- ARG MAX
    //   y=2: sqrt(0.01 * 0.79) = sqrt(0.0079) = 0.08888194
    //   Z = 0.40537305  =>  [0.31204..., 0.46876..., 0.21926...]
    //
    // The two pools pick DIFFERENT TOKENS. Token 0 is the one that a single
    // document loves and the other rejects: the OR promotes it, the AND vetoes
    // it. Token 1 is the one both documents merely tolerate: the AND rewards
    // the consensus, the OR is unimpressed. Neither is "a rounding difference"
    // from the other — they are different operations with different semantics,
    // and REPLUG specifies the first.
    let p1 = [0.80_f64, 0.19, 0.01];
    let p2 = [0.02_f64, 0.19, 0.79];
    let weights = [0.5_f64, 0.5];
    let log_weights = [0.5_f64.ln(), 0.5_f64.ln()];

    let log_p1: Vec<f64> = p1.iter().map(|x| x.ln()).collect();
    let log_p2: Vec<f64> = p2.iter().map(|x| x.ln()).collect();

    // The arithmetic mixture, checked against the hand arithmetic above.
    let arithmetic = exp_all(
        &mixture_log_probs(&log_weights, &[log_p1.clone(), log_p2.clone()]).expect("mixture"),
    );
    assert_close(arithmetic[0], 0.410, "arithmetic y=0");
    assert_close(arithmetic[1], 0.190, "arithmetic y=1");
    assert_close(arithmetic[2], 0.400, "arithmetic y=2");
    assert_is_distribution(&arithmetic, "arithmetic pool");

    // The geometric pool. Feeding the *logits* `ln p` is exactly "average the
    // logits, then softmax", since softmax(ln p) == p.
    let geometric =
        exp_all(&log_linear_pool_log_probs(&weights, &[log_p1, log_p2]).expect("log-linear pool"));
    assert_is_distribution(&geometric, "geometric pool");

    let g0 = (0.80_f64 * 0.02).sqrt();
    let g1 = (0.19_f64 * 0.19).sqrt();
    let g2 = (0.01_f64 * 0.79).sqrt();
    let z = g0 + g1 + g2;
    assert_close(
        geometric[0],
        g0 / z,
        "geometric y=0 == normalized sqrt product",
    );
    assert_close(
        geometric[1],
        g1 / z,
        "geometric y=1 == normalized sqrt product",
    );
    assert_close(
        geometric[2],
        g2 / z,
        "geometric y=2 == normalized sqrt product",
    );

    // THE POINT: the two pools disagree on which token to emit.
    assert_eq!(arg_max(&arithmetic), Some(0), "the mixture emits token 0");
    assert_eq!(
        arg_max(&geometric),
        Some(1),
        "the logit-average emits token 1"
    );
    assert_ne!(arg_max(&arithmetic), arg_max(&geometric));

    // ...and they differ elementwise, well outside any rounding tolerance.
    for index in 0..3 {
        assert!(
            (arithmetic[index] - geometric[index]).abs() > 1e-3,
            "pools should differ materially at {index}: {} vs {}",
            arithmetic[index],
            geometric[index]
        );
    }
}

#[test]
fn the_geometric_pool_lets_an_irrelevant_document_veto_the_answer() {
    // The semantic statement of the same fact, and the reason REPLUG cannot use
    // the geometric pool. Document 1 is certain the answer is token 0.
    // Document 2 is irrelevant to the question and assigns token 0 a tiny
    // probability. Under the arithmetic mixture, document 1's certainty
    // survives — no document can veto. Under the geometric pool, document 2
    // annihilates it.
    let confident = [0.98_f64, 0.01, 0.01];
    let irrelevant = [0.0001_f64, 0.4999, 0.5000];
    let weights = [0.5_f64, 0.5];
    let log_weights = [0.5_f64.ln(), 0.5_f64.ln()];
    let log_confident: Vec<f64> = confident.iter().map(|x| x.ln()).collect();
    let log_irrelevant: Vec<f64> = irrelevant.iter().map(|x| x.ln()).collect();

    let arithmetic = exp_all(
        &mixture_log_probs(
            &log_weights,
            &[log_confident.clone(), log_irrelevant.clone()],
        )
        .expect("mixture"),
    );
    // 0.5 * 0.98 + 0.5 * 0.0001 = 0.49 + 0.00005 = 0.49005 — bounded below by
    // λ_1 · p_1(0) = 0.49, exactly as the OR-semantics promises.
    assert_close(
        arithmetic[0],
        0.49005,
        "arithmetic keeps the confident answer",
    );
    assert!(
        arithmetic[0] >= 0.5 * 0.98,
        "no document can veto under the mixture"
    );
    assert_eq!(arg_max(&arithmetic), Some(0));

    let geometric = exp_all(
        &log_linear_pool_log_probs(&weights, &[log_confident, log_irrelevant])
            .expect("log-linear pool"),
    );
    // The geometric pool, hand-computed exactly (not bounded by a guessed
    // threshold — the point deserves the real numbers):
    //
    //   g_0 = sqrt(0.98 * 0.0001) = sqrt(0.000098)  = 0.00989949
    //   g_1 = sqrt(0.01 * 0.4999) = sqrt(0.004999)  = 0.07070361
    //   g_2 = sqrt(0.01 * 0.5000) = sqrt(0.005)     = 0.07071068
    //   Z   = 0.15131378
    //   =>  [0.06542362, 0.46726482, 0.46731156]
    //
    // The confident document's 0.98 has been pulled all the way down to 0.065 —
    // a 7.5x collapse from the mixture's 0.49005 — purely because ONE other
    // document, which knows nothing about the question, happened to assign the
    // token 1e-4. The token it was certain about now finishes LAST.
    let g0 = (0.98_f64 * 0.0001).sqrt();
    let g1 = (0.01_f64 * 0.4999).sqrt();
    let g2 = (0.01_f64 * 0.5000).sqrt();
    let z = g0 + g1 + g2;
    assert_close(
        geometric[0],
        g0 / z,
        "geometric y=0 == normalized sqrt product",
    );
    assert_close(
        geometric[1],
        g1 / z,
        "geometric y=1 == normalized sqrt product",
    );
    assert_close(
        geometric[2],
        g2 / z,
        "geometric y=2 == normalized sqrt product",
    );
    assert!(
        (geometric[0] - 0.06542362).abs() < 1e-7,
        "geometric p(0) literal, got {}",
        geometric[0]
    );

    // The veto, stated three ways.
    assert_ne!(arg_max(&geometric), Some(0), "the veto changed the answer");
    assert_eq!(
        arg_max(&geometric),
        Some(2),
        "the token the confident document rejected now WINS under the geometric pool"
    );
    assert!(
        arithmetic[0] / geometric[0] > 7.0,
        "the mixture should keep token 0 several times higher than the pool does: {} vs {}",
        arithmetic[0],
        geometric[0]
    );
    assert_is_distribution(&geometric, "geometric pool with a veto");
}

#[test]
fn the_two_pools_coincide_only_when_every_document_agrees() {
    // The one case where "average the logits" is harmless: identical
    // distributions. Both pools return that distribution. This is why the bug
    // is so easy to ship — it is invisible on any example where the documents
    // already agree.
    let p = [0.1_f64, 0.7, 0.2];
    let log_p: Vec<f64> = p.iter().map(|x| x.ln()).collect();
    let weights = [0.3_f64, 0.7];
    let log_weights: Vec<f64> = weights.iter().map(|x| x.ln()).collect();

    let arithmetic = exp_all(
        &mixture_log_probs(&log_weights, &[log_p.clone(), log_p.clone()]).expect("mixture"),
    );
    let geometric =
        exp_all(&log_linear_pool_log_probs(&weights, &[log_p.clone(), log_p]).expect("pool"));

    for index in 0..3 {
        assert_close(arithmetic[index], p[index], "arithmetic on agreement");
        assert_close(geometric[index], p[index], "geometric on agreement");
    }

    // And the divergence between them is exactly zero.
    let divergence = kl_divergence_from_log_probs(
        &arithmetic.iter().map(|x| x.ln()).collect::<Vec<f64>>(),
        &geometric.iter().map(|x| x.ln()).collect::<Vec<f64>>(),
    )
    .expect("kl");
    assert!(
        divergence < 1e-12,
        "KL should vanish on agreement, got {divergence}"
    );
}

#[test]
fn the_ensemble_promotes_a_token_that_no_single_document_would_have_chosen() {
    let model = consensus_runner_up_model();
    let engine = ReplugEngine::new(ReplugConfig::default()).expect("valid config");
    const CORRECT: usize = 2; // "gamma"

    // ── First: establish that each document, ON ITS OWN, gets it wrong. ──
    //
    // A single-document ensemble is exactly that document's distribution (λ = 1),
    // which the reduction test above proves — so this really is "what the LM
    // would have done with only this document in context".
    for (source, expected_argmax, doc_id) in [
        ("SOURCE-A: ...", 0_usize, "a"),
        ("SOURCE-B: ...", 1_usize, "b"),
    ] {
        let solo = vec![ReplugDocument::new(doc_id, source, 1.0)];
        let (log_probs, _, _) = engine
            .ensemble_next_token(&model, "which one?", &solo)
            .expect("solo ensemble");
        let probs = exp_all(&log_probs);
        assert_is_distribution(&probs, "solo distribution");

        assert_eq!(
            arg_max(&probs),
            Some(expected_argmax),
            "document {doc_id} alone should emit token {expected_argmax}"
        );
        assert_ne!(
            arg_max(&probs),
            Some(CORRECT),
            "document {doc_id} alone must NOT emit the correct token"
        );

        // And "gamma" is precisely RANK 2 under this document — the setup the
        // claim depends on.
        let mut ranking: Vec<usize> = (0..3).collect();
        ranking.sort_by(|&left, &right| probs[right].total_cmp(&probs[left]));
        assert_eq!(
            ranking[1], CORRECT,
            "the correct token should be the runner-up under document {doc_id}"
        );
    }

    // ── Now: the ensemble. Equal retrieval scores, so λ = [0.5, 0.5]. ──
    //
    //   y=alpha: 0.5*0.50 + 0.5*0.10 = 0.250 + 0.050 = 0.30
    //   y=beta : 0.5*0.10 + 0.5*0.50 = 0.050 + 0.250 = 0.30
    //   y=gamma: 0.5*0.40 + 0.5*0.40 = 0.200 + 0.200 = 0.40   <-- ARG MAX
    let documents = vec![
        ReplugDocument::new("a", "SOURCE-A: ...", 1.0),
        ReplugDocument::new("b", "SOURCE-B: ...", 1.0),
    ];
    let (log_probs, weights, _) = engine
        .ensemble_next_token(&model, "which one?", &documents)
        .expect("ensemble");
    let probs = exp_all(&log_probs);

    assert_close(weights[0], 0.5, "equal scores => equal weights");
    assert_close(weights[1], 0.5, "equal scores => equal weights");

    assert_close_f32(probs[0], 0.30, "ensembled p(alpha)");
    assert_close_f32(probs[1], 0.30, "ensembled p(beta)");
    assert_close_f32(probs[2], 0.40, "ensembled p(gamma)");
    assert_is_distribution(&probs, "the ensemble");

    // THE CLAIM: the ensemble emits the token that neither of its parts would.
    assert_eq!(
        arg_max(&probs),
        Some(CORRECT),
        "the ensemble must promote the consensus runner-up to rank 1"
    );

    // Strictly, not by a tie: p(gamma) beats BOTH alternatives.
    assert!(probs[CORRECT] > probs[0], "0.40 > 0.30");
    assert!(probs[CORRECT] > probs[1], "0.40 > 0.30");
}

#[test]
fn the_ensembles_win_is_in_the_ranking_and_the_likelihood_stays_inside_its_components() {
    // The honest companion to the test above, and a guard against overclaiming.
    //
    // A mixture is a CONVEX COMBINATION, so on any *single* token it is pinned
    // between its components: min_i p_i(y) ≤ Σ_i λ_i p_i(y) ≤ max_i p_i(y). The
    // ensemble therefore CANNOT beat the best individual document on the
    // likelihood of one token — and any implementation claiming otherwise is
    // computing something that is not a mixture.
    //
    // REPLUG's win is a RANKING effect, not a magic likelihood boost: mixing
    // rearranges which token is on top (proved above), because the bound applies
    // to every token separately and the *argmax* is free to move. Asserting the
    // bound here is what keeps the previous test's claim from being mistaken for
    // something stronger than it is.
    let asymmetric = ReplugStaticLanguageModel::new(vec![
        "alpha".to_string(),
        "beta".to_string(),
        "gamma".to_string(),
    ])
    .expect("vocab")
    // Document A is genuinely informative about the target.
    .with_probability_rule(vec!["SOURCE-A"], &[0.10, 0.10, 0.80])
    .expect("rule a")
    // Document B is not.
    .with_probability_rule(vec!["SOURCE-B"], &[0.45, 0.45, 0.10])
    .expect("rule b");

    let engine =
        ReplugEngine::new(ReplugConfig::default().with_temperature(1.0)).expect("valid config");
    let target = asymmetric
        .encode(&["gamma"])
        .expect("gamma is in the vocabulary");
    // Equal retrieval scores => λ = [0.5, 0.5].
    let documents = vec![
        ReplugDocument::new("a", "SOURCE-A: ...", 1.0),
        ReplugDocument::new("b", "SOURCE-B: ...", 1.0),
    ];

    let per_document = engine
        .document_target_log_likelihoods(&asymmetric, "which one?", &documents, &target)
        .expect("per-document likelihoods");
    assert_close_f32(per_document[0], 0.80_f64.ln(), "log P(gamma | A)");
    assert_close_f32(per_document[1], 0.10_f64.ln(), "log P(gamma | B)");

    let ensembled = engine
        .ensemble_target_log_likelihood(&asymmetric, "which one?", &documents, &target)
        .expect("ensembled likelihood");

    // 0.5 * 0.80 + 0.5 * 0.10 = 0.400 + 0.050 = 0.45
    assert_close_f32(
        ensembled,
        0.45_f64.ln(),
        "ensembled log P(gamma) == ln(0.45)",
    );

    let best_single = per_document
        .iter()
        .copied()
        .fold(f64::NEG_INFINITY, f64::max);
    let worst_single = per_document.iter().copied().fold(f64::INFINITY, f64::min);

    // The convex-combination bound, asserted in both directions.
    assert!(
        ensembled > worst_single,
        "the ensemble must strictly beat the uninformative document: {ensembled} vs {worst_single}"
    );
    assert!(
        ensembled < best_single,
        "the ensemble CANNOT beat the best single document on one token: {ensembled} vs {best_single}"
    );

    // And with a retriever that knows which document is good (τ → 0), the
    // ensemble approaches — but never exceeds — that best single document. This
    // is precisely the sense in which LSR's job is to make the retriever's λ
    // agree with the LM: λ is the only lever that moves the ensemble toward the
    // best component.
    let sharp =
        ReplugEngine::new(ReplugConfig::default().with_temperature(1e-6)).expect("valid config");
    let informed = vec![
        ReplugDocument::new("a", "SOURCE-A: ...", 2.0), // now correctly ranked first
        ReplugDocument::new("b", "SOURCE-B: ...", 1.0),
    ];
    let sharpened = sharp
        .ensemble_target_log_likelihood(&asymmetric, "which one?", &informed, &target)
        .expect("ensembled");
    assert_close(
        sharpened,
        best_single,
        "tau -> 0 recovers the best component",
    );
    assert!(
        sharpened > ensembled,
        "a better retriever gives a better ensemble"
    );
}
