#![allow(
    clippy::float_cmp,
    clippy::similar_names,
    clippy::cast_precision_loss,
    clippy::too_many_lines,
    clippy::unreadable_literal,
    clippy::manual_midpoint,
    clippy::items_after_statements,
    clippy::doc_markdown
)]
//! Tests for decode-time distribution contrast — `CAD`, contrastive decoding and
//! `DoLa`.
//!
//! The organising principle is the one `crate::replug`'s suite uses and the one
//! the module was commissioned under: **every claim is checked against numbers
//! computed by hand in the test itself**, and — for the pieces that overlap
//! `REPLUG`'s arithmetic — against a kernel this module did not write. Where a
//! test asserts a probability of `171.5 / 205`, that arithmetic is written out,
//! so the assertion is an independent check and not a restatement of the code.
//!
//! Sections, matching the commissioning brief's testing bar:
//! (a) cross-module numerical agreement with `replug::math::log_linear_pool_log_probs`;
//! (b) the divergence that agreement hides, and the constraint that fixes it;
//! (c) Jensen–Shannon correctness; (d) `DoLa` layer selection; (e) the `alpha = 0`
//! ablation; (f) faithfulness gain; then generation, errors and determinism.

use super::engine::{ContextAwareDecoder, ContrastiveDecoder, DoLaDecoder};
use super::math::{
    adaptive_plausibility_mask, contrast_log_probs, contrast_scores,
    jensen_shannon_divergence_from_log_probs, masked_contrast_log_probs,
};
use super::model::{
    ContextAwareLanguageModel, ContextAwareStaticLanguageModel, LayeredLanguageModel,
};
use super::types::{
    CadConfig, ContextAwareConfig, ContextAwareError, ContrastiveConfig, DecodeMode,
    DecodingStrategy, DolaConfig, LayerSelector,
};

use crate::replug::math::{log_linear_pool_log_probs, log_softmax};

/// Pure-`f64` tolerance. Every quantity here is `O(1)` through a handful of
/// `exp`/`ln`/`+`, so a few ULP of `f64` (~`2e-16`) is the real error scale;
/// `1e-12` is generous but still meaningful.
const TOL: f64 = 1e-12;

/// Tolerance for a quantity that has crossed the `f32` logit boundary of a
/// [`ContextAwareStaticLanguageModel`]. The fixture stores `ln(p) as f32`, so a
/// hand-written probability round-trips through a 24-bit mantissa (relative
/// epsilon ≈ `6e-8`); `1e-6` sits just above the resulting `|Δp| ≲ 3e-7`. It is a
/// real bound — a wrong contrast formula is wrong by `O(10⁻²)` (the divergence
/// test differs by `0.16`, the faithfulness test by `0.49`), so nothing incorrect
/// hides under it.
const F32_TOL: f64 = 1e-6;

fn assert_close(actual: f64, expected: f64, what: &str) {
    assert!(
        (actual - expected).abs() < TOL,
        "{what}: expected {expected}, got {actual} (diff {})",
        (actual - expected).abs()
    );
}

fn assert_close_f32(actual: f64, expected: f64, what: &str) {
    assert!(
        (actual - expected).abs() < F32_TOL,
        "{what}: expected {expected}, got {actual} (diff {}, f32 budget {F32_TOL})",
        (actual - expected).abs()
    );
}

/// Bit-for-bit `f64` equality — the strongest possible claim, used for the
/// cross-module agreement and the ablation, where the two sides are provably the
/// *same* floating-point operations and any difference at all would signal that
/// they are not.
fn assert_bit_equal(actual: &[f64], expected: &[f64], what: &str) {
    assert_eq!(actual.len(), expected.len(), "{what}: length");
    for (index, (&a, &e)) in actual.iter().zip(expected).enumerate() {
        assert_eq!(
            a.to_bits(),
            e.to_bits(),
            "{what}: entry {index} differs — got {a}, expected {e}"
        );
    }
}

/// Assert a log-probability vector is a proper distribution: exponentiates to
/// non-negative entries summing to `1`.
fn assert_is_log_distribution(log_probs: &[f64], what: &str) {
    let mut total = 0.0_f64;
    for (index, &lp) in log_probs.iter().enumerate() {
        let p = lp.exp();
        assert!(p.is_finite(), "{what}: entry {index} not finite: {p}");
        assert!(p >= 0.0, "{what}: entry {index} negative: {p}");
        total += p;
    }
    assert!(
        (total - 1.0).abs() < 1e-10,
        "{what}: sums to {total}, not 1"
    );
}

// ══════════════════════════════════════════════════════════════════════════════
// (c) Jensen–Shannon divergence correctness
// ══════════════════════════════════════════════════════════════════════════════

mod jsd_correctness {
    use super::*;
    use std::f64::consts::LN_2;

    /// A distribution given as probabilities, turned into a normalized
    /// log-probability vector the same way a real one would be: via the logs of
    /// its entries. Exact zeros become `-inf`, which the divergence must accept.
    fn log_dist(probs: &[f64]) -> Vec<f64> {
        probs
            .iter()
            .map(|&p| if p == 0.0 { f64::NEG_INFINITY } else { p.ln() })
            .collect()
    }

    #[test]
    fn jsd_of_a_distribution_with_itself_is_zero() {
        let p = log_dist(&[0.7, 0.2, 0.1]);
        let jsd = jensen_shannon_divergence_from_log_probs(&p, &p).expect("jsd");
        assert_close(jsd, 0.0, "JSD(P, P)");
    }

    #[test]
    fn jsd_is_symmetric() {
        let p = log_dist(&[0.7, 0.2, 0.1]);
        let q = log_dist(&[0.1, 0.3, 0.6]);
        let forward = jensen_shannon_divergence_from_log_probs(&p, &q).expect("pq");
        let backward = jensen_shannon_divergence_from_log_probs(&q, &p).expect("qp");
        // Exactly equal: the two KL terms are added, and f64 addition commutes.
        assert_eq!(
            forward.to_bits(),
            backward.to_bits(),
            "JSD(P,Q) = {forward} but JSD(Q,P) = {backward}"
        );
    }

    #[test]
    fn jsd_matches_hand_computed_three_symbol_value() {
        // P = [0.7, 0.2, 0.1], Q = [0.1, 0.2, 0.7]. M = (P+Q)/2 = [0.4, 0.2, 0.4].
        //
        //   KL(P‖M) = 0.7·ln(0.7/0.4) + 0.2·ln(1) + 0.1·ln(0.1/0.4)
        //           = 0.7·ln(1.75)    + 0        + 0.1·ln(0.25)
        //           = 0.7·0.55961579  + 0.1·(−1.38629436)
        //           = 0.39173105 − 0.13862944 = 0.25310161
        //   KL(Q‖M) = 0.25310161 by the P↔Q symmetry of this pair about M.
        //   JSD     = ½(0.25310161 + 0.25310161) = 0.25310161  nats.
        let p = log_dist(&[0.7, 0.2, 0.1]);
        let q = log_dist(&[0.1, 0.2, 0.7]);
        let jsd = jensen_shannon_divergence_from_log_probs(&p, &q).expect("jsd");
        assert_close(jsd, 0.253101615442806, "JSD three-symbol");
    }

    #[test]
    fn jsd_of_disjoint_supports_is_exactly_ln_two() {
        // P and Q share no support, so on P's support M = P/2 and on Q's support
        // M = Q/2. Each KL is then Σ p·ln 2 = ln 2, and JSD = ln 2 — the maximum a
        // Jensen–Shannon divergence can attain, and the reason it is bounded where
        // a KL (here +inf) is not.
        let p = log_dist(&[0.5, 0.5, 0.0, 0.0]);
        let q = log_dist(&[0.0, 0.0, 0.5, 0.5]);
        let jsd = jensen_shannon_divergence_from_log_probs(&p, &q).expect("jsd");
        assert_close(jsd, LN_2, "JSD(disjoint)");
    }

    #[test]
    fn jsd_is_bounded_below_by_zero_and_above_by_ln_two() {
        let cases = [
            (vec![0.5, 0.5], vec![0.5, 0.5]),
            (vec![0.99, 0.01], vec![0.01, 0.99]),
            (vec![0.4, 0.35, 0.25], vec![0.2, 0.2, 0.6]),
            (vec![0.9, 0.05, 0.05], vec![0.05, 0.9, 0.05]),
        ];
        for (pv, qv) in cases {
            let jsd = jensen_shannon_divergence_from_log_probs(&log_dist(&pv), &log_dist(&qv))
                .expect("jsd");
            assert!(
                (0.0..=LN_2 + TOL).contains(&jsd),
                "JSD({pv:?}, {qv:?}) = {jsd} outside [0, ln2]"
            );
        }
    }

    #[test]
    fn jsd_rejects_raw_logits() {
        // The one silent mistake the check exists to catch: raw (unnormalized)
        // logits handed to a divergence. `[2, 1, 0]` has logsumexp ≈ 2.41, so it is
        // not a distribution, and the divergence must refuse it rather than return
        // the plausible-looking divergence of nothing.
        let logits = vec![2.0, 1.0, 0.0];
        let dist = log_softmax(&logits);
        let error =
            jensen_shannon_divergence_from_log_probs(&logits, &dist).expect_err("must reject");
        assert!(
            matches!(error, ContextAwareError::UnnormalizedDistribution { .. }),
            "expected UnnormalizedDistribution, got {error:?}"
        );
    }

    #[test]
    fn jsd_rejects_length_mismatch_and_empty() {
        let p = log_dist(&[0.5, 0.5]);
        let q = log_dist(&[0.3, 0.3, 0.4]);
        assert!(matches!(
            jensen_shannon_divergence_from_log_probs(&p, &q),
            Err(ContextAwareError::VocabSizeMismatch { .. })
        ));
        assert!(matches!(
            jensen_shannon_divergence_from_log_probs(&[], &[]),
            Err(ContextAwareError::EmptyLogits)
        ));
    }
}

// ══════════════════════════════════════════════════════════════════════════════
// (a) Cross-module numerical agreement — the centrepiece.
//
// CAD's adjustment IS `replug::math::log_linear_pool_log_probs` evaluated at the
// weights (1+a, −a). Assert it bit-for-bit, on inputs where no truncation fires.
// That is ground truth this module did not author.
// ══════════════════════════════════════════════════════════════════════════════

mod cad_cross_module_agreement {
    use super::*;

    #[test]
    fn contrast_is_bit_equal_to_the_replug_log_linear_pool() {
        let positive = vec![1.3, 0.4, -0.2, 0.9, -1.1];
        let negative = vec![0.2, 0.7, 0.1, -0.5, 0.3];

        for alpha in [0.0, 0.5, 1.0, 2.5] {
            let weights = (1.0 + alpha, -alpha);

            // This module's contrast, unconstrained.
            let ours =
                contrast_log_probs(&positive, &negative, weights.0, weights.1).expect("ours");

            // REPLUG's log-linear pool at exactly those weights. Its own docs
            // reason about softmax weights only; (1+a, −a) is outside that regime,
            // which is the whole point — the arithmetic coincides, the semantics
            // do not.
            let theirs = log_linear_pool_log_probs(
                &[weights.0, weights.1],
                &[positive.clone(), negative.clone()],
            )
            .expect("theirs");

            assert_bit_equal(
                &ours,
                &theirs,
                &format!("CAD vs REPLUG pool, alpha={alpha}"),
            );
        }
    }

    #[test]
    fn the_prelude_alias_is_the_same_function() {
        // `replug_log_linear_pool` is the crate-prelude name for the same kernel;
        // assert the alias resolves to identical output, so the doc claim that the
        // formula is "already computable today" is checkable through the public
        // surface too.
        let positive = vec![0.5, -0.5, 0.2];
        let negative = vec![-0.1, 0.4, 0.0];
        let ours = contrast_log_probs(&positive, &negative, 2.0, -1.0).expect("ours");
        let via_alias = crate::replug::replug_log_linear_pool(&[2.0, -1.0], &[positive, negative])
            .expect("alias");
        assert_bit_equal(&ours, &via_alias, "prelude alias");
    }

    #[test]
    fn masked_contrast_with_vacuous_alpha_equals_unconstrained() {
        // alpha_plaus = 0 makes the constraint vacuous (ln 0 = −inf threshold), so
        // masked contrast is bit-for-bit the unconstrained contrast — the bridge
        // that lets the cross-module agreement above stand for the masked path.
        let positive = vec![1.0, 0.3, -0.7, 0.2];
        let negative = vec![0.1, 0.5, 0.0, -0.2];
        let unconstrained = contrast_log_probs(&positive, &negative, 2.0, -1.0).expect("unc");
        let (masked, plausible) =
            masked_contrast_log_probs(&positive, &negative, 2.0, -1.0, 0.0).expect("masked");
        assert!(
            plausible.iter().all(|&keep| keep),
            "alpha=0 keeps everything"
        );
        assert_bit_equal(&masked, &unconstrained, "masked@alpha=0 vs unconstrained");
    }
}

// ══════════════════════════════════════════════════════════════════════════════
// (b) Why this module exists: the divergence the kernel alone hands you, and the
// adaptive plausibility constraint that is this module's core novel content.
// ══════════════════════════════════════════════════════════════════════════════

mod plausibility_divergence {
    use super::*;

    // The example carried in the module documentation, at f64 exactness.
    //
    //   z⁺ = [ 0, −1, −2, −12 ]   p⁺ = [0.6652382, 0.2447275, 0.0900302, 4.09e−6]
    //   z⁻ = [ −3, 0, −1, −40 ]
    //   CAD score at a=1:  2·z⁺ − z⁻ = [3, −2, −3, 16].
    const Z_POS: [f64; 4] = [0.0, -1.0, -2.0, -12.0];
    const Z_NEG: [f64; 4] = [-3.0, 0.0, -1.0, -40.0];

    #[test]
    fn unconstrained_contrast_amplifies_the_token_the_negative_model_forbade() {
        // The 4th token is nearly impossible under p⁺ (4 parts per million) and
        // *far* more impossible under p⁻ — so the log-ratio blows it up. The
        // unconstrained kernel hands it p ≈ 0.99999. The expected value is written
        // to full f64 precision: this is pure f64 math (constants in, no model), so
        // it is checked against the tight TOL, not the f32 budget.
        let scores = contrast_scores(&Z_POS, &Z_NEG, 2.0, -1.0).expect("scores");
        assert_eq!(scores, vec![3.0, -2.0, -3.0, 16.0], "score = 2z⁺ − z⁻");

        let log_probs = contrast_log_probs(&Z_POS, &Z_NEG, 2.0, -1.0).expect("lp");
        let probs: Vec<f64> = log_probs.iter().map(|lp| lp.exp()).collect();
        assert_eq!(
            crate::replug::math::arg_max(&log_probs),
            Some(3),
            "unconstrained argmax is the forbidden token"
        );
        assert_close(probs[3], 0.9999977188430206, "unconstrained p(token 3)");
    }

    #[test]
    fn the_constraint_forbids_exactly_that_token_and_restores_the_argmax() {
        // Under p⁺ the max mass is 0.6652382; alpha_plaus = 0.1 sets the threshold
        // at 0.0665. Token 3, at 4e−6, falls below it and is masked to −inf. The
        // argmax returns to token 0 at p = 0.99086747 — hand-computed as
        // softmax([3, −2, −3]) = [e³, e⁻², e⁻³]/Σ = [20.0855, 0.13534, 0.049787]/
        // 20.2706.
        let mask = adaptive_plausibility_mask(&Z_POS, 0.1).expect("mask");
        assert_eq!(mask, vec![true, true, true, false], "token 3 masked");

        let (log_probs, plausible) =
            masked_contrast_log_probs(&Z_POS, &Z_NEG, 2.0, -1.0, 0.1).expect("masked");
        assert_eq!(plausible, mask, "mask reported on the outcome");
        assert_is_log_distribution(&log_probs, "constrained CAD");

        let probs: Vec<f64> = log_probs.iter().map(|lp| lp.exp()).collect();
        assert_eq!(
            crate::replug::math::arg_max(&log_probs),
            Some(0),
            "constrained argmax is the plausible token"
        );
        assert_eq!(probs[3], 0.0, "the forbidden token has exactly zero mass");
        assert_close(probs[0], 0.9908674725821728, "constrained p(token 0)");

        // The concrete divergence: the *same* input, one with the constraint and
        // one without, disagree on the arg-max — 3 vs 0. That is the whole reason
        // the module is not a wrapper around the REPLUG kernel.
        let unconstrained = contrast_log_probs(&Z_POS, &Z_NEG, 2.0, -1.0).expect("unc");
        assert_ne!(
            crate::replug::math::arg_max(&unconstrained),
            crate::replug::math::arg_max(&log_probs),
            "constrained and unconstrained must disagree on argmax here"
        );
    }

    #[test]
    fn the_constraint_can_never_empty_the_vocabulary() {
        // For every alpha in [0, 1] the arg-max of p⁺ survives, so at least one
        // token is always decodable. Checked at the boundary alpha = 1, where only
        // the arg-max (and its ties) remain.
        for alpha in [0.0, 0.1, 0.5, 0.9, 1.0] {
            let mask = adaptive_plausibility_mask(&Z_POS, alpha).expect("mask");
            assert!(
                mask.iter().any(|&keep| keep),
                "alpha={alpha} emptied the vocabulary"
            );
        }
        let full = adaptive_plausibility_mask(&Z_POS, 1.0).expect("mask");
        assert_eq!(
            full,
            vec![true, false, false, false],
            "alpha=1 keeps only argmax"
        );
    }

    #[test]
    fn alpha_above_one_is_rejected_not_clamped() {
        let error = adaptive_plausibility_mask(&Z_POS, 1.5).expect_err("must reject");
        assert!(matches!(
            error,
            ContextAwareError::InvalidPlausibilityAlpha { .. }
        ));
    }

    #[test]
    fn the_pathology_runs_through_the_real_cad_decoder() {
        // The same divergence, exhibited end-to-end through ContextAwareDecoder on
        // a fixture. The with-context prompt gives the "trap" token a small but
        // non-zero mass; the context-free prompt gives it far less.
        //
        // Vocab: 0="fact" (context-supported), 1="filler", 2="rare" (the trap).
        // p⁺ = [0.60, 0.35, 0.05]   — "rare" is implausible to the informed model
        // p⁻ = [0.30, 0.699, 0.001] — and near-impossible to the prior.
        // CAD mass ∝ p⁺²/p⁻ = [0.36/0.30, 0.1225/0.699, 0.0025/0.001] =
        // [1.20, 0.175, 2.50], so the trap (2.50) wins the unconstrained contrast.
        let model = ContextAwareStaticLanguageModel::new(vec![
            "fact".to_string(),
            "filler".to_string(),
            "rare".to_string(),
        ])
        .expect("vocab")
        .with_probability_rule(vec!["CTX", "Q"], &[0.60, 0.35, 0.05])
        .expect("with-context")
        .with_probability_rule(vec!["Q"], &[0.30, 0.699, 0.001])
        .expect("context-free");

        // Unconstrained (alpha_plaus = 0): the trap token wins the contrast.
        let unconstrained = ContextAwareDecoder::new(
            CadConfig::default()
                .with_alpha(1.0)
                .with_plausibility_alpha(0.0),
        )
        .expect("cfg")
        .next_token(&model, "CTX: fact.", "Q?")
        .expect("step");
        assert_eq!(
            unconstrained.argmax(),
            Some(2),
            "unconstrained picks the trap"
        );
        assert!(!unconstrained.truncated(), "alpha=0 truncates nothing");

        // Constrained (default alpha_plaus = 0.1): max p⁺ = 0.60, threshold 0.06,
        // so "rare" (p⁺ = 0.05) is masked and the argmax returns to "fact".
        let constrained = ContextAwareDecoder::new(CadConfig::default().with_alpha(1.0))
            .expect("cfg")
            .next_token(&model, "CTX: fact.", "Q?")
            .expect("step");
        assert_eq!(constrained.argmax(), Some(0), "constrained picks the fact");
        assert!(constrained.truncated(), "the constraint bound here");
        assert!(!constrained.plausible[2], "the trap token is masked");
        assert_eq!(constrained.probs()[2], 0.0, "trap has zero mass");
    }
}

// ══════════════════════════════════════════════════════════════════════════════
// (e) Ablation: alpha = 0 reproduces the plain with-context distribution
// bit-for-bit.
// ══════════════════════════════════════════════════════════════════════════════

mod cad_ablation {
    use super::*;

    #[test]
    fn cad_at_alpha_zero_is_the_identity_on_the_positive_distribution() {
        // At alpha = 0 the weights are (1, −0): score = 1·z⁺ + (−0)·z⁻ = z⁺
        // exactly, so log_softmax of it is the with-context distribution
        // bit-for-bit. Requires the constraint vacuous (alpha_plaus = 0), since a
        // firing constraint would legitimately differ.
        let model = ContextAwareStaticLanguageModel::new(vec![
            "a".to_string(),
            "b".to_string(),
            "c".to_string(),
            "d".to_string(),
        ])
        .expect("vocab")
        .with_probability_rule(vec!["CTX", "Q"], &[0.4, 0.3, 0.2, 0.1])
        .expect("with-context")
        .with_probability_rule(vec!["Q"], &[0.1, 0.2, 0.3, 0.4])
        .expect("context-free");

        let outcome = ContextAwareDecoder::new(
            CadConfig::default()
                .with_alpha(0.0)
                .with_plausibility_alpha(0.0),
        )
        .expect("cfg")
        .next_token(&model, "CTX: stuff.", "Q?")
        .expect("step");

        // The contrasted distribution equals the positive distribution, exactly.
        assert_bit_equal(
            &outcome.log_probs,
            &outcome.positive_log_probs,
            "CAD@alpha=0 vs positive distribution",
        );

        // And that positive distribution is genuinely the model's with-context
        // output — checked against the raw logits the fixture stored.
        let logits = model.next_token_logits("CTX: stuff.Q?").expect("logits");
        let promoted: Vec<f64> = logits.iter().map(|&z| f64::from(z)).collect();
        assert_bit_equal(
            &outcome.log_probs,
            &log_softmax(&promoted),
            "CAD@alpha=0 vs raw with-context log_softmax",
        );
    }

    #[test]
    fn contrastive_at_beta_zero_is_the_expert_distribution() {
        // The analogous ablation for contrastive decoding: beta = 0 subtracts
        // nothing, so the output is the expert's own distribution.
        let expert = ContextAwareStaticLanguageModel::new(vec![
            "x".to_string(),
            "y".to_string(),
            "z".to_string(),
        ])
        .expect("vocab")
        .with_probability_rule(vec!["Q"], &[0.5, 0.3, 0.2])
        .expect("expert");
        let amateur = ContextAwareStaticLanguageModel::new(vec![
            "x".to_string(),
            "y".to_string(),
            "z".to_string(),
        ])
        .expect("vocab")
        .with_probability_rule(vec!["Q"], &[0.2, 0.5, 0.3])
        .expect("amateur");

        let outcome = ContrastiveDecoder::new(
            ContrastiveConfig::default()
                .with_amateur_weight(0.0)
                .with_plausibility_alpha(0.0),
        )
        .expect("cfg")
        .next_token(&expert, &amateur, "Q?")
        .expect("step");

        assert_bit_equal(
            &outcome.log_probs,
            &outcome.positive_log_probs,
            "CD@beta=0 vs expert distribution",
        );
    }
}

// ══════════════════════════════════════════════════════════════════════════════
// (f) Faithfulness gain: where the context contradicts the no-context prior, CAD
// raises the context-supported token above the prior-favoured one; alpha = 0 does
// not.
// ══════════════════════════════════════════════════════════════════════════════

mod cad_faithfulness {
    use super::*;

    /// The quick-start fixture: 0 = "Sylvania" (what the context says), 1 =
    /// "Paris" (what the model remembers). WITH the context the model is nudged but
    /// still prefers Paris; WITHOUT it, it is sure of Paris.
    fn memory_vs_context_model() -> ContextAwareStaticLanguageModel {
        ContextAwareStaticLanguageModel::new(vec![
            "Sylvania".to_string(),
            "Paris".to_string(),
            "London".to_string(),
            "Zzyzx".to_string(),
        ])
        .expect("vocab")
        .with_probability_rule(vec!["CONTEXT", "QUESTION"], &[0.35, 0.40, 0.20, 0.05])
        .expect("with-context")
        .with_probability_rule(vec!["QUESTION"], &[0.05, 0.70, 0.20, 0.05])
        .expect("context-free")
    }

    const CONTEXT: &str = "CONTEXT: Freedonia's capital is Sylvania.";
    const QUERY: &str = "QUESTION: What is the capital?";

    #[test]
    fn alpha_zero_leaves_the_prior_favoured_token_on_top() {
        // At alpha = 0 CAD is the with-context distribution: p(Sylvania) = 0.35 <
        // p(Paris) = 0.40. The model has been given the context and still gets it
        // wrong. This is the failure CAD exists to fix.
        let model = memory_vs_context_model();
        let outcome = ContextAwareDecoder::new(
            CadConfig::default()
                .with_alpha(0.0)
                .with_plausibility_alpha(0.0),
        )
        .expect("cfg")
        .next_token(&model, CONTEXT, QUERY)
        .expect("step");

        let probs = outcome.probs();
        assert_close_f32(probs[0], 0.35, "alpha=0 p(Sylvania)");
        assert_close_f32(probs[1], 0.40, "alpha=0 p(Paris)");
        assert!(
            probs[0] < probs[1],
            "alpha=0: context-supported token is NOT on top"
        );
        assert_eq!(outcome.argmax(), Some(1), "alpha=0 argmax is the prior");
    }

    #[test]
    fn alpha_one_raises_the_context_supported_token_above_the_prior() {
        // At alpha = 1: p_cad(y) ∝ p⁺(y)²/p⁻(y). Unnormalized masses
        //   Sylvania: 0.35²/0.05 = 2.45
        //   Paris:    0.40²/0.70 = 0.22857
        //   London:   0.20²/0.20 = 0.20
        //   Zzyzx:    0.05²/0.05 = 0.05
        // Scaling by 70: [171.5, 16, 14, 3.5], sum 205. So p(Sylvania)=171.5/205 =
        // 0.83659 and p(Paris)=16/205 = 0.07805. The context now wins, by a 0.49
        // margin no f32 rounding could manufacture.
        let model = memory_vs_context_model();
        let outcome = ContextAwareDecoder::new(CadConfig::default().with_alpha(1.0))
            .expect("cfg")
            .next_token(&model, CONTEXT, QUERY)
            .expect("step");

        let probs = outcome.probs();
        assert_close_f32(probs[0], 171.5 / 205.0, "alpha=1 p(Sylvania)");
        assert_close_f32(probs[1], 16.0 / 205.0, "alpha=1 p(Paris)");
        assert!(
            probs[0] > probs[1],
            "alpha=1: context-supported {} must beat prior {}",
            probs[0],
            probs[1]
        );
        assert_eq!(outcome.argmax(), Some(0), "alpha=1 argmax is the context");
        // What the model would have said, recorded alongside, is still Paris.
        assert_eq!(
            outcome.positive_argmax(),
            Some(1),
            "the uncontrasted argmax"
        );
    }

    #[test]
    fn faithfulness_grows_monotonically_with_alpha() {
        // A stronger claim than two points: p(Sylvania) − p(Paris) increases with
        // alpha across the whole sweep. The contrast strength does what it says.
        let model = memory_vs_context_model();
        let mut previous_margin = f64::NEG_INFINITY;
        for alpha in [0.0, 0.25, 0.5, 1.0, 2.0] {
            let outcome = ContextAwareDecoder::new(CadConfig::default().with_alpha(alpha))
                .expect("cfg")
                .next_token(&model, CONTEXT, QUERY)
                .expect("step");
            let probs = outcome.probs();
            let margin = probs[0] - probs[1];
            assert!(
                margin > previous_margin - F32_TOL,
                "alpha={alpha}: margin {margin} did not grow from {previous_margin}"
            );
            previous_margin = margin;
        }
        assert!(previous_margin > 0.0, "the strongest contrast is faithful");
    }
}

// ══════════════════════════════════════════════════════════════════════════════
// (d) DoLa layer selection: build a fixture where a KNOWN layer is the maximally-
// divergent premature layer; assert the selector picks exactly it.
// ══════════════════════════════════════════════════════════════════════════════

mod dola_selection {
    use super::*;

    /// A 4-layer, 3-token model. The mature (final) layer peaks token 0; the layer
    /// stack was built so that layer 1 disagrees with it the most.
    ///
    ///   layer 0 = [0.60, 0.25, 0.15]   JSD vs mature ≈ 0.005834   (nearly settled)
    ///   layer 1 = [0.10, 0.10, 0.80]   JSD vs mature ≈ 0.289988   (the maximum)
    ///   layer 2 = [0.40, 0.30, 0.30]   JSD vs mature ≈ 0.051912
    ///   layer 3 = [0.70, 0.20, 0.10]   (mature)
    fn layered_model() -> ContextAwareStaticLanguageModel {
        ContextAwareStaticLanguageModel::with_layer_count(
            vec!["a".to_string(), "b".to_string(), "c".to_string()],
            4,
        )
        .expect("vocab")
        .with_layered_probability_rule(
            vec!["Q"],
            &[
                vec![0.60, 0.25, 0.15],
                vec![0.10, 0.10, 0.80],
                vec![0.40, 0.30, 0.30],
                vec![0.70, 0.20, 0.10],
            ],
        )
        .expect("layers")
    }

    #[test]
    fn dynamic_selector_picks_the_maximally_divergent_layer() {
        let model = layered_model();
        let decoder = DoLaDecoder::new(DolaConfig::default()).expect("cfg");
        let premature = decoder
            .select_premature_layer(&model, "Q?")
            .expect("select");

        assert_eq!(premature.layer, 1, "layer 1 is the JSD maximizer");
        assert_close_f32(premature.divergence_nats, 0.289988, "max JSD");

        // The full audit trail: every non-mature layer, ascending, with the min at
        // layer 0.
        let layers: Vec<usize> = premature.considered.iter().map(|&(l, _)| l).collect();
        assert_eq!(
            layers,
            vec![0, 1, 2],
            "candidates are all but mature layer 3"
        );
        assert_close_f32(
            premature.min_divergence_nats(),
            0.005834,
            "min JSD (layer 0)",
        );
        assert_close_f32(premature.max_divergence_nats(), 0.289988, "max JSD");
    }

    #[test]
    fn the_contrast_uses_the_selected_layer_and_masks_correctly() {
        // A full DoLa step: mature=[.7,.2,.1], premature=layer1=[.1,.1,.8].
        // score = log q_mature − log q_premature; the plausibility set is read off
        // q_mature (max 0.7, threshold 0.07), so all three tokens survive here.
        let model = layered_model();
        let outcome = DoLaDecoder::new(DolaConfig::default())
            .expect("cfg")
            .next_token(&model, "Q?")
            .expect("step");

        assert_eq!(outcome.mode, DecodeMode::Dola);
        assert_eq!(
            outcome.premature_layer.as_ref().map(|p| p.layer),
            Some(1),
            "step recorded layer 1"
        );
        assert_is_log_distribution(&outcome.log_probs, "DoLa distribution");

        // Hand-check the contrast: log(0.7/0.1)=1.9459, log(0.2/0.1)=0.6931,
        // log(0.1/0.8)=−2.0794 → softmax([1.9459, 0.6931, −2.0794]).
        // exp: [7.0, 2.0, 0.125], sum 9.125 → [0.76712, 0.21918, 0.01370].
        let probs = outcome.probs();
        assert_close_f32(probs[0], 7.0 / 9.125, "DoLa p(token 0)");
        assert_close_f32(probs[1], 2.0 / 9.125, "DoLa p(token 1)");
        assert_close_f32(probs[2], 0.125 / 9.125, "DoLa p(token 2)");
        // Token 0 is amplified (0.70 → 0.767): the mature layer's favourite, which
        // the early layer least anticipated, is boosted. Exactly DoLa's intent.
        assert!(probs[0] > 0.70, "the late-emerging token is amplified");
    }

    #[test]
    fn a_fixed_selector_overrides_the_jsd_maximizer() {
        // DoLa-static: contrast against layer 2 even though layer 1 diverges more.
        // Proves the selection is honoured, not merely defaulted.
        let model = layered_model();
        let outcome = DoLaDecoder::new(DolaConfig::new(LayerSelector::Fixed(2)))
            .expect("cfg")
            .next_token(&model, "Q?")
            .expect("step");
        assert_eq!(
            outcome.premature_layer.as_ref().map(|p| p.layer),
            Some(2),
            "fixed selector used layer 2"
        );
        // Only layer 2 was considered.
        assert_eq!(
            outcome.premature_layer.as_ref().unwrap().considered.len(),
            1
        );
    }

    #[test]
    fn a_bucket_selector_restricts_the_candidate_set() {
        let model = layered_model();
        // Even layers of {0,1,2,3} → {0, 2}; layer 1 (the true maximizer) is
        // excluded, so the selector must fall back to the best *within the bucket*,
        // which is layer 2.
        let selector = LayerSelector::bucket(0, 4, 2).expect("bucket");
        let outcome = DoLaDecoder::new(DolaConfig::new(selector))
            .expect("cfg")
            .next_token(&model, "Q?")
            .expect("step");
        let premature = outcome.premature_layer.expect("premature");
        assert_eq!(premature.layer, 2, "best even layer is 2");
        let layers: Vec<usize> = premature.considered.iter().map(|&(l, _)| l).collect();
        assert_eq!(layers, vec![0, 2], "bucket restricted candidates to evens");
    }

    #[test]
    fn contrasting_a_layer_against_itself_is_refused() {
        // The silent catastrophe DoLa's design avoids: were the mature layer a
        // candidate, JSD would be 0 and the contrast would be the uniform
        // distribution. Must be a loud error.
        let model = layered_model();
        let selector = LayerSelector::MaxJensenShannon {
            candidates: vec![3], // layer 3 is mature
        };
        let error = DoLaDecoder::new(DolaConfig::new(selector))
            .expect("cfg")
            .next_token(&model, "Q?")
            .expect_err("must refuse");
        assert!(matches!(
            error,
            ContextAwareError::MatureLayerInCandidates { layer: 3 }
        ));
    }

    #[test]
    fn a_single_layer_model_cannot_do_dola() {
        // No premature layer exists, so DoLa is not applicable — reported, not
        // approximated.
        let model = ContextAwareStaticLanguageModel::new(vec!["a".to_string(), "b".to_string()])
            .expect("vocab");
        let error = DoLaDecoder::new(DolaConfig::default())
            .expect("cfg")
            .next_token(&model, "Q?")
            .expect_err("must fail");
        assert!(matches!(error, ContextAwareError::EmptyCandidateSet));
    }

    #[test]
    fn an_out_of_range_candidate_is_rejected() {
        let model = layered_model();
        let selector = LayerSelector::MaxJensenShannon {
            candidates: vec![9],
        };
        let error = DoLaDecoder::new(DolaConfig::new(selector))
            .expect("cfg")
            .select_premature_layer(&model, "Q?")
            .expect_err("must fail");
        assert!(matches!(
            error,
            ContextAwareError::LayerOutOfRange {
                layer: 9,
                num_layers: 4
            }
        ));
    }

    #[test]
    fn the_mature_layer_contract_holds_by_construction() {
        // next_token_logits(ctx) must equal layer_logits(ctx, num_layers − 1), or
        // DoLa contrasts against a distribution the model never emitted. The
        // fixture satisfies it via a shared code path; assert it bit-for-bit.
        let model = layered_model();
        let mature = model.layer_logits("Q?", 3).expect("layer 3");
        let emitted = model.next_token_logits("Q?").expect("emitted");
        assert_eq!(mature, emitted, "mature layer == emitted logits");
    }
}

// ══════════════════════════════════════════════════════════════════════════════
// Contrastive decoding: the amateur-veto pathology, and its cure.
// ══════════════════════════════════════════════════════════════════════════════

mod contrastive_decoding {
    use super::*;

    #[test]
    fn the_amateur_can_veto_a_token_and_the_constraint_stops_it() {
        // expert p_exp = [0.55, 0.42, 0.03]; amateur p_ama = [0.50, 0.48, 0.02].
        // The log-ratio log(p_exp/p_ama) = [0.09531, −0.13353, 0.40546] is
        // maximized by token 2 — which the expert thinks is nearly impossible
        // (0.03), and which wins only because the amateur thinks it even more so.
        // Unconstrained, contrastive decoding emits the token neither is confident
        // about.
        let expert = ContextAwareStaticLanguageModel::new(vec![
            "keep".to_string(),
            "also".to_string(),
            "junk".to_string(),
        ])
        .expect("vocab")
        .with_probability_rule(vec!["Q"], &[0.55, 0.42, 0.03])
        .expect("expert");
        let amateur = ContextAwareStaticLanguageModel::new(vec![
            "keep".to_string(),
            "also".to_string(),
            "junk".to_string(),
        ])
        .expect("vocab")
        .with_probability_rule(vec!["Q"], &[0.50, 0.48, 0.02])
        .expect("amateur");

        let unconstrained =
            ContrastiveDecoder::new(ContrastiveConfig::default().with_plausibility_alpha(0.0))
                .expect("cfg")
                .next_token(&expert, &amateur, "Q?")
                .expect("step");
        assert_eq!(unconstrained.argmax(), Some(2), "unconstrained emits junk");

        // Default alpha_plaus = 0.1: max p_exp = 0.55, threshold 0.055, so junk
        // (0.03) is masked. The argmax returns to token 0. Hand value:
        // softmax([0.09531, −0.13353]) = [0.55696, 0.44304].
        let constrained = ContrastiveDecoder::new(ContrastiveConfig::default())
            .expect("cfg")
            .next_token(&expert, &amateur, "Q?")
            .expect("step");
        assert_eq!(
            constrained.argmax(),
            Some(0),
            "constrained keeps a real token"
        );
        assert!(constrained.truncated(), "the constraint bound");
        assert_close_f32(constrained.probs()[0], 0.556962, "constrained p(keep)");
        assert_close_f32(constrained.probs()[1], 0.443038, "constrained p(also)");
        assert_eq!(constrained.probs()[2], 0.0, "junk masked to zero");
    }

    #[test]
    fn a_tokenizer_mismatch_between_expert_and_amateur_is_refused() {
        let expert = ContextAwareStaticLanguageModel::new(vec!["a".to_string(), "b".to_string()])
            .expect("vocab");
        let amateur = ContextAwareStaticLanguageModel::new(vec![
            "a".to_string(),
            "b".to_string(),
            "c".to_string(),
        ])
        .expect("vocab");
        let error = ContrastiveDecoder::new(ContrastiveConfig::default())
            .expect("cfg")
            .next_token(&expert, &amateur, "Q?")
            .expect_err("must refuse");
        assert!(matches!(
            error,
            ContextAwareError::VocabSizeMismatch {
                expected: 2,
                actual: 3,
                ..
            }
        ));
    }

    #[test]
    fn amateur_temperature_reshapes_but_tau_one_is_inert() {
        // At tau = 1 the amateur temperature is a no-op (a renormalization the final
        // log_softmax absorbs), so the result equals the plain contrast.
        let expert = ContextAwareStaticLanguageModel::new(vec![
            "a".to_string(),
            "b".to_string(),
            "c".to_string(),
        ])
        .expect("vocab")
        .with_probability_rule(vec!["Q"], &[0.5, 0.3, 0.2])
        .expect("expert");
        let amateur = ContextAwareStaticLanguageModel::new(vec![
            "a".to_string(),
            "b".to_string(),
            "c".to_string(),
        ])
        .expect("vocab")
        .with_probability_rule(vec!["Q"], &[0.2, 0.5, 0.3])
        .expect("amateur");

        let plain =
            ContrastiveDecoder::new(ContrastiveConfig::default().with_plausibility_alpha(0.0))
                .expect("cfg")
                .next_token(&expert, &amateur, "Q?")
                .expect("step");
        let tau_one = ContrastiveDecoder::new(
            ContrastiveConfig::default()
                .with_amateur_temperature(1.0)
                .with_plausibility_alpha(0.0),
        )
        .expect("cfg")
        .next_token(&expert, &amateur, "Q?")
        .expect("step");
        assert_bit_equal(&plain.log_probs, &tau_one.log_probs, "tau=1 is inert");

        // A larger tau flattens the amateur, so it vetoes less; the contrasted
        // distribution genuinely changes.
        let tau_hot = ContrastiveDecoder::new(
            ContrastiveConfig::default()
                .with_amateur_temperature(5.0)
                .with_plausibility_alpha(0.0),
        )
        .expect("cfg")
        .next_token(&expert, &amateur, "Q?")
        .expect("step");
        let changed = plain
            .log_probs
            .iter()
            .zip(&tau_hot.log_probs)
            .any(|(&a, &b)| (a - b).abs() > 1e-6);
        assert!(changed, "tau=5 must reshape the amateur");
    }
}

// ══════════════════════════════════════════════════════════════════════════════
// Generation, statistics, EOS, determinism, and configuration validation.
// ══════════════════════════════════════════════════════════════════════════════

mod generation_and_edges {
    use super::*;

    /// A stateful CAD fixture: the with-context distribution depends on what has
    /// been generated, so generation is a genuine multi-step process rather than
    /// the same token emitted `n` times. `<end>` is token 3.
    ///
    /// The rule keys are chosen with the same care `crate::replug`'s stateful
    /// fixture uses, and for the same reason: the token surfaces (`X`/`Y`/`Z`) do
    /// not occur in the prompt markers (`SRC`, `GO`), so a step-1 rule keyed on a
    /// generated token cannot mis-fire at step 0; and the step-1 rules carry an
    /// *extra* required substring, so they are strictly more specific than the
    /// step-0 rules and reliably win once the token they key on has been emitted.
    /// (An earlier version keyed a rule on `"QA"` while passing the query `"Q?"`;
    /// the generated context was `"Q?X"`, which never matched, and generation
    /// looped forever — a fixture bug this very test caught. A later one used the
    /// marker `"CTX"`, which itself *contains* the token surface `"X"` — same trap,
    /// one layer down.)
    fn stateful_cad_model() -> ContextAwareStaticLanguageModel {
        ContextAwareStaticLanguageModel::new(vec![
            "X".to_string(),
            "Y".to_string(),
            "Z".to_string(),
            "<end>".to_string(),
        ])
        .expect("vocab")
        // Step 0, with context: prefer X.
        .with_probability_rule(vec!["SRC"], &[0.70, 0.15, 0.10, 0.05])
        .expect("s0 ctx")
        // Step 0, context-free.
        .with_probability_rule(vec!["GO"], &[0.25, 0.30, 0.25, 0.20])
        .expect("s0 free")
        // Step 1 (once "X" is generated), with context: prefer <end>.
        .with_probability_rule(vec!["SRC", "X"], &[0.05, 0.05, 0.10, 0.80])
        .expect("s1 ctx")
        // Step 1, context-free.
        .with_probability_rule(vec!["GO", "X"], &[0.10, 0.10, 0.20, 0.60])
        .expect("s1 free")
        .with_eos_token(3)
        .expect("eos")
    }

    #[test]
    fn generation_halts_at_eos_and_reports_consistent_stats() {
        let model = stateful_cad_model();
        let output =
            ContextAwareDecoder::new(CadConfig::default().with_alpha(1.0).with_max_tokens(10))
                .expect("cfg")
                .generate(&model, "SRC hint", "GO")
                .expect("generate");

        // Step 0 emits "A" (context boosts it), step 1 emits <eos>.
        assert_eq!(output.token_ids, vec![0, 3], "A then eos");
        assert_eq!(output.text, "X", "eos is not appended to the text");
        assert_eq!(output.stats.mode, DecodeMode::ContextAware);
        assert_eq!(output.stats.generated_tokens, 2);
        assert_eq!(output.stats.lm_calls, 4, "two prompts × two steps");
        assert_eq!(output.steps.len(), 2);

        // sequence_log_prob is the sum of the recorded per-step token log-probs.
        let summed: f64 = output.steps.iter().map(|s| s.token_log_prob).sum();
        assert_close(output.stats.sequence_log_prob, summed, "sequence log-prob");
        // The contrast raised the emitted tokens above their uncontrasted mass:
        // positive_sequence_log_prob is lower (more negative) than the contrasted.
        assert!(
            output.stats.sequence_log_prob > output.stats.positive_sequence_log_prob,
            "contrast improved the chosen tokens: {} vs {}",
            output.stats.sequence_log_prob,
            output.stats.positive_sequence_log_prob
        );
    }

    #[test]
    fn recorded_distributions_can_be_switched_off() {
        let model = stateful_cad_model();
        let output = ContextAwareDecoder::new(
            CadConfig::default()
                .with_alpha(1.0)
                .with_strategy(DecodingStrategy::Greedy)
                .with_max_tokens(1),
        )
        .expect("cfg")
        .generate(&model, "SRC hint", "GO")
        .expect("generate");
        // record_distributions defaults on; the step carries the full vectors.
        assert!(output.steps[0].log_distribution.is_some());
        assert!(output.steps[0].positive_log_distribution.is_some());

        let mut config = CadConfig::default().with_alpha(1.0).with_max_tokens(1);
        config.shared.record_distributions = false;
        let lean = ContextAwareDecoder::new(config)
            .expect("cfg")
            .generate(&model, "SRC hint", "GO")
            .expect("generate");
        assert!(lean.steps[0].log_distribution.is_none());
    }

    #[test]
    fn sampling_is_reproducible_for_a_fixed_seed() {
        let model = stateful_cad_model();
        let config = CadConfig::default()
            .with_alpha(1.0)
            .with_max_tokens(6)
            .with_strategy(DecodingStrategy::Sampling {
                temperature: 1.0,
                seed: 0xC0FFEE,
            });
        let first = ContextAwareDecoder::new(config.clone())
            .expect("cfg")
            .generate(&model, "SRC hint", "GO")
            .expect("gen1");
        let second = ContextAwareDecoder::new(config)
            .expect("cfg")
            .generate(&model, "SRC hint", "GO")
            .expect("gen2");
        assert_eq!(first.token_ids, second.token_ids, "same seed, same stream");
    }

    #[test]
    fn sampling_never_selects_a_masked_token() {
        // A hard mask must survive re-tempering: a plausibility-excluded token has
        // log p = −inf, so it can never be sampled however the RNG falls. Drive many
        // seeds through the divergence fixture, where token 2 is masked.
        let model = ContextAwareStaticLanguageModel::new(vec![
            "fact".to_string(),
            "filler".to_string(),
            "rare".to_string(),
        ])
        .expect("vocab")
        .with_probability_rule(vec!["CTX", "Q"], &[0.60, 0.399, 0.001])
        .expect("ctx")
        .with_probability_rule(vec!["Q"], &[0.10, 0.899, 0.000001])
        .expect("free");

        for seed in 0..64_u64 {
            let output = ContextAwareDecoder::new(
                CadConfig::default()
                    .with_alpha(1.0)
                    .with_max_tokens(1)
                    .with_strategy(DecodingStrategy::Sampling {
                        temperature: 2.0,
                        seed,
                    }),
            )
            .expect("cfg")
            .generate(&model, "CTX: fact.", "Q?")
            .expect("gen");
            assert_ne!(output.token_ids[0], 2, "seed {seed} sampled a masked token");
        }
    }

    #[test]
    fn a_non_finite_logit_is_caught_and_named() {
        // A model that emits an infinity must be rejected, and the error must say
        // which of the two distributions produced it — half an answer is useless in
        // a contrast.
        let model = ContextAwareStaticLanguageModel::new(vec!["a".to_string(), "b".to_string()])
            .expect("vocab")
            .with_default_logits(vec![f32::INFINITY, 0.0])
            .expect("default");
        let error = ContextAwareDecoder::new(CadConfig::default())
            .expect("cfg")
            .next_token(&model, "ctx", "q")
            .expect_err("must reject");
        assert!(matches!(
            error,
            ContextAwareError::NonFiniteScore { index: 0, .. }
        ));
    }

    #[test]
    fn configuration_is_validated() {
        // alpha_plaus outside [0,1].
        assert!(matches!(
            ContextAwareConfig::default()
                .with_plausibility_alpha(1.5)
                .validate(),
            Err(ContextAwareError::InvalidPlausibilityAlpha { .. })
        ));
        // NaN alpha_plaus.
        assert!(matches!(
            ContextAwareConfig::default()
                .with_plausibility_alpha(f64::NAN)
                .validate(),
            Err(ContextAwareError::InvalidPlausibilityAlpha { .. })
        ));
        // Zero max_tokens.
        assert!(matches!(
            ContextAwareConfig::default().with_max_tokens(0).validate(),
            Err(ContextAwareError::InvalidConfig { .. })
        ));
        // Negative CAD alpha (would push toward the prior — the opposite of CAD).
        assert!(matches!(
            CadConfig::default().with_alpha(-0.5).validate(),
            Err(ContextAwareError::InvalidConfig { .. })
        ));
        // Non-positive amateur temperature.
        assert!(matches!(
            ContrastiveConfig::default()
                .with_amateur_temperature(0.0)
                .validate(),
            Err(ContextAwareError::InvalidConfig { .. })
        ));
        // Valid configurations pass.
        assert!(CadConfig::default().validate().is_ok());
        assert!(ContrastiveConfig::default().validate().is_ok());
        assert!(DolaConfig::default().validate().is_ok());
    }

    #[test]
    fn the_negative_prompt_omits_the_context_but_keeps_the_generation() {
        // The two CAD prompts must be next-token distributions for the SAME
        // position: identical except for the presence of the context.
        let decoder = ContextAwareDecoder::new(CadConfig::default()).expect("cfg");
        let positive = decoder.positive_context("CTX", "Q", "gen");
        let negative = decoder.negative_context("Q", "gen");
        assert_eq!(
            positive, "CTX\n\nQgen",
            "context, separator, query, generation"
        );
        assert_eq!(negative, "Qgen", "context removed, generation kept");
    }
}
