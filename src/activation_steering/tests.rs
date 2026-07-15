#![allow(
    clippy::float_cmp,
    clippy::similar_names,
    clippy::cast_precision_loss,
    clippy::cast_possible_truncation,
    clippy::cast_sign_loss,
    clippy::many_single_char_names,
    clippy::too_many_lines,
    clippy::unreadable_literal,
    clippy::doc_markdown,
    clippy::needless_range_loop
)]
//! Tests for `activation_steering`.
//!
//! The bar here is **measurement against a known generating process**, not
//! self-consistency. The headline tests are:
//!
//! - `mass_mean_recovers_the_planted_direction` — synthesize head activations in
//!   which a *known* unit direction separates the classes, then demand the probe
//!   hand that direction back (cosine ≥ 0.98) and be near-orthogonal to an
//!   unrelated one.
//! - `top_k_selects_exactly_the_signal_heads` — build a dataset in which only `K`
//!   of `H` heads carry signal; demand that top-`K`-by-validation-accuracy is
//!   *exactly* those `K`.
//! - `iti_moves_the_output_by_alpha_times_sigma` — steer the fixture and measure
//!   that its activation moves by the predicted magnitude, and that `alpha = 0` is
//!   bit-identical to no steering.
//! - `captured_edit_does_not_propagate_to_a_later_layer` — prove honest-limit 1:
//!   editing a captured `ModelHiddenStates` cannot change a downstream layer,
//!   whereas the *same* edit through the mid-forward hook does.
//!
//! The logistic probe's `logistic_loss_is_monotonically_non_increasing` is the
//! quiet workhorse: a wrong gradient cannot pass it.

use crate::hidden_states::{HiddenStateTensor, LayerHiddenState, ModelHiddenStates, TensorShape};

use super::ActivationSteering;
use super::engine::split_pairs;
use super::model::{SteerableModel, SteeringFixtureModel};
use super::probe::LinearProbe;
use super::rng::SteeringRng;
use super::types::{
    ActivationPair, ContrastivePair, HeadIndex, Intervention, InterventionConfig, InterventionSite,
    MIN_DIRECTION_NORM, ProbeMethod, SteeringConfig, SteeringGeometry, SteeringPositions,
    SteeringVector,
};

// ── Synthetic generating process ─────────────────────────────────────────────

/// A vector of `dim` independent `N(0, std^2)` draws, as `f32`.
fn gaussian_vec(rng: &mut SteeringRng, dim: usize, std: f64) -> Vec<f32> {
    (0..dim)
        .map(|_| (rng.next_standard_normal() * std) as f32)
        .collect()
}

/// Build contrastive head-activation pairs with a **known** generating process:
/// every head is isotropic `N(0, noise^2)` nuisance, *except* the `signal_slots`,
/// where the positive class is pushed `+mu` along `direction` and the negative
/// class `-mu`. Recovering `direction` from a signal slot is therefore a check
/// against ground truth, and the non-signal slots are genuine distractors (noise,
/// not zeros) that a correct head-selector must reject.
fn synth_head_pairs(
    geometry: SteeringGeometry,
    signal_slots: &[usize],
    direction: &[f64],
    mu: f64,
    noise_std: f64,
    num_pairs: usize,
    seed: u64,
) -> Vec<ActivationPair> {
    let slots = geometry.num_head_slots();
    let head_dim = geometry.head_dim();
    assert_eq!(
        direction.len(),
        head_dim,
        "planted direction is head_dim wide"
    );
    let mut rng = SteeringRng::new(seed);
    let mut pairs = Vec::with_capacity(num_pairs);
    for _ in 0..num_pairs {
        let mut positive = Vec::with_capacity(slots);
        let mut negative = Vec::with_capacity(slots);
        for slot in 0..slots {
            let mut pos = gaussian_vec(&mut rng, head_dim, noise_std);
            let mut neg = gaussian_vec(&mut rng, head_dim, noise_std);
            if signal_slots.contains(&slot) {
                for (k, &component) in direction.iter().enumerate() {
                    pos[k] += (mu * component) as f32;
                    neg[k] -= (mu * component) as f32;
                }
            }
            positive.push(pos);
            negative.push(neg);
        }
        pairs.push(ActivationPair::new(positive, negative).expect("finite synthetic activations"));
    }
    pairs
}

/// The activations of one class at one slot, as the probe wants them.
fn class_at(pairs: &[ActivationPair], slot: usize, positive: bool) -> Vec<&[f32]> {
    pairs
        .iter()
        .map(|pair| {
            let class = if positive {
                &pair.positive
            } else {
                &pair.negative
            };
            class[slot].as_slice()
        })
        .collect()
}

/// A `head_dim`-wide basis vector `e_axis`.
fn basis(head_dim: usize, axis: usize) -> Vec<f64> {
    let mut v = vec![0.0; head_dim];
    v[axis] = 1.0;
    v
}

/// The Euclidean distance between two `f32` activations, computed in `f64`.
fn distance(a: &[f32], b: &[f32]) -> f64 {
    a.iter()
        .zip(b)
        .map(|(x, y)| {
            let d = f64::from(*x) - f64::from(*y);
            d * d
        })
        .sum::<f64>()
        .sqrt()
}

// ── (a) Known generating process: mass-mean recovers the planted direction ────

#[test]
fn mass_mean_recovers_the_planted_direction() {
    let geometry = SteeringGeometry::new(1, 1, 24).expect("valid geometry");
    let head_dim = geometry.head_dim();
    let planted = basis(head_dim, 0);
    let mu = 2.0;
    let noise = 0.5;

    let pairs = synth_head_pairs(geometry, &[0], &planted, mu, noise, 400, 0xC0FFEE);
    let positive = class_at(&pairs, 0, true);
    let negative = class_at(&pairs, 0, false);

    let probe = LinearProbe::fit_mass_mean(&positive, &negative).expect("fit");
    let recovered = probe
        .weight_vector()
        .expect("weights")
        .normalized()
        .expect("unit");

    // Ground truth: cosine to the planted direction is essentially 1.
    let planted_vec = SteeringVector::unit(planted).expect("unit");
    let cosine = recovered.cosine(&planted_vec).expect("cosine");
    assert!(
        cosine >= 0.98,
        "cosine to planted direction = {cosine}, want >= 0.98"
    );

    // ...and near-zero to an unrelated axis the signal was never placed along.
    let unrelated = SteeringVector::unit(basis(head_dim, 1)).expect("unit");
    let off_cosine = recovered.cosine(&unrelated).expect("cosine").abs();
    assert!(
        off_cosine < 0.1,
        "cosine to unrelated axis = {off_cosine}, want ~0"
    );

    // sigma = pooled std of the projections onto the recovered direction, which a
    // known N(±mu, noise^2) mixture makes sqrt(mu^2 + noise^2), *not* noise: the
    // class separation is part of the spread being measured.
    let mut projections = Vec::new();
    for activation in positive.iter().chain(negative.iter()) {
        projections.push(recovered.project(activation).expect("project"));
    }
    let count = projections.len() as f64;
    let mean = projections.iter().sum::<f64>() / count;
    let sigma = (projections
        .iter()
        .map(|p| (p - mean) * (p - mean))
        .sum::<f64>()
        / count)
        .sqrt();
    let predicted = (mu * mu + noise * noise).sqrt();
    assert!(
        (sigma - predicted).abs() < 0.1,
        "sigma = {sigma:.4}, predicted sqrt(mu^2+noise^2) = {predicted:.4}"
    );

    eprintln!(
        "(a) recovered cosine = {cosine:.4}, off-axis = {off_cosine:.4}, \
         sigma = {sigma:.4} (predicted {predicted:.4})"
    );
}

#[test]
fn logistic_probe_also_recovers_a_well_separated_direction() {
    // With strong separation the *discriminative* direction and the mass-mean
    // direction coincide, because there is no low-variance nuisance dimension for
    // the logistic fit to exploit.
    let geometry = SteeringGeometry::new(1, 1, 16).expect("valid geometry");
    let planted = basis(16, 3);
    let pairs = synth_head_pairs(geometry, &[0], &planted, 2.5, 0.4, 300, 7);
    let positive = class_at(&pairs, 0, true);
    let negative = class_at(&pairs, 0, false);

    let (probe, _, _) =
        LinearProbe::fit_logistic(&positive, &negative, 1e-3, 500, 1e-7).expect("fit");
    let recovered = probe
        .weight_vector()
        .expect("weights")
        .normalized()
        .expect("unit");
    let planted_vec = SteeringVector::unit(planted).expect("unit");
    let cosine = recovered.cosine(&planted_vec).expect("cosine");
    assert!(cosine >= 0.95, "logistic cosine = {cosine}, want >= 0.95");
}

// ── The logistic loss must decrease. A wrong gradient cannot pass this. ───────

#[test]
fn logistic_loss_is_monotonically_non_increasing() {
    let geometry = SteeringGeometry::new(1, 1, 12).expect("valid geometry");
    let pairs = synth_head_pairs(geometry, &[0], &basis(12, 5), 1.5, 1.0, 120, 99);
    let positive = class_at(&pairs, 0, true);
    let negative = class_at(&pairs, 0, false);

    let (_, loss_history, _) =
        LinearProbe::fit_logistic(&positive, &negative, 1e-2, 400, 1e-9).expect("fit");
    assert!(loss_history.len() >= 2, "expected several iterates");

    // The provably-safe step size guarantees this. If it ever fails, the gradient
    // is wrong.
    for window in loss_history.windows(2) {
        assert!(
            window[1] <= window[0] + 1e-12,
            "loss increased: {} -> {}",
            window[0],
            window[1]
        );
    }
    // And it actually made progress.
    assert!(
        loss_history[loss_history.len() - 1] < loss_history[0] - 1e-3,
        "loss did not decrease meaningfully"
    );
    eprintln!(
        "(loss) {} iterates, {:.5} -> {:.5}",
        loss_history.len(),
        loss_history[0],
        loss_history[loss_history.len() - 1]
    );
}

// ── (b) Head-selection ground truth ──────────────────────────────────────────

#[test]
fn top_k_selects_exactly_the_signal_heads() {
    let geometry = SteeringGeometry::new(2, 4, 16).expect("valid geometry");
    let signal_slots = [1usize, 3, 6]; // K = 3 of H = 8
    let pairs = synth_head_pairs(
        geometry,
        &signal_slots,
        &basis(16, 0),
        3.0,
        0.5,
        60,
        0x5E1EC7,
    );

    let config = SteeringConfig::default()
        .with_probe_method(ProbeMethod::MassMean)
        .with_top_k_heads(signal_slots.len())
        .with_seed(0xA5A5);
    let mut steering = ActivationSteering::new(geometry, config).expect("engine");
    let report = steering
        .fit_iti_from_activations(&pairs)
        .expect("fit")
        .clone();

    // Selection is EXACTLY the signal heads.
    let mut selected: Vec<HeadIndex> = report.selected.clone();
    selected.sort();
    let mut expected: Vec<HeadIndex> = signal_slots
        .iter()
        .map(|&slot| geometry.head_at(slot).expect("slot"))
        .collect();
    expected.sort();
    assert_eq!(selected, expected, "top-K did not recover the signal heads");

    // Report the numbers: every signal head classifies near-perfectly, and the
    // model-wide mean sits far below because 5 of 8 heads are pure noise.
    for &slot in &signal_slots {
        let head = geometry.head_at(slot).expect("slot");
        let accuracy = report.head(head).expect("head").result.accuracy;
        assert!(
            accuracy.validation >= 0.9,
            "signal head {head} validation = {}",
            accuracy.validation
        );
        eprintln!(
            "(b) signal head {head}: validation = {:.3}",
            accuracy.validation
        );
    }
    let mean = report.mean_validation_accuracy();
    assert!(
        mean < 0.75,
        "mean validation over all heads = {mean}, expected the noise to drag it down"
    );
    eprintln!("(b) mean validation over all 8 heads = {mean:.3}");

    // The worst signal head still beat the best noise head — the ranking is not a
    // coincidence of the tie-break.
    let worst_signal = signal_slots
        .iter()
        .map(|&slot| {
            report
                .head(geometry.head_at(slot).expect("slot"))
                .expect("head")
                .result
                .accuracy
                .validation
        })
        .fold(f64::INFINITY, f64::min);
    let best_noise = (0..geometry.num_head_slots())
        .filter(|slot| !signal_slots.contains(slot))
        .map(|slot| {
            report
                .head(geometry.head_at(slot).expect("slot"))
                .expect("head")
                .result
                .accuracy
                .validation
        })
        .fold(f64::NEG_INFINITY, f64::max);
    assert!(
        worst_signal > best_noise,
        "signal head at {worst_signal} did not beat best noise head at {best_noise}"
    );
}

#[test]
fn logistic_and_mass_mean_agree_on_which_heads_carry_signal() {
    let geometry = SteeringGeometry::new(1, 4, 16).expect("valid geometry");
    let signal_slots = [2usize];
    let pairs = synth_head_pairs(geometry, &signal_slots, &basis(16, 1), 2.5, 0.6, 80, 11);

    let selected_by = |method| {
        let config = SteeringConfig::default()
            .with_probe_method(method)
            .with_top_k_heads(1)
            .with_seed(3);
        let mut steering = ActivationSteering::new(geometry, config).expect("engine");
        steering
            .fit_iti_from_activations(&pairs)
            .expect("fit")
            .selected
            .clone()
    };
    let expected = vec![geometry.head_at(2).expect("slot")];
    assert_eq!(selected_by(ProbeMethod::MassMean), expected);
    assert_eq!(selected_by(ProbeMethod::Logistic), expected);
}

// ── (c) Intervention effect, magnitude, and the alpha = 0 ablation ────────────

/// A fixture with a single truthfulness concept planted in one **last-layer**
/// head, so exactly that head carries signal and no propagation muddies the
/// ground truth. Returns the model, the fitted engine, and the planted head.
fn fitted_single_head() -> (SteeringFixtureModel, ActivationSteering, HeadIndex) {
    let geometry = SteeringGeometry::new(3, 4, 8).expect("valid geometry");
    let last_layer = geometry.num_layers() - 1;
    let planted = HeadIndex::new(last_layer, 2);
    let shift = SteeringVector::unit(basis(8, 0)).expect("unit");
    let concept =
        Intervention::new(InterventionSite::Head(planted), shift, 3.0).expect("intervention");
    let model = SteeringFixtureModel::new(geometry, 16, 0xF1)
        .expect("model")
        .with_concept("TRUTHFUL", concept)
        .expect("concept");

    let pairs: Vec<ContrastivePair> = (0..40)
        .map(|i| {
            ContrastivePair::new(
                format!("TRUTHFUL statement number {i}"),
                format!("statement number {i}"),
            )
        })
        .collect();

    let config = SteeringConfig::default()
        .with_probe_method(ProbeMethod::MassMean)
        .with_top_k_heads(1)
        .with_seed(0xBEE5);
    let mut steering = ActivationSteering::new(geometry, config).expect("engine");
    steering.fit_iti(&model, &pairs).expect("fit");
    (model, steering, planted)
}

#[test]
fn iti_moves_the_output_by_alpha_times_sigma() {
    let (model, steering, planted) = fitted_single_head();

    // The planted head was found, and it is the only non-degenerate one.
    let report = steering.report().expect("fitted");
    assert_eq!(report.selected, vec![planted]);
    let sigma = report.head(planted).expect("head").sigma;
    assert!(
        sigma > 0.0,
        "sigma should be positive for a real signal head"
    );

    let alpha = 2.0;
    let config = InterventionConfig::with_alpha(alpha).expect("alpha");
    let interventions = steering.iti_interventions(&config).expect("interventions");
    assert_eq!(interventions.len(), 1);

    // The intervention's magnitude is exactly alpha * sigma, and — the direction
    // being unit — so is the length of the shift it will add.
    let intervention = &interventions[0];
    assert!((intervention.magnitude - alpha * sigma).abs() < 1e-12);
    assert!((intervention.delta_norm() - alpha * sigma).abs() < 1e-9);

    // Now MEASURE it on a text with no marker, so the concept does not fire and the
    // only change is our steering. With a single intervention the head's input
    // residual is unchanged, so its output moves by *exactly* the delta.
    let text = "statement number 500";
    let clean = model.forward_pass(text, &[]).expect("clean");
    let steered = model.forward_pass(text, &interventions).expect("steered");

    let clean_head = clean.heads.head(planted).expect("clean head");
    let steered_head = steered.heads.head(planted).expect("steered head");
    let moved = distance(clean_head, steered_head);
    assert!(
        (moved - alpha * sigma).abs() < 1e-4,
        "head moved by {moved:.6}, predicted alpha*sigma = {:.6}",
        alpha * sigma
    );

    // And it moved along the fitted direction, not just by the right amount.
    let displacement: Vec<f64> = clean_head
        .iter()
        .zip(steered_head)
        .map(|(c, s)| f64::from(*s) - f64::from(*c))
        .collect();
    let displacement = SteeringVector::new(displacement).expect("finite displacement");
    let cosine = displacement
        .cosine(&report.head(planted).expect("head").direction)
        .expect("cosine");
    assert!(cosine > 0.999, "shift direction cosine = {cosine}");

    // Doubling alpha doubles the shift: the effect is linear in alpha, as claimed.
    let double = InterventionConfig::with_alpha(2.0 * alpha).expect("alpha");
    let double_interventions = steering.iti_interventions(&double).expect("interventions");
    let doubled = model
        .forward_pass(text, &double_interventions)
        .expect("steered");
    let doubled_move = distance(clean_head, doubled.heads.head(planted).expect("head"));
    assert!(
        (doubled_move - 2.0 * moved).abs() < 1e-4,
        "doubling alpha did not double the shift"
    );

    eprintln!(
        "(c) sigma = {sigma:.4}, alpha = {alpha}, head moved {moved:.4} (predicted {:.4})",
        alpha * sigma
    );
}

#[test]
fn alpha_zero_is_bit_identical_to_no_steering() {
    let (model, steering, _) = fitted_single_head();
    let text = "statement number 7";

    let clean = model.forward(text).expect("clean logits");
    let ablated = steering
        .steer(
            &model,
            text,
            &InterventionConfig::with_alpha(0.0).expect("alpha"),
        )
        .expect("ablated logits");

    // Not "close": identical. Every delta is the zero vector, so nothing was added.
    assert_eq!(clean, ablated, "alpha = 0 must be the exact identity");

    // A positive alpha, by contrast, changes the logits.
    let steered = steering
        .steer(
            &model,
            text,
            &InterventionConfig::with_alpha(4.0).expect("alpha"),
        )
        .expect("steered logits");
    assert_ne!(clean, steered, "alpha > 0 must change the output");
}

#[test]
fn caa_edit_changes_the_logits_by_the_predicted_amount() {
    // On the fixture, a residual edit at the last layer changes logit[t] by exactly
    // U[t] . delta, because the last layer's residual *is* the unembedding's input.
    let geometry = SteeringGeometry::new(2, 3, 4).expect("valid geometry");
    let last = geometry.num_layers() - 1;
    let model = SteeringFixtureModel::new(geometry, 10, 0x2C).expect("model");

    let direction = SteeringVector::new(vec![
        0.3, -0.7, 0.5, 0.2, -0.1, 0.4, 0.6, -0.2, 0.1, 0.0, 0.3, -0.5,
    ])
    .expect("finite");
    let alpha = 1.5;
    let intervention = Intervention::new(
        InterventionSite::Residual { layer: last },
        direction.clone(),
        alpha,
    )
    .expect("intervention");

    let text = "alpha beta gamma";
    let clean = model.forward(text).expect("clean");
    let steered = model
        .forward_with_interventions(text, &[intervention])
        .expect("steered");

    let delta = direction.scaled(alpha).expect("scaled");
    for token in 0..model.vocab_size() {
        let row = model.unembedding_row(token).expect("row");
        let predicted: f64 = row.iter().zip(delta.as_slice()).map(|(u, d)| u * d).sum();
        let observed = f64::from(steered[token]) - f64::from(clean[token]);
        assert!(
            (observed - predicted).abs() < 1e-4,
            "logit {token}: observed change {observed:.6}, predicted U.delta {predicted:.6}"
        );
    }
}

// ── Mid-forward propagation: the intervention reaches later layers ────────────

#[test]
fn steering_an_early_head_propagates_to_a_later_layer_and_the_logits() {
    let geometry = SteeringGeometry::new(3, 2, 6).expect("valid geometry");
    let model = SteeringFixtureModel::new(geometry, 12, 0x9A).expect("model");
    let early = HeadIndex::new(0, 1);
    let direction = SteeringVector::unit(basis(6, 2)).expect("unit");
    let intervention =
        Intervention::new(InterventionSite::Head(early), direction, 5.0).expect("intervention");

    let text = "one two three four";
    let clean = model.forward_pass(text, &[]).expect("clean");
    let steered = model.forward_pass(text, &[intervention]).expect("steered");

    // Layer 0's steered head obviously changed. The claim is that layer 2's
    // residual changed too — with no intervention *there*, purely by propagation.
    let clean_l2 = clean.residual.layer(2).expect("layer 2");
    let steered_l2 = steered.residual.layer(2).expect("layer 2");
    assert!(
        distance(clean_l2, steered_l2) > 1e-3,
        "an edit at layer 0 did not reach layer 2 — propagation is broken"
    );
    // And it reached the output.
    assert_ne!(
        clean.logits, steered.logits,
        "the edit did not reach the logits"
    );
}

// ── (d) Probe generalization on a seeded split ───────────────────────────────

#[test]
fn probe_generalizes_to_a_held_out_split() {
    let geometry = SteeringGeometry::new(1, 1, 16).expect("valid geometry");

    // Strong separation: the held-out split is classified *perfectly*. This is a
    // number (1.0), not "> 0.5".
    let clean = synth_head_pairs(geometry, &[0], &basis(16, 0), 3.0, 0.3, 50, 1234);
    let config = SteeringConfig::default()
        .with_probe_method(ProbeMethod::Logistic)
        .with_top_k_heads(1)
        .with_train_fraction(0.8)
        .with_seed(0xD00D);
    let mut steering = ActivationSteering::new(geometry, config).expect("engine");
    let report = steering
        .fit_iti_from_activations(&clean)
        .expect("fit")
        .clone();
    let accuracy = report
        .head(HeadIndex::new(0, 0))
        .expect("head")
        .result
        .accuracy;
    assert_eq!(report.num_train_pairs, 40);
    assert_eq!(report.num_validation_pairs, 10);
    assert_eq!(
        accuracy.validation, 1.0,
        "well-separated data should generalize perfectly"
    );

    // Moderate separation: the held-out accuracy is a specific, reproducible value
    // strictly between chance and perfection — the honest generalization number.
    let noisy = synth_head_pairs(geometry, &[0], &basis(16, 0), 0.7, 1.0, 200, 4321);
    let noisy_config = SteeringConfig::default()
        .with_probe_method(ProbeMethod::Logistic)
        .with_top_k_heads(1)
        .with_train_fraction(0.8)
        .with_seed(0xD00D);
    let mut noisy_steering = ActivationSteering::new(geometry, noisy_config).expect("engine");
    let noisy_validation = noisy_steering
        .fit_iti_from_activations(&noisy)
        .expect("fit")
        .head(HeadIndex::new(0, 0))
        .expect("head")
        .result
        .accuracy
        .validation;
    assert!(
        noisy_validation > 0.55 && noisy_validation < 1.0,
        "moderate-separation validation = {noisy_validation}, expected a genuine number in (0.55, 1.0)"
    );

    // Reproducible: the seeded split makes the number an artifact, not a lucky draw.
    let repeat_config = SteeringConfig::default()
        .with_probe_method(ProbeMethod::Logistic)
        .with_top_k_heads(1)
        .with_train_fraction(0.8)
        .with_seed(0xD00D);
    let mut repeat = ActivationSteering::new(geometry, repeat_config).expect("engine");
    let repeat_validation = repeat
        .fit_iti_from_activations(&noisy)
        .expect("fit")
        .head(HeadIndex::new(0, 0))
        .expect("head")
        .result
        .accuracy
        .validation;
    assert_eq!(
        noisy_validation, repeat_validation,
        "the seeded split must be reproducible"
    );

    eprintln!("(d) strong-separation validation = 1.0, moderate = {noisy_validation:.4}");
}

#[test]
fn the_split_is_seeded_disjoint_and_covers_every_pair() {
    let (train, validation) = split_pairs(50, 0.8, 42).expect("split");
    assert_eq!(train.len(), 40);
    assert_eq!(validation.len(), 10);

    // Disjoint and complete.
    let mut all: Vec<usize> = train.iter().chain(&validation).copied().collect();
    all.sort_unstable();
    all.dedup();
    assert_eq!(
        all.len(),
        50,
        "the split must partition all 50 pairs with no overlap"
    );

    // Seeded: same seed, same split; different seed, (almost surely) different.
    let (train_again, _) = split_pairs(50, 0.8, 42).expect("split");
    assert_eq!(train, train_again);
    let (train_other, _) = split_pairs(50, 0.8, 43).expect("split");
    assert_ne!(train, train_other);

    // Both splits are always non-empty, even for the smallest admissible dataset.
    let (small_train, small_validation) = split_pairs(2, 0.99, 1).expect("split");
    assert_eq!(small_train.len(), 1);
    assert_eq!(small_validation.len(), 1);
    assert!(split_pairs(1, 0.8, 1).is_err(), "one pair cannot be split");
}

// ── (e) The honest-limit test: a captured snapshot has no forward coupling ────

#[test]
fn captured_edit_does_not_propagate_to_a_later_layer() {
    // Fit a CAA vector for layer 0, so we have a real steering direction to apply.
    let geometry = SteeringGeometry::new(2, 2, 4).expect("valid geometry");
    let hidden_dim = geometry.hidden_dim();
    let pairs = {
        let mut rng = SteeringRng::new(55);
        (0..30)
            .map(|_| {
                let mut positive = Vec::new();
                let mut negative = Vec::new();
                for _ in 0..geometry.num_layers() {
                    let mut pos = gaussian_vec(&mut rng, hidden_dim, 0.5);
                    let neg = gaussian_vec(&mut rng, hidden_dim, 0.5);
                    pos[0] += 2.0; // a real difference for CAA to find
                    positive.push(pos);
                    negative.push(neg);
                }
                ActivationPair::new(positive, negative).expect("finite")
            })
            .collect::<Vec<_>>()
    };
    let mut steering =
        ActivationSteering::new(geometry, SteeringConfig::default()).expect("engine");
    steering.fit_caa_from_activations(&pairs).expect("fit caa");

    // Build a captured ModelHiddenStates the way the crate's extractors do:
    // two layers, each [1, seq_len, hidden_dim].
    let seq_len = 3;
    let mut states = ModelHiddenStates::new("fixture", geometry.num_layers(), hidden_dim);
    states.sequence_length = seq_len;
    for layer in 0..geometry.num_layers() {
        let data: Vec<f32> = (0..seq_len * hidden_dim)
            .map(|k| (layer * 100 + k) as f32)
            .collect();
        let tensor =
            HiddenStateTensor::from_vec(data, TensorShape::new(vec![1, seq_len, hidden_dim]))
                .expect("tensor");
        states.add_layer(LayerHiddenState::new(layer, tensor));
    }

    let layer0_before = states
        .get_layer(0)
        .expect("layer 0")
        .hidden_state
        .data
        .clone();
    let layer1_before = states
        .get_layer(1)
        .expect("layer 1")
        .hidden_state
        .data
        .clone();

    // Edit layer 0 of the *captured snapshot*.
    let edited = steering
        .edit_captured_residual(
            &mut states,
            0,
            &InterventionConfig::with_alpha(3.0).expect("alpha"),
        )
        .expect("edit");
    assert_eq!(edited, seq_len, "every position should have been edited");

    let layer0_after = &states.get_layer(0).expect("layer 0").hidden_state.data;
    let layer1_after = &states.get_layer(1).expect("layer 1").hidden_state.data;

    // Layer 0 changed...
    assert_ne!(
        &layer0_before, layer0_after,
        "the edit should have changed layer 0"
    );
    // ...but layer 1 is BIT-IDENTICAL. There is no model here to recompute it from
    // the edited layer 0 — a captured ModelHiddenStates is an owned snapshot with
    // no forward coupling. This is honest-limit 1, proven rather than asserted.
    assert_eq!(
        &layer1_before, layer1_after,
        "editing a captured snapshot must not touch a later layer"
    );

    // Contrast: the SAME conceptual edit, applied through the mid-forward hook,
    // DOES change a later layer (and the logits). That is the whole reason this
    // module defines its own SteerableModel trait.
    let model = SteeringFixtureModel::new(geometry, 8, 0x77).expect("model");
    let text = "red green blue";
    let clean = model.forward_pass(text, &[]).expect("clean");
    let caa = steering.caa_vector(0).expect("caa layer 0");
    let intervention = caa
        .intervention(&InterventionConfig::with_alpha(3.0).expect("alpha"))
        .expect("intervention");
    let steered = model.forward_pass(text, &[intervention]).expect("steered");
    assert!(
        distance(
            clean.residual.layer(1).expect("l1"),
            steered.residual.layer(1).expect("l1")
        ) > 1e-6,
        "the mid-forward edit at layer 0 must reach layer 1"
    );
    assert_ne!(
        clean.logits, steered.logits,
        "the mid-forward edit must reach the logits"
    );
}

// ── CAA arithmetic ───────────────────────────────────────────────────────────

#[test]
fn caa_vector_is_the_difference_of_class_means() {
    // For a complete pairing, mean-of-paired-differences == difference-of-means.
    // Computed two ways, asserted equal.
    let geometry = SteeringGeometry::new(1, 2, 3).expect("valid geometry");
    let hidden_dim = geometry.hidden_dim();
    let mut rng = SteeringRng::new(2024);
    let pairs: Vec<ActivationPair> = (0..25)
        .map(|_| {
            let positive = vec![gaussian_vec(&mut rng, hidden_dim, 1.0)];
            let negative = vec![gaussian_vec(&mut rng, hidden_dim, 1.0)];
            ActivationPair::new(positive, negative).expect("finite")
        })
        .collect();

    let mut steering =
        ActivationSteering::new(geometry, SteeringConfig::default()).expect("engine");
    let vectors = steering
        .fit_caa_from_activations(&pairs)
        .expect("fit")
        .to_vec();
    assert_eq!(vectors.len(), 1);
    let caa = &vectors[0];
    assert_eq!(caa.num_pairs, 25);

    let count = pairs.len() as f64;
    for k in 0..hidden_dim {
        let pos_mean: f64 = pairs
            .iter()
            .map(|p| f64::from(p.positive[0][k]))
            .sum::<f64>()
            / count;
        let neg_mean: f64 = pairs
            .iter()
            .map(|p| f64::from(p.negative[0][k]))
            .sum::<f64>()
            / count;
        let expected = pos_mean - neg_mean;
        assert!(
            (caa.vector.as_slice()[k] - expected).abs() < 1e-9,
            "component {k}: {} vs difference-of-means {expected}",
            caa.vector.as_slice()[k]
        );
    }
    // The norm is the effect size, and it is positive for a genuine contrast.
    assert!(caa.norm() > 0.0);
}

#[test]
fn caa_can_target_a_subset_of_layers() {
    let geometry = SteeringGeometry::new(4, 2, 4).expect("valid geometry");
    let hidden_dim = geometry.hidden_dim();
    let mut rng = SteeringRng::new(9);
    let pairs: Vec<ActivationPair> = (0..12)
        .map(|_| {
            let mut positive = Vec::new();
            let mut negative = Vec::new();
            for _ in 0..geometry.num_layers() {
                positive.push(gaussian_vec(&mut rng, hidden_dim, 1.0));
                negative.push(gaussian_vec(&mut rng, hidden_dim, 1.0));
            }
            ActivationPair::new(positive, negative).expect("finite")
        })
        .collect();

    let config = SteeringConfig::default().with_caa_layers(vec![1, 3]);
    let mut steering = ActivationSteering::new(geometry, config).expect("engine");
    let vectors = steering.fit_caa_from_activations(&pairs).expect("fit");
    let layers: Vec<usize> = vectors.iter().map(|v| v.layer).collect();
    assert_eq!(
        layers,
        vec![1, 3],
        "only the configured layers should be fitted"
    );
}

// ── Degenerate heads, and the selection guarantees around them ────────────────

#[test]
fn a_head_that_cannot_separate_the_classes_is_flagged_degenerate() {
    // On the fixture, every head EXCEPT the planted one sees identical activations
    // for a pair's positive and negative (the marker is excluded from content), so
    // its mass-mean direction is exactly zero.
    let (_, steering, planted) = fitted_single_head();
    let report = steering.report().expect("fitted");

    assert!(
        !report.head(planted).expect("head").degenerate,
        "the planted head carries signal"
    );
    let expected_degenerate = report.geometry.num_head_slots() - 1;
    assert_eq!(report.num_degenerate, expected_degenerate);

    // A degenerate head has a zero direction and zero sigma, and is never selected.
    for report_entry in &report.heads {
        if report_entry.degenerate {
            assert!(report_entry.direction.norm() < MIN_DIRECTION_NORM);
            assert_eq!(report_entry.sigma, 0.0);
            assert!(!report.selected.contains(&report_entry.head));
        }
    }
}

#[test]
fn selecting_more_heads_than_carry_signal_is_an_error_not_a_fabrication() {
    // Same fixture, but ask for 2 heads when only 1 is non-degenerate. Padding the
    // selection with a zero-direction head would be a silent no-op reported as a
    // steering vector; instead it is a loud error.
    let geometry = SteeringGeometry::new(2, 2, 6).expect("valid geometry");
    let last = geometry.num_layers() - 1;
    let planted = HeadIndex::new(last, 0);
    let shift = SteeringVector::unit(basis(6, 0)).expect("unit");
    let concept =
        Intervention::new(InterventionSite::Head(planted), shift, 2.0).expect("intervention");
    let model = SteeringFixtureModel::new(geometry, 8, 3)
        .expect("model")
        .with_concept("MARK", concept)
        .expect("concept");
    let pairs: Vec<ContrastivePair> = (0..20)
        .map(|i| ContrastivePair::new(format!("MARK item {i}"), format!("item {i}")))
        .collect();

    let config = SteeringConfig::default()
        .with_top_k_heads(2)
        .with_probe_method(ProbeMethod::MassMean);
    let mut steering = ActivationSteering::new(geometry, config).expect("engine");
    let error = steering.fit_iti(&model, &pairs).unwrap_err();
    assert!(
        matches!(
            error,
            super::types::SteeringError::InsufficientHeads { available: 1, .. }
        ),
        "expected InsufficientHeads, got {error:?}"
    );
}

// ── Determinism ──────────────────────────────────────────────────────────────

#[test]
fn fitting_is_fully_deterministic() {
    let geometry = SteeringGeometry::new(2, 2, 8).expect("valid geometry");
    let pairs = synth_head_pairs(geometry, &[0, 3], &basis(8, 1), 2.0, 0.7, 40, 17);
    let config = SteeringConfig::default().with_top_k_heads(2).with_seed(88);

    let fit_once = || {
        let mut steering = ActivationSteering::new(geometry, config.clone()).expect("engine");
        steering
            .fit_iti_from_activations(&pairs)
            .expect("fit")
            .clone()
    };
    // Same inputs, byte-identical report.
    assert_eq!(fit_once(), fit_once());
}

// ── The fixture model's own guarantees ───────────────────────────────────────

#[test]
fn fixture_is_deterministic_and_has_the_right_shapes() {
    let geometry = SteeringGeometry::new(3, 4, 8).expect("valid geometry");
    let model = SteeringFixtureModel::new(geometry, 20, 42).expect("model");

    let a = model
        .forward_pass("the quick brown fox", &[])
        .expect("pass");
    let b = model
        .forward_pass("the quick brown fox", &[])
        .expect("pass");
    assert_eq!(a, b, "the fixture must be a pure function of its input");

    assert_eq!(a.heads.len(), geometry.num_head_slots());
    assert_eq!(
        a.heads.head(HeadIndex::new(1, 2)).expect("head").len(),
        geometry.head_dim()
    );
    assert_eq!(a.residual.num_layers(), geometry.num_layers());
    assert_eq!(
        a.residual.layer(0).expect("layer").len(),
        geometry.hidden_dim()
    );
    assert_eq!(a.logits.len(), model.vocab_size());

    // Different content, different activations.
    let c = model
        .forward_pass("a different sentence entirely", &[])
        .expect("pass");
    assert_ne!(a.logits, c.logits);
}

#[test]
fn steering_positions_select_the_right_tokens() {
    // The pure-function contract of SteeringPositions::includes.
    assert!(SteeringPositions::All.includes(0, 3));
    assert!(SteeringPositions::All.includes(2, 3));
    assert!(!SteeringPositions::All.includes(3, 3));
    assert!(!SteeringPositions::LastToken.includes(0, 3));
    assert!(!SteeringPositions::LastToken.includes(1, 3));
    assert!(SteeringPositions::LastToken.includes(2, 3));
    assert!(!SteeringPositions::LastToken.includes(0, 0));

    // Behaviorally, on a model with a *downstream* layer to read them: both
    // variants steer the final position identically, but only All also moves the
    // earlier positions, and that difference reaches the logits through layer 1's
    // causal mean. (A single-layer model could not show this: within one layer,
    // attention reads the layer input, so an earlier position's steered output is
    // never read again — which is exactly why the head is at an EARLY layer here.)
    let geometry = SteeringGeometry::new(2, 1, 4).expect("valid geometry");
    let model = SteeringFixtureModel::new(geometry, 6, 1).expect("model");
    let head = HeadIndex::new(0, 0);
    let direction = SteeringVector::unit(basis(4, 0)).expect("unit");

    let all = Intervention::new(InterventionSite::Head(head), direction.clone(), 2.0)
        .expect("intervention")
        .at_positions(SteeringPositions::All);
    let last = Intervention::new(InterventionSite::Head(head), direction, 2.0)
        .expect("intervention")
        .at_positions(SteeringPositions::LastToken);

    let text = "one two three";
    let clean = model.forward(text).expect("clean");
    let all_logits = model.forward_with_interventions(text, &[all]).expect("all");
    let last_logits = model
        .forward_with_interventions(text, &[last])
        .expect("last");

    assert_ne!(
        clean, last_logits,
        "last-token steering still changes the output"
    );
    assert_ne!(
        all_logits, last_logits,
        "all-position steering additionally moves the earlier positions"
    );
}

#[test]
fn intervention_arithmetic_matches_its_definition() {
    let direction = SteeringVector::new(vec![1.0, -2.0, 0.5]).expect("finite");
    let intervention = Intervention::new(InterventionSite::Residual { layer: 0 }, direction, 3.0)
        .expect("intervention");
    assert_eq!(intervention.delta(), vec![3.0, -6.0, 1.5]);
    assert_eq!(intervention.delta_f32(), vec![3.0f32, -6.0, 1.5]);
    // delta_norm = |magnitude| * ||direction||.
    let expected = 3.0 * (1.0f64 + 4.0 + 0.25).sqrt();
    assert!((intervention.delta_norm() - expected).abs() < 1e-12);
}

// ── Validation and error paths ───────────────────────────────────────────────

#[test]
fn configuration_and_geometry_reject_nonsense() {
    assert!(SteeringGeometry::new(0, 4, 8).is_err());
    assert!(SteeringGeometry::new(2, 0, 8).is_err());
    assert!(SteeringGeometry::new(2, 4, 0).is_err());

    let geometry = SteeringGeometry::new(2, 4, 8).expect("valid");
    assert_eq!(geometry.hidden_dim(), 32);
    assert_eq!(geometry.num_head_slots(), 8);
    assert_eq!(geometry.head_slice(3).expect("slice"), 24..32);
    assert!(geometry.head_slice(4).is_err());
    assert_eq!(geometry.flat_index(HeadIndex::new(1, 2)).expect("flat"), 6);
    assert_eq!(geometry.head_at(6).expect("head"), HeadIndex::new(1, 2));

    assert!(
        SteeringConfig::default()
            .with_top_k_heads(0)
            .validate()
            .is_err()
    );
    assert!(
        SteeringConfig::default()
            .with_train_fraction(0.0)
            .validate()
            .is_err()
    );
    assert!(
        SteeringConfig::default()
            .with_train_fraction(1.0)
            .validate()
            .is_err()
    );
    assert!(
        SteeringConfig::default()
            .with_l2_penalty(-1.0)
            .validate()
            .is_err()
    );
    assert!(InterventionConfig::with_alpha(f64::NAN).is_err());
}

#[test]
fn vectors_reject_non_finite_and_zero_directions() {
    assert!(SteeringVector::new(vec![]).is_err());
    assert!(SteeringVector::new(vec![1.0, f64::NAN]).is_err());
    assert!(SteeringVector::new(vec![1.0, f64::INFINITY]).is_err());

    let zero = SteeringVector::zeros(4).expect("zeros");
    assert!(zero.is_degenerate());
    assert!(zero.normalized().is_err());

    let unit = SteeringVector::unit(vec![3.0, 4.0]).expect("unit");
    assert!((unit.norm() - 1.0).abs() < 1e-12);
    assert_eq!(unit.as_slice(), &[0.6, 0.8]);

    // cosine to the zero vector is undefined, not silently zero.
    assert!(
        unit.cosine(&SteeringVector::zeros(2).expect("zeros"))
            .is_err()
    );
}

#[test]
fn probes_and_fits_reject_empty_or_ragged_input() {
    let empty: Vec<&[f32]> = vec![];
    let some: Vec<&[f32]> = vec![&[1.0, 2.0]];
    assert!(LinearProbe::fit_mass_mean(&empty, &some).is_err());
    assert!(LinearProbe::fit_mass_mean(&some, &empty).is_err());

    let ragged_a: Vec<&[f32]> = vec![&[1.0, 2.0]];
    let ragged_b: Vec<&[f32]> = vec![&[1.0, 2.0, 3.0]];
    assert!(LinearProbe::fit_mass_mean(&ragged_a, &ragged_b).is_err());

    let geometry = SteeringGeometry::new(1, 2, 4).expect("valid");
    let mut steering =
        ActivationSteering::new(geometry, SteeringConfig::default()).expect("engine");
    assert!(steering.report().is_err(), "unfitted engine has no report");
    assert!(
        steering.fit_iti_from_activations(&[]).is_err(),
        "no pairs is an error"
    );
}

#[test]
fn edit_captured_residual_rejects_a_malformed_capture() {
    let geometry = SteeringGeometry::new(1, 2, 4).expect("valid");
    let hidden_dim = geometry.hidden_dim();
    let mut rng = SteeringRng::new(5);
    let pairs: Vec<ActivationPair> = (0..8)
        .map(|_| {
            ActivationPair::new(
                vec![gaussian_vec(&mut rng, hidden_dim, 1.0)],
                vec![gaussian_vec(&mut rng, hidden_dim, 1.0)],
            )
            .expect("finite")
        })
        .collect();
    let mut steering =
        ActivationSteering::new(geometry, SteeringConfig::default()).expect("engine");
    steering.fit_caa_from_activations(&pairs).expect("fit");

    // A tensor of the wrong hidden width is rejected rather than half-written.
    let mut states = ModelHiddenStates::new("m", 1, hidden_dim);
    states.sequence_length = 2;
    let wrong = HiddenStateTensor::from_vec(
        vec![0.0; 2 * (hidden_dim + 1)],
        TensorShape::new(vec![1, 2, hidden_dim + 1]),
    )
    .expect("tensor");
    states.add_layer(LayerHiddenState::new(0, wrong));
    assert!(
        steering
            .edit_captured_residual(&mut states, 0, &InterventionConfig::default())
            .is_err()
    );
}
