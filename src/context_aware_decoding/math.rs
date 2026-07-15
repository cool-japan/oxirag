//! The four primitives of decode-time contrast: combine, constrain,
//! renormalize, and diverge.
//!
//! Everything here is built on `REPLUG`'s log-space `f64` kernels
//! ([`crate::replug::math`]) rather than on a second, subtly-different copy of
//! them. That is a deliberate dependency, not an accident of convenience: those
//! kernels are written against underflow, they are tested against logit spreads
//! of `±10⁴`, and a contrast is a *difference of two logarithms* — the single
//! most cancellation-prone operation a next-token distribution can be put
//! through. Re-deriving `log_softmax` here would mean re-deriving its failure
//! modes.
//!
//! # What this module adds that `REPLUG` does not have
//!
//! `REPLUG` interpolates. Its pooling weights are a softmax, so they are
//! non-negative and sum to one, and the result of
//! `Σ_i λ_i · p_i` is a convex combination — it can never leave the probability
//! simplex, and no `p_i` can ever be *subtracted*.
//!
//! This module extrapolates. Its second weight is **negative**. The combined
//! score
//!
//! ```text
//!   s(y)  =  w⁺ · v⁺(y)  +  w⁻ · v⁻(y) ,        w⁻ < 0
//! ```
//!
//! exponentiates to `p⁺(y)^{w⁺} / p⁻(y)^{|w⁻|}`, and that quotient **diverges as
//! `p⁻(y) → 0`**. A token that the negative distribution considers impossible is
//! amplified without bound — and the negative distribution is, by construction,
//! the *less informed* of the two, so it is exactly the distribution whose zeros
//! should carry the least authority. Unconstrained extrapolation hands them the
//! most.
//!
//! [`adaptive_plausibility_mask`] is the fix, and it is the piece a convex
//! mixture never needs.

use std::f64::consts::LN_2;

use crate::replug::math::{kl_divergence_from_log_probs, log_softmax, log_sum_exp};

use super::types::{ContextAwareError, ContextAwareResult};

/// How far `log Σ_y exp(v_y)` may stray from `0` before a vector is refused as a
/// log-probability vector.
///
/// A vector produced by `log_softmax` over a vocabulary of `V` tokens
/// accumulates `O(V)` roundings of relative size `2⁻⁵³`, so even a 128k
/// vocabulary lands within `~10⁻¹³` of zero. Raw logits, on the other hand, miss
/// by `O(1)` or more. `1e-9` sits comfortably between the two: loose enough that
/// no genuine distribution is ever rejected, tight enough that no logit vector
/// is ever accepted.
const NORMALIZATION_TOLERANCE: f64 = 1e-9;

// ── Validation ───────────────────────────────────────────────────────────────

/// Reject an empty score vector, and any `NaN` or infinite entry in it.
///
/// The finiteness requirement is specific to contrast and is not pedantry. The
/// combination multiplies each score by a weight that may be negative or zero:
///
/// - `-0.0 · -inf` is `NaN` — so a `CAD` run at `alpha = 0`, the *ablation*, is
///   where an infinite negative score would first poison the output, and it
///   would do so silently.
/// - `-1.0 · -inf` is `+inf` — an infinitely confident vote for precisely the
///   token the negative model ruled out.
///
/// `REPLUG`'s mixture, whose weights are all non-negative, meets neither hazard;
/// this is one of several places where the sign of the weight changes what the
/// code has to defend against.
fn require_finite_scores(scores: &[f64], origin: &str) -> ContextAwareResult<()> {
    if scores.is_empty() {
        return Err(ContextAwareError::EmptyLogits);
    }
    for (index, &value) in scores.iter().enumerate() {
        if !value.is_finite() {
            return Err(ContextAwareError::NonFiniteScore {
                index,
                origin: origin.to_string(),
                value,
            });
        }
    }
    Ok(())
}

/// Reject two score vectors of different lengths.
fn require_same_length(positive: &[f64], negative: &[f64], origin: &str) -> ContextAwareResult<()> {
    if positive.len() == negative.len() {
        Ok(())
    } else {
        Err(ContextAwareError::VocabSizeMismatch {
            expected: positive.len(),
            actual: negative.len(),
            origin: origin.to_string(),
        })
    }
}

// ── Combine ──────────────────────────────────────────────────────────────────

/// The contrast kernel: `s(y) = w⁺ · v⁺(y) + w⁻ · v⁻(y)`, **unnormalized**.
///
/// # Why the inputs are "score vectors" and not "log-probabilities"
///
/// Both are accepted, and it does not matter which you pass, because of a small
/// lemma that the three papers implicitly rely on and that is worth stating
/// once. Write a model's log-probabilities as `l(y) = z(y) − c`, where `z` are
/// its raw logits and `c = logsumexp(z)` is a constant that does not depend on
/// `y`. Then
///
/// ```text
///   w⁺ · l⁺(y) + w⁻ · l⁻(y)  =  w⁺ · z⁺(y) + w⁻ · z⁻(y)  −  (w⁺ · c⁺ + w⁻ · c⁻)
/// ```
///
/// and the bracketed term is *also* independent of `y`. A `log_softmax` is
/// invariant to an additive constant, so once [`contrast_log_probs`] normalizes,
/// the two forms are the same distribution — **for any weights, including
/// negative ones**. That is why Shi et al. can write `CAD` on logits, Chuang et
/// al. can write `DoLa` on log-probabilities, and both are correct.
///
/// The same invariance is why the plausibility mask can be read off the raw
/// scores (see [`adaptive_plausibility_mask`]).
///
/// This module's decoders pass **raw promoted logits**, for one reason: it makes
/// the `alpha = 0` ablation exact. `(1 + 0) · z⁺ + (−0) · z⁻` is bit-for-bit
/// `z⁺`, so `log_softmax` of it is bit-for-bit the model's own distribution,
/// whereas normalizing first and then normalizing again would differ from it in
/// the last few bits.
///
/// # Errors
///
/// Returns [`ContextAwareError::EmptyLogits`] for an empty vector,
/// [`ContextAwareError::NonFiniteScore`] for a `NaN` or infinite entry (see the
/// hazard note on this module),
/// [`ContextAwareError::VocabSizeMismatch`] when the two vectors differ in
/// length, and [`ContextAwareError::InvalidConfig`] for a non-finite weight.
pub fn contrast_scores(
    positive: &[f64],
    negative: &[f64],
    positive_weight: f64,
    negative_weight: f64,
) -> ContextAwareResult<Vec<f64>> {
    require_finite_scores(positive, "positive")?;
    require_finite_scores(negative, "negative")?;
    require_same_length(positive, negative, "negative distribution")?;
    if !positive_weight.is_finite() || !negative_weight.is_finite() {
        return Err(ContextAwareError::InvalidConfig {
            reason: format!(
                "contrast weights must be finite, got ({positive_weight}, {negative_weight})"
            ),
        });
    }

    Ok(positive
        .iter()
        .zip(negative)
        .map(|(&plus, &minus)| positive_weight * plus + negative_weight * minus)
        .collect())
}

/// The contrast kernel, renormalized: `log_softmax(w⁺ · v⁺ + w⁻ · v⁻)`.
///
/// **This is `REPLUG`'s log-linear pool, evaluated at weights that leave the
/// simplex.** `crate::replug::math::log_linear_pool_log_probs(&[w⁺, w⁻], &[v⁺,
/// v⁻])` computes the identical vector, and this module's tests assert exactly
/// that, bit-for-bit, as a cross-check against an implementation it did not
/// write.
///
/// The agreement is not an argument that the two modules do the same thing. That
/// function's own documentation reasons about its weights *exclusively* as a
/// softmax — non-negative, summing to one — under which it is a weighted
/// **geometric** mean of distributions, and it exists in `REPLUG` chiefly as a
/// foil: the "wrong pool", implemented so that the difference from the
/// arithmetic mixture can be measured. At `w⁻ < 0` its factor `p⁻(y)^{w⁻}`
/// becomes `1 / p⁻(y)^{|w⁻|}` and *diverges*, a regime neither its documentation
/// nor its tests cover. What this module contributes is not the eight-line
/// kernel; it is the two contexts, the layer selection, and above all the
/// constraint in [`masked_contrast_log_probs`] that makes the divergence safe.
///
/// # Errors
///
/// As [`contrast_scores`].
pub fn contrast_log_probs(
    positive: &[f64],
    negative: &[f64],
    positive_weight: f64,
    negative_weight: f64,
) -> ContextAwareResult<Vec<f64>> {
    let scores = contrast_scores(positive, negative, positive_weight, negative_weight)?;
    Ok(log_softmax(&scores))
}

// ── Constrain ────────────────────────────────────────────────────────────────

/// The **adaptive plausibility constraint**:
/// `V_valid = { y : p⁺(y) ≥ alpha · max_{y'} p⁺(y') }`.
///
/// Returns one `bool` per token: `true` for the tokens the contrast is allowed to
/// reorder, `false` for the tokens it must not touch. It is *adaptive* because
/// the threshold is a fraction of the running maximum rather than an absolute
/// probability: where the positive distribution is confident the surviving set is
/// tiny, and where it is uncertain the surviving set is large — the constraint
/// tightens exactly where the model has something to say.
///
/// # Why it is computed in log-space, from scores
///
/// The obvious implementation compares `p(y) ≥ alpha · max p` in probability
/// space, and it is wrong for the case that matters. The tokens this constraint
/// is *for* are the ones with tiny `p⁺` — and on a real vocabulary those
/// underflow. Once `p⁺(y)` and `alpha · max p⁺` are both `0.0`, the comparison
/// `0.0 ≥ 0.0` is `true` and the constraint silently admits the token it exists
/// to exclude.
///
/// In log-space the same predicate is
///
/// ```text
///   v(y) − max_{y'} v(y')  ≥  ln(alpha)
/// ```
///
/// which is a difference of two finite numbers and holds its resolution all the
/// way down. It also shows why the input may be **any** score vector — logits or
/// log-probabilities: the predicate depends only on `v(y) − max v`, which is
/// invariant to the additive normalizer that separates them.
///
/// # Two exact endpoints, and one theorem
///
/// - `alpha = 0`: `ln(0) = -inf` and every token satisfies `v(y) − max ≥ -inf`.
///   The constraint is **vacuous**, not approximately but exactly, so
///   [`masked_contrast_log_probs`] at `alpha = 0` is bit-for-bit
///   [`contrast_log_probs`].
/// - `alpha = 1`: `ln(1) = 0` and only the arg-max survives (with anything
///   exactly tied to it).
/// - For every `alpha ∈ [0, 1]` the result contains **at least one** `true`: the
///   arg-max satisfies `0 ≥ ln(alpha)` whenever `alpha ≤ 1`. The constraint can
///   therefore never empty the vocabulary and leave nothing to decode. This is
///   why `alpha > 1` is rejected rather than clamped.
///
/// # Errors
///
/// Returns [`ContextAwareError::EmptyLogits`] for an empty vector,
/// [`ContextAwareError::InvalidPlausibilityAlpha`] when `alpha` is outside
/// `[0, 1]` or `NaN`, and [`ContextAwareError::NonFiniteScore`] for a `NaN` or
/// infinite entry.
pub fn adaptive_plausibility_mask(scores: &[f64], alpha: f64) -> ContextAwareResult<Vec<bool>> {
    require_finite_scores(scores, "positive")?;
    // `NaN` fails every comparison, so `contains` rejects it without a special
    // case.
    if !(0.0..=1.0).contains(&alpha) {
        return Err(ContextAwareError::InvalidPlausibilityAlpha { alpha });
    }

    let max = scores.iter().copied().fold(f64::NEG_INFINITY, f64::max);
    // `alpha == 0` gives `ln(alpha) == -inf`, hence `threshold == -inf`, and
    // `finite >= -inf` is true for every token. The vacuous case falls out of the
    // arithmetic instead of being special-cased.
    let threshold = max + alpha.ln();

    Ok(scores.iter().map(|&score| score >= threshold).collect())
}

/// Contrast, constrain, renormalize — the whole operation, in the order that
/// makes it safe.
///
/// Returns `(log p, V_valid)`. The order of the three steps is the method:
///
/// 1. **Combine** the two score vectors ([`contrast_scores`]). At this point the
///    scores are not a distribution and may be arbitrarily large, positive or
///    negative — that is what leaving the simplex means.
/// 2. **Mask** every token outside `V_valid` to `-inf`, where `V_valid` is read
///    off the **positive** distribution ([`adaptive_plausibility_mask`]) and
///    *not* off the combined scores. Reading it off the combined scores instead
///    is the single most natural bug available here, and it is self-defeating:
///    the combined scores are exactly where the implausible token has already
///    won, so the mask would keep it and discard everything else.
/// 3. **Renormalize** with `log_softmax`, which is finite because at least one
///    token always survives step 2.
///
/// The mask is applied *after* the combination rather than to the inputs. In
/// exact arithmetic the two are the same; in floating point, writing `-inf` into
/// an input and then multiplying it by a negative weight produces `+inf`, and by
/// a zero weight produces `NaN`. Combining finite numbers and then overwriting
/// the result meets neither.
///
/// # Errors
///
/// As [`contrast_scores`] and [`adaptive_plausibility_mask`].
pub fn masked_contrast_log_probs(
    positive: &[f64],
    negative: &[f64],
    positive_weight: f64,
    negative_weight: f64,
    plausibility_alpha: f64,
) -> ContextAwareResult<(Vec<f64>, Vec<bool>)> {
    let mut scores = contrast_scores(positive, negative, positive_weight, negative_weight)?;
    let plausible = adaptive_plausibility_mask(positive, plausibility_alpha)?;

    for (score, &keep) in scores.iter_mut().zip(&plausible) {
        if !keep {
            *score = f64::NEG_INFINITY;
        }
    }

    Ok((log_softmax(&scores), plausible))
}

// ── Diverge ──────────────────────────────────────────────────────────────────

/// The **Jensen–Shannon divergence** between two distributions given as
/// log-probabilities, in **nats**.
///
/// ```text
///   M    = (P + Q) / 2
///   JSD  = ½ · KL(P ‖ M)  +  ½ · KL(Q ‖ M)
/// ```
///
/// `DoLa` needs this and nothing else needs it, so this is the crate's only
/// implementation. The properties it relies on — and which this module's tests
/// check against hand-computed values — are the ones the `KL` alone does not
/// have:
///
/// - **Symmetric.** `JSD(P, Q) = JSD(Q, P)`, exactly (the two `KL` terms are
///   added, and `f64` addition is commutative). "Which layer diverges most from
///   the final one" would otherwise depend on which way round you asked.
/// - **Bounded.** `0 ≤ JSD ≤ ln 2`, with `ln 2` attained **iff** `P` and `Q` put
///   their mass on disjoint token sets. A `KL` between two layers can be `+inf`
///   — an early layer routinely assigns a token a probability that underflows —
///   and an `arg max` over a set containing `+inf` selects on a rounding
///   artefact. The bound is what makes the selection well-posed.
/// - **Zero iff equal.** A layer that has already converged on the final
///   distribution contributes nothing and is never selected.
///
/// # Computing it without ever taking `ln` of a probability
///
/// The mixture `M` is needed in log-space, and forming it as
/// `ln((exp(log_p) + exp(log_q)) / 2)` would exponentiate two possibly-underflowed
/// probabilities, add two zeros, and take the logarithm of `0.0`. Instead
///
/// ```text
///   log M(y)  =  logaddexp( log P(y), log Q(y) )  −  ln 2
/// ```
///
/// which is [`crate::replug::math::log_sum_exp`] over a two-element slice: stable
/// for any inputs, and correct in the limit — if both entries are `-inf` the
/// result is `-inf` rather than a `NaN`, which is the one place the naive form
/// blows up. The two `KL` terms are then
/// [`crate::replug::math::kl_divergence_from_log_probs`], which is finite here by
/// construction, since `M(y) ≥ P(y) / 2 > 0` wherever `P(y) > 0`.
///
/// # `-inf` is allowed here, unlike everywhere else in this module
///
/// A genuinely-zero probability is meaningful input to a divergence — it is
/// precisely the case that attains the `ln 2` bound — and the `0 · log 0 = 0`
/// convention inside the `KL` handles it. The contrast kernel, by contrast,
/// refuses `-inf`, because it would multiply it by a negative weight.
///
/// # Errors
///
/// Returns [`ContextAwareError::EmptyLogits`] for empty inputs,
/// [`ContextAwareError::VocabSizeMismatch`] when the two differ in length,
/// [`ContextAwareError::NonFiniteScore`] for a `NaN` or `+inf` entry, and
/// [`ContextAwareError::UnnormalizedDistribution`] when either input is not a
/// normalized log-probability vector — which is what stops raw logits from being
/// silently accepted and turned into a plausible number that is the divergence of
/// nothing.
pub fn jensen_shannon_divergence_from_log_probs(
    log_p: &[f64],
    log_q: &[f64],
) -> ContextAwareResult<f64> {
    require_log_probabilities(log_p, "first")?;
    require_log_probabilities(log_q, "second")?;
    require_same_length(log_p, log_q, "second distribution")?;

    let log_m: Vec<f64> = log_p
        .iter()
        .zip(log_q)
        .map(|(&lp, &lq)| log_sum_exp(&[lp, lq]) - LN_2)
        .collect();

    let kl_p = kl_divergence_from_log_probs(log_p, &log_m)?;
    let kl_q = kl_divergence_from_log_probs(log_q, &log_m)?;

    Ok(0.5 * (kl_p + kl_q))
}

/// Reject anything that is not a normalized log-probability vector.
///
/// `-inf` entries are permitted (a token of probability exactly zero); `NaN` and
/// `+inf` are not; and the total mass must be `1`, checked as
/// `|logsumexp(v)| < NORMALIZATION_TOLERANCE`.
fn require_log_probabilities(log_probs: &[f64], origin: &str) -> ContextAwareResult<()> {
    if log_probs.is_empty() {
        return Err(ContextAwareError::EmptyLogits);
    }
    for (index, &value) in log_probs.iter().enumerate() {
        if value.is_nan() || (value.is_infinite() && value.is_sign_positive()) {
            return Err(ContextAwareError::NonFiniteScore {
                index,
                origin: origin.to_string(),
                value,
            });
        }
    }

    let log_total_mass = log_sum_exp(log_probs);
    if !log_total_mass.is_finite() || log_total_mass.abs() >= NORMALIZATION_TOLERANCE {
        return Err(ContextAwareError::UnnormalizedDistribution { log_total_mass });
    }
    Ok(())
}
