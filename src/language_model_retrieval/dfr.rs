//! Divergence From Randomness: the `PL2` and `DPH` weighting models.
//!
//! # The `DFR` idea
//!
//! Query likelihood ([`super::smoothing`]) is *generative*: it asks how
//! probable the query is under the document's language model. Divergence From
//! Randomness (Amati & van Rijsbergen, 2002) asks a different question
//! entirely, and answers it with two multiplied factors:
//!
//! ```text
//! score_w(D) = qtf(w) · Inf₁ · Inf₂
//!            = qtf(w) · (-log₂ Prob₁(tf | randomness)) · (1 - Prob₂(tf | eliteness))
//! ```
//!
//! - **`Inf₁`, the informative content.** Posit a *randomness model* under
//!   which the term's occurrences are sprinkled over the collection with no
//!   regard for topic. The less probable the observed `tf` is under that
//!   model, the more surprising it is, and the more information it carries
//!   about the document. `Inf₁` is exactly that surprisal, `-log₂ Prob₁`.
//! - **`Inf₂`, the after-effect (or "risk") factor.** Not all of the surprise
//!   should be believed. If a term is genuinely *elite* in a document — the
//!   document is *about* it — then seeing it once makes seeing it again
//!   likely, so each additional occurrence adds less evidence than the last.
//!   `Inf₂` discounts `Inf₁` by the probability that the next occurrence would
//!   have been observed anyway.
//!
//! The product is the weight. High weight means: *this frequency is very
//! unlikely to be an accident, and the unlikeliness is not merely an artifact
//! of the term being bursty.*
//!
//! # `PL2` = **P**oisson + **L**aplace after-effect + normalization **2**
//!
//! **Randomness model (P).** Under a Poisson process with rate
//! `λ_w = cf(w) / N` (`w`'s mean number of occurrences per document over the
//! `N` documents of the collection):
//!
//! ```text
//! Prob₁(tf) = e^(-λ) · λ^tf / tf!
//! ```
//!
//! **Normalization (2).** Raw `tf` is not comparable across documents of
//! different lengths, so `PL2` replaces it with the *normalized* frequency
//! that a document of average length would have exhibited at the same density:
//!
//! ```text
//! tfn = tf · log₂(1 + c · avgdl / dl)
//! ```
//!
//! `c > 0` controls the aggressiveness. As `dl → ∞` the factor tends to `0`
//! (an occurrence in an enormous document is worth almost nothing); at
//! `dl = c · avgdl` it is exactly `1`.
//!
//! **Informative content.** Take `Inf₁ = -log₂ Prob₁(tfn)` and expand:
//!
//! ```text
//! -log₂ Prob₁(tfn) = λ·log₂(e) - tfn·log₂(λ) + log₂(tfn!)
//! ```
//!
//! `tfn` is a real number, so `tfn!` must be approximated. Stirling's formula
//! `n! ≈ n^n · e^(-n) · √(2πn)` gives
//! `log₂(tfn!) ≈ tfn·log₂(tfn) - tfn·log₂(e) + ½·log₂(2π·tfn)`, hence
//!
//! ```text
//! Inf₁ = tfn·log₂(tfn / λ) + (λ - tfn)·log₂(e) + ½·log₂(2π·tfn)
//! ```
//!
//! **After-effect (L).** Laplace's law of succession estimates the probability
//! of *one more* occurrence, given `tfn` so far, as `tfn / (tfn + 1)`; so
//! `Inf₂ = 1 - tfn/(tfn + 1) = 1 / (tfn + 1)`.
//!
//! **The model.**
//!
//! ```text
//! score_w = qtf · ( tfn·log₂(tfn/λ) + (λ - tfn)·log₂(e) + ½·log₂(2π·tfn) ) / (tfn + 1)
//! ```
//!
//! ## The `tfn → 0` singularity
//!
//! This formula is *not* finite at `tfn = 0`. Two of its three terms blow up:
//! `tfn·log₂(tfn/λ)` is a `0 · (-∞)` indeterminate form (it does converge, to
//! `0`, since `x·log x → 0`), but `½·log₂(2π·tfn) → -∞` outright, and nothing
//! cancels it. Evaluated naively in floating point the formula yields `NaN`
//! and `-inf`. Two distinct cases reach it, and they need different answers.
//!
//! **`tf = 0` — the term is absent from the document.** The contribution is
//! defined to be exactly `0`, and this module does not evaluate the formula at
//! all. It is worth being precise about *why*, because the mathematics alone
//! suggests otherwise: the *exact* (non-Stirling) surprisal at `tf = 0` is
//! `-log₂(e^(-λ)) = λ·log₂(e)`, a perfectly finite positive number — the `-∞`
//! is purely an artifact of Stirling's formula, which approximates `0!` as
//! `√(2π·0) = 0` instead of `1`. But awarding every document a bonus of
//! `λ·log₂(e)` for a term it does **not** contain would be actively harmful:
//! `λ` grows with `cf(w)`, so the bonus would be *largest for the commonest
//! terms*, and a document containing none of the query's terms could outrank
//! one that contains them all. `DFR` is a model of *evidence of eliteness*,
//! and a term that never occurs supplies none. Zero it is — and this is what
//! every `DFR` implementation does, by scoring only the query terms' posting
//! lists.
//!
//! **`0 < tfn < ε` — the term is present, but normalized nearly out of
//! existence.** Here the document *is* a candidate, and it genuinely has to be
//! scored, but Stirling's series has broken down. `tfn` is floored at
//! [`PL2_MIN_TFN`] (`1e-9`), which bounds the surrogate's error rather than
//! emitting `-inf`. The resulting score, `≈ 1.44·λ - 13.6`, is a large finite
//! penalty — the correct qualitative verdict for a term whose normalized
//! frequency is indistinguishable from zero. Reaching this branch requires
//! `dl > 1.4e9 · c · avgdl`, so no real corpus ever does; it is a guard
//! against `c → 0` misconfiguration and adversarial input, not a modelling
//! decision.
//!
//! Note also that `PL2` scores can legitimately be **negative** (e.g.
//! `tfn = 0.01`, `λ = 0.1` gives `≈ -1.88`), because `½·log₂(2π·tfn)` is
//! negative for `tfn < 1/(2π)`. That is the model, not a bug, and this module
//! does not clamp it: clamping would flatten genuine distinctions among weakly
//! matching documents.
//!
//! # `DPH` — hyper-geometric, parameter-free
//!
//! `PL2`'s `c` has to be tuned. `DPH` (Amati, Ambrosi, Bianchi, Gaibisso &
//! Gambosi, 2007) removes it by deriving the normalization from the
//! within-document *relative* frequency `f = tf / dl` itself:
//!
//! ```text
//! score_w = qtf · ((1 - f)² / (tf + 1)) · ( tf·log₂( f · N / cf ) + ½·log₂( 2π·tf·(1 - f) ) )
//! ```
//!
//! The bracket is again a surprisal: `f · N / cf` is the ratio of the observed
//! within-document density to the density a random document would show, so its
//! log is an observed-versus-expected divergence, and `½·log₂(2π·tf·(1-f))` is
//! the Stirling correction of the corresponding binomial. The prefactor
//! `(1 - f)² / (tf + 1)` is the after-effect: `1/(tf + 1)` is Laplace again,
//! and `(1 - f)²` is Popper's normalized information gain, which vanishes as
//! the document becomes *nothing but* the term.
//!
//! ## The `tf = dl` singularity
//!
//! When a document consists solely of repetitions of the query term,
//! `f = 1`, and the formula is again indeterminate: the prefactor
//! `(1 - f)² = 0` multiplies a bracket containing
//! `½·log₂(2π·tf·(1 - f)) = ½·log₂(0) = -∞`, i.e. `0 · (-∞)`. Naive
//! evaluation gives `NaN`.
//!
//! The limit exists and is exactly `0`. Write `u = 1 - f → 0⁺`; the product
//! behaves as
//!
//! ```text
//! (u² / (tf+1)) · ( A + ½·log₂(2π·tf·u) )  =  (u²·A + ½·u²·log₂(2π·tf) + ½·u²·log₂ u) / (tf+1)
//! ```
//!
//! and `u² · log u → 0` as `u → 0⁺` (the quadratic beats the logarithm), so
//! every summand vanishes. This module returns `0` for `tf = dl`, which is the
//! true limit and not a patch. It also happens to be the right *retrieval*
//! behaviour: a document that is 100% the query term carries no discriminating
//! information about that term at all — Popper's factor says the observation
//! was a foregone conclusion.
//!
//! `tf = 0` is handled exactly as in `PL2`: contribution `0`, formula never
//! evaluated.
//!
//! ## A note on the `DPH` variant implemented here
//!
//! Terrier's reference `DPH` uses `(tf · avgdl / dl) · (|C| / cf)` inside the
//! logarithm, i.e. it compares a length-normalized `tf` against the collection
//! *token* probability. The formula implemented here compares the raw relative
//! frequency `tf / dl` against the per-document Poisson rate `cf / N` — the
//! same observed-over-expected ratio, expressed with the same statistics `PL2`
//! uses. Both are `DFR` models in good standing; this one is what
//! [`DfrModel::Dph`] computes, exactly and without deviation.

use std::f64::consts::{LOG2_E, TAU};

use super::types::{DfrModel, LmRetrievalError, PL2_MIN_TFN};

impl DfrModel {
    /// Validate the model's hyper-parameter.
    ///
    /// # Errors
    ///
    /// Returns [`LmRetrievalError::InvalidParameter`] if `PL2`'s `c` is not
    /// finite and strictly positive. [`DfrModel::Dph`] is parameter-free and
    /// always valid.
    pub fn validate(&self) -> Result<(), LmRetrievalError> {
        match *self {
            Self::Pl2 { c } => {
                if !c.is_finite() || c <= 0.0 {
                    return Err(LmRetrievalError::InvalidParameter {
                        parameter: "c".to_string(),
                        value: c,
                        reason: "the PL2 length-normalization constant must be finite and \
                                 strictly positive"
                            .to_string(),
                    });
                }
                Ok(())
            }
            Self::Dph => Ok(()),
        }
    }

    /// The model's contribution for a single query term.
    ///
    /// Returns `0.0` whenever the term does not occur in the document, and is
    /// finite for every input — see the module documentation for the two
    /// singularities and how each is resolved.
    #[must_use]
    pub fn term_score(
        &self,
        query_weight: f64,
        term_frequency: u64,
        doc_length: u64,
        avg_doc_length: f64,
        collection_frequency: u64,
        num_documents: u64,
    ) -> f64 {
        match *self {
            Self::Pl2 { c } => pl2_term_score(
                query_weight,
                term_frequency,
                doc_length,
                avg_doc_length,
                collection_frequency,
                num_documents,
                c,
            ),
            Self::Dph => dph_term_score(
                query_weight,
                term_frequency,
                doc_length,
                collection_frequency,
                num_documents,
            ),
        }
    }

    /// `PL2`'s normalized term frequency
    /// `tfn = tf · log₂(1 + c · avgdl / dl)`, before flooring.
    ///
    /// Returns `0.0` for [`DfrModel::Dph`], which has no such quantity, and
    /// for a zero-length document.
    #[must_use]
    #[allow(clippy::cast_precision_loss)]
    pub fn normalized_term_frequency(
        &self,
        term_frequency: u64,
        doc_length: u64,
        avg_doc_length: f64,
    ) -> f64 {
        match *self {
            Self::Pl2 { c } => {
                if doc_length == 0 || !avg_doc_length.is_finite() || avg_doc_length <= 0.0 {
                    return 0.0;
                }
                let tf = term_frequency as f64;
                let dl = doc_length as f64;
                tf * (c * avg_doc_length / dl).ln_1p() / std::f64::consts::LN_2
            }
            Self::Dph => 0.0,
        }
    }
}

/// `PL2`: Poisson randomness, Laplace after-effect, normalization 2.
///
/// See the module documentation for the derivation and for the treatment of
/// the `tfn → 0` singularity.
#[must_use]
#[allow(clippy::cast_precision_loss)]
pub fn pl2_term_score(
    query_weight: f64,
    term_frequency: u64,
    doc_length: u64,
    avg_doc_length: f64,
    collection_frequency: u64,
    num_documents: u64,
    c: f64,
) -> f64 {
    // A term the document does not contain contributes nothing: DFR weighs
    // *observed* divergence, and there is nothing to observe. (The remaining
    // guards are unreachable given tf > 0 -- tf > 0 implies dl > 0, cf > 0 and
    // N > 0 -- but they are cheap, and they keep the function total.)
    if term_frequency == 0
        || doc_length == 0
        || collection_frequency == 0
        || num_documents == 0
        || !avg_doc_length.is_finite()
        || avg_doc_length <= 0.0
        || !c.is_finite()
        || c <= 0.0
    {
        return 0.0;
    }

    let lambda = collection_frequency as f64 / num_documents as f64;
    if !lambda.is_finite() || lambda <= 0.0 {
        return 0.0;
    }

    let tf = term_frequency as f64;
    let dl = doc_length as f64;

    // tfn = tf * log2(1 + c * avgdl / dl). `ln_1p` keeps the small-argument
    // case (dl >> c * avgdl) accurate: log(1 + x) computed as ln(1.0 + x)
    // loses every significant digit once x drops below the f64 epsilon.
    let tfn_raw = tf * (c * avg_doc_length / dl).ln_1p() / std::f64::consts::LN_2;
    if !tfn_raw.is_finite() {
        return 0.0;
    }
    // Stirling's approximation of log2(tfn!) diverges to -inf as tfn -> 0,
    // although the exact value converges to log2(0!) = 0. Floor tfn rather
    // than propagate the artifact.
    let tfn = tfn_raw.max(PL2_MIN_TFN);

    let surprisal = (lambda - tfn).mul_add(
        LOG2_E,
        tfn.mul_add((tfn / lambda).log2(), 0.5 * (TAU * tfn).log2()),
    );

    query_weight * surprisal / (tfn + 1.0)
}

/// `DPH`: the parameter-free hyper-geometric `DFR` model.
///
/// See the module documentation for the derivation and for the proof that the
/// `tf = dl` limit is exactly zero.
#[must_use]
#[allow(clippy::cast_precision_loss)]
pub fn dph_term_score(
    query_weight: f64,
    term_frequency: u64,
    doc_length: u64,
    collection_frequency: u64,
    num_documents: u64,
) -> f64 {
    // Term absent: no observed divergence, no evidence, no contribution.
    if term_frequency == 0
        || doc_length == 0
        || collection_frequency == 0
        || num_documents == 0
        || term_frequency > doc_length
    {
        return 0.0;
    }

    // (1 - tf/dl) is computed from the integer difference so that it stays
    // exactly zero when tf == dl (rather than an epsilon of either sign) and
    // stays representable for very long documents.
    let tf = term_frequency as f64;
    let dl = doc_length as f64;
    let one_minus_f = (doc_length - term_frequency) as f64 / dl;

    // tf == dl: the document is nothing but this term. The prefactor
    // (1 - f)^2 vanishes quadratically while the bracket's Stirling term
    // diverges only logarithmically, so the product's limit is exactly 0.
    if one_minus_f <= 0.0 {
        return 0.0;
    }

    let f = tf / dl;
    let expected_ratio = f * (num_documents as f64 / collection_frequency as f64);
    let after_effect = one_minus_f * one_minus_f / (tf + 1.0);
    let surprisal = tf.mul_add(expected_ratio.log2(), 0.5 * (TAU * tf * one_minus_f).log2());

    query_weight * after_effect * surprisal
}
