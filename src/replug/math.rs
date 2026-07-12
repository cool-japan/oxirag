//! Numerically-stable probability primitives for the `REPLUG` ensemble.
//!
//! Every routine in this file is written against one adversary: **underflow**.
//! A next-token distribution over a real vocabulary routinely contains
//! probabilities below `1e-300`, and the `REPLUG` objective needs their
//! *logarithms*. The rule this file follows without exception is:
//!
//! > Never take the logarithm of a probability. Carry log-probabilities from
//! > the logits onward, and only exponentiate at the very last moment (if at
//! > all).
//!
//! # Why `f64` here when the model hands us `f32` logits
//!
//! A language model emits `f32` logits — that is what the hardware produces
//! and there is no information below that precision to recover. But the
//! *operations* this module performs on them are not the model's operations:
//! we sum `k` mixture components across a vocabulary, take logarithms of the
//! result, and then form a Kullback–Leibler divergence, which is a difference
//! of two nearly-equal logarithms and is therefore catastrophically
//! cancellation-prone. `f32` has ~7 decimal digits; a `KL` of `1e-8` between
//! two nearly-identical retrieval distributions is exactly the regime `REPLUG`
//! -`LSR` converges into, and in `f32` it is pure rounding noise. So logits
//! are promoted to `f64` at the trait boundary ([`promote_logits`]) and all
//! distribution-level arithmetic happens in `f64`.
//!
//! The underflow thresholds make the same point concretely. `exp(-88)` is
//! already `0` in `f32`; in `f64` you have room down to `exp(-745)`. Log-space
//! working *plus* `f64` is what keeps a logit spread of `±1e4` (see the
//! stability tests) from collapsing to `NaN`.

use super::types::{ReplugError, ReplugResult};

// ── Promotion ────────────────────────────────────────────────────────────────

/// Promote model-native `f32` logits to `f64`, rejecting any non-finite entry.
///
/// A `NaN` or infinite logit is *not* something to silently paper over: it
/// means the model produced garbage, and every downstream quantity (softmax,
/// mixture, `KL`) would inherit the poison while still looking like a number.
/// This is the one place we check, so that nothing after it has to.
///
/// # Errors
///
/// Returns [`ReplugError::EmptyLogits`] when `logits` is empty, and
/// [`ReplugError::NonFiniteLogit`] when any entry is `NaN` or infinite.
pub fn promote_logits(logits: &[f32]) -> ReplugResult<Vec<f64>> {
    if logits.is_empty() {
        return Err(ReplugError::EmptyLogits);
    }
    let mut out = Vec::with_capacity(logits.len());
    for (index, &logit) in logits.iter().enumerate() {
        if !logit.is_finite() {
            return Err(ReplugError::NonFiniteLogit {
                index,
                value: f64::from(logit),
            });
        }
        out.push(f64::from(logit));
    }
    Ok(out)
}

// ── log-sum-exp ──────────────────────────────────────────────────────────────

/// The stable `log Σ_i exp(x_i)`.
///
/// Computed as `m + log Σ_i exp(x_i − m)` with `m = max_i x_i`, so the largest
/// summand is exactly `exp(0) = 1` and cannot overflow, while summands far
/// below the max underflow harmlessly to `0` (they contribute nothing to the
/// sum anyway — that is the point of the shift, not a defect of it).
///
/// `-inf` entries are permitted and contribute exactly `0`, which is the
/// correct limit: a component with zero probability adds zero mass. If *every*
/// entry is `-inf` the result is `-inf` (log of an empty sum), and the shift is
/// skipped so that `-inf − -inf = NaN` never arises.
///
/// An empty slice yields `-inf` by the same convention (`Σ over ∅ = 0`).
#[must_use]
pub fn log_sum_exp(values: &[f64]) -> f64 {
    let max = values.iter().copied().fold(f64::NEG_INFINITY, f64::max);

    // Guard the `-inf - -inf = NaN` case: if the maximum is already `-inf`
    // then every entry is `-inf` (or the slice is empty) and the answer is
    // `-inf` without any shifting.
    if max == f64::NEG_INFINITY {
        return f64::NEG_INFINITY;
    }

    let sum: f64 = values.iter().map(|&value| (value - max).exp()).sum();
    max + sum.ln()
}

// ── softmax / log-softmax ────────────────────────────────────────────────────

/// Stable `log softmax(x)`: `x_i − logsumexp(x)`.
///
/// This — not [`softmax`] — is the canonical entry point for everything in
/// this module. Its output is finite for any finite input, *including* inputs
/// whose softmax underflows to exactly `0`. For logits `[0, -20_000]` the
/// softmax is `[1, 0]` and `log(0) = -inf` destroys the mixture; the log-
/// softmax is `[0, -20_000]`, which is perfectly informative.
#[must_use]
pub fn log_softmax(logits: &[f64]) -> Vec<f64> {
    let normalizer = log_sum_exp(logits);
    if !normalizer.is_finite() {
        // Every logit was `-inf`. There is no distribution to speak of; the
        // caller's validation should have rejected this, but returning `-inf`
        // is at least the honest answer rather than a fabricated uniform.
        return vec![f64::NEG_INFINITY; logits.len()];
    }
    logits.iter().map(|&logit| logit - normalizer).collect()
}

/// Stable `softmax(x)`, obtained by exponentiating [`log_softmax`].
///
/// Going through log-space costs one extra pass and buys exactness: the result
/// provably sums to `1` up to rounding, because every element is
/// `exp(x_i − logsumexp(x))` computed from the *same* normalizer.
#[must_use]
pub fn softmax(logits: &[f64]) -> Vec<f64> {
    log_softmax(logits)
        .into_iter()
        .map(f64::exp)
        .collect::<Vec<f64>>()
}

/// Temperature-scaled `log softmax`: `log softmax(x / τ)`, computed **without
/// ever forming `x / τ`**.
///
/// This is the workhorse behind the `REPLUG` document weights
/// `λ(d_i | q) = softmax(s_i / τ)` and it is where a naive implementation
/// breaks. The obvious code is "divide by `τ`, then softmax", and it overflows:
/// with `τ = 1e-40` and a score of `0.9`, `0.9 / τ` is `+inf` in `f32` (and
/// `9e39` is already near the edge in the promoted `f64`), so the softmax sees
/// `inf − inf = NaN` and the whole ensemble is poisoned by a *configuration*
/// value, not by any data.
///
/// The fix is to shift **before** dividing. Because softmax is invariant to a
/// constant shift of its argument,
///
/// ```text
///   softmax(x / τ)_i  =  softmax((x − max(x)) / τ)_i
/// ```
///
/// is an algebraic identity, and the right-hand side has every argument `≤ 0`,
/// so `exp` of it is in `(0, 1]` — bounded, always. The two temperature limits
/// then fall out for free and *exactly*:
///
/// - **`τ → 0⁺`**: for `x_i < max(x)`, `(x_i − max) / τ → −inf` and
///   `exp → 0`; for the maximizer, `0 / τ = 0` and `exp → 1`. The result is
///   the arg-max indicator (uniform over ties). No overflow occurs on the way
///   there, which is precisely what dividing first would not give you.
/// - **`τ → ∞`**: `(x_i − max) / τ → 0` for every `i`, so every weight is `1`
///   before normalization and the result is exactly uniform. `τ = f64::INFINITY`
///   is therefore a *usable* value here rather than a `NaN` generator.
///
/// # Errors
///
/// Returns [`ReplugError::InvalidTemperature`] when `temperature` is not
/// strictly positive or is `NaN`. `τ = 0` would compute `0 / 0 = NaN` for the
/// maximizing element, so it is rejected rather than approximated; use a small
/// positive `τ` to approach the arg-max limit.
/// Returns [`ReplugError::EmptyLogits`] when `values` is empty.
pub fn temperature_log_softmax(values: &[f64], temperature: f64) -> ReplugResult<Vec<f64>> {
    if values.is_empty() {
        return Err(ReplugError::EmptyLogits);
    }
    // `NaN` fails every comparison, so it must be rejected explicitly rather
    // than relying on `<= 0.0`. Note that `temperature = +inf` is deliberately
    // *allowed*: it is the exact uniform limit, and the shift-before-divide
    // below evaluates it as `finite / inf = 0` without any `NaN`.
    if temperature.is_nan() || temperature <= 0.0 {
        return Err(ReplugError::InvalidTemperature { temperature });
    }

    let max = values.iter().copied().fold(f64::NEG_INFINITY, f64::max);
    if !max.is_finite() {
        return Err(ReplugError::NonFiniteLogit {
            index: 0,
            value: max,
        });
    }

    // Shift first, divide second. `scaled_i <= 0` for every `i`.
    let scaled: Vec<f64> = values
        .iter()
        .map(|&value| (value - max) / temperature)
        .collect();

    Ok(log_softmax(&scaled))
}

/// Temperature-scaled `softmax(x / τ)`; the exponentiated [`temperature_log_softmax`].
///
/// # Errors
///
/// Propagates the errors of [`temperature_log_softmax`].
pub fn temperature_softmax(values: &[f64], temperature: f64) -> ReplugResult<Vec<f64>> {
    Ok(temperature_log_softmax(values, temperature)?
        .into_iter()
        .map(f64::exp)
        .collect())
}

// ── The REPLUG mixture ───────────────────────────────────────────────────────

/// The `REPLUG` ensemble: the **λ-weighted arithmetic mixture** of the
/// per-document next-token distributions, in log-space.
///
/// For every candidate token `y`,
///
/// ```text
///   p(y | q)  =  Σ_i  λ_i · p(y | d_i ⊕ q)
/// ```
///
/// and this function returns `log p(y | q)` for every `y`, computed as
///
/// ```text
///   log p(y | q)  =  logsumexp_i ( log λ_i  +  log p(y | d_i ⊕ q) )
/// ```
///
/// # Why the log-space form is not optional
///
/// The mixture is a sum, and one is tempted to just *do the sum*: exponentiate
/// each document's log-probabilities and accumulate `λ_i · p_i(y)`. That is
/// correct in exact arithmetic and wrong in floating point, because the mixture
/// is only the first half of the job — `REPLUG` then needs `log p(y | q)` (to
/// score a token, to accumulate a sequence log-likelihood, to compute an
/// entropy, to feed `LSR`). If any `p_i(y)` underflows to exactly `0`, the
/// probability-space sum happily returns `0.0` and the subsequent `ln` returns
/// `-inf`: an infinitely-confident claim that the token is impossible, derived
/// from nothing but a rounding step. In log-space, the same token carries a
/// finite `log p(y | q) ≈ -20_000` and every downstream quantity stays finite.
///
/// The `logsumexp` is taken **over the `k` documents**, not over the vocabulary
/// — the vocabulary axis is already normalized inside each `log_probs[i]`.
///
/// # Guarantees
///
/// The result is a proper log-probability vector: `Σ_y exp(result[y]) = 1` up
/// to rounding, because `Σ_y Σ_i λ_i p_i(y) = Σ_i λ_i · 1 = 1` whenever
/// `Σ_i λ_i = 1` (which [`temperature_log_softmax`] guarantees for λ).
///
/// # Errors
///
/// Returns [`ReplugError::NoDocuments`] when `log_weights` is empty,
/// [`ReplugError::WeightCountMismatch`] when the number of weights and the
/// number of per-document distributions disagree, and
/// [`ReplugError::VocabSizeMismatch`] when the per-document distributions are
/// not all the same length.
pub fn mixture_log_probs(log_weights: &[f64], log_probs: &[Vec<f64>]) -> ReplugResult<Vec<f64>> {
    if log_weights.is_empty() || log_probs.is_empty() {
        return Err(ReplugError::NoDocuments);
    }
    if log_weights.len() != log_probs.len() {
        return Err(ReplugError::WeightCountMismatch {
            weights: log_weights.len(),
            documents: log_probs.len(),
        });
    }

    let vocab_size = log_probs[0].len();
    if vocab_size == 0 {
        return Err(ReplugError::EmptyLogits);
    }
    for (index, per_doc) in log_probs.iter().enumerate() {
        if per_doc.len() != vocab_size {
            return Err(ReplugError::VocabSizeMismatch {
                expected: vocab_size,
                actual: per_doc.len(),
                document_index: index,
            });
        }
    }

    // Scratch buffer reused across the vocabulary so the mixture is one
    // allocation, not one per token.
    let mut terms = vec![0.0_f64; log_weights.len()];
    let mut out = Vec::with_capacity(vocab_size);

    for token in 0..vocab_size {
        for (slot, (&log_weight, per_doc)) in
            terms.iter_mut().zip(log_weights.iter().zip(log_probs))
        {
            // `log λ_i + log p_i(y)`. A document with λ_i == 0 has
            // `log λ_i == -inf`, and `-inf + finite == -inf`, which
            // `log_sum_exp` then treats as a zero-mass component. That is the
            // `0 · log 0 = 0` convention, obtained for free rather than
            // special-cased.
            *slot = log_weight + per_doc[token];
        }
        out.push(log_sum_exp(&terms));
    }

    Ok(out)
}

/// The **log-linear (geometric) pool** — `softmax` of the λ-weighted average of
/// the *logits*. This is **not** the `REPLUG` ensemble; it is here so that the
/// difference can be measured rather than merely asserted.
///
/// # The identity that makes this a different operation
///
/// Averaging logits and re-normalizing is often mistaken for "the same thing"
/// as mixing distributions. It is not. Write `z_i` for document `i`'s logits,
/// so `p_i = softmax(z_i)`, i.e. `exp(z_i(y)) = p_i(y) · Z_i` for a
/// `y`-independent normalizer `Z_i`. Then
///
/// ```text
///   softmax( Σ_i λ_i z_i )(y)  ∝  exp( Σ_i λ_i z_i(y) )
///                              =  Π_i exp( z_i(y) )^{λ_i}
///                              =  Π_i ( p_i(y) · Z_i )^{λ_i}
///                              =  ( Π_i p_i(y)^{λ_i} ) · Π_i Z_i^{λ_i}
///                              ∝  Π_i p_i(y)^{λ_i}
/// ```
///
/// because `Π_i Z_i^{λ_i}` does not depend on `y` and is absorbed by the
/// normalization. So averaging logits computes the **normalized weighted
/// geometric mean** of the per-document distributions (a *logarithmic opinion
/// pool*, a.k.a. product-of-experts), whereas `REPLUG` computes their
/// **weighted arithmetic mean** (a *linear opinion pool*).
///
/// # Why the distinction is semantic, not cosmetic
///
/// The two pools implement opposite logical connectives over the documents:
///
/// - The **arithmetic** mixture is an **OR**. Its value is bounded below by
///   `λ_i · p_i(y)` for every `i`, so *one* document that strongly supports a
///   token drags the ensemble toward it even if every other document has never
///   heard of it. No document can veto.
/// - The **geometric** pool is an **AND**. It contains the factor `p_i(y)^{λ_i}`,
///   so a single document assigning `p_i(y) ≈ 0` drives the pooled value to
///   `≈ 0` regardless of how enthusiastic the others are. Every document holds
///   a veto.
///
/// `REPLUG` retrieves `k` documents *independently*; most of them are expected
/// to be irrelevant to any given token. A pooling rule in which an irrelevant
/// document can veto the token that the one relevant document is certain about
/// is not a defensible reading of "ensemble the retrieved evidence" — which is
/// exactly why `REPLUG` specifies the arithmetic mixture, and why implementing
/// it as a logit average is a real (and quiet) bug.
///
/// # Errors
///
/// Same validation as [`mixture_log_probs`].
pub fn log_linear_pool_log_probs(weights: &[f64], logits: &[Vec<f64>]) -> ReplugResult<Vec<f64>> {
    if weights.is_empty() || logits.is_empty() {
        return Err(ReplugError::NoDocuments);
    }
    if weights.len() != logits.len() {
        return Err(ReplugError::WeightCountMismatch {
            weights: weights.len(),
            documents: logits.len(),
        });
    }

    let vocab_size = logits[0].len();
    if vocab_size == 0 {
        return Err(ReplugError::EmptyLogits);
    }
    for (index, per_doc) in logits.iter().enumerate() {
        if per_doc.len() != vocab_size {
            return Err(ReplugError::VocabSizeMismatch {
                expected: vocab_size,
                actual: per_doc.len(),
                document_index: index,
            });
        }
    }

    let averaged: Vec<f64> = (0..vocab_size)
        .map(|token| {
            weights
                .iter()
                .zip(logits)
                .map(|(&weight, per_doc)| weight * per_doc[token])
                .sum::<f64>()
        })
        .collect();

    Ok(log_softmax(&averaged))
}

// ── Kullback–Leibler divergence ──────────────────────────────────────────────

/// `KL(P ‖ Q) = Σ_i p_i · (log p_i − log q_i)`, taken over **log-probabilities**.
///
/// The signature is the whole point: this function refuses to accept
/// probabilities, because the caller would have had to compute `log p_i` by
/// calling `.ln()` on them, and `ln(0.0) = -inf` is how every naive `KL`
/// implementation acquires its `NaN`s. Feed it the output of [`log_softmax`],
/// which is finite by construction, and the divergence is finite too.
///
/// # The `0 · log 0 = 0` convention
///
/// The summand is `p_i · (log p_i − log q_i)`. Where `p_i = 0` the measure-
/// theoretic convention sets the contribution to `0` (the limit of
/// `x · log x` as `x → 0⁺` is `0`), and here that is a *load-bearing* guard,
/// not pedantry: when `log_p[i]` is a large negative number, `exp(log_p[i])`
/// underflows to exactly `0.0`, and if `log_q[i]` happens to be `-inf` the
/// product is `0.0 · inf = NaN`. The explicit `p_i == 0 → skip` below is what
/// stops that. (`p_i > 0` with `q_i = 0` is the genuinely infinite case — `P`
/// puts mass where `Q` has none — and is correctly reported as `+inf` rather
/// than clamped.)
///
/// # Non-negativity
///
/// `KL ≥ 0` (Gibbs' inequality), with equality **iff** `P = Q`. Floating-point
/// summation can push the computed value a hair below zero when `P ≈ Q`; the
/// result is clamped at `0` only in that rounding-sized neighbourhood, and a
/// genuinely negative value (which would signal that the inputs were not
/// normalized) is left alone so that it can be caught rather than hidden.
///
/// # Errors
///
/// Returns [`ReplugError::VocabSizeMismatch`] when the two log-probability
/// vectors differ in length, and [`ReplugError::EmptyLogits`] when they are
/// empty.
pub fn kl_divergence_from_log_probs(log_p: &[f64], log_q: &[f64]) -> ReplugResult<f64> {
    if log_p.is_empty() || log_q.is_empty() {
        return Err(ReplugError::EmptyLogits);
    }
    if log_p.len() != log_q.len() {
        return Err(ReplugError::VocabSizeMismatch {
            expected: log_p.len(),
            actual: log_q.len(),
            document_index: 0,
        });
    }

    let mut total = 0.0_f64;
    for (&lp, &lq) in log_p.iter().zip(log_q) {
        let p = lp.exp();
        // `0 · log 0 = 0`. Also catches an underflowed `p` whose partner
        // `lq` is `-inf`, which would otherwise be `0.0 * inf = NaN`.
        if p == 0.0 {
            continue;
        }
        if lq == f64::NEG_INFINITY {
            // `P` has mass where `Q` has none: the divergence is genuinely
            // infinite. Report it rather than silently clamping.
            return Ok(f64::INFINITY);
        }
        total += p * (lp - lq);
    }

    // Clamp only rounding-sized negatives (see the doc comment above).
    if total < 0.0 && total > -1e-9 {
        total = 0.0;
    }
    Ok(total)
}

/// Shannon entropy `H(p) = −Σ_i p_i log p_i` in **nats**, from log-probabilities.
///
/// Same discipline as [`kl_divergence_from_log_probs`]: the input is
/// `log_softmax` output, and `p_i = 0` contributes exactly `0`.
#[must_use]
pub fn entropy_from_log_probs(log_p: &[f64]) -> f64 {
    let mut total = 0.0_f64;
    for &lp in log_p {
        let p = lp.exp();
        if p == 0.0 {
            continue;
        }
        total -= p * lp;
    }
    total.max(0.0)
}

// ── Selection ────────────────────────────────────────────────────────────────

/// The index of the largest entry, ties broken by the **lowest index**.
///
/// Uses [`f64::total_cmp`] rather than `partial_cmp`, so it is a total order:
/// no `unwrap` on an `Option<Ordering>` and no undefined behaviour if a `NaN`
/// slips through (a `NaN` would sort above `+inf` under `total_cmp` and be
/// selected, which is a *visible* wrong answer rather than a panic — the
/// validation in [`promote_logits`] is what actually prevents it).
///
/// Returns `None` for an empty slice.
#[must_use]
pub fn arg_max(values: &[f64]) -> Option<usize> {
    values
        .iter()
        .enumerate()
        .max_by(|(left_index, left), (right_index, right)| {
            left.total_cmp(right)
                // `max_by` returns the *last* maximum on a tie; reversing the
                // index comparison makes it return the first.
                .then_with(|| right_index.cmp(left_index))
        })
        .map(|(index, _)| index)
}

// ── Deterministic PRNG ───────────────────────────────────────────────────────

/// `SplitMix64` — a small, fast, fully-deterministic PRNG.
///
/// Hand-rolled rather than pulled from `rand`, per this workspace's dependency
/// policy. `SplitMix64` (Steele, Lea & Flood 2014) is the standard seeding
/// generator for `xoshiro`/`xoroshiro`; it has a fixed period of `2^64`, passes
/// `BigCrush`, and — critically for a test suite — produces the identical
/// stream for an identical seed on every platform, because every operation is
/// an exact 64-bit integer operation.
#[derive(Debug, Clone)]
pub struct ReplugRng {
    state: u64,
}

impl ReplugRng {
    /// Seed the generator.
    #[must_use]
    pub fn new(seed: u64) -> Self {
        Self { state: seed }
    }

    /// The next raw 64-bit output.
    pub fn next_u64(&mut self) -> u64 {
        self.state = self.state.wrapping_add(0x9E37_79B9_7F4A_7C15);
        let mut z = self.state;
        z = (z ^ (z >> 30)).wrapping_mul(0xBF58_476D_1CE4_E5B9);
        z = (z ^ (z >> 27)).wrapping_mul(0x94D0_49BB_1331_11EB);
        z ^ (z >> 31)
    }

    /// A uniform draw from `[0, 1)`.
    ///
    /// Built from the top 53 bits, which is exactly the mantissa width of an
    /// `f64`: every representable value in `[0, 1)` at `2^-53` spacing is
    /// reachable and equally likely, and the result can never be exactly `1.0`
    /// (which would fall off the end of an inverse-CDF search).
    #[allow(clippy::cast_precision_loss)] // 53-bit integer -> f64 is exact.
    pub fn next_f64(&mut self) -> f64 {
        ((self.next_u64() >> 11) as f64) * (1.0 / 9_007_199_254_740_992.0) // 2^53
    }

    /// Sample a token index from a distribution given as **log**-probabilities,
    /// by inverse-CDF over the exponentiated mass.
    ///
    /// The accumulation is done in probability space (a CDF is a sum, and there
    /// is no stable log-space analogue of an inverse-CDF search), but the
    /// *inputs* are log-probabilities so the caller never had to materialize a
    /// possibly-underflowing `exp` twice. Tokens whose probability underflows to
    /// `0` are simply unreachable, which is the correct behaviour.
    ///
    /// The final `saturating_sub` guards the one floating-point edge case:
    /// `Σ p_i` can land a hair under the drawn `u ∈ [0, 1)` through rounding,
    /// in which case the loop falls off the end and the last index is returned
    /// rather than panicking or fabricating index `0`.
    pub fn sample_from_log_probs(&mut self, log_probs: &[f64]) -> Option<usize> {
        if log_probs.is_empty() {
            return None;
        }
        let draw = self.next_f64();
        let mut cumulative = 0.0_f64;
        for (index, &lp) in log_probs.iter().enumerate() {
            cumulative += lp.exp();
            if draw < cumulative {
                return Some(index);
            }
        }
        Some(log_probs.len().saturating_sub(1))
    }
}
