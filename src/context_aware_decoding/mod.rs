//! Decode-time **distribution contrast**: `CAD`, contrastive decoding and
//! `DoLa` — three methods that improve a language model's next-token
//! distribution by *subtracting* another one.
//!
//! Every other decoding-time technique in this crate makes the model's
//! distribution **better informed**: retrieve more, prompt differently, ensemble
//! more evidence, then read off the arg-max. The three methods here do something
//! categorically different. They run the model **twice**, take the two
//! distributions it produces, and *push the output away from the second one*.
//! What comes out is not a distribution the model would ever have emitted; it is
//! a distribution that isolates the **difference** between two things the model
//! believes.
//!
//! ```text
//!   score(y)  =  w⁺ · v⁺(y)  +  w⁻ · v⁻(y) ,          w⁻ < 0
//! ```
//!
//! and the three methods are three answers to one question — *which two
//! distributions?*
//!
//! | | positive `v⁺` | negative `v⁻` | what the subtraction removes |
//! |---|---|---|---|
//! | [`ContextAwareDecoder`] — `CAD`, Shi et al. 2024 | `p(y | c ⊕ x)`, **with** the retrieved context | `p(y | x)` — the *same model*, **without** it | what the model would have said from memory, so that only the retrieved evidence is left |
//! | [`ContrastiveDecoder`] — Li et al. 2023 | `p_expert(y | x)`, the large model | `p_amateur(y | x)` — a *smaller* model, same prompt | the failure modes that scale does not fix: repetition, genericness, the locally-likely token |
//! | [`DoLaDecoder`] — Chuang et al. 2024 | `q_mature(y | x)`, the **final** layer | `q_premature(y | x)` — an **earlier layer of the same forward pass** | the part of the prediction that was already settled low in the stack, leaving what the upper layers actually contributed |
//!
//! `CAD` exists for the failure that makes retrieval-augmented generation
//! untrustworthy: a model that has *memorized* a fact keeps asserting it even
//! when the retrieved passage says otherwise, because the passage is worth a few
//! logits and the memory is worth many. Retrieving harder does not fix that. The
//! only way to find out what the context actually contributed is to ask the model
//! the same question **without** it, and subtract the answer.
//!
//! # Distinct from `replug`
//!
//! Distinct from [`replug`](crate::replug), which **interpolates** `k` next-token
//! distributions with non-negative weights summing to one (a convex mixture over
//! retrieved documents — no distribution can ever be subtracted). This module
//! **extrapolates**: it applies a *negative* coefficient to a second distribution
//! in order to push the output *away* from it. That leaves the probability
//! simplex before renormalization, and therefore requires an adaptive
//! plausibility mass constraint — which a convex mixture never needs, and which
//! [`replug::math::log_linear_pool_log_probs`](crate::replug::math::log_linear_pool_log_probs)
//! does not provide.
//!
//! That is worth unpacking, because the *arithmetic* of the two modules overlaps
//! while the semantics do not.
//!
//! `crate::replug::math::log_linear_pool_log_probs` computes
//! `log_softmax(Σ_i w_i · z_i)`, and its validation checks only that the weights
//! and the logit vectors agree in number — it never requires the weights to be
//! non-negative, or to sum to one. So
//!
//! ```text
//!   log_linear_pool_log_probs(&[1 + alpha, -alpha], &[z_with_context, z_without_context])
//! ```
//!
//! **is** `CAD`'s formula, already computable today. This module's tests assert
//! exactly that, bit-for-bit, and treat it as ground truth: a cross-check against
//! a kernel written by somebody else, for another purpose, in another module.
//!
//! It does not make this module a duplicate, for three reasons.
//!
//! 1. **It is an eight-line kernel, not a method.** `CAD`, contrastive decoding
//!    and `DoLa` are: construct the two contexts (or find the two models, or
//!    select the two layers), query the model twice, combine — and, critically,
//!    **constrain**. The kernel supplies only "combine".
//!
//! 2. **The negative-weight regime is outside what that function's own
//!    documentation reasons about.** It interprets its weights *exclusively* as a
//!    softmax over documents: non-negative, summing to one, under which it is the
//!    normalized weighted **geometric** mean of the per-document distributions. It
//!    exists in `REPLUG` chiefly as a *regression-test foil* — the "wrong pool",
//!    implemented so that `REPLUG`'s arithmetic mixture can be measured against
//!    it. With a negative weight its factor `p⁻(y)^{w⁻}` becomes
//!    `1 / p⁻(y)^{|w⁻|}`, which **diverges as `p⁻(y) → 0`**. No test and no line
//!    of documentation over there covers that.
//!
//! 3. **The fix for that divergence is this module's core content.** It is called
//!    the *adaptive plausibility constraint*, it appears in all three papers, and
//!    a convex mixture never needs one — a convex mixture cannot leave the simplex,
//!    and so has nothing to be rescued from.
//!
//! # The divergence, and the constraint that fixes it
//!
//! The pathology is not hypothetical and it is not mild. Take a four-token
//! vocabulary and a `CAD` step at `alpha = 1`, with logits
//!
//! ```text
//!   z⁺ = [  0, -1, -2, -12 ]      p⁺ = [ 0.665, 0.245, 0.090, 0.0000041 ]
//!   z⁻ = [ -3,  0, -1, -40 ]
//! ```
//!
//! The last token is one the context-conditioned model considers essentially
//! impossible — four parts in a million. But the context-*free* model considers it
//! **very much more** impossible, and an unconstrained contrast reads that as
//! overwhelming evidence:
//!
//! ```text
//!   score = 2·z⁺ − z⁻ = [ 3, -2, -3, 16 ]
//! ```
//!
//! Unconstrained, `CAD` hands that token a probability of **0.99999**. A token the
//! informed model gave four parts in a million becomes a near-certainty, purely
//! because the *less* informed of the two distributions had never heard of it —
//! and the less informed distribution is precisely the one whose zeros should
//! carry the least authority. Unconstrained extrapolation gives them the most.
//!
//! The adaptive plausibility constraint
//!
//! ```text
//!   V_valid  =  { y : p⁺(y)  ≥  alpha_plaus · max_{y'} p⁺(y') }
//! ```
//!
//! keeps only the tokens the *positive* distribution finds credible, masks the
//! rest to `-inf` **before** renormalizing, and returns the arg-max to the first
//! token at `p = 0.991`. It is a mass constraint; it is *adaptive* because the
//! threshold is a fraction of the running maximum, so it tightens exactly where
//! the model is confident and relaxes where it is not; and it can never empty the
//! vocabulary, because the arg-max of `p⁺` survives it for every
//! `alpha_plaus ≤ 1`. This module's tests carry the whole example, with these
//! numbers.
//!
//! # Jensen–Shannon divergence
//!
//! `DoLa` selects its premature layer by `arg max_j JSD( q_mature ‖ q_j )`, and no
//! Jensen–Shannon divergence existed anywhere in this crate, so
//! [`math::jensen_shannon_divergence_from_log_probs`] is the crate's only one. It
//! is assembled from `REPLUG`'s log-space kernels —
//! `log M = logaddexp(log P, log Q) − ln 2`, then
//! `JSD = ½ KL(P ‖ M) + ½ KL(Q ‖ M)` — and the property that makes it the right
//! choice, where a `KL` is not, is that it is **bounded** by `ln 2`. A `KL` between
//! two layers is routinely `+inf`, because an early layer will happily assign a
//! token a probability that underflows; an `arg max` over a set containing `+inf`
//! selects on a rounding artefact rather than on a divergence.
//!
//! # Bring your own model
//!
//! `CAD` and contrastive decoding need only [`ContextAwareLanguageModel`]: *give
//! me the next-token logits for this context*. Any black box — an HTTP endpoint, a
//! quantized on-device model — satisfies it.
//!
//! `DoLa` needs strictly more, and the type system says so.
//! [`LayeredLanguageModel`] additionally demands `layer_logits(context, layer)`
//! and `num_layers()`: the ability to project an *intermediate* layer through the
//! output head. That is a white-box capability and a real deployment constraint,
//! which is why it is a trait bound rather than a runtime error.
//!
//! [`ContextAwareStaticLanguageModel`] implements both, deterministically and by
//! table lookup, so that every numeric claim above is checkable with no weights, no
//! randomness and no network.
//!
//! # Quick start
//!
//! The value proposition in miniature: a model whose parametric memory *overrides*
//! the retrieved context. It has been given the context, and it still prefers what
//! it remembers. `CAD` fixes that, and the numbers are exact.
//!
//! ```rust
//! # #[cfg(feature = "context-aware-decoding")]
//! # {
//! use oxirag::context_aware_decoding::{
//!     CadConfig, ContextAwareDecoder, ContextAwareStaticLanguageModel,
//! };
//!
//! // 0 = "Sylvania" (what the retrieved context says), 1 = "Paris" (what the
//! // model remembers), 2 = "London", 3 = "Zzyzx".
//! let vocab = vec![
//!     "Sylvania".to_string(), "Paris".to_string(),
//!     "London".to_string(), "Zzyzx".to_string(),
//! ];
//! let model = ContextAwareStaticLanguageModel::new(vocab)?
//!     // WITH the context. (The with-context prompt contains the query as a
//!     // substring, so this rule keys on both markers in order to be the more
//!     // specific one.) The context moved the model -- but not far enough: its
//!     // arg max is still the remembered answer.
//!     .with_probability_rule(vec!["CONTEXT", "QUESTION"], &[0.35, 0.40, 0.20, 0.05])?
//!     // WITHOUT it: the parametric prior, undisturbed.
//!     .with_probability_rule(vec!["QUESTION"], &[0.05, 0.70, 0.20, 0.05])?;
//!
//! let decoder = ContextAwareDecoder::new(CadConfig::default().with_alpha(1.0))?;
//! let outcome = decoder.next_token(
//!     &model,
//!     "CONTEXT: Freedonia's capital is Sylvania.",
//!     "QUESTION: What is the capital?",
//! )?;
//!
//! // Left alone, the model ignores the context and says what it remembers.
//! assert_eq!(outcome.positive_argmax(), Some(1)); // "Paris"
//! // Under the contrast it says what the evidence says.
//! assert_eq!(outcome.argmax(), Some(0));          // "Sylvania"
//!
//! // The numbers are hand-computable: p_cad(y) is proportional to
//! // p+(y) * (p+(y) / p-(y))^alpha, so at alpha = 1 the unnormalized masses are
//! // 0.35*(0.35/0.05) = 2.45, 0.40*(0.40/0.70) = 0.2285714, 0.20*1 and 0.05*1 --
//! // that is [171.5, 16, 14, 3.5] / 70, which normalizes to
//! // [171.5, 16, 14, 3.5] / 205.
//! let probs = outcome.probs();
//! assert!((probs[0] - 171.5 / 205.0).abs() < 1e-6); // 0.8366 -- was 0.35
//! assert!((probs[1] -  16.0 / 205.0).abs() < 1e-6); // 0.0780 -- was 0.40
//! # }
//! # Ok::<(), oxirag::context_aware_decoding::ContextAwareError>(())
//! ```

pub mod engine;
pub mod math;
pub mod model;
pub mod types;

#[cfg(all(test, not(target_arch = "wasm32")))]
mod tests;

pub use engine::{ContextAwareDecoder, ContrastiveDecoder, DoLaDecoder};

// The math helpers are re-exported under `cad_`-prefixed aliases. This crate
// funnels ~200 modules through a single flat prelude, so a bare `contrast_scores`
// at the root of a module's public surface is a collision waiting to happen.
// Inside this module they keep their natural names; only the re-exported aliases
// are prefixed.
pub use math::{
    adaptive_plausibility_mask as cad_adaptive_plausibility_mask,
    contrast_log_probs as cad_contrast_log_probs, contrast_scores as cad_contrast_scores,
    jensen_shannon_divergence_from_log_probs as cad_jensen_shannon_divergence,
    masked_contrast_log_probs as cad_masked_contrast_log_probs,
};
pub use model::{
    ContextAwareLanguageModel, ContextAwareRule, ContextAwareStaticLanguageModel,
    LayeredLanguageModel,
};
pub use types::{
    CadConfig, ContextAwareConfig, ContextAwareError, ContextAwareOutput, ContextAwareResult,
    ContextAwareStats, ContextAwareStep, ContrastOutcome, ContrastiveConfig, DecodeMode,
    DecodingStrategy, DolaConfig, LayerSelector, PrematureLayer,
};
