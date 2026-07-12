//! `REPLUG` (Shi et al., 2023, *"`REPLUG`: Retrieval-Augmented Black-Box
//! Language Models"*): ensembling a frozen language model's **output
//! probability distribution** across separately-retrieved documents, and
//! training the retriever from that model's feedback.
//!
//! Every retrieved document is prepended to the query **on its own**, the frozen
//! LM produces a next-token distribution for **each** such context
//! independently, and those `k` distributions are then combined into one:
//!
//! ```text
//!   p(y | q)  =  Σ_i  λ(d_i | q) · p(y | d_i ⊕ q)
//!   λ(d_i | q) = softmax( s(q, d_i) / τ )
//! ```
//!
//! The unit of combination is a **probability distribution over the vocabulary**
//! — not a ranked list, not a score, not a string. That is the whole idea, and
//! it is what makes `REPLUG` work with a genuinely black-box LM: it needs no
//! gradients, no fine-tuning, and no access to the model's internals beyond the
//! logits it already emits.
//!
//! # How this differs from the other combination modules in this crate
//!
//! Four modules here sound adjacent. None of them combines an output
//! distribution, and the distinction is one of *kind*, not of degree — they
//! operate on different objects at different points in the pipeline:
//!
//! | Module | Combines... | Is the LM involved? |
//! |---|---|---|
//! | `ensemble_retriever` | ranked candidate lists from `N` **retrievers** | **No.** There is no LM anywhere in it. |
//! | `rank_fusion` | already-produced **`(doc, score)` ranked lists** | **No.** Pure rank/score aggregation. |
//! | `answer_aggregator` | final **answer strings**, by voting over sentences | Only upstream — it sees the LM's *text*, after all the probability has been thrown away. |
//! | `fusion_in_decoder` | per-passage **extracted sentences** (a Jaccard-overlap heuristic) | **No.** Despite the name, it makes no LM call. |
//! | `replug` (this module) | the LM's **next-token probability distributions**, one per document | **Yes — `k` times per token.** The distributions *are* the thing being combined. |
//!
//! The practical consequence: `REPLUG` can change the model's mind about a token
//! that **no single document would have produced**. Two documents that each rank
//! the correct token *second* can, mixed, rank it *first* — the runner-up that
//! everyone agrees on beats the front-runners that nobody agrees on. That is not
//! reachable by fusing ranked lists (they contain no token-level information) or
//! by voting on strings (the LM has already committed to a token by then). This
//! module's test suite demonstrates it on hand-computed numbers.
//!
//! # The two numerical traps, and how this module avoids them
//!
//! ## 1. Mixing distributions is **not** averaging logits
//!
//! The single most common way to get `REPLUG` quietly wrong is to average the
//! per-document logits and softmax the result. That computes a **geometric**
//! mean of the distributions, not the arithmetic one:
//!
//! ```text
//!   softmax( Σ_i λ_i z_i )  ∝  Π_i p_i^{λ_i}     (geometric — an AND)
//!   Σ_i λ_i p_i                                   (arithmetic — an OR)
//! ```
//!
//! The two implement opposite logical connectives over the documents. Under the
//! geometric pool every document holds a **veto**: one document assigning ≈0 to a
//! token drives the pooled probability to ≈0 no matter how certain the others
//! are. Under the arithmetic mixture no document can veto anything — the value is
//! bounded below by `λ_i · p_i(y)` for every `i`. Since `REPLUG` retrieves `k`
//! documents independently and *expects* most of them to be irrelevant to any
//! given token, letting an irrelevant document veto the one relevant document's
//! answer is not a defensible reading of "ensemble the evidence". The derivation
//! is in [`math::log_linear_pool_log_probs`], both pools are implemented, and the
//! tests exhibit a concrete case where they disagree on the **arg-max**.
//!
//! ## 2. Never take the logarithm of a probability
//!
//! Real next-token distributions contain probabilities far below `f32`'s
//! underflow point (`exp(-88)` is already `0.0` there), and `REPLUG` needs their
//! logarithms — for the sequence log-likelihood, for the entropy, and above all
//! for the `LSR` `KL`. Compute the mixture in probability space and a token whose
//! per-document probabilities all underflow gets `p = 0.0`, hence
//! `log p = -inf`: an infinitely-confident claim that a token is *impossible*,
//! manufactured entirely by rounding. So the mixture is evaluated in log-space,
//!
//! ```text
//!   log p(y | q) = logsumexp_i ( log λ_i + log p(y | d_i ⊕ q) )
//! ```
//!
//! and stays finite for logit spreads of `±10⁴`. See [`math`].
//!
//! # `REPLUG`-`LSR`: the LM teaches the retriever
//!
//! The retriever was trained on its own contrastive objective; nobody ever asked
//! whether the documents it likes are the documents that make the **LM** better.
//! [`lsr`] closes that loop. With the LM frozen, its behaviour becomes the
//! supervision:
//!
//! ```text
//!   P_R(d | q)  = softmax( s(q, d) / τ )                    what the retriever believes
//!   Q_LM(d | q) = softmax( log P_LM(y* | d ⊕ q) / β )       what the LM's behaviour implies
//!   L           = KL( Q_LM ‖ P_R )
//!   ∂L/∂s_j     = ( P_R(d_j) − Q_LM(d_j) ) / τ
//! ```
//!
//! A document the LM valued more than the retriever did (`Q > P`) gets a negative
//! gradient, so a descent step **raises** its score and it climbs the ranking.
//! The derivation, and why the `KL` is taken in this order and not the other, are
//! in the [`lsr`] module docs.
//!
//! # Bring your own model
//!
//! `REPLUG` needs a full next-token distribution, which no existing trait in this
//! crate exposes (they return text plus scalar log-probabilities). So this module
//! states its own minimal requirement, [`ReplugLanguageModel`]: *give me the
//! logits for this context, and tell me how to decode a token id*. Implement it
//! over Candle, an HTTP endpoint, or anything else. For tests and for validating
//! your own wiring, [`ReplugStaticLanguageModel`] is a deterministic, table-driven
//! stand-in with no weights and no randomness.
//!
//! # Quick start
//!
//! The example below is the value proposition in miniature. Two documents; each
//! one, on its own, would have the LM emit a *different* token — and each ranks
//! the correct token (`"gamma"`) only **second**. The ensemble ranks it **first**.
//!
//! ```rust
//! # #[cfg(feature = "replug")]
//! # {
//! use oxirag::replug::{
//!     ReplugConfig, ReplugDocument, ReplugEngine, ReplugLanguageModel,
//!     ReplugStaticLanguageModel,
//! };
//!
//! let vocab = vec!["alpha".to_string(), "beta".to_string(), "gamma".to_string()];
//! let model = ReplugStaticLanguageModel::new(vocab)?
//!     // With doc A in context the LM prefers "alpha"; "gamma" is only 2nd.
//!     .with_probability_rule(vec!["SOURCE-A"], &[0.5, 0.1, 0.4])?
//!     // With doc B in context it prefers "beta"; "gamma" is again only 2nd.
//!     .with_probability_rule(vec!["SOURCE-B"], &[0.1, 0.5, 0.4])?;
//!
//! // Equal retrieval scores, so lambda = [0.5, 0.5].
//! let documents = vec![
//!     ReplugDocument::new("a", "SOURCE-A: ...", 1.0),
//!     ReplugDocument::new("b", "SOURCE-B: ...", 1.0),
//! ];
//!
//! let engine = ReplugEngine::new(ReplugConfig::default())?;
//! let (log_probs, weights, _) = engine.ensemble_next_token(&model, "q?", &documents)?;
//! let probs: Vec<f64> = log_probs.iter().map(|lp| lp.exp()).collect();
//!
//! // lambda comes from the f64 retrieval scores, so it is exact.
//! assert!((weights[0] - 0.5).abs() < 1e-12 && (weights[1] - 0.5).abs() < 1e-12);
//!
//! // 0.5*0.5 + 0.5*0.1 = 0.30   |   0.5*0.1 + 0.5*0.5 = 0.30   |   0.5*0.4 + 0.5*0.4 = 0.40
//! //
//! // Tolerance 1e-6, not 1e-12: `next_token_logits` returns `f32` (as a real
//! // model's projection does), so a probability written as `0.5` here round-trips
//! // as `exp(log_softmax(f64::from(ln(0.5) as f32)))`. f32's 24-bit mantissa —
//! // not the f64 mixture math — is the binding error term.
//! assert!((probs[0] - 0.30).abs() < 1e-6);
//! assert!((probs[1] - 0.30).abs() < 1e-6);
//! assert!((probs[2] - 0.40).abs() < 1e-6);
//!
//! // Neither document would have said "gamma". The ensemble does.
//! assert_eq!(model.decode(2)?, "gamma");
//! # }
//! # Ok::<(), oxirag::replug::ReplugError>(())
//! ```

pub mod ensemble;
pub mod lsr;
pub mod math;
pub mod model;
pub mod types;

#[cfg(test)]
mod tests;

pub use ensemble::ReplugEngine;

// The math helpers are re-exported under `replug_`-prefixed aliases. This crate
// funnels ~190 modules through a single flat prelude, so a bare `softmax` or
// `log_softmax` at the root of a module's public surface is a collision waiting
// to happen (`distillation::TemperatureScaling` already has both, as inherent
// methods). Inside this module they keep their natural names; only the
// re-exported aliases are prefixed.
pub use math::{
    ReplugRng, arg_max as replug_arg_max, entropy_from_log_probs as replug_entropy_from_log_probs,
    kl_divergence_from_log_probs as replug_kl_divergence,
    log_linear_pool_log_probs as replug_log_linear_pool, log_softmax as replug_log_softmax,
    log_sum_exp as replug_log_sum_exp, mixture_log_probs as replug_mixture_log_probs,
    promote_logits as replug_promote_logits, softmax as replug_softmax,
    temperature_log_softmax as replug_temperature_log_softmax,
    temperature_softmax as replug_temperature_softmax,
};
pub use model::{ReplugContextRule, ReplugLanguageModel, ReplugStaticLanguageModel};
pub use types::{
    ReplugConfig, ReplugDecoding, ReplugDocument, ReplugDocumentGradient, ReplugEnsembleOutput,
    ReplugError, ReplugLsrSignal, ReplugResult, ReplugStats, ReplugStep,
};
