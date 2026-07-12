#![allow(
    clippy::float_cmp,
    clippy::similar_names,
    clippy::cast_precision_loss,
    clippy::too_many_lines,
    clippy::unreadable_literal,
    clippy::manual_midpoint,
    clippy::items_after_statements
)]
//! Tests for the `REPLUG` output-distribution ensemble and `REPLUG`-`LSR`.
//!
//! The organising principle: **every claim the module's documentation makes is
//! checked against numbers computed by hand in the test itself**, never against
//! numbers computed by re-calling the code under test. Where a test asserts that
//! the mixture equals `0.5 · 0.5 + 0.5 · 0.1 = 0.30`, that arithmetic is written
//! out literally, so the assertion is an independent check on the implementation
//! rather than a restatement of it.
//!
//! Sections: math primitives → λ weighting and its two temperature limits → the
//! mixture (hand-computed) → arithmetic-vs-geometric → *the ensemble beats every
//! one of its parts* → generation → `LSR` → numerical stability → edge cases.

use super::ensemble::ReplugEngine;
use super::math::{
    ReplugRng, arg_max, entropy_from_log_probs, kl_divergence_from_log_probs,
    log_linear_pool_log_probs, log_softmax, log_sum_exp, mixture_log_probs, promote_logits,
    softmax, temperature_log_softmax, temperature_softmax,
};
use super::model::{ReplugLanguageModel, ReplugStaticLanguageModel};
use super::types::{ReplugConfig, ReplugDecoding, ReplugDocument, ReplugError};

/// Absolute tolerance for `f64` distribution arithmetic. Every quantity here is
/// `O(1)`, and the operations are a handful of `exp`/`ln`/`+`, so a few units in
/// the last place of `f64` (~`2e-16`) is the real error scale; `1e-12` is a
/// generous but still *meaningful* bound — loose enough not to be flaky, tight
/// enough that a genuinely wrong formula cannot slip under it.
const TOL: f64 = 1e-12;

/// Tolerance for a quantity that has crossed the **`f32` logit boundary**.
///
/// [`super::model::ReplugLanguageModel::next_token_logits`] returns `f32` —
/// deliberately, because that is what a real model's final projection emits. So
/// when a test writes "this document's distribution is `[0.5, 0.1, 0.4]`", the
/// fixture stores `ln(p) as f32` and the engine reads it back as
/// `exp(log_softmax(f64::from(logit)))`. The round-trip is therefore limited by
/// `f32`'s 24-bit mantissa (relative epsilon ≈ `6e-8`), **not** by the `f64`
/// mixture arithmetic.
///
/// The budget: `|Δ ln p| ≲ |ln p| · 6e-8 ≲ 1.4e-7` for `p ∈ [0.01, 1]`, and a
/// softmax is 1-Lipschitz in its logits, so `|Δp| ≲ 3e-7`. `1e-6` sits just
/// above that. It is a *real* bound, not a fudge factor: a genuinely wrong
/// mixture formula is wrong by `O(10⁻²)` or more (the arithmetic-vs-geometric
/// test differs by `0.2`), so nothing incorrect can hide under it.
///
/// Assertions on quantities that never touch the model — the pure-`f64` mixture,
/// λ (computed from `f64` retrieval scores), the `KL` — use the much tighter
/// [`TOL`].
const F32_TOL: f64 = 1e-6;

fn assert_close(actual: f64, expected: f64, what: &str) {
    assert!(
        (actual - expected).abs() < TOL,
        "{what}: expected {expected}, got {actual} (diff {})",
        (actual - expected).abs()
    );
}

/// [`assert_close`] for a value that has passed through the `f32` logits of a
/// [`ReplugStaticLanguageModel`]. See [`F32_TOL`].
fn assert_close_f32(actual: f64, expected: f64, what: &str) {
    assert!(
        (actual - expected).abs() < F32_TOL,
        "{what}: expected {expected}, got {actual} (diff {}, f32 budget {F32_TOL})",
        (actual - expected).abs()
    );
}

/// Assert that `probs` is a *proper* probability distribution: every entry
/// non-negative and finite, and the whole thing sums to exactly 1 up to
/// rounding. Called on the output of every mixture in this file.
fn assert_is_distribution(probs: &[f64], what: &str) {
    for (index, &p) in probs.iter().enumerate() {
        assert!(p.is_finite(), "{what}: entry {index} is not finite: {p}");
        assert!(p >= 0.0, "{what}: entry {index} is negative: {p}");
        assert!(p <= 1.0 + TOL, "{what}: entry {index} exceeds 1: {p}");
    }
    let total: f64 = probs.iter().sum();
    assert!(
        (total - 1.0).abs() < 1e-10,
        "{what}: does not sum to 1, sums to {total}"
    );
}

fn exp_all(log_probs: &[f64]) -> Vec<f64> {
    log_probs.iter().copied().map(f64::exp).collect()
}

// ── Shared fixtures ──────────────────────────────────────────────────────────
//
// Model fixtures and document builders reused across the test submodules.
// Child modules reach them via `use super::*`.

fn documents_for_weighting() -> Vec<ReplugDocument> {
    vec![
        ReplugDocument::new("top", "TOP", 0.9),
        ReplugDocument::new("mid", "MID", 0.6),
        ReplugDocument::new("low", "LOW", 0.1),
    ]
}

/// Vocabulary: token 0 = "alpha", 1 = "beta", 2 = "gamma". `"gamma"` is the
/// correct answer.
///
/// Document A makes the LM prefer `"alpha"`; document B makes it prefer
/// `"beta"`. **Neither document, alone, would produce `"gamma"`** — under each
/// one it is only the runner-up. The ensemble emits it anyway.
fn consensus_runner_up_model() -> ReplugStaticLanguageModel {
    ReplugStaticLanguageModel::new(vec![
        "alpha".to_string(),
        "beta".to_string(),
        "gamma".to_string(),
    ])
    .expect("vocab")
    // p(· | A ⊕ q) = [0.50, 0.10, 0.40] -> argmax "alpha", "gamma" is RANK 2
    .with_probability_rule(vec!["SOURCE-A"], &[0.50, 0.10, 0.40])
    .expect("rule a")
    // p(· | B ⊕ q) = [0.10, 0.50, 0.40] -> argmax "beta",  "gamma" is RANK 2
    .with_probability_rule(vec!["SOURCE-B"], &[0.10, 0.50, 0.40])
    .expect("rule b")
}

/// A fixture whose distribution depends on **both** the document and the tokens
/// generated so far, so that generation is a genuine multi-step process rather
/// than the same token emitted `n` times.
///
/// Vocabulary: `0 = "R"`, `1 = "G"`, `2 = "B"`, `3 = "<eos>"`.
fn stateful_model() -> ReplugStaticLanguageModel {
    ReplugStaticLanguageModel::new(vec![
        "R".to_string(),
        "G".to_string(),
        "B".to_string(),
        "<eos>".to_string(),
    ])
    .expect("vocab")
    // Step 0 — nothing generated yet. A prefers R, B prefers B; G is the
    // consensus runner-up and wins the mixture.
    .with_probability_rule(vec!["SOURCE-A"], &[0.50, 0.35, 0.10, 0.05])
    .expect("a step0")
    .with_probability_rule(vec!["SOURCE-B"], &[0.10, 0.35, 0.50, 0.05])
    .expect("b step0")
    // Step 1 — the two-substring rules are strictly more specific, so they win
    // once "G" is in context. Both documents now agree on "B".
    .with_probability_rule(vec!["SOURCE-A", "G"], &[0.05, 0.05, 0.80, 0.10])
    .expect("a step1")
    .with_probability_rule(vec!["SOURCE-B", "G"], &[0.05, 0.05, 0.80, 0.10])
    .expect("b step1")
    // Step 2 — with "GB" in context both documents want to stop.
    .with_probability_rule(vec!["SOURCE-A", "GB"], &[0.02, 0.02, 0.06, 0.90])
    .expect("a step2")
    .with_probability_rule(vec!["SOURCE-B", "GB"], &[0.02, 0.02, 0.06, 0.90])
    .expect("b step2")
    .with_eos_token(3)
    .expect("eos in vocab")
}

/// Vocabulary `["yes", "no"]`. The ground truth is `"yes"` (token 0).
///
/// - `HELPFUL` makes the LM assign `p("yes") = 0.9`.
/// - `USELESS` makes it assign `p("yes") = 0.2` — worse than a coin flip, i.e.
///   this document actively misleads the model.
///
/// The retriever, however, has them the **wrong way round**: it scores `USELESS`
/// above `HELPFUL`. `LSR` must fix that.
fn lsr_model() -> ReplugStaticLanguageModel {
    ReplugStaticLanguageModel::new(vec!["yes".to_string(), "no".to_string()])
        .expect("vocab")
        .with_probability_rule(vec!["HELPFUL"], &[0.90, 0.10])
        .expect("helpful rule")
        .with_probability_rule(vec!["USELESS"], &[0.20, 0.80])
        .expect("useless rule")
}

fn misranked_documents() -> Vec<ReplugDocument> {
    vec![
        // The retriever's favourite — and the LM's least favourite.
        ReplugDocument::new("useless", "USELESS text", 1.0),
        // What the LM actually needs, ranked second.
        ReplugDocument::new("helpful", "HELPFUL text", 0.5),
    ]
}

// ── Submodules ───────────────────────────────────────────────────────────────
mod generation;
mod lsr;
mod mixture;
mod primitives;
mod stability;
