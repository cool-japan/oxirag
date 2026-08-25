//! Self-Taught Reasoner (`STaR`) — bootstrap a reasoner from its own correct
//! rationales, rationalizing backward from the gold answer for the problems it
//! cannot yet solve.
//!
//! Implements `STaR` (Zelikman et al., 2022, *"`STaR`: Bootstrapping Reasoning
//! With Reasoning"*). The loop is:
//!
//! 1. **Forward generation.** For each problem, the generator produces a
//!    rationale and an answer.
//! 2. **Correctness filter.** Keep the `(problem, rationale, answer)` triples
//!    whose answer is correct; discard the rest.
//! 3. **Backward rationalization.** For each problem the forward pass *failed*,
//!    hand the generator the **gold answer as a hint** and ask it to produce a
//!    rationale that reaches it. Keep those that now reach the gold answer —
//!    unless they cheat (see below).
//! 4. **Accumulate & fine-tune.** Add the kept rationales to a bootstrapped
//!    training set and fine-tune the generator on it.
//! 5. **Iterate** to a fixed point: repeat until the accumulated set stops
//!    growing.
//!
//! # Distinct from `self_consistency`
//!
//! Distinct from `self_consistency`, which samples K rationales and
//! **marginalizes them away** at inference (the reasoning is a means to a voted
//! answer and is then discarded): this module **keeps** the rationales that
//! reached the right answer and **rationalizes backwards** from the gold answer
//! for the ones that did not, accumulating a bootstrapped training set that
//! improves the generator across rounds.
//!
//! # The cheat-rationale problem
//!
//! Backward rationalization is handed the answer, so a rationale can "reach" it
//! by simply restating it — "the answer is `X` because the answer is `X`" — with
//! no reasoning at all. Such rationales are poison as training signal, and
//! `STaR`'s whole validity depends on excluding them. This module detects and
//! rejects them ([`is_cheating_rationale`]): a kept rationalization must engage
//! the problem *and* show at least one computed intermediate, not merely echo the
//! hint. Both sides of that line are tested.
//!
//! # Bring your own model
//!
//! `STaR` needs a learner that can generate, rationalize toward a hint, and
//! **update its policy** — a capability no text-in/text-out trait in this crate
//! exposes. So this module states its own minimal requirement,
//! [`ReasoningModel`]. For tests and for validating your own wiring,
//! [`StaticReasoningModel`] is a deterministic, rule-inducing stand-in whose
//! policy genuinely improves as the accumulated set grows — no weights, no
//! randomness, no network.
//!
//! # Quick start
//!
//! A tiny task, `f(x) = (x + 4) mod 13`, seeded with one worked example. With
//! rationalization off, the model bootstraps purely forward: it induces the
//! offset from the seed and its solved region grows outward each round until it
//! covers the whole domain.
//!
//! ```
//! # #[cfg(feature = "self-taught-reasoner")]
//! # {
//! use oxirag::self_taught_reasoner::{
//!     SelfTaughtReasoner, StarConfig, StarProblem, StaticReasoningModel,
//! };
//!
//! let modulus: u64 = 13;
//! let offset: u64 = 4;
//! let problems: Vec<StarProblem> = (0..modulus)
//!     .map(|x| {
//!         StarProblem::new(
//!             format!("a{x}"),
//!             format!("[A] x={x}"),
//!             ((x + offset) % modulus).to_string(),
//!         )
//!     })
//!     .collect();
//!
//! // Seeded at x = 6 with a generalization radius of 2.
//! let mut model = StaticReasoningModel::new(modulus, 2)
//!     .with_seed("A", 6, ((6 + offset) % modulus).to_string());
//!
//! let reasoner = SelfTaughtReasoner::new(StarConfig::new().with_rationalization(false));
//! let outcome = reasoner.run(&problems, &mut model).unwrap();
//!
//! assert!(outcome.converged);
//! assert_eq!(outcome.final_coverage(), 1.0);
//! // Forward accuracy climbs monotonically to the ceiling.
//! let curve = outcome.accuracy_curve();
//! assert!(curve.first().copied().unwrap() < curve.last().copied().unwrap());
//! assert_eq!(curve.last().copied().unwrap(), 1.0);
//! # }
//! ```

pub mod engine;
pub mod model;
pub mod rng;
pub mod text;
pub mod types;

#[cfg(all(test, not(target_arch = "wasm32")))]
mod tests;

pub use engine::SelfTaughtReasoner;
pub use model::{RationalizationStyle, ReasoningModel, StaticReasoningModel};
pub use rng::{StarRng, mix_seed};
pub use text::{answers_equivalent, is_cheating_rationale, normalize_answer};
pub use types::{
    BootstrapRound, RationaleSet, RationaleSource, StarConfig, StarError, StarGeneration,
    StarOutcome, StarProblem, StarRationale,
};
