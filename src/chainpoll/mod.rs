//! `ChainPoll`: hallucination detection via deterministic multi-formulation
//! chain-of-thought polling (Friel & Sanyal, 2023, *"`ChainPoll`: A High
//! Efficacy Method for LLM Hallucination Detection"*).
//!
//! ## The mechanism: prompt-formulation polling, not sampling
//!
//! `ChainPoll`'s defining idea is easy to mistake for stochastic
//! self-consistency because both ask "does this claim hold up under
//! repetition?" -- but the *axis of repetition* is completely different from
//! this crate's other hallucination/consistency modules:
//!
//! - [`crate::selfcheckgpt`] fixes one prompt and draws `K` samples from a
//!   generator at **temperature > 0**, measuring how consistently the
//!   *content* recurs across stochastic re-generations.
//! - [`crate::eigenscore`] also draws `K` stochastic samples, but measures
//!   the **differential entropy of their embedding covariance** rather than
//!   lexical/semantic consistency.
//! - **`chainpoll`** never samples anything. It holds the generator's
//!   behaviour fixed (deterministic, temperature-independent) and instead
//!   varies the **prompt formulation** itself: `N` structurally distinct
//!   phrasings of the same yes/no "is this claim grounded in the context?"
//!   question -- different wording, word order, perspective, and even
//!   inverted framings (e.g. "does the context *contradict* the claim?") --
//!   each asked exactly once via a chain-of-thought judge. The claim's
//!   factuality is decided by majority vote across those `N` formulations,
//!   never by resampling any one of them.
//!
//! ## Algorithm
//!
//! 1. **Generate.** [`ChainPollScorer::generate_formulations`] renders the
//!    first `N` = [`ChainPollConfig::num_formulations`] entries of a fixed,
//!    ordered bank of prompt templates into concrete [`PollFormulation`]s for
//!    the claim and context. Purely deterministic: no randomness anywhere.
//! 2. **Judge.** Each [`PollFormulation`] is dispatched to a pluggable
//!    [`ChainOfThoughtJudge`], which returns a raw yes/no verdict plus its
//!    chain-of-thought reasoning. [`MockChainPollJudge`] is a deterministic
//!    lexical-overlap stand-in for tests; production callers wire this trait
//!    up to a real LLM chain-of-thought call.
//! 3. **Un-invert.** Some formulations are deliberately phrased as the
//!    *opposite* question (see [`PollFraming::Inverted`]) to diversify the
//!    prompt surface; each formulation's raw verdict is normalized to "claim
//!    supported by context" via [`PollFraming::resolve_supported`] before it
//!    is allowed to vote.
//! 4. **Majority vote.** [`ChainPollResult::hallucination_score`] is the
//!    fraction of the `N` un-inverted verdicts that voted "hallucinated" --
//!    calibrated so a unanimous "supported" poll scores near `0.0`, a
//!    unanimous "hallucinated" poll scores near `1.0`, and an evenly split
//!    poll scores near `0.5`. [`ChainPollResult::is_hallucination`] compares
//!    that score against [`ChainPollConfig::hallucination_threshold`], and
//!    [`ChainPollResult::confidence`] measures how decisive the majority is.
//!
//! # Example
//!
//! ```
//! use oxirag::chainpoll::{ChainPollConfig, ChainPollScorer, MockChainPollJudge};
//!
//! let scorer = ChainPollScorer::new(
//!     ChainPollConfig::default(),
//!     Box::new(MockChainPollJudge::default()),
//! );
//!
//! let context = "The Eiffel Tower was completed in 1889 and is located in Paris, France.";
//!
//! // A claim that closely echoes the context is judged supported by every
//! // formulation, so the hallucination score is near zero.
//! let grounded = scorer
//!     .poll(
//!         "The Eiffel Tower was completed in 1889 in Paris, France.",
//!         context,
//!     )
//!     .expect("valid claim and context");
//! assert!(!grounded.is_hallucination);
//! assert!(grounded.hallucination_score < 0.2);
//!
//! // A claim sharing almost no vocabulary with the context is judged
//! // unsupported by every formulation, so the score is near one.
//! let fabricated = scorer
//!     .poll("Kangaroos are native to the Amazon rainforest.", context)
//!     .expect("valid claim and context");
//! assert!(fabricated.is_hallucination);
//! assert!(fabricated.hallucination_score > 0.8);
//!
//! // Every one of the five default formulations produced a verdict.
//! assert_eq!(grounded.num_formulations(), 5);
//! ```

pub mod scorer;
#[cfg(test)]
mod tests;
pub mod types;

pub use scorer::ChainPollScorer;
pub use types::{
    ChainOfThoughtJudge, ChainPollConfig, ChainPollError, ChainPollFormulationVerdict,
    ChainPollResult, MockChainPollJudge, PollFormulation, PollFraming,
};
