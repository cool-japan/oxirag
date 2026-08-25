//! Self-Consistency — sample diverse reasoning paths and marginalize over them.
//!
//! Implements *self-consistency decoding* (Wang et al., 2022,
//! "Self-Consistency Improves Chain of Thought Reasoning in Language Models").
//! Instead of greedily decoding a single chain of thought, the engine samples
//! `K` diverse reasoning paths for a question, extracts each path's final
//! answer, **clusters** the answers by semantic equivalence, and
//! **marginalizes** over the latent reasoning paths: the answer cluster with
//! the highest aggregate vote wins, and the confidence is that cluster's share
//! of the total vote mass.
//!
//! This module is **distinct** from `answer_aggregator`: the aggregator *fuses*
//! externally-provided candidate texts into one synthesized answer, whereas
//! self-consistency *samples* its own diverse reasoning chains and selects the
//! modal answer by majority vote over those chains.
//!
//! # Example
//!
//! ```
//! use oxirag::self_consistency::{
//!     MockReasoningSampler, ReasoningPath, SelfConsistencyConfig, SelfConsistencyEngine,
//! };
//!
//! let sampler = MockReasoningSampler::new(vec![
//!     ReasoningPath::new("2 + 2 = 4, then double", "8"),
//!     ReasoningPath::new("count up: 8", "8"),
//!     ReasoningPath::new("guessed", "7"),
//! ]);
//! let engine = SelfConsistencyEngine::new(SelfConsistencyConfig::new().with_num_paths(3));
//! let out = engine.run("what is 2*4?", &sampler).unwrap();
//! assert_eq!(out.answer, "8");
//! ```

pub mod engine;
pub mod types;

#[cfg(all(test, not(target_arch = "wasm32")))]
mod tests;

pub use engine::SelfConsistencyEngine;
pub use types::{
    AnswerCluster, MockReasoningSampler, ReasoningPath, ReasoningSampler, SelfConsistencyConfig,
    SelfConsistencyError, SelfConsistencyOutput, VoteWeighting,
};
