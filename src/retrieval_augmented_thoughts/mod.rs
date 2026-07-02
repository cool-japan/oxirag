//! Retrieval-Augmented Thoughts (RAT; Wang et al., 2024) — per-step revision
//! of a drafted chain-of-thought, conditioned on step-targeted retrieval.
//!
//! RAT first drafts an ordinary chain-of-thought (`CoT`) for a question, then
//! walks the draft **left-to-right**, revising each thought step with
//! evidence retrieved specifically for that step. The retrieval query for
//! step `i` is built from the original question, the thoughts already
//! revised (`0..i`), and the current draft thought `i` — so retrieval is
//! conditioned on the cumulative reasoning so far, not just the bare
//! question. Once every step has been revised, the final answer is
//! synthesised from the fully revised chain.
//!
//! ## Distinct from `iterative_rag` and `self_ask`
//!
//! | Module | Unit of revision | Retrieval query source |
//! |--------|-------------------|-------------------------|
//! | `iterative_rag` (ITER-RETGEN) | The *whole answer*, regenerated fresh each round | Original query expanded with terms from the previous draft |
//! | `self_ask` | Explicit follow-up sub-questions, interleaved with sub-answers | Each self-asked follow-up |
//! | **`retrieval_augmented_thoughts`** | Each *thought step* of a pre-drafted `CoT`, revised once, in order | Question + already-revised prior steps + the step's own draft |
//!
//! # Example
//!
//! ```
//! use oxirag::retrieval_augmented_thoughts::{
//!     MockRatGenerator, MockRatRetriever, RatConfig, RatEngine,
//! };
//!
//! let generator = MockRatGenerator::new(vec![
//!     "The Eiffel Tower is in Berlin.".to_string(),
//!     "It was completed in 1990.".to_string(),
//! ]);
//! let retriever = MockRatRetriever::new(vec![
//!     "The Eiffel Tower is located in Paris, France.".to_string(),
//!     "Construction of the Eiffel Tower finished in 1889.".to_string(),
//! ]);
//!
//! let engine = RatEngine::new(RatConfig::default(), generator, retriever);
//! let trace = engine
//!     .run("Where and when was the Eiffel Tower built?")
//!     .unwrap();
//!
//! assert_eq!(trace.thoughts.len(), 2);
//! // The first thought's revision is grounded by retrieval mentioning Paris.
//! assert!(trace.thoughts[0].revised.contains("Paris"));
//! assert!(trace.thoughts[0].was_revised());
//! assert!(!trace.final_answer.is_empty());
//! ```

pub mod engine;
pub mod types;

#[cfg(test)]
mod tests;

pub use engine::RatEngine;
pub use types::{
    MockRatGenerator, MockRatRetriever, RatConfig, RatError, RatGenerator, RatRetriever,
    RatThought, RatTrace,
};
