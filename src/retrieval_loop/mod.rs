//! FLARE: Forward-Looking Active `REtrieval` augmented generation.
//!
//! This module implements the iterative generate-retrieve loop described by
//! Jiang et al. (2023) — "Active Retrieval Augmented Generation".
//!
//! # Algorithm overview
//!
//! ```text
//! ┌─────────────────────────────────────────────────────────────────────┐
//! │  FLARE Loop                                                         │
//! │                                                                     │
//! │  1. Generate a short draft (2 sentences)                           │
//! │     using the current context window.                              │
//! │                                                                     │
//! │  2. Score each sentence against the context via a frequency-based  │
//! │     heuristic (proxy for LLM token log-probabilities).             │
//! │                                                                     │
//! │  3. If the first uncertain sentence has a span long enough to be   │
//! │     a meaningful query:                                             │
//! │       a. Optionally prepend the original query (query augment).    │
//! │       b. Retrieve the top-k documents from the retriever backend.  │
//! │       c. Add the documents to the bounded context window.          │
//! │       d. Regenerate a full answer with the enriched context.       │
//! │                                                                     │
//! │  4. Repeat until all sentences are confident or max_iterations     │
//! │     is reached.                                                     │
//! └─────────────────────────────────────────────────────────────────────┘
//! ```
//!
//! # Quick start
//!
//! ```rust
//! # #[cfg(feature = "flare")]
//! # {
//! use oxirag::retrieval_loop::{
//!     engine::FlareEngine,
//!     generator::MockFlareGenerator,
//!     retriever::MockFlareRetriever,
//!     types::{FlareConfig, ContextDoc},
//! };
//!
//! # #[tokio::main]
//! # async fn main() {
//! // 1. Create a mock generator (replace with a real LLM adapter in production).
//! let generator = MockFlareGenerator::new_single(
//!     "Rust is a systems programming language.".to_string()
//! );
//!
//! // 2. Create a mock retriever (replace with your vector store adapter).
//! let retriever = MockFlareRetriever::new(vec![
//!     ContextDoc::new("Rust prioritises memory safety.", 0.9, "doc-1"),
//! ]);
//!
//! // 3. Configure the engine.
//! let config = FlareConfig::default()
//!     .with_max_iterations(5)
//!     .with_confidence_threshold(0.5);
//!
//! // 4. Run the loop.
//! let engine = FlareEngine::new(generator, retriever, config);
//! let output = engine.run_simple("What is Rust?").await.unwrap();
//!
//! println!("Answer: {}", output.final_answer);
//! println!("Iterations: {}", output.iterations.len());
//! println!("Retrieval rate: {:.1}%", output.retrieval_rate() * 100.0);
//! # }
//! # }
//! ```

pub mod confidence;
pub mod engine;
pub mod generator;
pub mod retriever;
pub mod types;

#[cfg(test)]
mod tests;

pub use confidence::ConfidenceEstimator;
pub use engine::FlareEngine;
pub use generator::{FlareGenerator, MockFlareGenerator, TemplateGenerator};
pub use retriever::{FlareRetriever, MockFlareRetriever, QueryAugmentedRetriever};
pub use types::{
    ContextDoc, ContextWindow, FlareConfig, FlareError, FlareOutput, IterationRecord,
    SentenceConfidence, TokenConfidence,
};
