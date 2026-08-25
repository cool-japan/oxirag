//! DRAGIN (Su et al. 2024) — *Dynamic Retrieval Augmented Generation based on
//! the real-time Information Need of Large Language Models*.
//!
//! DRAGIN decides **when** and **what** to retrieve from the model's own
//! token-level signals during generation:
//!
//! - **RIND** (Real-time Information Need Detection) fires retrieval the instant
//!   the model is about to emit an **uncertain content token** — a token whose
//!   confidence is below a threshold and which actually carries information
//!   (stopwords are ignored, since low confidence on "the" is not an
//!   information need).
//! - **QFS** (Query Formulation by Self-attention) builds the retrieval query
//!   from the **salient content tokens** of the recently generated context,
//!   rather than blindly reusing the original question. The self-attention
//!   signal is approximated by content-token salience within a bounded window.
//!
//! This is distinct from FLARE in `retrieval_loop`: FLARE triggers on the
//! confidence of the *next sentence* and reuses the look-ahead draft as the
//! query, whereas DRAGIN triggers at **token** granularity and forms its query
//! from attention-salient tokens.
//!
//! The generator and retriever are supplied by the caller as the
//! [`UncertaintyGenerator`] and [`Retriever`] traits. Deterministic
//! [`MockUncertaintyGenerator`] and [`MockRetriever`] implementations are
//! provided for tests.
//!
//! # Example
//!
//! ```
//! use oxirag::dragin::{
//!     DraginConfig, DraginEngine, MockRetriever, MockUncertaintyGenerator, TokenInfo,
//! };
//!
//! // The first segment ends on a low-confidence content token ("Nolan"),
//! // which triggers a retrieval; the second segment is all confident.
//! let generator = MockUncertaintyGenerator::new(vec![
//!     vec![
//!         TokenInfo::new("Inception", 0.9),
//!         TokenInfo::new("was", 0.9),
//!         TokenInfo::new("directed", 0.9),
//!         TokenInfo::new("by", 0.9),
//!         TokenInfo::new("Nolan", 0.2),
//!     ],
//!     vec![TokenInfo::new("indeed", 0.95)],
//! ]);
//! let retriever = MockRetriever::new(vec![(
//!     "directed".to_string(),
//!     vec!["Christopher Nolan directed Inception.".to_string()],
//! )]);
//!
//! let engine = DraginEngine::new(DraginConfig::default());
//! let trace = engine.run("Who directed Inception?", &generator, &retriever).unwrap();
//!
//! assert_eq!(trace.num_retrievals, 1);
//! assert_eq!(trace.triggers[0].trigger_token, "Nolan");
//! assert!(trace.generated.contains("Inception"));
//! ```

pub mod engine;
#[cfg(all(test, not(target_arch = "wasm32")))]
mod tests;
pub mod types;

pub use engine::DraginEngine;
pub use types::{
    DraginConfig, DraginError, DraginTrace, MockRetriever, MockUncertaintyGenerator,
    RetrievalTrigger, Retriever, TokenInfo, UncertaintyGenerator,
};
