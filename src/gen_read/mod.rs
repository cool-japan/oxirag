//! `GenRead` — generate-then-read context construction.
//!
//! Implements *`GenRead`* (Yu et al., 2023, "Generate rather than Retrieve:
//! Large Language Models are Strong Context Generators"). Instead of retrieving
//! passages from a corpus, `GenRead` prompts a generator to **write** several
//! diverse *contextual documents* for the query, **clusters** them for
//! diversity, picks **one representative per cluster**, and assembles the
//! representatives into a reading context that downstream readers consume.
//!
//! This module is **distinct** from `HyDE` (`advanced_retrieval::hyde`): `HyDE`
//! generates a *single* hypothetical document whose dense *embedding* drives a
//! similarity search, so its output is still a set of *retrieved* passages.
//! `GenRead` never touches a corpus — the *generated* documents themselves, after
//! clustering-based de-duplication, become the context.
//!
//! # Pipeline
//!
//! | Stage | Responsibility |
//! |-------|----------------|
//! | [`ContextGenerator`] | Generate `num_docs` diverse contextual documents |
//! | clustering | Group the documents into `<= num_clusters` diverse clusters |
//! | representative selection | Pick the most-central document of each cluster |
//! | [`GenReadEngine::run`] | Join the representatives into the reading context |
//!
//! # Example
//!
//! ```
//! use oxirag::gen_read::{GenReadConfig, GenReadEngine, MockContextGenerator};
//!
//! let generator = MockContextGenerator::new(vec![
//!     "Rust is a systems programming language focused on safety.".to_string(),
//!     "Rust guarantees memory safety without a garbage collector.".to_string(),
//!     "The borrow checker enforces ownership rules at compile time.".to_string(),
//!     "Cargo is the Rust build tool and package manager.".to_string(),
//!     "Crates are the unit of compilation and distribution in Rust.".to_string(),
//! ]);
//! let engine = GenReadEngine::new(GenReadConfig::new().with_num_clusters(2));
//! let out = engine.run("what is rust?", &generator).unwrap();
//!
//! assert_eq!(out.documents.len(), 5);
//! assert!(out.clusters <= 2);
//! assert!(!out.context.is_empty());
//! ```

pub mod engine;
pub mod types;

#[cfg(all(test, not(target_arch = "wasm32")))]
mod tests;

pub use engine::GenReadEngine;
pub use types::{
    ContextGenerator, GenReadConfig, GenReadError, GenReadOutput, GeneratedDoc,
    MockContextGenerator,
};
