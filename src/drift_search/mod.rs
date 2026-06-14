//! `GraphRAG` DRIFT search (Microsoft, 2024).
//!
//! DRIFT — *Dynamic Reasoning and Inference with Flexible Traversal* — combines
//! coarse **global** community-level search with fine **local** entity-level
//! search, iterating from one to the other via dynamically generated follow-up
//! sub-queries. It bridges the gap between Microsoft `GraphRAG`'s standalone
//! global and local strategies: a global pass picks the most relevant
//! communities, follow-up queries are spun out from their salient terms, and a
//! local pass drills into the passages those communities own.
//!
//! This module is **self-contained**: the caller supplies pre-computed
//! [`CommunityReport`]s plus the passages they reference, and the engine indexes
//! them with a deterministic FNV-1a lexical embedding. No model is invoked.
//!
//! # Workflow
//!
//! 1. **Global** — match the query against every community summary and keep the
//!    top [`DriftConfig::top_communities`].
//! 2. **Follow-ups** — synthesize up to [`DriftConfig::follow_ups`] sub-queries
//!    from the salient terms and entities of the selected communities.
//! 3. **Local** — for each follow-up, match against the passages belonging to
//!    the selected communities, keeping [`DriftConfig::top_local`] per step.
//! 4. **Aggregate** — de-duplicate and order the retrieved passages and
//!    synthesize a [`DriftAnswer::summary`] from the selected reports.
//!
//! # Architecture
//!
//! | Component | Responsibility |
//! |-----------|----------------|
//! | [`CommunityReport`] | Caller-supplied community summary + entities + passages |
//! | [`DriftSearchEngine`] | Indexes reports/passages and runs the DRIFT workflow |
//! | [`DriftAnswer`] | The recorded steps, communities, context, and summary |
//!
//! # Quick start
//!
//! ```rust
//! # #[cfg(feature = "drift-search")] {
//! use oxirag::drift_search::{CommunityReport, DriftConfig, DriftSearchEngine};
//! use oxirag::types::{Document, DocumentId};
//!
//! let passages = vec![
//!     Document::new("Rust guarantees memory safety without a garbage collector.")
//!         .with_id(DocumentId::from_string("p1")),
//!     Document::new("Ownership and borrowing enforce safety at compile time.")
//!         .with_id(DocumentId::from_string("p2")),
//! ];
//! let reports = vec![CommunityReport::new(
//!     0,
//!     "The Rust safety community covers ownership, borrowing, and memory safety.",
//!     vec!["Rust".into(), "ownership".into()],
//!     vec![DocumentId::from_string("p1"), DocumentId::from_string("p2")],
//! )];
//!
//! let mut engine = DriftSearchEngine::new(DriftConfig::default());
//! engine.build(reports, &passages).unwrap();
//! let answer = engine.search("How does Rust ensure safety?").unwrap();
//! assert!(!answer.final_context.is_empty());
//! # }
//! ```

pub mod engine;
#[cfg(test)]
mod tests;
pub mod types;

pub use engine::DriftSearchEngine;
pub use types::{CommunityReport, DriftAnswer, DriftConfig, DriftError, DriftStep, DriftStepKind};
