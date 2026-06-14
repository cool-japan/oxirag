//! Speculative RAG — draft from diverse document clusters, then verify.
//!
//! Implements *Speculative RAG* (Wang et al., 2024, "Speculative RAG: Enhancing
//! Retrieval Augmented Generation through Drafting"). Rather than asking one
//! generator to reason over the whole retrieved corpus, the pipeline first
//! **clusters** the documents into diverse subsets, generates **one draft
//! answer per cluster in parallel** (so each draft reflects a distinct
//! perspective), then **verifies** every draft (a support confidence score) and
//! combines it with **self-consistency** (agreement across the drafts). The
//! highest-scoring draft is returned.
//!
//! This module is **distinct** from `self_consistency`: that module samples
//! diverse *reasoning paths* from a single query and marginalizes over them,
//! whereas here the diversity comes from *disjoint document clusters* — each
//! draft is grounded in a different slice of the evidence.
//!
//! # Pipeline
//!
//! | Stage | Responsibility |
//! |-------|----------------|
//! | [`SpeculativeDrafter::cluster_docs`] | Partition the corpus into `<= num_clusters` diverse groups |
//! | [`Drafter`] | Draft one answer per cluster (run in parallel) |
//! | [`DraftVerifier`] | Score how well each draft is supported by its docs |
//! | self-consistency | Mean token-Jaccard agreement across drafts |
//! | [`SpeculativeDrafter::run`] | Blend the scores and pick the best draft |
//!
//! # Example
//!
//! ```
//! use oxirag::speculative_drafting::{
//!     MockDraftVerifier, MockDrafter, SpecDraftConfig, SpeculativeDrafter,
//! };
//! use oxirag::types::Document;
//!
//! let docs = vec![
//!     Document::new("Photosynthesis converts sunlight into chemical energy in plants."),
//!     Document::new("Chlorophyll in the chloroplast captures light for photosynthesis."),
//!     Document::new("The mitochondria produce ATP through cellular respiration."),
//! ];
//! let drafter = SpeculativeDrafter::new(SpecDraftConfig::new().with_num_clusters(2));
//! let out = drafter
//!     .run("how do plants make energy?", &docs, &MockDrafter::new(), &MockDraftVerifier::new())
//!     .unwrap();
//! assert!(!out.best.is_empty());
//! assert!((0.0..=1.0).contains(&out.confidence));
//! ```

pub mod cluster;
pub mod drafter;
pub mod types;

#[cfg(test)]
mod tests;

pub use drafter::SpeculativeDrafter;
pub use types::{
    DraftCandidate, DraftVerifier, Drafter, MockDraftVerifier, MockDrafter, SpecDraftConfig,
    SpecDraftError, SpeculativeOutput,
};
