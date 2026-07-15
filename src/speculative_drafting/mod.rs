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
//! Two paper-faithful capabilities are available **opt-in**, each defaulting
//! to today's behavior so existing callers see no change:
//!
//! - **Paper subset sampling.** [`DraftSubsetStrategy::OneRepresentativePerCluster`]
//!   forms `M` subsets, each holding one representative document sampled from
//!   *every* cluster — rather than one draft per whole cluster — matching the
//!   paper's own sampling scheme. Select it with
//!   [`SpecDraftConfig::with_subset_strategy`]; the default remains
//!   [`DraftSubsetStrategy::PerCluster`].
//! - **Self-reflection.** A [`Drafter`] that overrides
//!   [`Drafter::draft_with_rationale`] attaches a rationale to its draft; the
//!   verifier's [`DraftVerifier::reflect`] then scores a rationale-conditioned
//!   confidence `ρ_SR`, and the total score becomes `ρ_SC * ρ_SR` — the
//!   paper's verifier score — instead of the plain support/consistency blend
//!   `ρ_SC` alone. A drafter that never supplies a rationale (the default)
//!   leaves the score exactly as it was.
//!
//! This module is **distinct** from `self_consistency`: that module samples
//! diverse *reasoning paths* from a single query and marginalizes over them,
//! whereas here the diversity comes from *disjoint document clusters* — each
//! draft is grounded in a different slice of the evidence. It is also
//! **distinct** from [`layer2_speculator`](crate::layer2_speculator), which
//! performs *token-level* draft-and-verify speculative decoding (a small
//! model proposes tokens, a large model verifies them); this module performs
//! *document-level* draft-and-verify retrieval-augmented generation.
//!
//! # Pipeline
//!
//! | Stage | Responsibility |
//! |-------|----------------|
//! | [`SpeculativeDrafter::cluster_docs`] | Partition the corpus into `<= num_clusters` diverse groups |
//! | [`SpeculativeDrafter::draft_groups`] | One group per cluster, or `M` cross-cluster representative subsets (see [`DraftSubsetStrategy`]) |
//! | [`Drafter`] | Draft one answer per group, optionally with a rationale (run in parallel) |
//! | [`DraftVerifier`] | Score how well each draft is supported by its docs, and — given a rationale — self-reflection confidence |
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
    DraftCandidate, DraftSubsetStrategy, DraftVerifier, Drafter, MockDraftVerifier, MockDrafter,
    SpecDraftConfig, SpecDraftError, SpeculativeOutput,
};
