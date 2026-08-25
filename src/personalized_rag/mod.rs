//! Personalized RAG — reranking retrieval results with a per-user profile.
//!
//! Where [`relevance_feedback`](crate::relevance_feedback) adjusts a *query* with
//! the classic Rocchio algorithm from explicit per-result judgements, this module
//! personalizes ranking for a *user* across queries. A [`UserProfile`] carries
//! two long-lived signals — explicit topic-interest weights and an interaction
//! history — and the [`PersonalizedReranker`] blends each result's original
//! relevance with a *personal affinity* derived from that profile.
//!
//! # Affinity
//!
//! Affinity is a deterministic value in `[0, 1]` combining:
//!
//! * a **topic-interest match** — the weighted fraction of the user's interest
//!   topics that appear among the document's tokens, and
//! * a **history similarity** — the cosine similarity between the document's
//!   FNV-1a lexical embedding and the centroid of the user's interaction history.
//!
//! # Blending
//!
//! The final score is
//! `(1 - personalization_weight) * relevance + personalization_weight * affinity`.
//! With `personalization_weight = 0.0` the original ordering is preserved
//! exactly; with a high weight, a profile-aligned document can overtake a more
//! relevant but off-profile one.
//!
//! # Architecture
//!
//! | Component | Responsibility |
//! |-----------|----------------|
//! | [`UserProfile`] | Topic-interest weights + interaction history |
//! | [`PersonalizedConfig`] | Blend factor and embedding dimensionality |
//! | [`PersonalizedReranker`] | Computes affinity and reranks results |
//!
//! # Quick start
//!
//! ```rust,ignore
//! # #[cfg(feature = "personalized-rag")] {
//! use oxirag::personalized_rag::{PersonalizedConfig, PersonalizedReranker, UserProfile};
//!
//! let profile = UserProfile::new().with_interest("rust", 1.0);
//! let reranker = PersonalizedReranker::new(PersonalizedConfig::default());
//! let ranked = reranker.rerank(&profile, &results);
//! # let results: Vec<oxirag::types::SearchResult> = Vec::new();
//! # let _ = ranked;
//! # }
//! ```

pub mod reranker;
#[cfg(all(test, not(target_arch = "wasm32")))]
mod tests;
pub mod types;

pub use reranker::PersonalizedReranker;
pub use types::{PersonalizedConfig, PersonalizedError, UserProfile};
