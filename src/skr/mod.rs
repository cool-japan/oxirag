//! SKR — Self-Knowledge guided Retrieval augmentation (Wang et al. 2023,
//! "Self-Knowledge Guided Retrieval Augmentation for Large Language Models").
//!
//! SKR is a **binary retrieve-or-not gate**: for each incoming question it
//! decides whether the answering model already knows enough to answer
//! directly ([`SkrRetrievalChoice::Skip`]) or should be augmented with
//! retrieval ([`SkrRetrievalChoice::Retrieve`]). Unlike a hand-written
//! heuristic, the gate is **memory-based**: it consults an accumulated
//! [`SelfKnowledgePool`] of past `(question, was-answerable-without-
//! retrieval)` exemplars and takes a similarity-weighted vote of the `k`
//! nearest ones — a k-nearest-neighbor classifier over the model's own
//! track record, not a fixed rule.
//!
//! # Distinct from `adaptive_rag` and `self_route`
//!
//! Three modules in this crate all decide *something* about retrieval before
//! or during answering a query, but they decide different things from
//! different evidence:
//!
//! - `adaptive_rag` is **stateless**: a query-*complexity* heuristic (surface
//!   features of the question text alone — clause count, comparison words,
//!   multi-hop phrasing, …) classifies each query into one of three
//!   complexity tiers, and that tier selects the retrieval **depth**: no
//!   retrieval, one pass, or iterative multi-hop. It never looks at any prior
//!   question or outcome, and it never looks at retrieved context — its
//!   decision is a pure function of the query text.
//! - `self_route` inspects the **retrieved context** for a query that has
//!   *already* been retrieved, and decides whether that context is
//!   sufficient to answer cheaply with RAG or whether the query needs a
//!   long-context fallback. It answers "was retrieval enough?", strictly
//!   *after* retrieval has already happened.
//! - `skr` (this module) instead asks "does *this* model need retrieval *at
//!   all* for *this* question?", **before** any retrieval happens, and
//!   answers it from **accumulated self-knowledge**: a growing memory of
//!   which past questions this model could answer unaided and which it
//!   could not. It is stateful (the pool grows via
//!   [`SkrGate::observe_outcome`], so the gate's judgement improves with
//!   experience) and it never inspects retrieved context or query-surface
//!   complexity — only similarity to remembered self-knowledge exemplars.
//!
//! The three are complementary and can be layered: `skr` decides *whether*
//! to retrieve at all; if it says retrieve, `adaptive_rag` can decide *how
//! deep*; and once results come back, `self_route` can decide whether they
//! were *enough*.
//!
//! # Algorithm
//!
//! 1. **Pool.** A [`SelfKnowledgePool`] holds [`SkrExemplar`]s — questions
//!    labelled `answerable_without_retrieval: true` ("I know this") or
//!    `false` ("I don't know this") — each stored with a deterministic
//!    FNV-1a bucket-histogram pseudo-embedding of its question text.
//! 2. **Gate.** [`SkrGate::decide`] embeds the incoming question, finds the
//!    [`SkrConfig::k`] nearest pool exemplars by cosine similarity
//!    ([`SkrNeighbor`]s), and combines their labels into a single "known"
//!    score in `[0.0, 1.0]` — a similarity-weighted vote by default
//!    ([`SkrConfig::weight_by_similarity`]), or a plain majority vote when
//!    that is disabled.
//! 3. **Threshold.** A known score at or above
//!    [`SkrConfig::decision_threshold`] returns
//!    [`SkrRetrievalChoice::Skip`]; below it returns
//!    [`SkrRetrievalChoice::Retrieve`]. An empty pool has no evidence to vote
//!    with, so it falls back to the configurable
//!    [`SkrConfig::default_choice`] (default: `Retrieve`) instead of
//!    guessing.
//! 4. **Feedback loop.** [`SkrGate::observe_outcome`] (and the lower-level
//!    [`SkrGate::add_labeled`] / [`SkrGate::add_exemplar`]) append fresh
//!    datapoints to the pool as new questions are actually answered, so
//!    later decisions reflect the model's *current* self-knowledge. Once
//!    [`SkrConfig::pool_cap`] is reached, the oldest exemplars are evicted
//!    first (FIFO), keeping the pool bounded.
//!
//! # Quick start
//!
//! ```rust,ignore
//! use oxirag::skr::{SkrConfig, SkrGate, SkrRetrievalChoice};
//!
//! let mut gate = SkrGate::new(SkrConfig::default())?;
//!
//! // Seed the pool with self-knowledge from past interactions.
//! gate.add_labeled("What is the capital of France?", true)?;
//! gate.add_labeled("What is the boiling point of water at sea level?", true)?;
//! gate.add_labeled("What did our Q3 internal audit conclude?", false)?;
//!
//! // A question close to the "known" exemplars skips retrieval...
//! let decision = gate.decide("What is the capital of Germany?")?;
//! assert_eq!(decision.choice, SkrRetrievalChoice::Skip);
//!
//! // ...while one close to the "unknown" exemplar requires it.
//! let decision = gate.decide("What did our Q4 internal audit conclude?")?;
//! assert_eq!(decision.choice, SkrRetrievalChoice::Retrieve);
//!
//! // Feed real outcomes back in as they are observed, growing the pool.
//! gate.observe_outcome("What did our Q4 internal audit conclude?", false)?;
//! # Ok::<(), oxirag::skr::SkrError>(())
//! ```

pub mod engine;
#[cfg(test)]
mod tests;
pub mod types;

pub use engine::{SelfKnowledgePool, SkrGate};
pub use types::{
    SkrConfig, SkrDecision, SkrError, SkrExemplar, SkrNeighbor, SkrResult, SkrRetrievalChoice,
};
