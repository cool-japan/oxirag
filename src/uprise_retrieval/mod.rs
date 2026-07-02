//! UPRISE-style universal prompt retrieval.
//!
//! Implements the retrieval side of Cheng et al. 2023, *"UPRISE: Universal
//! Prompt Retrieval for Improving Zero-Shot Evaluation"*: a single pool of
//! reusable prompt exemplars spanning **many different tasks** is queried by
//! embedding similarity, and candidates are then **reranked by a weighted
//! combination of similarity and historical outcome quality** rather than by
//! similarity alone.
//!
//! # Relationship to `prompt_optimization` and `self_query`
//!
//! [`crate::prompt_optimization`] selects few-shot demonstrations from a
//! **fixed, single-task** pool (its `DemoPool` is meant to hold examples for
//! *one* task, and `DemoSelector` never reasons about historical outcomes —
//! only similarity/diversity/quality at selection time). [`crate::self_query`]
//! is not exemplar retrieval at all: it parses a query into a structured
//! metadata filter.
//!
//! `uprise_retrieval` is neither: its [`UpriseIndex`] deliberately mixes
//! exemplars from unrelated tasks in one pool, and [`UpriseRetriever`] is
//! explicitly willing to serve a query for task *X* with an exemplar
//! authored for task *Y* whenever that exemplar's `prompt_text` is
//! semantically close to the query — the [`PromptExemplar::task_label`] is
//! never used to gate a match, only (optionally) to *exclude* same-task reuse
//! via [`UpriseRetriever::retrieve_excluding_task`]. On top of that,
//! candidates are reranked by a config-weighted combination of similarity and
//! each exemplar's running [`outcome_quality`](PromptExemplar::outcome_quality)
//! — a signal fed back over time via an exponential moving average (see
//! [`UpriseIndex::update_outcome`]) — so exemplars with a strong track record
//! can outrank merely-more-similar ones.
//!
//! # Pipeline
//!
//! 1. [`UpriseIndex`] accumulates [`PromptExemplar`]s — `(task_label,
//!    prompt_text, outcome_quality)` triples — from any number of tasks.
//! 2. [`UpriseRetriever::retrieve`] embeds the query with the deterministic
//!    FNV-1a lexical pseudo-embedding used across `OxiRAG`, computes cosine
//!    similarity against every eligible exemplar's (possibly lazily
//!    computed) embedding, and combines that similarity with the exemplar's
//!    `outcome_quality` via [`UpriseConfig::combined_score`]:
//!    `similarity_weight * similarity + outcome_weight * outcome_quality`.
//!    The top [`UpriseConfig::top_k`] exemplars by that combined score are
//!    returned as [`UpriseHit`]s.
//! 3. After a retrieved exemplar is actually used downstream, feed the
//!    observed result back with [`UpriseIndex::update_outcome`], which
//!    applies an EMA update (`alpha * new_signal + (1 - alpha) *
//!    old_quality`) so the pool's quality signal improves over time without
//!    letting a single noisy observation overwrite history.
//!
//! # Determinism
//!
//! Everything here is pure, allocation-light `std`-only code: the
//! pseudo-embedding is a deterministic FNV-1a bucket histogram (no `rand`,
//! no ML runtime), and ties in the final ranking are broken by a stable sort
//! that preserves pool insertion order.
//!
//! # Quick start
//!
//! ```rust
//! # #[cfg(feature = "uprise-retrieval")]
//! # {
//! use oxirag::uprise_retrieval::{PromptExemplar, UpriseConfig, UpriseIndex, UpriseRetriever};
//!
//! let mut index = UpriseIndex::new();
//! // A high-performing QA exemplar...
//! index.add(
//!     PromptExemplar::new("qa", "Answer the question concisely and factually.", 0.9).unwrap(),
//! );
//! // ...and an unrelated, lower-performing summarization exemplar.
//! index.add(
//!     PromptExemplar::new("summarization", "Summarize the passage in one sentence.", 0.4)
//!         .unwrap(),
//! );
//!
//! let retriever = UpriseRetriever::new(UpriseConfig::default().with_top_k(1));
//! let hits = retriever
//!     .retrieve("What is the capital of France?", &index)
//!     .unwrap();
//! assert_eq!(hits.len(), 1);
//! assert_eq!(hits[0].task_label, "qa");
//! # }
//! ```

pub mod retriever;
#[cfg(test)]
mod tests;
pub mod types;

pub use retriever::{UpriseIndex, UpriseRetriever};
pub use types::{PromptExemplar, UpriseConfig, UpriseError, UpriseHit};
