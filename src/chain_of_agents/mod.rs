//! Chain-of-Agents: sequential multi-agent long-context processing (Zhang et
//! al., Google, 2024).
//!
//! A document too long to fit in one context window is split into chunks.
//! Then:
//!
//! 1. **Worker agents run sequentially, one per chunk.** Worker `i` receives
//!    `(query, chunk_i, communication_unit_from_worker_{i-1})` and returns an
//!    *updated* communication unit — see [`CoaWorker::process`]. Worker
//!    `i + 1` sees exactly what worker `i` returned. Information therefore
//!    flows **forward** along the chain: a fact surfaced while reading chunk
//!    2 is available to the worker reading chunk 7, five chunks later,
//!    without either worker ever holding the whole document in context.
//! 2. **A manager agent** then reads the *final* communication unit — and
//!    only the final communication unit, never a raw chunk — and produces
//!    the answer; see [`CoaManager::synthesize`].
//!
//! The defining property is the **sequential, forward-flowing communication
//! unit** ([`CoaCommunicationUnit`]): a real structured accumulator of
//! query-relevant evidence, a running partial answer, unresolved open
//! questions, and a completeness signal — bounded by
//! [`CoaConfig::evidence_budget`] / [`CoaConfig::open_question_budget`] so it
//! never grows without limit as the chain lengthens (see
//! [`CoaCommunicationUnit::merge_evidence`] for the exact eviction rule).
//!
//! # How this differs from every other multi-stage / multi-agent module here
//!
//! Several modules in this crate combine multiple pieces of generated text
//! into one output, or run more than one agent — but none of them has a
//! worker that reads what an *earlier* worker produced before it runs:
//!
//! | Module | Structure | Does information flow between stages/agents? |
//! |---|---|---|
//! | [`skeleton_of_thought`](crate::skeleton_of_thought) | Two-stage: outline, then expand each point | **No** — [`skeleton_of_thought::PointExpander`](crate::skeleton_of_thought::PointExpander) expands every skeleton point *independently*; no state passes between expansions |
//! | [`graph_summarization`](crate::graph_summarization) (`GlobalSearchEngine`) | Explicit map-reduce over community summaries | **No** — every summary is scored independently ("Map"), then reduced once at the end; no summary's scoring ever sees another summary's output |
//! | [`long_rag`](crate::long_rag) | Groups documents into long retrieval units by lexical/semantic clustering | **N/A** — not an agent workflow at all; there is no worker, no manager, and nothing sequential about it |
//! | [`pipeline_composer`](crate::pipeline_composer) | Linear chain of [`pipeline_composer::PipelineStage`](crate::pipeline_composer::PipelineStage)s | **Plumbing only** — stages are non-agentic deterministic text transforms threaded in sequence; there is no evolving *agent* state, no evidence accumulation, and no notion of a chunk of a longer input being read by successive stages |
//! | `chain_of_agents` (this module) | Sequential worker chain + manager | **Yes, by construction** — [`CoaWorker::process`]'s `incoming` parameter *is* everything the chain has learned so far; [`CoaEngine::run_chunks`] threads it worker-to-worker in strict chunk order, and reversing that order changes the result (see the module's tests) |
//!
//! `chain_of_agents` is the only module here where the very definition of
//! correctness depends on *order*: a fact from chunk 1 must still be visible
//! when chunk 9 is processed, which requires an accumulating, forward-passed
//! state rather than independent (parallel-safe) per-item processing.
//!
//! # The local chunk splitter
//!
//! This module does not depend on [`crate::chunking`] (gated behind the
//! separate `chunking` feature, which `chain-of-agents` does not enable) and
//! instead ships a small local sentence-packing splitter — see the
//! [`engine`] module's documentation for why, and [`CoaEngine::run_chunks`]
//! for the lower-level entry point that accepts pre-split chunks from any
//! source, including [`crate::chunking`] when that feature is enabled.
//!
//! # Quick start
//!
//! ```rust
//! # #[cfg(feature = "chain-of-agents")]
//! # {
//! use oxirag::chain_of_agents::{CoaConfig, CoaEngine, CoaLexicalManager, CoaLexicalWorker};
//!
//! // Chunk 0 introduces a rename; chunk 2 (two chunks later) states a fact
//! // only about the *new* name. Answering the query correctly requires
//! // carrying the rename forward across the intervening chunk.
//! let chunks = vec![
//!     "The Kepler probe was renamed Artemis.".to_string(),
//!     "Meanwhile, funding for unrelated lunar telescopes increased.".to_string(),
//!     "Artemis launched in 2031.".to_string(),
//! ];
//!
//! let engine = CoaEngine::new(CoaConfig::default());
//! let trace = engine
//!     .run_chunks(
//!         "When did the Kepler probe launch?",
//!         &chunks,
//!         &CoaLexicalWorker,
//!         &CoaLexicalManager,
//!     )
//!     .expect("well-formed chain-of-agents run");
//!
//! assert_eq!(trace.steps.len(), 3);
//! assert!(trace.answer.contains("Kepler") && trace.answer.contains("2031"));
//! # }
//! ```

pub mod engine;
pub mod types;
pub mod unit;
pub mod worker;

#[cfg(all(test, not(target_arch = "wasm32")))]
mod tests;

pub use engine::CoaEngine;
pub use types::{CoaConfig, CoaError, CoaEvidence, CoaTrace, CoaWorkerStep, MIN_CHUNK_SIZE};
pub use unit::CoaCommunicationUnit;
pub use worker::{CoaLexicalManager, CoaLexicalWorker, CoaManager, CoaWorker};
