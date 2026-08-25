//! Multi-turn conversational RAG with history management and query reformulation.
//!
//! This module provides a complete, production-grade framework for building
//! multi-turn RAG applications. It handles:
//!
//! - **Session lifecycle** — create, update, and delete conversation sessions
//!   with configurable history limits and optional rolling summaries.
//! - **History buffering** — four buffer strategies (full, sliding-window,
//!   summary, hybrid) that convert conversation history into a context string.
//! - **Query reformulation** — four strategies that transform a raw user query
//!   into a context-enriched form that the retrieval layer can use directly.
//! - **Follow-up detection** — heuristic analysis of pronoun density and query
//!   length to identify queries that are continuations of the current thread.
//!
//! # Quick start
//!
//! ```rust
//! # #[cfg(feature = "conversational")]
//! # {
//! # tokio_test::block_on(async {
//! use std::sync::Arc;
//! use oxirag::conversation::buffer::SlidingWindowBuffer;
//! use oxirag::conversation::reformulator::{QueryReformulator, ReformulationStrategy};
//! use oxirag::conversation::session::{ConversationalPipeline, InMemorySessionManager};
//! use oxirag::conversation::types::SessionConfig;
//!
//! // Build the pipeline.
//! let mgr = Arc::new(InMemorySessionManager::new_default());
//! let buf = SlidingWindowBuffer::new(6);
//! let reform = QueryReformulator::new(ReformulationStrategy::FollowUpResolution);
//! let pipeline = ConversationalPipeline::new(mgr, buf, reform);
//!
//! // Create a session for the user.
//! let session_id = pipeline.create_session(SessionConfig::default()).await.unwrap();
//!
//! // Process the first turn.
//! let aware = pipeline.process_turn(&session_id, "What is Rust?").await.unwrap();
//! // `aware.as_search_text()` is ready for retrieval.
//!
//! // After obtaining the answer, record both sides of the turn.
//! pipeline.record_user_turn(&session_id, "What is Rust?").await.unwrap();
//! pipeline.record_response(&session_id, "Rust is a systems programming language.").await.unwrap();
//!
//! // Process a follow-up turn.
//! let aware2 = pipeline.process_turn(&session_id, "Tell me more.").await.unwrap();
//! assert!(aware2.is_follow_up);
//! # });
//! # }
//! ```
//!
//! # Module layout
//!
//! | Sub-module | Purpose |
//! |---|---|
//! | [`types`] | Core data types: `Turn`, `Session`, `ConversationHistory`, errors |
//! | [`buffer`] | History → context string strategies |
//! | [`reformulator`] | Query reformulation strategies and follow-up detection |
//! | [`session`] | Session management and the `ConversationalPipeline` |

pub mod buffer;
pub mod reformulator;
pub mod session;
pub mod types;

#[cfg(all(test, not(target_arch = "wasm32")))]
mod tests;

// ── Convenience re-exports ────────────────────────────────────────────────────

pub use buffer::{
    FullHistoryBuffer, HistoryBuffer, HybridBuffer, SlidingWindowBuffer, SummaryBuffer,
};
pub use reformulator::{
    ConversationAwareQuery, FollowUpDetector, QueryReformulator, ReformulationStrategy,
};
pub use session::{ConversationalPipeline, InMemorySessionManager, SessionManager};
pub use types::{
    ConversationError, ConversationHistory, ConversationId, Session, SessionConfig, Turn, TurnRole,
};
