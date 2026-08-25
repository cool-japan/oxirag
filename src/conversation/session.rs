//! Session management and the conversational pipeline.
//!
//! This module provides:
//!
//! - [`SessionManager`]: an async trait for CRUD operations on [`Session`]s.
//! - [`InMemorySessionManager`]: an in-memory implementation backed by a
//!   `tokio::sync::RwLock<HashMap<ConversationId, Session>>`.
//! - [`ConversationalPipeline`]: a high-level wrapper that ties a
//!   `SessionManager`, a [`HistoryBuffer`], and a [`QueryReformulator`]
//!   together to process individual turns.

use std::collections::HashMap;
use std::sync::Arc;
use std::time::SystemTime;

use crate::sync::RwLock;
use async_trait::async_trait;
use tracing::{debug, instrument, warn};

use super::buffer::HistoryBuffer;
use super::reformulator::{ConversationAwareQuery, FollowUpDetector, QueryReformulator};
use super::types::{ConversationError, ConversationId, Session, SessionConfig, Turn, TurnRole};

// ── SessionManager ────────────────────────────────────────────────────────────

/// Async CRUD interface for conversation sessions.
///
/// Implementors are required to be `Send + Sync` so they can be shared
/// across threads in a Tokio multi-threaded runtime.
#[cfg_attr(not(target_arch = "wasm32"), async_trait)]
#[cfg_attr(target_arch = "wasm32", async_trait(?Send))]
pub trait SessionManager: Send + Sync {
    /// Create a new session with the given configuration and return its ID.
    async fn create_session(
        &self,
        config: SessionConfig,
    ) -> Result<ConversationId, ConversationError>;

    /// Retrieve a session by ID, returning a clone of the current state.
    async fn get_session(&self, id: &ConversationId) -> Result<Session, ConversationError>;

    /// Append a turn to an existing session's history.
    ///
    /// If the session's `max_turns` limit is exceeded the oldest turns are
    /// automatically discarded.
    async fn update_session(
        &self,
        id: &ConversationId,
        turn: Turn,
    ) -> Result<(), ConversationError>;

    /// Remove a session permanently.
    async fn delete_session(&self, id: &ConversationId) -> Result<(), ConversationError>;

    /// Return the IDs of all live sessions in an unspecified order.
    async fn list_sessions(&self) -> Vec<ConversationId>;

    /// Return the total number of active sessions.
    async fn session_count(&self) -> usize;
}

// ── InMemorySessionManager ────────────────────────────────────────────────────

/// A `SessionManager` backed by an in-memory hash-map protected by a
/// `RwLock`.
///
/// When the number of sessions reaches `max_sessions`, the oldest session
/// (by `created_at`) is evicted to make room for the new one.
///
/// # Example
///
/// ```rust
/// # #[cfg(feature = "conversational")]
/// # {
/// # tokio_test::block_on(async {
/// use oxirag::conversation::session::{InMemorySessionManager, SessionManager};
/// use oxirag::conversation::types::SessionConfig;
///
/// let mgr = InMemorySessionManager::new_default();
/// let id = mgr.create_session(SessionConfig::default()).await.unwrap();
/// assert_eq!(mgr.session_count().await, 1);
/// mgr.delete_session(&id).await.unwrap();
/// assert_eq!(mgr.session_count().await, 0);
/// # });
/// # }
/// ```
#[derive(Debug)]
pub struct InMemorySessionManager {
    sessions: Arc<RwLock<HashMap<ConversationId, Session>>>,
    max_sessions: usize,
}

impl InMemorySessionManager {
    /// Create a manager with the given maximum number of sessions.
    #[must_use]
    pub fn new(max_sessions: usize) -> Self {
        Self {
            sessions: Arc::new(RwLock::new(HashMap::new())),
            max_sessions,
        }
    }

    /// Create a manager with a default cap of 1 000 sessions.
    #[must_use]
    pub fn new_default() -> Self {
        Self::new(1_000)
    }

    /// Evict the oldest session (by `created_at`) from the store.
    ///
    /// Called while holding the write lock.
    fn evict_oldest(sessions: &mut HashMap<ConversationId, Session>) {
        if sessions.is_empty() {
            return;
        }
        let oldest_id = sessions
            .iter()
            .min_by_key(|(_, s)| {
                s.created_at
                    .duration_since(SystemTime::UNIX_EPOCH)
                    .unwrap_or_default()
            })
            .map(|(id, _)| id.clone());

        if let Some(id) = oldest_id {
            warn!(session_id = %id, "Evicting oldest session to stay within max_sessions limit");
            sessions.remove(&id);
        }
    }
}

#[cfg_attr(not(target_arch = "wasm32"), async_trait)]
#[cfg_attr(target_arch = "wasm32", async_trait(?Send))]
impl SessionManager for InMemorySessionManager {
    #[instrument(skip(self, config), fields(max_sessions = self.max_sessions))]
    async fn create_session(
        &self,
        config: SessionConfig,
    ) -> Result<ConversationId, ConversationError> {
        let mut guard = self.sessions.write().await;

        if guard.len() >= self.max_sessions {
            // Evict the oldest session rather than failing hard.
            Self::evict_oldest(&mut guard);
        }

        let session = Session::new(config);
        let id = session.id.clone();
        guard.insert(id.clone(), session);
        debug!(session_id = %id, "Created new conversation session");
        Ok(id)
    }

    #[instrument(skip(self))]
    async fn get_session(&self, id: &ConversationId) -> Result<Session, ConversationError> {
        let guard = self.sessions.read().await;
        guard
            .get(id)
            .cloned()
            .ok_or_else(|| ConversationError::SessionNotFound(id.to_string()))
    }

    #[instrument(skip(self, turn))]
    async fn update_session(
        &self,
        id: &ConversationId,
        turn: Turn,
    ) -> Result<(), ConversationError> {
        let mut guard = self.sessions.write().await;
        let session = guard
            .get_mut(id)
            .ok_or_else(|| ConversationError::SessionNotFound(id.to_string()))?;

        session.add_turn(turn);
        debug!(
            session_id = %id,
            turns = session.turn_count(),
            "Appended turn to session"
        );
        Ok(())
    }

    #[instrument(skip(self))]
    async fn delete_session(&self, id: &ConversationId) -> Result<(), ConversationError> {
        let mut guard = self.sessions.write().await;
        if guard.remove(id).is_none() {
            return Err(ConversationError::SessionNotFound(id.to_string()));
        }
        debug!(session_id = %id, "Deleted conversation session");
        Ok(())
    }

    async fn list_sessions(&self) -> Vec<ConversationId> {
        let guard = self.sessions.read().await;
        guard.keys().cloned().collect()
    }

    async fn session_count(&self) -> usize {
        let guard = self.sessions.read().await;
        guard.len()
    }
}

// ── ConversationalPipeline ────────────────────────────────────────────────────

/// A high-level wrapper that drives one full turn of a multi-turn conversation.
///
/// It coordinates three components:
///
/// 1. A [`SessionManager`] `S` — persists conversation state.
/// 2. A [`HistoryBuffer`] `B` — converts history to a context string.
/// 3. A [`QueryReformulator`] — transforms the user query into a retrieval-ready
///    form.
///
/// # Workflow
///
/// ```text
/// process_turn(session_id, user_query)
///   ├── get_session(session_id)           → Session
///   ├── buffer.get_context(history)       → context_string
///   ├── FollowUpDetector::is_follow_up()  → bool
///   ├── reformulator.reformulate()        → reformulated_query
///   └── ConversationAwareQuery { … }
///
/// record_response(session_id, response)
///   └── update_session(session_id, Turn::Assistant(response))
/// ```
///
/// # Example
///
/// ```rust
/// # #[cfg(feature = "conversational")]
/// # {
/// # tokio_test::block_on(async {
/// use oxirag::conversation::buffer::SlidingWindowBuffer;
/// use oxirag::conversation::reformulator::{QueryReformulator, ReformulationStrategy};
/// use oxirag::conversation::session::{ConversationalPipeline, InMemorySessionManager};
/// use oxirag::conversation::types::SessionConfig;
///
/// let mgr = std::sync::Arc::new(InMemorySessionManager::new_default());
/// let buf = SlidingWindowBuffer::new(6);
/// let reform = QueryReformulator::new(ReformulationStrategy::FollowUpResolution);
/// let pipeline = ConversationalPipeline::new(mgr, buf, reform);
///
/// let id = pipeline.create_session(SessionConfig::default()).await.unwrap();
/// let aware = pipeline.process_turn(&id, "What is Rust?").await.unwrap();
/// assert_eq!(aware.original_query, "What is Rust?");
/// # });
/// # }
/// ```
pub struct ConversationalPipeline<S, B>
where
    S: SessionManager,
    B: HistoryBuffer,
{
    /// The underlying session manager.
    pub session_manager: Arc<S>,
    buffer: B,
    reformulator: QueryReformulator,
}

impl<S, B> ConversationalPipeline<S, B>
where
    S: SessionManager,
    B: HistoryBuffer,
{
    /// Create a new pipeline.
    pub fn new(session_manager: Arc<S>, buffer: B, reformulator: QueryReformulator) -> Self {
        Self {
            session_manager,
            buffer,
            reformulator,
        }
    }

    /// Create a new session in the underlying store and return its ID.
    ///
    /// # Errors
    ///
    /// Returns [`ConversationError::CapacityExceeded`] if the session store
    /// is full and eviction fails.
    #[instrument(skip(self, config))]
    pub async fn create_session(
        &self,
        config: SessionConfig,
    ) -> Result<ConversationId, ConversationError> {
        self.session_manager.create_session(config).await
    }

    /// Process one user turn and return a retrieval-ready [`ConversationAwareQuery`].
    ///
    /// This does **not** automatically record the user's turn in the session
    /// history: call [`record_response`] after obtaining the answer to append
    /// both the question and the answer.
    ///
    /// # Errors
    ///
    /// - [`ConversationError::EmptyQuery`] if `user_query` is blank.
    /// - [`ConversationError::SessionNotFound`] if `session_id` does not exist.
    /// - [`ConversationError::ReformulationFailed`] if reformulation fails.
    ///
    /// [`record_response`]: Self::record_response
    #[instrument(skip(self, user_query), fields(query_len = user_query.len()))]
    pub async fn process_turn(
        &self,
        session_id: &ConversationId,
        user_query: &str,
    ) -> Result<ConversationAwareQuery, ConversationError> {
        let trimmed = user_query.trim();
        if trimmed.is_empty() {
            return Err(ConversationError::EmptyQuery);
        }

        // 1. Retrieve current session state.
        let session = self.session_manager.get_session(session_id).await?;

        // 2. Build context string from history.
        let context = self.buffer.get_context(&session.history).await;

        // 3. Detect follow-up.
        let is_follow_up = FollowUpDetector::is_follow_up(trimmed);

        // 4. Reformulate.
        let reformulated = self.reformulator.reformulate(trimmed, &context)?;

        debug!(
            session_id = %session_id,
            is_follow_up,
            original = trimmed,
            reformulated = %reformulated,
            "Processed conversational turn"
        );

        Ok(ConversationAwareQuery::new(
            trimmed,
            reformulated,
            context,
            is_follow_up,
        ))
    }

    /// Record the assistant's response by appending it to the session history.
    ///
    /// Call this after the RAG pipeline has produced an answer.
    ///
    /// If you need to record the user turn and assistant response separately,
    /// use [`record_user_turn`] and [`record_assistant_turn`] directly.
    ///
    /// # Errors
    ///
    /// Returns [`ConversationError::SessionNotFound`] if `session_id` does not
    /// exist in the session store.
    ///
    /// [`record_user_turn`]: Self::record_user_turn
    /// [`record_assistant_turn`]: Self::record_assistant_turn
    #[instrument(skip(self, response))]
    pub async fn record_response(
        &self,
        session_id: &ConversationId,
        response: &str,
    ) -> Result<(), ConversationError> {
        let turn = Turn::new(TurnRole::Assistant, response);
        self.session_manager.update_session(session_id, turn).await
    }

    /// Append a user turn directly to the session history.
    ///
    /// # Errors
    ///
    /// Returns [`ConversationError::SessionNotFound`] if `session_id` does not
    /// exist.
    #[instrument(skip(self, query))]
    pub async fn record_user_turn(
        &self,
        session_id: &ConversationId,
        query: &str,
    ) -> Result<(), ConversationError> {
        let turn = Turn::new(TurnRole::User, query);
        self.session_manager.update_session(session_id, turn).await
    }

    /// Append an assistant turn directly to the session history.
    ///
    /// # Errors
    ///
    /// Returns [`ConversationError::SessionNotFound`] if `session_id` does not
    /// exist.
    #[instrument(skip(self, response))]
    pub async fn record_assistant_turn(
        &self,
        session_id: &ConversationId,
        response: &str,
    ) -> Result<(), ConversationError> {
        self.record_response(session_id, response).await
    }

    /// Delete a session from the store.
    ///
    /// # Errors
    ///
    /// Returns [`ConversationError::SessionNotFound`] if `session_id` does not
    /// exist in the session store.
    #[instrument(skip(self))]
    pub async fn delete_session(
        &self,
        session_id: &ConversationId,
    ) -> Result<(), ConversationError> {
        self.session_manager.delete_session(session_id).await
    }

    /// Return the number of active sessions.
    pub async fn session_count(&self) -> usize {
        self.session_manager.session_count().await
    }

    /// Return all active session IDs.
    pub async fn list_sessions(&self) -> Vec<ConversationId> {
        self.session_manager.list_sessions().await
    }
}
