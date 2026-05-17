//! History buffer strategies for building conversational context.
//!
//! Each buffer variant implements the [`HistoryBuffer`] trait, which takes a
//! [`ConversationHistory`] and produces a plain-text context string that can be
//! fed into a [`QueryReformulator`].
//!
//! | Buffer | Behaviour |
//! |---|---|
//! | [`FullHistoryBuffer`] | Serialises every turn verbatim |
//! | [`SlidingWindowBuffer`] | Keeps only the last *N* turns |
//! | [`SummaryBuffer`] | Recent turns verbatim + rolling summary of older turns |
//! | [`HybridBuffer`] | Recent turns verbatim + character-limited summary |
//!
//! [`QueryReformulator`]: crate::conversation::reformulator::QueryReformulator

use std::sync::Arc;

use async_trait::async_trait;
use tokio::sync::RwLock;

use super::types::{ConversationHistory, TurnRole};

// ── HistoryBuffer ─────────────────────────────────────────────────────────────

/// Trait for context-building strategies over a [`ConversationHistory`].
///
/// Implementations are free to be stateful (e.g., maintaining a rolling
/// summary in a lock) which is why the method takes `&self` and is `async`.
#[async_trait]
pub trait HistoryBuffer: Send + Sync {
    /// Build a plain-text context string from `history`.
    ///
    /// The returned string is suitable for prepending to a query or injecting
    /// into a reformulation template.
    async fn get_context(&self, history: &ConversationHistory) -> String;
}

// ── Formatting helper ─────────────────────────────────────────────────────────

/// Format a slice of turns into a single string of the form:
/// ```text
/// User: …
/// Assistant: …
/// User: …
/// ```
fn format_turns(turns: &[super::types::Turn]) -> String {
    turns
        .iter()
        .map(|t| format!("{}: {}", t.role, t.text))
        .collect::<Vec<_>>()
        .join("\n")
}

// ── FullHistoryBuffer ─────────────────────────────────────────────────────────

/// A buffer that serialises the entire conversation history verbatim.
///
/// This is the most faithful representation of the dialogue but grows without
/// bound and may cause issues with very long conversations.
///
/// # Example
///
/// ```rust
/// # #[cfg(feature = "conversational")]
/// # {
/// # tokio_test::block_on(async {
/// use oxirag::conversation::buffer::{FullHistoryBuffer, HistoryBuffer};
/// use oxirag::conversation::types::{ConversationHistory, Turn, TurnRole};
///
/// let mut history = ConversationHistory::new();
/// history.add_turn(Turn::new(TurnRole::User, "Hello!"));
/// history.add_turn(Turn::new(TurnRole::Assistant, "Hi there!"));
///
/// let buf = FullHistoryBuffer;
/// let ctx = buf.get_context(&history).await;
/// assert!(ctx.contains("User: Hello!"));
/// assert!(ctx.contains("Assistant: Hi there!"));
/// # });
/// # }
/// ```
pub struct FullHistoryBuffer;

#[async_trait]
impl HistoryBuffer for FullHistoryBuffer {
    async fn get_context(&self, history: &ConversationHistory) -> String {
        format_turns(history.as_slice())
    }
}

// ── SlidingWindowBuffer ───────────────────────────────────────────────────────

/// A buffer that only considers the most recent `window_size` turns.
///
/// Older turns are silently discarded, keeping the context size bounded.
///
/// # Example
///
/// ```rust
/// # #[cfg(feature = "conversational")]
/// # {
/// # tokio_test::block_on(async {
/// use oxirag::conversation::buffer::{HistoryBuffer, SlidingWindowBuffer};
/// use oxirag::conversation::types::{ConversationHistory, Turn, TurnRole};
///
/// let mut history = ConversationHistory::new();
/// for i in 0..10 {
///     history.add_turn(Turn::new(TurnRole::User, format!("Q{i}")));
///     history.add_turn(Turn::new(TurnRole::Assistant, format!("A{i}")));
/// }
///
/// let buf = SlidingWindowBuffer::new(4);
/// let ctx = buf.get_context(&history).await;
/// // Only the last 4 turns should appear.
/// assert!(ctx.contains("Q9"));
/// # });
/// # }
/// ```
pub struct SlidingWindowBuffer {
    /// Number of turns to retain.
    pub window_size: usize,
}

impl SlidingWindowBuffer {
    /// Create a new sliding-window buffer with the given window size.
    #[must_use]
    pub fn new(window_size: usize) -> Self {
        Self { window_size }
    }
}

impl Default for SlidingWindowBuffer {
    fn default() -> Self {
        Self::new(6)
    }
}

#[async_trait]
impl HistoryBuffer for SlidingWindowBuffer {
    async fn get_context(&self, history: &ConversationHistory) -> String {
        format_turns(history.recent_turns(self.window_size))
    }
}

// ── SummaryBuffer ─────────────────────────────────────────────────────────────

/// A buffer that keeps the `max_recent` most recent turns verbatim and
/// maintains a rolling plain-text summary of older turns.
///
/// The summary is accumulated lazily: whenever the history has more turns than
/// `max_recent`, all turns that fall outside that window are condensed into the
/// rolling summary stored in an `Arc<RwLock<Option<String>>>`.
///
/// Because this module has no LLM dependency, the "summarisation" step is a
/// heuristic: it concatenates the text of older turns, separated by " | ", and
/// truncates the result to `max_summary_chars` characters.
///
/// # Example
///
/// ```rust
/// # #[cfg(feature = "conversational")]
/// # {
/// # tokio_test::block_on(async {
/// use oxirag::conversation::buffer::{HistoryBuffer, SummaryBuffer};
/// use oxirag::conversation::types::{ConversationHistory, Turn, TurnRole};
///
/// let mut history = ConversationHistory::new();
/// for i in 0..20 {
///     history.add_turn(Turn::new(TurnRole::User, format!("question {i}")));
///     history.add_turn(Turn::new(TurnRole::Assistant, format!("answer {i}")));
/// }
///
/// let buf = SummaryBuffer::new(4, 500);
/// let ctx = buf.get_context(&history).await;
/// assert!(!ctx.is_empty());
/// # });
/// # }
/// ```
pub struct SummaryBuffer {
    /// Number of recent turns to include verbatim.
    pub max_recent: usize,
    /// Maximum number of characters in the accumulated summary.
    pub max_summary_chars: usize,
    /// The rolling summary (shared so it can be updated across calls).
    summary: Arc<RwLock<Option<String>>>,
}

impl SummaryBuffer {
    /// Create a new summary buffer.
    ///
    /// * `max_recent` — how many recent turns to present verbatim.
    /// * `max_summary_chars` — character cap on the rolling summary.
    #[must_use]
    pub fn new(max_recent: usize, max_summary_chars: usize) -> Self {
        Self {
            max_recent,
            max_summary_chars,
            summary: Arc::new(RwLock::new(None)),
        }
    }
}

impl Default for SummaryBuffer {
    fn default() -> Self {
        Self::new(6, 1000)
    }
}

#[async_trait]
impl HistoryBuffer for SummaryBuffer {
    async fn get_context(&self, history: &ConversationHistory) -> String {
        let all_turns = history.as_slice();
        let total = all_turns.len();

        if total <= self.max_recent {
            // No summarisation needed — everything fits in the recent window.
            return format_turns(all_turns);
        }

        let split = total - self.max_recent;
        let older = &all_turns[..split];
        let recent = &all_turns[split..];

        // Build a summary of the older turns.
        let new_summary = self.build_summary(older);

        // Update the stored summary.
        {
            let mut guard = self.summary.write().await;
            *guard = Some(new_summary.clone());
        }

        let recent_ctx = format_turns(recent);
        if new_summary.is_empty() {
            recent_ctx
        } else {
            format!("[Summary of earlier conversation: {new_summary}]\n\n{recent_ctx}")
        }
    }
}

impl SummaryBuffer {
    /// Build a plain-text summary from a slice of turns by concatenating their
    /// texts and then truncating to `max_summary_chars`.
    fn build_summary(&self, turns: &[super::types::Turn]) -> String {
        let raw: String = turns
            .iter()
            .map(|t| {
                let role_label = match t.role {
                    TurnRole::User => "U",
                    TurnRole::Assistant => "A",
                };
                format!("{role_label}: {}", t.text)
            })
            .collect::<Vec<_>>()
            .join(" | ");

        if raw.len() <= self.max_summary_chars {
            raw
        } else {
            // Truncate at a word boundary when possible.
            let truncated = &raw[..self.max_summary_chars];
            match truncated.rfind(' ') {
                Some(pos) => format!("{}…", &truncated[..pos]),
                None => format!("{truncated}…"),
            }
        }
    }
}

// ── HybridBuffer ──────────────────────────────────────────────────────────────

/// A buffer combining verbatim recent turns with a character-capped summary of
/// older turns.
///
/// Unlike [`SummaryBuffer`], `HybridBuffer` is stateless — it recomputes the
/// summary on every call. This makes it cheaper to clone and simpler to reason
/// about correctness at the cost of recomputing the summary each time.
///
/// # Example
///
/// ```rust
/// # #[cfg(feature = "conversational")]
/// # {
/// # tokio_test::block_on(async {
/// use oxirag::conversation::buffer::{HybridBuffer, HistoryBuffer};
/// use oxirag::conversation::types::{ConversationHistory, Turn, TurnRole};
///
/// let mut history = ConversationHistory::new();
/// for i in 0..15 {
///     history.add_turn(Turn::new(TurnRole::User, format!("Q{i}")));
///     history.add_turn(Turn::new(TurnRole::Assistant, format!("A{i}")));
/// }
///
/// let buf = HybridBuffer::new(4, 300);
/// let ctx = buf.get_context(&history).await;
/// assert!(ctx.contains("Q14"));  // most recent user turn
/// # });
/// # }
/// ```
pub struct HybridBuffer {
    /// Number of recent turns to include verbatim.
    pub recent_turns: usize,
    /// Maximum characters for the summary of older turns.
    pub max_summary_chars: usize,
}

impl HybridBuffer {
    /// Create a hybrid buffer.
    #[must_use]
    pub fn new(recent_turns: usize, max_summary_chars: usize) -> Self {
        Self {
            recent_turns,
            max_summary_chars,
        }
    }
}

impl Default for HybridBuffer {
    fn default() -> Self {
        Self::new(6, 800)
    }
}

#[async_trait]
impl HistoryBuffer for HybridBuffer {
    async fn get_context(&self, history: &ConversationHistory) -> String {
        let all_turns = history.as_slice();
        let total = all_turns.len();

        if total <= self.recent_turns {
            return format_turns(all_turns);
        }

        let split = total - self.recent_turns;
        let older = &all_turns[..split];
        let recent = &all_turns[split..];

        // Build a compact summary of older turns.
        let raw_summary: String = older
            .iter()
            .map(|t| t.text.as_str())
            .collect::<Vec<_>>()
            .join(" ");

        let summary = if raw_summary.len() > self.max_summary_chars {
            let truncated = &raw_summary[..self.max_summary_chars];
            match truncated.rfind(' ') {
                Some(pos) => format!("{}…", &truncated[..pos]),
                None => format!("{truncated}…"),
            }
        } else {
            raw_summary
        };

        let recent_ctx = format_turns(recent);

        if summary.is_empty() {
            recent_ctx
        } else {
            format!("[Earlier context: {summary}]\n\n{recent_ctx}")
        }
    }
}
