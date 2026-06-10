//! Types for the `long_term_memory` module.
use chrono::{DateTime, Utc};
use thiserror::Error;
// ── MemoryKind ────────────────────────────────────────────────────────────────
/// The kind of a memory record.
#[derive(Debug, Clone, Default, PartialEq, Eq)]
pub enum MemoryKind {
    /// An observed event or experience.
    #[default]
    Observation,
    /// A higher-level reflection synthesised from observations.
    Reflection,
    /// A factual belief extracted from context.
    Fact,
    /// A planned action or intention.
    Plan,
}
impl MemoryKind {
    /// Human-readable label for this kind.
    #[must_use]
    pub fn as_str(&self) -> &str {
        match self {
            Self::Observation => "observation",
            Self::Reflection => "reflection",
            Self::Fact => "fact",
            Self::Plan => "plan",
        }
    }
}
// ── MemoryRecord ──────────────────────────────────────────────────────────────
/// A single record in the long-term memory stream.
#[derive(Debug, Clone)]
pub struct MemoryRecord {
    /// Unique string identifier.
    pub id: String,
    /// Textual content of the memory.
    pub content: String,
    /// Kind of memory.
    pub kind: MemoryKind,
    /// Importance score in [0.0, 1.0] (higher = more salient).
    pub importance: f32,
    /// When the memory was first created.
    pub created_at: DateTime<Utc>,
    /// When the memory was last accessed.
    pub last_accessed: DateTime<Utc>,
    /// Number of times this memory has been retrieved.
    pub access_count: u32,
    /// Cached lexical embedding vector (length = `LongTermMemoryConfig::dim`).
    pub embedding: Vec<f32>,
}
// ── ImportanceScorer ──────────────────────────────────────────────────────────
/// Synchronous scorer that assigns an importance value to new memory content.
pub trait ImportanceScorer {
    /// Score `content` for importance in [0.0, 1.0].
    fn score(&self, content: &str) -> f32;
}
// ── HeuristicImportanceScorer ─────────────────────────────────────────────────
/// Lexical heuristic importance scorer (affective words, numbers, named entities).
#[derive(Debug, Clone, Default)]
pub struct HeuristicImportanceScorer;
impl ImportanceScorer for HeuristicImportanceScorer {
    fn score(&self, content: &str) -> f32 {
        let l = content.to_lowercase();
        let affective = [
            "important",
            "critical",
            "urgent",
            "must",
            "always",
            "never",
            "key",
            "significant",
            "vital",
            "essential",
        ];
        #[allow(clippy::cast_precision_loss)]
        let mut s: f32 = affective.iter().filter(|w| l.contains(**w)).count() as f32 * 0.1;
        let num_count = l.chars().filter(char::is_ascii_digit).count();
        #[allow(clippy::cast_precision_loss)]
        {
            s += (num_count.min(3) as f32) * 0.05;
        }
        let capital_words = content
            .split_whitespace()
            .filter(|w| w.chars().next().is_some_and(char::is_uppercase) && w.len() > 2)
            .count();
        #[allow(clippy::cast_precision_loss)]
        {
            s += (capital_words.min(5) as f32) * 0.05;
        }
        s.min(1.0_f32)
    }
}
// ── MemoryQuery ───────────────────────────────────────────────────────────────
/// Query against the long-term memory store.
#[derive(Debug, Clone)]
pub struct MemoryQuery {
    /// Query text.
    pub text: String,
    /// Maximum number of records to return.
    pub top_k: usize,
    /// Current time used to compute recency decay.
    pub now: DateTime<Utc>,
}
impl MemoryQuery {
    /// Create a new query with current time.
    #[must_use]
    pub fn new(text: impl Into<String>, top_k: usize) -> Self {
        Self {
            text: text.into(),
            top_k,
            now: Utc::now(),
        }
    }
    /// Create a query with an explicit timestamp (useful for testing).
    #[must_use]
    pub fn with_now(mut self, now: DateTime<Utc>) -> Self {
        self.now = now;
        self
    }
}
// ── RetrievedMemory ───────────────────────────────────────────────────────────
/// A memory record with composite retrieval scores.
#[derive(Debug, Clone)]
pub struct RetrievedMemory {
    /// The memory record.
    pub record: MemoryRecord,
    /// Recency component score in [0.0, 1.0].
    pub recency: f32,
    /// Importance component score (= `record.importance`).
    pub importance: f32,
    /// Semantic relevance component in [0.0, 1.0].
    pub relevance: f32,
    /// Combined weighted score.
    pub combined: f32,
}
// ── LongTermMemoryConfig ──────────────────────────────────────────────────────
/// Configuration for `LongTermMemoryStore` and `MemoryRetriever`.
#[derive(Debug, Clone)]
pub struct LongTermMemoryConfig {
    /// Maximum number of memory records to retain. Defaults to `256`.
    pub capacity: usize,
    /// Weight of the recency component. Defaults to `0.35`.
    pub recency_weight: f32,
    /// Weight of the importance component. Defaults to `0.35`.
    pub importance_weight: f32,
    /// Weight of the semantic relevance component. Defaults to `0.30`.
    pub relevance_weight: f32,
    /// Half-life in seconds for recency decay. Defaults to `86400.0` (1 day).
    pub decay_half_life_secs: f64,
    /// Embedding dimension for FNV-1a pseudo-embeddings. Defaults to `128`.
    pub dim: usize,
}
impl Default for LongTermMemoryConfig {
    fn default() -> Self {
        Self {
            capacity: 256,
            recency_weight: 0.35,
            importance_weight: 0.35,
            relevance_weight: 0.30,
            decay_half_life_secs: 86_400.0,
            dim: 128,
        }
    }
}
impl LongTermMemoryConfig {
    /// Set the memory capacity.
    #[must_use]
    pub fn with_capacity(mut self, v: usize) -> Self {
        self.capacity = v;
        self
    }
    /// Set the recency weight.
    #[must_use]
    pub fn with_recency_weight(mut self, v: f32) -> Self {
        self.recency_weight = v;
        self
    }
    /// Set the importance weight.
    #[must_use]
    pub fn with_importance_weight(mut self, v: f32) -> Self {
        self.importance_weight = v;
        self
    }
    /// Set the relevance weight.
    #[must_use]
    pub fn with_relevance_weight(mut self, v: f32) -> Self {
        self.relevance_weight = v;
        self
    }
    /// Set the decay half-life in seconds.
    #[must_use]
    pub fn with_decay_half_life_secs(mut self, v: f64) -> Self {
        self.decay_half_life_secs = v;
        self
    }
    /// Set the embedding dimension.
    #[must_use]
    pub fn with_dim(mut self, v: usize) -> Self {
        self.dim = v;
        self
    }
}
// ── LongTermMemoryError ───────────────────────────────────────────────────────
/// Errors from the `long_term_memory` module.
#[derive(Debug, Error)]
pub enum LongTermMemoryError {
    /// The content string to store was empty.
    #[error("Memory content must not be empty")]
    EmptyContent,
    /// The store was empty when retrieval was attempted.
    #[error("Memory store is empty")]
    EmptyStore,
    /// The query text was empty.
    #[error("Memory query must not be empty")]
    EmptyQuery,
}
