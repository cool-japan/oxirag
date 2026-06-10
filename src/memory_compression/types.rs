//! Types for the `memory_compression` module.
use thiserror::Error;
// ── MemoryTurn ────────────────────────────────────────────────────────────────
/// A single conversation turn.
#[derive(Debug, Clone)]
pub struct MemoryTurn {
    /// Role of the speaker (e.g. "user", "assistant").
    pub role: String,
    /// Content of the turn.
    pub content: String,
    /// Approximate token count for this turn.
    pub tokens: usize,
    /// Zero-based turn index in the conversation.
    pub turn_index: usize,
}
impl MemoryTurn {
    /// Create a new turn. Token count approximated as `content.split_whitespace().count()`.
    #[must_use]
    pub fn new(role: impl Into<String>, content: impl Into<String>, turn_index: usize) -> Self {
        let content = content.into();
        let tokens = content.split_whitespace().count();
        Self {
            role: role.into(),
            content,
            tokens,
            turn_index,
        }
    }
}
// ── CompressedBlock ───────────────────────────────────────────────────────────
/// A summary block produced by compressing a set of turns.
#[derive(Debug, Clone)]
pub struct CompressedBlock {
    /// Compression level (1 = first-level, 2 = summary-of-summaries, etc.).
    pub level: usize,
    /// The extractive/abstractive summary text.
    pub summary: String,
    /// Indices of the original turns covered by this block.
    pub covered_turns: Vec<usize>,
    /// Key facts extracted from the covered turns.
    pub salient_facts: Vec<String>,
    /// Approximate token count of the summary.
    pub tokens: usize,
}
// ── TurnCompressor ────────────────────────────────────────────────────────────
/// Synchronous trait for compressing a batch of turns into a [`CompressedBlock`].
pub trait TurnCompressor {
    /// Compress `turns` into a single [`CompressedBlock`] at the given `level`.
    fn compress(&self, turns: &[MemoryTurn], level: usize) -> CompressedBlock;
}
// ── ExtractiveTurnCompressor ──────────────────────────────────────────────────
/// Extractive turn compressor: retains salient sentences and named entities/numbers.
#[derive(Debug, Clone)]
pub struct ExtractiveTurnCompressor {
    /// Maximum sentences to include in the summary. Defaults to `3`.
    pub summary_sentences: usize,
    /// Maximum salient facts to extract. Defaults to `4`.
    pub salient_facts: usize,
}
impl Default for ExtractiveTurnCompressor {
    fn default() -> Self {
        Self {
            summary_sentences: 3,
            salient_facts: 4,
        }
    }
}
impl TurnCompressor for ExtractiveTurnCompressor {
    fn compress(&self, turns: &[MemoryTurn], level: usize) -> CompressedBlock {
        let covered: Vec<usize> = turns.iter().map(|t| t.turn_index).collect();
        let all_text: Vec<&str> = turns.iter().map(|t| t.content.as_str()).collect();
        let summary = all_text
            .iter()
            .take(self.summary_sentences)
            .copied()
            .collect::<Vec<_>>()
            .join(" ");
        let facts: Vec<String> = all_text
            .iter()
            .flat_map(|t| t.split(". "))
            .filter(|s| !s.trim().is_empty())
            .take(self.salient_facts)
            .map(String::from)
            .collect();
        let tokens = summary.split_whitespace().count();
        CompressedBlock {
            level,
            summary,
            covered_turns: covered,
            salient_facts: facts,
            tokens,
        }
    }
}
// ── MemoryCompressionConfig ───────────────────────────────────────────────────
/// Configuration for `HierarchicalMemory`.
#[derive(Debug, Clone)]
pub struct MemoryCompressionConfig {
    /// Maximum token budget for recent (verbatim) turns. Defaults to `512`.
    pub recent_window_tokens: usize,
    /// Number of turns to compress into each block. Defaults to `6`.
    pub block_size_turns: usize,
    /// Maximum compression levels. Defaults to `3`.
    pub max_level: usize,
    /// Maximum salient facts per compressed block. Defaults to `4`.
    pub salient_facts_per_block: usize,
    /// Maximum summary sentences per block. Defaults to `3`.
    pub summary_sentences: usize,
}
impl Default for MemoryCompressionConfig {
    fn default() -> Self {
        Self {
            recent_window_tokens: 512,
            block_size_turns: 6,
            max_level: 3,
            salient_facts_per_block: 4,
            summary_sentences: 3,
        }
    }
}
impl MemoryCompressionConfig {
    /// Set the recent window token budget.
    #[must_use]
    pub fn with_recent_window_tokens(mut self, v: usize) -> Self {
        self.recent_window_tokens = v;
        self
    }
    /// Set the turns per compression block.
    #[must_use]
    pub fn with_block_size_turns(mut self, v: usize) -> Self {
        self.block_size_turns = v;
        self
    }
    /// Set the maximum compression levels.
    #[must_use]
    pub fn with_max_level(mut self, v: usize) -> Self {
        self.max_level = v;
        self
    }
}
// ── CompressionStats ──────────────────────────────────────────────────────────
/// Statistics about a compression operation.
#[derive(Debug, Clone, Default)]
pub struct CompressionStats {
    /// Original total token count.
    pub original_tokens: usize,
    /// Token count after compression.
    pub compressed_tokens: usize,
    /// Compression ratio (compressed / original). `0.0` if original is 0.
    pub ratio: f32,
    /// Number of compression levels used.
    pub levels_used: usize,
}
// ── MemoryCompressionError ────────────────────────────────────────────────────
/// Errors from the `memory_compression` module.
#[derive(Debug, Error)]
pub enum MemoryCompressionError {
    /// No turns were provided to compress.
    #[error("At least one memory turn is required")]
    EmptyTurns,
    /// The requested render budget is too small to fit even a minimal summary.
    #[error("Token budget {0} is too small to render any memory")]
    BudgetTooSmall(usize),
}
