//! Hierarchical memory implementation.
use crate::memory_compression::types::{
    CompressedBlock, CompressionStats, ExtractiveTurnCompressor, MemoryCompressionConfig,
    MemoryCompressionError, MemoryTurn, TurnCompressor,
};

// ── HierarchicalMemory ────────────────────────────────────────────────────────

/// Hierarchical memory that retains recent turns verbatim and compresses older turns.
#[derive(Debug, Clone)]
pub struct HierarchicalMemory {
    /// Recent turns retained verbatim.
    pub recent: Vec<MemoryTurn>,
    /// Compressed blocks from older turns (multiple levels).
    pub blocks: Vec<CompressedBlock>,
    /// Configuration.
    pub config: MemoryCompressionConfig,
}

impl HierarchicalMemory {
    /// Create a new, empty hierarchical memory.
    #[must_use]
    pub fn new(config: MemoryCompressionConfig) -> Self {
        Self {
            recent: Vec::new(),
            blocks: Vec::new(),
            config,
        }
    }

    /// Total approximate token count (recent + compressed).
    #[must_use]
    pub fn total_tokens(&self) -> usize {
        let recent: usize = self.recent.iter().map(|t| t.tokens).sum();
        let blocks: usize = self.blocks.iter().map(|b| b.tokens).sum();
        recent + blocks
    }

    /// Append a new turn, triggering compression if needed.
    ///
    /// # Errors
    ///
    /// Returns [`MemoryCompressionError::EmptyTurns`] if `turn` content is empty.
    pub fn push_turn(&mut self, turn: MemoryTurn) -> Result<(), MemoryCompressionError> {
        if turn.content.trim().is_empty() {
            return Err(MemoryCompressionError::EmptyTurns);
        }
        self.recent.push(turn);
        self.maybe_compress();
        Ok(())
    }

    /// Compress oldest turns into a block when the recent window overflows.
    pub fn maybe_compress(&mut self) {
        let recent_tokens: usize = self.recent.iter().map(|t| t.tokens).sum();
        if recent_tokens <= self.config.recent_window_tokens {
            return;
        }
        let batch_size = self.config.block_size_turns.max(1);
        if self.recent.len() < batch_size {
            return;
        }
        let to_compress: Vec<MemoryTurn> = self.recent.drain(..batch_size).collect();
        let compressor = ExtractiveTurnCompressor {
            summary_sentences: self.config.summary_sentences,
            salient_facts: self.config.salient_facts_per_block,
        };
        let block = compressor.compress(&to_compress, 1);
        // Cascade: merge level-N blocks into level-N+1 if enough accumulate
        self.blocks.push(block);
        if self.blocks.len() >= self.config.block_size_turns && self.config.max_level > 1 {
            let level1: Vec<_> = self.blocks.drain(..).collect();
            let texts: Vec<MemoryTurn> = level1
                .iter()
                .map(|b| MemoryTurn {
                    role: "summary".to_string(),
                    content: b.summary.clone(),
                    tokens: b.tokens,
                    turn_index: b.covered_turns.first().copied().unwrap_or(0),
                })
                .collect();
            let l2 = compressor.compress(&texts, 2);
            self.blocks.push(l2);
        }
    }

    /// Render the memory as a string within a token budget.
    ///
    /// # Errors
    ///
    /// Returns [`MemoryCompressionError::BudgetTooSmall`] if budget is 0.
    pub fn render(&self, budget: usize) -> Result<String, MemoryCompressionError> {
        if budget == 0 {
            return Err(MemoryCompressionError::BudgetTooSmall(budget));
        }
        let mut parts: Vec<String> = Vec::new();
        let mut used = 0usize;
        for b in &self.blocks {
            if used + b.tokens > budget {
                break;
            }
            parts.push(b.summary.clone());
            used += b.tokens;
        }
        for t in &self.recent {
            if used + t.tokens > budget {
                break;
            }
            parts.push(format!("{}: {}", t.role, t.content));
            used += t.tokens;
        }
        if parts.is_empty() {
            return Err(MemoryCompressionError::BudgetTooSmall(budget));
        }
        Ok(parts.join("\n"))
    }

    /// Compute compression statistics.
    #[must_use]
    pub fn stats(&self) -> CompressionStats {
        let original_tokens: usize = self.recent.iter().map(|t| t.tokens).sum::<usize>()
            + self
                .blocks
                .iter()
                .map(|b| b.covered_turns.len() * 50)
                .sum::<usize>();
        let compressed_tokens = self.total_tokens();
        #[allow(clippy::cast_precision_loss)]
        let ratio = if original_tokens == 0 {
            0.0
        } else {
            compressed_tokens as f32 / original_tokens as f32
        };
        let levels_used = self.blocks.iter().map(|b| b.level).max().unwrap_or(0);
        CompressionStats {
            original_tokens,
            compressed_tokens,
            ratio,
            levels_used,
        }
    }
}

impl Default for HierarchicalMemory {
    fn default() -> Self {
        Self::new(MemoryCompressionConfig::default())
    }
}
