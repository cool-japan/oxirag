//! Long-term memory store implementation.
use chrono::Utc;
use std::collections::HashMap;

use crate::long_term_memory::types::{
    HeuristicImportanceScorer, ImportanceScorer, LongTermMemoryConfig, LongTermMemoryError,
    MemoryKind, MemoryRecord,
};
use crate::types::DocumentId;

// ── embed helper ──────────────────────────────────────────────────────────────

const EMBED_OFFSET: u64 = 14_695_981_039_346_656_037;
const EMBED_PRIME: u64 = 1_099_511_628_211;

fn embed(text: &str, dim: usize) -> Vec<f32> {
    if dim == 0 {
        return Vec::new();
    }
    let tokens: Vec<&str> = text
        .split(|c: char| !c.is_alphanumeric())
        .filter(|t| t.len() >= 2)
        .collect();
    let mut buckets = vec![0.0_f32; dim];
    for tok in &tokens {
        #[allow(clippy::cast_possible_truncation)]
        let idx = {
            let mut h = EMBED_OFFSET;
            for b in tok.as_bytes() {
                h = h.wrapping_mul(EMBED_PRIME) ^ u64::from(*b);
            }
            (h % dim as u64) as usize
        };
        buckets[idx] += 1.0;
    }
    let norm: f32 = buckets.iter().map(|x| x * x).sum::<f32>().sqrt();
    if norm > 0.0 {
        for b in &mut buckets {
            *b /= norm;
        }
    }
    buckets
}

// ── LongTermMemoryStore ───────────────────────────────────────────────────────

/// Long-term memory store for generative-agents-style episodic memory.
#[derive(Debug, Clone)]
pub struct LongTermMemoryStore {
    /// All memory records, indexed by ID.
    pub records: HashMap<String, MemoryRecord>,
    /// Configuration.
    pub config: LongTermMemoryConfig,
}

impl LongTermMemoryStore {
    /// Create a new, empty store with the given config.
    #[must_use]
    pub fn new(config: LongTermMemoryConfig) -> Self {
        Self {
            records: HashMap::new(),
            config,
        }
    }

    /// Number of records currently stored.
    #[must_use]
    pub fn len(&self) -> usize {
        self.records.len()
    }
    /// Return true if the store is empty.
    #[must_use]
    pub fn is_empty(&self) -> bool {
        self.records.is_empty()
    }

    /// Insert a new memory record.
    ///
    /// # Errors
    ///
    /// Returns [`LongTermMemoryError::EmptyContent`] if `content` is empty.
    pub fn insert(
        &mut self,
        content: impl Into<String>,
        kind: MemoryKind,
        scorer: &dyn ImportanceScorer,
    ) -> Result<String, LongTermMemoryError> {
        let content = content.into();
        if content.trim().is_empty() {
            return Err(LongTermMemoryError::EmptyContent);
        }
        let importance = scorer.score(&content);
        let embedding = embed(&content, self.config.dim);
        let id = DocumentId::new().as_str().to_string();
        let now = Utc::now();
        let record = MemoryRecord {
            id: id.clone(),
            content,
            kind,
            importance,
            created_at: now,
            last_accessed: now,
            access_count: 0,
            embedding,
        };
        self.records.insert(id.clone(), record);
        if self.records.len() > self.config.capacity {
            self.prune();
        }
        Ok(id)
    }

    /// Insert with the default heuristic importance scorer.
    ///
    /// # Errors
    ///
    /// Returns [`LongTermMemoryError::EmptyContent`] if `content` is empty.
    pub fn insert_observation(
        &mut self,
        content: impl Into<String>,
    ) -> Result<String, LongTermMemoryError> {
        self.insert(content, MemoryKind::Observation, &HeuristicImportanceScorer)
    }

    /// Prune the lowest-scoring record to keep within capacity.
    pub fn prune(&mut self) {
        if let Some(min_id) = self
            .records
            .values()
            .min_by(|a, b| {
                a.importance
                    .partial_cmp(&b.importance)
                    .unwrap_or(std::cmp::Ordering::Equal)
            })
            .map(|r| r.id.clone())
        {
            self.records.remove(&min_id);
        }
    }

    /// Synthesise a reflection from high-importance observations.
    ///
    /// # Errors
    ///
    /// Returns [`LongTermMemoryError::EmptyStore`] if no observations exist.
    pub fn reflect(&mut self) -> Result<String, LongTermMemoryError> {
        let observations: Vec<&MemoryRecord> = self
            .records
            .values()
            .filter(|r| r.kind == MemoryKind::Observation && r.importance > 0.5)
            .collect();
        if observations.is_empty() {
            return Err(LongTermMemoryError::EmptyStore);
        }
        let summary = observations
            .iter()
            .take(3)
            .map(|r| r.content.as_str())
            .collect::<Vec<_>>()
            .join(". ");
        self.insert(
            summary.clone(),
            MemoryKind::Reflection,
            &HeuristicImportanceScorer,
        )?;
        Ok(summary)
    }
}

impl Default for LongTermMemoryStore {
    fn default() -> Self {
        Self::new(LongTermMemoryConfig::default())
    }
}
