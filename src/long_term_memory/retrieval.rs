//! Long-term memory retrieval.
use crate::long_term_memory::store::LongTermMemoryStore;
use crate::long_term_memory::types::{LongTermMemoryError, MemoryQuery, RetrievedMemory};

// ── cosine ────────────────────────────────────────────────────────────────────

fn cosine(a: &[f32], b: &[f32]) -> f32 {
    if a.len() != b.len() || a.is_empty() {
        return 0.0;
    }
    let dot: f32 = a.iter().zip(b).map(|(x, y)| x * y).sum();
    let na: f32 = a.iter().map(|x| x * x).sum::<f32>().sqrt();
    let nb: f32 = b.iter().map(|x| x * x).sum::<f32>().sqrt();
    if na == 0.0 || nb == 0.0 {
        0.0
    } else {
        (dot / (na * nb)).clamp(-1.0, 1.0)
    }
}

const EMBED_OFFSET: u64 = 14_695_981_039_346_656_037;
const EMBED_PRIME: u64 = 1_099_511_628_211;

fn embed_query(text: &str, dim: usize) -> Vec<f32> {
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

// ── MemoryRetriever ───────────────────────────────────────────────────────────

/// Retrieves memories from a [`LongTermMemoryStore`] using a combined score.
///
/// `combined = w_recency * decay(age) + w_importance * importance + w_relevance * cosine`
#[derive(Debug, Clone, Default)]
pub struct MemoryRetriever;

impl MemoryRetriever {
    /// Retrieve top-k memories from `store` for `query`.
    ///
    /// # Errors
    ///
    /// Returns [`LongTermMemoryError::EmptyStore`] if the store is empty.
    /// Returns [`LongTermMemoryError::EmptyQuery`] if the query text is empty.
    pub fn retrieve(
        &self,
        store: &mut LongTermMemoryStore,
        query: &MemoryQuery,
    ) -> Result<Vec<RetrievedMemory>, LongTermMemoryError> {
        if store.is_empty() {
            return Err(LongTermMemoryError::EmptyStore);
        }
        if query.text.trim().is_empty() {
            return Err(LongTermMemoryError::EmptyQuery);
        }

        let cfg = &store.config;
        let q_embed = embed_query(&query.text, cfg.dim);
        let half_life = cfg.decay_half_life_secs;

        let mut scored: Vec<RetrievedMemory> = store
            .records
            .values_mut()
            .map(|record| {
                let age_secs_i64 = (query.now - record.last_accessed).num_seconds().max(0);
                #[allow(clippy::cast_precision_loss)]
                let age_secs = age_secs_i64 as f64;
                #[allow(clippy::cast_possible_truncation)]
                let recency = 2.0_f64.powf(-age_secs / half_life) as f32;
                let relevance = cosine(&q_embed, &record.embedding);
                let combined = cfg.recency_weight * recency
                    + cfg.importance_weight * record.importance
                    + cfg.relevance_weight * relevance;
                record.access_count += 1;
                record.last_accessed = query.now;
                RetrievedMemory {
                    record: record.clone(),
                    recency,
                    importance: record.importance,
                    relevance,
                    combined,
                }
            })
            .collect();

        scored.sort_by(|a, b| {
            b.combined
                .partial_cmp(&a.combined)
                .unwrap_or(std::cmp::Ordering::Equal)
        });
        scored.truncate(query.top_k);
        Ok(scored)
    }
}
