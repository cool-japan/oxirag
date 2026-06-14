//! `LongRAG` retriever: build long units, then rank them against a query.

use crate::long_rag::grouper::{LongUnitGrouper, embed};
use crate::long_rag::types::{LongHit, LongRagConfig, LongRagError, LongUnit};
use crate::types::Document;

// ── Cosine similarity ─────────────────────────────────────────────────────────

/// Cosine similarity between two L2-normalised, equal-length vectors.
fn cosine(a: &[f32], b: &[f32]) -> f32 {
    if a.len() != b.len() || a.is_empty() {
        return 0.0;
    }
    a.iter().zip(b.iter()).map(|(x, y)| x * y).sum()
}

// ── LongRagRetriever ──────────────────────────────────────────────────────────

/// Retrieves over long units assembled by a [`LongUnitGrouper`].
///
/// Documents are grouped once at build time into a small number of long units;
/// queries are then scored directly against those units. Because there are fewer,
/// longer units than there would be short chunks, retrieval suffers less
/// fragmentation and tends to recover more complete context per hit.
#[derive(Debug, Clone)]
pub struct LongRagRetriever {
    /// Grouping and embedding configuration.
    config: LongRagConfig,
    /// The long units built from the indexed corpus.
    units: Vec<LongUnit>,
}

impl LongRagRetriever {
    /// Create a new, empty retriever with the given configuration.
    #[must_use]
    pub fn new(config: LongRagConfig) -> Self {
        Self {
            config,
            units: Vec::new(),
        }
    }

    /// Group `docs` into long units and store them for retrieval.
    ///
    /// Replaces any previously built units. The grouping strategy and token
    /// budget are taken from the retriever's configuration.
    ///
    /// # Errors
    ///
    /// Returns [`LongRagError::EmptyCorpus`] when `docs` is empty.
    pub fn build(&mut self, docs: &[Document]) -> Result<(), LongRagError> {
        if docs.is_empty() {
            return Err(LongRagError::EmptyCorpus);
        }
        let grouper = LongUnitGrouper::new(self.config.clone());
        self.units = grouper.group(docs);
        Ok(())
    }

    /// Number of long units held by the retriever.
    #[must_use]
    pub fn unit_count(&self) -> usize {
        self.units.len()
    }

    /// Return `true` when no long units have been built.
    #[must_use]
    pub fn is_empty(&self) -> bool {
        self.units.is_empty()
    }

    /// Access the long units held by the retriever.
    #[must_use]
    pub fn units(&self) -> &[LongUnit] {
        &self.units
    }

    /// Search for the `top_k` long units most similar to `query`.
    ///
    /// The query is embedded with the same lexical scheme as the units, scored by
    /// cosine similarity against every unit, sorted in descending score order
    /// (ties broken by unit id for determinism), and truncated to `top_k`.
    ///
    /// # Errors
    ///
    /// Returns [`LongRagError::EmptyCorpus`] when no units have been built, and
    /// [`LongRagError::EmptyQuery`] when `query` is blank.
    pub fn search(&self, query: &str, top_k: usize) -> Result<Vec<LongHit>, LongRagError> {
        if query.trim().is_empty() {
            return Err(LongRagError::EmptyQuery);
        }
        if self.units.is_empty() {
            return Err(LongRagError::EmptyCorpus);
        }
        let q_emb = embed(query, self.config.dim);
        let mut hits: Vec<LongHit> = self
            .units
            .iter()
            .map(|u| LongHit {
                unit: u.clone(),
                score: cosine(&q_emb, &u.embedding),
            })
            .collect();
        hits.sort_by(|a, b| {
            b.score
                .partial_cmp(&a.score)
                .unwrap_or(std::cmp::Ordering::Equal)
                .then_with(|| a.unit.id.cmp(&b.unit.id))
        });
        hits.truncate(top_k);
        Ok(hits)
    }
}
