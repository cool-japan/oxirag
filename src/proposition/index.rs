//! Proposition index: fine-grained (Dense-X) retrieval over a corpus.

use std::collections::HashMap;

use crate::proposition::extractor::{HeuristicPropositionExtractor, embed};
use crate::proposition::types::{Proposition, PropositionConfig, PropositionError, PropositionHit};
use crate::types::{Document, DocumentId};

// ── Cosine similarity ─────────────────────────────────────────────────────────

/// Cosine similarity between two L2-normalised, equal-length vectors.
fn cosine(a: &[f32], b: &[f32]) -> f32 {
    if a.len() != b.len() || a.is_empty() {
        return 0.0;
    }
    a.iter().zip(b.iter()).map(|(x, y)| x * y).sum()
}

// ── PropositionIndex ──────────────────────────────────────────────────────────

/// An in-memory index of propositions supporting fine-grained retrieval.
///
/// Each indexed document is decomposed into atomic propositions; queries are
/// matched against individual propositions, and parent documents are recovered
/// from the best-matching proposition.
#[derive(Debug, Clone)]
pub struct PropositionIndex {
    /// Extraction and embedding configuration.
    config: PropositionConfig,
    /// The heuristic extractor used to decompose documents.
    extractor: HeuristicPropositionExtractor,
    /// All propositions accumulated across indexed documents.
    props: Vec<Proposition>,
    /// Next proposition identifier to assign.
    next_id: usize,
}

impl PropositionIndex {
    /// Create a new, empty index with the given configuration.
    #[must_use]
    pub fn new(config: PropositionConfig) -> Self {
        let extractor = HeuristicPropositionExtractor::new(config.clone());
        Self {
            config,
            extractor,
            props: Vec::new(),
            next_id: 0,
        }
    }

    /// Index a single document, decomposing it into propositions.
    pub fn add_document(&mut self, doc: &Document) {
        for (sentence_idx, text) in self.extractor.extract_with_positions(doc) {
            let embedding = embed(&text, self.config.dim);
            self.props.push(Proposition {
                id: self.next_id,
                text,
                parent_id: doc.id.clone(),
                source_sentence: sentence_idx,
                embedding,
            });
            self.next_id += 1;
        }
    }

    /// Index a batch of documents.
    pub fn add_documents(&mut self, docs: &[Document]) {
        for doc in docs {
            self.add_document(doc);
        }
    }

    /// Number of documents represented in the index (distinct parent ids).
    #[must_use]
    pub fn len(&self) -> usize {
        let mut seen: Vec<&str> = self.props.iter().map(|p| p.parent_id.as_str()).collect();
        seen.sort_unstable();
        seen.dedup();
        seen.len()
    }

    /// Return `true` when no propositions have been indexed.
    #[must_use]
    pub fn is_empty(&self) -> bool {
        self.props.is_empty()
    }

    /// Total number of propositions held by the index.
    #[must_use]
    pub fn proposition_count(&self) -> usize {
        self.props.len()
    }

    /// Search for the `top_k` propositions most similar to `query`.
    ///
    /// # Errors
    ///
    /// Returns [`PropositionError::EmptyCorpus`] when the index is empty, and
    /// [`PropositionError::EmptyQuery`] when `query` is blank.
    pub fn search(
        &self,
        query: &str,
        top_k: usize,
    ) -> Result<Vec<PropositionHit>, PropositionError> {
        if query.trim().is_empty() {
            return Err(PropositionError::EmptyQuery);
        }
        if self.props.is_empty() {
            return Err(PropositionError::EmptyCorpus);
        }
        let q_emb = embed(query, self.config.dim);
        let mut hits: Vec<PropositionHit> = self
            .props
            .iter()
            .map(|p| PropositionHit {
                proposition: p.clone(),
                score: cosine(&q_emb, &p.embedding),
            })
            .collect();
        hits.sort_by(|a, b| {
            b.score
                .partial_cmp(&a.score)
                .unwrap_or(std::cmp::Ordering::Equal)
                .then_with(|| a.proposition.id.cmp(&b.proposition.id))
        });
        hits.truncate(top_k);
        Ok(hits)
    }

    /// Search for the `top_k` parent documents most relevant to `query`.
    ///
    /// Each document is scored by its single best-matching proposition.
    ///
    /// # Errors
    ///
    /// Returns [`PropositionError::EmptyCorpus`] when the index is empty, and
    /// [`PropositionError::EmptyQuery`] when `query` is blank.
    pub fn search_documents(
        &self,
        query: &str,
        top_k: usize,
    ) -> Result<Vec<(DocumentId, f32)>, PropositionError> {
        if query.trim().is_empty() {
            return Err(PropositionError::EmptyQuery);
        }
        if self.props.is_empty() {
            return Err(PropositionError::EmptyCorpus);
        }
        let q_emb = embed(query, self.config.dim);
        let mut best: HashMap<String, f32> = HashMap::new();
        for p in &self.props {
            let score = cosine(&q_emb, &p.embedding);
            let key = p.parent_id.as_str().to_string();
            best.entry(key)
                .and_modify(|s| {
                    if score > *s {
                        *s = score;
                    }
                })
                .or_insert(score);
        }
        let mut docs: Vec<(DocumentId, f32)> = best
            .into_iter()
            .map(|(k, v)| (DocumentId::from_string(k), v))
            .collect();
        docs.sort_by(|a, b| {
            b.1.partial_cmp(&a.1)
                .unwrap_or(std::cmp::Ordering::Equal)
                .then_with(|| a.0.as_str().cmp(b.0.as_str()))
        });
        docs.truncate(top_k);
        Ok(docs)
    }

    /// Access the raw propositions held by the index.
    #[must_use]
    pub fn propositions(&self) -> &[Proposition] {
        &self.props
    }
}
