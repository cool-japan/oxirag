//! Document summary index: match on summaries, return full parent documents.

use std::collections::HashMap;

use crate::summary_index::summarizer::{ExtractiveSummarizer, Summarizer, embed};
use crate::summary_index::types::{DocumentSummary, SummaryConfig, SummaryHit, SummaryIndexError};
use crate::types::{Document, DocumentId};

// ── Cosine similarity ─────────────────────────────────────────────────────────

/// Cosine similarity between two L2-normalised, equal-length vectors.
fn cosine(a: &[f32], b: &[f32]) -> f32 {
    if a.len() != b.len() || a.is_empty() {
        return 0.0;
    }
    a.iter().zip(b.iter()).map(|(x, y)| x * y).sum()
}

// ── SummaryIndex ──────────────────────────────────────────────────────────────

/// An in-memory document summary index (`LlamaIndex` `DocumentSummaryIndex`).
///
/// For each indexed document an extractive summary is produced, and the
/// **summary** embedding is what queries are matched against. Retrieval returns
/// the **full parent document**, giving precise, summary-driven recall without
/// losing any document context.
///
/// This is a *flat*, one-summary-per-document index. It is deliberately distinct
/// from RAPTOR, which builds a recursive multi-level tree of summaries.
#[derive(Debug, Clone)]
pub struct SummaryIndex<S: Summarizer> {
    /// Summarization and embedding configuration.
    config: SummaryConfig,
    /// The summarizer used to condense each document.
    summarizer: S,
    /// One extractive summary (with embedding) per indexed document.
    summaries: Vec<DocumentSummary>,
    /// Full parent documents, keyed by their identifier.
    docs: HashMap<DocumentId, Document>,
}

impl SummaryIndex<ExtractiveSummarizer> {
    /// Create a new, empty index backed by the default [`ExtractiveSummarizer`].
    #[must_use]
    pub fn new(config: SummaryConfig) -> Self {
        Self::with_summarizer(config, ExtractiveSummarizer::new())
    }
}

impl<S: Summarizer> SummaryIndex<S> {
    /// Create a new, empty index using a custom summarizer.
    #[must_use]
    pub fn with_summarizer(config: SummaryConfig, summarizer: S) -> Self {
        Self {
            config,
            summarizer,
            summaries: Vec::new(),
            docs: HashMap::new(),
        }
    }

    /// Index a single document: summarize it, embed the summary, store both.
    pub fn add_document(&mut self, doc: &Document) {
        let summary = self
            .summarizer
            .summarize(doc, self.config.summary_sentences);
        let embedding = embed(&summary, self.config.dim);
        self.summaries.push(DocumentSummary {
            doc_id: doc.id.clone(),
            summary,
            embedding,
        });
        self.docs.insert(doc.id.clone(), doc.clone());
    }

    /// Index a batch of documents.
    pub fn add_documents(&mut self, docs: &[Document]) {
        for doc in docs {
            self.add_document(doc);
        }
    }

    /// Number of documents held by the index.
    #[must_use]
    pub fn len(&self) -> usize {
        self.summaries.len()
    }

    /// Return `true` when no documents have been indexed.
    #[must_use]
    pub fn is_empty(&self) -> bool {
        self.summaries.is_empty()
    }

    /// Search for the `top_k` documents whose summaries best match `query`.
    ///
    /// The query is embedded with the same lexical scheme as the summaries,
    /// scored by cosine similarity against every summary, sorted in descending
    /// score order, and truncated to `top_k`. Each returned [`SummaryHit`]
    /// carries the **full parent document** alongside its summary and score.
    ///
    /// # Errors
    ///
    /// Returns [`SummaryIndexError::EmptyIndex`] when the index holds no
    /// documents, and [`SummaryIndexError::EmptyQuery`] when `query` is blank.
    pub fn search(&self, query: &str, top_k: usize) -> Result<Vec<SummaryHit>, SummaryIndexError> {
        if query.trim().is_empty() {
            return Err(SummaryIndexError::EmptyQuery);
        }
        if self.summaries.is_empty() {
            return Err(SummaryIndexError::EmptyIndex);
        }
        let q_emb = embed(query, self.config.dim);
        let mut hits: Vec<SummaryHit> = self
            .summaries
            .iter()
            .filter_map(|s| {
                self.docs.get(&s.doc_id).map(|doc| SummaryHit {
                    document: doc.clone(),
                    summary: s.summary.clone(),
                    score: cosine(&q_emb, &s.embedding),
                })
            })
            .collect();
        hits.sort_by(|a, b| {
            b.score
                .partial_cmp(&a.score)
                .unwrap_or(std::cmp::Ordering::Equal)
                .then_with(|| a.document.id.as_str().cmp(b.document.id.as_str()))
        });
        hits.truncate(top_k);
        Ok(hits)
    }

    /// Access the stored per-document summaries.
    #[must_use]
    pub fn summaries(&self) -> &[DocumentSummary] {
        &self.summaries
    }
}
