//! `DocumentChunker` — orchestrates document chunking.
//!
//! `DocumentChunker` holds a boxed [`ChunkStrategy`] and a [`ChunkConfig`],
//! providing a clean API for converting [`Document`]s into indexed fragments.
//!
//! Convenience constructors (`with_fixed_size`, `with_sentences`, etc.) make
//! it easy to pick a strategy without importing the concrete types.

use crate::chunking::chunk::Chunk;
use crate::chunking::config::ChunkConfig;
use crate::chunking::strategies::{
    ChunkStrategy, FixedSizeChunker, MarkdownChunker, RecursiveChunker, SentenceChunker,
};
use crate::types::Document;

/// Orchestrates document chunking via a pluggable [`ChunkStrategy`].
///
/// # Examples
///
/// ```rust
/// # #[cfg(feature = "chunking")]
/// # {
/// use oxirag::chunking::{ChunkConfig, DocumentChunker};
/// use oxirag::types::Document;
///
/// let config = ChunkConfig::default().with_chunk_size(200).with_chunk_overlap(50);
/// let chunker = DocumentChunker::with_fixed_size(config);
///
/// let doc = Document::new("A very long document that needs to be split into pieces.");
/// let indexed_docs = chunker.chunk_document(&doc);
/// println!("Produced {} indexable chunks", indexed_docs.len());
/// # }
/// ```
pub struct DocumentChunker {
    strategy: Box<dyn ChunkStrategy>,
    config: ChunkConfig,
}

impl DocumentChunker {
    /// Create a `DocumentChunker` with any [`ChunkStrategy`] implementation.
    pub fn new(strategy: impl ChunkStrategy + 'static, config: ChunkConfig) -> Self {
        Self {
            strategy: Box::new(strategy),
            config,
        }
    }

    /// Create a `DocumentChunker` backed by [`FixedSizeChunker`].
    #[must_use]
    pub fn with_fixed_size(config: ChunkConfig) -> Self {
        Self::new(FixedSizeChunker, config)
    }

    /// Create a `DocumentChunker` backed by [`SentenceChunker`].
    #[must_use]
    pub fn with_sentences(config: ChunkConfig) -> Self {
        Self::new(SentenceChunker, config)
    }

    /// Create a `DocumentChunker` backed by [`RecursiveChunker`].
    #[must_use]
    pub fn with_recursive(config: ChunkConfig) -> Self {
        Self::new(RecursiveChunker, config)
    }

    /// Create a `DocumentChunker` backed by [`MarkdownChunker`].
    #[must_use]
    pub fn with_markdown(config: ChunkConfig) -> Self {
        Self::new(MarkdownChunker, config)
    }

    /// The name of the underlying strategy (for logging / diagnostics).
    #[must_use]
    pub fn strategy_name(&self) -> &str {
        self.strategy.name()
    }

    /// Return raw [`Chunk`] objects for `doc` without converting to `Document`.
    ///
    /// Useful when the caller needs offset information or wants to inspect
    /// chunks before deciding whether to index them.
    #[must_use]
    pub fn raw_chunks(&self, doc: &Document) -> Vec<Chunk> {
        self.strategy.chunk(doc, &self.config)
    }

    /// Chunk `doc` and return each chunk converted to a [`Document`] ready for
    /// indexing.
    ///
    /// The returned documents contain all metadata from the source document,
    /// plus the chunk-specific keys `chunk_index`, `source_doc_id`,
    /// `start_char`, and `end_char`.
    #[must_use]
    pub fn chunk_document(&self, doc: &Document) -> Vec<Document> {
        self.strategy
            .chunk(doc, &self.config)
            .into_iter()
            .map(Chunk::into_document)
            .collect()
    }

    /// Chunk every document in `docs` and collect all resulting chunk-documents
    /// into a single flat vector.
    ///
    /// Documents are processed in order; chunks from earlier documents appear
    /// before chunks from later documents in the result.
    #[must_use]
    pub fn chunk_documents(&self, docs: &[Document]) -> Vec<Document> {
        docs.iter().flat_map(|d| self.chunk_document(d)).collect()
    }
}
