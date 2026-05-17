//! The `Chunk` type — a single fragment of a source document.

use serde::{Deserialize, Serialize};
use std::collections::HashMap;

use crate::types::Document;

/// A single chunk derived from a larger source document.
///
/// Each `Chunk` records its position within the original document so that
/// consumers can reconstruct context or highlight the matched span.
///
/// Chunks are cheaply converted to [`Document`] instances via
/// [`Chunk::into_document`], making them ready for indexing in the echo layer.
///
/// # Examples
///
/// ```rust
/// # #[cfg(feature = "chunking")]
/// # {
/// use oxirag::chunking::{Chunk, ChunkConfig, FixedSizeChunker, ChunkStrategy};
/// use oxirag::types::Document;
///
/// let doc = Document::new("Hello world. This is a test document.");
/// let config = ChunkConfig::default().with_chunk_size(20).with_chunk_overlap(0);
/// let chunker = FixedSizeChunker;
/// let chunks = chunker.chunk(&doc, &config);
/// for chunk in chunks {
///     let indexed_doc = chunk.into_document();
///     println!("{}", indexed_doc.content);
/// }
/// # }
/// ```
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Chunk {
    /// The text content of this chunk.
    pub content: String,
    /// String representation of the source document's [`DocumentId`].
    ///
    /// [`DocumentId`]: crate::types::DocumentId
    pub source_doc_id: String,
    /// Zero-based index of this chunk within the document.
    pub chunk_index: usize,
    /// Inclusive start character offset in the original document content.
    pub start_char: usize,
    /// Exclusive end character offset in the original document content.
    pub end_char: usize,
    /// Metadata carried over from the source document, augmented with
    /// `"chunk_index"` and `"source_doc_id"` keys.
    pub metadata: HashMap<String, String>,
}

impl Chunk {
    /// Create a new `Chunk`.
    #[must_use]
    pub fn new(
        content: impl Into<String>,
        source_doc_id: impl Into<String>,
        chunk_index: usize,
        start_char: usize,
        end_char: usize,
        metadata: HashMap<String, String>,
    ) -> Self {
        let source_doc_id = source_doc_id.into();
        let content = content.into();
        let mut metadata = metadata;
        metadata.insert("chunk_index".to_string(), chunk_index.to_string());
        metadata.insert("source_doc_id".to_string(), source_doc_id.clone());
        Self {
            content,
            source_doc_id,
            chunk_index,
            start_char,
            end_char,
            metadata,
        }
    }

    /// Convert this chunk into a [`Document`] that can be directly indexed.
    ///
    /// The resulting document's content is the chunk text; its metadata
    /// carries all chunk-level annotations including `chunk_index`,
    /// `source_doc_id`, `start_char`, and `end_char`.
    #[must_use]
    pub fn into_document(self) -> Document {
        let mut meta = self.metadata;
        meta.insert("start_char".to_string(), self.start_char.to_string());
        meta.insert("end_char".to_string(), self.end_char.to_string());

        let mut doc = Document::new(self.content);
        for (k, v) in meta {
            doc = doc.with_metadata(k, v);
        }
        doc
    }
}
