#![cfg(feature = "nodejs")]
#![allow(missing_docs)]
//! napi-rs wrappers for core `OxiRAG` types.

use napi_derive::napi;

use crate::types::{Document, Query, SearchResult};

// ─────────────────────────────────────────────────────────────────────────────
// NapiDocument
// ─────────────────────────────────────────────────────────────────────────────

/// A document to index or a retrieved search result.
///
/// @example
/// ```js
/// const doc = new Document("Rust is memory-safe.", "Rust Overview");
/// console.log(doc.id);      // UUID string
/// console.log(doc.content); // "Rust is memory-safe."
/// console.log(doc.title);   // "Rust Overview"
/// ```
#[napi]
pub struct NapiDocument {
    pub(crate) inner: Document,
}

#[napi]
impl NapiDocument {
    /// Create a new document from content and an optional title.
    #[napi(constructor)]
    #[must_use]
    pub fn new(content: String, title: Option<String>) -> Self {
        let mut doc = Document::new(content);
        if let Some(t) = title {
            doc = doc.with_title(t);
        }
        Self { inner: doc }
    }

    /// The document content.
    #[napi(getter)]
    #[must_use]
    pub fn content(&self) -> String {
        self.inner.content.clone()
    }

    /// Unique document identifier (UUID string).
    #[napi(getter)]
    #[must_use]
    pub fn id(&self) -> String {
        self.inner.id.to_string()
    }

    /// Optional document title.
    #[napi(getter)]
    #[must_use]
    pub fn title(&self) -> Option<String> {
        self.inner.title.clone()
    }

    /// Optional document source (URL or path).
    #[napi(getter)]
    #[must_use]
    pub fn source(&self) -> Option<String> {
        self.inner.source.clone()
    }

    /// Attach a metadata key-value pair and return a new document.
    ///
    /// The original document is not mutated; a new `NapiDocument` is returned.
    #[napi]
    #[must_use]
    pub fn with_metadata(&self, key: String, value: String) -> NapiDocument {
        NapiDocument {
            inner: self.inner.clone().with_metadata(key, value),
        }
    }

    /// Attach a source string and return a new document.
    #[napi]
    #[must_use]
    pub fn with_source(&self, source: String) -> NapiDocument {
        NapiDocument {
            inner: self.inner.clone().with_source(source),
        }
    }
}

// ─────────────────────────────────────────────────────────────────────────────
// NapiSearchResult
// ─────────────────────────────────────────────────────────────────────────────

/// A search result from the Echo layer.
///
/// Wraps the matched document alongside its similarity score and rank.
#[napi]
pub struct NapiSearchResult {
    pub(crate) inner: SearchResult,
}

#[napi]
impl NapiSearchResult {
    /// Similarity score in the range \[0.0, 1.0\].  Higher is more relevant.
    #[napi(getter)]
    #[must_use]
    pub fn score(&self) -> f64 {
        f64::from(self.inner.score)
    }

    /// Zero-indexed rank within the result set.
    #[napi(getter)]
    #[must_use]
    pub fn rank(&self) -> u32 {
        // Ranks are small counters; a result set larger than u32::MAX is not
        // reachable in practice.
        #[allow(clippy::cast_possible_truncation)]
        {
            self.inner.rank as u32
        }
    }

    /// The matching document.
    #[napi(getter)]
    #[must_use]
    pub fn document(&self) -> NapiDocument {
        NapiDocument {
            inner: self.inner.document.clone(),
        }
    }

    /// Convenience shortcut: the matched document's content string.
    #[napi(getter)]
    #[must_use]
    pub fn content(&self) -> String {
        self.inner.document.content.clone()
    }

    /// Convenience shortcut: the matched document ID string.
    #[napi(getter)]
    #[must_use]
    pub fn document_id(&self) -> String {
        self.inner.document.id.to_string()
    }
}

// ─────────────────────────────────────────────────────────────────────────────
// NapiQuery
// ─────────────────────────────────────────────────────────────────────────────

/// A query to run through the `OxiRAG` pipeline.
///
/// All setter methods return a **new** `NapiQuery`; the original is unchanged.
///
/// @example
/// ```js
/// const q = new Query("What is Rust?").withTopK(5).withMinScore(0.4);
/// ```
#[napi]
pub struct NapiQuery {
    pub(crate) inner: Query,
}

#[napi]
impl NapiQuery {
    /// Construct a query from a plain text string.
    #[napi(constructor)]
    #[must_use]
    pub fn new(text: String) -> Self {
        Self {
            inner: Query::new(text),
        }
    }

    /// Return a new query with `top_k` overridden.
    ///
    /// `top_k` controls the maximum number of Echo-layer results returned.
    #[napi]
    #[must_use]
    pub fn with_top_k(&self, top_k: u32) -> NapiQuery {
        NapiQuery {
            inner: self.inner.clone().with_top_k(top_k as usize),
        }
    }

    /// Return a new query with `min_score` overridden.
    ///
    /// Results with a similarity score below `score` are discarded.
    ///
    /// # Errors
    ///
    /// This method is infallible; returned for API symmetry only.
    #[napi]
    #[must_use]
    pub fn with_min_score(&self, score: f64) -> NapiQuery {
        #[allow(clippy::cast_possible_truncation)]
        let score_f32 = score as f32;
        NapiQuery {
            inner: self.inner.clone().with_min_score(score_f32),
        }
    }

    /// The query text.
    #[napi(getter)]
    #[must_use]
    pub fn text(&self) -> String {
        self.inner.text.clone()
    }

    /// Maximum number of results to retrieve.
    #[napi(getter)]
    #[must_use]
    pub fn top_k(&self) -> u32 {
        // top_k is a small limit value; truncation is not reachable in practice.
        #[allow(clippy::cast_possible_truncation)]
        {
            self.inner.top_k as u32
        }
    }

    /// Minimum similarity score threshold, or `null` if not set.
    #[napi(getter)]
    #[must_use]
    pub fn min_score(&self) -> Option<f64> {
        self.inner.min_score.map(f64::from)
    }
}

// ─────────────────────────────────────────────────────────────────────────────
// Unit tests (pure Rust, no N-API runtime needed)
// ─────────────────────────────────────────────────────────────────────────────

#[cfg(all(test, not(target_arch = "wasm32")))]
mod tests {
    use super::*;

    #[test]
    fn test_napi_document_roundtrip() {
        let doc = NapiDocument::new("test content".into(), Some("Test Title".into()));
        assert_eq!(doc.content(), "test content");
        assert_eq!(doc.title(), Some("Test Title".into()));
        assert!(!doc.id().is_empty(), "document id should not be empty");
    }

    #[test]
    fn test_napi_document_no_title() {
        let doc = NapiDocument::new("content only".into(), None);
        assert_eq!(doc.content(), "content only");
        assert!(doc.title().is_none());
        assert!(doc.source().is_none());
    }

    #[test]
    fn test_napi_document_with_metadata() {
        let doc = NapiDocument::new("base content".into(), None)
            .with_metadata("lang".into(), "en".into())
            .with_metadata("author".into(), "tester".into());
        assert_eq!(doc.content(), "base content");
        assert!(!doc.id().is_empty());
        // Inner fields are accessible within the crate.
        assert_eq!(doc.inner.metadata.get("lang"), Some(&"en".to_string()));
        assert_eq!(
            doc.inner.metadata.get("author"),
            Some(&"tester".to_string())
        );
    }

    #[test]
    fn test_napi_document_with_source() {
        let doc =
            NapiDocument::new("sourced".into(), None).with_source("https://example.com".into());
        assert_eq!(doc.source(), Some("https://example.com".into()));
    }

    #[test]
    fn test_napi_document_id_uniqueness() {
        let doc1 = NapiDocument::new("a".into(), None);
        let doc2 = NapiDocument::new("b".into(), None);
        assert_ne!(doc1.id(), doc2.id());
    }

    #[test]
    fn test_napi_query_defaults() {
        let q = NapiQuery::new("test query".into());
        assert_eq!(q.text(), "test query");
        assert_eq!(q.top_k(), 10, "default top_k should be 10");
        assert!(q.min_score().is_none());
    }

    #[test]
    fn test_napi_query_fluent() {
        let q = NapiQuery::new("hello world".into())
            .with_top_k(5)
            .with_min_score(0.5);
        assert_eq!(q.text(), "hello world");
        assert_eq!(q.top_k(), 5);
        assert!(
            (q.min_score().expect("min_score should be set") - 0.5).abs() < 1e-6,
            "min_score should be approximately 0.5"
        );
    }

    #[test]
    fn test_napi_query_chaining_immutable() {
        let q1 = NapiQuery::new("query".into());
        let q2 = q1.with_top_k(3);
        assert_eq!(q1.top_k(), 10, "original top_k should be unchanged");
        assert_eq!(q2.top_k(), 3);
    }

    #[test]
    fn test_napi_query_min_score_roundtrip() {
        let q = NapiQuery::new("x".into()).with_min_score(0.75);
        let score = q.min_score().expect("min_score should be set");
        assert!((score - 0.75).abs() < 1e-5, "expected ~0.75, got {score}");
    }
}
