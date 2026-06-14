//! Types for the `gen_read` module.

use thiserror::Error;

// ── ContextGenerator ─────────────────────────────────────────────────────────────

/// A generator that produces a *contextual document* for a query.
///
/// In `GenRead` (Yu et al., 2023, "Generate rather than Retrieve") a large
/// language model is prompted to *write* documents that plausibly contain the
/// answer, and those generated documents — not retrieved passages — become the
/// reading context. Sampling several diverse documents and clustering them is
/// what gives the method its breadth, so the `index` argument lets an
/// implementation vary its output deterministically across samples.
pub trait ContextGenerator {
    /// Generate a contextual document for the query.
    ///
    /// `index` enables deterministic diversity across samples: calling with
    /// `0, 1, 2, ...` should yield distinct documents, and the same `index`
    /// must always yield the same document for a given query.
    fn generate(&self, query: &str, index: usize) -> String;
}

// ── MockContextGenerator ─────────────────────────────────────────────────────────

/// Deterministic [`ContextGenerator`] that replays a scripted list of documents.
///
/// The document at position `index` (modulo the script length) is returned, so
/// the generator is fully determined by its script and the requested index.
/// When the script is empty a synthesized placeholder containing the query and
/// index is returned instead, guaranteeing non-empty, distinct output.
#[derive(Debug, Clone, Default)]
pub struct MockContextGenerator {
    /// The scripted documents, returned by index (wrapping on overflow).
    docs: Vec<String>,
}

impl MockContextGenerator {
    /// Create a generator from a list of scripted documents.
    #[must_use]
    pub fn new(docs: Vec<String>) -> Self {
        Self { docs }
    }

    /// The scripted documents backing this generator.
    #[must_use]
    pub fn docs(&self) -> &[String] {
        &self.docs
    }
}

impl ContextGenerator for MockContextGenerator {
    fn generate(&self, query: &str, index: usize) -> String {
        if self.docs.is_empty() {
            return format!("[gen-read doc {index}] {}", query.trim());
        }
        self.docs[index % self.docs.len()].clone()
    }
}

// ── GeneratedDoc ─────────────────────────────────────────────────────────────────

/// A single generated contextual document with its assigned cluster.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct GeneratedDoc {
    /// The generated document text.
    pub content: String,
    /// Index of the cluster (into the cluster list) this document belongs to.
    pub cluster_id: usize,
}

// ── GenReadOutput ────────────────────────────────────────────────────────────────

/// Result of a generate-then-read pass.
#[derive(Debug, Clone)]
pub struct GenReadOutput {
    /// Every generated document, each tagged with its cluster.
    pub documents: Vec<GeneratedDoc>,
    /// Number of non-empty clusters formed from the generated documents.
    pub clusters: usize,
    /// The assembled reading context: one representative per cluster, joined
    /// with the configured separator.
    pub context: String,
}

// ── GenReadConfig ────────────────────────────────────────────────────────────────

/// Configuration for [`GenReadEngine`](crate::gen_read::GenReadEngine).
///
/// `num_docs` contextual documents are generated, clustered into at most
/// `num_clusters` groups using `dim`-dimensional lexical pseudo-embeddings, and
/// one representative per cluster is joined with `join_separator` to form the
/// reading context.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct GenReadConfig {
    /// Number of contextual documents to generate. Defaults to `5`.
    pub num_docs: usize,
    /// Maximum number of clusters to group the documents into. Defaults to `2`.
    pub num_clusters: usize,
    /// Dimensionality of the lexical pseudo-embeddings used for clustering.
    ///
    /// Defaults to `128`.
    pub dim: usize,
    /// Separator joining the per-cluster representatives into the context.
    ///
    /// Defaults to `"\n\n"`.
    pub join_separator: String,
}

impl Default for GenReadConfig {
    fn default() -> Self {
        Self {
            num_docs: 5,
            num_clusters: 2,
            dim: 128,
            join_separator: "\n\n".to_string(),
        }
    }
}

impl GenReadConfig {
    /// Create a config with default values.
    #[must_use]
    pub fn new() -> Self {
        Self::default()
    }

    /// Set the number of contextual documents to generate.
    #[must_use]
    pub fn with_num_docs(mut self, num_docs: usize) -> Self {
        self.num_docs = num_docs;
        self
    }

    /// Set the maximum number of clusters.
    #[must_use]
    pub fn with_num_clusters(mut self, num_clusters: usize) -> Self {
        self.num_clusters = num_clusters;
        self
    }

    /// Set the pseudo-embedding dimensionality used for clustering.
    #[must_use]
    pub fn with_dim(mut self, dim: usize) -> Self {
        self.dim = dim;
        self
    }

    /// Set the separator joining per-cluster representatives.
    #[must_use]
    pub fn with_join_separator(mut self, join_separator: impl Into<String>) -> Self {
        self.join_separator = join_separator.into();
        self
    }
}

// ── GenReadError ─────────────────────────────────────────────────────────────────

/// Errors from the `gen_read` module.
#[derive(Debug, Error)]
pub enum GenReadError {
    /// The query was empty after trimming.
    #[error("query must not be empty")]
    EmptyQuery,
}
