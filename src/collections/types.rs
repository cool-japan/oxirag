//! Core types for the `collections` module.
//!
//! This module defines multi-tenant collection identifiers, configuration,
//! metadata, statistics, and error types.

use serde::{Deserialize, Serialize};
use std::time::SystemTime;
use thiserror::Error;

// ── CollectionId ─────────────────────────────────────────────────────────────

/// A normalised, URL-safe identifier for a vector-store collection.
///
/// Names are normalised on construction: all characters are lowercased,
/// spaces and hyphens are converted to underscores, and any character that is
/// neither alphanumeric nor an underscore is rejected with
/// [`CollectionError::InvalidName`].
#[derive(Debug, Clone, PartialEq, Eq, Hash, Serialize, Deserialize)]
pub struct CollectionId(String);

impl CollectionId {
    /// Create a new [`CollectionId`] from `name`, normalising it.
    ///
    /// Normalisation rules:
    /// - All ASCII letters are lowercased.
    /// - Spaces (`' '`) and hyphens (`'-'`) are mapped to underscores.
    /// - Any remaining character outside `[a-z0-9_]` causes an error.
    /// - Empty names (after normalisation) are rejected.
    ///
    /// # Errors
    ///
    /// Returns [`CollectionError::InvalidName`] if the normalised name would be
    /// empty or contain invalid characters.
    pub fn new(name: &str) -> Result<Self, CollectionError> {
        let normalised: String = name
            .chars()
            .map(|c| match c {
                ' ' | '-' => '_',
                other => other.to_ascii_lowercase(),
            })
            .collect();

        if normalised.is_empty() {
            return Err(CollectionError::InvalidName(format!(
                "collection name must not be empty (got {name:?})"
            )));
        }

        for ch in normalised.chars() {
            if !ch.is_ascii_alphanumeric() && ch != '_' {
                return Err(CollectionError::InvalidName(format!(
                    "collection name {name:?} contains invalid character {ch:?} after normalisation"
                )));
            }
        }

        Ok(Self(normalised))
    }

    /// Return the normalised name string.
    #[must_use]
    pub fn as_str(&self) -> &str {
        &self.0
    }
}

impl std::fmt::Display for CollectionId {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.write_str(&self.0)
    }
}

// ── CollectionMetadata ────────────────────────────────────────────────────────

/// Human-readable metadata attached to a collection.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct CollectionMetadata {
    /// Optional free-text description.
    pub description: Option<String>,
    /// When the collection was created.
    pub created_at: SystemTime,
    /// When the collection metadata was last modified.
    pub updated_at: SystemTime,
    /// Arbitrary user-defined tags for grouping / filtering collections.
    pub tags: Vec<String>,
}

impl Default for CollectionMetadata {
    fn default() -> Self {
        let now = crate::time::system_now();
        Self {
            description: None,
            created_at: now,
            updated_at: now,
            tags: Vec::new(),
        }
    }
}

impl CollectionMetadata {
    /// Create metadata with an optional description and tags.
    #[must_use]
    pub fn new(description: Option<String>, tags: Vec<String>) -> Self {
        let now = crate::time::system_now();
        Self {
            description,
            created_at: now,
            updated_at: now,
            tags,
        }
    }
}

// ── SimilarityMetric ──────────────────────────────────────────────────────────

/// The similarity metric used when searching within a collection.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize, Default)]
pub enum SimilarityMetric {
    /// Cosine similarity (angle between vectors).
    #[default]
    Cosine,
    /// Euclidean distance (converted to a similarity score).
    Euclidean,
    /// Raw dot product.
    DotProduct,
}

// ── CollectionConfig ──────────────────────────────────────────────────────────

/// Configuration parameters for a collection.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct CollectionConfig {
    /// The dimensionality of vectors stored in this collection.
    pub embedding_dimension: usize,
    /// Optional hard cap on the number of documents.  `None` means unlimited.
    pub max_documents: Option<usize>,
    /// Default `top_k` for searches that do not specify one.
    pub default_top_k: usize,
    /// Similarity metric used for ANN search.
    pub similarity_metric: SimilarityMetric,
}

impl Default for CollectionConfig {
    fn default() -> Self {
        Self {
            embedding_dimension: 384,
            max_documents: None,
            default_top_k: 10,
            similarity_metric: SimilarityMetric::Cosine,
        }
    }
}

impl CollectionConfig {
    /// Create a config with the given embedding dimension.
    #[must_use]
    pub fn new(embedding_dimension: usize) -> Self {
        Self {
            embedding_dimension,
            ..Default::default()
        }
    }

    /// Set the maximum number of documents.
    #[must_use]
    pub fn with_max_documents(mut self, max: usize) -> Self {
        self.max_documents = Some(max);
        self
    }

    /// Set the default top-k value.
    #[must_use]
    pub fn with_default_top_k(mut self, k: usize) -> Self {
        self.default_top_k = k;
        self
    }

    /// Set the similarity metric.
    #[must_use]
    pub fn with_similarity_metric(mut self, metric: SimilarityMetric) -> Self {
        self.similarity_metric = metric;
        self
    }
}

// ── Collection ────────────────────────────────────────────────────────────────

/// A named, configured vector-store collection.
///
/// Fields are read-only from outside the module; mutation happens through the
/// `CollectionStore` trait.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Collection {
    /// Normalised identifier.
    pub(crate) id: CollectionId,
    /// Configuration set at creation time.
    pub(crate) config: CollectionConfig,
    /// User-supplied metadata.
    pub(crate) metadata: CollectionMetadata,
    /// Current number of indexed documents.
    pub(crate) document_count: usize,
}

impl Collection {
    /// Create a new collection (internal constructor).
    pub(crate) fn new(
        id: CollectionId,
        config: CollectionConfig,
        metadata: CollectionMetadata,
    ) -> Self {
        Self {
            id,
            config,
            metadata,
            document_count: 0,
        }
    }

    /// Returns the collection's unique identifier.
    #[must_use]
    pub fn id(&self) -> &CollectionId {
        &self.id
    }

    /// Returns the collection's configuration.
    #[must_use]
    pub fn config(&self) -> &CollectionConfig {
        &self.config
    }

    /// Returns the collection's metadata.
    #[must_use]
    pub fn metadata(&self) -> &CollectionMetadata {
        &self.metadata
    }

    /// Returns the number of documents currently indexed in this collection.
    #[must_use]
    pub fn document_count(&self) -> usize {
        self.document_count
    }
}

// ── CollectionStats ───────────────────────────────────────────────────────────

/// Runtime statistics for a single collection.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct CollectionStats {
    /// The collection this stat snapshot belongs to.
    pub id: CollectionId,
    /// Number of documents currently indexed.
    pub document_count: usize,
    /// Running average search latency in milliseconds.
    pub avg_search_latency_ms: f64,
    /// Total number of search operations performed.
    pub total_searches: u64,
}

impl CollectionStats {
    /// Create initial (zero) stats for `id`.
    #[must_use]
    pub fn new(id: CollectionId) -> Self {
        Self {
            id,
            document_count: 0,
            avg_search_latency_ms: 0.0,
            total_searches: 0,
        }
    }

    /// Update the running average latency with a new measurement.
    ///
    /// Uses the recurrence: `avg_new = (avg_old * n + new) / (n + 1)`.
    pub fn record_search(&mut self, latency_ms: f64) {
        #[allow(clippy::cast_precision_loss)]
        let n = self.total_searches as f64;
        self.avg_search_latency_ms = (self.avg_search_latency_ms * n + latency_ms) / (n + 1.0);
        self.total_searches = self.total_searches.saturating_add(1);
    }
}

// ── FederatedResult ───────────────────────────────────────────────────────────

/// A search result from a cross-collection (federated) query.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct FederatedResult {
    /// Text content of the matched document.
    pub content: String,
    /// Reciprocal-Rank-Fusion score (higher is better).
    pub score: f32,
    /// Name of the collection this result came from.
    pub collection_id: String,
    /// Unique document identifier (as a string).
    pub document_id: String,
    /// Rank within the final merged result list (0-indexed).
    pub rank: usize,
}

// ── CollectionError ───────────────────────────────────────────────────────────

/// Errors that can occur in the `collections` subsystem.
#[derive(Debug, Error)]
pub enum CollectionError {
    /// No collection with the given name exists.
    #[error("collection not found: {0}")]
    NotFound(String),

    /// A collection with the given name already exists.
    #[error("collection already exists: {0}")]
    AlreadyExists(String),

    /// The embedding dimension provided does not match the collection's schema.
    #[error("dimension mismatch: expected {expected}, got {got}")]
    DimensionMismatch {
        /// The dimension the collection was created with.
        expected: usize,
        /// The dimension of the incoming vector.
        got: usize,
    },

    /// The collection has reached its document capacity.
    #[error("capacity exceeded: {0}")]
    CapacityExceeded(String),

    /// The supplied collection name is invalid.
    #[error("invalid collection name: {0}")]
    InvalidName(String),

    /// An underlying storage operation failed.
    #[error("storage error: {0}")]
    StorageError(String),
}
