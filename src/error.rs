//! Unified error types for `OxiRAG`.

use thiserror::Error;

/// The main error type for `OxiRAG` operations.
#[derive(Debug, Error)]
pub enum OxiRagError {
    /// Embedding-related errors
    #[error("Embedding error: {0}")]
    Embedding(#[from] EmbeddingError),

    /// Vector store errors
    #[error("Vector store error: {0}")]
    VectorStore(#[from] VectorStoreError),

    /// Speculator errors
    #[error("Speculator error: {0}")]
    Speculator(#[from] SpeculatorError),

    /// Judge errors
    #[error("Judge error: {0}")]
    Judge(#[from] JudgeError),

    /// Graph errors
    #[cfg(feature = "graphrag")]
    #[error("Graph error: {0}")]
    Graph(#[from] GraphError),

    /// Distillation errors
    #[cfg(feature = "distillation")]
    #[error("Distillation error: {0}")]
    Distillation(#[from] DistillationError),

    /// Prefix cache errors
    #[cfg(feature = "prefix-cache")]
    #[error("Prefix cache error: {0}")]
    PrefixCache(#[from] PrefixCacheError),

    /// Hidden state errors
    #[cfg(feature = "hidden-states")]
    #[error("Hidden state error: {0}")]
    HiddenState(#[from] HiddenStateError),

    /// Memory errors
    #[error("Memory error: {0}")]
    Memory(#[from] crate::memory::MemoryError),

    /// Pipeline errors
    #[error("Pipeline error: {0}")]
    Pipeline(#[from] PipelineError),

    /// Circuit breaker errors
    #[error("Circuit breaker error: {0}")]
    CircuitBreaker(#[from] CircuitBreakerError),

    /// Configuration errors
    #[error("Configuration error: {0}")]
    Config(String),

    /// IO errors (native only)
    #[cfg(feature = "native")]
    #[error("IO error: {0}")]
    Io(#[from] std::io::Error),

    /// IO errors (WASM - represented as string)
    #[cfg(not(feature = "native"))]
    #[error("IO error: {0}")]
    Io(String),

    /// Serialization errors
    #[error("Serialization error: {0}")]
    Serialization(#[from] serde_json::Error),
}

/// Errors related to embedding operations.
#[derive(Debug, Error)]
pub enum EmbeddingError {
    /// Model loading failed
    #[error("Failed to load embedding model: {0}")]
    ModelLoad(String),

    /// Tokenization failed
    #[error("Tokenization failed: {0}")]
    Tokenization(String),

    /// Inference failed
    #[error("Inference failed: {0}")]
    Inference(String),

    /// Dimension mismatch
    #[error("Dimension mismatch: expected {expected}, got {actual}")]
    DimensionMismatch {
        /// Expected dimension
        expected: usize,
        /// Actual dimension received
        actual: usize,
    },

    /// Empty input
    #[error("Empty input provided")]
    EmptyInput,

    /// Backend error
    #[error("Backend error: {0}")]
    Backend(String),
}

/// Errors related to vector store operations.
#[derive(Debug, Error)]
pub enum VectorStoreError {
    /// Document not found
    #[error("Document not found: {0}")]
    NotFound(String),

    /// Duplicate document ID
    #[error("Duplicate document ID: {0}")]
    DuplicateId(String),

    /// Storage capacity exceeded
    #[error("Storage capacity exceeded: max {max}, current {current}")]
    CapacityExceeded {
        /// Maximum allowed capacity
        max: usize,
        /// Current count
        current: usize,
    },

    /// Index error
    #[error("Index error: {0}")]
    Index(String),

    /// Dimension mismatch in vectors
    #[error("Vector dimension mismatch: expected {expected}, got {actual}")]
    DimensionMismatch {
        /// Expected vector dimension
        expected: usize,
        /// Actual vector dimension
        actual: usize,
    },
}

/// Errors related to speculator operations.
#[derive(Debug, Error)]
pub enum SpeculatorError {
    /// Model loading failed
    #[error("Failed to load speculator model: {0}")]
    ModelLoad(String),

    /// Generation failed
    #[error("Generation failed: {0}")]
    Generation(String),

    /// Verification failed
    #[error("Verification failed: {0}")]
    Verification(String),

    /// Invalid draft
    #[error("Invalid draft: {0}")]
    InvalidDraft(String),

    /// Context too long
    #[error("Context too long: {length} tokens exceeds max {max}")]
    ContextTooLong {
        /// Actual context length
        length: usize,
        /// Maximum allowed length
        max: usize,
    },

    /// Backend error
    #[error("Backend error: {0}")]
    Backend(String),
}

/// Errors related to judge operations.
#[derive(Debug, Error)]
pub enum JudgeError {
    /// Claim extraction failed
    #[error("Claim extraction failed: {0}")]
    ExtractionFailed(String),

    /// SMT encoding failed
    #[error("SMT encoding failed: {0}")]
    EncodingFailed(String),

    /// Solver error
    #[error("Solver error: {0}")]
    SolverError(String),

    /// Timeout during verification
    #[error("Verification timeout after {0}ms")]
    Timeout(u64),

    /// Inconsistent claims
    #[error("Inconsistent claims detected: {0}")]
    InconsistentClaims(String),

    /// Unsupported claim type
    #[error("Unsupported claim type: {0}")]
    UnsupportedClaim(String),
}

/// Errors related to pipeline operations.
#[derive(Debug, Error)]
pub enum PipelineError {
    /// Layer not configured
    #[error("Layer not configured: {0}")]
    LayerNotConfigured(String),

    /// Pipeline build error
    #[error("Pipeline build error: {0}")]
    BuildError(String),

    /// Execution error
    #[error("Pipeline execution error: {0}")]
    ExecutionError(String),

    /// Invalid threshold
    #[error("Invalid threshold: {0}")]
    InvalidThreshold(String),
}

/// Errors related to distillation operations.
#[cfg(feature = "distillation")]
#[derive(Debug, Error)]
pub enum DistillationError {
    /// Failed to track a query.
    #[error("Query tracking failed: {0}")]
    TrackingFailed(String),

    /// Failed to collect Q&A pair.
    #[error("Collection failed: {0}")]
    CollectionFailed(String),

    /// Invalid configuration.
    #[error("Invalid configuration: {0}")]
    InvalidConfig(String),

    /// Pattern not found.
    #[error("Pattern not found: {0}")]
    PatternNotFound(String),

    /// Storage error.
    #[error("Storage error: {0}")]
    StorageError(String),
}

/// Errors related to prefix cache operations.
#[cfg(feature = "prefix-cache")]
#[derive(Debug, Error)]
pub enum PrefixCacheError {
    /// Cache entry not found.
    #[error("Cache entry not found: {0}")]
    NotFound(String),

    /// Cache capacity exceeded.
    #[error("Cache capacity exceeded: {0}")]
    CapacityExceeded(String),

    /// Invalid cache configuration.
    #[error("Invalid configuration: {0}")]
    InvalidConfig(String),

    /// Fingerprint generation failed.
    #[error("Fingerprint generation failed: {0}")]
    FingerprintError(String),

    /// Entry expired.
    #[error("Entry expired: {0}")]
    Expired(String),

    /// Lock acquisition failed.
    #[error("Lock acquisition failed: {0}")]
    LockError(String),
}

/// Errors related to graph operations.
#[cfg(feature = "graphrag")]
#[derive(Debug, Error)]
pub enum GraphError {
    /// Entity not found in the graph.
    #[error("Entity not found: {0}")]
    EntityNotFound(String),

    /// Relationship not found in the graph.
    #[error("Relationship not found: {0}")]
    RelationshipNotFound(String),

    /// Entity extraction failed.
    #[error("Entity extraction failed: {0}")]
    ExtractionFailed(String),

    /// Graph traversal error.
    #[error("Traversal error: {0}")]
    TraversalError(String),

    /// Configuration error.
    #[error("Configuration error: {0}")]
    ConfigurationError(String),

    /// Storage error.
    #[error("Storage error: {0}")]
    StorageError(String),

    /// Invalid query.
    #[error("Invalid query: {0}")]
    InvalidQuery(String),
}

/// Errors related to circuit breaker operations.
#[derive(Debug, Error)]
pub enum CircuitBreakerError {
    /// Circuit breaker is open and rejecting requests.
    #[error("Circuit breaker is open: {0}")]
    Open(String),

    /// Circuit breaker is in half-open state with max requests reached.
    #[error("Circuit breaker is half-open, max requests reached: {0}")]
    HalfOpenMaxRequests(String),

    /// Invalid circuit breaker configuration.
    #[error("Invalid configuration: {0}")]
    InvalidConfig(String),
}

/// Errors related to hidden state operations.
#[cfg(feature = "hidden-states")]
#[derive(Debug, Clone, Error)]
pub enum HiddenStateError {
    /// Shape mismatch between tensors.
    #[error("Shape mismatch: expected {expected:?}, got {actual:?}")]
    ShapeMismatch {
        /// Expected shape dimensions.
        expected: Vec<usize>,
        /// Actual shape dimensions.
        actual: Vec<usize>,
    },

    /// Invalid dimension for operation.
    #[error("Invalid dimension: {0}")]
    InvalidDimension(String),

    /// Cache operation failed.
    #[error("Cache error: {0}")]
    CacheError(String),

    /// Provider operation failed.
    #[error("Provider error: {0}")]
    ProviderError(String),
}

/// A type alias for Results with [`OxiRagError`].
pub type Result<T> = std::result::Result<T, OxiRagError>;

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_error_display() {
        let err = OxiRagError::Config("invalid setting".into());
        assert_eq!(err.to_string(), "Configuration error: invalid setting");
    }

    #[test]
    fn test_embedding_error_conversion() {
        let emb_err = EmbeddingError::EmptyInput;
        let err: OxiRagError = emb_err.into();
        assert!(matches!(err, OxiRagError::Embedding(_)));
    }

    #[test]
    fn test_vector_store_error_conversion() {
        let vs_err = VectorStoreError::NotFound("doc123".into());
        let err: OxiRagError = vs_err.into();
        assert!(matches!(err, OxiRagError::VectorStore(_)));
    }
}
