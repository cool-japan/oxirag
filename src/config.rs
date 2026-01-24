//! Configuration management for `OxiRAG`.

use serde::{Deserialize, Serialize};
use std::path::PathBuf;

/// Global configuration for `OxiRAG`.
#[derive(Debug, Clone, Default, Serialize, Deserialize)]
pub struct OxiRagConfig {
    /// Echo layer configuration.
    pub echo: EchoConfig,
    /// Speculator layer configuration.
    pub speculator: SpeculatorConfig,
    /// Judge layer configuration.
    pub judge: JudgeConfig,
    /// Graph layer configuration.
    #[cfg(feature = "graphrag")]
    pub graph: GraphConfig,
    /// Pipeline configuration.
    pub pipeline: PipelineConfig,
}

/// Configuration for the Echo (semantic search) layer.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct EchoConfig {
    /// Embedding model identifier.
    pub model_id: String,
    /// Path to model weights (if local).
    pub model_path: Option<PathBuf>,
    /// Embedding dimension.
    pub dimension: usize,
    /// Maximum batch size for embedding.
    pub batch_size: usize,
    /// Similarity metric to use.
    pub similarity_metric: SimilarityMetricConfig,
    /// Whether to normalize embeddings.
    pub normalize: bool,
    /// Cache directory for models.
    pub cache_dir: Option<PathBuf>,
}

impl Default for EchoConfig {
    fn default() -> Self {
        Self {
            model_id: "sentence-transformers/all-MiniLM-L6-v2".to_string(),
            model_path: None,
            dimension: 384,
            batch_size: 32,
            similarity_metric: SimilarityMetricConfig::Cosine,
            normalize: true,
            cache_dir: None,
        }
    }
}

/// Similarity metric configuration.
#[derive(Debug, Clone, Copy, Default, PartialEq, Eq, Serialize, Deserialize)]
pub enum SimilarityMetricConfig {
    /// Cosine similarity.
    #[default]
    Cosine,
    /// Euclidean distance (converted to similarity).
    Euclidean,
    /// Dot product.
    DotProduct,
}

/// Configuration for the Speculator (draft verification) layer.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct SpeculatorConfig {
    /// Model identifier for the small language model.
    pub model_id: String,
    /// Path to model weights (if local).
    pub model_path: Option<PathBuf>,
    /// Temperature for generation.
    pub temperature: f32,
    /// Top-p sampling parameter.
    pub top_p: f32,
    /// Maximum context length.
    pub max_context_length: usize,
    /// Maximum tokens to generate.
    pub max_new_tokens: usize,
    /// Whether to use sampling.
    pub do_sample: bool,
    /// Cache directory for models.
    pub cache_dir: Option<PathBuf>,
}

impl Default for SpeculatorConfig {
    fn default() -> Self {
        Self {
            model_id: "microsoft/phi-2".to_string(),
            model_path: None,
            temperature: 0.7,
            top_p: 0.9,
            max_context_length: 2048,
            max_new_tokens: 256,
            do_sample: true,
            cache_dir: None,
        }
    }
}

/// Configuration for the Judge (logic verification) layer.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct JudgeConfig {
    /// Timeout for SMT solving in milliseconds.
    pub timeout_ms: u64,
    /// Maximum number of claims to verify.
    pub max_claims: usize,
    /// Whether to check consistency between claims.
    pub check_consistency: bool,
    /// Minimum confidence for claim extraction.
    pub min_claim_confidence: f32,
    /// Whether to generate counterexamples.
    pub generate_counterexamples: bool,
}

impl Default for JudgeConfig {
    fn default() -> Self {
        Self {
            timeout_ms: 5000,
            max_claims: 10,
            check_consistency: true,
            min_claim_confidence: 0.5,
            generate_counterexamples: true,
        }
    }
}

/// Configuration for retry logic with exponential backoff.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct RetryConfig {
    /// Maximum number of retry attempts.
    pub max_retries: usize,
    /// Initial delay between retries in milliseconds.
    pub initial_delay_ms: u64,
    /// Maximum delay between retries in milliseconds.
    pub max_delay_ms: u64,
    /// Multiplier for exponential backoff.
    pub backoff_multiplier: f64,
    /// Whether to add random jitter to delay.
    pub add_jitter: bool,
}

impl Default for RetryConfig {
    fn default() -> Self {
        Self {
            max_retries: 3,
            initial_delay_ms: 100,
            max_delay_ms: 5000,
            backoff_multiplier: 2.0,
            add_jitter: true,
        }
    }
}

impl RetryConfig {
    /// Create a new `RetryConfig` with default values.
    #[must_use]
    pub fn new() -> Self {
        Self::default()
    }

    /// Set the maximum number of retries.
    #[must_use]
    pub const fn with_max_retries(mut self, max_retries: usize) -> Self {
        self.max_retries = max_retries;
        self
    }

    /// Set the initial delay in milliseconds.
    #[must_use]
    pub const fn with_initial_delay_ms(mut self, initial_delay_ms: u64) -> Self {
        self.initial_delay_ms = initial_delay_ms;
        self
    }

    /// Set the maximum delay in milliseconds.
    #[must_use]
    pub const fn with_max_delay_ms(mut self, max_delay_ms: u64) -> Self {
        self.max_delay_ms = max_delay_ms;
        self
    }

    /// Set the backoff multiplier.
    #[must_use]
    pub const fn with_backoff_multiplier(mut self, backoff_multiplier: f64) -> Self {
        self.backoff_multiplier = backoff_multiplier;
        self
    }

    /// Set whether to add jitter.
    #[must_use]
    pub const fn with_jitter(mut self, add_jitter: bool) -> Self {
        self.add_jitter = add_jitter;
        self
    }
}

/// Configuration for the unified pipeline.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct PipelineConfig {
    /// Threshold for skipping speculation (high-confidence Echo results).
    pub fast_path_threshold: f32,
    /// Threshold for skipping verification (high-confidence speculation).
    pub skip_verification_threshold: f32,
    /// Whether to enable fast-path optimizations.
    pub enable_fast_path: bool,
    /// Maximum retries for failed operations (deprecated: use `retry_config` instead).
    pub max_retries: usize,
    /// Whether to run layers in parallel where possible.
    pub parallel_execution: bool,
    /// Retry configuration with exponential backoff.
    pub retry_config: Option<RetryConfig>,
}

impl Default for PipelineConfig {
    fn default() -> Self {
        Self {
            fast_path_threshold: 0.95,
            skip_verification_threshold: 0.9,
            enable_fast_path: true,
            max_retries: 3,
            parallel_execution: false,
            retry_config: Some(RetryConfig::default()),
        }
    }
}

/// Configuration for the Graph (knowledge graph) layer.
#[cfg(feature = "graphrag")]
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct GraphConfig {
    /// Maximum number of hops for graph traversal.
    pub max_hops: usize,
    /// Minimum confidence threshold for entity extraction.
    pub min_entity_confidence: f32,
    /// Minimum confidence threshold for relationship extraction.
    pub min_relationship_confidence: f32,
    /// Minimum confidence threshold for traversal paths.
    pub min_path_confidence: f32,
    /// Whether to enable hybrid search (combine vector + graph).
    pub enable_hybrid_search: bool,
    /// Weight for graph results in hybrid search (0.0 to 1.0).
    pub graph_weight: f32,
    /// Maximum entities to extract per document.
    pub max_entities_per_doc: usize,
    /// Maximum relationships to extract per document.
    pub max_relationships_per_doc: usize,
}

#[cfg(feature = "graphrag")]
impl Default for GraphConfig {
    fn default() -> Self {
        Self {
            max_hops: 3,
            min_entity_confidence: 0.5,
            min_relationship_confidence: 0.4,
            min_path_confidence: 0.3,
            enable_hybrid_search: true,
            graph_weight: 0.3,
            max_entities_per_doc: 50,
            max_relationships_per_doc: 100,
        }
    }
}

impl OxiRagConfig {
    /// Create a new configuration with default values.
    #[must_use]
    pub fn new() -> Self {
        Self::default()
    }

    /// Load configuration from a file (native only).
    ///
    /// # Errors
    ///
    /// Returns an error if the file cannot be read or parsed.
    #[cfg(feature = "native")]
    pub fn from_file(path: impl AsRef<std::path::Path>) -> crate::error::Result<Self> {
        let content = std::fs::read_to_string(path)?;
        let config: Self = serde_json::from_str(&content)?;
        Ok(config)
    }

    /// Save configuration to a file (native only).
    ///
    /// # Errors
    ///
    /// Returns an error if the file cannot be written.
    #[cfg(feature = "native")]
    pub fn to_file(&self, path: impl AsRef<std::path::Path>) -> crate::error::Result<()> {
        let content = serde_json::to_string_pretty(self)?;
        std::fs::write(path, content)?;
        Ok(())
    }

    /// Load configuration from a JSON string.
    ///
    /// # Errors
    ///
    /// Returns an error if the JSON cannot be parsed.
    pub fn from_json(json: &str) -> crate::error::Result<Self> {
        let config: Self = serde_json::from_str(json)?;
        Ok(config)
    }

    /// Serialize configuration to a JSON string.
    ///
    /// # Errors
    ///
    /// Returns an error if serialization fails.
    pub fn to_json(&self) -> crate::error::Result<String> {
        let content = serde_json::to_string_pretty(self)?;
        Ok(content)
    }

    /// Set Echo configuration.
    #[must_use]
    pub fn with_echo(mut self, echo: EchoConfig) -> Self {
        self.echo = echo;
        self
    }

    /// Set Speculator configuration.
    #[must_use]
    pub fn with_speculator(mut self, speculator: SpeculatorConfig) -> Self {
        self.speculator = speculator;
        self
    }

    /// Set Judge configuration.
    #[must_use]
    pub fn with_judge(mut self, judge: JudgeConfig) -> Self {
        self.judge = judge;
        self
    }

    /// Set Pipeline configuration.
    #[must_use]
    pub fn with_pipeline(mut self, pipeline: PipelineConfig) -> Self {
        self.pipeline = pipeline;
        self
    }

    /// Set Graph configuration.
    #[cfg(feature = "graphrag")]
    #[must_use]
    pub fn with_graph(mut self, graph: GraphConfig) -> Self {
        self.graph = graph;
        self
    }
}

#[cfg(test)]
#[allow(clippy::float_cmp)]
mod tests {
    use super::*;

    #[test]
    fn test_default_config() {
        let config = OxiRagConfig::default();
        assert_eq!(config.echo.dimension, 384);
        assert_eq!(config.speculator.temperature, 0.7);
        assert_eq!(config.judge.timeout_ms, 5000);
        assert_eq!(config.pipeline.fast_path_threshold, 0.95);
    }

    #[test]
    fn test_config_builder() {
        let config = OxiRagConfig::new()
            .with_echo(EchoConfig {
                dimension: 768,
                ..Default::default()
            })
            .with_pipeline(PipelineConfig {
                enable_fast_path: false,
                ..Default::default()
            });

        assert_eq!(config.echo.dimension, 768);
        assert!(!config.pipeline.enable_fast_path);
    }

    #[test]
    fn test_config_serialization() {
        let config = OxiRagConfig::default();
        let json = serde_json::to_string(&config).unwrap();
        let parsed: OxiRagConfig = serde_json::from_str(&json).unwrap();
        assert_eq!(config.echo.dimension, parsed.echo.dimension);
    }
}
