#![cfg(feature = "nodejs")]
#![allow(missing_docs)]
//! Fluent builder for [`NapiPipeline`].
//!
//! All setter methods return a **new** `NapiPipelineBuilder` (immutable fluent
//! API), matching the pattern used by the Python bindings.

use std::sync::Arc;

use napi_derive::napi;

use crate::layer1_echo::{EchoLayer, InMemoryVectorStore, MockEmbeddingProvider};
use crate::layer2_speculator::RuleBasedSpeculator;
use crate::layer3_judge::{AdvancedClaimExtractor, JudgeConfig, JudgeImpl, MockSmtVerifier};
use crate::nodejs::pipeline::NapiPipeline;
use crate::observability::MemoryObserver;
use crate::pipeline::{PipelineBuilder, PipelineConfig};

// ─────────────────────────────────────────────────────────────────────────────
// NapiPipelineBuilder
// ─────────────────────────────────────────────────────────────────────────────

/// Fluent builder for constructing a [`NapiPipeline`] from JavaScript.
///
/// All setter methods are immutable and return a new builder, making the API
/// safe to call in chains without mutation of the original.
///
/// @example
/// ```js
/// const pipeline = new PipelineBuilder()
///   .withDimension(384)
///   .withMaxResults(10)
///   .build();
/// ```
#[napi]
pub struct NapiPipelineBuilder {
    /// Embedding vector dimension (must match the provider).
    dimension: usize,
    /// Maximum Echo-layer search results forwarded to Layer 2 and Layer 3.
    max_results: usize,
    /// Whether to enable the fast-path short-circuit optimisation.
    enable_fast_path: bool,
    /// Fast-path confidence threshold (0.0–1.0).
    fast_path_threshold: f64,
}

impl Default for NapiPipelineBuilder {
    fn default() -> Self {
        Self {
            dimension: 384,
            max_results: 10,
            enable_fast_path: true,
            fast_path_threshold: 0.95,
        }
    }
}

#[napi]
impl NapiPipelineBuilder {
    /// Create a builder with sensible defaults.
    ///
    /// - `dimension`: 384
    /// - `maxResults`: 10
    /// - `fastPath`: enabled at threshold 0.95
    #[napi(constructor)]
    #[must_use]
    pub fn new() -> Self {
        Self::default()
    }

    /// Return a new builder with the embedding vector dimension overridden.
    ///
    /// The dimension must match the number of floats produced by the embedding
    /// provider.  The default mock provider supports any dimension.
    #[napi]
    #[must_use]
    pub fn with_dimension(&self, dim: u32) -> NapiPipelineBuilder {
        NapiPipelineBuilder {
            dimension: dim as usize,
            max_results: self.max_results,
            enable_fast_path: self.enable_fast_path,
            fast_path_threshold: self.fast_path_threshold,
        }
    }

    /// Return a new builder with the maximum search results overridden.
    ///
    /// Controls how many Echo-layer results are forwarded to Layer 2
    /// (Speculator) and Layer 3 (Judge).
    #[napi]
    #[must_use]
    pub fn with_max_results(&self, k: u32) -> NapiPipelineBuilder {
        NapiPipelineBuilder {
            dimension: self.dimension,
            max_results: k as usize,
            enable_fast_path: self.enable_fast_path,
            fast_path_threshold: self.fast_path_threshold,
        }
    }

    /// Return a new builder with the fast-path flag overridden.
    ///
    /// When enabled, queries whose top Echo score exceeds `fastPathThreshold`
    /// skip Layer 2 and Layer 3.
    #[napi]
    #[must_use]
    pub fn with_fast_path(&self, enable: bool) -> NapiPipelineBuilder {
        NapiPipelineBuilder {
            dimension: self.dimension,
            max_results: self.max_results,
            enable_fast_path: enable,
            fast_path_threshold: self.fast_path_threshold,
        }
    }

    /// Return a new builder with the fast-path confidence threshold overridden.
    ///
    /// The threshold must be in `[0.0, 1.0]`.  Default is 0.95.
    #[napi]
    #[must_use]
    pub fn with_fast_path_threshold(&self, threshold: f64) -> NapiPipelineBuilder {
        NapiPipelineBuilder {
            dimension: self.dimension,
            max_results: self.max_results,
            enable_fast_path: self.enable_fast_path,
            fast_path_threshold: threshold,
        }
    }

    /// Build and return a ready-to-use [`NapiPipeline`].
    ///
    /// This is a synchronous call — no `await` needed.
    ///
    /// # Errors
    ///
    /// Returns an `Error` if the internal pipeline construction fails.  This
    /// should not happen with the default in-memory configuration.
    #[napi]
    pub fn build(&self) -> napi::Result<NapiPipeline> {
        let echo = EchoLayer::new(
            MockEmbeddingProvider::new(self.dimension),
            InMemoryVectorStore::new(self.dimension),
        );
        let speculator = RuleBasedSpeculator::default();
        let judge = JudgeImpl::new(
            AdvancedClaimExtractor::new(),
            MockSmtVerifier::default(),
            JudgeConfig::default(),
        );
        #[allow(clippy::cast_possible_truncation)]
        let config = PipelineConfig {
            max_search_results: self.max_results,
            enable_fast_path: self.enable_fast_path,
            fast_path_threshold: self.fast_path_threshold as f32,
            ..PipelineConfig::default()
        };
        let observer = Arc::new(MemoryObserver::new());
        let pipeline = PipelineBuilder::new()
            .with_echo(echo)
            .with_speculator(speculator)
            .with_judge(judge)
            .with_config(config)
            .with_observers(vec![
                Arc::clone(&observer) as Arc<dyn crate::observability::SpanObserver>
            ])
            .build()
            .map_err(|e| napi::Error::from_reason(e.to_string()))?;
        Ok(NapiPipeline {
            inner: Arc::new(tokio::sync::Mutex::new(pipeline)),
            observer,
        })
    }
}
