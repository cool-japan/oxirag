//! Unified RAG pipeline combining all three layers.

use crate::time::Instant;
use async_trait::async_trait;
use std::sync::Arc;
use std::time::Duration;

use crate::config::RetryConfig;
use crate::error::{OxiRagError, PipelineError};
use crate::layer1_echo::Echo;
use crate::layer2_speculator::Speculator;
use crate::layer3_judge::Judge;
use crate::observability::{PipelineSpanContext, SpanObserver};
use crate::retry::RetryPolicy;
#[cfg(feature = "native")]
use crate::types::VerificationResult;
use crate::types::{Document, Draft, PipelineOutput, Query, SpeculationDecision};

/// Platform-agnostic sleep function.
#[cfg(feature = "native")]
async fn sleep_duration(duration: Duration) {
    tokio::time::sleep(duration).await;
}

/// Platform-agnostic sleep function for WASM.
#[cfg(all(target_arch = "wasm32", not(feature = "native")))]
async fn sleep_duration(duration: Duration) {
    use wasm_bindgen_futures::JsFuture;

    // Neither `expect` here is survivable: under `panic = "abort"` a missing
    // global or a refused `setTimeout` would abort the instance from inside a
    // retry. A sleep that cannot be scheduled resolves immediately instead —
    // the caller retries sooner than it asked to, which is a degraded backoff
    // rather than a dead page.
    let Some(scope) = crate::global_scope::GlobalScope::current() else {
        return;
    };
    let millis = i32::try_from(duration.as_millis()).unwrap_or(i32::MAX);
    let promise = js_sys::Promise::new(&mut |resolve, _| {
        if scope.set_timeout(&resolve, millis).is_err() {
            let _ = resolve.call0(&wasm_bindgen::JsValue::UNDEFINED);
        }
    });
    let _ = JsFuture::from(promise).await;
}

/// Fallback sleep for when neither native nor wasm features are enabled.
#[cfg(all(not(feature = "native"), not(target_arch = "wasm32")))]
async fn sleep_duration(_duration: Duration) {
    std::hint::spin_loop();
}

/// Configuration for the unified three-layer RAG pipeline.
///
/// Controls fast-path thresholds, parallelism, retry behaviour, and how many
/// results the Echo layer retrieves before feeding them into the Speculator and
/// Judge layers.
///
/// # Example
///
/// ```
/// use oxirag::pipeline::PipelineConfig;
///
/// let config = PipelineConfig {
///     fast_path_threshold: 0.98,
///     enable_fast_path: true,
///     max_search_results: 10,
///     ..PipelineConfig::default()
/// };
/// ```
#[derive(Debug, Clone)]
pub struct PipelineConfig {
    /// Threshold for fast-path (skip speculation if Echo confidence is high).
    pub fast_path_threshold: f32,
    /// Threshold for skipping verification.
    pub skip_verification_threshold: f32,
    /// Whether to enable fast-path optimizations.
    pub enable_fast_path: bool,
    /// Maximum retries for failed operations (deprecated: use `retry_config` instead).
    pub max_retries: usize,
    /// Whether to run layers in parallel where possible.
    pub parallel_execution: bool,
    /// Maximum number of search results to use.
    pub max_search_results: usize,
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
            max_search_results: 5,
            retry_config: Some(RetryConfig::default()),
        }
    }
}

/// The unified RAG pipeline trait.
///
/// Implementors drive a query through Echo → Speculator → Judge (and optionally
/// Graph), returning a [`PipelineOutput`] that contains the final answer together
/// with per-layer diagnostics.
#[cfg_attr(not(target_arch = "wasm32"), async_trait)]
#[cfg_attr(target_arch = "wasm32", async_trait(?Send))]
pub trait RagPipeline: Send + Sync {
    /// Run a query through all active pipeline layers and return the result.
    ///
    /// The concrete execution order is Echo (Layer 1) → Speculator (Layer 2) →
    /// Judge (Layer 3). A *fast-path* short-circuit skips Layers 2 and 3 when the
    /// Echo top-1 score exceeds [`PipelineConfig::fast_path_threshold`].
    ///
    /// # Errors
    ///
    /// Returns [`OxiRagError`] if any layer fails and the retry budget is exhausted.
    ///
    /// # Example
    ///
    /// ```rust,ignore
    /// use oxirag::prelude::*;
    ///
    /// let result = pipeline.process(Query::new("What is Rust?")).await?;
    /// println!("Answer: {}", result.final_answer);
    /// println!("Confidence: {:.2}", result.confidence);
    /// ```
    async fn process(&self, query: Query) -> Result<PipelineOutput, OxiRagError>;

    /// Process multiple queries through the pipeline concurrently.
    ///
    /// This method processes all queries in parallel, returning results in the same order
    /// as the input queries. Each query is processed independently.
    ///
    /// # Arguments
    ///
    /// * `queries` - A vector of queries to process.
    ///
    /// # Returns
    ///
    /// A vector of results, one for each input query. Each result is either
    /// a successful `PipelineOutput` or an error.
    async fn process_batch(&self, queries: Vec<Query>) -> Vec<Result<PipelineOutput, OxiRagError>>;

    /// Index a document for retrieval.
    async fn index(&mut self, document: Document) -> Result<(), OxiRagError>;

    /// Index multiple documents.
    async fn index_batch(&mut self, documents: Vec<Document>) -> Result<(), OxiRagError>;

    /// Get pipeline configuration.
    fn config(&self) -> &PipelineConfig;
}

/// The three-layer RAG pipeline.
pub struct Pipeline<E, S, J>
where
    E: Echo,
    S: Speculator,
    J: Judge,
{
    echo: E,
    speculator: S,
    judge: J,
    config: PipelineConfig,
    retry_policy: RetryPolicy,
    /// Observers seeded into every [`PipelineSpanContext`] created during `process`.
    observers: Vec<Arc<dyn SpanObserver>>,
}

impl<E, S, J> Pipeline<E, S, J>
where
    E: Echo,
    S: Speculator,
    J: Judge,
{
    /// Create a new pipeline with the given layers.
    #[must_use]
    pub fn new(echo: E, speculator: S, judge: J, config: PipelineConfig) -> Self {
        let retry_policy = config
            .retry_config
            .as_ref()
            .map_or_else(RetryPolicy::no_retry, |rc| RetryPolicy::new(rc.clone()));

        Self {
            echo,
            speculator,
            judge,
            config,
            retry_policy,
            observers: Vec::new(),
        }
    }

    /// Create a new pipeline seeded with span observers.
    #[must_use]
    pub fn new_with_observers(
        echo: E,
        speculator: S,
        judge: J,
        config: PipelineConfig,
        observers: Vec<Arc<dyn SpanObserver>>,
    ) -> Self {
        let retry_policy = config
            .retry_config
            .as_ref()
            .map_or_else(RetryPolicy::no_retry, |rc| RetryPolicy::new(rc.clone()));

        Self {
            echo,
            speculator,
            judge,
            config,
            retry_policy,
            observers,
        }
    }

    /// Get a reference to the Echo layer.
    #[must_use]
    pub fn echo(&self) -> &E {
        &self.echo
    }

    /// Get a mutable reference to the Echo layer.
    pub fn echo_mut(&mut self) -> &mut E {
        &mut self.echo
    }

    /// Get a reference to the Speculator layer.
    #[must_use]
    pub fn speculator(&self) -> &S {
        &self.speculator
    }

    /// Get a reference to the Judge layer.
    #[must_use]
    pub fn judge(&self) -> &J {
        &self.judge
    }

    /// Get a reference to the retry policy.
    #[must_use]
    pub const fn retry_policy(&self) -> &RetryPolicy {
        &self.retry_policy
    }

    /// Get a mutable reference to the pipeline configuration.
    ///
    /// Everything in [`PipelineConfig`] except `retry_config` takes effect on
    /// the next [`RagPipeline::process`] call, so a caller can turn the fast
    /// path on and off between queries rather than rebuilding the pipeline
    /// around a new config — which would mean rebuilding its layers, and with
    /// them the index they hold.
    ///
    /// `retry_config` is the exception: [`RetryPolicy`] is compiled from it at
    /// construction time, so changing that field here has no effect. Build a new
    /// pipeline to change the retry policy.
    pub const fn config_mut(&mut self) -> &mut PipelineConfig {
        &mut self.config
    }

    /// Generate a draft answer from search results.
    #[allow(clippy::unused_self, clippy::cast_precision_loss)]
    fn generate_draft(&self, query: &Query, context: &[crate::types::SearchResult]) -> Draft {
        if context.is_empty() {
            return Draft::new("No relevant information found.", &query.text).with_confidence(0.0);
        }

        // Simple draft generation: combine top results
        let combined: String = context
            .iter()
            .take(3)
            .map(|r| r.document.content.as_str())
            .collect::<Vec<_>>()
            .join(" ");

        let avg_score: f32 = context.iter().map(|r| r.score).sum::<f32>() / context.len() as f32;

        let mut draft = Draft::new(combined, &query.text).with_confidence(avg_score);

        for result in context {
            draft = draft.with_source(result.document.id.clone());
        }

        draft
    }

    /// Process query sequentially (Speculator then Judge).
    async fn process_sequential(
        &self,
        query: Query,
        search_results: Vec<crate::types::SearchResult>,
        draft: Draft,
        mut layers_used: Vec<String>,
        start: Instant,
    ) -> Result<PipelineOutput, OxiRagError> {
        // Layer 2: Speculator - Draft Verification (with retry)
        layers_used.push("Speculator".to_string());

        let speculation = self
            .retry_policy
            .retry(|| async { self.speculator.verify_draft(&draft, &search_results).await })
            .await
            .map_err(OxiRagError::Speculator)?;

        let (working_draft, speculation_result) = match speculation.decision {
            SpeculationDecision::Accept => (draft.clone(), Some(speculation)),
            SpeculationDecision::Revise => {
                let revised = self
                    .retry_policy
                    .retry(|| async {
                        self.speculator
                            .revise_draft(&draft, &search_results, &speculation)
                            .await
                    })
                    .await
                    .map_err(OxiRagError::Speculator)?;
                (revised, Some(speculation))
            }
            SpeculationDecision::Reject => {
                let rejection_draft = Draft::new(
                    "The retrieved information does not sufficiently answer the query.",
                    &query.text,
                )
                .with_confidence(0.2);
                (rejection_draft, Some(speculation))
            }
        };

        // Check if we can skip verification
        if self.config.enable_fast_path
            && speculation_result
                .as_ref()
                .is_some_and(|s| s.confidence >= self.config.skip_verification_threshold)
        {
            return Ok(PipelineOutput {
                query,
                search_results,
                draft,
                speculation: speculation_result,
                verification: None,
                final_answer: working_draft.content,
                confidence: working_draft.confidence,
                layers_used,
                total_duration_ms: u64::try_from(start.elapsed().as_millis()).unwrap_or(u64::MAX),
            });
        }

        // Layer 3: Judge - Logic Verification (with retry)
        layers_used.push("Judge".to_string());

        let verification = self
            .retry_policy
            .retry(|| async { self.judge.judge(&working_draft, &search_results).await })
            .await
            .map_err(OxiRagError::Judge)?;

        let final_confidence = f32::midpoint(working_draft.confidence, verification.confidence);

        Ok(PipelineOutput {
            query,
            search_results,
            draft,
            speculation: speculation_result,
            verification: Some(verification),
            final_answer: working_draft.content,
            confidence: final_confidence,
            layers_used,
            total_duration_ms: u64::try_from(start.elapsed().as_millis()).unwrap_or(u64::MAX),
        })
    }

    /// Process query with parallel execution (Speculator and Judge run concurrently).
    #[cfg(feature = "native")]
    async fn process_parallel(
        &self,
        query: Query,
        search_results: Vec<crate::types::SearchResult>,
        draft: Draft,
        mut layers_used: Vec<String>,
        start: Instant,
    ) -> Result<PipelineOutput, OxiRagError> {
        layers_used.push("Speculator".to_string());
        layers_used.push("Judge".to_string());

        tracing::debug!("Running Speculator and Judge in parallel");

        // Run Speculator and Judge in parallel on the initial draft
        let (speculation_result, judge_result) =
            run_parallel_verification(&self.speculator, &self.judge, &draft, &search_results).await;

        let speculation = speculation_result.map_err(OxiRagError::Speculator)?;

        match speculation.decision {
            SpeculationDecision::Accept => {
                // Use the parallel judge result directly
                let verification = judge_result.map_err(OxiRagError::Judge)?;
                let final_confidence = f32::midpoint(draft.confidence, verification.confidence);

                Ok(PipelineOutput {
                    query,
                    search_results,
                    draft: draft.clone(),
                    speculation: Some(speculation),
                    verification: Some(verification),
                    final_answer: draft.content,
                    confidence: final_confidence,
                    layers_used,
                    total_duration_ms: u64::try_from(start.elapsed().as_millis())
                        .unwrap_or(u64::MAX),
                })
            }
            SpeculationDecision::Revise => {
                // Need to revise draft and re-run judge
                let revised = self
                    .speculator
                    .revise_draft(&draft, &search_results, &speculation)
                    .await
                    .map_err(OxiRagError::Speculator)?;

                // Re-run judge on revised draft
                let verification = self
                    .judge
                    .judge(&revised, &search_results)
                    .await
                    .map_err(OxiRagError::Judge)?;

                let final_confidence = f32::midpoint(revised.confidence, verification.confidence);

                Ok(PipelineOutput {
                    query,
                    search_results,
                    draft,
                    speculation: Some(speculation),
                    verification: Some(verification),
                    final_answer: revised.content,
                    confidence: final_confidence,
                    layers_used,
                    total_duration_ms: u64::try_from(start.elapsed().as_millis())
                        .unwrap_or(u64::MAX),
                })
            }
            SpeculationDecision::Reject => {
                // Rejected - skip judge verification
                let rejection_draft = Draft::new(
                    "The retrieved information does not sufficiently answer the query.",
                    &query.text,
                )
                .with_confidence(0.2);

                // Remove Judge from layers_used since we're skipping it
                layers_used.retain(|l| l != "Judge");

                Ok(PipelineOutput {
                    query,
                    search_results,
                    draft,
                    speculation: Some(speculation),
                    verification: None,
                    final_answer: rejection_draft.content,
                    confidence: rejection_draft.confidence,
                    layers_used,
                    total_duration_ms: u64::try_from(start.elapsed().as_millis())
                        .unwrap_or(u64::MAX),
                })
            }
        }
    }
}

/// Parallel execution helper for running Speculator and Judge concurrently.
#[cfg(feature = "native")]
async fn run_parallel_verification<S, J>(
    speculator: &S,
    judge: &J,
    draft: &Draft,
    search_results: &[crate::types::SearchResult],
) -> (
    Result<crate::types::SpeculationResult, crate::error::SpeculatorError>,
    Result<VerificationResult, crate::error::JudgeError>,
)
where
    S: Speculator + Send + Sync,
    J: Judge + Send + Sync,
{
    let speculation_future = speculator.verify_draft(draft, search_results);
    let verification_future = judge.judge(draft, search_results);

    tokio::join!(speculation_future, verification_future)
}

#[cfg_attr(not(target_arch = "wasm32"), async_trait)]
#[cfg_attr(target_arch = "wasm32", async_trait(?Send))]
impl<E, S, J> RagPipeline for Pipeline<E, S, J>
where
    E: Echo + Send + Sync,
    S: Speculator + Send + Sync,
    J: Judge + Send + Sync,
{
    async fn process(&self, query: Query) -> Result<PipelineOutput, OxiRagError> {
        let start = Instant::now();
        let layers_used = vec!["Echo".to_string()];

        // Build a span context and seed it with all registered observers.
        let mut span_ctx = PipelineSpanContext::new();
        span_ctx.set_attribute("query", query.text.clone());
        for obs in &self.observers {
            span_ctx.add_observer(obs.clone());
        }

        // Layer 1: Echo - Semantic Search (with retry)
        let search_results = {
            let mut echo_span = span_ctx.begin_layer("echo");
            let result = self
                .retry_policy
                .retry(|| async {
                    self.echo
                        .search(&query.text, self.config.max_search_results, query.min_score)
                        .await
                })
                .await;
            match result {
                Ok(results) => {
                    echo_span.set_item_count(results.len());
                    echo_span.success();
                    results
                }
                Err(e) => {
                    echo_span.error(e.to_string());
                    span_ctx.finalize();
                    return Err(OxiRagError::Embedding(e));
                }
            }
        };

        // Generate draft from search results
        let draft = self.generate_draft(&query, &search_results);

        // Check for fast-path
        let top_score = search_results.first().map_or(0.0, |r| r.score);

        if self.config.enable_fast_path && top_score >= self.config.fast_path_threshold {
            // Fast path: high confidence search, skip speculation and verification
            tracing::debug!(
                "Fast path: top score {} >= threshold {}",
                top_score,
                self.config.fast_path_threshold
            );
            span_ctx.begin_layer("speculator").skip();
            span_ctx.begin_layer("judge").skip();
            span_ctx.finalize();

            return Ok(PipelineOutput {
                query,
                search_results,
                draft: draft.clone(),
                speculation: None,
                verification: None,
                final_answer: draft.content,
                confidence: top_score,
                layers_used,
                total_duration_ms: u64::try_from(start.elapsed().as_millis()).unwrap_or(u64::MAX),
            });
        }

        // Parallel or sequential execution of Speculator and Judge
        #[cfg(feature = "native")]
        if self.config.parallel_execution {
            // For parallel execution we finalize the context after the call returns.
            let output = self
                .process_parallel(query, search_results, draft, layers_used, start)
                .await;
            span_ctx.begin_layer("speculator").success();
            span_ctx.begin_layer("judge").success();
            span_ctx.finalize();
            return output;
        }

        // Sequential execution (default)
        let output = self
            .process_sequential(query, search_results, draft, layers_used, start)
            .await;
        span_ctx.begin_layer("speculator").success();
        span_ctx.begin_layer("judge").success();
        span_ctx.finalize();
        output
    }

    async fn index(&mut self, document: Document) -> Result<(), OxiRagError> {
        // Manual retry loop for mutable borrow context
        let mut last_error = None;
        let max_retries = self.retry_policy.config().max_retries;

        for attempt in 0..=max_retries {
            match self.echo.index(document.clone()).await {
                Ok(id) => {
                    let _ = id;
                    return Ok(());
                }
                Err(e) => {
                    if !crate::retry::Retryable::is_retryable(&e) || attempt == max_retries {
                        return Err(OxiRagError::Embedding(e));
                    }
                    tracing::warn!(
                        attempt = attempt + 1,
                        max_retries = max_retries,
                        "Index operation failed, retrying"
                    );
                    last_error = Some(e);
                    let delay = self.retry_policy.calculate_delay(attempt);
                    sleep_duration(delay).await;
                }
            }
        }

        Err(OxiRagError::Embedding(
            last_error.expect("retry loop should have returned"),
        ))
    }

    async fn index_batch(&mut self, documents: Vec<Document>) -> Result<(), OxiRagError> {
        // Manual retry loop for mutable borrow context
        let mut last_error = None;
        let max_retries = self.retry_policy.config().max_retries;

        for attempt in 0..=max_retries {
            match self.echo.index_batch(documents.clone()).await {
                Ok(_ids) => return Ok(()),
                Err(e) => {
                    if !crate::retry::Retryable::is_retryable(&e) || attempt == max_retries {
                        return Err(OxiRagError::Embedding(e));
                    }
                    tracing::warn!(
                        attempt = attempt + 1,
                        max_retries = max_retries,
                        "Index batch operation failed, retrying"
                    );
                    last_error = Some(e);
                    let delay = self.retry_policy.calculate_delay(attempt);
                    sleep_duration(delay).await;
                }
            }
        }

        Err(OxiRagError::Embedding(
            last_error.expect("retry loop should have returned"),
        ))
    }

    async fn process_batch(&self, queries: Vec<Query>) -> Vec<Result<PipelineOutput, OxiRagError>> {
        if queries.is_empty() {
            return Vec::new();
        }

        #[cfg(feature = "native")]
        {
            use futures::future::join_all;

            let futures: Vec<_> = queries.into_iter().map(|q| self.process(q)).collect();

            join_all(futures).await
        }

        #[cfg(all(target_arch = "wasm32", not(feature = "native")))]
        {
            // WASM: process sequentially as join_all may not work reliably
            let mut results = Vec::with_capacity(queries.len());
            for query in queries {
                results.push(self.process(query).await);
            }
            results
        }

        #[cfg(all(not(feature = "native"), not(target_arch = "wasm32")))]
        {
            // Fallback: sequential processing
            let mut results = Vec::with_capacity(queries.len());
            for query in queries {
                results.push(self.process(query).await);
            }
            results
        }
    }

    fn config(&self) -> &PipelineConfig {
        &self.config
    }
}

/// Fluent builder for constructing a type-safe [`Pipeline`].
///
/// Each of the three layers (Echo, Speculator, Judge) must be provided before
/// calling [`build`]; omitting any layer causes [`build`] to return an error.
///
/// # Example
///
/// ```rust,ignore
/// use oxirag::prelude::*;
///
/// let pipeline = PipelineBuilder::new()
///     .with_echo(EchoLayer::new(MockEmbeddingProvider::new(384), InMemoryVectorStore::new(384)))
///     .with_speculator(RuleBasedSpeculator::default())
///     .with_judge(JudgeImpl::new(AdvancedClaimExtractor::new(), MockSmtVerifier::default(), JudgeConfig::default()))
///     .with_config(PipelineConfig { enable_fast_path: false, ..Default::default() })
///     .build()
///     .expect("all layers configured");
/// ```
///
/// [`build`]: PipelineBuilder::build
pub struct PipelineBuilder<E, S, J> {
    echo: Option<E>,
    speculator: Option<S>,
    judge: Option<J>,
    config: PipelineConfig,
    observers: Vec<Arc<dyn SpanObserver>>,
}

impl<E, S, J> Default for PipelineBuilder<E, S, J> {
    fn default() -> Self {
        Self {
            echo: None,
            speculator: None,
            judge: None,
            config: PipelineConfig::default(),
            observers: Vec::new(),
        }
    }
}

impl<E, S, J> PipelineBuilder<E, S, J>
where
    E: Echo,
    S: Speculator,
    J: Judge,
{
    /// Create a new pipeline builder.
    #[must_use]
    #[allow(clippy::should_implement_trait)]
    pub fn new() -> Self {
        Self::default()
    }

    /// Set the Echo layer.
    #[must_use]
    pub fn with_echo(mut self, echo: E) -> Self {
        self.echo = Some(echo);
        self
    }

    /// Set the Speculator layer.
    #[must_use]
    pub fn with_speculator(mut self, speculator: S) -> Self {
        self.speculator = Some(speculator);
        self
    }

    /// Set the Judge layer.
    #[must_use]
    pub fn with_judge(mut self, judge: J) -> Self {
        self.judge = Some(judge);
        self
    }

    /// Set the pipeline configuration.
    #[must_use]
    pub fn with_config(mut self, config: PipelineConfig) -> Self {
        self.config = config;
        self
    }

    /// Register span observers that will be notified for every pipeline execution.
    ///
    /// Multiple calls append observers; they do not replace earlier ones.
    #[must_use]
    pub fn with_observers(mut self, observers: Vec<Arc<dyn SpanObserver>>) -> Self {
        self.observers.extend(observers);
        self
    }

    /// Build the pipeline.
    ///
    /// # Errors
    ///
    /// Returns an error if any required layer is not configured.
    pub fn build(self) -> Result<Pipeline<E, S, J>, PipelineError> {
        let echo = self
            .echo
            .ok_or_else(|| PipelineError::BuildError("Echo layer not configured".to_string()))?;
        let speculator = self.speculator.ok_or_else(|| {
            PipelineError::BuildError("Speculator layer not configured".to_string())
        })?;
        let judge = self
            .judge
            .ok_or_else(|| PipelineError::BuildError("Judge layer not configured".to_string()))?;

        Ok(Pipeline::new_with_observers(
            echo,
            speculator,
            judge,
            self.config,
            self.observers,
        ))
    }
}

#[cfg(all(test, not(target_arch = "wasm32")))]
mod tests {
    use super::*;
    use crate::layer1_echo::{EchoLayer, InMemoryVectorStore, MockEmbeddingProvider};
    use crate::layer2_speculator::RuleBasedSpeculator;
    use crate::layer3_judge::{AdvancedClaimExtractor, JudgeConfig, JudgeImpl, MockSmtVerifier};

    fn create_test_pipeline() -> Pipeline<
        EchoLayer<MockEmbeddingProvider, InMemoryVectorStore>,
        RuleBasedSpeculator,
        JudgeImpl<AdvancedClaimExtractor, MockSmtVerifier>,
    > {
        let echo = EchoLayer::new(MockEmbeddingProvider::new(64), InMemoryVectorStore::new(64));

        let speculator = RuleBasedSpeculator::default();

        let judge = JudgeImpl::new(
            AdvancedClaimExtractor::new(),
            MockSmtVerifier::default(),
            JudgeConfig::default(),
        );

        Pipeline::new(echo, speculator, judge, PipelineConfig::default())
    }

    #[tokio::test]
    async fn test_pipeline_empty_index() {
        let pipeline = create_test_pipeline();
        let query = Query::new("What is the meaning of life?");

        let result = pipeline
            .process(query)
            .await
            .expect("test operation should succeed");

        assert!(result.search_results.is_empty());
        assert!(result.final_answer.contains("No relevant information"));
    }

    #[tokio::test]
    async fn test_pipeline_with_documents() {
        let mut pipeline = create_test_pipeline();

        // Index some documents
        pipeline
            .index(Document::new("The capital of France is Paris."))
            .await
            .expect("test operation should succeed");
        pipeline
            .index(Document::new("Paris is known for the Eiffel Tower."))
            .await
            .expect("test operation should succeed");

        let query = Query::new("What is the capital of France?").with_top_k(3);

        let result = pipeline
            .process(query)
            .await
            .expect("test operation should succeed");

        assert!(!result.search_results.is_empty());
        assert!(result.layers_used.contains(&"Echo".to_string()));
    }

    #[tokio::test]
    async fn test_pipeline_fast_path() {
        let mut pipeline = Pipeline::new(
            EchoLayer::new(MockEmbeddingProvider::new(64), InMemoryVectorStore::new(64)),
            RuleBasedSpeculator::default(),
            JudgeImpl::new(
                AdvancedClaimExtractor::new(),
                MockSmtVerifier::default(),
                JudgeConfig::default(),
            ),
            PipelineConfig {
                fast_path_threshold: -1.0, // Threshold below cosine similarity minimum, always take fast path
                enable_fast_path: true,
                ..Default::default()
            },
        );

        pipeline
            .index(Document::new("Test document content."))
            .await
            .expect("test operation should succeed");

        let query = Query::new("test");
        let result = pipeline
            .process(query)
            .await
            .expect("test operation should succeed");

        // Should not have speculation or verification due to fast path
        assert!(result.speculation.is_none());
        assert!(result.verification.is_none());
    }

    #[tokio::test]
    async fn test_pipeline_builder() {
        let echo = EchoLayer::new(MockEmbeddingProvider::new(32), InMemoryVectorStore::new(32));
        let speculator = RuleBasedSpeculator::default();
        let judge = JudgeImpl::new(
            AdvancedClaimExtractor::new(),
            MockSmtVerifier::default(),
            JudgeConfig::default(),
        );

        let pipeline = PipelineBuilder::new()
            .with_echo(echo)
            .with_speculator(speculator)
            .with_judge(judge)
            .with_config(PipelineConfig {
                enable_fast_path: false,
                ..Default::default()
            })
            .build()
            .expect("test operation should succeed");

        assert!(!pipeline.config().enable_fast_path);
    }

    #[tokio::test]
    async fn test_pipeline_builder_missing_layer() {
        let echo = EchoLayer::new(MockEmbeddingProvider::new(32), InMemoryVectorStore::new(32));

        let result = PipelineBuilder::<
            _,
            RuleBasedSpeculator,
            JudgeImpl<AdvancedClaimExtractor, MockSmtVerifier>,
        >::new()
        .with_echo(echo)
        .build();

        assert!(result.is_err());
    }

    #[tokio::test]
    async fn test_pipeline_batch_index() {
        let mut pipeline = create_test_pipeline();

        let docs = vec![
            Document::new("First document"),
            Document::new("Second document"),
            Document::new("Third document"),
        ];

        pipeline
            .index_batch(docs)
            .await
            .expect("test operation should succeed");

        let query = Query::new("document").with_top_k(5);
        let result = pipeline
            .process(query)
            .await
            .expect("test operation should succeed");

        assert_eq!(result.search_results.len(), 3);
    }

    #[tokio::test]
    async fn test_pipeline_layers_used() {
        let mut pipeline = Pipeline::new(
            EchoLayer::new(MockEmbeddingProvider::new(64), InMemoryVectorStore::new(64)),
            RuleBasedSpeculator::default(),
            JudgeImpl::new(
                AdvancedClaimExtractor::new(),
                MockSmtVerifier::default(),
                JudgeConfig::default(),
            ),
            PipelineConfig {
                enable_fast_path: false, // Disable fast path to use all layers
                ..Default::default()
            },
        );

        pipeline
            .index(Document::new("Some test content here."))
            .await
            .expect("test operation should succeed");

        let query = Query::new("test");
        let result = pipeline
            .process(query)
            .await
            .expect("test operation should succeed");

        assert!(result.layers_used.contains(&"Echo".to_string()));
        assert!(result.layers_used.contains(&"Speculator".to_string()));
        assert!(result.layers_used.contains(&"Judge".to_string()));
    }

    #[test]
    fn test_pipeline_config_default_has_retry() {
        let config = PipelineConfig::default();
        assert!(config.retry_config.is_some());
        let retry = config.retry_config.expect("test operation should succeed");
        assert_eq!(retry.max_retries, 3);
        assert_eq!(retry.initial_delay_ms, 100);
        assert_eq!(retry.max_delay_ms, 5000);
        assert!((retry.backoff_multiplier - 2.0).abs() < f64::EPSILON);
        assert!(retry.add_jitter);
    }

    #[test]
    fn test_pipeline_with_custom_retry_config() {
        let config = PipelineConfig {
            retry_config: Some(RetryConfig {
                max_retries: 5,
                initial_delay_ms: 50,
                max_delay_ms: 2000,
                backoff_multiplier: 1.5,
                add_jitter: false,
            }),
            ..Default::default()
        };

        let echo = EchoLayer::new(MockEmbeddingProvider::new(32), InMemoryVectorStore::new(32));
        let speculator = RuleBasedSpeculator::default();
        let judge = JudgeImpl::new(
            AdvancedClaimExtractor::new(),
            MockSmtVerifier::default(),
            JudgeConfig::default(),
        );

        let pipeline = Pipeline::new(echo, speculator, judge, config);
        let retry_config = pipeline.retry_policy().config();

        assert_eq!(retry_config.max_retries, 5);
        assert_eq!(retry_config.initial_delay_ms, 50);
        assert!(!retry_config.add_jitter);
    }

    #[test]
    fn test_pipeline_without_retry() {
        let config = PipelineConfig {
            retry_config: None,
            ..Default::default()
        };

        let echo = EchoLayer::new(MockEmbeddingProvider::new(32), InMemoryVectorStore::new(32));
        let speculator = RuleBasedSpeculator::default();
        let judge = JudgeImpl::new(
            AdvancedClaimExtractor::new(),
            MockSmtVerifier::default(),
            JudgeConfig::default(),
        );

        let pipeline = Pipeline::new(echo, speculator, judge, config);
        // No retry means max_retries is 0
        assert_eq!(pipeline.retry_policy().config().max_retries, 0);
    }

    #[tokio::test]
    async fn test_pipeline_retry_policy_accessor() {
        let pipeline = create_test_pipeline();
        let policy = pipeline.retry_policy();
        assert_eq!(policy.config().max_retries, 3);
    }

    #[tokio::test]
    async fn test_pipeline_parallel_execution() {
        let mut pipeline = Pipeline::new(
            EchoLayer::new(MockEmbeddingProvider::new(64), InMemoryVectorStore::new(64)),
            RuleBasedSpeculator::default(),
            JudgeImpl::new(
                AdvancedClaimExtractor::new(),
                MockSmtVerifier::default(),
                JudgeConfig::default(),
            ),
            PipelineConfig {
                parallel_execution: true,
                enable_fast_path: false,
                ..Default::default()
            },
        );

        pipeline
            .index(Document::new("Parallel execution test document."))
            .await
            .expect("test operation should succeed");

        let query = Query::new("parallel test");
        let result = pipeline
            .process(query)
            .await
            .expect("test operation should succeed");

        // Both Speculator and Judge should be used
        assert!(result.layers_used.contains(&"Echo".to_string()));
        assert!(result.layers_used.contains(&"Speculator".to_string()));
        assert!(result.layers_used.contains(&"Judge".to_string()));
    }

    #[test]
    fn test_pipeline_config_parallel_execution_default() {
        let config = PipelineConfig::default();
        assert!(!config.parallel_execution);
    }

    #[tokio::test]
    async fn test_pipeline_process_batch_empty() {
        let pipeline = create_test_pipeline();
        let results = pipeline.process_batch(vec![]).await;
        assert!(results.is_empty());
    }

    #[tokio::test]
    async fn test_pipeline_process_batch_single() {
        let mut pipeline = create_test_pipeline();

        pipeline
            .index(Document::new("Rust is a systems programming language."))
            .await
            .expect("test operation should succeed");

        let queries = vec![Query::new("What is Rust?")];
        let results = pipeline.process_batch(queries).await;

        assert_eq!(results.len(), 1);
        assert!(results[0].is_ok());
    }

    #[tokio::test]
    async fn test_pipeline_process_batch_multiple() {
        let mut pipeline = create_test_pipeline();

        pipeline
            .index(Document::new("The capital of France is Paris."))
            .await
            .expect("test operation should succeed");
        pipeline
            .index(Document::new("Python is a programming language."))
            .await
            .expect("test operation should succeed");
        pipeline
            .index(Document::new("The Eiffel Tower is in Paris."))
            .await
            .expect("test operation should succeed");

        let queries = vec![
            Query::new("What is the capital of France?"),
            Query::new("What is Python?"),
            Query::new("Where is the Eiffel Tower?"),
        ];

        let results = pipeline.process_batch(queries).await;

        assert_eq!(results.len(), 3);
        for result in &results {
            assert!(result.is_ok());
        }
    }

    #[tokio::test]
    async fn test_pipeline_process_batch_preserves_order() {
        let mut pipeline = create_test_pipeline();

        pipeline
            .index(Document::new("Alpha document content."))
            .await
            .expect("test operation should succeed");
        pipeline
            .index(Document::new("Beta document content."))
            .await
            .expect("test operation should succeed");

        let queries = vec![Query::new("Alpha"), Query::new("Beta"), Query::new("Alpha")];

        let results = pipeline.process_batch(queries).await;

        assert_eq!(results.len(), 3);
        // All results should be successful
        assert!(results[0].is_ok());
        assert!(results[1].is_ok());
        assert!(results[2].is_ok());
    }
}
