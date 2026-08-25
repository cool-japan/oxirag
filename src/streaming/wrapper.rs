//! `StreamingPipeline` trait, `StreamingPipelineWrapper`, and `StreamingPipelineResult`.

#[cfg(feature = "native")]
use crate::time::Instant;
use async_trait::async_trait;

use crate::error::OxiRagError;
use crate::pipeline::RagPipeline;
use crate::types::{PipelineOutput, Query};

#[cfg(feature = "native")]
use super::types::{ChunkType, PipelineChunk, truncate_content};

#[cfg(feature = "native")]
use std::pin::Pin;

#[cfg(feature = "native")]
use futures::Stream;

#[cfg(feature = "native")]
use tokio::sync::mpsc;

/// Streaming pipeline result that allows consuming chunks as they arrive.
#[cfg(feature = "native")]
pub struct StreamingPipelineResult {
    pub(super) receiver: mpsc::Receiver<PipelineChunk>,
    pub(super) final_output: Option<PipelineOutput>,
    pub(super) collected_chunks: Vec<PipelineChunk>,
}

#[cfg(feature = "native")]
impl StreamingPipelineResult {
    /// Create a new streaming pipeline result with the given receiver.
    #[must_use]
    pub fn new(receiver: mpsc::Receiver<PipelineChunk>) -> Self {
        Self {
            receiver,
            final_output: None,
            collected_chunks: Vec::new(),
        }
    }

    /// Create a streaming pipeline result with a pre-computed output.
    #[must_use]
    pub fn from_output(output: PipelineOutput) -> Self {
        let (_, receiver) = mpsc::channel(1);
        Self {
            receiver,
            final_output: Some(output),
            collected_chunks: Vec::new(),
        }
    }

    /// Get the next chunk from the stream.
    ///
    /// Returns `None` when the stream is exhausted.
    pub async fn next(&mut self) -> Option<PipelineChunk> {
        let chunk = self.receiver.recv().await;
        if let Some(ref c) = chunk {
            self.collected_chunks.push(c.clone());
        }
        chunk
    }

    /// Convert this streaming result into a futures Stream.
    #[must_use]
    pub fn into_stream(self) -> Pin<Box<dyn Stream<Item = PipelineChunk> + Send>> {
        Box::pin(tokio_stream::wrappers::ReceiverStream::new(self.receiver))
    }

    /// Collect all remaining chunks and return the final output.
    ///
    /// # Errors
    ///
    /// Returns an error if the stream encountered an error during processing.
    pub async fn collect(mut self) -> Result<PipelineOutput, OxiRagError> {
        if let Some(output) = self.final_output {
            return Ok(output);
        }

        // Collect all remaining chunks
        let mut last_error: Option<String> = None;
        while let Some(chunk) = self.receiver.recv().await {
            if let ChunkType::Error(ref err) = chunk.chunk_type {
                last_error = Some(err.clone());
            }
            self.collected_chunks.push(chunk);
        }

        if let Some(err) = last_error {
            return Err(OxiRagError::Pipeline(
                crate::error::PipelineError::ExecutionError(err),
            ));
        }

        // Build a minimal output from collected chunks
        // In practice, the sender should have sent a final result
        Err(OxiRagError::Pipeline(
            crate::error::PipelineError::ExecutionError(
                "Stream ended without final output".to_string(),
            ),
        ))
    }

    /// Process chunks as they arrive with a callback.
    pub async fn for_each<F>(mut self, mut callback: F)
    where
        F: FnMut(PipelineChunk),
    {
        while let Some(chunk) = self.next().await {
            callback(chunk);
        }
    }

    /// Check if the stream has a pre-computed final output.
    #[must_use]
    pub fn has_final_output(&self) -> bool {
        self.final_output.is_some()
    }

    /// Get a reference to the collected chunks so far.
    #[must_use]
    pub fn collected_chunks(&self) -> &[PipelineChunk] {
        &self.collected_chunks
    }

    /// Get the number of collected chunks.
    #[must_use]
    pub fn chunk_count(&self) -> usize {
        self.collected_chunks.len()
    }
}

/// Non-native (WASM) placeholder for streaming pipeline result.
#[cfg(not(feature = "native"))]
pub struct StreamingPipelineResult {
    output: Option<PipelineOutput>,
    error: Option<OxiRagError>,
}

#[cfg(not(feature = "native"))]
impl StreamingPipelineResult {
    /// Create a streaming pipeline result from an output (WASM doesn't support true streaming).
    #[must_use]
    pub fn from_output(output: PipelineOutput) -> Self {
        Self {
            output: Some(output),
            error: None,
        }
    }

    /// Create a streaming pipeline result from an error.
    #[must_use]
    pub fn from_error(error: OxiRagError) -> Self {
        Self {
            output: None,
            error: Some(error),
        }
    }

    /// Collect the result (immediately available in WASM).
    ///
    /// # Errors
    ///
    /// The error the pipeline produced, or [`PipelineError::ExecutionError`] if
    /// this result carries neither an output nor an error.
    ///
    /// `async` with nothing to await is deliberate: this is the `wasm32` half of
    /// an API whose native half genuinely awaits a channel, and callers are
    /// written once against both.
    #[allow(clippy::unused_async)]
    pub async fn collect(self) -> Result<PipelineOutput, OxiRagError> {
        match (self.output, self.error) {
            (Some(output), _) => Ok(output),
            (None, Some(err)) => Err(err),
            (None, None) => Err(OxiRagError::Pipeline(
                crate::error::PipelineError::ExecutionError("No output available".to_string()),
            )),
        }
    }
}

/// Trait extension for streaming pipeline execution.
#[cfg_attr(not(target_arch = "wasm32"), async_trait)]
#[cfg_attr(target_arch = "wasm32", async_trait(?Send))]
pub trait StreamingPipeline: RagPipeline + Send + Sync {
    /// Process a query with streaming output.
    ///
    /// # Arguments
    /// * `query` - The query to process
    ///
    /// # Returns
    /// A streaming result that can be consumed chunk by chunk.
    ///
    /// # Errors
    ///
    /// Returns an error if the pipeline fails to start processing.
    async fn process_streaming(&self, query: Query)
    -> Result<StreamingPipelineResult, OxiRagError>;

    /// Process multiple queries with streaming output.
    ///
    /// # Arguments
    /// * `queries` - The queries to process
    ///
    /// # Returns
    /// A vector of streaming results, one for each query.
    async fn process_batch_streaming(&self, queries: Vec<Query>) -> Vec<StreamingPipelineResult>;
}

/// A wrapper that adds streaming capabilities to any pipeline.
pub struct StreamingPipelineWrapper<P: RagPipeline> {
    pub(super) inner: P,
    pub(super) chunk_buffer_size: usize,
}

impl<P: RagPipeline> StreamingPipelineWrapper<P> {
    /// Create a new streaming wrapper around an existing pipeline.
    #[must_use]
    pub fn new(pipeline: P) -> Self {
        Self {
            inner: pipeline,
            chunk_buffer_size: 32,
        }
    }

    /// Set the buffer size for chunks.
    #[must_use]
    pub fn with_buffer_size(mut self, size: usize) -> Self {
        self.chunk_buffer_size = size.max(1);
        self
    }

    /// Get a reference to the inner pipeline.
    #[must_use]
    pub fn inner(&self) -> &P {
        &self.inner
    }

    /// Get a mutable reference to the inner pipeline.
    #[must_use]
    pub fn inner_mut(&mut self) -> &mut P {
        &mut self.inner
    }

    /// Get the configured buffer size.
    #[must_use]
    pub fn buffer_size(&self) -> usize {
        self.chunk_buffer_size
    }
}

#[cfg_attr(not(target_arch = "wasm32"), async_trait)]
#[cfg_attr(target_arch = "wasm32", async_trait(?Send))]
impl<P: RagPipeline + Send + Sync> RagPipeline for StreamingPipelineWrapper<P> {
    async fn process(&self, query: Query) -> Result<PipelineOutput, OxiRagError> {
        self.inner.process(query).await
    }

    async fn process_batch(&self, queries: Vec<Query>) -> Vec<Result<PipelineOutput, OxiRagError>> {
        self.inner.process_batch(queries).await
    }

    async fn index(&mut self, document: crate::types::Document) -> Result<(), OxiRagError> {
        self.inner.index(document).await
    }

    async fn index_batch(
        &mut self,
        documents: Vec<crate::types::Document>,
    ) -> Result<(), OxiRagError> {
        self.inner.index_batch(documents).await
    }

    fn config(&self) -> &crate::pipeline::PipelineConfig {
        self.inner.config()
    }
}

/// Helper to get timestamp in milliseconds, saturating to `u64::MAX` for very long durations.
#[cfg(feature = "native")]
#[inline]
fn elapsed_ms(start: &Instant) -> u64 {
    // Saturating conversion for very long durations (> 584 million years)
    #[allow(clippy::cast_possible_truncation)]
    {
        start.elapsed().as_millis().min(u128::from(u64::MAX)) as u64
    }
}

/// Emits chunks for search results.
#[cfg(feature = "native")]
async fn emit_search_chunks(
    tx: &mpsc::Sender<PipelineChunk>,
    output: &PipelineOutput,
    chunk_id: &mut usize,
    start: &Instant,
) {
    // Emit search results
    for (rank, search_result) in output.search_results.iter().enumerate() {
        let chunk = PipelineChunk::new(
            *chunk_id,
            ChunkType::SearchResult {
                rank,
                score: search_result.score,
            },
            truncate_content(&search_result.document.content, 100),
        )
        .with_layer("Echo")
        .with_confidence(search_result.score)
        .with_timestamp(elapsed_ms(start));
        let _ = tx.send(chunk).await;
        *chunk_id += 1;
    }

    // Emit search completed
    let chunk = PipelineChunk::new(
        *chunk_id,
        ChunkType::SearchCompleted {
            total: output.search_results.len(),
        },
        format!("Found {} results", output.search_results.len()),
    )
    .with_layer("Echo")
    .with_timestamp(elapsed_ms(start));
    let _ = tx.send(chunk).await;
    *chunk_id += 1;
}

/// Emits chunks for speculation results.
#[cfg(feature = "native")]
async fn emit_speculation_chunks(
    tx: &mpsc::Sender<PipelineChunk>,
    speculation: &crate::types::SpeculationResult,
    chunk_id: &mut usize,
    start: &Instant,
) {
    let chunk = PipelineChunk::new(
        *chunk_id,
        ChunkType::SpeculationStarted,
        "Starting speculation verification",
    )
    .with_layer("Speculator")
    .with_timestamp(elapsed_ms(start));
    let _ = tx.send(chunk).await;
    *chunk_id += 1;

    // Emit speculation progress
    let chunk = PipelineChunk::new(
        *chunk_id,
        ChunkType::SpeculationProgress {
            stage: "verification".to_string(),
            confidence: speculation.confidence,
        },
        &speculation.explanation,
    )
    .with_layer("Speculator")
    .with_confidence(speculation.confidence)
    .with_timestamp(elapsed_ms(start));
    let _ = tx.send(chunk).await;
    *chunk_id += 1;

    // Emit speculation decision
    let chunk = PipelineChunk::new(
        *chunk_id,
        ChunkType::SpeculationDecision(speculation.decision.clone()),
        format!("Decision: {:?}", speculation.decision),
    )
    .with_layer("Speculator")
    .with_confidence(speculation.confidence)
    .with_timestamp(elapsed_ms(start));
    let _ = tx.send(chunk).await;
    *chunk_id += 1;
}

/// Emits chunks for verification results.
#[cfg(feature = "native")]
async fn emit_verification_chunks(
    tx: &mpsc::Sender<PipelineChunk>,
    verification: &crate::types::VerificationResult,
    chunk_id: &mut usize,
    start: &Instant,
) {
    let chunk = PipelineChunk::new(
        *chunk_id,
        ChunkType::VerificationStarted,
        "Starting logic verification",
    )
    .with_layer("Judge")
    .with_timestamp(elapsed_ms(start));
    let _ = tx.send(chunk).await;
    *chunk_id += 1;

    // Emit claim extractions and verifications
    for (i, claim_result) in verification.claim_results.iter().enumerate() {
        let chunk = PipelineChunk::new(
            *chunk_id,
            ChunkType::ClaimExtracted { claim_id: i },
            &claim_result.claim.text,
        )
        .with_layer("Judge")
        .with_confidence(claim_result.claim.confidence)
        .with_timestamp(elapsed_ms(start));
        let _ = tx.send(chunk).await;
        *chunk_id += 1;

        let chunk = PipelineChunk::new(
            *chunk_id,
            ChunkType::ClaimVerified {
                claim_id: i,
                status: format!("{:?}", claim_result.status),
            },
            claim_result
                .explanation
                .as_deref()
                .unwrap_or("No explanation"),
        )
        .with_layer("Judge")
        .with_duration(claim_result.duration_ms)
        .with_timestamp(elapsed_ms(start));
        let _ = tx.send(chunk).await;
        *chunk_id += 1;
    }

    // Emit verification completed
    let chunk = PipelineChunk::new(
        *chunk_id,
        ChunkType::VerificationCompleted,
        &verification.summary,
    )
    .with_layer("Judge")
    .with_confidence(verification.confidence)
    .with_duration(verification.total_duration_ms)
    .with_timestamp(elapsed_ms(start));
    let _ = tx.send(chunk).await;
    *chunk_id += 1;
}

#[cfg(feature = "native")]
#[cfg_attr(not(target_arch = "wasm32"), async_trait)]
#[cfg_attr(target_arch = "wasm32", async_trait(?Send))]
impl<P: RagPipeline + Send + Sync> StreamingPipeline for StreamingPipelineWrapper<P> {
    async fn process_streaming(
        &self,
        query: Query,
    ) -> Result<StreamingPipelineResult, OxiRagError> {
        let (tx, rx) = mpsc::channel(self.chunk_buffer_size);
        let query_text = query.text.clone();

        // Process the query and emit chunks
        let result = self.inner.process(query.clone()).await;

        // Spawn a task to emit chunks based on the result
        let start = Instant::now();

        tokio::spawn(async move {
            let mut chunk_id = 0;

            // Emit search started
            let chunk = PipelineChunk::new(
                chunk_id,
                ChunkType::SearchStarted,
                format!("Searching for: {query_text}"),
            )
            .with_layer("Echo")
            .with_timestamp(elapsed_ms(&start));
            let _ = tx.send(chunk).await;
            chunk_id += 1;

            match result {
                Ok(output) => {
                    emit_search_chunks(&tx, &output, &mut chunk_id, &start).await;

                    // Emit draft generated
                    let chunk = PipelineChunk::new(
                        chunk_id,
                        ChunkType::DraftGenerated,
                        truncate_content(&output.draft.content, 200),
                    )
                    .with_confidence(output.draft.confidence)
                    .with_timestamp(elapsed_ms(&start));
                    let _ = tx.send(chunk).await;
                    chunk_id += 1;

                    // Emit speculation if present
                    if let Some(ref speculation) = output.speculation {
                        emit_speculation_chunks(&tx, speculation, &mut chunk_id, &start).await;
                    }

                    // Emit verification if present
                    if let Some(ref verification) = output.verification {
                        emit_verification_chunks(&tx, verification, &mut chunk_id, &start).await;
                    }

                    // Emit final answer
                    let chunk =
                        PipelineChunk::new(chunk_id, ChunkType::FinalAnswer, &output.final_answer)
                            .with_confidence(output.confidence)
                            .with_duration(output.total_duration_ms)
                            .with_timestamp(elapsed_ms(&start));
                    let _ = tx.send(chunk).await;
                }
                Err(err) => {
                    let chunk = PipelineChunk::new(
                        chunk_id,
                        ChunkType::Error(err.to_string()),
                        format!("Pipeline error: {err}"),
                    )
                    .with_timestamp(elapsed_ms(&start));
                    let _ = tx.send(chunk).await;
                }
            }
        });

        Ok(StreamingPipelineResult::new(rx))
    }

    async fn process_batch_streaming(&self, queries: Vec<Query>) -> Vec<StreamingPipelineResult> {
        use futures::future::join_all;

        let futures: Vec<_> = queries
            .into_iter()
            .map(|q| async move {
                match self.process_streaming(q).await {
                    Ok(result) => result,
                    Err(err) => {
                        // Create a result that will emit an error chunk
                        let (tx, rx) = mpsc::channel(1);
                        let chunk = PipelineChunk::new(
                            0,
                            ChunkType::Error(err.to_string()),
                            format!("Failed to start streaming: {err}"),
                        );
                        let _ = tx.send(chunk).await;
                        StreamingPipelineResult::new(rx)
                    }
                }
            })
            .collect();

        join_all(futures).await
    }
}

#[cfg(not(feature = "native"))]
#[cfg_attr(not(target_arch = "wasm32"), async_trait)]
#[cfg_attr(target_arch = "wasm32", async_trait(?Send))]
impl<P: RagPipeline + Send + Sync> StreamingPipeline for StreamingPipelineWrapper<P> {
    async fn process_streaming(
        &self,
        query: Query,
    ) -> Result<StreamingPipelineResult, OxiRagError> {
        let result = self.inner.process(query).await;
        match result {
            Ok(output) => Ok(StreamingPipelineResult::from_output(output)),
            Err(err) => Ok(StreamingPipelineResult::from_error(err)),
        }
    }

    async fn process_batch_streaming(&self, queries: Vec<Query>) -> Vec<StreamingPipelineResult> {
        let mut results = Vec::with_capacity(queries.len());
        for query in queries {
            match self.process_streaming(query).await {
                Ok(result) => results.push(result),
                Err(err) => results.push(StreamingPipelineResult::from_error(err)),
            }
        }
        results
    }
}
