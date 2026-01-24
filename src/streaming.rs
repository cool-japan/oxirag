//! Async streaming API for the RAG pipeline.
//!
//! This module provides real-time streaming of pipeline results, allowing consumers
//! to receive partial results as they become available during processing.
//!
//! # Example
//!
//! ```rust,ignore
//! use oxirag::prelude::*;
//! use oxirag::streaming::{StreamingPipeline, StreamingPipelineWrapper};
//!
//! // Wrap an existing pipeline for streaming
//! let streaming = StreamingPipelineWrapper::new(pipeline);
//!
//! // Process with streaming output
//! let mut result = streaming.process_streaming(query).await?;
//!
//! // Consume chunks as they arrive
//! while let Some(chunk) = result.next().await {
//!     println!("Chunk {}: {:?}", chunk.chunk_id, chunk.chunk_type);
//! }
//! ```

use async_trait::async_trait;
use serde::{Deserialize, Serialize};
use std::time::Instant;

use crate::error::OxiRagError;
use crate::pipeline::RagPipeline;
use crate::types::{PipelineOutput, Query, SpeculationDecision};

#[cfg(feature = "native")]
use std::pin::Pin;

#[cfg(feature = "native")]
use futures::Stream;

#[cfg(feature = "native")]
use tokio::sync::mpsc;

/// A chunk of streaming pipeline output.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct PipelineChunk {
    /// Unique identifier for this chunk within the stream.
    pub chunk_id: usize,
    /// The type of this chunk.
    pub chunk_type: ChunkType,
    /// The content of this chunk.
    pub content: String,
    /// Metadata about this chunk.
    pub metadata: ChunkMetadata,
}

impl PipelineChunk {
    /// Create a new pipeline chunk.
    #[must_use]
    pub fn new(chunk_id: usize, chunk_type: ChunkType, content: impl Into<String>) -> Self {
        Self {
            chunk_id,
            chunk_type,
            content: content.into(),
            metadata: ChunkMetadata::default(),
        }
    }

    /// Set the metadata for this chunk.
    #[must_use]
    pub fn with_metadata(mut self, metadata: ChunkMetadata) -> Self {
        self.metadata = metadata;
        self
    }

    /// Set the timestamp for this chunk.
    #[must_use]
    pub fn with_timestamp(mut self, timestamp_ms: u64) -> Self {
        self.metadata.timestamp_ms = timestamp_ms;
        self
    }

    /// Set the layer for this chunk.
    #[must_use]
    pub fn with_layer(mut self, layer: impl Into<String>) -> Self {
        self.metadata.layer = Some(layer.into());
        self
    }

    /// Set the confidence for this chunk.
    #[must_use]
    pub fn with_confidence(mut self, confidence: f32) -> Self {
        self.metadata.confidence = Some(confidence);
        self
    }

    /// Set the duration for this chunk.
    #[must_use]
    pub fn with_duration(mut self, duration_ms: u64) -> Self {
        self.metadata.duration_ms = Some(duration_ms);
        self
    }
}

/// The type of a pipeline chunk.
#[derive(Debug, Clone, Serialize, Deserialize, PartialEq)]
pub enum ChunkType {
    /// Search phase has started.
    SearchStarted,
    /// A search result was found.
    SearchResult {
        /// Rank of this result (0-indexed).
        rank: usize,
        /// Similarity score.
        score: f32,
    },
    /// Search phase has completed.
    SearchCompleted {
        /// Total number of results found.
        total: usize,
    },
    /// A draft answer was generated.
    DraftGenerated,
    /// Speculation phase has started.
    SpeculationStarted,
    /// Speculation is making progress.
    SpeculationProgress {
        /// Current stage of speculation.
        stage: String,
        /// Current confidence level.
        confidence: f32,
    },
    /// Speculation has made a decision.
    SpeculationDecision(SpeculationDecision),
    /// Verification phase has started.
    VerificationStarted,
    /// A claim was extracted from the draft.
    ClaimExtracted {
        /// Unique identifier for this claim.
        claim_id: usize,
    },
    /// A claim was verified.
    ClaimVerified {
        /// Unique identifier for this claim.
        claim_id: usize,
        /// Verification status.
        status: String,
    },
    /// Verification phase has completed.
    VerificationCompleted,
    /// The final answer is ready.
    FinalAnswer,
    /// An error occurred.
    Error(String),
}

/// Metadata about a pipeline chunk.
#[derive(Debug, Clone, Default, Serialize, Deserialize)]
pub struct ChunkMetadata {
    /// Timestamp in milliseconds since stream start.
    pub timestamp_ms: u64,
    /// Which layer produced this chunk.
    pub layer: Option<String>,
    /// Confidence score at this point.
    pub confidence: Option<f32>,
    /// Duration of the operation in milliseconds.
    pub duration_ms: Option<u64>,
}

impl ChunkMetadata {
    /// Create new metadata with the given timestamp.
    #[must_use]
    pub fn new(timestamp_ms: u64) -> Self {
        Self {
            timestamp_ms,
            layer: None,
            confidence: None,
            duration_ms: None,
        }
    }

    /// Set the layer.
    #[must_use]
    pub fn with_layer(mut self, layer: impl Into<String>) -> Self {
        self.layer = Some(layer.into());
        self
    }

    /// Set the confidence.
    #[must_use]
    pub fn with_confidence(mut self, confidence: f32) -> Self {
        self.confidence = Some(confidence);
        self
    }

    /// Set the duration.
    #[must_use]
    pub fn with_duration(mut self, duration_ms: u64) -> Self {
        self.duration_ms = Some(duration_ms);
        self
    }
}

/// Streaming pipeline result that allows consuming chunks as they arrive.
#[cfg(feature = "native")]
pub struct StreamingPipelineResult {
    receiver: mpsc::Receiver<PipelineChunk>,
    final_output: Option<PipelineOutput>,
    collected_chunks: Vec<PipelineChunk>,
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
#[async_trait]
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
    inner: P,
    chunk_buffer_size: usize,
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

#[async_trait]
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
#[async_trait]
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
#[async_trait]
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

/// Progress reporter for pipeline stages.
#[cfg(feature = "native")]
pub struct ProgressReporter {
    sender: mpsc::Sender<PipelineChunk>,
    chunk_counter: usize,
    start_time: Instant,
}

#[cfg(feature = "native")]
impl ProgressReporter {
    /// Create a new progress reporter.
    #[must_use]
    pub fn new(sender: mpsc::Sender<PipelineChunk>) -> Self {
        Self {
            sender,
            chunk_counter: 0,
            start_time: Instant::now(),
        }
    }

    /// Get the current timestamp in milliseconds.
    fn timestamp_ms(&self) -> u64 {
        elapsed_ms(&self.start_time)
    }

    /// Get the next chunk ID and increment the counter.
    fn next_chunk_id(&mut self) -> usize {
        let id = self.chunk_counter;
        self.chunk_counter += 1;
        id
    }

    /// Report a generic chunk.
    pub async fn report(&mut self, chunk_type: ChunkType, content: &str) {
        let chunk = PipelineChunk::new(self.next_chunk_id(), chunk_type, content)
            .with_timestamp(self.timestamp_ms());
        let _ = self.sender.send(chunk).await;
    }

    /// Report a search result.
    pub async fn report_search_result(&mut self, rank: usize, score: f32, doc_preview: &str) {
        let chunk = PipelineChunk::new(
            self.next_chunk_id(),
            ChunkType::SearchResult { rank, score },
            doc_preview,
        )
        .with_layer("Echo")
        .with_confidence(score)
        .with_timestamp(self.timestamp_ms());
        let _ = self.sender.send(chunk).await;
    }

    /// Report speculation progress.
    pub async fn report_speculation(&mut self, stage: &str, confidence: f32) {
        let chunk = PipelineChunk::new(
            self.next_chunk_id(),
            ChunkType::SpeculationProgress {
                stage: stage.to_string(),
                confidence,
            },
            format!("Speculation stage: {stage}"),
        )
        .with_layer("Speculator")
        .with_confidence(confidence)
        .with_timestamp(self.timestamp_ms());
        let _ = self.sender.send(chunk).await;
    }

    /// Report a claim extraction.
    pub async fn report_claim(&mut self, claim_id: usize, claim_text: &str) {
        let chunk = PipelineChunk::new(
            self.next_chunk_id(),
            ChunkType::ClaimExtracted { claim_id },
            claim_text,
        )
        .with_layer("Judge")
        .with_timestamp(self.timestamp_ms());
        let _ = self.sender.send(chunk).await;
    }

    /// Report an error.
    pub async fn report_error(&mut self, error: &str) {
        let chunk = PipelineChunk::new(
            self.next_chunk_id(),
            ChunkType::Error(error.to_string()),
            error,
        )
        .with_timestamp(self.timestamp_ms());
        let _ = self.sender.send(chunk).await;
    }

    /// Report the final answer.
    pub async fn report_final(&mut self, answer: &str, confidence: f32) {
        let chunk = PipelineChunk::new(self.next_chunk_id(), ChunkType::FinalAnswer, answer)
            .with_confidence(confidence)
            .with_timestamp(self.timestamp_ms());
        let _ = self.sender.send(chunk).await;
    }

    /// Report search started.
    pub async fn report_search_started(&mut self, query: &str) {
        let chunk = PipelineChunk::new(
            self.next_chunk_id(),
            ChunkType::SearchStarted,
            format!("Searching for: {query}"),
        )
        .with_layer("Echo")
        .with_timestamp(self.timestamp_ms());
        let _ = self.sender.send(chunk).await;
    }

    /// Report search completed.
    pub async fn report_search_completed(&mut self, total: usize) {
        let chunk = PipelineChunk::new(
            self.next_chunk_id(),
            ChunkType::SearchCompleted { total },
            format!("Found {total} results"),
        )
        .with_layer("Echo")
        .with_timestamp(self.timestamp_ms());
        let _ = self.sender.send(chunk).await;
    }

    /// Report draft generated.
    pub async fn report_draft(&mut self, draft_preview: &str, confidence: f32) {
        let chunk = PipelineChunk::new(
            self.next_chunk_id(),
            ChunkType::DraftGenerated,
            draft_preview,
        )
        .with_confidence(confidence)
        .with_timestamp(self.timestamp_ms());
        let _ = self.sender.send(chunk).await;
    }

    /// Report speculation started.
    pub async fn report_speculation_started(&mut self) {
        let chunk = PipelineChunk::new(
            self.next_chunk_id(),
            ChunkType::SpeculationStarted,
            "Starting speculation verification",
        )
        .with_layer("Speculator")
        .with_timestamp(self.timestamp_ms());
        let _ = self.sender.send(chunk).await;
    }

    /// Report speculation decision.
    pub async fn report_speculation_decision(
        &mut self,
        decision: SpeculationDecision,
        confidence: f32,
    ) {
        let chunk = PipelineChunk::new(
            self.next_chunk_id(),
            ChunkType::SpeculationDecision(decision.clone()),
            format!("Decision: {decision:?}"),
        )
        .with_layer("Speculator")
        .with_confidence(confidence)
        .with_timestamp(self.timestamp_ms());
        let _ = self.sender.send(chunk).await;
    }

    /// Report verification started.
    pub async fn report_verification_started(&mut self) {
        let chunk = PipelineChunk::new(
            self.next_chunk_id(),
            ChunkType::VerificationStarted,
            "Starting logic verification",
        )
        .with_layer("Judge")
        .with_timestamp(self.timestamp_ms());
        let _ = self.sender.send(chunk).await;
    }

    /// Report claim verified.
    pub async fn report_claim_verified(
        &mut self,
        claim_id: usize,
        status: &str,
        explanation: &str,
    ) {
        let chunk = PipelineChunk::new(
            self.next_chunk_id(),
            ChunkType::ClaimVerified {
                claim_id,
                status: status.to_string(),
            },
            explanation,
        )
        .with_layer("Judge")
        .with_timestamp(self.timestamp_ms());
        let _ = self.sender.send(chunk).await;
    }

    /// Report verification completed.
    pub async fn report_verification_completed(
        &mut self,
        summary: &str,
        confidence: f32,
        duration_ms: u64,
    ) {
        let chunk = PipelineChunk::new(
            self.next_chunk_id(),
            ChunkType::VerificationCompleted,
            summary,
        )
        .with_layer("Judge")
        .with_confidence(confidence)
        .with_duration(duration_ms)
        .with_timestamp(self.timestamp_ms());
        let _ = self.sender.send(chunk).await;
    }

    /// Get the current chunk count.
    #[must_use]
    pub fn chunk_count(&self) -> usize {
        self.chunk_counter
    }

    /// Get the elapsed time since the reporter was created.
    #[must_use]
    pub fn elapsed_ms(&self) -> u64 {
        self.timestamp_ms()
    }
}

/// Helper function to truncate content for display.
fn truncate_content(content: &str, max_len: usize) -> String {
    if content.len() <= max_len {
        content.to_string()
    } else {
        format!("{}...", &content[..max_len.saturating_sub(3)])
    }
}

#[cfg(all(test, feature = "native"))]
mod tests {
    use super::*;
    use crate::layer1_echo::{EchoLayer, InMemoryVectorStore, MockEmbeddingProvider};
    use crate::layer2_speculator::RuleBasedSpeculator;
    use crate::layer3_judge::{AdvancedClaimExtractor, JudgeConfig, JudgeImpl, MockSmtVerifier};
    use crate::pipeline::{Pipeline, PipelineConfig};
    use crate::types::Document;
    use tokio::sync::mpsc;

    type TestPipeline = Pipeline<
        EchoLayer<MockEmbeddingProvider, InMemoryVectorStore>,
        RuleBasedSpeculator,
        JudgeImpl<AdvancedClaimExtractor, MockSmtVerifier>,
    >;

    fn create_test_pipeline() -> TestPipeline {
        let echo = EchoLayer::new(MockEmbeddingProvider::new(64), InMemoryVectorStore::new(64));
        let speculator = RuleBasedSpeculator::default();
        let judge = JudgeImpl::new(
            AdvancedClaimExtractor::new(),
            MockSmtVerifier::default(),
            JudgeConfig::default(),
        );
        Pipeline::new(
            echo,
            speculator,
            judge,
            PipelineConfig {
                enable_fast_path: false,
                ..Default::default()
            },
        )
    }

    #[tokio::test]
    async fn test_pipeline_chunk_creation() {
        let chunk = PipelineChunk::new(0, ChunkType::SearchStarted, "test content")
            .with_timestamp(100)
            .with_layer("Echo")
            .with_confidence(0.9);

        assert_eq!(chunk.chunk_id, 0);
        assert_eq!(chunk.content, "test content");
        assert_eq!(chunk.metadata.timestamp_ms, 100);
        assert_eq!(chunk.metadata.layer, Some("Echo".to_string()));
        assert_eq!(chunk.metadata.confidence, Some(0.9));
    }

    #[tokio::test]
    async fn test_chunk_metadata_builder() {
        let metadata = ChunkMetadata::new(50)
            .with_layer("Speculator")
            .with_confidence(0.85)
            .with_duration(200);

        assert_eq!(metadata.timestamp_ms, 50);
        assert_eq!(metadata.layer, Some("Speculator".to_string()));
        assert_eq!(metadata.confidence, Some(0.85));
        assert_eq!(metadata.duration_ms, Some(200));
    }

    #[tokio::test]
    async fn test_chunk_type_equality() {
        assert_eq!(ChunkType::SearchStarted, ChunkType::SearchStarted);
        assert_eq!(
            ChunkType::SearchResult {
                rank: 0,
                score: 0.9
            },
            ChunkType::SearchResult {
                rank: 0,
                score: 0.9
            }
        );
        assert_ne!(ChunkType::SearchStarted, ChunkType::DraftGenerated);
    }

    #[tokio::test]
    async fn test_streaming_wrapper_creation() {
        let pipeline = create_test_pipeline();
        let wrapper = StreamingPipelineWrapper::new(pipeline).with_buffer_size(64);

        assert_eq!(wrapper.buffer_size(), 64);
    }

    #[tokio::test]
    async fn test_streaming_wrapper_buffer_minimum() {
        let pipeline = create_test_pipeline();
        let wrapper = StreamingPipelineWrapper::new(pipeline).with_buffer_size(0);

        assert_eq!(wrapper.buffer_size(), 1);
    }

    #[tokio::test]
    async fn test_streaming_empty_index() {
        let pipeline = create_test_pipeline();
        let wrapper = StreamingPipelineWrapper::new(pipeline);
        let query = Query::new("What is the meaning of life?");

        let mut result = wrapper.process_streaming(query).await.unwrap();

        let mut chunks = Vec::new();
        while let Some(chunk) = result.next().await {
            chunks.push(chunk);
        }

        // Should have at least search started, search completed, draft generated, and final answer
        assert!(chunks.len() >= 4);
        assert!(matches!(chunks[0].chunk_type, ChunkType::SearchStarted));
    }

    #[tokio::test]
    async fn test_streaming_with_documents() {
        let mut pipeline = create_test_pipeline();
        pipeline
            .index(Document::new("The capital of France is Paris."))
            .await
            .unwrap();

        let wrapper = StreamingPipelineWrapper::new(pipeline);
        let query = Query::new("What is the capital of France?");

        let mut result = wrapper.process_streaming(query).await.unwrap();

        let mut has_search_result = false;
        let mut has_final_answer = false;

        while let Some(chunk) = result.next().await {
            if matches!(chunk.chunk_type, ChunkType::SearchResult { .. }) {
                has_search_result = true;
            }
            if matches!(chunk.chunk_type, ChunkType::FinalAnswer) {
                has_final_answer = true;
            }
        }

        assert!(has_search_result);
        assert!(has_final_answer);
    }

    #[tokio::test]
    async fn test_streaming_chunk_ordering() {
        let mut pipeline = create_test_pipeline();
        pipeline
            .index(Document::new("Test document content."))
            .await
            .unwrap();

        let wrapper = StreamingPipelineWrapper::new(pipeline);
        let query = Query::new("test");

        let mut result = wrapper.process_streaming(query).await.unwrap();

        let mut chunks = Vec::new();
        while let Some(chunk) = result.next().await {
            chunks.push(chunk);
        }

        // Verify chunk IDs are in order
        for (i, chunk) in chunks.iter().enumerate() {
            assert_eq!(chunk.chunk_id, i);
        }
    }

    #[tokio::test]
    async fn test_streaming_into_stream() {
        use futures::StreamExt;

        let mut pipeline = create_test_pipeline();
        pipeline
            .index(Document::new("Stream test document."))
            .await
            .unwrap();

        let wrapper = StreamingPipelineWrapper::new(pipeline);
        let query = Query::new("stream");

        let result = wrapper.process_streaming(query).await.unwrap();
        let mut stream = result.into_stream();

        let mut count = 0;
        while let Some(chunk) = stream.next().await {
            count += 1;
            assert!(chunk.chunk_id < 100); // Sanity check
        }

        assert!(count > 0);
    }

    #[tokio::test]
    async fn test_streaming_batch() {
        let mut pipeline = create_test_pipeline();
        pipeline
            .index(Document::new("Alpha document."))
            .await
            .unwrap();
        pipeline
            .index(Document::new("Beta document."))
            .await
            .unwrap();

        let wrapper = StreamingPipelineWrapper::new(pipeline);
        let queries = vec![Query::new("alpha"), Query::new("beta")];

        let results = wrapper.process_batch_streaming(queries).await;

        assert_eq!(results.len(), 2);
    }

    #[tokio::test]
    async fn test_streaming_collected_chunks() {
        let pipeline = create_test_pipeline();
        let wrapper = StreamingPipelineWrapper::new(pipeline);
        let query = Query::new("test");

        let mut result = wrapper.process_streaming(query).await.unwrap();

        // Consume some chunks
        let _ = result.next().await;
        let _ = result.next().await;

        assert!(result.collected_chunks().len() >= 2);
        assert!(result.chunk_count() >= 2);
    }

    #[tokio::test]
    async fn test_streaming_for_each() {
        let pipeline = create_test_pipeline();
        let wrapper = StreamingPipelineWrapper::new(pipeline);
        let query = Query::new("test");

        let result = wrapper.process_streaming(query).await.unwrap();

        let mut count = 0;
        result
            .for_each(|_chunk| {
                count += 1;
            })
            .await;

        assert!(count > 0);
    }

    #[tokio::test]
    async fn test_progress_reporter_basic() {
        let (tx, mut rx) = mpsc::channel(32);
        let mut reporter = ProgressReporter::new(tx);

        reporter.report_search_started("test query").await;
        reporter.report_search_result(0, 0.9, "Test document").await;
        reporter.report_search_completed(1).await;

        let chunk1 = rx.recv().await.unwrap();
        assert!(matches!(chunk1.chunk_type, ChunkType::SearchStarted));

        let chunk2 = rx.recv().await.unwrap();
        assert!(matches!(
            chunk2.chunk_type,
            ChunkType::SearchResult { rank: 0, .. }
        ));

        let chunk3 = rx.recv().await.unwrap();
        assert!(matches!(
            chunk3.chunk_type,
            ChunkType::SearchCompleted { total: 1 }
        ));
    }

    #[tokio::test]
    async fn test_progress_reporter_speculation() {
        let (tx, mut rx) = mpsc::channel(32);
        let mut reporter = ProgressReporter::new(tx);

        reporter.report_speculation_started().await;
        reporter.report_speculation("verification", 0.8).await;
        reporter
            .report_speculation_decision(SpeculationDecision::Accept, 0.9)
            .await;

        let chunk1 = rx.recv().await.unwrap();
        assert!(matches!(chunk1.chunk_type, ChunkType::SpeculationStarted));

        let chunk2 = rx.recv().await.unwrap();
        assert!(matches!(
            chunk2.chunk_type,
            ChunkType::SpeculationProgress { .. }
        ));

        let chunk3 = rx.recv().await.unwrap();
        assert!(matches!(
            chunk3.chunk_type,
            ChunkType::SpeculationDecision(SpeculationDecision::Accept)
        ));
    }

    #[tokio::test]
    async fn test_progress_reporter_verification() {
        let (tx, mut rx) = mpsc::channel(32);
        let mut reporter = ProgressReporter::new(tx);

        reporter.report_verification_started().await;
        reporter.report_claim(0, "Test claim").await;
        reporter
            .report_claim_verified(0, "Verified", "Explanation")
            .await;
        reporter
            .report_verification_completed("Summary", 0.85, 100)
            .await;

        let chunk1 = rx.recv().await.unwrap();
        assert!(matches!(chunk1.chunk_type, ChunkType::VerificationStarted));

        let chunk2 = rx.recv().await.unwrap();
        assert!(matches!(
            chunk2.chunk_type,
            ChunkType::ClaimExtracted { claim_id: 0 }
        ));

        let chunk3 = rx.recv().await.unwrap();
        assert!(matches!(
            chunk3.chunk_type,
            ChunkType::ClaimVerified { claim_id: 0, .. }
        ));

        let chunk4 = rx.recv().await.unwrap();
        assert!(matches!(
            chunk4.chunk_type,
            ChunkType::VerificationCompleted
        ));
    }

    #[tokio::test]
    async fn test_progress_reporter_error() {
        let (tx, mut rx) = mpsc::channel(32);
        let mut reporter = ProgressReporter::new(tx);

        reporter.report_error("Test error").await;

        let chunk = rx.recv().await.unwrap();
        assert!(matches!(chunk.chunk_type, ChunkType::Error(_)));
    }

    #[tokio::test]
    async fn test_progress_reporter_final_answer() {
        let (tx, mut rx) = mpsc::channel(32);
        let mut reporter = ProgressReporter::new(tx);

        reporter.report_final("The answer is 42", 0.95).await;

        let chunk = rx.recv().await.unwrap();
        assert!(matches!(chunk.chunk_type, ChunkType::FinalAnswer));
        assert_eq!(chunk.content, "The answer is 42");
        assert_eq!(chunk.metadata.confidence, Some(0.95));
    }

    #[tokio::test]
    async fn test_progress_reporter_chunk_counter() {
        let (tx, _rx) = mpsc::channel(32);
        let mut reporter = ProgressReporter::new(tx);

        assert_eq!(reporter.chunk_count(), 0);

        reporter.report(ChunkType::SearchStarted, "test").await;
        assert_eq!(reporter.chunk_count(), 1);

        reporter.report(ChunkType::DraftGenerated, "test").await;
        assert_eq!(reporter.chunk_count(), 2);
    }

    #[tokio::test]
    async fn test_truncate_content() {
        assert_eq!(truncate_content("short", 10), "short");
        assert_eq!(
            truncate_content("a longer string that needs truncation", 10),
            "a longe..."
        );
        assert_eq!(truncate_content("exactly10c", 10), "exactly10c");
    }

    #[tokio::test]
    async fn test_streaming_result_from_output() {
        let query = Query::new("test");
        let draft = crate::types::Draft::new("Test answer", "test");
        let output = PipelineOutput::new(query, draft);

        let result = StreamingPipelineResult::from_output(output);
        assert!(result.has_final_output());
    }

    #[tokio::test]
    async fn test_chunk_with_metadata() {
        let metadata = ChunkMetadata::new(100)
            .with_layer("Test")
            .with_confidence(0.5)
            .with_duration(50);

        let chunk = PipelineChunk::new(0, ChunkType::SearchStarted, "test").with_metadata(metadata);

        assert_eq!(chunk.metadata.timestamp_ms, 100);
        assert_eq!(chunk.metadata.layer, Some("Test".to_string()));
        assert_eq!(chunk.metadata.confidence, Some(0.5));
        assert_eq!(chunk.metadata.duration_ms, Some(50));
    }

    #[tokio::test]
    async fn test_concurrent_streaming_queries() {
        let mut pipeline = create_test_pipeline();
        pipeline
            .index(Document::new("First document content."))
            .await
            .unwrap();
        pipeline
            .index(Document::new("Second document content."))
            .await
            .unwrap();

        let wrapper = std::sync::Arc::new(StreamingPipelineWrapper::new(pipeline));

        let handles: Vec<_> = (0..3)
            .map(|i| {
                let wrapper = wrapper.clone();
                tokio::spawn(async move {
                    let query = Query::new(format!("query {i}"));
                    let mut result = wrapper.process_streaming(query).await.unwrap();
                    let mut count = 0;
                    while let Some(_chunk) = result.next().await {
                        count += 1;
                    }
                    count
                })
            })
            .collect();

        for handle in handles {
            let count = handle.await.unwrap();
            assert!(count > 0);
        }
    }

    #[tokio::test]
    async fn test_streaming_wrapper_inner_access() {
        let pipeline = create_test_pipeline();
        let wrapper = StreamingPipelineWrapper::new(pipeline);

        // Test inner() returns a reference
        let _ = wrapper.inner();

        // Test config access through inner
        let config = wrapper.inner().config();
        assert!(!config.enable_fast_path);
    }

    #[tokio::test]
    async fn test_streaming_wrapper_inner_mut() {
        let pipeline = create_test_pipeline();
        let mut wrapper = StreamingPipelineWrapper::new(pipeline);

        // Test inner_mut() returns a mutable reference
        let _inner = wrapper.inner_mut();
    }
}
