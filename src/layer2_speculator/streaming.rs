//! Streaming verification support for the Speculator layer.
//!
//! This module provides streaming capabilities for verification results,
//! allowing consumers to receive partial results as they become available.

use async_trait::async_trait;
use std::pin::Pin;

use crate::error::SpeculatorError;
use crate::layer2_speculator::traits::Speculator;
use crate::types::{Draft, SearchResult, SpeculationDecision, SpeculationResult};

#[cfg(feature = "native")]
use futures::Stream;
#[cfg(feature = "native")]
use tokio::sync::mpsc;

/// A chunk of streaming verification output.
#[derive(Debug, Clone)]
pub struct VerificationChunk {
    /// Unique identifier for this chunk within the stream.
    pub chunk_id: usize,
    /// The content of this chunk.
    pub content: String,
    /// Whether this is the final chunk in the stream.
    pub is_final: bool,
    /// Partial decision if available at this point.
    pub partial_decision: Option<SpeculationDecision>,
    /// Partial confidence score (0.0 to 1.0).
    pub partial_confidence: f32,
}

impl VerificationChunk {
    /// Create a new verification chunk.
    #[must_use]
    pub fn new(chunk_id: usize, content: impl Into<String>) -> Self {
        Self {
            chunk_id,
            content: content.into(),
            is_final: false,
            partial_decision: None,
            partial_confidence: 0.0,
        }
    }

    /// Mark this chunk as final.
    #[must_use]
    pub fn with_final(mut self) -> Self {
        self.is_final = true;
        self
    }

    /// Set a partial decision for this chunk.
    #[must_use]
    pub fn with_decision(mut self, decision: SpeculationDecision) -> Self {
        self.partial_decision = Some(decision);
        self
    }

    /// Set the partial confidence for this chunk.
    #[must_use]
    pub fn with_confidence(mut self, confidence: f32) -> Self {
        self.partial_confidence = confidence;
        self
    }
}

impl Default for VerificationChunk {
    fn default() -> Self {
        Self::new(0, "")
    }
}

/// Streaming verification result that allows consuming chunks as they arrive.
#[cfg(feature = "native")]
pub struct StreamingVerification {
    receiver: mpsc::Receiver<VerificationChunk>,
    final_result: Option<SpeculationResult>,
    collected_chunks: Vec<VerificationChunk>,
}

#[cfg(feature = "native")]
impl StreamingVerification {
    /// Create a new streaming verification with the given receiver.
    #[must_use]
    pub fn new(receiver: mpsc::Receiver<VerificationChunk>) -> Self {
        Self {
            receiver,
            final_result: None,
            collected_chunks: Vec::new(),
        }
    }

    /// Create a streaming verification with a pre-computed result.
    #[must_use]
    pub fn from_result(result: SpeculationResult) -> Self {
        let (_, receiver) = mpsc::channel(1);
        Self {
            receiver,
            final_result: Some(result),
            collected_chunks: Vec::new(),
        }
    }

    /// Get the next chunk from the stream.
    ///
    /// Returns `None` when the stream is exhausted.
    pub async fn next_chunk(&mut self) -> Option<VerificationChunk> {
        let chunk = self.receiver.recv().await;
        if let Some(ref c) = chunk {
            self.collected_chunks.push(c.clone());
        }
        chunk
    }

    /// Convert this streaming verification into a futures Stream.
    #[must_use]
    pub fn into_stream(self) -> Pin<Box<dyn Stream<Item = VerificationChunk> + Send>> {
        Box::pin(tokio_stream::wrappers::ReceiverStream::new(self.receiver))
    }

    /// Collect all remaining chunks and return the final result.
    pub async fn collect(mut self) -> SpeculationResult {
        if let Some(result) = self.final_result {
            return result;
        }

        // Collect all remaining chunks
        while let Some(chunk) = self.receiver.recv().await {
            self.collected_chunks.push(chunk);
        }

        // Build the final result from collected chunks
        self.build_result_from_chunks()
    }

    /// Build a speculation result from collected chunks.
    fn build_result_from_chunks(&self) -> SpeculationResult {
        if self.collected_chunks.is_empty() {
            return SpeculationResult::new(SpeculationDecision::Revise, 0.5)
                .with_explanation("No chunks received");
        }

        // Find the final chunk or use the last one
        let final_chunk = self
            .collected_chunks
            .iter()
            .rfind(|c| c.is_final)
            .or_else(|| self.collected_chunks.last());

        let (decision, confidence) = if let Some(chunk) = final_chunk {
            (
                chunk
                    .partial_decision
                    .clone()
                    .unwrap_or(SpeculationDecision::Revise),
                chunk.partial_confidence,
            )
        } else {
            (SpeculationDecision::Revise, 0.5)
        };

        // Combine all content
        let explanation: String = self
            .collected_chunks
            .iter()
            .map(|c| c.content.as_str())
            .collect::<Vec<_>>()
            .join("");

        SpeculationResult::new(decision, confidence).with_explanation(explanation)
    }

    /// Check if the stream has a pre-computed final result.
    #[must_use]
    pub fn has_final_result(&self) -> bool {
        self.final_result.is_some()
    }

    /// Get a reference to the collected chunks so far.
    #[must_use]
    pub fn collected_chunks(&self) -> &[VerificationChunk] {
        &self.collected_chunks
    }
}

/// Non-native (WASM) placeholder for streaming verification.
#[cfg(not(feature = "native"))]
pub struct StreamingVerification {
    result: SpeculationResult,
}

#[cfg(not(feature = "native"))]
impl StreamingVerification {
    /// Create a streaming verification from a result (WASM doesn't support true streaming).
    #[must_use]
    pub fn from_result(result: SpeculationResult) -> Self {
        Self { result }
    }

    /// Collect the result (immediately available in WASM).
    pub async fn collect(self) -> SpeculationResult {
        self.result
    }
}

/// Trait extension for streaming verification capabilities.
#[async_trait]
pub trait StreamingSpeculator: Speculator {
    /// Verify a draft with streaming output.
    ///
    /// # Arguments
    /// * `draft` - The draft answer to verify
    /// * `context` - The retrieved documents providing context
    ///
    /// # Returns
    /// A streaming verification that can be consumed chunk by chunk.
    async fn verify_draft_streaming(
        &self,
        draft: &Draft,
        context: &[SearchResult],
    ) -> Result<StreamingVerification, SpeculatorError>;
}

/// A wrapper that adds streaming capabilities to any Speculator.
pub struct StreamingSpeculatorWrapper<S: Speculator> {
    inner: S,
    chunk_size: usize,
}

impl<S: Speculator> StreamingSpeculatorWrapper<S> {
    /// Create a new streaming wrapper around an existing speculator.
    #[must_use]
    pub fn new(inner: S) -> Self {
        Self {
            inner,
            chunk_size: 50, // Default: 50 characters per chunk
        }
    }

    /// Set the chunk size for streaming output.
    #[must_use]
    pub fn with_chunk_size(mut self, chunk_size: usize) -> Self {
        self.chunk_size = chunk_size.max(1);
        self
    }

    /// Get a reference to the inner speculator.
    #[must_use]
    pub fn inner(&self) -> &S {
        &self.inner
    }

    /// Get the configured chunk size.
    #[must_use]
    pub fn chunk_size(&self) -> usize {
        self.chunk_size
    }
}

#[cfg(feature = "native")]
#[async_trait]
impl<S: Speculator + Send + Sync> StreamingSpeculator for StreamingSpeculatorWrapper<S> {
    async fn verify_draft_streaming(
        &self,
        draft: &Draft,
        context: &[SearchResult],
    ) -> Result<StreamingVerification, SpeculatorError> {
        // Get the full result first
        let result = self.inner.verify_draft(draft, context).await?;

        // Create a channel for streaming
        let (tx, rx) = mpsc::channel(32);
        let chunk_size = self.chunk_size;
        let explanation = result.explanation.clone();
        let decision = result.decision.clone();
        let confidence = result.confidence;

        // Spawn a task to send chunks
        tokio::spawn(async move {
            let chars: Vec<char> = explanation.chars().collect();
            let total_chunks = chars.len().div_ceil(chunk_size);

            for (i, chunk_chars) in chars.chunks(chunk_size).enumerate() {
                let chunk_content: String = chunk_chars.iter().collect();
                let is_final = i == total_chunks - 1;

                #[allow(clippy::cast_precision_loss)]
                let progress = (i + 1) as f32 / total_chunks as f32;

                let chunk =
                    VerificationChunk::new(i, chunk_content).with_confidence(confidence * progress);

                // Add decision only on final chunk
                let chunk = if is_final {
                    chunk.with_final().with_decision(decision.clone())
                } else {
                    chunk
                };

                if tx.send(chunk).await.is_err() {
                    break;
                }

                // Small delay to simulate streaming
                tokio::time::sleep(tokio::time::Duration::from_millis(1)).await;
            }
        });

        Ok(StreamingVerification::new(rx))
    }
}

#[cfg(not(feature = "native"))]
#[async_trait]
impl<S: Speculator + Send + Sync> StreamingSpeculator for StreamingSpeculatorWrapper<S> {
    async fn verify_draft_streaming(
        &self,
        draft: &Draft,
        context: &[SearchResult],
    ) -> Result<StreamingVerification, SpeculatorError> {
        let result = self.inner.verify_draft(draft, context).await?;
        Ok(StreamingVerification::from_result(result))
    }
}

#[async_trait]
impl<S: Speculator + Send + Sync> Speculator for StreamingSpeculatorWrapper<S> {
    async fn verify_draft(
        &self,
        draft: &Draft,
        context: &[SearchResult],
    ) -> Result<SpeculationResult, SpeculatorError> {
        self.inner.verify_draft(draft, context).await
    }

    async fn revise_draft(
        &self,
        draft: &Draft,
        context: &[SearchResult],
        speculation: &SpeculationResult,
    ) -> Result<Draft, SpeculatorError> {
        self.inner.revise_draft(draft, context, speculation).await
    }

    fn config(&self) -> &crate::layer2_speculator::traits::SpeculatorConfig {
        self.inner.config()
    }
}

#[cfg(all(test, feature = "native"))]
#[allow(clippy::float_cmp)]
mod tests {
    use super::*;
    use crate::layer2_speculator::traits::RuleBasedSpeculator;
    use crate::types::Document;

    fn create_context() -> Vec<SearchResult> {
        vec![
            SearchResult::new(
                Document::new("The capital of France is Paris. It is known for the Eiffel Tower."),
                0.9,
                0,
            ),
            SearchResult::new(
                Document::new("Paris has a population of about 2 million in the city proper."),
                0.85,
                1,
            ),
        ]
    }

    #[tokio::test]
    async fn test_verification_chunk_creation() {
        let chunk = VerificationChunk::new(0, "test content")
            .with_confidence(0.8)
            .with_decision(SpeculationDecision::Accept)
            .with_final();

        assert_eq!(chunk.chunk_id, 0);
        assert_eq!(chunk.content, "test content");
        assert!(chunk.is_final);
        assert_eq!(chunk.partial_confidence, 0.8);
        assert!(matches!(
            chunk.partial_decision,
            Some(SpeculationDecision::Accept)
        ));
    }

    #[tokio::test]
    async fn test_streaming_wrapper_creation() {
        let speculator = RuleBasedSpeculator::default();
        let wrapper = StreamingSpeculatorWrapper::new(speculator).with_chunk_size(100);

        assert_eq!(wrapper.chunk_size(), 100);
    }

    #[tokio::test]
    async fn test_streaming_verification() {
        let speculator = RuleBasedSpeculator::default();
        let wrapper = StreamingSpeculatorWrapper::new(speculator).with_chunk_size(10);

        let draft = Draft::new(
            "The capital of France is Paris, which is famous for the Eiffel Tower.",
            "What is the capital of France?",
        )
        .with_confidence(0.9);

        let context = create_context();
        let streaming = wrapper
            .verify_draft_streaming(&draft, &context)
            .await
            .unwrap();
        let result = streaming.collect().await;

        assert!(result.confidence > 0.0);
    }

    #[tokio::test]
    async fn test_streaming_next_chunk() {
        let speculator = RuleBasedSpeculator::default();
        let wrapper = StreamingSpeculatorWrapper::new(speculator).with_chunk_size(5);

        let draft = Draft::new(
            "Paris is the capital of France with the Eiffel Tower.",
            "What is the capital of France?",
        )
        .with_confidence(0.9);

        let context = create_context();
        let mut streaming = wrapper
            .verify_draft_streaming(&draft, &context)
            .await
            .unwrap();

        let mut chunk_count = 0;
        while let Some(chunk) = streaming.next_chunk().await {
            chunk_count += 1;
            assert!(chunk.chunk_id < 100); // Sanity check

            if chunk.is_final {
                assert!(chunk.partial_decision.is_some());
                break;
            }
        }

        assert!(chunk_count > 0);
    }

    #[tokio::test]
    async fn test_streaming_from_result() {
        let result = SpeculationResult::new(SpeculationDecision::Accept, 0.95)
            .with_explanation("Test explanation");

        let streaming = StreamingVerification::from_result(result.clone());
        assert!(streaming.has_final_result());

        let collected = streaming.collect().await;
        assert!(matches!(collected.decision, SpeculationDecision::Accept));
        assert_eq!(collected.confidence, 0.95);
    }

    #[tokio::test]
    async fn test_wrapper_delegates_verify() {
        let speculator = RuleBasedSpeculator::default();
        let wrapper = StreamingSpeculatorWrapper::new(speculator);

        let draft = Draft::new(
            "Paris is the capital of France.",
            "What is the capital of France?",
        )
        .with_confidence(0.8);

        let context = create_context();
        let result = wrapper.verify_draft(&draft, &context).await.unwrap();

        assert!(result.confidence > 0.0);
    }

    #[tokio::test]
    async fn test_wrapper_delegates_revise() {
        let speculator = RuleBasedSpeculator::default();
        let wrapper = StreamingSpeculatorWrapper::new(speculator);

        let draft = Draft::new("Paris", "What is the capital of France?");
        let context = create_context();

        let speculation = wrapper.verify_draft(&draft, &context).await.unwrap();
        let revised = wrapper
            .revise_draft(&draft, &context, &speculation)
            .await
            .unwrap();

        assert!(revised.content.len() > draft.content.len());
    }

    #[tokio::test]
    async fn test_chunk_size_minimum() {
        let speculator = RuleBasedSpeculator::default();
        let wrapper = StreamingSpeculatorWrapper::new(speculator).with_chunk_size(0);

        // Should be clamped to at least 1
        assert_eq!(wrapper.chunk_size(), 1);
    }

    #[tokio::test]
    async fn test_streaming_collected_chunks() {
        let speculator = RuleBasedSpeculator::default();
        let wrapper = StreamingSpeculatorWrapper::new(speculator).with_chunk_size(10);

        let draft = Draft::new("Paris is the capital of France.", "What is the capital?")
            .with_confidence(0.9);

        let context = create_context();
        let mut streaming = wrapper
            .verify_draft_streaming(&draft, &context)
            .await
            .unwrap();

        // Consume some chunks
        let _ = streaming.next_chunk().await;
        let _ = streaming.next_chunk().await;

        assert!(streaming.collected_chunks().len() >= 2);
    }

    #[tokio::test]
    async fn test_verification_chunk_default() {
        let chunk = VerificationChunk::default();
        assert_eq!(chunk.chunk_id, 0);
        assert!(chunk.content.is_empty());
        assert!(!chunk.is_final);
        assert!(chunk.partial_decision.is_none());
        assert_eq!(chunk.partial_confidence, 0.0);
    }
}
