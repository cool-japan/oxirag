//! `ProgressReporter` for pipeline stages (native/tokio only).

#[cfg(feature = "native")]
use crate::time::Instant;

#[cfg(feature = "native")]
use tokio::sync::mpsc;

#[cfg(feature = "native")]
use crate::types::SpeculationDecision;

#[cfg(feature = "native")]
use super::types::{ChunkType, PipelineChunk};

/// Helper to get timestamp in milliseconds for the progress reporter.
#[cfg(feature = "native")]
#[inline]
fn elapsed_ms_since(start: &Instant) -> u64 {
    #[allow(clippy::cast_possible_truncation)]
    {
        start.elapsed().as_millis().min(u128::from(u64::MAX)) as u64
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
        elapsed_ms_since(&self.start_time)
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
