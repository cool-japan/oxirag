//! Core types for the streaming pipeline API.

use serde::{Deserialize, Serialize};

use crate::types::SpeculationDecision;

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

/// Helper function to truncate content for display.
pub(super) fn truncate_content(content: &str, max_len: usize) -> String {
    if content.len() <= max_len {
        content.to_string()
    } else {
        format!("{}...", &content[..max_len.saturating_sub(3)])
    }
}
