//! Layer 2: Speculator - Draft verification using small language models.
//!
//! The Speculator layer provides:
//! - Draft answer verification against retrieved context
//! - Decision making: Accept, Revise, or Reject
//! - Draft revision when needed
//! - Streaming verification support
//! - Confidence calibration
//! - Multi-stage verification pipelines
//! - SLM (Small Language Model) interfaces
//! - Hidden state-based speculative decoding

pub mod calibration;
pub mod candle_slm;
pub mod hidden_state_speculator;
pub mod hidden_states;
pub mod slm;
pub mod speculative;
pub mod streaming;
pub mod traits;
pub mod verification;

// Re-export main types from traits
pub use traits::{RuleBasedSpeculator, Speculator, SpeculatorConfig, prompts};

// Re-export candle types
pub use candle_slm::{CandleSlmConfig, CandleSlmDevice, MockSlmSpeculator};

#[cfg(feature = "speculator")]
pub use candle_slm::CandleSLM;

// Re-export streaming types
#[cfg(feature = "native")]
pub use streaming::{
    StreamingSpeculator, StreamingSpeculatorWrapper, StreamingVerification, VerificationChunk,
};

#[cfg(not(feature = "native"))]
pub use streaming::{StreamingVerification, VerificationChunk};

// Re-export calibration types
pub use calibration::{CalibrationMethod, CalibrationStats, ConfidenceCalibrator};

// Re-export SLM types
pub use slm::{FinishReason, GenerationOutput, MockSlm, SlmBuilder, SlmConfig, SmallLanguageModel};

// Re-export verification pipeline types
pub use verification::{
    AggregationMethod, FactualConsistencyStage, KeywordMatchStage, PipelineBuilder,
    SemanticSimilarityStage, StageResult, VerificationPipeline, VerificationStage,
};

// Re-export hidden states types
pub use hidden_states::{
    HiddenStateCache, HiddenStateCacheConfig, HiddenStateProvider, LayerHiddenState,
    MockHiddenStateProvider, ModelHiddenStates, ModelKVCache, PrefixReuseStrategy,
    StateReuseStrategy,
};

// Re-export speculative decoding types
pub use speculative::{
    MockSpeculativeDecoder, SpeculativeDecoder, SpeculativeDecodingConfig, SpeculativeOutput,
    SpeculativeStats, SpeculativeStep, TokenWithProb,
    VerificationResult as SpeculativeVerificationResult,
};

// Re-export hidden state speculator types
pub use hidden_state_speculator::{
    DivergencePoint, HiddenStateSpeculator, HiddenStateSpeculatorConfig, MockHiddenStateSpeculator,
};

#[cfg(feature = "speculator")]
pub use candle_slm::CandleSlmSpeculator;
