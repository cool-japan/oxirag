//! Composable RAG pipeline stages.
//!
//! Provides a synchronous, linear pipeline of [`PipelineStage`] processors
//! that thread context text through each stage in order.  Stages may block
//! the pipeline on policy violations or transform/filter the context.

pub mod composer;
pub mod stage;
#[cfg(all(test, not(target_arch = "wasm32")))]
mod tests;
pub mod types;

pub use composer::ComposedPipeline;
#[cfg(feature = "guardrails")]
pub use stage::GuardrailStage;
pub use stage::{FormatStage, KeywordFilterStage, PassThroughStage, SanitizerStage, TruncateStage};
pub use types::{ComposerError, PipelineStage, StageInput, StageOutput};
