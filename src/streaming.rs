//! Async streaming pipeline API.

pub mod progress;
pub mod types;
pub mod wrapper;

#[cfg(feature = "native")]
pub use progress::ProgressReporter;
pub use types::{ChunkMetadata, ChunkType, PipelineChunk};
pub use wrapper::{StreamingPipeline, StreamingPipelineResult, StreamingPipelineWrapper};

#[cfg(all(test, not(target_arch = "wasm32")))]
mod tests;
