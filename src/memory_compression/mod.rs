//! Hierarchical memory compression — recursive turn summarization.
pub mod compressor;
pub mod hierarchy;
#[cfg(test)]
mod tests;
pub mod types;
pub use hierarchy::HierarchicalMemory;
pub use types::{
    CompressedBlock, CompressionStats, ExtractiveTurnCompressor, MemoryCompressionConfig,
    MemoryCompressionError, MemoryTurn, TurnCompressor,
};
