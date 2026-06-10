//! Contextual Retrieval — prepend doc-level context to each chunk before indexing.
pub mod contextualizer;
pub mod index;
#[cfg(test)]
mod tests;
pub mod types;
pub use types::{
    ChunkContext, ChunkNeighborhood, ContextualChunk, ContextualConfig, ContextualIndexBuilder,
    ContextualRetrievalError, Contextualizer, ExtractiveContextualizer,
};
