//! Long-term memory stream — generative-agents-style importance + recency + relevance retrieval.
pub mod retrieval;
pub mod store;
#[cfg(test)]
mod tests;
pub mod types;
pub use retrieval::MemoryRetriever;
pub use store::LongTermMemoryStore;
pub use types::{
    HeuristicImportanceScorer, ImportanceScorer, LongTermMemoryConfig, LongTermMemoryError,
    MemoryKind, MemoryQuery, MemoryRecord, RetrievedMemory,
};
