//! Entity memory — tracks per-entity knowledge across conversation turns.
pub mod store;
#[cfg(test)]
mod tests;
pub mod tracker;
pub mod types;
pub use types::{
    EntityCategory, EntityKnowledge, EntityMemoryConfig, EntityMemoryError, EntityMemoryStore,
    EntityMentionExtractor, EntityMentionSpan, HeuristicEntityMentionExtractor,
};
