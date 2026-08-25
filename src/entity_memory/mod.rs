//! Entity memory — tracks per-entity knowledge across conversation turns.
pub mod store;
#[cfg(all(test, not(target_arch = "wasm32")))]
mod tests;
pub mod tracker;
pub mod types;
pub use types::{
    EntityCategory, EntityKnowledge, EntityMemoryConfig, EntityMemoryError, EntityMemoryStore,
    EntityMentionExtractor, EntityMentionSpan, HeuristicEntityMentionExtractor,
};
