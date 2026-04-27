//! Storage implementations for the Echo layer.

mod memory;
#[cfg(feature = "echo-redb")]
mod redb;

pub use memory::InMemoryVectorStore;
#[cfg(feature = "echo-redb")]
pub use redb::RedbVectorStore;
