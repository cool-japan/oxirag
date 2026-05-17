//! Storage implementations for the Echo layer.

#[cfg(all(target_arch = "wasm32", feature = "wasm-indexeddb"))]
pub mod indexeddb;
mod memory;
#[cfg(feature = "echo-redb")]
mod redb;

#[cfg(all(target_arch = "wasm32", feature = "wasm-indexeddb"))]
pub use indexeddb::IndexedDbVectorStore;
pub use memory::InMemoryVectorStore;
#[cfg(feature = "echo-redb")]
pub use redb::RedbVectorStore;
