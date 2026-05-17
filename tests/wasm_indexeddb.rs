//! WASM integration tests for the IndexedDB vector store backend.
//!
//! These tests run in a real browser via `wasm-pack test --chrome --headless`.
//! They are compiled only when `target_arch = "wasm32"`.

#![cfg(target_arch = "wasm32")]

use wasm_bindgen_test::*;

wasm_bindgen_test_configure!(run_in_browser);

#[cfg(all(target_arch = "wasm32", feature = "wasm-indexeddb"))]
mod indexeddb_tests {
    use super::*;
    use oxirag::layer1_echo::storage::IndexedDbVectorStore;
    use oxirag::layer1_echo::traits::{IndexedDocument, VectorStore};
    use oxirag::types::Document;

    /// A freshly created store should report zero documents.
    #[wasm_bindgen_test]
    async fn test_indexeddb_empty_on_create() {
        let mut store = IndexedDbVectorStore::new(4);
        // Clear any leftover state from previous test runs.
        store.clear().await.expect("clear should succeed");
        assert_eq!(store.count().await, 0, "store should be empty after clear");
    }

    /// Inserting one document should increment the count to 1.
    #[wasm_bindgen_test]
    async fn test_indexeddb_insert_and_count() {
        let mut store = IndexedDbVectorStore::new(4);
        store.clear().await.expect("clear should succeed");

        let doc = Document::new("test content");
        let embedding = vec![1.0_f32, 0.0, 0.0, 0.0];
        let indexed = IndexedDocument::new(doc, embedding);

        store.insert(indexed).await.expect("insert should succeed");
        assert_eq!(store.count().await, 1, "count should be 1 after one insert");
    }

    /// `dimension()` must reflect the value supplied at construction.
    #[wasm_bindgen_test]
    async fn test_indexeddb_dimension() {
        let store = IndexedDbVectorStore::new(64);
        assert_eq!(store.dimension(), 64);

        let store2 = IndexedDbVectorStore::new(384);
        assert_eq!(store2.dimension(), 384);
    }

    /// Searching in an empty store should return an empty slice, not an error.
    #[wasm_bindgen_test]
    async fn test_indexeddb_search_empty_store() {
        let mut store = IndexedDbVectorStore::new(4);
        store.clear().await.expect("clear should succeed");

        let results = store
            .search(&[1.0, 0.0, 0.0, 0.0], 5, None)
            .await
            .expect("search on empty store should succeed");

        assert!(results.is_empty(), "empty store should return no results");
    }

    /// Insert several documents and verify the retrieved count.
    #[wasm_bindgen_test]
    async fn test_indexeddb_insert_multiple() {
        let mut store = IndexedDbVectorStore::new(4);
        store.clear().await.expect("clear should succeed");

        for i in 0..3_usize {
            let doc = Document::new(format!("document {i}"));
            let embedding: Vec<f32> = (0..4).map(|j| if j == i % 4 { 1.0 } else { 0.0 }).collect();
            let indexed = IndexedDocument::new(doc, embedding);
            store.insert(indexed).await.expect("insert should succeed");
        }

        assert_eq!(store.count().await, 3, "expected 3 documents");
    }

    /// Clearing a populated store should bring the count back to zero.
    #[wasm_bindgen_test]
    async fn test_indexeddb_clear_resets_count() {
        let mut store = IndexedDbVectorStore::new(4);

        // Ensure there is at least one document before clearing.
        let doc = Document::new("to be cleared");
        let indexed = IndexedDocument::new(doc, vec![1.0, 0.0, 0.0, 0.0]);
        store.insert(indexed).await.expect("insert should succeed");

        store.clear().await.expect("clear should succeed");
        assert_eq!(store.count().await, 0, "count should be 0 after clear");
    }
}
