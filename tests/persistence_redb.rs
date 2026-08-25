//! Integration tests for redb-backed persistent storage backends.
//!
//! These tests write data, drop the DB, reopen it, and assert recovery.
//! Uses tempfile::TempDir for automatic cleanup.

#![cfg(not(target_arch = "wasm32"))]

#[cfg(feature = "full")]
mod persistence_tests {
    use tempfile::TempDir;

    use oxirag::layer1_echo::storage::RedbVectorStore;
    use oxirag::layer1_echo::traits::{IndexedDocument, VectorStore};
    use oxirag::layer4_graph::traits::GraphStore;
    use oxirag::layer4_graph::types::RelationshipType;
    use oxirag::layer4_graph::{EntityType, GraphEntity, GraphRelationship, RedbGraphStore};
    use oxirag::prefix_cache::traits::PrefixCacheStore;
    use oxirag::prefix_cache::{
        ContextFingerprint, KVCacheEntry, PrefixCacheConfig, RedbPrefixCache,
    };
    use oxirag::types::{Document, DocumentId};

    // ─── vector store persistence ────────────────────────────────────────────

    /// Insert 5 documents into `RedbVectorStore`, drop it, reopen from the same
    /// path, and assert the 5 documents are still present (`count()` returns 5).
    #[tokio::test]
    async fn test_redb_vector_store_persistence() {
        let dir = TempDir::new().expect("temp dir should be created");
        let db_path = dir.path().join("vector_store.redb");

        const DIM: usize = 32;

        // Write phase — open, insert, close.
        {
            let mut store =
                RedbVectorStore::new(&db_path, DIM).expect("redb vector store should open");

            for i in 0_u32..5 {
                let doc = Document::new(format!("Document {i}"))
                    .with_id(DocumentId::from_string(format!("doc-{i}")));
                let embedding: Vec<f32> = (0..DIM).map(|j| (i as f32 + j as f32) / 100.0).collect();
                let norm: f32 = embedding.iter().map(|x| x * x).sum::<f32>().sqrt();
                let embedding: Vec<f32> = embedding.iter().map(|x| x / norm.max(1e-9)).collect();
                let indexed = IndexedDocument::new(doc, embedding);
                store.insert(indexed).await.expect("insert should succeed");
            }

            assert_eq!(store.count().await, 5, "store should hold 5 documents");
            // `store` drops here — database file is flushed.
        }

        // Read phase — reopen from same path and verify count.
        {
            let store =
                RedbVectorStore::new(&db_path, DIM).expect("redb vector store should reopen");
            let count = store.count().await;
            assert_eq!(count, 5, "reopened store should still hold 5 documents");
        }
    }

    // ─── graph store persistence ──────────────────────────────────────────────

    /// Insert an entity + relationship into `RedbGraphStore`, drop it, reopen,
    /// and assert the entity can still be retrieved.
    #[tokio::test]
    async fn test_redb_graph_store_persistence() {
        let dir = TempDir::new().expect("temp dir should be created");
        let db_path = dir.path().join("graph_store.redb");

        let entity_id: String;

        // Write phase.
        {
            let mut store = RedbGraphStore::new(&db_path).expect("redb graph store should open");

            let entity = GraphEntity::new("Rust", EntityType::Technology).with_id("rust-lang");
            entity_id = store
                .add_entity(entity)
                .await
                .expect("add_entity should succeed");

            let target = GraphEntity::new("LLVM", EntityType::Technology).with_id("llvm-project");
            store
                .add_entity(target)
                .await
                .expect("add second entity should succeed");

            let rel =
                GraphRelationship::new(entity_id.clone(), "llvm-project", RelationshipType::Uses);
            store
                .add_relationship(rel)
                .await
                .expect("add_relationship should succeed");
        }

        // Read phase.
        {
            let store = RedbGraphStore::new(&db_path).expect("redb graph store should reopen");

            let found = store
                .get_entity(&entity_id)
                .await
                .expect("get_entity should not error");

            assert!(
                found.is_some(),
                "entity '{entity_id}' should still be present after reopen"
            );

            let entity = found.expect("entity is some");
            assert_eq!(entity.name, "Rust");
        }
    }

    // ─── prefix cache persistence ─────────────────────────────────────────────

    /// Put a `KVCacheEntry`, drop the cache, reopen, and assert `get()` returns
    /// `Some`.
    #[tokio::test]
    async fn test_redb_prefix_cache_persistence() {
        let dir = TempDir::new().expect("temp dir should be created");
        let db_path = dir.path().join("prefix_cache.redb");

        let fingerprint = ContextFingerprint::new(0xDEAD_BEEF, 128, "persistent test context");

        // Write phase.
        {
            let mut cache = RedbPrefixCache::new(&db_path, PrefixCacheConfig::default())
                .expect("redb prefix cache should open");

            let entry = KVCacheEntry::new(
                "persistent-key",
                fingerprint.clone(),
                vec![0.1_f32; 64],
                128,
            );
            cache.put(entry).await.expect("put should succeed");
        }

        // Read phase.
        {
            let cache = RedbPrefixCache::new(&db_path, PrefixCacheConfig::default())
                .expect("redb prefix cache should reopen");

            let hit = cache.get(&fingerprint).await;
            assert!(hit.is_some(), "cache entry should survive drop and reopen");
        }
    }
}
