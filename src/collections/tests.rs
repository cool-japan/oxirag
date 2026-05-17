//! Tests for the `collections` module.

use super::index::CollectionIndex;
use super::store::{CollectionStore, InMemoryCollectionStore};
use super::types::{
    CollectionConfig, CollectionError, CollectionId, CollectionMetadata, SimilarityMetric,
};
use crate::types::Document;

// ── CollectionId normalisation ────────────────────────────────────────────────

#[test]
fn test_collection_id_lowercase() {
    let id = CollectionId::new("MyCollection").expect("valid name");
    assert_eq!(id.as_str(), "mycollection");
}

#[test]
fn test_collection_id_spaces_to_underscores() {
    let id = CollectionId::new("my collection name").expect("valid name");
    assert_eq!(id.as_str(), "my_collection_name");
}

#[test]
fn test_collection_id_hyphens_to_underscores() {
    let id = CollectionId::new("my-collection-name").expect("valid name");
    assert_eq!(id.as_str(), "my_collection_name");
}

#[test]
fn test_collection_id_mixed() {
    let id = CollectionId::new("My-Collection Name").expect("valid name");
    assert_eq!(id.as_str(), "my_collection_name");
}

#[test]
fn test_collection_id_already_valid() {
    let id = CollectionId::new("valid_name_123").expect("valid name");
    assert_eq!(id.as_str(), "valid_name_123");
}

#[test]
fn test_collection_id_empty_rejected() {
    let err = CollectionId::new("").expect_err("empty name must fail");
    assert!(matches!(err, CollectionError::InvalidName(_)));
}

#[test]
fn test_collection_id_invalid_char_rejected() {
    // '@' is invalid after normalisation.
    let err = CollectionId::new("my@collection").expect_err("invalid char must fail");
    assert!(matches!(err, CollectionError::InvalidName(_)));
}

#[test]
fn test_collection_id_dot_rejected() {
    let err = CollectionId::new("my.collection").expect_err("dot must fail");
    assert!(matches!(err, CollectionError::InvalidName(_)));
}

#[test]
fn test_collection_id_display() {
    let id = CollectionId::new("hello world").expect("valid name");
    assert_eq!(id.to_string(), "hello_world");
}

// ── InMemoryCollectionStore CRUD ──────────────────────────────────────────────

#[tokio::test]
async fn test_store_create_and_get() {
    let store = InMemoryCollectionStore::new();
    let id = CollectionId::new("test").expect("valid name");
    let config = CollectionConfig::new(64);
    let meta = CollectionMetadata::default();

    let collection = store
        .create(id.clone(), config, meta)
        .await
        .expect("create should succeed");
    assert_eq!(collection.id().as_str(), "test");
    assert_eq!(collection.config().embedding_dimension, 64);

    let fetched = store.get(&id).await.expect("get should succeed");
    assert_eq!(fetched.id().as_str(), "test");
}

#[tokio::test]
async fn test_store_create_duplicate_rejected() {
    let store = InMemoryCollectionStore::new();
    let id = CollectionId::new("dup").expect("valid name");
    let config = CollectionConfig::new(64);

    store
        .create(id.clone(), config.clone(), CollectionMetadata::default())
        .await
        .expect("first create");
    let err = store
        .create(id, config, CollectionMetadata::default())
        .await
        .expect_err("duplicate must fail");
    assert!(matches!(err, CollectionError::AlreadyExists(_)));
}

#[tokio::test]
async fn test_store_capacity_limit() {
    let store = InMemoryCollectionStore::with_max_collections(2);

    for i in 0..2_usize {
        let id = CollectionId::new(&format!("col{i}")).expect("valid name");
        store
            .create(id, CollectionConfig::new(4), CollectionMetadata::default())
            .await
            .expect("should succeed");
    }

    let extra = CollectionId::new("extra").expect("valid name");
    let err = store
        .create(
            extra,
            CollectionConfig::new(4),
            CollectionMetadata::default(),
        )
        .await
        .expect_err("capacity exceeded");
    assert!(matches!(err, CollectionError::CapacityExceeded(_)));
}

#[tokio::test]
async fn test_store_delete() {
    let store = InMemoryCollectionStore::new();
    let id = CollectionId::new("to_delete").expect("valid name");
    store
        .create(
            id.clone(),
            CollectionConfig::new(4),
            CollectionMetadata::default(),
        )
        .await
        .expect("create");

    store.delete(&id).await.expect("delete should succeed");
    let err = store.get(&id).await.expect_err("should be gone");
    assert!(matches!(err, CollectionError::NotFound(_)));
}

#[tokio::test]
async fn test_store_delete_nonexistent() {
    let store = InMemoryCollectionStore::new();
    let id = CollectionId::new("ghost").expect("valid name");
    let err = store.delete(&id).await.expect_err("should fail");
    assert!(matches!(err, CollectionError::NotFound(_)));
}

#[tokio::test]
async fn test_store_list() {
    let store = InMemoryCollectionStore::new();
    assert_eq!(store.list().await.len(), 0);

    for i in 0..3_usize {
        let id = CollectionId::new(&format!("col{i}")).expect("valid name");
        store
            .create(id, CollectionConfig::new(4), CollectionMetadata::default())
            .await
            .expect("create");
    }

    assert_eq!(store.list().await.len(), 3);
}

#[tokio::test]
async fn test_store_collection_count() {
    let store = InMemoryCollectionStore::new();
    assert_eq!(store.collection_count().await, 0);

    let id = CollectionId::new("one").expect("valid name");
    store
        .create(id, CollectionConfig::new(4), CollectionMetadata::default())
        .await
        .expect("create");
    assert_eq!(store.collection_count().await, 1);
}

#[tokio::test]
async fn test_store_update_metadata() {
    let store = InMemoryCollectionStore::new();
    let id = CollectionId::new("meta_test").expect("valid name");
    store
        .create(
            id.clone(),
            CollectionConfig::new(4),
            CollectionMetadata::default(),
        )
        .await
        .expect("create");

    let new_meta = CollectionMetadata::new(
        Some("Updated description".to_string()),
        vec!["tag1".to_string()],
    );
    store
        .update_metadata(&id, new_meta)
        .await
        .expect("update_metadata should succeed");

    let col = store.get(&id).await.expect("get");
    assert_eq!(
        col.metadata().description.as_deref(),
        Some("Updated description")
    );
    assert_eq!(col.metadata().tags, vec!["tag1".to_string()]);
}

// ── CollectionIndex indexing & search ─────────────────────────────────────────

#[tokio::test]
async fn test_index_create_and_index_document() {
    let index = CollectionIndex::new(InMemoryCollectionStore::new());
    let id = CollectionId::new("books").expect("valid name");

    index
        .create_collection(
            id.clone(),
            CollectionConfig::new(4),
            CollectionMetadata::default(),
        )
        .await
        .expect("create");

    let doc = Document::new("Rust programming guide");
    let embedding = vec![1.0_f32, 0.0, 0.0, 0.0];
    index
        .index_document(&id, doc, embedding)
        .await
        .expect("index_document");

    let count = index.document_count(&id).await.expect("count");
    assert_eq!(count, 1);
}

#[tokio::test]
async fn test_index_search_returns_results() {
    let index = CollectionIndex::new(InMemoryCollectionStore::new());
    let id = CollectionId::new("search_test").expect("valid name");

    index
        .create_collection(
            id.clone(),
            CollectionConfig::new(4),
            CollectionMetadata::default(),
        )
        .await
        .expect("create");

    for i in 0..5_usize {
        #[allow(clippy::cast_precision_loss)]
        let emb = vec![1.0_f32 - i as f32 * 0.2, i as f32 * 0.2, 0.0, 0.0];
        index
            .index_document(&id, Document::new(format!("doc {i}")), emb)
            .await
            .expect("index");
    }

    let query = vec![1.0_f32, 0.0, 0.0, 0.0];
    let results = index.search(&id, &query, 3).await.expect("search");
    assert!(!results.is_empty());
    assert!(results.len() <= 3);
}

#[tokio::test]
async fn test_index_dimension_mismatch_on_insert() {
    let index = CollectionIndex::new(InMemoryCollectionStore::new());
    let id = CollectionId::new("dim_test").expect("valid name");

    index
        .create_collection(
            id.clone(),
            CollectionConfig::new(4),
            CollectionMetadata::default(),
        )
        .await
        .expect("create");

    let doc = Document::new("wrong dim");
    let bad_emb = vec![1.0_f32, 0.0]; // 2-dim, not 4
    let err = index
        .index_document(&id, doc, bad_emb)
        .await
        .expect_err("must fail");
    assert!(matches!(
        err,
        CollectionError::DimensionMismatch {
            expected: 4,
            got: 2
        }
    ));
}

#[tokio::test]
async fn test_index_dimension_mismatch_on_search() {
    let index = CollectionIndex::new(InMemoryCollectionStore::new());
    let id = CollectionId::new("dim_search").expect("valid name");

    index
        .create_collection(
            id.clone(),
            CollectionConfig::new(4),
            CollectionMetadata::default(),
        )
        .await
        .expect("create");

    let bad_query = vec![1.0_f32, 0.0]; // 2-dim
    let err = index
        .search(&id, &bad_query, 5)
        .await
        .expect_err("must fail");
    assert!(matches!(
        err,
        CollectionError::DimensionMismatch {
            expected: 4,
            got: 2
        }
    ));
}

// ── Cross-collection search with RRF ──────────────────────────────────────────

#[tokio::test]
async fn test_cross_collection_search_rrf_fusion() {
    let index = CollectionIndex::new(InMemoryCollectionStore::new());
    let dim = 4_usize;

    // Create three collections.
    let cids: Vec<CollectionId> = (0..3_usize)
        .map(|i| CollectionId::new(&format!("col{i}")).expect("valid name"))
        .collect();

    for cid in &cids {
        index
            .create_collection(
                cid.clone(),
                CollectionConfig::new(dim),
                CollectionMetadata::default(),
            )
            .await
            .expect("create");
    }

    // Index one unique document per collection.
    let embeddings: [Vec<f32>; 3] = [
        vec![1.0_f32, 0.0, 0.0, 0.0],
        vec![0.9_f32, 0.1, 0.0, 0.0],
        vec![0.8_f32, 0.2, 0.0, 0.0],
    ];

    for (i, (cid, emb)) in cids.iter().zip(embeddings.iter()).enumerate() {
        index
            .index_document(cid, Document::new(format!("doc from col {i}")), emb.clone())
            .await
            .expect("index");
    }

    let query = vec![1.0_f32, 0.0, 0.0, 0.0];
    let results = index
        .cross_collection_search(&cids, &query, 2)
        .await
        .expect("cross search");

    assert!(!results.is_empty());
    // Results should have collection_id set.
    assert!(results.iter().all(|r| !r.collection_id.is_empty()));
    // Ranks should be assigned.
    assert!(results.iter().enumerate().all(|(i, r)| r.rank == i));
}

#[tokio::test]
async fn test_cross_collection_search_empty_collections() {
    let index = CollectionIndex::new(InMemoryCollectionStore::new());
    let results = index
        .cross_collection_search(&[], &[1.0_f32, 0.0], 5)
        .await
        .expect("empty cross search");
    assert!(results.is_empty());
}

#[tokio::test]
async fn test_cross_collection_search_not_found() {
    let index = CollectionIndex::new(InMemoryCollectionStore::new());
    let ghost = CollectionId::new("ghost").expect("valid name");
    let err = index
        .cross_collection_search(&[ghost], &[1.0_f32, 0.0], 5)
        .await
        .expect_err("must fail");
    assert!(matches!(err, CollectionError::NotFound(_)));
}

// ── search_all ────────────────────────────────────────────────────────────────

#[tokio::test]
async fn test_search_all_empty() {
    let index = CollectionIndex::new(InMemoryCollectionStore::new());
    let results = index
        .search_all(&[1.0_f32, 0.0], 5)
        .await
        .expect("search_all");
    assert!(results.is_empty());
}

#[tokio::test]
async fn test_search_all_multiple_collections() {
    let index = CollectionIndex::new(InMemoryCollectionStore::new());
    let dim = 2_usize;

    for i in 0..3_usize {
        let cid = CollectionId::new(&format!("all{i}")).expect("valid name");
        index
            .create_collection(
                cid.clone(),
                CollectionConfig::new(dim),
                CollectionMetadata::default(),
            )
            .await
            .expect("create");
        index
            .index_document(
                &cid,
                Document::new(format!("all doc {i}")),
                vec![1.0_f32, 0.0],
            )
            .await
            .expect("index");
    }

    let results = index
        .search_all(&[1.0_f32, 0.0], 2)
        .await
        .expect("search_all");
    // Expect at most top_k * n_collections results.
    assert!(results.len() <= 2 * 3);
    assert!(!results.is_empty());
}

// ── CollectionConfig defaults ─────────────────────────────────────────────────

#[test]
fn test_config_defaults() {
    let cfg = CollectionConfig::default();
    assert_eq!(cfg.default_top_k, 10);
    assert_eq!(cfg.similarity_metric, SimilarityMetric::Cosine);
    assert!(cfg.max_documents.is_none());
}

#[test]
fn test_config_builder() {
    let cfg = CollectionConfig::new(128)
        .with_max_documents(1000)
        .with_default_top_k(5)
        .with_similarity_metric(SimilarityMetric::Euclidean);

    assert_eq!(cfg.embedding_dimension, 128);
    assert_eq!(cfg.max_documents, Some(1000));
    assert_eq!(cfg.default_top_k, 5);
    assert_eq!(cfg.similarity_metric, SimilarityMetric::Euclidean);
}

// ── CollectionStats running average ──────────────────────────────────────────

#[test]
fn test_stats_record_search() {
    let id = CollectionId::new("stat_test").expect("valid name");
    let mut stats = super::types::CollectionStats::new(id);

    stats.record_search(10.0);
    stats.record_search(20.0);
    stats.record_search(30.0);

    assert_eq!(stats.total_searches, 3);
    assert!((stats.avg_search_latency_ms - 20.0).abs() < 1e-3);
}
