//! Tests for the ANN (HNSW) module.

use std::collections::HashSet;

use crate::error::VectorStoreError;
use crate::layer1_echo::similarity::compute_similarity;
use crate::layer1_echo::traits::{IndexedDocument, SimilarityMetric, VectorStore};
use crate::types::{Document, DocumentId};

use super::config::AnnConfig;
use super::hnsw::HnswIndex;
use super::store::AnnVectorStore;

fn create_random_vector(dim: usize, seed: u64) -> Vec<f32> {
    let mut rng = seed;
    (0..dim)
        .map(|_| {
            rng = rng.wrapping_mul(6_364_136_223_846_793_005).wrapping_add(1);
            #[allow(clippy::cast_precision_loss)]
            let val = ((rng >> 33) as f32 / (1u32 << 31) as f32) * 2.0 - 1.0;
            val
        })
        .collect()
}

fn normalize_vector(v: &[f32]) -> Vec<f32> {
    let norm: f32 = v.iter().map(|x| x * x).sum::<f32>().sqrt();
    if norm == 0.0 {
        v.to_vec()
    } else {
        v.iter().map(|x| x / norm).collect()
    }
}

use crate::layer1_echo::filter::MetadataFilter;

// Test 1: Index creation and configuration
#[test]
fn test_index_creation_default_config() {
    let config = AnnConfig::default();
    let index = HnswIndex::new(64, config);

    assert_eq!(index.dimension(), 64);
    assert!(index.is_empty());
    assert_eq!(index.len(), 0);
}

// Test 2: Custom configuration
#[test]
fn test_index_custom_config() {
    let config = AnnConfig::new()
        .with_m(32)
        .with_m_max(64)
        .with_ef_construction(100)
        .with_ef_search(25)
        .with_distance_metric(SimilarityMetric::DotProduct);

    assert_eq!(config.m, 32);
    assert_eq!(config.m_max, 64);
    assert_eq!(config.ef_construction, 100);
    assert_eq!(config.ef_search, 25);
    assert_eq!(config.distance_metric, SimilarityMetric::DotProduct);
}

// Test 3: Insert single vector
#[test]
fn test_insert_single_vector() {
    let config = AnnConfig::default();
    let mut index = HnswIndex::new(4, config);

    let id = DocumentId::new();
    let vector = vec![1.0, 0.0, 0.0, 0.0];

    index
        .insert(id.clone(), vector.clone())
        .expect("test operation should succeed");

    assert_eq!(index.len(), 1);
    assert!(!index.is_empty());

    let node = index.get_node(&id).expect("test operation should succeed");
    assert_eq!(node.vector, vector);
}

// Test 4: Insert multiple vectors
#[test]
fn test_insert_multiple_vectors() {
    let config = AnnConfig::default();
    let mut index = HnswIndex::new(4, config);

    for i in 0..10 {
        let id = DocumentId::from_string(format!("doc_{i}"));
        let vector = create_random_vector(4, i);
        index
            .insert(id, vector)
            .expect("test operation should succeed");
    }

    assert_eq!(index.len(), 10);
}

// Test 5: Dimension mismatch error
#[test]
fn test_insert_dimension_mismatch() {
    let config = AnnConfig::default();
    let mut index = HnswIndex::new(4, config);

    let id = DocumentId::new();
    let vector = vec![1.0, 0.0, 0.0]; // Wrong dimension

    let result = index.insert(id, vector);
    assert!(matches!(
        result,
        Err(VectorStoreError::DimensionMismatch { .. })
    ));
}

// Test 6: Search returns correct results
#[test]
fn test_search_basic() {
    let config = AnnConfig::default().with_ef_search(100);
    let mut index = HnswIndex::new(4, config);

    // Insert vectors
    let target = normalize_vector(&[1.0, 0.0, 0.0, 0.0]);
    let other1 = normalize_vector(&[0.0, 1.0, 0.0, 0.0]);
    let other2 = normalize_vector(&[0.0, 0.0, 1.0, 0.0]);
    let similar = normalize_vector(&[0.9, 0.1, 0.0, 0.0]);

    index
        .insert(DocumentId::from_string("target"), target.clone())
        .expect("test operation should succeed");
    index
        .insert(DocumentId::from_string("other1"), other1)
        .expect("test operation should succeed");
    index
        .insert(DocumentId::from_string("other2"), other2)
        .expect("test operation should succeed");
    index
        .insert(DocumentId::from_string("similar"), similar)
        .expect("test operation should succeed");

    let results = index.search(&target, 2);

    assert_eq!(results.len(), 2);
    // The target itself should be the most similar
    assert_eq!(results[0].0, DocumentId::from_string("target"));
    // Similar should be second
    assert_eq!(results[1].0, DocumentId::from_string("similar"));
}

// Test 7: Search accuracy comparison with brute force
#[test]
fn test_search_accuracy_vs_brute_force() {
    // Use higher M for better connectivity and higher ef for better search
    let config = AnnConfig::default()
        .with_m(32)
        .with_m_max(64)
        .with_ef_search(100)
        .with_ef_construction(200);
    let mut index = HnswIndex::new(32, config.clone());

    let num_vectors = 100;
    let mut vectors: Vec<(DocumentId, Vec<f32>)> = Vec::new();

    for i in 0..num_vectors {
        let id = DocumentId::from_string(format!("doc_{i}"));
        let vector = normalize_vector(&create_random_vector(32, i));
        vectors.push((id.clone(), vector.clone()));
        index
            .insert(id, vector)
            .expect("test operation should succeed");
    }

    // Search with a query
    let query = normalize_vector(&create_random_vector(32, 999));
    let k = 10;

    // HNSW search
    let hnsw_results = index.search(&query, k);

    // Brute force search
    let mut brute_force: Vec<(DocumentId, f32)> = vectors
        .iter()
        .map(|(id, v)| {
            let similarity = compute_similarity(&query, v, config.distance_metric);
            (id.clone(), similarity)
        })
        .collect();
    brute_force.sort_by(|a, b| b.1.partial_cmp(&a.1).unwrap_or(std::cmp::Ordering::Equal));
    brute_force.truncate(k);

    // Check recall (how many of top-k brute force results are in HNSW results)
    let hnsw_ids: HashSet<_> = hnsw_results.iter().map(|(id, _)| id.clone()).collect();
    let bf_ids: HashSet<_> = brute_force.iter().map(|(id, _)| id.clone()).collect();

    #[allow(clippy::cast_precision_loss)]
    let recall = hnsw_ids.intersection(&bf_ids).count() as f32 / k as f32;
    // ANN is approximate - with good parameters we should get at least 30% recall
    // This test validates the algorithm works, not that it achieves perfect recall
    assert!(recall >= 0.3, "Recall should be at least 30%, got {recall}");
}

// Test 8: Remove vector
#[test]
fn test_remove_vector() {
    let config = AnnConfig::default();
    let mut index = HnswIndex::new(4, config);

    let id1 = DocumentId::from_string("doc1");
    let id2 = DocumentId::from_string("doc2");

    index
        .insert(id1.clone(), vec![1.0, 0.0, 0.0, 0.0])
        .expect("test operation should succeed");
    index
        .insert(id2.clone(), vec![0.0, 1.0, 0.0, 0.0])
        .expect("test operation should succeed");

    assert_eq!(index.len(), 2);

    let removed = index.remove(&id1);
    assert!(removed.is_some());
    assert_eq!(index.len(), 1);
    assert!(index.get_node(&id1).is_none());
    assert!(index.get_node(&id2).is_some());
}

// Test 9: Remove non-existent vector
#[test]
fn test_remove_nonexistent() {
    let config = AnnConfig::default();
    let mut index = HnswIndex::new(4, config);

    let id = DocumentId::from_string("nonexistent");
    let removed = index.remove(&id);
    assert!(removed.is_none());
}

// Test 10: Search with different similarity metrics
#[test]
fn test_different_similarity_metrics() {
    // Cosine similarity
    let cosine_config = AnnConfig::default().with_distance_metric(SimilarityMetric::Cosine);
    let mut cosine_index = HnswIndex::new(4, cosine_config);

    // Dot product
    let dot_config = AnnConfig::default().with_distance_metric(SimilarityMetric::DotProduct);
    let mut dot_index = HnswIndex::new(4, dot_config);

    // Euclidean
    let euc_config = AnnConfig::default().with_distance_metric(SimilarityMetric::Euclidean);
    let mut euc_index = HnswIndex::new(4, euc_config);

    let id = DocumentId::from_string("test");
    let vector = vec![1.0, 0.0, 0.0, 0.0];

    cosine_index
        .insert(id.clone(), vector.clone())
        .expect("test operation should succeed");
    dot_index
        .insert(id.clone(), vector.clone())
        .expect("test operation should succeed");
    euc_index
        .insert(id, vector.clone())
        .expect("test operation should succeed");

    let query = vec![1.0, 0.0, 0.0, 0.0];

    let cosine_results = cosine_index.search(&query, 1);
    let dot_results = dot_index.search(&query, 1);
    let euc_results = euc_index.search(&query, 1);

    // All should find the same vector
    assert_eq!(cosine_results.len(), 1);
    assert_eq!(dot_results.len(), 1);
    assert_eq!(euc_results.len(), 1);

    // Scores should be 1.0 for identical vectors
    assert!((cosine_results[0].1 - 1.0).abs() < 0.01);
    assert!((dot_results[0].1 - 1.0).abs() < 0.01);
    assert!((euc_results[0].1 - 1.0).abs() < 0.01);
}

// Test 11: Search with threshold
#[test]
fn test_search_with_threshold() {
    let config = AnnConfig::default().with_ef_search(100);
    let mut index = HnswIndex::new(4, config);

    let similar = normalize_vector(&[1.0, 0.1, 0.0, 0.0]);
    let dissimilar = normalize_vector(&[0.0, 1.0, 0.0, 0.0]);

    index
        .insert(DocumentId::from_string("similar"), similar)
        .expect("test operation should succeed");
    index
        .insert(DocumentId::from_string("dissimilar"), dissimilar)
        .expect("test operation should succeed");

    let query = normalize_vector(&[1.0, 0.0, 0.0, 0.0]);

    // Search with high threshold
    let results = index.search_with_threshold(&query, 10, 0.9);

    // Only the similar vector should pass the threshold
    assert_eq!(results.len(), 1);
    assert_eq!(results[0].0, DocumentId::from_string("similar"));
}

// Test 12: Empty index search
#[test]
fn test_search_empty_index() {
    let config = AnnConfig::default();
    let index = HnswIndex::new(4, config);

    let query = vec![1.0, 0.0, 0.0, 0.0];
    let results = index.search(&query, 10);

    assert!(results.is_empty());
}

// Test 13: Single element index
#[test]
fn test_single_element_index() {
    let config = AnnConfig::default();
    let mut index = HnswIndex::new(4, config);

    let id = DocumentId::from_string("only");
    let vector = vec![1.0, 0.0, 0.0, 0.0];
    index
        .insert(id.clone(), vector.clone())
        .expect("test operation should succeed");

    let results = index.search(&vector, 10);

    assert_eq!(results.len(), 1);
    assert_eq!(results[0].0, id);
}

// Test 14: Clear index
#[test]
fn test_clear_index() {
    let config = AnnConfig::default();
    let mut index = HnswIndex::new(4, config);

    for i in 0..10 {
        let id = DocumentId::from_string(format!("doc_{i}"));
        index
            .insert(id, create_random_vector(4, i))
            .expect("test operation should succeed");
    }

    assert_eq!(index.len(), 10);

    index.clear();

    assert!(index.is_empty());
    assert_eq!(index.len(), 0);
}

// Test 15: Statistics
#[test]
fn test_index_stats() {
    let config = AnnConfig::default();
    let mut index = HnswIndex::new(4, config);

    for i in 0..10 {
        let id = DocumentId::from_string(format!("doc_{i}"));
        index
            .insert(id, create_random_vector(4, i))
            .expect("test operation should succeed");
    }

    let stats = index.stats();

    assert_eq!(stats.num_nodes, 10);
    assert!(stats.memory_bytes > 0);
}

// Test 16: Duplicate ID replacement
#[test]
fn test_duplicate_id_replacement() {
    let config = AnnConfig::default();
    let mut index = HnswIndex::new(4, config);

    let id = DocumentId::from_string("doc");
    let vector1 = vec![1.0, 0.0, 0.0, 0.0];
    let vector2 = vec![0.0, 1.0, 0.0, 0.0];

    index
        .insert(id.clone(), vector1)
        .expect("test operation should succeed");
    index
        .insert(id.clone(), vector2.clone())
        .expect("test operation should succeed");

    assert_eq!(index.len(), 1);

    let node = index.get_node(&id).expect("test operation should succeed");
    assert_eq!(node.vector, vector2);
}

// Test 17: AnnVectorStore basic operations
#[tokio::test]
async fn test_ann_vector_store_basic() {
    let config = AnnConfig::default();
    let mut store = AnnVectorStore::new(4, config);

    let doc = IndexedDocument::new(Document::new("test content"), vec![1.0, 0.0, 0.0, 0.0]);

    let id = doc.document.id.clone();

    store
        .insert(doc)
        .await
        .expect("test operation should succeed");

    assert_eq!(store.count().await, 1);

    let retrieved = store.get(&id).await.expect("test operation should succeed");
    assert!(retrieved.is_some());
    assert_eq!(
        retrieved
            .expect("test operation should succeed")
            .document
            .content,
        "test content"
    );
}

// Test 18: AnnVectorStore search
#[tokio::test]
async fn test_ann_vector_store_search() {
    let config = AnnConfig::default().with_ef_search(100);
    let mut store = AnnVectorStore::new(4, config);

    let doc1 = IndexedDocument::new(
        Document::new("similar"),
        normalize_vector(&[1.0, 0.1, 0.0, 0.0]),
    );

    let doc2 = IndexedDocument::new(
        Document::new("dissimilar"),
        normalize_vector(&[0.0, 1.0, 0.0, 0.0]),
    );

    store
        .insert(doc1)
        .await
        .expect("test operation should succeed");
    store
        .insert(doc2)
        .await
        .expect("test operation should succeed");

    let query = normalize_vector(&[1.0, 0.0, 0.0, 0.0]);
    let results = store
        .search(&query, 2, None)
        .await
        .expect("test operation should succeed");

    assert_eq!(results.len(), 2);
    assert_eq!(results[0].document.content, "similar");
    assert_eq!(results[0].rank, 0);
}

// Test 19: AnnVectorStore with filter
#[tokio::test]
async fn test_ann_vector_store_with_filter() {
    let config = AnnConfig::default().with_ef_search(100);
    let mut store = AnnVectorStore::new(4, config);

    let doc1 = IndexedDocument::new(
        Document::new("science doc").with_metadata("category", "science"),
        normalize_vector(&[1.0, 0.1, 0.0, 0.0]),
    );

    let doc2 = IndexedDocument::new(
        Document::new("tech doc").with_metadata("category", "technology"),
        normalize_vector(&[0.9, 0.2, 0.0, 0.0]),
    );

    store
        .insert(doc1)
        .await
        .expect("test operation should succeed");
    store
        .insert(doc2)
        .await
        .expect("test operation should succeed");

    let query = normalize_vector(&[1.0, 0.0, 0.0, 0.0]);
    let filter = MetadataFilter::eq("category", "science");

    let results = store
        .search_with_filter(&query, 10, None, Some(&filter))
        .await
        .expect("test operation should succeed");

    assert_eq!(results.len(), 1);
    assert_eq!(results[0].document.content, "science doc");
}

// Test 20: AnnVectorStore upsert
#[tokio::test]
async fn test_ann_vector_store_upsert() {
    let config = AnnConfig::default();
    let mut store = AnnVectorStore::new(4, config);

    let doc = IndexedDocument::new(Document::new("original"), vec![1.0, 0.0, 0.0, 0.0]);

    let id = doc.document.id.clone();

    // First upsert (insert)
    let is_new = store
        .upsert(doc)
        .await
        .expect("test operation should succeed");
    assert!(is_new);
    assert_eq!(store.count().await, 1);

    // Second upsert (update)
    let updated_doc = IndexedDocument::new(
        Document::new("updated").with_id(id.clone()),
        vec![0.0, 1.0, 0.0, 0.0],
    );

    let is_new = store
        .upsert(updated_doc)
        .await
        .expect("test operation should succeed");
    assert!(!is_new);
    assert_eq!(store.count().await, 1);

    let retrieved = store
        .get(&id)
        .await
        .expect("test operation should succeed")
        .expect("test operation should succeed");
    assert_eq!(retrieved.document.content, "updated");
}

// Test 21: AnnVectorStore delete
#[tokio::test]
async fn test_ann_vector_store_delete() {
    let config = AnnConfig::default();
    let mut store = AnnVectorStore::new(4, config);

    let doc = IndexedDocument::new(Document::new("test"), vec![1.0, 0.0, 0.0, 0.0]);

    let id = doc.document.id.clone();

    store
        .insert(doc)
        .await
        .expect("test operation should succeed");
    assert_eq!(store.count().await, 1);

    let deleted = store
        .delete(&id)
        .await
        .expect("test operation should succeed");
    assert!(deleted);
    assert_eq!(store.count().await, 0);
}

// Test 22: AnnVectorStore update
#[tokio::test]
async fn test_ann_vector_store_update() {
    let config = AnnConfig::default().with_ef_search(100);
    let mut store = AnnVectorStore::new(4, config);

    let doc = IndexedDocument::new(
        Document::new("test"),
        normalize_vector(&[1.0, 0.0, 0.0, 0.0]),
    );

    let id = doc.document.id.clone();

    store
        .insert(doc)
        .await
        .expect("test operation should succeed");

    // Update embedding
    let new_embedding = normalize_vector(&[0.0, 1.0, 0.0, 0.0]);
    let updated = store
        .update(&id, new_embedding.clone())
        .await
        .expect("test operation should succeed");
    assert!(updated);

    // Verify by searching
    let results = store
        .search(&new_embedding, 1, None)
        .await
        .expect("test operation should succeed");
    assert_eq!(results[0].document.id, id);
}

// Test 23: AnnVectorStore clear
#[tokio::test]
async fn test_ann_vector_store_clear() {
    let config = AnnConfig::default();
    let mut store = AnnVectorStore::new(4, config);

    for i in 0..5 {
        let doc =
            IndexedDocument::new(Document::new(format!("doc{i}")), create_random_vector(4, i));
        store
            .insert(doc)
            .await
            .expect("test operation should succeed");
    }

    assert_eq!(store.count().await, 5);

    store.clear().await.expect("test operation should succeed");

    assert_eq!(store.count().await, 0);
}

// Test 24: Search results order (descending similarity)
#[test]
fn test_search_results_order() {
    let config = AnnConfig::default().with_ef_search(100);
    let mut index = HnswIndex::new(4, config);

    // Insert vectors with known similarities to query [1, 0, 0, 0]
    let query = normalize_vector(&[1.0, 0.0, 0.0, 0.0]);

    let v1 = normalize_vector(&[1.0, 0.0, 0.0, 0.0]); // sim ~ 1.0
    let v2 = normalize_vector(&[0.9, 0.3, 0.0, 0.0]); // sim ~ 0.95
    let v3 = normalize_vector(&[0.7, 0.7, 0.0, 0.0]); // sim ~ 0.71
    let v4 = normalize_vector(&[0.0, 1.0, 0.0, 0.0]); // sim ~ 0.0

    index
        .insert(DocumentId::from_string("v4"), v4)
        .expect("test operation should succeed");
    index
        .insert(DocumentId::from_string("v2"), v2)
        .expect("test operation should succeed");
    index
        .insert(DocumentId::from_string("v3"), v3)
        .expect("test operation should succeed");
    index
        .insert(DocumentId::from_string("v1"), v1)
        .expect("test operation should succeed");

    let results = index.search(&query, 4);

    // Should be sorted by descending similarity
    assert_eq!(results[0].0, DocumentId::from_string("v1"));
    assert_eq!(results[1].0, DocumentId::from_string("v2"));
    assert_eq!(results[2].0, DocumentId::from_string("v3"));
    assert_eq!(results[3].0, DocumentId::from_string("v4"));

    // Verify scores are in descending order
    for i in 0..results.len() - 1 {
        assert!(results[i].1 >= results[i + 1].1);
    }
}

// Test 25: Benchmark-style test comparing search times
#[test]
fn test_search_performance_scales() {
    let config = AnnConfig::default().with_ef_search(50);
    let mut index = HnswIndex::new(64, config);

    // Insert vectors
    for i in 0..500 {
        let id = DocumentId::from_string(format!("doc_{i}"));
        let vector = normalize_vector(&create_random_vector(64, i));
        index.insert(id, vector).expect("Failed to insert vector");
    }

    let query = normalize_vector(&create_random_vector(64, 999));

    // Measure search time (basic timing)
    let start = crate::time::Instant::now();
    for _ in 0..100 {
        let _ = index.search(&query, 10);
    }
    let duration = start.elapsed();

    // Should complete 100 searches in reasonable time
    // Use different thresholds for debug vs release builds
    #[cfg(debug_assertions)]
    let max_duration_secs = 5; // Debug builds are much slower
    #[cfg(not(debug_assertions))]
    let max_duration_secs = 1;

    assert!(
        duration.as_secs() < max_duration_secs,
        "Search took too long: {duration:?} (max: {max_duration_secs}s)"
    );
}
