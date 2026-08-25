//! Hybrid search combining dense (vector) and sparse (BM25) retrieval.
//!
//! This module provides a comprehensive hybrid search implementation that combines:
//! - Dense retrieval using embedding vectors (semantic similarity)
//! - Sparse retrieval using BM25 weighting (lexical matching)
//!
//! The combination of both approaches typically yields better retrieval performance
//! than either method alone, as they capture complementary aspects of relevance.

pub mod bm25;
pub mod fusion;
pub mod types;

// Re-export everything at the module level to keep the same public API surface.
pub use bm25::{BM25Encoder, InMemorySparseStore};
pub use fusion::HybridSearcher;
pub use types::{
    BM25Params, FusionStrategy, HybridConfig, HybridResult, SparseVector, SparseVectorStore,
};

#[cfg(all(test, not(target_arch = "wasm32")))]
mod tests {
    use super::*;
    use crate::types::DocumentId;

    #[test]
    fn test_sparse_vector_new() {
        let v = SparseVector::new(vec![0, 2, 5], vec![1.0, 2.0, 3.0], 10);
        assert_eq!(v.indices, vec![0, 2, 5]);
        assert_eq!(v.values, vec![1.0, 2.0, 3.0]);
        assert_eq!(v.dimension, 10);
        assert_eq!(v.nnz(), 3);
    }

    #[test]
    fn test_sparse_vector_empty() {
        let v = SparseVector::empty(100);
        assert!(v.is_empty());
        assert_eq!(v.nnz(), 0);
        assert_eq!(v.dimension, 100);
    }

    #[test]
    fn test_sparse_vector_dot() {
        let a = SparseVector::new(vec![0, 2, 4], vec![1.0, 2.0, 3.0], 10);
        let b = SparseVector::new(vec![1, 2, 4], vec![1.0, 2.0, 1.0], 10);
        // Matching indices: 2 (2.0*2.0=4.0) and 4 (3.0*1.0=3.0)
        assert!((a.dot(&b) - 7.0).abs() < 1e-6);
    }

    #[test]
    fn test_sparse_vector_norm() {
        let v = SparseVector::new(vec![0, 1], vec![3.0, 4.0], 10);
        assert!((v.norm() - 5.0).abs() < 1e-6);
    }

    #[test]
    fn test_sparse_vector_cosine_similarity() {
        let a = SparseVector::new(vec![0, 1], vec![1.0, 0.0], 10);
        let b = SparseVector::new(vec![0, 1], vec![1.0, 0.0], 10);
        assert!((a.cosine_similarity(&b) - 1.0).abs() < 1e-6);

        let c = SparseVector::new(vec![0, 1], vec![0.0, 1.0], 10);
        assert!(a.cosine_similarity(&c).abs() < 1e-6);
    }

    #[test]
    fn test_sparse_vector_to_dense() {
        let v = SparseVector::new(vec![0, 2, 4], vec![1.0, 2.0, 3.0], 5);
        let dense = v.to_dense();
        assert_eq!(dense, vec![1.0, 0.0, 2.0, 0.0, 3.0]);
    }

    #[test]
    fn test_sparse_vector_from_dense() {
        let dense = vec![1.0, 0.0, 2.0, 0.0, 3.0];
        let sparse = SparseVector::from_dense(&dense);
        assert_eq!(sparse.indices, vec![0, 2, 4]);
        assert_eq!(sparse.values, vec![1.0, 2.0, 3.0]);
        assert_eq!(sparse.dimension, 5);
    }

    #[test]
    fn test_bm25_encoder_tokenize() {
        let tokens = BM25Encoder::tokenize("Hello, World! This is a test.");
        assert!(tokens.contains(&"hello".to_string()));
        assert!(tokens.contains(&"world".to_string()));
        assert!(tokens.contains(&"this".to_string()));
        assert!(tokens.contains(&"test".to_string()));
        assert!(tokens.contains(&"is".to_string())); // "is" has length 2, so it's included
        // Single character words should be filtered
        assert!(!tokens.contains(&"a".to_string()));
    }

    #[test]
    fn test_bm25_encoder_fit() {
        let mut encoder = BM25Encoder::new();
        let docs = vec!["the quick brown fox", "the lazy dog", "quick brown dog"];
        encoder.fit(&docs);

        assert_eq!(encoder.total_documents(), 3);
        assert!(encoder.vocab_size() > 0);
        // "the" appears in 2 docs, should have lower IDF
        // "fox" appears in 1 doc, should have higher IDF
    }

    #[test]
    fn test_bm25_encoder_encode() {
        let mut encoder = BM25Encoder::new();
        let docs = vec![
            "the quick brown fox jumps",
            "the lazy dog sleeps",
            "quick brown dog runs",
        ];
        encoder.fit(&docs);

        let query = "quick brown";
        let sparse = encoder.encode(query);

        assert!(!sparse.is_empty());
        assert!(sparse.nnz() > 0);
    }

    #[test]
    fn test_bm25_encoder_encode_batch() {
        let mut encoder = BM25Encoder::new();
        let docs = vec!["document one", "document two"];
        encoder.fit(&docs);

        let queries = vec!["query one", "query two"];
        let vectors = encoder.encode_batch(&queries);

        assert_eq!(vectors.len(), 2);
    }

    #[test]
    fn test_bm25_params_default() {
        let params = BM25Params::default();
        assert!((params.k1 - 1.5).abs() < 1e-6);
        assert!((params.b - 0.75).abs() < 1e-6);
    }

    #[test]
    fn test_hybrid_result_new() {
        let result = HybridResult::new(DocumentId::from("doc1"), 0.8, 0.6, 0.7, 0);
        assert_eq!(result.document_id.as_str(), "doc1");
        assert!((result.dense_score - 0.8).abs() < 1e-6);
        assert!((result.sparse_score - 0.6).abs() < 1e-6);
        assert!((result.combined_score - 0.7).abs() < 1e-6);
        assert_eq!(result.rank, 0);
    }

    #[test]
    fn test_fusion_strategy_default() {
        let strategy = FusionStrategy::default();
        match strategy {
            FusionStrategy::WeightedSum {
                dense_weight,
                sparse_weight,
            } => {
                assert!((dense_weight - 0.5).abs() < 1e-6);
                assert!((sparse_weight - 0.5).abs() < 1e-6);
            }
            _ => panic!("Expected WeightedSum as default"),
        }
    }

    #[test]
    fn test_fusion_strategy_rrf() {
        let strategy = FusionStrategy::rrf(60);
        match strategy {
            FusionStrategy::ReciprocalRankFusion { k } => {
                assert_eq!(k, 60);
            }
            _ => panic!("Expected ReciprocalRankFusion"),
        }
    }

    #[test]
    fn test_hybrid_config_default() {
        let config = HybridConfig::default();
        assert!((config.dense_weight - 0.5).abs() < 1e-6);
        assert!((config.sparse_weight - 0.5).abs() < 1e-6);
        assert!(config.normalize_scores);
        assert!(config.min_score.is_none());
    }

    #[test]
    fn test_hybrid_config_weighted_sum() {
        let config = HybridConfig::weighted_sum(0.7, 0.3);
        assert!((config.dense_weight - 0.7).abs() < 1e-6);
        assert!((config.sparse_weight - 0.3).abs() < 1e-6);
    }

    #[test]
    fn test_hybrid_config_rrf() {
        let config = HybridConfig::rrf(60);
        match config.fusion_strategy {
            FusionStrategy::ReciprocalRankFusion { k } => {
                assert_eq!(k, 60);
            }
            _ => panic!("Expected RRF fusion strategy"),
        }
    }

    #[tokio::test]
    async fn test_in_memory_sparse_store_insert_and_get() {
        let mut store = InMemorySparseStore::new();
        let id = DocumentId::from("doc1");
        let vector = SparseVector::new(vec![0, 2, 4], vec![1.0, 2.0, 3.0], 10);

        store
            .insert(id.clone(), vector.clone())
            .await
            .expect("test operation should succeed");

        let retrieved = store.get(&id).await.expect("test operation should succeed");
        assert!(retrieved.is_some());
        let retrieved = retrieved.expect("test operation should succeed");
        assert_eq!(retrieved.indices, vector.indices);
        assert_eq!(retrieved.values, vector.values);
    }

    #[tokio::test]
    async fn test_in_memory_sparse_store_search() {
        let mut store = InMemorySparseStore::new();

        // Insert documents
        store
            .insert(
                DocumentId::from("doc1"),
                SparseVector::new(vec![0, 1], vec![1.0, 0.5], 10),
            )
            .await
            .expect("test operation should succeed");
        store
            .insert(
                DocumentId::from("doc2"),
                SparseVector::new(vec![0, 2], vec![0.5, 1.0], 10),
            )
            .await
            .expect("test operation should succeed");
        store
            .insert(
                DocumentId::from("doc3"),
                SparseVector::new(vec![3, 4], vec![1.0, 1.0], 10),
            )
            .await
            .expect("test operation should succeed");

        // Search with query that matches doc1 and doc2
        let query = SparseVector::new(vec![0, 1], vec![1.0, 1.0], 10);
        let results = store
            .search(&query, 10)
            .await
            .expect("test operation should succeed");

        // doc1 should have highest score: 1.0*1.0 + 0.5*1.0 = 1.5
        // doc2 should have lower score: 0.5*1.0 = 0.5
        // doc3 should not match (no overlapping indices)
        assert_eq!(results.len(), 2);
        assert_eq!(results[0].0.as_str(), "doc1");
        assert!((results[0].1 - 1.5).abs() < 1e-6);
    }

    #[tokio::test]
    async fn test_in_memory_sparse_store_delete() {
        let mut store = InMemorySparseStore::new();
        let id = DocumentId::from("doc1");
        let vector = SparseVector::new(vec![0, 2], vec![1.0, 2.0], 10);

        store
            .insert(id.clone(), vector)
            .await
            .expect("test operation should succeed");
        assert_eq!(store.count().await, 1);

        let deleted = store
            .delete(&id)
            .await
            .expect("test operation should succeed");
        assert!(deleted);
        assert_eq!(store.count().await, 0);

        let deleted = store
            .delete(&id)
            .await
            .expect("test operation should succeed");
        assert!(!deleted);
    }

    #[tokio::test]
    async fn test_in_memory_sparse_store_clear() {
        let mut store = InMemorySparseStore::new();

        store
            .insert(
                DocumentId::from("doc1"),
                SparseVector::new(vec![0], vec![1.0], 10),
            )
            .await
            .expect("test operation should succeed");
        store
            .insert(
                DocumentId::from("doc2"),
                SparseVector::new(vec![1], vec![1.0], 10),
            )
            .await
            .expect("test operation should succeed");

        assert_eq!(store.count().await, 2);

        store.clear().await.expect("test operation should succeed");
        assert_eq!(store.count().await, 0);
    }

    #[tokio::test]
    async fn test_hybrid_searcher_weighted_sum() {
        let mut encoder = BM25Encoder::new();
        let docs = vec!["quick brown fox", "lazy dog", "brown dog"];
        encoder.fit(&docs);

        let mut store = InMemorySparseStore::new();
        for (i, doc) in docs.iter().enumerate() {
            let id = DocumentId::from(format!("doc{}", i + 1));
            let vector = encoder.encode(doc);
            store
                .insert(id, vector)
                .await
                .expect("test operation should succeed");
        }

        let config = HybridConfig::weighted_sum(0.5, 0.5);
        let searcher = HybridSearcher::new(store, encoder, config);

        // Dense results (simulated)
        let dense_results = vec![
            (DocumentId::from("doc1"), 0.9),
            (DocumentId::from("doc2"), 0.5),
            (DocumentId::from("doc3"), 0.7),
        ];

        let results = searcher
            .search("brown fox", &dense_results, 3)
            .await
            .expect("test operation should succeed");

        assert!(!results.is_empty());
        // Results should be sorted by combined score
        for i in 1..results.len() {
            assert!(results[i - 1].combined_score >= results[i].combined_score);
        }
    }

    #[tokio::test]
    async fn test_hybrid_searcher_rrf() {
        let mut encoder = BM25Encoder::new();
        let docs = vec!["quick brown fox", "lazy dog", "brown dog"];
        encoder.fit(&docs);

        let mut store = InMemorySparseStore::new();
        for (i, doc) in docs.iter().enumerate() {
            let id = DocumentId::from(format!("doc{}", i + 1));
            let vector = encoder.encode(doc);
            store
                .insert(id, vector)
                .await
                .expect("test operation should succeed");
        }

        let config = HybridConfig::rrf(60);
        let searcher = HybridSearcher::new(store, encoder, config);

        let dense_results = vec![
            (DocumentId::from("doc1"), 0.9),
            (DocumentId::from("doc3"), 0.7),
            (DocumentId::from("doc2"), 0.5),
        ];

        let results = searcher
            .search("brown", &dense_results, 3)
            .await
            .expect("test operation should succeed");

        assert!(!results.is_empty());
        // Results should be sorted by RRF score
        for i in 1..results.len() {
            assert!(results[i - 1].combined_score >= results[i].combined_score);
        }
    }

    #[tokio::test]
    async fn test_hybrid_searcher_distribution_based() {
        let mut encoder = BM25Encoder::new();
        let docs = vec!["quick brown fox", "lazy dog", "brown dog"];
        encoder.fit(&docs);

        let mut store = InMemorySparseStore::new();
        for (i, doc) in docs.iter().enumerate() {
            let id = DocumentId::from(format!("doc{}", i + 1));
            let vector = encoder.encode(doc);
            store
                .insert(id, vector)
                .await
                .expect("test operation should succeed");
        }

        let config = HybridConfig {
            fusion_strategy: FusionStrategy::distribution_based(0.5, 0.5),
            dense_weight: 0.5,
            sparse_weight: 0.5,
            normalize_scores: true,
            min_score: None,
        };
        let searcher = HybridSearcher::new(store, encoder, config);

        let dense_results = vec![
            (DocumentId::from("doc1"), 0.9),
            (DocumentId::from("doc2"), 0.5),
            (DocumentId::from("doc3"), 0.7),
        ];

        let results = searcher
            .search("brown fox", &dense_results, 3)
            .await
            .expect("test operation should succeed");

        assert!(!results.is_empty());
    }

    #[tokio::test]
    async fn test_hybrid_searcher_with_min_score() {
        let mut encoder = BM25Encoder::new();
        let docs = vec!["quick brown fox", "lazy dog"];
        encoder.fit(&docs);

        let mut store = InMemorySparseStore::new();
        for (i, doc) in docs.iter().enumerate() {
            let id = DocumentId::from(format!("doc{}", i + 1));
            let vector = encoder.encode(doc);
            store
                .insert(id, vector)
                .await
                .expect("test operation should succeed");
        }

        let config = HybridConfig::weighted_sum(0.5, 0.5).with_min_score(0.8);
        let searcher = HybridSearcher::new(store, encoder, config);

        let dense_results = vec![
            (DocumentId::from("doc1"), 0.9),
            (DocumentId::from("doc2"), 0.3),
        ];

        let results = searcher
            .search("quick fox", &dense_results, 10)
            .await
            .expect("test operation should succeed");

        // All results should have combined_score >= 0.8
        for result in &results {
            assert!(result.combined_score >= 0.8);
        }
    }

    #[test]
    fn test_normalize_score() {
        // Test normalization
        let score = HybridSearcher::<InMemorySparseStore>::normalize_score(0.5, 0.0, 1.0);
        assert!((score - 0.5).abs() < 1e-6);

        let score = HybridSearcher::<InMemorySparseStore>::normalize_score(5.0, 0.0, 10.0);
        assert!((score - 0.5).abs() < 1e-6);

        // Test edge case: min == max
        let score = HybridSearcher::<InMemorySparseStore>::normalize_score(5.0, 5.0, 5.0);
        assert!((score - 0.5).abs() < 1e-6);
    }

    #[test]
    fn test_score_stats() {
        let results = vec![
            (DocumentId::from("a"), 1.0),
            (DocumentId::from("b"), 2.0),
            (DocumentId::from("c"), 3.0),
        ];

        let (mean, std) = HybridSearcher::<InMemorySparseStore>::score_stats(&results);
        assert!((mean - 2.0).abs() < 1e-6);
        // std = sqrt(((1-2)^2 + (2-2)^2 + (3-2)^2) / 3) = sqrt(2/3)
        let expected_std = (2.0_f32 / 3.0).sqrt();
        assert!((std - expected_std).abs() < 1e-6);
    }

    #[test]
    fn test_z_normalize() {
        let normalized = HybridSearcher::<InMemorySparseStore>::z_normalize(5.0, 3.0, 2.0);
        // (5 - 3) / 2 = 1.0
        assert!((normalized - 1.0).abs() < 1e-6);
    }

    #[test]
    fn test_bm25_encoder_with_custom_params() {
        let params = BM25Params {
            k1: 2.0,
            b: 0.5,
            delta: 1.0,
        };
        let encoder = BM25Encoder::with_params(params);
        assert!((encoder.params().k1 - 2.0).abs() < 1e-6);
        assert!((encoder.params().b - 0.5).abs() < 1e-6);
    }

    #[test]
    fn test_sparse_vector_default() {
        let v = SparseVector::default();
        assert!(v.is_empty());
        assert_eq!(v.dimension, 0);
    }

    #[tokio::test]
    async fn test_hybrid_searcher_empty_results() {
        let encoder = BM25Encoder::new();
        let store = InMemorySparseStore::new();
        let config = HybridConfig::default();
        let searcher = HybridSearcher::new(store, encoder, config);

        let results = searcher
            .search("query", &[], 10)
            .await
            .expect("test operation should succeed");
        assert!(results.is_empty());
    }

    #[tokio::test]
    async fn test_hybrid_searcher_search_with_sparse() {
        let mut encoder = BM25Encoder::new();
        let docs = vec!["hello world", "world peace"];
        encoder.fit(&docs);

        let mut store = InMemorySparseStore::new();
        for (i, doc) in docs.iter().enumerate() {
            let id = DocumentId::from(format!("doc{}", i + 1));
            let vector = encoder.encode(doc);
            store
                .insert(id, vector)
                .await
                .expect("test operation should succeed");
        }

        let config = HybridConfig::default();
        let searcher = HybridSearcher::new(store, encoder, config);

        let sparse_query = searcher.encoder().encode("world");
        let dense_results = vec![
            (DocumentId::from("doc1"), 0.8),
            (DocumentId::from("doc2"), 0.6),
        ];

        let results = searcher
            .search_with_sparse(&sparse_query, &dense_results, 10)
            .await
            .expect("test operation should succeed");
        assert!(!results.is_empty());
    }
}
