//! Tests for the semantic cache module.

use std::collections::HashMap;

use crate::semantic_cache::{
    CacheEntry, InMemorySemanticCache, SemanticCache, SemanticCacheConfig,
};
use crate::types::{Document, DocumentId, Draft, PipelineOutput, Query, SearchResult};

// ── Helpers ──────────────────────────────────────────────────────────────────

fn make_output(answer: &str) -> PipelineOutput {
    let query = Query::new("test query");
    let draft = Draft::new(answer, "test query");
    let mut output = PipelineOutput::new(query, draft);
    output.final_answer = answer.to_string();
    output
}

fn make_result(content: &str, score: f32) -> SearchResult {
    SearchResult::new(Document::new(content), score, 0)
}

fn unit_vec(dim: usize, nonzero_idx: usize) -> Vec<f32> {
    let mut v = vec![0.0_f32; dim];
    if nonzero_idx < dim {
        v[nonzero_idx] = 1.0;
    }
    v
}

fn uniform_vec(dim: usize) -> Vec<f32> {
    #[allow(clippy::cast_precision_loss)]
    let val = 1.0_f32 / (dim as f32).sqrt();
    vec![val; dim]
}

fn default_cache() -> InMemorySemanticCache {
    InMemorySemanticCache::new(SemanticCacheConfig::default())
}

// ── Config tests ─────────────────────────────────────────────────────────────

#[test]
fn test_config_defaults() {
    let cfg = SemanticCacheConfig::default();
    assert!((cfg.similarity_threshold - 0.92).abs() < 1e-6);
    assert_eq!(cfg.max_entries, 1000);
    assert!(cfg.ttl_secs.is_none());
}

#[test]
fn test_config_builder_methods() {
    let cfg = SemanticCacheConfig::default()
        .with_threshold(0.85)
        .with_max_entries(50)
        .with_ttl(120);

    assert!((cfg.similarity_threshold - 0.85).abs() < 1e-6);
    assert_eq!(cfg.max_entries, 50);
    assert_eq!(cfg.ttl_secs, Some(120));
}

// ── Basic store + lookup ──────────────────────────────────────────────────────

#[tokio::test]
async fn test_store_and_exact_lookup() {
    let config = SemanticCacheConfig::default().with_threshold(0.90);
    let mut cache = InMemorySemanticCache::new(config);

    let emb = uniform_vec(8);
    let output = make_output("Rust is fast.");

    cache
        .store(emb.clone(), "What is Rust?", output.clone())
        .await;

    let hit = cache.lookup(&emb).await;
    assert!(hit.is_some());
    assert_eq!(hit.unwrap().final_answer, "Rust is fast.");
}

#[tokio::test]
async fn test_lookup_no_match_below_threshold() {
    let config = SemanticCacheConfig::default().with_threshold(0.99);
    let mut cache = InMemorySemanticCache::new(config);

    // Store with unit vector along axis 0
    let stored = unit_vec(4, 0); // [1, 0, 0, 0]
    cache
        .store(stored, "query A", make_output("answer A"))
        .await;

    // Lookup with unit vector along axis 1 (orthogonal, similarity = 0)
    let probe = unit_vec(4, 1); // [0, 1, 0, 0]
    let hit = cache.lookup(&probe).await;
    assert!(hit.is_none());
}

#[tokio::test]
async fn test_lookup_empty_cache_returns_none() {
    let cache = default_cache();
    let hit = cache.lookup(&uniform_vec(4)).await;
    assert!(hit.is_none());
}

#[tokio::test]
async fn test_multiple_entries_returns_first_match() {
    let config = SemanticCacheConfig::default().with_threshold(0.90);
    let mut cache = InMemorySemanticCache::new(config);

    // Two very different embeddings
    let emb_a = unit_vec(4, 0);
    let emb_b = unit_vec(4, 1);

    cache
        .store(emb_a.clone(), "query A", make_output("answer A"))
        .await;
    cache
        .store(emb_b.clone(), "query B", make_output("answer B"))
        .await;

    let hit_a = cache.lookup(&emb_a).await;
    assert!(hit_a.is_some());
    assert_eq!(hit_a.unwrap().final_answer, "answer A");

    let hit_b = cache.lookup(&emb_b).await;
    assert!(hit_b.is_some());
    assert_eq!(hit_b.unwrap().final_answer, "answer B");
}

// ── LRU eviction ─────────────────────────────────────────────────────────────

#[tokio::test]
async fn test_lru_eviction_removes_oldest() {
    let config = SemanticCacheConfig::default()
        .with_threshold(0.95)
        .with_max_entries(3);
    let mut cache = InMemorySemanticCache::new(config);

    let emb0 = unit_vec(8, 0);
    let emb1 = unit_vec(8, 1);
    let emb2 = unit_vec(8, 2);
    let emb3 = unit_vec(8, 3);

    cache.store(emb0.clone(), "q0", make_output("a0")).await;
    cache.store(emb1.clone(), "q1", make_output("a1")).await;
    cache.store(emb2.clone(), "q2", make_output("a2")).await;

    assert_eq!(cache.len(), 3);

    // Insert a 4th entry — emb0 should be evicted
    cache.store(emb3.clone(), "q3", make_output("a3")).await;
    assert_eq!(cache.len(), 3);

    // emb0 should be gone
    assert!(cache.lookup(&emb0).await.is_none());
    // emb3 should be present
    assert!(cache.lookup(&emb3).await.is_some());
}

#[tokio::test]
async fn test_max_entries_one_keeps_only_latest() {
    let config = SemanticCacheConfig::default()
        .with_threshold(0.95)
        .with_max_entries(1);
    let mut cache = InMemorySemanticCache::new(config);

    let emb0 = unit_vec(4, 0);
    let emb1 = unit_vec(4, 1);

    cache.store(emb0.clone(), "q0", make_output("a0")).await;
    cache.store(emb1.clone(), "q1", make_output("a1")).await;

    assert_eq!(cache.len(), 1);
    assert!(cache.lookup(&emb0).await.is_none()); // evicted
    assert!(cache.lookup(&emb1).await.is_some());
}

// ── TTL ───────────────────────────────────────────────────────────────────────

#[test]
fn test_is_expired_no_ttl() {
    // Entries created "just now" are never expired when ttl_secs is None
    let entry = CacheEntry::new("q", vec![1.0], make_output("a"));
    assert!(!InMemorySemanticCache::is_expired(&entry, None));
}

#[test]
fn test_is_expired_zero_ttl() {
    // TTL of 0 — even a freshly-created entry should expire after 0 seconds.
    // In practice elapsed() may be < 1 s, but is_expired checks >= secs.
    // For 0 secs the entry is expired only once elapsed >= 0, which is always
    // true (elapsed is always >= 0). So this should return true.
    let entry = CacheEntry::new("q", vec![1.0], make_output("a"));
    // elapsed() starts at 0 ns; 0 >= 0 is true
    assert!(InMemorySemanticCache::is_expired(&entry, Some(0)));
}

#[tokio::test]
async fn test_ttl_expired_entry_is_miss() {
    // TTL = 0 means everything expires immediately
    let config = SemanticCacheConfig::default()
        .with_threshold(0.80)
        .with_ttl(0);
    let mut cache = InMemorySemanticCache::new(config);

    let emb = uniform_vec(4);
    cache.store(emb.clone(), "q", make_output("a")).await;

    // Entry should be immediately expired (ttl=0 seconds)
    let hit = cache.lookup(&emb).await;
    assert!(hit.is_none());
}

// ── CacheStats ────────────────────────────────────────────────────────────────

#[tokio::test]
async fn test_stats_initial_state() {
    let cache = default_cache();
    let stats = cache.stats();
    assert_eq!(stats.entries, 0);
    assert_eq!(stats.total_hits, 0);
    assert_eq!(stats.total_lookups, 0);
    assert!((stats.hit_rate - 0.0).abs() < 1e-6);
}

#[tokio::test]
async fn test_stats_hit_rate_computation() {
    let config = SemanticCacheConfig::default().with_threshold(0.90);
    let mut cache = InMemorySemanticCache::new(config);

    let emb = uniform_vec(4);
    cache.store(emb.clone(), "q", make_output("a")).await;

    // 1 hit
    let _ = cache.lookup(&emb).await;
    // 1 miss (orthogonal)
    let _ = cache.lookup(&unit_vec(4, 0)).await;

    let stats = cache.stats();
    assert_eq!(stats.total_lookups, 2);
    assert_eq!(stats.total_hits, 1);
    assert!((stats.hit_rate - 0.5).abs() < 1e-4);
}

#[tokio::test]
async fn test_stats_zero_lookups_hit_rate() {
    let cache = default_cache();
    let stats = cache.stats();
    assert!((stats.hit_rate - 0.0).abs() < 1e-6);
}

#[tokio::test]
async fn test_stats_entries_after_clear() {
    let mut cache = default_cache();
    let emb = uniform_vec(4);
    cache.store(emb.clone(), "q", make_output("a")).await;
    assert_eq!(cache.stats().entries, 1);

    cache.clear().await;
    let stats = cache.stats();
    assert_eq!(stats.entries, 0);
    assert_eq!(stats.total_hits, 0);
    assert_eq!(stats.total_lookups, 0);
}

// ── Clear ─────────────────────────────────────────────────────────────────────

#[tokio::test]
async fn test_clear_removes_all_entries() {
    let mut cache = default_cache();
    for i in 0..5 {
        cache
            .store(unit_vec(4, i % 4), &format!("q{i}"), make_output("a"))
            .await;
    }
    assert!(!cache.is_empty());
    cache.clear().await;
    assert!(cache.is_empty());
}

// ── Edge cases ────────────────────────────────────────────────────────────────

#[test]
fn test_cosine_similarity_identical() {
    let v = vec![1.0_f32, 2.0, 3.0];
    let sim = InMemorySemanticCache::cosine_similarity(&v, &v);
    assert!(
        (sim - 1.0).abs() < 1e-5,
        "Identical vectors should have similarity 1.0, got {sim}"
    );
}

#[test]
fn test_cosine_similarity_orthogonal() {
    let a = vec![1.0_f32, 0.0, 0.0];
    let b = vec![0.0_f32, 1.0, 0.0];
    let sim = InMemorySemanticCache::cosine_similarity(&a, &b);
    assert!(
        (sim - 0.0).abs() < 1e-5,
        "Orthogonal vectors should have similarity 0.0, got {sim}"
    );
}

#[test]
fn test_cosine_similarity_anti_parallel() {
    let a = vec![1.0_f32, 0.0];
    let b = vec![-1.0_f32, 0.0];
    let sim = InMemorySemanticCache::cosine_similarity(&a, &b);
    assert!(
        sim <= 0.0,
        "Anti-parallel vectors should have similarity <= 0.0, got {sim}"
    );
}

#[test]
fn test_cosine_similarity_zero_vector() {
    let a = vec![0.0_f32, 0.0, 0.0];
    let b = vec![1.0_f32, 0.0, 0.0];
    let sim = InMemorySemanticCache::cosine_similarity(&a, &b);
    assert!(
        (sim - 0.0).abs() < 1e-5,
        "Zero vector should return 0.0, got {sim}"
    );
}

#[test]
fn test_cosine_similarity_both_zero_vectors() {
    let a = vec![0.0_f32; 4];
    let b = vec![0.0_f32; 4];
    let sim = InMemorySemanticCache::cosine_similarity(&a, &b);
    assert!(
        (sim - 0.0).abs() < 1e-5,
        "Two zero vectors should return 0.0, got {sim}"
    );
}

#[test]
fn test_cosine_similarity_dimension_mismatch() {
    let a = vec![1.0_f32, 0.0];
    let b = vec![1.0_f32, 0.0, 0.0];
    let sim = InMemorySemanticCache::cosine_similarity(&a, &b);
    assert!((sim - 0.0).abs() < 1e-5);
}

#[tokio::test]
async fn test_zero_length_embedding_returns_miss() {
    let config = SemanticCacheConfig::default().with_threshold(0.90);
    let mut cache = InMemorySemanticCache::new(config);

    // Storing an empty embedding is technically allowed, but
    // any lookup should be a miss (cosine_similarity → 0.0 < 0.90).
    cache
        .store(vec![], "empty query", make_output("answer"))
        .await;

    let hit = cache.lookup(&[]).await;
    // cosine_similarity([], []) returns 0.0 (both zero-magnitude) which is < 0.90
    assert!(hit.is_none());
}

// ── is_empty ──────────────────────────────────────────────────────────────────

#[tokio::test]
async fn test_is_empty_initial() {
    let cache = default_cache();
    assert!(cache.is_empty());
}

#[tokio::test]
async fn test_is_empty_after_store() {
    let mut cache = default_cache();
    cache.store(uniform_vec(4), "q", make_output("a")).await;
    assert!(!cache.is_empty());
}

// ── search_results in output survives cache round-trip ────────────────────────

#[tokio::test]
async fn test_output_with_search_results_survives_roundtrip() {
    let config = SemanticCacheConfig::default().with_threshold(0.90);
    let mut cache = InMemorySemanticCache::new(config);

    let emb = uniform_vec(8);
    let mut output = make_output("detailed answer");
    output.search_results = vec![
        make_result("relevant doc 1", 0.95),
        make_result("relevant doc 2", 0.88),
    ];

    cache.store(emb.clone(), "q", output).await;

    let hit = cache.lookup(&emb).await.expect("should be a cache hit");
    assert_eq!(hit.search_results.len(), 2);
    assert!((hit.search_results[0].score - 0.95).abs() < 1e-5);
}

// ── CacheEntry construction ───────────────────────────────────────────────────

#[test]
fn test_cache_entry_initial_hit_count() {
    let entry = CacheEntry::new("q", vec![1.0, 0.0], make_output("a"));
    assert_eq!(entry.hit_count, 0);
    assert_eq!(entry.query_text, "q");
}

// Suppress unused import warnings for HashMap if not used elsewhere
#[allow(unused_imports)]
fn _use_hashmap(_m: HashMap<String, String>, _id: DocumentId) {}
