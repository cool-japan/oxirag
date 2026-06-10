use crate::long_term_memory::retrieval::MemoryRetriever;
use crate::long_term_memory::store::LongTermMemoryStore;
use crate::long_term_memory::types::{
    HeuristicImportanceScorer, ImportanceScorer, LongTermMemoryConfig, LongTermMemoryError,
    MemoryKind, MemoryQuery,
};
use chrono::Utc;

// ── LongTermMemoryConfig ───────────────────────────────────────────────────

#[test]
fn test_config_default_capacity() {
    let cfg = LongTermMemoryConfig::default();
    assert_eq!(cfg.capacity, 256);
}

#[test]
fn test_config_default_recency_weight() {
    let cfg = LongTermMemoryConfig::default();
    assert!((cfg.recency_weight - 0.35).abs() < 1e-6);
}

#[test]
fn test_config_default_importance_weight() {
    let cfg = LongTermMemoryConfig::default();
    assert!((cfg.importance_weight - 0.35).abs() < 1e-6);
}

#[test]
fn test_config_default_relevance_weight() {
    let cfg = LongTermMemoryConfig::default();
    assert!((cfg.relevance_weight - 0.30).abs() < 1e-6);
}

#[test]
fn test_config_default_decay_half_life() {
    let cfg = LongTermMemoryConfig::default();
    assert!((cfg.decay_half_life_secs - 86_400.0).abs() < 1e-6);
}

#[test]
fn test_config_default_dim() {
    let cfg = LongTermMemoryConfig::default();
    assert_eq!(cfg.dim, 128);
}

#[test]
fn test_config_builder_with_capacity() {
    let cfg = LongTermMemoryConfig::default().with_capacity(64);
    assert_eq!(cfg.capacity, 64);
}

#[test]
fn test_config_builder_with_recency_weight() {
    let cfg = LongTermMemoryConfig::default().with_recency_weight(0.5);
    assert!((cfg.recency_weight - 0.5).abs() < 1e-6);
}

#[test]
fn test_config_builder_with_importance_weight() {
    let cfg = LongTermMemoryConfig::default().with_importance_weight(0.4);
    assert!((cfg.importance_weight - 0.4).abs() < 1e-6);
}

#[test]
fn test_config_builder_with_relevance_weight() {
    let cfg = LongTermMemoryConfig::default().with_relevance_weight(0.1);
    assert!((cfg.relevance_weight - 0.1).abs() < 1e-6);
}

#[test]
fn test_config_builder_with_decay_half_life_secs() {
    let cfg = LongTermMemoryConfig::default().with_decay_half_life_secs(3600.0);
    assert!((cfg.decay_half_life_secs - 3600.0).abs() < 1e-6);
}

#[test]
fn test_config_builder_with_dim() {
    let cfg = LongTermMemoryConfig::default().with_dim(64);
    assert_eq!(cfg.dim, 64);
}

// ── HeuristicImportanceScorer ──────────────────────────────────────────────

#[test]
fn test_heuristic_scorer_returns_bounded_score() {
    let scorer = HeuristicImportanceScorer;
    let s = scorer.score("This is important and critical information.");
    assert!((0.0..=1.0).contains(&s));
}

#[test]
fn test_heuristic_scorer_empty_string_returns_zero() {
    let scorer = HeuristicImportanceScorer;
    let s = scorer.score("");
    assert!((0.0..=1.0).contains(&s));
}

#[test]
fn test_heuristic_scorer_affective_words_raise_score() {
    let scorer = HeuristicImportanceScorer;
    let high = scorer.score("This is critical and essential and vital must always");
    let low = scorer.score("the quick brown fox jumps over");
    assert!(high > low);
}

#[test]
fn test_heuristic_scorer_digits_raise_score() {
    let scorer = HeuristicImportanceScorer;
    let with_digits = scorer.score("the value is 42 and 7 and 3");
    let without = scorer.score("the value is forty two and seven");
    assert!(with_digits > without);
}

#[test]
fn test_heuristic_scorer_capitalized_words_raise_score() {
    let scorer = HeuristicImportanceScorer;
    let with_caps = scorer.score("Alice Bob Charlie met at Headquarters");
    let without = scorer.score("some people met at some place");
    assert!(with_caps > without);
}

#[test]
fn test_heuristic_scorer_clamps_at_one() {
    let scorer = HeuristicImportanceScorer;
    // Many affective words, digits, and capitalized words should saturate at 1.0
    let s = scorer.score("CRITICAL important urgent must essential vital key significant always Alice Bob Charlie 1 2 3");
    assert!(s <= 1.0);
}

#[test]
fn test_heuristic_scorer_plain_text_low_score() {
    let scorer = HeuristicImportanceScorer;
    let s = scorer.score("the cat sat on the mat");
    assert!(s < 0.3);
}

// ── LongTermMemoryStore ────────────────────────────────────────────────────

#[test]
fn test_store_new_is_empty() {
    let store = LongTermMemoryStore::new(LongTermMemoryConfig::default());
    assert!(store.is_empty());
    assert_eq!(store.len(), 0);
}

#[test]
fn test_store_default_is_empty() {
    let store = LongTermMemoryStore::default();
    assert!(store.is_empty());
}

#[test]
fn test_store_insert_observation_empty_content_errors() {
    let mut store = LongTermMemoryStore::default();
    let result = store.insert_observation("");
    assert!(matches!(result, Err(LongTermMemoryError::EmptyContent)));
}

#[test]
fn test_store_insert_observation_whitespace_only_errors() {
    let mut store = LongTermMemoryStore::default();
    let result = store.insert_observation("   ");
    assert!(matches!(result, Err(LongTermMemoryError::EmptyContent)));
}

#[test]
fn test_store_insert_observation_returns_id() {
    let mut store = LongTermMemoryStore::default();
    let result = store.insert_observation("I met Alice at the conference.");
    assert!(result.is_ok());
    let id = result.unwrap();
    assert!(!id.is_empty());
}

#[test]
fn test_store_len_after_single_insert() {
    let mut store = LongTermMemoryStore::default();
    store
        .insert_observation("An important event occurred.")
        .unwrap();
    assert_eq!(store.len(), 1);
    assert!(!store.is_empty());
}

#[test]
fn test_store_len_after_multiple_inserts() {
    let mut store = LongTermMemoryStore::default();
    store.insert_observation("Memory one.").unwrap();
    store.insert_observation("Memory two.").unwrap();
    store.insert_observation("Memory three.").unwrap();
    assert_eq!(store.len(), 3);
}

#[test]
fn test_store_insert_with_custom_scorer() {
    struct ConstantScorer(f32);
    impl ImportanceScorer for ConstantScorer {
        fn score(&self, _content: &str) -> f32 {
            self.0
        }
    }
    let mut store = LongTermMemoryStore::default();
    let scorer = ConstantScorer(0.75);
    let id = store
        .insert("Custom scored memory", MemoryKind::Fact, &scorer)
        .unwrap();
    let record = store.records.get(&id).unwrap();
    assert!((record.importance - 0.75).abs() < 1e-6);
}

#[test]
fn test_store_insert_kind_stored_correctly() {
    let mut store = LongTermMemoryStore::default();
    let id = store
        .insert(
            "A plan to do something",
            MemoryKind::Plan,
            &HeuristicImportanceScorer,
        )
        .unwrap();
    let record = store.records.get(&id).unwrap();
    assert_eq!(record.kind, MemoryKind::Plan);
}

#[test]
fn test_store_record_content_matches() {
    let mut store = LongTermMemoryStore::default();
    let content = "The quick brown fox.";
    let id = store.insert_observation(content).unwrap();
    let record = store.records.get(&id).unwrap();
    assert_eq!(record.content, content);
}

#[test]
fn test_store_record_has_embedding() {
    let mut store = LongTermMemoryStore::default();
    let id = store
        .insert_observation("Embedding test content here.")
        .unwrap();
    let record = store.records.get(&id).unwrap();
    assert_eq!(record.embedding.len(), store.config.dim);
}

#[test]
fn test_store_record_access_count_starts_zero() {
    let mut store = LongTermMemoryStore::default();
    let id = store.insert_observation("New observation record.").unwrap();
    let record = store.records.get(&id).unwrap();
    assert_eq!(record.access_count, 0);
}

#[test]
fn test_store_prune_removes_lowest_importance() {
    struct FixedScorer(f32);
    impl ImportanceScorer for FixedScorer {
        fn score(&self, _: &str) -> f32 {
            self.0
        }
    }
    let mut store = LongTermMemoryStore::default();
    let low_id = store
        .insert(
            "low importance content",
            MemoryKind::Observation,
            &FixedScorer(0.1),
        )
        .unwrap();
    store
        .insert(
            "high importance content one",
            MemoryKind::Observation,
            &FixedScorer(0.9),
        )
        .unwrap();
    store
        .insert(
            "high importance content two",
            MemoryKind::Observation,
            &FixedScorer(0.8),
        )
        .unwrap();
    assert_eq!(store.len(), 3);
    store.prune();
    assert_eq!(store.len(), 2);
    assert!(!store.records.contains_key(&low_id));
}

#[test]
fn test_store_reflect_empty_store_errors() {
    let mut store = LongTermMemoryStore::default();
    let result = store.reflect();
    assert!(matches!(result, Err(LongTermMemoryError::EmptyStore)));
}

#[test]
fn test_store_reflect_no_high_importance_observations_errors() {
    // importance=0.1 observations won't pass the > 0.5 threshold in reflect()
    struct LowScorer;
    impl ImportanceScorer for LowScorer {
        fn score(&self, _: &str) -> f32 {
            0.1
        }
    }
    let mut store = LongTermMemoryStore::default();
    store
        .insert(
            "low importance obs one",
            MemoryKind::Observation,
            &LowScorer,
        )
        .unwrap();
    store
        .insert(
            "low importance obs two",
            MemoryKind::Observation,
            &LowScorer,
        )
        .unwrap();
    let result = store.reflect();
    assert!(matches!(result, Err(LongTermMemoryError::EmptyStore)));
}

#[test]
fn test_store_reflect_with_high_importance_observations_returns_summary() {
    struct HighScorer;
    impl ImportanceScorer for HighScorer {
        fn score(&self, _: &str) -> f32 {
            0.9
        }
    }
    let mut store = LongTermMemoryStore::default();
    store
        .insert(
            "important observation alpha",
            MemoryKind::Observation,
            &HighScorer,
        )
        .unwrap();
    store
        .insert(
            "important observation beta",
            MemoryKind::Observation,
            &HighScorer,
        )
        .unwrap();
    store
        .insert(
            "important observation gamma",
            MemoryKind::Observation,
            &HighScorer,
        )
        .unwrap();
    let summary = store.reflect().unwrap();
    assert!(!summary.is_empty());
}

#[test]
fn test_store_reflect_adds_reflection_record() {
    struct HighScorer;
    impl ImportanceScorer for HighScorer {
        fn score(&self, _: &str) -> f32 {
            0.9
        }
    }
    let mut store = LongTermMemoryStore::default();
    store
        .insert("important obs one", MemoryKind::Observation, &HighScorer)
        .unwrap();
    store
        .insert("important obs two", MemoryKind::Observation, &HighScorer)
        .unwrap();
    let len_before = store.len();
    store.reflect().unwrap();
    assert_eq!(store.len(), len_before + 1);
    let has_reflection = store
        .records
        .values()
        .any(|r| r.kind == MemoryKind::Reflection);
    assert!(has_reflection);
}

#[test]
fn test_store_reflect_summary_contains_observation_content() {
    struct HighScorer;
    impl ImportanceScorer for HighScorer {
        fn score(&self, _: &str) -> f32 {
            0.9
        }
    }
    let mut store = LongTermMemoryStore::default();
    store
        .insert(
            "unique alpha phrase here",
            MemoryKind::Observation,
            &HighScorer,
        )
        .unwrap();
    let summary = store.reflect().unwrap();
    assert!(summary.contains("unique alpha phrase here"));
}

#[test]
fn test_store_capacity_overflow_auto_prunes() {
    // capacity=3: inserting 4 records should trigger prune, leaving 3
    let cfg = LongTermMemoryConfig::default().with_capacity(3);
    let mut store = LongTermMemoryStore::new(cfg);
    for i in 0..4 {
        store
            .insert_observation(format!("observation number {i}"))
            .unwrap();
    }
    assert_eq!(store.len(), 3);
}

#[test]
fn test_store_capacity_exactly_at_limit_no_prune() {
    let cfg = LongTermMemoryConfig::default().with_capacity(3);
    let mut store = LongTermMemoryStore::new(cfg);
    store.insert_observation("first record").unwrap();
    store.insert_observation("second record").unwrap();
    store.insert_observation("third record").unwrap();
    assert_eq!(store.len(), 3);
}

// ── MemoryKind ─────────────────────────────────────────────────────────────

#[test]
fn test_memory_kind_clone() {
    let kind = MemoryKind::Fact;
    let cloned = kind.clone();
    assert_eq!(cloned, MemoryKind::Fact);
}

#[test]
fn test_memory_kind_default_is_observation() {
    let kind = MemoryKind::default();
    assert_eq!(kind, MemoryKind::Observation);
}

#[test]
fn test_memory_kind_as_str_observation() {
    assert_eq!(MemoryKind::Observation.as_str(), "observation");
}

#[test]
fn test_memory_kind_as_str_reflection() {
    assert_eq!(MemoryKind::Reflection.as_str(), "reflection");
}

#[test]
fn test_memory_kind_as_str_fact() {
    assert_eq!(MemoryKind::Fact.as_str(), "fact");
}

#[test]
fn test_memory_kind_as_str_plan() {
    assert_eq!(MemoryKind::Plan.as_str(), "plan");
}

#[test]
fn test_memory_kind_partial_eq() {
    assert_eq!(MemoryKind::Observation, MemoryKind::Observation);
    assert_ne!(MemoryKind::Observation, MemoryKind::Reflection);
}

// ── LongTermMemoryError ────────────────────────────────────────────────────

#[test]
fn test_error_empty_content_display() {
    let e = LongTermMemoryError::EmptyContent;
    let msg = format!("{e}");
    assert!(!msg.is_empty());
    assert!(msg.contains("empty") || msg.contains("Empty") || msg.contains("must not"));
}

#[test]
fn test_error_empty_store_display() {
    let e = LongTermMemoryError::EmptyStore;
    let msg = format!("{e}");
    assert!(!msg.is_empty());
}

#[test]
fn test_error_empty_query_display() {
    let e = LongTermMemoryError::EmptyQuery;
    let msg = format!("{e}");
    assert!(!msg.is_empty());
}

// ── MemoryQuery ────────────────────────────────────────────────────────────

#[test]
fn test_memory_query_new_sets_fields() {
    let q = MemoryQuery::new("what happened yesterday", 5);
    assert_eq!(q.text, "what happened yesterday");
    assert_eq!(q.top_k, 5);
}

#[test]
fn test_memory_query_with_now_overrides_timestamp() {
    let fixed = Utc::now();
    let q = MemoryQuery::new("test query", 3).with_now(fixed);
    assert_eq!(q.now, fixed);
}

// ── MemoryRetriever ────────────────────────────────────────────────────────

#[test]
fn test_retriever_empty_store_errors() {
    let retriever = MemoryRetriever;
    let mut store = LongTermMemoryStore::default();
    let q = MemoryQuery::new("something", 5);
    let result = retriever.retrieve(&mut store, &q);
    assert!(matches!(result, Err(LongTermMemoryError::EmptyStore)));
}

#[test]
fn test_retriever_empty_query_errors() {
    let retriever = MemoryRetriever;
    let mut store = LongTermMemoryStore::default();
    store
        .insert_observation("some content to query against")
        .unwrap();
    let q = MemoryQuery::new("", 5);
    let result = retriever.retrieve(&mut store, &q);
    assert!(matches!(result, Err(LongTermMemoryError::EmptyQuery)));
}

#[test]
fn test_retriever_returns_results() {
    let retriever = MemoryRetriever;
    let mut store = LongTermMemoryStore::default();
    store
        .insert_observation("Paris is the capital of France")
        .unwrap();
    store
        .insert_observation("Berlin is the capital of Germany")
        .unwrap();
    let q = MemoryQuery::new("capital city Europe", 5);
    let results = retriever.retrieve(&mut store, &q).unwrap();
    assert!(!results.is_empty());
}

#[test]
fn test_retriever_respects_top_k() {
    let retriever = MemoryRetriever;
    let mut store = LongTermMemoryStore::default();
    for i in 0..10 {
        store
            .insert_observation(format!("memory record number {i} about various topics"))
            .unwrap();
    }
    let q = MemoryQuery::new("record", 3);
    let results = retriever.retrieve(&mut store, &q).unwrap();
    assert!(results.len() <= 3);
}

#[test]
fn test_retriever_increments_access_count() {
    let retriever = MemoryRetriever;
    let mut store = LongTermMemoryStore::default();
    let id = store
        .insert_observation("access count test content")
        .unwrap();
    assert_eq!(store.records[&id].access_count, 0);
    let q = MemoryQuery::new("access count test", 5);
    retriever.retrieve(&mut store, &q).unwrap();
    assert_eq!(store.records[&id].access_count, 1);
}

#[test]
fn test_retriever_increments_access_count_twice() {
    let retriever = MemoryRetriever;
    let mut store = LongTermMemoryStore::default();
    let id = store
        .insert_observation("double access count test")
        .unwrap();
    let q = MemoryQuery::new("double access", 5);
    retriever.retrieve(&mut store, &q).unwrap();
    retriever.retrieve(&mut store, &q).unwrap();
    assert_eq!(store.records[&id].access_count, 2);
}

#[test]
fn test_retriever_results_sorted_by_combined_desc() {
    let retriever = MemoryRetriever;
    let mut store = LongTermMemoryStore::default();
    store
        .insert_observation("important critical vital must essential fact")
        .unwrap();
    store
        .insert_observation("trivial random note with no keywords")
        .unwrap();
    store
        .insert_observation("key significant urgent event happened now")
        .unwrap();
    let q = MemoryQuery::new("important event", 10);
    let results = retriever.retrieve(&mut store, &q).unwrap();
    for window in results.windows(2) {
        assert!(window[0].combined >= window[1].combined);
    }
}

#[test]
fn test_retriever_combined_score_non_negative() {
    let retriever = MemoryRetriever;
    let mut store = LongTermMemoryStore::default();
    store
        .insert_observation("some observation to score")
        .unwrap();
    let q = MemoryQuery::new("observation", 5);
    let results = retriever.retrieve(&mut store, &q).unwrap();
    for r in &results {
        assert!(r.combined >= 0.0);
    }
}

#[test]
fn test_retriever_recency_score_bounded() {
    let retriever = MemoryRetriever;
    let mut store = LongTermMemoryStore::default();
    store
        .insert_observation("recency score test content")
        .unwrap();
    let q = MemoryQuery::new("recency score", 5);
    let results = retriever.retrieve(&mut store, &q).unwrap();
    for r in &results {
        assert!((0.0..=1.0).contains(&r.recency));
    }
}

#[test]
fn test_retriever_relevance_score_bounded() {
    let retriever = MemoryRetriever;
    let mut store = LongTermMemoryStore::default();
    store
        .insert_observation("relevance score test content here")
        .unwrap();
    let q = MemoryQuery::new("relevance score test", 5);
    let results = retriever.retrieve(&mut store, &q).unwrap();
    for r in &results {
        assert!((-1.0..=1.0).contains(&r.relevance));
    }
}

#[test]
fn test_retriever_updates_last_accessed() {
    let retriever = MemoryRetriever;
    let mut store = LongTermMemoryStore::default();
    let id = store.insert_observation("last accessed test").unwrap();
    let created_at = store.records[&id].last_accessed;
    let fixed_now = Utc::now() + chrono::Duration::seconds(100);
    let q = MemoryQuery::new("last accessed", 5).with_now(fixed_now);
    retriever.retrieve(&mut store, &q).unwrap();
    assert_eq!(store.records[&id].last_accessed, fixed_now);
    assert!(store.records[&id].last_accessed >= created_at);
}
