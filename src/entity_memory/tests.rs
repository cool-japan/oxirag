use crate::entity_memory::types::{
    EntityCategory, EntityKnowledge, EntityMemoryConfig, EntityMemoryError, EntityMemoryStore,
    EntityMentionExtractor, HeuristicEntityMentionExtractor,
};

// ── EntityCategory ─────────────────────────────────────────────────────────

#[test]
fn test_entity_category_as_str_person() {
    assert_eq!(EntityCategory::Person.as_str(), "person");
}

#[test]
fn test_entity_category_as_str_organization() {
    assert_eq!(EntityCategory::Organization.as_str(), "organization");
}

#[test]
fn test_entity_category_as_str_location() {
    assert_eq!(EntityCategory::Location.as_str(), "location");
}

#[test]
fn test_entity_category_as_str_concept() {
    assert_eq!(EntityCategory::Concept.as_str(), "concept");
}

#[test]
fn test_entity_category_as_str_product() {
    assert_eq!(EntityCategory::Product.as_str(), "product");
}

#[test]
fn test_entity_category_as_str_other() {
    assert_eq!(EntityCategory::Other.as_str(), "other");
}

#[test]
fn test_entity_category_default_is_concept() {
    let cat = EntityCategory::default();
    assert_eq!(cat, EntityCategory::Concept);
}

#[test]
fn test_entity_category_clone() {
    let cat = EntityCategory::Person;
    let cloned = cat.clone();
    assert_eq!(cloned, EntityCategory::Person);
}

#[test]
fn test_entity_category_partial_eq() {
    assert_eq!(EntityCategory::Location, EntityCategory::Location);
    assert_ne!(EntityCategory::Location, EntityCategory::Person);
}

// ── EntityKnowledge ────────────────────────────────────────────────────────

#[test]
fn test_entity_knowledge_new_sets_name() {
    let ek = EntityKnowledge::new("Alice", EntityCategory::Person);
    assert_eq!(ek.name, "Alice");
}

#[test]
fn test_entity_knowledge_new_sets_category() {
    let ek = EntityKnowledge::new("OpenAI", EntityCategory::Organization);
    assert_eq!(ek.category, EntityCategory::Organization);
}

#[test]
fn test_entity_knowledge_new_mention_count_zero() {
    let ek = EntityKnowledge::new("Paris", EntityCategory::Location);
    assert_eq!(ek.mention_count, 0);
}

#[test]
fn test_entity_knowledge_new_salience_zero() {
    let ek = EntityKnowledge::new("Rust", EntityCategory::Concept);
    assert!((ek.salience - 0.0).abs() < 1e-6);
}

#[test]
fn test_entity_knowledge_new_facts_empty() {
    let ek = EntityKnowledge::new("Widget", EntityCategory::Product);
    assert!(ek.facts.is_empty());
}

#[test]
fn test_entity_knowledge_new_summary_empty() {
    let ek = EntityKnowledge::new("SomeThing", EntityCategory::Other);
    assert!(ek.summary.is_empty());
}

#[test]
fn test_entity_knowledge_fields_accessible() {
    let ek = EntityKnowledge::new("Berlin", EntityCategory::Location);
    let _ = ek.first_seen;
    let _ = ek.last_seen;
    let _ = ek.mention_count;
    let _ = ek.salience;
    let _ = &ek.facts;
    let _ = &ek.summary;
}

// ── HeuristicEntityMentionExtractor ────────────────────────────────────────

#[test]
fn test_heuristic_extractor_new() {
    let ext = HeuristicEntityMentionExtractor::new(3);
    assert_eq!(ext.min_len, 3);
}

#[test]
fn test_heuristic_extractor_default_min_len() {
    let ext = HeuristicEntityMentionExtractor::default();
    assert_eq!(ext.min_len, 0); // default() is derived: fields default to 0
}

#[test]
fn test_heuristic_extractor_extract_finds_capitalized_words() {
    let ext = HeuristicEntityMentionExtractor::new(2);
    let spans = ext.extract("Hello World today");
    let texts: Vec<&str> = spans.iter().map(|s| s.text.as_str()).collect();
    assert!(texts.contains(&"Hello"));
    assert!(texts.contains(&"World"));
}

#[test]
fn test_heuristic_extractor_extract_skips_lowercase() {
    let ext = HeuristicEntityMentionExtractor::new(2);
    let spans = ext.extract("the quick brown fox");
    assert!(spans.is_empty());
}

#[test]
fn test_heuristic_extractor_extract_finds_single_entity() {
    let ext = HeuristicEntityMentionExtractor::new(2);
    let spans = ext.extract("Paris is beautiful");
    assert_eq!(spans.len(), 1);
    assert_eq!(spans[0].text, "Paris");
}

#[test]
fn test_heuristic_extractor_extract_respects_min_len() {
    // min_len=5 — "Hi" (len 2) should be skipped, "Hello" (len 5) kept
    let ext = HeuristicEntityMentionExtractor::new(5);
    let spans = ext.extract("Hi Hello there");
    let texts: Vec<&str> = spans.iter().map(|s| s.text.as_str()).collect();
    assert!(!texts.contains(&"Hi"));
    assert!(texts.contains(&"Hello"));
}

#[test]
fn test_heuristic_extractor_span_start_end_set() {
    let ext = HeuristicEntityMentionExtractor::new(2);
    let text = "Paris is great";
    let spans = ext.extract(text);
    assert!(!spans.is_empty());
    let span = &spans[0];
    assert!(span.start < span.end);
    assert!(span.end <= text.len());
}

#[test]
fn test_heuristic_extractor_category_is_other() {
    let ext = HeuristicEntityMentionExtractor::new(2);
    let spans = ext.extract("Berlin");
    assert!(!spans.is_empty());
    assert_eq!(spans[0].category, EntityCategory::Other);
}

#[test]
fn test_heuristic_extractor_empty_text_returns_empty() {
    let ext = HeuristicEntityMentionExtractor::new(2);
    let spans = ext.extract("");
    assert!(spans.is_empty());
}

// ── EntityMemoryConfig ─────────────────────────────────────────────────────

#[test]
fn test_config_default_max_entities() {
    let cfg = EntityMemoryConfig::default();
    assert_eq!(cfg.max_entities, 128);
}

#[test]
fn test_config_default_max_facts_per_entity() {
    let cfg = EntityMemoryConfig::default();
    assert_eq!(cfg.max_facts_per_entity, 8);
}

#[test]
fn test_config_default_min_mention_len() {
    let cfg = EntityMemoryConfig::default();
    assert_eq!(cfg.min_mention_len, 2);
}

#[test]
fn test_config_default_decay() {
    let cfg = EntityMemoryConfig::default();
    assert!((cfg.decay - 0.9).abs() < 1e-6);
}

#[test]
fn test_config_builder_with_max_entities() {
    let cfg = EntityMemoryConfig::default().with_max_entities(64);
    assert_eq!(cfg.max_entities, 64);
}

#[test]
fn test_config_builder_with_max_facts_per_entity() {
    let cfg = EntityMemoryConfig::default().with_max_facts_per_entity(4);
    assert_eq!(cfg.max_facts_per_entity, 4);
}

#[test]
fn test_config_builder_with_min_mention_len() {
    let cfg = EntityMemoryConfig::default().with_min_mention_len(3);
    assert_eq!(cfg.min_mention_len, 3);
}

#[test]
fn test_config_builder_with_decay() {
    let cfg = EntityMemoryConfig::default().with_decay(0.5);
    assert!((cfg.decay - 0.5).abs() < 1e-6);
}

// ── EntityMemoryStore ──────────────────────────────────────────────────────

#[test]
fn test_store_new_is_empty() {
    let store = EntityMemoryStore::new(EntityMemoryConfig::default());
    assert!(store.entities.is_empty());
}

#[test]
fn test_store_default_is_empty() {
    let store = EntityMemoryStore::default();
    assert!(store.entities.is_empty());
}

#[test]
fn test_store_observe_empty_text_errors() {
    let mut store = EntityMemoryStore::default();
    let result = store.observe("");
    assert!(matches!(result, Err(EntityMemoryError::EmptyText)));
}

#[test]
fn test_store_observe_whitespace_only_errors() {
    let mut store = EntityMemoryStore::default();
    let result = store.observe("   ");
    assert!(matches!(result, Err(EntityMemoryError::EmptyText)));
}

#[test]
fn test_store_observe_returns_entity_keys() {
    let mut store = EntityMemoryStore::default();
    let keys = store.observe("Paris is a great city in France").unwrap();
    assert!(keys.contains(&"paris".to_string()) || keys.contains(&"france".to_string()));
}

#[test]
fn test_store_observe_entity_lowercased_key() {
    let mut store = EntityMemoryStore::default();
    store.observe("Paris is beautiful").unwrap();
    // Key must be lowercase
    assert!(store.entities.contains_key("paris"));
    assert!(!store.entities.contains_key("Paris"));
}

#[test]
fn test_store_get_existing_entity() {
    let mut store = EntityMemoryStore::default();
    store.observe("Paris is beautiful").unwrap();
    let result = store.get("paris");
    assert!(result.is_ok());
    let ek = result.unwrap();
    assert_eq!(ek.name, "Paris");
}

#[test]
fn test_store_get_nonexistent_entity_errors() {
    let store = EntityMemoryStore::default();
    let result = store.get("nonexistent_entity");
    assert!(matches!(result, Err(EntityMemoryError::EntityNotFound(_))));
}

#[test]
fn test_store_get_not_found_error_contains_name() {
    let store = EntityMemoryStore::default();
    let result = store.get("Atlantis");
    match result {
        Err(EntityMemoryError::EntityNotFound(name)) => {
            assert_eq!(name, "Atlantis");
        }
        _ => panic!("expected EntityNotFound"),
    }
}

#[test]
fn test_store_observe_increments_mention_count() {
    let mut store = EntityMemoryStore::default();
    store.observe("Paris is beautiful").unwrap();
    assert_eq!(store.get("paris").unwrap().mention_count, 1);
    store.observe("Paris is the capital of France").unwrap();
    assert_eq!(store.get("paris").unwrap().mention_count, 2);
}

#[test]
fn test_store_observe_sets_salience_above_zero() {
    let mut store = EntityMemoryStore::default();
    store.observe("Paris is great").unwrap();
    let ek = store.get("paris").unwrap();
    assert!(ek.salience > 0.0);
}

#[test]
fn test_store_observe_salience_increases_on_repeat() {
    let mut store = EntityMemoryStore::default();
    store.observe("Paris is great").unwrap();
    let sal_after_one = store.get("paris").unwrap().salience;
    store.observe("Paris is also beautiful").unwrap();
    let sal_after_two = store.get("paris").unwrap().salience;
    assert!(sal_after_two > sal_after_one);
}

#[test]
fn test_store_observe_salience_capped_at_one() {
    let mut store = EntityMemoryStore::default();
    // 6 observations of 0.2 each would saturate at 1.0
    for _ in 0..10 {
        store.observe("Paris is mentioned again").unwrap();
    }
    let ek = store.get("paris").unwrap();
    assert!(ek.salience <= 1.0);
}

#[test]
fn test_store_top_salient_returns_bounded_count() {
    let mut store = EntityMemoryStore::default();
    store.observe("Paris is a city").unwrap();
    store.observe("Berlin is a city").unwrap();
    store.observe("Tokyo is a city").unwrap();
    store.observe("London is a city").unwrap();
    let top = store.top_salient(2);
    assert!(top.len() <= 2);
}

#[test]
fn test_store_top_salient_sorted_descending() {
    let mut store = EntityMemoryStore::default();
    store.observe("Paris is mentioned").unwrap();
    store.observe("Paris is mentioned again").unwrap();
    store.observe("Berlin is mentioned").unwrap();
    let top = store.top_salient(10);
    for window in top.windows(2) {
        assert!(window[0].salience >= window[1].salience);
    }
}

#[test]
fn test_store_top_salient_empty_store_returns_empty() {
    let store = EntityMemoryStore::default();
    let top = store.top_salient(5);
    assert!(top.is_empty());
}

#[test]
fn test_store_context_for_known_entity() {
    let mut store = EntityMemoryStore::default();
    store.observe("Paris is the capital of France").unwrap();
    let ctx = store.context_for("query about paris");
    assert!(!ctx.is_empty());
}

#[test]
fn test_store_context_for_unknown_entity_returns_empty() {
    let mut store = EntityMemoryStore::default();
    store.observe("Paris is great").unwrap();
    let ctx = store.context_for("query about something unknown here");
    assert!(ctx.is_empty());
}

#[test]
fn test_store_observe_appends_facts() {
    let mut store = EntityMemoryStore::default();
    store.observe("Paris is the capital of France").unwrap();
    let ek = store.get("paris").unwrap();
    assert!(!ek.facts.is_empty());
}

#[test]
fn test_store_max_facts_per_entity_cap_respected() {
    let cfg = EntityMemoryConfig::default().with_max_facts_per_entity(2);
    let mut store = EntityMemoryStore::new(cfg);
    // Observe 5 times with different texts to trigger fact insertion
    store.observe("Paris is great today").unwrap();
    store.observe("Paris is the capital here").unwrap();
    store.observe("Paris is well known now").unwrap();
    store.observe("Paris has many attractions visible").unwrap();
    store
        .observe("Paris attracts millions of visitors")
        .unwrap();
    let ek = store.get("paris").unwrap();
    assert!(ek.facts.len() <= 2);
}

#[test]
fn test_store_multiple_entities_tracked_simultaneously() {
    let mut store = EntityMemoryStore::default();
    store.observe("Alice met Bob in Berlin today").unwrap();
    // At least some of Alice, Bob, Berlin should be tracked
    assert!(!store.entities.is_empty());
}

#[test]
fn test_store_entity_names_are_lowercase_keys() {
    let mut store = EntityMemoryStore::default();
    store.observe("Germany is a country").unwrap();
    // The key must be lowercased
    let has_lower = store.entities.contains_key("germany");
    assert!(has_lower);
}

// ── EntityMemoryError ──────────────────────────────────────────────────────

#[test]
fn test_error_empty_text_display() {
    let e = EntityMemoryError::EmptyText;
    let msg = format!("{e}");
    assert!(!msg.is_empty());
}

#[test]
fn test_error_entity_not_found_display_includes_name() {
    let e = EntityMemoryError::EntityNotFound("Foo".to_string());
    let msg = format!("{e}");
    assert!(msg.contains("Foo"));
}

#[test]
fn test_error_entity_not_found_wraps_name() {
    let e = EntityMemoryError::EntityNotFound("UniqueNameXYZ".to_string());
    let msg = format!("{e}");
    assert!(msg.contains("UniqueNameXYZ"));
}
