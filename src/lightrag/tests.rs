use crate::lightrag::engine::{LightRagEngine, LightRagIndex};
use crate::lightrag::types::{
    LightRagChunk, LightRagConfig, LightRagDualKeywords, LightRagEntityKind, LightRagError,
    LightRagIndexStats, LightRagMode, LightRagRelation, LightRagResult, relation_dedup_key,
};

// ── Test fixtures ────────────────────────────────────────────────────────────

/// A small, hand-traceable 3-chunk corpus: Marie Curie <-> University of
/// Paris <-> France (2 hops), plus Marie Curie <-> Pierre Curie (1 hop).
fn curie_index() -> LightRagIndex {
    let mut index = LightRagIndex::new(128, true);
    index
        .insert_chunk(LightRagChunk::new(
            "c1",
            "Marie Curie discovered radium. Marie Curie worked at the University of Paris.",
        ))
        .unwrap();
    index
        .insert_chunk(LightRagChunk::new(
            "c2",
            "The University of Paris is a research institution in France.",
        ))
        .unwrap();
    index
        .insert_chunk(LightRagChunk::new(
            "c3",
            "Pierre Curie collaborated with Marie Curie on radioactivity research.",
        ))
        .unwrap();
    index
}

fn curie_engine() -> LightRagEngine {
    let mut engine = LightRagEngine::new(LightRagConfig::default());
    engine
        .insert_chunk(LightRagChunk::new(
            "c1",
            "Marie Curie discovered radium. Marie Curie worked at the University of Paris.",
        ))
        .unwrap();
    engine
        .insert_chunk(LightRagChunk::new(
            "c2",
            "The University of Paris is a research institution in France.",
        ))
        .unwrap();
    engine
        .insert_chunk(LightRagChunk::new(
            "c3",
            "Pierre Curie collaborated with Marie Curie on radioactivity research.",
        ))
        .unwrap();
    engine
}

/// A small corpus where "Zenith" connects to both "Acme Corp" (in a sentence
/// matching the high-level query keyword "partnership") and "Osaka" (in a
/// sentence that does not) — used to test Hybrid fusion boosting a
/// dual-path entity above a single-path one.
fn zenith_engine() -> LightRagEngine {
    let mut engine = LightRagEngine::new(LightRagConfig::default());
    engine
        .insert_chunk(LightRagChunk::new(
            "z1",
            "Zenith formed a partnership with Acme Corp.",
        ))
        .unwrap();
    engine
        .insert_chunk(LightRagChunk::new(
            "z2",
            "Zenith relocated its headquarters to Osaka.",
        ))
        .unwrap();
    engine
}

/// Three isolated, single-entity chunks sharing vocabulary with a query, for
/// testing local top-k truncation without any neighborhood expansion noise.
fn languages_engine() -> LightRagEngine {
    let mut engine = LightRagEngine::new(LightRagConfig::default());
    engine
        .insert_chunk(LightRagChunk::new(
            "r1",
            "Rust provides memory safety guarantees.",
        ))
        .unwrap();
    engine
        .insert_chunk(LightRagChunk::new(
            "p1",
            "Python provides dynamic typing features.",
        ))
        .unwrap();
    engine
        .insert_chunk(LightRagChunk::new(
            "j1",
            "JavaScript provides flexible scripting.",
        ))
        .unwrap();
    engine
}

// ── LightRagMode ─────────────────────────────────────────────────────────────

#[test]
fn mode_default_is_hybrid() {
    assert_eq!(LightRagMode::default(), LightRagMode::Hybrid);
}

#[test]
fn mode_label() {
    assert_eq!(LightRagMode::Local.label(), "local");
    assert_eq!(LightRagMode::Global.label(), "global");
    assert_eq!(LightRagMode::Hybrid.label(), "hybrid");
}

#[test]
fn mode_uses_local_and_global() {
    assert!(LightRagMode::Local.uses_local());
    assert!(!LightRagMode::Local.uses_global());
    assert!(LightRagMode::Global.uses_global());
    assert!(!LightRagMode::Global.uses_local());
    assert!(LightRagMode::Hybrid.uses_local());
    assert!(LightRagMode::Hybrid.uses_global());
}

// ── LightRagEntityKind ───────────────────────────────────────────────────────

#[test]
fn entity_kind_label() {
    assert_eq!(LightRagEntityKind::Person.label(), "person");
    assert_eq!(LightRagEntityKind::Organization.label(), "organization");
    assert_eq!(LightRagEntityKind::Location.label(), "location");
    assert_eq!(LightRagEntityKind::Date.label(), "date");
    assert_eq!(LightRagEntityKind::Concept.label(), "concept");
    assert_eq!(LightRagEntityKind::Other.label(), "other");
}

// ── LightRagConfig ───────────────────────────────────────────────────────────

#[test]
fn config_default_values() {
    let config = LightRagConfig::default();
    assert_eq!(config.top_k_entities, 5);
    assert_eq!(config.top_k_relations, 5);
    assert_eq!(config.default_mode, LightRagMode::Hybrid);
    assert_eq!(config.embedding_dim, 128);
    assert!(config.dedup_enabled);
    assert_eq!(config.hop_depth, 1);
}

#[test]
fn config_builder_methods() {
    let config = LightRagConfig::new()
        .with_top_k_entities(3)
        .with_top_k_relations(4)
        .with_default_mode(LightRagMode::Local)
        .with_embedding_dim(64)
        .with_dedup_enabled(false)
        .with_hop_depth(2);
    assert_eq!(config.top_k_entities, 3);
    assert_eq!(config.top_k_relations, 4);
    assert_eq!(config.default_mode, LightRagMode::Local);
    assert_eq!(config.embedding_dim, 64);
    assert!(!config.dedup_enabled);
    assert_eq!(config.hop_depth, 2);
}

// ── LightRagChunk / LightRagDualKeywords / LightRagIndexStats ───────────────

#[test]
fn chunk_new() {
    let chunk = LightRagChunk::new("id1", "some text");
    assert_eq!(chunk.id, "id1");
    assert_eq!(chunk.text, "some text");
}

#[test]
fn dual_keywords_new_is_empty_len() {
    let empty = LightRagDualKeywords::default();
    assert!(empty.is_empty());
    assert_eq!(empty.len(), 0);
    let kw = LightRagDualKeywords::new(
        vec!["a".to_string()],
        vec!["b".to_string(), "c".to_string()],
    );
    assert!(!kw.is_empty());
    assert_eq!(kw.len(), 3);
}

#[test]
fn index_stats_total_mentions() {
    let stats = LightRagIndexStats {
        entities_added: 2,
        entities_merged: 3,
        relations_added: 1,
        relations_merged: 4,
    };
    assert_eq!(stats.total_entity_mentions(), 5);
    assert_eq!(stats.total_relation_mentions(), 5);
}

// ── LightRagRelation helpers ─────────────────────────────────────────────────

#[test]
fn relation_dedup_key_order_independent() {
    assert_eq!(relation_dedup_key("a", "b"), relation_dedup_key("b", "a"));
    assert_eq!(
        relation_dedup_key("a", "b"),
        ("a".to_string(), "b".to_string())
    );
}

#[test]
fn relation_other_end() {
    let relation = LightRagRelation::new("alice", "bob", "desc", vec![], "c1");
    assert_eq!(relation.other_end("alice"), Some("bob"));
    assert_eq!(relation.other_end("bob"), Some("alice"));
    assert_eq!(relation.other_end("carol"), None);
}

#[test]
fn relation_new_sets_occurrence_count_and_source_chunks() {
    let relation = LightRagRelation::new("alice", "bob", "desc", vec!["kw".to_string()], "c1");
    assert_eq!(relation.occurrence_count, 1);
    assert_eq!(relation.source_chunks, vec!["c1".to_string()]);
    assert_eq!(
        relation.dedup_key(),
        ("alice".to_string(), "bob".to_string())
    );
}

// ── LightRagResult helpers ───────────────────────────────────────────────────

#[test]
fn result_is_empty_and_entity_keys() {
    let result = LightRagResult {
        query: "q".to_string(),
        mode: LightRagMode::Local,
        keywords: LightRagDualKeywords::default(),
        entities: vec![],
        relations: vec![],
        source_chunks: vec![],
        context: String::new(),
    };
    assert!(result.is_empty());
    assert!(result.entity_keys().is_empty());
}

// ── LightRagIndex / LightRagEngine construction ─────────────────────────────

#[test]
fn index_new_is_empty() {
    let index = LightRagIndex::new(128, true);
    assert_eq!(index.entity_count(), 0);
    assert_eq!(index.relation_count(), 0);
    assert_eq!(index.chunk_count(), 0);
    assert_eq!(index.embedding_dim(), 128);
    assert!(index.dedup_enabled());
}

#[test]
fn engine_new_exposes_config_and_empty_index() {
    let config = LightRagConfig::default().with_embedding_dim(64);
    let engine = LightRagEngine::new(config.clone());
    assert_eq!(engine.config(), &config);
    assert_eq!(engine.index().entity_count(), 0);
}

#[test]
fn engine_insert_chunk_delegates_to_index() {
    let mut engine = LightRagEngine::new(LightRagConfig::default());
    let stats = engine
        .insert_chunk(LightRagChunk::new(
            "c1",
            "Acme Corporation announced a new product.",
        ))
        .unwrap();
    assert_eq!(stats.entities_added, 1);
    assert_eq!(engine.index().entity_count(), 1);
    assert_eq!(engine.index().chunk_count(), 1);
}

// ── insert_chunk validation ──────────────────────────────────────────────────

#[test]
fn insert_chunk_empty_text_error() {
    let mut index = LightRagIndex::new(128, true);
    let err = index
        .insert_chunk(LightRagChunk::new("c1", "   "))
        .unwrap_err();
    assert!(matches!(err, LightRagError::EmptyChunkText));
}

#[test]
fn insert_chunk_empty_id_error() {
    let mut index = LightRagIndex::new(128, true);
    let err = index
        .insert_chunk(LightRagChunk::new("  ", "some text"))
        .unwrap_err();
    assert!(matches!(err, LightRagError::EmptyChunkId));
}

#[test]
fn insert_chunk_duplicate_id_error() {
    let mut index = LightRagIndex::new(128, true);
    index
        .insert_chunk(LightRagChunk::new(
            "c1",
            "Acme Corporation announced a product.",
        ))
        .unwrap();
    let err = index
        .insert_chunk(LightRagChunk::new("c1", "Different text entirely."))
        .unwrap_err();
    match err {
        LightRagError::DuplicateChunkId(id) => assert_eq!(id, "c1"),
        other => panic!("expected DuplicateChunkId, got {other:?}"),
    }
}

// ── Entity extraction ────────────────────────────────────────────────────────

#[test]
fn entity_extraction_basic_capitalized_span() {
    let mut index = LightRagIndex::new(128, true);
    index
        .insert_chunk(LightRagChunk::new(
            "c1",
            "Acme Corporation announced a new product.",
        ))
        .unwrap();
    let entity = index.get_entity("acme corporation").unwrap();
    assert_eq!(entity.name, "Acme Corporation");
}

#[test]
fn entity_extraction_multi_word_connector() {
    let mut index = LightRagIndex::new(128, true);
    index
        .insert_chunk(LightRagChunk::new(
            "c1",
            "She studied at the University of Tokyo.",
        ))
        .unwrap();
    assert!(index.get_entity("university of tokyo").is_some());
}

#[test]
fn entity_extraction_acronym() {
    let mut index = LightRagIndex::new(128, true);
    index
        .insert_chunk(LightRagChunk::new("c1", "NASA launched a satellite."))
        .unwrap();
    assert!(index.get_entity("nasa").is_some());
}

#[test]
fn entity_extraction_skips_sentence_initial_stopword() {
    let mut index = LightRagIndex::new(128, true);
    let stats = index
        .insert_chunk(LightRagChunk::new("c1", "The company grew rapidly."))
        .unwrap();
    assert_eq!(stats.entities_added, 0);
    assert_eq!(index.entity_count(), 0);
}

#[test]
fn entity_extraction_sentence_initial_non_stopword_is_captured() {
    let mut index = LightRagIndex::new(128, true);
    index
        .insert_chunk(LightRagChunk::new("c1", "Kyoto has many temples."))
        .unwrap();
    assert!(index.get_entity("kyoto").is_some());
}

// ── Entity kind inference ────────────────────────────────────────────────────

#[test]
fn entity_kind_organization() {
    let mut index = LightRagIndex::new(128, true);
    index
        .insert_chunk(LightRagChunk::new(
            "c1",
            "Acme Corporation announced a new product.",
        ))
        .unwrap();
    assert_eq!(
        index.get_entity("acme corporation").unwrap().kind,
        LightRagEntityKind::Organization
    );
}

#[test]
fn entity_kind_location_gazetteer() {
    let mut index = LightRagIndex::new(128, true);
    index
        .insert_chunk(LightRagChunk::new(
            "c1",
            "The expedition reached Lake Victoria.",
        ))
        .unwrap();
    assert_eq!(
        index.get_entity("lake victoria").unwrap().kind,
        LightRagEntityKind::Location
    );
}

#[test]
fn entity_kind_location_preposition() {
    let mut index = LightRagIndex::new(128, true);
    index
        .insert_chunk(LightRagChunk::new("c1", "The event happened in Kyoto."))
        .unwrap();
    assert_eq!(
        index.get_entity("kyoto").unwrap().kind,
        LightRagEntityKind::Location
    );
}

#[test]
fn entity_kind_person_two_words() {
    let mut index = LightRagIndex::new(128, true);
    index
        .insert_chunk(LightRagChunk::new("c1", "Marie Curie discovered radium."))
        .unwrap();
    assert_eq!(
        index.get_entity("marie curie").unwrap().kind,
        LightRagEntityKind::Person
    );
}

#[test]
fn entity_kind_date_year() {
    let mut index = LightRagIndex::new(128, true);
    index
        .insert_chunk(LightRagChunk::new("c1", "The treaty was signed in 1905."))
        .unwrap();
    assert_eq!(
        index.get_entity("1905").unwrap().kind,
        LightRagEntityKind::Date
    );
}

#[test]
fn entity_kind_concept_single_word_fallback() {
    let mut index = LightRagIndex::new(128, true);
    index
        .insert_chunk(LightRagChunk::new(
            "c1",
            "Photosynthesis converts sunlight into energy.",
        ))
        .unwrap();
    assert_eq!(
        index.get_entity("photosynthesis").unwrap().kind,
        LightRagEntityKind::Concept
    );
}

// ── Relation extraction ──────────────────────────────────────────────────────

#[test]
fn relation_extraction_basic_two_entities() {
    let index = curie_index();
    assert!(
        index
            .get_relation("marie curie", "university of paris")
            .is_some()
    );
}

#[test]
fn relation_extraction_none_for_single_entity() {
    let mut index = LightRagIndex::new(128, true);
    index
        .insert_chunk(LightRagChunk::new("c1", "Marie Curie was a physicist."))
        .unwrap();
    assert_eq!(index.entity_count(), 1);
    assert_eq!(index.relation_count(), 0);
}

#[test]
fn relation_keywords_extracted() {
    let index = curie_index();
    let relation = index
        .get_relation("marie curie", "university of paris")
        .unwrap();
    assert_eq!(relation.keywords, vec!["worked".to_string()]);
}

// ── Incremental dedup: entities ──────────────────────────────────────────────

#[test]
fn dedup_merge_keeps_entity_count_stable() {
    let mut index = LightRagIndex::new(128, true);
    index
        .insert_chunk(LightRagChunk::new(
            "c1",
            "Acme Corporation announced a product.",
        ))
        .unwrap();
    assert_eq!(index.entity_count(), 1);
    let stats = index
        .insert_chunk(LightRagChunk::new(
            "c2",
            "Acme Corporation expanded overseas.",
        ))
        .unwrap();
    assert_eq!(index.entity_count(), 1);
    assert_eq!(stats.entities_added, 0);
    assert_eq!(stats.entities_merged, 1);
}

#[test]
fn dedup_merge_accumulates_description_and_sources() {
    let mut index = LightRagIndex::new(128, true);
    index
        .insert_chunk(LightRagChunk::new(
            "c1",
            "Acme Corporation announced a product.",
        ))
        .unwrap();
    index
        .insert_chunk(LightRagChunk::new(
            "c2",
            "Acme Corporation expanded overseas.",
        ))
        .unwrap();
    let entity = index.get_entity("acme corporation").unwrap();
    assert!(entity.description.contains("announced a product"));
    assert!(entity.description.contains("expanded overseas"));
    assert_eq!(
        entity.source_chunks,
        vec!["c1".to_string(), "c2".to_string()]
    );
}

#[test]
fn dedup_merge_increments_occurrence_count() {
    let mut index = LightRagIndex::new(128, true);
    index
        .insert_chunk(LightRagChunk::new(
            "c1",
            "Acme Corporation announced a product.",
        ))
        .unwrap();
    index
        .insert_chunk(LightRagChunk::new(
            "c2",
            "Acme Corporation expanded overseas.",
        ))
        .unwrap();
    assert_eq!(
        index
            .get_entity("acme corporation")
            .unwrap()
            .occurrence_count,
        2
    );
}

#[test]
fn dedup_disabled_creates_distinct_entities() {
    let mut index = LightRagIndex::new(128, false);
    index
        .insert_chunk(LightRagChunk::new(
            "c1",
            "Acme Corporation announced a product.",
        ))
        .unwrap();
    let stats2 = index
        .insert_chunk(LightRagChunk::new(
            "c2",
            "Acme Corporation expanded overseas.",
        ))
        .unwrap();
    assert_eq!(index.entity_count(), 2);
    assert_eq!(stats2.entities_added, 1);
    assert_eq!(stats2.entities_merged, 0);
}

// ── Incremental dedup: relations ─────────────────────────────────────────────

#[test]
fn relation_dedup_merges_reversed_direction() {
    let mut index = LightRagIndex::new(128, true);
    index
        .insert_chunk(LightRagChunk::new("c1", "Alice met Bob."))
        .unwrap();
    index
        .insert_chunk(LightRagChunk::new("c2", "Bob greeted Alice warmly."))
        .unwrap();
    assert_eq!(index.relation_count(), 1);
    let relation = index.get_relation("alice", "bob").unwrap();
    assert_eq!(relation.src, "alice");
    assert_eq!(relation.dst, "bob");
    assert_eq!(relation.occurrence_count, 2);
    assert_eq!(
        relation.source_chunks,
        vec!["c1".to_string(), "c2".to_string()]
    );
}

#[test]
fn relation_dedup_accumulates_keywords_and_sources() {
    let mut index = LightRagIndex::new(128, true);
    index
        .insert_chunk(LightRagChunk::new("c1", "Alice met Bob."))
        .unwrap();
    index
        .insert_chunk(LightRagChunk::new("c2", "Bob greeted Alice warmly."))
        .unwrap();
    let relation = index.get_relation("alice", "bob").unwrap();
    assert_eq!(
        relation.keywords,
        vec![
            "met".to_string(),
            "greeted".to_string(),
            "warmly".to_string()
        ]
    );
}

// ── Dual-keyword extraction ──────────────────────────────────────────────────

#[test]
fn dual_keywords_separates_low_and_high() {
    let engine = LightRagEngine::new(LightRagConfig::default());
    let kw = engine
        .extract_dual_keywords(
            "What is the relationship between Marie Curie and the University of Paris?",
        )
        .unwrap();
    assert!(kw.low_level.contains(&"marie curie".to_string()));
    assert!(kw.low_level.contains(&"university of paris".to_string()));
    assert!(kw.high_level.contains(&"relationship".to_string()));
    assert!(!kw.high_level.iter().any(|k| k.contains("marie")));
}

#[test]
fn dual_keywords_empty_query_error() {
    let engine = LightRagEngine::new(LightRagConfig::default());
    let err = engine.extract_dual_keywords("   ").unwrap_err();
    assert!(matches!(err, LightRagError::EmptyQuery));
}

#[test]
fn dual_keywords_all_lowercase_fallback_nonempty_low_level() {
    let engine = LightRagEngine::new(LightRagConfig::default());
    let kw = engine
        .extract_dual_keywords("explain quantum encryption mechanisms")
        .unwrap();
    assert!(!kw.low_level.is_empty());
    assert_eq!(kw.low_level.len(), 1);
}

#[test]
fn dual_keywords_determinism() {
    let engine = LightRagEngine::new(LightRagConfig::default());
    let kw1 = engine
        .extract_dual_keywords("Marie Curie research institution")
        .unwrap();
    let kw2 = engine
        .extract_dual_keywords("Marie Curie research institution")
        .unwrap();
    assert_eq!(kw1, kw2);
}

#[test]
fn dual_keywords_numeric_token_is_low_level() {
    let engine = LightRagEngine::new(LightRagConfig::default());
    let kw = engine
        .extract_dual_keywords("What happened in 1905?")
        .unwrap();
    assert!(kw.low_level.contains(&"1905".to_string()));
}

// ── Query validation ──────────────────────────────────────────────────────────

#[test]
fn query_empty_query_error() {
    let engine = zenith_engine();
    let err = engine.query("   ", LightRagMode::Hybrid).unwrap_err();
    assert!(matches!(err, LightRagError::EmptyQuery));
}

#[test]
fn query_empty_index_error() {
    let engine = LightRagEngine::new(LightRagConfig::default());
    let err = engine.query("some text", LightRagMode::Hybrid).unwrap_err();
    assert!(matches!(err, LightRagError::EmptyIndex));
}

#[test]
fn query_no_match_returns_empty_result() {
    let engine = zenith_engine();
    let result = engine
        .query("xyzzy plugh nonsense", LightRagMode::Hybrid)
        .unwrap();
    assert!(result.is_empty());
    assert!(result.context.is_empty());
}

// ── Local mode ────────────────────────────────────────────────────────────────

#[test]
fn local_mode_returns_entity_and_neighbors() {
    let engine = curie_engine();
    let result = engine
        .query("information about Marie Curie", LightRagMode::Local)
        .unwrap();
    let keys: Vec<&str> = result.entities.iter().map(|e| e.key.as_str()).collect();
    // "marie curie" is the exact lexical match, so it must be the top-ranked
    // seed, ahead of any entity that only scores via vector overlap.
    assert_eq!(result.entities.first().unwrap().key, "marie curie");
    assert!(keys.contains(&"marie curie"));
    assert!(keys.contains(&"university of paris"));
    assert!(keys.contains(&"pierre curie"));
}

#[test]
fn local_mode_neighborhood_includes_incident_relations() {
    let engine = curie_engine();
    let result = engine
        .query("information about Marie Curie", LightRagMode::Local)
        .unwrap();
    assert!(
        result.relations.iter().any(
            |r| r.dedup_key() == ("marie curie".to_string(), "university of paris".to_string())
        )
    );
    assert!(
        result
            .relations
            .iter()
            .any(|r| r.dedup_key() == ("marie curie".to_string(), "pierre curie".to_string()))
    );
}

#[test]
fn local_mode_hop_depth_2_expands_further() {
    let config = LightRagConfig::default().with_hop_depth(2);
    let mut engine = LightRagEngine::new(config);
    engine
        .insert_chunk(LightRagChunk::new(
            "c1",
            "Marie Curie discovered radium. Marie Curie worked at the University of Paris.",
        ))
        .unwrap();
    engine
        .insert_chunk(LightRagChunk::new(
            "c2",
            "The University of Paris is a research institution in France.",
        ))
        .unwrap();
    let result = engine
        .query("information about Marie Curie", LightRagMode::Local)
        .unwrap();
    let keys: Vec<&str> = result.entities.iter().map(|e| e.key.as_str()).collect();
    assert!(keys.contains(&"france"));
}

// ── Global mode ───────────────────────────────────────────────────────────────

#[test]
fn global_mode_returns_relation_centric_results() {
    let engine = curie_engine();
    let result = engine
        .query("research institution", LightRagMode::Global)
        .unwrap();
    assert!(
        result
            .relations
            .iter()
            .any(|r| r.dedup_key() == ("france".to_string(), "university of paris".to_string()))
    );
    let keys: Vec<&str> = result.entities.iter().map(|e| e.key.as_str()).collect();
    assert!(keys.contains(&"university of paris"));
    assert!(keys.contains(&"france"));
}

// ── Hybrid mode ───────────────────────────────────────────────────────────────

#[test]
fn hybrid_mode_fuses_without_duplicate_entities() {
    let engine = zenith_engine();
    let result = engine
        .query("Zenith partnership", LightRagMode::Hybrid)
        .unwrap();
    let mut seen = std::collections::HashSet::new();
    for entity in &result.entities {
        assert!(
            seen.insert(entity.key.clone()),
            "duplicate entity key: {}",
            entity.key
        );
    }
    assert!(!result.entities.is_empty());
}

#[test]
fn hybrid_mode_dual_path_entity_ranks_above_single_path() {
    let engine = zenith_engine();
    let result = engine
        .query("Zenith partnership", LightRagMode::Hybrid)
        .unwrap();
    let acme_pos = result.entities.iter().position(|e| e.key == "acme corp");
    let osaka_pos = result.entities.iter().position(|e| e.key == "osaka");
    assert!(
        acme_pos.is_some(),
        "acme corp should be present: {result:?}"
    );
    assert!(osaka_pos.is_some(), "osaka should be present: {result:?}");
    assert!(
        acme_pos < osaka_pos,
        "dual-path entity should rank above single-path entity"
    );
}

// ── Ranking correctness ───────────────────────────────────────────────────────

#[test]
fn ranking_correctness_lexical_match_ranks_first() {
    let engine = languages_engine();
    let result = engine
        .query(
            "I want to know about Rust memory safety.",
            LightRagMode::Local,
        )
        .unwrap();
    assert!(!result.entities.is_empty());
    assert_eq!(result.entities[0].key, "rust");
}

// ── 1-hop / N-hop neighborhood correctness ──────────────────────────────────

#[test]
fn one_hop_neighborhood_correctness() {
    let index = curie_index();
    let (entities, relations) = index.neighborhood(&["marie curie".to_string()], 1);
    assert_eq!(
        entities,
        vec![
            "marie curie".to_string(),
            "pierre curie".to_string(),
            "university of paris".to_string()
        ]
    );
    assert_eq!(
        relations,
        vec![
            ("marie curie".to_string(), "pierre curie".to_string()),
            ("marie curie".to_string(), "university of paris".to_string()),
        ]
    );
}

#[test]
fn neighborhood_excludes_two_hop_at_depth_one() {
    let index = curie_index();
    let (entities, _) = index.neighborhood(&["marie curie".to_string()], 1);
    assert!(!entities.contains(&"france".to_string()));
}

#[test]
fn neighborhood_includes_two_hop_at_depth_two() {
    let index = curie_index();
    let (entities, relations) = index.neighborhood(&["marie curie".to_string()], 2);
    assert!(entities.contains(&"france".to_string()));
    assert!(relations.contains(&("france".to_string(), "university of paris".to_string())));
}

#[test]
fn neighbors_of_empty_for_isolated_entity() {
    let mut index = LightRagIndex::new(128, true);
    index
        .insert_chunk(LightRagChunk::new(
            "c1",
            "Photosynthesis converts sunlight.",
        ))
        .unwrap();
    assert!(index.neighbors_of("photosynthesis").is_empty());
}

// ── Index accessors ───────────────────────────────────────────────────────────

#[test]
fn index_accessors_get_entity_relation_chunk() {
    let index = curie_index();
    assert!(index.get_entity("marie curie").is_some());
    assert!(index.get_entity("nonexistent").is_none());
    assert!(
        index
            .get_relation("marie curie", "university of paris")
            .is_some()
    );
    assert!(
        index
            .get_relation("university of paris", "marie curie")
            .is_some()
    );
    assert!(index.get_relation("marie curie", "nonexistent").is_none());
    assert!(index.get_chunk("c1").is_some());
    assert!(index.get_chunk("nonexistent").is_none());
    assert_eq!(index.entity_count(), 4);
    assert_eq!(index.relation_count(), 3);
    assert_eq!(index.chunk_count(), 3);
}

#[test]
fn index_entities_and_relations_iterators_cover_everything() {
    let index = curie_index();
    assert_eq!(index.entities().count(), index.entity_count());
    assert_eq!(index.relations().count(), index.relation_count());
}

// ── Config effects / context / determinism ──────────────────────────────────

#[test]
fn top_k_entities_truncates_local_seeds() {
    let config = LightRagConfig::default().with_top_k_entities(2);
    let mut engine = LightRagEngine::new(config);
    engine
        .insert_chunk(LightRagChunk::new(
            "a",
            "Alpha supports quantum encryption protocols.",
        ))
        .unwrap();
    engine
        .insert_chunk(LightRagChunk::new(
            "b",
            "Beta supports quantum encryption protocols.",
        ))
        .unwrap();
    engine
        .insert_chunk(LightRagChunk::new(
            "c",
            "Gamma supports quantum encryption protocols.",
        ))
        .unwrap();
    engine
        .insert_chunk(LightRagChunk::new(
            "d",
            "Delta supports quantum encryption protocols.",
        ))
        .unwrap();
    let result = engine.query("encryption", LightRagMode::Local).unwrap();
    assert_eq!(result.entities.len(), 2);
}

#[test]
fn top_k_relations_truncates_global_relations() {
    let config = LightRagConfig::default().with_top_k_relations(2);
    let mut engine = LightRagEngine::new(config);
    engine
        .insert_chunk(LightRagChunk::new(
            "a",
            "Alpha teaches Beta about encryption.",
        ))
        .unwrap();
    engine
        .insert_chunk(LightRagChunk::new(
            "b",
            "Gamma teaches Delta about encryption.",
        ))
        .unwrap();
    engine
        .insert_chunk(LightRagChunk::new(
            "c",
            "Epsilon teaches Zeta about encryption.",
        ))
        .unwrap();
    let result = engine
        .query("teaches encryption process", LightRagMode::Global)
        .unwrap();
    assert_eq!(result.relations.len(), 2);
}

#[test]
fn context_synthesis_contains_entities_and_relations() {
    let engine = curie_engine();
    let result = engine.query("Marie Curie", LightRagMode::Local).unwrap();
    assert!(result.context.contains("Entities:"));
    assert!(result.context.contains("Marie Curie"));
    assert!(result.context.contains("Relations:"));
}

#[test]
fn query_default_uses_config_mode() {
    let config = LightRagConfig::default().with_default_mode(LightRagMode::Local);
    let mut engine = LightRagEngine::new(config);
    engine
        .insert_chunk(LightRagChunk::new(
            "c1",
            "Marie Curie discovered radium. Marie Curie worked at the University of Paris.",
        ))
        .unwrap();
    let explicit = engine.query("Marie Curie", LightRagMode::Local).unwrap();
    let default = engine.query_default("Marie Curie").unwrap();
    assert_eq!(explicit, default);
    assert_eq!(default.mode, LightRagMode::Local);
}

#[test]
fn determinism_same_query_twice_identical_result() {
    let engine = curie_engine();
    let r1 = engine
        .query("Marie Curie research", LightRagMode::Hybrid)
        .unwrap();
    let r2 = engine
        .query("Marie Curie research", LightRagMode::Hybrid)
        .unwrap();
    assert_eq!(r1, r2);
}

#[test]
fn determinism_across_fresh_engines() {
    let r1 = curie_engine()
        .query("Marie Curie research", LightRagMode::Hybrid)
        .unwrap();
    let r2 = curie_engine()
        .query("Marie Curie research", LightRagMode::Hybrid)
        .unwrap();
    assert_eq!(r1, r2);
}

#[test]
fn source_chunks_are_resolved_and_deduplicated() {
    let engine = curie_engine();
    let result = engine.query("Marie Curie", LightRagMode::Local).unwrap();
    assert!(!result.source_chunks.is_empty());
    let mut ids: Vec<&str> = result.source_chunks.iter().map(|c| c.id.as_str()).collect();
    let before_len = ids.len();
    ids.sort_unstable();
    ids.dedup();
    assert_eq!(
        ids.len(),
        before_len,
        "source_chunks must not contain duplicates"
    );
}
