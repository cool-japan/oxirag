#![allow(
    clippy::float_cmp,
    clippy::similar_names,
    clippy::cast_precision_loss,
    clippy::useless_vec,
    clippy::map_unwrap_or,
    clippy::manual_range_contains,
    clippy::many_single_char_names,
    clippy::match_wildcard_for_single_variants,
    clippy::default_constructed_unit_structs,
    clippy::doc_markdown,
    clippy::too_many_lines
)]
//! Tests for the `entity_linking` module.

use super::catalog::EntityCatalog;
use super::linker::{EntityLinker, cosine, embed};
use super::types::{
    CanonicalEntity, EntityLinkConfig, EntityLinkError, EntityMention, LinkedEntity,
};

// ── helpers ─────────────────────────────────────────────────────────────────

/// Build a catalog whose single surface form "Mercury" is ambiguous between a
/// planet (`Q1`) and a chemical element (`Q2`).
fn ambiguous_mercury_catalog() -> EntityCatalog {
    let mut catalog = EntityCatalog::new();
    catalog.add(
        CanonicalEntity::new("Q1", "Mercury Planet")
            .with_alias("Mercury")
            .with_description("the smallest planet orbiting the sun in the solar system"),
    );
    catalog.add(
        CanonicalEntity::new("Q2", "Mercury Element")
            .with_alias("Mercury")
            .with_description("a silvery liquid metal chemical element with symbol Hg"),
    );
    catalog
}

// ── CanonicalEntity ─────────────────────────────────────────────────────────

#[test]
fn test_canonical_entity_new_defaults() {
    let entity = CanonicalEntity::new("id1", "Name One");
    assert_eq!(entity.id, "id1");
    assert_eq!(entity.name, "Name One");
    assert!(entity.aliases.is_empty(), "aliases start empty");
    assert!(entity.description.is_empty(), "description starts blank");
}

#[test]
fn test_canonical_entity_with_alias() {
    let entity = CanonicalEntity::new("id1", "Name").with_alias("Nick");
    assert_eq!(entity.aliases, vec!["Nick".to_string()]);
}

#[test]
fn test_canonical_entity_with_aliases() {
    let entity = CanonicalEntity::new("id1", "Name").with_aliases(["A", "B", "C"]);
    assert_eq!(
        entity.aliases,
        vec!["A".to_string(), "B".to_string(), "C".to_string()]
    );
}

#[test]
fn test_canonical_entity_with_description() {
    let entity = CanonicalEntity::new("id1", "Name").with_description("a gloss");
    assert_eq!(entity.description, "a gloss");
}

#[test]
fn test_canonical_entity_builder_chain() {
    let entity = CanonicalEntity::new("id1", "Name")
        .with_alias("X")
        .with_aliases(vec!["Y", "Z"])
        .with_description("d");
    assert_eq!(entity.aliases, vec!["X", "Y", "Z"]);
    assert_eq!(entity.description, "d");
}

// ── EntityMention ───────────────────────────────────────────────────────────

#[test]
fn test_entity_mention_new() {
    let mention = EntityMention::new("Paris", 4, 9);
    assert_eq!(mention.text, "Paris");
    assert_eq!(mention.start, 4);
    assert_eq!(mention.end, 9);
}

// ── LinkedEntity ────────────────────────────────────────────────────────────

#[test]
fn test_linked_entity_linked() {
    let linked = LinkedEntity::linked("Paris", "Q90", 1.0);
    assert_eq!(linked.mention, "Paris");
    assert_eq!(linked.entity_id.as_deref(), Some("Q90"));
    assert_eq!(linked.confidence, 1.0);
    assert!(linked.is_linked());
}

#[test]
fn test_linked_entity_nil() {
    let nil = LinkedEntity::nil("Unknown");
    assert_eq!(nil.mention, "Unknown");
    assert!(nil.entity_id.is_none());
    assert_eq!(nil.confidence, 0.0);
    assert!(!nil.is_linked());
}

// ── EntityLinkConfig ────────────────────────────────────────────────────────

#[test]
fn test_config_default_min_confidence() {
    let cfg = EntityLinkConfig::default();
    assert_eq!(
        cfg.min_confidence, 0.0,
        "default min_confidence should be 0.0"
    );
}

#[test]
fn test_config_default_dim() {
    let cfg = EntityLinkConfig::default();
    assert_eq!(cfg.dim, 128, "default dim should be 128");
}

#[test]
fn test_config_new_equals_default() {
    assert_eq!(EntityLinkConfig::new(), EntityLinkConfig::default());
}

#[test]
fn test_config_with_min_confidence() {
    let cfg = EntityLinkConfig::new().with_min_confidence(0.3);
    assert_eq!(cfg.min_confidence, 0.3);
}

#[test]
fn test_config_with_dim() {
    let cfg = EntityLinkConfig::new().with_dim(64);
    assert_eq!(cfg.dim, 64);
}

#[test]
fn test_config_builder_chain() {
    let cfg = EntityLinkConfig::new()
        .with_min_confidence(0.5)
        .with_dim(256);
    assert_eq!(cfg.min_confidence, 0.5);
    assert_eq!(cfg.dim, 256);
}

// ── EntityCatalog: add / len / is_empty ─────────────────────────────────────

#[test]
fn test_catalog_new_is_empty() {
    let catalog = EntityCatalog::new();
    assert!(catalog.is_empty());
    assert_eq!(catalog.len(), 0);
}

#[test]
fn test_catalog_default_is_empty() {
    let catalog = EntityCatalog::default();
    assert!(catalog.is_empty());
}

#[test]
fn test_catalog_add_increments_len() {
    let mut catalog = EntityCatalog::new();
    catalog.add(CanonicalEntity::new("a", "Apple"));
    catalog.add(CanonicalEntity::new("b", "Banana"));
    assert_eq!(catalog.len(), 2);
    assert!(!catalog.is_empty());
}

#[test]
fn test_catalog_entity_access() {
    let mut catalog = EntityCatalog::new();
    catalog.add(CanonicalEntity::new("a", "Apple"));
    assert_eq!(catalog.entity(0).map(|e| e.name.as_str()), Some("Apple"));
    assert!(
        catalog.entity(1).is_none(),
        "out-of-range index yields None"
    );
}

#[test]
fn test_catalog_entities_slice() {
    let mut catalog = EntityCatalog::new();
    catalog.add(CanonicalEntity::new("a", "Apple"));
    catalog.add(CanonicalEntity::new("b", "Banana"));
    let names: Vec<&str> = catalog.entities().iter().map(|e| e.name.as_str()).collect();
    assert_eq!(names, vec!["Apple", "Banana"]);
}

// ── EntityCatalog: candidates ───────────────────────────────────────────────

#[test]
fn test_candidates_by_name() {
    let mut catalog = EntityCatalog::new();
    catalog.add(CanonicalEntity::new("a", "Apple"));
    assert_eq!(catalog.candidates("Apple"), vec![0]);
}

#[test]
fn test_candidates_by_alias() {
    let mut catalog = EntityCatalog::new();
    catalog.add(CanonicalEntity::new("a", "Apple Inc").with_alias("Apple"));
    assert_eq!(catalog.candidates("Apple"), vec![0]);
}

#[test]
fn test_candidates_case_insensitive_name() {
    let mut catalog = EntityCatalog::new();
    catalog.add(CanonicalEntity::new("a", "Apple"));
    assert_eq!(catalog.candidates("apple"), vec![0]);
    assert_eq!(catalog.candidates("APPLE"), vec![0]);
    assert_eq!(catalog.candidates("ApPlE"), vec![0]);
}

#[test]
fn test_candidates_case_insensitive_alias() {
    let mut catalog = EntityCatalog::new();
    catalog.add(CanonicalEntity::new("a", "Apple Inc").with_alias("AAPL"));
    assert_eq!(catalog.candidates("aapl"), vec![0]);
}

#[test]
fn test_candidates_trims_surface_form() {
    let mut catalog = EntityCatalog::new();
    catalog.add(CanonicalEntity::new("a", "Apple"));
    assert_eq!(catalog.candidates("  Apple  "), vec![0]);
}

#[test]
fn test_candidates_none_for_unknown() {
    let mut catalog = EntityCatalog::new();
    catalog.add(CanonicalEntity::new("a", "Apple"));
    assert!(catalog.candidates("Orange").is_empty());
}

#[test]
fn test_candidates_multiple_shared_alias() {
    let catalog = ambiguous_mercury_catalog();
    let cands = catalog.candidates("Mercury");
    assert_eq!(
        cands,
        vec![0, 1],
        "shared alias returns both entities in order"
    );
}

#[test]
fn test_candidates_no_duplicate_when_alias_equals_name() {
    let mut catalog = EntityCatalog::new();
    catalog.add(CanonicalEntity::new("a", "Apple").with_alias("Apple"));
    assert_eq!(
        catalog.candidates("Apple"),
        vec![0],
        "name == alias must not duplicate the candidate"
    );
}

#[test]
fn test_candidates_ascending_order() {
    let mut catalog = EntityCatalog::new();
    catalog.add(CanonicalEntity::new("a", "Shared"));
    catalog.add(CanonicalEntity::new("b", "Other"));
    catalog.add(CanonicalEntity::new("c", "Shared"));
    assert_eq!(catalog.candidates("Shared"), vec![0, 2]);
}

// ── embed / cosine ──────────────────────────────────────────────────────────

#[test]
fn test_embed_dim_zero_empty() {
    assert!(embed("anything here", 0).is_empty());
}

#[test]
fn test_embed_length_matches_dim() {
    assert_eq!(embed("hello world", 64).len(), 64);
}

#[test]
fn test_embed_normalised() {
    let v = embed("alpha beta gamma", 128);
    let norm: f32 = v.iter().map(|x| x * x).sum::<f32>().sqrt();
    assert!(
        (norm - 1.0).abs() < 1e-5,
        "embedding should be L2-normalised"
    );
}

#[test]
fn test_embed_case_insensitive() {
    assert_eq!(embed("Hello World", 64), embed("hello world", 64));
}

#[test]
fn test_cosine_identical_is_one() {
    let v = embed("planet orbiting sun", 128);
    assert!((cosine(&v, &v) - 1.0).abs() < 1e-5);
}

#[test]
fn test_cosine_mismatched_length_zero() {
    assert_eq!(cosine(&[1.0, 0.0], &[1.0]), 0.0);
}

#[test]
fn test_cosine_empty_zero() {
    assert_eq!(cosine(&[], &[]), 0.0);
}

#[test]
fn test_cosine_disjoint_vocab_zero() {
    let a = embed("planet orbiting sun", 256);
    let b = embed("silvery liquid metal", 256);
    assert!(
        cosine(&a, &b).abs() < 1e-6,
        "disjoint vocab should be orthogonal"
    );
}

// ── detect_mentions ─────────────────────────────────────────────────────────

#[test]
fn test_detect_mentions_single_word() {
    let linker = EntityLinker::new(EntityLinkConfig::default(), EntityCatalog::new());
    let mentions = linker.detect_mentions("we visit Paris today.");
    let texts: Vec<&str> = mentions.iter().map(|m| m.text.as_str()).collect();
    assert_eq!(texts, vec!["Paris"]);
}

#[test]
fn test_detect_mentions_contiguous_joined() {
    let linker = EntityLinker::new(EntityLinkConfig::default(), EntityCatalog::new());
    let mentions = linker.detect_mentions("New York is large.");
    let texts: Vec<&str> = mentions.iter().map(|m| m.text.as_str()).collect();
    assert_eq!(texts, vec!["New York"], "contiguous capitalised words join");
}

#[test]
fn test_detect_mentions_multiple_spans() {
    let linker = EntityLinker::new(EntityLinkConfig::default(), EntityCatalog::new());
    let mentions = linker.detect_mentions("today Alice met Bob in London.");
    let texts: Vec<&str> = mentions.iter().map(|m| m.text.as_str()).collect();
    assert_eq!(texts, vec!["Alice", "Bob", "London"]);
}

#[test]
fn test_detect_mentions_skips_lowercase() {
    let linker = EntityLinker::new(EntityLinkConfig::default(), EntityCatalog::new());
    let mentions = linker.detect_mentions("the cat sat on the mat");
    assert!(
        mentions.is_empty(),
        "no capitalised words means no mentions"
    );
}

#[test]
fn test_detect_mentions_skips_single_char() {
    let linker = EntityLinker::new(EntityLinkConfig::default(), EntityCatalog::new());
    let mentions = linker.detect_mentions("A quick fox");
    assert!(
        mentions.is_empty(),
        "single capital letter 'A' is below the length floor"
    );
}

#[test]
fn test_detect_mentions_offsets() {
    let linker = EntityLinker::new(EntityLinkConfig::default(), EntityCatalog::new());
    let text = "we visit Paris now";
    let mentions = linker.detect_mentions(text);
    assert_eq!(mentions.len(), 1);
    let m = &mentions[0];
    assert_eq!(&text[m.start..m.end], "Paris");
}

#[test]
fn test_detect_mentions_trailing_punctuation_stripped() {
    let linker = EntityLinker::new(EntityLinkConfig::default(), EntityCatalog::new());
    let mentions = linker.detect_mentions("we love Rust and Cargo.");
    let texts: Vec<&str> = mentions.iter().map(|m| m.text.as_str()).collect();
    // Trailing punctuation is stripped so the surface form is lookup-ready.
    assert_eq!(texts, vec!["Rust", "Cargo"]);
}

#[test]
fn test_detect_mentions_internal_punctuation_kept() {
    let linker = EntityLinker::new(EntityLinkConfig::default(), EntityCatalog::new());
    // "Rust," and "Cargo" are contiguous capitalised words and join; the comma
    // sits inside the joined run, so it is preserved.
    let mentions = linker.detect_mentions("we use Rust, Cargo daily.");
    let texts: Vec<&str> = mentions.iter().map(|m| m.text.as_str()).collect();
    assert_eq!(texts, vec!["Rust, Cargo"]);
}

// ── link: single candidate ──────────────────────────────────────────────────

#[test]
fn test_link_single_candidate_sets_id() {
    let mut catalog = EntityCatalog::new();
    catalog.add(CanonicalEntity::new("Q90", "Paris").with_description("capital of France"));
    let linker = EntityLinker::new(EntityLinkConfig::default(), catalog);
    let linked = linker.link("Paris", "a trip to France");
    assert_eq!(linked.entity_id.as_deref(), Some("Q90"));
    assert_eq!(
        linked.confidence, 1.0,
        "single candidate links at full confidence"
    );
}

#[test]
fn test_link_single_candidate_via_alias() {
    let mut catalog = EntityCatalog::new();
    catalog.add(
        CanonicalEntity::new("Q95", "Apple Inc")
            .with_alias("Apple")
            .with_description("technology company"),
    );
    let linker = EntityLinker::new(EntityLinkConfig::default(), catalog);
    let linked = linker.link("Apple", "the company released a phone");
    assert_eq!(linked.entity_id.as_deref(), Some("Q95"));
}

// ── link: ambiguous, disambiguated by context ───────────────────────────────

#[test]
fn test_link_ambiguous_context_planet() {
    let linker = EntityLinker::new(EntityLinkConfig::default(), ambiguous_mercury_catalog());
    let linked = linker.link(
        "Mercury",
        "the planet orbiting the sun in the solar system is hot",
    );
    assert_eq!(
        linked.entity_id.as_deref(),
        Some("Q1"),
        "planet context should select the planet entity"
    );
    assert!(linked.confidence > 0.0);
}

#[test]
fn test_link_ambiguous_context_element() {
    let linker = EntityLinker::new(EntityLinkConfig::default(), ambiguous_mercury_catalog());
    let linked = linker.link(
        "Mercury",
        "the silvery liquid metal chemical element is toxic",
    );
    assert_eq!(
        linked.entity_id.as_deref(),
        Some("Q2"),
        "element context should select the element entity"
    );
}

#[test]
fn test_link_ambiguous_confidence_in_unit_range() {
    let linker = EntityLinker::new(EntityLinkConfig::default(), ambiguous_mercury_catalog());
    let linked = linker.link("Mercury", "the planet orbiting the sun");
    assert!(linked.confidence >= 0.0 && linked.confidence <= 1.0);
}

#[test]
fn test_link_ambiguous_picks_a_candidate_with_blank_context() {
    // With no overlapping context tokens, all scores are 0.0; the linker still
    // resolves to a candidate (the first, by tie-break) rather than NIL.
    let linker = EntityLinker::new(EntityLinkConfig::default(), ambiguous_mercury_catalog());
    let linked = linker.link("Mercury", "");
    assert_eq!(linked.entity_id.as_deref(), Some("Q1"));
}

// ── link: NIL outcomes ──────────────────────────────────────────────────────

#[test]
fn test_link_no_candidate_is_nil() {
    let mut catalog = EntityCatalog::new();
    catalog.add(CanonicalEntity::new("Q1", "Paris"));
    let linker = EntityLinker::new(EntityLinkConfig::default(), catalog);
    let linked = linker.link("Berlin", "a European capital");
    assert!(linked.entity_id.is_none(), "unknown mention links to NIL");
    assert_eq!(linked.confidence, 0.0);
}

#[test]
fn test_link_below_min_confidence_is_nil() {
    // A demanding floor rejects the ambiguous best when context overlap is weak.
    let cfg = EntityLinkConfig::new().with_min_confidence(0.9);
    let linker = EntityLinker::new(cfg, ambiguous_mercury_catalog());
    let linked = linker.link("Mercury", "unrelated words about gardening tools");
    assert!(
        linked.entity_id.is_none(),
        "best confidence below floor should yield NIL"
    );
}

#[test]
fn test_link_above_min_confidence_links() {
    // Single candidate scores 1.0, which clears even a strict floor.
    let cfg = EntityLinkConfig::new().with_min_confidence(0.9);
    let mut catalog = EntityCatalog::new();
    catalog.add(CanonicalEntity::new("Q90", "Paris").with_description("capital of France"));
    let linker = EntityLinker::new(cfg, catalog);
    let linked = linker.link("Paris", "France");
    assert_eq!(linked.entity_id.as_deref(), Some("Q90"));
}

#[test]
fn test_link_zero_floor_accepts_zero_score() {
    // The default floor of 0.0 accepts an ambiguous pick even at score 0.0.
    let linker = EntityLinker::new(EntityLinkConfig::default(), ambiguous_mercury_catalog());
    let linked = linker.link("Mercury", "");
    assert!(
        linked.is_linked(),
        "zero floor accepts a zero-score best candidate"
    );
}

// ── link_all ────────────────────────────────────────────────────────────────

#[test]
fn test_link_all_over_sentence() {
    let mut catalog = EntityCatalog::new();
    catalog.add(CanonicalEntity::new("Q90", "Paris").with_description("capital of France"));
    catalog.add(CanonicalEntity::new("Q64", "Berlin").with_description("capital of Germany"));
    let linker = EntityLinker::new(EntityLinkConfig::default(), catalog);
    let results = linker
        .link_all("Paris and Berlin are capitals.")
        .expect("non-empty catalog links");
    let ids: Vec<Option<&str>> = results.iter().map(|r| r.entity_id.as_deref()).collect();
    assert_eq!(ids, vec![Some("Q90"), Some("Q64")]);
}

#[test]
fn test_link_all_mixes_linked_and_nil() {
    let mut catalog = EntityCatalog::new();
    catalog.add(CanonicalEntity::new("Q90", "Paris").with_description("capital of France"));
    let linker = EntityLinker::new(EntityLinkConfig::default(), catalog);
    let results = linker
        .link_all("Paris and Atlantis are places.")
        .expect("non-empty catalog links");
    assert_eq!(results.len(), 2);
    assert_eq!(results[0].entity_id.as_deref(), Some("Q90"));
    assert!(results[1].entity_id.is_none(), "unknown 'Atlantis' is NIL");
}

#[test]
fn test_link_all_empty_text_no_mentions() {
    let mut catalog = EntityCatalog::new();
    catalog.add(CanonicalEntity::new("Q1", "Paris"));
    let linker = EntityLinker::new(EntityLinkConfig::default(), catalog);
    let results = linker.link_all("nothing capitalised here").expect("links");
    assert!(results.is_empty());
}

// ── EmptyCatalog error ──────────────────────────────────────────────────────

#[test]
fn test_link_all_empty_catalog_errors() {
    let linker = EntityLinker::new(EntityLinkConfig::default(), EntityCatalog::new());
    let err = linker.link_all("Paris is nice").unwrap_err();
    assert!(matches!(err, EntityLinkError::EmptyCatalog));
}

#[test]
fn test_empty_catalog_error_message() {
    let err = EntityLinkError::EmptyCatalog;
    assert_eq!(err.to_string(), "catalog is empty");
}

// ── accessors ───────────────────────────────────────────────────────────────

#[test]
fn test_linker_config_accessor() {
    let cfg = EntityLinkConfig::new().with_min_confidence(0.42);
    let linker = EntityLinker::new(cfg, EntityCatalog::new());
    assert_eq!(linker.config().min_confidence, 0.42);
}

#[test]
fn test_linker_catalog_accessor() {
    let mut catalog = EntityCatalog::new();
    catalog.add(CanonicalEntity::new("a", "Apple"));
    let linker = EntityLinker::new(EntityLinkConfig::default(), catalog);
    assert_eq!(linker.catalog().len(), 1);
}

// ── determinism ─────────────────────────────────────────────────────────────

#[test]
fn test_link_is_deterministic() {
    let linker = EntityLinker::new(EntityLinkConfig::default(), ambiguous_mercury_catalog());
    let a = linker.link("Mercury", "the planet orbiting the sun");
    let b = linker.link("Mercury", "the planet orbiting the sun");
    assert_eq!(a, b, "linking the same inputs must be deterministic");
}

#[test]
fn test_link_all_is_deterministic() {
    let mut catalog = EntityCatalog::new();
    catalog.add(CanonicalEntity::new("Q90", "Paris").with_description("capital of France"));
    catalog.add(CanonicalEntity::new("Q64", "Berlin").with_description("capital of Germany"));
    let linker = EntityLinker::new(EntityLinkConfig::default(), catalog);
    let a = linker.link_all("Paris and Berlin").unwrap();
    let b = linker.link_all("Paris and Berlin").unwrap();
    assert_eq!(a, b);
}

#[test]
fn test_embed_is_deterministic() {
    assert_eq!(
        embed("planet orbiting sun", 128),
        embed("planet orbiting sun", 128)
    );
}

#[test]
fn test_detect_mentions_is_deterministic() {
    let linker = EntityLinker::new(EntityLinkConfig::default(), EntityCatalog::new());
    let a = linker.detect_mentions("Alice met Bob in New York");
    let b = linker.detect_mentions("Alice met Bob in New York");
    assert_eq!(a, b);
}
