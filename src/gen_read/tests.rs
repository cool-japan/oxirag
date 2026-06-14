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

//! Tests for the `gen_read` module.

use super::engine::GenReadEngine;
use super::types::{
    ContextGenerator, GenReadConfig, GenReadError, GeneratedDoc, MockContextGenerator,
};

// ── Test fixtures ────────────────────────────────────────────────────────────────

/// A scripted generator with five topically clusterable documents: three about
/// plants/photosynthesis and two about computers/CPUs.
fn scripted_generator() -> MockContextGenerator {
    MockContextGenerator::new(vec![
        "Photosynthesis converts sunlight into chemical energy in green plants.".to_string(),
        "Chlorophyll inside the chloroplast captures sunlight for photosynthesis.".to_string(),
        "Plants release oxygen as a byproduct of photosynthesis in the leaves.".to_string(),
        "A computer processor executes instructions fetched from memory.".to_string(),
        "The CPU cache stores frequently accessed instructions and memory data.".to_string(),
    ])
}

/// A generator whose output varies purely by `index` for diversity testing.
struct IndexedGenerator;

impl ContextGenerator for IndexedGenerator {
    fn generate(&self, query: &str, index: usize) -> String {
        format!("document number {index} answering the query {query}")
    }
}

// ── GenReadConfig: defaults ──────────────────────────────────────────────────────

#[test]
fn config_default_num_docs() {
    assert_eq!(GenReadConfig::default().num_docs, 5);
}

#[test]
fn config_default_num_clusters() {
    assert_eq!(GenReadConfig::default().num_clusters, 2);
}

#[test]
fn config_default_dim() {
    assert_eq!(GenReadConfig::default().dim, 128);
}

#[test]
fn config_default_join_separator() {
    assert_eq!(GenReadConfig::default().join_separator, "\n\n");
}

#[test]
fn config_new_equals_default() {
    assert_eq!(GenReadConfig::new(), GenReadConfig::default());
}

// ── GenReadConfig: builders ──────────────────────────────────────────────────────

#[test]
fn config_with_num_docs() {
    let cfg = GenReadConfig::new().with_num_docs(8);
    assert_eq!(cfg.num_docs, 8);
}

#[test]
fn config_with_num_clusters() {
    let cfg = GenReadConfig::new().with_num_clusters(4);
    assert_eq!(cfg.num_clusters, 4);
}

#[test]
fn config_with_dim() {
    let cfg = GenReadConfig::new().with_dim(64);
    assert_eq!(cfg.dim, 64);
}

#[test]
fn config_with_join_separator() {
    let cfg = GenReadConfig::new().with_join_separator(" --- ");
    assert_eq!(cfg.join_separator, " --- ");
}

#[test]
fn config_builders_chain() {
    let cfg = GenReadConfig::new()
        .with_num_docs(7)
        .with_num_clusters(3)
        .with_dim(256)
        .with_join_separator("|");
    assert_eq!(cfg.num_docs, 7);
    assert_eq!(cfg.num_clusters, 3);
    assert_eq!(cfg.dim, 256);
    assert_eq!(cfg.join_separator, "|");
}

#[test]
fn config_builders_do_not_touch_other_fields() {
    let cfg = GenReadConfig::new().with_num_docs(9);
    assert_eq!(cfg.num_clusters, 2);
    assert_eq!(cfg.dim, 128);
    assert_eq!(cfg.join_separator, "\n\n");
}

// ── run: document count ──────────────────────────────────────────────────────────

#[test]
fn run_generates_exactly_num_docs() {
    let engine = GenReadEngine::new(GenReadConfig::new());
    let out = engine
        .run("how do plants make energy?", &scripted_generator())
        .unwrap();
    assert_eq!(out.documents.len(), 5);
}

#[test]
fn run_generates_exactly_num_docs_custom() {
    let engine = GenReadEngine::new(GenReadConfig::new().with_num_docs(3));
    let out = engine.run("query text", &IndexedGenerator).unwrap();
    assert_eq!(out.documents.len(), 3);
}

#[test]
fn run_generates_exactly_num_docs_large() {
    let engine = GenReadEngine::new(GenReadConfig::new().with_num_docs(12).with_num_clusters(4));
    let out = engine.run("query text", &IndexedGenerator).unwrap();
    assert_eq!(out.documents.len(), 12);
}

#[test]
fn run_num_docs_zero_clamps_to_one() {
    let engine = GenReadEngine::new(GenReadConfig::new().with_num_docs(0));
    let out = engine.run("query", &IndexedGenerator).unwrap();
    assert_eq!(out.documents.len(), 1);
}

// ── run: cluster ids ─────────────────────────────────────────────────────────────

#[test]
fn run_each_doc_cluster_id_below_num_clusters() {
    let engine = GenReadEngine::new(GenReadConfig::new().with_num_clusters(2));
    let out = engine
        .run("photosynthesis and processors", &scripted_generator())
        .unwrap();
    for doc in &out.documents {
        assert!(doc.cluster_id < 2, "cluster_id {} not < 2", doc.cluster_id);
    }
}

#[test]
fn run_each_doc_cluster_id_below_actual_cluster_count() {
    let engine = GenReadEngine::new(GenReadConfig::new().with_num_clusters(3));
    let out = engine
        .run("photosynthesis and processors", &scripted_generator())
        .unwrap();
    for doc in &out.documents {
        assert!(
            doc.cluster_id < out.clusters,
            "cluster_id {} not < clusters {}",
            doc.cluster_id,
            out.clusters
        );
    }
}

#[test]
fn run_clusters_count_at_most_num_clusters() {
    let engine = GenReadEngine::new(GenReadConfig::new().with_num_clusters(2));
    let out = engine
        .run("photosynthesis and processors", &scripted_generator())
        .unwrap();
    assert!(out.clusters <= 2);
}

#[test]
fn run_clusters_count_at_most_num_clusters_high() {
    let engine = GenReadEngine::new(GenReadConfig::new().with_num_clusters(3));
    let out = engine
        .run("photosynthesis and processors", &scripted_generator())
        .unwrap();
    assert!(out.clusters <= 3);
}

#[test]
fn run_clusters_at_least_one() {
    let engine = GenReadEngine::new(GenReadConfig::new());
    let out = engine.run("any query", &scripted_generator()).unwrap();
    assert!(out.clusters >= 1);
}

#[test]
fn run_clusters_never_exceed_documents() {
    let engine = GenReadEngine::new(GenReadConfig::new().with_num_docs(2).with_num_clusters(5));
    let out = engine.run("query", &IndexedGenerator).unwrap();
    assert!(out.clusters <= out.documents.len());
}

#[test]
fn run_topical_split_forms_two_clusters() {
    // Three plant docs vs. two computer docs should split into two clusters.
    let engine = GenReadEngine::new(GenReadConfig::new().with_num_clusters(2));
    let out = engine
        .run("explain the topics", &scripted_generator())
        .unwrap();
    assert_eq!(out.clusters, 2);
}

#[test]
fn run_cluster_ids_cover_zero_to_count() {
    let engine = GenReadEngine::new(GenReadConfig::new().with_num_clusters(2));
    let out = engine
        .run("explain the topics", &scripted_generator())
        .unwrap();
    let mut seen = vec![false; out.clusters];
    for doc in &out.documents {
        seen[doc.cluster_id] = true;
    }
    // Every cluster is non-empty, so every id in 0..clusters must appear.
    assert!(
        seen.iter().all(|&s| s),
        "some cluster id was never assigned"
    );
}

// ── run: context assembly ────────────────────────────────────────────────────────

#[test]
fn run_context_non_empty() {
    let engine = GenReadEngine::new(GenReadConfig::new());
    let out = engine
        .run("how do plants make energy?", &scripted_generator())
        .unwrap();
    assert!(!out.context.is_empty());
}

#[test]
fn run_context_joins_one_representative_per_cluster() {
    let engine = GenReadEngine::new(GenReadConfig::new().with_num_clusters(2));
    let out = engine
        .run("explain the topics", &scripted_generator())
        .unwrap();
    // With two clusters and the default separator, the context is two
    // representatives joined by exactly one separator occurrence.
    let parts: Vec<&str> = out.context.split("\n\n").collect();
    assert_eq!(parts.len(), out.clusters);
    assert_eq!(parts.len(), 2);
}

#[test]
fn run_context_uses_custom_separator() {
    let engine = GenReadEngine::new(
        GenReadConfig::new()
            .with_num_clusters(2)
            .with_join_separator(" ||| "),
    );
    let out = engine
        .run("explain the topics", &scripted_generator())
        .unwrap();
    assert!(out.context.contains(" ||| "));
    let parts: Vec<&str> = out.context.split(" ||| ").collect();
    assert_eq!(parts.len(), out.clusters);
}

#[test]
fn run_context_representatives_are_generated_docs() {
    let engine = GenReadEngine::new(GenReadConfig::new().with_num_clusters(2));
    let out = engine
        .run("explain the topics", &scripted_generator())
        .unwrap();
    // Each piece of the context must be the content of one generated document.
    let contents: Vec<&str> = out.documents.iter().map(|d| d.content.as_str()).collect();
    for part in out.context.split("\n\n") {
        assert!(
            contents.contains(&part),
            "context piece not a generated doc: {part}"
        );
    }
}

#[test]
fn run_single_cluster_context_has_no_separator() {
    // One cluster ⇒ a single representative ⇒ default separator absent.
    let engine = GenReadEngine::new(GenReadConfig::new().with_num_clusters(1));
    let out = engine
        .run("explain the topics", &scripted_generator())
        .unwrap();
    assert_eq!(out.clusters, 1);
    assert!(!out.context.contains("\n\n"));
    assert!(!out.context.is_empty());
}

#[test]
fn run_context_pieces_count_matches_clusters() {
    let engine = GenReadEngine::new(GenReadConfig::new().with_num_clusters(3));
    let out = engine
        .run("explain the topics", &scripted_generator())
        .unwrap();
    let pieces = out.context.split("\n\n").count();
    assert_eq!(pieces, out.clusters);
}

// ── run: representative selection determinism ────────────────────────────────────

#[test]
fn run_representative_selection_deterministic() {
    let engine = GenReadEngine::new(GenReadConfig::new().with_num_clusters(2));
    let generator = scripted_generator();
    let a = engine.run("explain the topics", &generator).unwrap();
    let b = engine.run("explain the topics", &generator).unwrap();
    assert_eq!(a.context, b.context);
}

#[test]
fn run_representative_is_most_central_singleton() {
    // A two-cluster split where one cluster is a singleton: its representative
    // must be that single document. With `num_docs = 3` and `num_clusters = 2`
    // the deterministic seeds are documents 0 and 1; placing the lone "zeta"
    // document first makes it its own cluster while the two "alpha" documents
    // group together.
    let generator = MockContextGenerator::new(vec![
        "completely unrelated zeta eta theta iota".to_string(),
        "alpha beta gamma delta epsilon".to_string(),
        "alpha beta gamma delta epsilon".to_string(),
    ]);
    let engine = GenReadEngine::new(GenReadConfig::new().with_num_docs(3).with_num_clusters(2));
    let out = engine.run("query", &generator).unwrap();
    assert_eq!(out.clusters, 2);
    // The singleton cluster's representative is its only member's content.
    assert!(out.context.contains("zeta eta theta iota"));
    // The shared cluster contributes exactly one "alpha" representative.
    assert!(out.context.contains("alpha beta gamma delta epsilon"));
}

// ── run: determinism ─────────────────────────────────────────────────────────────

#[test]
fn run_determinism_identical_output_documents() {
    let engine = GenReadEngine::new(GenReadConfig::new());
    let generator = scripted_generator();
    let a = engine
        .run("how do plants make energy?", &generator)
        .unwrap();
    let b = engine
        .run("how do plants make energy?", &generator)
        .unwrap();
    assert_eq!(a.documents, b.documents);
}

#[test]
fn run_determinism_identical_cluster_count() {
    let engine = GenReadEngine::new(GenReadConfig::new());
    let generator = scripted_generator();
    let a = engine.run("query", &generator).unwrap();
    let b = engine.run("query", &generator).unwrap();
    assert_eq!(a.clusters, b.clusters);
}

#[test]
fn run_determinism_identical_context() {
    let engine = GenReadEngine::new(GenReadConfig::new());
    let generator = scripted_generator();
    let a = engine.run("query", &generator).unwrap();
    let b = engine.run("query", &generator).unwrap();
    assert_eq!(a.context, b.context);
}

#[test]
fn run_determinism_across_engine_instances() {
    let generator = scripted_generator();
    let a = GenReadEngine::new(GenReadConfig::new())
        .run("q", &generator)
        .unwrap();
    let b = GenReadEngine::new(GenReadConfig::new())
        .run("q", &generator)
        .unwrap();
    assert_eq!(a.context, b.context);
    assert_eq!(a.documents, b.documents);
}

#[test]
fn run_determinism_indexed_generator() {
    let engine = GenReadEngine::new(GenReadConfig::new().with_num_docs(6).with_num_clusters(3));
    let a = engine.run("query", &IndexedGenerator).unwrap();
    let b = engine.run("query", &IndexedGenerator).unwrap();
    assert_eq!(a.documents, b.documents);
    assert_eq!(a.context, b.context);
    assert_eq!(a.clusters, b.clusters);
}

// ── run: empty query error ───────────────────────────────────────────────────────

#[test]
fn run_empty_query_errors() {
    let engine = GenReadEngine::new(GenReadConfig::new());
    let err = engine.run("", &scripted_generator()).unwrap_err();
    assert!(matches!(err, GenReadError::EmptyQuery));
}

#[test]
fn run_whitespace_query_errors() {
    let engine = GenReadEngine::new(GenReadConfig::new());
    let err = engine.run("   \t\n  ", &scripted_generator()).unwrap_err();
    assert!(matches!(err, GenReadError::EmptyQuery));
}

#[test]
fn empty_query_error_display() {
    let err = GenReadError::EmptyQuery;
    assert_eq!(err.to_string(), "query must not be empty");
}

#[test]
fn run_non_empty_query_ok() {
    let engine = GenReadEngine::new(GenReadConfig::new());
    assert!(engine.run("valid", &scripted_generator()).is_ok());
}

// ── MockContextGenerator: scripted docs by index ─────────────────────────────────

#[test]
fn mock_generator_returns_scripted_by_index() {
    let generator = MockContextGenerator::new(vec![
        "first".to_string(),
        "second".to_string(),
        "third".to_string(),
    ]);
    assert_eq!(generator.generate("q", 0), "first");
    assert_eq!(generator.generate("q", 1), "second");
    assert_eq!(generator.generate("q", 2), "third");
}

#[test]
fn mock_generator_wraps_on_overflow() {
    let generator = MockContextGenerator::new(vec!["a".to_string(), "b".to_string()]);
    assert_eq!(generator.generate("q", 2), "a");
    assert_eq!(generator.generate("q", 3), "b");
    assert_eq!(generator.generate("q", 4), "a");
}

#[test]
fn mock_generator_ignores_query_text() {
    let generator = MockContextGenerator::new(vec!["scripted".to_string()]);
    assert_eq!(generator.generate("anything", 0), "scripted");
    assert_eq!(generator.generate("different", 0), "scripted");
}

#[test]
fn mock_generator_empty_script_synthesizes_non_empty() {
    let generator = MockContextGenerator::new(vec![]);
    let d0 = generator.generate("the query", 0);
    let d1 = generator.generate("the query", 1);
    assert!(!d0.is_empty());
    assert!(!d1.is_empty());
    assert_ne!(d0, d1, "empty-script docs should vary by index");
    assert!(d0.contains("the query"));
}

#[test]
fn mock_generator_docs_accessor() {
    let generator = MockContextGenerator::new(vec!["x".to_string(), "y".to_string()]);
    assert_eq!(generator.docs(), &["x".to_string(), "y".to_string()]);
}

#[test]
fn mock_generator_default_is_empty() {
    let generator = MockContextGenerator::default();
    assert!(generator.docs().is_empty());
    // Default (empty) still produces non-empty synthesized output.
    assert!(!generator.generate("q", 0).is_empty());
}

#[test]
fn mock_generator_deterministic() {
    let generator = scripted_generator();
    assert_eq!(generator.generate("q", 0), generator.generate("q", 0));
    assert_eq!(generator.generate("q", 3), generator.generate("q", 3));
}

// ── GeneratedDoc ─────────────────────────────────────────────────────────────────

#[test]
fn generated_doc_fields_accessible() {
    let doc = GeneratedDoc {
        content: "hello".to_string(),
        cluster_id: 1,
    };
    assert_eq!(doc.content, "hello");
    assert_eq!(doc.cluster_id, 1);
}

#[test]
fn generated_doc_equality() {
    let a = GeneratedDoc {
        content: "x".to_string(),
        cluster_id: 0,
    };
    let b = GeneratedDoc {
        content: "x".to_string(),
        cluster_id: 0,
    };
    assert_eq!(a, b);
}

// ── Output structure sanity ──────────────────────────────────────────────────────

#[test]
fn run_documents_preserve_generation_order() {
    // With a single cluster the documents stay in generation order (indices
    // 0..num_docs), so their contents match the script positions.
    let generator = scripted_generator();
    let engine = GenReadEngine::new(GenReadConfig::new().with_num_clusters(1));
    let out = engine.run("query", &generator).unwrap();
    for (i, doc) in out.documents.iter().enumerate() {
        assert_eq!(doc.content, generator.generate("query", i));
    }
}

#[test]
fn run_with_indexed_generator_distinct_documents() {
    let engine = GenReadEngine::new(GenReadConfig::new().with_num_docs(5));
    let out = engine.run("topic", &IndexedGenerator).unwrap();
    // IndexedGenerator embeds the index, so all five documents differ.
    let mut contents: Vec<&str> = out.documents.iter().map(|d| d.content.as_str()).collect();
    contents.sort_unstable();
    contents.dedup();
    assert_eq!(contents.len(), 5);
}
