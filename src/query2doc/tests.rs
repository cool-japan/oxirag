//! Tests for the `query2doc` module.

use super::expander::Query2DocExpander;
use super::types::{
    MockPseudoDocGenerator, PseudoDocGenerator, PseudoDocument, Query2DocConfig, Query2DocError,
    Query2DocVariant,
};

// ── Test helpers ──────────────────────────────────────────────────────────────

/// A [`PseudoDocGenerator`] that always returns the same fixed string,
/// regardless of the query. Used to make repetition/weighting/truncation
/// assertions unambiguous (the [`MockPseudoDocGenerator`]'s fallback template
/// embeds the query itself, which would otherwise confound exact-count checks).
#[derive(Debug, Clone)]
struct FixedGenerator(String);

impl FixedGenerator {
    fn new(text: impl Into<String>) -> Self {
        Self(text.into())
    }
}

impl PseudoDocGenerator for FixedGenerator {
    fn generate(&self, _query: &str) -> String {
        self.0.clone()
    }
}

fn default_expander_fixed(pseudo_doc: &str) -> Query2DocExpander<FixedGenerator> {
    Query2DocExpander::new(Query2DocConfig::default(), FixedGenerator::new(pseudo_doc))
}

// ── Query2DocVariant ──────────────────────────────────────────────────────────

#[test]
fn variant_default_is_sparse() {
    assert_eq!(Query2DocVariant::default(), Query2DocVariant::Sparse);
}

#[test]
fn variant_as_str_sparse() {
    assert_eq!(Query2DocVariant::Sparse.as_str(), "sparse");
}

#[test]
fn variant_as_str_dense() {
    assert_eq!(Query2DocVariant::Dense.as_str(), "dense");
}

#[test]
fn variant_display_sparse() {
    assert_eq!(format!("{}", Query2DocVariant::Sparse), "sparse");
}

#[test]
fn variant_display_dense() {
    assert_eq!(format!("{}", Query2DocVariant::Dense), "dense");
}

#[test]
fn variant_equality() {
    assert_eq!(Query2DocVariant::Sparse, Query2DocVariant::Sparse);
    assert_eq!(Query2DocVariant::Dense, Query2DocVariant::Dense);
}

#[test]
fn variant_inequality() {
    assert_ne!(Query2DocVariant::Sparse, Query2DocVariant::Dense);
}

#[test]
fn variant_copy_clone() {
    // `Query2DocVariant` is `Copy`, so `v` remains usable after being copied
    // into `copied` below (a genuine `.clone()` call would be redundant and
    // is rejected by `clippy::clone_on_copy`).
    let v = Query2DocVariant::Dense;
    let copied = v;
    assert_eq!(v, copied);
}

// ── Query2DocConfig ───────────────────────────────────────────────────────────

#[test]
fn config_default_query_repetitions() {
    assert_eq!(Query2DocConfig::default().query_repetitions, 5);
}

#[test]
fn config_default_max_pseudo_len() {
    assert_eq!(Query2DocConfig::default().max_pseudo_len, 400);
}

#[test]
fn config_default_variant() {
    assert_eq!(Query2DocConfig::default().variant, Query2DocVariant::Sparse);
}

#[test]
fn config_new_equals_default() {
    assert_eq!(Query2DocConfig::new(), Query2DocConfig::default());
}

#[test]
fn config_builder_query_repetitions() {
    let cfg = Query2DocConfig::new().with_query_repetitions(10);
    assert_eq!(cfg.query_repetitions, 10);
    // Other fields untouched.
    assert_eq!(cfg.max_pseudo_len, 400);
}

#[test]
fn config_builder_max_pseudo_len() {
    let cfg = Query2DocConfig::new().with_max_pseudo_len(128);
    assert_eq!(cfg.max_pseudo_len, 128);
}

#[test]
fn config_builder_variant() {
    let cfg = Query2DocConfig::new().with_variant(Query2DocVariant::Dense);
    assert_eq!(cfg.variant, Query2DocVariant::Dense);
}

#[test]
fn config_builder_chaining() {
    let cfg = Query2DocConfig::new()
        .with_query_repetitions(3)
        .with_max_pseudo_len(50)
        .with_variant(Query2DocVariant::Dense);
    assert_eq!(cfg.query_repetitions, 3);
    assert_eq!(cfg.max_pseudo_len, 50);
    assert_eq!(cfg.variant, Query2DocVariant::Dense);
}

#[test]
fn config_validate_ok_default() {
    assert!(Query2DocConfig::default().validate().is_ok());
}

#[test]
fn config_validate_zero_repetitions_err() {
    let cfg = Query2DocConfig::new().with_query_repetitions(0);
    assert!(matches!(
        cfg.validate(),
        Err(Query2DocError::InvalidConfig(_))
    ));
}

#[test]
fn config_validate_zero_max_len_err() {
    let cfg = Query2DocConfig::new().with_max_pseudo_len(0);
    assert!(matches!(
        cfg.validate(),
        Err(Query2DocError::InvalidConfig(_))
    ));
}

#[test]
fn config_validate_err_message_repetitions() {
    let cfg = Query2DocConfig::new().with_query_repetitions(0);
    let err = cfg.validate().unwrap_err();
    assert!(err.to_string().contains("query_repetitions"));
}

#[test]
fn config_validate_err_message_max_len() {
    let cfg = Query2DocConfig::new().with_max_pseudo_len(0);
    let err = cfg.validate().unwrap_err();
    assert!(err.to_string().contains("max_pseudo_len"));
}

// ── Query2DocError ────────────────────────────────────────────────────────────

#[test]
fn error_empty_query_display() {
    assert_eq!(Query2DocError::EmptyQuery.to_string(), "query is empty");
}

#[test]
fn error_invalid_config_display() {
    let err = Query2DocError::InvalidConfig("bad value".to_string());
    assert_eq!(err.to_string(), "invalid configuration: bad value");
}

#[test]
fn error_equality() {
    assert_eq!(Query2DocError::EmptyQuery, Query2DocError::EmptyQuery);
    assert_eq!(
        Query2DocError::InvalidConfig("x".to_string()),
        Query2DocError::InvalidConfig("x".to_string())
    );
}

#[test]
fn error_inequality() {
    assert_ne!(
        Query2DocError::EmptyQuery,
        Query2DocError::InvalidConfig("x".to_string())
    );
}

#[test]
fn error_clone() {
    let err = Query2DocError::InvalidConfig("clone me".to_string());
    assert_eq!(err.clone(), err);
}

// ── MockPseudoDocGenerator ────────────────────────────────────────────────────

#[test]
fn mock_generate_empty_query_returns_empty() {
    let mock_gen = MockPseudoDocGenerator::new();
    assert_eq!(mock_gen.generate(""), "");
}

#[test]
fn mock_generate_whitespace_query_returns_empty() {
    let mock_gen = MockPseudoDocGenerator::new();
    assert_eq!(mock_gen.generate("   \t  "), "");
}

#[test]
fn mock_generate_deterministic_same_output() {
    let mock_gen = MockPseudoDocGenerator::new();
    let a = mock_gen.generate("What is Rust?");
    let b = mock_gen.generate("What is Rust?");
    assert_eq!(a, b);
}

#[test]
fn mock_generate_fallback_template_no_knowledge() {
    let mock_gen = MockPseudoDocGenerator::new();
    let doc = mock_gen.generate("What is Rust?");
    assert_eq!(
        doc,
        "Rust is closely related to Rust. This passage elaborates on Rust and provides the context needed to answer \"What is Rust?\"."
    );
}

#[test]
fn mock_generate_with_knowledge_hit() {
    let mock_gen = MockPseudoDocGenerator::new().with_knowledge("rust", "is great");
    let doc = mock_gen.generate("What is Rust?");
    assert_eq!(doc, "Rust is great");
}

#[test]
fn mock_generate_with_knowledge_case_insensitive() {
    let mock_gen = MockPseudoDocGenerator::new().with_knowledge("Rust", "is great");
    let doc = mock_gen.generate("what is RUST?");
    assert_eq!(doc, "RUST is great");
}

#[test]
fn mock_generate_knowledge_miss_uses_fallback() {
    let mock_gen = MockPseudoDocGenerator::new().with_knowledge("python", "is dynamic");
    let doc = mock_gen.generate("What is Rust?");
    assert_eq!(
        doc,
        "Rust is closely related to Rust. This passage elaborates on Rust and provides the context needed to answer \"What is Rust?\"."
    );
}

#[test]
fn mock_default_has_empty_knowledge() {
    let mock_gen = MockPseudoDocGenerator::default();
    assert!(mock_gen.knowledge.is_empty());
}

#[test]
fn mock_with_knowledge_builder_chaining() {
    let mock_gen = MockPseudoDocGenerator::new()
        .with_knowledge("rust", "fact-rust")
        .with_knowledge("python", "fact-python");
    assert_eq!(mock_gen.knowledge.len(), 2);
    assert_eq!(mock_gen.generate("Tell me about Rust"), "Rust fact-rust");
    assert_eq!(
        mock_gen.generate("Tell me about Python"),
        "Python fact-python"
    );
}

#[test]
fn mock_generate_single_word_query() {
    let mock_gen = MockPseudoDocGenerator::new();
    let doc = mock_gen.generate("Rust");
    assert_eq!(
        doc,
        "Rust is closely related to Rust. This passage elaborates on Rust and provides the context needed to answer \"Rust\"."
    );
}

#[test]
fn mock_generate_keyword_order_preserved() {
    // Only "python" is registered; "Rust" appears first in the query but has
    // no canned fact, so the first *matching* keyword ("Python") is used.
    let mock_gen = MockPseudoDocGenerator::new().with_knowledge("python", "fact-python");
    let doc = mock_gen.generate("What is Rust and Python?");
    assert_eq!(doc, "Python fact-python");
}

#[test]
fn mock_generate_all_stopwords_query_uses_generic_fallback() {
    let mock_gen = MockPseudoDocGenerator::new();
    let doc = mock_gen.generate("What is this?");
    assert_eq!(
        doc,
        "This passage provides background information relevant to the question \"What is this?\"."
    );
}

#[test]
fn mock_generate_never_empty_for_nonempty_query() {
    let mock_gen = MockPseudoDocGenerator::new();
    assert!(!mock_gen.generate("anything at all").is_empty());
}

// ── Query2DocExpander — sparse variant ────────────────────────────────────────

#[test]
fn expand_sparse_default_repetitions_count() {
    let expander = default_expander_fixed("PSEUDODOC");
    let result = expander.expand("hello world").unwrap();
    assert_eq!(result.expanded_query.matches("hello world").count(), 5);
}

#[test]
fn expand_sparse_exact_string_with_fixed_generator() {
    let expander = default_expander_fixed("PSEUDODOC");
    let result = expander.expand("hello world").unwrap();
    assert_eq!(
        result.expanded_query,
        "hello world hello world hello world hello world hello world PSEUDODOC"
    );
}

#[test]
fn expand_sparse_custom_repetitions() {
    let config = Query2DocConfig::new().with_query_repetitions(3);
    let expander = Query2DocExpander::new(config, FixedGenerator::new("PSEUDODOC"));
    let result = expander.expand("hello").unwrap();
    assert_eq!(result.expanded_query, "hello hello hello PSEUDODOC");
}

#[test]
fn expand_sparse_single_repetition() {
    let config = Query2DocConfig::new().with_query_repetitions(1);
    let expander = Query2DocExpander::new(config, FixedGenerator::new("PSEUDODOC"));
    let result = expander.expand("hello").unwrap();
    assert_eq!(result.expanded_query, "hello PSEUDODOC");
}

#[test]
fn expand_sparse_large_repetitions_value() {
    let config = Query2DocConfig::new().with_query_repetitions(20);
    let expander = Query2DocExpander::new(config, FixedGenerator::new("DOC"));
    let result = expander.expand("q").unwrap();
    // 20 repetitions of the single-character query "q", separated by spaces,
    // followed by the (q-free) pseudo-document "DOC".
    assert_eq!(result.expanded_query.matches('q').count(), 20);
    assert_eq!(
        result.expanded_query,
        format!("{} DOC", vec!["q"; 20].join(" "))
    );
}

#[test]
fn expand_sparse_query_field_trimmed() {
    let expander = default_expander_fixed("PSEUDODOC");
    let result = expander.expand("  hello world  ").unwrap();
    assert_eq!(result.query, "hello world");
}

#[test]
fn expand_sparse_pseudo_doc_field_matches_generator_output_untruncated() {
    let expander = default_expander_fixed("A SHORT DOC");
    let result = expander.expand("q").unwrap();
    assert_eq!(result.pseudo_doc, "A SHORT DOC");
}

#[test]
fn expand_sparse_default_variant_matches_explicit_sparse() {
    let default_result = default_expander_fixed("DOC").expand("q").unwrap();
    let explicit_config = Query2DocConfig::new().with_variant(Query2DocVariant::Sparse);
    let explicit_result = Query2DocExpander::new(explicit_config, FixedGenerator::new("DOC"))
        .expand("q")
        .unwrap();
    assert_eq!(default_result, explicit_result);
}

// ── Query2DocExpander — dense variant ─────────────────────────────────────────

#[test]
fn expand_dense_contains_sep_once() {
    let config = Query2DocConfig::new().with_variant(Query2DocVariant::Dense);
    let expander = Query2DocExpander::new(config, FixedGenerator::new("PSEUDODOC"));
    let result = expander.expand("hello world").unwrap();
    assert_eq!(result.expanded_query.matches("[SEP]").count(), 1);
}

#[test]
fn expand_dense_query_appears_once() {
    let config = Query2DocConfig::new().with_variant(Query2DocVariant::Dense);
    let expander = Query2DocExpander::new(config, FixedGenerator::new("unrelated pseudo text"));
    let result = expander.expand("rust ownership").unwrap();
    assert_eq!(result.expanded_query.matches("rust ownership").count(), 1);
}

#[test]
fn expand_dense_exact_string_with_fixed_generator() {
    let config = Query2DocConfig::new().with_variant(Query2DocVariant::Dense);
    let expander = Query2DocExpander::new(config, FixedGenerator::new("PSEUDODOC"));
    let result = expander.expand("hello world").unwrap();
    assert_eq!(result.expanded_query, "hello world [SEP] PSEUDODOC");
}

#[test]
fn expand_dense_ignores_query_repetitions_config() {
    // query_repetitions is set high, but the dense variant must still emit
    // the query exactly once.
    let config = Query2DocConfig::new()
        .with_variant(Query2DocVariant::Dense)
        .with_query_repetitions(10);
    let expander = Query2DocExpander::new(config, FixedGenerator::new("DOC"));
    let result = expander.expand("hello world").unwrap();
    assert_eq!(result.expanded_query, "hello world [SEP] DOC");
}

// ── Query2DocExpander — errors ────────────────────────────────────────────────

#[test]
fn expand_empty_query_errors() {
    let expander = default_expander_fixed("DOC");
    assert_eq!(expander.expand(""), Err(Query2DocError::EmptyQuery));
}

#[test]
fn expand_whitespace_query_errors() {
    let expander = default_expander_fixed("DOC");
    assert_eq!(
        expander.expand("   \n\t  "),
        Err(Query2DocError::EmptyQuery)
    );
}

#[test]
fn expand_zero_repetitions_config_errors() {
    let config = Query2DocConfig::new().with_query_repetitions(0);
    let expander = Query2DocExpander::new(config, FixedGenerator::new("DOC"));
    assert!(matches!(
        expander.expand("valid query"),
        Err(Query2DocError::InvalidConfig(_))
    ));
}

#[test]
fn expand_zero_max_len_config_errors() {
    let config = Query2DocConfig::new().with_max_pseudo_len(0);
    let expander = Query2DocExpander::new(config, FixedGenerator::new("DOC"));
    assert!(matches!(
        expander.expand("valid query"),
        Err(Query2DocError::InvalidConfig(_))
    ));
}

#[test]
fn expand_invalid_config_checked_before_empty_query() {
    // Even when the query is also empty, an invalid configuration is
    // reported first (config validation is deterministic and independent of
    // the query).
    let config = Query2DocConfig::new().with_query_repetitions(0);
    let expander = Query2DocExpander::new(config, FixedGenerator::new("DOC"));
    assert!(matches!(
        expander.expand(""),
        Err(Query2DocError::InvalidConfig(_))
    ));
}

// ── Truncation ────────────────────────────────────────────────────────────────

#[test]
fn expand_truncates_pseudo_doc_to_max_len() {
    let long_doc = "A".repeat(1000);
    let config = Query2DocConfig::new().with_max_pseudo_len(10);
    let expander = Query2DocExpander::new(config, FixedGenerator::new(long_doc));
    let result = expander.expand("q").unwrap();
    assert_eq!(result.pseudo_doc, "A".repeat(10));
    assert_eq!(result.pseudo_doc.chars().count(), 10);
}

#[test]
fn expand_no_truncation_when_within_limit() {
    let expander = default_expander_fixed("SHORT");
    let result = expander.expand("q").unwrap();
    assert_eq!(result.pseudo_doc, "SHORT");
}

#[test]
fn expand_truncation_respects_char_boundary_multibyte() {
    // Each character below is a multi-byte UTF-8 scalar value; truncating on
    // a byte boundary instead of a char boundary would panic or corrupt data.
    let multibyte_doc = "日本語テキスト";
    let config = Query2DocConfig::new().with_max_pseudo_len(3);
    let expander = Query2DocExpander::new(config, FixedGenerator::new(multibyte_doc));
    let result = expander.expand("q").unwrap();
    assert_eq!(result.pseudo_doc, "日本語");
    assert_eq!(result.pseudo_doc.chars().count(), 3);
}

#[test]
fn expand_truncation_applies_before_concatenation() {
    let long_doc = "X".repeat(50);
    let config = Query2DocConfig::new()
        .with_max_pseudo_len(5)
        .with_query_repetitions(1);
    let expander = Query2DocExpander::new(config, FixedGenerator::new(long_doc));
    let result = expander.expand("q").unwrap();
    assert_eq!(result.expanded_query, format!("q {}", "X".repeat(5)));
}

// ── Term-weighting (core Query2Doc property) ─────────────────────────────────

#[test]
fn expand_sparse_term_weighting_repeats_query_terms() {
    let config = Query2DocConfig::default(); // 5 repetitions
    let expander = Query2DocExpander::new(
        config,
        FixedGenerator::new("A completely unrelated passage about memory management."),
    );
    let result = expander.expand("rust ownership").unwrap();
    assert_eq!(
        result.expanded_query.matches("rust ownership").count(),
        5,
        "sparse variant must repeat the query 5 times by default"
    );
}

#[test]
fn expand_dense_term_weighting_single_occurrence() {
    let config = Query2DocConfig::new().with_variant(Query2DocVariant::Dense);
    let expander = Query2DocExpander::new(
        config,
        FixedGenerator::new("A completely unrelated passage about memory management."),
    );
    let result = expander.expand("rust ownership").unwrap();
    assert_eq!(
        result.expanded_query.matches("rust ownership").count(),
        1,
        "dense variant must emit the query exactly once"
    );
}

#[test]
fn expand_sparse_repetitions_scale_linearly() {
    let doc = "UNRELATED_TEXT_MARKER";
    for reps in 1..=8usize {
        let config = Query2DocConfig::new().with_query_repetitions(reps);
        let expander = Query2DocExpander::new(config, FixedGenerator::new(doc));
        let result = expander.expand("marker_query").unwrap();
        assert_eq!(
            result.expanded_query.matches("marker_query").count(),
            reps,
            "expected {reps} occurrences of the query for query_repetitions={reps}"
        );
    }
}

// ── PseudoDocument ────────────────────────────────────────────────────────────

#[test]
fn pseudo_document_fields_accessible() {
    let expander = default_expander_fixed("DOC");
    let result = expander.expand("hello").unwrap();
    let query_field: &str = &result.query;
    let pseudo_doc_field: &str = &result.pseudo_doc;
    let expanded_query_field: &str = &result.expanded_query;
    assert!(!query_field.is_empty());
    assert!(!pseudo_doc_field.is_empty());
    assert!(!expanded_query_field.is_empty());
}

#[test]
fn pseudo_document_equality() {
    let a = PseudoDocument {
        query: "q".to_string(),
        pseudo_doc: "d".to_string(),
        expanded_query: "q d".to_string(),
    };
    let b = a.clone();
    assert_eq!(a, b);
}

#[test]
fn pseudo_document_inequality() {
    let a = PseudoDocument {
        query: "q1".to_string(),
        pseudo_doc: "d".to_string(),
        expanded_query: "q1 d".to_string(),
    };
    let b = PseudoDocument {
        query: "q2".to_string(),
        pseudo_doc: "d".to_string(),
        expanded_query: "q2 d".to_string(),
    };
    assert_ne!(a, b);
}

#[test]
fn pseudo_document_clone_independent() {
    let a = PseudoDocument {
        query: "q".to_string(),
        pseudo_doc: "d".to_string(),
        expanded_query: "q d".to_string(),
    };
    let mut b = a.clone();
    b.query.push_str("-modified");
    assert_ne!(a.query, b.query);
}

// ── Trait-object compatibility ────────────────────────────────────────────────

fn generate_via_dyn(generator: &dyn PseudoDocGenerator, query: &str) -> String {
    generator.generate(query)
}

#[test]
fn pseudo_doc_generator_is_dyn_compatible() {
    let mock = MockPseudoDocGenerator::new();
    let via_dyn = generate_via_dyn(&mock, "hello there");
    let direct = mock.generate("hello there");
    assert_eq!(via_dyn, direct);
}

// ── Expander cloning / determinism ────────────────────────────────────────────

#[test]
fn expander_clone_identical_results() {
    let expander =
        Query2DocExpander::new(Query2DocConfig::default(), MockPseudoDocGenerator::new());
    let cloned = expander.clone();
    let r1 = expander.expand("What is Rust?").unwrap();
    let r2 = cloned.expand("What is Rust?").unwrap();
    assert_eq!(r1, r2);
}

#[test]
fn expand_is_pure_repeated_calls_identical() {
    let expander = default_expander_fixed("DOC");
    let r1 = expander.expand("same query").unwrap();
    let r2 = expander.expand("same query").unwrap();
    assert_eq!(r1, r2);
}

#[test]
fn expand_different_queries_produce_different_results() {
    let expander = default_expander_fixed("DOC");
    let r1 = expander.expand("query one").unwrap();
    let r2 = expander.expand("query two").unwrap();
    assert_ne!(r1, r2);
}

// ── Config field access on expander ───────────────────────────────────────────

#[test]
fn expander_exposes_config_and_generator() {
    let expander = Query2DocExpander::new(
        Query2DocConfig::new().with_query_repetitions(7),
        FixedGenerator::new("DOC"),
    );
    assert_eq!(expander.config.query_repetitions, 7);
    assert_eq!(expander.generator.generate("anything"), "DOC");
}
