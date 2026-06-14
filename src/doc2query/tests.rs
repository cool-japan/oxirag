//! Tests for the `doc2query` module.
#![allow(clippy::similar_names)]

use crate::doc2query::expander::Doc2QueryExpander;
use crate::doc2query::generator::HeuristicQueryGenerator;
use crate::doc2query::types::{Doc2QueryConfig, Doc2QueryError, ExpandedDocument, QueryGenerator};
use crate::types::{Document, DocumentId};
use std::collections::HashSet;

// ── helpers ──────────────────────────────────────────────────────────────────

fn sample_doc() -> Document {
    Document::new(
        "Photosynthesis converts sunlight into chemical energy. \
         Chlorophyll absorbs light in plant cells. \
         Oxygen is released as a byproduct of photosynthesis.",
    )
    .with_id(DocumentId::from_string("doc-sample"))
}

fn entity_doc() -> Document {
    Document::new(
        "Einstein developed relativity. Newton formulated gravity. \
         Einstein and Newton shaped modern physics.",
    )
    .with_id(DocumentId::from_string("doc-entity"))
}

// ── Doc2QueryConfig defaults ──────────────────────────────────────────────────

#[test]
fn test_config_default_num_queries() {
    let c = Doc2QueryConfig::default();
    assert_eq!(c.num_queries, 5, "default num_queries should be 5");
}

#[test]
fn test_config_default_include_definitional() {
    let c = Doc2QueryConfig::default();
    assert!(
        c.include_definitional,
        "default include_definitional should be true"
    );
}

#[test]
fn test_config_default_include_relational() {
    let c = Doc2QueryConfig::default();
    assert!(
        c.include_relational,
        "default include_relational should be true"
    );
}

#[test]
fn test_config_default_append_separator() {
    let c = Doc2QueryConfig::default();
    assert_eq!(
        c.append_separator, "\n",
        "default append_separator should be newline"
    );
}

#[test]
fn test_config_new_matches_default() {
    let c = Doc2QueryConfig::new();
    assert_eq!(c.num_queries, 5, "new() should match Default num_queries");
}

// ── Doc2QueryConfig builders ──────────────────────────────────────────────────

#[test]
fn test_config_with_num_queries_builder() {
    let c = Doc2QueryConfig::new().with_num_queries(3);
    assert_eq!(c.num_queries, 3, "with_num_queries should set num_queries");
}

#[test]
fn test_config_with_include_definitional_builder() {
    let c = Doc2QueryConfig::new().with_include_definitional(false);
    assert!(
        !c.include_definitional,
        "with_include_definitional(false) should disable definitional"
    );
}

#[test]
fn test_config_with_include_relational_builder() {
    let c = Doc2QueryConfig::new().with_include_relational(false);
    assert!(
        !c.include_relational,
        "with_include_relational(false) should disable relational"
    );
}

#[test]
fn test_config_with_append_separator_builder() {
    let c = Doc2QueryConfig::new().with_append_separator(" | ");
    assert_eq!(
        c.append_separator, " | ",
        "with_append_separator should set the separator"
    );
}

// ── HeuristicQueryGenerator ───────────────────────────────────────────────────

#[test]
fn test_generator_new_stores_config() {
    let cfg = Doc2QueryConfig::new().with_num_queries(7);
    let gen_ = HeuristicQueryGenerator::new(cfg);
    assert_eq!(
        gen_.config.num_queries, 7,
        "new should store the provided config"
    );
}

#[test]
fn test_generate_respects_num_queries_upper_bound() {
    let gen_ = HeuristicQueryGenerator::new(Doc2QueryConfig::default());
    let questions = gen_.generate(&sample_doc(), 3);
    assert!(
        questions.len() <= 3,
        "generate should return at most n questions, got {}",
        questions.len()
    );
}

#[test]
fn test_generate_zero_yields_empty() {
    let gen_ = HeuristicQueryGenerator::new(Doc2QueryConfig::default());
    let questions = gen_.generate(&sample_doc(), 0);
    assert!(questions.is_empty(), "generate(.., 0) should be empty");
}

#[test]
fn test_generate_nonempty_for_normal_doc() {
    let gen_ = HeuristicQueryGenerator::new(Doc2QueryConfig::default());
    let questions = gen_.generate(&sample_doc(), 5);
    assert!(
        !questions.is_empty(),
        "generate should produce questions for a normal document"
    );
}

#[test]
fn test_generate_questions_contain_salient_term() {
    let gen_ = HeuristicQueryGenerator::new(Doc2QueryConfig::default());
    let questions = gen_.generate(&sample_doc(), 5);
    let joined = questions.join(" ").to_lowercase();
    assert!(
        joined.contains("photosynthesis"),
        "questions should reference a salient document term, got: {questions:?}"
    );
}

#[test]
fn test_generate_questions_reference_real_tokens_only() {
    let doc = sample_doc();
    let gen_ = HeuristicQueryGenerator::new(Doc2QueryConfig::default());
    let questions = gen_.generate(&doc, 5);
    // Every alphabetic token in the questions that is not a template word must
    // appear in the document text (no hallucinated content tokens).
    let template_words: HashSet<&str> = [
        "what",
        "is",
        "does",
        "mean",
        "how",
        "the",
        "connection",
        "between",
        "and",
        "relate",
        "to",
        "which",
        "details",
        "concern",
    ]
    .into_iter()
    .collect();
    let doc_tokens: HashSet<String> = doc
        .content
        .split(|c: char| !c.is_alphanumeric())
        .filter(|t| t.len() >= 2)
        .map(str::to_lowercase)
        .collect();
    let all_grounded = questions.iter().all(|q| {
        q.split(|c: char| !c.is_alphanumeric())
            .filter(|t| t.len() >= 2)
            .map(str::to_lowercase)
            .all(|t| template_words.contains(t.as_str()) || doc_tokens.contains(&t))
    });
    assert!(
        all_grounded,
        "all question tokens must be template words or document tokens: {questions:?}"
    );
}

#[test]
fn test_generate_definitional_present_by_default() {
    let gen_ = HeuristicQueryGenerator::new(Doc2QueryConfig::default());
    let questions = gen_.generate(&sample_doc(), 5);
    let has_definitional = questions.iter().any(|q| q.starts_with("What is "));
    assert!(
        has_definitional,
        "default config should yield 'What is' questions, got: {questions:?}"
    );
}

#[test]
fn test_generate_include_definitional_false_suppresses_what_is() {
    let cfg = Doc2QueryConfig::new()
        .with_include_definitional(false)
        .with_include_relational(false);
    let gen_ = HeuristicQueryGenerator::new(cfg);
    let questions = gen_.generate(&sample_doc(), 5);
    let has_what_is = questions.iter().any(|q| q.starts_with("What is "));
    assert!(
        !has_what_is,
        "include_definitional=false should suppress 'What is' questions, got: {questions:?}"
    );
}

#[test]
fn test_generate_include_definitional_false_suppresses_what_does_mean() {
    let cfg = Doc2QueryConfig::new()
        .with_include_definitional(false)
        .with_include_relational(false);
    let gen_ = HeuristicQueryGenerator::new(cfg);
    let questions = gen_.generate(&sample_doc(), 5);
    let has_mean = questions.iter().any(|q| q.contains(" mean?"));
    assert!(
        !has_mean,
        "include_definitional=false should suppress 'mean?' questions, got: {questions:?}"
    );
}

#[test]
fn test_generate_relational_present_for_entity_doc() {
    let cfg = Doc2QueryConfig::new().with_num_queries(10);
    let gen_ = HeuristicQueryGenerator::new(cfg);
    let questions = gen_.generate(&entity_doc(), 10);
    let has_relational = questions
        .iter()
        .any(|q| q.contains("relate to") || q.contains("connection between"));
    assert!(
        has_relational,
        "entity-rich doc should yield relational questions, got: {questions:?}"
    );
}

#[test]
fn test_generate_include_relational_false_suppresses_relational() {
    let cfg = Doc2QueryConfig::new()
        .with_include_relational(false)
        .with_num_queries(10);
    let gen_ = HeuristicQueryGenerator::new(cfg);
    let questions = gen_.generate(&entity_doc(), 10);
    let has_relational = questions
        .iter()
        .any(|q| q.contains("relate to") || q.contains("connection between"));
    assert!(
        !has_relational,
        "include_relational=false should suppress relational questions, got: {questions:?}"
    );
}

#[test]
fn test_generate_relational_references_entities() {
    let cfg = Doc2QueryConfig::new()
        .with_include_definitional(false)
        .with_num_queries(10);
    let gen_ = HeuristicQueryGenerator::new(cfg);
    let questions = gen_.generate(&entity_doc(), 10);
    let mentions_entity = questions
        .iter()
        .any(|q| q.contains("Einstein") || q.contains("Newton"));
    assert!(
        mentions_entity,
        "relational questions should reference detected entities, got: {questions:?}"
    );
}

#[test]
fn test_generate_single_sentence_still_yields_questions() {
    let doc = Document::new("Rust is a systems programming language");
    let gen_ = HeuristicQueryGenerator::new(Doc2QueryConfig::default());
    let questions = gen_.generate(&doc, 5);
    assert!(
        !questions.is_empty(),
        "a single-sentence doc should still yield questions"
    );
}

#[test]
fn test_generate_no_duplicate_questions() {
    let doc =
        Document::new("Rust rust rust safety safety memory memory ownership ownership borrow.");
    let gen_ = HeuristicQueryGenerator::new(Doc2QueryConfig::default());
    let questions = gen_.generate(&doc, 5);
    let unique: HashSet<&String> = questions.iter().collect();
    assert_eq!(
        unique.len(),
        questions.len(),
        "generated questions must be deduplicated, got: {questions:?}"
    );
}

#[test]
fn test_generate_deterministic_across_calls() {
    let doc = sample_doc();
    let gen_ = HeuristicQueryGenerator::new(Doc2QueryConfig::default());
    let first = gen_.generate(&doc, 5);
    let second = gen_.generate(&doc, 5);
    assert_eq!(
        first, second,
        "generation must be deterministic across calls"
    );
}

#[test]
fn test_generate_uses_title_terms() {
    let doc = Document::new("It operates on tabular records.").with_title("Quantum Cryptography");
    let gen_ = HeuristicQueryGenerator::new(Doc2QueryConfig::default());
    let questions = gen_.generate(&doc, 5);
    let joined = questions.join(" ").to_lowercase();
    assert!(
        joined.contains("quantum") || joined.contains("cryptography"),
        "title terms should feed question generation, got: {questions:?}"
    );
}

// ── ExpandedDocument / expand ─────────────────────────────────────────────────

#[test]
fn test_expand_returns_original_document() {
    let doc = sample_doc();
    let expander = Doc2QueryExpander::new(Doc2QueryConfig::default());
    let expanded = expander.expand(&doc).unwrap();
    assert_eq!(
        expanded.original.id.as_str(),
        "doc-sample",
        "expand should retain the original document id"
    );
}

#[test]
fn test_expand_content_contains_original_content() {
    let doc = sample_doc();
    let expander = Doc2QueryExpander::new(Doc2QueryConfig::default());
    let expanded = expander.expand(&doc).unwrap();
    assert!(
        expanded.expanded_content.contains(&doc.content),
        "expanded_content must contain the original content"
    );
}

#[test]
fn test_expand_content_contains_all_generated_queries() {
    let doc = sample_doc();
    let expander = Doc2QueryExpander::new(Doc2QueryConfig::default());
    let expanded = expander.expand(&doc).unwrap();
    let all_present = expanded
        .generated_queries
        .iter()
        .all(|q| expanded.expanded_content.contains(q));
    assert!(
        all_present,
        "expanded_content must contain every generated query"
    );
}

#[test]
fn test_expand_content_uses_separator() {
    let doc = Document::new("Rust enables fearless concurrency in software.");
    let cfg = Doc2QueryConfig::new().with_append_separator(" <SEP> ");
    let expander = Doc2QueryExpander::new(cfg);
    let expanded = expander.expand(&doc).unwrap();
    assert!(
        expanded.expanded_content.contains(" <SEP> "),
        "expanded_content should use the configured separator when queries exist"
    );
}

#[test]
fn test_expand_generated_queries_nonempty() {
    let doc = sample_doc();
    let expander = Doc2QueryExpander::new(Doc2QueryConfig::default());
    let expanded = expander.expand(&doc).unwrap();
    assert!(
        !expanded.generated_queries.is_empty(),
        "expand should populate generated_queries for a normal doc"
    );
}

#[test]
fn test_expand_query_count_within_config_limit() {
    let cfg = Doc2QueryConfig::new().with_num_queries(2);
    let expander = Doc2QueryExpander::new(cfg);
    let expanded = expander.expand(&sample_doc()).unwrap();
    assert!(
        expanded.generated_queries.len() <= 2,
        "generated query count must respect num_queries, got {}",
        expanded.generated_queries.len()
    );
}

#[test]
fn test_expand_empty_document_errors() {
    let doc = Document::new("");
    let expander = Doc2QueryExpander::new(Doc2QueryConfig::default());
    let err = expander.expand(&doc).unwrap_err();
    assert!(
        matches!(err, Doc2QueryError::EmptyDocument),
        "empty document should return EmptyDocument error"
    );
}

#[test]
fn test_expand_whitespace_document_errors() {
    let doc = Document::new("   \n  ");
    let expander = Doc2QueryExpander::new(Doc2QueryConfig::default());
    let err = expander.expand(&doc).unwrap_err();
    assert!(
        matches!(err, Doc2QueryError::EmptyDocument),
        "whitespace-only document should return EmptyDocument error"
    );
}

#[test]
fn test_expand_deterministic() {
    let doc = sample_doc();
    let expander = Doc2QueryExpander::new(Doc2QueryConfig::default());
    let first = expander.expand(&doc).unwrap();
    let second = expander.expand(&doc).unwrap();
    assert_eq!(
        first.expanded_content, second.expanded_content,
        "expand must be deterministic"
    );
}

// ── to_indexable_document ─────────────────────────────────────────────────────

#[test]
fn test_to_indexable_preserves_id() {
    let doc = sample_doc();
    let expander = Doc2QueryExpander::new(Doc2QueryConfig::default());
    let indexable = expander.to_indexable_document(&doc).unwrap();
    assert_eq!(
        indexable.id.as_str(),
        "doc-sample",
        "to_indexable_document must preserve the document id"
    );
}

#[test]
fn test_to_indexable_preserves_metadata() {
    let doc = sample_doc().with_metadata("lang", "en");
    let expander = Doc2QueryExpander::new(Doc2QueryConfig::default());
    let indexable = expander.to_indexable_document(&doc).unwrap();
    assert_eq!(
        indexable.metadata.get("lang").map(String::as_str),
        Some("en"),
        "to_indexable_document must preserve metadata"
    );
}

#[test]
fn test_to_indexable_changes_content() {
    let doc = sample_doc();
    let original_content = doc.content.clone();
    let expander = Doc2QueryExpander::new(Doc2QueryConfig::default());
    let indexable = expander.to_indexable_document(&doc).unwrap();
    assert_ne!(
        indexable.content, original_content,
        "to_indexable_document should append queries to the content"
    );
}

#[test]
fn test_to_indexable_content_contains_original() {
    let doc = sample_doc();
    let expander = Doc2QueryExpander::new(Doc2QueryConfig::default());
    let indexable = expander.to_indexable_document(&doc).unwrap();
    assert!(
        indexable.content.starts_with(&doc.content),
        "indexable content should begin with the original content"
    );
}

#[test]
fn test_to_indexable_preserves_title() {
    let doc = sample_doc().with_title("Biology Basics");
    let expander = Doc2QueryExpander::new(Doc2QueryConfig::default());
    let indexable = expander.to_indexable_document(&doc).unwrap();
    assert_eq!(
        indexable.title.as_deref(),
        Some("Biology Basics"),
        "to_indexable_document must preserve the title"
    );
}

#[test]
fn test_to_indexable_empty_document_errors() {
    let doc = Document::new("");
    let expander = Doc2QueryExpander::new(Doc2QueryConfig::default());
    let err = expander.to_indexable_document(&doc).unwrap_err();
    assert!(
        matches!(err, Doc2QueryError::EmptyDocument),
        "to_indexable_document on empty doc should error"
    );
}

// ── expand_corpus ─────────────────────────────────────────────────────────────

#[test]
fn test_expand_corpus_length_matches_input() {
    let docs = vec![sample_doc(), entity_doc(), Document::new("Pure Rust code.")];
    let expander = Doc2QueryExpander::new(Doc2QueryConfig::default());
    let expanded = expander.expand_corpus(&docs).unwrap();
    assert_eq!(
        expanded.len(),
        3,
        "expand_corpus output length must match input length"
    );
}

#[test]
fn test_expand_corpus_empty_input_yields_empty() {
    let expander = Doc2QueryExpander::new(Doc2QueryConfig::default());
    let expanded = expander.expand_corpus(&[]).unwrap();
    assert!(
        expanded.is_empty(),
        "expand_corpus on empty input should yield empty output"
    );
}

#[test]
fn test_expand_corpus_propagates_empty_document_error() {
    let docs = vec![sample_doc(), Document::new("")];
    let expander = Doc2QueryExpander::new(Doc2QueryConfig::default());
    let err = expander.expand_corpus(&docs).unwrap_err();
    assert!(
        matches!(err, Doc2QueryError::EmptyDocument),
        "expand_corpus should propagate EmptyDocument for an empty member"
    );
}

#[test]
fn test_expand_corpus_preserves_order() {
    let docs = vec![
        sample_doc(),
        entity_doc().with_id(DocumentId::from_string("second")),
    ];
    let expander = Doc2QueryExpander::new(Doc2QueryConfig::default());
    let expanded = expander.expand_corpus(&docs).unwrap();
    assert_eq!(
        expanded[1].original.id.as_str(),
        "second",
        "expand_corpus must preserve input order"
    );
}

// ── custom generator via with_generator ───────────────────────────────────────

/// A trivial deterministic generator used to test `with_generator`.
#[derive(Debug)]
struct StubGenerator;

impl QueryGenerator for StubGenerator {
    fn generate(&self, _doc: &Document, n: usize) -> Vec<String> {
        (0..n.min(2)).map(|i| format!("stub query {i}")).collect()
    }
}

#[test]
fn test_with_generator_uses_custom_generator() {
    let cfg = Doc2QueryConfig::new().with_num_queries(2);
    let expander = Doc2QueryExpander::with_generator(cfg, StubGenerator);
    let expanded = expander.expand(&sample_doc()).unwrap();
    assert_eq!(
        expanded.generated_queries,
        vec!["stub query 0".to_string(), "stub query 1".to_string()],
        "with_generator should delegate to the custom generator"
    );
}

#[test]
fn test_with_generator_appends_custom_queries_to_content() {
    let cfg = Doc2QueryConfig::new().with_num_queries(2);
    let expander = Doc2QueryExpander::with_generator(cfg, StubGenerator);
    let expanded = expander.expand(&sample_doc()).unwrap();
    assert!(
        expanded.expanded_content.contains("stub query 0"),
        "custom generator queries should be appended to the content"
    );
}

// ── Doc2QueryError display ────────────────────────────────────────────────────

#[test]
fn test_error_empty_document_display() {
    let e = Doc2QueryError::EmptyDocument;
    assert!(
        e.to_string().contains("empty"),
        "EmptyDocument display should mention 'empty', got: {e}"
    );
}

// ── ExpandedDocument field access ─────────────────────────────────────────────

#[test]
fn test_expanded_document_struct_field_access() {
    let original = Document::new("content").with_id(DocumentId::from_string("ed"));
    let ed = ExpandedDocument {
        original: original.clone(),
        generated_queries: vec!["What is content?".to_string()],
        expanded_content: "content\nWhat is content?".to_string(),
    };
    assert_eq!(
        ed.generated_queries.len(),
        1,
        "ExpandedDocument should expose generated_queries"
    );
}
