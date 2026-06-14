#![allow(clippy::float_cmp, clippy::similar_names)]
//! Tests for the `summary_index` module.

use crate::summary_index::index::SummaryIndex;
use crate::summary_index::summarizer::{
    ExtractiveSummarizer, Summarizer, embed, split_sentences, tokenize,
};
use crate::summary_index::types::{SummaryConfig, SummaryIndexError};
use crate::types::Document;

// ── helpers ───────────────────────────────────────────────────────────────────

fn sentence_count(summary: &str) -> usize {
    split_sentences(summary).len()
}

fn doc_with_id(id: &str, content: &str) -> Document {
    Document::new(content).with_id(id)
}

// ── SummaryConfig: defaults & builders ──────────────────────────────────────────

#[test]
fn config_default_summary_sentences() {
    assert_eq!(SummaryConfig::default().summary_sentences, 3);
}

#[test]
fn config_default_dim() {
    assert_eq!(SummaryConfig::default().dim, 128);
}

#[test]
fn config_new_matches_default() {
    let a = SummaryConfig::new();
    let b = SummaryConfig::default();
    assert_eq!(a.summary_sentences, b.summary_sentences);
    assert_eq!(a.dim, b.dim);
}

#[test]
fn config_with_summary_sentences() {
    let c = SummaryConfig::new().with_summary_sentences(5);
    assert_eq!(c.summary_sentences, 5);
}

#[test]
fn config_with_dim() {
    let c = SummaryConfig::new().with_dim(64);
    assert_eq!(c.dim, 64);
}

#[test]
fn config_builders_chain() {
    let c = SummaryConfig::new().with_summary_sentences(2).with_dim(256);
    assert_eq!(c.summary_sentences, 2);
    assert_eq!(c.dim, 256);
}

#[test]
fn config_builders_are_independent() {
    let base = SummaryConfig::default();
    let modified = base.clone().with_summary_sentences(7);
    assert_eq!(base.summary_sentences, 3);
    assert_eq!(modified.summary_sentences, 7);
}

// ── tokenizer & sentence splitter ───────────────────────────────────────────────

#[test]
fn tokenize_lowercases() {
    assert_eq!(tokenize("Rust IS Fast"), vec!["rust", "is", "fast"]);
}

#[test]
fn tokenize_drops_short_tokens() {
    // "a" has length 1 and is dropped; "is" is kept.
    assert_eq!(tokenize("a is x"), vec!["is"]);
}

#[test]
fn tokenize_splits_on_non_alphanumeric() {
    assert_eq!(tokenize("foo,bar;baz"), vec!["foo", "bar", "baz"]);
}

#[test]
fn split_sentences_basic() {
    let s = split_sentences("One. Two! Three?");
    assert_eq!(s, vec!["One", "Two", "Three"]);
}

#[test]
fn split_sentences_ignores_empty() {
    let s = split_sentences("Hello...  world.");
    assert_eq!(s, vec!["Hello", "world"]);
}

#[test]
fn split_sentences_trims() {
    let s = split_sentences("  spaced sentence  .  next  .");
    assert_eq!(s, vec!["spaced sentence", "next"]);
}

// ── embedding ───────────────────────────────────────────────────────────────────

#[test]
fn embed_is_l2_normalised() {
    let v = embed("rust is fast and safe", 128);
    let norm: f32 = v.iter().map(|x| x * x).sum::<f32>().sqrt();
    assert!((norm - 1.0).abs() < 1e-5);
}

#[test]
fn embed_zero_dim_is_empty() {
    assert!(embed("anything", 0).is_empty());
}

#[test]
fn embed_empty_text_is_zero_vector() {
    let v = embed("", 32);
    assert_eq!(v.len(), 32);
    assert!(v.iter().all(|x| *x == 0.0));
}

#[test]
fn embed_is_deterministic() {
    assert_eq!(embed("repeatable text", 64), embed("repeatable text", 64));
}

// ── ExtractiveSummarizer: sentence budget ───────────────────────────────────────

#[test]
fn summary_respects_max_sentences() {
    let doc = Document::new(
        "Alpha beta gamma. Delta epsilon zeta. Eta theta iota. Kappa lambda mu. Nu xi omicron.",
    );
    let s = ExtractiveSummarizer::new().summarize(&doc, 3);
    assert!(sentence_count(&s) <= 3);
}

#[test]
fn summary_never_exceeds_budget_of_two() {
    let doc =
        Document::new("One sentence here. Two sentence here. Three sentence here. Four here.");
    let s = ExtractiveSummarizer::new().summarize(&doc, 2);
    assert!(sentence_count(&s) <= 2);
}

#[test]
fn summary_of_short_doc_returns_whole_doc() {
    // Document has fewer sentences than the budget → whole doc is returned.
    let doc = Document::new("Only one sentence.");
    let s = ExtractiveSummarizer::new().summarize(&doc, 3);
    assert_eq!(s, "Only one sentence");
}

#[test]
fn summary_of_two_sentence_doc_with_budget_three() {
    let doc = Document::new("First sentence. Second sentence.");
    let s = ExtractiveSummarizer::new().summarize(&doc, 3);
    assert_eq!(s, "First sentence Second sentence");
}

#[test]
fn summary_zero_budget_is_empty() {
    let doc = Document::new("Something here. Another thing.");
    let s = ExtractiveSummarizer::new().summarize(&doc, 0);
    assert!(s.is_empty());
}

#[test]
fn summary_empty_doc_is_empty() {
    let doc = Document::new("");
    let s = ExtractiveSummarizer::new().summarize(&doc, 3);
    assert!(s.is_empty());
}

// ── ExtractiveSummarizer: salience ──────────────────────────────────────────────

#[test]
fn summary_includes_salient_sentence() {
    // "retrieval" recurs heavily, making its sentences the most salient.
    let doc = Document::new(
        "The weather is nice today. \
         Retrieval augmented generation uses retrieval to ground answers. \
         Retrieval systems index documents for fast retrieval lookups. \
         Cats are popular pets. \
         A random unrelated filler statement about gardening.",
    );
    let s = ExtractiveSummarizer::new().summarize(&doc, 2);
    assert!(s.to_lowercase().contains("retrieval"));
}

#[test]
fn summary_picks_higher_tf_sentences() {
    // Two sentences share repeated tokens; one is isolated filler.
    let doc = Document::new(
        "Vector vector vector search powers modern systems. \
         Vector search uses vector embeddings extensively. \
         Bananas grow in the tropics.",
    );
    let s = ExtractiveSummarizer::new().summarize(&doc, 2);
    assert!(s.to_lowercase().contains("vector"));
    assert!(!s.to_lowercase().contains("bananas"));
}

#[test]
fn summary_drops_low_salience_filler() {
    let doc = Document::new(
        "Database indexing database indexing database performance matters greatly. \
         Indexing strategies improve database query database speed. \
         An off topic remark.",
    );
    let s = ExtractiveSummarizer::new().summarize(&doc, 1);
    assert!(s.to_lowercase().contains("database") || s.to_lowercase().contains("indexing"));
    assert!(!s.to_lowercase().contains("off topic"));
}

// ── ExtractiveSummarizer: original order preserved ──────────────────────────────

#[test]
fn summary_preserves_original_order() {
    // The most-salient terms appear in a LATE sentence and an EARLY sentence;
    // the selected ones must still come out in document order.
    let doc = Document::new(
        "Apple apple apple is an early salient sentence. \
         A neutral middle sentence with little weight. \
         Apple apple is a later salient sentence too.",
    );
    let s = ExtractiveSummarizer::new().summarize(&doc, 2);
    let early = s.find("early salient");
    let later = s.find("later salient");
    assert!(early.is_some(), "early sentence missing: {s}");
    assert!(later.is_some(), "later sentence missing: {s}");
    assert!(early < later, "order not preserved: {s}");
}

#[test]
fn summary_order_with_three_picks() {
    let doc = Document::new(
        "Token token token one. Token token two. Token three. Filler four word word.",
    );
    let s = ExtractiveSummarizer::new().summarize(&doc, 3);
    let one = s.find("one");
    let two = s.find("two");
    let three = s.find("three");
    if let (Some(o), Some(t), Some(th)) = (one, two, three) {
        assert!(o < t && t < th, "order not preserved: {s}");
    }
}

#[test]
fn summary_joined_by_single_space() {
    let doc = Document::new("First salient salient salient. Second salient salient. Third one.");
    let s = ExtractiveSummarizer::new().summarize(&doc, 2);
    assert!(!s.contains("  "), "double space found: {s}");
}

// ── ExtractiveSummarizer: determinism ───────────────────────────────────────────

#[test]
fn summary_is_deterministic() {
    let doc =
        Document::new("Alpha alpha beta. Gamma delta delta. Epsilon zeta. Eta theta theta theta.");
    let summ = ExtractiveSummarizer::new();
    let a = summ.summarize(&doc, 2);
    let b = summ.summarize(&doc, 2);
    assert_eq!(a, b);
}

#[test]
fn summary_tie_break_prefers_earlier_sentence() {
    // Two sentences have identical salience; the earlier should be chosen.
    let doc = Document::new("Equal weight here. Equal weight here. Distinct filler tail.");
    let s = ExtractiveSummarizer::new().summarize(&doc, 1);
    assert_eq!(sentence_count(&s), 1);
    assert!(s.contains("Equal weight here"));
}

// ── SummaryIndex: add / len / is_empty ──────────────────────────────────────────

#[test]
fn index_starts_empty() {
    let index = SummaryIndex::new(SummaryConfig::default());
    assert!(index.is_empty());
    assert_eq!(index.len(), 0);
}

#[test]
fn index_add_document_increments_len() {
    let mut index = SummaryIndex::new(SummaryConfig::default());
    index.add_document(&Document::new("Some content. More content here."));
    assert_eq!(index.len(), 1);
    assert!(!index.is_empty());
}

#[test]
fn index_add_documents_batch() {
    let mut index = SummaryIndex::new(SummaryConfig::default());
    let docs = vec![
        Document::new("First document body. Second sentence."),
        Document::new("Another document entirely. With more text."),
        Document::new("Third one here. Trailing sentence."),
    ];
    index.add_documents(&docs);
    assert_eq!(index.len(), 3);
}

#[test]
fn index_stores_a_summary_per_document() {
    let mut index = SummaryIndex::new(SummaryConfig::default());
    index.add_document(&Document::new("Lorem ipsum dolor. Sit amet consectetur."));
    index.add_document(&Document::new("Quick brown fox. Jumps over dog."));
    assert_eq!(index.summaries().len(), 2);
}

#[test]
fn index_summary_embedding_has_config_dim() {
    let mut index = SummaryIndex::new(SummaryConfig::default().with_dim(64));
    index.add_document(&Document::new(
        "Embedding dimension test sentence. Another one here.",
    ));
    assert_eq!(index.summaries()[0].embedding.len(), 64);
}

#[test]
fn index_summary_truncated_to_budget() {
    let mut index = SummaryIndex::new(SummaryConfig::default().with_summary_sentences(2));
    index.add_document(&Document::new(
        "One one one. Two two. Three three three three. Four. Five.",
    ));
    let summary = &index.summaries()[0].summary;
    assert!(sentence_count(summary) <= 2);
}

// ── SummaryIndex: search returns FULL document ──────────────────────────────────

#[test]
fn search_returns_full_parent_document_not_summary() {
    let content = "Machine learning is powerful. \
         Neural networks learn representations from data. \
         Gradient descent optimizes the network weights. \
         Deep learning stacks many layers. \
         Backpropagation computes the gradients efficiently.";
    let mut index = SummaryIndex::new(SummaryConfig::default());
    index.add_document(&Document::new(content));
    let hits = index
        .search("gradient descent weights", 1)
        .expect("search ok");
    assert_eq!(hits.len(), 1);
    // The returned document content must be the ORIGINAL full content.
    assert_eq!(hits[0].document.content, content);
}

#[test]
fn search_document_content_longer_than_summary() {
    let content = "Sentence one here. Sentence two here. Sentence three here. \
         Sentence four here. Sentence five here. Sentence six here.";
    let mut index = SummaryIndex::new(SummaryConfig::default().with_summary_sentences(2));
    index.add_document(&Document::new(content));
    let hits = index.search("sentence", 1).expect("search ok");
    // Full document is strictly longer than its (2-sentence) summary.
    assert!(hits[0].document.content.len() > hits[0].summary.len());
    assert_eq!(hits[0].document.content, content);
}

#[test]
fn search_hit_carries_summary_distinct_from_content() {
    let content = "First. Second. Third. Fourth. Fifth. Sixth. Seventh.";
    let mut index = SummaryIndex::new(SummaryConfig::default().with_summary_sentences(2));
    index.add_document(&Document::new(content));
    let hits = index.search("third", 1).expect("search ok");
    assert_ne!(hits[0].summary, hits[0].document.content);
}

#[test]
fn search_preserves_document_id() {
    let mut index = SummaryIndex::new(SummaryConfig::default());
    index.add_document(&doc_with_id(
        "doc-xyz",
        "Content sentence one. Content sentence two.",
    ));
    let hits = index.search("content", 1).expect("search ok");
    assert_eq!(hits[0].document.id.as_str(), "doc-xyz");
}

#[test]
fn search_preserves_document_metadata() {
    let mut index = SummaryIndex::new(SummaryConfig::default());
    let doc = Document::new("Topic sentence one. Topic sentence two.")
        .with_title("My Title")
        .with_metadata("lang", "en");
    index.add_document(&doc);
    let hits = index.search("topic", 1).expect("search ok");
    assert_eq!(hits[0].document.title.as_deref(), Some("My Title"));
    assert_eq!(
        hits[0].document.metadata.get("lang"),
        Some(&"en".to_string())
    );
}

// ── SummaryIndex: ranking & top_k ───────────────────────────────────────────────

#[test]
fn search_ranks_relevant_document_first() {
    let mut index = SummaryIndex::new(SummaryConfig::default());
    index.add_document(&doc_with_id(
        "cooking",
        "Cooking pasta requires boiling water. Add salt and stir the pasta well.",
    ));
    index.add_document(&doc_with_id(
        "quantum",
        "Quantum computing uses qubits. Superposition enables quantum parallelism.",
    ));
    let hits = index
        .search("qubits superposition quantum", 2)
        .expect("search ok");
    assert_eq!(hits[0].document.id.as_str(), "quantum");
}

#[test]
fn search_scores_are_descending() {
    let mut index = SummaryIndex::new(SummaryConfig::default());
    index.add_document(&Document::new(
        "Rust programming is fast. Rust ensures memory safety.",
    ));
    index.add_document(&Document::new("Python is dynamic. Python is widely used."));
    index.add_document(&Document::new(
        "Cooking recipes vary. Many enjoy baking bread.",
    ));
    let hits = index.search("rust memory safety", 3).expect("search ok");
    for pair in hits.windows(2) {
        assert!(pair[0].score >= pair[1].score, "scores not descending");
    }
}

#[test]
fn search_respects_top_k() {
    let mut index = SummaryIndex::new(SummaryConfig::default());
    for i in 0..6 {
        index.add_document(&Document::new(format!(
            "Document number {i}. Body text here {i}."
        )));
    }
    let hits = index.search("document body text", 3).expect("search ok");
    assert_eq!(hits.len(), 3);
}

#[test]
fn search_top_k_larger_than_corpus() {
    let mut index = SummaryIndex::new(SummaryConfig::default());
    index.add_document(&Document::new("Single document here. With two sentences."));
    let hits = index.search("document", 10).expect("search ok");
    assert_eq!(hits.len(), 1);
}

#[test]
fn search_top_k_zero_returns_empty() {
    let mut index = SummaryIndex::new(SummaryConfig::default());
    index.add_document(&Document::new("Content one. Content two."));
    let hits = index.search("content", 0).expect("search ok");
    assert!(hits.is_empty());
}

#[test]
fn search_returns_all_documents() {
    let mut index = SummaryIndex::new(SummaryConfig::default());
    index.add_document(&Document::new("Topic alpha here. Alpha continues here."));
    index.add_document(&Document::new("Topic beta here. Beta continues here."));
    let hits = index.search("topic", 5).expect("search ok");
    assert_eq!(hits.len(), 2);
}

// ── SummaryIndex: errors ────────────────────────────────────────────────────────

#[test]
fn search_empty_index_errors() {
    let index = SummaryIndex::new(SummaryConfig::default());
    let err = index.search("anything", 5).unwrap_err();
    assert!(matches!(err, SummaryIndexError::EmptyIndex));
}

#[test]
fn search_empty_query_errors() {
    let mut index = SummaryIndex::new(SummaryConfig::default());
    index.add_document(&Document::new("Some content. More content."));
    let err = index.search("", 5).unwrap_err();
    assert!(matches!(err, SummaryIndexError::EmptyQuery));
}

#[test]
fn search_whitespace_query_errors() {
    let mut index = SummaryIndex::new(SummaryConfig::default());
    index.add_document(&Document::new("Some content. More content."));
    let err = index.search("   \t  ", 5).unwrap_err();
    assert!(matches!(err, SummaryIndexError::EmptyQuery));
}

#[test]
fn empty_query_checked_before_empty_index() {
    // Empty query on an empty index → EmptyQuery wins (checked first).
    let index = SummaryIndex::new(SummaryConfig::default());
    let err = index.search("", 5).unwrap_err();
    assert!(matches!(err, SummaryIndexError::EmptyQuery));
}

#[test]
fn error_display_messages() {
    assert_eq!(SummaryIndexError::EmptyIndex.to_string(), "index is empty");
    assert_eq!(
        SummaryIndexError::EmptyQuery.to_string(),
        "query must not be empty"
    );
}

// ── SummaryIndex: determinism ───────────────────────────────────────────────────

#[test]
fn search_is_deterministic() {
    let build = || {
        let mut index = SummaryIndex::new(SummaryConfig::default());
        index.add_document(&doc_with_id(
            "a",
            "Vector search vector embeddings. Vector indexes data.",
        ));
        index.add_document(&doc_with_id(
            "b",
            "Graph traversal explores nodes. Edges connect nodes.",
        ));
        index.add_document(&doc_with_id(
            "c",
            "Cooking food requires heat. Recipes guide cooking.",
        ));
        index
    };
    let h1 = build().search("vector embeddings", 3).expect("search ok");
    let h2 = build().search("vector embeddings", 3).expect("search ok");
    assert_eq!(h1.len(), h2.len());
    for (a, b) in h1.iter().zip(h2.iter()) {
        assert_eq!(a.document.id.as_str(), b.document.id.as_str());
        assert_eq!(a.score, b.score);
    }
}

#[test]
fn search_tie_break_is_stable_by_doc_id() {
    // Two identical documents (distinct ids) → equal scores, ordered by id.
    let mut index = SummaryIndex::new(SummaryConfig::default());
    index.add_document(&doc_with_id(
        "zzz",
        "Identical body sentence. Second identical sentence.",
    ));
    index.add_document(&doc_with_id(
        "aaa",
        "Identical body sentence. Second identical sentence.",
    ));
    let hits = index
        .search("identical body sentence", 2)
        .expect("search ok");
    assert_eq!(hits[0].score, hits[1].score);
    assert_eq!(hits[0].document.id.as_str(), "aaa");
    assert_eq!(hits[1].document.id.as_str(), "zzz");
}

// ── SummaryIndex: custom summarizer ─────────────────────────────────────────────

/// A trivial summarizer that always returns a fixed marker string.
#[derive(Debug, Default)]
struct FixedSummarizer;

impl Summarizer for FixedSummarizer {
    fn summarize(&self, _doc: &Document, _max_sentences: usize) -> String {
        "fixed marker summary".to_string()
    }
}

#[test]
fn with_summarizer_uses_custom_implementation() {
    let mut index = SummaryIndex::with_summarizer(SummaryConfig::default(), FixedSummarizer);
    index.add_document(&Document::new(
        "Original document content. Second sentence here.",
    ));
    assert_eq!(index.summaries()[0].summary, "fixed marker summary");
}

#[test]
fn with_summarizer_still_returns_full_document() {
    let content = "Original document content. Second sentence here. Third sentence too.";
    let mut index = SummaryIndex::with_summarizer(SummaryConfig::default(), FixedSummarizer);
    index.add_document(&Document::new(content));
    let hits = index.search("fixed marker", 1).expect("search ok");
    // Even with a custom summarizer, the full parent document is returned.
    assert_eq!(hits[0].document.content, content);
    assert_eq!(hits[0].summary, "fixed marker summary");
}

// ── integration ─────────────────────────────────────────────────────────────────

#[test]
fn end_to_end_summary_index_flow() {
    let mut index = SummaryIndex::new(SummaryConfig::default().with_summary_sentences(2));
    index.add_documents(&[
        doc_with_id(
            "ml",
            "Machine learning trains models from data. \
             Supervised learning needs labeled examples. \
             Models generalize to unseen inputs. \
             Overfitting hurts generalization badly.",
        ),
        doc_with_id(
            "db",
            "Databases store structured records. \
             Indexes accelerate query execution. \
             Transactions guarantee atomicity. \
             Normalization reduces data redundancy.",
        ),
    ]);
    assert_eq!(index.len(), 2);

    let hits = index
        .search("indexes accelerate query execution", 2)
        .expect("search ok");
    assert_eq!(hits.len(), 2);
    assert_eq!(hits[0].document.id.as_str(), "db");
    // Returned document retains all four original sentences.
    assert!(
        hits[0]
            .document
            .content
            .contains("Normalization reduces data redundancy")
    );
    assert!(hits[0].score >= hits[1].score);
}
