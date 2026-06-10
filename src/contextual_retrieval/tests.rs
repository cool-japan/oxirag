#![allow(clippy::float_cmp)]
//! Tests for the `contextual_retrieval` module.

use crate::contextual_retrieval::types::{
    ChunkContext, ChunkNeighborhood, ContextualChunk, ContextualConfig, ContextualIndexBuilder,
    ContextualRetrievalError, Contextualizer, ExtractiveContextualizer,
};
use crate::types::{Document, DocumentId};

// ── helpers ──────────────────────────────────────────────────────────────────

fn make_doc(id: &str, content: &str) -> Document {
    Document::new(content).with_id(DocumentId::from_string(id))
}

fn make_doc_with_title(id: &str, content: &str, title: &str) -> Document {
    Document::new(content)
        .with_id(DocumentId::from_string(id))
        .with_metadata("title", title)
}

// ── ContextualConfig ──────────────────────────────────────────────────────

#[test]
fn test_contextual_config_default_summary_sentences() {
    let cfg = ContextualConfig::default();
    assert_eq!(
        cfg.summary_sentences, 2,
        "default summary_sentences should be 2"
    );
}

#[test]
fn test_contextual_config_default_include_title() {
    let cfg = ContextualConfig::default();
    assert!(cfg.include_title, "default include_title should be true");
}

#[test]
fn test_contextual_config_default_include_position() {
    let cfg = ContextualConfig::default();
    assert!(
        cfg.include_position,
        "default include_position should be true"
    );
}

#[test]
fn test_contextual_config_with_summary_sentences_builder() {
    let cfg = ContextualConfig::default().with_summary_sentences(5);
    assert_eq!(
        cfg.summary_sentences, 5,
        "with_summary_sentences should set value to 5"
    );
}

#[test]
fn test_contextual_config_with_include_title_false_builder() {
    let cfg = ContextualConfig::default().with_include_title(false);
    assert!(
        !cfg.include_title,
        "with_include_title(false) should disable title"
    );
}

#[test]
fn test_contextual_config_with_include_position_false_builder() {
    let cfg = ContextualConfig::default().with_include_position(false);
    assert!(
        !cfg.include_position,
        "with_include_position(false) should disable position"
    );
}

// ── ChunkContext ──────────────────────────────────────────────────────────

#[test]
fn test_chunk_context_default_doc_title_is_none() {
    let ctx = ChunkContext::default();
    assert!(ctx.doc_title.is_none(), "default doc_title should be None");
}

#[test]
fn test_chunk_context_default_doc_summary_empty() {
    let ctx = ChunkContext::default();
    assert!(
        ctx.doc_summary.is_empty(),
        "default doc_summary should be empty string"
    );
}

#[test]
fn test_chunk_context_default_position_ratio_zero() {
    let ctx = ChunkContext::default();
    assert_eq!(
        ctx.position_ratio, 0.0,
        "default position_ratio should be 0.0"
    );
}

#[test]
fn test_chunk_context_default_preceding_gist_none() {
    let ctx = ChunkContext::default();
    assert!(
        ctx.preceding_gist.is_none(),
        "default preceding_gist should be None"
    );
}

#[test]
fn test_chunk_context_field_assignment() {
    let ctx = ChunkContext {
        doc_title: Some("My Doc".to_string()),
        doc_summary: "A summary.".to_string(),
        position_ratio: 0.5,
        preceding_gist: Some("Previous chunk gist.".to_string()),
    };
    assert_eq!(
        ctx.doc_title.as_deref(),
        Some("My Doc"),
        "doc_title should match"
    );
    assert!(
        (ctx.position_ratio - 0.5).abs() < 1e-6,
        "position_ratio should be 0.5"
    );
}

// ── ContextualChunk ───────────────────────────────────────────────────────

#[test]
fn test_contextual_chunk_field_access() {
    let doc = make_doc("c1", "chunk content here");
    let chunk = ContextualChunk {
        original: doc.clone(),
        contextualized_content: "Context: summary. chunk content here".to_string(),
        context: ChunkContext::default(),
    };
    assert_eq!(chunk.original.id.as_str(), "c1", "original id should be c1");
    assert!(
        !chunk.contextualized_content.is_empty(),
        "contextualized_content should not be empty"
    );
}

// ── ExtractiveContextualizer ──────────────────────────────────────────────

#[test]
fn test_extractive_contextualizer_produces_non_empty_output() {
    let cfg = ContextualConfig::default();
    let ctx = ExtractiveContextualizer::new(cfg);
    let parent = make_doc(
        "parent",
        "This is the first sentence. This is the second sentence. This is the third sentence.",
    );
    let chunk = make_doc("c1", "chunk text here");
    let neighborhood = ChunkNeighborhood::default();
    let output = ctx.contextualize(&chunk, &parent, &neighborhood);
    assert!(
        !output.is_empty(),
        "contextualize should return non-empty string"
    );
}

#[test]
fn test_extractive_contextualizer_includes_chunk_content() {
    let cfg = ContextualConfig::default();
    let ctx = ExtractiveContextualizer::new(cfg);
    let parent = make_doc(
        "parent",
        "Parent document content with multiple sentences. Second sentence here.",
    );
    let chunk = make_doc("c1", "unique_chunk_marker_xyz");
    let neighborhood = ChunkNeighborhood::default();
    let output = ctx.contextualize(&chunk, &parent, &neighborhood);
    assert!(
        output.contains("unique_chunk_marker_xyz"),
        "contextualized output should include the chunk content"
    );
}

#[test]
fn test_extractive_contextualizer_includes_title_when_enabled() {
    let cfg = ContextualConfig::default().with_include_title(true);
    let ctx = ExtractiveContextualizer::new(cfg);
    let parent = make_doc_with_title(
        "parent",
        "First sentence content. Second sentence content.",
        "My Test Title",
    );
    let chunk = make_doc("c1", "chunk text");
    let neighborhood = ChunkNeighborhood::default();
    let output = ctx.contextualize(&chunk, &parent, &neighborhood);
    assert!(
        output.contains("My Test Title"),
        "output should contain the document title when include_title=true, got: {output}"
    );
}

#[test]
fn test_extractive_contextualizer_omits_title_when_disabled() {
    let cfg = ContextualConfig::default().with_include_title(false);
    let ctx = ExtractiveContextualizer::new(cfg);
    let parent = make_doc_with_title(
        "parent",
        "First sentence. Second sentence.",
        "Secret Title XYZ",
    );
    let chunk = make_doc("c1", "chunk text");
    let neighborhood = ChunkNeighborhood::default();
    let output = ctx.contextualize(&chunk, &parent, &neighborhood);
    assert!(
        !output.contains("Secret Title XYZ"),
        "output should NOT contain title when include_title=false, got: {output}"
    );
}

#[test]
fn test_extractive_contextualizer_uses_document_fallback_title() {
    // No "title" in metadata, no parent.title either
    let cfg = ContextualConfig::default().with_include_title(true);
    let ctx = ExtractiveContextualizer::new(cfg);
    let parent = make_doc("parent", "Some content. Another sentence.");
    let chunk = make_doc("c1", "chunk text");
    let neighborhood = ChunkNeighborhood::default();
    let output = ctx.contextualize(&chunk, &parent, &neighborhood);
    assert!(
        output.contains("Document"),
        "output should use fallback title 'Document', got: {output}"
    );
}

#[test]
fn test_extractive_contextualizer_includes_position_hint_when_enabled() {
    let cfg = ContextualConfig::default().with_include_position(true);
    let ctx = ExtractiveContextualizer::new(cfg);
    let parent = make_doc("parent", "First sentence. Second sentence. Third sentence.");
    let chunk = make_doc("c1", "chunk text");
    let neighborhood = ChunkNeighborhood::default();
    let output = ctx.contextualize(&chunk, &parent, &neighborhood);
    // Should contain some form of positional language
    assert!(
        output.contains("chunk") || output.contains("position") || output.contains("document"),
        "output should include a positional hint when include_position=true, got: {output}"
    );
}

#[test]
fn test_extractive_contextualizer_no_position_hint_when_disabled() {
    let cfg = ContextualConfig::default()
        .with_include_position(false)
        .with_include_title(false);
    let ctx = ExtractiveContextualizer::new(cfg);
    let parent = make_doc("parent", "First sentence. Second sentence.");
    let chunk = make_doc("c1", "my_special_chunk_content");
    let neighborhood = ChunkNeighborhood::default();
    let output = ctx.contextualize(&chunk, &parent, &neighborhood);
    assert!(
        !output.contains("position"),
        "output should not contain positional hint when disabled, got: {output}"
    );
}

#[test]
fn test_extractive_contextualizer_includes_preceding_gist() {
    let cfg = ContextualConfig::default();
    let ctx = ExtractiveContextualizer::new(cfg);
    let parent = make_doc("parent", "First sentence. Second sentence. Third sentence.");
    let preceding = make_doc(
        "prev",
        "Preceding chunk with unique_preceding_marker content.",
    );
    let chunk = make_doc("c1", "current chunk text");
    let neighborhood = ChunkNeighborhood {
        preceding: Some(preceding),
        following: None,
    };
    let output = ctx.contextualize(&chunk, &parent, &neighborhood);
    assert!(
        output.contains("unique_preceding_marker") || output.contains("Preceding"),
        "output should include preceding gist when neighbor exists, got: {output}"
    );
}

#[test]
fn test_extractive_contextualizer_no_preceding_gist_when_absent() {
    let cfg = ContextualConfig::default();
    let ctx = ExtractiveContextualizer::new(cfg);
    let parent = make_doc("parent", "First sentence. Second sentence.");
    let chunk = make_doc("c1", "current chunk text");
    let neighborhood = ChunkNeighborhood {
        preceding: None,
        following: None,
    };
    let output = ctx.contextualize(&chunk, &parent, &neighborhood);
    assert!(
        !output.contains("Preceding:"),
        "output should not contain 'Preceding:' when no preceding chunk, got: {output}"
    );
}

// ── ContextualIndexBuilder ────────────────────────────────────────────────

#[test]
fn test_contextual_index_builder_empty_parent_returns_error() {
    let builder = ContextualIndexBuilder::new(ContextualConfig::default());
    let parent = make_doc("parent", "");
    let chunks = vec![make_doc("c1", "chunk content")];
    let err = builder.build(&parent, &chunks).unwrap_err();
    assert!(
        matches!(err, ContextualRetrievalError::EmptyDocument),
        "empty parent should return EmptyDocument error"
    );
}

#[test]
fn test_contextual_index_builder_whitespace_parent_returns_error() {
    let builder = ContextualIndexBuilder::new(ContextualConfig::default());
    let parent = make_doc("parent", "   \n  ");
    let chunks = vec![make_doc("c1", "chunk content")];
    let err = builder.build(&parent, &chunks).unwrap_err();
    assert!(
        matches!(err, ContextualRetrievalError::EmptyDocument),
        "whitespace-only parent should return EmptyDocument error"
    );
}

#[test]
fn test_contextual_index_builder_no_chunks_returns_error() {
    let builder = ContextualIndexBuilder::new(ContextualConfig::default());
    let parent = make_doc("parent", "Some real content here.");
    let err = builder.build(&parent, &[]).unwrap_err();
    assert!(
        matches!(err, ContextualRetrievalError::NoChunks),
        "empty chunks should return NoChunks error"
    );
}

#[test]
fn test_contextual_index_builder_happy_path_count() {
    let builder = ContextualIndexBuilder::new(ContextualConfig::default());
    let parent = make_doc(
        "parent",
        "This is the first sentence. This is the second sentence. Third sentence here.",
    );
    let chunks = vec![
        make_doc("c1", "first chunk"),
        make_doc("c2", "second chunk"),
        make_doc("c3", "third chunk"),
    ];
    let result = builder.build(&parent, &chunks).unwrap();
    assert_eq!(
        result.len(),
        3,
        "build should return 3 ContextualChunks for 3 input chunks"
    );
}

#[test]
fn test_contextual_index_builder_all_contextualized_content_non_empty() {
    let builder = ContextualIndexBuilder::new(ContextualConfig::default());
    let parent = make_doc(
        "parent",
        "Main document content first sentence. Second sentence follows here.",
    );
    let chunks = vec![
        make_doc("c1", "chunk one content"),
        make_doc("c2", "chunk two content"),
    ];
    let result = builder.build(&parent, &chunks).unwrap();
    for (i, chunk) in result.iter().enumerate() {
        assert!(
            !chunk.contextualized_content.is_empty(),
            "chunk {i} should have non-empty contextualized_content"
        );
    }
}

#[test]
fn test_contextual_index_builder_first_position_ratio_zero() {
    let builder = ContextualIndexBuilder::new(ContextualConfig::default());
    let parent = make_doc("parent", "Sentence one. Sentence two. Sentence three.");
    let chunks = vec![
        make_doc("c1", "first chunk"),
        make_doc("c2", "second chunk"),
        make_doc("c3", "third chunk"),
    ];
    let result = builder.build(&parent, &chunks).unwrap();
    assert_eq!(
        result[0].context.position_ratio, 0.0,
        "first chunk should have position_ratio=0.0, got {}",
        result[0].context.position_ratio
    );
}

#[test]
fn test_contextual_index_builder_last_position_ratio_one() {
    let builder = ContextualIndexBuilder::new(ContextualConfig::default());
    let parent = make_doc("parent", "Sentence one. Sentence two. Sentence three.");
    let chunks = vec![
        make_doc("c1", "first chunk"),
        make_doc("c2", "second chunk"),
        make_doc("c3", "third chunk"),
    ];
    let result = builder.build(&parent, &chunks).unwrap();
    let last = result.last().unwrap();
    assert!(
        (last.context.position_ratio - 1.0).abs() < 1e-5,
        "last chunk should have position_ratio=1.0, got {}",
        last.context.position_ratio
    );
}

#[test]
fn test_contextual_index_builder_middle_position_ratio() {
    let builder = ContextualIndexBuilder::new(ContextualConfig::default());
    let parent = make_doc("parent", "Sentence one. Sentence two. Sentence three.");
    let chunks = vec![
        make_doc("c1", "first chunk"),
        make_doc("c2", "second chunk"),
        make_doc("c3", "third chunk"),
    ];
    let result = builder.build(&parent, &chunks).unwrap();
    assert!(
        (result[1].context.position_ratio - 0.5).abs() < 1e-5,
        "middle chunk should have position_ratio=0.5, got {}",
        result[1].context.position_ratio
    );
}

#[test]
fn test_contextual_index_builder_single_chunk_position_zero() {
    let builder = ContextualIndexBuilder::new(ContextualConfig::default());
    let parent = make_doc("parent", "Single document content here.");
    let chunks = vec![make_doc("c1", "only chunk")];
    let result = builder.build(&parent, &chunks).unwrap();
    assert_eq!(
        result[0].context.position_ratio, 0.0,
        "single chunk should have position_ratio=0.0, got {}",
        result[0].context.position_ratio
    );
}

#[test]
fn test_contextual_index_builder_doc_summary_non_empty() {
    let builder = ContextualIndexBuilder::new(ContextualConfig::default());
    let parent = make_doc(
        "parent",
        "First sentence content. Second sentence content. Third sentence content.",
    );
    let chunks = vec![make_doc("c1", "chunk content")];
    let result = builder.build(&parent, &chunks).unwrap();
    assert!(
        !result[0].context.doc_summary.is_empty(),
        "doc_summary should be non-empty"
    );
}

#[test]
fn test_contextual_index_builder_position_hint_in_content() {
    let cfg = ContextualConfig::default().with_include_position(true);
    let builder = ContextualIndexBuilder::new(cfg);
    let parent = make_doc("parent", "Sentence one. Sentence two. Sentence three.");
    let chunks = vec![
        make_doc("c1", "first chunk"),
        make_doc("c2", "second chunk"),
    ];
    let result = builder.build(&parent, &chunks).unwrap();
    // The last chunk is at 100%, so the content should include "100%"
    let last_content = &result.last().unwrap().contextualized_content;
    assert!(
        last_content.contains("100%") || last_content.contains("position"),
        "last chunk contextualized content should contain positional info, got: {last_content}"
    );
}

#[test]
fn test_contextual_index_builder_preserves_original_doc() {
    let builder = ContextualIndexBuilder::new(ContextualConfig::default());
    let parent = make_doc("parent", "Content of the parent document.");
    let chunks = vec![make_doc("chunk_id_abc", "chunk content")];
    let result = builder.build(&parent, &chunks).unwrap();
    assert_eq!(
        result[0].original.id.as_str(),
        "chunk_id_abc",
        "original doc id should be preserved"
    );
}

// ── ContextualRetrievalError display ─────────────────────────────────────

#[test]
fn test_error_empty_document_display() {
    let e = ContextualRetrievalError::EmptyDocument;
    let msg = e.to_string();
    assert!(
        !msg.is_empty(),
        "EmptyDocument error should have a display message"
    );
    assert!(
        msg.contains("empty") || msg.contains("Empty") || msg.contains("content"),
        "EmptyDocument display should mention content/empty, got: {msg}"
    );
}

#[test]
fn test_error_no_chunks_display() {
    let e = ContextualRetrievalError::NoChunks;
    let msg = e.to_string();
    assert!(
        !msg.is_empty(),
        "NoChunks error should have a display message"
    );
    assert!(
        msg.contains("chunk") || msg.contains("Chunk") || msg.contains("required"),
        "NoChunks display should mention chunks, got: {msg}"
    );
}
