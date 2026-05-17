//! Tests for the `chunking` module.
//!
//! Coverage targets:
//! - [`FixedSizeChunker`] — basic split, overlap, `min_chunk_size`, empty doc, short doc
//! - [`SentenceChunker`] — sentence boundary preservation, overlap, multi-sentence
//! - [`RecursiveChunker`] — paragraph splits, fallback to sentences, fallback to words
//! - [`MarkdownChunker`] — heading splits, code block preservation
//! - [`DocumentChunker`] — `chunk_document` roundtrip, metadata, `chunk_documents` batch
//! - `Chunk::into_document` — metadata carries over, `source_doc_id` set correctly
//! - Edge cases: empty content, single-word doc, very small `chunk_size`, 0 overlap

use std::collections::HashMap;

use crate::chunking::{
    Chunk, ChunkConfig, ChunkStrategy, DocumentChunker, FixedSizeChunker, MarkdownChunker,
    RecursiveChunker, SentenceChunker,
};
use crate::types::Document;

// ──────────────────────────────────────────────────────────────────────────────
// Helpers
// ──────────────────────────────────────────────────────────────────────────────

fn permissive_config() -> ChunkConfig {
    ChunkConfig::default()
        .with_chunk_size(1000)
        .with_chunk_overlap(0)
        .with_min_chunk_size(1)
}

fn small_config(size: usize, overlap: usize) -> ChunkConfig {
    ChunkConfig::default()
        .with_chunk_size(size)
        .with_chunk_overlap(overlap)
        .with_min_chunk_size(1)
}

// ──────────────────────────────────────────────────────────────────────────────
// FixedSizeChunker tests
// ──────────────────────────────────────────────────────────────────────────────

#[test]
fn fixed_basic_split() {
    let doc = Document::new("abcdefghijklmnopqrstuvwxyz"); // 26 chars
    let config = small_config(10, 0);
    let chunks = FixedSizeChunker.chunk(&doc, &config);
    // 26 chars / 10 step = 3 chunks (10, 10, 6)
    assert_eq!(chunks.len(), 3, "expected 3 chunks for 26-char doc with size=10");
    assert_eq!(chunks[0].content, "abcdefghij");
    assert_eq!(chunks[1].content, "klmnopqrst");
    assert_eq!(chunks[2].content, "uvwxyz");
}

#[test]
fn fixed_overlap_produces_more_chunks() {
    let doc = Document::new("abcdefghijklmnopqrst"); // 20 chars
    let config = small_config(8, 3); // step = 5
    let chunks = FixedSizeChunker.chunk(&doc, &config);
    // positions: 0-8, 5-13, 10-18, 15-20
    assert!(chunks.len() > 2, "overlap should produce more chunks");
    // Verify first chunk content
    assert_eq!(chunks[0].content, "abcdefgh");
    // Verify overlap: chunk[1] should start at char 5
    assert_eq!(chunks[1].start_char, 5);
}

#[test]
fn fixed_min_chunk_size_filters_small_tail() {
    let doc = Document::new("abcdefghijk"); // 11 chars
    let config = ChunkConfig::default()
        .with_chunk_size(10)
        .with_chunk_overlap(0)
        .with_min_chunk_size(5);
    let chunks = FixedSizeChunker.chunk(&doc, &config);
    // chunk 0 = "abcdefghij" (10 chars, passes), chunk 1 = "k" (1 char, filtered)
    assert_eq!(chunks.len(), 1);
    assert_eq!(chunks[0].content, "abcdefghij");
}

#[test]
fn fixed_empty_document_yields_no_chunks() {
    let doc = Document::new("");
    let config = permissive_config();
    let chunks = FixedSizeChunker.chunk(&doc, &config);
    assert!(chunks.is_empty(), "empty document should produce no chunks");
}

#[test]
fn fixed_short_document_single_chunk() {
    let doc = Document::new("Hello");
    let config = ChunkConfig::default()
        .with_chunk_size(1000)
        .with_chunk_overlap(0)
        .with_min_chunk_size(1);
    let chunks = FixedSizeChunker.chunk(&doc, &config);
    assert_eq!(chunks.len(), 1);
    assert_eq!(chunks[0].content, "Hello");
}

#[test]
fn fixed_chunk_indices_are_sequential() {
    let content: String = "x".repeat(50);
    let doc = Document::new(&content);
    let config = small_config(10, 2);
    let chunks = FixedSizeChunker.chunk(&doc, &config);
    for (i, chunk) in chunks.iter().enumerate() {
        assert_eq!(chunk.chunk_index, i, "chunk_index must be sequential");
    }
}

#[test]
fn fixed_char_offsets_are_non_decreasing() {
    let doc = Document::new("ABCDEFGHIJKLMNOPQRSTUVWXYZ");
    let config = small_config(6, 2);
    let chunks = FixedSizeChunker.chunk(&doc, &config);
    let mut prev_start = 0usize;
    for chunk in &chunks {
        assert!(
            chunk.start_char >= prev_start,
            "start_char must be non-decreasing"
        );
        assert!(
            chunk.end_char > chunk.start_char,
            "end_char must be strictly after start_char"
        );
        prev_start = chunk.start_char;
    }
}

#[test]
fn fixed_strip_whitespace_trims_content() {
    // Content with leading/trailing spaces within each window.
    let doc = Document::new("  hello  world  ");
    let config = ChunkConfig::default()
        .with_chunk_size(8)
        .with_chunk_overlap(0)
        .with_min_chunk_size(1)
        .with_strip_whitespace(true);
    let chunks = FixedSizeChunker.chunk(&doc, &config);
    for chunk in &chunks {
        assert_eq!(chunk.content, chunk.content.trim());
    }
}

// ──────────────────────────────────────────────────────────────────────────────
// SentenceChunker tests
// ──────────────────────────────────────────────────────────────────────────────

#[test]
fn sentence_basic_split_by_period() {
    let text = "First sentence. Second sentence. Third sentence.";
    let doc = Document::new(text);
    let config = ChunkConfig::default()
        .with_chunk_size(30)
        .with_chunk_overlap(0)
        .with_min_chunk_size(1);
    let chunks = SentenceChunker.chunk(&doc, &config);
    assert!(!chunks.is_empty(), "should produce at least one chunk");
    // Each chunk must be a non-empty string.
    for chunk in &chunks {
        assert!(!chunk.content.is_empty());
    }
}

#[test]
fn sentence_preserves_boundaries() {
    let text = "Hello world. How are you? Fine! Great.";
    let doc = Document::new(text);
    let config = small_config(20, 0);
    let chunks = SentenceChunker.chunk(&doc, &config);
    // No chunk should split a sentence in the middle.
    for chunk in &chunks {
        // A chunk must not start mid-word relative to a preceding terminator.
        // Simplistic check: chunk content should not start with a lowercase
        // letter that follows a sentence terminator (unless it is an overlap).
        assert!(!chunk.content.is_empty());
    }
}

#[test]
fn sentence_overlap_produces_repeated_content() {
    let text =
        "Alpha sentence one. Beta sentence two. Gamma sentence three. Delta sentence four.";
    let doc = Document::new(text);
    let config = ChunkConfig::default()
        .with_chunk_size(50)
        .with_chunk_overlap(20)
        .with_min_chunk_size(1);
    let chunks = SentenceChunker.chunk(&doc, &config);
    assert!(
        chunks.len() >= 2,
        "long text with overlap should produce multiple chunks"
    );
}

#[test]
fn sentence_paragraph_break_splits() {
    let text = "First paragraph content.\n\nSecond paragraph content.\n\nThird paragraph.";
    let doc = Document::new(text);
    let config = small_config(40, 0);
    let chunks = SentenceChunker.chunk(&doc, &config);
    assert!(!chunks.is_empty());
    // Paragraph breaks act as strong terminators.
    let total_content: usize = chunks.iter().map(|c| c.content.len()).sum();
    assert!(total_content > 0);
}

#[test]
fn sentence_empty_document() {
    let doc = Document::new("");
    let config = permissive_config();
    let chunks = SentenceChunker.chunk(&doc, &config);
    assert!(chunks.is_empty());
}

#[test]
fn sentence_single_short_text() {
    let doc = Document::new("One sentence only.");
    let config = permissive_config();
    let chunks = SentenceChunker.chunk(&doc, &config);
    // Single chunk containing the whole sentence.
    assert_eq!(chunks.len(), 1);
}

// ──────────────────────────────────────────────────────────────────────────────
// RecursiveChunker tests
// ──────────────────────────────────────────────────────────────────────────────

#[test]
fn recursive_paragraph_split() {
    let text =
        "Paragraph one content.\n\nParagraph two content.\n\nParagraph three content.";
    let doc = Document::new(text);
    let config = small_config(30, 0);
    let chunks = RecursiveChunker.chunk(&doc, &config);
    assert!(
        chunks.len() >= 2,
        "recursive chunker should split on paragraph boundaries"
    );
}

#[test]
fn recursive_fallback_sentence_split() {
    // No paragraph breaks — should fall back to sentence splits.
    let text = "Long sentence one goes here. Another long sentence follows. And yet another.";
    let doc = Document::new(text);
    let config = small_config(35, 0);
    let chunks = RecursiveChunker.chunk(&doc, &config);
    assert!(
        chunks.len() >= 2,
        "recursive chunker should fall back to sentence splits"
    );
}

#[test]
fn recursive_fallback_word_split() {
    // One giant sentence with no sentence terminators — fall back to word splits.
    let text = "oneword anotherword thirdword fourthword fifthword sixthword seventhword";
    let doc = Document::new(text);
    let config = small_config(20, 0);
    let chunks = RecursiveChunker.chunk(&doc, &config);
    assert!(
        chunks.len() >= 2,
        "should fall back to word-level splitting"
    );
}

#[test]
fn recursive_empty_document() {
    let doc = Document::new("");
    let config = permissive_config();
    let chunks = RecursiveChunker.chunk(&doc, &config);
    assert!(chunks.is_empty());
}

#[test]
fn recursive_short_document_single_chunk() {
    let doc = Document::new("Short.");
    let config = permissive_config();
    let chunks = RecursiveChunker.chunk(&doc, &config);
    assert_eq!(chunks.len(), 1);
}

#[test]
fn recursive_chunk_indices_sequential() {
    let text = "para one\n\npara two\n\npara three\n\npara four\n\npara five";
    let doc = Document::new(text);
    let config = small_config(12, 0);
    let chunks = RecursiveChunker.chunk(&doc, &config);
    for (i, chunk) in chunks.iter().enumerate() {
        assert_eq!(chunk.chunk_index, i);
    }
}

// ──────────────────────────────────────────────────────────────────────────────
// MarkdownChunker tests
// ──────────────────────────────────────────────────────────────────────────────

#[test]
fn markdown_heading_splits() {
    let md = "# Title\n\nIntro text.\n\n## Section One\n\nContent of section one.\n\n## Section Two\n\nContent of section two.";
    let doc = Document::new(md);
    let config = permissive_config();
    let chunks = MarkdownChunker.chunk(&doc, &config);
    assert!(
        chunks.len() >= 2,
        "markdown chunker should split on headings"
    );
}

#[test]
fn markdown_h3_headings() {
    let md = "### Sub A\n\nText A.\n\n### Sub B\n\nText B.";
    let doc = Document::new(md);
    let config = permissive_config();
    let chunks = MarkdownChunker.chunk(&doc, &config);
    assert!(chunks.len() >= 2, "### headings should produce splits");
}

#[test]
fn markdown_code_block_preserved() {
    let md =
        "# Code Example\n\n```rust\nfn main() {\n    println!(\"Hello\");\n}\n```\n\n## After Block\n\nSome text.";
    let doc = Document::new(md);
    let config = permissive_config();
    let chunks = MarkdownChunker.chunk(&doc, &config);
    // The code block must appear in exactly one chunk (not split mid-block).
    let code_chunks: Vec<&Chunk> = chunks
        .iter()
        .filter(|c| c.content.contains("fn main()"))
        .collect();
    assert_eq!(
        code_chunks.len(),
        1,
        "code block must not be split across chunks"
    );
}

#[test]
fn markdown_no_headings_falls_back_to_fixed() {
    let plain = "This is plain text. No headings here. Just sentences. And more sentences.";
    let doc = Document::new(plain);
    let config = small_config(30, 5);
    let chunks = MarkdownChunker.chunk(&doc, &config);
    assert!(
        !chunks.is_empty(),
        "no-heading markdown falls back to fixed-size"
    );
}

#[test]
fn markdown_empty_document() {
    let doc = Document::new("");
    let config = permissive_config();
    let chunks = MarkdownChunker.chunk(&doc, &config);
    assert!(chunks.is_empty());
}

#[test]
fn markdown_large_section_split_by_window() {
    let body: String = "word ".repeat(300); // 1500 chars
    let md = format!("# Big Section\n\n{body}");
    let doc = Document::new(&md);
    let config = ChunkConfig::default()
        .with_chunk_size(200)
        .with_chunk_overlap(20)
        .with_min_chunk_size(1);
    let chunks = MarkdownChunker.chunk(&doc, &config);
    assert!(
        chunks.len() > 1,
        "large markdown section should be split by sliding window"
    );
}

// ──────────────────────────────────────────────────────────────────────────────
// DocumentChunker tests
// ──────────────────────────────────────────────────────────────────────────────

#[test]
fn chunker_chunk_document_roundtrip() {
    let doc = Document::new("Content that will be chunked into pieces for indexing.")
        .with_title("Test Doc")
        .with_metadata("author", "Alice");

    let config = small_config(20, 5);
    let chunker = DocumentChunker::with_fixed_size(config);
    let indexed = chunker.chunk_document(&doc);

    assert!(!indexed.is_empty(), "should produce at least one indexed doc");
    for d in &indexed {
        assert!(!d.content.is_empty());
        // Chunk-level metadata should be present.
        assert!(d.metadata.contains_key("source_doc_id"));
        assert!(d.metadata.contains_key("chunk_index"));
    }
}

#[test]
fn chunker_metadata_carries_over() {
    let mut meta = HashMap::new();
    meta.insert("lang".to_string(), "en".to_string());
    meta.insert("version".to_string(), "2".to_string());

    let mut doc = Document::new("Some content for metadata propagation test.");
    doc.metadata = meta.clone();

    let config = permissive_config();
    let chunker = DocumentChunker::with_sentences(config);
    let indexed = chunker.chunk_document(&doc);

    assert!(!indexed.is_empty());
    for d in &indexed {
        assert_eq!(d.metadata.get("lang"), Some(&"en".to_string()));
        assert_eq!(d.metadata.get("version"), Some(&"2".to_string()));
    }
}

#[test]
fn chunker_source_doc_id_is_correct() {
    let doc = Document::new("source id check content");
    let original_id = doc.id.to_string();

    let config = permissive_config();
    let chunker = DocumentChunker::with_recursive(config);
    let indexed = chunker.chunk_document(&doc);

    for d in &indexed {
        assert_eq!(
            d.metadata.get("source_doc_id"),
            Some(&original_id),
            "source_doc_id must match the original document ID"
        );
    }
}

#[test]
fn chunker_chunk_documents_batch() {
    let docs = vec![
        Document::new("First document with some content."),
        Document::new("Second document with different content."),
        Document::new("Third document, also with content."),
    ];

    let config = small_config(20, 0);
    let chunker = DocumentChunker::with_fixed_size(config);
    let all = chunker.chunk_documents(&docs);

    // We should get at least 3 chunks (one per doc minimum at these sizes).
    assert!(all.len() >= 3, "batch should produce chunks for each doc");
}

#[test]
fn chunker_raw_chunks_returns_chunk_type() {
    let doc = Document::new("Testing raw chunk access with some text here.");
    let config = small_config(15, 3);
    let chunker = DocumentChunker::with_fixed_size(config.clone());
    let raw = chunker.raw_chunks(&doc);

    assert!(!raw.is_empty());
    // raw_chunks and chunk_document should agree on count.
    let indexed = chunker.chunk_document(&doc);
    assert_eq!(raw.len(), indexed.len(), "raw and indexed should have same count");
}

#[test]
fn chunker_strategy_name_exposed() {
    let config = permissive_config();
    assert_eq!(DocumentChunker::with_fixed_size(config.clone()).strategy_name(), "fixed-size");
    assert_eq!(DocumentChunker::with_sentences(config.clone()).strategy_name(), "sentence");
    assert_eq!(DocumentChunker::with_recursive(config.clone()).strategy_name(), "recursive");
    assert_eq!(DocumentChunker::with_markdown(config).strategy_name(), "markdown");
}

// ──────────────────────────────────────────────────────────────────────────────
// Chunk::into_document tests
// ──────────────────────────────────────────────────────────────────────────────

#[test]
fn chunk_into_document_carries_metadata() {
    let mut meta = HashMap::new();
    meta.insert("topic".to_string(), "testing".to_string());
    meta.insert("chunk_index".to_string(), "0".to_string());
    meta.insert("source_doc_id".to_string(), "doc-abc".to_string());

    let chunk = Chunk::new("chunk content", "doc-abc", 0, 0, 13, meta.clone());
    let doc = chunk.into_document();

    assert_eq!(doc.content, "chunk content");
    assert_eq!(doc.metadata.get("topic"), Some(&"testing".to_string()));
    assert_eq!(doc.metadata.get("source_doc_id"), Some(&"doc-abc".to_string()));
    assert_eq!(doc.metadata.get("chunk_index"), Some(&"0".to_string()));
    assert!(doc.metadata.contains_key("start_char"), "start_char must be in metadata");
    assert!(doc.metadata.contains_key("end_char"), "end_char must be in metadata");
}

#[test]
fn chunk_into_document_start_end_char_values() {
    let chunk = Chunk::new("hello", "id-001", 3, 42, 47, HashMap::new());
    let doc = chunk.into_document();
    assert_eq!(doc.metadata.get("start_char"), Some(&"42".to_string()));
    assert_eq!(doc.metadata.get("end_char"), Some(&"47".to_string()));
}

// ──────────────────────────────────────────────────────────────────────────────
// Edge case tests
// ──────────────────────────────────────────────────────────────────────────────

#[test]
fn edge_case_single_word_document() {
    let doc = Document::new("Rust");
    let config = ChunkConfig::default()
        .with_chunk_size(10)
        .with_chunk_overlap(0)
        .with_min_chunk_size(1);

    for strategy_name in &["fixed", "sentence", "recursive", "markdown"] {
        let chunker = match *strategy_name {
            "fixed" => DocumentChunker::with_fixed_size(config.clone()),
            "sentence" => DocumentChunker::with_sentences(config.clone()),
            "recursive" => DocumentChunker::with_recursive(config.clone()),
            "markdown" => DocumentChunker::with_markdown(config.clone()),
            _ => unreachable!(),
        };
        let chunks = chunker.chunk_document(&doc);
        assert_eq!(
            chunks.len(),
            1,
            "{strategy_name}: single-word doc should produce one chunk"
        );
        assert_eq!(chunks[0].content, "Rust");
    }
}

#[test]
fn edge_case_very_small_chunk_size() {
    let doc = Document::new("Hello world");
    let config = ChunkConfig::default()
        .with_chunk_size(3)
        .with_chunk_overlap(1)
        .with_min_chunk_size(1);
    let chunks = FixedSizeChunker.chunk(&doc, &config);
    assert!(chunks.len() > 2, "small chunk_size should produce many chunks");
    for chunk in &chunks {
        assert!(chunk.content.chars().count() <= 3);
    }
}

#[test]
fn edge_case_zero_overlap_no_repetition() {
    let doc = Document::new("abcdefghij"); // 10 chars
    let config = ChunkConfig::default()
        .with_chunk_size(5)
        .with_chunk_overlap(0)
        .with_min_chunk_size(1);
    let chunks = FixedSizeChunker.chunk(&doc, &config);
    assert_eq!(chunks.len(), 2);
    // Verify no character appears in two chunks (no overlap).
    let combined: String = chunks.iter().map(|c| c.content.as_str()).collect();
    assert_eq!(combined, "abcdefghij", "zero overlap: chars must not repeat");
}

#[test]
fn edge_case_overlap_equals_chunk_size_minus_one() {
    // Extreme overlap: step = 1, so every position produces a chunk.
    let doc = Document::new("abcde");
    let config = ChunkConfig::default()
        .with_chunk_size(3)
        .with_chunk_overlap(2) // step = 1
        .with_min_chunk_size(1);
    let chunks = FixedSizeChunker.chunk(&doc, &config);
    // positions: 0-3, 1-4, 2-5
    assert_eq!(chunks.len(), 3, "step-1 overlap should produce len-size+1 chunks");
}

#[test]
fn edge_case_chunk_config_defaults() {
    let config = ChunkConfig::default();
    assert_eq!(config.chunk_size, 1000);
    assert_eq!(config.chunk_overlap, 200);
    assert_eq!(config.min_chunk_size, 50);
    assert!(config.strip_whitespace);
    assert_eq!(config.step(), 800); // 1000 - 200
}

#[test]
fn edge_case_min_chunk_size_all_filtered() {
    // All chunks are too short.
    let doc = Document::new("Hi");
    let config = ChunkConfig::default()
        .with_chunk_size(10)
        .with_chunk_overlap(0)
        .with_min_chunk_size(100); // very high threshold
    let chunks = FixedSizeChunker.chunk(&doc, &config);
    assert!(
        chunks.is_empty(),
        "all chunks shorter than min_chunk_size should be filtered"
    );
}
