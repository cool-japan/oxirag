//! Tests for the `late_chunking` module.
#![allow(clippy::float_cmp)]

use super::chunker::LateChunker;
use super::types::{LateChunkConfig, LateChunkError, LatePooling};

// ── Helpers ───────────────────────────────────────────────────────────────────

/// Build a document with `n` distinct alphanumeric tokens.
fn doc_with_tokens(n: usize) -> String {
    (0..n)
        .map(|i| format!("tok{i}"))
        .collect::<Vec<_>>()
        .join(" ")
}

/// L2 norm of a vector.
fn norm(v: &[f32]) -> f32 {
    v.iter().map(|x| x * x).sum::<f32>().sqrt()
}

// ── Default config ────────────────────────────────────────────────────────────

#[test]
fn test_default_dim() {
    assert_eq!(
        LateChunkConfig::default().dim,
        128,
        "default dim should be 128"
    );
}

#[test]
fn test_default_chunk_size_tokens() {
    assert_eq!(
        LateChunkConfig::default().chunk_size_tokens,
        64,
        "default chunk_size_tokens should be 64"
    );
}

#[test]
fn test_default_overlap_tokens() {
    assert_eq!(
        LateChunkConfig::default().overlap_tokens,
        8,
        "default overlap_tokens should be 8"
    );
}

#[test]
fn test_default_context_weight() {
    assert_eq!(
        LateChunkConfig::default().context_weight,
        0.3,
        "default context_weight should be 0.3"
    );
}

#[test]
fn test_default_pooling_is_mean() {
    assert_eq!(
        LateChunkConfig::default().pooling,
        LatePooling::Mean,
        "default pooling should be Mean"
    );
}

#[test]
fn test_late_pooling_default_is_mean() {
    assert_eq!(
        LatePooling::default(),
        LatePooling::Mean,
        "LatePooling::default should be Mean"
    );
}

#[test]
fn test_new_equals_default() {
    assert_eq!(
        LateChunkConfig::new(),
        LateChunkConfig::default(),
        "new() should equal default()"
    );
}

// ── Builders ──────────────────────────────────────────────────────────────────

#[test]
fn test_with_dim() {
    assert_eq!(
        LateChunkConfig::new().with_dim(256).dim,
        256,
        "with_dim should set dim"
    );
}

#[test]
fn test_with_chunk_size_tokens() {
    assert_eq!(
        LateChunkConfig::new()
            .with_chunk_size_tokens(32)
            .chunk_size_tokens,
        32,
        "with_chunk_size_tokens should set chunk_size_tokens"
    );
}

#[test]
fn test_with_overlap_tokens() {
    assert_eq!(
        LateChunkConfig::new().with_overlap_tokens(4).overlap_tokens,
        4,
        "with_overlap_tokens should set overlap_tokens"
    );
}

#[test]
fn test_with_context_weight() {
    assert_eq!(
        LateChunkConfig::new()
            .with_context_weight(0.75)
            .context_weight,
        0.75,
        "with_context_weight should set context_weight"
    );
}

#[test]
fn test_with_pooling() {
    assert_eq!(
        LateChunkConfig::new()
            .with_pooling(LatePooling::Max)
            .pooling,
        LatePooling::Max,
        "with_pooling should set pooling"
    );
}

#[test]
fn test_builders_chain() {
    let cfg = LateChunkConfig::new()
        .with_dim(64)
        .with_chunk_size_tokens(16)
        .with_overlap_tokens(4)
        .with_context_weight(0.5)
        .with_pooling(LatePooling::Max);
    assert_eq!(cfg.dim, 64, "chained dim should be 64");
    assert_eq!(
        cfg.chunk_size_tokens, 16,
        "chained chunk_size_tokens should be 16"
    );
    assert_eq!(cfg.overlap_tokens, 4, "chained overlap_tokens should be 4");
    assert_eq!(
        cfg.context_weight, 0.5,
        "chained context_weight should be 0.5"
    );
    assert_eq!(
        cfg.pooling,
        LatePooling::Max,
        "chained pooling should be Max"
    );
}

// ── Empty-document errors ─────────────────────────────────────────────────────

#[test]
fn test_encode_document_empty_string_errors() {
    let chunker = LateChunker::new(LateChunkConfig::default());
    assert!(
        matches!(
            chunker.encode_document(""),
            Err(LateChunkError::EmptyDocument)
        ),
        "empty string should yield EmptyDocument"
    );
}

#[test]
fn test_encode_document_only_punctuation_errors() {
    let chunker = LateChunker::new(LateChunkConfig::default());
    assert!(
        matches!(
            chunker.encode_document("!!! ... ???"),
            Err(LateChunkError::EmptyDocument)
        ),
        "punctuation-only doc should yield EmptyDocument"
    );
}

#[test]
fn test_naive_chunks_empty_string_errors() {
    let chunker = LateChunker::new(LateChunkConfig::default());
    assert!(
        matches!(chunker.naive_chunks(""), Err(LateChunkError::EmptyDocument)),
        "empty string should yield EmptyDocument for naive_chunks"
    );
}

#[test]
fn test_naive_chunks_only_punctuation_errors() {
    let chunker = LateChunker::new(LateChunkConfig::default());
    assert!(
        matches!(
            chunker.naive_chunks("- - -"),
            Err(LateChunkError::EmptyDocument)
        ),
        "punctuation-only doc should yield EmptyDocument for naive_chunks"
    );
}

#[test]
fn test_single_char_tokens_dropped_errors() {
    // Tokens of length < 2 are dropped, so a doc of single chars is empty.
    let chunker = LateChunker::new(LateChunkConfig::default());
    assert!(
        matches!(
            chunker.encode_document("a b c d"),
            Err(LateChunkError::EmptyDocument)
        ),
        "single-char tokens should be dropped, yielding EmptyDocument"
    );
}

// ── Span tiling ───────────────────────────────────────────────────────────────

#[test]
fn test_spans_tile_with_overlap() {
    // 10 tokens, window 4, overlap 1 => stride 3 => spans [0,4) [3,7) [6,10).
    let cfg = LateChunkConfig::new()
        .with_dim(16)
        .with_chunk_size_tokens(4)
        .with_overlap_tokens(1);
    let chunker = LateChunker::new(cfg);
    let chunks = chunker.encode_document(&doc_with_tokens(10)).unwrap();
    let spans: Vec<(usize, usize)> = chunks
        .iter()
        .map(|c| (c.token_start, c.token_end))
        .collect();
    assert_eq!(
        spans,
        vec![(0, 4), (3, 7), (6, 10)],
        "spans should tile with stride 3"
    );
}

#[test]
fn test_consecutive_spans_share_overlap() {
    let cfg = LateChunkConfig::new()
        .with_dim(16)
        .with_chunk_size_tokens(5)
        .with_overlap_tokens(2);
    let chunker = LateChunker::new(cfg);
    let chunks = chunker.encode_document(&doc_with_tokens(11)).unwrap();
    // stride = 3, overlap = 2: each next start = prev_start + 3.
    for pair in chunks.windows(2) {
        assert_eq!(
            pair[1].token_start,
            pair[0].token_start + 3,
            "consecutive chunk starts should advance by the stride (3)"
        );
    }
}

#[test]
fn test_last_span_ends_at_token_count() {
    let cfg = LateChunkConfig::new()
        .with_dim(16)
        .with_chunk_size_tokens(4)
        .with_overlap_tokens(1);
    let chunker = LateChunker::new(cfg);
    let chunks = chunker.encode_document(&doc_with_tokens(10)).unwrap();
    assert_eq!(
        chunks.last().unwrap().token_end,
        10,
        "last span should end at the token count"
    );
}

#[test]
fn test_first_span_starts_at_zero() {
    let chunker = LateChunker::new(LateChunkConfig::default());
    let chunks = chunker.encode_document(&doc_with_tokens(200)).unwrap();
    assert_eq!(chunks[0].token_start, 0, "first span should start at 0");
}

#[test]
fn test_no_overlap_partitions_exactly() {
    // 12 tokens, window 4, overlap 0 => spans [0,4) [4,8) [8,12).
    let cfg = LateChunkConfig::new()
        .with_dim(16)
        .with_chunk_size_tokens(4)
        .with_overlap_tokens(0);
    let chunker = LateChunker::new(cfg);
    let chunks = chunker.encode_document(&doc_with_tokens(12)).unwrap();
    let spans: Vec<(usize, usize)> = chunks
        .iter()
        .map(|c| (c.token_start, c.token_end))
        .collect();
    assert_eq!(
        spans,
        vec![(0, 4), (4, 8), (8, 12)],
        "zero overlap should partition exactly"
    );
}

#[test]
fn test_overlap_geq_chunk_size_clamped() {
    // overlap >= chunk_size must be clamped so progress is still made.
    let cfg = LateChunkConfig::new()
        .with_dim(16)
        .with_chunk_size_tokens(4)
        .with_overlap_tokens(10);
    let chunker = LateChunker::new(cfg);
    let chunks = chunker.encode_document(&doc_with_tokens(9)).unwrap();
    // Clamped overlap = 3 => stride 1 => spans start at 0,1,2,...,5 then end at 9.
    assert!(
        chunks.len() >= 2,
        "clamped overlap should still produce multiple chunks, got {}",
        chunks.len()
    );
    assert_eq!(
        chunks.last().unwrap().token_end,
        9,
        "clamped overlap last span should reach token count"
    );
}

// ── Chunk counts ──────────────────────────────────────────────────────────────

#[test]
fn test_chunk_count_no_overlap() {
    // 12 tokens, window 4, overlap 0 => 3 chunks.
    let cfg = LateChunkConfig::new()
        .with_dim(16)
        .with_chunk_size_tokens(4)
        .with_overlap_tokens(0);
    let chunker = LateChunker::new(cfg);
    let chunks = chunker.encode_document(&doc_with_tokens(12)).unwrap();
    assert_eq!(chunks.len(), 3, "12 tokens / window 4 => 3 chunks");
}

#[test]
fn test_chunk_count_with_overlap() {
    // 10 tokens, window 4, overlap 1 => stride 3 => 3 chunks.
    let cfg = LateChunkConfig::new()
        .with_dim(16)
        .with_chunk_size_tokens(4)
        .with_overlap_tokens(1);
    let chunker = LateChunker::new(cfg);
    let chunks = chunker.encode_document(&doc_with_tokens(10)).unwrap();
    assert_eq!(
        chunks.len(),
        3,
        "10 tokens, window 4, overlap 1 => 3 chunks"
    );
}

#[test]
fn test_doc_shorter_than_chunk_one_chunk() {
    // 5 tokens, window 64 => single chunk spanning the whole doc.
    let chunker = LateChunker::new(LateChunkConfig::default());
    let chunks = chunker.encode_document(&doc_with_tokens(5)).unwrap();
    assert_eq!(
        chunks.len(),
        1,
        "doc shorter than chunk size should be one chunk"
    );
}

#[test]
fn test_doc_equal_to_chunk_one_chunk() {
    let cfg = LateChunkConfig::new()
        .with_dim(16)
        .with_chunk_size_tokens(8);
    let chunker = LateChunker::new(cfg);
    let chunks = chunker.encode_document(&doc_with_tokens(8)).unwrap();
    assert_eq!(
        chunks.len(),
        1,
        "doc equal to chunk size should be one chunk"
    );
}

#[test]
fn test_single_token_doc_one_chunk() {
    let chunker = LateChunker::new(LateChunkConfig::default());
    let chunks = chunker.encode_document("solitary").unwrap();
    assert_eq!(chunks.len(), 1, "single-token doc should produce one chunk");
    assert_eq!(chunks[0].token_start, 0, "single-token chunk starts at 0");
    assert_eq!(chunks[0].token_end, 1, "single-token chunk ends at 1");
}

// ── Span text reconstruction ──────────────────────────────────────────────────

#[test]
fn test_chunk_text_reconstruction() {
    let cfg = LateChunkConfig::new()
        .with_dim(16)
        .with_chunk_size_tokens(3)
        .with_overlap_tokens(0);
    let chunker = LateChunker::new(cfg);
    let chunks = chunker.encode_document("Alpha Beta Gamma Delta").unwrap();
    assert_eq!(
        chunks[0].text, "alpha beta gamma",
        "first chunk text should be the lower-cased first three tokens"
    );
}

// ── Embedding length ──────────────────────────────────────────────────────────

#[test]
fn test_embedding_length_matches_dim() {
    let cfg = LateChunkConfig::new()
        .with_dim(96)
        .with_chunk_size_tokens(8);
    let chunker = LateChunker::new(cfg);
    let chunks = chunker.encode_document(&doc_with_tokens(20)).unwrap();
    for c in &chunks {
        assert_eq!(
            c.embedding.len(),
            96,
            "chunk embedding length should equal dim"
        );
    }
}

#[test]
fn test_naive_embedding_length_matches_dim() {
    let cfg = LateChunkConfig::new()
        .with_dim(48)
        .with_chunk_size_tokens(8);
    let chunker = LateChunker::new(cfg);
    let chunks = chunker.naive_chunks(&doc_with_tokens(20)).unwrap();
    for c in &chunks {
        assert_eq!(
            c.embedding.len(),
            48,
            "naive chunk embedding length should equal dim"
        );
    }
}

#[test]
fn test_query_embedding_length_matches_dim() {
    let cfg = LateChunkConfig::new().with_dim(72);
    let chunker = LateChunker::new(cfg);
    let q = chunker.encode_query("what is the answer to everything");
    assert_eq!(q.len(), 72, "query embedding length should equal dim");
}

// ── L2 normalisation ──────────────────────────────────────────────────────────

#[test]
fn test_chunk_embeddings_l2_normalized_mean() {
    let chunker = LateChunker::new(LateChunkConfig::default());
    let chunks = chunker.encode_document(&doc_with_tokens(150)).unwrap();
    for c in &chunks {
        let n = norm(&c.embedding);
        assert!(
            (n - 1.0).abs() < 1e-5,
            "mean-pooled chunk embedding should be unit-normalised: norm={n}"
        );
    }
}

#[test]
fn test_chunk_embeddings_l2_normalized_max() {
    let cfg = LateChunkConfig::default().with_pooling(LatePooling::Max);
    let chunker = LateChunker::new(cfg);
    let chunks = chunker.encode_document(&doc_with_tokens(150)).unwrap();
    for c in &chunks {
        let n = norm(&c.embedding);
        assert!(
            (n - 1.0).abs() < 1e-5,
            "max-pooled chunk embedding should be unit-normalised: norm={n}"
        );
    }
}

#[test]
fn test_naive_embeddings_l2_normalized() {
    let chunker = LateChunker::new(LateChunkConfig::default());
    let chunks = chunker.naive_chunks(&doc_with_tokens(150)).unwrap();
    for c in &chunks {
        let n = norm(&c.embedding);
        assert!(
            (n - 1.0).abs() < 1e-5,
            "naive chunk embedding should be unit-normalised: norm={n}"
        );
    }
}

#[test]
fn test_query_embedding_l2_normalized() {
    let chunker = LateChunker::new(LateChunkConfig::default());
    let q = chunker.encode_query("rust systems programming language");
    let n = norm(&q);
    assert!(
        (n - 1.0).abs() < 1e-5,
        "query embedding should be unit-normalised: norm={n}"
    );
}

#[test]
fn test_empty_query_zero_vector() {
    let chunker = LateChunker::new(LateChunkConfig::default());
    let q = chunker.encode_query("!!!");
    assert!(
        q.iter().all(|x| *x == 0.0),
        "query with no tokens should produce a zero vector"
    );
}

// ── context_weight == 0 => late equals naive ──────────────────────────────────

#[test]
fn test_zero_context_weight_late_equals_naive() {
    let cfg = LateChunkConfig::new()
        .with_dim(64)
        .with_chunk_size_tokens(8)
        .with_overlap_tokens(2)
        .with_context_weight(0.0);
    let chunker = LateChunker::new(cfg);
    let text = doc_with_tokens(40);
    let late = chunker.encode_document(&text).unwrap();
    let naive = chunker.naive_chunks(&text).unwrap();
    assert_eq!(
        late.len(),
        naive.len(),
        "late and naive should produce the same number of chunks"
    );
    for (l, n) in late.iter().zip(naive.iter()) {
        for (a, b) in l.embedding.iter().zip(n.embedding.iter()) {
            assert!(
                (a - b).abs() < 1e-6,
                "with context_weight=0 late embeddings should equal naive: {a} vs {b}"
            );
        }
    }
}

#[test]
fn test_zero_context_weight_late_equals_naive_max() {
    let cfg = LateChunkConfig::new()
        .with_dim(64)
        .with_chunk_size_tokens(8)
        .with_overlap_tokens(0)
        .with_context_weight(0.0)
        .with_pooling(LatePooling::Max);
    let chunker = LateChunker::new(cfg);
    let text = doc_with_tokens(32);
    let late = chunker.encode_document(&text).unwrap();
    let naive = chunker.naive_chunks(&text).unwrap();
    for (l, n) in late.iter().zip(naive.iter()) {
        for (a, b) in l.embedding.iter().zip(n.embedding.iter()) {
            assert!(
                (a - b).abs() < 1e-6,
                "with context_weight=0 and max pooling late should equal naive: {a} vs {b}"
            );
        }
    }
}

#[test]
fn test_nonzero_context_weight_differs_from_naive() {
    let cfg = LateChunkConfig::new()
        .with_dim(64)
        .with_chunk_size_tokens(8)
        .with_overlap_tokens(0)
        .with_context_weight(0.5);
    let chunker = LateChunker::new(cfg);
    let text = doc_with_tokens(40);
    let late = chunker.encode_document(&text).unwrap();
    let naive = chunker.naive_chunks(&text).unwrap();
    let total_diff: f32 = late
        .iter()
        .zip(naive.iter())
        .flat_map(|(l, n)| {
            l.embedding
                .iter()
                .zip(n.embedding.iter())
                .map(|(a, b)| (a - b).abs())
        })
        .sum();
    assert!(
        total_diff > 1e-3,
        "non-zero context weight should make late chunks differ from naive: total_diff={total_diff}"
    );
}

// ── Mean vs Max pooling differ ────────────────────────────────────────────────

#[test]
fn test_mean_and_max_pooling_differ() {
    // With a small dimensionality, the six distinct tokens collide into an uneven
    // bucket distribution ([3, 1, 2]). Mean pooling weights buckets by token
    // count while max pooling caps each bucket, so the two pooled vectors point in
    // different directions even after L2-normalisation.
    let text = doc_with_tokens(6);
    let mean_chunker = LateChunker::new(
        LateChunkConfig::new()
            .with_dim(3)
            .with_chunk_size_tokens(6)
            .with_pooling(LatePooling::Mean),
    );
    let max_chunker = LateChunker::new(
        LateChunkConfig::new()
            .with_dim(3)
            .with_chunk_size_tokens(6)
            .with_pooling(LatePooling::Max),
    );
    let mean_chunk = &mean_chunker.encode_document(&text).unwrap()[0];
    let max_chunk = &max_chunker.encode_document(&text).unwrap()[0];
    let diff: f32 = mean_chunk
        .embedding
        .iter()
        .zip(max_chunk.embedding.iter())
        .map(|(a, b)| (a - b).abs())
        .sum();
    assert!(
        diff > 1e-3,
        "mean and max pooling should differ on a multi-token chunk: diff={diff}"
    );
}

#[test]
fn test_single_token_mean_equals_max() {
    // With a single token in a span, mean and max pooling coincide.
    let text = "lonely";
    let mean_chunk = LateChunker::new(
        LateChunkConfig::new()
            .with_dim(32)
            .with_pooling(LatePooling::Mean),
    )
    .encode_document(text)
    .unwrap()[0]
        .embedding
        .clone();
    let max_chunk = LateChunker::new(
        LateChunkConfig::new()
            .with_dim(32)
            .with_pooling(LatePooling::Max),
    )
    .encode_document(text)
    .unwrap()[0]
        .embedding
        .clone();
    for (a, b) in mean_chunk.iter().zip(max_chunk.iter()) {
        assert!(
            (a - b).abs() < 1e-6,
            "single-token mean and max pooling should match: {a} vs {b}"
        );
    }
}

// ── Cosine ────────────────────────────────────────────────────────────────────

#[test]
fn test_cosine_identical_is_one() {
    let v = vec![0.3f32, 0.4, 0.5, 0.1];
    let c = LateChunker::cosine(&v, &v);
    assert!(
        (c - 1.0).abs() < 1e-5,
        "cosine of identical vectors should be 1: {c}"
    );
}

#[test]
fn test_cosine_orthogonal_is_zero() {
    let a = vec![1.0f32, 0.0, 0.0];
    let b = vec![0.0f32, 1.0, 0.0];
    let c = LateChunker::cosine(&a, &b);
    assert!(
        (c - 0.0).abs() < 1e-6,
        "cosine of orthogonal vectors should be 0: {c}"
    );
}

#[test]
fn test_cosine_opposite_is_negative_one() {
    let a = vec![1.0f32, 0.0];
    let b = vec![-1.0f32, 0.0];
    let c = LateChunker::cosine(&a, &b);
    assert!(
        (c + 1.0).abs() < 1e-6,
        "cosine of opposite vectors should be -1: {c}"
    );
}

#[test]
fn test_cosine_length_mismatch_is_zero() {
    let a = vec![1.0f32, 0.0];
    let b = vec![1.0f32, 0.0, 0.0];
    assert_eq!(
        LateChunker::cosine(&a, &b),
        0.0,
        "cosine of mismatched-length vectors should be 0"
    );
}

#[test]
fn test_cosine_empty_is_zero() {
    let a: Vec<f32> = Vec::new();
    let b: Vec<f32> = Vec::new();
    assert_eq!(
        LateChunker::cosine(&a, &b),
        0.0,
        "cosine of empty vectors should be 0"
    );
}

#[test]
fn test_cosine_zero_vector_is_zero() {
    let a = vec![0.0f32, 0.0, 0.0];
    let b = vec![1.0f32, 0.0, 0.0];
    assert_eq!(
        LateChunker::cosine(&a, &b),
        0.0,
        "cosine with a zero vector should be 0"
    );
}

#[test]
fn test_cosine_same_token_query_chunk_high() {
    // A query and a single-token doc sharing the exact token should align.
    let chunker = LateChunker::new(
        LateChunkConfig::new()
            .with_dim(128)
            .with_context_weight(0.0),
    );
    let chunk = &chunker.encode_document("photosynthesis").unwrap()[0];
    let q = chunker.encode_query("photosynthesis");
    let c = LateChunker::cosine(&chunk.embedding, &q);
    assert!(
        (c - 1.0).abs() < 1e-5,
        "identical single token chunk and query should have cosine ~1: {c}"
    );
}

// ── Determinism ───────────────────────────────────────────────────────────────

#[test]
fn test_encode_document_deterministic() {
    let chunker = LateChunker::new(LateChunkConfig::default());
    let text = doc_with_tokens(120);
    let a = chunker.encode_document(&text).unwrap();
    let b = chunker.encode_document(&text).unwrap();
    assert_eq!(
        a.len(),
        b.len(),
        "two runs should produce equal chunk counts"
    );
    for (ca, cb) in a.iter().zip(b.iter()) {
        assert_eq!(ca.text, cb.text, "chunk text should be deterministic");
        assert_eq!(
            ca.token_start, cb.token_start,
            "token_start should be deterministic"
        );
        assert_eq!(
            ca.token_end, cb.token_end,
            "token_end should be deterministic"
        );
        assert_eq!(
            ca.embedding, cb.embedding,
            "embeddings should be deterministic"
        );
    }
}

#[test]
fn test_naive_chunks_deterministic() {
    let chunker = LateChunker::new(LateChunkConfig::default());
    let text = doc_with_tokens(80);
    let a = chunker.naive_chunks(&text).unwrap();
    let b = chunker.naive_chunks(&text).unwrap();
    for (ca, cb) in a.iter().zip(b.iter()) {
        assert_eq!(
            ca.embedding, cb.embedding,
            "naive embeddings should be deterministic"
        );
    }
}

#[test]
fn test_encode_query_deterministic() {
    let chunker = LateChunker::new(LateChunkConfig::default());
    let a = chunker.encode_query("deterministic query encoding test");
    let b = chunker.encode_query("deterministic query encoding test");
    assert_eq!(a, b, "query encoding should be deterministic");
}

// ── Document-context effect ───────────────────────────────────────────────────

#[test]
fn test_higher_context_weight_pulls_chunks_together() {
    // With more document context, chunk embeddings from the same document should
    // become more similar to one another (they share the doc mean).
    let text = format!("{} {}", doc_with_tokens(20), {
        (20..40)
            .map(|i| format!("wd{i}"))
            .collect::<Vec<_>>()
            .join(" ")
    });

    let low = LateChunker::new(
        LateChunkConfig::new()
            .with_dim(128)
            .with_chunk_size_tokens(10)
            .with_overlap_tokens(0)
            .with_context_weight(0.05),
    );
    let high = LateChunker::new(
        LateChunkConfig::new()
            .with_dim(128)
            .with_chunk_size_tokens(10)
            .with_overlap_tokens(0)
            .with_context_weight(0.9),
    );

    let low_chunks = low.encode_document(&text).unwrap();
    let high_chunks = high.encode_document(&text).unwrap();
    assert!(
        low_chunks.len() >= 2,
        "need at least two chunks for this test"
    );

    let low_sim = LateChunker::cosine(&low_chunks[0].embedding, &low_chunks[1].embedding);
    let high_sim = LateChunker::cosine(&high_chunks[0].embedding, &high_chunks[1].embedding);
    assert!(
        high_sim > low_sim,
        "higher context weight should increase inter-chunk similarity: high={high_sim}, low={low_sim}"
    );
}

#[test]
fn test_late_differs_from_naive_for_default_config() {
    let chunker = LateChunker::new(
        LateChunkConfig::default()
            .with_chunk_size_tokens(20)
            .with_overlap_tokens(0),
    );
    let text = doc_with_tokens(80);
    let late = chunker.encode_document(&text).unwrap();
    let naive = chunker.naive_chunks(&text).unwrap();
    let diff: f32 = late
        .iter()
        .zip(naive.iter())
        .flat_map(|(l, n)| {
            l.embedding
                .iter()
                .zip(n.embedding.iter())
                .map(|(a, b)| (a - b).abs())
        })
        .sum();
    assert!(
        diff > 1e-3,
        "default config late chunking should differ from naive: diff={diff}"
    );
}

// ── Edge configs ──────────────────────────────────────────────────────────────

#[test]
fn test_zero_dim_yields_empty_embeddings() {
    let cfg = LateChunkConfig::new().with_dim(0).with_chunk_size_tokens(4);
    let chunker = LateChunker::new(cfg);
    let chunks = chunker.encode_document(&doc_with_tokens(8)).unwrap();
    for c in &chunks {
        assert!(
            c.embedding.is_empty(),
            "zero dim should produce empty embeddings"
        );
    }
}

#[test]
fn test_chunk_size_one_one_chunk_per_token() {
    let cfg = LateChunkConfig::new()
        .with_dim(16)
        .with_chunk_size_tokens(1)
        .with_overlap_tokens(0);
    let chunker = LateChunker::new(cfg);
    let chunks = chunker.encode_document(&doc_with_tokens(5)).unwrap();
    assert_eq!(
        chunks.len(),
        5,
        "chunk size 1 should produce one chunk per token"
    );
    for (i, c) in chunks.iter().enumerate() {
        assert_eq!(c.token_start, i, "chunk {i} should start at token {i}");
        assert_eq!(
            c.token_end,
            i + 1,
            "chunk {i} should end at token {}",
            i + 1
        );
    }
}

#[test]
fn test_all_chunks_cover_every_token() {
    // Union of all spans should cover [0, n).
    let cfg = LateChunkConfig::new()
        .with_dim(16)
        .with_chunk_size_tokens(5)
        .with_overlap_tokens(2);
    let chunker = LateChunker::new(cfg);
    let n = 23;
    let chunks = chunker.encode_document(&doc_with_tokens(n)).unwrap();
    let mut covered = vec![false; n];
    for c in &chunks {
        for slot in covered.iter_mut().take(c.token_end).skip(c.token_start) {
            *slot = true;
        }
    }
    assert!(
        covered.iter().all(|&x| x),
        "every token should be covered by some chunk"
    );
}
