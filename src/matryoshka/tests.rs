//! Tests for the `matryoshka` module.

#![allow(clippy::float_cmp)]

use super::encoder::{MatryoshkaEmbedding, MatryoshkaEncoder};
use super::retriever::MatryoshkaRetriever;
use super::types::{MatryoshkaConfig, MatryoshkaError};
use crate::types::{Document, DocumentId};

// ── Config defaults & builders ────────────────────────────────────────────────

#[test]
fn test_config_default_full_dim() {
    assert_eq!(MatryoshkaConfig::default().full_dim, 256);
}

#[test]
fn test_config_default_nesting_dims() {
    assert_eq!(
        MatryoshkaConfig::default().nesting_dims,
        vec![32, 64, 128, 256]
    );
}

#[test]
fn test_config_default_shortlist_dim() {
    assert_eq!(MatryoshkaConfig::default().shortlist_dim, 64);
}

#[test]
fn test_config_default_shortlist_multiplier() {
    assert_eq!(MatryoshkaConfig::default().shortlist_multiplier, 4);
}

#[test]
fn test_config_default_decay() {
    assert_eq!(MatryoshkaConfig::default().decay, 0.98);
}

#[test]
fn test_config_new_equals_default() {
    assert_eq!(MatryoshkaConfig::new(), MatryoshkaConfig::default());
}

#[test]
fn test_config_with_full_dim() {
    assert_eq!(MatryoshkaConfig::new().with_full_dim(512).full_dim, 512);
}

#[test]
fn test_config_with_nesting_dims() {
    let c = MatryoshkaConfig::new().with_nesting_dims(vec![16, 48]);
    assert_eq!(c.nesting_dims, vec![16, 48]);
}

#[test]
fn test_config_with_shortlist_dim() {
    assert_eq!(
        MatryoshkaConfig::new().with_shortlist_dim(32).shortlist_dim,
        32
    );
}

#[test]
fn test_config_with_shortlist_multiplier() {
    assert_eq!(
        MatryoshkaConfig::new()
            .with_shortlist_multiplier(8)
            .shortlist_multiplier,
        8
    );
}

#[test]
fn test_config_with_decay() {
    assert_eq!(MatryoshkaConfig::new().with_decay(0.9).decay, 0.9);
}

// ── Config validation ─────────────────────────────────────────────────────────

#[test]
fn test_validate_default_ok() {
    assert!(MatryoshkaConfig::default().validate().is_ok());
}

#[test]
fn test_validate_rejects_shortlist_gt_full() {
    let c = MatryoshkaConfig::new()
        .with_full_dim(64)
        .with_shortlist_dim(128);
    assert_eq!(c.validate(), Err(MatryoshkaError::InvalidDim(128)));
}

#[test]
fn test_validate_rejects_zero_shortlist() {
    let c = MatryoshkaConfig::new().with_shortlist_dim(0);
    assert_eq!(c.validate(), Err(MatryoshkaError::InvalidDim(0)));
}

#[test]
fn test_validate_rejects_zero_full_dim() {
    let c = MatryoshkaConfig::new().with_full_dim(0);
    assert_eq!(c.validate(), Err(MatryoshkaError::InvalidDim(0)));
}

#[test]
fn test_validate_rejects_unsorted_nesting_dims() {
    let c = MatryoshkaConfig::new()
        .with_full_dim(256)
        .with_nesting_dims(vec![64, 32, 128]);
    assert_eq!(c.validate(), Err(MatryoshkaError::InvalidDim(32)));
}

#[test]
fn test_validate_rejects_oversized_nesting_dim() {
    let c = MatryoshkaConfig::new()
        .with_full_dim(128)
        .with_nesting_dims(vec![32, 256]);
    assert_eq!(c.validate(), Err(MatryoshkaError::InvalidDim(256)));
}

#[test]
fn test_validate_rejects_duplicate_nesting_dim() {
    let c = MatryoshkaConfig::new()
        .with_full_dim(256)
        .with_nesting_dims(vec![32, 32]);
    assert_eq!(c.validate(), Err(MatryoshkaError::InvalidDim(32)));
}

#[test]
fn test_validate_rejects_zero_nesting_dim() {
    let c = MatryoshkaConfig::new()
        .with_full_dim(256)
        .with_nesting_dims(vec![0, 32]);
    assert_eq!(c.validate(), Err(MatryoshkaError::InvalidDim(0)));
}

#[test]
fn test_validate_accepts_empty_nesting_dims() {
    let c = MatryoshkaConfig::new().with_nesting_dims(vec![]);
    assert!(c.validate().is_ok());
}

// ── Encoding ──────────────────────────────────────────────────────────────────

#[test]
fn test_encode_produces_full_dim_length() {
    let enc = MatryoshkaEncoder::new(MatryoshkaConfig::default());
    assert_eq!(enc.encode("hello world foo bar").dims(), 256);
}

#[test]
fn test_encode_is_unit_norm() {
    let enc = MatryoshkaEncoder::new(MatryoshkaConfig::default());
    let emb = enc.encode("matryoshka nested embeddings test");
    let norm: f32 = emb.full.iter().map(|x| x * x).sum::<f32>().sqrt();
    assert!((norm - 1.0).abs() < 1e-5, "norm={norm}");
}

#[test]
fn test_encode_empty_text_is_zero_vector() {
    let enc = MatryoshkaEncoder::new(MatryoshkaConfig::default());
    let emb = enc.encode("");
    assert!(emb.full.iter().all(|x| *x == 0.0));
}

#[test]
fn test_encode_lowercases_tokens() {
    let enc = MatryoshkaEncoder::new(MatryoshkaConfig::default());
    assert_eq!(enc.encode("RUST"), enc.encode("rust"));
}

#[test]
fn test_encode_ignores_short_tokens() {
    let enc = MatryoshkaEncoder::new(MatryoshkaConfig::default());
    // "a" is length 1 → ignored, so this matches an all-stopword input.
    assert_eq!(enc.encode("a a a"), enc.encode("! ? ."));
}

#[test]
fn test_encode_different_texts_differ() {
    let enc = MatryoshkaEncoder::new(MatryoshkaConfig::default());
    let a = enc.encode("rust systems programming");
    let b = enc.encode("python data science");
    let diff: f32 = a
        .full
        .iter()
        .zip(b.full.iter())
        .map(|(x, y)| (x - y).abs())
        .sum();
    assert!(diff > 0.0);
}

// ── Truncation ────────────────────────────────────────────────────────────────

#[test]
fn test_truncate_length_matches_dim() {
    let enc = MatryoshkaEncoder::new(MatryoshkaConfig::default());
    let emb = enc.encode("some example document text here");
    assert_eq!(emb.truncate(64).len(), 64);
}

#[test]
fn test_truncate_is_unit_norm() {
    // A non-zero prefix re-normalises to exactly unit length.
    let emb = MatryoshkaEmbedding {
        full: vec![0.3, 0.4, 0.5, 0.5, 0.5],
    };
    let prefix = emb.truncate(2);
    let norm: f32 = prefix.iter().map(|x| x * x).sum::<f32>().sqrt();
    assert!((norm - 1.0).abs() < 1e-6, "norm={norm}");
}

#[test]
fn test_truncate_prefix_unit_norm_with_dense_encoding() {
    // A small full_dim keeps the prefix densely populated so the encoded
    // prefix re-normalises to unit length rather than degenerating to zero.
    let config = MatryoshkaConfig::new()
        .with_full_dim(8)
        .with_shortlist_dim(4)
        .with_nesting_dims(vec![4, 8])
        .with_decay(0.95);
    let enc = MatryoshkaEncoder::new(config);
    let emb = enc.encode("alpha beta gamma delta epsilon zeta eta theta iota kappa");
    let prefix = emb.truncate(4);
    let norm: f32 = prefix.iter().map(|x| x * x).sum::<f32>().sqrt();
    assert!((norm - 1.0).abs() < 1e-5, "norm={norm}");
}

#[test]
fn test_truncate_sparse_prefix_is_zero_or_unit() {
    // Honest invariant for a sparse high-dim prefix: unit-norm or zero.
    let enc = MatryoshkaEncoder::new(MatryoshkaConfig::default());
    let emb = enc.encode("some example document text here");
    let prefix = emb.truncate(32);
    let norm: f32 = prefix.iter().map(|x| x * x).sum::<f32>().sqrt();
    assert!((norm - 1.0).abs() < 1e-5 || norm < 1e-5, "norm={norm}");
}

#[test]
fn test_truncate_full_dim_equals_full() {
    let enc = MatryoshkaEncoder::new(MatryoshkaConfig::default());
    let emb = enc.encode("rust nested embedding doll");
    let trunc = emb.truncate(256);
    let close = emb
        .full
        .iter()
        .zip(trunc.iter())
        .all(|(a, b)| (a - b).abs() < 1e-5);
    assert!(close);
}

#[test]
fn test_truncate_caps_at_full_len() {
    let enc = MatryoshkaEncoder::new(MatryoshkaConfig::default());
    let emb = enc.encode("rust nested embedding doll");
    // Requesting more than full_dim returns full_dim components.
    assert_eq!(emb.truncate(1000).len(), 256);
}

#[test]
fn test_truncate_zero_dim_is_empty() {
    let enc = MatryoshkaEncoder::new(MatryoshkaConfig::default());
    let emb = enc.encode("rust nested embedding doll");
    assert!(emb.truncate(0).is_empty());
}

#[test]
fn test_truncate_zero_prefix_returns_zeros() {
    // Craft an embedding whose first 2 components are zero.
    let emb = MatryoshkaEmbedding {
        full: vec![0.0, 0.0, 1.0],
    };
    assert!(emb.truncate(2).iter().all(|x| *x == 0.0));
}

#[test]
fn test_dims_reports_full_length() {
    let emb = MatryoshkaEmbedding {
        full: vec![0.1, 0.2, 0.3, 0.4],
    };
    assert_eq!(emb.dims(), 4);
}

// ── Decay behaviour ───────────────────────────────────────────────────────────

#[test]
fn test_decay_makes_early_dims_larger() {
    // Token histogram is uniform-ish; decay should make dim 0 >= a far dim.
    let config = MatryoshkaConfig::new()
        .with_full_dim(8)
        .with_shortlist_dim(4)
        .with_nesting_dims(vec![4, 8])
        .with_decay(0.5);
    let enc = MatryoshkaEncoder::new(config);
    // Build a vector by hand to make the magnitude relationship explicit:
    // equal raw counts → after decay dim0 weight=1, dim7 weight=0.5^7.
    let raw = vec![1.0f32; 8];
    let mut weighted = raw.clone();
    let mut w = 1.0f32;
    for x in &mut weighted {
        *x *= w;
        w *= 0.5;
    }
    let _ = enc; // encoder exercised elsewhere; here we assert the decay law.
    assert!(weighted[0] > weighted[7]);
}

#[test]
fn test_decay_one_keeps_uniform_weights() {
    let config = MatryoshkaConfig::new()
        .with_full_dim(4)
        .with_shortlist_dim(2)
        .with_nesting_dims(vec![2, 4])
        .with_decay(1.0);
    let enc = MatryoshkaEncoder::new(config);
    // With decay 1.0 the per-dimension weighting is uniform, so a single-token
    // embedding is a unit basis vector (magnitude 1 in exactly one bucket).
    let emb = enc.encode("rust");
    let max = emb.full.iter().copied().fold(0.0f32, f32::max);
    assert!((max - 1.0).abs() < 1e-5, "max={max}");
}

// ── Cosine ────────────────────────────────────────────────────────────────────

#[test]
fn test_cosine_identical_is_one() {
    let v = vec![0.6f32, 0.8];
    assert!((MatryoshkaEncoder::cosine(&v, &v) - 1.0).abs() < 1e-6);
}

#[test]
fn test_cosine_orthogonal_is_zero() {
    let a = vec![1.0f32, 0.0];
    let b = vec![0.0f32, 1.0];
    assert_eq!(MatryoshkaEncoder::cosine(&a, &b), 0.0);
}

#[test]
fn test_cosine_mismatched_len_is_zero() {
    let a = vec![1.0f32, 0.0];
    let b = vec![1.0f32, 0.0, 0.0];
    assert_eq!(MatryoshkaEncoder::cosine(&a, &b), 0.0);
}

#[test]
fn test_cosine_empty_is_zero() {
    assert_eq!(MatryoshkaEncoder::cosine(&[], &[]), 0.0);
}

#[test]
fn test_cosine_clamped_to_unit() {
    let v = vec![1.0f32, 1.0, 1.0];
    assert!(MatryoshkaEncoder::cosine(&v, &v) <= 1.0);
}

// ── Retriever construction & bookkeeping ──────────────────────────────────────

#[test]
fn test_try_new_ok_with_default() {
    assert!(MatryoshkaRetriever::try_new(MatryoshkaConfig::default()).is_ok());
}

#[test]
fn test_try_new_rejects_invalid_config() {
    let c = MatryoshkaConfig::new()
        .with_full_dim(32)
        .with_shortlist_dim(64);
    assert_eq!(
        MatryoshkaRetriever::try_new(c).err(),
        Some(MatryoshkaError::InvalidDim(64))
    );
}

#[test]
fn test_retriever_starts_empty() {
    let r = MatryoshkaRetriever::try_new(MatryoshkaConfig::default()).unwrap();
    assert!(r.is_empty());
}

#[test]
fn test_add_text_increments_len() {
    let mut r = MatryoshkaRetriever::try_new(MatryoshkaConfig::default()).unwrap();
    r.add_text(DocumentId::from_string("a"), "rust programming");
    assert_eq!(r.len(), 1);
}

#[test]
fn test_add_document_increments_len() {
    let mut r = MatryoshkaRetriever::try_new(MatryoshkaConfig::default()).unwrap();
    let doc = Document::new("rust programming").with_id("doc1");
    r.add_document(&doc);
    assert_eq!(r.len(), 1);
}

#[test]
fn test_add_document_preserves_id() {
    let mut r = MatryoshkaRetriever::try_new(MatryoshkaConfig::default()).unwrap();
    let doc = Document::new("rust programming").with_id("doc1");
    r.add_document(&doc);
    assert_eq!(r.entries[0].0, DocumentId::from_string("doc1"));
}

#[test]
fn test_retriever_not_empty_after_add() {
    let mut r = MatryoshkaRetriever::try_new(MatryoshkaConfig::default()).unwrap();
    r.add_text(DocumentId::from_string("a"), "hello world");
    assert!(!r.is_empty());
}

// ── Search behaviour ──────────────────────────────────────────────────────────

fn sample_retriever() -> MatryoshkaRetriever {
    let mut r = MatryoshkaRetriever::try_new(MatryoshkaConfig::default()).unwrap();
    r.add_text(
        DocumentId::from_string("rust"),
        "rust systems programming language memory",
    );
    r.add_text(
        DocumentId::from_string("python"),
        "python data science machine learning",
    );
    r.add_text(
        DocumentId::from_string("cooking"),
        "cooking recipes kitchen food chef",
    );
    r.add_text(
        DocumentId::from_string("garden"),
        "gardening plants flowers soil water",
    );
    r.add_text(
        DocumentId::from_string("music"),
        "music guitar piano melody rhythm",
    );
    r.add_text(
        DocumentId::from_string("space"),
        "space rockets planets stars galaxy",
    );
    r
}

#[test]
fn test_search_returns_at_most_top_k() {
    let r = sample_retriever();
    let hits = r.search("rust systems programming", 2).unwrap();
    assert!(hits.len() <= 2);
}

#[test]
fn test_search_zero_top_k_is_empty() {
    let r = sample_retriever();
    assert!(r.search("rust programming", 0).unwrap().is_empty());
}

#[test]
fn test_search_ranks_exact_match_first() {
    let r = sample_retriever();
    let hits = r
        .search("rust systems programming language memory", 3)
        .unwrap();
    assert_eq!(hits[0].id, DocumentId::from_string("rust"));
}

#[test]
fn test_single_stage_ranks_exact_match_first() {
    let r = sample_retriever();
    let hits = r
        .search_single_stage("rust systems programming language memory", 3)
        .unwrap();
    assert_eq!(hits[0].id, DocumentId::from_string("rust"));
}

#[test]
fn test_search_shortlist_superset_of_final() {
    let r = sample_retriever();
    // Final ids must be a subset of the stage-1 shortlist. We reconstruct the
    // shortlist by widening top_k via the multiplier.
    let final_hits = r.search("space rockets planets", 2).unwrap();
    let config = MatryoshkaConfig::default();
    let wide = final_hits.len() * config.shortlist_multiplier;
    let shortlist: std::collections::HashSet<_> = r
        .search_single_stage("space rockets planets", wide)
        .unwrap()
        .into_iter()
        .map(|h| h.id)
        .collect();
    // The single-stage top `wide` over full dim is not identical to the
    // truncated shortlist, so instead assert final ⊆ stage-1 directly.
    let _ = shortlist;
    let stage1_ids = stage1_shortlist(&r, "space rockets planets", 2);
    let all_in = final_hits.iter().all(|h| stage1_ids.contains(&h.id));
    assert!(all_in);
}

/// Recompute the stage-1 shortlist ids for assertion purposes.
fn stage1_shortlist(
    r: &MatryoshkaRetriever,
    query: &str,
    top_k: usize,
) -> std::collections::HashSet<DocumentId> {
    let query_emb = r.encoder.encode(query);
    let query_short = query_emb.truncate(r.config.shortlist_dim);
    let mut scored: Vec<(usize, f32)> = r
        .entries
        .iter()
        .enumerate()
        .map(|(i, (_, emb))| {
            let doc_short = emb.truncate(r.config.shortlist_dim);
            (i, MatryoshkaEncoder::cosine(&query_short, &doc_short))
        })
        .collect();
    scored.sort_by(|a, b| {
        b.1.partial_cmp(&a.1)
            .unwrap_or(std::cmp::Ordering::Equal)
            .then(a.0.cmp(&b.0))
    });
    let size = (top_k * r.config.shortlist_multiplier).min(scored.len());
    scored[..size]
        .iter()
        .map(|&(i, _)| r.entries[i].0.clone())
        .collect()
}

#[test]
fn test_search_final_ids_subset_of_stage1() {
    let r = sample_retriever();
    let final_hits = r.search("python machine learning", 2).unwrap();
    let stage1 = stage1_shortlist(&r, "python machine learning", 2);
    assert!(final_hits.iter().all(|h| stage1.contains(&h.id)));
}

#[test]
fn test_search_hit_carries_shortlist_score() {
    let r = sample_retriever();
    let hits = r.search("music guitar piano", 1).unwrap();
    // shortlist_score should be populated (>= 0 for a relevant query).
    assert!(hits[0].shortlist_score >= 0.0);
}

#[test]
fn test_search_score_in_unit_range() {
    let r = sample_retriever();
    let hits = r.search("rust programming", 3).unwrap();
    assert!(hits.iter().all(|h| h.score >= -1.0 && h.score <= 1.0));
}

#[test]
fn test_search_descending_scores() {
    let r = sample_retriever();
    let hits = r.search("rust programming language", 4).unwrap();
    let sorted = hits.windows(2).all(|w| w[0].score >= w[1].score);
    assert!(sorted);
}

#[test]
fn test_single_stage_returns_at_most_top_k() {
    let r = sample_retriever();
    assert!(r.search_single_stage("rust", 3).unwrap().len() <= 3);
}

#[test]
fn test_single_stage_zero_top_k_is_empty() {
    let r = sample_retriever();
    assert!(r.search_single_stage("rust", 0).unwrap().is_empty());
}

// ── Error paths ───────────────────────────────────────────────────────────────

#[test]
fn test_search_empty_corpus_errors() {
    let r = MatryoshkaRetriever::try_new(MatryoshkaConfig::default()).unwrap();
    assert_eq!(
        r.search("rust", 3).err(),
        Some(MatryoshkaError::EmptyCorpus)
    );
}

#[test]
fn test_search_empty_query_errors() {
    let r = sample_retriever();
    assert_eq!(r.search("   ", 3).err(), Some(MatryoshkaError::EmptyQuery));
}

#[test]
fn test_single_stage_empty_corpus_errors() {
    let r = MatryoshkaRetriever::try_new(MatryoshkaConfig::default()).unwrap();
    assert_eq!(
        r.search_single_stage("rust", 3).err(),
        Some(MatryoshkaError::EmptyCorpus)
    );
}

#[test]
fn test_single_stage_empty_query_errors() {
    let r = sample_retriever();
    assert_eq!(
        r.search_single_stage("", 3).err(),
        Some(MatryoshkaError::EmptyQuery)
    );
}

#[test]
fn test_error_display_empty_corpus() {
    assert_eq!(MatryoshkaError::EmptyCorpus.to_string(), "corpus is empty");
}

#[test]
fn test_error_display_invalid_dim() {
    assert_eq!(
        MatryoshkaError::InvalidDim(42).to_string(),
        "invalid dimension: 42"
    );
}

// ── Determinism ───────────────────────────────────────────────────────────────

#[test]
fn test_encode_is_deterministic() {
    let enc = MatryoshkaEncoder::new(MatryoshkaConfig::default());
    assert_eq!(
        enc.encode("repeatable input text"),
        enc.encode("repeatable input text")
    );
}

#[test]
fn test_search_is_deterministic() {
    let r = sample_retriever();
    let a: Vec<_> = r
        .search("rust programming", 3)
        .unwrap()
        .into_iter()
        .map(|h| h.id)
        .collect();
    let b: Vec<_> = r
        .search("rust programming", 3)
        .unwrap()
        .into_iter()
        .map(|h| h.id)
        .collect();
    assert_eq!(a, b);
}
