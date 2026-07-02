//! Comprehensive tests for the `instruction_embed` module (instruction-
//! conditioned embeddings — INSTRUCTOR, Su et al. 2023; TART, Asai et al.
//! 2023).

#![allow(
    clippy::float_cmp,
    clippy::similar_names,
    clippy::cast_precision_loss,
    clippy::cast_possible_truncation,
    clippy::too_many_lines,
    clippy::unreadable_literal,
    clippy::doc_markdown,
    clippy::many_single_char_names,
    clippy::default_trait_access
)]

use super::engine::{InstructionEmbedder, InstructionIndex, InstructionRegistry};
use super::types::{
    InstructionEmbedConfig, InstructionEmbedError, InstructionEmbedding, TaskInstruction,
};
use crate::types::Document;

// ── Helpers ───────────────────────────────────────────────────────────────────

fn doc(id: &str, content: &str) -> Document {
    Document::new(content).with_id(id)
}

fn norm(vector: &[f32]) -> f32 {
    vector.iter().map(|x| x * x).sum::<f32>().sqrt()
}

/// A corpus with two topically disjoint documents, used throughout the
/// crafted-ranking tests.
fn ranking_corpus() -> Vec<Document> {
    vec![
        doc(
            "physics",
            "quantum entanglement particle wavefunction physics laboratory experiment",
        ),
        doc(
            "cooking",
            "tomato basil garlic olive pasta recipe kitchen dinner",
        ),
    ]
}

fn physics_query_instruction() -> TaskInstruction {
    TaskInstruction::new(
        "physics_query",
        "quantum entanglement particle wavefunction physics laboratory experiment",
    )
}

fn cooking_query_instruction() -> TaskInstruction {
    TaskInstruction::new(
        "cooking_query",
        "tomato basil garlic olive pasta recipe kitchen dinner",
    )
}

/// A config with a strong-enough influence and enough dimensions for the
/// crafted-corpus ranking-flip demonstrations to have a comfortable margin.
fn ranking_config() -> InstructionEmbedConfig {
    InstructionEmbedConfig::new()
        .with_dim(64)
        .with_instruction_influence(2.5)
}

// ── InstructionEmbedConfig: defaults & builders ──────────────────────────────

#[test]
fn test_config_default_dim() {
    assert_eq!(InstructionEmbedConfig::default().dim, 128);
}

#[test]
fn test_config_default_instruction_influence() {
    assert_eq!(
        InstructionEmbedConfig::default().instruction_influence,
        0.65
    );
}

#[test]
fn test_config_default_normalize() {
    assert!(InstructionEmbedConfig::default().normalize);
}

#[test]
fn test_config_default_top_k() {
    assert_eq!(InstructionEmbedConfig::default().default_top_k, 10);
}

#[test]
fn test_config_new_equals_default() {
    assert_eq!(
        InstructionEmbedConfig::new(),
        InstructionEmbedConfig::default()
    );
}

#[test]
fn test_config_with_dim() {
    assert_eq!(InstructionEmbedConfig::new().with_dim(64).dim, 64);
}

#[test]
fn test_config_with_instruction_influence() {
    assert_eq!(
        InstructionEmbedConfig::new()
            .with_instruction_influence(1.5)
            .instruction_influence,
        1.5
    );
}

#[test]
fn test_config_with_normalize() {
    assert!(
        !InstructionEmbedConfig::new()
            .with_normalize(false)
            .normalize
    );
}

#[test]
fn test_config_with_default_top_k() {
    assert_eq!(
        InstructionEmbedConfig::new()
            .with_default_top_k(5)
            .default_top_k,
        5
    );
}

// ── InstructionEmbedConfig: validate ─────────────────────────────────────────

#[test]
fn test_validate_default_ok() {
    assert!(InstructionEmbedConfig::default().validate().is_ok());
}

#[test]
fn test_validate_rejects_zero_dim() {
    let config = InstructionEmbedConfig::new().with_dim(0);
    assert_eq!(config.validate(), Err(InstructionEmbedError::InvalidDim(0)));
}

#[test]
fn test_validate_nonzero_dim_ok() {
    let config = InstructionEmbedConfig::new().with_dim(1);
    assert!(config.validate().is_ok());
}

// ── TaskInstruction ───────────────────────────────────────────────────────────

#[test]
fn test_task_instruction_new_sets_name_and_text() {
    let instruction = TaskInstruction::new("sci_doc", "Represent the document for retrieval:");
    assert_eq!(instruction.name, "sci_doc");
    assert_eq!(instruction.text, "Represent the document for retrieval:");
}

#[test]
fn test_task_instruction_from_text_name_equals_text() {
    let instruction = TaskInstruction::from_text("Represent the document for retrieval:");
    assert_eq!(instruction.name, instruction.text);
}

#[test]
fn test_task_instruction_is_blank_true_for_empty() {
    assert!(TaskInstruction::new("x", "").is_blank());
}

#[test]
fn test_task_instruction_is_blank_true_for_whitespace() {
    assert!(TaskInstruction::new("x", "   \t\n").is_blank());
}

#[test]
fn test_task_instruction_is_blank_false_for_nonblank() {
    assert!(!TaskInstruction::new("x", "represent the document").is_blank());
}

#[test]
fn test_task_instruction_equality() {
    assert_eq!(
        TaskInstruction::new("a", "text"),
        TaskInstruction::new("a", "text")
    );
    assert_ne!(
        TaskInstruction::new("a", "text"),
        TaskInstruction::new("b", "text")
    );
}

// ── InstructionEmbedding ──────────────────────────────────────────────────────

#[test]
fn test_embedding_dims() {
    let embedding = InstructionEmbedding {
        vector: vec![0.1, 0.2, 0.3, 0.4],
        instruction_name: "x".to_string(),
    };
    assert_eq!(embedding.dims(), 4);
}

#[test]
fn test_embedding_norm_unit() {
    let embedding = InstructionEmbedding {
        vector: vec![0.6, 0.8],
        instruction_name: "x".to_string(),
    };
    assert!((embedding.norm() - 1.0).abs() < 1e-6);
}

#[test]
fn test_embedding_norm_zero_vector() {
    let embedding = InstructionEmbedding {
        vector: vec![0.0, 0.0, 0.0],
        instruction_name: "x".to_string(),
    };
    assert_eq!(embedding.norm(), 0.0);
}

// ── InstructionEmbedder: construction ────────────────────────────────────────

#[test]
fn test_embedder_config_accessor() {
    let config = InstructionEmbedConfig::new().with_dim(16);
    let embedder = InstructionEmbedder::new(config.clone());
    assert_eq!(embedder.config(), &config);
}

// ── InstructionEmbedder: determinism ──────────────────────────────────────────

#[test]
fn test_embed_same_text_same_instruction_is_deterministic() {
    let embedder = InstructionEmbedder::new(InstructionEmbedConfig::default());
    let instruction = TaskInstruction::new("sci", "Represent the science document:");
    let a = embedder
        .embed("rust systems programming", &instruction)
        .unwrap();
    let b = embedder
        .embed("rust systems programming", &instruction)
        .unwrap();
    assert_eq!(a, b);
}

#[test]
fn test_embed_deterministic_across_fresh_embedders() {
    let instruction = TaskInstruction::new("sci", "Represent the science document:");
    let a = InstructionEmbedder::new(InstructionEmbedConfig::default())
        .embed("rust systems programming", &instruction)
        .unwrap();
    let b = InstructionEmbedder::new(InstructionEmbedConfig::default())
        .embed("rust systems programming", &instruction)
        .unwrap();
    assert_eq!(a.vector, b.vector);
}

#[test]
fn test_embed_records_instruction_name() {
    let embedder = InstructionEmbedder::new(InstructionEmbedConfig::default());
    let instruction = TaskInstruction::new("my_instruction", "Represent the text:");
    let embedding = embedder.embed("some text", &instruction).unwrap();
    assert_eq!(embedding.instruction_name, "my_instruction");
}

// ── InstructionEmbedder: instruction conditioning changes the embedding ─────

#[test]
fn test_embed_different_instructions_produce_different_vectors() {
    let embedder = InstructionEmbedder::new(InstructionEmbedConfig::default());
    let instr_a = TaskInstruction::new("a", "Represent the science document for retrieval:");
    let instr_b = TaskInstruction::new(
        "b",
        "Represent the question for retrieving supporting documents:",
    );
    let a = embedder
        .embed("rust systems programming", &instr_a)
        .unwrap();
    let b = embedder
        .embed("rust systems programming", &instr_b)
        .unwrap();
    assert_ne!(a.vector, b.vector);
}

#[test]
fn test_embed_different_instructions_cosine_less_than_one() {
    let embedder = InstructionEmbedder::new(InstructionEmbedConfig::default());
    let instr_a = TaskInstruction::new("a", "Represent the science document for retrieval:");
    let instr_b = TaskInstruction::new(
        "b",
        "Represent the question for retrieving supporting documents:",
    );
    let a = embedder
        .embed("rust systems programming", &instr_a)
        .unwrap();
    let b = embedder
        .embed("rust systems programming", &instr_b)
        .unwrap();
    let cosine = InstructionEmbedder::cosine(&a.vector, &b.vector);
    assert!(cosine < 1.0, "cosine={cosine}");
}

#[test]
fn test_embed_same_instruction_different_text_differ() {
    let embedder = InstructionEmbedder::new(InstructionEmbedConfig::default());
    let instruction = TaskInstruction::new("x", "Represent the document for retrieval:");
    let a = embedder
        .embed("rust systems programming", &instruction)
        .unwrap();
    let b = embedder.embed("python data science", &instruction).unwrap();
    assert_ne!(a.vector, b.vector);
}

#[test]
fn test_embed_instruction_conditioning_is_not_a_trivial_concat() {
    // A trivial "concat the instruction hash onto the text embedding and
    // renormalise" scheme would still rank two *documents* identically
    // regardless of instruction, since a shared additive offset applied
    // equally to a fixed pair of vectors does not by itself change *within
    // group* ordering when compared to a *third*, externally fixed vector in
    // every possible construction. Demonstrate the stronger property this
    // module actually provides: conditioning the SAME text under two
    // instructions moves it by a *different* amount depending on how much
    // the instruction's own vocabulary overlaps that text — i.e. the
    // instruction genuinely re-weights the representation rather than
    // appending a constant.
    let embedder = InstructionEmbedder::new(ranking_config());
    let overlapping = TaskInstruction::new(
        "overlap",
        "quantum entanglement particle wavefunction physics laboratory experiment",
    );
    let disjoint = TaskInstruction::new(
        "disjoint",
        "tomato basil garlic olive pasta recipe kitchen dinner",
    );
    let text = "quantum entanglement particle wavefunction physics laboratory experiment";
    let base = InstructionEmbedder::new(
        InstructionEmbedConfig::new()
            .with_dim(64)
            .with_instruction_influence(0.0),
    )
    .embed(text, &overlapping)
    .unwrap();
    let with_overlap = embedder.embed(text, &overlapping).unwrap();
    let with_disjoint = embedder.embed(text, &disjoint).unwrap();

    let cos_overlap = InstructionEmbedder::cosine(&base.vector, &with_overlap.vector);
    let cos_disjoint = InstructionEmbedder::cosine(&base.vector, &with_disjoint.vector);
    assert!(
        cos_overlap > cos_disjoint,
        "cos_overlap={cos_overlap} cos_disjoint={cos_disjoint}"
    );
}

// ── InstructionEmbedder: influence = 0 boundary ──────────────────────────────

#[test]
fn test_embed_zero_influence_collapses_across_instructions() {
    let config = InstructionEmbedConfig::new()
        .with_dim(32)
        .with_instruction_influence(0.0);
    let embedder = InstructionEmbedder::new(config);
    let instr_a = TaskInstruction::new("a", "Represent the science document for retrieval:");
    let instr_b = TaskInstruction::new(
        "b",
        "Represent the question for retrieving supporting documents:",
    );
    let a = embedder
        .embed("rust systems programming", &instr_a)
        .unwrap();
    let b = embedder
        .embed("rust systems programming", &instr_b)
        .unwrap();
    assert_eq!(a.vector, b.vector);
}

#[test]
fn test_embed_zero_influence_collapses_even_for_wildly_different_instructions() {
    let config = InstructionEmbedConfig::new()
        .with_dim(32)
        .with_instruction_influence(0.0);
    let embedder = InstructionEmbedder::new(config);
    let instr_a = TaskInstruction::from_text("");
    let instr_b = TaskInstruction::from_text(
        "a very long and completely unrelated instruction about cooking recipes",
    );
    let a = embedder
        .embed("rust systems programming", &instr_a)
        .unwrap();
    let b = embedder
        .embed("rust systems programming", &instr_b)
        .unwrap();
    assert_eq!(a.vector, b.vector);
}

#[test]
fn test_embed_zero_influence_still_deterministic() {
    let config = InstructionEmbedConfig::new()
        .with_dim(32)
        .with_instruction_influence(0.0);
    let embedder = InstructionEmbedder::new(config);
    let instruction = TaskInstruction::from_text("Represent the document:");
    let a = embedder.embed("some text here", &instruction).unwrap();
    let b = embedder.embed("some text here", &instruction).unwrap();
    assert_eq!(a.vector, b.vector);
}

// ── InstructionEmbedder: normalization ───────────────────────────────────────

#[test]
fn test_embed_default_is_unit_norm() {
    let embedder = InstructionEmbedder::new(InstructionEmbedConfig::default());
    let instruction = TaskInstruction::new("x", "Represent the document for retrieval:");
    let embedding = embedder
        .embed("rust systems programming language", &instruction)
        .unwrap();
    assert!(
        (norm(&embedding.vector) - 1.0).abs() < 1e-4,
        "norm={}",
        norm(&embedding.vector)
    );
}

#[test]
fn test_embed_unit_norm_under_strong_influence() {
    let embedder = InstructionEmbedder::new(ranking_config());
    let embedding = embedder
        .embed(&ranking_corpus()[0].content, &physics_query_instruction())
        .unwrap();
    assert!(
        (norm(&embedding.vector) - 1.0).abs() < 1e-4,
        "norm={}",
        norm(&embedding.vector)
    );
}

#[test]
fn test_embed_normalize_disabled_generally_not_unit_norm() {
    let config = InstructionEmbedConfig::new()
        .with_dim(32)
        .with_instruction_influence(0.5)
        .with_normalize(false);
    let embedder = InstructionEmbedder::new(config);
    let instruction = TaskInstruction::new("x", "Represent the document for retrieval:");
    let embedding = embedder
        .embed("alpha beta gamma delta epsilon zeta eta", &instruction)
        .unwrap();
    assert!(
        (norm(&embedding.vector) - 1.0).abs() > 0.05,
        "norm={}",
        norm(&embedding.vector)
    );
}

#[test]
fn test_embed_normalize_true_vs_false_same_direction() {
    // Disabling normalisation should not change the vector's *direction*,
    // only its magnitude: cosine between the normalised and unnormalised
    // outputs (for the same inputs) is 1.0.
    let base_config = InstructionEmbedConfig::new()
        .with_dim(32)
        .with_instruction_influence(0.5);
    let normalized = InstructionEmbedder::new(base_config.clone().with_normalize(true));
    let unnormalized = InstructionEmbedder::new(base_config.with_normalize(false));
    let instruction = TaskInstruction::new("x", "Represent the document for retrieval:");
    let a = normalized
        .embed("alpha beta gamma delta", &instruction)
        .unwrap();
    let b = unnormalized
        .embed("alpha beta gamma delta", &instruction)
        .unwrap();
    let cosine = InstructionEmbedder::cosine(&a.vector, &b.vector);
    assert!((cosine - 1.0).abs() < 1e-4, "cosine={cosine}");
}

// ── InstructionEmbedder: error paths ─────────────────────────────────────────

#[test]
fn test_embed_empty_text_errors() {
    let embedder = InstructionEmbedder::new(InstructionEmbedConfig::default());
    let instruction = TaskInstruction::from_text("Represent the document:");
    assert_eq!(
        embedder.embed("", &instruction).err(),
        Some(InstructionEmbedError::EmptyText)
    );
}

#[test]
fn test_embed_whitespace_only_text_errors() {
    let embedder = InstructionEmbedder::new(InstructionEmbedConfig::default());
    let instruction = TaskInstruction::from_text("Represent the document:");
    assert_eq!(
        embedder.embed("   \n\t", &instruction).err(),
        Some(InstructionEmbedError::EmptyText)
    );
}

#[test]
fn test_embed_zero_dim_errors_invalid_dim() {
    let embedder = InstructionEmbedder::new(InstructionEmbedConfig::new().with_dim(0));
    let instruction = TaskInstruction::from_text("Represent the document:");
    assert_eq!(
        embedder.embed("some text", &instruction).err(),
        Some(InstructionEmbedError::InvalidDim(0))
    );
}

#[test]
fn test_embed_blank_instruction_text_does_not_error() {
    // An empty *instruction* is not itself an error (only empty *text* is);
    // it just means the shift vector degenerates to zero.
    let embedder = InstructionEmbedder::new(InstructionEmbedConfig::default());
    let instruction = TaskInstruction::from_text("");
    assert!(embedder.embed("some text", &instruction).is_ok());
}

// ── InstructionEmbedder::cosine ───────────────────────────────────────────────

#[test]
fn test_cosine_identical_is_one() {
    let v = vec![0.6f32, 0.8];
    assert!((InstructionEmbedder::cosine(&v, &v) - 1.0).abs() < 1e-6);
}

#[test]
fn test_cosine_orthogonal_is_zero() {
    let a = vec![1.0f32, 0.0];
    let b = vec![0.0f32, 1.0];
    assert_eq!(InstructionEmbedder::cosine(&a, &b), 0.0);
}

#[test]
fn test_cosine_mismatched_len_is_zero() {
    let a = vec![1.0f32, 0.0];
    let b = vec![1.0f32, 0.0, 0.0];
    assert_eq!(InstructionEmbedder::cosine(&a, &b), 0.0);
}

#[test]
fn test_cosine_empty_is_zero() {
    assert_eq!(InstructionEmbedder::cosine(&[], &[]), 0.0);
}

#[test]
fn test_cosine_clamped_to_unit_range() {
    let v = vec![1.0f32, 1.0, 1.0];
    let cosine = InstructionEmbedder::cosine(&v, &v);
    assert!((-1.0..=1.0).contains(&cosine));
}

#[test]
fn test_cosine_opposite_vectors_is_negative_one() {
    let a = vec![1.0f32, 0.0];
    let b = vec![-1.0f32, 0.0];
    assert!((InstructionEmbedder::cosine(&a, &b) - (-1.0)).abs() < 1e-6);
}

// ── InstructionRegistry ────────────────────────────────────────────────────────

#[test]
fn test_registry_new_is_empty() {
    assert!(InstructionRegistry::new().is_empty());
}

#[test]
fn test_registry_default_is_empty() {
    assert!(InstructionRegistry::default().is_empty());
}

#[test]
fn test_registry_len_zero_initially() {
    assert_eq!(InstructionRegistry::new().len(), 0);
}

#[test]
fn test_registry_register_first_returns_none() {
    let mut registry = InstructionRegistry::new();
    let previous = registry.register(TaskInstruction::new("sci", "Represent the document:"));
    assert!(previous.is_none());
}

#[test]
fn test_registry_register_increments_len() {
    let mut registry = InstructionRegistry::new();
    registry.register(TaskInstruction::new("sci", "Represent the document:"));
    assert_eq!(registry.len(), 1);
}

#[test]
fn test_registry_register_replace_returns_previous() {
    let mut registry = InstructionRegistry::new();
    registry.register(TaskInstruction::new("sci", "first text"));
    let previous = registry.register(TaskInstruction::new("sci", "second text"));
    assert_eq!(previous, Some(TaskInstruction::new("sci", "first text")));
}

#[test]
fn test_registry_register_replace_keeps_len_one() {
    let mut registry = InstructionRegistry::new();
    registry.register(TaskInstruction::new("sci", "first text"));
    registry.register(TaskInstruction::new("sci", "second text"));
    assert_eq!(registry.len(), 1);
}

#[test]
fn test_registry_get_hit() {
    let mut registry = InstructionRegistry::new();
    registry.register(TaskInstruction::new("sci", "Represent the document:"));
    assert_eq!(
        registry.get("sci").map(|i| i.text.as_str()),
        Some("Represent the document:")
    );
}

#[test]
fn test_registry_get_miss_is_none() {
    let registry = InstructionRegistry::new();
    assert!(registry.get("missing").is_none());
}

#[test]
fn test_registry_try_get_hit() {
    let mut registry = InstructionRegistry::new();
    registry.register(TaskInstruction::new("sci", "Represent the document:"));
    assert!(registry.try_get("sci").is_ok());
}

#[test]
fn test_registry_try_get_unknown_errors() {
    let registry = InstructionRegistry::new();
    assert_eq!(
        registry.try_get("missing").err(),
        Some(InstructionEmbedError::UnknownInstruction(
            "missing".to_string()
        ))
    );
}

#[test]
fn test_registry_contains_true_after_register() {
    let mut registry = InstructionRegistry::new();
    registry.register(TaskInstruction::new("sci", "text"));
    assert!(registry.contains("sci"));
}

#[test]
fn test_registry_contains_false_before_register() {
    let registry = InstructionRegistry::new();
    assert!(!registry.contains("sci"));
}

#[test]
fn test_registry_remove_existing_returns_some() {
    let mut registry = InstructionRegistry::new();
    registry.register(TaskInstruction::new("sci", "text"));
    assert!(registry.remove("sci").is_some());
}

#[test]
fn test_registry_remove_missing_returns_none() {
    let mut registry = InstructionRegistry::new();
    assert!(registry.remove("sci").is_none());
}

#[test]
fn test_registry_remove_decrements_len() {
    let mut registry = InstructionRegistry::new();
    registry.register(TaskInstruction::new("sci", "text"));
    registry.remove("sci");
    assert_eq!(registry.len(), 0);
}

#[test]
fn test_registry_names_sorted_ascending() {
    let mut registry = InstructionRegistry::new();
    registry.register(TaskInstruction::new("zeta", "z"));
    registry.register(TaskInstruction::new("alpha", "a"));
    registry.register(TaskInstruction::new("mu", "m"));
    assert_eq!(registry.names(), vec!["alpha", "mu", "zeta"]);
}

#[test]
fn test_registry_is_empty_false_after_register() {
    let mut registry = InstructionRegistry::new();
    registry.register(TaskInstruction::new("sci", "text"));
    assert!(!registry.is_empty());
}

// ── InstructionIndex: construction & bookkeeping ─────────────────────────────

#[test]
fn test_index_new_is_empty() {
    let index = InstructionIndex::new(InstructionEmbedConfig::default());
    assert!(index.is_empty());
}

#[test]
fn test_index_new_len_zero() {
    let index = InstructionIndex::new(InstructionEmbedConfig::default());
    assert_eq!(index.len(), 0);
}

#[test]
fn test_index_new_is_not_indexed() {
    let index = InstructionIndex::new(InstructionEmbedConfig::default());
    assert!(!index.is_indexed());
}

#[test]
fn test_index_new_doc_instruction_name_none() {
    let index = InstructionIndex::new(InstructionEmbedConfig::default());
    assert!(index.doc_instruction_name().is_none());
}

#[test]
fn test_index_config_accessor() {
    let config = InstructionEmbedConfig::new().with_dim(16);
    let index = InstructionIndex::new(config.clone());
    assert_eq!(index.config(), &config);
}

// ── InstructionIndex::index ──────────────────────────────────────────────────

#[test]
fn test_index_populates_entries() {
    let mut index = InstructionIndex::new(InstructionEmbedConfig::default());
    let instruction = TaskInstruction::from_text("Represent the document for retrieval:");
    index.index(&ranking_corpus(), &instruction).unwrap();
    assert_eq!(index.len(), 2);
}

#[test]
fn test_index_sets_indexed_true() {
    let mut index = InstructionIndex::new(InstructionEmbedConfig::default());
    let instruction = TaskInstruction::from_text("Represent the document for retrieval:");
    index.index(&ranking_corpus(), &instruction).unwrap();
    assert!(index.is_indexed());
}

#[test]
fn test_index_records_doc_instruction_name() {
    let mut index = InstructionIndex::new(InstructionEmbedConfig::default());
    let instruction = TaskInstruction::new("doc_task", "Represent the document for retrieval:");
    index.index(&ranking_corpus(), &instruction).unwrap();
    assert_eq!(index.doc_instruction_name(), Some("doc_task"));
}

#[test]
fn test_index_empty_corpus_errors() {
    let mut index = InstructionIndex::new(InstructionEmbedConfig::default());
    let instruction = TaskInstruction::from_text("Represent the document for retrieval:");
    assert_eq!(
        index.index(&[], &instruction).err(),
        Some(InstructionEmbedError::EmptyCorpus)
    );
}

#[test]
fn test_reindex_replaces_not_accumulates() {
    let mut index = InstructionIndex::new(InstructionEmbedConfig::default());
    let instruction = TaskInstruction::from_text("Represent the document for retrieval:");
    index.index(&ranking_corpus(), &instruction).unwrap();
    index.index(&ranking_corpus(), &instruction).unwrap();
    assert_eq!(index.len(), 2);
}

#[test]
fn test_index_zero_dim_config_errors_invalid_dim() {
    let mut index = InstructionIndex::new(InstructionEmbedConfig::new().with_dim(0));
    let instruction = TaskInstruction::from_text("Represent the document for retrieval:");
    assert_eq!(
        index.index(&ranking_corpus(), &instruction).err(),
        Some(InstructionEmbedError::InvalidDim(0))
    );
}

#[test]
fn test_index_preserves_document_ids() {
    let mut index = InstructionIndex::new(InstructionEmbedConfig::default());
    let instruction = TaskInstruction::from_text("Represent the document for retrieval:");
    index.index(&ranking_corpus(), &instruction).unwrap();
    let hits = index.search("physics laboratory", &instruction, 2).unwrap();
    let ids: Vec<String> = hits.iter().map(|(id, _)| id.as_str().to_string()).collect();
    assert!(ids.contains(&"physics".to_string()));
    assert!(ids.contains(&"cooking".to_string()));
}

// ── InstructionIndex::search ─────────────────────────────────────────────────

#[test]
fn test_search_not_indexed_errors() {
    let index = InstructionIndex::new(InstructionEmbedConfig::default());
    let instruction = TaskInstruction::from_text("Represent the query:");
    assert_eq!(
        index.search("some query", &instruction, 3).err(),
        Some(InstructionEmbedError::NotIndexed)
    );
}

#[test]
fn test_search_empty_query_errors() {
    let mut index = InstructionIndex::new(InstructionEmbedConfig::default());
    let instruction = TaskInstruction::from_text("Represent the document:");
    index.index(&ranking_corpus(), &instruction).unwrap();
    assert_eq!(
        index.search("   ", &instruction, 3).err(),
        Some(InstructionEmbedError::EmptyQuery)
    );
}

#[test]
fn test_search_after_clear_errors_not_indexed() {
    let mut index = InstructionIndex::new(InstructionEmbedConfig::default());
    let instruction = TaskInstruction::from_text("Represent the document:");
    index.index(&ranking_corpus(), &instruction).unwrap();
    index.clear();
    assert_eq!(
        index.search("physics", &instruction, 3).err(),
        Some(InstructionEmbedError::NotIndexed)
    );
}

#[test]
fn test_search_returns_at_most_k() {
    let mut index = InstructionIndex::new(InstructionEmbedConfig::default());
    let instruction = TaskInstruction::from_text("Represent the document:");
    index.index(&ranking_corpus(), &instruction).unwrap();
    let hits = index.search("physics laboratory", &instruction, 1).unwrap();
    assert!(hits.len() <= 1);
}

#[test]
fn test_search_k_zero_is_empty() {
    let mut index = InstructionIndex::new(InstructionEmbedConfig::default());
    let instruction = TaskInstruction::from_text("Represent the document:");
    index.index(&ranking_corpus(), &instruction).unwrap();
    assert!(
        index
            .search("physics laboratory", &instruction, 0)
            .unwrap()
            .is_empty()
    );
}

#[test]
fn test_search_descending_scores() {
    let mut index = InstructionIndex::new(InstructionEmbedConfig::default());
    let instruction = TaskInstruction::from_text("Represent the document:");
    index.index(&ranking_corpus(), &instruction).unwrap();
    let hits = index
        .search("physics laboratory experiment", &instruction, 2)
        .unwrap();
    assert!(hits.windows(2).all(|w| w[0].1 >= w[1].1));
}

#[test]
fn test_search_score_in_unit_range() {
    let mut index = InstructionIndex::new(InstructionEmbedConfig::default());
    let instruction = TaskInstruction::from_text("Represent the document:");
    index.index(&ranking_corpus(), &instruction).unwrap();
    let hits = index
        .search("physics laboratory experiment", &instruction, 2)
        .unwrap();
    assert!(hits.iter().all(|(_, score)| (-1.0..=1.0).contains(score)));
}

#[test]
fn test_search_default_uses_config_top_k() {
    let config = InstructionEmbedConfig::new().with_default_top_k(1);
    let mut index = InstructionIndex::new(config);
    let instruction = TaskInstruction::from_text("Represent the document:");
    index.index(&ranking_corpus(), &instruction).unwrap();
    let hits = index
        .search_default("physics laboratory experiment", &instruction)
        .unwrap();
    assert_eq!(hits.len(), 1);
}

#[test]
fn test_search_ties_broken_by_ascending_id() {
    let mut index = InstructionIndex::new(InstructionEmbedConfig::default());
    let instruction = TaskInstruction::from_text("Represent the document:");
    // Identical content under identical ids other than the id itself, so the
    // two documents score an exact tie.
    let docs = vec![
        doc("zzz", "identical shared content"),
        doc("aaa", "identical shared content"),
    ];
    index.index(&docs, &instruction).unwrap();
    let hits = index
        .search("identical shared content", &instruction, 2)
        .unwrap();
    assert_eq!(hits[0].0.as_str(), "aaa");
    assert_eq!(hits[1].0.as_str(), "zzz");
}

#[test]
fn test_search_finds_relevant_doc_via_shared_vocabulary() {
    let mut index = InstructionIndex::new(InstructionEmbedConfig::default());
    let instruction = TaskInstruction::from_text("Represent the document:");
    index.index(&ranking_corpus(), &instruction).unwrap();
    let hits = index
        .search(
            "quantum entanglement particle wavefunction physics laboratory experiment",
            &instruction,
            2,
        )
        .unwrap();
    assert_eq!(hits[0].0.as_str(), "physics");
}

// ── Crafted-corpus ranking flip (the defining property) ──────────────────────

#[test]
fn test_query_instruction_changes_top_ranked_document() {
    let mut index = InstructionIndex::new(ranking_config());
    let doc_instruction = TaskInstruction::from_text("Represent the document for retrieval:");
    index.index(&ranking_corpus(), &doc_instruction).unwrap();

    // A deliberately topic-neutral query: it shares no vocabulary with
    // either document, so the base (instruction-agnostic) query embedding
    // alone has no strong reason to prefer one over the other. The task
    // instruction alone supplies the topical pull.
    let query = "tell me more about this";

    let physics_hits = index
        .search(query, &physics_query_instruction(), 1)
        .unwrap();
    let cooking_hits = index
        .search(query, &cooking_query_instruction(), 1)
        .unwrap();

    assert_eq!(physics_hits[0].0.as_str(), "physics");
    assert_eq!(cooking_hits[0].0.as_str(), "cooking");
    assert_ne!(physics_hits[0].0, cooking_hits[0].0);
}

#[test]
fn test_query_instruction_changes_ranking_with_clear_margin() {
    // Same as `test_query_instruction_changes_top_ranked_document`, but also
    // asserts the winning margin is substantial (not a razor-thin, possibly
    // coincidental edge) — the instruction should *decisively* favour the
    // topically matching document.
    let mut index = InstructionIndex::new(ranking_config());
    let doc_instruction = TaskInstruction::from_text("Represent the document for retrieval:");
    index.index(&ranking_corpus(), &doc_instruction).unwrap();
    let query = "tell me more about this";

    let physics_hits = index
        .search(query, &physics_query_instruction(), 2)
        .unwrap();
    let cooking_hits = index
        .search(query, &cooking_query_instruction(), 2)
        .unwrap();

    let physics_top_score = physics_hits[0].1;
    let physics_second_score = physics_hits[1].1;
    let cooking_top_score = cooking_hits[0].1;
    let cooking_second_score = cooking_hits[1].1;

    assert!(physics_top_score - physics_second_score > 0.1);
    assert!(cooking_top_score - cooking_second_score > 0.1);
}

#[test]
fn test_reindexing_under_different_instruction_pair_changes_ranking() {
    // Re-index the *same* corpus twice, once per task instruction pair, and
    // query with the *same* neutral query text each time. The document that
    // ranks first flips with the task.
    let mut index = InstructionIndex::new(ranking_config());
    let neutral_query = "tell me more about this";

    let physics_doc_instruction = TaskInstruction::new(
        "physics_doc",
        "Represent the physics document for retrieval:",
    );
    let cooking_doc_instruction = TaskInstruction::new(
        "cooking_doc",
        "Represent the cooking document for retrieval:",
    );

    index
        .index(&ranking_corpus(), &physics_doc_instruction)
        .unwrap();
    let physics_task_hits = index
        .search(neutral_query, &physics_query_instruction(), 1)
        .unwrap();

    index
        .index(&ranking_corpus(), &cooking_doc_instruction)
        .unwrap();
    let cooking_task_hits = index
        .search(neutral_query, &cooking_query_instruction(), 1)
        .unwrap();

    assert_eq!(physics_task_hits[0].0.as_str(), "physics");
    assert_eq!(cooking_task_hits[0].0.as_str(), "cooking");
}

#[test]
fn test_reindexing_same_instruction_preserves_ranking() {
    // Contrast case: re-indexing under the *same* instruction (not a
    // different one) leaves the ranking unchanged, confirming the flips
    // above are caused by the instruction change and not by the act of
    // re-indexing itself.
    let mut index = InstructionIndex::new(ranking_config());
    let doc_instruction = TaskInstruction::from_text("Represent the document for retrieval:");
    let query = "tell me more about this";

    index.index(&ranking_corpus(), &doc_instruction).unwrap();
    let first = index
        .search(query, &physics_query_instruction(), 2)
        .unwrap();

    index.index(&ranking_corpus(), &doc_instruction).unwrap();
    let second = index
        .search(query, &physics_query_instruction(), 2)
        .unwrap();

    assert_eq!(first, second);
}

// ── InstructionIndex::clear ──────────────────────────────────────────────────

#[test]
fn test_clear_resets_indexed_flag() {
    let mut index = InstructionIndex::new(InstructionEmbedConfig::default());
    let instruction = TaskInstruction::from_text("Represent the document:");
    index.index(&ranking_corpus(), &instruction).unwrap();
    index.clear();
    assert!(!index.is_indexed());
}

#[test]
fn test_clear_empties_entries() {
    let mut index = InstructionIndex::new(InstructionEmbedConfig::default());
    let instruction = TaskInstruction::from_text("Represent the document:");
    index.index(&ranking_corpus(), &instruction).unwrap();
    index.clear();
    assert_eq!(index.len(), 0);
    assert!(index.is_empty());
}

#[test]
fn test_clear_resets_doc_instruction_name() {
    let mut index = InstructionIndex::new(InstructionEmbedConfig::default());
    let instruction = TaskInstruction::from_text("Represent the document:");
    index.index(&ranking_corpus(), &instruction).unwrap();
    index.clear();
    assert!(index.doc_instruction_name().is_none());
}

// ── Error Display messages ────────────────────────────────────────────────────

#[test]
fn test_error_display_empty_text() {
    assert_eq!(
        InstructionEmbedError::EmptyText.to_string(),
        "text must not be empty"
    );
}

#[test]
fn test_error_display_empty_query() {
    assert_eq!(
        InstructionEmbedError::EmptyQuery.to_string(),
        "query must not be empty"
    );
}

#[test]
fn test_error_display_empty_corpus() {
    assert_eq!(
        InstructionEmbedError::EmptyCorpus.to_string(),
        "corpus is empty"
    );
}

#[test]
fn test_error_display_not_indexed() {
    assert_eq!(
        InstructionEmbedError::NotIndexed.to_string(),
        "index has not been built; call `index` first"
    );
}

#[test]
fn test_error_display_invalid_dim() {
    assert_eq!(
        InstructionEmbedError::InvalidDim(42).to_string(),
        "invalid embedding dimension: 42"
    );
}

#[test]
fn test_error_display_unknown_instruction() {
    assert_eq!(
        InstructionEmbedError::UnknownInstruction("ghost".to_string()).to_string(),
        "unknown instruction: ghost"
    );
}
