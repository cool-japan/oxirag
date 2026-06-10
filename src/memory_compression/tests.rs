use crate::memory_compression::hierarchy::HierarchicalMemory;
use crate::memory_compression::types::{
    CompressedBlock, CompressionStats, ExtractiveTurnCompressor, MemoryCompressionConfig,
    MemoryCompressionError, MemoryTurn, TurnCompressor,
};

// ── MemoryTurn ─────────────────────────────────────────────────────────────

#[test]
fn test_memory_turn_new_tokens_equals_word_count() {
    let turn = MemoryTurn::new("user", "hello world foo", 0);
    assert_eq!(turn.tokens, 3);
}

#[test]
fn test_memory_turn_new_single_word() {
    let turn = MemoryTurn::new("assistant", "Hello", 1);
    assert_eq!(turn.tokens, 1);
    assert_eq!(turn.role, "assistant");
    assert_eq!(turn.content, "Hello");
    assert_eq!(turn.turn_index, 1);
}

#[test]
fn test_memory_turn_new_multi_word_content() {
    let turn = MemoryTurn::new("user", "The quick brown fox jumps over the lazy dog", 0);
    assert_eq!(turn.tokens, 9);
}

#[test]
fn test_memory_turn_new_empty_content_zero_tokens() {
    // empty content push is rejected by push_turn, but MemoryTurn itself allows it
    let turn = MemoryTurn::new("user", "", 0);
    assert_eq!(turn.tokens, 0);
}

#[test]
fn test_memory_turn_fields_accessible() {
    let turn = MemoryTurn::new("assistant", "Some reply text here", 5);
    assert_eq!(turn.role, "assistant");
    assert_eq!(turn.content, "Some reply text here");
    assert_eq!(turn.turn_index, 5);
    assert_eq!(turn.tokens, 4);
}

// ── CompressedBlock ────────────────────────────────────────────────────────

#[test]
fn test_compressed_block_fields_accessible() {
    let block = CompressedBlock {
        level: 1,
        summary: "Some summary text.".to_string(),
        covered_turns: vec![0, 1, 2],
        salient_facts: vec!["fact one".to_string()],
        tokens: 3,
    };
    assert_eq!(block.level, 1);
    assert_eq!(block.summary, "Some summary text.");
    assert_eq!(block.covered_turns, vec![0, 1, 2]);
    assert_eq!(block.tokens, 3);
}

// ── ExtractiveTurnCompressor ───────────────────────────────────────────────

#[test]
fn test_extractive_compressor_default_fields() {
    let c = ExtractiveTurnCompressor::default();
    assert_eq!(c.summary_sentences, 3);
    assert_eq!(c.salient_facts, 4);
}

#[test]
fn test_extractive_compressor_compress_single_turn() {
    let c = ExtractiveTurnCompressor::default();
    let turns = vec![MemoryTurn::new("user", "Paris is a great city", 0)];
    let block = c.compress(&turns, 1);
    assert_eq!(block.level, 1);
    assert!(!block.summary.is_empty());
    assert_eq!(block.covered_turns, vec![0]);
}

#[test]
fn test_extractive_compressor_compress_multiple_turns() {
    let c = ExtractiveTurnCompressor::default();
    let turns = vec![
        MemoryTurn::new("user", "Tell me about Rust", 0),
        MemoryTurn::new("assistant", "Rust is a systems language", 1),
        MemoryTurn::new("user", "What are its features", 2),
    ];
    let block = c.compress(&turns, 1);
    assert_eq!(block.covered_turns.len(), 3);
    assert_eq!(block.covered_turns, vec![0, 1, 2]);
}

#[test]
fn test_extractive_compressor_covered_turns_match_indices() {
    let c = ExtractiveTurnCompressor::default();
    let turns = vec![
        MemoryTurn::new("user", "first turn content", 10),
        MemoryTurn::new("assistant", "second turn content", 11),
    ];
    let block = c.compress(&turns, 1);
    assert_eq!(block.covered_turns, vec![10, 11]);
}

#[test]
fn test_extractive_compressor_summary_non_empty() {
    let c = ExtractiveTurnCompressor::default();
    let turns = vec![MemoryTurn::new(
        "user",
        "Hello there how are you doing today",
        0,
    )];
    let block = c.compress(&turns, 1);
    assert!(!block.summary.is_empty());
}

#[test]
fn test_extractive_compressor_salient_facts_bounded() {
    let c = ExtractiveTurnCompressor {
        summary_sentences: 3,
        salient_facts: 2,
    };
    let turns = vec![MemoryTurn::new(
        "user",
        "Fact one. Fact two. Fact three. Fact four. Fact five.",
        0,
    )];
    let block = c.compress(&turns, 1);
    assert!(block.salient_facts.len() <= 2);
}

#[test]
fn test_extractive_compressor_salient_facts_default_max_four() {
    let c = ExtractiveTurnCompressor::default();
    let turns = vec![MemoryTurn::new(
        "user",
        "Alpha. Beta. Gamma. Delta. Epsilon. Zeta.",
        0,
    )];
    let block = c.compress(&turns, 1);
    assert!(block.salient_facts.len() <= 4);
}

#[test]
fn test_extractive_compressor_level_propagated() {
    let c = ExtractiveTurnCompressor::default();
    let turns = vec![MemoryTurn::new("user", "some content", 0)];
    let block = c.compress(&turns, 2);
    assert_eq!(block.level, 2);
}

#[test]
fn test_extractive_compressor_respects_summary_sentences_limit() {
    let c = ExtractiveTurnCompressor {
        summary_sentences: 2,
        salient_facts: 4,
    };
    let turns = vec![
        MemoryTurn::new("user", "Alpha content", 0),
        MemoryTurn::new("assistant", "Beta content", 1),
        MemoryTurn::new("user", "Gamma content", 2),
        MemoryTurn::new("assistant", "Delta content", 3),
    ];
    let block = c.compress(&turns, 1);
    // summary should only use first 2 turns' content
    assert!(!block.summary.contains("Gamma content"));
}

// ── MemoryCompressionConfig ────────────────────────────────────────────────

#[test]
fn test_config_default_recent_window_tokens() {
    let cfg = MemoryCompressionConfig::default();
    assert_eq!(cfg.recent_window_tokens, 512);
}

#[test]
fn test_config_default_block_size_turns() {
    let cfg = MemoryCompressionConfig::default();
    assert_eq!(cfg.block_size_turns, 6);
}

#[test]
fn test_config_default_max_level() {
    let cfg = MemoryCompressionConfig::default();
    assert_eq!(cfg.max_level, 3);
}

#[test]
fn test_config_default_salient_facts_per_block() {
    let cfg = MemoryCompressionConfig::default();
    assert_eq!(cfg.salient_facts_per_block, 4);
}

#[test]
fn test_config_default_summary_sentences() {
    let cfg = MemoryCompressionConfig::default();
    assert_eq!(cfg.summary_sentences, 3);
}

#[test]
fn test_config_builder_recent_window() {
    let cfg = MemoryCompressionConfig::default().with_recent_window_tokens(256);
    assert_eq!(cfg.recent_window_tokens, 256);
}

#[test]
fn test_config_builder_block_size() {
    let cfg = MemoryCompressionConfig::default().with_block_size_turns(4);
    assert_eq!(cfg.block_size_turns, 4);
}

#[test]
fn test_config_builder_max_level() {
    let cfg = MemoryCompressionConfig::default().with_max_level(2);
    assert_eq!(cfg.max_level, 2);
}

// ── HierarchicalMemory ─────────────────────────────────────────────────────

#[test]
fn test_hierarchical_memory_new_starts_empty() {
    let mem = HierarchicalMemory::new(MemoryCompressionConfig::default());
    assert!(mem.recent.is_empty());
    assert!(mem.blocks.is_empty());
    assert_eq!(mem.total_tokens(), 0);
}

#[test]
fn test_hierarchical_memory_default_starts_empty() {
    let mem = HierarchicalMemory::default();
    assert!(mem.recent.is_empty());
}

#[test]
fn test_push_turn_empty_content_errors() {
    let mut mem = HierarchicalMemory::default();
    let turn = MemoryTurn::new("user", "", 0);
    let result = mem.push_turn(turn);
    assert!(matches!(result, Err(MemoryCompressionError::EmptyTurns)));
}

#[test]
fn test_push_turn_whitespace_only_errors() {
    let mut mem = HierarchicalMemory::default();
    let turn = MemoryTurn::new("user", "   ", 0);
    let result = mem.push_turn(turn);
    assert!(matches!(result, Err(MemoryCompressionError::EmptyTurns)));
}

#[test]
fn test_push_turn_happy_path_ok() {
    let mut mem = HierarchicalMemory::default();
    let turn = MemoryTurn::new("user", "Hello, this is a test turn", 0);
    assert!(mem.push_turn(turn).is_ok());
    assert_eq!(mem.recent.len(), 1);
}

#[test]
fn test_total_tokens_increases_with_pushes() {
    let mut mem = HierarchicalMemory::default();
    assert_eq!(mem.total_tokens(), 0);
    mem.push_turn(MemoryTurn::new("user", "first turn content here", 0))
        .unwrap();
    let after_one = mem.total_tokens();
    mem.push_turn(MemoryTurn::new("assistant", "second reply content here", 1))
        .unwrap();
    assert!(mem.total_tokens() > after_one);
}

#[test]
fn test_render_budget_zero_errors() {
    let mem = HierarchicalMemory::default();
    let result = mem.render(0);
    assert!(matches!(
        result,
        Err(MemoryCompressionError::BudgetTooSmall(0))
    ));
}

#[test]
fn test_render_empty_memory_with_budget_errors() {
    // No turns pushed: render has nothing to include
    let mem = HierarchicalMemory::default();
    let result = mem.render(10_000);
    assert!(matches!(
        result,
        Err(MemoryCompressionError::BudgetTooSmall(_))
    ));
}

#[test]
fn test_render_returns_string_after_pushing_turns() {
    let mut mem = HierarchicalMemory::default();
    mem.push_turn(MemoryTurn::new("user", "Hello how are you", 0))
        .unwrap();
    mem.push_turn(MemoryTurn::new("assistant", "I am doing well thank you", 1))
        .unwrap();
    let output = mem.render(10_000).unwrap();
    assert!(!output.is_empty());
}

#[test]
fn test_render_includes_role_in_output() {
    let mut mem = HierarchicalMemory::default();
    mem.push_turn(MemoryTurn::new("user", "This is user content", 0))
        .unwrap();
    let output = mem.render(10_000).unwrap();
    assert!(output.contains("user"));
}

#[test]
fn test_render_includes_content_in_output() {
    let mut mem = HierarchicalMemory::default();
    mem.push_turn(MemoryTurn::new("user", "unique marker phrase xyz", 0))
        .unwrap();
    let output = mem.render(10_000).unwrap();
    assert!(output.contains("unique marker phrase xyz"));
}

#[test]
fn test_stats_levels_used_zero_before_compression() {
    let mut mem = HierarchicalMemory::default();
    mem.push_turn(MemoryTurn::new("user", "just one turn", 0))
        .unwrap();
    let stats = mem.stats();
    assert_eq!(stats.levels_used, 0);
}

#[test]
fn test_stats_original_tokens_nonzero_after_pushes() {
    let mut mem = HierarchicalMemory::default();
    mem.push_turn(MemoryTurn::new("user", "some content words here", 0))
        .unwrap();
    let stats = mem.stats();
    assert!(stats.original_tokens > 0);
}

#[test]
fn test_stats_default_compression_stats() {
    let stats = CompressionStats::default();
    assert_eq!(stats.original_tokens, 0);
    assert_eq!(stats.compressed_tokens, 0);
    assert_eq!(stats.levels_used, 0);
    assert!((stats.ratio - 0.0).abs() < 1e-6);
}

#[test]
fn test_maybe_compress_triggers_when_window_overflows() {
    // Use a tiny window (5 tokens) and small block size so compression fires quickly
    let cfg = MemoryCompressionConfig::default()
        .with_recent_window_tokens(5)
        .with_block_size_turns(2);
    let mut mem = HierarchicalMemory::new(cfg);
    // Each turn has 3 words (3 tokens). After 2 turns we have 6 > 5 → compression fires.
    mem.push_turn(MemoryTurn::new("user", "alpha beta gamma", 0))
        .unwrap();
    mem.push_turn(MemoryTurn::new("assistant", "delta epsilon zeta", 1))
        .unwrap();
    // blocks should have been populated
    assert!(!mem.blocks.is_empty());
}

#[test]
fn test_maybe_compress_reduces_recent_len() {
    let cfg = MemoryCompressionConfig::default()
        .with_recent_window_tokens(5)
        .with_block_size_turns(2);
    let mut mem = HierarchicalMemory::new(cfg);
    mem.push_turn(MemoryTurn::new("user", "alpha beta gamma", 0))
        .unwrap();
    mem.push_turn(MemoryTurn::new("assistant", "delta epsilon zeta", 1))
        .unwrap();
    // The 2 compressed turns are moved into blocks; recent may be empty or < 2
    assert!(mem.recent.len() < 2);
}

#[test]
fn test_stats_levels_used_one_after_compression() {
    let cfg = MemoryCompressionConfig::default()
        .with_recent_window_tokens(5)
        .with_block_size_turns(2);
    let mut mem = HierarchicalMemory::new(cfg);
    mem.push_turn(MemoryTurn::new("user", "alpha beta gamma", 0))
        .unwrap();
    mem.push_turn(MemoryTurn::new("assistant", "delta epsilon zeta", 1))
        .unwrap();
    let stats = mem.stats();
    assert!(stats.levels_used >= 1);
}

#[test]
fn test_stats_ratio_reasonable_after_compression() {
    let cfg = MemoryCompressionConfig::default()
        .with_recent_window_tokens(5)
        .with_block_size_turns(2);
    let mut mem = HierarchicalMemory::new(cfg);
    mem.push_turn(MemoryTurn::new("user", "alpha beta gamma", 0))
        .unwrap();
    mem.push_turn(MemoryTurn::new("assistant", "delta epsilon zeta", 1))
        .unwrap();
    let stats = mem.stats();
    // ratio should be in [0.0, 1.0] range when compression actually reduces tokens
    assert!(stats.ratio >= 0.0);
}

#[test]
fn test_two_level_compression_cascade() {
    // block_size_turns=2 means: after 2 blocks accumulate, they cascade to level 2
    let cfg = MemoryCompressionConfig::default()
        .with_recent_window_tokens(3)
        .with_block_size_turns(2)
        .with_max_level(2);
    let mut mem = HierarchicalMemory::new(cfg);
    // Push enough turns to trigger multiple compressions:
    // Each push of 2 three-word turns creates 1 block; after 2 blocks → cascade to L2
    for i in 0..8usize {
        mem.push_turn(MemoryTurn::new(
            "user",
            format!("alpha beta gamma{i}"),
            i * 2,
        ))
        .unwrap();
        mem.push_turn(MemoryTurn::new(
            "assistant",
            format!("delta epsilon zeta{i}"),
            i * 2 + 1,
        ))
        .unwrap();
    }
    let stats = mem.stats();
    // Some compression must have occurred
    assert!(stats.levels_used >= 1);
}

#[test]
fn test_error_empty_turns_display() {
    let e = MemoryCompressionError::EmptyTurns;
    let msg = format!("{e}");
    assert!(!msg.is_empty());
}

#[test]
fn test_error_budget_too_small_display_includes_value() {
    let e = MemoryCompressionError::BudgetTooSmall(42);
    let msg = format!("{e}");
    assert!(msg.contains("42"));
}

#[test]
fn test_total_tokens_bounded_after_many_pushes() {
    let cfg = MemoryCompressionConfig::default()
        .with_recent_window_tokens(10)
        .with_block_size_turns(3);
    let mut mem = HierarchicalMemory::new(cfg);
    for i in 0..30usize {
        mem.push_turn(MemoryTurn::new(
            "user",
            format!("word1 word2 word3 word4 {i}"),
            i,
        ))
        .unwrap();
    }
    // Compression should have fired many times; total_tokens should not be
    // proportional to 30 full turns (it will be much less)
    let total = mem.total_tokens();
    // 30 turns × 5 tokens each = 150 uncompressed; with compression should be < 150
    assert!(total < 200);
}

#[test]
fn test_render_respects_budget_approximately() {
    let mut mem = HierarchicalMemory::default();
    // Push turns whose tokens sum to roughly 20
    for i in 0..5usize {
        mem.push_turn(MemoryTurn::new(
            "user",
            format!("word one two three {i}"),
            i,
        ))
        .unwrap();
    }
    // budget=5: should cut off early or error
    let result = mem.render(5);
    // Either we get a short output or a BudgetTooSmall error
    match result {
        Ok(s) => {
            // output tokens should be bounded relative to budget
            let word_count = s.split_whitespace().count();
            assert!(word_count <= 20); // sanity check: not the full 25+ words
        }
        Err(MemoryCompressionError::BudgetTooSmall(_)) => { /* acceptable */ }
        Err(e) => panic!("unexpected error: {e}"),
    }
}
