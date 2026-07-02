#![allow(
    clippy::float_cmp,
    clippy::similar_names,
    clippy::too_many_lines,
    clippy::cast_precision_loss,
    clippy::cast_possible_truncation,
    clippy::cast_sign_loss
)]

//! Tests for the `llmlingua` module: surrogate-LM correctness, surprisal
//! ranking, budget allocation, achieved-vs-requested ratios, and edge cases.

use super::budget::BudgetController;
use super::compressor::PerplexityCompressor;
use super::ngram_model::{PerplexityModel, normalize_token, split_sentences, surface_tokens};
use super::types::{CompressionTarget, LlmLinguaConfig, LlmLinguaError};

// ── helpers ─────────────────────────────────────────────────────────────────────

/// Whitespace-tokenise a lowercase string into model tokens.
fn toks(text: &str) -> Vec<String> {
    text.split_whitespace().map(str::to_string).collect()
}

/// Sum `P(w | context)` over the full vocabulary plus one out-of-vocabulary
/// token (the reserved `<unk>` mass). A correctly normalised model returns ~1.
fn probability_mass(model: &PerplexityModel, context: &[&str]) -> f64 {
    let mut mass = 0.0;
    for w in model.vocabulary() {
        mass += model.probability(context, w);
    }
    mass += model.probability(context, "zzz_definitely_out_of_vocab");
    mass
}

// ── text helpers ─────────────────────────────────────────────────────────────────

#[test]
fn split_sentences_basic() {
    let segments = split_sentences("Hello world. How are you? I am fine!");
    assert_eq!(segments.len(), 3);
    assert_eq!(segments[0], "Hello world.");
    assert_eq!(segments[1], "How are you?");
    assert_eq!(segments[2], "I am fine!");
}

#[test]
fn split_sentences_newline_and_no_boundary() {
    let segments = split_sentences("first line\nsecond line");
    assert_eq!(segments, vec!["first line", "second line"]);
    let single = split_sentences("no terminal punctuation here");
    assert_eq!(single, vec!["no terminal punctuation here"]);
}

#[test]
fn surface_tokens_preserve_punctuation() {
    let tokens = surface_tokens("The 2024 report, revised.");
    assert_eq!(tokens, vec!["The", "2024", "report,", "revised."]);
}

#[test]
fn normalize_token_strips_and_lowercases() {
    assert_eq!(normalize_token("France."), "france");
    assert_eq!(normalize_token("(hello)"), "hello");
    assert_eq!(normalize_token("2024"), "2024");
    assert_eq!(normalize_token("---"), "<punct>");
}

// ── n-gram model correctness ─────────────────────────────────────────────────────

#[test]
fn probabilities_sum_to_one_over_vocabulary() {
    let tokens = toks("the cat sat on the mat the cat ran fast");
    let model = PerplexityModel::fit(&tokens, 3, 1.0, 0.7);

    // Unigram (empty context).
    assert!((probability_mass(&model, &[]) - 1.0).abs() < 1e-6);
    // Seen bigram context.
    assert!((probability_mass(&model, &["the"]) - 1.0).abs() < 1e-6);
    // Seen trigram context.
    assert!((probability_mass(&model, &["the", "cat"]) - 1.0).abs() < 1e-6);
    // Unseen higher-order context (must back off yet stay normalised).
    assert!((probability_mass(&model, &["never", "seen"]) - 1.0).abs() < 1e-6);
}

#[test]
fn smoothing_avoids_zero_probabilities() {
    let tokens = toks("alpha beta gamma alpha beta");
    let model = PerplexityModel::fit(&tokens, 3, 1.0, 0.7);

    // Every vocabulary token, in any context, has strictly positive probability.
    for w in model.vocabulary() {
        assert!(model.probability(&[], w) > 0.0);
        assert!(model.probability(&["alpha"], w) > 0.0);
        assert!(model.probability(&["alpha", "beta"], w) > 0.0);
    }
    // An entirely unseen token also gets non-zero (unk) mass, so surprisal is
    // finite rather than infinite.
    let p_unseen = model.probability(&["alpha", "beta"], "quux");
    assert!(p_unseen > 0.0);
    assert!(model.surprisal(&["alpha", "beta"], "quux").is_finite());
}

#[test]
fn model_reports_vocab_and_totals() {
    let tokens = toks("a b c a b");
    let model = PerplexityModel::fit(&tokens, 2, 1.0, 0.6);
    assert_eq!(model.order(), 2);
    assert_eq!(model.total_tokens(), 5);
    assert_eq!(model.vocab_size(), 3); // a, b, c
}

#[test]
fn reference_corpus_widens_support() {
    let context = toks("the report is short");
    let reference = toks("the annual report is detailed the report is important");
    let model = PerplexityModel::fit_with_reference(&context, &reference, 3, 1.0, 0.7);
    // Reference tokens contribute counts, so total exceeds the context alone.
    assert_eq!(model.total_tokens(), context.len() + reference.len());
    assert!((probability_mass(&model, &["the", "report"]) - 1.0).abs() < 1e-6);
}

// ── surprisal ranking ────────────────────────────────────────────────────────────

#[test]
fn predictable_phrase_has_lower_surprisal_than_novel_phrase() {
    // "the sky is blue" repeats three times (predictable); the tail introduces
    // rare, novel tokens.
    let tokens =
        toks("the sky is blue the sky is blue the sky is blue a xylophone hums quietly today");
    let model = PerplexityModel::fit(&tokens, 3, 1.0, 0.7);

    let predictable = model.surprisal(&["sky", "is"], "blue");
    let novel = model.surprisal(&["blue", "a"], "xylophone");

    assert!(
        novel > predictable,
        "novel token surprisal {novel} should exceed predictable {predictable}"
    );
}

#[test]
fn frequent_token_has_lower_surprisal_than_rare_token_in_unseen_context() {
    let tokens = toks("data data data data data one rare occurrence appears");
    let model = PerplexityModel::fit(&tokens, 3, 1.0, 0.7);
    // Both contexts are unseen at trigram order, forcing backoff to unigram,
    // where "data" (frequent) beats "rare" (seen once).
    let frequent = model.surprisal(&["zzz", "qqq"], "data");
    let rare = model.surprisal(&["zzz", "qqq"], "rare");
    assert!(rare > frequent);
}

// ── BudgetController ─────────────────────────────────────────────────────────────

#[test]
fn allocation_gives_denser_segments_more_tokens() {
    let controller = BudgetController::new(0.9, 1.0);
    let lengths = [10usize, 10];
    let densities = [1.0f32, 5.0];
    let protected = [0usize, 0];

    let keep = controller.allocate(&lengths, &densities, &protected, 10);

    // Total budget met exactly and the denser segment keeps strictly more.
    assert_eq!(keep.iter().sum::<usize>(), 10);
    assert!(
        keep[1] > keep[0],
        "denser segment {} should keep more than sparser {}",
        keep[1],
        keep[0]
    );
    // Bounds respected.
    for &k in &keep {
        assert!((1..=10).contains(&k));
    }
}

#[test]
fn allocation_zero_emphasis_is_uniform_for_equal_lengths() {
    let controller = BudgetController::new(0.9, 0.0);
    let keep = controller.allocate(&[10, 10], &[1.0, 9.0], &[0, 0], 10);
    assert_eq!(keep[0], keep[1]);
    assert_eq!(keep.iter().sum::<usize>(), 10);
}

#[test]
fn allocation_respects_floors_when_budget_too_tight() {
    // max_local_drop_ratio 0.2 => keep at least 80% of each segment.
    let controller = BudgetController::new(0.2, 1.0);
    let keep = controller.allocate(&[10, 10], &[1.0, 1.0], &[0, 0], 5);
    // Floors (8 each) dominate the tiny budget, so the total lands above it.
    assert_eq!(keep, vec![8, 8]);
}

#[test]
fn allocation_protected_tokens_set_the_floor() {
    let controller = BudgetController::new(1.0, 1.0);
    // Segment 0 has 6 protected tokens; even under a tight budget it cannot
    // drop below them.
    let keep = controller.allocate(&[10, 10], &[1.0, 1.0], &[6, 0], 4);
    assert!(keep[0] >= 6);
}

#[test]
fn allocation_empty_input_is_empty() {
    let controller = BudgetController::new(0.5, 1.0);
    assert!(controller.allocate(&[], &[], &[], 5).is_empty());
}

// ── compression: ratios ──────────────────────────────────────────────────────────

fn multi_sentence_text() -> &'static str {
    "the weather in london turned unusually cold during november this year. \
     researchers measured record snowfall across several northern districts overnight. \
     local transport authorities suspended every train until conditions improved greatly."
}

#[test]
fn achieved_ratio_is_close_to_requested() {
    let config = LlmLinguaConfig::new()
        .with_ratio(0.5)
        .with_protect_numbers(false)
        .with_protect_capitalized(false)
        .with_protect_boundaries(false)
        .with_max_local_drop_ratio(1.0);
    let compressor = PerplexityCompressor::new(config);
    let result = compressor
        .compress(multi_sentence_text())
        .expect("non-empty");

    assert_eq!(result.requested_ratio, 0.5);
    assert!(
        (result.achieved_ratio - 0.5).abs() < 0.1,
        "achieved {} should be close to requested 0.5",
        result.achieved_ratio
    );
    assert!(result.compressed_token_count < result.original_token_count);
}

#[test]
fn ratio_one_keeps_everything() {
    let compressor = PerplexityCompressor::new(LlmLinguaConfig::new().with_ratio(1.0));
    let result = compressor
        .compress(multi_sentence_text())
        .expect("non-empty");
    assert_eq!(result.achieved_ratio, 1.0);
    assert_eq!(result.compressed_token_count, result.original_token_count);
    assert!(result.segment_stats.iter().all(|s| s.retained));
}

#[test]
fn near_zero_ratio_compresses_hard_but_stays_nonempty() {
    let compressor = PerplexityCompressor::new(LlmLinguaConfig::new().with_ratio(0.01));
    let result = compressor
        .compress(multi_sentence_text())
        .expect("non-empty");
    assert!(result.compressed_token_count >= 1);
    assert!(result.compressed_token_count < result.original_token_count);
    assert!(!result.compressed_text.is_empty());
    // Cannot actually reach 1% because of floors; achieved therefore exceeds it.
    assert!(result.achieved_ratio >= result.requested_ratio);
}

#[test]
fn token_budget_target_is_honored() {
    // Single sentence, protection off, unlimited local drop: budget hit exactly.
    let text = "one two three four five six seven eight nine ten eleven twelve";
    let config = LlmLinguaConfig::new()
        .with_token_budget(5)
        .with_protect_numbers(false)
        .with_protect_capitalized(false)
        .with_protect_boundaries(false)
        .with_max_local_drop_ratio(1.0);
    let compressor = PerplexityCompressor::new(config);
    let result = compressor.compress(text).expect("non-empty");
    assert_eq!(result.compressed_token_count, 5);
    assert_eq!(result.original_token_count, 12);
    assert!((result.requested_ratio - 5.0 / 12.0).abs() < 1e-6);
}

// ── compression: coarse stage & density ──────────────────────────────────────────

#[test]
fn dense_segment_outranks_boilerplate_segment() {
    // Segment 0 is highly repetitive boilerplate (low density); segment 1 is
    // information-rich novel content (high density).
    let text = "the the the the the the the the. \
                quantum entanglement violates classical locality assumptions dramatically.";
    let compressor = PerplexityCompressor::new(LlmLinguaConfig::new().with_ratio(0.99));
    let result = compressor.compress(text).expect("non-empty");
    assert_eq!(result.segment_stats.len(), 2);
    assert!(
        result.segment_stats[0].density < result.segment_stats[1].density,
        "boilerplate density {} should be below novel density {}",
        result.segment_stats[0].density,
        result.segment_stats[1].density
    );
}

#[test]
fn coarse_stage_drops_lowest_density_segment_first() {
    // Under a tight budget the low-density boilerplate segment is dropped whole.
    // The target (~6 tokens) is below the novel segment's length, so dropping
    // the 10-token boilerplate is affordable without undershooting the budget.
    let text = "the the the the the the the the the the. \
                rare novel entropic assertions surprise attentive skeptical readers everywhere.";
    let config = LlmLinguaConfig::new()
        .with_ratio(0.3)
        .with_min_segments_retained(1);
    let compressor = PerplexityCompressor::new(config);
    let result = compressor.compress(text).expect("non-empty");

    let boilerplate = &result.segment_stats[0];
    let novel = &result.segment_stats[1];
    assert!(!boilerplate.retained, "boilerplate should be dropped");
    assert!(novel.retained, "novel segment should survive");
    assert!(boilerplate.text.is_empty());
    assert_eq!(boilerplate.retained_tokens, 0);
}

#[test]
fn min_segments_retained_floor_is_respected() {
    let text = "aaa aaa aaa. bbb bbb bbb. ccc ccc ccc. ddd ddd ddd.";
    let config = LlmLinguaConfig::new()
        .with_ratio(0.05)
        .with_min_segments_retained(3);
    let compressor = PerplexityCompressor::new(config);
    let result = compressor.compress(text).expect("non-empty");
    let retained = result.segment_stats.iter().filter(|s| s.retained).count();
    assert!(retained >= 3);
}

// ── compression: protection ──────────────────────────────────────────────────────

#[test]
fn protected_tokens_survive_aggressive_compression() {
    let text = "Contact Alice about the 2024 Berlin summit immediately.";
    let compressor = PerplexityCompressor::new(LlmLinguaConfig::new().with_ratio(0.3));
    let result = compressor.compress(text).expect("non-empty");
    // Single sentence => always retained; protected tokens must remain.
    assert!(result.compressed_text.contains("2024"), "number kept");
    assert!(
        result.compressed_text.contains("Berlin"),
        "proper noun kept"
    );
    assert!(result.compressed_text.contains("Alice"), "proper noun kept");
    // Boundary tokens kept.
    assert!(result.compressed_text.starts_with("Contact"));
}

#[test]
fn disabling_protection_allows_dropping_numbers() {
    let text = "one two three four 5 6 7 8 nine ten eleven twelve thirteen fourteen";
    let protected_cfg = LlmLinguaConfig::new().with_ratio(0.3);
    let unprotected_cfg = LlmLinguaConfig::new()
        .with_ratio(0.3)
        .with_protect_numbers(false)
        .with_protect_capitalized(false)
        .with_protect_boundaries(false)
        .with_max_local_drop_ratio(1.0);

    let with_protection = PerplexityCompressor::new(protected_cfg)
        .compress(text)
        .expect("non-empty");
    let without = PerplexityCompressor::new(unprotected_cfg)
        .compress(text)
        .expect("non-empty");

    // Removing protection permits equal-or-stronger compression.
    assert!(without.compressed_token_count <= with_protection.compressed_token_count);
}

// ── compression: edge cases ──────────────────────────────────────────────────────

#[test]
fn single_sentence_input() {
    let text = "the small brown fox jumped over the lazy sleeping dog quietly.";
    let compressor = PerplexityCompressor::new(LlmLinguaConfig::new().with_ratio(0.5));
    let result = compressor.compress(text).expect("non-empty");
    assert_eq!(result.segment_stats.len(), 1);
    assert!(result.segment_stats[0].retained);
    assert!(!result.compressed_text.is_empty());
    assert!(result.compressed_token_count <= result.original_token_count);
    assert!(result.compressed_token_count >= 1);
}

#[test]
fn empty_input_errors() {
    let compressor = PerplexityCompressor::default();
    assert_eq!(compressor.compress(""), Err(LlmLinguaError::EmptyInput));
    assert_eq!(
        compressor.compress("   \n  \t "),
        Err(LlmLinguaError::EmptyInput)
    );
}

#[test]
fn result_convenience_methods() {
    let compressor = PerplexityCompressor::new(
        LlmLinguaConfig::new()
            .with_ratio(0.5)
            .with_protect_numbers(false)
            .with_protect_capitalized(false)
            .with_protect_boundaries(false)
            .with_max_local_drop_ratio(1.0),
    );
    let result = compressor
        .compress(multi_sentence_text())
        .expect("non-empty");
    assert_eq!(
        result.tokens_saved(),
        result.original_token_count - result.compressed_token_count
    );
    assert!((result.compression_rate() - (1.0 - result.achieved_ratio)).abs() < 1e-6);
    // Per-segment dropped_tokens is consistent.
    for stat in &result.segment_stats {
        assert_eq!(
            stat.dropped_tokens(),
            stat.original_tokens - stat.retained_tokens
        );
    }
}

// ── configuration validation ─────────────────────────────────────────────────────

#[test]
fn config_validation_rejects_bad_values() {
    let cases = vec![
        LlmLinguaConfig::new().with_ngram_order(0),
        LlmLinguaConfig::new().with_ngram_order(9),
        LlmLinguaConfig::new().with_add_k(0.0),
        LlmLinguaConfig::new().with_add_k(-1.0),
        LlmLinguaConfig::new().with_interpolation_lambda(0.0),
        LlmLinguaConfig::new().with_interpolation_lambda(1.5),
        LlmLinguaConfig::new().with_ratio(0.0),
        LlmLinguaConfig::new().with_ratio(1.5),
        LlmLinguaConfig::new().with_token_budget(0),
        LlmLinguaConfig::new().with_min_segments_retained(0),
        LlmLinguaConfig::new().with_max_local_drop_ratio(1.5),
        LlmLinguaConfig::new().with_density_emphasis(-0.5),
    ];
    for config in cases {
        assert!(
            matches!(config.validate(), Err(LlmLinguaError::InvalidConfig(_))),
            "expected invalid config to be rejected: {config:?}"
        );
    }
}

#[test]
fn config_validation_accepts_defaults_and_valid_targets() {
    assert!(LlmLinguaConfig::new().validate().is_ok());
    assert!(
        LlmLinguaConfig::new()
            .with_target(CompressionTarget::TokenBudget(10))
            .validate()
            .is_ok()
    );
    assert!(
        LlmLinguaConfig::new()
            .with_ngram_order(1)
            .validate()
            .is_ok()
    );
}

#[test]
fn invalid_config_surfaces_from_compress() {
    let compressor = PerplexityCompressor::new(LlmLinguaConfig::new().with_ratio(2.0));
    assert!(matches!(
        compressor.compress("some text here."),
        Err(LlmLinguaError::InvalidConfig(_))
    ));
}
