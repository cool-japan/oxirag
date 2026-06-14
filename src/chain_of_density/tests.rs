#![allow(
    clippy::float_cmp,
    clippy::similar_names,
    clippy::cast_precision_loss,
    clippy::useless_vec,
    clippy::map_unwrap_or,
    clippy::manual_range_contains,
    clippy::many_single_char_names,
    clippy::match_wildcard_for_single_variants,
    clippy::default_constructed_unit_structs,
    clippy::doc_markdown,
    clippy::too_many_lines,
    clippy::items_after_statements
)]
//! Tests for the `chain_of_density` module.

use super::engine::{
    ChainOfDensityEngine, extract_entities, initial_summary, is_entity, is_filler, raw_tokens,
    trim_filler, word_count,
};
use super::types::{ChainOfDensityOutput, CodConfig, CodError, DensityStep};

// ── Test fixtures ─────────────────────────────────────────────────────────────

/// A dense source mentioning many distinct entities and numbers.
const DENSE_SOURCE: &str = "Ada Lovelace worked with Charles Babbage in London during 1843. \
The Analytical Engine was a mechanical computer designed by Babbage. \
Lovelace wrote the first published algorithm for the machine. \
Her notes referenced Bernoulli numbers and the year 1837. \
The British mathematician collaborated with Babbage on the Difference Engine.";

/// A short source with only a couple of entities.
const SHORT_SOURCE: &str = "Rust is a systems language. It was created at Mozilla.";

fn default_engine() -> ChainOfDensityEngine {
    ChainOfDensityEngine::new(CodConfig::default())
}

/// Returns `true` when every word in `entity` appears as a whole token in
/// `summary` (case-insensitive).
fn summary_contains_entity(summary: &str, entity: &str) -> bool {
    let tokens: Vec<String> = raw_tokens(summary)
        .iter()
        .map(|t| t.to_lowercase())
        .collect();
    raw_tokens(entity)
        .iter()
        .all(|part| tokens.contains(&part.to_lowercase()))
}

// ── CodConfig tests ───────────────────────────────────────────────────────────

#[test]
fn test_config_default_values() {
    let cfg = CodConfig::default();
    assert_eq!(cfg.iterations, 3);
    assert_eq!(cfg.target_words, 60);
    assert_eq!(cfg.entities_per_step, 2);
}

#[test]
fn test_config_new_matches_default() {
    assert_eq!(CodConfig::new(), CodConfig::default());
}

#[test]
fn test_config_with_iterations() {
    let cfg = CodConfig::new().with_iterations(7);
    assert_eq!(cfg.iterations, 7);
    // Other fields unchanged.
    assert_eq!(cfg.target_words, 60);
    assert_eq!(cfg.entities_per_step, 2);
}

#[test]
fn test_config_with_target_words() {
    let cfg = CodConfig::new().with_target_words(120);
    assert_eq!(cfg.target_words, 120);
    assert_eq!(cfg.iterations, 3);
}

#[test]
fn test_config_with_entities_per_step() {
    let cfg = CodConfig::new().with_entities_per_step(5);
    assert_eq!(cfg.entities_per_step, 5);
    assert_eq!(cfg.iterations, 3);
}

#[test]
fn test_config_builder_chain() {
    let cfg = CodConfig::new()
        .with_iterations(4)
        .with_target_words(80)
        .with_entities_per_step(3);
    assert_eq!(cfg.iterations, 4);
    assert_eq!(cfg.target_words, 80);
    assert_eq!(cfg.entities_per_step, 3);
}

#[test]
fn test_config_clone_eq() {
    let cfg = CodConfig::new().with_iterations(9);
    let cloned = cfg.clone();
    assert_eq!(cfg, cloned);
}

// ── Entity extraction tests ───────────────────────────────────────────────────

#[test]
fn test_is_entity_capitalized() {
    assert!(is_entity("London"));
    assert!(is_entity("Babbage"));
    assert!(is_entity("Ab"));
}

#[test]
fn test_is_entity_numbers() {
    assert!(is_entity("1843"));
    assert!(is_entity("2026"));
    assert!(is_entity("0"));
}

#[test]
fn test_is_entity_rejects_lowercase() {
    assert!(!is_entity("london"));
    assert!(!is_entity("the"));
    assert!(!is_entity("machine"));
}

#[test]
fn test_is_entity_rejects_single_capital() {
    // Single capitalized character is not a multi-char entity.
    assert!(!is_entity("A"));
    assert!(!is_entity("I"));
}

#[test]
fn test_is_entity_rejects_empty() {
    assert!(!is_entity(""));
}

#[test]
fn test_is_entity_rejects_mixed_alnum() {
    // Mixed letters and digits starting with a digit is not a pure number.
    assert!(!is_entity("12ab"));
}

#[test]
fn test_extract_entities_order_preserved() {
    let entities = extract_entities("Charlie met Alice then Bob met Charlie.");
    assert_eq!(entities, vec!["Charlie", "Alice", "Bob"]);
}

#[test]
fn test_extract_entities_dedup_case_insensitive() {
    let entities = extract_entities("Rust and RUST and rust differ. Rust again.");
    // "Rust" appears once; "rust" (lowercase) is not an entity.
    assert_eq!(entities, vec!["Rust"]);
}

#[test]
fn test_extract_entities_includes_numbers() {
    let entities = extract_entities("In 1837 and 1843 things happened.");
    assert!(entities.contains(&"1837".to_string()));
    assert!(entities.contains(&"1843".to_string()));
}

#[test]
fn test_extract_entities_empty_source() {
    assert!(extract_entities("").is_empty());
    assert!(extract_entities("the and of is").is_empty());
}

#[test]
fn test_extract_entities_dense_source_many() {
    let entities = extract_entities(DENSE_SOURCE);
    // Should find a healthy number of distinct entities.
    assert!(entities.len() >= 8, "got {}", entities.len());
    assert!(entities.contains(&"Lovelace".to_string()));
    assert!(entities.contains(&"Babbage".to_string()));
    assert!(entities.contains(&"London".to_string()));
}

// ── Helper tests ──────────────────────────────────────────────────────────────

#[test]
fn test_word_count_basic() {
    assert_eq!(word_count("one two three"), 3);
    assert_eq!(word_count("   spaced   out   words  "), 3);
    assert_eq!(word_count(""), 0);
}

#[test]
fn test_is_filler_recognises_common_words() {
    assert!(is_filler("the"));
    assert!(is_filler("The"));
    assert!(is_filler("a"));
    assert!(is_filler("with,"));
}

#[test]
fn test_is_filler_rejects_content_words() {
    assert!(!is_filler("Lovelace"));
    assert!(!is_filler("algorithm"));
    assert!(!is_filler("1843"));
}

#[test]
fn test_trim_filler_reduces_to_budget() {
    let text = "the quick brown Fox jumps over the lazy Dog";
    let trimmed = trim_filler(text, 6);
    assert!(word_count(&trimmed) <= 6, "got {}", word_count(&trimmed));
}

#[test]
fn test_trim_filler_keeps_under_budget_unchanged() {
    let text = "Fox Dog Cat";
    let trimmed = trim_filler(text, 10);
    assert_eq!(trimmed, text);
}

#[test]
fn test_trim_filler_preserves_last_word() {
    let text = "Alice and Bob and the";
    let trimmed = trim_filler(text, 1);
    // Cannot remove the final word, so at least one word remains.
    assert!(!trimmed.is_empty());
    assert!(trimmed.ends_with("the"));
}

#[test]
fn test_trim_filler_no_filler_no_change() {
    // No filler words to remove → text returned even though over budget.
    let text = "Alice Bob Charlie Dave";
    let trimmed = trim_filler(text, 2);
    assert_eq!(trimmed, text);
}

#[test]
fn test_initial_summary_lead_sentence() {
    let sentences: Vec<String> = vec!["First sentence here".to_string(), "Second one".to_string()];
    let summary = initial_summary(&sentences, 60);
    assert!(summary.starts_with("First sentence here"));
}

#[test]
fn test_initial_summary_respects_budget() {
    let sentences: Vec<String> = vec![
        "alpha beta gamma delta epsilon zeta eta theta".to_string(),
        "another sentence".to_string(),
    ];
    let summary = initial_summary(&sentences, 4);
    assert!(word_count(&summary) <= 4, "got {}", word_count(&summary));
}

#[test]
fn test_initial_summary_empty() {
    let sentences: Vec<String> = Vec::new();
    assert!(initial_summary(&sentences, 60).is_empty());
}

#[test]
fn test_initial_summary_ends_with_period() {
    let sentences: Vec<String> = vec!["No terminal punctuation here".to_string()];
    let summary = initial_summary(&sentences, 60);
    assert!(summary.ends_with('.'));
}

// ── summarize: structural tests ───────────────────────────────────────────────

#[test]
fn test_summarize_step_count_equals_iterations() {
    let engine = default_engine();
    let output = engine.summarize(DENSE_SOURCE).unwrap();
    assert_eq!(output.steps.len(), 3);
}

#[test]
fn test_summarize_custom_iterations() {
    let engine = ChainOfDensityEngine::new(CodConfig::new().with_iterations(5));
    let output = engine.summarize(DENSE_SOURCE).unwrap();
    assert_eq!(output.steps.len(), 5);
}

#[test]
fn test_summarize_zero_iterations() {
    let engine = ChainOfDensityEngine::new(CodConfig::new().with_iterations(0));
    let output = engine.summarize(DENSE_SOURCE).unwrap();
    assert!(output.steps.is_empty());
}

#[test]
fn test_summarize_final_equals_last_step() {
    let engine = default_engine();
    let output = engine.summarize(DENSE_SOURCE).unwrap();
    let last = output.steps.last().unwrap();
    assert_eq!(output.final_summary, last.summary);
}

#[test]
fn test_summarize_non_empty_summaries() {
    let engine = default_engine();
    let output = engine.summarize(DENSE_SOURCE).unwrap();
    for step in &output.steps {
        assert!(!step.summary.trim().is_empty());
    }
}

#[test]
fn test_summarize_word_count_field_matches_summary() {
    let engine = default_engine();
    let output = engine.summarize(DENSE_SOURCE).unwrap();
    for step in &output.steps {
        assert_eq!(step.word_count, word_count(&step.summary));
    }
}

// ── summarize: added entities are real & novel ────────────────────────────────

#[test]
fn test_added_entities_are_source_entities() {
    let engine = default_engine();
    let output = engine.summarize(DENSE_SOURCE).unwrap();
    let source_entities = extract_entities(DENSE_SOURCE);
    for step in &output.steps {
        for added in &step.added_entities {
            assert!(
                source_entities.contains(added),
                "{added} is not a source entity"
            );
        }
    }
}

#[test]
fn test_added_entities_not_previously_present() {
    let engine = default_engine();
    let output = engine.summarize(DENSE_SOURCE).unwrap();

    // Reconstruct: before each step, none of its added entities were present in
    // the *previous* step's summary (or the initial summary for step 0).
    let sentences: Vec<String> = DENSE_SOURCE
        .split(['.', '!', '?'])
        .map(str::trim)
        .filter(|s| !s.is_empty())
        .map(str::to_string)
        .collect();
    let initial = initial_summary(&sentences, CodConfig::default().target_words);

    let mut prev_summary = initial;
    for step in &output.steps {
        for added in &step.added_entities {
            assert!(
                !summary_contains_entity(&prev_summary, added),
                "entity {added} was already present before its step"
            );
        }
        prev_summary = step.summary.clone();
    }
}

#[test]
fn test_added_entities_appear_in_step_summary() {
    let engine = default_engine();
    let output = engine.summarize(DENSE_SOURCE).unwrap();
    for step in &output.steps {
        for added in &step.added_entities {
            assert!(
                summary_contains_entity(&step.summary, added),
                "entity {added} not present in its own step summary"
            );
        }
    }
}

#[test]
fn test_added_entities_unique_across_steps() {
    let engine = default_engine();
    let output = engine.summarize(DENSE_SOURCE).unwrap();
    let mut seen: Vec<String> = Vec::new();
    for step in &output.steps {
        for added in &step.added_entities {
            assert!(!seen.contains(added), "entity {added} added twice");
            seen.push(added.clone());
        }
    }
}

#[test]
fn test_added_entities_count_within_step_limit() {
    let engine = ChainOfDensityEngine::new(CodConfig::new().with_entities_per_step(2));
    let output = engine.summarize(DENSE_SOURCE).unwrap();
    for step in &output.steps {
        assert!(step.added_entities.len() <= 2);
    }
}

// ── summarize: entity_count monotonicity ──────────────────────────────────────

#[test]
fn test_entity_count_non_decreasing() {
    let engine = default_engine();
    let output = engine.summarize(DENSE_SOURCE).unwrap();
    for window in output.steps.windows(2) {
        assert!(
            window[1].entity_count >= window[0].entity_count,
            "entity_count decreased: {} -> {}",
            window[0].entity_count,
            window[1].entity_count
        );
    }
}

#[test]
fn test_entity_count_strictly_increases_when_added() {
    let engine = default_engine();
    let output = engine.summarize(DENSE_SOURCE).unwrap();
    for window in output.steps.windows(2) {
        if !window[1].added_entities.is_empty() {
            assert!(
                window[1].entity_count > window[0].entity_count,
                "expected strict increase when {} entities added",
                window[1].added_entities.len()
            );
        }
    }
}

#[test]
fn test_entity_count_increases_from_initial() {
    let engine = default_engine();
    let output = engine.summarize(DENSE_SOURCE).unwrap();
    // First step should not have fewer entities than zero and should reflect
    // additions when any occurred.
    let first = &output.steps[0];
    if !first.added_entities.is_empty() {
        assert!(first.entity_count >= first.added_entities.len());
    }
}

#[test]
fn test_entity_count_matches_field() {
    let engine = default_engine();
    let output = engine.summarize(DENSE_SOURCE).unwrap();
    let source_entities = extract_entities(DENSE_SOURCE);
    for step in &output.steps {
        let actual = source_entities
            .iter()
            .filter(|e| summary_contains_entity(&step.summary, e))
            .count();
        assert_eq!(step.entity_count, actual);
    }
}

// ── summarize: word budget ────────────────────────────────────────────────────

#[test]
fn test_word_count_stays_near_target() {
    let target = 60;
    let engine = ChainOfDensityEngine::new(CodConfig::new().with_target_words(target));
    let output = engine.summarize(DENSE_SOURCE).unwrap();
    // Allow a small slack for the entity clause that pushes over before trim.
    let slack = 12;
    for step in &output.steps {
        assert!(
            step.word_count <= target + slack,
            "word_count {} exceeds target {} + slack {}",
            step.word_count,
            target,
            slack
        );
    }
}

#[test]
fn test_word_count_small_target_compresses() {
    let target = 20;
    let engine = ChainOfDensityEngine::new(
        CodConfig::new()
            .with_target_words(target)
            .with_iterations(4),
    );
    let output = engine.summarize(DENSE_SOURCE).unwrap();
    let slack = 12;
    for step in &output.steps {
        assert!(
            step.word_count <= target + slack,
            "word_count {} too large for target {}",
            step.word_count,
            target
        );
    }
}

#[test]
fn test_word_count_does_not_explode_over_iterations() {
    let engine =
        ChainOfDensityEngine::new(CodConfig::new().with_iterations(8).with_target_words(40));
    let output = engine.summarize(DENSE_SOURCE).unwrap();
    let first = output.steps.first().unwrap().word_count;
    let last = output.steps.last().unwrap().word_count;
    // Length stays bounded rather than growing linearly with iterations.
    assert!(last <= first + 12, "first {first} last {last}");
}

// ── summarize: densification behaviour ────────────────────────────────────────

#[test]
fn test_dense_source_densifies() {
    let engine = default_engine();
    let output = engine.summarize(DENSE_SOURCE).unwrap();
    let first = output.steps.first().unwrap();
    let last = output.steps.last().unwrap();
    // The final summary should have at least as many entities as the first.
    assert!(last.entity_count >= first.entity_count);
    // And density should rise (entities per word) given a bounded length.
    assert!(last.density() >= first.density());
}

#[test]
fn test_many_entities_get_incorporated() {
    let engine = ChainOfDensityEngine::new(CodConfig::new().with_iterations(4));
    let output = engine.summarize(DENSE_SOURCE).unwrap();
    let total_added: usize = output.steps.iter().map(|s| s.added_entities.len()).sum();
    assert!(total_added >= 4, "only {total_added} entities incorporated");
}

#[test]
fn test_final_entity_count_helper() {
    let engine = default_engine();
    let output = engine.summarize(DENSE_SOURCE).unwrap();
    assert_eq!(
        output.final_entity_count(),
        output.steps.last().unwrap().entity_count
    );
}

#[test]
fn test_step_count_helper() {
    let engine = default_engine();
    let output = engine.summarize(DENSE_SOURCE).unwrap();
    assert_eq!(output.step_count(), output.steps.len());
}

#[test]
fn test_short_source_still_produces_output() {
    let engine = default_engine();
    let output = engine.summarize(SHORT_SOURCE).unwrap();
    assert_eq!(output.steps.len(), 3);
    assert!(!output.final_summary.is_empty());
}

#[test]
fn test_source_with_no_entities() {
    // Lowercase-only source has no entities; steps still recorded, no additions.
    let engine = default_engine();
    let output = engine
        .summarize("the cat sat on the mat near the door")
        .unwrap();
    assert_eq!(output.steps.len(), 3);
    for step in &output.steps {
        assert!(step.added_entities.is_empty());
        assert_eq!(step.entity_count, 0);
    }
}

#[test]
fn test_entities_run_out_gracefully() {
    // Only two entities but four iterations: later steps add nothing.
    let source = "Alice met Bob. They talked about things and other matters at length.";
    let engine = ChainOfDensityEngine::new(
        CodConfig::new()
            .with_iterations(4)
            .with_entities_per_step(1),
    );
    let output = engine.summarize(source).unwrap();
    assert_eq!(output.steps.len(), 4);
    let total_added: usize = output.steps.iter().map(|s| s.added_entities.len()).sum();
    // At most the number of distinct entities can ever be added.
    assert!(total_added <= extract_entities(source).len());
}

// ── Error tests ───────────────────────────────────────────────────────────────

#[test]
fn test_empty_source_error() {
    let engine = default_engine();
    let err = engine.summarize("").unwrap_err();
    assert!(matches!(err, CodError::EmptySource));
}

#[test]
fn test_whitespace_source_error() {
    let engine = default_engine();
    let err = engine.summarize("   \n\t  ").unwrap_err();
    assert!(matches!(err, CodError::EmptySource));
}

#[test]
fn test_error_display_message() {
    let err = CodError::EmptySource;
    assert_eq!(err.to_string(), "source must not be empty");
}

// ── Determinism tests ─────────────────────────────────────────────────────────

#[test]
fn test_summarize_is_deterministic() {
    let engine = default_engine();
    let a = engine.summarize(DENSE_SOURCE).unwrap();
    let b = engine.summarize(DENSE_SOURCE).unwrap();
    assert_eq!(a.final_summary, b.final_summary);
    assert_eq!(a.steps, b.steps);
}

#[test]
fn test_separate_engines_same_output() {
    let a = default_engine().summarize(DENSE_SOURCE).unwrap();
    let b = ChainOfDensityEngine::default()
        .summarize(DENSE_SOURCE)
        .unwrap();
    assert_eq!(a, b);
}

#[test]
fn test_determinism_across_configs() {
    let cfg = CodConfig::new()
        .with_iterations(5)
        .with_target_words(45)
        .with_entities_per_step(3);
    let a = ChainOfDensityEngine::new(cfg.clone())
        .summarize(DENSE_SOURCE)
        .unwrap();
    let b = ChainOfDensityEngine::new(cfg)
        .summarize(DENSE_SOURCE)
        .unwrap();
    assert_eq!(a, b);
}

// ── DensityStep / Output type tests ───────────────────────────────────────────

#[test]
fn test_density_step_density_zero_when_empty() {
    let step = DensityStep {
        summary: String::new(),
        added_entities: Vec::new(),
        word_count: 0,
        entity_count: 0,
    };
    assert_eq!(step.density(), 0.0);
}

#[test]
fn test_density_step_density_value() {
    let step = DensityStep {
        summary: "Alice Bob extra words here".to_string(),
        added_entities: vec!["Alice".to_string(), "Bob".to_string()],
        word_count: 5,
        entity_count: 2,
    };
    assert!((step.density() - 0.4).abs() < 1e-6);
}

#[test]
fn test_output_default_helpers_on_empty() {
    let output = ChainOfDensityOutput {
        steps: Vec::new(),
        final_summary: String::new(),
    };
    assert_eq!(output.step_count(), 0);
    assert_eq!(output.final_entity_count(), 0);
}

#[test]
fn test_engine_config_accessor() {
    let cfg = CodConfig::new().with_iterations(6);
    let engine = ChainOfDensityEngine::new(cfg.clone());
    assert_eq!(engine.config(), &cfg);
}

#[test]
fn test_output_clone_eq() {
    let engine = default_engine();
    let output = engine.summarize(DENSE_SOURCE).unwrap();
    let cloned = output.clone();
    assert_eq!(output, cloned);
}
