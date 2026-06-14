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
    clippy::too_many_lines
)]
//! Tests for the `context_pruning` module.

use crate::types::Document;

use super::pruner::TokenPruner;
use super::types::{ContextPruningError, PruneConfig, PrunedContext};

// ── Test helpers ──────────────────────────────────────────────────────────────

fn words(s: &str) -> Vec<String> {
    s.split_whitespace().map(str::to_string).collect()
}

/// Return the words of `text` in order, used to assert order preservation.
fn order_of(text: &str) -> Vec<String> {
    words(text)
}

/// Check `sub` appears as a subsequence of `full` (same relative order).
fn is_subsequence(sub: &[String], full: &[String]) -> bool {
    let mut it = full.iter();
    sub.iter().all(|s| it.any(|f| f == s))
}

// ── PruneConfig defaults & builders ───────────────────────────────────────────

#[test]
fn test_config_default_values() {
    let cfg = PruneConfig::default();
    assert_eq!(cfg.target_ratio, 0.5);
    assert_eq!(cfg.min_tokens, 5);
    assert!(cfg.preserve_entities);
}

#[test]
fn test_config_new_matches_default() {
    assert_eq!(PruneConfig::new(), PruneConfig::default());
}

#[test]
fn test_config_with_target_ratio() {
    let cfg = PruneConfig::new().with_target_ratio(0.25);
    assert_eq!(cfg.target_ratio, 0.25);
}

#[test]
fn test_config_with_min_tokens() {
    let cfg = PruneConfig::new().with_min_tokens(3);
    assert_eq!(cfg.min_tokens, 3);
}

#[test]
fn test_config_with_preserve_entities() {
    let cfg = PruneConfig::new().with_preserve_entities(false);
    assert!(!cfg.preserve_entities);
}

#[test]
fn test_config_builder_chain() {
    let cfg = PruneConfig::new()
        .with_target_ratio(0.3)
        .with_min_tokens(2)
        .with_preserve_entities(false);
    assert_eq!(cfg.target_ratio, 0.3);
    assert_eq!(cfg.min_tokens, 2);
    assert!(!cfg.preserve_entities);
}

#[test]
fn test_config_clone_eq() {
    let cfg = PruneConfig::new().with_target_ratio(0.7);
    assert_eq!(cfg.clone(), cfg);
}

// ── keep ≈ target_ratio ───────────────────────────────────────────────────────

#[test]
fn test_keeps_half_by_default() {
    // 20 distinct content words, ratio 0.5 → keep 10.
    let pruner = TokenPruner::new(PruneConfig::new().with_min_tokens(1));
    let text = "alpha beta gamma delta epsilon zeta eta theta iota kappa \
                lambda mu nu xi omicron pi rho sigma tau upsilon";
    let pruned = pruner.prune(text, None).expect("ok");
    assert_eq!(pruned.original_tokens, 20);
    assert_eq!(pruned.kept_tokens, 10);
}

#[test]
fn test_keep_count_quarter_ratio() {
    let pruner = TokenPruner::new(
        PruneConfig::new()
            .with_target_ratio(0.25)
            .with_min_tokens(1),
    );
    let text = "alpha beta gamma delta epsilon zeta eta theta iota kappa \
                lambda mu nu xi omicron pi rho sigma tau upsilon";
    let pruned = pruner.prune(text, None).expect("ok");
    // round(0.25 * 20) == 5
    assert_eq!(pruned.kept_tokens, 5);
}

#[test]
fn test_keep_count_rounds() {
    // 7 words, ratio 0.5 → round(3.5) == 4 (banker's? Rust round() rounds half away from zero → 4).
    let pruner = TokenPruner::new(PruneConfig::new().with_min_tokens(1));
    let text = "alpha beta gamma delta epsilon zeta eta";
    let pruned = pruner.prune(text, None).expect("ok");
    assert_eq!(pruned.original_tokens, 7);
    assert_eq!(pruned.kept_tokens, 4);
}

#[test]
fn test_full_ratio_keeps_all() {
    let pruner = TokenPruner::new(PruneConfig::new().with_target_ratio(1.0));
    let text = "alpha beta gamma delta epsilon zeta";
    let pruned = pruner.prune(text, None).expect("ok");
    assert_eq!(pruned.kept_tokens, pruned.original_tokens);
}

#[test]
fn test_ratio_clamped_above_one() {
    let pruner = TokenPruner::new(PruneConfig::new().with_target_ratio(5.0));
    let text = "alpha beta gamma delta epsilon zeta";
    let pruned = pruner.prune(text, None).expect("ok");
    assert_eq!(pruned.kept_tokens, pruned.original_tokens);
}

#[test]
fn test_zero_ratio_floored_to_min_tokens() {
    let pruner = TokenPruner::new(PruneConfig::new().with_target_ratio(0.0).with_min_tokens(3));
    let text = "alpha beta gamma delta epsilon zeta eta theta";
    let pruned = pruner.prune(text, None).expect("ok");
    // round(0) == 0, floored to min_tokens 3.
    assert_eq!(pruned.kept_tokens, 3);
}

// ── min_tokens floor ──────────────────────────────────────────────────────────

#[test]
fn test_min_tokens_floor_on_short_context() {
    // 6 words, ratio 0.5 → round(3) == 3, but min_tokens 5 floors to 5.
    let pruner = TokenPruner::new(PruneConfig::default());
    let text = "alpha beta gamma delta epsilon zeta";
    let pruned = pruner.prune(text, None).expect("ok");
    assert_eq!(pruned.kept_tokens, 5);
}

#[test]
fn test_min_tokens_capped_at_n() {
    // Context shorter than min_tokens → keep everything.
    let pruner = TokenPruner::new(PruneConfig::default());
    let text = "alpha beta gamma";
    let pruned = pruner.prune(text, None).expect("ok");
    assert_eq!(pruned.kept_tokens, 3);
    assert_eq!(pruned.original_tokens, 3);
}

#[test]
fn test_single_word_keeps_one() {
    let pruner = TokenPruner::new(PruneConfig::default());
    let pruned = pruner.prune("solitary", None).expect("ok");
    assert_eq!(pruned.kept_tokens, 1);
    assert_eq!(pruned.text, "solitary");
}

// ── high-idf survives, stopwords dropped first ────────────────────────────────

#[test]
fn test_stopwords_dropped_before_content_words() {
    // Mix of stopwords and rare content words; keep half.
    let pruner = TokenPruner::new(
        PruneConfig::new()
            .with_min_tokens(1)
            .with_preserve_entities(false),
    );
    let text = "the quark and the gluon are in the proton";
    // 9 words → keep round(4.5) == 5. Stopwords: the(x3), and, are, in → 6 stopwords.
    // Content: quark, gluon, proton → 3. All 3 content kept + 2 stopwords.
    let pruned = pruner.prune(text, None).expect("ok");
    assert_eq!(pruned.kept_tokens, 5);
    let kept = order_of(&pruned.text);
    assert!(kept.contains(&"quark".to_string()));
    assert!(kept.contains(&"gluon".to_string()));
    assert!(kept.contains(&"proton".to_string()));
}

#[test]
fn test_rare_word_outranks_common_word() {
    let pruner = TokenPruner::new(PruneConfig::new().with_preserve_entities(false));
    // "data" repeated many times (low idf), "synapse" once (high idf).
    let text = "data data data data data data data synapse";
    let imp = pruner.token_importance(&text.split_whitespace().collect::<Vec<_>>(), None);
    let last = imp[imp.len() - 1]; // synapse
    let first = imp[0]; // data
    assert!(
        last > first,
        "rare 'synapse' {last} should outrank common 'data' {first}"
    );
}

#[test]
fn test_repeated_filler_dropped() {
    let pruner = TokenPruner::new(
        PruneConfig::new()
            .with_target_ratio(0.4)
            .with_min_tokens(1)
            .with_preserve_entities(false),
    );
    let text = "noise noise noise noise noise signal";
    // 6 words → keep round(2.4) == 2. "signal" (rare) must be kept.
    let pruned = pruner.prune(text, None).expect("ok");
    assert!(order_of(&pruned.text).contains(&"signal".to_string()));
}

#[test]
fn test_all_stopwords_still_keeps_min() {
    let pruner = TokenPruner::new(PruneConfig::new().with_min_tokens(2).with_target_ratio(0.0));
    let text = "the and of to in on";
    let pruned = pruner.prune(text, None).expect("ok");
    assert_eq!(pruned.kept_tokens, 2);
}

// ── preserve_entities ─────────────────────────────────────────────────────────

#[test]
fn test_preserve_entities_keeps_rare_capitalized() {
    // Build a long context of common words plus a single capitalized entity.
    let pruner = TokenPruner::new(
        PruneConfig::new()
            .with_target_ratio(0.1)
            .with_min_tokens(1)
            .with_preserve_entities(true),
    );
    let text = "data data data data data data data data data Einstein";
    // ratio 0.1 of 10 → keep round(1.0)==1; Einstein must be that one.
    let pruned = pruner.prune(text, None).expect("ok");
    assert!(
        order_of(&pruned.text).contains(&"Einstein".to_string()),
        "capitalized entity must survive, got: {}",
        pruned.text
    );
}

#[test]
fn test_preserve_entities_disabled_may_drop_entity() {
    let pruner = TokenPruner::new(
        PruneConfig::new()
            .with_target_ratio(0.1)
            .with_min_tokens(1)
            .with_preserve_entities(false),
    );
    // Entity appears many times → low idf → droppable when preservation off.
    let text = "Acme Acme Acme Acme Acme Acme Acme Acme Acme novelty";
    let pruned = pruner.prune(text, None).expect("ok");
    // "novelty" is rarer than "acme" → it wins the single slot.
    assert!(order_of(&pruned.text).contains(&"novelty".to_string()));
}

#[test]
fn test_preserve_entities_multiple() {
    let pruner = TokenPruner::new(
        PruneConfig::new()
            .with_target_ratio(0.2)
            .with_min_tokens(1)
            .with_preserve_entities(true),
    );
    let text = "common common common common common Paris common common Tokyo common";
    let pruned = pruner.prune(text, None).expect("ok");
    let kept = order_of(&pruned.text);
    assert!(kept.contains(&"Paris".to_string()));
    assert!(kept.contains(&"Tokyo".to_string()));
}

#[test]
fn test_entity_importance_is_max() {
    let pruner = TokenPruner::new(PruneConfig::new().with_preserve_entities(true));
    let imp = pruner.token_importance(&["hello", "World"], None);
    assert_eq!(imp[1], f32::MAX);
    assert!(imp[0] < f32::MAX);
}

#[test]
fn test_entity_lowercase_not_preserved() {
    let pruner = TokenPruner::new(PruneConfig::new().with_preserve_entities(true));
    let imp = pruner.token_importance(&["world"], None);
    assert!(imp[0] < f32::MAX);
}

#[test]
fn test_entity_with_leading_punctuation() {
    // First alphabetic char is uppercase even with a leading quote.
    let pruner = TokenPruner::new(PruneConfig::new().with_preserve_entities(true));
    let imp = pruner.token_importance(&["\"Newton"], None);
    assert_eq!(imp[0], f32::MAX);
}

// ── original word order preserved ─────────────────────────────────────────────

#[test]
fn test_output_preserves_original_order() {
    let pruner = TokenPruner::new(
        PruneConfig::new()
            .with_min_tokens(1)
            .with_preserve_entities(false),
    );
    let text = "zebra apple mango banana cherry kiwi";
    let pruned = pruner.prune(text, None).expect("ok");
    let kept = order_of(&pruned.text);
    let full = order_of(text);
    assert!(
        is_subsequence(&kept, &full),
        "kept {kept:?} must be a subsequence of {full:?}"
    );
}

#[test]
fn test_order_preserved_with_query() {
    let pruner = TokenPruner::new(
        PruneConfig::new()
            .with_min_tokens(1)
            .with_preserve_entities(false),
    );
    let text = "one rare two common three signal four noise five";
    let pruned = pruner.prune(text, Some("signal rare")).expect("ok");
    let kept = order_of(&pruned.text);
    let full = order_of(text);
    assert!(is_subsequence(&kept, &full));
}

#[test]
fn test_order_preserved_entities() {
    let pruner = TokenPruner::new(
        PruneConfig::new()
            .with_target_ratio(0.5)
            .with_min_tokens(1)
            .with_preserve_entities(true),
    );
    let text = "Alpha bravo Charlie delta Echo foxtrot";
    let pruned = pruner.prune(text, None).expect("ok");
    let kept = order_of(&pruned.text);
    let full = order_of(text);
    assert!(is_subsequence(&kept, &full));
}

#[test]
fn test_kept_words_are_subset_of_input() {
    let pruner = TokenPruner::new(PruneConfig::default());
    let text = "the rapid prototyping framework accelerates iteration cycles dramatically";
    let pruned = pruner.prune(text, None).expect("ok");
    let full: std::collections::HashSet<String> = order_of(text).into_iter().collect();
    for w in order_of(&pruned.text) {
        assert!(full.contains(&w), "word {w} not from input");
    }
}

#[test]
fn test_surface_form_preserved() {
    // Punctuation in the surface form is retained verbatim when kept.
    let pruner = TokenPruner::new(PruneConfig::new().with_target_ratio(1.0));
    let text = "neutrino, photon. electron!";
    let pruned = pruner.prune(text, None).expect("ok");
    assert!(pruned.text.contains("neutrino,"));
    assert!(pruned.text.contains("photon."));
    assert!(pruned.text.contains("electron!"));
}

// ── compression_ratio == kept/original ────────────────────────────────────────

#[test]
fn test_compression_ratio_value() {
    let pruner = TokenPruner::new(PruneConfig::new().with_min_tokens(1));
    let text = "alpha beta gamma delta epsilon zeta eta theta iota kappa";
    let pruned = pruner.prune(text, None).expect("ok");
    let expected = pruned.kept_tokens as f32 / pruned.original_tokens as f32;
    assert!((pruned.compression_ratio - expected).abs() < 1e-6);
}

#[test]
fn test_compression_ratio_half() {
    let pruner = TokenPruner::new(PruneConfig::new().with_min_tokens(1));
    let text = "alpha beta gamma delta epsilon zeta eta theta iota kappa";
    let pruned = pruner.prune(text, None).expect("ok");
    assert!((pruned.compression_ratio - 0.5).abs() < 1e-6);
}

#[test]
fn test_compression_ratio_full() {
    let pruner = TokenPruner::new(PruneConfig::new().with_target_ratio(1.0));
    let text = "alpha beta gamma";
    let pruned = pruner.prune(text, None).expect("ok");
    assert!((pruned.compression_ratio - 1.0).abs() < 1e-6);
}

#[test]
fn test_compression_ratio_in_unit_range() {
    let pruner = TokenPruner::new(PruneConfig::default());
    let text =
        "the model compresses tokens efficiently using inverse frequency weighting heuristics";
    let pruned = pruner.prune(text, None).expect("ok");
    assert!(pruned.compression_ratio >= 0.0 && pruned.compression_ratio <= 1.0);
}

// ── query-conditioned pruning ─────────────────────────────────────────────────

#[test]
fn test_query_token_survives() {
    let pruner = TokenPruner::new(
        PruneConfig::new()
            .with_target_ratio(0.3)
            .with_min_tokens(1)
            .with_preserve_entities(false),
    );
    // All content words equally rare; query should pull "mitochondria" up.
    let text = "ribosome lysosome mitochondria vacuole cytoplasm nucleus membrane";
    let pruned = pruner.prune(text, Some("mitochondria")).expect("ok");
    assert!(
        order_of(&pruned.text).contains(&"mitochondria".to_string()),
        "query token must survive: {}",
        pruned.text
    );
}

#[test]
fn test_query_bonus_raises_importance() {
    let pruner = TokenPruner::new(PruneConfig::new().with_preserve_entities(false));
    let w = vec!["alpha", "beta", "gamma"];
    let without = pruner.token_importance(&w, None);
    let with = pruner.token_importance(&w, Some("beta"));
    assert!(with[1] > without[1], "query bonus should raise 'beta'");
    assert_eq!(with[0], without[0], "non-query tokens unchanged");
}

#[test]
fn test_query_overlap_multiple_tokens() {
    let pruner = TokenPruner::new(
        PruneConfig::new()
            .with_target_ratio(0.4)
            .with_min_tokens(1)
            .with_preserve_entities(false),
    );
    let text = "apple banana cherry date elderberry fig grape";
    let pruned = pruner.prune(text, Some("banana grape")).expect("ok");
    let kept = order_of(&pruned.text);
    assert!(kept.contains(&"banana".to_string()));
    assert!(kept.contains(&"grape".to_string()));
}

#[test]
fn test_query_case_insensitive() {
    let pruner = TokenPruner::new(PruneConfig::new().with_preserve_entities(false));
    let w = vec!["enzyme", "protein"];
    let imp = pruner.token_importance(&w, Some("ENZYME"));
    let base = pruner.token_importance(&w, None);
    assert!(imp[0] > base[0]);
}

#[test]
fn test_query_none_no_bonus() {
    let pruner = TokenPruner::new(PruneConfig::new().with_preserve_entities(false));
    let w = vec!["alpha", "beta"];
    let a = pruner.token_importance(&w, None);
    let b = pruner.token_importance(&w, Some(""));
    assert_eq!(a, b);
}

#[test]
fn test_non_query_stopword_below_content() {
    // Without a query, a stopword stays below a content word due to the penalty.
    let pruner = TokenPruner::new(PruneConfig::new().with_preserve_entities(false));
    let w = vec!["the", "neuroscience"];
    let imp = pruner.token_importance(&w, None);
    assert!(imp[1] > imp[0], "content word outranks penalized stopword");
}

#[test]
fn test_query_stopword_gets_boost() {
    // An explicit stopword in the query still receives the additive overlap bonus.
    let pruner = TokenPruner::new(PruneConfig::new().with_preserve_entities(false));
    let w = vec!["the", "neuroscience"];
    let base = pruner.token_importance(&w, None);
    let boosted = pruner.token_importance(&w, Some("the"));
    assert!(boosted[0] > base[0], "query overlap raises even a stopword");
}

// ── token_importance basics ───────────────────────────────────────────────────

#[test]
fn test_importance_length_matches_input() {
    let pruner = TokenPruner::new(PruneConfig::default());
    let w = vec!["one", "two", "three", "four"];
    assert_eq!(pruner.token_importance(&w, None).len(), 4);
}

#[test]
fn test_importance_empty_input() {
    let pruner = TokenPruner::new(PruneConfig::default());
    assert!(pruner.token_importance(&[], None).is_empty());
}

#[test]
fn test_pure_punctuation_zero_importance() {
    let pruner = TokenPruner::new(PruneConfig::new().with_preserve_entities(false));
    let imp = pruner.token_importance(&["---", "word"], None);
    assert_eq!(imp[0], 0.0);
    assert!(imp[1] > 0.0);
}

#[test]
fn test_importance_all_positive_for_content() {
    let pruner = TokenPruner::new(PruneConfig::new().with_preserve_entities(false));
    let imp = pruner.token_importance(&["alpha", "beta", "gamma"], None);
    assert!(imp.iter().all(|&v| v > 0.0));
}

// ── EmptyContext error ────────────────────────────────────────────────────────

#[test]
fn test_empty_context_errors() {
    let pruner = TokenPruner::new(PruneConfig::default());
    let err = pruner.prune("", None).expect_err("should fail");
    assert_eq!(err, ContextPruningError::EmptyContext);
}

#[test]
fn test_whitespace_only_context_errors() {
    let pruner = TokenPruner::new(PruneConfig::default());
    let err = pruner.prune("   \t\n  ", None).expect_err("should fail");
    assert!(matches!(err, ContextPruningError::EmptyContext));
}

#[test]
fn test_error_display() {
    let err = ContextPruningError::EmptyContext;
    assert_eq!(err.to_string(), "context must not be empty");
}

// ── determinism ───────────────────────────────────────────────────────────────

#[test]
fn test_prune_is_deterministic() {
    let pruner = TokenPruner::new(PruneConfig::default());
    let text = "the rapid prototyping framework accelerates iteration cycles dramatically today";
    let a = pruner.prune(text, Some("framework")).expect("ok");
    let b = pruner.prune(text, Some("framework")).expect("ok");
    assert_eq!(a, b);
}

#[test]
fn test_importance_is_deterministic() {
    let pruner = TokenPruner::new(PruneConfig::default());
    let w = vec!["alpha", "beta", "alpha", "gamma"];
    assert_eq!(
        pruner.token_importance(&w, Some("gamma")),
        pruner.token_importance(&w, Some("gamma"))
    );
}

#[test]
fn test_tie_break_prefers_earlier_index() {
    // Two equally-scored content words; with a tight budget the earlier one wins.
    let pruner = TokenPruner::new(
        PruneConfig::new()
            .with_target_ratio(0.5)
            .with_min_tokens(1)
            .with_preserve_entities(false),
    );
    // Four equally-rare words → keep 2; ties resolved by original order.
    let text = "alpha beta gamma delta";
    let pruned = pruner.prune(text, None).expect("ok");
    let kept = order_of(&pruned.text);
    assert_eq!(kept, vec!["alpha".to_string(), "beta".to_string()]);
}

// ── prune_documents ───────────────────────────────────────────────────────────

#[test]
fn test_prune_documents_basic() {
    let pruner = TokenPruner::new(PruneConfig::new().with_min_tokens(1));
    let docs = vec![
        Document::new("the quantum entanglement phenomenon links distant particles instantly"),
        Document::new("classical mechanics describes macroscopic motion under gravity precisely"),
    ];
    let out = pruner.prune_documents(&docs, None).expect("ok");
    assert_eq!(out.len(), 2);
    assert!(out.iter().all(|p| p.kept_tokens > 0));
}

#[test]
fn test_prune_documents_empty_slice() {
    let pruner = TokenPruner::new(PruneConfig::default());
    let out = pruner.prune_documents(&[], None).expect("ok");
    assert!(out.is_empty());
}

#[test]
fn test_prune_documents_propagates_empty_error() {
    let pruner = TokenPruner::new(PruneConfig::default());
    let docs = vec![Document::new("valid content here words"), Document::new("")];
    let err = pruner
        .prune_documents(&docs, None)
        .expect_err("should fail");
    assert_eq!(err, ContextPruningError::EmptyContext);
}

#[test]
fn test_prune_documents_with_query() {
    let pruner = TokenPruner::new(
        PruneConfig::new()
            .with_target_ratio(0.3)
            .with_min_tokens(1)
            .with_preserve_entities(false),
    );
    let docs = vec![Document::new(
        "ribosome lysosome mitochondria vacuole cytoplasm nucleus membrane organelle",
    )];
    let out = pruner
        .prune_documents(&docs, Some("mitochondria"))
        .expect("ok");
    assert!(order_of(&out[0].text).contains(&"mitochondria".to_string()));
}

// ── PrunedContext helpers ─────────────────────────────────────────────────────

#[test]
fn test_pruned_context_is_empty_false() {
    let pruner = TokenPruner::new(PruneConfig::default());
    let pruned = pruner.prune("alpha beta gamma", None).expect("ok");
    assert!(!pruned.is_empty());
}

#[test]
fn test_pruned_context_fields_consistent() {
    let pruner = TokenPruner::new(PruneConfig::new().with_min_tokens(1));
    let text = "alpha beta gamma delta epsilon zeta eta theta";
    let pruned: PrunedContext = pruner.prune(text, None).expect("ok");
    assert_eq!(pruned.kept_tokens, order_of(&pruned.text).len());
    assert!(pruned.kept_tokens <= pruned.original_tokens);
}

#[test]
fn test_pruned_context_clone_eq() {
    let pruner = TokenPruner::new(PruneConfig::default());
    let pruned = pruner.prune("alpha beta gamma", None).expect("ok");
    assert_eq!(pruned.clone(), pruned);
}

// ── default pruner ────────────────────────────────────────────────────────────

#[test]
fn test_default_pruner_uses_default_config() {
    let pruner = TokenPruner::default();
    assert_eq!(pruner.config, PruneConfig::default());
}

#[test]
fn test_realistic_paragraph_prunes_and_keeps_entities() {
    let pruner = TokenPruner::new(PruneConfig::new().with_target_ratio(0.5));
    let text = "Albert Einstein developed the theory of relativity which fundamentally \
                changed our understanding of space and time in modern physics";
    let pruned = pruner.prune(text, Some("relativity")).expect("ok");
    let kept = order_of(&pruned.text);
    // Entities preserved.
    assert!(kept.contains(&"Albert".to_string()));
    assert!(kept.contains(&"Einstein".to_string()));
    // Query token preserved.
    assert!(kept.contains(&"relativity".to_string()));
    // Roughly half kept.
    assert!(pruned.kept_tokens <= pruned.original_tokens);
    assert!(pruned.kept_tokens >= pruned.original_tokens / 3);
    // Order preserved.
    assert!(is_subsequence(&kept, &order_of(text)));
}
