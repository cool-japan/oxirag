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

use crate::dragin::engine::DraginEngine;
use crate::dragin::types::{
    DraginConfig, DraginError, MockRetriever, MockUncertaintyGenerator, Retriever, TokenInfo,
    UncertaintyGenerator,
};

// ── Test helpers ──────────────────────────────────────────────────────────────

/// Build a `TokenInfo` segment from `(token, confidence)` pairs.
fn seg(pairs: &[(&str, f32)]) -> Vec<TokenInfo> {
    pairs.iter().map(|(t, c)| TokenInfo::new(*t, *c)).collect()
}

/// An engine with default config.
fn default_engine() -> DraginEngine {
    DraginEngine::new(DraginConfig::default())
}

// ── DraginConfig: defaults ────────────────────────────────────────────────────

#[test]
fn config_default_uncertainty_threshold() {
    assert_eq!(DraginConfig::default().uncertainty_threshold, 0.5);
}

#[test]
fn config_default_max_retrievals() {
    assert_eq!(DraginConfig::default().max_retrievals, 5);
}

#[test]
fn config_default_query_window() {
    assert_eq!(DraginConfig::default().query_window, 10);
}

#[test]
fn config_new_equals_default() {
    assert_eq!(DraginConfig::new(), DraginConfig::default());
}

// ── DraginConfig: builders ────────────────────────────────────────────────────

#[test]
fn config_with_uncertainty_threshold() {
    let cfg = DraginConfig::new().with_uncertainty_threshold(0.3);
    assert_eq!(cfg.uncertainty_threshold, 0.3);
}

#[test]
fn config_with_max_retrievals() {
    let cfg = DraginConfig::new().with_max_retrievals(2);
    assert_eq!(cfg.max_retrievals, 2);
}

#[test]
fn config_with_query_window() {
    let cfg = DraginConfig::new().with_query_window(4);
    assert_eq!(cfg.query_window, 4);
}

#[test]
fn config_builders_chain() {
    let cfg = DraginConfig::new()
        .with_uncertainty_threshold(0.25)
        .with_max_retrievals(3)
        .with_query_window(7);
    assert_eq!(cfg.uncertainty_threshold, 0.25);
    assert_eq!(cfg.max_retrievals, 3);
    assert_eq!(cfg.query_window, 7);
}

#[test]
fn config_builders_do_not_mutate_others() {
    let cfg = DraginConfig::new().with_query_window(1);
    // Untouched fields keep their defaults.
    assert_eq!(cfg.uncertainty_threshold, 0.5);
    assert_eq!(cfg.max_retrievals, 5);
}

// ── is_content_token ──────────────────────────────────────────────────────────

#[test]
fn content_token_stopword_the_is_not_content() {
    assert!(!default_engine().is_content_token("the"));
}

#[test]
fn content_token_stopword_and_is_not_content() {
    assert!(!default_engine().is_content_token("and"));
}

#[test]
fn content_token_stopword_with_is_not_content() {
    assert!(!default_engine().is_content_token("with"));
}

#[test]
fn content_token_stopword_uppercase_is_not_content() {
    // Membership is case-insensitive.
    assert!(!default_engine().is_content_token("The"));
    assert!(!default_engine().is_content_token("WITH"));
}

#[test]
fn content_token_long_word_is_content() {
    assert!(default_engine().is_content_token("Inception"));
}

#[test]
fn content_token_director_is_content() {
    assert!(default_engine().is_content_token("director"));
}

#[test]
fn content_token_short_word_is_not_content() {
    // Fewer than three characters → not content.
    assert!(!default_engine().is_content_token("of"));
    assert!(!default_engine().is_content_token("a"));
    assert!(!default_engine().is_content_token("hi"));
}

#[test]
fn content_token_three_char_non_stopword_is_content() {
    // Exactly three chars and not a stopword.
    assert!(default_engine().is_content_token("cat"));
}

#[test]
fn content_token_strips_surrounding_punctuation() {
    // Punctuation is trimmed before the length / stopword check.
    assert!(default_engine().is_content_token("Nolan."));
    assert!(default_engine().is_content_token("(Inception)"));
    assert!(!default_engine().is_content_token("the,"));
}

#[test]
fn content_token_empty_is_not_content() {
    assert!(!default_engine().is_content_token(""));
    assert!(!default_engine().is_content_token("..."));
}

// ── tokenize ──────────────────────────────────────────────────────────────────

#[test]
fn tokenize_splits_on_non_alphanumeric() {
    let toks = DraginEngine::tokenize("Hello, world! 42");
    assert_eq!(toks, vec!["Hello", "world", "42"]);
}

#[test]
fn tokenize_drops_empty_fragments() {
    let toks = DraginEngine::tokenize("  --foo--bar-- ");
    assert_eq!(toks, vec!["foo", "bar"]);
}

#[test]
fn tokenize_empty_string_is_empty() {
    assert!(DraginEngine::tokenize("").is_empty());
    assert!(DraginEngine::tokenize("   ").is_empty());
}

// ── QFS: formulate_query ──────────────────────────────────────────────────────

#[test]
fn qfs_keeps_only_content_tokens() {
    let engine = default_engine();
    let toks = vec!["the", "Inception", "was", "directed", "by", "Nolan"];
    let q = engine.formulate_query(&toks);
    // Stopwords ("the", "was", "by") and the short "the" are dropped.
    assert_eq!(q, "Inception directed Nolan");
}

#[test]
fn qfs_respects_query_window() {
    // window = 3 → only the last three tokens are considered.
    let engine = DraginEngine::new(DraginConfig::new().with_query_window(3));
    let toks = vec!["Alpha", "Beta", "Gamma", "Delta", "Epsilon"];
    let q = engine.formulate_query(&toks);
    assert_eq!(q, "Gamma Delta Epsilon");
}

#[test]
fn qfs_window_with_stopwords_inside() {
    let engine = DraginEngine::new(DraginConfig::new().with_query_window(4));
    let toks = vec!["history", "of", "the", "Roman", "Empire"];
    // Last 4 = ["of","the","Roman","Empire"]; content = ["Roman","Empire"].
    let q = engine.formulate_query(&toks);
    assert_eq!(q, "Roman Empire");
}

#[test]
fn qfs_all_stopwords_falls_back_to_window_join() {
    let engine = DraginEngine::new(DraginConfig::new().with_query_window(3));
    let toks = vec!["the", "and", "for"];
    // No content tokens → join the window verbatim.
    let q = engine.formulate_query(&toks);
    assert_eq!(q, "the and for");
}

#[test]
fn qfs_empty_input_is_empty_query() {
    let engine = default_engine();
    let toks: Vec<&str> = Vec::new();
    assert_eq!(engine.formulate_query(&toks), "");
}

#[test]
fn qfs_single_content_token() {
    let engine = default_engine();
    let toks = vec!["Mitochondria"];
    assert_eq!(engine.formulate_query(&toks), "Mitochondria");
}

#[test]
fn qfs_window_larger_than_input_uses_all() {
    let engine = DraginEngine::new(DraginConfig::new().with_query_window(100));
    let toks = vec!["Quantum", "entanglement"];
    assert_eq!(engine.formulate_query(&toks), "Quantum entanglement");
}

// ── Mock generator behaviour ──────────────────────────────────────────────────

#[test]
fn mock_generator_consumes_segments_in_order() {
    let g = MockUncertaintyGenerator::new(vec![seg(&[("first", 0.9)]), seg(&[("second", 0.9)])]);
    assert_eq!(g.remaining(), 2);
    assert_eq!(g.generate("ctx")[0].token, "first");
    assert_eq!(g.generate("ctx")[0].token, "second");
    assert_eq!(g.remaining(), 0);
}

#[test]
fn mock_generator_empty_when_exhausted() {
    let g = MockUncertaintyGenerator::new(vec![seg(&[("only", 0.9)])]);
    let _ = g.generate("ctx");
    assert!(g.generate("ctx").is_empty());
}

#[test]
fn mock_generator_default_is_empty() {
    let g = MockUncertaintyGenerator::default();
    assert!(g.generate("ctx").is_empty());
}

// ── Mock retriever behaviour ──────────────────────────────────────────────────

#[test]
fn mock_retriever_echo_returns_query() {
    let r = MockRetriever::echo();
    assert_eq!(r.retrieve("hello"), vec!["hello".to_string()]);
}

#[test]
fn mock_retriever_matches_substring() {
    let r = MockRetriever::new(vec![(
        "director".to_string(),
        vec!["doc-a".to_string(), "doc-b".to_string()],
    )]);
    assert_eq!(
        r.retrieve("who is the director here"),
        vec!["doc-a".to_string(), "doc-b".to_string()]
    );
}

#[test]
fn mock_retriever_no_match_echoes() {
    let r = MockRetriever::new(vec![("zzz".to_string(), vec!["never".to_string()])]);
    assert_eq!(r.retrieve("the query"), vec!["the query".to_string()]);
}

#[test]
fn mock_retriever_first_matching_mapping_wins() {
    let r = MockRetriever::new(vec![
        ("cat".to_string(), vec!["feline".to_string()]),
        ("cataract".to_string(), vec!["eye".to_string()]),
    ]);
    // "cat" is contained first → its passages are returned.
    assert_eq!(r.retrieve("cataract surgery"), vec!["feline".to_string()]);
}

// ── RIND: triggers on a low-confidence content token ──────────────────────────

#[test]
fn rind_triggers_on_low_confidence_content_token() {
    let generator = MockUncertaintyGenerator::new(vec![
        seg(&[("Inception", 0.9), ("directed", 0.9), ("Nolan", 0.2)]),
        // Second segment all confident → loop stops after handling the trigger.
        seg(&[("confirmed", 0.95)]),
    ]);
    let retriever = MockRetriever::echo();
    let trace = default_engine()
        .run("question", &generator, &retriever)
        .unwrap();
    assert_eq!(trace.num_retrievals, 1);
    assert_eq!(trace.triggers[0].trigger_token, "Nolan");
}

#[test]
fn rind_trigger_position_recorded() {
    let generator = MockUncertaintyGenerator::new(vec![seg(&[
        ("alpha", 0.9),
        ("beta", 0.9),
        ("Heisenberg", 0.1),
    ])]);
    let retriever = MockRetriever::echo();
    let trace = default_engine().run("q", &generator, &retriever).unwrap();
    // Trigger token is at index 2 within its segment.
    assert_eq!(trace.triggers[0].position, 2);
    assert_eq!(trace.triggers[0].trigger_token, "Heisenberg");
}

#[test]
fn rind_first_uncertain_content_token_wins() {
    let generator = MockUncertaintyGenerator::new(vec![seg(&[("Mercury", 0.1), ("Venus", 0.1)])]);
    let retriever = MockRetriever::echo();
    let trace = default_engine().run("q", &generator, &retriever).unwrap();
    // First uncertain content token ("Mercury") triggers; only one retrieval
    // per segment.
    assert_eq!(trace.triggers[0].trigger_token, "Mercury");
    assert_eq!(trace.triggers[0].position, 0);
}

#[test]
fn rind_confidence_at_threshold_does_not_trigger() {
    // confidence == threshold is NOT < threshold, so no trigger.
    let generator = MockUncertaintyGenerator::new(vec![seg(&[("Galaxy", 0.5)])]);
    let retriever = MockRetriever::echo();
    let trace = default_engine().run("q", &generator, &retriever).unwrap();
    assert_eq!(trace.num_retrievals, 0);
}

// ── RIND: does NOT trigger on a low-confidence stopword ───────────────────────

#[test]
fn rind_ignores_low_confidence_stopword() {
    let generator = MockUncertaintyGenerator::new(vec![seg(&[
        ("Newton", 0.9),
        ("the", 0.01), // low confidence but a stopword → no trigger
        ("law", 0.9),
    ])]);
    let retriever = MockRetriever::echo();
    let trace = default_engine().run("q", &generator, &retriever).unwrap();
    assert_eq!(trace.num_retrievals, 0);
}

#[test]
fn rind_ignores_low_confidence_short_token() {
    // "of" is shorter than three chars → never a trigger even at 0.0.
    let generator =
        MockUncertaintyGenerator::new(vec![seg(&[("speed", 0.9), ("of", 0.0), ("light", 0.9)])]);
    let retriever = MockRetriever::echo();
    let trace = default_engine().run("q", &generator, &retriever).unwrap();
    assert_eq!(trace.num_retrievals, 0);
}

#[test]
fn rind_skips_uncertain_stopword_then_triggers_on_content() {
    let generator = MockUncertaintyGenerator::new(vec![
        seg(&[
            ("and", 0.01),        // stopword → skipped
            ("Schrodinger", 0.2), // content + uncertain → trigger
        ]),
        seg(&[("end", 0.99)]),
    ]);
    let retriever = MockRetriever::echo();
    let trace = default_engine().run("q", &generator, &retriever).unwrap();
    assert_eq!(trace.num_retrievals, 1);
    assert_eq!(trace.triggers[0].trigger_token, "Schrodinger");
    assert_eq!(trace.triggers[0].position, 1);
}

// ── No-trigger case (all confident) → 0 retrievals ────────────────────────────

#[test]
fn no_trigger_when_all_confident() {
    let generator = MockUncertaintyGenerator::new(vec![seg(&[
        ("Photosynthesis", 0.95),
        ("converts", 0.9),
        ("sunlight", 0.92),
    ])]);
    let retriever = MockRetriever::echo();
    let trace = default_engine().run("q", &generator, &retriever).unwrap();
    assert_eq!(trace.num_retrievals, 0);
    assert!(trace.triggers.is_empty());
}

#[test]
fn no_trigger_still_accumulates_generated_text() {
    let generator = MockUncertaintyGenerator::new(vec![seg(&[
        ("Mitochondria", 0.95),
        ("produce", 0.9),
        ("ATP", 0.92),
    ])]);
    let retriever = MockRetriever::echo();
    let trace = default_engine().run("q", &generator, &retriever).unwrap();
    assert_eq!(trace.generated, "Mitochondria produce ATP");
}

#[test]
fn empty_first_segment_yields_empty_run() {
    let generator = MockUncertaintyGenerator::new(vec![Vec::new()]);
    let retriever = MockRetriever::echo();
    let trace = default_engine().run("q", &generator, &retriever).unwrap();
    assert_eq!(trace.num_retrievals, 0);
    assert_eq!(trace.generated, "");
}

// ── QFS within run: query built from salient tokens ───────────────────────────

#[test]
fn run_forms_query_from_salient_generated_tokens() {
    let generator = MockUncertaintyGenerator::new(vec![
        seg(&[
            ("The", 0.9),
            ("Eiffel", 0.9),
            ("Tower", 0.9),
            ("stands", 0.9),
            ("in", 0.9),
            ("Paris", 0.2), // trigger
        ]),
        seg(&[("done", 0.99)]),
    ]);
    let retriever = MockRetriever::echo();
    let trace = default_engine()
        .run("Where is the Eiffel Tower?", &generator, &retriever)
        .unwrap();
    // QFS over the generated tokens keeps content tokens only.
    assert_eq!(trace.triggers[0].formed_query, "Eiffel Tower stands Paris");
    // Echo retriever returns the formed query.
    assert_eq!(
        trace.triggers[0].retrieved,
        vec!["Eiffel Tower stands Paris".to_string()]
    );
}

#[test]
fn run_query_respects_window_over_generated() {
    let engine = DraginEngine::new(DraginConfig::new().with_query_window(2));
    let generator = MockUncertaintyGenerator::new(vec![seg(&[
        ("Alpha", 0.9),
        ("Beta", 0.9),
        ("Gamma", 0.2), // trigger; window = last 2 generated tokens
    ])]);
    let retriever = MockRetriever::echo();
    let trace = engine.run("q", &generator, &retriever).unwrap();
    // Generated tokens so far: [Alpha, Beta, Gamma]; last 2 = [Beta, Gamma].
    assert_eq!(trace.triggers[0].formed_query, "Beta Gamma");
}

// ── Trace records all fields ──────────────────────────────────────────────────

#[test]
fn trace_records_retrieved_passages() {
    let generator =
        MockUncertaintyGenerator::new(vec![seg(&[("Composer", 0.2)]), seg(&[("ok", 0.99)])]);
    let retriever = MockRetriever::new(vec![(
        "Composer".to_string(),
        vec!["passage-1".to_string(), "passage-2".to_string()],
    )]);
    let trace = default_engine().run("q", &generator, &retriever).unwrap();
    assert_eq!(
        trace.triggers[0].retrieved,
        vec!["passage-1".to_string(), "passage-2".to_string()]
    );
}

#[test]
fn trace_num_retrievals_matches_triggers_len() {
    let generator = MockUncertaintyGenerator::new(vec![
        seg(&[("Alpha", 0.2)]),
        seg(&[("Bravo", 0.2)]),
        seg(&[("done", 0.99)]),
    ]);
    let retriever = MockRetriever::echo();
    let trace = default_engine().run("q", &generator, &retriever).unwrap();
    assert_eq!(trace.num_retrievals, trace.triggers.len());
    assert_eq!(trace.num_retrievals, 2);
}

#[test]
fn trace_generated_accumulates_across_segments() {
    let generator = MockUncertaintyGenerator::new(vec![
        seg(&[("Quantum", 0.2)]),
        seg(&[("mechanics", 0.95), ("rules", 0.95)]),
    ]);
    let retriever = MockRetriever::echo();
    let trace = default_engine().run("q", &generator, &retriever).unwrap();
    // Both segments are appended to `generated`.
    assert_eq!(trace.generated, "Quantum mechanics rules");
}

#[test]
fn trace_records_query_per_trigger() {
    let generator = MockUncertaintyGenerator::new(vec![
        seg(&[("Newton", 0.2)]),
        seg(&[("Einstein", 0.2)]),
        seg(&[("fin", 0.99)]),
    ]);
    let retriever = MockRetriever::echo();
    let trace = default_engine().run("q", &generator, &retriever).unwrap();
    // First query: only "Newton" generated so far.
    assert_eq!(trace.triggers[0].formed_query, "Newton");
    // Second query: "Newton Einstein" generated so far.
    assert_eq!(trace.triggers[1].formed_query, "Newton Einstein");
}

// ── max_retrievals cap ────────────────────────────────────────────────────────

#[test]
fn max_retrievals_cap_halts_loop() {
    let engine = DraginEngine::new(DraginConfig::new().with_max_retrievals(2));
    let generator = MockUncertaintyGenerator::new(vec![
        seg(&[("Alpha", 0.2)]),
        seg(&[("Bravo", 0.2)]),
        seg(&[("Charlie", 0.2)]), // would trigger a 3rd, but cap is 2
        seg(&[("Delta", 0.2)]),
    ]);
    let retriever = MockRetriever::echo();
    let trace = engine.run("q", &generator, &retriever).unwrap();
    assert_eq!(trace.num_retrievals, 2);
}

#[test]
fn max_retrievals_zero_never_retrieves() {
    let engine = DraginEngine::new(DraginConfig::new().with_max_retrievals(0));
    let generator = MockUncertaintyGenerator::new(vec![seg(&[("Uncertain", 0.1)])]);
    let retriever = MockRetriever::echo();
    let trace = engine.run("q", &generator, &retriever).unwrap();
    assert_eq!(trace.num_retrievals, 0);
}

#[test]
fn max_retrievals_cap_stops_before_consuming_all_segments() {
    let engine = DraginEngine::new(DraginConfig::new().with_max_retrievals(1));
    let generator = MockUncertaintyGenerator::new(vec![
        seg(&[("First", 0.2)]),  // triggers; cap reached after this
        seg(&[("Second", 0.2)]), // generated, but cap halts before retrieval
        seg(&[("Third", 0.2)]),  // never generated
    ]);
    let retriever = MockRetriever::echo();
    let trace = engine.run("q", &generator, &retriever).unwrap();
    assert_eq!(trace.num_retrievals, 1);
    // The loop breaks on the second segment's trigger (cap reached); the third
    // segment is never consumed.
    assert_eq!(generator.remaining(), 1);
}

// ── Determinism ───────────────────────────────────────────────────────────────

#[test]
fn run_is_deterministic_across_repeated_runs() {
    let build = || {
        MockUncertaintyGenerator::new(vec![
            seg(&[("Relativity", 0.2)]),
            seg(&[("explains", 0.9), ("gravity", 0.2)]),
            seg(&[("end", 0.99)]),
        ])
    };
    let retriever = MockRetriever::echo();
    let engine = default_engine();

    let t1 = engine.run("q", &build(), &retriever).unwrap();
    let t2 = engine.run("q", &build(), &retriever).unwrap();
    assert_eq!(t1, t2);
}

#[test]
fn trace_equality_is_structural() {
    let g1 = MockUncertaintyGenerator::new(vec![seg(&[("Atom", 0.2)]), seg(&[("ok", 0.99)])]);
    let g2 = MockUncertaintyGenerator::new(vec![seg(&[("Atom", 0.2)]), seg(&[("ok", 0.99)])]);
    let r = MockRetriever::echo();
    let engine = default_engine();
    assert_eq!(
        engine.run("same", &g1, &r).unwrap(),
        engine.run("same", &g2, &r).unwrap()
    );
}

// ── EmptyQuery error ──────────────────────────────────────────────────────────

#[test]
fn empty_query_errors() {
    let generator = MockUncertaintyGenerator::default();
    let retriever = MockRetriever::echo();
    let err = default_engine()
        .run("", &generator, &retriever)
        .unwrap_err();
    assert!(matches!(err, DraginError::EmptyQuery));
}

#[test]
fn whitespace_only_query_errors() {
    let generator = MockUncertaintyGenerator::default();
    let retriever = MockRetriever::echo();
    let err = default_engine()
        .run("   \t\n ", &generator, &retriever)
        .unwrap_err();
    assert!(matches!(err, DraginError::EmptyQuery));
}

#[test]
fn empty_query_error_displays_message() {
    let err = DraginError::EmptyQuery;
    assert_eq!(err.to_string(), "query must not be empty");
}

#[test]
fn non_empty_query_does_not_error() {
    let generator = MockUncertaintyGenerator::new(vec![seg(&[("ok", 0.99)])]);
    let retriever = MockRetriever::echo();
    assert!(default_engine().run("real", &generator, &retriever).is_ok());
}

// ── Engine construction ───────────────────────────────────────────────────────

#[test]
fn engine_default_uses_default_config() {
    let engine = DraginEngine::default();
    assert_eq!(engine.config, DraginConfig::default());
}

#[test]
fn engine_new_stores_config() {
    let cfg = DraginConfig::new().with_max_retrievals(9);
    let engine = DraginEngine::new(cfg.clone());
    assert_eq!(engine.config, cfg);
}

// ── TokenInfo construction ────────────────────────────────────────────────────

#[test]
fn token_info_new_sets_fields() {
    let ti = TokenInfo::new("word", 0.42);
    assert_eq!(ti.token, "word");
    assert_eq!(ti.confidence, 0.42);
}

// ── Integration: retrieved context is folded back ─────────────────────────────

#[test]
fn retrieved_context_influences_continuation() {
    // The mock generator ignores context, but we still verify the loop drives
    // through multiple retrievals and stops cleanly on a confident tail.
    let generator = MockUncertaintyGenerator::new(vec![
        seg(&[("Capital", 0.2)]),
        seg(&[("Population", 0.2)]),
        seg(&[("Summary", 0.95), ("complete", 0.95)]),
    ]);
    // QFS accumulates all content tokens generated so far, so by the second
    // trigger the formed query is "Capital Population". The retriever checks
    // "Population" first so the second event resolves to the population passage.
    let retriever = MockRetriever::new(vec![
        (
            "Population".to_string(),
            vec!["The population is large.".to_string()],
        ),
        (
            "Capital".to_string(),
            vec!["Paris is the capital.".to_string()],
        ),
    ]);
    let trace = default_engine()
        .run("Tell me about France", &generator, &retriever)
        .unwrap();
    assert_eq!(trace.num_retrievals, 2);
    // First trigger: only "Capital" generated → matches the capital passage.
    assert_eq!(trace.triggers[0].formed_query, "Capital");
    assert_eq!(
        trace.triggers[0].retrieved,
        vec!["Paris is the capital.".to_string()]
    );
    // Second trigger: "Capital Population" generated → "Population" matched first.
    assert_eq!(trace.triggers[1].formed_query, "Capital Population");
    assert_eq!(
        trace.triggers[1].retrieved,
        vec!["The population is large.".to_string()]
    );
    assert!(trace.generated.ends_with("Summary complete"));
}
