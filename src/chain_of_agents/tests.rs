#![allow(clippy::float_cmp, clippy::similar_names, clippy::too_many_lines)]
//! Tests for the `chain_of_agents` module.

use super::engine::{CoaEngine, split_into_chunks, split_sentences};
use super::types::CoaConfig;
use super::types::CoaError;
use super::types::CoaEvidence;
use super::unit::CoaCommunicationUnit;
use super::worker::{CoaLexicalManager, CoaLexicalWorker, CoaManager, CoaWorker};

// ── Test fixtures ────────────────────────────────────────────────────────────

/// Two chunks whose only query-relevant sentence ties in relevance score,
/// so which one survives a tight evidence budget depends purely on chunk
/// order — used by both the "manager only sees the final unit" test and the
/// "order matters" test.
fn capitals_chunks() -> Vec<String> {
    vec![
        "Rivertown is the capital of Elmsworth.".to_string(),
        "Mosshaven is the capital of Duskvale.".to_string(),
    ]
}

const CAPITALS_QUERY: &str = "Which cities are capitals of nearby regions?";

/// A 9-chunk fixture where the answer requires combining a fact from the
/// *first* chunk (a rename) with a fact from the *last* chunk (a launch
/// date stated only about the new name) — the headline forward-flow proof.
/// Every chunk in between is deliberately irrelevant to the query.
fn kepler_chunks() -> Vec<String> {
    vec![
        "The Kepler probe was renamed Artemis.".to_string(),
        "The committee reviewed quarterly budget reports.".to_string(),
        "A new species of beetle was discovered in the rainforest.".to_string(),
        "Local schools announced a longer summer break this year.".to_string(),
        "The museum unveiled an exhibit on ancient pottery.".to_string(),
        "Farmers reported a strong harvest across the valley.".to_string(),
        "The city council approved funding for a new bridge.".to_string(),
        "Researchers published a study on coral reef bleaching.".to_string(),
        "Artemis launched in 2031 after years of preparation.".to_string(),
    ]
}

const KEPLER_QUERY: &str = "When did the Kepler probe launch?";

// ── Headline: forward flow, and the ablation that proves it's load-bearing ──

#[test]
fn test_forward_flow_answers_a_question_spanning_first_and_last_chunk() {
    let engine = CoaEngine::new(CoaConfig::default());
    let chunks = kepler_chunks();

    let trace = engine
        .run_chunks(KEPLER_QUERY, &chunks, &CoaLexicalWorker, &CoaLexicalManager)
        .expect("well-formed chain-of-agents run");

    assert_eq!(trace.steps.len(), 9);
    assert!(
        trace.answer.contains("Kepler") && trace.answer.contains("2031"),
        "expected the chain to combine the rename (chunk 0) with the launch date \
         (chunk 8) into one answer, got: {:?}",
        trace.answer
    );
}

#[test]
fn test_zeroed_communication_unit_ablation_fails_to_answer() {
    // Same query, same chunks, same deterministic worker/manager as the
    // success case above — but here every worker is handed a *fresh, empty*
    // communication unit instead of the previous worker's output, exactly
    // as if the chain had no memory at all. This is the load-bearing proof:
    // if the ablation answered just as well as the real chain, the
    // communication unit would be decorative, not the mechanism.
    let chunks = kepler_chunks();
    let worker = CoaLexicalWorker;
    let manager = CoaLexicalManager;

    let mut last_unit = CoaCommunicationUnit::new(6, 4);
    for chunk in &chunks {
        let empty_incoming = CoaCommunicationUnit::new(6, 4);
        last_unit = worker
            .process(KEPLER_QUERY, chunk, &empty_incoming)
            .expect("worker should not fail on a single isolated chunk");
    }
    let ablated_answer = manager
        .synthesize(KEPLER_QUERY, &last_unit)
        .expect("manager should not fail");

    // The last chunk alone ("Artemis launched in 2031...") never mentions
    // Kepler, and with no communication unit carrying the rename forward,
    // the manager has no way to learn that Artemis *is* the Kepler probe.
    assert!(
        !ablated_answer.contains("Kepler"),
        "the zeroed-unit ablation should never learn the Kepler/Artemis alias, \
         got: {ablated_answer:?}"
    );

    // Cross-check against the real, threaded run: it *does* combine both
    // facts, unlike the ablation.
    let engine = CoaEngine::new(CoaConfig::default());
    let real_trace = engine
        .run_chunks(KEPLER_QUERY, &chunks, &worker, &manager)
        .expect("well-formed chain-of-agents run");
    assert!(real_trace.answer.contains("Kepler") && real_trace.answer.contains("2031"));
    assert_ne!(real_trace.answer, ablated_answer);
}

// ── The manager never sees a raw chunk, only the final unit ─────────────────

#[test]
fn test_manager_only_sees_evidence_that_survived_in_the_final_unit() {
    // Both sentences tie in relevance (see `capitals_chunks`); with a
    // budget of exactly 1, only the more recently discovered one survives.
    let engine = CoaEngine::new(CoaConfig::new().with_evidence_budget(1));
    let chunks = capitals_chunks();

    let trace = engine
        .run_chunks(
            CAPITALS_QUERY,
            &chunks,
            &CoaLexicalWorker,
            &CoaLexicalManager,
        )
        .expect("well-formed chain-of-agents run");

    // "Rivertown"/"Elmsworth" appeared in a chunk, but lost the eviction
    // fight and never made it into the final unit.
    assert_eq!(trace.final_unit.evidence.len(), 1);
    assert!(trace.final_unit.evidence[0].text.contains("Mosshaven"));
    assert!(!trace.final_unit.evidence[0].text.contains("Rivertown"));

    // The manager, having only ever seen `final_unit`, cannot mention a
    // fact that was dropped from it before the manager ever ran.
    assert!(trace.answer.contains("Mosshaven"));
    assert!(
        !trace.answer.contains("Rivertown") && !trace.answer.contains("Elmsworth"),
        "the manager must not be able to produce text derived from evidence \
         that was evicted from the communication unit, got: {:?}",
        trace.answer
    );
}

// ── Order matters: this is sequential, not map-reduce ────────────────────────

#[test]
fn test_reversing_chunk_order_changes_the_result() {
    let engine = CoaEngine::new(CoaConfig::new().with_evidence_budget(1));
    let forward_chunks = capitals_chunks();
    let mut reversed_chunks = forward_chunks.clone();
    reversed_chunks.reverse();

    let forward = engine
        .run_chunks(
            CAPITALS_QUERY,
            &forward_chunks,
            &CoaLexicalWorker,
            &CoaLexicalManager,
        )
        .expect("well-formed chain-of-agents run");
    let reversed = engine
        .run_chunks(
            CAPITALS_QUERY,
            &reversed_chunks,
            &CoaLexicalWorker,
            &CoaLexicalManager,
        )
        .expect("well-formed chain-of-agents run");

    // Same two facts, same tie in relevance score — but the tie-break rule
    // favors whichever chunk was seen *most recently*, so which one
    // survives the budget of 1 depends entirely on chunk order. A
    // map-reduce style aggregation (independent per-item scoring, order
    // irrelevant) could never produce this difference.
    assert_ne!(forward.final_unit.evidence, reversed.final_unit.evidence);
    assert_ne!(forward.answer, reversed.answer);
    assert!(forward.final_unit.evidence[0].text.contains("Mosshaven"));
    assert!(reversed.final_unit.evidence[0].text.contains("Rivertown"));
}

// ── Bounded state: no linear growth with chunk count ─────────────────────────

const WIDGET_QUERY: &str = "What is widget performance?";

fn widget_chunks(count: usize) -> Vec<String> {
    (0..count)
        .map(|i| format!("Widget variant {i} demonstrated strong performance results today."))
        .collect()
}

#[test]
fn test_evidence_stays_bounded_regardless_of_chunk_count() {
    let engine = CoaEngine::new(
        CoaConfig::new()
            .with_evidence_budget(5)
            .with_open_question_budget(3),
    );

    let small = engine
        .run_chunks(
            WIDGET_QUERY,
            &widget_chunks(8),
            &CoaLexicalWorker,
            &CoaLexicalManager,
        )
        .expect("well-formed chain-of-agents run");
    let large = engine
        .run_chunks(
            WIDGET_QUERY,
            &widget_chunks(80),
            &CoaLexicalWorker,
            &CoaLexicalManager,
        )
        .expect("well-formed chain-of-agents run");

    // Both runs have far more relevant chunks than the budget allows,
    // and both saturate to *exactly* the budget rather than scaling with
    // chunk count — this is the failure mode bounded merge exists to
    // prevent.
    assert_eq!(small.final_unit.evidence.len(), 5);
    assert_eq!(large.final_unit.evidence.len(), 5);
    assert!(small.final_unit.open_questions.len() <= 3);
    assert!(large.final_unit.open_questions.len() <= 3);
}

#[test]
fn test_evidence_below_budget_is_not_padded() {
    let engine = CoaEngine::new(CoaConfig::new().with_evidence_budget(5));
    let trace = engine
        .run_chunks(
            WIDGET_QUERY,
            &widget_chunks(3),
            &CoaLexicalWorker,
            &CoaLexicalManager,
        )
        .expect("well-formed chain-of-agents run");
    // Only 3 relevant chunks, well under the budget: nothing evicted.
    assert_eq!(trace.final_unit.evidence.len(), 3);
}

// ── Edge cases ────────────────────────────────────────────────────────────────

#[test]
fn test_single_chunk_reduces_to_one_worker_and_manager() {
    let engine = CoaEngine::new(CoaConfig::default());
    let trace = engine
        .run(
            "What is the capital of France?",
            "Paris is the capital of France.",
            &CoaLexicalWorker,
            &CoaLexicalManager,
        )
        .expect("well-formed chain-of-agents run");

    assert_eq!(trace.chunk_count, 1);
    assert_eq!(trace.workers_run, 1);
    assert_eq!(trace.steps.len(), 1);
    assert!(!trace.truncated);
    assert!(trace.answer.contains("Paris"));
}

#[test]
fn test_empty_document_is_an_honest_error() {
    let engine = CoaEngine::new(CoaConfig::default());
    let result = engine.run("a query", "   ", &CoaLexicalWorker, &CoaLexicalManager);
    assert_eq!(result, Err(CoaError::EmptyDocument));
}

#[test]
fn test_empty_chunks_slice_is_an_honest_error() {
    let engine = CoaEngine::new(CoaConfig::default());
    let result = engine.run_chunks("a query", &[], &CoaLexicalWorker, &CoaLexicalManager);
    assert_eq!(result, Err(CoaError::EmptyDocument));
}

#[test]
fn test_empty_query_is_an_honest_error() {
    let engine = CoaEngine::new(CoaConfig::default());
    let result = engine.run(
        "   ",
        "some document text.",
        &CoaLexicalWorker,
        &CoaLexicalManager,
    );
    assert_eq!(result, Err(CoaError::EmptyQuery));
}

#[test]
fn test_invalid_config_is_an_honest_error() {
    let engine = CoaEngine::new(CoaConfig::new().with_evidence_budget(0));
    let result = engine.run_chunks(
        "a query",
        &["a chunk".to_string()],
        &CoaLexicalWorker,
        &CoaLexicalManager,
    );
    assert!(matches!(result, Err(CoaError::InvalidConfig { .. })));
}

#[test]
fn test_chunk_with_no_relevant_content_passes_through_essentially_unchanged() {
    let engine = CoaEngine::new(CoaConfig::default());
    let chunks = vec![
        "The population of Springfield reached fifty thousand this year.".to_string(),
        "The weather in the mountains stayed calm and pleasant.".to_string(),
    ];

    let trace = engine
        .run_chunks(
            "What is the population of Springfield?",
            &chunks,
            &CoaLexicalWorker,
            &CoaLexicalManager,
        )
        .expect("well-formed chain-of-agents run");

    let incoming = &trace.steps[1].incoming_unit;
    let outgoing = &trace.steps[1].outgoing_unit;

    // The only field that is allowed to change on a chunk with zero
    // query-relevant content is the chunk-counter bookkeeping field.
    assert_eq!(outgoing.chunks_seen, incoming.chunks_seen + 1);
    assert_eq!(outgoing.evidence, incoming.evidence);
    assert_eq!(outgoing.partial_answer, incoming.partial_answer);
    assert_eq!(outgoing.open_questions, incoming.open_questions);
    assert_eq!(outgoing.completeness, incoming.completeness);
}

#[test]
fn test_more_chunks_than_max_workers_is_documented_not_silent() {
    let engine = CoaEngine::new(CoaConfig::new().with_max_workers(3));
    let chunks: Vec<String> = (0..7)
        .map(|i| format!("chunk content number {i}."))
        .collect();

    let trace = engine
        .run_chunks(
            "irrelevant query text",
            &chunks,
            &CoaLexicalWorker,
            &CoaLexicalManager,
        )
        .expect("well-formed chain-of-agents run");

    assert_eq!(trace.chunk_count, 7);
    assert_eq!(trace.workers_run, 3);
    assert!(trace.truncated);
    assert_eq!(trace.steps.len(), 3);
    // The *first* max_workers chunks are processed, in order — not a
    // silent drop, not a sample.
    for (i, step) in trace.steps.iter().enumerate() {
        assert_eq!(step.chunk, chunks[i]);
        assert_eq!(step.worker_index, i);
        assert_eq!(step.chunk_index, i);
    }
}

// ── Trait-object usage (object safety) ───────────────────────────────────────

#[test]
fn test_engine_accepts_trait_objects() {
    let worker: &dyn CoaWorker = &CoaLexicalWorker;
    let manager: &dyn CoaManager = &CoaLexicalManager;
    let engine = CoaEngine::new(CoaConfig::default());
    let chunks = vec!["Paris is the capital of France.".to_string()];

    let trace = engine
        .run_chunks("What is the capital of France?", &chunks, worker, manager)
        .expect("run should succeed with trait objects");
    assert_eq!(trace.workers_run, 1);
}

// ── CoaWorker / CoaManager direct error handling ─────────────────────────────

#[test]
fn test_worker_rejects_empty_query() {
    let unit = CoaCommunicationUnit::new(4, 4);
    let result = CoaLexicalWorker.process("", "some chunk text", &unit);
    assert_eq!(result, Err(CoaError::EmptyQuery));
}

#[test]
fn test_manager_rejects_empty_query() {
    let unit = CoaCommunicationUnit::new(4, 4);
    let result = CoaLexicalManager.synthesize("", &unit);
    assert_eq!(result, Err(CoaError::EmptyQuery));
}

// ── CoaCommunicationUnit: bounded merge and eviction rules ───────────────────

#[test]
fn test_merge_evidence_sorts_by_relevance_and_truncates() {
    let mut unit = CoaCommunicationUnit::new(2, 4);
    unit.merge_evidence(vec![
        CoaEvidence::new("a", 0, 0.2),
        CoaEvidence::new("b", 1, 0.9),
        CoaEvidence::new("c", 2, 0.5),
    ]);

    assert_eq!(unit.evidence.len(), 2);
    assert_eq!(unit.evidence[0].text, "b");
    assert_eq!(unit.evidence[1].text, "c");
    assert!(!unit.evidence.iter().any(|e| e.text == "a"));
}

#[test]
fn test_merge_evidence_dedup_keeps_higher_scoring_copy() {
    let mut unit = CoaCommunicationUnit::new(5, 4);
    unit.merge_evidence(vec![CoaEvidence::new("x", 0, 0.3)]);
    unit.merge_evidence(vec![CoaEvidence::new("x", 3, 0.8)]);

    assert_eq!(unit.evidence.len(), 1);
    assert_eq!(unit.evidence[0].relevance_score, 0.8);
    assert_eq!(unit.evidence[0].source_chunk_index, 3);
}

#[test]
fn test_merge_evidence_tie_break_prefers_more_recent_chunk() {
    let mut unit = CoaCommunicationUnit::new(1, 4);
    unit.merge_evidence(vec![
        CoaEvidence::new("p", 0, 0.5),
        CoaEvidence::new("q", 5, 0.5),
    ]);

    assert_eq!(unit.evidence.len(), 1);
    assert_eq!(unit.evidence[0].text, "q");
}

#[test]
fn test_merge_evidence_is_idempotent_with_no_new_items() {
    let mut unit = CoaCommunicationUnit::new(3, 4);
    unit.merge_evidence(vec![
        CoaEvidence::new("a", 0, 0.4),
        CoaEvidence::new("b", 1, 0.6),
    ]);
    let before = unit.clone();
    unit.merge_evidence(Vec::new());
    assert_eq!(unit, before);
}

#[test]
fn test_merge_open_questions_fifo_eviction() {
    let mut unit = CoaCommunicationUnit::new(5, 2);
    unit.merge_open_questions(vec!["q1".to_string(), "q2".to_string(), "q3".to_string()]);
    assert_eq!(
        unit.open_questions,
        vec!["q2".to_string(), "q3".to_string()]
    );
}

#[test]
fn test_merge_open_questions_deduplicates() {
    let mut unit = CoaCommunicationUnit::new(5, 5);
    unit.merge_open_questions(vec!["q1".to_string()]);
    unit.merge_open_questions(vec!["q1".to_string(), "q2".to_string()]);
    assert_eq!(
        unit.open_questions,
        vec!["q1".to_string(), "q2".to_string()]
    );
}

#[test]
fn test_resolve_open_questions_removes_matching_entries() {
    let mut unit = CoaCommunicationUnit::new(5, 5);
    unit.open_questions = vec!["about foo".to_string(), "about bar".to_string()];
    unit.resolve_open_questions(|q| q.contains("foo"));
    assert_eq!(unit.open_questions, vec!["about bar".to_string()]);
}

#[test]
fn test_enforce_budgets_is_idempotent_and_bounds_evidence() {
    let mut unit = CoaCommunicationUnit::new(2, 4);
    unit.evidence = vec![
        CoaEvidence::new("a", 0, 0.1),
        CoaEvidence::new("b", 1, 0.9),
        CoaEvidence::new("c", 2, 0.5),
        CoaEvidence::new("d", 3, 0.7),
    ];
    unit.enforce_budgets();
    assert_eq!(unit.evidence.len(), 2);
    assert_eq!(unit.evidence[0].text, "b");
    assert_eq!(unit.evidence[1].text, "d");

    let after_first = unit.clone();
    unit.enforce_budgets();
    assert_eq!(unit, after_first);
}

#[test]
fn test_evidence_new_clamps_relevance_score() {
    assert_eq!(CoaEvidence::new("t", 0, 1.5).relevance_score, 1.0);
    assert_eq!(CoaEvidence::new("t", 0, -0.5).relevance_score, 0.0);
}

// ── CoaConfig validation ──────────────────────────────────────────────────────

#[test]
fn test_config_default_validates() {
    assert!(CoaConfig::default().validate().is_ok());
}

#[test]
fn test_config_rejects_tiny_chunk_size() {
    let result = CoaConfig::new().with_chunk_size(4).validate();
    assert!(matches!(result, Err(CoaError::InvalidConfig { .. })));
}

#[test]
fn test_config_rejects_overlap_not_smaller_than_chunk_size() {
    let result = CoaConfig::new()
        .with_chunk_size(100)
        .with_chunk_overlap(100)
        .validate();
    assert!(matches!(result, Err(CoaError::InvalidConfig { .. })));
}

#[test]
fn test_config_rejects_zero_evidence_budget() {
    let result = CoaConfig::new().with_evidence_budget(0).validate();
    assert!(matches!(result, Err(CoaError::InvalidConfig { .. })));
}

#[test]
fn test_config_rejects_zero_max_workers() {
    let result = CoaConfig::new().with_max_workers(0).validate();
    assert!(matches!(result, Err(CoaError::InvalidConfig { .. })));
}

// ── Local sentence splitter / chunk packer ───────────────────────────────────

#[test]
fn test_split_sentences_basic_punctuation() {
    let sentences = split_sentences("Hello world. How are you? Fine!");
    assert_eq!(sentences, vec!["Hello world.", "How are you?", "Fine!"]);
}

#[test]
fn test_split_sentences_newline_boundaries() {
    let sentences = split_sentences("Line one\nLine two\n");
    assert_eq!(sentences, vec!["Line one", "Line two"]);
}

#[test]
fn test_split_into_chunks_empty_document() {
    assert!(split_into_chunks("", 100, 10).is_empty());
    assert!(split_into_chunks("   ", 100, 10).is_empty());
}

#[test]
fn test_split_into_chunks_oversized_single_sentence_is_not_cut() {
    let doc = "This sentence has no terminal punctuation and is quite long indeed";
    let chunks = split_into_chunks(doc, 20, 5);
    assert_eq!(chunks, vec![doc.to_string()]);
}

#[test]
fn test_split_into_chunks_packs_multiple_and_overlaps() {
    let sentences: Vec<String> = (0..12)
        .map(|i| format!("Sentence number {i} appears here for overlap testing."))
        .collect();
    let doc = sentences.join(" ");

    let chunks = split_into_chunks(&doc, 90, 55);
    assert!(
        chunks.len() > 1,
        "expected the document to be split into multiple chunks"
    );

    let mut found_overlap = false;
    for window in chunks.windows(2) {
        for sentence in &sentences {
            if window[0].contains(sentence.as_str()) && window[1].contains(sentence.as_str()) {
                found_overlap = true;
            }
        }
    }
    assert!(
        found_overlap,
        "expected at least one sentence shared between consecutive chunks due to overlap, chunks: {chunks:?}"
    );
}

#[test]
fn test_engine_run_splits_a_long_document_into_multiple_chunks() {
    let sentences: Vec<String> = (0..12)
        .map(|i| format!("Widget report {i} contains performance details worth noting."))
        .collect();
    let document = sentences.join(" ");

    let engine = CoaEngine::new(CoaConfig::new().with_chunk_size(90).with_chunk_overlap(30));
    let trace = engine
        .run(
            WIDGET_QUERY,
            &document,
            &CoaLexicalWorker,
            &CoaLexicalManager,
        )
        .expect("well-formed chain-of-agents run");

    assert!(trace.chunk_count > 1);
    assert_eq!(trace.workers_run, trace.chunk_count);
    assert_eq!(trace.steps.len(), trace.chunk_count);
}
