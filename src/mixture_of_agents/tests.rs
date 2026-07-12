#![allow(clippy::float_cmp, clippy::similar_names, clippy::too_many_lines)]
//! Tests for the `mixture_of_agents` module.

use std::cell::RefCell;

use super::aggregator::{MoaSynthesisAggregator, cluster_sentences, content_terms, jaccard};
use super::engine::{MoaEngine, MockMoaProposer};
use super::types::{
    MoaAggregator, MoaConfig, MoaContextMode, MoaError, MoaLayer, MoaLayerStats, MoaProposer,
    MoaResponse, MoaTrace,
};

// ── Test helpers ─────────────────────────────────────────────────────────────

/// Records every `prior_responses` slice it was called with, so tests can
/// verify the engine hands each proposer exactly the context
/// [`super::types::MoaContextMode`] dictates.
#[derive(Default)]
struct RecordingMoaProposer {
    response_text: String,
    seen_snapshots: RefCell<Vec<Vec<MoaResponse>>>,
}

impl RecordingMoaProposer {
    fn new(response_text: impl Into<String>) -> Self {
        Self {
            response_text: response_text.into(),
            seen_snapshots: RefCell::new(Vec::new()),
        }
    }
}

impl MoaProposer for RecordingMoaProposer {
    fn propose(&self, _query: &str, prior_responses: &[MoaResponse]) -> Result<String, MoaError> {
        self.seen_snapshots
            .borrow_mut()
            .push(prior_responses.to_vec());
        Ok(self.response_text.clone())
    }
}

/// A proposer whose output changes only once it is given non-empty prior
/// context, for tests that need a clean, single-step "improves with
/// context" signal (as opposed to [`MockMoaProposer`]'s incremental
/// incorporation, which is exercised elsewhere).
struct RefiningMoaProposer {
    base_fact: String,
    refinement_fact: String,
}

impl MoaProposer for RefiningMoaProposer {
    fn propose(&self, _query: &str, prior_responses: &[MoaResponse]) -> Result<String, MoaError> {
        if prior_responses.is_empty() {
            Ok(self.base_fact.clone())
        } else {
            Ok(format!("{} {}", self.base_fact, self.refinement_fact))
        }
    }
}

/// A [`MoaProposer`] that always fails, for error-propagation tests.
struct FailingMoaProposer;

impl MoaProposer for FailingMoaProposer {
    fn propose(&self, _query: &str, _prior_responses: &[MoaResponse]) -> Result<String, MoaError> {
        Err(MoaError::ProposerFailed {
            reason: "boom".to_string(),
        })
    }
}

/// A [`MoaAggregator`] that always fails, for error-propagation tests.
struct FailingMoaAggregator;

impl MoaAggregator for FailingMoaAggregator {
    fn aggregate(&self, _query: &str, _proposals: &[MoaResponse]) -> Result<String, MoaError> {
        Err(MoaError::AggregatorFailed {
            reason: "boom".to_string(),
        })
    }
}

/// A [`MoaProposer`] that always returns an empty string, regardless of
/// context.
struct EmptyMoaProposer;

impl MoaProposer for EmptyMoaProposer {
    fn propose(&self, _query: &str, _prior_responses: &[MoaResponse]) -> Result<String, MoaError> {
        Ok(String::new())
    }
}

// ── MoaConfig ────────────────────────────────────────────────────────────────

#[test]
fn config_default_values() {
    let config = MoaConfig::default();
    assert_eq!(config.num_layers, 3);
    assert_eq!(config.proposers_per_layer, 3);
    assert_eq!(config.context_mode, MoaContextMode::AggregateAndProposals);
    assert_eq!(config.early_stop_similarity, None);
}

#[test]
fn config_new_equals_default() {
    assert_eq!(MoaConfig::new(), MoaConfig::default());
}

#[test]
fn config_with_num_layers() {
    assert_eq!(MoaConfig::new().with_num_layers(7).num_layers, 7);
}

#[test]
fn config_with_proposers_per_layer() {
    assert_eq!(
        MoaConfig::new()
            .with_proposers_per_layer(5)
            .proposers_per_layer,
        5
    );
}

#[test]
fn config_with_context_mode() {
    assert_eq!(
        MoaConfig::new()
            .with_context_mode(MoaContextMode::ProposalsOnly)
            .context_mode,
        MoaContextMode::ProposalsOnly
    );
}

#[test]
fn config_with_early_stop_similarity() {
    assert_eq!(
        MoaConfig::new()
            .with_early_stop_similarity(0.9)
            .early_stop_similarity,
        Some(0.9)
    );
}

#[test]
fn config_without_early_stop_clears_threshold() {
    let config = MoaConfig::new()
        .with_early_stop_similarity(0.9)
        .without_early_stop();
    assert_eq!(config.early_stop_similarity, None);
}

#[test]
fn config_builder_chain() {
    let config = MoaConfig::new()
        .with_num_layers(4)
        .with_proposers_per_layer(2)
        .with_context_mode(MoaContextMode::AggregateOnly)
        .with_early_stop_similarity(0.95);
    assert_eq!(config.num_layers, 4);
    assert_eq!(config.proposers_per_layer, 2);
    assert_eq!(config.context_mode, MoaContextMode::AggregateOnly);
    assert_eq!(config.early_stop_similarity, Some(0.95));
}

#[test]
fn context_mode_default_is_aggregate_and_proposals() {
    assert_eq!(
        MoaContextMode::default(),
        MoaContextMode::AggregateAndProposals
    );
}

// ── MoaResponse ──────────────────────────────────────────────────────────────

#[test]
fn response_proposal_constructor() {
    let response = MoaResponse::proposal(2, 1, "hello");
    assert_eq!(response.proposer_id, 2);
    assert_eq!(response.layer, 1);
    assert_eq!(response.text, "hello");
    assert!(!response.is_aggregate);
}

#[test]
fn response_aggregate_constructor() {
    let response = MoaResponse::aggregate(3, "merged");
    assert_eq!(response.proposer_id, usize::MAX);
    assert_eq!(response.layer, 3);
    assert_eq!(response.text, "merged");
    assert!(response.is_aggregate);
}

// ── MoaError ─────────────────────────────────────────────────────────────────

#[test]
fn error_empty_query_display() {
    assert_eq!(MoaError::EmptyQuery.to_string(), "query must not be empty");
}

#[test]
fn error_zero_layers_display() {
    assert_eq!(
        MoaError::ZeroLayers.to_string(),
        "num_layers must be at least 1, got 0"
    );
}

#[test]
fn error_zero_proposers_display() {
    assert_eq!(
        MoaError::ZeroProposers.to_string(),
        "at least 1 proposer is required, got 0"
    );
}

#[test]
fn error_proposer_count_mismatch_display() {
    assert_eq!(
        MoaError::ProposerCountMismatch {
            expected: 3,
            actual: 1
        }
        .to_string(),
        "expected 3 proposer(s) (per MoaConfig::proposers_per_layer), got 1"
    );
}

#[test]
fn error_invalid_similarity_threshold_display() {
    assert_eq!(
        MoaError::InvalidSimilarityThreshold { value: 1.5 }.to_string(),
        "early_stop_similarity must be within [0.0, 1.0], got 1.5"
    );
}

#[test]
fn error_proposer_failed_display() {
    assert_eq!(
        MoaError::ProposerFailed {
            reason: "timeout".to_string()
        }
        .to_string(),
        "proposer failed to produce a response: timeout"
    );
}

#[test]
fn error_aggregator_failed_display() {
    assert_eq!(
        MoaError::AggregatorFailed {
            reason: "oops".to_string()
        }
        .to_string(),
        "aggregator failed to synthesize a response: oops"
    );
}

#[test]
fn error_empty_proposals_display() {
    assert_eq!(
        MoaError::EmptyProposals.to_string(),
        "aggregator was called with zero proposals to synthesize"
    );
}

#[test]
fn error_is_clone_and_partial_eq() {
    let first = MoaError::ZeroLayers;
    let second = first.clone();
    assert_eq!(first, second);
    assert_ne!(MoaError::ZeroLayers, MoaError::EmptyQuery);
}

// ── MoaTrace ─────────────────────────────────────────────────────────────────

#[test]
fn trace_final_layer_returns_last() {
    let layer0 = MoaLayer {
        layer_index: 0,
        proposals: vec![],
        aggregate: MoaResponse::aggregate(0, "a"),
        stats: dummy_stats(0, None),
    };
    let layer1 = MoaLayer {
        layer_index: 1,
        proposals: vec![],
        aggregate: MoaResponse::aggregate(1, "b"),
        stats: dummy_stats(1, Some(0.0)),
    };
    let trace = MoaTrace {
        query: "q".to_string(),
        layers: vec![layer0, layer1],
        final_response: "b".to_string(),
        layers_run: 2,
        stopped_early: false,
        early_stop_reason: None,
    };
    assert_eq!(trace.final_layer().unwrap().aggregate.text, "b");
    assert_eq!(trace.layer(0).unwrap().aggregate.text, "a");
    assert!(trace.layer(5).is_none());
}

fn dummy_stats(layer_index: usize, change_from_previous: Option<f32>) -> MoaLayerStats {
    MoaLayerStats {
        layer_index,
        agreement: 1.0,
        distinct_claims: 1,
        covered_claims: 1,
        coverage_ratio: 1.0,
        change_from_previous,
    }
}

// ── lexical helpers (aggregator module) ─────────────────────────────────────

#[test]
fn content_terms_filters_stopwords_and_short_tokens() {
    let terms = content_terms("The cat is on a mat, and it is red");
    assert!(!terms.contains("the"));
    assert!(!terms.contains("is"));
    assert!(!terms.contains("on"));
    assert!(terms.contains("cat"));
    assert!(terms.contains("red"));
}

#[test]
fn content_terms_lowercases() {
    let terms = content_terms("CATS Dogs");
    assert!(terms.contains("cats"));
    assert!(terms.contains("dogs"));
}

#[test]
fn jaccard_identical_sets_is_one() {
    let a = content_terms("cats dogs birds");
    let b = content_terms("cats dogs birds");
    assert_eq!(jaccard(&a, &b), 1.0);
}

#[test]
fn jaccard_disjoint_sets_is_zero() {
    let a = content_terms("cats dogs");
    let b = content_terms("rockets planets");
    assert_eq!(jaccard(&a, &b), 0.0);
}

#[test]
fn jaccard_both_empty_is_zero() {
    assert_eq!(jaccard(&content_terms("a an is"), &content_terms("")), 0.0);
}

#[test]
fn cluster_sentences_groups_near_duplicates() {
    let proposals = vec![
        MoaResponse::proposal(0, 0, "The mitochondria is the powerhouse of the cell."),
        MoaResponse::proposal(1, 0, "Mitochondria serve as the powerhouse of the cell."),
    ];
    let clusters = cluster_sentences(&proposals, 0.5);
    assert_eq!(clusters.len(), 1);
    assert_eq!(clusters[0].proposer_ids.len(), 2);
}

#[test]
fn cluster_sentences_keeps_distinct_claims_separate() {
    let proposals = vec![
        MoaResponse::proposal(
            0,
            0,
            "Photosynthesis converts sunlight into chemical energy.",
        ),
        MoaResponse::proposal(1, 0, "Tectonic plates drift slowly across the mantle."),
    ];
    let clusters = cluster_sentences(&proposals, 0.5);
    assert_eq!(clusters.len(), 2);
}

#[test]
fn cluster_sentences_skips_contentless_sentences() {
    // Every token here is either under three characters or a stopword, so
    // neither sentence carries a single content term.
    let proposals = vec![MoaResponse::proposal(0, 0, "So it is. It was he.")];
    let clusters = cluster_sentences(&proposals, 0.5);
    assert!(clusters.is_empty());
}

// ── MoaSynthesisAggregator ───────────────────────────────────────────────────

#[test]
fn aggregator_default_threshold() {
    assert_eq!(MoaSynthesisAggregator::default().similarity_threshold, 0.6);
}

#[test]
fn aggregator_new_stores_threshold() {
    assert_eq!(MoaSynthesisAggregator::new(0.7).similarity_threshold, 0.7);
}

#[test]
fn aggregator_errors_on_empty_query() {
    let aggregator = MoaSynthesisAggregator::default();
    let proposals = vec![MoaResponse::proposal(0, 0, "some text")];
    assert_eq!(
        aggregator.aggregate("   ", &proposals),
        Err(MoaError::EmptyQuery)
    );
}

#[test]
fn aggregator_errors_on_empty_proposals() {
    let aggregator = MoaSynthesisAggregator::default();
    assert_eq!(
        aggregator.aggregate("q", &[]),
        Err(MoaError::EmptyProposals)
    );
}

/// The defining property: **synthesis, not selection**. Proposer A knows
/// only fact X, proposer B knows only fact Y — the aggregate must contain
/// both. A "pick the best proposal" aggregator would drop one.
#[test]
fn aggregator_synthesizes_both_facts_not_just_one() {
    let aggregator = MoaSynthesisAggregator::default();
    let proposal_a = MoaResponse::proposal(0, 0, "The Eiffel Tower was completed in 1889.");
    let proposal_b = MoaResponse::proposal(1, 0, "Quantum entanglement was studied by Einstein.");
    let merged = aggregator
        .aggregate("Tell me two facts.", &[proposal_a, proposal_b])
        .unwrap();
    let lower = merged.to_lowercase();
    assert!(
        lower.contains("eiffel"),
        "aggregate dropped fact X: {merged}"
    );
    assert!(
        lower.contains("entanglement"),
        "aggregate dropped fact Y: {merged}"
    );
}

#[test]
fn aggregator_retains_a_claim_made_by_only_one_of_three_proposers() {
    let aggregator = MoaSynthesisAggregator::default();
    let proposals = vec![
        MoaResponse::proposal(0, 0, "Water boils at one hundred degrees Celsius."),
        MoaResponse::proposal(1, 0, "Water boils at one hundred degrees Celsius."),
        MoaResponse::proposal(2, 0, "Copper conducts electricity remarkably well."),
    ];
    let merged = aggregator.aggregate("q", &proposals).unwrap();
    assert!(
        merged.to_lowercase().contains("copper"),
        "single-proposer claim was dropped: {merged}"
    );
}

#[test]
fn aggregator_orders_agreed_upon_content_before_minority_content() {
    let aggregator = MoaSynthesisAggregator::default();
    let proposals = vec![
        MoaResponse::proposal(
            0,
            0,
            "Solar panels convert sunlight into electricity efficiently.",
        ),
        MoaResponse::proposal(
            1,
            0,
            "Wind turbines are unrelated to solar generation entirely.",
        ),
        MoaResponse::proposal(
            2,
            0,
            "Solar panels convert sunlight into electricity efficiently.",
        ),
    ];
    let merged = aggregator.aggregate("q", &proposals).unwrap();
    let solar_pos = merged.to_lowercase().find("solar panels").unwrap();
    let wind_pos = merged.to_lowercase().find("wind turbines").unwrap();
    assert!(
        solar_pos < wind_pos,
        "content agreed on by 2/3 proposers should rank before a 1/3 minority claim: {merged}"
    );
}

#[test]
fn aggregator_does_not_duplicate_identical_proposals() {
    let aggregator = MoaSynthesisAggregator::default();
    let text = "Copper conducts electricity efficiently.";
    let proposals = vec![
        MoaResponse::proposal(0, 0, text),
        MoaResponse::proposal(1, 0, text),
        MoaResponse::proposal(2, 0, text),
    ];
    let merged = aggregator.aggregate("q", &proposals).unwrap();
    assert_eq!(merged.to_lowercase().matches("conducts").count(), 1);
}

#[test]
fn aggregator_single_proposal_reconstructs_it() {
    let aggregator = MoaSynthesisAggregator::default();
    let text = "Solo answers the question. The answer is forty-two.";
    let proposals = vec![MoaResponse::proposal(0, 0, text)];
    let merged = aggregator.aggregate("q", &proposals).unwrap();
    assert_eq!(merged, text);
}

#[test]
fn aggregator_handles_empty_proposal_text_gracefully() {
    let aggregator = MoaSynthesisAggregator::default();
    let proposals = vec![
        MoaResponse::proposal(0, 0, ""),
        MoaResponse::proposal(1, 0, "Granite is an igneous rock."),
    ];
    let merged = aggregator.aggregate("q", &proposals).unwrap();
    assert!(merged.to_lowercase().contains("granite"));
}

#[test]
fn aggregator_all_empty_proposals_yields_empty_string() {
    let aggregator = MoaSynthesisAggregator::default();
    let proposals = vec![
        MoaResponse::proposal(0, 0, ""),
        MoaResponse::proposal(1, 0, ""),
    ];
    let merged = aggregator.aggregate("q", &proposals).unwrap();
    assert!(merged.is_empty());
}

#[test]
fn aggregator_higher_threshold_keeps_near_duplicates_separate() {
    let proposals = vec![
        MoaResponse::proposal(0, 0, "The mitochondria is the powerhouse of the cell."),
        MoaResponse::proposal(
            1,
            0,
            "Ribosomes are found throughout the mitochondria matrix.",
        ),
    ];
    let loose = MoaSynthesisAggregator::new(0.1);
    let strict = MoaSynthesisAggregator::new(0.9);
    let loose_merged = loose.aggregate("q", &proposals).unwrap();
    let strict_merged = strict.aggregate("q", &proposals).unwrap();
    // A very strict threshold should never merge more than a loose one, so
    // its output cannot be shorter than the loose output's; a stricter
    // aggregator only ever splits clusters further, never coarsens them.
    assert!(strict_merged.len() >= loose_merged.len());
}

// ── MockMoaProposer ──────────────────────────────────────────────────────────

#[test]
fn mock_proposer_no_context_uses_facts_only() {
    let proposer = MockMoaProposer::new("A").with_fact("Fact one.");
    let response = proposer.propose("q", &[]).unwrap();
    assert!(response.contains("Fact one."));
    assert!(!response.contains("Incorporating"));
}

#[test]
fn mock_proposer_determinism() {
    let proposer = MockMoaProposer::new("A").with_fact("Fact one.");
    let first = proposer.propose("q", &[]).unwrap();
    let second = proposer.propose("q", &[]).unwrap();
    assert_eq!(first, second);
}

#[test]
fn mock_proposer_incorporates_new_prior_content() {
    let proposer = MockMoaProposer::new("A").with_fact("Sodium reacts vigorously with water.");
    let prior = vec![MoaResponse::aggregate(
        0,
        "Potassium ignites spontaneously in air.",
    )];
    let response = proposer.propose("q", &prior).unwrap();
    assert!(response.to_lowercase().contains("potassium"));
}

#[test]
fn mock_proposer_does_not_repeat_already_known_content() {
    let proposer = MockMoaProposer::new("A").with_fact("Sodium reacts vigorously with water.");
    let prior = vec![MoaResponse::aggregate(
        0,
        "Sodium reacts vigorously with water.",
    )];
    let response = proposer.propose("q", &prior).unwrap();
    assert_eq!(
        response.to_lowercase().matches("sodium").count(),
        1,
        "already-known content should not be re-appended: {response}"
    );
}

#[test]
fn mock_proposer_empty_label_uses_generic_name() {
    let proposer = MockMoaProposer::default();
    let response = proposer.propose("q", &[]).unwrap();
    assert!(response.starts_with("Proposer answers"));
}

#[test]
fn mock_proposer_with_fact_appends_in_order() {
    let proposer = MockMoaProposer::new("A")
        .with_fact("First fact.")
        .with_fact("Second fact.");
    assert_eq!(proposer.facts, vec!["First fact.", "Second fact."]);
}

// ── MoaEngine: validation / errors ──────────────────────────────────────────

#[test]
fn engine_error_empty_query() {
    let proposer = MockMoaProposer::new("A");
    let proposers: [&dyn MoaProposer; 1] = [&proposer];
    let engine = MoaEngine::new(MoaConfig::new().with_proposers_per_layer(1));
    let aggregator = MoaSynthesisAggregator::default();
    assert_eq!(
        engine.run("   ", &proposers, &aggregator),
        Err(MoaError::EmptyQuery)
    );
}

#[test]
fn engine_error_zero_layers() {
    let proposer = MockMoaProposer::new("A");
    let proposers: [&dyn MoaProposer; 1] = [&proposer];
    let engine = MoaEngine::new(
        MoaConfig::new()
            .with_num_layers(0)
            .with_proposers_per_layer(1),
    );
    let aggregator = MoaSynthesisAggregator::default();
    assert_eq!(
        engine.run("q", &proposers, &aggregator),
        Err(MoaError::ZeroLayers)
    );
}

#[test]
fn engine_error_zero_proposers() {
    let engine = MoaEngine::new(MoaConfig::new().with_proposers_per_layer(0));
    let aggregator = MoaSynthesisAggregator::default();
    let proposers: [&dyn MoaProposer; 0] = [];
    assert_eq!(
        engine.run("q", &proposers, &aggregator),
        Err(MoaError::ZeroProposers)
    );
}

#[test]
fn engine_error_proposer_count_mismatch() {
    let proposer = MockMoaProposer::new("A");
    let proposers: [&dyn MoaProposer; 1] = [&proposer];
    let engine = MoaEngine::new(MoaConfig::new().with_proposers_per_layer(3));
    let aggregator = MoaSynthesisAggregator::default();
    assert_eq!(
        engine.run("q", &proposers, &aggregator),
        Err(MoaError::ProposerCountMismatch {
            expected: 3,
            actual: 1
        })
    );
}

#[test]
fn engine_error_invalid_similarity_threshold() {
    let proposer = MockMoaProposer::new("A");
    let proposers: [&dyn MoaProposer; 1] = [&proposer];
    let engine = MoaEngine::new(
        MoaConfig::new()
            .with_proposers_per_layer(1)
            .with_early_stop_similarity(1.5),
    );
    let aggregator = MoaSynthesisAggregator::default();
    assert_eq!(
        engine.run("q", &proposers, &aggregator),
        Err(MoaError::InvalidSimilarityThreshold { value: 1.5 })
    );
}

#[test]
fn engine_proposer_error_propagates() {
    let failing = FailingMoaProposer;
    let proposers: [&dyn MoaProposer; 1] = [&failing];
    let engine = MoaEngine::new(MoaConfig::new().with_proposers_per_layer(1));
    let aggregator = MoaSynthesisAggregator::default();
    assert_eq!(
        engine.run("q", &proposers, &aggregator),
        Err(MoaError::ProposerFailed {
            reason: "boom".to_string()
        })
    );
}

#[test]
fn engine_aggregator_error_propagates() {
    let proposer = MockMoaProposer::new("A");
    let proposers: [&dyn MoaProposer; 1] = [&proposer];
    let engine = MoaEngine::new(MoaConfig::new().with_proposers_per_layer(1));
    let failing_aggregator = FailingMoaAggregator;
    assert_eq!(
        engine.run("q", &proposers, &failing_aggregator),
        Err(MoaError::AggregatorFailed {
            reason: "boom".to_string()
        })
    );
}

// ── MoaEngine: the layering is real ─────────────────────────────────────────

/// The core structural claim of mixture-of-agents: a layer-2 proposer
/// actually *receives* the layer-1 aggregate in `prior_responses`. This is
/// exactly the inter-layer communication `self_consistency` never has.
#[test]
fn engine_layer2_proposer_receives_layer1_aggregate() {
    let known_fact = MockMoaProposer::new("A").with_fact("Paris is the capital of France.");
    let recorder = RecordingMoaProposer::new("recorded response");
    let proposers: [&dyn MoaProposer; 2] = [&known_fact, &recorder];
    let engine = MoaEngine::new(
        MoaConfig::new()
            .with_num_layers(2)
            .with_proposers_per_layer(2)
            .with_context_mode(MoaContextMode::AggregateOnly),
    );
    let aggregator = MoaSynthesisAggregator::default();
    let _trace = engine
        .run("What is the capital of France?", &proposers, &aggregator)
        .unwrap();

    let snapshots = recorder.seen_snapshots.borrow();
    assert_eq!(
        snapshots.len(),
        2,
        "the proposer should be called once per layer"
    );
    assert!(
        snapshots[0].is_empty(),
        "layer 0 must receive no prior context"
    );
    assert_eq!(
        snapshots[1].len(),
        1,
        "AggregateOnly should expose exactly the layer-0 aggregate"
    );
    assert!(snapshots[1][0].is_aggregate);
    assert!(
        snapshots[1][0].text.to_lowercase().contains("paris"),
        "layer-2 proposer did not receive layer-1's synthesized aggregate: {:?}",
        snapshots[1][0].text
    );
}

/// A proposer that produces different output when given prior context
/// demonstrably changes the layer-2 output — proving the layering has a
/// real causal effect, not merely that data is threaded through unused.
#[test]
fn engine_layer2_output_changes_because_it_saw_layer1_aggregate() {
    let refining = RefiningMoaProposer {
        base_fact: "Base fact only.".to_string(),
        refinement_fact: "Refined with prior context.".to_string(),
    };
    let proposers: [&dyn MoaProposer; 1] = [&refining];
    let engine = MoaEngine::new(
        MoaConfig::new()
            .with_num_layers(2)
            .with_proposers_per_layer(1),
    );
    let aggregator = MoaSynthesisAggregator::default();
    let trace = engine.run("q", &proposers, &aggregator).unwrap();

    assert_eq!(trace.layers[0].proposals[0].text, "Base fact only.");
    assert_eq!(
        trace.layers[1].proposals[0].text,
        "Base fact only. Refined with prior context."
    );
    assert_ne!(
        trace.layers[0].aggregate.text,
        trace.layers[1].aggregate.text
    );
}

// ── MoaEngine: context mode ──────────────────────────────────────────────────

#[test]
fn engine_context_mode_aggregate_only() {
    let fact = MockMoaProposer::new("A").with_fact("Sodium reacts vigorously with water.");
    let recorder = RecordingMoaProposer::new("r");
    let proposers: [&dyn MoaProposer; 2] = [&fact, &recorder];
    let engine = MoaEngine::new(
        MoaConfig::new()
            .with_num_layers(2)
            .with_proposers_per_layer(2)
            .with_context_mode(MoaContextMode::AggregateOnly),
    );
    let aggregator = MoaSynthesisAggregator::default();
    let _ = engine.run("q", &proposers, &aggregator).unwrap();
    let snapshots = recorder.seen_snapshots.borrow();
    assert_eq!(snapshots[1].len(), 1);
    assert!(snapshots[1][0].is_aggregate);
}

#[test]
fn engine_context_mode_proposals_only() {
    let fact = MockMoaProposer::new("A").with_fact("Sodium reacts vigorously with water.");
    let recorder = RecordingMoaProposer::new("r");
    let proposers: [&dyn MoaProposer; 2] = [&fact, &recorder];
    let engine = MoaEngine::new(
        MoaConfig::new()
            .with_num_layers(2)
            .with_proposers_per_layer(2)
            .with_context_mode(MoaContextMode::ProposalsOnly),
    );
    let aggregator = MoaSynthesisAggregator::default();
    let _ = engine.run("q", &proposers, &aggregator).unwrap();
    let snapshots = recorder.seen_snapshots.borrow();
    assert_eq!(snapshots[1].len(), 2);
    assert!(snapshots[1].iter().all(|r| !r.is_aggregate));
}

#[test]
fn engine_context_mode_aggregate_and_proposals() {
    let fact = MockMoaProposer::new("A").with_fact("Sodium reacts vigorously with water.");
    let recorder = RecordingMoaProposer::new("r");
    let proposers: [&dyn MoaProposer; 2] = [&fact, &recorder];
    let engine = MoaEngine::new(
        MoaConfig::new()
            .with_num_layers(2)
            .with_proposers_per_layer(2)
            .with_context_mode(MoaContextMode::AggregateAndProposals),
    );
    let aggregator = MoaSynthesisAggregator::default();
    let _ = engine.run("q", &proposers, &aggregator).unwrap();
    let snapshots = recorder.seen_snapshots.borrow();
    assert_eq!(snapshots[1].len(), 3);
    assert_eq!(snapshots[1].iter().filter(|r| r.is_aggregate).count(), 1);
}

// ── MoaEngine: synthesis, not selection (end to end) ────────────────────────

#[test]
fn engine_final_response_contains_facts_from_every_proposer() {
    let proposer_a =
        MockMoaProposer::new("A").with_fact("Fact X concerns photosynthesis in plants.");
    let proposer_b =
        MockMoaProposer::new("B").with_fact("Fact Y concerns tectonic plate movement.");
    let proposers: [&dyn MoaProposer; 2] = [&proposer_a, &proposer_b];
    let engine = MoaEngine::new(
        MoaConfig::new()
            .with_num_layers(1)
            .with_proposers_per_layer(2),
    );
    let aggregator = MoaSynthesisAggregator::default();
    let trace = engine
        .run("What do you know?", &proposers, &aggregator)
        .unwrap();
    let lower = trace.final_response.to_lowercase();
    assert!(lower.contains("photosynthesis"));
    assert!(lower.contains("tectonic"));
}

// ── MoaEngine: monotone improvement with depth ──────────────────────────────

#[test]
fn engine_distinct_claims_are_non_decreasing_across_layers() {
    let proposer_a = RefiningMoaProposer {
        base_fact: "Aardvarks are nocturnal burrowing mammals.".to_string(),
        refinement_fact: "Aardvarks primarily eat ants and termites.".to_string(),
    };
    let proposer_b = RefiningMoaProposer {
        base_fact: "Basalt is a common volcanic rock.".to_string(),
        refinement_fact: "Basalt forms from rapidly cooling lava.".to_string(),
    };
    let proposers: [&dyn MoaProposer; 2] = [&proposer_a, &proposer_b];
    let engine = MoaEngine::new(
        MoaConfig::new()
            .with_num_layers(3)
            .with_proposers_per_layer(2),
    );
    let aggregator = MoaSynthesisAggregator::default();
    let trace = engine
        .run("Tell me about nature.", &proposers, &aggregator)
        .unwrap();

    assert_eq!(trace.layers.len(), 3);
    let claim_counts: Vec<usize> = trace
        .layers
        .iter()
        .map(|l| l.stats.distinct_claims)
        .collect();
    for window in claim_counts.windows(2) {
        assert!(
            window[1] >= window[0],
            "distinct claim coverage regressed across layers: {claim_counts:?}"
        );
    }
    // The refinement facts are only visible from layer 1 onward, so
    // coverage should have strictly grown by the second layer.
    assert!(
        claim_counts[1] > claim_counts[0],
        "claim coverage did not grow once proposers saw prior context: {claim_counts:?}"
    );

    // The layer-0 aggregate must not already contain layer-1-only content.
    assert!(
        !trace.layers[0]
            .aggregate
            .text
            .to_lowercase()
            .contains("termites")
    );
    assert!(
        trace.layers[2]
            .aggregate
            .text
            .to_lowercase()
            .contains("termites")
    );
    assert!(
        trace.layers[2]
            .aggregate
            .text
            .to_lowercase()
            .contains("lava")
    );
}

// ── MoaEngine: early stopping ────────────────────────────────────────────────

#[test]
fn engine_early_stop_fires_when_aggregate_stabilizes() {
    let proposer_a =
        MockMoaProposer::new("A").with_fact("The mitochondria is the powerhouse of the cell.");
    let proposer_b =
        MockMoaProposer::new("B").with_fact("Ribosomes synthesize proteins within the cell.");
    let proposers: [&dyn MoaProposer; 2] = [&proposer_a, &proposer_b];
    let engine = MoaEngine::new(
        MoaConfig::new()
            .with_num_layers(6)
            .with_proposers_per_layer(2)
            .with_early_stop_similarity(0.99),
    );
    let aggregator = MoaSynthesisAggregator::default();
    let trace = engine
        .run("Explain cell biology.", &proposers, &aggregator)
        .unwrap();

    assert!(trace.stopped_early);
    assert!(trace.early_stop_reason.is_some());
    assert!(trace.layers_run < 6);
    assert_eq!(trace.layers.len(), trace.layers_run);

    let last_stats = trace.layers.last().unwrap().stats;
    let change = last_stats.change_from_previous.unwrap();
    assert!(
        1.0 - change >= 0.99,
        "run should only stop once similarity actually cleared the threshold: change={change}"
    );

    for layer in &trace.layers {
        assert_eq!(layer.proposals.len(), 2);
        assert!(!layer.aggregate.text.is_empty());
    }
}

#[test]
fn engine_early_stop_disabled_runs_full_layers() {
    let proposer_a = MockMoaProposer::new("A").with_fact("A stable fact.");
    let proposer_b = MockMoaProposer::new("B").with_fact("Another stable fact.");
    let proposers: [&dyn MoaProposer; 2] = [&proposer_a, &proposer_b];
    let engine = MoaEngine::new(
        MoaConfig::new()
            .with_num_layers(4)
            .with_proposers_per_layer(2),
    );
    let aggregator = MoaSynthesisAggregator::default();
    let trace = engine.run("q", &proposers, &aggregator).unwrap();
    assert_eq!(trace.layers_run, 4);
    assert!(!trace.stopped_early);
    assert!(trace.early_stop_reason.is_none());
}

#[test]
fn engine_trace_records_every_layer() {
    let proposer_a = MockMoaProposer::new("A").with_fact("Fact one.");
    let proposer_b = MockMoaProposer::new("B").with_fact("Fact two.");
    let proposers: [&dyn MoaProposer; 2] = [&proposer_a, &proposer_b];
    let engine = MoaEngine::new(
        MoaConfig::new()
            .with_num_layers(3)
            .with_proposers_per_layer(2),
    );
    let aggregator = MoaSynthesisAggregator::default();
    let trace = engine.run("q", &proposers, &aggregator).unwrap();
    assert_eq!(trace.layers.len(), 3);
    for (index, layer) in trace.layers.iter().enumerate() {
        assert_eq!(layer.layer_index, index);
        assert_eq!(layer.proposals.len(), 2);
        assert_eq!(layer.proposals[0].proposer_id, 0);
        assert_eq!(layer.proposals[1].proposer_id, 1);
        assert!(!layer.aggregate.text.is_empty());
    }
    assert_eq!(trace.layers[0].stats.change_from_previous, None);
    assert!(trace.layers[1].stats.change_from_previous.is_some());
    assert!(trace.layers[2].stats.change_from_previous.is_some());
}

// ── MoaEngine: edge cases ────────────────────────────────────────────────────

#[test]
fn engine_single_proposer_aggregate_approximates_the_proposal() {
    // Deliberately punctuation-free query: `MockMoaProposer`'s template
    // quotes the query verbatim, and this module's sentence splitter is a
    // simple `.`/`!`/`?`-boundary scanner with no quote-awareness, so a
    // query containing its own terminal punctuation would itself introduce
    // an extra (contentless, and therefore correctly dropped) sentence
    // boundary — this test is about single-proposal reconstruction, not
    // about that unrelated tokenizer edge case.
    let proposer = MockMoaProposer::new("Solo").with_fact("The answer is forty-two.");
    let proposers: [&dyn MoaProposer; 1] = [&proposer];
    let engine = MoaEngine::new(
        MoaConfig::new()
            .with_num_layers(1)
            .with_proposers_per_layer(1),
    );
    let aggregator = MoaSynthesisAggregator::default();
    let trace = engine.run("the answer", &proposers, &aggregator).unwrap();
    let solo_output = proposer.propose("the answer", &[]).unwrap();
    assert_eq!(trace.final_response, solo_output);
}

#[test]
fn engine_single_layer_reduces_to_propose_then_aggregate() {
    let proposer_a = MockMoaProposer::new("A").with_fact("Fact one.");
    let proposer_b = MockMoaProposer::new("B").with_fact("Fact two.");
    let proposers: [&dyn MoaProposer; 2] = [&proposer_a, &proposer_b];
    let engine = MoaEngine::new(
        MoaConfig::new()
            .with_num_layers(1)
            .with_proposers_per_layer(2),
    );
    let aggregator = MoaSynthesisAggregator::default();
    let trace = engine.run("q", &proposers, &aggregator).unwrap();
    assert_eq!(trace.layers.len(), 1);
    assert_eq!(trace.layers_run, 1);
    assert!(!trace.stopped_early);
    assert_eq!(trace.layers[0].stats.change_from_previous, None);
    assert_eq!(trace.final_response, trace.layers[0].aggregate.text);
}

#[test]
fn engine_handles_empty_proposer_output_gracefully() {
    let empty = EmptyMoaProposer;
    let proposer_b = MockMoaProposer::new("B").with_fact("Water boils at 100 degrees Celsius.");
    let proposers: [&dyn MoaProposer; 2] = [&empty, &proposer_b];
    let engine = MoaEngine::new(
        MoaConfig::new()
            .with_num_layers(1)
            .with_proposers_per_layer(2),
    );
    let aggregator = MoaSynthesisAggregator::default();
    let trace = engine.run("q", &proposers, &aggregator).unwrap();
    assert_eq!(trace.layers[0].proposals[0].text, "");
    assert!(trace.final_response.to_lowercase().contains("boils"));
}

#[test]
fn engine_all_proposers_identical_output_not_duplicated() {
    struct FixedMoaProposer(&'static str);
    impl MoaProposer for FixedMoaProposer {
        fn propose(&self, _query: &str, _prior: &[MoaResponse]) -> Result<String, MoaError> {
            Ok(self.0.to_string())
        }
    }
    let fixed = FixedMoaProposer("Copper conducts electricity efficiently.");
    let proposers: [&dyn MoaProposer; 3] = [&fixed, &fixed, &fixed];
    let engine = MoaEngine::new(
        MoaConfig::new()
            .with_num_layers(1)
            .with_proposers_per_layer(3),
    );
    let aggregator = MoaSynthesisAggregator::default();
    let trace = engine.run("q", &proposers, &aggregator).unwrap();
    assert_eq!(
        trace
            .final_response
            .to_lowercase()
            .matches("conducts")
            .count(),
        1
    );
}

// ── MoaEngine: determinism / defaults ───────────────────────────────────────

#[test]
fn engine_determinism_same_inputs_twice() {
    let proposer_a = MockMoaProposer::new("A").with_fact("Fact A.");
    let proposer_b = MockMoaProposer::new("B").with_fact("Fact B.");
    let proposers: [&dyn MoaProposer; 2] = [&proposer_a, &proposer_b];
    let engine = MoaEngine::new(
        MoaConfig::new()
            .with_num_layers(2)
            .with_proposers_per_layer(2),
    );
    let aggregator = MoaSynthesisAggregator::default();
    let first = engine.run("q", &proposers, &aggregator).unwrap();
    let second = engine.run("q", &proposers, &aggregator).unwrap();
    assert_eq!(first, second);
}

#[test]
fn engine_default_uses_default_config() {
    let engine = MoaEngine::default();
    assert_eq!(engine.config, MoaConfig::default());
}

#[test]
fn engine_new_stores_config() {
    let config = MoaConfig::new().with_num_layers(9);
    let engine = MoaEngine::new(config.clone());
    assert_eq!(engine.config, config);
}

// ── MoaLayerStats ────────────────────────────────────────────────────────────

#[test]
fn layer_stats_agreement_is_one_for_single_proposal() {
    let proposer = MockMoaProposer::new("Solo").with_fact("The answer is forty-two.");
    let proposers: [&dyn MoaProposer; 1] = [&proposer];
    let engine = MoaEngine::new(
        MoaConfig::new()
            .with_num_layers(1)
            .with_proposers_per_layer(1),
    );
    let aggregator = MoaSynthesisAggregator::default();
    let trace = engine.run("q", &proposers, &aggregator).unwrap();
    assert_eq!(trace.layers[0].stats.agreement, 1.0);
}

#[test]
fn layer_stats_coverage_ratio_is_one_for_synthesis_aggregator() {
    let proposer_a = MockMoaProposer::new("A")
        .with_fact("Photosynthesis converts sunlight into chemical energy.");
    let proposer_b =
        MockMoaProposer::new("B").with_fact("Tectonic plates drift across the planetary mantle.");
    let proposers: [&dyn MoaProposer; 2] = [&proposer_a, &proposer_b];
    let engine = MoaEngine::new(
        MoaConfig::new()
            .with_num_layers(1)
            .with_proposers_per_layer(2),
    );
    let aggregator = MoaSynthesisAggregator::default();
    let trace = engine.run("q", &proposers, &aggregator).unwrap();
    assert_eq!(trace.layers[0].stats.coverage_ratio, 1.0);
    assert_eq!(
        trace.layers[0].stats.covered_claims,
        trace.layers[0].stats.distinct_claims
    );
}

/// A selective (non-synthesizing) aggregator should show materially lower
/// coverage than [`MoaSynthesisAggregator`] on the same inputs — the
/// [`MoaLayerStats::coverage_ratio`] signal is meant to be able to catch
/// exactly this kind of "pick a winner" anti-pattern.
#[test]
fn layer_stats_coverage_ratio_is_lower_for_a_selecting_aggregator() {
    struct PickFirstAggregator;
    impl MoaAggregator for PickFirstAggregator {
        fn aggregate(&self, query: &str, proposals: &[MoaResponse]) -> Result<String, MoaError> {
            if query.trim().is_empty() {
                return Err(MoaError::EmptyQuery);
            }
            proposals
                .first()
                .map(|p| p.text.clone())
                .ok_or(MoaError::EmptyProposals)
        }
    }

    let proposer_a = MockMoaProposer::new("A")
        .with_fact("Photosynthesis converts sunlight into chemical energy.");
    let proposer_b =
        MockMoaProposer::new("B").with_fact("Tectonic plates drift across the planetary mantle.");
    let proposers: [&dyn MoaProposer; 2] = [&proposer_a, &proposer_b];
    let engine = MoaEngine::new(
        MoaConfig::new()
            .with_num_layers(1)
            .with_proposers_per_layer(2),
    );
    let selecting = PickFirstAggregator;
    let trace = engine.run("q", &proposers, &selecting).unwrap();
    assert!(
        trace.layers[0].stats.coverage_ratio < 1.0,
        "a selecting aggregator should be caught by a coverage_ratio below 1.0: {}",
        trace.layers[0].stats.coverage_ratio
    );
}

#[test]
fn layer_stats_change_from_previous_is_none_for_first_layer() {
    let proposer = MockMoaProposer::new("A").with_fact("Fact one.");
    let proposers: [&dyn MoaProposer; 1] = [&proposer];
    let engine = MoaEngine::new(
        MoaConfig::new()
            .with_num_layers(2)
            .with_proposers_per_layer(1),
    );
    let aggregator = MoaSynthesisAggregator::default();
    let trace = engine.run("q", &proposers, &aggregator).unwrap();
    assert_eq!(trace.layers[0].stats.change_from_previous, None);
    assert!(trace.layers[1].stats.change_from_previous.is_some());
}
