//! Tests for the `uprise_retrieval` module.
#![allow(clippy::float_cmp, clippy::similar_names)]

use super::retriever::{UpriseIndex, UpriseRetriever};
use super::types::{PromptExemplar, UpriseConfig, UpriseError, UpriseHit};

// ── helpers ─────────────────────────────────────────────────────────────────

/// Shorthand for constructing a valid exemplar in test setup.
fn ex(task_label: &str, prompt_text: &str, outcome_quality: f64) -> PromptExemplar {
    PromptExemplar::new(task_label, prompt_text, outcome_quality).expect("valid exemplar")
}

fn approx_eq(a: f64, b: f64) -> bool {
    (a - b).abs() < 1e-9
}

/// A four-exemplar pool spanning four distinct tasks, used by the cross-task
/// retrieval tests below. `fact_checking` is near-duplicate text of the
/// canonical query; `qa` is lexically related but from a different task;
/// `sentiment_classification` and `translation` are near-disjoint decoys with
/// deliberately high outcome quality (to prove outcome weighting does not
/// override an overwhelming similarity gap).
fn cross_task_pool() -> UpriseIndex {
    let mut index = UpriseIndex::new();
    index.add_many([
        ex(
            "qa",
            "Identify factual errors and inconsistencies between the generated answer and the source document.",
            0.85,
        ),
        ex(
            "translation",
            "Translate the following sentence from English to Spanish accurately.",
            0.99,
        ),
        ex(
            "sentiment_classification",
            "Classify the sentiment of this tweet as positive or negative.",
            0.99,
        ),
        ex(
            "fact_checking",
            "Identify factual errors and inconsistencies in this AI-generated summary of a news article, flagging any unsupported claims.",
            0.9,
        ),
    ]);
    index
}

const CROSS_TASK_QUERY: &str =
    "Identify factual errors and inconsistencies in this AI-generated summary of a news article.";

// ── PromptExemplar ────────────────────────────────────────────────────────────

#[test]
fn test_prompt_exemplar_new_valid() {
    let e = PromptExemplar::new("qa", "Answer the question.", 0.5).expect("valid");
    assert_eq!(e.task_label, "qa");
    assert_eq!(e.prompt_text, "Answer the question.");
    assert!(approx_eq(e.outcome_quality, 0.5));
    assert!(e.embedding.is_empty());
}

#[test]
fn test_prompt_exemplar_new_boundary_values_valid() {
    assert!(PromptExemplar::new("t", "p", 0.0).is_ok());
    assert!(PromptExemplar::new("t", "p", 1.0).is_ok());
}

#[test]
fn test_prompt_exemplar_new_rejects_negative_outcome() {
    let err = PromptExemplar::new("qa", "p", -0.01).expect_err("should error");
    assert_eq!(err, UpriseError::InvalidOutcomeQuality(-0.01));
}

#[test]
fn test_prompt_exemplar_new_rejects_outcome_above_one() {
    let err = PromptExemplar::new("qa", "p", 1.5).expect_err("should error");
    assert_eq!(err, UpriseError::InvalidOutcomeQuality(1.5));
}

#[test]
fn test_prompt_exemplar_new_rejects_nan_outcome() {
    let err = PromptExemplar::new("qa", "p", f64::NAN).expect_err("should error");
    assert!(matches!(err, UpriseError::InvalidOutcomeQuality(v) if v.is_nan()));
}

#[test]
fn test_prompt_exemplar_new_rejects_infinite_outcome() {
    let err = PromptExemplar::new("qa", "p", f64::INFINITY).expect_err("should error");
    assert!(matches!(err, UpriseError::InvalidOutcomeQuality(v) if v.is_infinite()));
}

#[test]
fn test_prompt_exemplar_with_embedding_builder() {
    let e = ex("qa", "text", 0.5).with_embedding(vec![1.0, 0.0, 0.0]);
    assert_eq!(e.embedding, vec![1.0, 0.0, 0.0]);
}

// ── UpriseIndex: pool management ──────────────────────────────────────────────

#[test]
fn test_index_new_is_empty() {
    let index = UpriseIndex::new();
    assert!(index.is_empty());
    assert_eq!(index.len(), 0);
}

#[test]
fn test_index_default_is_empty() {
    assert!(UpriseIndex::default().is_empty());
}

#[test]
fn test_index_add_increases_len() {
    let mut index = UpriseIndex::new();
    index.add(ex("qa", "answer this", 0.5));
    assert_eq!(index.len(), 1);
    assert!(!index.is_empty());
}

#[test]
fn test_index_add_many() {
    let mut index = UpriseIndex::new();
    index.add_many([ex("qa", "a", 0.1), ex("summarization", "b", 0.2)]);
    assert_eq!(index.len(), 2);
}

#[test]
fn test_index_task_labels_sorted_and_deduped() {
    let mut index = UpriseIndex::new();
    index.add_many([
        ex("summarization", "a", 0.1),
        ex("qa", "b", 0.2),
        ex("qa", "c", 0.3),
    ]);
    assert_eq!(index.task_labels(), vec!["qa", "summarization"]);
}

#[test]
fn test_index_task_labels_empty_pool() {
    assert!(UpriseIndex::new().task_labels().is_empty());
}

// ── UpriseIndex: EMA outcome update ───────────────────────────────────────────

#[test]
fn test_update_outcome_single_step_matches_ema_formula() {
    let mut index = UpriseIndex::new();
    index.add(ex("qa", "p", 0.5));
    // new_quality = alpha * new_signal + (1 - alpha) * old_quality
    //             = 0.2 * 1.0 + 0.8 * 0.5 = 0.6
    index.update_outcome(0, 1.0, 0.2).expect("valid update");
    assert!(approx_eq(index.exemplars[0].outcome_quality, 0.6));
}

#[test]
fn test_update_outcome_repeated_feedback_converges_geometrically() {
    let mut index = UpriseIndex::new();
    index.add(ex("qa", "p", 0.5));
    let alpha = 0.2;
    let expected = [0.6, 0.68, 0.744];
    for exp in expected {
        index.update_outcome(0, 1.0, alpha).expect("valid update");
        assert!(
            approx_eq(index.exemplars[0].outcome_quality, exp),
            "got {}, expected {exp}",
            index.exemplars[0].outcome_quality
        );
    }
}

#[test]
fn test_update_outcome_repeated_low_feedback_decays_towards_zero() {
    let mut index = UpriseIndex::new();
    index.add(ex("qa", "p", 0.9));
    for _ in 0..50 {
        index.update_outcome(0, 0.0, 0.3).expect("valid update");
    }
    // Repeated zero-signal feedback should drive the EMA arbitrarily close to 0.
    assert!(index.exemplars[0].outcome_quality < 1e-6);
}

#[test]
fn test_update_outcome_ema_alpha_one_overwrites_immediately() {
    let mut index = UpriseIndex::new();
    index.add(ex("qa", "p", 0.1));
    index.update_outcome(0, 0.9, 1.0).expect("valid update");
    assert!(approx_eq(index.exemplars[0].outcome_quality, 0.9));
}

#[test]
fn test_update_outcome_rejects_invalid_new_signal() {
    let mut index = UpriseIndex::new();
    index.add(ex("qa", "p", 0.5));
    let err = index.update_outcome(0, 1.5, 0.2).expect_err("should error");
    assert_eq!(err, UpriseError::InvalidOutcomeQuality(1.5));
    // The stored quality must be untouched on error.
    assert!(approx_eq(index.exemplars[0].outcome_quality, 0.5));
}

#[test]
fn test_update_outcome_rejects_zero_alpha() {
    let mut index = UpriseIndex::new();
    index.add(ex("qa", "p", 0.5));
    let err = index.update_outcome(0, 0.8, 0.0).expect_err("should error");
    assert_eq!(err, UpriseError::InvalidEmaAlpha(0.0));
}

#[test]
fn test_update_outcome_rejects_alpha_above_one() {
    let mut index = UpriseIndex::new();
    index.add(ex("qa", "p", 0.5));
    let err = index.update_outcome(0, 0.8, 1.1).expect_err("should error");
    assert_eq!(err, UpriseError::InvalidEmaAlpha(1.1));
}

#[test]
fn test_update_outcome_rejects_nan_alpha() {
    let mut index = UpriseIndex::new();
    index.add(ex("qa", "p", 0.5));
    let err = index
        .update_outcome(0, 0.8, f64::NAN)
        .expect_err("should error");
    assert!(matches!(err, UpriseError::InvalidEmaAlpha(v) if v.is_nan()));
}

#[test]
fn test_update_outcome_index_out_of_bounds() {
    let mut index = UpriseIndex::new();
    index.add(ex("qa", "p", 0.5));
    let err = index.update_outcome(5, 0.8, 0.2).expect_err("should error");
    assert_eq!(err, UpriseError::IndexOutOfBounds(5));
}

#[test]
fn test_update_outcome_for_locates_by_content_and_updates() {
    let mut index = UpriseIndex::new();
    index.add(ex("qa", "answer concisely", 0.5));
    index
        .update_outcome_for("qa", "answer concisely", 1.0, 0.5)
        .expect("found and updated");
    assert!(approx_eq(index.exemplars[0].outcome_quality, 0.75));
}

#[test]
fn test_update_outcome_for_no_match_errors_with_pool_len() {
    let mut index = UpriseIndex::new();
    index.add(ex("qa", "answer concisely", 0.5));
    let err = index
        .update_outcome_for("qa", "nonexistent text", 1.0, 0.5)
        .expect_err("should error");
    assert_eq!(err, UpriseError::IndexOutOfBounds(1));
}

// ── UpriseConfig ──────────────────────────────────────────────────────────────

#[test]
fn test_config_defaults() {
    let cfg = UpriseConfig::default();
    assert_eq!(cfg.top_k, 5);
    assert!(approx_eq(cfg.similarity_weight, 0.6));
    assert!(approx_eq(cfg.outcome_weight, 0.4));
    assert!(approx_eq(cfg.ema_alpha, 0.3));
    assert_eq!(cfg.dim, 128);
}

#[test]
fn test_config_builder_chain() {
    let cfg = UpriseConfig::default()
        .with_top_k(3)
        .with_similarity_weight(0.7)
        .with_outcome_weight(0.3)
        .with_ema_alpha(0.5)
        .with_dim(64);
    assert_eq!(cfg.top_k, 3);
    assert!(approx_eq(cfg.similarity_weight, 0.7));
    assert!(approx_eq(cfg.outcome_weight, 0.3));
    assert!(approx_eq(cfg.ema_alpha, 0.5));
    assert_eq!(cfg.dim, 64);
}

#[test]
fn test_config_combined_score_formula() {
    let cfg = UpriseConfig::default()
        .with_similarity_weight(0.6)
        .with_outcome_weight(0.4);
    let score = cfg.combined_score(0.8, 0.5);
    assert!(approx_eq(score, 0.6 * 0.8 + 0.4 * 0.5));
}

// ── UpriseRetriever: error handling / empty index ─────────────────────────────

#[test]
fn test_retrieve_on_empty_index_errors() {
    let retriever = UpriseRetriever::new(UpriseConfig::default());
    let index = UpriseIndex::new();
    let err = retriever
        .retrieve("anything", &index)
        .expect_err("should error");
    assert_eq!(err, UpriseError::EmptyIndex);
}

#[test]
fn test_retrieve_empty_query_errors() {
    let retriever = UpriseRetriever::new(UpriseConfig::default());
    let index = cross_task_pool();
    let err = retriever.retrieve("", &index).expect_err("should error");
    assert_eq!(err, UpriseError::EmptyQuery);
}

#[test]
fn test_retrieve_whitespace_only_query_errors() {
    let retriever = UpriseRetriever::new(UpriseConfig::default());
    let index = cross_task_pool();
    let err = retriever
        .retrieve("   \t  ", &index)
        .expect_err("should error");
    assert_eq!(err, UpriseError::EmptyQuery);
}

#[test]
fn test_rank_by_similarity_empty_index_errors() {
    let retriever = UpriseRetriever::new(UpriseConfig::default());
    let index = UpriseIndex::new();
    let err = retriever
        .rank_by_similarity("anything", &index)
        .expect_err("should error");
    assert_eq!(err, UpriseError::EmptyIndex);
}

#[test]
fn test_retrieve_excluding_task_that_removes_everything_errors() {
    let retriever = UpriseRetriever::new(UpriseConfig::default());
    let mut index = UpriseIndex::new();
    index.add(ex("qa", "answer the question", 0.5));
    let err = retriever
        .retrieve_excluding_task("a question", &index, "qa")
        .expect_err("should error");
    assert_eq!(err, UpriseError::NoEligibleExemplars);
}

#[test]
fn test_retrieve_filtered_custom_predicate() {
    let retriever = UpriseRetriever::new(UpriseConfig::default());
    let index = cross_task_pool();
    let hits = retriever
        .retrieve_filtered(CROSS_TASK_QUERY, &index, |e| e.outcome_quality > 0.95)
        .expect("ok");
    assert!(hits.iter().all(|h| {
        index
            .exemplars
            .iter()
            .any(|e| e.task_label == h.task_label && e.outcome_quality > 0.95)
    }));
    assert!(!hits.is_empty());
}

// ── UpriseRetriever: cross-task retrieval (centerpiece) ───────────────────────

#[test]
fn test_cross_task_retrieval_finds_exemplar_from_unrelated_task_label() {
    // The query describes an unseen "fact_checking"-flavoured task, but the
    // pool (minus the exact "fact_checking" match) only offers a "qa"-labelled
    // exemplar that is lexically related. It must still be surfaced as the top
    // hit once the same-labelled exemplar is excluded.
    let retriever = UpriseRetriever::new(UpriseConfig::default().with_top_k(1));
    let index = cross_task_pool();
    let hits = retriever
        .retrieve_excluding_task(CROSS_TASK_QUERY, &index, "fact_checking")
        .expect("ok");
    assert_eq!(hits.len(), 1);
    assert_eq!(hits[0].task_label, "qa");
}

#[test]
fn test_cross_task_retrieval_ranks_relevant_task_above_high_outcome_decoys() {
    // "translation" and "sentiment_classification" carry a near-maximal
    // outcome_quality (0.99) but share essentially no vocabulary with the
    // query; outcome weighting alone must not let them beat a lexically
    // relevant cross-task exemplar with a comparatively modest outcome score.
    let retriever = UpriseRetriever::new(UpriseConfig::default().with_top_k(4));
    let index = cross_task_pool();
    let hits = retriever.retrieve(CROSS_TASK_QUERY, &index).expect("ok");
    assert_eq!(hits.len(), 4);
    let rank_of = |label: &str| {
        hits.iter()
            .position(|h| h.task_label == label)
            .expect("present")
    };
    assert!(rank_of("qa") < rank_of("translation"));
    assert!(rank_of("qa") < rank_of("sentiment_classification"));
}

#[test]
fn test_cross_task_retrieval_all_tasks_eligible_by_default() {
    // Without exclusion, the near-duplicate "fact_checking" exemplar wins
    // outright (both highest similarity and high outcome) — proving retrieval
    // is not gated by task_label at all; it happens to also be same-labelled
    // here only because we didn't ask for exclusion.
    let retriever = UpriseRetriever::new(UpriseConfig::default().with_top_k(1));
    let index = cross_task_pool();
    let hits = retriever.retrieve(CROSS_TASK_QUERY, &index).expect("ok");
    assert_eq!(hits[0].task_label, "fact_checking");
}

// ── UpriseRetriever: outcome-weighted reranking (centerpiece) ─────────────────

#[test]
fn test_outcome_weighted_reranking_flips_pure_similarity_order() {
    // Exemplar A is near-duplicate wording of the query (very high similarity)
    // but has a poor historical track record. Exemplar B is only moderately
    // similar (different task, overlapping vocabulary) but has an excellent
    // track record. Outcome-weighted reranking must promote B above A even
    // though A is the more similar candidate.
    let query = "Classify the sentiment of this movie review as positive, negative, or neutral.";
    let mut index = UpriseIndex::new();
    index.add(ex(
        "sentiment_classification",
        "Classify the sentiment of this movie review as positive, negative, or neutral sentiment analysis task.",
        0.05,
    ));
    index.add(ex(
        "summarization",
        "Summarize the key sentiment and emotional tone expressed in this passage about a movie.",
        0.95,
    ));

    let config = UpriseConfig::default()
        .with_top_k(2)
        .with_similarity_weight(0.6)
        .with_outcome_weight(0.4);
    let retriever = UpriseRetriever::new(config);

    // Pure-similarity order: the near-duplicate wins.
    let by_similarity = retriever.rank_by_similarity(query, &index).expect("ok");
    assert_eq!(by_similarity[0].task_label, "sentiment_classification");
    assert_eq!(by_similarity[1].task_label, "summarization");
    assert!(by_similarity[0].similarity > by_similarity[1].similarity);

    // Outcome-weighted order: the historically strong performer overtakes it.
    let reranked = retriever.retrieve(query, &index).expect("ok");
    assert_eq!(reranked[0].task_label, "summarization");
    assert_eq!(reranked[1].task_label, "sentiment_classification");
    assert!(reranked[0].combined_score > reranked[1].combined_score);

    // The flip is real: the winner by combined score is the loser by raw
    // similarity, and vice versa.
    assert_ne!(by_similarity[0].task_label, reranked[0].task_label);
}

#[test]
fn test_combined_score_matches_config_formula_for_each_hit() {
    let query = "Classify the sentiment of this movie review as positive, negative, or neutral.";
    let mut index = UpriseIndex::new();
    index.add(ex(
        "sentiment_classification",
        "Classify the sentiment of this movie review as positive, negative, or neutral sentiment analysis task.",
        0.05,
    ));
    index.add(ex(
        "summarization",
        "Summarize the key sentiment and emotional tone expressed in this passage about a movie.",
        0.95,
    ));
    let config = UpriseConfig::default();
    let retriever = UpriseRetriever::new(config.clone());
    for hit in retriever.retrieve(query, &index).expect("ok") {
        let expected = config.combined_score(hit.similarity, hit.outcome_quality);
        assert!(approx_eq(hit.combined_score, expected));
    }
}

#[test]
fn test_zero_outcome_weight_recovers_pure_similarity_order() {
    let query = "Classify the sentiment of this movie review as positive, negative, or neutral.";
    let mut index = UpriseIndex::new();
    index.add(ex(
        "sentiment_classification",
        "Classify the sentiment of this movie review as positive, negative, or neutral sentiment analysis task.",
        0.05,
    ));
    index.add(ex(
        "summarization",
        "Summarize the key sentiment and emotional tone expressed in this passage about a movie.",
        0.95,
    ));
    let config = UpriseConfig::default()
        .with_similarity_weight(1.0)
        .with_outcome_weight(0.0);
    let retriever = UpriseRetriever::new(config);
    let by_similarity = retriever.rank_by_similarity(query, &index).expect("ok");
    let by_combined = retriever.retrieve(query, &index).expect("ok");
    let sim_order: Vec<&str> = by_similarity
        .iter()
        .map(|h| h.task_label.as_str())
        .collect();
    let combined_order: Vec<&str> = by_combined.iter().map(|h| h.task_label.as_str()).collect();
    assert_eq!(sim_order, combined_order);
}

#[test]
fn test_zero_similarity_weight_ranks_purely_by_outcome_quality() {
    let query = "Classify the sentiment of this movie review as positive, negative, or neutral.";
    let mut index = UpriseIndex::new();
    index.add(ex(
        "sentiment_classification",
        "Classify the sentiment of this movie review as positive, negative, or neutral sentiment analysis task.",
        0.05,
    ));
    index.add(ex(
        "summarization",
        "Summarize the key sentiment and emotional tone expressed in this passage about a movie.",
        0.95,
    ));
    let config = UpriseConfig::default()
        .with_similarity_weight(0.0)
        .with_outcome_weight(1.0);
    let retriever = UpriseRetriever::new(config);
    let by_combined = retriever.retrieve(query, &index).expect("ok");
    // Purely by outcome_quality: 0.95 beats 0.05, regardless of similarity.
    assert_eq!(by_combined[0].task_label, "summarization");
    assert!(approx_eq(by_combined[0].combined_score, 0.95));
    assert!(approx_eq(by_combined[1].combined_score, 0.05));
}

// ── UpriseRetriever: top_k / truncation ───────────────────────────────────────

#[test]
fn test_top_k_truncates_results() {
    let retriever = UpriseRetriever::new(UpriseConfig::default().with_top_k(2));
    let index = cross_task_pool();
    let hits = retriever.retrieve(CROSS_TASK_QUERY, &index).expect("ok");
    assert_eq!(hits.len(), 2);
}

#[test]
fn test_top_k_zero_returns_empty_without_error() {
    let retriever = UpriseRetriever::new(UpriseConfig::default().with_top_k(0));
    let index = cross_task_pool();
    let hits = retriever.retrieve(CROSS_TASK_QUERY, &index).expect("ok");
    assert!(hits.is_empty());
}

#[test]
fn test_top_k_larger_than_pool_returns_all() {
    let retriever = UpriseRetriever::new(UpriseConfig::default().with_top_k(1000));
    let index = cross_task_pool();
    let hits = retriever.retrieve(CROSS_TASK_QUERY, &index).expect("ok");
    assert_eq!(hits.len(), index.len());
}

// ── UpriseRetriever: precomputed embeddings ───────────────────────────────────

#[test]
fn test_precomputed_embedding_of_matching_dim_overrides_recompute() {
    let config = UpriseConfig::default().with_dim(8);
    let retriever = UpriseRetriever::new(config.clone());

    let mut baseline_index = UpriseIndex::new();
    baseline_index.add(ex(
        "qa",
        "totally unrelated filler text about gardening",
        0.5,
    ));
    let baseline_hit = &retriever
        .rank_by_similarity("machine learning question answering", &baseline_index)
        .expect("ok")[0];

    let mut overridden_index = UpriseIndex::new();
    // Same prompt_text, but a forced embedding that is *not* what recomputing
    // from the text would produce.
    overridden_index.add(
        ex("qa", "totally unrelated filler text about gardening", 0.5)
            .with_embedding(vec![1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0]),
    );
    let overridden_hit = &retriever
        .rank_by_similarity("machine learning question answering", &overridden_index)
        .expect("ok")[0];

    assert_ne!(baseline_hit.similarity, overridden_hit.similarity);
}

#[test]
fn test_mismatched_dim_embedding_falls_back_to_recompute() {
    let config = UpriseConfig::default().with_dim(8);
    let retriever = UpriseRetriever::new(config.clone());

    let mut baseline_index = UpriseIndex::new();
    baseline_index.add(ex(
        "qa",
        "totally unrelated filler text about gardening",
        0.5,
    ));
    let baseline_hit = &retriever
        .rank_by_similarity("machine learning question answering", &baseline_index)
        .expect("ok")[0];

    let mut mismatched_index = UpriseIndex::new();
    // Wrong length (2, not 8) — must be ignored and recomputed from prompt_text.
    mismatched_index.add(
        ex("qa", "totally unrelated filler text about gardening", 0.5)
            .with_embedding(vec![1.0, 0.0]),
    );
    let mismatched_hit = &retriever
        .rank_by_similarity("machine learning question answering", &mismatched_index)
        .expect("ok")[0];

    assert!(approx_eq(
        baseline_hit.similarity,
        mismatched_hit.similarity
    ));
}

// ── determinism ───────────────────────────────────────────────────────────────

#[test]
fn test_retrieve_is_deterministic_across_repeated_calls() {
    let retriever = UpriseRetriever::new(UpriseConfig::default().with_top_k(4));
    let index = cross_task_pool();
    let first = retriever.retrieve(CROSS_TASK_QUERY, &index).expect("ok");
    for _ in 0..16 {
        let again = retriever.retrieve(CROSS_TASK_QUERY, &index).expect("ok");
        assert_eq!(first, again);
    }
}

// ── UpriseHit / struct sanity ──────────────────────────────────────────────────

#[test]
fn test_hit_struct_fields_populated_from_source_exemplar() {
    let retriever = UpriseRetriever::new(UpriseConfig::default().with_top_k(1));
    let mut index = UpriseIndex::new();
    index.add(ex("qa", "answer the question", 0.42));
    let hits = retriever
        .retrieve("answer the question", &index)
        .expect("ok");
    let hit: &UpriseHit = &hits[0];
    assert_eq!(hit.task_label, "qa");
    assert_eq!(hit.prompt_text, "answer the question");
    assert!(approx_eq(hit.outcome_quality, 0.42));
    assert!(hit.similarity > 0.99); // identical text ⇒ ~1.0 cosine similarity
}

// ── UpriseError Display ────────────────────────────────────────────────────────

#[test]
fn test_error_display_messages() {
    assert_eq!(
        UpriseError::EmptyQuery.to_string(),
        "query must not be empty"
    );
    assert_eq!(
        UpriseError::EmptyIndex.to_string(),
        "prompt exemplar index is empty"
    );
    assert_eq!(
        UpriseError::NoEligibleExemplars.to_string(),
        "no eligible exemplars remain after filtering"
    );
    assert_eq!(
        UpriseError::InvalidOutcomeQuality(1.5).to_string(),
        "outcome quality must be finite and within [0.0, 1.0], got 1.5"
    );
    assert_eq!(
        UpriseError::InvalidEmaAlpha(0.0).to_string(),
        "ema_alpha must be finite and within (0.0, 1.0], got 0"
    );
    assert_eq!(
        UpriseError::IndexOutOfBounds(3).to_string(),
        "exemplar index 3 out of bounds"
    );
}
