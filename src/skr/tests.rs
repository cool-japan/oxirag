use crate::skr::engine::{SelfKnowledgePool, SkrGate};
use crate::skr::types::{
    SkrConfig, SkrDecision, SkrError, SkrExemplar, SkrNeighbor, SkrRetrievalChoice,
};

// ── SkrRetrievalChoice ────────────────────────────────────────────────────────

#[test]
fn retrieval_choice_as_str_and_display() {
    assert_eq!(SkrRetrievalChoice::Skip.as_str(), "skip");
    assert_eq!(SkrRetrievalChoice::Retrieve.as_str(), "retrieve");
    assert_eq!(SkrRetrievalChoice::Skip.to_string(), "skip");
    assert_eq!(SkrRetrievalChoice::Retrieve.to_string(), "retrieve");
}

#[test]
fn retrieval_choice_should_retrieve() {
    assert!(!SkrRetrievalChoice::Skip.should_retrieve());
    assert!(SkrRetrievalChoice::Retrieve.should_retrieve());
}

#[test]
fn retrieval_choice_default_is_retrieve() {
    assert_eq!(SkrRetrievalChoice::default(), SkrRetrievalChoice::Retrieve);
}

// ── SkrExemplar ───────────────────────────────────────────────────────────────

#[test]
fn exemplar_new_has_no_embedding() {
    let exemplar = SkrExemplar::new("q", true);
    assert_eq!(exemplar.question, "q");
    assert!(exemplar.answerable_without_retrieval);
    assert!(exemplar.embedding.is_empty());
}

#[test]
fn exemplar_with_embedding_attaches_vector() {
    let exemplar = SkrExemplar::new("q", false).with_embedding(vec![1.0, 2.0]);
    assert_eq!(exemplar.embedding, vec![1.0, 2.0]);
    assert!(!exemplar.answerable_without_retrieval);
}

// ── SkrDecision helpers ───────────────────────────────────────────────────────

#[test]
fn decision_fallback_helpers() {
    let fallback = SkrDecision {
        choice: SkrRetrievalChoice::Retrieve,
        known_score: 0.0,
        neighbors: Vec::new(),
        reason: "pool empty".to_string(),
    };
    assert!(fallback.is_fallback());
    assert!(fallback.closest_neighbor().is_none());
    assert!(fallback.should_retrieve());
}

#[test]
fn decision_non_fallback_helpers() {
    let neighbor = SkrNeighbor {
        question: "q".into(),
        answerable_without_retrieval: true,
        similarity: 0.9,
    };
    let decision = SkrDecision {
        choice: SkrRetrievalChoice::Skip,
        known_score: 0.9,
        neighbors: vec![neighbor.clone()],
        reason: "high score".to_string(),
    };
    assert!(!decision.is_fallback());
    assert_eq!(decision.closest_neighbor(), Some(&neighbor));
    assert!(!decision.should_retrieve());
}

// ── SkrConfig ─────────────────────────────────────────────────────────────────

#[test]
fn config_defaults() {
    let config = SkrConfig::default();
    assert_eq!(config.k, 5);
    assert!((config.decision_threshold - 0.5).abs() < f32::EPSILON);
    assert_eq!(config.default_choice, SkrRetrievalChoice::Retrieve);
    assert_eq!(config.embedding_dim, 128);
    assert_eq!(config.pool_cap, 500);
    assert!(config.weight_by_similarity);
    assert!(config.validate().is_ok());
}

#[test]
fn config_new_equals_default() {
    assert_eq!(SkrConfig::new(), SkrConfig::default());
}

#[test]
fn config_builders_set_every_field() {
    let config = SkrConfig::new()
        .with_k(7)
        .with_decision_threshold(0.75)
        .with_default_choice(SkrRetrievalChoice::Skip)
        .with_embedding_dim(64)
        .with_pool_cap(10)
        .with_weight_by_similarity(false);
    assert_eq!(config.k, 7);
    assert!((config.decision_threshold - 0.75).abs() < f32::EPSILON);
    assert_eq!(config.default_choice, SkrRetrievalChoice::Skip);
    assert_eq!(config.embedding_dim, 64);
    assert_eq!(config.pool_cap, 10);
    assert!(!config.weight_by_similarity);
    assert!(config.validate().is_ok());
}

#[test]
fn config_validate_rejects_zero_k() {
    let err = SkrConfig::new().with_k(0).validate().unwrap_err();
    assert!(matches!(err, SkrError::InvalidK));
}

#[test]
fn config_validate_rejects_zero_embedding_dim() {
    let err = SkrConfig::new()
        .with_embedding_dim(0)
        .validate()
        .unwrap_err();
    assert!(matches!(err, SkrError::InvalidEmbeddingDim));
}

#[test]
fn config_validate_rejects_out_of_range_threshold() {
    assert!(matches!(
        SkrConfig::new()
            .with_decision_threshold(-0.01)
            .validate()
            .unwrap_err(),
        SkrError::InvalidThreshold(_)
    ));
    assert!(matches!(
        SkrConfig::new()
            .with_decision_threshold(1.01)
            .validate()
            .unwrap_err(),
        SkrError::InvalidThreshold(_)
    ));
}

#[test]
fn config_validate_rejects_non_finite_threshold() {
    assert!(matches!(
        SkrConfig::new()
            .with_decision_threshold(f32::NAN)
            .validate()
            .unwrap_err(),
        SkrError::InvalidThreshold(_)
    ));
    assert!(matches!(
        SkrConfig::new()
            .with_decision_threshold(f32::INFINITY)
            .validate()
            .unwrap_err(),
        SkrError::InvalidThreshold(_)
    ));
}

#[test]
fn config_validate_accepts_threshold_boundaries() {
    assert!(
        SkrConfig::new()
            .with_decision_threshold(0.0)
            .validate()
            .is_ok()
    );
    assert!(
        SkrConfig::new()
            .with_decision_threshold(1.0)
            .validate()
            .is_ok()
    );
}

// ── SkrError ──────────────────────────────────────────────────────────────────

#[test]
fn error_messages_are_specific() {
    assert!(SkrError::EmptyQuestion.to_string().contains("empty"));
    assert!(SkrError::InvalidK.to_string().contains("neighbor"));
    assert!(
        SkrError::InvalidEmbeddingDim
            .to_string()
            .contains("dimension")
    );
    let msg = SkrError::InvalidThreshold(1.5).to_string();
    assert!(msg.contains("1.5"));
}

#[test]
fn error_is_clone_and_partial_eq() {
    let a = SkrError::InvalidK;
    let b = a.clone();
    assert_eq!(a, b);
    assert_ne!(SkrError::InvalidK, SkrError::EmptyQuestion);
}

// ── SelfKnowledgePool: basics ─────────────────────────────────────────────────

#[test]
fn pool_new_is_empty() {
    let pool = SelfKnowledgePool::new();
    assert!(pool.is_empty());
    assert_eq!(pool.len(), 0);
    assert_eq!(pool.known_count(), 0);
    assert_eq!(pool.unknown_count(), 0);
}

#[test]
fn pool_default_is_empty() {
    let pool = SelfKnowledgePool::default();
    assert!(pool.is_empty());
}

#[test]
fn pool_add_labeled_and_observe_outcome_grow_counts() {
    let config = SkrConfig::default();
    let mut pool = SelfKnowledgePool::new();
    pool.add_labeled("q1", true, &config);
    pool.add_labeled("q2", false, &config);
    pool.observe_outcome("q3", true, &config);

    assert_eq!(pool.len(), 3);
    assert_eq!(pool.known_count(), 2);
    assert_eq!(pool.unknown_count(), 1);
    assert!(!pool.is_empty());
}

// ── SelfKnowledgePool: embedding computation ─────────────────────────────────

#[test]
fn pool_add_exemplar_computes_embedding_when_missing() {
    let config = SkrConfig::new().with_embedding_dim(32);
    let mut pool = SelfKnowledgePool::new();
    pool.add_exemplar(SkrExemplar::new("hello world", true), &config);
    assert_eq!(pool.exemplars[0].embedding.len(), 32);
}

#[test]
fn pool_add_exemplar_preserves_matching_precomputed_embedding() {
    let config = SkrConfig::new().with_embedding_dim(3);
    let mut pool = SelfKnowledgePool::new();
    let custom = vec![0.1_f32, 0.2, 0.3];
    pool.add_exemplar(
        SkrExemplar::new("hello", true).with_embedding(custom.clone()),
        &config,
    );
    assert_eq!(pool.exemplars[0].embedding, custom);
}

#[test]
fn pool_add_exemplar_recomputes_mismatched_precomputed_embedding() {
    let config = SkrConfig::new().with_embedding_dim(16);
    let mut pool = SelfKnowledgePool::new();
    pool.add_exemplar(
        SkrExemplar::new("hello", true).with_embedding(vec![1.0, 2.0]),
        &config,
    );
    assert_eq!(pool.exemplars[0].embedding.len(), 16);
}

// ── SelfKnowledgePool: cap eviction ───────────────────────────────────────────

#[test]
fn pool_cap_evicts_oldest_first() {
    let config = SkrConfig::new().with_pool_cap(2);
    let mut pool = SelfKnowledgePool::new();
    pool.add_labeled("first", true, &config);
    pool.add_labeled("second", false, &config);
    pool.add_labeled("third", true, &config);

    assert_eq!(pool.len(), 2);
    let questions: Vec<&str> = pool.exemplars.iter().map(|e| e.question.as_str()).collect();
    assert_eq!(questions, vec!["second", "third"]);
}

#[test]
fn pool_cap_zero_is_unbounded() {
    let config = SkrConfig::new().with_pool_cap(0);
    let mut pool = SelfKnowledgePool::new();
    for i in 0..50 {
        pool.add_labeled(format!("question {i}"), i % 2 == 0, &config);
    }
    assert_eq!(pool.len(), 50);
}

// ── SkrGate: construction ─────────────────────────────────────────────────────

#[test]
fn gate_new_accepts_valid_config() {
    let gate = SkrGate::new(SkrConfig::default()).unwrap();
    assert!(gate.pool.is_empty());
}

#[test]
fn gate_new_rejects_zero_k() {
    let err = SkrGate::new(SkrConfig::new().with_k(0)).unwrap_err();
    assert!(matches!(err, SkrError::InvalidK));
}

#[test]
fn gate_new_rejects_zero_embedding_dim() {
    let err = SkrGate::new(SkrConfig::new().with_embedding_dim(0)).unwrap_err();
    assert!(matches!(err, SkrError::InvalidEmbeddingDim));
}

#[test]
fn gate_new_rejects_invalid_threshold() {
    let err = SkrGate::new(SkrConfig::new().with_decision_threshold(1.5)).unwrap_err();
    assert!(matches!(err, SkrError::InvalidThreshold(_)));

    let err_nan = SkrGate::new(SkrConfig::new().with_decision_threshold(f32::NAN)).unwrap_err();
    assert!(matches!(err_nan, SkrError::InvalidThreshold(_)));

    let err_neg = SkrGate::new(SkrConfig::new().with_decision_threshold(-0.1)).unwrap_err();
    assert!(matches!(err_neg, SkrError::InvalidThreshold(_)));
}

#[test]
fn gate_default_is_valid_and_empty() {
    let gate = SkrGate::default();
    assert_eq!(gate.config, SkrConfig::default());
    assert!(gate.pool.is_empty());
    let decision = gate.decide("any question").unwrap();
    assert!(decision.is_fallback());
}

#[test]
fn gate_with_pool_seeds_existing_pool() {
    let mut pool = SelfKnowledgePool::new();
    let config = SkrConfig::default();
    pool.add_labeled("seeded question", true, &config);

    let gate = SkrGate::with_pool(config, pool).unwrap();
    assert_eq!(gate.pool.len(), 1);
}

#[test]
fn gate_with_pool_rejects_invalid_config() {
    let pool = SelfKnowledgePool::new();
    let err = SkrGate::with_pool(SkrConfig::new().with_k(0), pool).unwrap_err();
    assert!(matches!(err, SkrError::InvalidK));
}

// ── SkrGate: pool maintenance ─────────────────────────────────────────────────

#[test]
fn gate_add_exemplar_direct() {
    let mut gate = SkrGate::new(SkrConfig::default()).unwrap();
    gate.add_exemplar(SkrExemplar::new("direct question", true));
    assert_eq!(gate.pool.len(), 1);
    assert_eq!(gate.pool.exemplars[0].question, "direct question");
}

#[test]
fn gate_add_labeled_and_observe_outcome_grow_pool() {
    let mut gate = SkrGate::new(SkrConfig::default()).unwrap();
    gate.add_labeled("labeled question", true).unwrap();
    gate.observe_outcome("observed question", false).unwrap();
    assert_eq!(gate.pool.len(), 2);
    assert_eq!(gate.pool.known_count(), 1);
    assert_eq!(gate.pool.unknown_count(), 1);
}

#[test]
fn add_labeled_and_observe_outcome_reject_empty_question() {
    let mut gate = SkrGate::new(SkrConfig::default()).unwrap();
    assert!(matches!(
        gate.add_labeled("  ", true).unwrap_err(),
        SkrError::EmptyQuestion
    ));
    assert!(matches!(
        gate.observe_outcome("", false).unwrap_err(),
        SkrError::EmptyQuestion
    ));
    assert!(gate.pool.is_empty());
}

#[test]
fn pool_cap_eviction_via_gate() {
    let mut gate = SkrGate::new(SkrConfig::new().with_pool_cap(3)).unwrap();
    for i in 0..10 {
        gate.add_labeled(format!("q{i}"), true).unwrap();
    }
    assert_eq!(gate.pool.len(), 3);
    let questions: Vec<&str> = gate
        .pool
        .exemplars
        .iter()
        .map(|e| e.question.as_str())
        .collect();
    assert_eq!(questions, vec!["q7", "q8", "q9"]);
}

// ── SkrGate: k_nearest ────────────────────────────────────────────────────────

#[test]
fn k_nearest_rejects_empty_question() {
    let gate = SkrGate::new(SkrConfig::default()).unwrap();
    assert!(matches!(
        gate.k_nearest("   ").unwrap_err(),
        SkrError::EmptyQuestion
    ));
}

#[test]
fn k_nearest_on_empty_pool_returns_empty_vec_not_error() {
    let gate = SkrGate::new(SkrConfig::default()).unwrap();
    let neighbors = gate.k_nearest("some question").unwrap();
    assert!(neighbors.is_empty());
}

#[test]
fn k_nearest_picks_exact_match_as_closest() {
    let mut pool = SelfKnowledgePool::new();
    let config = SkrConfig::default();
    pool.add_labeled("What is the capital of France?", true, &config);
    pool.add_labeled("How do I bake a chocolate cake?", false, &config);
    pool.add_labeled("What year did the French revolution begin?", false, &config);

    let gate = SkrGate::with_pool(config.with_k(1), pool).unwrap();
    let neighbors = gate.k_nearest("What is the capital of France?").unwrap();

    assert_eq!(neighbors.len(), 1);
    assert_eq!(neighbors[0].question, "What is the capital of France?");
    assert!((neighbors[0].similarity - 1.0).abs() < 1e-5);
    assert!(neighbors[0].answerable_without_retrieval);
}

#[test]
fn k_nearest_returns_k_ranked_by_similarity_desc() {
    let query = "distinct filler question about widgets";
    let mut pool = SelfKnowledgePool::new();
    let config = SkrConfig::default();
    pool.add_labeled(query, true, &config);
    pool.add_labeled(
        "totally unrelated text about gardening tools",
        false,
        &config,
    );
    pool.add_labeled(
        "another unrelated sentence about kitchen utensils",
        false,
        &config,
    );
    pool.add_labeled(
        "yet another distinct topic entirely about vehicles",
        true,
        &config,
    );

    let gate = SkrGate::with_pool(config.with_k(3), pool).unwrap();
    let neighbors = gate.k_nearest(query).unwrap();

    assert_eq!(neighbors.len(), 3);
    assert_eq!(neighbors[0].question, query);
    assert!((neighbors[0].similarity - 1.0).abs() < 1e-5);
    for pair in neighbors.windows(2) {
        assert!(pair[0].similarity >= pair[1].similarity);
    }
}

#[test]
fn k_nearest_truncates_to_pool_size_when_k_exceeds_it() {
    let mut pool = SelfKnowledgePool::new();
    let config = SkrConfig::default();
    pool.add_labeled("question alpha", true, &config);
    pool.add_labeled("question beta", false, &config);

    let gate = SkrGate::with_pool(config.with_k(10), pool).unwrap();
    let neighbors = gate.k_nearest("question alpha").unwrap();
    assert_eq!(neighbors.len(), 2);
}

#[test]
fn decide_neighbors_match_k_nearest() {
    let mut gate = SkrGate::new(SkrConfig::new().with_k(2)).unwrap();
    gate.add_labeled("alpha question", true).unwrap();
    gate.add_labeled("beta question", false).unwrap();
    gate.add_labeled("gamma question", true).unwrap();

    let decision = gate.decide("alpha question").unwrap();
    let neighbors = gate.k_nearest("alpha question").unwrap();
    assert_eq!(decision.neighbors, neighbors);
}

// ── SkrGate::decide: error paths ──────────────────────────────────────────────

#[test]
fn decide_rejects_empty_question() {
    let gate = SkrGate::new(SkrConfig::default()).unwrap();
    assert!(matches!(
        gate.decide("   ").unwrap_err(),
        SkrError::EmptyQuestion
    ));
    assert!(matches!(
        gate.decide("").unwrap_err(),
        SkrError::EmptyQuestion
    ));
}

#[test]
fn decide_surfaces_invalid_config_even_with_populated_pool() {
    let mut gate = SkrGate::new(SkrConfig::default()).unwrap();
    gate.add_labeled("some question", true).unwrap();
    gate.config.k = 0; // mutate to an invalid state after construction
    let err = gate.decide("some question").unwrap_err();
    assert!(matches!(err, SkrError::InvalidK));
}

// ── SkrGate::decide: empty pool fallback ──────────────────────────────────────

#[test]
fn empty_pool_falls_back_to_default_choice_retrieve() {
    let gate = SkrGate::new(SkrConfig::default()).unwrap();
    assert!(gate.pool.is_empty());
    let decision = gate.decide("any question").unwrap();
    assert_eq!(decision.choice, SkrRetrievalChoice::Retrieve);
    assert!(decision.known_score.abs() < 1e-9);
    assert!(decision.neighbors.is_empty());
    assert!(decision.is_fallback());
}

#[test]
fn empty_pool_falls_back_to_configured_default_choice_skip() {
    let gate =
        SkrGate::new(SkrConfig::new().with_default_choice(SkrRetrievalChoice::Skip)).unwrap();
    let decision = gate.decide("any question").unwrap();
    assert_eq!(decision.choice, SkrRetrievalChoice::Skip);
    assert!(decision.is_fallback());
}

// ── SkrGate::decide: known / unknown clusters ─────────────────────────────────

#[test]
fn question_near_known_cluster_skips_retrieval() {
    let mut gate = SkrGate::new(SkrConfig::new().with_k(3)).unwrap();
    gate.add_labeled("What is the capital of France?", true)
        .unwrap();
    gate.add_labeled("What is the capital of Germany?", true)
        .unwrap();
    gate.add_labeled("What is the capital of Italy?", true)
        .unwrap();
    gate.add_labeled("What is the internal budget of Acme Corp for Q3?", false)
        .unwrap();

    let decision = gate.decide("What is the capital of Spain?").unwrap();
    assert_eq!(decision.choice, SkrRetrievalChoice::Skip);
    assert!(decision.known_score > 0.5);
    assert!(decision.reason.contains("skip"));
}

#[test]
fn question_near_unknown_cluster_requires_retrieval() {
    let mut gate = SkrGate::new(SkrConfig::new().with_k(3)).unwrap();
    gate.add_labeled("What was Acme Corp's internal Q1 revenue?", false)
        .unwrap();
    gate.add_labeled("What was Acme Corp's internal Q2 revenue?", false)
        .unwrap();
    gate.add_labeled("What was Acme Corp's internal Q3 revenue?", false)
        .unwrap();
    gate.add_labeled("What is the capital of France?", true)
        .unwrap();

    let decision = gate
        .decide("What was Acme Corp's internal Q4 revenue?")
        .unwrap();
    assert_eq!(decision.choice, SkrRetrievalChoice::Retrieve);
    assert!(decision.known_score < 0.5);
    assert!(decision.reason.contains("retrieve"));
}

// ── SkrGate::decide: similarity weighting & threshold ─────────────────────────

#[test]
fn similarity_weighting_makes_closer_exemplar_count_more() {
    let query = "What is the capital of France?";
    let mut pool = SelfKnowledgePool::new();
    let config = SkrConfig::new().with_k(2);
    pool.add_labeled(query, true, &config);
    pool.add_labeled(
        "A completely different sentence about zoo animal feeding schedules",
        false,
        &config,
    );
    // k == pool.len(), so both exemplars are always the "k nearest"
    // regardless of ranking, but their similarity magnitudes still differ:
    // the first is an exact textual match to the query (similarity == 1.0);
    // the second is unrelated text, whose similarity is provably < 1.0.

    let weighted =
        SkrGate::with_pool(config.clone().with_weight_by_similarity(true), pool.clone()).unwrap();
    let unweighted = SkrGate::with_pool(config.with_weight_by_similarity(false), pool).unwrap();

    let weighted_decision = weighted.decide(query).unwrap();
    let unweighted_decision = unweighted.decide(query).unwrap();

    // Unweighted: exactly 1 positive out of 2 neighbors, regardless of
    // similarity magnitude.
    assert!((unweighted_decision.known_score - 0.5).abs() < 1e-6);
    // Weighted: the exact-match (similarity 1.0) positive neighbor
    // outweighs the more-distant negative one, so the weighted score must
    // strictly exceed the unweighted 50/50 split.
    assert!(
        weighted_decision.known_score > unweighted_decision.known_score,
        "weighted={}, unweighted={}",
        weighted_decision.known_score,
        unweighted_decision.known_score
    );
}

#[test]
fn unweighted_majority_vote_counts_labels_equally() {
    let config = SkrConfig::new().with_k(4).with_weight_by_similarity(false);
    let mut pool = SelfKnowledgePool::new();
    pool.add_labeled("a", true, &config);
    pool.add_labeled("b", true, &config);
    pool.add_labeled("c", true, &config);
    pool.add_labeled("d", false, &config);
    // k == pool.len() == 4, so all four are always consulted regardless of
    // similarity ranking: 3 positive / 4 total = 0.75, exactly.
    let gate = SkrGate::with_pool(config, pool).unwrap();
    let decision = gate.decide("anything").unwrap();
    assert!((decision.known_score - 0.75).abs() < 1e-6);
}

#[test]
fn negative_similarity_neighbor_is_clamped_to_zero_weight() {
    let dim = 8;
    let config = SkrConfig::new().with_k(2).with_embedding_dim(dim);
    let mut pool = SelfKnowledgePool::new();
    // Adversarial "positive" exemplar: an all-negative custom embedding, so
    // its cosine similarity to any naturally-embedded (non-negative
    // bucket-histogram) query is <= 0.
    pool.add_exemplar(
        SkrExemplar::new("adversarial embedding exemplar", true).with_embedding(vec![-1.0; dim]),
        &config,
    );
    // Genuine "negative" (unknown) exemplar, naturally embedded, an exact
    // textual match to the query so its similarity is a known 1.0.
    let query = "what is the capital of testland";
    pool.add_labeled(query, false, &config);

    let gate = SkrGate::with_pool(config, pool).unwrap();
    let decision = gate.decide(query).unwrap();

    // Without clamping, the adversarial negative-similarity "positive"
    // neighbor combined with a negative total weight could push the ratio
    // wildly outside [0.0, 1.0]. With clamping its vote weight floors at 0,
    // so it contributes nothing positive; since the only other neighbor is
    // labelled "unknown", the known score must be exactly 0.0.
    assert!(
        decision.known_score.abs() < 1e-9,
        "expected known_score == 0.0, got {}",
        decision.known_score
    );
    assert_eq!(decision.choice, SkrRetrievalChoice::Retrieve);
}

#[test]
fn raising_threshold_makes_gate_retrieve_more() {
    let mut pool = SelfKnowledgePool::new();
    let config = SkrConfig::new().with_k(2).with_weight_by_similarity(false);
    pool.add_labeled("some known question about topic A", true, &config);
    pool.add_labeled("some unknown question about topic B", false, &config);
    // pool has exactly 2 exemplars and k == 2, so both are always the "k
    // nearest" regardless of the query text or actual similarity values: the
    // unweighted majority vote is therefore exactly 1/2 = 0.5 for any query.

    let query = "an entirely different query about topic C";

    let low_threshold =
        SkrGate::with_pool(config.clone().with_decision_threshold(0.49), pool.clone()).unwrap();
    let high_threshold = SkrGate::with_pool(config.with_decision_threshold(0.51), pool).unwrap();

    let low_decision = low_threshold.decide(query).unwrap();
    let high_decision = high_threshold.decide(query).unwrap();

    assert!((low_decision.known_score - 0.5).abs() < 1e-6);
    assert!((high_decision.known_score - 0.5).abs() < 1e-6);
    assert_eq!(low_decision.choice, SkrRetrievalChoice::Skip);
    assert_eq!(high_decision.choice, SkrRetrievalChoice::Retrieve);
}

// ── SkrGate::observe_outcome: online learning ─────────────────────────────────

#[test]
fn observe_outcome_grows_pool_and_can_shift_decision() {
    let mut gate = SkrGate::new(
        SkrConfig::new()
            .with_k(2)
            .with_weight_by_similarity(false)
            .with_decision_threshold(0.5),
    )
    .unwrap();
    gate.add_labeled("first known question", true).unwrap();
    gate.add_labeled("second unknown question", false).unwrap();
    assert_eq!(gate.pool.len(), 2);

    let query = "a brand new query text";
    // k == pool.len() == 2, so both existing exemplars are always consulted:
    // unweighted score = 1/2 = 0.5 >= threshold 0.5 => Skip.
    let before = gate.decide(query).unwrap();
    assert_eq!(before.choice, SkrRetrievalChoice::Skip);
    assert!((before.known_score - 0.5).abs() < 1e-6);

    // Feed back two more negative outcomes, growing the pool from 2 to 4
    // exemplars. Raise k to match so every exemplar is still consulted,
    // keeping the vote fully deterministic.
    gate.observe_outcome("third unknown question", false)
        .unwrap();
    gate.observe_outcome("fourth unknown question", false)
        .unwrap();
    assert_eq!(gate.pool.len(), 4);
    gate.config.k = 4;

    let after = gate.decide(query).unwrap();
    // Now 1 positive out of 4 => 0.25, below the 0.5 threshold => Retrieve.
    assert!((after.known_score - 0.25).abs() < 1e-6);
    assert_eq!(after.choice, SkrRetrievalChoice::Retrieve);
}

// ── determinism ───────────────────────────────────────────────────────────────

#[test]
fn decide_is_deterministic_across_repeated_calls() {
    let mut gate = SkrGate::new(SkrConfig::default()).unwrap();
    gate.add_labeled("What is the capital of France?", true)
        .unwrap();
    gate.add_labeled("What is our internal roadmap?", false)
        .unwrap();

    let first = gate.decide("What is the capital of Spain?").unwrap();
    let second = gate.decide("What is the capital of Spain?").unwrap();
    assert_eq!(first, second);
}

#[test]
fn decide_is_deterministic_across_independently_built_gates() {
    fn build() -> SkrGate {
        let mut gate = SkrGate::new(SkrConfig::default()).unwrap();
        gate.add_labeled("What is the capital of France?", true)
            .unwrap();
        gate.add_labeled("What is our internal roadmap?", false)
            .unwrap();
        gate
    }
    let gate_a = build();
    let gate_b = build();

    let decision_a = gate_a.decide("What is the capital of Spain?").unwrap();
    let decision_b = gate_b.decide("What is the capital of Spain?").unwrap();
    assert_eq!(decision_a, decision_b);
}
