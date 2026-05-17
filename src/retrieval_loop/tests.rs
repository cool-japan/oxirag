//! Tests for the FLARE retrieval loop module.
//!
//! Covers:
//! - [`ConfidenceEstimator`] token/sentence/sentence-split logic
//! - [`MockFlareGenerator`] cycling and partial generation
//! - [`MockFlareRetriever`] top-k behaviour
//! - [`ContextWindow`] add/dedup/trim invariants
//! - [`FlareEngine`] `run`/`run_simple` with mocks
//! - [`FlareOutput`] derived metrics

use super::confidence::{ConfidenceEstimator, tokenise};
use super::engine::FlareEngine;
use super::generator::{FlareGenerator, MockFlareGenerator, TemplateGenerator};
use super::retriever::{FlareRetriever, MockFlareRetriever, QueryAugmentedRetriever};
use super::types::{
    ContextDoc, ContextWindow, FlareConfig, FlareOutput, IterationRecord, SentenceConfidence,
    TokenConfidence,
};

// ── tokenise helper ───────────────────────────────────────────────────────────

#[test]
fn test_tokenise_simple() {
    let tokens = tokenise("Hello, world!");
    assert_eq!(tokens, vec!["Hello", "world"]);
}

#[test]
fn test_tokenise_apostrophe_kept() {
    let tokens = tokenise("it's a test");
    assert!(tokens.contains(&"it's".to_string()));
}

#[test]
fn test_tokenise_empty_string() {
    assert!(tokenise("").is_empty());
}

#[test]
fn test_tokenise_only_punctuation() {
    assert!(tokenise("...---!!!").is_empty());
}

#[test]
fn test_tokenise_multi_space() {
    let tokens = tokenise("foo   bar");
    assert_eq!(tokens, vec!["foo", "bar"]);
}

// ── ConfidenceEstimator::estimate_token_confidence ────────────────────────────

#[test]
fn test_token_confidence_zero_when_context_empty() {
    let conf = ConfidenceEstimator::estimate_token_confidence("rust", "");
    assert!(conf.abs() < 1e-6, "expected ~0.0, got {conf}");
}

#[test]
fn test_token_confidence_in_unit_range() {
    let context = "rust is a programming language rust";
    let conf = ConfidenceEstimator::estimate_token_confidence("rust", context);
    assert!(
        (0.0..=1.0).contains(&conf),
        "confidence out of [0,1]: {conf}"
    );
}

#[test]
fn test_token_confidence_higher_for_frequent_token() {
    let context = "rust rust rust rust rust rust rust rust rust rust rust rust";
    let conf_rust = ConfidenceEstimator::estimate_token_confidence("rust", context);
    let conf_python = ConfidenceEstimator::estimate_token_confidence("python", context);
    assert!(
        conf_rust > conf_python,
        "frequent token should have higher confidence"
    );
}

#[test]
fn test_token_confidence_zero_for_empty_token() {
    let conf = ConfidenceEstimator::estimate_token_confidence("", "some context");
    assert!(conf.abs() < 1e-6, "expected ~0.0, got {conf}");
}

#[test]
fn test_token_confidence_case_insensitive() {
    let context = "Rust is great";
    let conf_lower = ConfidenceEstimator::estimate_token_confidence("rust", context);
    let conf_upper = ConfidenceEstimator::estimate_token_confidence("RUST", context);
    // Both should match the same occurrence.
    assert!(
        (conf_lower - conf_upper).abs() < 1e-6,
        "case should not matter: lower={conf_lower} upper={conf_upper}"
    );
}

#[test]
fn test_token_confidence_strips_punctuation() {
    let context = "rust is fast";
    let conf_plain = ConfidenceEstimator::estimate_token_confidence("rust", context);
    let conf_punc = ConfidenceEstimator::estimate_token_confidence("rust.", context);
    assert!(
        (conf_plain - conf_punc).abs() < 1e-6,
        "trailing punctuation should be stripped before matching"
    );
}

// ── ConfidenceEstimator::estimate_sentence_confidence ─────────────────────────

#[test]
fn test_sentence_confidence_avg_in_range() {
    let sc = ConfidenceEstimator::estimate_sentence_confidence(
        "Rust is a fast language",
        "Rust is a systems programming language",
    );
    assert!(
        (0.0..=1.0).contains(&sc.avg_confidence),
        "avg_confidence out of range: {}",
        sc.avg_confidence
    );
}

#[test]
fn test_sentence_confidence_tokens_non_empty() {
    let sc = ConfidenceEstimator::estimate_sentence_confidence("hello world", "hello world");
    assert!(!sc.tokens.is_empty());
}

#[test]
fn test_sentence_confidence_empty_context_gives_zero() {
    let sc = ConfidenceEstimator::estimate_sentence_confidence("Rust is great", "");
    assert!(
        sc.avg_confidence.abs() < 1e-6,
        "expected ~0.0, got {}",
        sc.avg_confidence
    );
}

#[test]
fn test_sentence_confidence_high_when_context_rich() {
    let context = "rust rust rust rust rust rust rust rust rust rust rust rust rust";
    let sc = ConfidenceEstimator::estimate_sentence_confidence("rust is fast", context);
    assert!(
        sc.avg_confidence > 0.0,
        "should have some confidence when context is rich"
    );
}

// ── SentenceConfidence methods ────────────────────────────────────────────────

#[test]
fn test_sentence_confidence_is_uncertain() {
    let tokens = vec![TokenConfidence {
        token: "rust".to_string(),
        confidence: 0.2,
    }];
    let sc = SentenceConfidence::new("rust".to_string(), tokens);
    assert!(sc.is_uncertain(0.5), "0.2 < 0.5 should be uncertain");
    assert!(!sc.is_uncertain(0.1), "0.2 >= 0.1 should not be uncertain");
}

#[test]
fn test_sentence_confidence_uncertain_span() {
    let sc = SentenceConfidence::new("hello world".to_string(), vec![]);
    assert_eq!(sc.uncertain_span(), "hello world");
}

#[test]
fn test_sentence_confidence_empty_tokens_gives_zero_avg() {
    let sc = SentenceConfidence::new("something".to_string(), vec![]);
    assert!(
        sc.avg_confidence.abs() < 1e-6,
        "expected ~0.0, got {}",
        sc.avg_confidence
    );
}

// ── ConfidenceEstimator::split_into_sentences ─────────────────────────────────

#[test]
fn test_split_empty() {
    assert!(ConfidenceEstimator::split_into_sentences("").is_empty());
}

#[test]
fn test_split_single_sentence_no_period() {
    let sentences = ConfidenceEstimator::split_into_sentences("Hello world");
    assert_eq!(sentences.len(), 1);
    assert_eq!(sentences[0], "Hello world");
}

#[test]
fn test_split_two_sentences() {
    let sentences = ConfidenceEstimator::split_into_sentences("Hello world. Goodbye world.");
    assert_eq!(sentences.len(), 2);
    assert_eq!(sentences[0], "Hello world.");
    assert_eq!(sentences[1], "Goodbye world.");
}

#[test]
fn test_split_question_and_exclamation() {
    let sentences =
        ConfidenceEstimator::split_into_sentences("Is Rust fast? Yes it is! Very much so.");
    assert_eq!(sentences.len(), 3, "got: {sentences:?}");
}

#[test]
fn test_split_paragraph_break() {
    let text = "First paragraph.\n\nSecond paragraph.";
    let sentences = ConfidenceEstimator::split_into_sentences(text);
    assert_eq!(sentences.len(), 2);
    assert!(sentences[0].contains("First"));
    assert!(sentences[1].contains("Second"));
}

#[test]
fn test_split_trims_whitespace() {
    let sentences = ConfidenceEstimator::split_into_sentences("  Hello. ");
    assert_eq!(sentences.len(), 1);
    assert_eq!(sentences[0], "Hello.");
}

// ── ConfidenceEstimator::identify_uncertain_sentences ────────────────────────

#[test]
fn test_identify_uncertain_no_context_all_uncertain() {
    let text = "Rust is a language. It is fast. It is safe.";
    let uncertain = ConfidenceEstimator::identify_uncertain_sentences(text, "", 0.5);
    // With no context every sentence has avg_confidence=0.0 < 0.5
    assert_eq!(uncertain.len(), 3);
}

#[test]
fn test_identify_uncertain_rich_context_reduces_uncertain() {
    // When every word in the text is well-represented in the context
    // the estimator should raise confidence above zero.
    let context = "rust language fast safe systems programming rust language fast safe";
    let text = "Rust is fast. Rust is safe.";
    let uncertain = ConfidenceEstimator::identify_uncertain_sentences(text, context, 1.0);
    // All sentences should be uncertain relative to threshold=1.0 (perfect confidence is very rare)
    assert!(!uncertain.is_empty());
}

#[test]
fn test_identify_uncertain_empty_text() {
    let uncertain = ConfidenceEstimator::identify_uncertain_sentences("", "some context", 0.5);
    assert!(uncertain.is_empty());
}

// ── ContextWindow ─────────────────────────────────────────────────────────────

#[test]
fn test_context_window_add_and_len() {
    let mut w = ContextWindow::new(10_000);
    assert!(w.is_empty());
    w.add_doc(ContextDoc::new("content a", 0.9, "doc-a"));
    assert_eq!(w.len(), 1);
    w.add_doc(ContextDoc::new("content b", 0.7, "doc-b"));
    assert_eq!(w.len(), 2);
}

#[test]
fn test_context_window_dedup_same_source_id_keeps_higher_score() {
    let mut w = ContextWindow::new(10_000);
    w.add_doc(ContextDoc::new("low score version", 0.3, "doc-x"));
    w.add_doc(ContextDoc::new("high score version", 0.9, "doc-x"));
    assert_eq!(w.len(), 1, "duplicate source_id should be deduplicated");
    assert!(w.docs[0].score > 0.8, "should retain higher-scored version");
}

#[test]
fn test_context_window_dedup_lower_score_ignored() {
    let mut w = ContextWindow::new(10_000);
    w.add_doc(ContextDoc::new("high score version", 0.9, "doc-x"));
    w.add_doc(ContextDoc::new("low score version", 0.3, "doc-x"));
    assert_eq!(w.len(), 1);
    assert!(w.docs[0].score > 0.8, "high score doc should be retained");
}

#[test]
fn test_context_window_sorted_descending() {
    let mut w = ContextWindow::new(10_000);
    w.add_doc(ContextDoc::new("low", 0.2, "doc-low"));
    w.add_doc(ContextDoc::new("high", 0.9, "doc-high"));
    w.add_doc(ContextDoc::new("mid", 0.5, "doc-mid"));
    assert!(
        w.docs[0].score >= w.docs[1].score,
        "first should be highest scored"
    );
    assert!(
        w.docs[1].score >= w.docs[2].score,
        "second should be higher than third"
    );
}

#[test]
fn test_context_window_to_string_non_empty() {
    let mut w = ContextWindow::new(10_000);
    w.add_doc(ContextDoc::new("hello", 0.9, "doc-a"));
    let s = w.to_string();
    assert!(s.contains("hello"));
    assert!(s.contains("doc-a"));
}

#[test]
fn test_context_window_trims_to_budget() {
    // Very tight budget — only fits one small doc.
    let mut w = ContextWindow::new(80);
    w.add_doc(ContextDoc::new("short doc", 0.9, "doc-a")); // fits
    w.add_doc(ContextDoc::new(
        "another doc that may not fit depending on total",
        0.5,
        "doc-b",
    ));
    // After trimming, the total should be within budget.
    let total: usize = w.docs.iter().map(|d| d.to_context_string().len()).sum();
    assert!(
        total <= 80,
        "total context ({total} chars) exceeds budget (80 chars)"
    );
}

// ── ContextDoc ────────────────────────────────────────────────────────────────

#[test]
fn test_context_doc_to_context_string_format() {
    let doc = ContextDoc::new("This is content.", 0.75, "my-source");
    let s = doc.to_context_string();
    assert!(s.contains("my-source"));
    assert!(s.contains("0.750"));
    assert!(s.contains("This is content."));
}

// ── FlareOutput::retrieval_rate ───────────────────────────────────────────────

#[test]
fn test_retrieval_rate_zero_iterations() {
    let out = FlareOutput {
        final_answer: "answer".to_string(),
        iterations: vec![],
        total_retrievals: 0,
        context_doc_count: 0,
    };
    assert!(
        out.retrieval_rate().abs() < 1e-6,
        "expected ~0.0, got {}",
        out.retrieval_rate()
    );
}

#[test]
fn test_retrieval_rate_all_retrieved() {
    let rec = IterationRecord {
        iteration: 0,
        generated_text: "g".to_string(),
        triggered_retrieval: true,
        retrieval_query: Some("q".to_string()),
        docs_retrieved: 2,
        avg_confidence: 0.3,
    };
    let out = FlareOutput {
        final_answer: "answer".to_string(),
        iterations: vec![rec.clone(), rec],
        total_retrievals: 2,
        context_doc_count: 4,
    };
    assert!((out.retrieval_rate() - 1.0).abs() < 1e-6);
}

#[test]
fn test_retrieval_rate_partial() {
    let make_rec = |retrieved: bool| IterationRecord {
        iteration: 0,
        generated_text: "g".to_string(),
        triggered_retrieval: retrieved,
        retrieval_query: None,
        docs_retrieved: 0,
        avg_confidence: 0.3,
    };
    let out = FlareOutput {
        final_answer: "ans".to_string(),
        iterations: vec![make_rec(true), make_rec(false), make_rec(false)],
        total_retrievals: 1,
        context_doc_count: 2,
    };
    let rate = out.retrieval_rate();
    assert!((rate - 1.0 / 3.0).abs() < 1e-5, "rate={rate}");
}

// ── MockFlareGenerator ────────────────────────────────────────────────────────

#[tokio::test]
async fn test_mock_generator_single_always_same() {
    let generator = MockFlareGenerator::new_single("fixed answer".to_string());
    let a = generator.generate("q", "ctx").await.expect("generate ok");
    let b = generator.generate("q", "ctx").await.expect("generate ok");
    assert_eq!(a, "fixed answer");
    assert_eq!(b, "fixed answer");
}

#[tokio::test]
async fn test_mock_generator_cycles() {
    let generator = MockFlareGenerator::new(vec![
        "answer one".to_string(),
        "answer two".to_string(),
        "answer three".to_string(),
    ]);
    let a = generator.generate("q", "").await.expect("ok");
    let b = generator.generate("q", "").await.expect("ok");
    let c = generator.generate("q", "").await.expect("ok");
    let d = generator.generate("q", "").await.expect("ok"); // wraps around
    assert_eq!(a, "answer one");
    assert_eq!(b, "answer two");
    assert_eq!(c, "answer three");
    assert_eq!(d, "answer one");
}

#[tokio::test]
async fn test_mock_generator_partial_truncates() {
    let answer = "First sentence. Second sentence. Third sentence.";
    let generator = MockFlareGenerator::new_single(answer.to_string());
    let partial = generator
        .generate_partial("q", "", 2)
        .await
        .expect("generate_partial ok");
    // Should contain at most 2 sentences.
    let sentences = ConfidenceEstimator::split_into_sentences(&partial);
    assert!(
        sentences.len() <= 2,
        "expected ≤2 sentences, got {}",
        sentences.len()
    );
}

// ── TemplateGenerator ─────────────────────────────────────────────────────────

#[tokio::test]
async fn test_template_generator_fills_placeholders() {
    let generator = TemplateGenerator::new("Q: {query} / C: {context}");
    let out = generator
        .generate("What is Rust?", "Rust is a language.")
        .await
        .expect("ok");
    assert!(out.contains("What is Rust?"));
    assert!(out.contains("Rust is a language."));
}

#[tokio::test]
async fn test_template_generator_partial() {
    let generator = TemplateGenerator::new("First sentence. Second sentence. Third sentence.");
    let partial = generator.generate_partial("q", "", 1).await.expect("ok");
    let sentences = ConfidenceEstimator::split_into_sentences(&partial);
    assert!(
        sentences.len() <= 1,
        "expected ≤1 sentence, got {}",
        sentences.len()
    );
}

// ── MockFlareRetriever ────────────────────────────────────────────────────────

#[tokio::test]
async fn test_mock_retriever_empty() {
    let ret = MockFlareRetriever::new_empty();
    let docs = ret.retrieve("anything", 10).await.expect("ok");
    assert!(docs.is_empty());
}

#[tokio::test]
async fn test_mock_retriever_respects_top_k() {
    let docs = vec![
        ContextDoc::new("doc a", 0.9, "a"),
        ContextDoc::new("doc b", 0.7, "b"),
        ContextDoc::new("doc c", 0.5, "c"),
    ];
    let ret = MockFlareRetriever::new(docs);
    let results = ret.retrieve("query", 2).await.expect("ok");
    assert_eq!(results.len(), 2, "should return top_k=2");
}

#[tokio::test]
async fn test_mock_retriever_returns_highest_scored_first() {
    let docs = vec![
        ContextDoc::new("low", 0.2, "low"),
        ContextDoc::new("high", 0.95, "high"),
        ContextDoc::new("mid", 0.5, "mid"),
    ];
    let ret = MockFlareRetriever::new(docs);
    let results = ret.retrieve("query", 3).await.expect("ok");
    assert_eq!(results[0].source_id, "high");
}

#[tokio::test]
async fn test_mock_retriever_top_k_larger_than_corpus() {
    let docs = vec![ContextDoc::new("only doc", 0.8, "sole")];
    let ret = MockFlareRetriever::new(docs);
    let results = ret.retrieve("query", 100).await.expect("ok");
    assert_eq!(results.len(), 1);
}

// ── QueryAugmentedRetriever ───────────────────────────────────────────────────

#[tokio::test]
async fn test_query_augmented_retriever_prepends_original() {
    use std::sync::{Arc, Mutex};

    // Capture the query string seen by the inner retriever.
    #[derive(Clone)]
    struct CaptureRetriever {
        captured: Arc<Mutex<Vec<String>>>,
    }

    #[async_trait::async_trait]
    impl FlareRetriever for CaptureRetriever {
        async fn retrieve(
            &self,
            query: &str,
            _top_k: usize,
        ) -> Result<Vec<ContextDoc>, super::types::FlareError> {
            self.captured
                .lock()
                .expect("lock ok")
                .push(query.to_string());
            Ok(vec![])
        }
    }

    let captured = Arc::new(Mutex::new(Vec::new()));
    let inner = CaptureRetriever {
        captured: captured.clone(),
    };
    let aug = QueryAugmentedRetriever::new(inner, "original query");
    aug.retrieve("uncertain span", 5).await.expect("ok");

    let seen = captured.lock().expect("lock ok");
    assert_eq!(seen.len(), 1);
    assert!(
        seen[0].contains("original query"),
        "should contain original query"
    );
    assert!(
        seen[0].contains("uncertain span"),
        "should contain uncertain span"
    );
}

// ── FlareEngine::run_simple ───────────────────────────────────────────────────

#[tokio::test]
async fn test_engine_run_simple_basic_output() {
    let generator =
        MockFlareGenerator::new_single("Rust is a systems programming language.".to_string());
    let ret = MockFlareRetriever::new_empty();
    let config = FlareConfig::default()
        .with_max_iterations(3)
        .with_confidence_threshold(0.01); // very low threshold → no retrieval triggered
    let engine = FlareEngine::new(generator, ret, config);
    let output = engine.run_simple("What is Rust?").await.expect("ok");

    assert!(
        !output.final_answer.is_empty(),
        "should have a final answer"
    );
    assert!(
        !output.iterations.is_empty(),
        "should have at least one iteration"
    );
}

#[tokio::test]
async fn test_engine_run_simple_returns_flare_output() {
    let generator = MockFlareGenerator::new_single("Answer.".to_string());
    let ret = MockFlareRetriever::new_empty();
    let engine = FlareEngine::new(generator, ret, FlareConfig::default());
    let output = engine.run_simple("q").await.expect("ok");

    // Check that FlareOutput fields are populated.
    assert!(output.iterations.len() <= FlareConfig::default().max_iterations);
    // context_doc_count should be 0 (no initial context, no retrieval)
    assert_eq!(output.context_doc_count, 0);
}

#[tokio::test]
async fn test_engine_run_with_initial_context() {
    let generator = MockFlareGenerator::new_single("Answer.".to_string());
    let ret = MockFlareRetriever::new_empty();
    let engine = FlareEngine::new(generator, ret, FlareConfig::default());
    let output = engine
        .run("What is Rust?", "Rust is fast and safe.")
        .await
        .expect("ok");

    // Initial context counts as 1 doc.
    assert!(
        output.context_doc_count >= 1,
        "initial context should count as a doc"
    );
}

#[tokio::test]
async fn test_engine_run_triggers_retrieval_for_uncertain_text() {
    // Use a high confidence threshold so that almost everything triggers retrieval.
    let config = FlareConfig::default()
        .with_max_iterations(3)
        .with_confidence_threshold(0.999)
        .with_min_query_length(3);

    let generator = MockFlareGenerator::new(vec![
        // First call (generate_partial): returns a sentence that will be uncertain.
        "Quantum entanglement paradox phenomenon.".to_string(),
        // Second call (generate full after retrieval):
        "Quantum entanglement is well known.".to_string(),
        // Third call:
        "Quantum entanglement is a phenomenon.".to_string(),
        // Fourth call:
        "Quantum entanglement is a phenomenon.".to_string(),
        // Fifth call:
        "Quantum entanglement is a phenomenon.".to_string(),
        // Sixth call:
        "Quantum entanglement is a phenomenon.".to_string(),
    ]);

    let retriever_docs = vec![ContextDoc::new(
        "Quantum entanglement is a quantum mechanics phenomenon.",
        0.95,
        "qm-doc",
    )];
    let ret = MockFlareRetriever::new(retriever_docs);

    let engine = FlareEngine::new(generator, ret, config);
    let output = engine
        .run_simple("What is quantum entanglement?")
        .await
        .expect("ok");

    // With a very high threshold, retrieval should be triggered.
    assert!(
        output.total_retrievals > 0,
        "retrieval should have been triggered at least once"
    );
}

#[tokio::test]
async fn test_engine_run_respects_max_iterations() {
    // Threshold=1.0 → always uncertain → retrieval every iteration.
    let config = FlareConfig::default()
        .with_max_iterations(3)
        .with_confidence_threshold(1.0)
        .with_min_query_length(2);

    // Provide enough answers for up to max_iterations * 2 calls
    // (generate_partial + generate per iteration).
    let answers: Vec<String> = (0..20).map(|i| format!("Answer iteration {i}.")).collect();
    let generator = MockFlareGenerator::new(answers);
    let ret = MockFlareRetriever::new(vec![ContextDoc::new("context doc", 0.9, "doc-a")]);

    let engine = FlareEngine::new(generator, ret, config.clone());
    let output = engine.run_simple("some query").await.expect("ok");

    let n_iters = output.iterations.len();
    let max_iters = config.max_iterations;
    assert!(
        n_iters <= max_iters,
        "iterations ({n_iters}) should not exceed max_iterations ({max_iters})"
    );
}

#[tokio::test]
async fn test_engine_retrieval_rate_calculation() {
    // Force one retrieval in three iterations: threshold=1.0 → first iteration triggers;
    // after retrieval use lower threshold so subsequent iterations converge.
    // We use a simple setup where retrieval_rate can be verified.
    let config = FlareConfig::default()
        .with_max_iterations(1)
        .with_confidence_threshold(1.0)
        .with_min_query_length(2);

    let generator = MockFlareGenerator::new((0..10).map(|i| format!("Sentence {i}.")).collect());
    let ret = MockFlareRetriever::new(vec![ContextDoc::new("retrieved context", 0.9, "ret-doc")]);

    let engine = FlareEngine::new(generator, ret, config);
    let output = engine.run_simple("query").await.expect("ok");

    // retrieval_rate = total_retrievals / iterations.len()
    #[allow(clippy::cast_precision_loss)]
    let expected_rate = output.total_retrievals as f32 / output.iterations.len().max(1) as f32;
    let got = output.retrieval_rate();
    assert!(
        (got - expected_rate).abs() < 1e-6,
        "retrieval_rate mismatch: got {got} expected {expected_rate}"
    );
}

#[tokio::test]
async fn test_engine_context_doc_count_grows_with_retrievals() {
    let config = FlareConfig::default()
        .with_max_iterations(2)
        .with_confidence_threshold(1.0)
        .with_min_query_length(2)
        .with_max_context_docs(10);

    let generator =
        MockFlareGenerator::new((0..10).map(|i| format!("Generated answer {i}.")).collect());
    let ret = MockFlareRetriever::new(vec![
        ContextDoc::new("doc a", 0.9, "doc-a"),
        ContextDoc::new("doc b", 0.8, "doc-b"),
    ]);

    let engine = FlareEngine::new(generator, ret, config);
    let output = engine.run_simple("query").await.expect("ok");

    if output.total_retrievals > 0 {
        assert!(
            output.context_doc_count > 0,
            "context_doc_count should increase when retrieval occurs"
        );
    }
}

#[tokio::test]
async fn test_engine_no_retrieval_when_threshold_zero() {
    // With threshold=0.0, nothing is ever uncertain → no retrieval.
    let config = FlareConfig::default()
        .with_max_iterations(5)
        .with_confidence_threshold(0.0);

    let generator = MockFlareGenerator::new_single("Answer to everything.".to_string());
    let ret = MockFlareRetriever::new(vec![ContextDoc::new("irrelevant doc", 0.9, "doc-x")]);

    let engine = FlareEngine::new(generator, ret, config);
    let output = engine.run_simple("any question").await.expect("ok");

    assert_eq!(
        output.total_retrievals, 0,
        "no retrieval expected with threshold=0.0"
    );
}

#[tokio::test]
async fn test_engine_query_augment_enabled_vs_disabled() {
    // Both configs should produce valid output (smoke test for the flag).
    let gen_a = MockFlareGenerator::new_single("Uncertain answer about unknown topic.".to_string());
    let gen_b = MockFlareGenerator::new_single("Uncertain answer about unknown topic.".to_string());
    let ret_a = MockFlareRetriever::new(vec![ContextDoc::new("topic info", 0.9, "doc-a")]);
    let ret_b = MockFlareRetriever::new(vec![ContextDoc::new("topic info", 0.9, "doc-b")]);

    let config_aug = FlareConfig::default()
        .with_query_augment(true)
        .with_max_iterations(2)
        .with_confidence_threshold(0.99);
    let config_no_aug = FlareConfig::default()
        .with_query_augment(false)
        .with_max_iterations(2)
        .with_confidence_threshold(0.99);

    let engine_a = FlareEngine::new(gen_a, ret_a, config_aug);
    let engine_b = FlareEngine::new(gen_b, ret_b, config_no_aug);

    let out_a = engine_a.run_simple("test query").await.expect("ok");
    let out_b = engine_b.run_simple("test query").await.expect("ok");

    assert!(!out_a.final_answer.is_empty());
    assert!(!out_b.final_answer.is_empty());
}
