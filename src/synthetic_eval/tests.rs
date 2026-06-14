//! Tests for the `synthetic_eval` module.
#![allow(clippy::float_cmp)]

use crate::synthetic_eval::generator::SyntheticEvalGenerator;
use crate::synthetic_eval::templater::{
    HeuristicTemplater, QuestionTemplater, sentences, tokenize,
};
use crate::synthetic_eval::types::{
    QuestionType, SyntheticEvalConfig, SyntheticEvalError, SyntheticQa,
};
use crate::types::{Document, DocumentId};

// ── Helpers ───────────────────────────────────────────────────────────────────

fn doc(id: &str, content: &str) -> Document {
    Document::new(content).with_id(DocumentId::from_string(id))
}

/// A small, lexically diverse corpus with overlapping salient terms.
fn corpus() -> Vec<Document> {
    vec![
        doc(
            "d1",
            "Ada Lovelace wrote the first algorithm for the Analytical Engine. \
             She is widely regarded as the first computer programmer.",
        ),
        doc(
            "d2",
            "Charles Babbage designed the Analytical Engine during the eighteen thirties. \
             Babbage is often called the father of the computer.",
        ),
        doc(
            "d3",
            "Alan Turing formulated the concept of the universal computing machine. \
             Turing also helped break the Enigma cipher during the war.",
        ),
        doc(
            "d4",
            "Grace Hopper developed the first compiler for a programming language. \
             Hopper popularized the term debugging in computer science.",
        ),
    ]
}

fn collect_types(qa: &[SyntheticQa]) -> std::collections::HashSet<QuestionType> {
    qa.iter().map(|q| q.question_type).collect()
}

// ── QuestionType ────────────────────────────────────────────────────────────────

#[test]
fn question_type_all_has_four_variants() {
    assert_eq!(QuestionType::all().len(), 4);
}

#[test]
fn question_type_all_is_ordered() {
    assert_eq!(QuestionType::all()[0], QuestionType::Factoid);
}

#[test]
fn question_type_as_str_cloze() {
    assert_eq!(QuestionType::Cloze.as_str(), "cloze");
}

#[test]
fn question_type_display_definitional() {
    assert_eq!(format!("{}", QuestionType::Definitional), "definitional");
}

// ── Config defaults & builders ───────────────────────────────────────────────────

#[test]
fn config_default_questions_per_doc() {
    assert_eq!(SyntheticEvalConfig::default().questions_per_doc, 3);
}

#[test]
fn config_default_num_distractors() {
    assert_eq!(SyntheticEvalConfig::default().num_distractors, 2);
}

#[test]
fn config_default_min_sentence_tokens() {
    assert_eq!(SyntheticEvalConfig::default().min_sentence_tokens, 5);
}

#[test]
fn config_default_types_is_all_four() {
    assert_eq!(SyntheticEvalConfig::default().types.len(), 4);
}

#[test]
fn config_new_equals_default() {
    assert_eq!(SyntheticEvalConfig::new(), SyntheticEvalConfig::default());
}

#[test]
fn config_with_questions_per_doc() {
    assert_eq!(
        SyntheticEvalConfig::new()
            .with_questions_per_doc(7)
            .questions_per_doc,
        7
    );
}

#[test]
fn config_with_num_distractors() {
    assert_eq!(
        SyntheticEvalConfig::new()
            .with_num_distractors(5)
            .num_distractors,
        5
    );
}

#[test]
fn config_with_min_sentence_tokens() {
    assert_eq!(
        SyntheticEvalConfig::new()
            .with_min_sentence_tokens(9)
            .min_sentence_tokens,
        9
    );
}

#[test]
fn config_with_types_restricts() {
    assert_eq!(
        SyntheticEvalConfig::new()
            .with_types(vec![QuestionType::Cloze])
            .types,
        vec![QuestionType::Cloze]
    );
}

// ── Tokenizer & sentence splitter ────────────────────────────────────────────────

#[test]
fn tokenize_lowercases() {
    assert!(tokenize("Hello World").contains(&"hello".to_string()));
}

#[test]
fn tokenize_drops_short_tokens() {
    assert!(!tokenize("a I am ok").contains(&"a".to_string()));
}

#[test]
fn sentences_splits_on_period() {
    assert_eq!(sentences("One thing. Two things.").len(), 2);
}

#[test]
fn sentences_drops_empty() {
    assert!(sentences("...").is_empty());
}

#[test]
fn sentences_splits_on_question_mark() {
    assert_eq!(sentences("Why? Because.").len(), 2);
}

// ── Empty corpus error ───────────────────────────────────────────────────────────

#[test]
fn empty_corpus_errors() {
    let generator = SyntheticEvalGenerator::new(SyntheticEvalConfig::default());
    assert_eq!(
        generator.generate(&[]),
        Err(SyntheticEvalError::EmptyCorpus)
    );
}

#[test]
fn empty_corpus_error_display() {
    assert_eq!(
        SyntheticEvalError::EmptyCorpus.to_string(),
        "corpus is empty"
    );
}

// ── Basic generation invariants ──────────────────────────────────────────────────

#[test]
fn generate_non_empty_corpus_produces_output() {
    let generator = SyntheticEvalGenerator::new(SyntheticEvalConfig::default());
    let qa = generator.generate(&corpus()).expect("non-empty");
    assert!(!qa.is_empty());
}

#[test]
fn questions_per_doc_cap_respected() {
    let config = SyntheticEvalConfig::default().with_questions_per_doc(2);
    let generator = SyntheticEvalGenerator::new(config);
    let c = corpus();
    let qa = generator.generate_for(&c[0], &c);
    assert!(qa.len() <= 2);
}

#[test]
fn questions_per_doc_zero_yields_nothing() {
    let config = SyntheticEvalConfig::default().with_questions_per_doc(0);
    let generator = SyntheticEvalGenerator::new(config);
    let c = corpus();
    assert!(generator.generate_for(&c[0], &c).is_empty());
}

#[test]
fn every_qa_has_non_empty_question() {
    let generator = SyntheticEvalGenerator::new(SyntheticEvalConfig::default());
    let qa = generator.generate(&corpus()).expect("non-empty");
    assert!(qa.iter().all(|q| !q.question.trim().is_empty()));
}

#[test]
fn every_qa_has_non_empty_answer() {
    let generator = SyntheticEvalGenerator::new(SyntheticEvalConfig::default());
    let qa = generator.generate(&corpus()).expect("non-empty");
    assert!(qa.iter().all(|q| !q.answer.trim().is_empty()));
}

#[test]
fn every_qa_has_non_empty_source_sentence() {
    let generator = SyntheticEvalGenerator::new(SyntheticEvalConfig::default());
    let qa = generator.generate(&corpus()).expect("non-empty");
    assert!(qa.iter().all(|q| !q.source_sentence.trim().is_empty()));
}

#[test]
fn source_id_matches_origin_document() {
    let config = SyntheticEvalConfig::default();
    let generator = SyntheticEvalGenerator::new(config);
    let c = corpus();
    let qa = generator.generate_for(&c[0], &c);
    assert!(qa.iter().all(|q| q.source_id == c[0].id));
}

// ── Cloze ────────────────────────────────────────────────────────────────────────

#[test]
fn cloze_question_contains_blank() {
    let config = SyntheticEvalConfig::default().with_types(vec![QuestionType::Cloze]);
    let generator = SyntheticEvalGenerator::new(config);
    let qa = generator.generate(&corpus()).expect("non-empty");
    assert!(qa.iter().all(|q| q.question.contains("___")));
}

#[test]
fn cloze_answer_is_blanked_term_present_in_sentence() {
    let config = SyntheticEvalConfig::default().with_types(vec![QuestionType::Cloze]);
    let generator = SyntheticEvalGenerator::new(config);
    let qa = generator.generate(&corpus()).expect("non-empty");
    assert!(
        qa.iter()
            .all(|q| tokenize(&q.source_sentence).contains(&q.answer.to_lowercase()))
    );
}

#[test]
fn cloze_answer_tokens_subset_of_sentence() {
    let config = SyntheticEvalConfig::default().with_types(vec![QuestionType::Cloze]);
    let generator = SyntheticEvalGenerator::new(config);
    let qa = generator.generate(&corpus()).expect("non-empty");
    let sentence_tokens: std::collections::HashSet<String> =
        tokenize(&qa[0].source_sentence).into_iter().collect();
    assert!(
        tokenize(&qa[0].answer)
            .iter()
            .all(|t| sentence_tokens.contains(t))
    );
}

#[test]
fn cloze_blank_appears_once() {
    let config = SyntheticEvalConfig::default().with_types(vec![QuestionType::Cloze]);
    let generator = SyntheticEvalGenerator::new(config);
    let qa = generator.generate(&corpus()).expect("non-empty");
    assert_eq!(qa[0].question.matches("___").count(), 1);
}

// ── Definitional ─────────────────────────────────────────────────────────────────

#[test]
fn definitional_question_contains_what_is() {
    let config = SyntheticEvalConfig::default().with_types(vec![QuestionType::Definitional]);
    let generator = SyntheticEvalGenerator::new(config);
    let qa = generator.generate(&corpus()).expect("non-empty");
    assert!(qa.iter().all(|q| q.question.contains("What is")));
}

#[test]
fn definitional_answer_tokens_subset_of_sentence() {
    let config = SyntheticEvalConfig::default().with_types(vec![QuestionType::Definitional]);
    let generator = SyntheticEvalGenerator::new(config);
    let qa = generator.generate(&corpus()).expect("non-empty");
    let sentence_tokens: std::collections::HashSet<String> =
        tokenize(&qa[0].source_sentence).into_iter().collect();
    assert!(
        tokenize(&qa[0].answer)
            .iter()
            .all(|t| sentence_tokens.contains(t))
    );
}

#[test]
fn definitional_answer_equals_source_sentence() {
    let config = SyntheticEvalConfig::default().with_types(vec![QuestionType::Definitional]);
    let generator = SyntheticEvalGenerator::new(config);
    let qa = generator.generate(&corpus()).expect("non-empty");
    assert_eq!(qa[0].answer, qa[0].source_sentence);
}

// ── Factoid ──────────────────────────────────────────────────────────────────────

#[test]
fn factoid_question_starts_with_what_did() {
    let config = SyntheticEvalConfig::default().with_types(vec![QuestionType::Factoid]);
    let generator = SyntheticEvalGenerator::new(config);
    let qa = generator.generate(&corpus()).expect("non-empty");
    assert!(qa.iter().all(|q| q.question.starts_with("What did")));
}

#[test]
fn factoid_answer_tokens_subset_of_sentence() {
    let config = SyntheticEvalConfig::default().with_types(vec![QuestionType::Factoid]);
    let generator = SyntheticEvalGenerator::new(config);
    let qa = generator.generate(&corpus()).expect("non-empty");
    let sentence_tokens: std::collections::HashSet<String> =
        tokenize(&qa[0].source_sentence).into_iter().collect();
    assert!(
        tokenize(&qa[0].answer)
            .iter()
            .all(|t| sentence_tokens.contains(t))
    );
}

// ── Relational ───────────────────────────────────────────────────────────────────

#[test]
fn relational_requires_two_entities() {
    let templater = HeuristicTemplater::new();
    let single = doc("x", "Babbage worked carefully.");
    let produced = templater.generate(
        "Babbage worked carefully",
        &single,
        &[QuestionType::Relational],
    );
    assert!(produced.is_empty());
}

#[test]
fn relational_emitted_with_two_entities() {
    let templater = HeuristicTemplater::new();
    let d = doc("x", "ignored");
    let produced = templater.generate(
        "Ada Lovelace collaborated with Charles Babbage",
        &d,
        &[QuestionType::Relational],
    );
    assert_eq!(produced.len(), 1);
}

#[test]
fn relational_question_contains_relationship() {
    let templater = HeuristicTemplater::new();
    let d = doc("x", "ignored");
    let produced = templater.generate(
        "Ada Lovelace collaborated with Charles Babbage",
        &d,
        &[QuestionType::Relational],
    );
    assert!(produced[0].0.contains("relationship between"));
}

// ── Type filtering ───────────────────────────────────────────────────────────────

#[test]
fn types_filter_only_cloze() {
    let config = SyntheticEvalConfig::default().with_types(vec![QuestionType::Cloze]);
    let generator = SyntheticEvalGenerator::new(config);
    let qa = generator.generate(&corpus()).expect("non-empty");
    assert!(qa.iter().all(|q| q.question_type == QuestionType::Cloze));
}

#[test]
fn types_filter_only_definitional() {
    let config = SyntheticEvalConfig::default().with_types(vec![QuestionType::Definitional]);
    let generator = SyntheticEvalGenerator::new(config);
    let qa = generator.generate(&corpus()).expect("non-empty");
    assert!(
        collect_types(&qa)
            .iter()
            .all(|t| *t == QuestionType::Definitional)
    );
}

#[test]
fn types_filter_excludes_unrequested() {
    let config = SyntheticEvalConfig::default()
        .with_types(vec![QuestionType::Cloze, QuestionType::Definitional]);
    let generator = SyntheticEvalGenerator::new(config);
    let qa = generator.generate(&corpus()).expect("non-empty");
    assert!(!collect_types(&qa).contains(&QuestionType::Factoid));
}

// ── Sentence-length filtering ────────────────────────────────────────────────────

#[test]
fn short_sentences_skipped() {
    let config = SyntheticEvalConfig::default().with_min_sentence_tokens(50);
    let generator = SyntheticEvalGenerator::new(config);
    let c = corpus();
    assert!(generator.generate_for(&c[0], &c).is_empty());
}

#[test]
fn short_single_sentence_doc_yields_nothing() {
    let config = SyntheticEvalConfig::default().with_min_sentence_tokens(5);
    let generator = SyntheticEvalGenerator::new(config);
    let tiny = vec![doc("t1", "Short text here.")];
    assert!(generator.generate_for(&tiny[0], &tiny).is_empty());
}

// ── Distractor mining ────────────────────────────────────────────────────────────

#[test]
fn distractors_exclude_source_id() {
    let generator = SyntheticEvalGenerator::new(SyntheticEvalConfig::default());
    let qa = generator.generate(&corpus()).expect("non-empty");
    assert!(qa.iter().all(|q| !q.distractor_ids.contains(&q.source_id)));
}

#[test]
fn distractor_count_respects_cap() {
    let config = SyntheticEvalConfig::default().with_num_distractors(1);
    let generator = SyntheticEvalGenerator::new(config);
    let qa = generator.generate(&corpus()).expect("non-empty");
    assert!(qa.iter().all(|q| q.distractor_ids.len() <= 1));
}

#[test]
fn zero_distractors_yields_empty() {
    let config = SyntheticEvalConfig::default().with_num_distractors(0);
    let generator = SyntheticEvalGenerator::new(config);
    let qa = generator.generate(&corpus()).expect("non-empty");
    assert!(qa.iter().all(|q| q.distractor_ids.is_empty()));
}

#[test]
fn distractors_share_salient_term_with_question() {
    let config = SyntheticEvalConfig::default().with_num_distractors(3);
    let generator = SyntheticEvalGenerator::new(config);
    let c = corpus();
    let qa = generator.generate(&c).expect("non-empty");
    let by_id = |id: &DocumentId| c.iter().find(|d| &d.id == id).expect("present");
    let related = qa.iter().all(|q| {
        q.distractor_ids.iter().all(|did| {
            let q_terms: std::collections::HashSet<String> =
                tokenize(&q.question).into_iter().collect();
            let d_terms: std::collections::HashSet<String> =
                tokenize(&by_id(did).content).into_iter().collect();
            q_terms.intersection(&d_terms).next().is_some()
        })
    });
    assert!(related);
}

#[test]
fn distractors_are_unique() {
    let config = SyntheticEvalConfig::default().with_num_distractors(3);
    let generator = SyntheticEvalGenerator::new(config);
    let qa = generator.generate(&corpus()).expect("non-empty");
    let all_unique = qa.iter().all(|q| {
        let set: std::collections::HashSet<&DocumentId> = q.distractor_ids.iter().collect();
        set.len() == q.distractor_ids.len()
    });
    assert!(all_unique);
}

#[test]
fn unrelated_corpus_mines_no_distractors() {
    // Two documents with entirely disjoint vocabularies.
    let c = vec![
        doc(
            "a",
            "Quantum entanglement links distant particles instantly forever.",
        ),
        doc(
            "b",
            "Bakers knead dough overnight before sunrise daily routines.",
        ),
    ];
    let config = SyntheticEvalConfig::default().with_num_distractors(2);
    let generator = SyntheticEvalGenerator::new(config);
    let qa = generator.generate_for(&c[0], &c);
    assert!(qa.iter().all(|q| q.distractor_ids.is_empty()));
}

// ── Multi-document corpus ────────────────────────────────────────────────────────

#[test]
fn multi_doc_corpus_covers_multiple_sources() {
    let generator = SyntheticEvalGenerator::new(SyntheticEvalConfig::default());
    let qa = generator.generate(&corpus()).expect("non-empty");
    let sources: std::collections::HashSet<&DocumentId> = qa.iter().map(|q| &q.source_id).collect();
    assert!(sources.len() >= 2);
}

#[test]
fn multi_doc_total_within_global_cap() {
    let config = SyntheticEvalConfig::default().with_questions_per_doc(2);
    let generator = SyntheticEvalGenerator::new(config);
    let c = corpus();
    let qa = generator.generate(&c).expect("non-empty");
    assert!(qa.len() <= 2 * c.len());
}

// ── Determinism ──────────────────────────────────────────────────────────────────

#[test]
fn generation_is_deterministic() {
    let generator = SyntheticEvalGenerator::new(SyntheticEvalConfig::default());
    let first = generator.generate(&corpus()).expect("non-empty");
    let second = generator.generate(&corpus()).expect("non-empty");
    assert_eq!(first, second);
}

#[test]
fn generate_for_is_deterministic() {
    let generator = SyntheticEvalGenerator::new(SyntheticEvalConfig::default());
    let c = corpus();
    let first = generator.generate_for(&c[1], &c);
    let second = generator.generate_for(&c[1], &c);
    assert_eq!(first, second);
}

#[test]
fn distractor_order_is_deterministic() {
    let generator = SyntheticEvalGenerator::new(SyntheticEvalConfig::default());
    let c = corpus();
    let first: Vec<Vec<DocumentId>> = generator
        .generate_for(&c[0], &c)
        .into_iter()
        .map(|q| q.distractor_ids)
        .collect();
    let second: Vec<Vec<DocumentId>> = generator
        .generate_for(&c[0], &c)
        .into_iter()
        .map(|q| q.distractor_ids)
        .collect();
    assert_eq!(first, second);
}

// ── Custom templater ─────────────────────────────────────────────────────────────

/// A trivial templater used to verify the `with_templater` path.
#[derive(Debug, Clone, Copy)]
struct ConstTemplater;

impl QuestionTemplater for ConstTemplater {
    fn generate(
        &self,
        sentence: &str,
        _doc: &Document,
        _allowed: &[QuestionType],
    ) -> Vec<(String, String, QuestionType)> {
        vec![(
            "What is this about?".to_string(),
            sentence.to_string(),
            QuestionType::Factoid,
        )]
    }
}

#[test]
fn with_templater_uses_custom_questions() {
    let config = SyntheticEvalConfig::default().with_questions_per_doc(1);
    let generator = SyntheticEvalGenerator::with_templater(config, ConstTemplater);
    let c = corpus();
    let qa = generator.generate_for(&c[0], &c);
    assert_eq!(qa[0].question, "What is this about?");
}

#[test]
fn with_templater_respects_cap() {
    let config = SyntheticEvalConfig::default().with_questions_per_doc(1);
    let generator = SyntheticEvalGenerator::with_templater(config, ConstTemplater);
    let c = corpus();
    assert!(generator.generate_for(&c[0], &c).len() <= 1);
}

// ── Templater direct ─────────────────────────────────────────────────────────────

#[test]
fn templater_orders_by_canonical_template_order() {
    let templater = HeuristicTemplater::new();
    let d = doc("x", "ignored");
    let produced = templater.generate(
        "Ada Lovelace studied with Charles Babbage",
        &d,
        &QuestionType::all(),
    );
    assert_eq!(produced[0].2, QuestionType::Factoid);
}

#[test]
fn templater_empty_allowed_yields_nothing() {
    let templater = HeuristicTemplater::new();
    let d = doc("x", "ignored");
    let produced = templater.generate("Ada Lovelace studied mathematics", &d, &[]);
    assert!(produced.is_empty());
}

#[test]
fn templater_no_entity_skips_definitional() {
    let templater = HeuristicTemplater::new();
    let d = doc("x", "ignored");
    let produced = templater.generate(
        "the small device turned quickly around",
        &d,
        &[QuestionType::Definitional],
    );
    assert!(produced.is_empty());
}
