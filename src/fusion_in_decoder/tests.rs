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
//! Tests for the `fusion_in_decoder` module.

use crate::fusion_in_decoder::fusion::FusionInDecoder;
use crate::fusion_in_decoder::types::{FidConfig, FidError, FusedAnswer, PassageEvidence};
use crate::types::{Document, DocumentId};

// ── helpers ───────────────────────────────────────────────────────────────────

fn doc(id: &str, content: &str) -> Document {
    Document::new(content).with_id(id)
}

fn default_engine() -> FusionInDecoder {
    FusionInDecoder::new(FidConfig::default())
}

fn ids(answer: &FusedAnswer) -> Vec<String> {
    answer
        .attributions
        .iter()
        .map(|d| d.as_str().to_string())
        .collect()
}

// ── FidConfig defaults & builders ───────────────────────────────────────────────

#[test]
fn config_default_values() {
    let cfg = FidConfig::default();
    assert_eq!(cfg.top_passages, 5);
    assert_eq!(cfg.dim, 128);
    assert_eq!(cfg.dedup_threshold, 0.8);
}

#[test]
fn config_new_matches_default() {
    assert_eq!(FidConfig::new(), FidConfig::default());
}

#[test]
fn config_with_top_passages() {
    let cfg = FidConfig::new().with_top_passages(3);
    assert_eq!(cfg.top_passages, 3);
    assert_eq!(cfg.dim, 128);
    assert_eq!(cfg.dedup_threshold, 0.8);
}

#[test]
fn config_with_dim() {
    let cfg = FidConfig::new().with_dim(256);
    assert_eq!(cfg.dim, 256);
}

#[test]
fn config_with_dedup_threshold() {
    let cfg = FidConfig::new().with_dedup_threshold(0.5);
    assert_eq!(cfg.dedup_threshold, 0.5);
}

#[test]
fn config_builders_chain() {
    let cfg = FidConfig::new()
        .with_top_passages(2)
        .with_dim(64)
        .with_dedup_threshold(0.9);
    assert_eq!(cfg.top_passages, 2);
    assert_eq!(cfg.dim, 64);
    assert_eq!(cfg.dedup_threshold, 0.9);
}

#[test]
fn config_clone_eq() {
    let cfg = FidConfig::new().with_top_passages(7);
    assert_eq!(cfg.clone(), cfg);
}

// ── PassageEvidence basics ──────────────────────────────────────────────────────

#[test]
fn passage_evidence_new() {
    let pe = PassageEvidence::new(DocumentId::from("x"), "hello world", 0.5);
    assert_eq!(pe.passage_id.as_str(), "x");
    assert_eq!(pe.evidence, "hello world");
    assert_eq!(pe.relevance, 0.5);
}

#[test]
fn passage_evidence_is_relevant() {
    let pe = PassageEvidence::new(DocumentId::from("x"), "e", 0.4);
    assert!(pe.is_relevant(0.3));
    assert!(pe.is_relevant(0.4));
    assert!(!pe.is_relevant(0.5));
}

#[test]
fn passage_evidence_clone_eq() {
    let pe = PassageEvidence::new(DocumentId::from("x"), "e", 0.4);
    assert_eq!(pe.clone(), pe);
}

// ── FusionInDecoder construction ────────────────────────────────────────────────

#[test]
fn engine_new_stores_config() {
    let engine = FusionInDecoder::new(FidConfig::new().with_top_passages(9));
    assert_eq!(engine.config().top_passages, 9);
}

#[test]
fn engine_default_uses_default_config() {
    let engine = FusionInDecoder::default();
    assert_eq!(engine.config(), &FidConfig::default());
}

// ── extract_evidence: one evidence per passage ──────────────────────────────────

#[test]
fn extract_one_evidence_per_passage() {
    let docs = vec![
        doc("a", "Cats are mammals. Dogs are loyal animals."),
        doc("b", "The sky is blue. Water boils at one hundred degrees."),
    ];
    let ev = default_engine().extract_evidence("loyal animals", &docs);
    assert_eq!(ev.len(), 2);
}

#[test]
fn extract_best_sentence_chosen() {
    let docs = vec![doc(
        "a",
        "The capital of France is Paris. Bananas are yellow fruit.",
    )];
    let ev = default_engine().extract_evidence("capital of France", &docs);
    assert_eq!(ev.len(), 1);
    assert!(ev[0].evidence.contains("capital of France"));
    assert!(ev[0].relevance > 0.0);
}

#[test]
fn extract_evidence_has_passage_id() {
    let docs = vec![doc("doc-123", "Relevant query content here.")];
    let ev = default_engine().extract_evidence("query content", &docs);
    assert_eq!(ev[0].passage_id.as_str(), "doc-123");
}

#[test]
fn extract_relevance_is_overlap_score() {
    // query and the single sentence share exactly half their union tokens.
    let docs = vec![doc("a", "alpha beta")];
    let ev = default_engine().extract_evidence("alpha gamma", &docs);
    // tokens sentence={alpha,beta}, query={alpha,gamma}; intersection=1, union=3.
    assert!((ev[0].relevance - (1.0 / 3.0)).abs() < 1e-6);
}

#[test]
fn extract_relevance_zero_for_no_overlap() {
    let docs = vec![doc("a", "completely unrelated tokens here.")];
    let ev = default_engine().extract_evidence("xylophone zebra", &docs);
    assert_eq!(ev[0].relevance, 0.0);
}

#[test]
fn extract_handles_no_sentence_boundary() {
    let docs = vec![doc("a", "plain text without punctuation about apples")];
    let ev = default_engine().extract_evidence("apples", &docs);
    assert_eq!(ev.len(), 1);
    assert!(ev[0].evidence.contains("apples"));
    assert!(ev[0].relevance > 0.0);
}

#[test]
fn extract_empty_content_gives_empty_evidence() {
    let docs = vec![doc("a", "")];
    let ev = default_engine().extract_evidence("anything", &docs);
    assert_eq!(ev.len(), 1);
    assert_eq!(ev[0].evidence, "");
    assert_eq!(ev[0].relevance, 0.0);
}

// ── extract_evidence: sorting & capping ─────────────────────────────────────────

#[test]
fn extract_sorted_descending() {
    let docs = vec![
        doc("low", "irrelevant filler content."),
        doc("high", "machine learning models train on data."),
        doc("mid", "data is collected somewhere."),
    ];
    let ev = default_engine().extract_evidence("machine learning data", &docs);
    for pair in ev.windows(2) {
        assert!(pair[0].relevance >= pair[1].relevance);
    }
    // the most relevant should be the dedicated ML passage.
    assert_eq!(ev[0].passage_id.as_str(), "high");
}

#[test]
fn extract_capped_at_top_passages() {
    let docs: Vec<Document> = (0..10)
        .map(|i| doc(&format!("d{i}"), "query token appears here."))
        .collect();
    let engine = FusionInDecoder::new(FidConfig::new().with_top_passages(3));
    let ev = engine.extract_evidence("query token", &docs);
    assert_eq!(ev.len(), 3);
}

#[test]
fn extract_top_passages_one() {
    let docs = vec![
        doc("a", "alpha relevance high alpha."),
        doc("b", "beta relevance lower."),
    ];
    let engine = FusionInDecoder::new(FidConfig::new().with_top_passages(1));
    let ev = engine.extract_evidence("alpha relevance", &docs);
    assert_eq!(ev.len(), 1);
}

#[test]
fn extract_top_passages_zero_yields_empty() {
    let docs = vec![doc("a", "content here.")];
    let engine = FusionInDecoder::new(FidConfig::new().with_top_passages(0));
    let ev = engine.extract_evidence("content", &docs);
    assert!(ev.is_empty());
}

#[test]
fn extract_fewer_docs_than_cap() {
    let docs = vec![doc("a", "only one passage with query terms.")];
    let ev = default_engine().extract_evidence("query terms", &docs);
    assert_eq!(ev.len(), 1);
}

// ── fuse: concatenation ─────────────────────────────────────────────────────────

#[test]
fn fuse_concatenates_evidence() {
    let docs = vec![
        doc("a", "Paris is the capital of France."),
        doc("b", "Berlin is the capital of Germany."),
    ];
    let fused = default_engine().fuse("capital", &docs).unwrap();
    assert!(fused.answer.contains("Paris"));
    assert!(fused.answer.contains("Berlin"));
}

#[test]
fn fuse_answer_non_empty() {
    let docs = vec![doc("a", "Relevant fact about quantum physics.")];
    let fused = default_engine().fuse("quantum physics", &docs).unwrap();
    assert!(!fused.answer.is_empty());
}

#[test]
fn fuse_evidence_count_matches_survivors() {
    let docs = vec![
        doc("a", "First distinct evidence about cats."),
        doc("b", "Second distinct evidence about rockets."),
    ];
    let fused = default_engine().fuse("evidence", &docs).unwrap();
    assert_eq!(fused.evidence.len(), 2);
}

#[test]
fn fuse_single_doc() {
    let docs = vec![doc("solo", "The lone passage about astronomy.")];
    let fused = default_engine().fuse("astronomy", &docs).unwrap();
    assert_eq!(fused.attributions.len(), 1);
    assert_eq!(fused.attributions[0].as_str(), "solo");
}

#[test]
fn fuse_skips_empty_evidence_in_answer() {
    let docs = vec![doc("a", "Meaningful query content here."), doc("b", "")];
    let fused = default_engine().fuse("query content", &docs).unwrap();
    // Empty-evidence passage must not introduce stray separators / leading space.
    assert!(!fused.answer.starts_with(' '));
    assert!(!fused.answer.ends_with(' '));
    assert!(!fused.answer.contains("  "));
}

// ── fuse: attributions ──────────────────────────────────────────────────────────

#[test]
fn fuse_attributions_list_contributors() {
    let docs = vec![
        doc("a", "Apples grow on trees."),
        doc("b", "Oranges are citrus fruit."),
    ];
    let fused = default_engine().fuse("fruit trees citrus", &docs).unwrap();
    let id_list = ids(&fused);
    assert!(id_list.contains(&"a".to_string()));
    assert!(id_list.contains(&"b".to_string()));
}

#[test]
fn fuse_attributions_no_duplicates() {
    let docs = vec![
        doc("a", "Topic one sentence."),
        doc("b", "Topic two sentence."),
        doc("c", "Topic three sentence."),
    ];
    let fused = default_engine().fuse("topic sentence", &docs).unwrap();
    let mut seen = std::collections::HashSet::new();
    for id in &fused.attributions {
        assert!(seen.insert(id.clone()), "duplicate attribution found");
    }
}

#[test]
fn fuse_attributions_match_evidence() {
    let docs = vec![
        doc("a", "Distinct one about birds."),
        doc("b", "Distinct two about planes."),
    ];
    let fused = default_engine().fuse("distinct", &docs).unwrap();
    for ev in &fused.evidence {
        assert!(fused.attributions.contains(&ev.passage_id));
    }
}

#[test]
fn fuse_passage_count_helper() {
    let docs = vec![
        doc("a", "Alpha distinct content."),
        doc("b", "Beta distinct content."),
    ];
    let fused = default_engine().fuse("distinct content", &docs).unwrap();
    assert_eq!(fused.passage_count(), fused.attributions.len());
}

// ── fuse: dedup ─────────────────────────────────────────────────────────────────

#[test]
fn fuse_dedup_removes_near_identical() {
    // Two passages whose best sentences are identical → dedup to one.
    let docs = vec![
        doc("a", "The quick brown fox jumps over the dog."),
        doc("b", "The quick brown fox jumps over the dog."),
    ];
    let fused = default_engine().fuse("quick brown fox", &docs).unwrap();
    assert_eq!(
        fused.evidence.len(),
        1,
        "near-identical evidence not deduped"
    );
    assert_eq!(fused.attributions.len(), 1);
}

#[test]
fn fuse_dedup_keeps_higher_relevance_copy() {
    // Both best sentences are near-identical (token-Jaccard ~0.86 >= 0.8 → dedup),
    // but `keep`'s sentence has no diluting extra token, so it is strictly more
    // query-relevant, ranks first, and is the survivor.
    let docs = vec![
        doc("drop", "alpha beta gamma delta epsilon zeta extra."),
        doc("keep", "alpha beta gamma delta epsilon zeta."),
    ];
    let fused = default_engine()
        .fuse("alpha beta gamma delta", &docs)
        .unwrap();
    // keep relevance = 4/6 ≈ 0.667 > drop relevance = 4/7 ≈ 0.571.
    assert_eq!(
        fused.evidence.len(),
        1,
        "near-identical evidence not deduped"
    );
    // The survivor is the higher-relevance passage.
    assert_eq!(fused.evidence[0].passage_id.as_str(), "keep");
    assert_eq!(fused.attributions, vec![DocumentId::from("keep")]);
}

#[test]
fn fuse_distinct_evidence_kept() {
    let docs = vec![
        doc("a", "Photosynthesis converts sunlight into energy."),
        doc("b", "Gravity attracts masses toward each other."),
    ];
    let fused = default_engine().fuse("energy masses", &docs).unwrap();
    assert_eq!(fused.evidence.len(), 2, "distinct evidence wrongly deduped");
}

#[test]
fn fuse_low_dedup_threshold_collapses_overlap() {
    // Sentences share some tokens; a very low threshold treats them as dupes.
    let docs = vec![
        doc("a", "shared query token alpha here."),
        doc("b", "shared query token beta there."),
    ];
    let engine = FusionInDecoder::new(FidConfig::new().with_dedup_threshold(0.1));
    let fused = engine.fuse("shared query token", &docs).unwrap();
    assert_eq!(fused.evidence.len(), 1);
}

#[test]
fn fuse_high_dedup_threshold_keeps_both() {
    let docs = vec![
        doc("a", "shared query token alpha here."),
        doc("b", "shared query token beta there."),
    ];
    let engine = FusionInDecoder::new(FidConfig::new().with_dedup_threshold(0.99));
    let fused = engine.fuse("shared query token", &docs).unwrap();
    assert_eq!(fused.evidence.len(), 2);
}

#[test]
fn fuse_dedup_three_identical_to_one() {
    let docs = vec![
        doc("a", "Identical evidence sentence about science."),
        doc("b", "Identical evidence sentence about science."),
        doc("c", "Identical evidence sentence about science."),
    ];
    let fused = default_engine().fuse("evidence science", &docs).unwrap();
    assert_eq!(fused.evidence.len(), 1);
    assert_eq!(fused.attributions.len(), 1);
}

// ── fuse: ordering by relevance ─────────────────────────────────────────────────

#[test]
fn fuse_more_relevant_appears_earlier() {
    let docs = vec![
        doc("weak", "This passage barely mentions data once."),
        doc(
            "strong",
            "machine learning data pipelines process data efficiently.",
        ),
    ];
    let fused = default_engine()
        .fuse("machine learning data pipelines", &docs)
        .unwrap();
    let strong_pos = fused.answer.find("machine learning data pipelines");
    let weak_pos = fused.answer.find("barely mentions");
    assert!(strong_pos.is_some());
    if let (Some(s), Some(w)) = (strong_pos, weak_pos) {
        assert!(s < w, "more relevant evidence should appear earlier");
    }
    // And the evidence vector is ordered the same way.
    assert_eq!(fused.evidence[0].passage_id.as_str(), "strong");
}

#[test]
fn fuse_evidence_sorted_descending() {
    let docs = vec![
        doc("a", "alpha beta gamma delta epsilon all here."),
        doc("b", "alpha beta only."),
        doc("c", "nothing in common whatsoever."),
    ];
    let fused = default_engine()
        .fuse("alpha beta gamma delta epsilon", &docs)
        .unwrap();
    for pair in fused.evidence.windows(2) {
        assert!(pair[0].relevance >= pair[1].relevance);
    }
}

#[test]
fn fuse_first_attribution_is_most_relevant() {
    let docs = vec![
        doc("lo", "tangential mention of topic."),
        doc("hi", "topic topic topic core central topic discussion."),
    ];
    let fused = default_engine().fuse("topic core central", &docs).unwrap();
    assert_eq!(fused.attributions[0].as_str(), "hi");
}

// ── errors ──────────────────────────────────────────────────────────────────────

#[test]
fn fuse_empty_query_errors() {
    let docs = vec![doc("a", "some content.")];
    let err = default_engine().fuse("", &docs).unwrap_err();
    assert!(matches!(err, FidError::EmptyQuery));
}

#[test]
fn fuse_whitespace_query_errors() {
    let docs = vec![doc("a", "some content.")];
    let err = default_engine().fuse("   \t\n", &docs).unwrap_err();
    assert!(matches!(err, FidError::EmptyQuery));
}

#[test]
fn fuse_empty_corpus_errors() {
    let docs: Vec<Document> = vec![];
    let err = default_engine().fuse("a query", &docs).unwrap_err();
    assert!(matches!(err, FidError::EmptyCorpus));
}

#[test]
fn fuse_error_messages() {
    assert_eq!(FidError::EmptyQuery.to_string(), "query must not be empty");
    assert_eq!(FidError::EmptyCorpus.to_string(), "corpus is empty");
}

#[test]
fn fuse_empty_query_takes_precedence_over_empty_corpus() {
    let docs: Vec<Document> = vec![];
    let err = default_engine().fuse("", &docs).unwrap_err();
    assert!(matches!(err, FidError::EmptyQuery));
}

// ── determinism ─────────────────────────────────────────────────────────────────

#[test]
fn fuse_is_deterministic() {
    let docs = vec![
        doc("a", "First fact about renewable energy sources."),
        doc("b", "Second fact about solar power and wind energy."),
        doc("c", "Third fact about hydroelectric dams."),
    ];
    let engine = default_engine();
    let a = engine.fuse("energy power", &docs).unwrap();
    let b = engine.fuse("energy power", &docs).unwrap();
    assert_eq!(a, b);
}

#[test]
fn extract_evidence_is_deterministic() {
    let docs = vec![
        doc("a", "Alpha content about science."),
        doc("b", "Beta content about science."),
    ];
    let engine = default_engine();
    let first = engine.extract_evidence("science", &docs);
    let second = engine.extract_evidence("science", &docs);
    assert_eq!(first, second);
}

#[test]
fn fuse_tie_break_is_stable() {
    // Two passages with equal relevance but distinct evidence → tie broken by id.
    let docs = vec![
        doc("zzz", "query token unique alpha."),
        doc("aaa", "query token unique beta."),
    ];
    let engine = FusionInDecoder::new(FidConfig::new().with_dedup_threshold(0.99));
    let fused = engine.fuse("query token unique", &docs).unwrap();
    // Equal relevance ⇒ lexicographically smaller id ("aaa") comes first.
    assert_eq!(fused.attributions[0].as_str(), "aaa");
    assert_eq!(fused.attributions[1].as_str(), "zzz");
}

#[test]
fn fuse_order_independent_of_input_order() {
    let docs_fwd = vec![
        doc("a", "strong alpha beta gamma signal here."),
        doc("b", "weak gamma only."),
    ];
    let docs_rev = vec![
        doc("b", "weak gamma only."),
        doc("a", "strong alpha beta gamma signal here."),
    ];
    let engine = default_engine();
    let fwd = engine.fuse("alpha beta gamma", &docs_fwd).unwrap();
    let rev = engine.fuse("alpha beta gamma", &docs_rev).unwrap();
    assert_eq!(fwd.attributions, rev.attributions);
    assert_eq!(fwd.answer, rev.answer);
}

// ── FusedAnswer helpers ─────────────────────────────────────────────────────────

#[test]
fn fused_answer_is_empty_false_when_evidence() {
    let docs = vec![doc("a", "Relevant content here.")];
    let fused = default_engine().fuse("content", &docs).unwrap();
    assert!(!fused.is_empty());
}

#[test]
fn fused_answer_clone_eq() {
    let docs = vec![doc("a", "Content about topic.")];
    let fused = default_engine().fuse("topic", &docs).unwrap();
    assert_eq!(fused.clone(), fused);
}

// ── integration-ish coverage ─────────────────────────────────────────────────────

#[test]
fn fuse_realistic_multi_passage() {
    let docs = vec![
        doc(
            "p1",
            "The Great Wall of China was built to protect against invasions. \
             Construction spanned many dynasties.",
        ),
        doc(
            "p2",
            "The wall is a long fortification. It stretches thousands of miles.",
        ),
        doc(
            "p3",
            "Pandas are native to China and eat bamboo. They are an endangered species.",
        ),
    ];
    let fused = default_engine()
        .fuse("Why was the Great Wall of China built?", &docs)
        .unwrap();
    // The "built to protect" passage shares the most query terms and leads.
    assert_eq!(fused.evidence[0].passage_id.as_str(), "p1");
    assert!(fused.answer.contains("protect"));
    // Panda passage is least relevant; still attributed but ranked last.
    assert!(fused.attributions.contains(&DocumentId::from("p3")));
    assert_eq!(fused.attributions.last().unwrap().as_str(), "p3");
}

#[test]
fn fuse_all_irrelevant_still_succeeds() {
    let docs = vec![
        doc("a", "Random unrelated sentence one."),
        doc("b", "Random unrelated sentence two."),
    ];
    let fused = default_engine().fuse("xylophone quasar", &docs).unwrap();
    // No overlap → relevance 0, but evidence/attribution still produced.
    assert_eq!(fused.evidence.len(), 2);
    assert_eq!(fused.attributions.len(), 2);
}

#[test]
fn extract_then_fuse_consistent_count() {
    let docs = vec![
        doc("a", "Distinct one about volcanoes."),
        doc("b", "Distinct two about glaciers."),
        doc("c", "Distinct three about deserts."),
    ];
    let engine = default_engine();
    let extracted = engine.extract_evidence("distinct", &docs);
    let fused = engine.fuse("distinct", &docs).unwrap();
    // Without dedup collisions, fused evidence count equals extracted count.
    assert_eq!(extracted.len(), fused.evidence.len());
}
