//! Miscellaneous [`CorpusCurator`] wiring tests: error paths, empty inputs,
//! and the `Flag` vs `Drop` decontamination behaviours that the pillar tests
//! don't individually cover.

use crate::corpus_curation::{
    BenchmarkItem, ClassifierExample, ContaminationConfig, CorpusCurator, CurationConfig,
    CurationError, DecontaminationAction, NearDupConfig, RejectionStage,
};

use super::fixtures::doc;

/// Pad `sentence` with generated filler words until the whole text safely
/// clears the default heuristic rule set's 50-word minimum (with margin),
/// so tests that are not *about* the heuristic gate don't accidentally trip
/// it.
fn long_enough_text(sentence: &str) -> String {
    // Interleaving a real stop word ("the") with each distinct filler token
    // clears both the 50-word minimum and the stop-word-ratio minimum of the
    // default heuristic rule set with generous headroom, regardless of how
    // many words (or stop words) `sentence` itself contributes.
    let padding = (0..60)
        .map(|i| format!("the padword{i}"))
        .collect::<Vec<_>>()
        .join(" ");
    format!("{sentence}. {padding}.")
}

#[test]
fn empty_corpus_curates_to_an_empty_but_valid_report() {
    let curator = CorpusCurator::default();
    let report = curator
        .curate(&[], None)
        .expect("empty corpus is not an error");
    assert!(report.admitted_ids.is_empty());
    assert!(report.rejected.is_empty());
    assert_eq!(report.summary.total_documents, 0);
    assert_eq!(report.summary.admitted, 0);
}

#[test]
fn classifier_stage_without_a_trained_classifier_errors() {
    let config = CurationConfig::new().with_use_classifier(true);
    let curator = CorpusCurator::new(config);
    let docs = vec![doc("d1", long_enough_text("irrelevant content"))];

    assert_eq!(
        curator.score_quality(&docs[0]).unwrap_err(),
        CurationError::NoClassifierTrained
    );
    assert_eq!(
        curator.quality_report(&docs).unwrap_err(),
        CurationError::NoClassifierTrained
    );
    assert_eq!(
        curator.curate(&docs, None).unwrap_err(),
        CurationError::NoClassifierTrained
    );
}

#[test]
fn training_a_classifier_unblocks_the_classifier_stage() {
    let config = CurationConfig::new().with_use_classifier(true);
    let mut curator = CorpusCurator::new(config);
    assert!(curator.classifier().is_none());

    let examples = vec![
        ClassifierExample::good("well written informative helpful content"),
        ClassifierExample::bad("buy now click free win prize"),
    ];
    curator
        .train_classifier(&examples)
        .expect("training succeeds");
    assert!(curator.classifier().is_some());

    let document = doc(
        "d1",
        long_enough_text("well written informative helpful content"),
    );
    // No longer errors now that a classifier is attached.
    curator
        .score_quality(&document)
        .expect("classifier is attached");
}

#[test]
fn flag_action_keeps_contaminated_documents_admitted_but_flagged() {
    let config = CurationConfig::new()
        .with_contamination(ContaminationConfig::new().with_action(DecontaminationAction::Flag));
    let curator = CorpusCurator::new(config);

    let benchmark_text =
        "a distinctive benchmark passage about a fictional harbor and its old lighthouse keeper";
    let benchmark = vec![BenchmarkItem::new("b1", benchmark_text)];
    let host = long_enough_text("This paragraph exists only to carry the planted passage below");
    let docs = vec![doc("leaked", format!("{host} {benchmark_text} {host}"))];

    let report = curator
        .curate(&docs, Some(&benchmark))
        .expect("curation succeeds");
    assert!(
        report.is_admitted("leaked"),
        "Flag must not remove the document"
    );
    assert_eq!(report.flagged_contaminated_ids, vec!["leaked".to_string()]);
    assert!(report.rejection("leaked").is_none());
}

#[test]
fn drop_action_removes_contaminated_documents() {
    let config = CurationConfig::new()
        .with_contamination(ContaminationConfig::new().with_action(DecontaminationAction::Drop));
    let curator = CorpusCurator::new(config);

    let benchmark_text =
        "a distinctive benchmark passage about a fictional harbor and its old lighthouse keeper";
    let benchmark = vec![BenchmarkItem::new("b1", benchmark_text)];
    let host = long_enough_text("This paragraph exists only to carry the planted passage below");
    let docs = vec![doc("leaked", format!("{host} {benchmark_text} {host}"))];

    let report = curator
        .curate(&docs, Some(&benchmark))
        .expect("curation succeeds");
    assert!(
        !report.is_admitted("leaked"),
        "Drop must remove the document"
    );
    assert_eq!(report.flagged_contaminated_ids, vec!["leaked".to_string()]);
    let rejection = report.rejection("leaked").expect("rejected");
    assert_eq!(rejection.stage, RejectionStage::Contamination);
}

#[test]
fn near_dup_stage_can_be_disabled() {
    let config = CurationConfig::new().with_near_dup(NearDupConfig::new().with_enabled(false));
    let curator = CorpusCurator::new(config);

    let long_text =
        long_enough_text("near-duplicate handling is the only thing this test exercises");
    let docs = vec![doc("a", &long_text), doc("b", &long_text)];
    let report = curator.curate(&docs, None).expect("curation succeeds");

    // With the pillar disabled, exact duplicates are NOT deduplicated.
    assert_eq!(report.admitted_ids.len(), 2);
    assert!(report.near_duplicate_clusters.is_empty());
}

#[test]
fn near_dup_stage_removes_exact_duplicates_when_enabled() {
    let curator = CorpusCurator::default(); // near_dup.enabled defaults to true
    let long_text = long_enough_text("this document will be inserted twice under different ids");
    let docs = vec![doc("a", &long_text), doc("b", &long_text)];
    let report = curator.curate(&docs, None).expect("curation succeeds");

    assert_eq!(
        report.admitted_ids.len(),
        1,
        "one of the two exact duplicates is removed"
    );
    assert_eq!(report.near_duplicate_clusters.len(), 1);
}
