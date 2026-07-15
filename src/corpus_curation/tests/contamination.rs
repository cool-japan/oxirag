//! Pillar 3 tests: eval-set contamination detection with **planted ground
//! truth**.

use crate::corpus_curation::{
    BenchmarkItem, ContaminationConfig, ContaminationDetector, ContaminationKind, CurationRng,
};

use super::fixtures::{BENCH_VOCAB, CLEAN_VOCAB, doc, draw_text};

/// Change the two words at the midpoint of `text` to fixed tokens that never
/// occur naturally in [`BENCH_VOCAB`] draws, guaranteeing the result is *not*
/// a verbatim copy of `text` while leaving the great majority of its word
/// n-grams untouched.
fn perturb_two_adjacent_words(text: &str) -> String {
    let mut words: Vec<&str> = text.split_whitespace().collect();
    let mid = words.len() / 2;
    words[mid] = "zzzalpha";
    words[mid + 1] = "zzzbeta";
    words.join(" ")
}

/// (c) Plant `K = 6` benchmark items (3 verbatim, 3 near-verbatim) into a
/// corpus of `N = 20` documents; assert all 6 are detected, that the other 14
/// ("clean") documents are never flagged, and that the reported corpus-level
/// contamination rate equals the known planted fraction `K / N = 0.3`.
#[test]
#[allow(clippy::too_many_lines)] // one deliberately end-to-end scenario: plant, detect, verify
// every angle (detection, false positives, rate, kind classification) against the same corpus.
fn planted_contamination_is_fully_detected_with_no_false_positives() {
    let mut rng = CurationRng::new(0x1357_9BDF_2468_ACE0);

    // 6 benchmark items, each a distinctive 40-word passage.
    let benchmark: Vec<BenchmarkItem> = (0..6)
        .map(|i| BenchmarkItem::new(format!("bench-{i}"), draw_text(&mut rng, BENCH_VOCAB, 40)))
        .collect();

    let mut corpus = Vec::new();
    let mut verbatim_ids = Vec::new();
    let mut near_verbatim_ids = Vec::new();

    // 3 corpus documents embed a benchmark item verbatim, wrapped in
    // unrelated host filler.
    for item in benchmark.iter().take(3) {
        let host = draw_text(&mut rng, CLEAN_VOCAB, 20);
        let content = format!("{host} {} {host}", item.text);
        let id = format!("verbatim-of-{}", item.id);
        corpus.push(doc(&id, content));
        verbatim_ids.push(id);
    }

    // 3 corpus documents embed a near-verbatim copy (2 adjacent words
    // changed): not an exact substring, but high n-gram overlap.
    for item in benchmark.iter().skip(3).take(3) {
        let host = draw_text(&mut rng, CLEAN_VOCAB, 20);
        let perturbed = perturb_two_adjacent_words(&item.text);
        let content = format!("{host} {perturbed} {host}");
        let id = format!("near-verbatim-of-{}", item.id);
        corpus.push(doc(&id, content));
        near_verbatim_ids.push(id);
    }

    // 14 clean documents sharing no vocabulary with the benchmark at all.
    let mut clean_ids = Vec::new();
    for i in 0..14 {
        let id = format!("clean-{i}");
        corpus.push(doc(&id, draw_text(&mut rng, CLEAN_VOCAB, 40)));
        clean_ids.push(id);
    }

    assert_eq!(corpus.len(), 20, "N = 20 corpus documents");
    assert_eq!(
        verbatim_ids.len() + near_verbatim_ids.len(),
        6,
        "K = 6 planted items"
    );

    let config = ContaminationConfig::new()
        .with_ngram_size(6)
        .with_near_verbatim_threshold(0.6);
    let detector = ContaminationDetector::new(config).expect("valid config");
    let report = detector.detect(&benchmark, &corpus);

    // All K = 6 planted items are detected.
    assert_eq!(
        report.dirty_item_count(),
        6,
        "all 6 planted items must be detected"
    );
    for verdict in &report.verdicts {
        assert!(
            verdict.dirty,
            "benchmark item {} should be dirty",
            verdict.benchmark_id
        );
    }

    // Exactly the 6 planted documents are implicated: zero false positives
    // among the 14 clean documents.
    let mut expected_dirty: Vec<String> = verbatim_ids
        .iter()
        .chain(near_verbatim_ids.iter())
        .cloned()
        .collect();
    expected_dirty.sort();
    assert_eq!(report.dirty_document_ids, expected_dirty);

    let false_positive_count = report
        .dirty_document_ids
        .iter()
        .filter(|id| clean_ids.contains(id))
        .count();
    assert_eq!(
        false_positive_count, 0,
        "no clean document should be flagged"
    );

    // Corpus-level contamination rate equals the known planted fraction
    // K / N = 6 / 20 = 0.3 exactly.
    assert!(
        (report.contamination_rate - 0.3).abs() < 1e-9,
        "rate {} should equal 6/20 = 0.3",
        report.contamination_rate
    );

    // Verbatim-planted documents are classified Verbatim; near-verbatim
    // ones are classified NearVerbatim (never Verbatim, since their text
    // was deliberately altered).
    for (item, expected_id) in benchmark.iter().take(3).zip(verbatim_ids.iter()) {
        let verdict = report
            .verdicts
            .iter()
            .find(|v| v.benchmark_id == item.id)
            .unwrap();
        let matched = verdict
            .matches
            .iter()
            .find(|m| &m.document_id == expected_id)
            .unwrap();
        assert_eq!(matched.kind, ContaminationKind::Verbatim);
        assert!((matched.ngram_overlap - 1.0).abs() < f64::EPSILON);
    }
    for (item, expected_id) in benchmark
        .iter()
        .skip(3)
        .take(3)
        .zip(near_verbatim_ids.iter())
    {
        let verdict = report
            .verdicts
            .iter()
            .find(|v| v.benchmark_id == item.id)
            .unwrap();
        let matched = verdict
            .matches
            .iter()
            .find(|m| &m.document_id == expected_id)
            .unwrap();
        assert_eq!(matched.kind, ContaminationKind::NearVerbatim);
        assert!(
            matched.ngram_overlap >= 0.6,
            "near-verbatim overlap {} should be >= the configured 0.6 threshold",
            matched.ngram_overlap
        );
        assert!(
            matched.ngram_overlap < 1.0,
            "a near-verbatim match must not be a perfect (verbatim) overlap"
        );
    }
}

#[test]
fn empty_benchmark_item_text_is_reported_clean() {
    let config = ContaminationConfig::new();
    let detector = ContaminationDetector::new(config).expect("valid config");
    let benchmark = vec![BenchmarkItem::new("empty", "   ")];
    let corpus = vec![doc("d1", "some perfectly ordinary unrelated content")];
    let report = detector.detect(&benchmark, &corpus);
    assert_eq!(report.dirty_item_count(), 0);
    assert!(report.dirty_document_ids.is_empty());
}

#[test]
fn empty_corpus_yields_zero_contamination_rate() {
    let config = ContaminationConfig::new();
    let detector = ContaminationDetector::new(config).expect("valid config");
    let benchmark = vec![BenchmarkItem::new("q1", "anything")];
    let report = detector.detect(&benchmark, &[]);
    assert!((report.contamination_rate - 0.0).abs() < f64::EPSILON);
    assert_eq!(report.dirty_item_count(), 0);
}

#[test]
fn invalid_config_is_rejected() {
    let config = ContaminationConfig::new().with_ngram_size(0);
    assert!(ContaminationDetector::new(config).is_err());

    let config = ContaminationConfig::new().with_near_verbatim_threshold(1.5);
    assert!(ContaminationDetector::new(config).is_err());
}
