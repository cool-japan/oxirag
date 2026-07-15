//! (e) End-to-end ablation: on a corpus salted with low-quality, redundant,
//! and contaminated documents, curation must remove **exactly** the injected
//! bad documents, and a downstream retrieval metric must **strictly
//! improve**.

use std::collections::HashSet;

use crate::corpus_curation::{
    BenchmarkItem, ContaminationConfig, CorpusCurator, CurationConfig, CurationRng,
    DecontaminationAction,
};
use crate::types::Document;

use super::fixtures::{BENCH_VOCAB, CLEAN_VOCAB, GOOD_VOCAB, doc, draw_text};

/// Naive lexical retrieval score: total occurrences of every keyword,
/// case-insensitively. Deliberately dumb (no normalisation, no length
/// discounting) so that keyword-stuffed spam — which repeats the query terms
/// dozens of times — outscores genuine on-topic prose that mentions them
/// naturally only a couple of times.
fn keyword_score(content: &str, keywords: &[&str]) -> usize {
    let lower = content.to_lowercase();
    keywords.iter().map(|k| lower.matches(k).count()).sum()
}

/// Precision at `k`: the fraction of the top-`k` documents (by
/// [`keyword_score`]) that are in `relevant_ids`.
fn precision_at_k(
    docs: &[&Document],
    keywords: &[&str],
    relevant_ids: &HashSet<&str>,
    k: usize,
) -> f64 {
    let mut scored: Vec<(&Document, usize)> = docs
        .iter()
        .map(|d| (*d, keyword_score(&d.content, keywords)))
        .collect();
    scored.sort_by(|a, b| {
        b.1.cmp(&a.1)
            .then_with(|| a.0.id.as_str().cmp(b.0.id.as_str()))
    });
    let top_k = scored.into_iter().take(k);
    let relevant_in_top_k = top_k
        .filter(|(d, _)| relevant_ids.contains(d.id.as_str()))
        .count();
    #[allow(clippy::cast_precision_loss)]
    {
        relevant_in_top_k as f64 / k as f64
    }
}

#[test]
#[allow(clippy::too_many_lines)] // one deliberately end-to-end scenario; splitting it would
// scatter the corpus construction away from the assertions that depend on it.
fn curation_removes_exactly_the_injected_bad_documents_and_improves_precision() {
    let mut rng = CurationRng::new(0x4242_1337_ABCD_EF01);
    let keywords = ["quantum", "computing"];

    // 10 genuine on-topic documents: real prose that happens to mention the
    // query terms a couple of times each, plus plenty of unrelated filler.
    let mut good_docs: Vec<Document> = Vec::new();
    for i in 0..10 {
        let filler = draw_text(&mut rng, GOOD_VOCAB, 55);
        let content = format!(
            "Recent quantum computing research suggests that {filler}. The quantum \
             computing approach continues to improve steadily in report number {i}."
        );
        good_docs.push(doc(&format!("good-{i}"), content));
    }

    // 3 keyword-stuffed spam documents: fail the heuristic stop-word-ratio
    // rule (near-zero stop words) while scoring enormously higher than any
    // genuine document on naive keyword count.
    let mut spam_docs: Vec<Document> = Vec::new();
    for i in 0..3 {
        let content = "quantum computing ".repeat(60);
        spam_docs.push(doc(&format!("spam-{i}"), content.trim().to_string()));
    }

    // 2 documents that leak a held-out benchmark item verbatim, on an
    // unrelated topic so they never compete for the query's top-k anyway;
    // their removal is checked directly via the curation report.
    let benchmark: Vec<BenchmarkItem> = (0..2)
        .map(|i| BenchmarkItem::new(format!("bench-{i}"), draw_text(&mut rng, BENCH_VOCAB, 40)))
        .collect();
    let mut contaminated_docs: Vec<Document> = Vec::new();
    for item in &benchmark {
        let host = draw_text(&mut rng, CLEAN_VOCAB, 20);
        let content = format!("{host} {} {host}", item.text);
        contaminated_docs.push(doc(&format!("contaminated-{}", item.id), content));
    }

    // 2 exact near-duplicate copies of good-0: redundant, not "irrelevant",
    // but still injected junk that a curator should remove.
    let duplicate_docs = [
        doc("dup-copy-1", good_docs[0].content.clone()),
        doc("dup-copy-2", good_docs[0].content.clone()),
    ];

    let mut all_docs = good_docs.clone();
    all_docs.extend(spam_docs.iter().cloned());
    all_docs.extend(contaminated_docs.iter().cloned());
    all_docs.extend(duplicate_docs.iter().cloned());
    assert_eq!(
        all_docs.len(),
        17,
        "10 good + 3 spam + 2 contaminated + 2 duplicate = 17"
    );

    let config = CurationConfig::new()
        .with_contamination(ContaminationConfig::new().with_action(DecontaminationAction::Drop));
    let curator = CorpusCurator::new(config);
    let report = curator
        .curate(&all_docs, Some(&benchmark))
        .expect("curation succeeds with default (no classifier) configuration");

    // Curation removes exactly the 7 injected bad documents: 3 spam + 2
    // contaminated + 2 duplicate copies.
    let good_ids: Vec<String> = good_docs
        .iter()
        .map(|d| d.id.as_str().to_string())
        .collect();
    let mut expected_admitted = good_ids.clone();
    expected_admitted.sort();
    let mut actual_admitted = report.admitted_ids.clone();
    actual_admitted.sort();
    assert_eq!(
        actual_admitted, expected_admitted,
        "admitted set must equal exactly the 10 good docs"
    );

    let mut bad_ids: Vec<String> = spam_docs
        .iter()
        .map(|d| d.id.as_str().to_string())
        .collect();
    bad_ids.extend(contaminated_docs.iter().map(|d| d.id.as_str().to_string()));
    bad_ids.extend(duplicate_docs.iter().map(|d| d.id.as_str().to_string()));
    assert_eq!(bad_ids.len(), 7, "K = 7 injected bad documents");
    for bad_id in &bad_ids {
        assert!(
            report.rejection(bad_id).is_some(),
            "{bad_id} must be rejected"
        );
        assert!(!report.is_admitted(bad_id), "{bad_id} must not be admitted");
    }
    assert_eq!(
        report.summary.rejected_heuristic, 3,
        "3 spam docs rejected on heuristics"
    );
    assert_eq!(
        report.summary.rejected_contamination, 2,
        "2 contaminated docs dropped"
    );
    assert_eq!(
        report.summary.rejected_near_duplicate, 2,
        "2 duplicate copies rejected"
    );
    assert_eq!(report.summary.admitted, 10);

    // Downstream metric: naive keyword-count precision@10, relevant = the 10
    // genuine docs plus the 2 (content-identical) duplicate copies, since
    // both are genuinely about the query topic.
    let mut relevant_ids: HashSet<&str> = good_ids.iter().map(String::as_str).collect();
    relevant_ids.insert("dup-copy-1");
    relevant_ids.insert("dup-copy-2");

    let uncurated_refs: Vec<&Document> = all_docs.iter().collect();
    let uncurated_precision = precision_at_k(&uncurated_refs, &keywords, &relevant_ids, 10);

    let curated_docs: Vec<&Document> = all_docs
        .iter()
        .filter(|d| report.is_admitted(d.id.as_str()))
        .collect();
    let curated_precision = precision_at_k(&curated_docs, &keywords, &relevant_ids, 10);
    eprintln!(
        "[ablation measurement] uncurated_precision@10={uncurated_precision:.4} curated_precision@10={curated_precision:.4}"
    );

    assert!(
        curated_precision > uncurated_precision,
        "curated precision@10 {curated_precision} must strictly exceed uncurated {uncurated_precision}"
    );
    // Measured on this seed: uncurated 0.7 (spam dominates 3 of the top 10
    // slots), curated 1.0 (only genuine documents remain).
    assert!(
        (curated_precision - 1.0).abs() < f64::EPSILON,
        "curated precision@10 should be a perfect 1.0, got {curated_precision}"
    );
    assert!(
        uncurated_precision <= 0.8,
        "uncurated precision@10 should be visibly degraded by spam, got {uncurated_precision}"
    );
}
