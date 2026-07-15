//! Pillar 4 tests: near-duplicate composition on
//! [`crate::lsh_index::MinHashIndex`] actually composes — it clusters known
//! duplicates (`Jaccard` above threshold) and separates known non-duplicates
//! (below), using the existing index rather than new hashing code.

use crate::corpus_curation::{CurationRng, NearDupConfig, find_near_duplicate_clusters};
use crate::lsh_index::{LshConfig, MinHashIndex};

use super::fixtures::{NEARDUP_VOCAB, doc, draw_text};

/// (d) `find_near_duplicate_clusters` clusters three mutually near-identical
/// documents and leaves three mutually unrelated documents unclustered.
#[test]
fn composed_minhash_index_clusters_duplicates_and_separates_non_duplicates() {
    let mut rng = CurationRng::new(0x2468_ACE0_1357_9BDF);
    let base_text = draw_text(&mut rng, NEARDUP_VOCAB, 30);
    let mut base_words: Vec<&str> = base_text.split_whitespace().collect();

    // Group A: three near-duplicates of `base_text`, differing from it (and
    // from each other) only at word position 10.
    base_words[10] = "aurora";
    let variant_a = base_words.join(" ");
    base_words[10] = "basalt";
    let variant_b = base_words.join(" ");
    base_words[10] = "cascade";
    let variant_c = base_words.join(" ");

    // Group B: three mutually unrelated documents (independent draws).
    let unrelated_a = draw_text(&mut rng, NEARDUP_VOCAB, 30);
    let unrelated_b = draw_text(&mut rng, NEARDUP_VOCAB, 30);
    let unrelated_c = draw_text(&mut rng, NEARDUP_VOCAB, 30);

    let docs = vec![
        doc("dup-1", variant_a),
        doc("dup-2", variant_b),
        doc("dup-3", variant_c),
        doc("uniq-1", unrelated_a),
        doc("uniq-2", unrelated_b),
        doc("uniq-3", unrelated_c),
    ];

    let config = NearDupConfig::new()
        .with_shingle_size(4)
        .with_jaccard_threshold(0.5);
    let clusters =
        find_near_duplicate_clusters(&docs, &config).expect("index composition succeeds");

    assert_eq!(
        clusters.len(),
        1,
        "exactly one duplicate cluster should form"
    );
    let mut members = clusters[0].member_ids.clone();
    members.sort();
    assert_eq!(
        members,
        vec![
            "dup-1".to_string(),
            "dup-2".to_string(),
            "dup-3".to_string()
        ]
    );
    assert_eq!(
        clusters[0].representative_id, "dup-1",
        "lowest input index wins as representative"
    );
    let mut duplicates = clusters[0].duplicates();
    duplicates.sort_unstable();
    assert_eq!(duplicates, vec!["dup-2", "dup-3"]);

    for id in ["uniq-1", "uniq-2", "uniq-3"] {
        assert!(
            clusters
                .iter()
                .all(|c| !c.member_ids.iter().any(|m| m == id)),
            "{id} must not be clustered as a near-duplicate of anything"
        );
    }
}

/// A second, lower-level measurement: query the exact same
/// [`MinHashIndex`] the pillar composes and check the raw `Jaccard` scores it
/// reports for a pair with a hand-computed high overlap versus a pair with a
/// hand-computed low overlap, so "above/below threshold" is read off an
/// exact, checked number rather than inferred from cluster membership.
/// Element sets are literal (not text-derived) so the expected `Jaccard` is
/// exact by construction: `search` re-ranks its candidates by *exact*
/// `Jaccard` over the raw sets (only *gathering* candidates uses min-hash),
/// so the score returned here is not an estimate at all.
#[test]
fn composed_index_reports_jaccard_above_threshold_for_duplicates_and_below_for_others() {
    // base = {0, 1, ..., 19} (20 elements).
    let base_set: Vec<u64> = (0..20).collect();
    // near_dup shares 18 of base's 20 elements, plus 2 of its own:
    // |intersection| = 18, |union| = 22, Jaccard = 18/22 ~= 0.818.
    let near_dup_set: Vec<u64> = (0..18).chain([1_000, 1_001]).collect();
    // unrelated shares only 4 of base's 20 elements, plus 16 of its own:
    // |intersection| = 4, |union| = 36, Jaccard = 4/36 ~= 0.111.
    let unrelated_set: Vec<u64> = (0..4).chain(2_000..2_016).collect();

    // 64 single-row bands: candidacy only needs ONE of 64 independent
    // min-hash functions to agree, which is overwhelmingly likely for both
    // pairs here (true Jaccard 0.818 and 0.111 respectively) even though
    // only the second is below the clustering threshold.
    let lsh_config = LshConfig::new().with_num_bands(64).with_rows_per_band(1);
    let mut index = MinHashIndex::new(lsh_config).expect("valid config");
    index
        .insert("base", base_set.clone())
        .expect("non-empty element set");
    index
        .insert("near_dup", near_dup_set)
        .expect("non-empty element set");
    index
        .insert("unrelated", unrelated_set)
        .expect("non-empty element set");

    let hits = index.search(&base_set, 3).expect("index is non-empty");
    let near_dup_score = hits
        .iter()
        .find(|h| h.id == "near_dup")
        .expect("a Jaccard-0.818 pair must collide in at least one of 64 bands")
        .score;
    let unrelated_score = hits
        .iter()
        .find(|h| h.id == "unrelated")
        .expect("a Jaccard-0.111 pair must collide in at least one of 64 bands")
        .score;

    assert!(
        (f64::from(near_dup_score) - 18.0 / 22.0).abs() < 1e-6,
        "near_dup exact Jaccard {near_dup_score} should equal 18/22"
    );
    assert!(
        (f64::from(unrelated_score) - 4.0 / 36.0).abs() < 1e-6,
        "unrelated exact Jaccard {unrelated_score} should equal 4/36"
    );
    assert!(
        near_dup_score >= 0.5,
        "near-duplicate Jaccard {near_dup_score} should be >= 0.5"
    );
    assert!(
        unrelated_score < 0.5,
        "unrelated Jaccard {unrelated_score} should be < 0.5"
    );
}

#[test]
fn empty_and_single_document_corpora_yield_no_clusters() {
    let config = NearDupConfig::new();
    assert_eq!(
        find_near_duplicate_clusters(&[], &config).unwrap(),
        Vec::new()
    );

    let docs = vec![doc(
        "only",
        "a single lonely document with nothing to duplicate",
    )];
    assert_eq!(
        find_near_duplicate_clusters(&docs, &config).unwrap(),
        Vec::new()
    );
}

#[test]
fn documents_with_no_extractable_shingles_are_never_clustered() {
    let config = NearDupConfig::new();
    let docs = vec![
        doc("empty-1", ""),
        doc("empty-2", ""),
        doc("real", "some real content here"),
    ];
    let clusters = find_near_duplicate_clusters(&docs, &config).expect("succeeds");
    assert!(clusters.is_empty());
}
