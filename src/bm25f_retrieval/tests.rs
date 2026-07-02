#![allow(
    clippy::float_cmp,
    clippy::similar_names,
    clippy::cast_precision_loss,
    clippy::cast_possible_truncation,
    clippy::too_many_lines,
    clippy::unreadable_literal
)]
//! Tests for the `bm25f_retrieval` module.

use crate::bm25f_retrieval::engine::{Bm25fIndex, tokenize};
use crate::bm25f_retrieval::types::{
    Bm25fConfig, Bm25fDocument, Bm25fError, Bm25fField, Bm25fFieldWeight, Bm25fResult,
};

// ── Test helpers ─────────────────────────────────────────────────────────────

/// Build a document with exactly one named field.
fn single_field_doc(id: &str, field: &str, text: &str) -> Bm25fDocument {
    Bm25fDocument::new(id).with_field(field, text)
}

// ── Bm25fField ───────────────────────────────────────────────────────────────

#[test]
fn bm25f_field_new() {
    let field = Bm25fField::new("title", "hello world");
    assert_eq!(field.name, "title");
    assert_eq!(field.text, "hello world");
}

// ── Bm25fDocument ────────────────────────────────────────────────────────────

#[test]
fn bm25f_document_new_has_no_fields() {
    let doc = Bm25fDocument::new("d1");
    assert_eq!(doc.id, "d1");
    assert!(doc.fields.is_empty());
}

#[test]
fn bm25f_document_with_field_builder_appends_in_order() {
    let doc = Bm25fDocument::new("d1")
        .with_field("title", "hello")
        .with_field("body", "world");
    assert_eq!(doc.fields.len(), 2);
    assert_eq!(doc.fields[0].name, "title");
    assert_eq!(doc.fields[1].name, "body");
}

#[test]
fn bm25f_document_field_accessor_found() {
    let doc = Bm25fDocument::new("d1").with_field("title", "hello world");
    assert_eq!(doc.field("title"), Some("hello world"));
}

#[test]
fn bm25f_document_field_accessor_missing() {
    let doc = Bm25fDocument::new("d1").with_field("title", "hello world");
    assert_eq!(doc.field("body"), None);
}

#[test]
fn bm25f_document_field_accessor_first_match_when_duplicate_names() {
    let doc = Bm25fDocument::new("d1")
        .with_field("tags", "first")
        .with_field("tags", "second");
    assert_eq!(doc.field("tags"), Some("first"));
}

// ── Bm25fFieldWeight ─────────────────────────────────────────────────────────

#[test]
fn bm25f_field_weight_default() {
    let w = Bm25fFieldWeight::default();
    assert!((w.weight - 1.0).abs() < f32::EPSILON);
    assert!((w.b - 0.75).abs() < f32::EPSILON);
}

#[test]
fn bm25f_field_weight_new() {
    let w = Bm25fFieldWeight::new(2.5, 0.3);
    assert!((w.weight - 2.5).abs() < f32::EPSILON);
    assert!((w.b - 0.3).abs() < f32::EPSILON);
}

// ── Bm25fConfig ──────────────────────────────────────────────────────────────

#[test]
fn bm25f_config_default_values() {
    let config = Bm25fConfig::default();
    assert!((config.k1 - 1.2).abs() < f32::EPSILON);
    assert!((config.idf_smoothing - 0.5).abs() < f32::EPSILON);
    assert!(config.field_weights.is_empty());
}

#[test]
fn bm25f_config_new_equals_default() {
    assert_eq!(Bm25fConfig::new(), Bm25fConfig::default());
}

#[test]
fn bm25f_config_with_k1() {
    let config = Bm25fConfig::new().with_k1(2.0);
    assert!((config.k1 - 2.0).abs() < f32::EPSILON);
}

#[test]
fn bm25f_config_with_idf_smoothing() {
    let config = Bm25fConfig::new().with_idf_smoothing(1.0);
    assert!((config.idf_smoothing - 1.0).abs() < f32::EPSILON);
}

#[test]
fn bm25f_config_with_field_weight_overrides() {
    let config = Bm25fConfig::new().with_field_weight("title", Bm25fFieldWeight::new(2.0, 0.1));
    assert_eq!(config.field_weights.len(), 1);
    assert_eq!(
        config.field_weights["title"],
        Bm25fFieldWeight::new(2.0, 0.1)
    );
}

#[test]
fn bm25f_config_field_weight_returns_default_when_unset() {
    let config = Bm25fConfig::new();
    assert_eq!(config.field_weight("anything"), Bm25fFieldWeight::default());
}

#[test]
fn bm25f_config_field_weight_returns_configured_when_set() {
    let config = Bm25fConfig::new().with_field_weight("title", Bm25fFieldWeight::new(2.0, 0.1));
    assert_eq!(
        config.field_weight("title"),
        Bm25fFieldWeight::new(2.0, 0.1)
    );
    assert_eq!(config.field_weight("body"), Bm25fFieldWeight::default());
}

// ── Bm25fError / Bm25fResult ─────────────────────────────────────────────────

#[test]
fn bm25f_error_display_messages() {
    assert_eq!(Bm25fError::EmptyCorpus.to_string(), "corpus is empty");
    assert_eq!(
        Bm25fError::EmptyQuery.to_string(),
        "query must not be empty"
    );
    assert_eq!(Bm25fError::NotBuilt.to_string(), "index not built");
}

#[test]
fn bm25f_error_equality() {
    assert_eq!(Bm25fError::EmptyCorpus, Bm25fError::EmptyCorpus);
    assert_ne!(Bm25fError::EmptyCorpus, Bm25fError::EmptyQuery);
}

#[test]
fn bm25f_result_type_alias_is_usable() {
    fn helper(ok: bool) -> Bm25fResult<i32> {
        if ok {
            Ok(42)
        } else {
            Err(Bm25fError::EmptyQuery)
        }
    }
    assert_eq!(helper(true).unwrap(), 42);
    assert!(helper(false).is_err());
}

// ── tokenize ─────────────────────────────────────────────────────────────────

#[test]
fn tokenize_lowercases_and_strips_punctuation() {
    let tokens = tokenize("Rust, Rocks! (really)");
    assert_eq!(tokens, vec!["rust", "rocks", "really"]);
}

#[test]
fn tokenize_splits_on_whitespace_and_punctuation() {
    // '-' and '_' are both non-alphanumeric, so both act as split points too.
    let tokens = tokenize("alpha-beta_gamma delta.epsilon");
    assert_eq!(tokens, vec!["alpha", "beta", "gamma", "delta", "epsilon"]);
}

#[test]
fn tokenize_empty_text_yields_nothing() {
    assert!(tokenize("").is_empty());
    assert!(tokenize("   ...!!!   ").is_empty());
}

// ── Bm25fIndex: construction & validation ───────────────────────────────────

#[test]
fn build_rejects_empty_corpus() {
    let mut index = Bm25fIndex::new(Bm25fConfig::default());
    let empty: Vec<Bm25fDocument> = Vec::new();
    let err = index.build(&empty).unwrap_err();
    assert!(matches!(err, Bm25fError::EmptyCorpus));
}

#[test]
fn search_before_build_rejects_not_built() {
    let index = Bm25fIndex::new(Bm25fConfig::default());
    let err = index.search("term", 10).unwrap_err();
    assert!(matches!(err, Bm25fError::NotBuilt));
}

#[test]
fn search_rejects_empty_string_query() {
    let mut index = Bm25fIndex::new(Bm25fConfig::default());
    index
        .build(&[single_field_doc("d1", "body", "alpha")])
        .unwrap();
    let err = index.search("", 10).unwrap_err();
    assert!(matches!(err, Bm25fError::EmptyQuery));
}

#[test]
fn search_rejects_whitespace_only_query() {
    let mut index = Bm25fIndex::new(Bm25fConfig::default());
    index
        .build(&[single_field_doc("d1", "body", "alpha")])
        .unwrap();
    let err = index.search("   \t  ", 10).unwrap_err();
    assert!(matches!(err, Bm25fError::EmptyQuery));
}

#[test]
fn build_all_fields_empty_or_missing_returns_ok_and_search_returns_empty_results() {
    let corpus = vec![
        Bm25fDocument::new("d1"),
        Bm25fDocument::new("d2").with_field("body", ""),
        Bm25fDocument::new("d3").with_field("title", "   "),
    ];
    let mut index = Bm25fIndex::new(Bm25fConfig::default());
    assert!(index.build(&corpus).is_ok());
    assert_eq!(index.len(), 3);
    assert!(index.is_built());

    let hits = index.search("anything meaningful here", 10).unwrap();
    assert!(hits.is_empty());
}

#[test]
fn build_documents_with_zero_fields_does_not_panic() {
    let mut index = Bm25fIndex::new(Bm25fConfig::default());
    let corpus = vec![Bm25fDocument::new("d1")];
    assert!(index.build(&corpus).is_ok());
    let hits = index.search("anything", 10).unwrap();
    assert!(hits.is_empty());
}

#[test]
fn index_len_and_is_empty_before_and_after_build() {
    let mut index = Bm25fIndex::new(Bm25fConfig::default());
    assert_eq!(index.len(), 0);
    assert!(index.is_empty());

    let corpus = vec![
        single_field_doc("d1", "body", "alpha"),
        single_field_doc("d2", "body", "beta"),
    ];
    index.build(&corpus).unwrap();
    assert_eq!(index.len(), 2);
    assert!(!index.is_empty());
}

#[test]
fn index_is_built_flag_before_and_after_build() {
    let mut index = Bm25fIndex::new(Bm25fConfig::default());
    assert!(!index.is_built());
    index
        .build(&[single_field_doc("d1", "body", "alpha")])
        .unwrap();
    assert!(index.is_built());
}

#[test]
fn index_default_produces_empty_unbuilt_index() {
    let index = Bm25fIndex::default();
    assert!(index.is_empty());
    assert!(!index.is_built());
    assert_eq!(index.config(), &Bm25fConfig::default());
}

#[test]
fn index_config_accessor_returns_constructed_config() {
    let config = Bm25fConfig::new().with_k1(3.3);
    let index = Bm25fIndex::new(config.clone());
    assert_eq!(index.config(), &config);
}

#[test]
fn rebuilding_index_replaces_previous_statistics() {
    let mut index = Bm25fIndex::new(Bm25fConfig::default());
    index
        .build(&[single_field_doc("d1", "body", "foo")])
        .unwrap();
    assert_eq!(index.document_frequency("foo"), 1);

    index
        .build(&[
            single_field_doc("e1", "body", "bar"),
            single_field_doc("e2", "body", "bar"),
        ])
        .unwrap();
    assert_eq!(index.document_frequency("foo"), 0);
    assert_eq!(index.document_frequency("bar"), 2);
    assert_eq!(index.len(), 2);
}

// ── Bm25fIndex: pseudo-frequency (hand-calculated) ──────────────────────────

#[test]
fn pseudo_frequency_hand_calculated_two_field_single_document() {
    let config = Bm25fConfig::new()
        .with_field_weight("title", Bm25fFieldWeight::new(2.0, 0.5))
        .with_field_weight("body", Bm25fFieldWeight::new(1.0, 0.5));

    let corpus = vec![
        Bm25fDocument::new("d1")
            .with_field("title", "alpha alpha beta")
            .with_field("body", "alpha gamma gamma gamma"),
    ];

    let mut index = Bm25fIndex::new(config);
    index.build(&corpus).unwrap();

    // Single-document corpus: avg_len_title = 3/1 = 3.0, avg_len_body = 4/1
    // = 4.0, so each field's length ratio is exactly 1.0 and
    // norm = 1 - b + b*1 = 1.0 regardless of b. This isolates the
    // per-field weighting and cross-field summation; length-normalisation
    // behaviour itself is covered separately below.
    let expected_alpha = 2.0_f32 * 2.0 / 1.0 + 1.0_f32 * 1.0 / 1.0; // title(w*tf) + body(w*tf)
    let expected_beta = 2.0_f32 * 1.0 / 1.0; // title only
    let expected_gamma = 1.0_f32 * 3.0 / 1.0; // body only
    assert_eq!(expected_alpha, 5.0);
    assert_eq!(expected_beta, 2.0);
    assert_eq!(expected_gamma, 3.0);

    assert!((index.pseudo_frequency("d1", "alpha") - expected_alpha).abs() < 1e-4);
    assert!((index.pseudo_frequency("d1", "beta") - expected_beta).abs() < 1e-4);
    assert!((index.pseudo_frequency("d1", "gamma") - expected_gamma).abs() < 1e-4);
}

// ── Bm25fIndex: field-weight isolation ──────────────────────────────────────

#[test]
fn term_in_high_weight_field_scores_higher_than_low_weight_field() {
    let config = Bm25fConfig::new()
        .with_field_weight("title", Bm25fFieldWeight::new(3.0, 0.0))
        .with_field_weight("body", Bm25fFieldWeight::new(1.0, 0.0));

    // All else is symmetric between the two documents (identical field
    // lengths, identical other content); only *which* field carries
    // "signal" differs.
    let corpus = vec![
        Bm25fDocument::new("doc_a")
            .with_field("title", "signal report")
            .with_field("body", "alpha beta"),
        Bm25fDocument::new("doc_b")
            .with_field("title", "alpha beta")
            .with_field("body", "signal report"),
    ];

    let mut index = Bm25fIndex::new(config);
    index.build(&corpus).unwrap();

    let tf_a = index.pseudo_frequency("doc_a", "signal");
    let tf_b = index.pseudo_frequency("doc_b", "signal");
    assert!((tf_a - 3.0).abs() < 1e-4, "tf_a = {tf_a}");
    assert!((tf_b - 1.0).abs() < 1e-4, "tf_b = {tf_b}");
    assert!(tf_a > tf_b);

    let hits = index.search("signal", 10).unwrap();
    assert_eq!(hits.len(), 2);
    assert_eq!(hits[0].doc_id, "doc_a");
    assert_eq!(hits[1].doc_id, "doc_b");
    assert!(hits[0].score > hits[1].score);
}

#[test]
fn field_weight_zero_excludes_field_from_scoring() {
    let config = Bm25fConfig::new()
        .with_field_weight("hidden", Bm25fFieldWeight::new(0.0, 0.5))
        .with_field_weight("visible", Bm25fFieldWeight::new(1.0, 0.5));

    let corpus = vec![
        Bm25fDocument::new("d1")
            .with_field("hidden", "unique_word")
            .with_field("visible", "other text"),
    ];

    let mut index = Bm25fIndex::new(config);
    index.build(&corpus).unwrap();

    // Document frequency is tracked regardless of field weight (spec: "any
    // field"), so IDF is still well-defined and positive...
    assert_eq!(index.document_frequency("unique_word"), 1);
    assert!(index.idf("unique_word") > 0.0);
    // ...but the combined pseudo-frequency is zero, since the only field
    // carrying the term has weight 0, so it can never actually score.
    assert!(index.pseudo_frequency("d1", "unique_word").abs() < 1e-6);

    let hits = index.search("unique_word", 10).unwrap();
    assert!(
        hits.is_empty(),
        "a term only present in a zero-weight field should never produce a hit: {hits:?}"
    );
}

// ── Bm25fIndex: per-field b independence ────────────────────────────────────

#[test]
fn per_field_b_zero_does_not_penalize_long_field_length() {
    let config = Bm25fConfig::new()
        .with_field_weight("title", Bm25fFieldWeight::new(1.0, 0.0))
        .with_field_weight("body", Bm25fFieldWeight::new(1.0, 0.75));

    let corpus = vec![
        Bm25fDocument::new("short_title")
            .with_field("title", "focus")
            .with_field("body", "filler"),
        Bm25fDocument::new("long_title")
            .with_field(
                "title",
                "focus filler filler filler filler filler filler filler filler filler",
            )
            .with_field("body", "filler"),
    ];

    let mut index = Bm25fIndex::new(config);
    index.build(&corpus).unwrap();

    assert!((index.average_field_length("title") - 5.5).abs() < 1e-4);

    let tf_short = index.pseudo_frequency("short_title", "focus");
    let tf_long = index.pseudo_frequency("long_title", "focus");
    assert!(
        (tf_short - tf_long).abs() < 1e-4,
        "b_title = 0 must make length irrelevant: tf_short = {tf_short}, tf_long = {tf_long}"
    );
    assert!((tf_short - 1.0).abs() < 1e-4);
}

#[test]
fn per_field_b_positive_penalizes_long_field_length() {
    let config = Bm25fConfig::new()
        .with_field_weight("title", Bm25fFieldWeight::new(1.0, 0.0))
        .with_field_weight("body", Bm25fFieldWeight::new(1.0, 0.75));

    let corpus = vec![
        Bm25fDocument::new("short_body")
            .with_field("title", "filler")
            .with_field("body", "focus"),
        Bm25fDocument::new("long_body")
            .with_field("title", "filler")
            .with_field(
                "body",
                "focus filler filler filler filler filler filler filler filler filler",
            ),
    ];

    let mut index = Bm25fIndex::new(config);
    index.build(&corpus).unwrap();

    let avg_body = index.average_field_length("body");
    assert!((avg_body - 5.5).abs() < 1e-4);

    let tf_short = index.pseudo_frequency("short_body", "focus");
    let tf_long = index.pseudo_frequency("long_body", "focus");

    let ratio_short = 1.0_f32 / avg_body;
    let norm_short = 1.0 - 0.75 + 0.75 * ratio_short;
    let expected_short = 1.0_f32 / norm_short;

    let ratio_long = 10.0_f32 / avg_body;
    let norm_long = 1.0 - 0.75 + 0.75 * ratio_long;
    let expected_long = 1.0_f32 / norm_long;

    assert!((tf_short - expected_short).abs() < 1e-4);
    assert!((tf_long - expected_long).abs() < 1e-4);
    assert!(
        tf_short > tf_long,
        "b_body = 0.75 must penalize the longer field: tf_short = {tf_short}, tf_long = {tf_long}"
    );
}

// ── Bm25fIndex: IDF ──────────────────────────────────────────────────────────

#[test]
fn idf_rare_term_scores_higher_than_common_term() {
    let corpus = vec![
        single_field_doc("d1", "body", "rareterm apple"),
        single_field_doc("d2", "body", "apple banana"),
        single_field_doc("d3", "body", "apple cherry"),
        single_field_doc("d4", "body", "apple date"),
        single_field_doc("d5", "body", "apple fig"),
    ];
    let mut index = Bm25fIndex::new(Bm25fConfig::default());
    index.build(&corpus).unwrap();

    assert_eq!(index.document_frequency("apple"), 5);
    assert_eq!(index.document_frequency("rareterm"), 1);
    assert!(index.idf("rareterm") > index.idf("apple"));
}

#[test]
fn idf_hand_calculated_matches_formula() {
    let corpus = vec![
        single_field_doc("d1", "body", "rareterm apple"),
        single_field_doc("d2", "body", "apple banana"),
        single_field_doc("d3", "body", "apple cherry"),
        single_field_doc("d4", "body", "apple date"),
        single_field_doc("d5", "body", "apple fig"),
    ];
    let config = Bm25fConfig::default();
    let smoothing = config.idf_smoothing;
    let mut index = Bm25fIndex::new(config);
    index.build(&corpus).unwrap();

    let n = 5.0_f32;
    let expected_rareterm = (1.0 + (n - 1.0 + 0.5) / (1.0 + smoothing)).ln();
    let expected_apple = (1.0 + (n - 5.0 + 0.5) / (5.0 + smoothing)).ln();

    assert!((index.idf("rareterm") - expected_rareterm).abs() < 1e-4);
    assert!((index.idf("apple") - expected_apple).abs() < 1e-4);
}

// ── Bm25fIndex: multi-term scoring / duplicate query terms ─────────────────

#[test]
fn multi_term_query_score_equals_sum_of_per_term_contributions() {
    let config = Bm25fConfig::new().with_field_weight("body", Bm25fFieldWeight::new(1.0, 0.0));
    let corpus = vec![
        single_field_doc("d1", "body", "alpha alpha beta gamma"),
        single_field_doc("d2", "body", "alpha delta"),
        single_field_doc("d3", "body", "epsilon zeta"),
    ];
    let mut index = Bm25fIndex::new(config);
    index.build(&corpus).unwrap();

    let k1 = index.config().k1;
    let idf_alpha = index.idf("alpha");
    let idf_beta = index.idf("beta");
    let tf_alpha = index.pseudo_frequency("d1", "alpha");
    let tf_beta = index.pseudo_frequency("d1", "beta");
    assert!((tf_alpha - 2.0).abs() < 1e-4);
    assert!((tf_beta - 1.0).abs() < 1e-4);

    let expected = idf_alpha * (k1 + 1.0) * tf_alpha / (k1 + tf_alpha)
        + idf_beta * (k1 + 1.0) * tf_beta / (k1 + tf_beta);

    let hits = index.search("alpha beta", 10).unwrap();
    let hit_d1 = hits.iter().find(|h| h.doc_id == "d1").unwrap();
    assert!((hit_d1.score - expected).abs() < 1e-4);
}

#[test]
fn duplicate_query_terms_do_not_double_count() {
    let corpus = vec![
        single_field_doc("d1", "body", "alpha alpha beta gamma"),
        single_field_doc("d2", "body", "alpha delta"),
    ];
    let mut index = Bm25fIndex::new(Bm25fConfig::default());
    index.build(&corpus).unwrap();

    let once = index.search("alpha beta", 10).unwrap();
    let repeated = index.search("alpha alpha alpha beta beta", 10).unwrap();
    assert_eq!(once, repeated);
}

// ── Bm25fIndex: saturation ───────────────────────────────────────────────────

#[test]
fn saturation_factor_never_reaches_limit_for_any_finite_tf() {
    let k1 = 1.2_f32;
    let limit = k1 + 1.0;
    // Mathematically (in real numbers) tf*(k1+1)/(k1+tf) < (k1+1) strictly
    // for every finite tf > 0. Up to around 1e5 the gap to the limit
    // (roughly limit * k1/tf) is still comfortably larger than f32's own
    // rounding error at this magnitude, so the strict inequality holds in
    // f32 too.
    for &tf in &[0.001_f32, 1.0, 10.0, 1_000.0, 100_000.0] {
        let factor = tf * (k1 + 1.0) / (k1 + tf);
        assert!(
            factor < limit,
            "factor {factor} should stay strictly below the limit {limit} for tf = {tf}"
        );
    }

    // At extreme tf (>= ~1e9), k1 becomes smaller than f32's precision at
    // that magnitude, so `k1 + tf` rounds to the same representable value
    // as `tf` itself and the computed factor legitimately rounds to
    // *exactly* the limit -- not a bug, just f32 saturating at its own
    // resolution. The invariant that actually matters for correctness still
    // holds: the factor never *exceeds* the limit, even in the face of
    // accumulated floating-point rounding across the multiply/add/divide.
    let extreme_factor = 1.0e9_f32 * (k1 + 1.0) / (k1 + 1.0e9_f32);
    assert!(
        extreme_factor <= limit,
        "factor {extreme_factor} must never exceed the limit {limit}, even under f32 rounding"
    );
}

#[test]
fn saturation_scores_monotonically_increase_and_approach_limit_with_high_repeat_count() {
    let config = Bm25fConfig::new().with_field_weight("body", Bm25fFieldWeight::new(1.0, 0.0));
    let repeat_counts = [1usize, 10, 100, 10_000, 100_000];
    let corpus: Vec<Bm25fDocument> = repeat_counts
        .iter()
        .enumerate()
        .map(|(i, &count)| {
            let text = "word ".repeat(count);
            single_field_doc(&format!("doc{i}"), "body", text.trim())
        })
        .collect();

    let mut index = Bm25fIndex::new(config);
    index.build(&corpus).unwrap();

    // Every document contains "word", so document frequency equals corpus
    // size for every document: IDF is one shared constant, isolating the
    // saturation curve's own shape from any IDF confound.
    assert_eq!(index.document_frequency("word"), repeat_counts.len());

    let hits = index.search("word", repeat_counts.len()).unwrap();
    assert_eq!(hits.len(), repeat_counts.len());

    let mut scores_in_repeat_order = vec![0.0_f32; repeat_counts.len()];
    for hit in &hits {
        let idx: usize = hit.doc_id.trim_start_matches("doc").parse().unwrap();
        scores_in_repeat_order[idx] = hit.score;
    }

    for window in scores_in_repeat_order.windows(2) {
        assert!(
            window[1] > window[0],
            "scores must strictly increase with repeat count: {scores_in_repeat_order:?}"
        );
    }

    let k1 = index.config().k1;
    let idf = index.idf("word");
    let limit = idf * (k1 + 1.0);
    for &score in &scores_in_repeat_order {
        assert!(
            score < limit,
            "score {score} must stay below the limit {limit}"
        );
    }

    let highest = *scores_in_repeat_order.last().unwrap();
    assert!(
        (limit - highest).abs() < limit * 1e-3,
        "with 100,000 repeats the score should be within 0.1% of the asymptote: \
         highest = {highest}, limit = {limit}"
    );
}

// ── Bm25fIndex: determinism ──────────────────────────────────────────────────

#[test]
fn determinism_two_independent_builds_produce_identical_hits() {
    let config = Bm25fConfig::new()
        .with_field_weight("title", Bm25fFieldWeight::new(2.0, 0.3))
        .with_field_weight("body", Bm25fFieldWeight::new(1.0, 0.75))
        .with_field_weight("tags", Bm25fFieldWeight::new(0.5, 0.1));

    let corpus = vec![
        Bm25fDocument::new("d1")
            .with_field("title", "Rust systems programming")
            .with_field(
                "body",
                "Rust is a systems programming language emphasizing safety and speed",
            )
            .with_field("tags", "rust systems safety"),
        Bm25fDocument::new("d2")
            .with_field("title", "Python data science")
            .with_field(
                "body",
                "Python is a dynamic programming language widely used in data science",
            )
            .with_field("tags", "python data"),
        Bm25fDocument::new("d3")
            .with_field("body", "Assorted notes with no title or tags field at all"),
        Bm25fDocument::new("d4")
            .with_field("title", "Rust and Python compared")
            .with_field("tags", "rust python comparison"),
    ];

    let mut results = Vec::new();
    for _ in 0..3 {
        let mut index = Bm25fIndex::new(config.clone());
        index.build(&corpus).unwrap();
        let hits = index
            .search("rust python programming language", 10)
            .unwrap();
        results.push(hits);
    }

    assert_eq!(results[0], results[1]);
    assert_eq!(results[1], results[2]);
    assert!(!results[0].is_empty());
}

#[test]
fn determinism_repeated_search_calls_produce_identical_hits() {
    let corpus = vec![
        single_field_doc("d1", "body", "alpha beta gamma"),
        single_field_doc("d2", "body", "alpha beta"),
    ];
    let mut index = Bm25fIndex::new(Bm25fConfig::default());
    index.build(&corpus).unwrap();

    let first = index.search("alpha beta gamma", 10).unwrap();
    for _ in 0..5 {
        let repeat = index.search("alpha beta gamma", 10).unwrap();
        assert_eq!(first, repeat);
    }
}

// ── Bm25fIndex: missing-field documents ─────────────────────────────────────

#[test]
fn missing_field_documents_build_and_search_without_panicking() {
    let config = Bm25fConfig::new()
        .with_field_weight("title", Bm25fFieldWeight::new(1.0, 0.75))
        .with_field_weight("body", Bm25fFieldWeight::new(1.0, 0.75));

    let corpus = vec![
        Bm25fDocument::new("d1")
            .with_field("title", "shared term")
            .with_field("body", "shared term extra"),
        Bm25fDocument::new("d2").with_field("title", "shared term"),
        Bm25fDocument::new("d3").with_field("body", "shared term extra words here"),
    ];

    let mut index = Bm25fIndex::new(config);
    index.build(&corpus).unwrap();

    let hits = index.search("shared term", 10).unwrap();
    assert_eq!(hits.len(), 3);
}

#[test]
fn missing_field_average_length_uses_full_corpus_denominator() {
    let config = Bm25fConfig::new()
        .with_field_weight("title", Bm25fFieldWeight::new(1.0, 0.75))
        .with_field_weight("body", Bm25fFieldWeight::new(1.0, 0.75));

    let corpus = vec![
        Bm25fDocument::new("d1")
            .with_field("title", "shared term")
            .with_field("body", "shared term extra"),
        Bm25fDocument::new("d2").with_field("title", "shared term"),
        Bm25fDocument::new("d3").with_field("body", "shared term extra words here"),
    ];

    let mut index = Bm25fIndex::new(config);
    index.build(&corpus).unwrap();

    // "body" tokens: d1 -> 3 ("shared term extra"), d2 -> missing (0), d3 ->
    // 5 ("shared term extra words here"); the average divides by the whole
    // corpus (3 documents), not just the 2 that actually carry a body.
    let expected_avg_body = (3.0_f32 + 0.0 + 5.0) / 3.0;
    assert!((index.average_field_length("body") - expected_avg_body).abs() < 1e-4);
}

#[test]
fn missing_field_pseudo_frequency_uses_only_present_fields() {
    let config = Bm25fConfig::new()
        .with_field_weight("title", Bm25fFieldWeight::new(1.0, 0.75))
        .with_field_weight("body", Bm25fFieldWeight::new(1.0, 0.75));

    let corpus = vec![
        Bm25fDocument::new("d1")
            .with_field("title", "shared term")
            .with_field("body", "shared term extra"),
        Bm25fDocument::new("d2").with_field("title", "shared term"),
        Bm25fDocument::new("d3").with_field("body", "shared term extra words here"),
    ];

    let mut index = Bm25fIndex::new(config);
    index.build(&corpus).unwrap();

    // d2 has no "body" field at all; its pseudo-frequency for "term" must
    // come from the title contribution alone, with no panic or NaN from the
    // missing field.
    let avg_title = index.average_field_length("title");
    let ratio = 2.0_f32 / avg_title;
    let norm = 1.0 - 0.75 + 0.75 * ratio;
    let expected_d2_term = 1.0_f32 / norm;

    let actual = index.pseudo_frequency("d2", "term");
    assert!((actual - expected_d2_term).abs() < 1e-4);
}

// ── Bm25fIndex: duplicate field name within one document ───────────────────

#[test]
fn duplicate_field_name_within_document_sums_independently() {
    let config = Bm25fConfig::new().with_field_weight("tags", Bm25fFieldWeight::new(1.0, 0.0));
    let corpus = vec![
        Bm25fDocument::new("d1")
            .with_field("tags", "red")
            .with_field("tags", "red blue"),
    ];

    let mut index = Bm25fIndex::new(config);
    index.build(&corpus).unwrap();

    // Each `Bm25fField` occurrence is its own additive term in the Σ_f sum;
    // same-named fields are not pre-merged before normalisation.
    assert!((index.pseudo_frequency("d1", "red") - 2.0).abs() < 1e-4);
    assert!((index.pseudo_frequency("d1", "blue") - 1.0).abs() < 1e-4);
}

// ── Bm25fIndex: ranking correctness ──────────────────────────────────────────

#[test]
fn ranking_correctness_hand_built_multi_field_corpus() {
    let config = Bm25fConfig::new()
        .with_field_weight("title", Bm25fFieldWeight::new(2.0, 0.0))
        .with_field_weight("body", Bm25fFieldWeight::new(1.0, 0.75));

    let corpus = vec![
        Bm25fDocument::new("doc_r")
            .with_field("title", "Rust programming")
            .with_field(
                "body",
                "Rust is a systems programming language emphasizing safety",
            ),
        Bm25fDocument::new("doc_p")
            .with_field("title", "Python tutorial")
            .with_field(
                "body",
                "Python is a dynamic programming language for scripting",
            ),
        Bm25fDocument::new("doc_c")
            .with_field("title", "Cooking recipes")
            .with_field(
                "body",
                "A collection of recipes for cooking pasta and bread",
            ),
    ];

    let mut index = Bm25fIndex::new(config);
    index.build(&corpus).unwrap();

    // Hand-reasoned expected ranking: "programming" appears in doc_r's
    // *title* (weight 2.0) as well as its body, but only in doc_p's body;
    // "language" appears in both docs' bodies only, with equal field
    // lengths, so it contributes equally to both. doc_r must therefore
    // outrank doc_p, and doc_c -- which matches neither query term at all
    // -- must be excluded entirely rather than ranked last at score 0.
    let hits = index.search("programming language", 10).unwrap();
    assert_eq!(hits.len(), 2, "doc_c should not appear at all: {hits:?}");
    assert_eq!(hits[0].doc_id, "doc_r");
    assert_eq!(hits[1].doc_id, "doc_p");
    assert!(hits[0].score > hits[1].score);

    // Cross-check the exact scores against the documented formula, composed
    // from the independently hand-verified idf / pseudo_frequency
    // primitives (see the dedicated IDF and pseudo-frequency tests above).
    let k1 = index.config().k1;
    let expected_score = |doc_id: &str| -> f32 {
        ["programming", "language"]
            .iter()
            .map(|term| {
                let idf = index.idf(term);
                let tf = index.pseudo_frequency(doc_id, term);
                if tf <= 0.0 {
                    0.0
                } else {
                    idf * (k1 + 1.0) * tf / (k1 + tf)
                }
            })
            .sum()
    };
    assert!((hits[0].score - expected_score("doc_r")).abs() < 1e-4);
    assert!((hits[1].score - expected_score("doc_p")).abs() < 1e-4);
}

// ── Bm25fIndex: search mechanics ─────────────────────────────────────────────

#[test]
fn search_respects_top_k_truncation() {
    let corpus = vec![
        single_field_doc("d1", "body", "alpha alpha alpha alpha"),
        single_field_doc("d2", "body", "alpha alpha alpha"),
        single_field_doc("d3", "body", "alpha alpha"),
        single_field_doc("d4", "body", "alpha"),
    ];
    let mut index = Bm25fIndex::new(Bm25fConfig::default());
    index.build(&corpus).unwrap();

    let full = index.search("alpha", 10).unwrap();
    assert_eq!(full.len(), 4);

    let truncated = index.search("alpha", 2).unwrap();
    assert_eq!(truncated.len(), 2);
    assert_eq!(truncated.as_slice(), &full[..2]);
}

#[test]
fn search_k_zero_returns_empty() {
    let corpus = vec![single_field_doc("d1", "body", "alpha")];
    let mut index = Bm25fIndex::new(Bm25fConfig::default());
    index.build(&corpus).unwrap();

    let hits = index.search("alpha", 0).unwrap();
    assert!(hits.is_empty());
}

#[test]
fn search_results_sorted_descending_by_score() {
    let corpus = vec![
        single_field_doc("d1", "body", "alpha alpha alpha alpha"),
        single_field_doc("d2", "body", "alpha alpha"),
        single_field_doc("d3", "body", "alpha"),
    ];
    let mut index = Bm25fIndex::new(Bm25fConfig::default());
    index.build(&corpus).unwrap();

    let hits = index.search("alpha", 10).unwrap();
    for window in hits.windows(2) {
        assert!(window[0].score >= window[1].score);
    }
}

#[test]
fn search_tie_break_by_doc_id_ascending() {
    let corpus = vec![
        single_field_doc("zebra", "body", "shared word"),
        single_field_doc("alpha", "body", "shared word"),
    ];
    let mut index = Bm25fIndex::new(Bm25fConfig::default());
    index.build(&corpus).unwrap();

    let hits = index.search("shared word", 10).unwrap();
    assert_eq!(hits.len(), 2);
    assert!(
        (hits[0].score - hits[1].score).abs() < 1e-6,
        "scores should tie: {hits:?}"
    );
    assert_eq!(hits[0].doc_id, "alpha");
    assert_eq!(hits[1].doc_id, "zebra");
}

#[test]
fn search_query_with_unknown_and_known_terms_only_known_contributes() {
    let corpus = vec![
        single_field_doc("d1", "body", "alpha beta"),
        single_field_doc("d2", "body", "gamma delta"),
    ];
    let mut index = Bm25fIndex::new(Bm25fConfig::default());
    index.build(&corpus).unwrap();

    let known_only = index.search("alpha", 10).unwrap();
    let mixed = index.search("alpha unknownxyz", 10).unwrap();
    assert_eq!(known_only, mixed);
}

// ── Bm25fIndex: accessor edge cases ──────────────────────────────────────────

#[test]
fn document_frequency_unknown_term_is_zero() {
    let mut index = Bm25fIndex::new(Bm25fConfig::default());
    index
        .build(&[single_field_doc("d1", "body", "alpha")])
        .unwrap();
    assert_eq!(index.document_frequency("nonexistent"), 0);
}

#[test]
fn idf_unknown_term_is_zero() {
    let mut index = Bm25fIndex::new(Bm25fConfig::default());
    index
        .build(&[single_field_doc("d1", "body", "alpha")])
        .unwrap();
    assert_eq!(index.idf("nonexistent"), 0.0);
}

#[test]
fn pseudo_frequency_unknown_doc_is_zero() {
    let mut index = Bm25fIndex::new(Bm25fConfig::default());
    index
        .build(&[single_field_doc("d1", "body", "alpha")])
        .unwrap();
    assert_eq!(index.pseudo_frequency("nonexistent_doc", "alpha"), 0.0);
}

#[test]
fn pseudo_frequency_unknown_term_is_zero() {
    let mut index = Bm25fIndex::new(Bm25fConfig::default());
    index
        .build(&[single_field_doc("d1", "body", "alpha")])
        .unwrap();
    assert_eq!(index.pseudo_frequency("d1", "nonexistent"), 0.0);
}

#[test]
fn average_field_length_unknown_field_is_zero() {
    let mut index = Bm25fIndex::new(Bm25fConfig::default());
    index
        .build(&[single_field_doc("d1", "body", "alpha")])
        .unwrap();
    assert_eq!(index.average_field_length("nonexistent_field"), 0.0);
}
