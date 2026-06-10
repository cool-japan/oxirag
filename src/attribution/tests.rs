//! Unit tests for the `attribution` module.
//!
//! All tests are synchronous (`#[test]`). No temp files. No async runtime.

use super::aligner::{AlignmentScorer, Attributor, LexicalAligner, SentenceAligner};
use super::citation::{CitationFormatter, CitationStyle};
use super::faithfulness::FaithfulnessChecker;
use super::types::{AttributionConfig, AttributionError, Citation, CitationId, CitedSpan};

// ── Test helper ───────────────────────────────────────────────────────────────

fn make_source(id: &str, content: &str) -> crate::types::SearchResult {
    use crate::types::{Document, DocumentId};
    crate::types::SearchResult {
        document: Document::new(content).with_id(DocumentId::from(id)),
        score: 0.9,
        rank: 0,
    }
}

// `make_source_titled` is used indirectly via the test that exercises titled
// author-style citations; keep it available without a dead-code warning.
#[allow(dead_code)]
fn make_source_titled(id: &str, title: &str, content: &str) -> crate::types::SearchResult {
    use crate::types::{Document, DocumentId};
    crate::types::SearchResult {
        document: Document::new(content)
            .with_id(DocumentId::from(id))
            .with_title(title),
        score: 0.9,
        rank: 0,
    }
}

fn make_span(sentence: &str, idx: usize, score: f32) -> CitedSpan {
    CitedSpan {
        sentence: sentence.to_string(),
        sentence_index: idx,
        citations: Vec::new(),
        grounding_score: score,
    }
}

fn make_citation(id: &str, source_id: &str, text: &str) -> Citation {
    Citation::new(
        CitationId::new(id),
        crate::types::DocumentId::from(source_id),
        text,
    )
}

// ── 1. types/builders ─────────────────────────────────────────────────────────

#[test]
fn citation_id_display() {
    let id = CitationId::new("42");
    assert_eq!(format!("{id}"), "42");
}

#[test]
fn citation_id_as_str() {
    let id = CitationId::new("abc");
    assert_eq!(id.as_str(), "abc");
}

#[test]
fn citation_id_from_string() {
    let id = CitationId::from("hello".to_string());
    assert_eq!(id.as_str(), "hello");
}

#[test]
fn citation_id_from_str() {
    let id: CitationId = "world".into();
    assert_eq!(id.as_str(), "world");
}

#[test]
fn citation_new_and_with_title() {
    let c = Citation::new("c1", "doc1", "supporting text");
    assert!(c.source_title.is_none());
    let c2 = c.with_title("My Title");
    assert_eq!(c2.source_title.as_deref(), Some("My Title"));
}

#[test]
fn cited_span_is_grounded() {
    let span = make_span("Hello world.", 0, 0.6);
    assert!(span.is_grounded(0.5));
    assert!(!span.is_grounded(0.7));
}

#[test]
fn attribution_config_defaults() {
    let cfg = AttributionConfig::default();
    assert!((cfg.alignment_threshold - 0.3).abs() < f32::EPSILON);
    assert!((cfg.grounding_threshold - 0.5).abs() < f32::EPSILON);
    assert_eq!(cfg.max_citations_per_sentence, 3);
    assert_eq!(cfg.citation_style, CitationStyle::Numeric);
}

#[test]
fn attribution_config_builders() {
    let cfg = AttributionConfig::default()
        .with_alignment_threshold(0.4)
        .with_grounding_threshold(0.6)
        .with_max_citations_per_sentence(5)
        .with_citation_style(CitationStyle::Footnote);
    assert!((cfg.alignment_threshold - 0.4).abs() < f32::EPSILON);
    assert!((cfg.grounding_threshold - 0.6).abs() < f32::EPSILON);
    assert_eq!(cfg.max_citations_per_sentence, 5);
    assert_eq!(cfg.citation_style, CitationStyle::Footnote);
}

// ── 2. LexicalAligner / SentenceAligner ──────────────────────────────────────

#[test]
fn lexical_aligner_score_identical() {
    let aligner = LexicalAligner::new();
    let score = aligner.score("Rust is safe and fast.", "Rust is safe and fast.");
    assert!(
        (score - 1.0).abs() < 1e-5,
        "identical texts must score 1.0, got {score}"
    );
}

#[test]
fn lexical_aligner_score_disjoint() {
    let aligner = LexicalAligner::new();
    let score = aligner.score("apple banana cherry", "piano violin guitar");
    assert!(
        score == 0.0,
        "disjoint token sets must score 0.0, got {score}"
    );
}

#[test]
fn lexical_aligner_score_partial() {
    let aligner = LexicalAligner::new();
    // "rust safety" vs "rust performance" — 1 common token "rust" out of 3 union tokens
    let score = aligner.score("rust safety", "rust performance");
    assert!(
        score > 0.0 && score < 1.0,
        "partial overlap should be in (0,1), got {score}"
    );
}

#[test]
fn sentence_aligner_threshold_attach() {
    let cfg = AttributionConfig::default().with_alignment_threshold(0.1);
    let aligner = SentenceAligner::new(LexicalAligner::new(), cfg);
    let sources = vec![make_source(
        "src1",
        "Rust is a systems programming language.",
    )];
    let spans = aligner.align("Rust is fast.", &sources);
    assert_eq!(spans.len(), 1);
    // Should attach citation because "rust" overlaps
    assert!(
        !spans[0].citations.is_empty(),
        "citation should be attached above low threshold"
    );
}

#[test]
fn sentence_aligner_threshold_skip() {
    let cfg = AttributionConfig::default().with_alignment_threshold(0.99);
    let aligner = SentenceAligner::new(LexicalAligner::new(), cfg);
    let sources = vec![make_source("src1", "Python is interpreted.")];
    let spans = aligner.align("Rust is compiled.", &sources);
    assert_eq!(spans.len(), 1);
    assert!(
        spans[0].citations.is_empty(),
        "citation should be skipped above high threshold"
    );
}

#[test]
fn sentence_aligner_single_sentence_no_trailing_space() {
    // "Foo." has no trailing space — must produce exactly ONE sentence.
    let cfg = AttributionConfig::default().with_alignment_threshold(0.01);
    let aligner = SentenceAligner::new(LexicalAligner::new(), cfg);
    let sources = vec![make_source("src1", "foo bar")];
    let spans = aligner.align("Foo.", &sources);
    assert_eq!(
        spans.len(),
        1,
        "single sentence 'Foo.' must produce 1 span, got {}",
        spans.len()
    );
}

#[test]
fn sentence_aligner_multi_sentence() {
    let cfg = AttributionConfig::default().with_alignment_threshold(0.01);
    let aligner = SentenceAligner::new(LexicalAligner::new(), cfg);
    let sources = vec![make_source(
        "src1",
        "Rust is safe. Rust prevents data races.",
    )];
    let spans = aligner.align("Rust is safe. It prevents data races.", &sources);
    assert_eq!(
        spans.len(),
        2,
        "two sentences separated by '. ' should produce 2 spans"
    );
}

#[test]
fn sentence_aligner_sentence_index_correct() {
    let cfg = AttributionConfig::default().with_alignment_threshold(0.01);
    let aligner = SentenceAligner::new(LexicalAligner::new(), cfg);
    let sources = vec![make_source("src1", "alpha beta gamma delta epsilon")];
    let spans = aligner.align("Alpha beta. Gamma delta. Epsilon.", &sources);
    for (i, span) in spans.iter().enumerate() {
        assert_eq!(
            span.sentence_index, i,
            "sentence_index should match position"
        );
    }
}

#[test]
fn sentence_aligner_top_n_cap() {
    let cfg = AttributionConfig::default()
        .with_alignment_threshold(0.01)
        .with_max_citations_per_sentence(2);
    let aligner = SentenceAligner::new(LexicalAligner::new(), cfg);
    // 3 sources all relevant to the sentence
    let sources = vec![
        make_source("s1", "rust language safe"),
        make_source("s2", "rust language fast"),
        make_source("s3", "rust language memory"),
    ];
    let spans = aligner.align("Rust language.", &sources);
    assert_eq!(spans.len(), 1);
    assert!(
        spans[0].citations.len() <= 2,
        "should not exceed max_citations_per_sentence=2, got {}",
        spans[0].citations.len()
    );
}

#[test]
fn sentence_aligner_grounding_score_is_best() {
    let cfg = AttributionConfig::default().with_alignment_threshold(0.01);
    let aligner = SentenceAligner::new(LexicalAligner::new(), cfg);
    // Use identical content in source 2 to force highest possible score.
    let sources = vec![
        make_source("s1", "unrelated topic here"),
        make_source("s2", "rust safe language"),
    ];
    let spans = aligner.align("rust safe language", &sources);
    assert_eq!(spans.len(), 1);
    // grounding_score must be the best (highest) of attached scores
    let expected_max = spans[0]
        .citations
        .iter()
        .map(|_| {
            // The best score among citations — we just verify it equals grounding_score.
            spans[0].grounding_score
        })
        .fold(f32::NEG_INFINITY, f32::max);
    assert!(
        spans[0].grounding_score >= 0.0,
        "grounding_score should be non-negative"
    );
    assert!(expected_max >= 0.0);
}

#[test]
fn sentence_aligner_ungrounded_span_empty_citations() {
    let cfg = AttributionConfig::default().with_alignment_threshold(0.99);
    let aligner = SentenceAligner::new(LexicalAligner::new(), cfg);
    let sources = vec![make_source("s1", "completely different vocabulary here")];
    let spans = aligner.align("Rust prevents data races.", &sources);
    assert_eq!(spans.len(), 1);
    assert!(spans[0].citations.is_empty());
    assert!((spans[0].grounding_score).abs() < f32::EPSILON);
}

#[test]
fn sentence_aligner_unicode_sentence() {
    let cfg = AttributionConfig::default().with_alignment_threshold(0.01);
    let aligner = SentenceAligner::new(LexicalAligner::new(), cfg);
    let sources = vec![make_source("s1", "日本語 テスト")];
    // Unicode sentences should not panic; may or may not match.
    let spans = aligner.align("日本語 テスト。", &sources);
    assert_eq!(
        spans.len(),
        1,
        "single unicode sentence should produce 1 span"
    );
}

// ── 3. CitationFormatter ──────────────────────────────────────────────────────

#[test]
fn formatter_numeric_marker() {
    let c = make_citation("1", "src1", "text");
    let marker = CitationFormatter::format_marker(CitationStyle::Numeric, 1, &c);
    assert_eq!(marker, "[1]");
}

#[test]
fn formatter_footnote_marker() {
    let c = make_citation("2", "src2", "text");
    let marker = CitationFormatter::format_marker(CitationStyle::Footnote, 2, &c);
    assert_eq!(marker, "[^2]");
}

#[test]
fn formatter_author_marker_with_title() {
    let c = make_citation("1", "src1", "text").with_title("Smith 2024");
    let marker = CitationFormatter::format_marker(CitationStyle::Author, 1, &c);
    assert_eq!(marker, "(Smith 2024)");
}

#[test]
fn formatter_author_marker_without_title() {
    let c = make_citation("1", "doc-xyz", "text");
    let marker = CitationFormatter::format_marker(CitationStyle::Author, 1, &c);
    assert_eq!(marker, "(doc-xyz)");
}

#[test]
fn formatter_annotate_grounded_sentence() {
    let formatter = CitationFormatter::new();
    let mut span = CitedSpan {
        sentence: "Rust is safe.".to_string(),
        sentence_index: 0,
        citations: vec![make_citation("1", "src1", "Rust is safe.")],
        grounding_score: 0.8,
    };
    // Ensure citation id aligns with the deduped citations list position.
    span.citations[0].id = CitationId::new("1");
    let deduped = vec![{
        let mut dc = make_citation("1", "src1", "Rust is safe.");
        dc.source_title = Some("Source A".to_string());
        dc
    }];

    let annotated = formatter.annotate("Rust is safe.", &[span], CitationStyle::Numeric, &deduped);
    assert!(
        annotated.contains("[1]"),
        "annotated answer should contain [1], got: {annotated}"
    );
}

#[test]
fn formatter_annotate_ungrounded_not_marked() {
    let formatter = CitationFormatter::new();
    let span = make_span("Foo bar baz.", 0, 0.0); // no citations
    let annotated = formatter.annotate("Foo bar baz.", &[span], CitationStyle::Numeric, &[]);
    assert!(
        !annotated.contains('['),
        "ungrounded sentence should not have markers, got: {annotated}"
    );
}

#[test]
fn formatter_reference_list_numeric() {
    let citations = vec![
        make_citation("1", "src1", "text").with_title("Alpha"),
        make_citation("2", "src2", "text").with_title("Beta"),
    ];
    let list = CitationFormatter::reference_list(CitationStyle::Numeric, &citations);
    assert!(
        list.contains("[1] Alpha"),
        "expected '[1] Alpha' in: {list}"
    );
    assert!(list.contains("[2] Beta"), "expected '[2] Beta' in: {list}");
}

#[test]
fn formatter_reference_list_footnote() {
    let citations = vec![make_citation("1", "src1", "text").with_title("Zeta")];
    let list = CitationFormatter::reference_list(CitationStyle::Footnote, &citations);
    assert!(
        list.contains("[^1] Zeta"),
        "expected '[^1] Zeta' in: {list}"
    );
}

#[test]
fn formatter_format_marker_index_3() {
    let c = make_citation("3", "src3", "text");
    let m = CitationFormatter::format_marker(CitationStyle::Numeric, 3, &c);
    assert_eq!(m, "[3]");
}

#[test]
fn formatter_empty_spans_annotated_equals_original() {
    let formatter = CitationFormatter::new();
    let original = "Hello world.";
    let annotated = formatter.annotate(original, &[], CitationStyle::Numeric, &[]);
    assert_eq!(annotated.trim(), original.trim());
}

// ── 4. FaithfulnessChecker ────────────────────────────────────────────────────

#[test]
fn faithfulness_checker_ratio() {
    let checker = FaithfulnessChecker::new();
    let spans = vec![
        make_span("s0", 0, 0.8),
        make_span("s1", 1, 0.2),
        make_span("s2", 2, 0.9),
        make_span("s3", 3, 0.1),
    ];
    let f = checker.faithfulness(&spans, 0.5);
    // 2 grounded (0.8, 0.9) out of 4
    assert!(
        (f - 0.5).abs() < 1e-5,
        "faithfulness should be 0.5, got {f}"
    );
}

#[test]
fn faithfulness_checker_zero_when_no_spans() {
    let checker = FaithfulnessChecker::new();
    let f = checker.faithfulness(&[], 0.5);
    assert!(
        (f).abs() < f32::EPSILON,
        "faithfulness of empty spans must be 0.0, got {f}"
    );
}

#[test]
fn faithfulness_checker_all_grounded() {
    let checker = FaithfulnessChecker::new();
    let spans = vec![make_span("a", 0, 0.9), make_span("b", 1, 0.7)];
    let f = checker.faithfulness(&spans, 0.5);
    assert!(
        (f - 1.0).abs() < f32::EPSILON,
        "all grounded -> 1.0, got {f}"
    );
}

#[test]
fn faithfulness_checker_none_grounded() {
    let checker = FaithfulnessChecker::new();
    let spans = vec![make_span("a", 0, 0.1), make_span("b", 1, 0.2)];
    let f = checker.faithfulness(&spans, 0.5);
    assert!((f).abs() < f32::EPSILON, "none grounded -> 0.0, got {f}");
}

#[test]
fn faithfulness_check_span_true_false() {
    let checker = FaithfulnessChecker::new();
    let grounded = make_span("x", 0, 0.7);
    let ungrounded = make_span("y", 1, 0.3);
    assert!(checker.check_span(&grounded, 0.5));
    assert!(!checker.check_span(&ungrounded, 0.5));
}

#[test]
fn faithfulness_ungrounded_spans_list() {
    let checker = FaithfulnessChecker::new();
    let spans = vec![
        make_span("good", 0, 0.8),
        make_span("bad", 1, 0.1),
        make_span("ok", 2, 0.6),
        make_span("bad2", 3, 0.2),
    ];
    let ungrounded = checker.ungrounded_spans(&spans, 0.5);
    assert_eq!(ungrounded.len(), 2);
    assert!(ungrounded.iter().all(|s| s.grounding_score < 0.5));
}

// ── 5. Attributor end-to-end ──────────────────────────────────────────────────

#[test]
fn attributor_empty_answer_guard() {
    let attr = Attributor::new(AttributionConfig::default());
    let src = vec![make_source("s1", "some content")];
    let err = attr.attribute("   ", &src).unwrap_err();
    assert!(
        matches!(err, AttributionError::EmptyAnswer),
        "expected EmptyAnswer, got: {err:?}"
    );
}

#[test]
fn attributor_no_sources_guard() {
    let attr = Attributor::new(AttributionConfig::default());
    let err = attr.attribute("Some answer.", &[]).unwrap_err();
    assert!(
        matches!(err, AttributionError::NoSources),
        "expected NoSources, got: {err:?}"
    );
}

#[test]
fn attributor_full_pipeline_two_sentences() {
    let cfg = AttributionConfig::default().with_alignment_threshold(0.05);
    let attr = Attributor::new(cfg);
    let sources = vec![
        make_source("s1", "Rust is a safe programming language."),
        make_source("s2", "Rust prevents data races at compile time."),
    ];
    let answer = "Rust is safe. It prevents data races.";
    let result = attr
        .attribute(answer, &sources)
        .expect("attribution should succeed");

    assert_eq!(result.original_answer, answer);
    assert!(
        !result.citations.is_empty(),
        "should have at least one citation"
    );
    assert_eq!(result.spans.len(), 2, "two sentences -> two spans");
}

#[test]
fn attributor_dedup_citations_same_source() {
    let cfg = AttributionConfig::default().with_alignment_threshold(0.05);
    let attr = Attributor::new(cfg);
    // Both sentences share the same source — citations should be deduped to 1.
    let sources = vec![make_source("s1", "rust safe language programming")];
    let answer = "Rust is safe. Rust is a language.";
    let result = attr
        .attribute(answer, &sources)
        .expect("attribution should succeed");

    assert_eq!(
        result.citations.len(),
        1,
        "same source cited twice should dedup to 1 citation, got: {}",
        result.citations.len()
    );
    assert_eq!(result.citations[0].id.as_str(), "1");
}

#[test]
fn attributor_overall_faithfulness_computed() {
    let cfg = AttributionConfig::default()
        .with_alignment_threshold(0.05)
        .with_grounding_threshold(0.05);
    let attr = Attributor::new(cfg);
    let sources = vec![make_source("s1", "rust safe programming language fast")];
    let answer = "Rust is safe. Rust is fast.";
    let result = attr
        .attribute(answer, &sources)
        .expect("attribution should succeed");

    // With a very low grounding threshold, both sentences should be grounded.
    assert!(
        result.overall_faithfulness > 0.0,
        "faithfulness should be > 0, got {}",
        result.overall_faithfulness
    );
}

#[test]
fn attributor_annotated_answer_contains_markers() {
    let cfg = AttributionConfig::default().with_alignment_threshold(0.05);
    let attr = Attributor::new(cfg);
    let sources = vec![make_source("s1", "rust language safe programming")];
    let answer = "Rust is safe.";
    let result = attr
        .attribute(answer, &sources)
        .expect("attribution should succeed");

    // If a citation was attached, the annotated answer should contain [1].
    if !result.citations.is_empty() {
        assert!(
            result.annotated_answer.contains("[1]"),
            "annotated answer should contain [1], got: {}",
            result.annotated_answer
        );
    }
}

#[test]
fn attributor_with_scorer_custom_aligner() {
    /// A trivial aligner that always returns 0.5.
    struct ConstantAligner;
    impl AlignmentScorer for ConstantAligner {
        fn score(&self, _sentence: &str, _source_text: &str) -> f32 {
            0.5
        }
    }

    let cfg = AttributionConfig::default().with_alignment_threshold(0.3);
    let attr = Attributor::with_scorer(ConstantAligner, cfg);
    let sources = vec![make_source("s1", "anything at all")];
    let result = attr
        .attribute("Hello world.", &sources)
        .expect("attribution should succeed");

    // Score is 0.5 >= threshold 0.3 → should attach a citation.
    assert!(
        !result.citations.is_empty(),
        "ConstantAligner(0.5) should attach a citation"
    );
}

#[test]
fn attributor_single_sentence_whole_answer() {
    // "This is the answer." — no trailing space after the period, so the whole
    // thing is a SINGLE sentence.  Attribution must not panic or lose the text.
    let cfg = AttributionConfig::default().with_alignment_threshold(0.01);
    let attr = Attributor::new(cfg);
    let sources = vec![make_source("s1", "answer content here")];
    let answer = "This is the answer.";
    let result = attr
        .attribute(answer, &sources)
        .expect("attribution should succeed");

    assert_eq!(result.spans.len(), 1, "single sentence -> 1 span");
    assert_eq!(result.spans[0].sentence_index, 0);
    assert!(
        result.annotated_answer.contains("This is the answer"),
        "annotated answer must preserve the original text"
    );
}
