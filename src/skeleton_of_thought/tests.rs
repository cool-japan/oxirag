//! Unit tests for the `skeleton_of_thought` module.
#![allow(clippy::float_cmp, clippy::similar_names)]

use crate::skeleton_of_thought::engine::SkeletonOfThoughtEngine;
use crate::skeleton_of_thought::types::{
    MockPointExpander, MockSkeletonGenerator, PointExpander, SkeletonConfig, SkeletonGenerator,
    SkeletonPoint, SotError,
};

// ── helpers ───────────────────────────────────────────────────────────────────

fn engine_default() -> SkeletonOfThoughtEngine {
    SkeletonOfThoughtEngine::new(SkeletonConfig::default())
}

fn three_headers() -> Vec<String> {
    vec![
        "First point".to_string(),
        "Second point".to_string(),
        "Third point".to_string(),
    ]
}

/// Expander that records the `(query, point)` arguments it receives, to prove
/// the engine forwards the original query and the exact header.
#[derive(Debug, Default)]
struct RecordingExpander {
    calls: std::cell::RefCell<Vec<(String, String)>>,
}

impl PointExpander for RecordingExpander {
    fn expand(&self, query: &str, point: &str) -> String {
        self.calls
            .borrow_mut()
            .push((query.to_string(), point.to_string()));
        format!("{point} expanded")
    }
}

// ── SkeletonConfig ────────────────────────────────────────────────────────────

#[test]
fn config_default_max_points_is_ten() {
    assert_eq!(SkeletonConfig::default().max_points, 10);
}

#[test]
fn config_default_separator_is_newline() {
    assert_eq!(SkeletonConfig::default().join_separator, "\n");
}

#[test]
fn config_new_equals_default() {
    assert_eq!(SkeletonConfig::new(), SkeletonConfig::default());
}

#[test]
fn config_builder_sets_max_points() {
    assert_eq!(SkeletonConfig::new().with_max_points(3).max_points, 3);
}

#[test]
fn config_builder_zero_max_points() {
    assert_eq!(SkeletonConfig::new().with_max_points(0).max_points, 0);
}

#[test]
fn config_builder_sets_separator() {
    assert_eq!(
        SkeletonConfig::new()
            .with_join_separator(" | ")
            .join_separator,
        " | ",
    );
}

#[test]
fn config_builder_chains() {
    let cfg = SkeletonConfig::new()
        .with_max_points(4)
        .with_join_separator("; ");
    assert_eq!(cfg.max_points, 4);
    assert_eq!(cfg.join_separator, "; ");
}

#[test]
fn config_clone_equals_original() {
    let cfg = SkeletonConfig::new()
        .with_max_points(7)
        .with_join_separator("--");
    assert_eq!(cfg.clone(), cfg);
}

// ── SkeletonPoint ─────────────────────────────────────────────────────────────

#[test]
fn skeleton_point_new_sets_index() {
    assert_eq!(SkeletonPoint::new(2, "h", "c").index, 2);
}

#[test]
fn skeleton_point_new_sets_header() {
    assert_eq!(SkeletonPoint::new(0, "h", "c").header, "h");
}

#[test]
fn skeleton_point_new_sets_content() {
    assert_eq!(SkeletonPoint::new(0, "h", "c").content, "c");
}

#[test]
fn skeleton_point_equality() {
    assert_eq!(
        SkeletonPoint::new(0, "h", "c"),
        SkeletonPoint::new(0, "h", "c")
    );
}

// ── MockSkeletonGenerator ─────────────────────────────────────────────────────

#[test]
fn mock_generator_returns_points() {
    let gen_mock = MockSkeletonGenerator::new(three_headers());
    assert_eq!(gen_mock.skeleton("anything"), three_headers());
}

#[test]
fn mock_generator_ignores_query() {
    let gen_mock = MockSkeletonGenerator::new(three_headers());
    assert_eq!(gen_mock.skeleton("q1"), gen_mock.skeleton("q2"));
}

#[test]
fn mock_generator_default_is_empty() {
    assert!(MockSkeletonGenerator::default().skeleton("q").is_empty());
}

// ── MockPointExpander ─────────────────────────────────────────────────────────

#[test]
fn mock_expander_echo_formats_header() {
    let exp = MockPointExpander::echo();
    assert_eq!(exp.expand("q", "Topic"), "Topic: details");
}

#[test]
fn mock_expander_uses_mapped_expansion() {
    let exp = MockPointExpander::new(vec![("Topic".to_string(), "Full text".to_string())]);
    assert_eq!(exp.expand("q", "Topic"), "Full text");
}

#[test]
fn mock_expander_falls_back_to_echo_when_unmapped() {
    let exp = MockPointExpander::new(vec![("Other".to_string(), "x".to_string())]);
    assert_eq!(exp.expand("q", "Topic"), "Topic: details");
}

#[test]
fn mock_expander_first_match_wins() {
    let exp = MockPointExpander::new(vec![
        ("Topic".to_string(), "first".to_string()),
        ("Topic".to_string(), "second".to_string()),
    ]);
    assert_eq!(exp.expand("q", "Topic"), "first");
}

// ── run: skeleton production ───────────────────────────────────────────────────

#[test]
fn run_produces_one_point_per_header() {
    let gen_mock = MockSkeletonGenerator::new(three_headers());
    let out = engine_default()
        .run("query", &gen_mock, &MockPointExpander::echo())
        .unwrap();
    assert_eq!(out.points.len(), 3);
}

#[test]
fn run_preserves_header_text() {
    let gen_mock = MockSkeletonGenerator::new(three_headers());
    let out = engine_default()
        .run("query", &gen_mock, &MockPointExpander::echo())
        .unwrap();
    let headers: Vec<&str> = out.points.iter().map(|p| p.header.as_str()).collect();
    assert_eq!(headers, vec!["First point", "Second point", "Third point"]);
}

// ── run: expansion ─────────────────────────────────────────────────────────────

#[test]
fn run_each_content_is_non_empty() {
    let gen_mock = MockSkeletonGenerator::new(three_headers());
    let out = engine_default()
        .run("query", &gen_mock, &MockPointExpander::echo())
        .unwrap();
    assert!(out.points.iter().all(|p| !p.content.is_empty()));
}

#[test]
fn run_each_content_references_its_header() {
    let gen_mock = MockSkeletonGenerator::new(three_headers());
    let out = engine_default()
        .run("query", &gen_mock, &MockPointExpander::echo())
        .unwrap();
    assert!(
        out.points
            .iter()
            .all(|p| p.content.contains(p.header.as_str()))
    );
}

#[test]
fn run_uses_mapped_expansions() {
    let gen_mock = MockSkeletonGenerator::new(vec!["A".to_string(), "B".to_string()]);
    let exp = MockPointExpander::new(vec![
        ("A".to_string(), "alpha".to_string()),
        ("B".to_string(), "beta".to_string()),
    ]);
    let out = engine_default().run("query", &gen_mock, &exp).unwrap();
    assert_eq!(out.points[0].content, "alpha");
    assert_eq!(out.points[1].content, "beta");
}

// ── run: max_points cap ────────────────────────────────────────────────────────

#[test]
fn run_caps_points_at_max() {
    let headers = (0..8).map(|i| format!("point {i}")).collect();
    let gen_mock = MockSkeletonGenerator::new(headers);
    let engine = SkeletonOfThoughtEngine::new(SkeletonConfig::new().with_max_points(3));
    let out = engine
        .run("query", &gen_mock, &MockPointExpander::echo())
        .unwrap();
    assert_eq!(out.points.len(), 3);
}

#[test]
fn run_cap_keeps_leading_points_in_order() {
    let headers = (0..8).map(|i| format!("point {i}")).collect();
    let gen_mock = MockSkeletonGenerator::new(headers);
    let engine = SkeletonOfThoughtEngine::new(SkeletonConfig::new().with_max_points(3));
    let out = engine
        .run("query", &gen_mock, &MockPointExpander::echo())
        .unwrap();
    let headers: Vec<&str> = out.points.iter().map(|p| p.header.as_str()).collect();
    assert_eq!(headers, vec!["point 0", "point 1", "point 2"]);
}

#[test]
fn run_does_not_cap_below_count() {
    let gen_mock = MockSkeletonGenerator::new(three_headers());
    let engine = SkeletonOfThoughtEngine::new(SkeletonConfig::new().with_max_points(10));
    let out = engine
        .run("query", &gen_mock, &MockPointExpander::echo())
        .unwrap();
    assert_eq!(out.points.len(), 3);
}

// ── run: empty headers dropped ─────────────────────────────────────────────────

#[test]
fn run_drops_empty_headers() {
    let gen_mock = MockSkeletonGenerator::new(vec![
        "kept".to_string(),
        String::new(),
        "   ".to_string(),
        "also kept".to_string(),
    ]);
    let out = engine_default()
        .run("query", &gen_mock, &MockPointExpander::echo())
        .unwrap();
    let headers: Vec<&str> = out.points.iter().map(|p| p.header.as_str()).collect();
    assert_eq!(headers, vec!["kept", "also kept"]);
}

#[test]
fn run_drops_whitespace_only_headers() {
    let gen_mock = MockSkeletonGenerator::new(vec!["\t\n".to_string(), "real".to_string()]);
    let out = engine_default()
        .run("query", &gen_mock, &MockPointExpander::echo())
        .unwrap();
    assert_eq!(out.points.len(), 1);
    assert_eq!(out.points[0].header, "real");
}

#[test]
fn run_cap_applies_after_dropping_empties() {
    // Empties are dropped first, so the cap counts only non-empty headers.
    let gen_mock = MockSkeletonGenerator::new(vec![
        String::new(),
        "a".to_string(),
        String::new(),
        "b".to_string(),
        "c".to_string(),
    ]);
    let engine = SkeletonOfThoughtEngine::new(SkeletonConfig::new().with_max_points(2));
    let out = engine
        .run("query", &gen_mock, &MockPointExpander::echo())
        .unwrap();
    let headers: Vec<&str> = out.points.iter().map(|p| p.header.as_str()).collect();
    assert_eq!(headers, vec!["a", "b"]);
}

// ── run: index ordering ────────────────────────────────────────────────────────

#[test]
fn run_indices_are_zero_based_and_in_order() {
    let headers = (0..5).map(|i| format!("h{i}")).collect();
    let gen_mock = MockSkeletonGenerator::new(headers);
    let out = engine_default()
        .run("query", &gen_mock, &MockPointExpander::echo())
        .unwrap();
    let indices: Vec<usize> = out.points.iter().map(|p| p.index).collect();
    assert_eq!(indices, vec![0, 1, 2, 3, 4]);
}

#[test]
fn run_indices_match_position_after_dropping_empties() {
    let gen_mock = MockSkeletonGenerator::new(vec![
        String::new(),
        "a".to_string(),
        String::new(),
        "b".to_string(),
    ]);
    let out = engine_default()
        .run("query", &gen_mock, &MockPointExpander::echo())
        .unwrap();
    assert_eq!(out.points[0].index, 0);
    assert_eq!(out.points[1].index, 1);
}

// ── run: answer joining ────────────────────────────────────────────────────────

#[test]
fn run_answer_joins_with_default_separator() {
    let gen_mock = MockSkeletonGenerator::new(vec!["A".to_string(), "B".to_string()]);
    let exp = MockPointExpander::new(vec![
        ("A".to_string(), "alpha".to_string()),
        ("B".to_string(), "beta".to_string()),
    ]);
    let out = engine_default().run("query", &gen_mock, &exp).unwrap();
    assert_eq!(out.answer, "alpha\nbeta");
}

#[test]
fn run_answer_uses_custom_separator() {
    let gen_mock = MockSkeletonGenerator::new(vec!["A".to_string(), "B".to_string()]);
    let exp = MockPointExpander::new(vec![
        ("A".to_string(), "alpha".to_string()),
        ("B".to_string(), "beta".to_string()),
    ]);
    let engine = SkeletonOfThoughtEngine::new(SkeletonConfig::new().with_join_separator(" | "));
    let out = engine.run("query", &gen_mock, &exp).unwrap();
    assert_eq!(out.answer, "alpha | beta");
}

#[test]
fn run_answer_preserves_skeleton_order() {
    let gen_mock = MockSkeletonGenerator::new(three_headers());
    let exp = MockPointExpander::new(vec![
        ("First point".to_string(), "1".to_string()),
        ("Second point".to_string(), "2".to_string()),
        ("Third point".to_string(), "3".to_string()),
    ]);
    let engine = SkeletonOfThoughtEngine::new(SkeletonConfig::new().with_join_separator(","));
    let out = engine.run("query", &gen_mock, &exp).unwrap();
    assert_eq!(out.answer, "1,2,3");
}

#[test]
fn run_answer_equals_join_of_point_contents() {
    let gen_mock = MockSkeletonGenerator::new(three_headers());
    let out = engine_default()
        .run("query", &gen_mock, &MockPointExpander::echo())
        .unwrap();
    let expected = out
        .points
        .iter()
        .map(|p| p.content.clone())
        .collect::<Vec<_>>()
        .join("\n");
    assert_eq!(out.answer, expected);
}

#[test]
fn run_single_point_answer_has_no_separator() {
    let gen_mock = MockSkeletonGenerator::new(vec!["only".to_string()]);
    let exp = MockPointExpander::new(vec![("only".to_string(), "solo".to_string())]);
    let engine = SkeletonOfThoughtEngine::new(SkeletonConfig::new().with_join_separator(" | "));
    let out = engine.run("query", &gen_mock, &exp).unwrap();
    assert_eq!(out.answer, "solo");
}

// ── run: errors ────────────────────────────────────────────────────────────────

#[test]
fn run_empty_query_errors() {
    let gen_mock = MockSkeletonGenerator::new(three_headers());
    let err = engine_default()
        .run("", &gen_mock, &MockPointExpander::echo())
        .unwrap_err();
    assert!(matches!(err, SotError::EmptyQuery));
}

#[test]
fn run_whitespace_query_errors() {
    let gen_mock = MockSkeletonGenerator::new(three_headers());
    let err = engine_default()
        .run("   \t\n", &gen_mock, &MockPointExpander::echo())
        .unwrap_err();
    assert!(matches!(err, SotError::EmptyQuery));
}

#[test]
fn run_empty_skeleton_errors() {
    let gen_mock = MockSkeletonGenerator::default();
    let err = engine_default()
        .run("query", &gen_mock, &MockPointExpander::echo())
        .unwrap_err();
    assert!(matches!(err, SotError::EmptySkeleton));
}

#[test]
fn run_all_empty_headers_errors_with_empty_skeleton() {
    let gen_mock = MockSkeletonGenerator::new(vec![String::new(), "  ".to_string()]);
    let err = engine_default()
        .run("query", &gen_mock, &MockPointExpander::echo())
        .unwrap_err();
    assert!(matches!(err, SotError::EmptySkeleton));
}

#[test]
fn error_messages_render() {
    assert_eq!(SotError::EmptyQuery.to_string(), "query must not be empty");
    assert_eq!(SotError::EmptySkeleton.to_string(), "skeleton is empty");
}

// ── run: determinism ───────────────────────────────────────────────────────────

#[test]
fn run_is_deterministic_with_mocks() {
    let gen_mock = MockSkeletonGenerator::new(three_headers());
    let exp = MockPointExpander::new(vec![
        ("First point".to_string(), "1".to_string()),
        ("Second point".to_string(), "2".to_string()),
        ("Third point".to_string(), "3".to_string()),
    ]);
    let engine = engine_default();
    let a = engine.run("query", &gen_mock, &exp).unwrap();
    let b = engine.run("query", &gen_mock, &exp).unwrap();
    assert_eq!(a, b);
}

// ── run: expander argument forwarding ──────────────────────────────────────────

#[test]
fn expander_receives_original_query() {
    let gen_mock = MockSkeletonGenerator::new(three_headers());
    let recorder = RecordingExpander::default();
    engine_default()
        .run("the original query", &gen_mock, &recorder)
        .unwrap();
    assert!(
        recorder
            .calls
            .borrow()
            .iter()
            .all(|(query, _)| query == "the original query")
    );
}

#[test]
fn expander_receives_each_header_in_order() {
    let gen_mock = MockSkeletonGenerator::new(three_headers());
    let recorder = RecordingExpander::default();
    engine_default().run("query", &gen_mock, &recorder).unwrap();
    let points: Vec<String> = recorder
        .calls
        .borrow()
        .iter()
        .map(|(_, point)| point.clone())
        .collect();
    assert_eq!(points, vec!["First point", "Second point", "Third point"]);
}

#[test]
fn expander_called_once_per_kept_point() {
    let gen_mock =
        MockSkeletonGenerator::new(vec!["a".to_string(), String::new(), "b".to_string()]);
    let recorder = RecordingExpander::default();
    engine_default().run("query", &gen_mock, &recorder).unwrap();
    assert_eq!(recorder.calls.borrow().len(), 2);
}

// ── engine construction ────────────────────────────────────────────────────────

#[test]
fn engine_new_stores_config() {
    let cfg = SkeletonConfig::new().with_max_points(2);
    let engine = SkeletonOfThoughtEngine::new(cfg.clone());
    assert_eq!(engine.config, cfg);
}

#[test]
fn engine_default_uses_default_config() {
    assert_eq!(
        SkeletonOfThoughtEngine::default().config,
        SkeletonConfig::default()
    );
}
