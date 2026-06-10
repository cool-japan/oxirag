//! Tests for the `lost_in_middle` module.

use crate::lost_in_middle::types::{
    LostInMiddleError, LostInMiddleReorderer, ReorderConfig, ReorderReport, ReorderStrategy,
};
use crate::types::{Document, DocumentId, SearchResult};

// ── helpers ──────────────────────────────────────────────────────────────────

fn make_result(id: &str, score: f32) -> SearchResult {
    SearchResult {
        document: Document::new(format!("content for {id}")).with_id(DocumentId::from_string(id)),
        score,
        rank: 0,
    }
}

// ── ReorderStrategy::as_str ───────────────────────────────────────────────

#[test]
fn test_reorder_strategy_as_str_sandwich() {
    assert_eq!(
        ReorderStrategy::Sandwich.as_str(),
        "sandwich",
        "Sandwich.as_str() should return 'sandwich'"
    );
}

#[test]
fn test_reorder_strategy_as_str_head_tail() {
    assert_eq!(
        ReorderStrategy::HeadTail.as_str(),
        "head_tail",
        "HeadTail.as_str() should return 'head_tail'"
    );
}

#[test]
fn test_reorder_strategy_as_str_descending() {
    assert_eq!(
        ReorderStrategy::Descending.as_str(),
        "descending",
        "Descending.as_str() should return 'descending'"
    );
}

#[test]
fn test_reorder_strategy_as_str_custom() {
    assert_eq!(
        ReorderStrategy::Custom.as_str(),
        "custom",
        "Custom.as_str() should return 'custom'"
    );
}

#[test]
fn test_reorder_strategy_default_is_sandwich() {
    assert_eq!(
        ReorderStrategy::default(),
        ReorderStrategy::Sandwich,
        "default ReorderStrategy should be Sandwich"
    );
}

// ── ReorderConfig ─────────────────────────────────────────────────────────

#[test]
fn test_reorder_config_default_strategy_is_sandwich() {
    let cfg = ReorderConfig::default();
    assert_eq!(
        cfg.strategy,
        ReorderStrategy::Sandwich,
        "default strategy should be Sandwich"
    );
}

#[test]
fn test_reorder_config_default_preserve_top_k_zero() {
    let cfg = ReorderConfig::default();
    assert_eq!(cfg.preserve_top_k, 0, "default preserve_top_k should be 0");
}

#[test]
fn test_reorder_config_with_strategy_builder() {
    let cfg = ReorderConfig::default().with_strategy(ReorderStrategy::Descending);
    assert_eq!(
        cfg.strategy,
        ReorderStrategy::Descending,
        "with_strategy should set Descending"
    );
}

#[test]
fn test_reorder_config_with_preserve_top_k_builder() {
    let cfg = ReorderConfig::default().with_preserve_top_k(3);
    assert_eq!(
        cfg.preserve_top_k, 3,
        "with_preserve_top_k should set value to 3"
    );
}

#[test]
fn test_reorder_config_builder_chaining() {
    let cfg = ReorderConfig::default()
        .with_strategy(ReorderStrategy::HeadTail)
        .with_preserve_top_k(5);
    assert_eq!(
        cfg.strategy,
        ReorderStrategy::HeadTail,
        "chained strategy should be HeadTail"
    );
    assert_eq!(cfg.preserve_top_k, 5, "chained preserve_top_k should be 5");
}

// ── ReorderReport ─────────────────────────────────────────────────────────

#[test]
fn test_reorder_report_field_access() {
    let report = ReorderReport {
        original_order: vec!["a".to_string(), "b".to_string()],
        new_order: vec!["b".to_string(), "a".to_string()],
        strategy: ReorderStrategy::Sandwich,
    };
    assert_eq!(
        report.original_order.len(),
        2,
        "original_order should have 2 entries"
    );
    assert_eq!(report.new_order[0], "b", "new_order[0] should be 'b'");
    assert_eq!(
        report.strategy,
        ReorderStrategy::Sandwich,
        "strategy should be Sandwich"
    );
}

// ── LostInMiddleReorderer ──────────────────────────────────────────────────

#[test]
fn test_reorderer_new_stores_config() {
    let cfg = ReorderConfig::default().with_strategy(ReorderStrategy::Descending);
    let r = LostInMiddleReorderer::new(cfg);
    assert_eq!(
        r.config.strategy,
        ReorderStrategy::Descending,
        "new should store provided config"
    );
}

#[test]
fn test_reorderer_default_config_is_sandwich() {
    let r = LostInMiddleReorderer::default();
    assert_eq!(
        r.config.strategy,
        ReorderStrategy::Sandwich,
        "default reorderer should use Sandwich strategy"
    );
}

#[test]
fn test_reorder_empty_results_returns_error() {
    let r = LostInMiddleReorderer::default();
    let err = r.reorder(&[]).unwrap_err();
    assert!(
        matches!(err, LostInMiddleError::EmptyResults),
        "empty results should return EmptyResults error"
    );
}

// ── Sandwich strategy ─────────────────────────────────────────────────────

#[test]
fn test_sandwich_first_result_is_highest_scored() {
    let cfg = ReorderConfig::default().with_strategy(ReorderStrategy::Sandwich);
    let r = LostInMiddleReorderer::new(cfg);
    let results = vec![
        make_result("low", 0.1),
        make_result("high", 1.0),
        make_result("mid", 0.5),
    ];
    let reordered = r.reorder(&results).unwrap();
    assert_eq!(
        reordered[0].document.id.as_str(),
        "high",
        "Sandwich: first result should be highest-scored (1.0)"
    );
}

#[test]
fn test_sandwich_last_result_is_second_highest_scored() {
    let cfg = ReorderConfig::default().with_strategy(ReorderStrategy::Sandwich);
    let r = LostInMiddleReorderer::new(cfg);
    let results = vec![
        make_result("low", 0.1),
        make_result("high", 1.0),
        make_result("second", 0.9),
    ];
    let reordered = r.reorder(&results).unwrap();
    assert_eq!(
        reordered.last().unwrap().document.id.as_str(),
        "second",
        "Sandwich: last result should be 2nd-highest (0.9)"
    );
}

#[test]
fn test_sandwich_middle_has_lowest_score() {
    let cfg = ReorderConfig::default().with_strategy(ReorderStrategy::Sandwich);
    let r = LostInMiddleReorderer::new(cfg);
    let results = vec![
        make_result("highest", 1.0),
        make_result("second", 0.9),
        make_result("lowest", 0.1),
    ];
    let reordered = r.reorder(&results).unwrap();
    // With 3 elements: index 0 = rank 1 (highest), index 2 = rank 2 (second), index 1 = rank 3 (lowest)
    let middle_score = reordered[1].score;
    let first_score = reordered[0].score;
    let last_score = reordered[2].score;
    assert!(
        middle_score <= first_score && middle_score <= last_score,
        "Sandwich: middle result should have the lowest score; first={first_score}, middle={middle_score}, last={last_score}"
    );
}

// ── HeadTail strategy ─────────────────────────────────────────────────────

#[test]
fn test_head_tail_first_result_is_highest_scored() {
    let cfg = ReorderConfig::default().with_strategy(ReorderStrategy::HeadTail);
    let r = LostInMiddleReorderer::new(cfg);
    let results = vec![
        make_result("low", 0.1),
        make_result("high", 1.0),
        make_result("mid", 0.5),
    ];
    let reordered = r.reorder(&results).unwrap();
    assert_eq!(
        reordered[0].document.id.as_str(),
        "high",
        "HeadTail: first result should be highest-scored"
    );
}

#[test]
fn test_head_tail_same_alternating_behavior_as_sandwich() {
    // HeadTail uses the same alternating algorithm as Sandwich
    let sandwich_cfg = ReorderConfig::default().with_strategy(ReorderStrategy::Sandwich);
    let headtail_cfg = ReorderConfig::default().with_strategy(ReorderStrategy::HeadTail);
    let s = LostInMiddleReorderer::new(sandwich_cfg);
    let h = LostInMiddleReorderer::new(headtail_cfg);
    let results = vec![
        make_result("a", 0.9),
        make_result("b", 0.7),
        make_result("c", 0.5),
        make_result("d", 0.3),
    ];
    let sr = s.reorder(&results).unwrap();
    let hr = h.reorder(&results).unwrap();
    let s_ids: Vec<&str> = sr.iter().map(|r| r.document.id.as_str()).collect();
    let h_ids: Vec<&str> = hr.iter().map(|r| r.document.id.as_str()).collect();
    assert_eq!(
        s_ids, h_ids,
        "Sandwich and HeadTail should produce identical orderings"
    );
}

// ── Descending strategy ───────────────────────────────────────────────────

#[test]
fn test_descending_output_order_is_score_descending() {
    let cfg = ReorderConfig::default().with_strategy(ReorderStrategy::Descending);
    let r = LostInMiddleReorderer::new(cfg);
    let results = vec![
        make_result("c", 0.3),
        make_result("a", 0.9),
        make_result("b", 0.6),
    ];
    let reordered = r.reorder(&results).unwrap();
    assert_eq!(
        reordered[0].document.id.as_str(),
        "a",
        "Descending[0] should be 'a' (score 0.9)"
    );
    assert_eq!(
        reordered[1].document.id.as_str(),
        "b",
        "Descending[1] should be 'b' (score 0.6)"
    );
    assert_eq!(
        reordered[2].document.id.as_str(),
        "c",
        "Descending[2] should be 'c' (score 0.3)"
    );
}

#[test]
fn test_descending_scores_are_non_increasing() {
    let cfg = ReorderConfig::default().with_strategy(ReorderStrategy::Descending);
    let r = LostInMiddleReorderer::new(cfg);
    let results = vec![
        make_result("e", 0.2),
        make_result("c", 0.5),
        make_result("a", 0.9),
        make_result("b", 0.7),
        make_result("d", 0.3),
    ];
    let reordered = r.reorder(&results).unwrap();
    for i in 1..reordered.len() {
        assert!(
            reordered[i - 1].score >= reordered[i].score,
            "Descending: scores should be non-increasing at position {i}"
        );
    }
}

// ── Custom strategy ───────────────────────────────────────────────────────

#[test]
fn test_custom_output_is_score_sorted_descending() {
    let cfg = ReorderConfig::default().with_strategy(ReorderStrategy::Custom);
    let r = LostInMiddleReorderer::new(cfg);
    let results = vec![
        make_result("x", 0.4),
        make_result("y", 0.8),
        make_result("z", 0.2),
    ];
    let reordered = r.reorder(&results).unwrap();
    assert_eq!(
        reordered[0].document.id.as_str(),
        "y",
        "Custom[0] should be 'y' (score 0.8)"
    );
    assert_eq!(
        reordered[1].document.id.as_str(),
        "x",
        "Custom[1] should be 'x' (score 0.4)"
    );
    assert_eq!(
        reordered[2].document.id.as_str(),
        "z",
        "Custom[2] should be 'z' (score 0.2)"
    );
}

// ── reorder_with_report ───────────────────────────────────────────────────

#[test]
fn test_reorder_with_report_original_order_correct_ids() {
    let r = LostInMiddleReorderer::default();
    let results = vec![
        make_result("first", 0.9),
        make_result("second", 0.5),
        make_result("third", 0.1),
    ];
    let (_, report) = r.reorder_with_report(&results).unwrap();
    assert_eq!(
        report.original_order[0], "first",
        "original_order[0] should be 'first'"
    );
    assert_eq!(
        report.original_order[1], "second",
        "original_order[1] should be 'second'"
    );
    assert_eq!(
        report.original_order[2], "third",
        "original_order[2] should be 'third'"
    );
}

#[test]
fn test_reorder_with_report_new_order_differs_from_original_when_applicable() {
    // Use inputs that are NOT already in sandwich order, so new_order differs
    let cfg = ReorderConfig::default().with_strategy(ReorderStrategy::Sandwich);
    let r = LostInMiddleReorderer::new(cfg);
    let results = vec![
        make_result("low", 0.1),
        make_result("high", 1.0),
        make_result("mid", 0.5),
    ];
    let (_, report) = r.reorder_with_report(&results).unwrap();
    // Original: [low, high, mid]; new: [high, low, mid] or [high, mid, low] etc.
    assert_ne!(
        report.original_order, report.new_order,
        "new_order should differ from original_order for unsorted input"
    );
}

#[test]
fn test_reorder_with_report_strategy_matches_config() {
    let cfg = ReorderConfig::default().with_strategy(ReorderStrategy::Descending);
    let r = LostInMiddleReorderer::new(cfg);
    let results = vec![make_result("a", 0.9), make_result("b", 0.5)];
    let (_, report) = r.reorder_with_report(&results).unwrap();
    assert_eq!(
        report.strategy,
        ReorderStrategy::Descending,
        "report strategy should match the configured strategy"
    );
}

// ── General invariants ────────────────────────────────────────────────────

#[test]
fn test_reorder_all_input_results_present_in_output() {
    let r = LostInMiddleReorderer::default();
    let results = vec![
        make_result("a", 0.9),
        make_result("b", 0.7),
        make_result("c", 0.5),
        make_result("d", 0.3),
    ];
    let reordered = r.reorder(&results).unwrap();
    assert_eq!(
        reordered.len(),
        4,
        "output should contain all 4 input results"
    );
    let ids: Vec<&str> = reordered.iter().map(|r| r.document.id.as_str()).collect();
    for expected in &["a", "b", "c", "d"] {
        assert!(
            ids.contains(expected),
            "output should contain id '{expected}'"
        );
    }
}

#[test]
fn test_reorder_ranks_are_zero_indexed_consecutive() {
    let r = LostInMiddleReorderer::default();
    let results = vec![
        make_result("a", 0.9),
        make_result("b", 0.7),
        make_result("c", 0.5),
    ];
    let reordered = r.reorder(&results).unwrap();
    for (i, result) in reordered.iter().enumerate() {
        assert_eq!(
            result.rank, i,
            "rank at position {i} should be {i}, got {}",
            result.rank
        );
    }
}

#[test]
fn test_reorder_single_result_returned_unchanged() {
    let r = LostInMiddleReorderer::default();
    let results = vec![make_result("only", 0.7)];
    let reordered = r.reorder(&results).unwrap();
    assert_eq!(
        reordered.len(),
        1,
        "single result should yield 1-element output"
    );
    assert_eq!(
        reordered[0].document.id.as_str(),
        "only",
        "single result id should be 'only'"
    );
}

#[test]
fn test_reorder_two_results_sandwich_puts_highest_first() {
    let cfg = ReorderConfig::default().with_strategy(ReorderStrategy::Sandwich);
    let r = LostInMiddleReorderer::new(cfg);
    let results = vec![make_result("low", 0.3), make_result("high", 0.8)];
    let reordered = r.reorder(&results).unwrap();
    assert_eq!(
        reordered[0].document.id.as_str(),
        "high",
        "two-result sandwich: highest score should be first"
    );
}

#[test]
#[allow(clippy::cast_precision_loss)]
fn test_reorder_large_input_output_same_count() {
    let r = LostInMiddleReorderer::default();
    let results: Vec<SearchResult> = (0..10)
        .map(|i| make_result(&format!("doc{i}"), 1.0 - i as f32 * 0.1))
        .collect();
    let reordered = r.reorder(&results).unwrap();
    assert_eq!(
        reordered.len(),
        10,
        "large input: output should have same count (10)"
    );
}

#[test]
#[allow(clippy::cast_precision_loss)]
fn test_reorder_large_input_all_ranks_unique() {
    let r = LostInMiddleReorderer::default();
    let results: Vec<SearchResult> = (0..10)
        .map(|i| make_result(&format!("doc{i}"), 1.0 - i as f32 * 0.1))
        .collect();
    let reordered = r.reorder(&results).unwrap();
    let mut ranks: Vec<usize> = reordered.iter().map(|r| r.rank).collect();
    ranks.sort_unstable();
    let expected: Vec<usize> = (0..10).collect();
    assert_eq!(
        ranks, expected,
        "ranks should be exactly 0..9 after reorder"
    );
}

#[test]
fn test_error_empty_results_display_message() {
    let e = LostInMiddleError::EmptyResults;
    let msg = e.to_string();
    assert!(
        !msg.is_empty(),
        "EmptyResults error should have a display message"
    );
    assert!(
        msg.contains("empty") || msg.contains("Results") || msg.contains("results"),
        "EmptyResults display should be descriptive, got: {msg}"
    );
}

#[test]
fn test_preserve_top_k_zero_means_reorder_all() {
    // preserve_top_k=0 is the default — all results are reordered
    let cfg = ReorderConfig::default()
        .with_strategy(ReorderStrategy::Descending)
        .with_preserve_top_k(0);
    let r = LostInMiddleReorderer::new(cfg);
    let results = vec![
        make_result("low", 0.2),
        make_result("high", 0.9),
        make_result("mid", 0.5),
    ];
    let reordered = r.reorder(&results).unwrap();
    // All 3 should be returned
    assert_eq!(
        reordered.len(),
        3,
        "preserve_top_k=0 should include all results"
    );
    // With Descending, order should be high, mid, low
    assert_eq!(
        reordered[0].document.id.as_str(),
        "high",
        "first should be highest-scored"
    );
}
