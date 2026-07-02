#![allow(clippy::float_cmp, clippy::similar_names, clippy::too_many_lines)]
//! Unit tests for the `chain_of_table` module.

use std::collections::HashSet;

use crate::chain_of_table::engine::ChainOfTableEngine;
use crate::chain_of_table::ops::{apply, apply_with_config};
use crate::chain_of_table::types::{
    ChainOfTableConfig, ChainOfTableError, CotAddRule, CotAggregate, CotAnswerValue, CotCell,
    CotColumn, CotColumnType, CotComparator, CotEnabledOperations, CotOperationKind, CotPredicate,
    CotRow, CotTableOperation, CotTableState, cot_cell_cmp, format_number,
};

// ── helpers ──────────────────────────────────────────────────────────────────

/// A small products table: product (text), category (text), price (number),
/// quantity (number).
fn products_table() -> CotTableState {
    CotTableState::parse(
        &["product", "category", "price", "quantity"],
        &[
            vec!["Book A", "Books", "12", "3"],
            vec!["Toy B", "Toys", "8", "5"],
            vec!["Book C", "Books", "15", "2"],
            vec!["Pen D", "Office", "3", "10"],
        ],
    )
    .expect("valid products table")
}

/// A sales-by-region table used for grouping/aggregation.
fn sales_table() -> CotTableState {
    CotTableState::parse(
        &["region", "sales"],
        &[
            vec!["East", "100"],
            vec!["West", "50"],
            vec!["East", "40"],
            vec!["West", "60"],
        ],
    )
    .expect("valid sales table")
}

fn default_engine() -> ChainOfTableEngine {
    ChainOfTableEngine::new(ChainOfTableConfig::default())
}

// ── CotCell parsing / accessors ──────────────────────────────────────────────

#[test]
fn cell_parse_number() {
    assert_eq!(CotCell::parse("42"), CotCell::Number(42.0));
    assert_eq!(CotCell::parse(" 3.5 "), CotCell::Number(3.5));
    assert_eq!(CotCell::parse("-7"), CotCell::Number(-7.0));
}

#[test]
fn cell_parse_text() {
    assert_eq!(CotCell::parse("Books"), CotCell::Text("Books".to_string()));
    assert_eq!(CotCell::parse(" hi "), CotCell::Text("hi".to_string()));
}

#[test]
fn cell_parse_empty() {
    assert_eq!(CotCell::parse(""), CotCell::Empty);
    assert_eq!(CotCell::parse("   "), CotCell::Empty);
}

#[test]
fn cell_parse_rejects_non_finite() {
    // "NaN"/"inf" parse as f64 but are not finite, so they become text.
    assert_eq!(CotCell::parse("NaN"), CotCell::Text("NaN".to_string()));
    assert_eq!(CotCell::parse("inf"), CotCell::Text("inf".to_string()));
}

#[test]
fn cell_accessors() {
    let number = CotCell::Number(9.0);
    let text = CotCell::Text("x".to_string());
    let empty = CotCell::Empty;
    assert!(number.is_number() && !number.is_text() && !number.is_empty());
    assert!(text.is_text() && !text.is_number());
    assert!(empty.is_empty());
    assert_eq!(number.as_number(), Some(9.0));
    assert_eq!(text.as_number(), None);
    assert_eq!(text.as_text(), Some("x"));
    assert_eq!(number.as_text(), None);
}

#[test]
fn cell_display_string() {
    assert_eq!(CotCell::Number(12.0).to_display_string(), "12");
    assert_eq!(CotCell::Number(3.5).to_display_string(), "3.5");
    assert_eq!(CotCell::Text("hi".to_string()).to_display_string(), "hi");
    assert_eq!(CotCell::Empty.to_display_string(), "");
}

#[test]
fn cell_inferred_type() {
    assert_eq!(CotCell::Number(1.0).inferred_type(), CotColumnType::Number);
    assert_eq!(
        CotCell::Text("a".to_string()).inferred_type(),
        CotColumnType::Text
    );
    assert_eq!(CotCell::Empty.inferred_type(), CotColumnType::Text);
}

#[test]
fn format_number_integral_vs_fractional() {
    assert_eq!(format_number(5.0), "5");
    assert_eq!(format_number(5.25), "5.25");
    assert_eq!(format_number(-2.0), "-2");
}

// ── cot_cell_cmp ─────────────────────────────────────────────────────────────

#[test]
fn cell_cmp_empty_is_smallest() {
    use std::cmp::Ordering;
    assert_eq!(
        cot_cell_cmp(&CotCell::Empty, &CotCell::Number(0.0), false),
        Ordering::Less
    );
    assert_eq!(
        cot_cell_cmp(&CotCell::Empty, &CotCell::Empty, false),
        Ordering::Equal
    );
}

#[test]
fn cell_cmp_numbers_before_text() {
    use std::cmp::Ordering;
    assert_eq!(
        cot_cell_cmp(
            &CotCell::Number(999.0),
            &CotCell::Text("a".to_string()),
            false
        ),
        Ordering::Less
    );
}

#[test]
fn cell_cmp_numbers_by_value() {
    use std::cmp::Ordering;
    assert_eq!(
        cot_cell_cmp(&CotCell::Number(2.0), &CotCell::Number(10.0), false),
        Ordering::Less
    );
}

#[test]
fn cell_cmp_text_case_sensitivity() {
    use std::cmp::Ordering;
    let upper = CotCell::Text("Apple".to_string());
    let lower = CotCell::Text("apple".to_string());
    // Case-insensitive: equal; case-sensitive: 'A' < 'a'.
    assert_eq!(cot_cell_cmp(&upper, &lower, false), Ordering::Equal);
    assert_eq!(cot_cell_cmp(&upper, &lower, true), Ordering::Less);
}

// ── table parsing + type inference ───────────────────────────────────────────

#[test]
fn parse_infers_column_types() {
    let table = products_table();
    assert_eq!(table.column_count(), 4);
    assert_eq!(table.row_count(), 4);
    assert_eq!(table.columns[0].column_type, CotColumnType::Text);
    assert_eq!(table.columns[1].column_type, CotColumnType::Text);
    assert_eq!(table.columns[2].column_type, CotColumnType::Number);
    assert_eq!(table.columns[3].column_type, CotColumnType::Number);
    assert_eq!(table.cell(0, 2), Some(&CotCell::Number(12.0)));
    assert_eq!(table.cell(0, 0), Some(&CotCell::Text("Book A".to_string())));
}

#[test]
fn parse_empty_cells_become_empty() {
    let table =
        CotTableState::parse(&["a", "b"], &[vec!["1", ""], vec!["", "x"]]).expect("valid table");
    assert_eq!(table.cell(0, 1), Some(&CotCell::Empty));
    assert_eq!(table.cell(1, 0), Some(&CotCell::Empty));
}

#[test]
fn parse_mixed_column_is_text_and_coerces_numbers() {
    // A column with one text value forces the whole column to text; the
    // numeric-looking values are coerced to their text rendering.
    let table =
        CotTableState::parse(&["code"], &[vec!["1"], vec!["x"], vec!["2"]]).expect("valid table");
    assert_eq!(table.columns[0].column_type, CotColumnType::Text);
    assert_eq!(table.cell(0, 0), Some(&CotCell::Text("1".to_string())));
    assert_eq!(table.cell(2, 0), Some(&CotCell::Text("2".to_string())));
}

#[test]
fn parse_all_empty_column_defaults_to_text() {
    let table = CotTableState::parse(&["a"], &[vec![""], vec![""]]).expect("valid table");
    assert_eq!(table.columns[0].column_type, CotColumnType::Text);
}

#[test]
fn parse_empty_headers_errors() {
    let headers: [&str; 0] = [];
    let rows: Vec<Vec<&str>> = Vec::new();
    let error = CotTableState::parse(&headers, &rows).expect_err("empty headers");
    assert_eq!(error, ChainOfTableError::EmptyTable);
}

#[test]
fn parse_ragged_row_errors() {
    let error =
        CotTableState::parse(&["a", "b"], &[vec!["1", "2"], vec!["3"]]).expect_err("ragged row");
    assert_eq!(
        error,
        ChainOfTableError::RaggedRow {
            row: 1,
            expected: 2,
            found: 1,
        }
    );
}

#[test]
fn parse_duplicate_column_errors() {
    let error = CotTableState::parse(&["a", "a"], &[vec!["1", "2"]]).expect_err("duplicate column");
    assert_eq!(
        error,
        ChainOfTableError::DuplicateColumn {
            column: "a".to_string(),
        }
    );
}

#[test]
fn parse_zero_rows_is_valid() {
    let rows: Vec<Vec<&str>> = Vec::new();
    let table = CotTableState::parse(&["a", "b"], &rows).expect("header-only table");
    assert_eq!(table.column_count(), 2);
    assert_eq!(table.row_count(), 0);
    assert!(!table.is_empty());
}

// ── CotTableState helpers ────────────────────────────────────────────────────

#[test]
fn resolve_column_case_sensitivity() {
    let table = products_table();
    assert_eq!(table.resolve_column("price", true), Some(2));
    assert_eq!(table.resolve_column("PRICE", true), None);
    assert_eq!(table.resolve_column("PRICE", false), Some(2));
}

#[test]
fn column_names_and_scalar_detection() {
    let table = products_table();
    assert_eq!(
        table.column_names(),
        vec![
            "product".to_string(),
            "category".to_string(),
            "price".to_string(),
            "quantity".to_string()
        ]
    );
    assert!(!table.is_scalar());
    let scalar = CotTableState::new(
        vec![CotColumn::number("n")],
        vec![CotRow::new(vec![CotCell::Number(1.0)])],
    );
    assert!(scalar.is_scalar());
}

// ── f_select_row ─────────────────────────────────────────────────────────────

#[test]
fn select_row_numeric_greater_than() {
    let table = products_table();
    let op = CotTableOperation::SelectRow {
        predicate: CotPredicate::new("price", CotComparator::Gt, CotCell::Number(10.0)),
    };
    let result = apply(&op, &table).expect("select_row");
    assert_eq!(result.row_count(), 2); // 12 and 15
    for row in &result.rows {
        assert!(row.get(2).and_then(CotCell::as_number).unwrap() > 10.0);
    }
}

#[test]
fn select_row_text_equality() {
    let table = products_table();
    let op = CotTableOperation::SelectRow {
        predicate: CotPredicate::new(
            "category",
            CotComparator::Eq,
            CotCell::Text("Books".to_string()),
        ),
    };
    let result = apply(&op, &table).expect("select_row");
    assert_eq!(result.row_count(), 2);
}

#[test]
fn select_row_not_equal() {
    let table = products_table();
    let op = CotTableOperation::SelectRow {
        predicate: CotPredicate::new(
            "category",
            CotComparator::Ne,
            CotCell::Text("Books".to_string()),
        ),
    };
    let result = apply(&op, &table).expect("select_row");
    assert_eq!(result.row_count(), 2); // Toys, Office
}

#[test]
fn select_row_le_and_ge() {
    let table = products_table();
    let le = apply(
        &CotTableOperation::SelectRow {
            predicate: CotPredicate::new("price", CotComparator::Le, CotCell::Number(8.0)),
        },
        &table,
    )
    .expect("le");
    assert_eq!(le.row_count(), 2); // 8, 3

    let ge = apply(
        &CotTableOperation::SelectRow {
            predicate: CotPredicate::new("quantity", CotComparator::Ge, CotCell::Number(5.0)),
        },
        &table,
    )
    .expect("ge");
    assert_eq!(ge.row_count(), 2); // 5, 10
}

#[test]
fn select_row_case_sensitivity() {
    let table = products_table();
    let op = CotTableOperation::SelectRow {
        predicate: CotPredicate::new(
            "category",
            CotComparator::Eq,
            CotCell::Text("books".to_string()),
        ),
    };
    let insensitive = apply(&op, &table).expect("insensitive");
    assert_eq!(insensitive.row_count(), 2);

    let sensitive_config = ChainOfTableConfig::default().with_case_sensitive(true);
    let sensitive = apply_with_config(&op, &table, &sensitive_config).expect("sensitive");
    assert_eq!(sensitive.row_count(), 0); // "books" != "Books"
}

#[test]
fn select_row_ordering_ignores_empty_and_cross_type() {
    let table = CotTableState::parse(&["n"], &[vec!["5"], vec![""], vec!["12"]]).expect("table");
    let op = CotTableOperation::SelectRow {
        predicate: CotPredicate::new("n", CotComparator::Gt, CotCell::Number(4.0)),
    };
    let result = apply(&op, &table).expect("select");
    // Empty cell does not satisfy an ordering comparator.
    assert_eq!(result.row_count(), 2);
}

#[test]
fn select_row_unknown_column_errors() {
    let table = products_table();
    let op = CotTableOperation::SelectRow {
        predicate: CotPredicate::new("nope", CotComparator::Eq, CotCell::Number(1.0)),
    };
    let error = apply(&op, &table).expect_err("unknown column");
    assert_eq!(
        error,
        ChainOfTableError::UnknownColumn {
            column: "nope".to_string(),
        }
    );
}

// ── f_select_column ──────────────────────────────────────────────────────────

#[test]
fn select_column_projects_subset_in_order() {
    let table = products_table();
    let op = CotTableOperation::SelectColumn {
        columns: vec!["price".to_string(), "product".to_string()],
    };
    let result = apply(&op, &table).expect("projection");
    assert_eq!(
        result.column_names(),
        vec!["price".to_string(), "product".to_string()]
    );
    assert_eq!(result.row_count(), 4);
    assert_eq!(result.cell(0, 0), Some(&CotCell::Number(12.0)));
    assert_eq!(
        result.cell(0, 1),
        Some(&CotCell::Text("Book A".to_string()))
    );
}

#[test]
fn select_column_empty_projection_errors() {
    let table = products_table();
    let op = CotTableOperation::SelectColumn {
        columns: Vec::new(),
    };
    let error = apply(&op, &table).expect_err("empty projection");
    assert_eq!(error, ChainOfTableError::EmptyProjection);
}

#[test]
fn select_column_unknown_column_errors() {
    let table = products_table();
    let op = CotTableOperation::SelectColumn {
        columns: vec!["ghost".to_string()],
    };
    let error = apply(&op, &table).expect_err("unknown column");
    assert_eq!(
        error,
        ChainOfTableError::UnknownColumn {
            column: "ghost".to_string(),
        }
    );
}

// ── f_sort_by ────────────────────────────────────────────────────────────────

#[test]
fn sort_by_ascending() {
    let table = products_table();
    let op = CotTableOperation::SortBy {
        column: "price".to_string(),
        ascending: true,
    };
    let result = apply(&op, &table).expect("sort");
    let prices: Vec<f64> = result
        .rows
        .iter()
        .filter_map(|row| row.get(2).and_then(CotCell::as_number))
        .collect();
    assert_eq!(prices, vec![3.0, 8.0, 12.0, 15.0]);
}

#[test]
fn sort_by_descending() {
    let table = products_table();
    let op = CotTableOperation::SortBy {
        column: "price".to_string(),
        ascending: false,
    };
    let result = apply(&op, &table).expect("sort");
    let prices: Vec<f64> = result
        .rows
        .iter()
        .filter_map(|row| row.get(2).and_then(CotCell::as_number))
        .collect();
    assert_eq!(prices, vec![15.0, 12.0, 8.0, 3.0]);
}

#[test]
fn sort_by_text_lexicographic() {
    let table = products_table();
    let op = CotTableOperation::SortBy {
        column: "product".to_string(),
        ascending: true,
    };
    let result = apply(&op, &table).expect("sort");
    let names: Vec<String> = result
        .rows
        .iter()
        .filter_map(|row| row.get(0).and_then(|c| c.as_text().map(str::to_string)))
        .collect();
    assert_eq!(
        names,
        vec![
            "Book A".to_string(),
            "Book C".to_string(),
            "Pen D".to_string(),
            "Toy B".to_string()
        ]
    );
}

#[test]
fn sort_by_is_stable() {
    // Two rows with equal sort keys keep their original relative order.
    let table = CotTableState::parse(
        &["k", "tag"],
        &[vec!["1", "first"], vec!["1", "second"], vec!["0", "third"]],
    )
    .expect("table");
    let result = apply(
        &CotTableOperation::SortBy {
            column: "k".to_string(),
            ascending: true,
        },
        &table,
    )
    .expect("sort");
    assert_eq!(result.cell(0, 1), Some(&CotCell::Text("third".to_string())));
    assert_eq!(result.cell(1, 1), Some(&CotCell::Text("first".to_string())));
    assert_eq!(
        result.cell(2, 1),
        Some(&CotCell::Text("second".to_string()))
    );
}

// ── f_group_by ───────────────────────────────────────────────────────────────

#[test]
fn group_by_reorders_and_tags() {
    let table = sales_table();
    let result = apply(
        &CotTableOperation::GroupBy {
            column: "region".to_string(),
        },
        &table,
    )
    .expect("group_by");
    assert_eq!(result.group_key(), Some("region"));
    // First-appearance order: all East rows, then all West rows.
    let regions: Vec<String> = result
        .rows
        .iter()
        .filter_map(|row| row.get(0).and_then(|c| c.as_text().map(str::to_string)))
        .collect();
    assert_eq!(
        regions,
        vec![
            "East".to_string(),
            "East".to_string(),
            "West".to_string(),
            "West".to_string()
        ]
    );
    assert_eq!(result.row_count(), 4); // grouping does not collapse
}

#[test]
fn group_by_unknown_column_errors() {
    let table = sales_table();
    let error = apply(
        &CotTableOperation::GroupBy {
            column: "zone".to_string(),
        },
        &table,
    )
    .expect_err("unknown column");
    assert_eq!(
        error,
        ChainOfTableError::UnknownColumn {
            column: "zone".to_string(),
        }
    );
}

// ── f_aggregate (flat) ───────────────────────────────────────────────────────

fn aggregate_scalar(table: &CotTableState, column: &str, agg: CotAggregate) -> f64 {
    let result = apply(
        &CotTableOperation::Aggregate {
            column: column.to_string(),
            agg,
        },
        table,
    )
    .expect("aggregate");
    assert!(result.is_scalar());
    result
        .cell(0, 0)
        .and_then(CotCell::as_number)
        .expect("numeric scalar")
}

#[test]
fn aggregate_count() {
    let table = products_table();
    assert_eq!(
        aggregate_scalar(&table, "product", CotAggregate::Count),
        4.0
    );
}

#[test]
fn aggregate_sum_mean_max_min() {
    let table = products_table();
    assert_eq!(aggregate_scalar(&table, "price", CotAggregate::Sum), 38.0);
    assert_eq!(aggregate_scalar(&table, "price", CotAggregate::Mean), 9.5);
    assert_eq!(aggregate_scalar(&table, "price", CotAggregate::Max), 15.0);
    assert_eq!(aggregate_scalar(&table, "price", CotAggregate::Min), 3.0);
}

#[test]
fn aggregate_count_ignores_empty_cells() {
    let table = CotTableState::parse(&["a"], &[vec!["x"], vec![""], vec!["y"]]).expect("table");
    assert_eq!(aggregate_scalar(&table, "a", CotAggregate::Count), 2.0);
}

#[test]
fn aggregate_sum_skips_empty() {
    let table = CotTableState::parse(&["n"], &[vec!["10"], vec![""], vec!["5"]]).expect("table");
    assert_eq!(aggregate_scalar(&table, "n", CotAggregate::Sum), 15.0);
}

#[test]
fn aggregate_over_text_type_mismatch() {
    let table = products_table();
    let error = apply(
        &CotTableOperation::Aggregate {
            column: "category".to_string(),
            agg: CotAggregate::Sum,
        },
        &table,
    )
    .expect_err("sum over text");
    assert_eq!(
        error,
        ChainOfTableError::TypeMismatch {
            operation: "f_aggregate".to_string(),
            column: "category".to_string(),
        }
    );
}

#[test]
fn aggregate_mean_over_empty_numeric_errors() {
    let table = CotTableState::parse(&["n"], &[vec![""], vec![""]]).expect("table");
    // Column has no numeric values; parse types it as text, so mean reports an
    // empty aggregate only when the column is numeric. Build a numeric-empty
    // column explicitly.
    let numeric_empty = CotTableState::new(
        vec![CotColumn::number("n")],
        vec![
            CotRow::new(vec![CotCell::Empty]),
            CotRow::new(vec![CotCell::Empty]),
        ],
    );
    let _ = table;
    let error = apply(
        &CotTableOperation::Aggregate {
            column: "n".to_string(),
            agg: CotAggregate::Mean,
        },
        &numeric_empty,
    )
    .expect_err("mean over empty");
    assert_eq!(
        error,
        ChainOfTableError::EmptyAggregate {
            aggregate: "mean".to_string(),
            column: "n".to_string(),
        }
    );
}

#[test]
fn aggregate_sum_over_no_values_is_zero() {
    let numeric_empty = CotTableState::new(
        vec![CotColumn::number("n")],
        vec![CotRow::new(vec![CotCell::Empty])],
    );
    assert_eq!(
        aggregate_scalar(&numeric_empty, "n", CotAggregate::Sum),
        0.0
    );
}

#[test]
fn aggregate_unknown_column_errors() {
    let table = products_table();
    let error = apply(
        &CotTableOperation::Aggregate {
            column: "missing".to_string(),
            agg: CotAggregate::Count,
        },
        &table,
    )
    .expect_err("unknown");
    assert_eq!(
        error,
        ChainOfTableError::UnknownColumn {
            column: "missing".to_string(),
        }
    );
}

// ── group_by + aggregate ─────────────────────────────────────────────────────

fn grouped_totals(agg: CotAggregate) -> CotTableState {
    let table = sales_table();
    let grouped = apply(
        &CotTableOperation::GroupBy {
            column: "region".to_string(),
        },
        &table,
    )
    .expect("group_by");
    apply(
        &CotTableOperation::Aggregate {
            column: "sales".to_string(),
            agg,
        },
        &grouped,
    )
    .expect("aggregate")
}

#[test]
fn group_by_then_sum() {
    let result = grouped_totals(CotAggregate::Sum);
    assert_eq!(result.row_count(), 2);
    assert_eq!(
        result.column_names(),
        vec!["region".to_string(), "sum".to_string()]
    );
    assert_eq!(result.cell(0, 0), Some(&CotCell::Text("East".to_string())));
    assert_eq!(result.cell(0, 1), Some(&CotCell::Number(140.0)));
    assert_eq!(result.cell(1, 0), Some(&CotCell::Text("West".to_string())));
    assert_eq!(result.cell(1, 1), Some(&CotCell::Number(110.0)));
    assert_eq!(result.group_key(), None); // aggregate consumes the group key
}

#[test]
fn group_by_then_count() {
    let result = grouped_totals(CotAggregate::Count);
    assert_eq!(result.cell(0, 1), Some(&CotCell::Number(2.0)));
    assert_eq!(result.cell(1, 1), Some(&CotCell::Number(2.0)));
}

#[test]
fn group_by_then_mean_max_min() {
    let mean = grouped_totals(CotAggregate::Mean);
    assert_eq!(mean.cell(0, 1), Some(&CotCell::Number(70.0))); // (100+40)/2
    let max = grouped_totals(CotAggregate::Max);
    assert_eq!(max.cell(0, 1), Some(&CotCell::Number(100.0)));
    let min = grouped_totals(CotAggregate::Min);
    assert_eq!(min.cell(0, 1), Some(&CotCell::Number(40.0)));
}

// ── f_add_column ─────────────────────────────────────────────────────────────

#[test]
fn add_column_sum_rule() {
    let table = products_table();
    let op = CotTableOperation::AddColumn {
        name: "total_units".to_string(),
        rule: CotAddRule::Sum {
            columns: vec!["price".to_string(), "quantity".to_string()],
        },
    };
    let result = apply(&op, &table).expect("add_column");
    assert_eq!(result.column_count(), 5);
    assert_eq!(result.columns[4].column_type, CotColumnType::Number);
    assert_eq!(result.cell(0, 4), Some(&CotCell::Number(15.0))); // 12 + 3
}

#[test]
fn add_column_difference_rule() {
    let table = products_table();
    let op = CotTableOperation::AddColumn {
        name: "gap".to_string(),
        rule: CotAddRule::Difference {
            left: "price".to_string(),
            right: "quantity".to_string(),
        },
    };
    let result = apply(&op, &table).expect("add_column");
    assert_eq!(result.cell(0, 4), Some(&CotCell::Number(9.0))); // 12 - 3
    assert_eq!(result.cell(3, 4), Some(&CotCell::Number(-7.0))); // 3 - 10
}

#[test]
fn add_column_product_rule() {
    let table = products_table();
    let op = CotTableOperation::AddColumn {
        name: "revenue".to_string(),
        rule: CotAddRule::Product {
            columns: vec!["price".to_string(), "quantity".to_string()],
        },
    };
    let result = apply(&op, &table).expect("add_column");
    assert_eq!(result.cell(0, 4), Some(&CotCell::Number(36.0))); // 12 * 3
    assert_eq!(result.cell(1, 4), Some(&CotCell::Number(40.0))); // 8 * 5
}

#[test]
fn add_column_ratio_rule_and_zero_denominator() {
    let table = CotTableState::parse(
        &["a", "b"],
        &[vec!["10", "2"], vec!["9", "0"], vec!["7", "1"]],
    )
    .expect("table");
    let op = CotTableOperation::AddColumn {
        name: "ratio".to_string(),
        rule: CotAddRule::Ratio {
            numerator: "a".to_string(),
            denominator: "b".to_string(),
        },
    };
    let result = apply(&op, &table).expect("add_column");
    assert_eq!(result.cell(0, 2), Some(&CotCell::Number(5.0)));
    assert_eq!(result.cell(1, 2), Some(&CotCell::Empty)); // divide by zero → empty
    assert_eq!(result.cell(2, 2), Some(&CotCell::Number(7.0)));
}

#[test]
fn add_column_constant_rule() {
    let table = products_table();
    let op = CotTableOperation::AddColumn {
        name: "flag".to_string(),
        rule: CotAddRule::Constant {
            value: CotCell::Text("yes".to_string()),
        },
    };
    let result = apply(&op, &table).expect("add_column");
    assert_eq!(result.columns[4].column_type, CotColumnType::Text);
    for row in 0..result.row_count() {
        assert_eq!(result.cell(row, 4), Some(&CotCell::Text("yes".to_string())));
    }
}

#[test]
fn add_column_concat_rule() {
    let table = products_table();
    let op = CotTableOperation::AddColumn {
        name: "label".to_string(),
        rule: CotAddRule::Concat {
            columns: vec!["product".to_string(), "category".to_string()],
            separator: " / ".to_string(),
        },
    };
    let result = apply(&op, &table).expect("add_column");
    assert_eq!(
        result.cell(0, 4),
        Some(&CotCell::Text("Book A / Books".to_string()))
    );
}

#[test]
fn add_column_duplicate_name_errors() {
    let table = products_table();
    let op = CotTableOperation::AddColumn {
        name: "price".to_string(),
        rule: CotAddRule::Constant {
            value: CotCell::Number(0.0),
        },
    };
    let error = apply(&op, &table).expect_err("duplicate");
    assert_eq!(
        error,
        ChainOfTableError::DuplicateColumn {
            column: "price".to_string(),
        }
    );
}

#[test]
fn add_column_numeric_rule_over_text_errors() {
    let table = products_table();
    let op = CotTableOperation::AddColumn {
        name: "bad".to_string(),
        rule: CotAddRule::Sum {
            columns: vec!["category".to_string()],
        },
    };
    let error = apply(&op, &table).expect_err("type mismatch");
    assert_eq!(
        error,
        ChainOfTableError::TypeMismatch {
            operation: "f_add_column".to_string(),
            column: "category".to_string(),
        }
    );
}

// ── apply on empty table ─────────────────────────────────────────────────────

#[test]
fn apply_on_empty_table_errors() {
    let empty = CotTableState::default();
    let op = CotTableOperation::Aggregate {
        column: "x".to_string(),
        agg: CotAggregate::Count,
    };
    let error = apply(&op, &empty).expect_err("empty table");
    assert_eq!(error, ChainOfTableError::EmptyTable);
}

// ── CotTableOperation metadata ───────────────────────────────────────────────

#[test]
fn operation_kind_and_name() {
    let op = CotTableOperation::GroupBy {
        column: "c".to_string(),
    };
    assert_eq!(op.kind(), CotOperationKind::GroupBy);
    assert_eq!(op.name(), "f_group_by");
    assert_eq!(
        CotTableOperation::Aggregate {
            column: "c".to_string(),
            agg: CotAggregate::Count
        }
        .name(),
        "f_aggregate"
    );
}

#[test]
fn operation_kind_names_cover_vocabulary() {
    let names: Vec<&str> = CotOperationKind::all().iter().map(|k| k.name()).collect();
    assert!(names.contains(&"f_add_column"));
    assert!(names.contains(&"f_select_row"));
    assert!(names.contains(&"f_select_column"));
    assert!(names.contains(&"f_group_by"));
    assert!(names.contains(&"f_sort_by"));
    assert!(names.contains(&"f_aggregate"));
    assert_eq!(names.len(), 6);
}

#[test]
fn aggregate_metadata() {
    assert_eq!(CotAggregate::Count.as_str(), "count");
    assert!(!CotAggregate::Count.requires_numeric());
    assert!(CotAggregate::Sum.requires_numeric());
    assert_eq!(CotAggregate::default(), CotAggregate::Count);
}

#[test]
fn comparator_metadata() {
    assert_eq!(CotComparator::Gt.as_str(), ">");
    assert!(CotComparator::Gt.is_ordering());
    assert!(!CotComparator::Eq.is_ordering());
}

// ── CotEnabledOperations ─────────────────────────────────────────────────────

#[test]
fn enabled_operations_default_all() {
    let enabled = CotEnabledOperations::default();
    for kind in CotOperationKind::all() {
        assert!(enabled.is_enabled(kind));
    }
}

#[test]
fn enabled_operations_none_and_with() {
    let none = CotEnabledOperations::none();
    assert!(!none.is_enabled(CotOperationKind::Aggregate));
    let one = none.with(CotOperationKind::SortBy, true);
    assert!(one.is_enabled(CotOperationKind::SortBy));
    assert!(!one.is_enabled(CotOperationKind::Aggregate));
}

// ── config builders ──────────────────────────────────────────────────────────

#[test]
fn config_builders() {
    let config = ChainOfTableConfig::new()
        .with_max_steps(3)
        .with_case_sensitive(true)
        .with_default_aggregate(CotAggregate::Sum)
        .with_numeric_epsilon(1e-6)
        .with_enabled_operations(CotEnabledOperations::none());
    assert_eq!(config.max_steps, 3);
    assert!(config.case_sensitive);
    assert_eq!(config.default_aggregate, CotAggregate::Sum);
    assert_eq!(config.numeric_epsilon, 1e-6);
    assert!(
        !config
            .enabled_operations
            .is_enabled(CotOperationKind::SortBy)
    );
}

// ── engine: end-to-end chains ────────────────────────────────────────────────

#[test]
fn engine_how_many_where_chain() {
    let engine = default_engine();
    let table = products_table();
    let answer = engine
        .run("How many products where category is Books?", &table)
        .expect("run");
    assert_eq!(
        answer.operation_names(),
        vec!["f_select_row", "f_aggregate"]
    );
    assert_eq!(answer.as_number(), Some(2.0));
    assert!(answer.value.is_scalar());
    assert_eq!(answer.step_count(), 2);
}

#[test]
fn engine_which_has_the_most_chain() {
    let engine = default_engine();
    let table = CotTableState::parse(
        &["product", "price"],
        &[
            vec!["Widget", "10"],
            vec!["Gadget", "30"],
            vec!["Gizmo", "20"],
        ],
    )
    .expect("table");
    let answer = engine
        .run("Which product has the highest price?", &table)
        .expect("run");
    assert_eq!(answer.operation_names(), vec!["f_sort_by"]);
    // Answer is the top (highest-price) row.
    let top = answer.value.as_table().expect("table answer");
    assert_eq!(top.row_count(), 1);
    assert_eq!(top.cell(0, 0), Some(&CotCell::Text("Gadget".to_string())));
    assert_eq!(top.cell(0, 1), Some(&CotCell::Number(30.0)));
}

#[test]
fn engine_group_then_aggregate_chain() {
    let engine = default_engine();
    let table = sales_table();
    let answer = engine
        .run("What is the total sales per region?", &table)
        .expect("run");
    assert_eq!(answer.operation_names(), vec!["f_group_by", "f_aggregate"]);
    let result = answer.value.as_table().expect("table answer");
    assert_eq!(result.row_count(), 2);
    assert_eq!(result.cell(0, 0), Some(&CotCell::Text("East".to_string())));
    assert_eq!(result.cell(0, 1), Some(&CotCell::Number(140.0)));
}

#[test]
fn engine_average_scalar_chain() {
    let engine = default_engine();
    let table = products_table();
    let answer = engine
        .run("What is the average price?", &table)
        .expect("run");
    assert_eq!(answer.operation_names(), vec!["f_aggregate"]);
    assert_eq!(answer.as_number(), Some(9.5));
}

#[test]
fn engine_max_scalar_without_which() {
    let engine = default_engine();
    let table = products_table();
    // "highest price" without "which" wants the max VALUE, not the row.
    let answer = engine
        .run("What is the highest price?", &table)
        .expect("run");
    assert_eq!(answer.operation_names(), vec!["f_aggregate"]);
    assert_eq!(answer.as_number(), Some(15.0));
}

#[test]
fn engine_add_column_difference_chain() {
    let engine = default_engine();
    let table = CotTableState::parse(
        &["item", "revenue", "cost"],
        &[vec!["A", "100", "60"], vec!["B", "80", "30"]],
    )
    .expect("table");
    let answer = engine
        .run("What is the difference between revenue and cost?", &table)
        .expect("run");
    assert!(answer.operation_names().contains(&"f_add_column"));
    let derived = &answer.final_state;
    let gap_index = derived
        .resolve_column("difference", false)
        .expect("difference column");
    assert_eq!(derived.cell(0, gap_index), Some(&CotCell::Number(40.0)));
    assert_eq!(derived.cell(1, gap_index), Some(&CotCell::Number(50.0)));
}

// ── engine: control-flow / determinism / termination ─────────────────────────

#[test]
fn engine_determinism() {
    let engine = default_engine();
    let table = products_table();
    let first = engine
        .run("How many products where category is Books?", &table)
        .expect("run 1");
    let second = engine
        .run("How many products where category is Books?", &table)
        .expect("run 2");
    assert_eq!(first, second);
}

#[test]
fn engine_max_steps_termination() {
    let config = ChainOfTableConfig::default().with_max_steps(1);
    let engine = ChainOfTableEngine::new(config);
    let table = products_table();
    let answer = engine
        .run("How many products where category is Books?", &table)
        .expect("run");
    // Only the first operation (select_row) runs; the aggregate never fires.
    assert_eq!(answer.step_count(), 1);
    assert_eq!(answer.operation_names(), vec!["f_select_row"]);
    assert!(answer.value.as_table().is_some());
}

#[test]
fn engine_disabled_operation_is_not_emitted() {
    let config = ChainOfTableConfig::default().with_enabled_operations(
        CotEnabledOperations::all().with(CotOperationKind::Aggregate, false),
    );
    let engine = ChainOfTableEngine::new(config);
    let table = products_table();
    let answer = engine
        .run("How many products where category is Books?", &table)
        .expect("run");
    // select_row still runs, but the disabled aggregate does not.
    assert_eq!(answer.operation_names(), vec!["f_select_row"]);
    assert!(!answer.value.is_scalar());
}

#[test]
fn engine_empty_question_errors() {
    let engine = default_engine();
    let table = products_table();
    let error = engine.run("   ", &table).expect_err("empty question");
    assert_eq!(error, ChainOfTableError::EmptyQuestion);
}

#[test]
fn engine_empty_table_errors() {
    let engine = default_engine();
    let empty = CotTableState::default();
    let error = engine.run("count rows", &empty).expect_err("empty table");
    assert_eq!(error, ChainOfTableError::EmptyTable);
}

#[test]
fn engine_no_operations_enabled_errors() {
    let config =
        ChainOfTableConfig::default().with_enabled_operations(CotEnabledOperations::none());
    let engine = ChainOfTableEngine::new(config);
    let table = products_table();
    let error = engine.run("how many?", &table).expect_err("no ops");
    assert_eq!(error, ChainOfTableError::NoOperationsEnabled);
}

#[test]
fn engine_no_matching_intent_returns_input_table() {
    let engine = default_engine();
    let table = products_table();
    // A question with no recognised intent leaves the table unchanged.
    let answer = engine.run("Tell me about the data.", &table).expect("run");
    assert_eq!(answer.step_count(), 0);
    let result = answer.value.as_table().expect("table");
    assert_eq!(result.row_count(), 4);
    assert_eq!(result.column_count(), 4);
}

// ── engine: trace records the full chain ─────────────────────────────────────

#[test]
fn engine_trace_records_full_chain() {
    let engine = default_engine();
    let table = products_table();
    let answer = engine
        .run("How many products where category is Books?", &table)
        .expect("run");
    assert_eq!(answer.chain.len(), 2);

    let select = &answer.chain[0];
    assert_eq!(select.step, 0);
    assert_eq!(select.operation.kind(), CotOperationKind::SelectRow);
    assert_eq!(select.rows_before, 4);
    assert_eq!(select.rows_after, 2);
    assert!(select.description.contains("category"));

    let aggregate = &answer.chain[1];
    assert_eq!(aggregate.step, 1);
    assert_eq!(aggregate.operation.kind(), CotOperationKind::Aggregate);
    assert_eq!(aggregate.rows_before, 2);
    assert_eq!(aggregate.rows_after, 1);
    assert_eq!(aggregate.columns_after, 1);
}

// ── engine: planner unit ─────────────────────────────────────────────────────

#[test]
fn plan_next_skips_applied_kinds() {
    let engine = default_engine();
    let table = products_table();
    let mut applied = HashSet::new();
    let first = engine
        .plan_next(
            "How many products where category is Books?",
            &table,
            &applied,
        )
        .expect("first op");
    assert_eq!(first.kind(), CotOperationKind::SelectRow);
    applied.insert(CotOperationKind::SelectRow);

    // After filtering, the same question should now plan the aggregate.
    let filtered = apply(&first, &table).expect("apply");
    let second = engine
        .plan_next(
            "How many products where category is Books?",
            &filtered,
            &applied,
        )
        .expect("second op");
    assert_eq!(second.kind(), CotOperationKind::Aggregate);
}

#[test]
fn plan_next_returns_none_when_nothing_matches() {
    let engine = default_engine();
    let table = products_table();
    let applied = HashSet::new();
    let planned = engine.plan_next("Nothing relevant here.", &table, &applied);
    assert!(planned.is_none());
}

#[test]
fn answer_value_helpers() {
    let scalar = CotAnswerValue::Scalar(CotCell::Number(3.0));
    assert!(scalar.is_scalar());
    assert_eq!(scalar.as_scalar(), Some(&CotCell::Number(3.0)));
    assert!(scalar.as_table().is_none());

    let table_value = CotAnswerValue::Table(products_table());
    assert!(!table_value.is_scalar());
    assert!(table_value.as_table().is_some());
    assert!(table_value.as_scalar().is_none());
}
