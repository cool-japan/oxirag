//! Tests for the `table_rag` module.
#![allow(clippy::float_cmp, clippy::similar_names, clippy::too_many_lines)]

use super::engine::TableRagEngine;
use super::types::{
    CellProbe, TableColumnSpec, TableColumnType, TableRagConfig, TableRagError, TableRagIndex,
    TableRagResult, TableRagSubTable, TableRagTable, TableSchema,
};

/// Tolerance for comparing derived (sqrt/division-chain) `f32` scores
/// computed independently in Rust vs. hand-verified reference values.
const EPS: f32 = 1e-3;

// ── fixtures ──────────────────────────────────────────────────────────────

fn employees_schema() -> TableSchema {
    TableSchema::new(vec![
        TableColumnSpec::new("name", TableColumnType::Text),
        TableColumnSpec::new("department", TableColumnType::Text),
        TableColumnSpec::new("salary", TableColumnType::Number),
    ])
}

fn employees_table() -> TableRagTable {
    TableRagTable::new(
        "employees",
        employees_schema(),
        vec![
            vec![
                "Alice".to_string(),
                "Engineering".to_string(),
                "95000".to_string(),
            ],
            vec!["Bob".to_string(), "Sales".to_string(), "65000".to_string()],
            vec![
                "Carol".to_string(),
                "Engineering".to_string(),
                "120000".to_string(),
            ],
            vec![
                "Dave".to_string(),
                "Marketing".to_string(),
                "72000".to_string(),
            ],
        ],
    )
    .expect("valid employees table")
}

fn products_schema() -> TableSchema {
    TableSchema::new(vec![
        TableColumnSpec::new("product_name", TableColumnType::Text),
        TableColumnSpec::new("category", TableColumnType::Text),
        TableColumnSpec::new("price", TableColumnType::Number),
    ])
}

fn products_table() -> TableRagTable {
    TableRagTable::new(
        "products",
        products_schema(),
        vec![
            vec![
                "Widget".to_string(),
                "Hardware".to_string(),
                "19".to_string(),
            ],
            vec![
                "Gadget".to_string(),
                "Electronics".to_string(),
                "49".to_string(),
            ],
        ],
    )
    .expect("valid products table")
}

fn employees_index() -> TableRagIndex {
    TableRagIndex::new().with_table(employees_table())
}

fn multi_table_index() -> TableRagIndex {
    TableRagIndex::new()
        .with_table(employees_table())
        .with_table(products_table())
}

/// 30 distinct, hand-verified collision-free single-word values (no pair is
/// a substring of another, and none fuzzy-collides with "cat" or "kite" —
/// see the module's distinct-value-capping tests) used to exercise
/// [`TableRagConfig::max_distinct_values_per_column`] against a column much
/// larger than any reasonable cap.
const TOKEN_WORDS: [&str; 30] = [
    "cat", "dog", "sun", "moon", "star", "tree", "rock", "lake", "wind", "fire", "snow", "rain",
    "leaf", "bird", "fish", "frog", "wolf", "bear", "lion", "deer", "hawk", "mole", "crow", "swan",
    "goat", "lamb", "colt", "calf", "kite", "drum",
];

fn tokens_table() -> TableRagTable {
    let schema = TableSchema::new(vec![TableColumnSpec::new("token", TableColumnType::Text)]);
    let rows = TOKEN_WORDS.iter().map(|w| vec![(*w).to_string()]).collect();
    TableRagTable::new("tokens", schema, rows).expect("valid tokens table")
}

fn tokens_index() -> TableRagIndex {
    TableRagIndex::new().with_table(tokens_table())
}

// ── TableColumnType ───────────────────────────────────────────────────────

#[test]
fn table_column_type_name_all_variants() {
    assert_eq!(TableColumnType::Text.name(), "text");
    assert_eq!(TableColumnType::Number.name(), "number");
    assert_eq!(TableColumnType::Date.name(), "date");
    assert_eq!(TableColumnType::Boolean.name(), "boolean");
}

#[test]
fn table_column_type_keywords_text_is_empty() {
    assert!(TableColumnType::Text.keywords().is_empty());
}

#[test]
fn table_column_type_keywords_number_and_date_membership() {
    assert!(TableColumnType::Number.keywords().contains(&"total"));
    assert!(TableColumnType::Number.keywords().contains(&"much"));
    assert!(!TableColumnType::Number.keywords().contains(&"year"));
    assert!(TableColumnType::Date.keywords().contains(&"year"));
    assert!(TableColumnType::Date.keywords().contains(&"when"));
    assert!(TableColumnType::Boolean.keywords().contains(&"whether"));
}

// ── TableColumnSpec ───────────────────────────────────────────────────────

#[test]
fn table_column_spec_new_sets_fields() {
    let spec = TableColumnSpec::new("salary", TableColumnType::Number);
    assert_eq!(spec.name, "salary");
    assert_eq!(spec.column_type, TableColumnType::Number);
}

#[test]
fn table_column_spec_equality_and_inequality() {
    let a = TableColumnSpec::new("salary", TableColumnType::Number);
    let b = TableColumnSpec::new("salary", TableColumnType::Number);
    let c = TableColumnSpec::new("salary", TableColumnType::Text);
    assert_eq!(a, b);
    assert_ne!(a, c);
}

// ── TableSchema ───────────────────────────────────────────────────────────

#[test]
fn table_schema_new_sets_columns() {
    let schema = employees_schema();
    assert_eq!(schema.len(), 3);
    assert!(!schema.is_empty());
}

#[test]
fn table_schema_with_column_builder_appends() {
    let schema =
        TableSchema::new(vec![]).with_column(TableColumnSpec::new("id", TableColumnType::Number));
    assert_eq!(schema.len(), 1);
    assert_eq!(schema.columns[0].name, "id");
}

#[test]
fn table_schema_default_is_empty() {
    let schema = TableSchema::default();
    assert!(schema.is_empty());
    assert_eq!(schema.len(), 0);
}

#[test]
fn table_schema_column_index_case_insensitive_found() {
    let schema = employees_schema();
    assert_eq!(schema.column_index("DEPARTMENT"), Some(1));
    assert_eq!(schema.column_index("salary"), Some(2));
}

#[test]
fn table_schema_column_index_not_found_is_none() {
    let schema = employees_schema();
    assert_eq!(schema.column_index("nonexistent"), None);
}

// ── TableRagTable ─────────────────────────────────────────────────────────

#[test]
fn table_rag_table_new_ok_valid_rows() {
    let table = employees_table();
    assert_eq!(table.name, "employees");
    assert_eq!(table.row_count(), 4);
    assert_eq!(table.column_count(), 3);
}

#[test]
fn table_rag_table_new_rejects_empty_schema() {
    let err = TableRagTable::new("empty", TableSchema::default(), vec![]).unwrap_err();
    match err {
        TableRagError::EmptyColumns { table_name } => assert_eq!(table_name, "empty"),
        other => panic!("expected EmptyColumns, got {other:?}"),
    }
}

#[test]
fn table_rag_table_new_rejects_row_column_mismatch() {
    let err = TableRagTable::new(
        "bad",
        employees_schema(),
        vec![vec!["only".to_string(), "two".to_string()]],
    )
    .unwrap_err();
    match err {
        TableRagError::RowColumnMismatch {
            table_name,
            row_index,
            expected,
            found,
        } => {
            assert_eq!(table_name, "bad");
            assert_eq!(row_index, 0);
            assert_eq!(expected, 3);
            assert_eq!(found, 2);
        }
        other => panic!("expected RowColumnMismatch, got {other:?}"),
    }
}

#[test]
fn table_rag_table_row_count_and_column_count() {
    let table = products_table();
    assert_eq!(table.row_count(), 2);
    assert_eq!(table.column_count(), 3);
}

#[test]
fn table_rag_table_cell_returns_value() {
    let table = employees_table();
    assert_eq!(table.cell(0, 0), Some("Alice"));
    assert_eq!(table.cell(2, 1), Some("Engineering"));
}

#[test]
fn table_rag_table_cell_out_of_bounds_none() {
    let table = employees_table();
    assert_eq!(table.cell(99, 0), None);
    assert_eq!(table.cell(0, 99), None);
}

// ── TableRagIndex ─────────────────────────────────────────────────────────

#[test]
fn table_rag_index_new_is_empty() {
    let index = TableRagIndex::new();
    assert!(index.is_empty());
    assert_eq!(index.table_count(), 0);
}

#[test]
fn table_rag_index_with_table_builder() {
    let index = TableRagIndex::new().with_table(employees_table());
    assert!(!index.is_empty());
    assert_eq!(index.table_count(), 1);
}

#[test]
fn table_rag_index_add_table_in_place() {
    let mut index = TableRagIndex::new();
    index.add_table(employees_table());
    index.add_table(products_table());
    assert_eq!(index.table_count(), 2);
}

#[test]
fn table_rag_index_table_count_multi() {
    let index = multi_table_index();
    assert_eq!(index.table_count(), 2);
}

#[test]
fn table_rag_index_table_by_name_found() {
    let index = multi_table_index();
    let table = index.table_by_name("products").expect("products table");
    assert_eq!(table.row_count(), 2);
}

#[test]
fn table_rag_index_table_by_name_missing() {
    let index = employees_index();
    assert!(index.table_by_name("nonexistent").is_none());
}

// ── CellProbe ─────────────────────────────────────────────────────────────

#[test]
fn cell_probe_new_sets_fields() {
    let probe = CellProbe::new("department", "engineering", 1.0);
    assert_eq!(probe.column_name, "department");
    assert_eq!(probe.probe_value, "engineering");
    assert!((probe.score - 1.0).abs() < EPS);
}

#[test]
fn cell_probe_equality() {
    let a = CellProbe::new("department", "engineering", 1.0);
    let b = CellProbe::new("department", "engineering", 1.0);
    let c = CellProbe::new("department", "sales", 1.0);
    assert_eq!(a, b);
    assert_ne!(a, c);
}

// ── TableRagSubTable ──────────────────────────────────────────────────────

#[test]
fn sub_table_is_empty_no_rows() {
    let sub_table = TableRagSubTable {
        table_name: "t".to_string(),
        columns: vec![TableColumnSpec::new("c", TableColumnType::Text)],
        column_indices: vec![0],
        rows: vec![],
        row_indices: vec![],
    };
    assert!(sub_table.is_empty());
}

#[test]
fn sub_table_is_empty_no_columns() {
    let sub_table = TableRagSubTable {
        table_name: "t".to_string(),
        columns: vec![],
        column_indices: vec![],
        rows: vec![vec!["x".to_string()]],
        row_indices: vec![0],
    };
    assert!(sub_table.is_empty());
}

#[test]
fn sub_table_row_and_column_count() {
    let sub_table = TableRagSubTable {
        table_name: "t".to_string(),
        columns: vec![
            TableColumnSpec::new("a", TableColumnType::Text),
            TableColumnSpec::new("b", TableColumnType::Text),
        ],
        column_indices: vec![0, 1],
        rows: vec![vec!["x".to_string(), "y".to_string()]],
        row_indices: vec![3],
    };
    assert_eq!(sub_table.column_count(), 2);
    assert_eq!(sub_table.row_count(), 1);
    assert!(!sub_table.is_empty());
}

#[test]
fn sub_table_to_encoded_text_header_and_rows() {
    let sub_table = TableRagSubTable {
        table_name: "employees".to_string(),
        columns: vec![TableColumnSpec::new("department", TableColumnType::Text)],
        column_indices: vec![1],
        rows: vec![vec!["Engineering".to_string()], vec!["Sales".to_string()]],
        row_indices: vec![0, 1],
    };
    let text = sub_table.to_encoded_text();
    let lines: Vec<&str> = text.lines().collect();
    assert_eq!(lines[0], "Table: employees");
    assert_eq!(lines[1], "Columns: department");
    assert_eq!(lines[2], "Row 0: Engineering");
    assert_eq!(lines[3], "Row 1: Sales");
    assert_eq!(lines.len(), 4);
}

#[test]
fn sub_table_to_encoded_text_empty_rows_still_shows_header() {
    let sub_table = TableRagSubTable {
        table_name: "employees".to_string(),
        columns: vec![TableColumnSpec::new("department", TableColumnType::Text)],
        column_indices: vec![1],
        rows: vec![],
        row_indices: vec![],
    };
    let text = sub_table.to_encoded_text();
    let lines: Vec<&str> = text.lines().collect();
    assert_eq!(lines, vec!["Table: employees", "Columns: department"]);
}

#[test]
fn sub_table_to_encoded_text_multi_column_join() {
    let sub_table = TableRagSubTable {
        table_name: "employees".to_string(),
        columns: vec![
            TableColumnSpec::new("department", TableColumnType::Text),
            TableColumnSpec::new("salary", TableColumnType::Number),
        ],
        column_indices: vec![1, 2],
        rows: vec![vec!["Engineering".to_string(), "95000".to_string()]],
        row_indices: vec![0],
    };
    let text = sub_table.to_encoded_text();
    assert!(text.contains("Columns: department | salary"));
    assert!(text.contains("Row 0: Engineering | 95000"));
}

// ── TableRagResult ────────────────────────────────────────────────────────

#[test]
fn table_rag_result_is_empty_delegates() {
    let empty_sub = TableRagSubTable {
        table_name: "t".to_string(),
        columns: vec![TableColumnSpec::new("c", TableColumnType::Text)],
        column_indices: vec![0],
        rows: vec![],
        row_indices: vec![],
    };
    let result = TableRagResult {
        query: "q".to_string(),
        encoded_text: empty_sub.to_encoded_text(),
        sub_table: empty_sub,
        column_scores: vec![],
        row_scores: vec![],
        probes: vec![],
        capped_columns: vec![],
    };
    assert!(result.is_empty());
}

// ── TableRagConfig ────────────────────────────────────────────────────────

#[test]
fn table_rag_config_default_values() {
    let config = TableRagConfig::default();
    assert_eq!(config.top_k_columns, 6);
    assert_eq!(config.top_k_rows, 10);
    assert_eq!(config.max_distinct_values_per_column, 64);
    assert!((config.lexical_weight - 0.6).abs() < EPS);
    assert!((config.embedding_weight - 0.4).abs() < EPS);
    assert!((config.min_column_score - 0.05).abs() < EPS);
    assert!((config.min_cell_score - 0.05).abs() < EPS);
    assert_eq!(config.pseudo_embedding_dim, 64);
}

#[test]
fn table_rag_config_new_equals_default() {
    assert_eq!(TableRagConfig::new(), TableRagConfig::default());
}

#[test]
fn table_rag_config_builder_chain_sets_all_fields() {
    let config = TableRagConfig::new()
        .with_top_k_columns(3)
        .with_top_k_rows(5)
        .with_max_distinct_values_per_column(10)
        .with_lexical_weight(0.7)
        .with_embedding_weight(0.3)
        .with_min_column_score(0.1)
        .with_min_cell_score(0.2)
        .with_pseudo_embedding_dim(32);
    assert_eq!(config.top_k_columns, 3);
    assert_eq!(config.top_k_rows, 5);
    assert_eq!(config.max_distinct_values_per_column, 10);
    assert!((config.lexical_weight - 0.7).abs() < EPS);
    assert!((config.embedding_weight - 0.3).abs() < EPS);
    assert!((config.min_column_score - 0.1).abs() < EPS);
    assert!((config.min_cell_score - 0.2).abs() < EPS);
    assert_eq!(config.pseudo_embedding_dim, 32);
}

// ── TableRagError ─────────────────────────────────────────────────────────

#[test]
fn table_rag_error_display_messages() {
    assert_eq!(
        TableRagError::EmptyQuery.to_string(),
        "query must not be empty"
    );
    assert_eq!(
        TableRagError::EmptyIndex.to_string(),
        "table-rag index contains no tables"
    );
    assert_eq!(
        TableRagError::EmptyColumns {
            table_name: "t".to_string()
        }
        .to_string(),
        "table schema for 't' declares no columns"
    );
    assert_eq!(
        TableRagError::RowColumnMismatch {
            table_name: "t".to_string(),
            row_index: 2,
            expected: 3,
            found: 1,
        }
        .to_string(),
        "table 't' row 2 has 1 cells, expected 3 (one per schema column)"
    );
    assert_eq!(
        TableRagError::NoRelevantColumns.to_string(),
        "no column scored above the relevance threshold for this query"
    );
}

#[test]
fn table_rag_error_equality() {
    assert_eq!(TableRagError::EmptyQuery, TableRagError::EmptyQuery);
    assert_ne!(TableRagError::EmptyQuery, TableRagError::EmptyIndex);
}

// ── Engine: validation / empty paths ──────────────────────────────────────

#[test]
fn query_empty_string_is_error() {
    let engine = TableRagEngine::with_default_config(employees_index());
    assert_eq!(engine.query("").unwrap_err(), TableRagError::EmptyQuery);
}

#[test]
fn query_whitespace_only_is_error() {
    let engine = TableRagEngine::with_default_config(employees_index());
    assert_eq!(
        engine.query("   \t\n").unwrap_err(),
        TableRagError::EmptyQuery
    );
}

#[test]
fn query_empty_index_is_error() {
    let engine = TableRagEngine::with_default_config(TableRagIndex::new());
    assert_eq!(
        engine.query("anything").unwrap_err(),
        TableRagError::EmptyIndex
    );
}

#[test]
fn query_min_column_score_unreachable_is_no_relevant_columns_error() {
    // Max achievable score with default weights (0.6 + 0.4 = 1.0 total) is
    // 1.0; a threshold above that can never be cleared, by any query.
    let config = TableRagConfig::default().with_min_column_score(1.5);
    let engine = TableRagEngine::new(employees_index(), config);
    assert_eq!(
        engine.query("department").unwrap_err(),
        TableRagError::NoRelevantColumns
    );
}

// ── Engine: schema retrieval ──────────────────────────────────────────────

#[test]
fn schema_retrieval_selects_relevant_drops_irrelevant_isolated_lexical() {
    // Isolated lexical-only weights make column scoring hash-independent:
    // "department" lexically matches only the "department" column name.
    let config = TableRagConfig::default()
        .with_lexical_weight(1.0)
        .with_embedding_weight(0.0);
    let engine = TableRagEngine::new(employees_index(), config);
    let result = engine.query("department").expect("should succeed");

    assert_eq!(result.sub_table.columns.len(), 1);
    assert_eq!(result.sub_table.columns[0].name, "department");
    assert_eq!(result.column_scores.len(), 1);
    assert!((result.column_scores[0].1 - 1.0).abs() < EPS);
}

#[test]
fn schema_retrieval_type_keyword_boosts_number_column() {
    // "total" has zero lexical overlap with any column name; the only
    // possible signal is the Number column-type keyword bonus.
    let config = TableRagConfig::default()
        .with_lexical_weight(1.0)
        .with_embedding_weight(0.0);
    let engine = TableRagEngine::new(employees_index(), config);
    let result = engine.query("total").expect("should succeed");

    assert_eq!(result.sub_table.columns.len(), 1);
    assert_eq!(result.sub_table.columns[0].name, "salary");
    assert!((result.column_scores[0].1 - 0.25).abs() < EPS);
}

#[test]
fn schema_retrieval_embedding_only_matches_identical_column_name() {
    // Isolated embedding-only weights: querying the exact column name text
    // guarantees cosine similarity of 1.0 against that column (identical
    // pseudo-embedding vectors), independent of hashing specifics.
    let config = TableRagConfig::default()
        .with_lexical_weight(0.0)
        .with_embedding_weight(1.0);
    let engine = TableRagEngine::new(employees_index(), config);
    let result = engine.query("salary").expect("should succeed");

    assert_eq!(result.sub_table.columns.len(), 1);
    assert_eq!(result.sub_table.columns[0].name, "salary");
    assert!((result.column_scores[0].1 - 1.0).abs() < EPS);
}

#[test]
fn schema_retrieval_min_column_score_filters_weak_column() {
    let query = "department total";
    let permissive = TableRagEngine::new(
        employees_index(),
        TableRagConfig::default()
            .with_lexical_weight(1.0)
            .with_embedding_weight(0.0),
    );
    let strict = TableRagEngine::new(
        employees_index(),
        TableRagConfig::default()
            .with_lexical_weight(1.0)
            .with_embedding_weight(0.0)
            .with_min_column_score(0.5),
    );

    let permissive_result = permissive.query(query).expect("should succeed");
    let mut permissive_names: Vec<&str> = permissive_result
        .sub_table
        .columns
        .iter()
        .map(|c| c.name.as_str())
        .collect();
    permissive_names.sort_unstable();
    assert_eq!(permissive_names, vec!["department", "salary"]);

    let strict_result = strict.query(query).expect("should succeed");
    assert_eq!(strict_result.sub_table.columns.len(), 1);
    assert_eq!(strict_result.sub_table.columns[0].name, "department");
}

#[test]
fn schema_retrieval_top_k_columns_truncates_ties() {
    // All three column names appear verbatim in the query, in distinct
    // pseudo-embedding buckets, so by symmetry all three tie exactly.
    let config = TableRagConfig::default().with_top_k_columns(1);
    let engine = TableRagEngine::new(employees_index(), config);
    let result = engine
        .query("name department salary")
        .expect("should succeed");

    assert_eq!(result.sub_table.columns.len(), 1);
    // Ties are broken by ascending original column index; "name" is column 0.
    assert_eq!(result.sub_table.columns[0].name, "name");
}

// ── Engine: cell retrieval ────────────────────────────────────────────────

#[test]
fn cell_retrieval_selects_matching_rows_via_probes() {
    let engine = TableRagEngine::with_default_config(employees_index());
    let result = engine
        .query("Which employees work in the Engineering department?")
        .expect("should succeed");

    assert_eq!(result.sub_table.columns.len(), 1);
    assert_eq!(result.sub_table.columns[0].name, "department");
    assert_eq!(result.sub_table.row_indices, vec![0, 2]);
    assert_eq!(
        result.sub_table.rows,
        vec![
            vec!["Engineering".to_string()],
            vec!["Engineering".to_string()]
        ]
    );
}

#[test]
fn cell_retrieval_excludes_non_matching_rows() {
    let engine = TableRagEngine::with_default_config(employees_index());
    let result = engine
        .query("Which employees work in the Engineering department?")
        .expect("should succeed");

    assert!(!result.sub_table.row_indices.contains(&1)); // Bob / Sales
    assert!(!result.sub_table.row_indices.contains(&3)); // Dave / Marketing
}

#[test]
fn cell_retrieval_no_matching_rows_returns_empty_subtable_ok() {
    // "department" is clearly schema-relevant, but nothing in the query
    // textually matches any of the column's actual values (Engineering /
    // Sales / Marketing) — a legitimate "no cell match" outcome, not a
    // failure.
    let engine = TableRagEngine::with_default_config(employees_index());
    let result = engine
        .query("What department is Carol in?")
        .expect("should succeed, not error");

    assert_eq!(result.sub_table.columns.len(), 1);
    assert_eq!(result.sub_table.columns[0].name, "department");
    assert!(result.sub_table.rows.is_empty());
    assert!(result.sub_table.is_empty());
    assert!(result.is_empty());
}

#[test]
fn cell_retrieval_top_k_rows_truncates() {
    let config = TableRagConfig::default().with_top_k_rows(1);
    let engine = TableRagEngine::new(employees_index(), config);
    let result = engine
        .query("Which employees work in the Engineering department?")
        .expect("should succeed");

    assert_eq!(result.sub_table.row_count(), 1);
    // Ties are broken by ascending original row index; Alice is row 0.
    assert_eq!(result.sub_table.row_indices, vec![0]);
}

#[test]
fn cell_retrieval_min_cell_score_filters_all_probes_ok_empty() {
    // 1.0 is the maximum possible probe score; a threshold above that
    // filters every probe, but the column remains relevant.
    let config = TableRagConfig::default().with_min_cell_score(1.5);
    let engine = TableRagEngine::new(employees_index(), config);
    let result = engine
        .query("Which employees work in the Engineering department?")
        .expect("should succeed, not error");

    assert_eq!(result.sub_table.columns.len(), 1);
    assert!(result.sub_table.rows.is_empty());
    assert!(result.probes.is_empty());
}

// ── Engine: probe provenance ──────────────────────────────────────────────

#[test]
fn result_probes_recorded_for_relevant_column_only() {
    let engine = TableRagEngine::with_default_config(employees_index());
    let result = engine
        .query("Which employees work in the Engineering department?")
        .expect("should succeed");

    assert!(!result.probes.is_empty());
    assert!(result.probes.iter().all(|p| p.column_name == "department"));
}

#[test]
fn result_probes_sorted_best_first() {
    let engine = TableRagEngine::with_default_config(employees_index());
    let result = engine
        .query("Which employees work in the Engineering department?")
        .expect("should succeed");

    for window in result.probes.windows(2) {
        assert!(window[0].score >= window[1].score);
    }
    // The unigram "engineering" is an exact match (score 1.0) and must rank
    // first among the department probes.
    assert_eq!(result.probes[0].probe_value, "engineering");
    assert!((result.probes[0].score - 1.0).abs() < EPS);
}

// ── Engine: sub-table assembly / provenance ───────────────────────────────

#[test]
fn sub_table_intersection_shape_matches_selected_columns_rows() {
    let engine = TableRagEngine::with_default_config(employees_index());
    let result = engine
        .query("How much do employees in the Engineering department earn?")
        .expect("should succeed");

    assert_eq!(result.sub_table.columns.len(), 2);
    assert_eq!(result.sub_table.column_indices.len(), 2);
    for row in &result.sub_table.rows {
        assert_eq!(row.len(), result.sub_table.columns.len());
    }
    assert_eq!(
        result.sub_table.rows.len(),
        result.sub_table.row_indices.len()
    );
}

#[test]
fn sub_table_provenance_table_name_and_indices() {
    let engine = TableRagEngine::with_default_config(employees_index());
    let result = engine
        .query("Which employees work in the Engineering department?")
        .expect("should succeed");

    assert_eq!(result.sub_table.table_name, "employees");
    assert_eq!(result.sub_table.column_indices, vec![1]); // "department" is schema index 1
    assert_eq!(result.sub_table.row_indices, vec![0, 2]);
}

#[test]
fn sub_table_column_values_align_with_ranked_columns() {
    let engine = TableRagEngine::with_default_config(employees_index());
    let result = engine
        .query("How much do employees in the Engineering department earn?")
        .expect("should succeed");

    // department (higher score) ranks before salary.
    assert_eq!(result.sub_table.columns[0].name, "department");
    assert_eq!(result.sub_table.columns[1].name, "salary");

    assert_eq!(result.sub_table.row_indices, vec![0, 2, 3]);
    assert_eq!(
        result.sub_table.rows,
        vec![
            vec!["Engineering".to_string(), "95000".to_string()],
            vec!["Engineering".to_string(), "120000".to_string()],
            vec!["Marketing".to_string(), "72000".to_string()],
        ]
    );
}

// ── Engine: multi-table selection ─────────────────────────────────────────

#[test]
fn multi_table_index_picks_products_table() {
    let engine = TableRagEngine::with_default_config(multi_table_index());
    let result = engine
        .query("What is the product name of the Widget?")
        .expect("should succeed");

    assert_eq!(result.sub_table.table_name, "products");
    assert_eq!(result.sub_table.row_indices, vec![0]);
    assert_eq!(result.sub_table.rows[0][0], "Widget");
}

#[test]
fn multi_table_index_picks_employees_table_with_empty_rows() {
    let engine = TableRagEngine::with_default_config(multi_table_index());
    let result = engine
        .query("What department is Carol in?")
        .expect("should succeed");

    assert_eq!(result.sub_table.table_name, "employees");
    assert_eq!(result.sub_table.columns[0].name, "department");
    assert!(result.sub_table.rows.is_empty());
}

// ── Engine: distinct-value capping ────────────────────────────────────────

#[test]
fn distinct_value_capping_within_cap_is_retrievable() {
    let config = TableRagConfig::default().with_max_distinct_values_per_column(5);
    let engine = TableRagEngine::new(tokens_index(), config);
    // "cat" is row 0 — among the first 5 distinct values encountered, so it
    // stays inside the capped dictionary.
    let result = engine.query("What is token cat?").expect("should succeed");

    assert_eq!(result.sub_table.row_indices, vec![0]);
    assert_eq!(result.capped_columns, vec!["token".to_string()]);
}

#[test]
fn distinct_value_capping_beyond_cap_is_unreachable() {
    let config = TableRagConfig::default().with_max_distinct_values_per_column(5);
    let engine = TableRagEngine::new(tokens_index(), config);
    // "kite" is row 28 — far beyond the first 5 distinct values, so the cap
    // makes it genuinely unreachable through probe matching.
    let result = engine.query("What is token kite?").expect("should succeed");

    assert!(!result.sub_table.row_indices.contains(&28));
    assert_eq!(result.capped_columns, vec!["token".to_string()]);
}

#[test]
fn distinct_value_capping_control_uncapped_finds_value() {
    // Same query as the "beyond cap" test above, but with a cap comfortably
    // larger than the column's 30 distinct values: this proves the previous
    // test's empty result was genuinely caused by capping, not some other
    // defect — without the cap, row 28 ("kite") is found directly.
    let config = TableRagConfig::default().with_max_distinct_values_per_column(64);
    let engine = TableRagEngine::new(tokens_index(), config);
    let result = engine.query("What is token kite?").expect("should succeed");

    assert!(result.sub_table.row_indices.contains(&28));
    assert!(result.capped_columns.is_empty());
}

#[test]
fn distinct_value_capping_not_reported_when_column_smaller_than_cap() {
    // employees.department has only 3 distinct values, well under the
    // default cap of 64: no capping occurs.
    let engine = TableRagEngine::with_default_config(employees_index());
    let result = engine
        .query("Which employees work in the Engineering department?")
        .expect("should succeed");
    assert!(result.capped_columns.is_empty());
}

// ── Engine: determinism ───────────────────────────────────────────────────

#[test]
fn determinism_repeated_queries_identical() {
    let engine = TableRagEngine::with_default_config(multi_table_index());
    let first = engine
        .query("Which employees work in the Engineering department?")
        .expect("should succeed");
    let second = engine
        .query("Which employees work in the Engineering department?")
        .expect("should succeed");
    assert_eq!(first, second);
}

#[test]
fn determinism_across_fresh_engines_identical() {
    let engine_a = TableRagEngine::with_default_config(multi_table_index());
    let engine_b = TableRagEngine::with_default_config(multi_table_index());
    let result_a = engine_a
        .query("What is the product name of the Widget?")
        .expect("should succeed");
    let result_b = engine_b
        .query("What is the product name of the Widget?")
        .expect("should succeed");
    assert_eq!(result_a, result_b);
}

// ── Engine: ranking correctness ────────────────────────────────────────────

#[test]
fn ranking_correctness_exact_matches_outrank_fuzzy_match() {
    let engine = TableRagEngine::with_default_config(employees_index());
    let result = engine
        .query("How much do employees in the Engineering department earn?")
        .expect("should succeed");

    // Rows 0 and 2 (Engineering, exact-tier department match) must outrank
    // row 3 (Marketing, only an incidental fuzzy match).
    assert_eq!(result.row_scores.len(), 3);
    assert_eq!(result.row_scores[0].0, 0);
    assert_eq!(result.row_scores[1].0, 2);
    assert_eq!(result.row_scores[2].0, 3);
    assert!(result.row_scores[0].1 > result.row_scores[2].1);
    assert!(result.row_scores[1].1 > result.row_scores[2].1);
    assert!((result.row_scores[0].1 - 1.0).abs() < EPS);
    assert!((result.row_scores[1].1 - 1.0).abs() < EPS);
}

#[test]
fn ranking_correctness_columns_ranked_best_first() {
    let engine = TableRagEngine::with_default_config(employees_index());
    let result = engine
        .query("How much do employees in the Engineering department earn?")
        .expect("should succeed");

    assert_eq!(result.column_scores.len(), 2);
    assert_eq!(result.column_scores[0].0, "department");
    assert_eq!(result.column_scores[1].0, "salary");
    assert!(result.column_scores[0].1 > result.column_scores[1].1);
}

#[test]
fn ranking_correctness_rows_positionally_aligned_with_scores() {
    let engine = TableRagEngine::with_default_config(employees_index());
    let result = engine
        .query("How much do employees in the Engineering department earn?")
        .expect("should succeed");

    assert_eq!(result.row_scores.len(), result.sub_table.row_indices.len());
    for (score_entry, &row_index) in result
        .row_scores
        .iter()
        .zip(result.sub_table.row_indices.iter())
    {
        assert_eq!(score_entry.0, row_index);
    }
}

// ── Engine: encoded text ──────────────────────────────────────────────────

#[test]
fn encoded_text_round_trip_shape_matches_subtable() {
    let engine = TableRagEngine::with_default_config(employees_index());
    let result = engine
        .query("How much do employees in the Engineering department earn?")
        .expect("should succeed");

    // Round-trip the encoded text back into (columns, rows) shape via a
    // minimal parser mirroring `TableRagSubTable::to_encoded_text`'s format,
    // and check it matches the structured `sub_table` exactly.
    let lines: Vec<&str> = result.encoded_text.lines().collect();
    assert!(lines[0].starts_with("Table: "));
    assert_eq!(&lines[0]["Table: ".len()..], result.sub_table.table_name);

    let header = lines[1].strip_prefix("Columns: ").expect("Columns header");
    let parsed_columns: Vec<&str> = header.split(" | ").collect();
    assert_eq!(parsed_columns.len(), result.sub_table.columns.len());
    for (parsed, column) in parsed_columns.iter().zip(result.sub_table.columns.iter()) {
        assert_eq!(*parsed, column.name);
    }

    let row_lines = &lines[2..];
    assert_eq!(row_lines.len(), result.sub_table.rows.len());
    for (line, row) in row_lines.iter().zip(result.sub_table.rows.iter()) {
        let after_prefix = line.split_once(": ").expect("Row prefix").1;
        let parsed_cells: Vec<&str> = after_prefix.split(" | ").collect();
        assert_eq!(parsed_cells.len(), row.len());
        for (parsed, cell) in parsed_cells.iter().zip(row.iter()) {
            assert_eq!(parsed, cell);
        }
    }
}

#[test]
fn encoded_text_contains_expected_content() {
    let engine = TableRagEngine::with_default_config(employees_index());
    let result = engine
        .query("Which employees work in the Engineering department?")
        .expect("should succeed");

    assert!(result.encoded_text.contains("Table: employees"));
    assert!(result.encoded_text.contains("Columns: department"));
    assert!(result.encoded_text.contains("Engineering"));
}

#[test]
fn encoded_text_empty_subtable_has_header_no_rows() {
    let engine = TableRagEngine::with_default_config(employees_index());
    let result = engine
        .query("What department is Carol in?")
        .expect("should succeed");

    let lines: Vec<&str> = result.encoded_text.lines().collect();
    assert_eq!(lines.len(), 2); // Table + Columns header, zero Row lines
    assert!(!result.encoded_text.contains("Row "));
}

// ── Engine: constructors ───────────────────────────────────────────────────

#[test]
fn engine_new_and_with_default_config_use_supplied_index() {
    let engine = TableRagEngine::new(employees_index(), TableRagConfig::default());
    assert_eq!(engine.index.table_count(), 1);

    let engine2 = TableRagEngine::with_default_config(multi_table_index());
    assert_eq!(engine2.index.table_count(), 2);
    assert_eq!(engine2.config, TableRagConfig::default());
}
