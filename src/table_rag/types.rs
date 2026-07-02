//! Types for the `table_rag` module.
//!
//! Defines the tabular data model (`TableSchema`/`TableColumnSpec`/
//! `TableRagTable`/`TableRagIndex`), the two-stage retrieval vocabulary
//! (`CellProbe`), the retrieval output (`TableRagSubTable`/`TableRagResult`),
//! the tunable [`TableRagConfig`], and the `thiserror` error enum
//! [`TableRagError`].

use thiserror::Error;

// ── TableColumnType ────────────────────────────────────────────────────────

/// The semantic type of a [`TableRagTable`] column.
///
/// Drives two things during schema retrieval: a small, type-specific
/// keyword vocabulary (see [`TableColumnType::keywords`]) that nudges a
/// column's relevance score up when the query contains a type-indicative
/// word (e.g. `"how many"` hints at a [`TableColumnType::Number`] column),
/// and the human-readable label returned by [`TableColumnType::name`].
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum TableColumnType {
    /// Free text (names, descriptions, categories, ...).
    Text,
    /// A numeric quantity, compared/matched as text but semantically
    /// numeric (price, count, age, ...).
    Number,
    /// A date, year, or other point in time.
    Date,
    /// A `true`/`false`-style flag.
    Boolean,
}

impl TableColumnType {
    /// Human-readable, lowercase name of this column type.
    #[must_use]
    pub fn name(&self) -> &'static str {
        match self {
            Self::Text => "text",
            Self::Number => "number",
            Self::Date => "date",
            Self::Boolean => "boolean",
        }
    }

    /// Query keywords that hint a natural-language question is asking about
    /// a column of this type.
    ///
    /// Used by schema retrieval as a small additive bonus: when the query
    /// contains one of these tokens, columns of this type score higher,
    /// independent of how well the query text lexically or
    /// embedding-wise resembles the column's *name*. [`TableColumnType::Text`]
    /// has no distinguishing keywords (free text is the default, unmarked
    /// category), and [`TableColumnType::Boolean`]'s list is deliberately
    /// short and specific to avoid matching on common words.
    #[must_use]
    pub fn keywords(&self) -> &'static [&'static str] {
        match self {
            Self::Text => &[],
            Self::Number => &[
                "number", "count", "total", "sum", "average", "price", "cost", "amount",
                "quantity", "many", "much",
            ],
            Self::Date => &["date", "year", "when", "time", "month", "day"],
            Self::Boolean => &["whether", "true", "false"],
        }
    }
}

// ── TableColumnSpec ───────────────────────────────────────────────────────

/// The specification of a single column in a [`TableSchema`]: its name and
/// semantic [`TableColumnType`].
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct TableColumnSpec {
    /// The column's name, as it appears in the source table.
    pub name: String,
    /// The column's semantic type.
    pub column_type: TableColumnType,
}

impl TableColumnSpec {
    /// Create a new column specification.
    #[must_use]
    pub fn new(name: impl Into<String>, column_type: TableColumnType) -> Self {
        Self {
            name: name.into(),
            column_type,
        }
    }
}

// ── TableSchema ───────────────────────────────────────────────────────────

/// The ordered list of columns describing a [`TableRagTable`]'s shape.
///
/// Every row of the owning table has exactly [`TableSchema::len`] cells,
/// positionally aligned with [`TableSchema::columns`].
#[derive(Debug, Clone, Default, PartialEq, Eq)]
pub struct TableSchema {
    /// The columns, in source order.
    pub columns: Vec<TableColumnSpec>,
}

impl TableSchema {
    /// Create a schema from an ordered list of columns.
    #[must_use]
    pub fn new(columns: Vec<TableColumnSpec>) -> Self {
        Self { columns }
    }

    /// Append a column (builder).
    #[must_use]
    pub fn with_column(mut self, column: TableColumnSpec) -> Self {
        self.columns.push(column);
        self
    }

    /// Find the position of the column named `name` (case-insensitive).
    #[must_use]
    pub fn column_index(&self, name: &str) -> Option<usize> {
        self.columns
            .iter()
            .position(|c| c.name.eq_ignore_ascii_case(name))
    }

    /// The number of columns in the schema.
    #[must_use]
    pub fn len(&self) -> usize {
        self.columns.len()
    }

    /// Returns `true` when the schema declares no columns.
    #[must_use]
    pub fn is_empty(&self) -> bool {
        self.columns.is_empty()
    }
}

// ── TableRagTable ─────────────────────────────────────────────────────────

/// A single table indexed by `table_rag`: a name, a [`TableSchema`], and its
/// rows of cell values.
///
/// Every row is a `Vec<String>` whose length equals `schema.len()`; cell
/// values are stored verbatim (as text) regardless of the declared
/// [`TableColumnType`] — types only steer *retrieval scoring*, they are not
/// a parsing/validation constraint on cell content.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct TableRagTable {
    /// The table's name (used for provenance and multi-table selection).
    pub name: String,
    /// The table's column schema.
    pub schema: TableSchema,
    /// The table's rows; `rows[r][c]` is the cell at row `r`, column `c`.
    pub rows: Vec<Vec<String>>,
}

impl TableRagTable {
    /// Construct a new table, validating that every row has exactly one
    /// cell per schema column.
    ///
    /// # Errors
    ///
    /// - [`TableRagError::EmptyColumns`] when `schema` declares no columns.
    /// - [`TableRagError::RowColumnMismatch`] when some row's length does not
    ///   equal `schema.len()`.
    pub fn new(
        name: impl Into<String>,
        schema: TableSchema,
        rows: Vec<Vec<String>>,
    ) -> Result<Self, TableRagError> {
        let name = name.into();
        if schema.is_empty() {
            return Err(TableRagError::EmptyColumns { table_name: name });
        }
        let expected = schema.len();
        for (row_index, row) in rows.iter().enumerate() {
            if row.len() != expected {
                return Err(TableRagError::RowColumnMismatch {
                    table_name: name,
                    row_index,
                    expected,
                    found: row.len(),
                });
            }
        }
        Ok(Self { name, schema, rows })
    }

    /// The number of rows in the table.
    #[must_use]
    pub fn row_count(&self) -> usize {
        self.rows.len()
    }

    /// The number of columns in the table.
    #[must_use]
    pub fn column_count(&self) -> usize {
        self.schema.len()
    }

    /// The cell at `(row, col)`, or `None` when either index is out of
    /// bounds.
    #[must_use]
    pub fn cell(&self, row: usize, col: usize) -> Option<&str> {
        self.rows
            .get(row)
            .and_then(|r| r.get(col))
            .map(String::as_str)
    }
}

// ── TableRagIndex ─────────────────────────────────────────────────────────

/// A collection of [`TableRagTable`]s that [`crate::table_rag::TableRagEngine`]
/// retrieves over.
#[derive(Debug, Clone, Default, PartialEq, Eq)]
pub struct TableRagIndex {
    /// The indexed tables, in insertion order.
    pub tables: Vec<TableRagTable>,
}

impl TableRagIndex {
    /// Create an empty index.
    #[must_use]
    pub fn new() -> Self {
        Self::default()
    }

    /// Append a table to the index (builder).
    #[must_use]
    pub fn with_table(mut self, table: TableRagTable) -> Self {
        self.tables.push(table);
        self
    }

    /// Append a table to the index in place.
    pub fn add_table(&mut self, table: TableRagTable) {
        self.tables.push(table);
    }

    /// Returns `true` when the index contains no tables.
    #[must_use]
    pub fn is_empty(&self) -> bool {
        self.tables.is_empty()
    }

    /// The number of tables in the index.
    #[must_use]
    pub fn table_count(&self) -> usize {
        self.tables.len()
    }

    /// Find a table by name (exact match).
    #[must_use]
    pub fn table_by_name(&self, name: &str) -> Option<&TableRagTable> {
        self.tables.iter().find(|t| t.name == name)
    }
}

// ── CellProbe ─────────────────────────────────────────────────────────────

/// A candidate `(column, cell-value)` pair generated by expanding a query
/// during cell retrieval, together with the score of its best match against
/// the column's actual (distinct-value-encoded) data.
///
/// Only probes that cleared [`TableRagConfig::min_cell_score`] against at
/// least one of a column's tracked distinct values are kept — a
/// [`CellProbe`] is evidence that `probe_value` plausibly denotes a real
/// cell in `column_name`, not merely a candidate that was tried.
#[derive(Debug, Clone, PartialEq)]
pub struct CellProbe {
    /// The name of the column this probe was matched against.
    pub column_name: String,
    /// The candidate value (a query token, bigram, or the full query)
    /// that was matched.
    pub probe_value: String,
    /// The best match score achieved against the column's tracked distinct
    /// values, in (roughly) `[0.0, 1.0]`.
    pub score: f32,
}

impl CellProbe {
    /// Create a new cell probe.
    #[must_use]
    pub fn new(column_name: impl Into<String>, probe_value: impl Into<String>, score: f32) -> Self {
        Self {
            column_name: column_name.into(),
            probe_value: probe_value.into(),
            score,
        }
    }
}

// ── TableRagSubTable ──────────────────────────────────────────────────────

/// The compact sub-table produced by intersecting schema retrieval's
/// relevant columns with cell retrieval's relevant rows.
///
/// Carries full provenance back to the source [`TableRagTable`]:
/// [`TableRagSubTable::column_indices`] and [`TableRagSubTable::row_indices`]
/// record the *original* column/row positions each entry of
/// [`TableRagSubTable::columns`]/[`TableRagSubTable::rows`] came from, in the
/// same (relevance-ranked) order.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct TableRagSubTable {
    /// The name of the source table.
    pub table_name: String,
    /// The selected columns, ranked most-relevant-first.
    pub columns: Vec<TableColumnSpec>,
    /// The original schema index of each entry of `columns`, positionally
    /// aligned with it.
    pub column_indices: Vec<usize>,
    /// The selected rows, ranked most-relevant-first. Each row has exactly
    /// `columns.len()` cells, positionally aligned with `columns`.
    pub rows: Vec<Vec<String>>,
    /// The original row index of each entry of `rows`, positionally aligned
    /// with it.
    pub row_indices: Vec<usize>,
}

impl TableRagSubTable {
    /// Returns `true` when there are no columns or no rows (nothing useful
    /// to show downstream).
    #[must_use]
    pub fn is_empty(&self) -> bool {
        self.columns.is_empty() || self.rows.is_empty()
    }

    /// The number of rows in the sub-table.
    #[must_use]
    pub fn row_count(&self) -> usize {
        self.rows.len()
    }

    /// The number of columns in the sub-table.
    #[must_use]
    pub fn column_count(&self) -> usize {
        self.columns.len()
    }

    /// Render this sub-table as a compact, deterministic text block suitable
    /// as downstream-RAG context: a `Table:` line, a `Columns:` header
    /// (`" | "`-separated), then one `Row <original index>: ...` line per
    /// row (also `" | "`-separated), in ranked order.
    #[must_use]
    pub fn to_encoded_text(&self) -> String {
        use std::fmt::Write as _;

        let mut out = String::new();
        // `write!`/`writeln!` on a `String` are infallible (the error type
        // is `fmt::Error`, which a `String` sink never produces), so the
        // `Result` is intentionally discarded rather than propagated.
        let _ = writeln!(out, "Table: {}", self.table_name);
        let header = self
            .columns
            .iter()
            .map(|c| c.name.as_str())
            .collect::<Vec<_>>()
            .join(" | ");
        let _ = writeln!(out, "Columns: {header}");
        for (row_index, row) in self.row_indices.iter().zip(self.rows.iter()) {
            let _ = writeln!(out, "Row {row_index}: {}", row.join(" | "));
        }
        out
    }
}

// ── TableRagResult ────────────────────────────────────────────────────────

/// The complete output of a [`crate::table_rag::TableRagEngine::query`] call.
#[derive(Debug, Clone, PartialEq)]
pub struct TableRagResult {
    /// The (trimmed) query text that produced this result.
    pub query: String,
    /// The assembled sub-table: relevant columns × relevant rows.
    pub sub_table: TableRagSubTable,
    /// [`TableRagSubTable::to_encoded_text`] of `sub_table`, provided
    /// directly for downstream RAG consumption.
    pub encoded_text: String,
    /// `(column name, score)` pairs for every column selected during schema
    /// retrieval, ranked most-relevant-first — positionally aligned with
    /// `sub_table.columns`.
    pub column_scores: Vec<(String, f32)>,
    /// `(original row index, score)` pairs for every row selected during
    /// cell retrieval, ranked most-relevant-first — positionally aligned
    /// with `sub_table.rows`.
    pub row_scores: Vec<(usize, f32)>,
    /// Every [`CellProbe`] that matched during cell retrieval, ranked
    /// best-first.
    pub probes: Vec<CellProbe>,
    /// Names of columns whose distinct-value dictionary was truncated at
    /// [`TableRagConfig::max_distinct_values_per_column`] while answering
    /// this query — an honest signal that some of that column's rarer
    /// values were not considered during cell retrieval.
    pub capped_columns: Vec<String>,
}

impl TableRagResult {
    /// Returns `true` when the assembled [`TableRagSubTable`] is empty (see
    /// [`TableRagSubTable::is_empty`]).
    #[must_use]
    pub fn is_empty(&self) -> bool {
        self.sub_table.is_empty()
    }
}

// ── TableRagConfig ────────────────────────────────────────────────────────

/// Configuration for [`crate::table_rag::TableRagEngine`].
#[derive(Debug, Clone, PartialEq)]
pub struct TableRagConfig {
    /// Maximum number of columns kept after schema retrieval. Defaults to
    /// `6`.
    pub top_k_columns: usize,
    /// Maximum number of rows kept after cell retrieval. Defaults to `10`.
    pub top_k_rows: usize,
    /// Maximum number of distinct values tracked per column during cell
    /// retrieval (the "distinct-value encoding" that bounds cost on large
    /// columns: matching cost per column is `O(probes × cap)`, independent
    /// of row count). Defaults to `64`.
    pub max_distinct_values_per_column: usize,
    /// Weight of the lexical (token-overlap + type-keyword) component of a
    /// column's schema-retrieval score. Defaults to `0.6`.
    pub lexical_weight: f32,
    /// Weight of the pseudo-embedding cosine-similarity component of a
    /// column's schema-retrieval score. Defaults to `0.4`.
    pub embedding_weight: f32,
    /// Minimum schema-retrieval score for a column to be considered
    /// relevant. Defaults to `0.05`.
    pub min_column_score: f32,
    /// Minimum match score for a [`CellProbe`] (and hence the rows it
    /// matches) to be considered relevant. Defaults to `0.05`.
    pub min_cell_score: f32,
    /// Dimensionality of the deterministic FNV-1a pseudo-embeddings used for
    /// both column-name and cell-value similarity. Defaults to `64`.
    pub pseudo_embedding_dim: usize,
}

impl Default for TableRagConfig {
    fn default() -> Self {
        Self {
            top_k_columns: 6,
            top_k_rows: 10,
            max_distinct_values_per_column: 64,
            lexical_weight: 0.6,
            embedding_weight: 0.4,
            min_column_score: 0.05,
            min_cell_score: 0.05,
            pseudo_embedding_dim: 64,
        }
    }
}

impl TableRagConfig {
    /// Create a new configuration with default values.
    #[must_use]
    pub fn new() -> Self {
        Self::default()
    }

    /// Set the maximum number of columns kept after schema retrieval
    /// (builder).
    #[must_use]
    pub fn with_top_k_columns(mut self, top_k_columns: usize) -> Self {
        self.top_k_columns = top_k_columns;
        self
    }

    /// Set the maximum number of rows kept after cell retrieval (builder).
    #[must_use]
    pub fn with_top_k_rows(mut self, top_k_rows: usize) -> Self {
        self.top_k_rows = top_k_rows;
        self
    }

    /// Set the per-column distinct-value cap (builder).
    #[must_use]
    pub fn with_max_distinct_values_per_column(mut self, max_distinct_values: usize) -> Self {
        self.max_distinct_values_per_column = max_distinct_values;
        self
    }

    /// Set the lexical-score weight (builder).
    #[must_use]
    pub fn with_lexical_weight(mut self, lexical_weight: f32) -> Self {
        self.lexical_weight = lexical_weight;
        self
    }

    /// Set the embedding-score weight (builder).
    #[must_use]
    pub fn with_embedding_weight(mut self, embedding_weight: f32) -> Self {
        self.embedding_weight = embedding_weight;
        self
    }

    /// Set the minimum column relevance score (builder).
    #[must_use]
    pub fn with_min_column_score(mut self, min_column_score: f32) -> Self {
        self.min_column_score = min_column_score;
        self
    }

    /// Set the minimum cell/probe relevance score (builder).
    #[must_use]
    pub fn with_min_cell_score(mut self, min_cell_score: f32) -> Self {
        self.min_cell_score = min_cell_score;
        self
    }

    /// Set the pseudo-embedding dimensionality (builder).
    #[must_use]
    pub fn with_pseudo_embedding_dim(mut self, pseudo_embedding_dim: usize) -> Self {
        self.pseudo_embedding_dim = pseudo_embedding_dim;
        self
    }
}

// ── TableRagError ─────────────────────────────────────────────────────────

/// Errors produced by the `table_rag` module.
#[derive(Debug, Error, Clone, PartialEq, Eq)]
pub enum TableRagError {
    /// The supplied query was empty (after trimming).
    #[error("query must not be empty")]
    EmptyQuery,
    /// The [`TableRagIndex`] contains no tables.
    #[error("table-rag index contains no tables")]
    EmptyIndex,
    /// A [`TableRagTable`] was constructed with a schema that declares no
    /// columns.
    #[error("table schema for '{table_name}' declares no columns")]
    EmptyColumns {
        /// The name of the offending table.
        table_name: String,
    },
    /// A [`TableRagTable`] row's cell count did not match its schema's
    /// column count.
    #[error(
        "table '{table_name}' row {row_index} has {found} cells, expected {expected} (one per schema column)"
    )]
    RowColumnMismatch {
        /// The name of the offending table.
        table_name: String,
        /// The index of the offending row.
        row_index: usize,
        /// The number of cells the schema requires (`schema.len()`).
        expected: usize,
        /// The number of cells the offending row actually has.
        found: usize,
    },
    /// Schema retrieval found no column, in any table of the index, scoring
    /// at or above [`TableRagConfig::min_column_score`] for this query.
    #[error("no column scored above the relevance threshold for this query")]
    NoRelevantColumns,
}
