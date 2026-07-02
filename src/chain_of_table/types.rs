//! Types, configuration, and errors for the `chain_of_table` module.
//!
//! Chain-of-Table (Wang et al. 2024, "Chain-of-Table: Evolving Tables in the
//! Reasoning Chain for Table Understanding") reasons over *tabular* data by
//! evolving an explicit **relational table state** through a chain of
//! symbolic, table-mutating operations. This file defines that state and the
//! operation vocabulary:
//!
//! * [`CotCell`] — a single table cell (`Text`, `Number`, or `Empty`).
//! * [`CotColumn`] / [`CotColumnType`] — a column header and its inferred type.
//! * [`CotRow`] — one row of cells.
//! * [`CotTableState`] — the whole table: columns, rows, and an optional
//!   `group_key` marking the table as grouped by a column (set by
//!   `f_group_by`, consumed by `f_aggregate`).
//! * [`CotTableOperation`] — the operation vocabulary
//!   (`f_add_column`, `f_select_row`, `f_select_column`, `f_group_by`,
//!   `f_sort_by`, `f_aggregate`), each with typed arguments
//!   ([`CotAddRule`], [`CotPredicate`], [`CotComparator`], [`CotAggregate`]).
//! * [`CotOperationTrace`] — a record of one applied operation.
//! * [`CotAnswer`] / [`CotAnswerValue`] — the extracted answer (a scalar or a
//!   reduced table) plus the full operation chain.
//! * [`ChainOfTableConfig`] / [`CotEnabledOperations`] — engine configuration.
//! * [`ChainOfTableError`] / [`ChainOfTableResult`] — the module error type
//!   and its `Result` alias.
//!
//! Unlike `program_of_thought`, which emits an arithmetic-scalar DSL executed
//! over *numbers*, every operation here transforms a *relational table* into
//! another relational table; the reasoning chain is a sequence of table
//! states, not a numeric program.

use std::cmp::Ordering;

use thiserror::Error;

// ── ChainOfTableResult ───────────────────────────────────────────────────────

/// Convenience `Result` alias for the `chain_of_table` module.
pub type ChainOfTableResult<T> = Result<T, ChainOfTableError>;

// ── CotCell ──────────────────────────────────────────────────────────────────

/// A single table cell.
///
/// A parsed table's cells are homogeneous per column: a column inferred as
/// [`CotColumnType::Number`] holds [`CotCell::Number`] / [`CotCell::Empty`],
/// and a column inferred as [`CotColumnType::Text`] holds [`CotCell::Text`] /
/// [`CotCell::Empty`]. Derived columns ([`CotTableOperation::AddColumn`]) may
/// introduce cells of the type produced by their [`CotAddRule`].
#[derive(Debug, Clone, PartialEq)]
pub enum CotCell {
    /// A textual value.
    Text(String),
    /// A finite numeric value. Parsing rejects non-finite values (`NaN`,
    /// `inf`), so a `Number` cell always holds a finite `f64`.
    Number(f64),
    /// A missing / blank value.
    Empty,
}

impl CotCell {
    /// Parse a raw string into a cell, inferring the type: a value that parses
    /// as a **finite** `f64` becomes [`CotCell::Number`]; an empty (or
    /// all-whitespace) string becomes [`CotCell::Empty`]; anything else
    /// becomes [`CotCell::Text`] (trimmed).
    #[must_use]
    pub fn parse(raw: &str) -> Self {
        let trimmed = raw.trim();
        if trimmed.is_empty() {
            return Self::Empty;
        }
        match trimmed.parse::<f64>() {
            Ok(value) if value.is_finite() => Self::Number(value),
            _ => Self::Text(trimmed.to_string()),
        }
    }

    /// Return `true` for [`CotCell::Empty`].
    #[must_use]
    pub fn is_empty(&self) -> bool {
        matches!(self, Self::Empty)
    }

    /// Return `true` for [`CotCell::Number`].
    #[must_use]
    pub fn is_number(&self) -> bool {
        matches!(self, Self::Number(_))
    }

    /// Return `true` for [`CotCell::Text`].
    #[must_use]
    pub fn is_text(&self) -> bool {
        matches!(self, Self::Text(_))
    }

    /// Return the numeric value when this is a [`CotCell::Number`], else
    /// `None`.
    #[must_use]
    pub fn as_number(&self) -> Option<f64> {
        match self {
            Self::Number(value) => Some(*value),
            _ => None,
        }
    }

    /// Return the text when this is a [`CotCell::Text`], else `None`.
    #[must_use]
    pub fn as_text(&self) -> Option<&str> {
        match self {
            Self::Text(text) => Some(text.as_str()),
            _ => None,
        }
    }

    /// Render the cell to a display string: the text itself for
    /// [`CotCell::Text`], a compact numeric rendering for [`CotCell::Number`]
    /// (integral values print without a trailing `.0`), and the empty string
    /// for [`CotCell::Empty`].
    #[must_use]
    pub fn to_display_string(&self) -> String {
        match self {
            Self::Text(text) => text.clone(),
            Self::Number(value) => format_number(*value),
            Self::Empty => String::new(),
        }
    }

    /// The [`CotColumnType`] a column consisting solely of this cell would be
    /// assigned ([`CotCell::Empty`] is treated as text-compatible).
    #[must_use]
    pub fn inferred_type(&self) -> CotColumnType {
        match self {
            Self::Number(_) => CotColumnType::Number,
            Self::Text(_) | Self::Empty => CotColumnType::Text,
        }
    }
}

/// Total ordering used by [`CotTableOperation::SortBy`] and by min/max
/// aggregation: `Empty` sorts before every value, all numbers sort before all
/// text, numbers order by value, and text orders lexicographically.
///
/// When `case_sensitive` is `false`, text is compared by its lowercased form.
/// Because parsed numbers are always finite, the numeric branch never observes
/// an incomparable pair.
#[must_use]
pub fn cot_cell_cmp(left: &CotCell, right: &CotCell, case_sensitive: bool) -> Ordering {
    match (left, right) {
        (CotCell::Empty, CotCell::Empty) => Ordering::Equal,
        (CotCell::Number(a), CotCell::Number(b)) => a.partial_cmp(b).unwrap_or(Ordering::Equal),
        (CotCell::Text(a), CotCell::Text(b)) => {
            if case_sensitive {
                a.cmp(b)
            } else {
                a.to_lowercase().cmp(&b.to_lowercase())
            }
        }
        // Empty sorts before any value; all numbers sort before all text.
        (CotCell::Empty, _) | (CotCell::Number(_), CotCell::Text(_)) => Ordering::Less,
        (_, CotCell::Empty) | (CotCell::Text(_), CotCell::Number(_)) => Ordering::Greater,
    }
}

/// Render `value` compactly: integral finite values print without a fractional
/// part (`12`, not `12.0`); everything else uses the default `f64` formatting.
#[must_use]
pub fn format_number(value: f64) -> String {
    if value.is_finite() && value.fract() == 0.0 {
        format!("{value:.0}")
    } else {
        format!("{value}")
    }
}

// ── CotColumnType ────────────────────────────────────────────────────────────

/// The inferred type of a column.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum CotColumnType {
    /// A column whose non-empty cells are all textual.
    Text,
    /// A column whose non-empty cells all parse as finite numbers.
    Number,
}

impl CotColumnType {
    /// Return a stable lowercase string name for the type.
    #[must_use]
    pub fn as_str(self) -> &'static str {
        match self {
            Self::Text => "text",
            Self::Number => "number",
        }
    }

    /// Return `true` for [`CotColumnType::Number`].
    #[must_use]
    pub fn is_numeric(self) -> bool {
        matches!(self, Self::Number)
    }
}

// ── CotColumn ────────────────────────────────────────────────────────────────

/// A table column: a header name plus its inferred [`CotColumnType`].
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct CotColumn {
    /// The column header text.
    pub name: String,
    /// The inferred type of the column's values.
    pub column_type: CotColumnType,
}

impl CotColumn {
    /// Construct a column from a name and an explicit type.
    #[must_use]
    pub fn new(name: impl Into<String>, column_type: CotColumnType) -> Self {
        Self {
            name: name.into(),
            column_type,
        }
    }

    /// Construct a text column.
    #[must_use]
    pub fn text(name: impl Into<String>) -> Self {
        Self::new(name, CotColumnType::Text)
    }

    /// Construct a numeric column.
    #[must_use]
    pub fn number(name: impl Into<String>) -> Self {
        Self::new(name, CotColumnType::Number)
    }
}

// ── CotRow ───────────────────────────────────────────────────────────────────

/// A single table row: an ordered vector of [`CotCell`]s aligned with the
/// table's columns.
#[derive(Debug, Clone, PartialEq, Default)]
pub struct CotRow {
    /// The row's cells, one per column, in column order.
    pub cells: Vec<CotCell>,
}

impl CotRow {
    /// Construct a row from a vector of cells.
    #[must_use]
    pub fn new(cells: Vec<CotCell>) -> Self {
        Self { cells }
    }

    /// Return the cell at `index`, or `None` when out of bounds.
    #[must_use]
    pub fn get(&self, index: usize) -> Option<&CotCell> {
        self.cells.get(index)
    }

    /// Return the number of cells in the row.
    #[must_use]
    pub fn len(&self) -> usize {
        self.cells.len()
    }

    /// Return `true` when the row has no cells.
    #[must_use]
    pub fn is_empty(&self) -> bool {
        self.cells.is_empty()
    }
}

// ── CotTableState ────────────────────────────────────────────────────────────

/// The evolving relational table state that a Chain-of-Table run transforms.
///
/// A state carries its `columns` (headers + inferred types), its `rows`, and
/// an optional `group_key`. `group_key` is `None` for a flat table; it is set
/// to `Some(column)` by [`CotTableOperation::GroupBy`], which reorders the
/// rows so equal group values are contiguous (in first-appearance order) but
/// does *not* collapse them. A subsequent [`CotTableOperation::Aggregate`]
/// consumes the `group_key` to produce one aggregated row per group.
#[derive(Debug, Clone, PartialEq, Default)]
pub struct CotTableState {
    /// The table's columns, in order.
    pub columns: Vec<CotColumn>,
    /// The table's rows; each row's cell count matches `columns.len()`.
    pub rows: Vec<CotRow>,
    /// The column this table is grouped by, or `None` for a flat table.
    pub group_key: Option<String>,
}

impl CotTableState {
    /// Construct a flat table state from columns and rows without validation.
    ///
    /// Callers are responsible for keeping each row's length equal to
    /// `columns.len()`; [`CotTableState::parse`] enforces this when building
    /// from raw strings.
    #[must_use]
    pub fn new(columns: Vec<CotColumn>, rows: Vec<CotRow>) -> Self {
        Self {
            columns,
            rows,
            group_key: None,
        }
    }

    /// Parse a table from string `headers` and string `rows`, inferring each
    /// column's [`CotColumnType`].
    ///
    /// A column is typed [`CotColumnType::Number`] when every one of its
    /// non-empty cells parses as a finite number, and [`CotColumnType::Text`]
    /// otherwise. Cells are built to match: numeric columns hold
    /// [`CotCell::Number`] / [`CotCell::Empty`]; text columns hold
    /// [`CotCell::Text`] / [`CotCell::Empty`].
    ///
    /// # Errors
    ///
    /// - [`ChainOfTableError::EmptyTable`] when `headers` is empty.
    /// - [`ChainOfTableError::DuplicateColumn`] when two headers are equal.
    /// - [`ChainOfTableError::RaggedRow`] when a row's length differs from the
    ///   header count.
    pub fn parse<S>(headers: &[S], rows: &[Vec<S>]) -> ChainOfTableResult<Self>
    where
        S: AsRef<str>,
    {
        if headers.is_empty() {
            return Err(ChainOfTableError::EmptyTable);
        }

        let header_names: Vec<String> = headers
            .iter()
            .map(|h| h.as_ref().trim().to_string())
            .collect();
        for i in 0..header_names.len() {
            for j in (i + 1)..header_names.len() {
                if header_names[i] == header_names[j] {
                    return Err(ChainOfTableError::DuplicateColumn {
                        column: header_names[i].clone(),
                    });
                }
            }
        }

        let width = header_names.len();
        for (index, row) in rows.iter().enumerate() {
            if row.len() != width {
                return Err(ChainOfTableError::RaggedRow {
                    row: index,
                    expected: width,
                    found: row.len(),
                });
            }
        }

        // First pass: parse every cell and infer per-column types.
        let parsed: Vec<Vec<CotCell>> = rows
            .iter()
            .map(|row| {
                row.iter()
                    .map(|value| CotCell::parse(value.as_ref()))
                    .collect()
            })
            .collect();

        let mut column_types: Vec<CotColumnType> = Vec::with_capacity(width);
        for col in 0..width {
            let mut has_value = false;
            let mut all_numeric = true;
            for row in &parsed {
                match &row[col] {
                    CotCell::Empty => {}
                    CotCell::Number(_) => has_value = true,
                    CotCell::Text(_) => {
                        has_value = true;
                        all_numeric = false;
                    }
                }
            }
            // A column with no non-empty values defaults to text.
            column_types.push(if has_value && all_numeric {
                CotColumnType::Number
            } else {
                CotColumnType::Text
            });
        }

        // Second pass: coerce each cell to its column's type (a lone numeric
        // string inside an otherwise-text column becomes text).
        let mut built_rows: Vec<CotRow> = Vec::with_capacity(parsed.len());
        for row in parsed {
            let mut cells: Vec<CotCell> = Vec::with_capacity(width);
            for (col, cell) in row.into_iter().enumerate() {
                let coerced = match (column_types[col], cell) {
                    (_, CotCell::Empty) => CotCell::Empty,
                    (CotColumnType::Number, other) => other,
                    (CotColumnType::Text, CotCell::Number(value)) => {
                        CotCell::Text(format_number(value))
                    }
                    (CotColumnType::Text, text) => text,
                };
                cells.push(coerced);
            }
            built_rows.push(CotRow::new(cells));
        }

        let columns = header_names
            .into_iter()
            .zip(column_types)
            .map(|(name, column_type)| CotColumn::new(name, column_type))
            .collect();

        Ok(Self::new(columns, built_rows))
    }

    /// Return the number of columns.
    #[must_use]
    pub fn column_count(&self) -> usize {
        self.columns.len()
    }

    /// Return the number of rows.
    #[must_use]
    pub fn row_count(&self) -> usize {
        self.rows.len()
    }

    /// Return `true` when the table has no columns (a structurally empty
    /// table). A header-only table with zero rows is *not* considered empty.
    #[must_use]
    pub fn is_empty(&self) -> bool {
        self.columns.is_empty()
    }

    /// Return the column header names, in order.
    #[must_use]
    pub fn column_names(&self) -> Vec<String> {
        self.columns.iter().map(|c| c.name.clone()).collect()
    }

    /// Resolve a column by name to its index, honouring `case_sensitive`.
    ///
    /// With `case_sensitive == false`, an ASCII-case-insensitive comparison is
    /// used. Returns the first matching index, or `None` when no column
    /// matches.
    #[must_use]
    pub fn resolve_column(&self, name: &str, case_sensitive: bool) -> Option<usize> {
        let needle = name.trim();
        self.columns.iter().position(|column| {
            if case_sensitive {
                column.name == needle
            } else {
                column.name.eq_ignore_ascii_case(needle)
            }
        })
    }

    /// Return the [`CotColumn`] at `index`, or `None` when out of bounds.
    #[must_use]
    pub fn column(&self, index: usize) -> Option<&CotColumn> {
        self.columns.get(index)
    }

    /// Return the cell at `(row, column)`, or `None` when out of bounds.
    #[must_use]
    pub fn cell(&self, row: usize, column: usize) -> Option<&CotCell> {
        self.rows.get(row).and_then(|r| r.get(column))
    }

    /// Return a copy of this state tagged as grouped by `column`.
    #[must_use]
    pub fn with_group_key(mut self, column: impl Into<String>) -> Self {
        self.group_key = Some(column.into());
        self
    }

    /// Return the group key, or `None` when the table is flat.
    #[must_use]
    pub fn group_key(&self) -> Option<&str> {
        self.group_key.as_deref()
    }

    /// Return `true` when the table is a single cell (one column, one row) —
    /// the shape produced by aggregating a flat table to a scalar.
    #[must_use]
    pub fn is_scalar(&self) -> bool {
        self.column_count() == 1 && self.row_count() == 1
    }
}

// ── CotComparator ────────────────────────────────────────────────────────────

/// A comparison operator used by [`CotPredicate`] in
/// [`CotTableOperation::SelectRow`].
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum CotComparator {
    /// Equal (`=`).
    Eq,
    /// Not equal (`!=`).
    Ne,
    /// Strictly less than (`<`).
    Lt,
    /// Less than or equal (`<=`).
    Le,
    /// Strictly greater than (`>`).
    Gt,
    /// Greater than or equal (`>=`).
    Ge,
}

impl CotComparator {
    /// Return the source symbol denoting this comparator.
    #[must_use]
    pub fn as_str(self) -> &'static str {
        match self {
            Self::Eq => "=",
            Self::Ne => "!=",
            Self::Lt => "<",
            Self::Le => "<=",
            Self::Gt => ">",
            Self::Ge => ">=",
        }
    }

    /// Return `true` when this comparator is an ordering comparator
    /// (`<`, `<=`, `>`, `>=`) rather than (in)equality.
    #[must_use]
    pub fn is_ordering(self) -> bool {
        matches!(self, Self::Lt | Self::Le | Self::Gt | Self::Ge)
    }
}

// ── CotPredicate ─────────────────────────────────────────────────────────────

/// A row predicate of the form `column <comparator> value`, evaluated per row
/// by [`CotTableOperation::SelectRow`].
#[derive(Debug, Clone, PartialEq)]
pub struct CotPredicate {
    /// The name of the column whose cell is tested.
    pub column: String,
    /// The comparison operator.
    pub comparator: CotComparator,
    /// The right-hand comparison value.
    pub value: CotCell,
}

impl CotPredicate {
    /// Construct a predicate.
    #[must_use]
    pub fn new(column: impl Into<String>, comparator: CotComparator, value: CotCell) -> Self {
        Self {
            column: column.into(),
            comparator,
            value,
        }
    }
}

// ── CotAggregate ─────────────────────────────────────────────────────────────

/// An aggregation function applied by [`CotTableOperation::Aggregate`].
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default)]
pub enum CotAggregate {
    /// Count of non-empty cells in the column.
    #[default]
    Count,
    /// Sum of the column's numeric cells (empty cells skipped; the additive
    /// identity `0` for no values).
    Sum,
    /// Arithmetic mean of the column's numeric cells.
    Mean,
    /// Maximum numeric value in the column.
    Max,
    /// Minimum numeric value in the column.
    Min,
}

impl CotAggregate {
    /// Return a stable lowercase name for the aggregation.
    #[must_use]
    pub fn as_str(self) -> &'static str {
        match self {
            Self::Count => "count",
            Self::Sum => "sum",
            Self::Mean => "mean",
            Self::Max => "max",
            Self::Min => "min",
        }
    }

    /// Return `true` when the aggregation requires a numeric column
    /// (everything except [`CotAggregate::Count`], which counts non-empty
    /// cells of any type).
    #[must_use]
    pub fn requires_numeric(self) -> bool {
        !matches!(self, Self::Count)
    }
}

// ── CotAddRule ───────────────────────────────────────────────────────────────

/// A deterministic per-row derivation rule for [`CotTableOperation::AddColumn`].
///
/// Each rule computes one [`CotCell`] per row from that row's existing cells.
/// The arithmetic rules ([`CotAddRule::Sum`], [`CotAddRule::Difference`],
/// [`CotAddRule::Product`], [`CotAddRule::Ratio`]) require their referenced
/// columns to be numeric and yield a numeric column;
/// [`CotAddRule::Concat`] yields text; [`CotAddRule::Constant`] fills every
/// row with the same value.
#[derive(Debug, Clone, PartialEq)]
pub enum CotAddRule {
    /// Sum of the referenced numeric columns (empty operands treated as `0`).
    Sum {
        /// Names of the numeric columns to add together.
        columns: Vec<String>,
    },
    /// `left - right` over two numeric columns (empty operand ⇒ empty result).
    Difference {
        /// The minuend column name.
        left: String,
        /// The subtrahend column name.
        right: String,
    },
    /// Product of the referenced numeric columns (empty operand ⇒ empty
    /// result).
    Product {
        /// Names of the numeric columns to multiply together.
        columns: Vec<String>,
    },
    /// `numerator / denominator` over two numeric columns; a zero or empty
    /// denominator (or empty numerator) yields an empty result rather than a
    /// division error.
    Ratio {
        /// The numerator column name.
        numerator: String,
        /// The denominator column name.
        denominator: String,
    },
    /// A constant value assigned to every row.
    Constant {
        /// The constant cell value.
        value: CotCell,
    },
    /// Text concatenation of the referenced columns' display strings, joined
    /// by `separator`.
    Concat {
        /// Names of the columns to concatenate, in order.
        columns: Vec<String>,
        /// The separator inserted between values.
        separator: String,
    },
}

// ── CotTableOperation ────────────────────────────────────────────────────────

/// The Chain-of-Table operation vocabulary: symbolic transforms that each map
/// a [`CotTableState`] to a new [`CotTableState`].
///
/// The variant names are idiomatic Rust; each variant's paper-style operation
/// name (`f_add_column`, `f_select_row`, …) is available via
/// [`CotTableOperation::name`].
#[derive(Debug, Clone, PartialEq)]
pub enum CotTableOperation {
    /// `f_add_column`: derive a new column named `name` via `rule`.
    AddColumn {
        /// The name of the new column.
        name: String,
        /// The per-row derivation rule.
        rule: CotAddRule,
    },
    /// `f_select_row`: keep only rows satisfying `predicate`.
    SelectRow {
        /// The row-selection predicate.
        predicate: CotPredicate,
    },
    /// `f_select_column`: project to the named subset of columns, in the given
    /// order.
    SelectColumn {
        /// The column names to keep, in output order.
        columns: Vec<String>,
    },
    /// `f_group_by`: mark the table as grouped by `column`, reordering rows so
    /// equal group values are contiguous.
    GroupBy {
        /// The column to group by.
        column: String,
    },
    /// `f_sort_by`: order rows by `column`, ascending when `ascending` is
    /// `true`.
    SortBy {
        /// The column to sort by.
        column: String,
        /// Sort direction: ascending when `true`, descending when `false`.
        ascending: bool,
    },
    /// `f_aggregate`: reduce `column` with `agg`. On a flat table this yields a
    /// single scalar cell; on a grouped table it yields one row per group.
    Aggregate {
        /// The column to aggregate.
        column: String,
        /// The aggregation function.
        agg: CotAggregate,
    },
}

impl CotTableOperation {
    /// Return the [`CotOperationKind`] discriminant of this operation.
    #[must_use]
    pub fn kind(&self) -> CotOperationKind {
        match self {
            Self::AddColumn { .. } => CotOperationKind::AddColumn,
            Self::SelectRow { .. } => CotOperationKind::SelectRow,
            Self::SelectColumn { .. } => CotOperationKind::SelectColumn,
            Self::GroupBy { .. } => CotOperationKind::GroupBy,
            Self::SortBy { .. } => CotOperationKind::SortBy,
            Self::Aggregate { .. } => CotOperationKind::Aggregate,
        }
    }

    /// Return the paper-style operation name (`f_add_column`, `f_select_row`,
    /// `f_select_column`, `f_group_by`, `f_sort_by`, `f_aggregate`).
    #[must_use]
    pub fn name(&self) -> &'static str {
        self.kind().name()
    }
}

// ── CotOperationKind ─────────────────────────────────────────────────────────

/// The kind (discriminant) of a [`CotTableOperation`], independent of its
/// arguments. Used for enable/disable gating and de-duplication in the
/// planner.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum CotOperationKind {
    /// [`CotTableOperation::AddColumn`].
    AddColumn,
    /// [`CotTableOperation::SelectRow`].
    SelectRow,
    /// [`CotTableOperation::SelectColumn`].
    SelectColumn,
    /// [`CotTableOperation::GroupBy`].
    GroupBy,
    /// [`CotTableOperation::SortBy`].
    SortBy,
    /// [`CotTableOperation::Aggregate`].
    Aggregate,
}

impl CotOperationKind {
    /// Return the paper-style operation name for this kind.
    #[must_use]
    pub fn name(self) -> &'static str {
        match self {
            Self::AddColumn => "f_add_column",
            Self::SelectRow => "f_select_row",
            Self::SelectColumn => "f_select_column",
            Self::GroupBy => "f_group_by",
            Self::SortBy => "f_sort_by",
            Self::Aggregate => "f_aggregate",
        }
    }

    /// Every operation kind, in the planner's priority order.
    #[must_use]
    pub fn all() -> [CotOperationKind; 6] {
        [
            Self::AddColumn,
            Self::SelectRow,
            Self::GroupBy,
            Self::SortBy,
            Self::Aggregate,
            Self::SelectColumn,
        ]
    }
}

// ── CotEnabledOperations ─────────────────────────────────────────────────────

/// Which operation kinds the planner is permitted to emit. All are enabled by
/// default.
///
/// One boolean per operation in the six-operation vocabulary — the flags model
/// the vocabulary directly, so the "excessive bools" heuristic does not apply.
#[allow(clippy::struct_excessive_bools)]
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct CotEnabledOperations {
    /// Whether `f_add_column` is enabled.
    pub add_column: bool,
    /// Whether `f_select_row` is enabled.
    pub select_row: bool,
    /// Whether `f_select_column` is enabled.
    pub select_column: bool,
    /// Whether `f_group_by` is enabled.
    pub group_by: bool,
    /// Whether `f_sort_by` is enabled.
    pub sort_by: bool,
    /// Whether `f_aggregate` is enabled.
    pub aggregate: bool,
}

impl Default for CotEnabledOperations {
    fn default() -> Self {
        Self::all()
    }
}

impl CotEnabledOperations {
    /// All operations enabled.
    #[must_use]
    pub fn all() -> Self {
        Self {
            add_column: true,
            select_row: true,
            select_column: true,
            group_by: true,
            sort_by: true,
            aggregate: true,
        }
    }

    /// All operations disabled.
    #[must_use]
    pub fn none() -> Self {
        Self {
            add_column: false,
            select_row: false,
            select_column: false,
            group_by: false,
            sort_by: false,
            aggregate: false,
        }
    }

    /// Return `true` when `kind` is enabled.
    #[must_use]
    pub fn is_enabled(self, kind: CotOperationKind) -> bool {
        match kind {
            CotOperationKind::AddColumn => self.add_column,
            CotOperationKind::SelectRow => self.select_row,
            CotOperationKind::SelectColumn => self.select_column,
            CotOperationKind::GroupBy => self.group_by,
            CotOperationKind::SortBy => self.sort_by,
            CotOperationKind::Aggregate => self.aggregate,
        }
    }

    /// Return a copy with `kind` set to `enabled`.
    #[must_use]
    pub fn with(mut self, kind: CotOperationKind, enabled: bool) -> Self {
        match kind {
            CotOperationKind::AddColumn => self.add_column = enabled,
            CotOperationKind::SelectRow => self.select_row = enabled,
            CotOperationKind::SelectColumn => self.select_column = enabled,
            CotOperationKind::GroupBy => self.group_by = enabled,
            CotOperationKind::SortBy => self.sort_by = enabled,
            CotOperationKind::Aggregate => self.aggregate = enabled,
        }
        self
    }
}

// ── ChainOfTableConfig ───────────────────────────────────────────────────────

/// Configuration for [`crate::chain_of_table::ChainOfTableEngine`].
#[derive(Debug, Clone, PartialEq)]
pub struct ChainOfTableConfig {
    /// Maximum number of operations the drive loop will apply before stopping,
    /// regardless of whether the planner would suggest more. Defaults to `6`
    /// (one of each operation kind).
    pub max_steps: usize,
    /// Whether column-name matching and text-value comparison are
    /// case-sensitive. Defaults to `false`.
    pub case_sensitive: bool,
    /// Which operation kinds the planner may emit. Defaults to all enabled.
    pub enabled_operations: CotEnabledOperations,
    /// The aggregation used when the question asks to aggregate but names no
    /// specific function. Defaults to [`CotAggregate::Count`].
    pub default_aggregate: CotAggregate,
    /// Absolute tolerance for numeric equality in [`CotComparator::Eq`] /
    /// [`CotComparator::Ne`]. Defaults to `1e-9`.
    pub numeric_epsilon: f64,
}

impl Default for ChainOfTableConfig {
    fn default() -> Self {
        Self {
            max_steps: 6,
            case_sensitive: false,
            enabled_operations: CotEnabledOperations::all(),
            default_aggregate: CotAggregate::Count,
            numeric_epsilon: 1e-9,
        }
    }
}

impl ChainOfTableConfig {
    /// Create a configuration with default values.
    #[must_use]
    pub fn new() -> Self {
        Self::default()
    }

    /// Set the maximum number of drive-loop steps.
    #[must_use]
    pub fn with_max_steps(mut self, max_steps: usize) -> Self {
        self.max_steps = max_steps;
        self
    }

    /// Set whether name matching and text comparison are case-sensitive.
    #[must_use]
    pub fn with_case_sensitive(mut self, case_sensitive: bool) -> Self {
        self.case_sensitive = case_sensitive;
        self
    }

    /// Set which operation kinds the planner may emit.
    #[must_use]
    pub fn with_enabled_operations(mut self, enabled_operations: CotEnabledOperations) -> Self {
        self.enabled_operations = enabled_operations;
        self
    }

    /// Set the default aggregation function.
    #[must_use]
    pub fn with_default_aggregate(mut self, default_aggregate: CotAggregate) -> Self {
        self.default_aggregate = default_aggregate;
        self
    }

    /// Set the numeric-equality tolerance.
    #[must_use]
    pub fn with_numeric_epsilon(mut self, numeric_epsilon: f64) -> Self {
        self.numeric_epsilon = numeric_epsilon;
        self
    }
}

// ── CotOperationTrace ────────────────────────────────────────────────────────

/// A record of one applied operation in a Chain-of-Table run: the operation
/// itself plus a summary of its effect on the table shape.
#[derive(Debug, Clone, PartialEq)]
pub struct CotOperationTrace {
    /// Zero-based position of this operation in the chain.
    pub step: usize,
    /// The operation that was applied.
    pub operation: CotTableOperation,
    /// Row count of the table *before* the operation.
    pub rows_before: usize,
    /// Row count of the table *after* the operation.
    pub rows_after: usize,
    /// Column count of the table *after* the operation.
    pub columns_after: usize,
    /// A short human-readable description of what the operation did.
    pub description: String,
}

impl CotOperationTrace {
    /// Construct a trace record.
    #[must_use]
    pub fn new(
        step: usize,
        operation: CotTableOperation,
        rows_before: usize,
        rows_after: usize,
        columns_after: usize,
        description: impl Into<String>,
    ) -> Self {
        Self {
            step,
            operation,
            rows_before,
            rows_after,
            columns_after,
            description: description.into(),
        }
    }
}

// ── CotAnswerValue / CotAnswer ───────────────────────────────────────────────

/// The extracted value of a Chain-of-Table run: either a single scalar cell
/// (from a fully-reduced aggregate) or a small resulting table.
#[derive(Debug, Clone, PartialEq)]
pub enum CotAnswerValue {
    /// A single scalar cell.
    Scalar(CotCell),
    /// A resulting table (e.g. a filtered/sorted/grouped result, or the top
    /// row of a superlative query).
    Table(CotTableState),
}

impl CotAnswerValue {
    /// Return the scalar cell when this is a [`CotAnswerValue::Scalar`], else
    /// `None`.
    #[must_use]
    pub fn as_scalar(&self) -> Option<&CotCell> {
        match self {
            Self::Scalar(cell) => Some(cell),
            Self::Table(_) => None,
        }
    }

    /// Return the table when this is a [`CotAnswerValue::Table`], else `None`.
    #[must_use]
    pub fn as_table(&self) -> Option<&CotTableState> {
        match self {
            Self::Table(table) => Some(table),
            Self::Scalar(_) => None,
        }
    }

    /// Return `true` for a [`CotAnswerValue::Scalar`].
    #[must_use]
    pub fn is_scalar(&self) -> bool {
        matches!(self, Self::Scalar(_))
    }
}

/// The complete output of a [`crate::chain_of_table::ChainOfTableEngine`] run:
/// the extracted [`CotAnswerValue`], the final [`CotTableState`], and the full
/// chain of [`CotOperationTrace`]s applied to reach it.
#[derive(Debug, Clone, PartialEq)]
pub struct CotAnswer {
    /// The original question.
    pub question: String,
    /// The extracted answer value.
    pub value: CotAnswerValue,
    /// The final table state after the last operation.
    pub final_state: CotTableState,
    /// The ordered chain of operation traces.
    pub chain: Vec<CotOperationTrace>,
}

impl CotAnswer {
    /// Return the scalar answer cell when the answer is scalar, else `None`.
    #[must_use]
    pub fn scalar(&self) -> Option<&CotCell> {
        self.value.as_scalar()
    }

    /// Return the numeric scalar answer when the answer is a numeric scalar,
    /// else `None`.
    #[must_use]
    pub fn as_number(&self) -> Option<f64> {
        self.value.as_scalar().and_then(CotCell::as_number)
    }

    /// Return the number of operations applied.
    #[must_use]
    pub fn step_count(&self) -> usize {
        self.chain.len()
    }

    /// Return the ordered paper-style names of every operation in the chain.
    #[must_use]
    pub fn operation_names(&self) -> Vec<&'static str> {
        self.chain.iter().map(|t| t.operation.name()).collect()
    }
}

// ── ChainOfTableError ────────────────────────────────────────────────────────

/// Errors from the `chain_of_table` module.
#[derive(Debug, Error, Clone, PartialEq)]
pub enum ChainOfTableError {
    /// The question was empty or whitespace-only.
    #[error("question must not be empty")]
    EmptyQuestion,
    /// The table has no columns.
    #[error("table must have at least one column")]
    EmptyTable,
    /// A row's cell count did not match the header count during parsing.
    #[error("row {row} has {found} cells but the table has {expected} columns")]
    RaggedRow {
        /// Zero-based index of the offending row.
        row: usize,
        /// The expected cell count (the header count).
        expected: usize,
        /// The actual cell count.
        found: usize,
    },
    /// Two columns share the same header name.
    #[error("duplicate column name: {column}")]
    DuplicateColumn {
        /// The duplicated column name.
        column: String,
    },
    /// An operation referenced a column that is not present in the table.
    #[error("unknown column: {column}")]
    UnknownColumn {
        /// The unresolved column name.
        column: String,
    },
    /// An operation required a numeric column but the referenced column (or a
    /// cell within it) was not numeric.
    #[error("operation {operation} requires numeric column '{column}' but it is not numeric")]
    TypeMismatch {
        /// The paper-style operation name that required numeric data.
        operation: String,
        /// The offending column name.
        column: String,
    },
    /// An aggregation that needs at least one value (mean, max, or min) was
    /// applied to a column with no numeric values.
    #[error("aggregate {aggregate} over column '{column}' has no numeric values")]
    EmptyAggregate {
        /// The aggregation name.
        aggregate: String,
        /// The column name.
        column: String,
    },
    /// A [`CotTableOperation::SelectColumn`] projected to zero columns.
    #[error("column projection must keep at least one column")]
    EmptyProjection,
    /// The planner produced no applicable operation and the configuration
    /// enabled none, so no reasoning could take place.
    #[error("no operations are enabled")]
    NoOperationsEnabled,
}
