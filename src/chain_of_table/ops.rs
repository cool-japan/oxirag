//! The Chain-of-Table operation executor.
//!
//! [`apply`] (and its configurable sibling [`apply_with_config`]) is the heart
//! of the module: it takes one [`CotTableOperation`] and a [`CotTableState`]
//! and returns the *new* table state that results from applying the operation.
//! Every operation is a total, deterministic function from one relational
//! table to another — no arithmetic-scalar DSL, no floating evaluation of a
//! program, just table-to-table transforms.
//!
//! The six operations:
//!
//! * `f_add_column` — [`apply_add_column`]: derive a new column per row via a
//!   [`CotAddRule`].
//! * `f_select_row` — [`apply_select_row`]: keep rows satisfying a
//!   [`CotPredicate`].
//! * `f_select_column` — [`apply_select_column`]: project to a subset of
//!   columns.
//! * `f_group_by` — [`apply_group_by`]: tag the table as grouped and reorder
//!   rows so equal group values are contiguous.
//! * `f_sort_by` — [`apply_sort_by`]: order rows by a column.
//! * `f_aggregate` — [`apply_aggregate`]: reduce a column with an aggregation,
//!   yielding a scalar (flat table) or one row per group (grouped table).

use std::cmp::Ordering;

use super::types::{
    ChainOfTableConfig, ChainOfTableError, ChainOfTableResult, CotAddRule, CotAggregate, CotCell,
    CotColumn, CotColumnType, CotComparator, CotPredicate, CotRow, CotTableOperation,
    CotTableState, cot_cell_cmp,
};

// ── Public executor ──────────────────────────────────────────────────────────

/// Apply `operation` to `state`, returning the resulting table state.
///
/// This convenience entry point uses [`ChainOfTableConfig::default`] (which is
/// case-*insensitive* column matching with a `1e-9` numeric-equality
/// tolerance). Use [`apply_with_config`] to control case sensitivity and the
/// numeric tolerance.
///
/// # Errors
///
/// See [`apply_with_config`].
pub fn apply(
    operation: &CotTableOperation,
    state: &CotTableState,
) -> ChainOfTableResult<CotTableState> {
    apply_with_config(operation, state, &ChainOfTableConfig::default())
}

/// Apply `operation` to `state` under `config`, returning the resulting table
/// state.
///
/// # Errors
///
/// - [`ChainOfTableError::EmptyTable`] when `state` has no columns.
/// - [`ChainOfTableError::UnknownColumn`] when an operation references a
///   column not present in the table.
/// - [`ChainOfTableError::DuplicateColumn`] when `f_add_column` would create a
///   column whose name already exists.
/// - [`ChainOfTableError::TypeMismatch`] when a numeric operation touches a
///   non-numeric column or cell.
/// - [`ChainOfTableError::EmptyAggregate`] when `mean`/`max`/`min` is applied
///   to a column with no numeric values.
/// - [`ChainOfTableError::EmptyProjection`] when `f_select_column` keeps no
///   columns.
pub fn apply_with_config(
    operation: &CotTableOperation,
    state: &CotTableState,
    config: &ChainOfTableConfig,
) -> ChainOfTableResult<CotTableState> {
    if state.is_empty() {
        return Err(ChainOfTableError::EmptyTable);
    }
    match operation {
        CotTableOperation::AddColumn { name, rule } => apply_add_column(state, name, rule, config),
        CotTableOperation::SelectRow { predicate } => apply_select_row(state, predicate, config),
        CotTableOperation::SelectColumn { columns } => apply_select_column(state, columns, config),
        CotTableOperation::GroupBy { column } => apply_group_by(state, column, config),
        CotTableOperation::SortBy { column, ascending } => {
            apply_sort_by(state, column, *ascending, config)
        }
        CotTableOperation::Aggregate { column, agg } => {
            apply_aggregate(state, column, *agg, config)
        }
    }
}

// ── column resolution ────────────────────────────────────────────────────────

/// Resolve `name` to a column index, or return [`ChainOfTableError::UnknownColumn`].
fn resolve(
    state: &CotTableState,
    name: &str,
    config: &ChainOfTableConfig,
) -> ChainOfTableResult<usize> {
    state
        .resolve_column(name, config.case_sensitive)
        .ok_or_else(|| ChainOfTableError::UnknownColumn {
            column: name.to_string(),
        })
}

// ── f_add_column ─────────────────────────────────────────────────────────────

/// Execute `f_add_column`: derive a new column and append it.
fn apply_add_column(
    state: &CotTableState,
    name: &str,
    rule: &CotAddRule,
    config: &ChainOfTableConfig,
) -> ChainOfTableResult<CotTableState> {
    if state.resolve_column(name, config.case_sensitive).is_some() {
        return Err(ChainOfTableError::DuplicateColumn {
            column: name.to_string(),
        });
    }

    let (column_type, cells) = derive_column(state, rule, config)?;

    let mut columns = state.columns.clone();
    columns.push(CotColumn::new(name, column_type));

    let rows = state
        .rows
        .iter()
        .zip(cells)
        .map(|(row, cell)| {
            let mut new_cells = row.cells.clone();
            new_cells.push(cell);
            CotRow::new(new_cells)
        })
        .collect();

    Ok(CotTableState {
        columns,
        rows,
        group_key: state.group_key.clone(),
    })
}

/// Compute the derived column's type and its per-row cells.
fn derive_column(
    state: &CotTableState,
    rule: &CotAddRule,
    config: &ChainOfTableConfig,
) -> ChainOfTableResult<(CotColumnType, Vec<CotCell>)> {
    match rule {
        CotAddRule::Sum { columns } => {
            let indices = numeric_indices(state, columns, config)?;
            let cells = state
                .rows
                .iter()
                .map(|row| {
                    // Missing operands contribute the additive identity, so a
                    // sum is always defined.
                    let total: f64 = indices
                        .iter()
                        .filter_map(|&idx| row.get(idx).and_then(CotCell::as_number))
                        .sum();
                    CotCell::Number(total)
                })
                .collect();
            Ok((CotColumnType::Number, cells))
        }
        CotAddRule::Product { columns } => {
            let indices = numeric_indices(state, columns, config)?;
            let cells = state
                .rows
                .iter()
                .map(|row| product_cell(row, &indices))
                .collect();
            Ok((CotColumnType::Number, cells))
        }
        CotAddRule::Difference { left, right } => {
            let left_idx = numeric_index(state, left, config)?;
            let right_idx = numeric_index(state, right, config)?;
            let cells = state
                .rows
                .iter()
                .map(|row| {
                    match (
                        row.get(left_idx).and_then(CotCell::as_number),
                        row.get(right_idx).and_then(CotCell::as_number),
                    ) {
                        (Some(a), Some(b)) => CotCell::Number(a - b),
                        _ => CotCell::Empty,
                    }
                })
                .collect();
            Ok((CotColumnType::Number, cells))
        }
        CotAddRule::Ratio {
            numerator,
            denominator,
        } => {
            let num_idx = numeric_index(state, numerator, config)?;
            let den_idx = numeric_index(state, denominator, config)?;
            let cells = state
                .rows
                .iter()
                .map(|row| {
                    match (
                        row.get(num_idx).and_then(CotCell::as_number),
                        row.get(den_idx).and_then(CotCell::as_number),
                    ) {
                        (Some(a), Some(b)) if b != 0.0 => CotCell::Number(a / b),
                        _ => CotCell::Empty,
                    }
                })
                .collect();
            Ok((CotColumnType::Number, cells))
        }
        CotAddRule::Constant { value } => {
            let cells = vec![value.clone(); state.rows.len()];
            Ok((value.inferred_type(), cells))
        }
        CotAddRule::Concat { columns, separator } => {
            let indices = columns
                .iter()
                .map(|name| resolve(state, name, config))
                .collect::<ChainOfTableResult<Vec<usize>>>()?;
            let cells = state
                .rows
                .iter()
                .map(|row| {
                    let parts: Vec<String> = indices
                        .iter()
                        .map(|&idx| {
                            row.get(idx)
                                .map(CotCell::to_display_string)
                                .unwrap_or_default()
                        })
                        .collect();
                    CotCell::Text(parts.join(separator))
                })
                .collect();
            Ok((CotColumnType::Text, cells))
        }
    }
}

/// Compute a row's product cell: [`CotCell::Empty`] if any operand is missing,
/// otherwise the product of all operands.
fn product_cell(row: &CotRow, indices: &[usize]) -> CotCell {
    let mut product = 1.0;
    for &idx in indices {
        match row.get(idx).and_then(CotCell::as_number) {
            Some(value) => product *= value,
            None => return CotCell::Empty,
        }
    }
    CotCell::Number(product)
}

/// Resolve every name in `columns` to an index, requiring each to be numeric.
fn numeric_indices(
    state: &CotTableState,
    columns: &[String],
    config: &ChainOfTableConfig,
) -> ChainOfTableResult<Vec<usize>> {
    columns
        .iter()
        .map(|name| numeric_index(state, name, config))
        .collect()
}

/// Resolve `name` to an index, requiring the column to be numeric.
fn numeric_index(
    state: &CotTableState,
    name: &str,
    config: &ChainOfTableConfig,
) -> ChainOfTableResult<usize> {
    let idx = resolve(state, name, config)?;
    if state.columns[idx].column_type.is_numeric() {
        Ok(idx)
    } else {
        Err(ChainOfTableError::TypeMismatch {
            operation: "f_add_column".to_string(),
            column: name.to_string(),
        })
    }
}

// ── f_select_row ─────────────────────────────────────────────────────────────

/// Execute `f_select_row`: keep rows satisfying `predicate`.
fn apply_select_row(
    state: &CotTableState,
    predicate: &CotPredicate,
    config: &ChainOfTableConfig,
) -> ChainOfTableResult<CotTableState> {
    let idx = resolve(state, &predicate.column, config)?;
    let rows = state
        .rows
        .iter()
        .filter(|row| {
            row.get(idx).is_some_and(|cell| {
                cell_satisfies(
                    cell,
                    predicate.comparator,
                    &predicate.value,
                    config.case_sensitive,
                    config.numeric_epsilon,
                )
            })
        })
        .cloned()
        .collect();

    Ok(CotTableState {
        columns: state.columns.clone(),
        rows,
        group_key: state.group_key.clone(),
    })
}

/// Return `true` when `cell` satisfies `comparator` against `value`.
///
/// Equality (`=` / `!=`) compares numbers within `eps`, text with the
/// configured case sensitivity, and treats empty-vs-empty as equal;
/// mismatched types are unequal. Ordering comparators (`<`, `<=`, `>`, `>=`)
/// are defined only for same-typed number/number and text/text pairs; any
/// empty or cross-type operand makes an ordering comparison `false`.
#[must_use]
fn cell_satisfies(
    cell: &CotCell,
    comparator: CotComparator,
    value: &CotCell,
    case_sensitive: bool,
    eps: f64,
) -> bool {
    match comparator {
        CotComparator::Eq => cells_equal(cell, value, case_sensitive, eps),
        CotComparator::Ne => !cells_equal(cell, value, case_sensitive, eps),
        CotComparator::Lt | CotComparator::Le | CotComparator::Gt | CotComparator::Ge => {
            let Some(order) = order_cells(cell, value, case_sensitive) else {
                return false;
            };
            match comparator {
                CotComparator::Lt => order == Ordering::Less,
                CotComparator::Le => order != Ordering::Greater,
                CotComparator::Gt => order == Ordering::Greater,
                CotComparator::Ge => order != Ordering::Less,
                CotComparator::Eq | CotComparator::Ne => false,
            }
        }
    }
}

/// Value equality used by `=` / `!=` predicates.
fn cells_equal(left: &CotCell, right: &CotCell, case_sensitive: bool, eps: f64) -> bool {
    match (left, right) {
        (CotCell::Empty, CotCell::Empty) => true,
        (CotCell::Number(a), CotCell::Number(b)) => (a - b).abs() <= eps,
        (CotCell::Text(a), CotCell::Text(b)) => {
            if case_sensitive {
                a == b
            } else {
                a.eq_ignore_ascii_case(b)
            }
        }
        _ => false,
    }
}

/// Ordering used by `<` / `<=` / `>` / `>=` predicates. Returns `None` for
/// empty or cross-type operands, which makes those comparisons `false`.
fn order_cells(left: &CotCell, right: &CotCell, case_sensitive: bool) -> Option<Ordering> {
    match (left, right) {
        (CotCell::Number(a), CotCell::Number(b)) => a.partial_cmp(b),
        (CotCell::Text(a), CotCell::Text(b)) => Some(if case_sensitive {
            a.cmp(b)
        } else {
            a.to_lowercase().cmp(&b.to_lowercase())
        }),
        _ => None,
    }
}

// ── f_select_column ──────────────────────────────────────────────────────────

/// Execute `f_select_column`: project to the named columns, in order.
fn apply_select_column(
    state: &CotTableState,
    columns: &[String],
    config: &ChainOfTableConfig,
) -> ChainOfTableResult<CotTableState> {
    if columns.is_empty() {
        return Err(ChainOfTableError::EmptyProjection);
    }
    let indices = columns
        .iter()
        .map(|name| resolve(state, name, config))
        .collect::<ChainOfTableResult<Vec<usize>>>()?;

    let new_columns: Vec<CotColumn> = indices
        .iter()
        .map(|&idx| state.columns[idx].clone())
        .collect();
    let rows = state
        .rows
        .iter()
        .map(|row| {
            let cells = indices
                .iter()
                .map(|&idx| row.get(idx).cloned().unwrap_or(CotCell::Empty))
                .collect();
            CotRow::new(cells)
        })
        .collect();

    // A group key survives projection only when its column is kept.
    let group_key = state.group_key.clone().filter(|key| {
        new_columns
            .iter()
            .any(|column| column_name_matches(&column.name, key, config.case_sensitive))
    });

    Ok(CotTableState {
        columns: new_columns,
        rows,
        group_key,
    })
}

/// Compare two column names under the configured case sensitivity.
fn column_name_matches(a: &str, b: &str, case_sensitive: bool) -> bool {
    if case_sensitive {
        a == b
    } else {
        a.eq_ignore_ascii_case(b)
    }
}

// ── f_group_by ───────────────────────────────────────────────────────────────

/// Execute `f_group_by`: reorder rows so equal group values are contiguous (in
/// first-appearance order) and tag the table with the group key.
fn apply_group_by(
    state: &CotTableState,
    column: &str,
    config: &ChainOfTableConfig,
) -> ChainOfTableResult<CotTableState> {
    let idx = resolve(state, column, config)?;

    // Preserve first-appearance order of group keys and, within each group,
    // the original row order — both fully deterministic.
    let mut order: Vec<String> = Vec::new();
    let mut buckets: Vec<Vec<CotRow>> = Vec::new();
    for row in &state.rows {
        let key = group_key_string(row.get(idx), config.case_sensitive);
        if let Some(pos) = order.iter().position(|existing| existing == &key) {
            buckets[pos].push(row.clone());
        } else {
            order.push(key);
            buckets.push(vec![row.clone()]);
        }
    }
    let rows: Vec<CotRow> = buckets.into_iter().flatten().collect();

    Ok(CotTableState {
        columns: state.columns.clone(),
        rows,
        group_key: Some(state.columns[idx].name.clone()),
    })
}

/// Canonical bucket key for a group cell, honouring case sensitivity.
fn group_key_string(cell: Option<&CotCell>, case_sensitive: bool) -> String {
    let raw = cell.map(CotCell::to_display_string).unwrap_or_default();
    if case_sensitive {
        raw
    } else {
        raw.to_lowercase()
    }
}

// ── f_sort_by ────────────────────────────────────────────────────────────────

/// Execute `f_sort_by`: stably order rows by `column`.
fn apply_sort_by(
    state: &CotTableState,
    column: &str,
    ascending: bool,
    config: &ChainOfTableConfig,
) -> ChainOfTableResult<CotTableState> {
    let idx = resolve(state, column, config)?;
    let mut rows = state.rows.clone();
    rows.sort_by(|a, b| {
        let ordering = cot_cell_cmp(
            a.get(idx).unwrap_or(&CotCell::Empty),
            b.get(idx).unwrap_or(&CotCell::Empty),
            config.case_sensitive,
        );
        if ascending {
            ordering
        } else {
            ordering.reverse()
        }
    });

    Ok(CotTableState {
        columns: state.columns.clone(),
        rows,
        group_key: state.group_key.clone(),
    })
}

// ── f_aggregate ──────────────────────────────────────────────────────────────

/// Execute `f_aggregate`: reduce `column` with `agg`.
///
/// On a flat table this collapses to a single scalar cell (a 1×1 table). On a
/// grouped table it collapses each group to one row `[group value, aggregate]`
/// and clears the group key.
fn apply_aggregate(
    state: &CotTableState,
    column: &str,
    agg: CotAggregate,
    config: &ChainOfTableConfig,
) -> ChainOfTableResult<CotTableState> {
    let idx = resolve(state, column, config)?;

    if let Some(group_column) = state.group_key.as_ref() {
        aggregate_grouped(state, group_column, idx, agg, config)
    } else {
        let cells: Vec<&CotCell> = state.rows.iter().filter_map(|row| row.get(idx)).collect();
        let value = compute_aggregate(&cells, agg, column)?;
        Ok(CotTableState::new(
            vec![CotColumn::number(agg.as_str())],
            vec![CotRow::new(vec![CotCell::Number(value)])],
        ))
    }
}

/// Aggregate within groups: one output row per distinct group value, in
/// first-appearance order.
fn aggregate_grouped(
    state: &CotTableState,
    group_column: &str,
    value_idx: usize,
    agg: CotAggregate,
    config: &ChainOfTableConfig,
) -> ChainOfTableResult<CotTableState> {
    let group_idx = resolve(state, group_column, config)?;

    let mut order: Vec<String> = Vec::new();
    let mut representative: Vec<CotCell> = Vec::new();
    let mut group_cells: Vec<Vec<&CotCell>> = Vec::new();
    for row in &state.rows {
        let key = group_key_string(row.get(group_idx), config.case_sensitive);
        let position = if let Some(pos) = order.iter().position(|existing| existing == &key) {
            pos
        } else {
            order.push(key);
            representative.push(row.get(group_idx).cloned().unwrap_or(CotCell::Empty));
            group_cells.push(Vec::new());
            order.len() - 1
        };
        if let Some(cell) = row.get(value_idx) {
            group_cells[position].push(cell);
        }
    }

    let column_name = state.columns[value_idx].name.clone();
    let mut rows: Vec<CotRow> = Vec::with_capacity(order.len());
    for (group_value, cells) in representative.into_iter().zip(group_cells) {
        let value = compute_aggregate(&cells, agg, &column_name)?;
        rows.push(CotRow::new(vec![group_value, CotCell::Number(value)]));
    }

    let columns = vec![
        state.columns[group_idx].clone(),
        CotColumn::number(agg.as_str()),
    ];
    Ok(CotTableState::new(columns, rows))
}

/// Compute a single aggregate value over `cells`.
///
/// `Count` counts non-empty cells (of any type); `Sum`/`Mean`/`Max`/`Min`
/// operate on numeric cells and reject a text cell with
/// [`ChainOfTableError::TypeMismatch`]. `Mean`/`Max`/`Min` over no numeric
/// values yield [`ChainOfTableError::EmptyAggregate`]; `Sum` over no values is
/// `0`.
#[allow(clippy::cast_precision_loss)]
fn compute_aggregate(
    cells: &[&CotCell],
    agg: CotAggregate,
    column_name: &str,
) -> ChainOfTableResult<f64> {
    let type_mismatch = || ChainOfTableError::TypeMismatch {
        operation: "f_aggregate".to_string(),
        column: column_name.to_string(),
    };

    match agg {
        CotAggregate::Count => Ok(cells.iter().filter(|cell| !cell.is_empty()).count() as f64),
        CotAggregate::Sum => {
            let mut sum = 0.0;
            for cell in cells {
                match cell {
                    CotCell::Number(value) => sum += value,
                    CotCell::Empty => {}
                    CotCell::Text(_) => return Err(type_mismatch()),
                }
            }
            Ok(sum)
        }
        CotAggregate::Mean => {
            let mut sum = 0.0;
            let mut count = 0usize;
            for cell in cells {
                match cell {
                    CotCell::Number(value) => {
                        sum += value;
                        count += 1;
                    }
                    CotCell::Empty => {}
                    CotCell::Text(_) => return Err(type_mismatch()),
                }
            }
            if count == 0 {
                return Err(ChainOfTableError::EmptyAggregate {
                    aggregate: agg.as_str().to_string(),
                    column: column_name.to_string(),
                });
            }
            Ok(sum / count as f64)
        }
        CotAggregate::Max | CotAggregate::Min => {
            let mut best: Option<f64> = None;
            for cell in cells {
                match cell {
                    CotCell::Number(value) => {
                        best = Some(match best {
                            Some(current) => reduce_extremum(current, *value, agg),
                            None => *value,
                        });
                    }
                    CotCell::Empty => {}
                    CotCell::Text(_) => return Err(type_mismatch()),
                }
            }
            best.ok_or_else(|| ChainOfTableError::EmptyAggregate {
                aggregate: agg.as_str().to_string(),
                column: column_name.to_string(),
            })
        }
    }
}

/// Fold a running extremum for [`CotAggregate::Max`] / [`CotAggregate::Min`].
fn reduce_extremum(current: f64, candidate: f64, agg: CotAggregate) -> f64 {
    match agg {
        CotAggregate::Max => {
            if candidate > current {
                candidate
            } else {
                current
            }
        }
        _ => {
            if candidate < current {
                candidate
            } else {
                current
            }
        }
    }
}
