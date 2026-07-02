//! Chain-of-Table (Wang et al. 2024, "Chain-of-Table: Evolving Tables in the
//! Reasoning Chain for Table Understanding").
//!
//! Chain-of-Table is a **tabular** reasoning technique. Rather than answering a
//! question about a table in one shot, it evolves an explicit **relational
//! table state** through a *chain* of symbolic, table-mutating operations —
//! filter these rows, derive that column, group by this key, sort, aggregate —
//! until the answer falls out of the transformed table. The chain of
//! intermediate tables *is* the reasoning trace.
//!
//! # Differentiator vs [`program_of_thought`](crate::program_of_thought)
//!
//! Both modules externalise reasoning into an executable artifact, but the
//! artifact is fundamentally different:
//!
//! * `program_of_thought` emits an **arithmetic-scalar DSL** — a little
//!   program of variable assignments over `+ - * /` — and a deterministic
//!   interpreter runs it *over numbers* to produce a single numeric value. The
//!   state it threads is a `name → f64` environment.
//! * `chain_of_table` (this module) evolves a **relational table state**. Its
//!   vocabulary ([`CotTableOperation`]) is a set of *table-to-table*
//!   transforms (`f_add_column`, `f_select_row`, `f_select_column`,
//!   `f_group_by`, `f_sort_by`, `f_aggregate`); the executor
//!   ([`apply`]) maps one [`CotTableState`] to the next. The state it threads
//!   is a whole table (columns + rows), not a scalar environment, and only the
//!   final `f_aggregate` step (optionally) collapses it to a scalar.
//!
//! So `program_of_thought` computes a number *with* a program, whereas
//! `chain_of_table` reasons *by reshaping a table*.
//!
//! # Pieces
//!
//! * [`types`] — the table model ([`CotTableState`], [`CotCell`],
//!   [`CotColumn`], [`CotRow`]), the operation vocabulary
//!   ([`CotTableOperation`], [`CotAddRule`], [`CotPredicate`],
//!   [`CotComparator`], [`CotAggregate`]), configuration
//!   ([`ChainOfTableConfig`], [`CotEnabledOperations`]), and the error type
//!   ([`ChainOfTableError`]).
//! * [`ops`] — the executor [`apply`] / [`apply_with_config`]: apply one
//!   operation to a table state, deterministically.
//! * [`engine`] — [`ChainOfTableEngine`]: the heuristic planner (pick the next
//!   operation from the question and current table) and the drive loop that
//!   builds the chain and extracts a [`CotAnswer`].
//!
//! # Example
//!
//! ```
//! use oxirag::chain_of_table::{ChainOfTableConfig, ChainOfTableEngine, CotTableState};
//!
//! let headers = ["product", "category", "price"];
//! let rows = vec![
//!     vec!["Book A", "Books", "12"],
//!     vec!["Toy B", "Toys", "8"],
//!     vec!["Book C", "Books", "15"],
//! ];
//! let table = CotTableState::parse(&headers, &rows).expect("valid table");
//!
//! let engine = ChainOfTableEngine::new(ChainOfTableConfig::default());
//! let answer = engine
//!     .run("How many products where category is Books?", &table)
//!     .expect("run succeeds");
//!
//! // The chain filtered to category = Books, then counted the rows.
//! assert_eq!(answer.operation_names(), ["f_select_row", "f_aggregate"]);
//! assert_eq!(answer.as_number(), Some(2.0));
//! ```
//!
//! Operations can also be applied directly, without the engine:
//!
//! ```
//! use oxirag::chain_of_table::{
//!     apply, CotAggregate, CotTableOperation, CotTableState,
//! };
//!
//! let table = CotTableState::parse(
//!     &["city", "sales"],
//!     &[vec!["A", "10"], vec!["B", "30"], vec!["A", "20"]],
//! )
//! .expect("valid table");
//!
//! let grouped = apply(
//!     &CotTableOperation::GroupBy { column: "city".to_string() },
//!     &table,
//! )
//! .expect("group_by");
//! let totals = apply(
//!     &CotTableOperation::Aggregate {
//!         column: "sales".to_string(),
//!         agg: CotAggregate::Sum,
//!     },
//!     &grouped,
//! )
//! .expect("aggregate");
//!
//! // One row per city: A → 30, B → 30.
//! assert_eq!(totals.row_count(), 2);
//! ```

pub mod engine;
pub mod ops;
#[cfg(test)]
mod tests;
pub mod types;

pub use engine::ChainOfTableEngine;
pub use ops::{apply, apply_with_config};
pub use types::{
    ChainOfTableConfig, ChainOfTableError, ChainOfTableResult, CotAddRule, CotAggregate, CotAnswer,
    CotAnswerValue, CotCell, CotColumn, CotColumnType, CotComparator, CotEnabledOperations,
    CotOperationKind, CotOperationTrace, CotPredicate, CotRow, CotTableOperation, CotTableState,
    cot_cell_cmp, format_number,
};
