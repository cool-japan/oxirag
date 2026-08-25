//! `TableRAG` (Chen et al. 2024, "`TableRAG`: Million-Token Table Understanding
//! with Language Models") — two-stage **retrieval** over existing large
//! tables: schema retrieval, then cell retrieval, to select a relevant
//! sub-table small enough to fit an LLM's context.
//!
//! # Not `structured_extraction`: opposite direction
//!
//! It is easy to conflate this module with [`crate::structured_extraction`],
//! but the two solve inverse problems:
//!
//! - [`crate::structured_extraction`] goes **free-text → typed records**: it
//!   reads unstructured prose and pulls out a handful of typed fields
//!   (`price`, `date`, ...) via keyword-proximity heuristics, *creating*
//!   small structured records that did not exist as data before.
//! - `table_rag` goes the other way: it starts from data that is *already*
//!   structured — one or more large [`TableRagTable`]s, each with a
//!   [`TableSchema`] and many rows — and *retrieves* a small, relevant
//!   **sub-table** out of it. Nothing is extracted or typed; the input and
//!   output are both tables, just different sizes.
//!
//! # Not `chain_of_table`: retrieval, not a manipulation chain
//!
//! `table_rag` is also distinct from its sibling `chain_of_table`
//! (Wang et al. 2024, "`Chain-of-Table`"), which has an LLM iteratively
//! *transform* a table through an explicit chain of atomic operations
//! (`add_column`, `select_row`, `group_by`, ...) as a visible reasoning
//! trace. `table_rag` performs no iterative transformation and keeps no
//! operation trace: it is a single two-stage **retrieval** pass — schema
//! retrieval, then cell retrieval — that selects a sub-table once, up front,
//! as context for a downstream generator.
//!
//! # Two-stage retrieval
//!
//! 1. **Schema retrieval** ([`TableRagEngine::query`], stage 1). Every
//!    column of every [`TableRagTable`] in the [`TableRagIndex`] is scored
//!    against the query using a weighted mix of lexical (token-overlap +
//!    column-type-keyword) and pseudo-embedding cosine-similarity signals
//!    (see [`TableRagConfig::lexical_weight`] /
//!    [`TableRagConfig::embedding_weight`]). The table containing the single
//!    highest-scoring column is selected, and that table's own columns are
//!    ranked and filtered down to
//!    [`TableRagConfig::top_k_columns`] relevant [`TableColumnSpec`]s.
//! 2. **Cell retrieval** (stage 2). The query is expanded into candidate
//!    [`CellProbe`] values (unigrams, bigrams, and the full query text) and
//!    matched against each relevant column's *distinct-value* dictionary —
//!    a deterministic, first-seen-order encoding capped at
//!    [`TableRagConfig::max_distinct_values_per_column`] that bounds
//!    matching cost independently of row count. Matching probes vote for
//!    the rows holding their matched value; the highest-voted
//!    [`TableRagConfig::top_k_rows`] rows are kept.
//! 3. **Sub-table assembly.** The relevant columns and relevant rows are
//!    intersected into a compact [`TableRagSubTable`], carrying full
//!    provenance ([`TableRagSubTable::table_name`],
//!    [`TableRagSubTable::column_indices`],
//!    [`TableRagSubTable::row_indices`]) back to the source table, plus a
//!    serialized [`TableRagSubTable::to_encoded_text`] rendering for
//!    downstream RAG consumption.
//!
//! Everything is pure Rust, `std`-only, and fully deterministic: the
//! pseudo-embeddings are `FNV-1a` bucket histograms (no `rand`, no ML
//! runtime), so identical input always produces an identical
//! [`TableRagResult`].
//!
//! # Quick start
//!
//! ```
//! use oxirag::table_rag::{
//!     TableColumnSpec, TableColumnType, TableRagEngine, TableRagIndex, TableRagTable,
//!     TableSchema,
//! };
//!
//! let schema = TableSchema::new(vec![
//!     TableColumnSpec::new("name", TableColumnType::Text),
//!     TableColumnSpec::new("department", TableColumnType::Text),
//!     TableColumnSpec::new("salary", TableColumnType::Number),
//! ]);
//! let table = TableRagTable::new(
//!     "employees",
//!     schema,
//!     vec![
//!         vec!["Alice".to_string(), "Engineering".to_string(), "95000".to_string()],
//!         vec!["Bob".to_string(), "Sales".to_string(), "65000".to_string()],
//!     ],
//! )
//! .expect("valid table");
//!
//! let index = TableRagIndex::new().with_table(table);
//! let engine = TableRagEngine::with_default_config(index);
//! let result = engine
//!     .query("Which employees work in the Engineering department?")
//!     .expect("query should succeed");
//!
//! assert_eq!(result.sub_table.table_name, "employees");
//! assert!(result.sub_table.row_count() >= 1);
//! assert!(result.encoded_text.contains("Engineering"));
//! ```

pub mod engine;
#[cfg(all(test, not(target_arch = "wasm32")))]
mod tests;
pub mod types;

pub use engine::TableRagEngine;
pub use types::{
    CellProbe, TableColumnSpec, TableColumnType, TableRagConfig, TableRagError, TableRagIndex,
    TableRagResult, TableRagSubTable, TableRagTable, TableSchema,
};
