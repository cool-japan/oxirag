//! Closed-form knowledge editing: a **locate-then-edit** rank-1 update of a parametric linear
//! memory, plus a persistent, **non-evicting** discrete edit codebook consulted at inference.
//!
//! # What this module does
//!
//! A transformer's `MLP` down-projection behaves as a linear associative memory: a
//! post-activation vector `k` is a *key*, the matrix–vector product `W k` is the *value* read
//! out of it, and a fact lives in `W` as the association between one key and one value. To make
//! the model assert a new fact, you can therefore edit `W` directly, in closed form, so that it
//! answers a chosen value `v*` for a chosen key `k*` while disturbing its answers everywhere else
//! as little as possible. That is the `ROME` construction (Meng et al., 2022, *Locating and
//! Editing Factual Associations in GPT*), and its many-edits-at-once generalization is `MEMIT`
//! (Meng et al., 2023). This module implements both, in `f64`, with the collateral damage
//! **measured** rather than assumed:
//!
//! 1. **The rank-1 edit** ([`RankOneEdit`]). Solves
//!    `min tr(Delta C Delta^T)` subject to `(W + Delta) k* = v*`, whose closed form is the
//!    rank-1 matrix
//!    `Delta = (v* - W k*) (C^-1 k*)^T / (k*^T C^-1 k*)`, where `C = E[k k^T]` is the
//!    second-moment matrix of the keys the edit must *preserve*. The two post-conditions
//!    `W' k* = v*` (exactly) and `||W' k_i - W k_i|| <= ||v* - W k*|| * sqrt(q_i / q)` for every
//!    preserved key (a **derived**, tight Cauchy–Schwarz bound) are what make the edit real, and
//!    they are asserted in the tests.
//!
//! 2. **The non-evicting codebook** ([`EditCodebook`]). A `GRACE`/`SERAC`-style discrete
//!    key → value memory consulted at inference: a query within a deferral radius `epsilon` of an
//!    edit key is served the edited value; otherwise the system defers to the base model. An edit
//!    is **never silently evicted** — there is no capacity and no `TTL`, only an explicit,
//!    by-id [`EditCodebook::retract`].
//!
//! 3. **The sequential / batched multi-edit path** ([`EditBatch`] → [`MultiEdit`], the `MEMIT`
//!    construction). Applies many edits and **bounds the accumulated drift** on the preserved
//!    keys — which is exactly where multi-edit methods fail in practice, so the module measures
//!    it and reports the drift curve rather than trusting that it stays small.
//!
//! # Distinct from
//!
//! Distinct from `knowledge_unlearning` (which *deletes documents from a store* and re-audits for
//! residual leakage) and from `belief_revision` (which updates a *posterior over a fixed
//! hypothesis set*): `knowledge_editing` performs a closed-form **rank-1 edit of a parametric
//! linear memory matrix** — with the exact post-conditions `W' k* = v*` and `W' k_i ~= W k_i` for
//! preserved keys — plus a persistent, **non-evicting** discrete edit codebook consulted at
//! inference within a deferral radius. Unlike `semantic_cache`, which is structurally a
//! threshold-gated key → value lookup but is an `LRU` *memoization* keyed on the query embedding
//! and never alters what the system believes, an edit here must never be silently evicted.
//!
//! # Design notes
//!
//! * **`C^-1` is never formed.** `C = E[k k^T]` is symmetric positive definite, so it is factored
//!   once by Cholesky and applied by triangular substitution forever after. This is faster, more
//!   accurate, and — critically — makes the edit's denominator `k^T C^-1 k` a **sum of squares**
//!   (`||L^-1 k||^2`), hence structurally non-negative rather than clamped. See [`linalg`].
//! * **The linear algebra is hand-rolled** rather than imported from
//!   `crate::bandit_ranker`, whose `linalg` module contains the very same kernels. The two are
//!   independent Cargo features; importing across them would silently entangle them, exactly as
//!   `knowledge_unlearning` hand-rolls its own `MinHash` rather than depending on
//!   `semantic_dedup`. The error type is [`KnowledgeEditLinalgError`], *not* `LinalgError`, which
//!   is already taken by the prelude.
//! * **Every headline number is a measurement.** Post-condition residuals, collateral drift, and
//!   cumulative cost are all read back out of the committed memory, and a write that fails its
//!   post-condition is **rolled back** — the editor never returns a memory it has not verified.
//!
//! # Quick start
//!
//! ```rust
//! # #[cfg(feature = "knowledge-editing")]
//! # {
//! use oxirag::knowledge_editing::{
//!     EditConfig, EditKey, EditRequest, EditValue, EditableMemory, KnowledgeEditor,
//! };
//!
//! // A 2 -> 2 linear memory that currently maps everything through the identity, with a
//! // handful of preserved keys whose readouts the edit must leave alone.
//! let preserved = vec![vec![1.0, 0.0], vec![0.0, 1.0], vec![1.0, 1.0]];
//! let site = EditableMemory::from_preserved_keys(
//!     "mlp.down_proj",
//!     vec![1.0, 0.0, 0.0, 1.0], // row-major 2x2 identity
//!     2,
//!     2,
//!     &preserved,
//!     1e-6,
//! )
//! .expect("valid site");
//!
//! let mut editor =
//!     KnowledgeEditor::new(EditConfig::new(2, 2), vec![site]).expect("valid config");
//!
//! // Make the memory answer [9, 9] for the key [2, -1].
//! let key = EditKey::new(vec![2.0, -1.0]).unwrap();
//! let value = EditValue::new(vec![9.0, 9.0]).unwrap();
//! let result = editor
//!     .apply(&EditRequest::new(key.clone(), value.clone()))
//!     .expect("edit applies");
//!
//! // The post-condition holds to machine precision, and the drift on the preserved keys is
//! // bounded by the C-weighted norm of the delta.
//! assert!(result.postcondition_residual < 1e-9);
//!
//! // At inference the edited key is served exactly; a far-away key defers to the memory.
//! let served = editor.read(0, key.as_slice()).unwrap();
//! assert!(served.verdict.is_served());
//! # }
//! ```

pub mod codebook;
pub mod editor;
pub mod linalg;
pub mod memory;
pub mod rank_one;
#[cfg(test)]
mod tests;
pub mod types;

pub use codebook::EditCodebook;
pub use editor::KnowledgeEditor;
pub use linalg::KnowledgeEditLinalgError;
pub use memory::EditableMemory;
pub use rank_one::{EditBatch, MultiEdit, RankOneEdit};
pub use types::{
    CodebookStats, DEFAULT_COVARIANCE_RIDGE, DEFAULT_DEFERRAL_RADIUS,
    DEFAULT_POSTCONDITION_TOLERANCE, EditConfig, EditId, EditKey, EditRead, EditRecord,
    EditRequest, EditResult, EditScope, EditStrategy, EditValue, EditVerdict, KnowledgeEditError,
};
