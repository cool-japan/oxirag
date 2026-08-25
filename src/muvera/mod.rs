//! MUVERA — Multi-Vector Retrieval via Fixed Dimensional Encodings
//! (Dhulipala et al., 2024).
//!
//! Late-interaction models (`ColBERT`, `ColPali`, …) represent a document or a
//! query as a **set of token vectors** and score a pair by the Chamfer /
//! `MaxSim` similarity
//!
//! ```text
//! Chamfer(Q, D) = Σ_{q∈Q} max_{d∈D} ⟨q, d⟩ .
//! ```
//!
//! MUVERA collapses each such *set* into a **single** fixed-dimensional vector,
//! its *Fixed Dimensional Encoding* (FDE), engineered so that the ordinary dot
//! product of two FDEs approximates the Chamfer similarity of the underlying
//! sets:
//!
//! ```text
//! ⟨FDE_query(Q), FDE_doc(D)⟩ ≈ Chamfer(Q, D) .
//! ```
//!
//! Multi-vector retrieval is thereby reduced to a **single-vector maximum
//! inner-product search (MIPS)**: no per-query `MaxSim` scan is ever required.
//!
//! # How the FDE is built
//!
//! 1. **`SimHash` space partition.** `k_sim` deterministic Gaussian hyperplanes
//!    (seeded, no RNG crate) assign each token a `k_sim`-bit cell id, splitting
//!    the space into `B = 2^k_sim` cells.
//! 2. **Asymmetric per-cell aggregation.** Within each cell the **query** side
//!    **sums** its tokens while the **document** side **averages** its tokens;
//!    empty document cells optionally borrow the Hamming-nearest non-empty
//!    centroid. This asymmetry is precisely what makes the FDE dot product
//!    telescope into `Σ_{q∈Q} ⟨q, centroid(cell(q))⟩ ≈ Chamfer(Q, D)`.
//! 3. **Inner projection.** Each cell block is optionally projected to `d_proj`
//!    dimensions by a deterministic `±1` Johnson–Lindenstrauss matrix (scaled
//!    `1/√d_proj`), which preserves inner products in expectation.
//! 4. **Repetitions.** Steps 1–3 are repeated `r_reps` times with independent
//!    seeds and concatenated, reducing the estimator's variance. The final
//!    dimension is `r_reps · 2^k_sim · inner_dim`.
//!
//! See [`fde`] for the construction and [`index`] for the store + search.
//!
//! # How this differs from its neighbours in the crate
//!
//! * **vs [`crate::plaid_retrieval`]** — PLAID *keeps* the multi-vector
//!   representation and still performs a full `MaxSim` late-interaction scan at
//!   query time (it only *prunes* candidates with centroid probing). MUVERA
//!   instead *eliminates* `MaxSim` from the query path altogether: the whole
//!   token set becomes one FDE and ranking is a single dot product per
//!   document. (An optional exact-Chamfer re-rank of the FDE top candidates is
//!   available for faithful final ordering.)
//! * **vs [`crate::matryoshka`]** — Matryoshka truncates *one* embedding to a
//!   shorter nested prefix (a single-vector-to-single-vector dimensionality
//!   reduction). MUVERA maps a *whole set* of token vectors to one vector so
//!   that a single dot product recovers a set-to-set (Chamfer) similarity —
//!   a different problem entirely.
//!
//! # Quick start
//!
//! ```rust
//! # #[cfg(feature = "muvera")] {
//! use oxirag::muvera::{MuveraConfig, MuveraIndex};
//!
//! // 3-dimensional token vectors, coarse partition, exact re-rank on.
//! let config = MuveraConfig::new(3)
//!     .unwrap()
//!     .with_k_sim(2)
//!     .with_d_proj(0)
//!     .with_r_reps(4)
//!     .with_rerank(true);
//! let mut index = MuveraIndex::new(config).unwrap();
//!
//! // A document whose token set contains a vector identical to the query.
//! index.add("hit", vec![vec![1.0, 0.0, 0.0], vec![0.0, 1.0, 0.0]]).unwrap();
//! // A document whose tokens are all orthogonal to the query.
//! index.add("miss", vec![vec![0.0, 0.0, 1.0]]).unwrap();
//!
//! let query = vec![vec![1.0_f32, 0.0, 0.0]];
//! let hits = index.search(&query, 2).unwrap();
//! assert_eq!(hits[0].doc_id, "hit");
//! # }
//! ```

pub mod fde;
pub mod index;
#[cfg(all(test, not(target_arch = "wasm32")))]
mod tests;
pub mod types;

pub use fde::{FixedDimEncoding, MuveraEncoder};
pub use index::MuveraIndex;
pub use types::{
    DEFAULT_MUVERA_SEED, MAX_K_SIM, MuveraConfig, MuveraDocument, MuveraError, MuveraResult,
    MuveraSimilarity,
};
