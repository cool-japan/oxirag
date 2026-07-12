//! Exact top-k early termination over inverted lists: the WAND family.
//!
//! Ranked retrieval over an inverted index has an obvious, embarrassing
//! inefficiency. To answer "give me the ten best documents" the textbook
//! algorithm scores *every* document that contains *any* query term — often
//! hundreds of thousands of them — and then throws all but ten away. The
//! ninety-nine-point-something percent of that work spent on documents that
//! were never going to place is not merely wasted; for a common term like
//! "the" it dominates the query entirely.
//!
//! **Dynamic pruning** removes that waste *without changing the answer*. The
//! trick is to precompute, for every term, a cheap ceiling on how much that
//! term could ever contribute to any document's score. Once the algorithm holds
//! `k` documents in hand, their k-th best score `θ` becomes a bar: any document
//! whose *ceiling* falls below `θ` can be discarded without ever being scored,
//! because even in its best case it loses to `k` documents already found. As
//! `θ` climbs, the bar rises, and progressively more of the index becomes
//! provably irrelevant and can be leapt over.
//!
//! This module implements the three classical members of that family, plus the
//! exhaustive scan they are validated against:
//!
//! | [`PruningStrategy`] | Bound used | What it skips |
//! |---|---|---|
//! | [`Exhaustive`](PruningStrategy::Exhaustive) | none | nothing — the ground truth |
//! | [`Wand`](PruningStrategy::Wand) | global per-term ceiling `U_t` | every document below the pivot |
//! | [`BlockMaxWand`](PruningStrategy::BlockMaxWand) | per-block ceiling `U_{t,β}` | whole blocks of postings at a time |
//! | [`MaxScore`](PruningStrategy::MaxScore) | prefix sums of the `U_t` | entire *lists*, as candidate generators |
//!
//! # The load-bearing invariant: these are **exact**, not approximate
//!
//! All four strategies return the **identical ranked top-k** — the same
//! document ids, in the same order, with the same scores. This is not an
//! aspiration or a quality metric; it is the definition of correctness for this
//! module, and it is what separates dynamic pruning from the approximate
//! shortcuts (sampling, static index pruning, blunt early cut-offs) that
//! superficially resemble it. A pruning strategy earns its speed by proving
//! work *unnecessary*, never by guessing that it is unimportant.
//!
//! Two design decisions carry that invariant, and both are worth stating
//! plainly, because a plausible-looking implementation gets them wrong and then
//! "almost always" agrees with the baseline:
//!
//! ### 1. Ties are resolved by a strict total order
//!
//! Results are ranked by descending score and then by **ascending insertion
//! ordinal**. Ordinals are unique, so no two candidates ever compare equal: the
//! top-k is a uniquely determined sequence, not one of several acceptable
//! answers. Without such a rule, two strategies that reach tied documents in a
//! different order would return different — and both "correct" — rankings, and
//! the exactness claim would quietly become untestable.
//!
//! The tie-break interacts with the pruning test in a way worth stating
//! carefully, because it is easy to get the reasoning backwards. All four
//! traversals visit documents in strictly *ascending* ordinal order, so every
//! document already in the heap has a smaller ordinal than any candidate still
//! to come — which means a candidate that merely *ties* the k-th best score
//! always loses the tie-break, and could in principle be pruned. The test used
//! here is nonetheless `ceiling ≥ θ` rather than `ceiling > θ`, because the
//! exactness of `>` would rest on a coincidence between the direction of the
//! tie-break and the direction of the traversal, and would silently break if
//! either ever changed. `≥` is unconditionally safe.
//!
//! ### 2. Scores are summed in a canonical order
//!
//! Floating-point addition is not associative, and the strategies naturally
//! accumulate a document's term contributions in three different orders — query
//! order, cursor order, essential-then-non-essential order. Left alone, that
//! yields scores differing in the last bit, which is enough to reorder
//! genuinely-tied documents and break the invariant. Every full evaluation
//! therefore routes through a fixed register file indexed by query position, and
//! the score is always the left-to-right sum over it, with absent terms
//! contributing an exact `+0.0`. All four strategies consequently produce
//! **bit-for-bit identical** scores, and the equivalence tests can assert on
//! rank order rather than merely on score proximity.
//!
//! The pruning comparisons themselves absorb a small relative tolerance
//! (`1e-9`) because they compare a *sum of ceilings* against a *sum of scores*,
//! and those two sums round independently. That relaxation is strictly
//! conservative — it can only cause the traversal to score *more* documents than
//! strictly necessary, never fewer — so exactness survives it untouched.
//!
//! # The ceilings
//!
//! Scoring is BM25 (see [`bm25_idf`] and [`bm25_term_score`]), computed inside
//! this module over this module's own index. Each posting `(d, tf)` of term `t`
//! carries its precomputed contribution `s_t(d)`, and both ceilings are exact
//! maxima over *those very `f64` values* — never analytic over-estimates — so
//!
//! ```text
//! s_t(d)  ≤  U_{t, β(d)}  ≤  U_t
//! ```
//!
//! holds in `f64` with zero slack, where `β(d)` is the block holding `d`. The
//! full skipping proofs are written out in the source of the two traversal
//! files: `wand.rs` carries WAND's pivot argument and Block-Max WAND's
//! whole-block jump, and `maxscore.rs` carries the essential/non-essential split
//! and its early-abandonment rule.
//!
//! # Relationship to the crate's other lexical retrievers
//!
//! This module is deliberately self-contained, because dynamic pruning is a
//! property of the *access path*, and none of the crate's existing lexical
//! retrievers has one:
//!
//! (The three modules named below sit behind their own feature flags, so they
//! are referenced here by name rather than by link.)
//!
//! | Module | Layout | Why it cannot prune |
//! |---|---|---|
//! | `sparse_retrieval` | `Vec<(DocumentId, SparseVector)>` | not an inverted index at all: search is a full dot product against every document |
//! | `bm25f_retrieval` | forward index | search loops over every document exhaustively |
//! | `coil_retrieval` | genuine postings | but scores every matching posting, with no ceiling to skip against |
//!
//! None of them stores a per-term score ceiling, a block-max summary, or
//! document-ordinal-sorted postings that a cursor could leap through — so
//! `dynamic_pruning` builds its own inverted index rather than retrofitting
//! theirs.
//!
//! # Example
//!
//! ```
//! use oxirag::dynamic_pruning::{DynamicPruningIndex, PruningConfig, PruningStrategy};
//!
//! # fn main() -> Result<(), Box<dyn std::error::Error>> {
//! let config = PruningConfig {
//!     top_k: 2,
//!     ..PruningConfig::default()
//! };
//! let mut index = DynamicPruningIndex::new(config)?;
//! index.add_document("a", ["rust", "systems", "programming"])?;
//! index.add_document("b", ["rust", "rust", "ownership"])?;
//! index.add_document("c", ["python", "scripting"])?;
//! index.build();
//!
//! let query = ["rust", "ownership"];
//! let baseline = index.search_with(&query, PruningStrategy::Exhaustive)?;
//! let pruned = index.search_with(&query, PruningStrategy::BlockMaxWand)?;
//!
//! // Exactly the same answer -- that is the whole point.
//! assert_eq!(pruned.document_ids(), vec!["b", "a"]);
//! assert_eq!(baseline.document_ids(), pruned.document_ids());
//! assert_eq!(baseline.hits[0].score, pruned.hits[0].score);
//!
//! // The baseline never skips anything; that is what it is for.
//! assert_eq!(baseline.stats.postings_skipped, 0);
//! # Ok(())
//! # }
//! ```
//!
//! # References
//!
//! * Broder, Carmel, Herscovici, Soffer & Zien (2003), *Efficient Query
//!   Evaluation using a Two-Level Retrieval Process* — WAND.
//! * Ding & Suel (2011), *Faster Top-k Document Retrieval Using Block-Max
//!   Indexes* — Block-Max WAND.
//! * Turtle & Flood (1995), *Query Evaluation: Strategies and Optimizations* —
//!   `MaxScore`.

mod cursor;
pub mod index;
mod maxscore;
pub mod types;
mod wand;

#[cfg(test)]
mod tests;

pub use index::{
    DynamicPruningIndex, PruningBlock, PruningPosting, PruningPostingList, bm25_idf,
    bm25_term_score,
};
pub use types::{
    DynamicPruningError, DynamicPruningResult, PruningConfig, PruningHit, PruningSearchResult,
    PruningStats, PruningStrategy,
};
