//! Predicate-constrained approximate nearest-neighbor search — ANN retrieval
//! that respects a metadata filter *while* it searches, rather than in spite of
//! it.
//!
//! Every other ANN structure in this crate (`hnsw_index`, `disk_ann`, `spann`,
//! `ivf_index`, `lsh_index`, `rp_tree_index`, `product_quantization`, ...)
//! answers exactly one question: *what are the `k` nearest vectors to `q`?*
//! Real retrieval almost never wants that. It wants the `k` nearest vectors to
//! `q` **that are in Japanese, from after 2022, not deprecated, and visible to
//! this tenant** — and that "and" turns out to be one of the genuinely hard
//! problems in vector search.
//!
//! # Why filtering is hard
//!
//! There are two obvious ways to bolt a filter onto an ANN index, and both are
//! broken in complementary ways.
//!
//! **Post-filtering** searches unconstrained, over-fetches by some fixed
//! multiple `m` of `k`, and throws away the candidates that fail the predicate.
//! It is trivial to implement — which is why it is what almost every system
//! reaches for first, and why it is what this crate's own legacy vector store
//! does today with a hard-coded `m = 10`. It also cannot work. If the predicate
//! is independent of the query's proximity ranking, then of the unconstrained
//! top `k * m`, only about `s * k * m` satisfy it, where `s` is the predicate's
//! selectivity. Filling the result set therefore requires
//!
//! ```text
//! s * k * m >= k    <=>    s >= 1 / m
//! ```
//!
//! With `m = 10`, *any* predicate matching under 10% of the corpus starts
//! returning short. At `s = 1%` the method returns roughly **one** correct hit
//! out of ten — a recall near zero. And the fix is worse than the disease:
//! restoring recall means `m >= 1 / s`, i.e. fetching `k / s` candidates, which
//! at `s = 0.1%` is a thousand `k`. That is a scan wearing a graph search as a
//! disguise.
//!
//! **Pre-filtering** goes the other way: materialize the matching set, scan it
//! exhaustively. This is *exact* — it is, after all, brute force — and it is
//! wonderful when the predicate is selective. It is also linear in the size of
//! the matching set, so a predicate matching 60% of a hundred-million-vector
//! corpus turns your ANN index into a very expensive `for` loop.
//!
//! And the third option — the one that seems most natural — is the one that
//! fails most interestingly. **Just refuse to traverse into non-matching
//! nodes.** Run the ordinary greedy graph search, but skip any neighbor that
//! fails the predicate. This does not work either, and the reason is
//! structural. Greedy graph search works only because the proximity graph is
//! *navigable*: from anywhere, a short path of ever-closer nodes leads to the
//! query. Navigability is a property of the whole graph. Delete the
//! non-matching nodes and what remains is the **induced subgraph** on the
//! matching set — and an induced subgraph of a sparse navigable graph is
//! generally neither navigable nor even *connected*. At selectivity `s`, a node
//! of degree `R` keeps on average `s * R` of its edges; once `s * R < 1` —
//! which for a typical `R = 32` means any predicate more selective than about
//! 3% — the subgraph falls below the percolation threshold and shatters into
//! isolated fragments. The traversal dead-ends after a hop or two. No beam
//! width repairs a disconnected graph.
//!
//! # ACORN: separate what may be *returned* from what may be *traversed*
//!
//! [`FilterStrategy::InFilter`] implements the ACORN family of predicate-aware
//! traversals, whose central insight is that the naive filtered search conflates
//! two entirely different questions:
//!
//! - **What may be returned?** Only nodes satisfying the predicate. Always.
//!   Unconditionally.
//! - **What may be traversed?** *Everything.* A node that fails the predicate is
//!   still a perfectly good stepping stone.
//!
//! So the traversal admits non-matching nodes to its candidate pool as **routing
//! nodes** — expandable, never returnable — and thereby inherits the *whole*
//! graph's navigability. Restricting the results costs nothing structurally,
//! because the results were never what made the graph navigable.
//!
//! On top of that, ACORN adds the **two-hop expansion** that gives the method
//! its teeth. When expanding a node `u`, a neighbor `v` that fails the predicate
//! would ordinarily contribute nothing but a pool slot. Instead the traversal
//! reaches straight *through* `v` and harvests the members of `v`'s own neighbor
//! list that do satisfy the predicate. One expansion thus reaches `O(R^2)`
//! candidates rather than `O(R)` — which is exactly the surplus needed to keep a
//! usable supply of matching candidates flowing when `s * R < 1`. It also
//! rescues matches that would otherwise be lost outright, since `v` may well be
//! evicted from the bounded pool before it is ever popped.
//!
//! The expansion is **adaptive**: it is skipped entirely whenever `u` already
//! has at least [`min_predicate_neighbors`](FilteredSearchConfig::min_predicate_neighbors)
//! matching neighbors, because at that density the predicate subgraph navigates
//! perfectly well on its own and the two-hop tax buys nothing. `InFilter`
//! therefore costs about what an unfiltered search costs when `s` is large, and
//! pays its full `gamma`-fold price only where that price is the difference
//! between results and garbage.
//!
//! And that price is much lower than it looks. The two-hop step computes a
//! distance only for the second-hop nodes that actually *match*, and memoizes
//! its predicate evaluations, so most of its extra work is walking neighbor
//! lists already in cache. Measured on this module's own fixture at 1%
//! selectivity, raising
//! [`neighbor_expansion_gamma`](FilteredSearchConfig::neighbor_expansion_gamma)
//! from 3 to 16 moved recall@10 from **0.77 to 0.96** for a mere 10% increase in
//! distance computations (630 to 696). Being timid with `gamma` is the expensive
//! choice.
//!
//! The traversal also applies a *matching-reserve* rule, which stops routing
//! nodes — which cluster tightly around the query, being simply its nearest
//! neighbors — from monopolizing every slot of the bounded candidate pool and
//! leaving the discovered matches forever unexpanded. Reserving a slice of the
//! pool for matching nodes guarantees the search actually *follows* the matches
//! it finds into the region of the graph where their (likely also matching)
//! neighbors live.
//!
//! # Choosing between the three
//!
//! None of the three strategies dominates. [`StrategySelector`] estimates the
//! predicate's selectivity *before touching a single vector* — from per-attribute
//! value counts and equi-width histograms, composed over the predicate AST — and
//! routes each query to the plan that suits it:
//!
//! | Estimated `s` | Plan | Distance computations | Recall |
//! |---|---|---|---|
//! | low (`<= 1%` **and** few enough matches to scan) | [`PreFilter`](FilterStrategy::PreFilter) | `~ s * N` | **1.0**, exact by construction |
//! | middling | [`InFilter`](FilterStrategy::InFilter) | `~ L * R * gamma` | high |
//! | high (`>= 80%`) | [`PostFilter`](FilterStrategy::PostFilter) | `~ max(L, k*m) * R` | `~ min(1, s*m)` |
//!
//! The regimes are not arbitrary; each is the region where the chosen plan's
//! weakness does not bite:
//!
//! - At **low `s`**, `PostFilter` is broken (see above) and `InFilter` is at its
//!   most expensive (almost no neighbor matches, so nearly every expansion
//!   triggers the two-hop step, and the traversal cannot terminate early because
//!   its result heap may never fill). Meanwhile `s * N` is *small* — so the
//!   exhaustive scan is both cheaper **and** exact. It wins outright. Note the
//!   gate is on the ratio *and* the absolute count: `s <= 1%` means ten records
//!   on a corpus of a thousand and ten million on a corpus of a billion, and
//!   only one of those is worth scanning.
//! - At **high `s`**, almost everything matches, so the unconstrained top
//!   `k * m` is almost entirely made of matches; filtering it discards almost
//!   nothing, and `PostFilter`'s recall approaches the graph's *unfiltered*
//!   recall, which is the best any plan can do. [`FilteredSearchConfig::validate`]
//!   enforces `postfilter_threshold >= 1 / post_filter_multiplier`, which is
//!   precisely the condition making the over-fetch expected to hold `k` matches
//!   — the planner is structurally incapable of routing a query into
//!   `PostFilter`'s failure regime.
//! - **In between**, there are too many matches to scan and too few for a
//!   bounded over-fetch to capture reliably. This is the regime ACORN was
//!   invented for.
//!
//! **A mis-estimate costs time, never correctness.** Every plan verifies the
//! real predicate against the real metadata before admitting a hit, so a bad
//! estimate can only pick a slower plan (or, for `PostFilter`, a plan that
//! returns short). It can never return a vector that fails the filter.
//!
//! # Guarantees
//!
//! - **Soundness, unconditional.** Every hit returned by every strategy
//!   satisfies the predicate.
//! - **`PreFilter` is exact.** It returns the true top-`k` of the matching set,
//!   always. The inverted index may hand it a superset of that set or refuse to
//!   answer at all; the subsequent verification pass makes correctness
//!   independent of the resolver's cleverness — only the *cost* depends on that.
//! - **`InFilter` is exhaustive in the limit.** Routing nodes let the frontier
//!   reach every node in the entry points' connected component, so with a large
//!   enough beam and visit ceiling it too returns the exact filtered top-`k`.
//!   Its recall is a function of *budget*, not a structural ceiling — which is
//!   exactly what the naive filtered traversal cannot claim, since no budget
//!   reconnects a shattered subgraph.
//! - **`PostFilter` is bounded by its over-fetch.** Its recall cannot exceed
//!   `min(1, s * m)` in expectation, and the module's headline test measures
//!   precisely that collapse.
//! - **Determinism.** All randomness comes from a hand-rolled `SplitMix64` seeded
//!   from [`FilteredSearchConfig::seed`]; ties are broken by ascending node id
//!   everywhere. Identical inputs produce identical indexes and identical
//!   rankings.
//!
//! # Selectivity estimation
//!
//! [`SelectivityEstimator`] keeps, per attribute, exact value counts up to a
//! cardinality cap (a most-common-values list) and an equi-width histogram over
//! the numeric values. Leaf selectivities come from those; `And`, `Or` and `Not`
//! are folded together with the standard **independence assumption**
//! (`s(And) = Π s(i)`, `s(Or) = 1 - Π (1 - s(i))`, `s(Not) = 1 - s`). For
//! correlated attributes that assumption is arbitrarily wrong, and the estimate
//! says so: [`SelectivityEstimate::independence_assumed`] and
//! [`SelectivityEstimate::exact_leaves`] tell a caller exactly how much to trust
//! the number. See [`selectivity`] for the details and for the surprisingly many
//! cases in which a histogram range estimate is nonetheless *provably exact*.
//!
//! # Quick start
//!
//! ```rust
//! # #[cfg(feature = "filtered-vector-search")]
//! # {
//! use oxirag::filtered_vector_search::{
//!     FilterPredicate, FilterStrategy, FilteredMetadata, FilteredSearchConfig,
//!     FilteredVectorIndex, FilteredVectorRecord,
//! };
//!
//! // A tiny corpus: three vectors, two languages.
//! let records = vec![
//!     FilteredVectorRecord::new(
//!         "doc-en",
//!         vec![1.0, 0.0],
//!         FilteredMetadata::new().with("lang", "en").with("year", 2024_i64),
//!     ),
//!     FilteredVectorRecord::new(
//!         "doc-ja",
//!         vec![0.8, 0.2],
//!         FilteredMetadata::new().with("lang", "ja").with("year", 2023_i64),
//!     ),
//!     FilteredVectorRecord::new(
//!         "doc-ja-old",
//!         vec![0.0, 1.0],
//!         FilteredMetadata::new().with("lang", "ja").with("year", 2019_i64),
//!     ),
//! ];
//!
//! let index = FilteredVectorIndex::build(2, FilteredSearchConfig::default(), records)
//!     .expect("records are well-formed");
//!
//! // "Japanese, from 2020 onwards."
//! let predicate = FilterPredicate::and([
//!     FilterPredicate::eq("lang", "ja"),
//!     FilterPredicate::at_least("year", 2020.0),
//! ]);
//!
//! // The planner reports what it would do, and why, without running anything.
//! let (strategy, estimate) = index.plan(&predicate, 5).expect("valid predicate");
//! assert!(estimate.selectivity > 0.0);
//! assert_ne!(strategy, FilterStrategy::Auto);
//!
//! let hits = index.search(&[1.0, 0.0], 5, &predicate).expect("valid query");
//!
//! // "doc-en" is nearest, but it fails the filter; "doc-ja-old" passes `lang`
//! // but fails `year`. Only "doc-ja" survives both.
//! assert_eq!(hits.len(), 1);
//! assert_eq!(hits[0].id, "doc-ja");
//! # }
//! ```

mod attr_index;
mod graph;

pub mod index;
pub mod predicate;
pub mod selectivity;
pub mod types;

#[cfg(all(test, not(target_arch = "wasm32")))]
mod tests;

pub use index::{FilteredVectorIndex, FilteredVectorRecord};
pub use predicate::FilterPredicate;
pub use selectivity::{
    FilteredAttributeStats, FilteredNumericHistogram, SelectivityEstimator, StrategySelector,
};
pub use types::{
    AttrValue, FilterBound, FilterStrategy, FilteredDistanceMetric, FilteredHit, FilteredMetadata,
    FilteredSearchConfig, FilteredSearchError, FilteredSearchStats, SelectivityEstimate,
};
