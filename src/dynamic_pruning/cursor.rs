//! Shared traversal machinery: posting-list cursors, the canonical score
//! accumulator, the top-k heap, and the threshold comparison that every
//! pruning decision funnels through.
//!
//! Two invariants live here, and the exactness of the whole module rests on
//! both of them.
//!
//! # 1. The canonical summation order ([`ScoreAccumulator`])
//!
//! Floating-point addition is not associative. `Exhaustive` naturally sums a
//! document's term contributions in query order; WAND sums them in *cursor*
//! order (which is sorted by document id and therefore query-dependent);
//! `MaxScore` sums the essential terms first and then refines with the
//! non-essential ones in descending-bound order. Three different orders, three
//! results that can differ in the last bit or two.
//!
//! A last-bit difference is not cosmetic: two documents whose scores are
//! *mathematically* tied would then compare differently under `total_cmp`, and
//! the strategies would return the same documents in a different order — an
//! exactness failure.
//!
//! [`ScoreAccumulator`] removes the problem at the root. Every strategy writes
//! each term's contribution into a slot indexed by that term's position in the
//! (de-duplicated) query, and the final score is *always* the left-to-right
//! sum over all slots, absent terms contributing an exact `+0.0`. Since
//! `x + 0.0 == x` bitwise for every non-negative finite `x`, and BM25
//! contributions are non-negative, all four strategies produce **bit-for-bit
//! identical** scores for the same document.
//!
//! # 2. The tolerant threshold ([`can_reach`])
//!
//! Per-posting scores are bounded *exactly*: `s_t(d) ≤ U_{t,β} ≤ U_t` holds in
//! `f64` with no slack, because each bound is a maximum over the very same
//! stored values. But a pruning decision compares a *sum of bounds*
//! `fl(Σ U_t)` against a *sum of scores* `fl(Σ s_t)` (the threshold θ, which is
//! some document's accumulated score), and those two sums round independently.
//! In a pathological case `fl(Σ U_t)` can land a few ulps *below* `Σ s_t`, and
//! a naive `if upper_bound < theta { skip }` would then discard a document
//! that genuinely belongs in the top-k.
//!
//! [`can_reach`] therefore refuses to prune unless the bound is below θ by a
//! margin that comfortably exceeds any achievable summation error. This is a
//! strictly **conservative** relaxation: it can only ever cause the traversal
//! to score *more* documents than the ideal, never fewer, so exactness is
//! preserved by construction while the extra work is immeasurable.

use std::cmp::{Ordering, Reverse};
use std::collections::BinaryHeap;

use crate::dynamic_pruning::index::{EXHAUSTED, PruningPostingList};

/// Relative slack applied to every pruning comparison, to absorb the rounding
/// error of summing a handful of `f64` bounds. Query lengths are far below
/// `1 / (16 · f64::EPSILON) ≈ 2.8 · 10^{14}` terms, so `1e-9` is many orders of
/// magnitude larger than any reachable error and many orders of magnitude
/// smaller than the `1e-6` tolerance at which two scores are considered equal.
pub(crate) const PRUNING_TOLERANCE: f64 = 1e-9;

/// Whether a document whose score is bounded above by `upper_bound` might still
/// beat the current top-k threshold `threshold`, and therefore must be scored.
///
/// `threshold` is `None` while the heap holds fewer than `k` documents — until
/// then *every* document is a candidate and nothing may be pruned.
///
/// The comparison is deliberately generous (`≥ θ - tol` rather than `> θ`) for
/// two reasons:
///
/// * **Rounding — the necessary one.** See the module comment: `fl(Σ U_t)` and
///   `fl(Σ s_t)` round independently, so a ceiling that bounds a score in exact
///   arithmetic can come out a few ulps *below* it in `f64`. A bare
///   `upper_bound < θ ⇒ skip` would then discard a document that genuinely
///   belongs in the top-k. This is the reason the tolerance exists, and
///   `internals::can_reach_absorbs_summation_rounding` pins it down.
///
/// * **Independence from the traversal order — the defensive one.** It is worth
///   being precise about what `≥` is *not* needed for. One might argue that a
///   document scoring exactly θ must never be pruned, because the tie-break
///   favours the lower ordinal and such a document could displace the incumbent
///   k-th entry. That cannot actually happen *here*: all four traversals visit
///   documents in strictly ascending ordinal order, so every entry already in
///   the heap has a smaller ordinal than any candidate still to come, and a
///   score-tie therefore always loses. A strict `>` would, as it happens, still
///   be exact. But that exactness is a *coincidence* between the direction of
///   the tie-break and the direction of the traversal, and it would evaporate
///   the moment either changed. `≥` is unconditionally safe and costs nothing
///   measurable, so it is what we use.
///
/// Soundness of the pruning it *does* permit: if this returns `false` then
/// `score(d) ≤ upper_bound < θ`, and the heap already holds `k` documents whose
/// scores are all `≥ θ > score(d)`. Every one of them strictly outranks `d`, so
/// `d` cannot be in the top-k — regardless of ordinals, since the loss is
/// strict on score alone. And θ only ever increases, so the conclusion cannot
/// be invalidated later.
#[inline]
pub(crate) fn can_reach(upper_bound: f64, threshold: Option<f64>) -> bool {
    match threshold {
        None => true,
        Some(theta) => {
            if !theta.is_finite() {
                return true;
            }
            upper_bound >= theta - PRUNING_TOLERANCE * theta.abs().max(1.0)
        }
    }
}

// ── RankedEntry ──────────────────────────────────────────────────────────────

/// A scored document inside the top-k heap.
///
/// `Ord` is the module's **strict total order** on candidates: higher score
/// first, and among bit-for-bit equal scores, *lower* ordinal first (the
/// document that was indexed earlier wins). Ordinals are unique, so no two
/// distinct candidates ever compare `Equal` — which is precisely why the top-k
/// is uniquely determined and why every strategy is obliged to reproduce it.
///
/// `a > b` means "`a` outranks `b`".
#[derive(Debug, Clone, Copy)]
pub(crate) struct RankedEntry {
    /// The document's BM25 score for the query.
    pub score: f64,
    /// The document's insertion ordinal, used to break score ties.
    pub document_ordinal: u32,
}

impl PartialEq for RankedEntry {
    fn eq(&self, other: &Self) -> bool {
        self.cmp(other) == Ordering::Equal
    }
}

impl Eq for RankedEntry {}

impl PartialOrd for RankedEntry {
    fn partial_cmp(&self, other: &Self) -> Option<Ordering> {
        Some(self.cmp(other))
    }
}

impl Ord for RankedEntry {
    fn cmp(&self, other: &Self) -> Ordering {
        // `total_cmp` gives a total order on f64 without any `unwrap`, and
        // treats -0.0 < 0.0 (irrelevant here: BM25 scores are non-negative).
        self.score
            .total_cmp(&other.score)
            .then_with(|| other.document_ordinal.cmp(&self.document_ordinal))
    }
}

// ── TopKHeap ─────────────────────────────────────────────────────────────────

/// A bounded min-heap of the best `k` [`RankedEntry`]s seen so far.
///
/// "Min" with respect to [`RankedEntry`]'s total order, i.e. the heap's root is
/// the *worst* entry currently held — the one a new candidate must beat, and
/// the one whose score is the pruning threshold θ.
#[derive(Debug)]
pub(crate) struct TopKHeap {
    top_k: usize,
    heap: BinaryHeap<Reverse<RankedEntry>>,
}

impl TopKHeap {
    pub(crate) fn new(top_k: usize) -> Self {
        Self {
            top_k,
            heap: BinaryHeap::with_capacity(top_k.saturating_add(1)),
        }
    }

    /// The current pruning threshold θ: the score of the k-th best document,
    /// or `None` while fewer than `k` documents have been found (in which case
    /// nothing may be pruned).
    pub(crate) fn threshold(&self) -> Option<f64> {
        if self.heap.len() < self.top_k {
            return None;
        }
        self.heap.peek().map(|Reverse(entry)| entry.score)
    }

    /// Offer a fully scored document to the heap. Returns whether it was
    /// admitted (and therefore whether θ may have moved).
    ///
    /// θ is monotonically non-decreasing across calls: an admission replaces
    /// the heap's minimum with something strictly greater in the total order,
    /// so the new minimum's score is at least the old minimum's score.
    pub(crate) fn offer(&mut self, entry: RankedEntry) -> bool {
        if self.top_k == 0 {
            return false;
        }
        if self.heap.len() < self.top_k {
            self.heap.push(Reverse(entry));
            return true;
        }
        let Some(Reverse(worst)) = self.heap.peek().copied() else {
            return false;
        };
        if entry > worst {
            self.heap.pop();
            self.heap.push(Reverse(entry));
            true
        } else {
            false
        }
    }

    /// Drain the heap into a ranked list, best first.
    pub(crate) fn into_ranked(self) -> Vec<RankedEntry> {
        let mut entries: Vec<RankedEntry> =
            self.heap.into_iter().map(|Reverse(entry)| entry).collect();
        // Descending in the total order: best first.
        entries.sort_unstable_by(|left, right| right.cmp(left));
        entries
    }
}

// ── ScoreAccumulator ─────────────────────────────────────────────────────────

/// A fixed-length register file, one slot per de-duplicated query term, used to
/// assemble a document's score in a *query-order* — and hence
/// strategy-independent — summation.
///
/// See the module comment for why this exists.
#[derive(Debug)]
pub(crate) struct ScoreAccumulator {
    contributions: Vec<f64>,
}

impl ScoreAccumulator {
    pub(crate) fn new(term_count: usize) -> Self {
        Self {
            contributions: vec![0.0; term_count],
        }
    }

    /// Clear every slot back to `+0.0`, ready for the next document.
    pub(crate) fn reset(&mut self) {
        for contribution in &mut self.contributions {
            *contribution = 0.0;
        }
    }

    /// Record term `term_index`'s contribution to the document under
    /// evaluation.
    pub(crate) fn set(&mut self, term_index: usize, contribution: f64) {
        if let Some(slot) = self.contributions.get_mut(term_index) {
            *slot = contribution;
        }
    }

    /// The canonical score: a left-to-right sum over every slot, starting from
    /// `0.0`. Terms the document does not contain contribute an exact `+0.0`
    /// and therefore do not perturb the result by so much as a single bit.
    pub(crate) fn total(&self) -> f64 {
        self.contributions
            .iter()
            .fold(0.0f64, |sum, contribution| sum + contribution)
    }
}

// ── TermCursor ───────────────────────────────────────────────────────────────

/// A read head over one query term's posting list.
///
/// Carries the term's query-term-frequency weight `qtf`, so all scores and all
/// bounds it reports are already multiplied by it; the traversal algorithms
/// never need to know that duplicate query terms were folded.
#[derive(Debug)]
pub(crate) struct TermCursor<'a> {
    list: &'a PruningPostingList,
    /// Index of the current posting; `list.len()` once exhausted.
    position: usize,
    /// The term's index in the de-duplicated query, i.e. its
    /// [`ScoreAccumulator`] slot. Also the deterministic tie-break key for
    /// every cursor-selection heuristic in the module.
    term_index: usize,
    /// How many times the term occurred in the query.
    query_term_frequency: f64,
    /// `qtf · U_t`: the term's global upper bound, pre-scaled.
    max_score: f64,
}

impl<'a> TermCursor<'a> {
    pub(crate) fn new(
        list: &'a PruningPostingList,
        term_index: usize,
        query_term_frequency: f64,
    ) -> Self {
        Self {
            list,
            position: 0,
            term_index,
            query_term_frequency,
            max_score: query_term_frequency * list.max_score(),
        }
    }

    /// The term's [`ScoreAccumulator`] slot.
    pub(crate) const fn term_index(&self) -> usize {
        self.term_index
    }

    /// How many postings the underlying list holds (the term's document
    /// frequency). Used by the "advance the longest list" heuristic.
    pub(crate) fn document_frequency(&self) -> usize {
        self.list.len()
    }

    /// `qtf · U_t`: the largest contribution this term can make to *any*
    /// document.
    pub(crate) const fn max_score(&self) -> f64 {
        self.max_score
    }

    /// The current document ordinal, or [`EXHAUSTED`] if the cursor has run off
    /// the end of its list.
    pub(crate) fn document(&self) -> u32 {
        self.list
            .document_ordinals()
            .get(self.position)
            .copied()
            .unwrap_or(EXHAUSTED)
    }

    /// Whether the cursor has run off the end of its list.
    pub(crate) fn is_exhausted(&self) -> bool {
        self.position >= self.list.len()
    }

    /// `qtf · s_t(d)` for the current document `d`. Zero once exhausted.
    pub(crate) fn score(&self) -> f64 {
        self.list
            .scores()
            .get(self.position)
            .map_or(0.0, |score| self.query_term_frequency * score)
    }

    /// Step to the next posting.
    pub(crate) fn advance(&mut self) {
        if self.position < self.list.len() {
            self.position += 1;
        }
    }

    /// Move to the first posting whose document ordinal is at least `target`,
    /// exhausting the cursor if there is none.
    ///
    /// Never moves backwards: a `target` at or below the current document is a
    /// no-op. (`MaxScore` relies on that when it probes a non-essential list
    /// whose cursor has already been carried past the candidate.)
    ///
    /// Implemented as a binary search over the block maxima followed by a
    /// binary search inside the one block that can contain `target` — never a
    /// linear walk over the postings being skipped, which is the entire point:
    /// a skip must not cost what it saves.
    pub(crate) fn seek(&mut self, target: u32) {
        if self.is_exhausted() || self.document() >= target {
            return;
        }
        let blocks = self.list.blocks();
        let base = self.list.block_of(self.position);
        // The blocks partition the ordinal axis into ascending, disjoint,
        // half-open intervals, so the first block with `max_document_ordinal
        // >= target` is the *only* block that can contain `target`.
        let offset = blocks[base..].partition_point(|block| block.max_document_ordinal() < target);
        let block_index = base + offset;
        let Some(block) = blocks.get(block_index) else {
            self.position = self.list.len();
            return;
        };
        // Within that block, search only the postings at or after the current
        // position (everything before it is already known to be < target).
        let start = if block_index == base {
            self.position
        } else {
            block.start()
        };
        let ordinals = &self.list.document_ordinals()[start..block.end()];
        let within = ordinals.partition_point(|&ordinal| ordinal < target);
        self.position = start + within;
    }

    /// The index of the *shallow* block for `target`: the first block whose
    /// `max_document_ordinal` is at least `target`, searched from the block
    /// that owns the current posting.
    ///
    /// "Shallow" because it moves no posting pointer and reads no posting — it
    /// only consults the block maxima. Returns `None` when every remaining
    /// block ends before `target`, i.e. the term cannot contribute to `target`
    /// or to anything after it.
    fn shallow_block(&self, target: u32) -> Option<usize> {
        if self.is_exhausted() {
            return None;
        }
        let blocks = self.list.blocks();
        let base = self.list.block_of(self.position);
        let offset = blocks[base..].partition_point(|block| block.max_document_ordinal() < target);
        let block_index = base + offset;
        if block_index < blocks.len() {
            Some(block_index)
        } else {
            None
        }
    }

    /// `qtf · U_{t,β}` where `β` is the shallow block for `target`: an upper
    /// bound on this term's contribution to `target` *and to every document
    /// between the cursor's position and the end of `β`*, obtained without
    /// touching a posting.
    ///
    /// Sound because `target` — if the term contains it at all — must lie in
    /// `β`, and `U_{t,β}` is an exact maximum over `β`'s stored scores.
    /// Returns `0.0` when the term can no longer contribute to `target`.
    pub(crate) fn block_max_score(&self, target: u32) -> f64 {
        self.shallow_block(target).map_or(0.0, |block_index| {
            self.query_term_frequency * self.list.blocks()[block_index].max_score()
        })
    }

    /// The last document ordinal covered by the shallow block for `target`.
    ///
    /// This is the ordinal past which the cursor's block-max bound changes, and
    /// therefore the ordinal a failed block-max check may safely jump to
    /// (`+ 1`). Returns [`EXHAUSTED`] when there is no such block.
    pub(crate) fn block_max_document(&self, target: u32) -> u32 {
        self.shallow_block(target).map_or(EXHAUSTED, |block_index| {
            self.list.blocks()[block_index].max_document_ordinal()
        })
    }
}
