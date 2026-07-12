//! Document-at-a-time traversals: the exhaustive baseline, WAND, and
//! Block-Max WAND.
//!
//! # WAND's skipping argument, in full
//!
//! Keep the cursors sorted by their current document ordinal, `d_0 ≤ d_1 ≤ …
//! ≤ d_{n-1}` (exhausted cursors sort last). Let `U_i` be cursor `i`'s global
//! upper bound and let `θ` be the score of the current k-th best document.
//! Walk the sorted cursors accumulating `U_i` and stop at the first index `p`
//! with
//!
//! ```text
//! U_0 + U_1 + … + U_p  ≥  θ .
//! ```
//!
//! `p` is the **pivot** and `d_p` the **pivot document**. The claim is that
//! *no document below `d_p` can enter the top-k*, so the traversal may leap
//! straight to `d_p`.
//!
//! Take any document `d < d_p`. A term can contribute to `d` only if its
//! cursor still has a posting for it, and a cursor's remaining postings are all
//! `≥` its current document — so only cursors with `d_i ≤ d` can contribute.
//! Since the cursors are sorted and `d < d_p`, every such cursor has an index
//! `< p`. Hence
//!
//! ```text
//! score(d)  ≤  Σ_{i : d_i ≤ d} U_i  ≤  U_0 + … + U_{p-1}  <  θ ,
//! ```
//!
//! the last inequality because `p` was the *first* index at which the running
//! sum reached `θ`. The heap already holds `k` documents scoring at least `θ`,
//! all of which strictly outrank `d`; therefore `d` is not in the top-k. And
//! `θ` never decreases, so the verdict can never be overturned. ∎
//!
//! When the pivot is reached, either `d_0 == d_p` — every cursor that could
//! contribute to `d_p` is already sitting on it, so score it — or some cursor
//! below the pivot lags behind, in which case `seek` it up to `d_p` and
//! re-derive the pivot. Both branches move at least one cursor strictly
//! forward, so the loop terminates.
//!
//! # What block-max buys
//!
//! The bound above is *global*: `U_i` is the largest score term `i` achieves
//! anywhere in the corpus, which is wildly pessimistic for the particular
//! stretch of the document axis the cursor happens to be in. Block-Max WAND
//! re-checks the pivot against the per-block bounds of the blocks that actually
//! cover `d_p`:
//!
//! ```text
//! score(d_p)  ≤  Σ_{i ≤ p} U_{i, β_i(d_p)}   where β_i(d_p) is cursor i's
//!                                            shallow block for d_p
//! ```
//!
//! and that sum is typically far smaller than `Σ_{i ≤ p} U_i`. When it fails to
//! reach `θ`, the pivot is rejected *without a single posting being read* — and
//! more than that, the same bound covers every document up to the earliest
//! block end among the pivot's cursors, so a whole block can be jumped in one
//! step. Formally, for any `d` with `d_p ≤ d ≤ min_{i ≤ p} maxdoc(β_i(d_p))`
//! and `d < d_{p+1}`:
//!
//! * terms `i > p` cannot contain `d` (their cursors are already beyond it);
//! * for `i ≤ p`, `d` lies inside `β_i(d_p)` — the blocks partition the
//!   ordinal axis into ascending disjoint intervals, `β_i(d_p)` is the *only*
//!   block of list `i` that can hold anything in `[d_p, maxdoc(β_i(d_p))]`, and
//!   `d` is in that range — so term `i`'s contribution to `d` is at most
//!   `U_{i, β_i(d_p)}`.
//!
//! Hence `score(d) ≤ Σ_{i ≤ p} U_{i, β_i(d_p)} < θ` for every such `d`, and the
//! traversal may jump to `min( min_{i ≤ p} maxdoc(β_i(d_p)) + 1, d_{p+1} )`. ∎

use crate::dynamic_pruning::cursor::{
    RankedEntry, ScoreAccumulator, TermCursor, TopKHeap, can_reach,
};
use crate::dynamic_pruning::index::EXHAUSTED;
use crate::dynamic_pruning::types::PruningStats;

/// Sort cursors by current document ordinal, breaking ties on the term's query
/// position so the traversal is fully deterministic.
fn sort_by_document(cursors: &mut [TermCursor<'_>]) {
    cursors.sort_unstable_by(|left, right| {
        left.document()
            .cmp(&right.document())
            .then_with(|| left.term_index().cmp(&right.term_index()))
    });
}

/// The pivot: the first index at which the accumulated global upper bounds of
/// the doc-ordered cursors could reach `threshold`.
///
/// `None` means no document anywhere in the remaining postings can reach the
/// threshold — the traversal is finished.
fn select_pivot(cursors: &[TermCursor<'_>], threshold: Option<f64>) -> Option<usize> {
    let mut accumulated = 0.0f64;
    for (index, cursor) in cursors.iter().enumerate() {
        if cursor.is_exhausted() {
            // Sorted by document ordinal and `EXHAUSTED` is the largest, so
            // every cursor from here on is exhausted too.
            break;
        }
        accumulated += cursor.max_score();
        if can_reach(accumulated, threshold) {
            return Some(index);
        }
    }
    None
}

/// Pick which lagging cursor to pull up to the pivot document.
///
/// Considers only cursors strictly below the pivot document — pulling up a
/// cursor already sitting on it would be a no-op and would spin the loop. The
/// heuristic (advance the longest list, since it has the most to gain from a
/// skip) is the standard one; ties break on the term's query position for
/// determinism.
///
/// The caller only reaches this when `cursors[0].document() < pivot_document`,
/// so index `0` always qualifies and the fallback is never taken.
fn select_lagging_cursor(cursors: &[TermCursor<'_>], pivot: usize, pivot_document: u32) -> usize {
    let mut best: Option<usize> = None;
    for index in 0..pivot {
        if cursors[index].document() >= pivot_document {
            continue;
        }
        best = Some(match best {
            None => index,
            Some(current) => {
                if is_better_advance_choice(&cursors[index], &cursors[current]) {
                    index
                } else {
                    current
                }
            }
        });
    }
    best.unwrap_or(0)
}

fn is_better_advance_choice(candidate: &TermCursor<'_>, incumbent: &TermCursor<'_>) -> bool {
    let candidate_length = candidate.document_frequency();
    let incumbent_length = incumbent.document_frequency();
    candidate_length > incumbent_length
        || (candidate_length == incumbent_length && candidate.term_index() < incumbent.term_index())
}

/// Fully score `document` from every cursor currently sitting on it, offer it
/// to the heap, and step those cursors past it.
///
/// The score is assembled through the [`ScoreAccumulator`], i.e. summed in
/// query order, which is what makes it bit-for-bit identical to the one
/// `Exhaustive` and `MaxScore` compute for the same document.
fn evaluate_document(
    cursors: &mut [TermCursor<'_>],
    document: u32,
    accumulator: &mut ScoreAccumulator,
    heap: &mut TopKHeap,
    stats: &mut PruningStats,
) {
    accumulator.reset();
    for cursor in cursors.iter_mut() {
        if cursor.document() == document {
            accumulator.set(cursor.term_index(), cursor.score());
            stats.postings_scored += 1;
            cursor.advance();
        }
    }
    stats.full_evaluations += 1;
    heap.offer(RankedEntry {
        score: accumulator.total(),
        document_ordinal: document,
    });
}

// ── Exhaustive ───────────────────────────────────────────────────────────────

/// Document-at-a-time full scan: every posting of every query term is read and
/// every matching document is fully scored. No bound is ever consulted.
///
/// This is the ground truth. The top-k it produces — under the strict total
/// order "higher score, then lower ordinal" — is by definition *the* correct
/// answer, and the three pruning strategies are obliged to reproduce it
/// exactly.
pub(crate) fn search_exhaustive(
    cursors: &mut [TermCursor<'_>],
    term_count: usize,
    top_k: usize,
    stats: &mut PruningStats,
) -> Vec<RankedEntry> {
    let mut heap = TopKHeap::new(top_k);
    let mut accumulator = ScoreAccumulator::new(term_count);

    loop {
        let mut current = EXHAUSTED;
        for cursor in cursors.iter() {
            let document = cursor.document();
            if document < current {
                current = document;
            }
        }
        if current == EXHAUSTED {
            break;
        }
        evaluate_document(cursors, current, &mut accumulator, &mut heap, stats);
    }

    heap.into_ranked()
}

// ── WAND ─────────────────────────────────────────────────────────────────────

/// Weak-AND (Broder et al., 2003). See the module comment for the proof that
/// everything it skips provably cannot enter the top-k.
pub(crate) fn search_wand(
    cursors: &mut [TermCursor<'_>],
    term_count: usize,
    top_k: usize,
    stats: &mut PruningStats,
) -> Vec<RankedEntry> {
    let mut heap = TopKHeap::new(top_k);
    let mut accumulator = ScoreAccumulator::new(term_count);

    loop {
        sort_by_document(cursors);
        let threshold = heap.threshold();
        let Some(pivot) = select_pivot(cursors, threshold) else {
            // No remaining document can reach the threshold: done.
            break;
        };
        let pivot_document = cursors[pivot].document();
        if pivot_document == EXHAUSTED {
            break;
        }

        if cursors[0].document() == pivot_document {
            // Every cursor that can contribute to the pivot document is on it.
            evaluate_document(cursors, pivot_document, &mut accumulator, &mut heap, stats);
        } else {
            // Some cursor lags behind; pull it up. Everything it steps over is
            // skipped unscored, which is exactly the win.
            let lagging = select_lagging_cursor(cursors, pivot, pivot_document);
            cursors[lagging].seek(pivot_document);
        }
    }

    heap.into_ranked()
}

// ── Block-Max WAND ───────────────────────────────────────────────────────────

/// Block-Max WAND (Ding & Suel, 2011). Identical top-k to
/// [`search_wand`], strictly less work: the pivot is re-checked against the
/// tight per-block bounds before any posting is touched, and a failed check
/// jumps a whole block rather than a single document.
pub(crate) fn search_block_max_wand(
    cursors: &mut [TermCursor<'_>],
    term_count: usize,
    top_k: usize,
    stats: &mut PruningStats,
) -> Vec<RankedEntry> {
    let mut heap = TopKHeap::new(top_k);
    let mut accumulator = ScoreAccumulator::new(term_count);

    loop {
        sort_by_document(cursors);
        let threshold = heap.threshold();
        let Some(mut pivot) = select_pivot(cursors, threshold) else {
            break;
        };
        let pivot_document = cursors[pivot].document();
        if pivot_document == EXHAUSTED {
            break;
        }

        // Extend the pivot across the whole run of cursors already sitting on
        // the pivot document. They *do* contribute to it, so their block bounds
        // belong in the block-level check; leaving them out would make the
        // bound unsound.
        while pivot + 1 < cursors.len() && cursors[pivot + 1].document() == pivot_document {
            pivot += 1;
        }

        let mut block_bound = 0.0f64;
        for cursor in &cursors[..=pivot] {
            block_bound += cursor.block_max_score(pivot_document);
        }

        if can_reach(block_bound, threshold) {
            if cursors[0].document() == pivot_document {
                evaluate_document(cursors, pivot_document, &mut accumulator, &mut heap, stats);
            } else {
                let lagging = select_lagging_cursor(cursors, pivot, pivot_document);
                cursors[lagging].seek(pivot_document);
            }
        } else {
            // The block-level bound proved that nothing from the pivot document
            // up to the earliest block end can place. Jump the whole run.
            stats.blocks_skipped += 1;
            let next = next_block_candidate(cursors, pivot, pivot_document);
            let chosen = select_block_skip_cursor(cursors, pivot);
            cursors[chosen].seek(next);
        }
    }

    heap.into_ranked()
}

/// The first document ordinal at which the rejected block configuration could
/// possibly change: one past the earliest block end among the pivot's cursors,
/// or the next cursor's document, whichever comes first.
///
/// Guaranteed to be strictly greater than `pivot_document`, so the traversal
/// always makes progress:
///
/// * every `block_max_document(pivot_document)` is `≥ pivot_document` (a
///   shallow block *covers* its target by construction), so each `+ 1` term is
///   `> pivot_document`;
/// * `cursors[pivot + 1]` is strictly past `pivot_document` because the pivot
///   was extended across every cursor that equalled it.
fn next_block_candidate(cursors: &[TermCursor<'_>], pivot: usize, pivot_document: u32) -> u32 {
    let mut next = pivot_document.saturating_add(1);
    for cursor in &cursors[..=pivot] {
        let block_end = cursor.block_max_document(pivot_document);
        if block_end != EXHAUSTED {
            next = next.min(block_end.saturating_add(1));
        }
    }
    if let Some(following) = cursors.get(pivot + 1) {
        let document = following.document();
        if document != EXHAUSTED {
            next = next.min(document);
        }
    }
    next
}

/// Pick which cursor to drive forward after a failed block-max check.
///
/// Every cursor at or below the pivot is strictly below the jump target, so any
/// choice makes progress; the longest list is chosen because it stands to skip
/// the most. Ties break on the term's query position for determinism.
fn select_block_skip_cursor(cursors: &[TermCursor<'_>], pivot: usize) -> usize {
    let mut best = 0usize;
    for index in 1..=pivot {
        if is_better_advance_choice(&cursors[index], &cursors[best]) {
            best = index;
        }
    }
    best
}
