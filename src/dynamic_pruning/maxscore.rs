//! `MaxScore` (Turtle & Flood, 1995): term-at-a-time-flavoured pruning by
//! splitting the query into *essential* and *non-essential* terms.
//!
//! # The essential / non-essential split
//!
//! Sort the query's cursors by their global upper bound, ascending:
//! `U_0 ≤ U_1 ≤ … ≤ U_{n-1}`. Write `P(e) = U_0 + … + U_{e-1}` for the prefix
//! sums, and let `θ` be the current k-th best score. Choose the split point
//!
//! ```text
//! e = max { j : P(j) < θ } .
//! ```
//!
//! Terms `0 … e-1` are **non-essential**, terms `e … n-1` are **essential**.
//! The name is earned by a one-line argument: a document that contains *no*
//! essential term draws its entire score from the non-essential ones, so
//!
//! ```text
//! score(d)  ≤  Σ_{i < e, d ∈ L_i} U_i  ≤  P(e)  <  θ ,
//! ```
//!
//! and — the heap already holding `k` documents that score at least `θ` — it
//! strictly loses to all of them. **Such documents need never be enumerated at
//! all.** So `MaxScore` iterates only the essential lists to *generate*
//! candidates. The non-essential lists are never walked; they are only ever
//! probed, by `seek`, at the specific ordinals the essential lists throw up.
//!
//! Because `θ` only ever grows and the `P(j)` are fixed, `e` only ever grows
//! too: the essential set shrinks monotonically as the heap fills with better
//! documents, and the traversal narrows onto the rarest, highest-bound terms.
//! When `e` reaches `n`, even *all* terms together cannot reach `θ` and the
//! search is over.
//!
//! # Early abandonment during refinement
//!
//! A candidate `d` is first scored on the essential lists. The non-essential
//! lists are then folded in **from the highest bound downwards**, `i = e-1`
//! down to `0`. Before probing list `i`, everything still outstanding is
//! exactly lists `0 … i`, whose total possible contribution is `P(i + 1)`. So
//! if
//!
//! ```text
//! partial(d)  +  P(i + 1)  <  θ
//! ```
//!
//! the candidate is abandoned on the spot: no assignment of the remaining
//! postings could lift it to `θ`. Descending order of bound is what makes this
//! bite early — the biggest contribution is resolved first, so `partial` climbs
//! (or fails to climb) as fast as it possibly can.
//!
//! # Why the probes are always forward
//!
//! Candidates are strictly increasing: every essential cursor sitting on a
//! candidate is stepped past it before the next candidate is drawn, and the
//! essential set only ever shrinks, so the minimum over it can only rise.
//! `seek` therefore never has to travel backwards — and when a non-essential
//! cursor is already *beyond* the candidate (because it was essential a moment
//! ago and got stepped past), `seek` is a no-op and the term correctly
//! contributes nothing: a cursor only ever lands at or after a posting it was
//! asked for, so having overshot the candidate proves it holds no posting for
//! it.

use crate::dynamic_pruning::cursor::{
    RankedEntry, ScoreAccumulator, TermCursor, TopKHeap, can_reach,
};
use crate::dynamic_pruning::index::EXHAUSTED;
use crate::dynamic_pruning::types::PruningStats;

/// `MaxScore`. Identical top-k to the exhaustive scan; see the module comment
/// for the bound arguments.
pub(crate) fn search_max_score(
    cursors: &mut [TermCursor<'_>],
    term_count: usize,
    top_k: usize,
    stats: &mut PruningStats,
) -> Vec<RankedEntry> {
    let mut heap = TopKHeap::new(top_k);
    let mut accumulator = ScoreAccumulator::new(term_count);

    // Ascending by global upper bound. The order is fixed for the whole
    // traversal (the bounds are static), unlike WAND's per-step doc-id sort.
    // Ties break on the term's query position, for determinism.
    cursors.sort_by(|left, right| {
        left.max_score()
            .total_cmp(&right.max_score())
            .then_with(|| left.term_index().cmp(&right.term_index()))
    });

    let cursor_count = cursors.len();
    // prefix[j] = U_0 + ... + U_{j-1}, so prefix[0] == 0 and
    // prefix[cursor_count] is the score ceiling of the entire query.
    let mut prefix = vec![0.0f64; cursor_count + 1];
    for index in 0..cursor_count {
        prefix[index + 1] = prefix[index] + cursors[index].max_score();
    }

    // cursors[..non_essential] are non-essential. Monotonically non-decreasing.
    let mut non_essential = 0usize;

    loop {
        let threshold = heap.threshold();

        // Grow the non-essential prefix as far as the current threshold allows.
        // `!can_reach(prefix[j + 1], threshold)` says: even if a document
        // contained every one of terms 0..=j and maxed all of them out, it
        // could not reach theta.
        while non_essential < cursor_count && !can_reach(prefix[non_essential + 1], threshold) {
            non_essential += 1;
        }
        if non_essential == cursor_count {
            // Even the whole query cannot reach the threshold any more.
            break;
        }

        // The next candidate is the smallest document ordinal held by any
        // essential cursor. Documents that appear only in non-essential lists
        // are, by the split's defining property, unable to place -- so they are
        // never even generated.
        let mut candidate = EXHAUSTED;
        for cursor in &cursors[non_essential..] {
            let document = cursor.document();
            if document < candidate {
                candidate = document;
            }
        }
        if candidate == EXHAUSTED {
            break;
        }

        accumulator.reset();
        let mut partial = 0.0f64;
        for cursor in &mut cursors[non_essential..] {
            if cursor.document() == candidate {
                let contribution = cursor.score();
                accumulator.set(cursor.term_index(), contribution);
                partial += contribution;
                stats.postings_scored += 1;
            }
        }
        stats.full_evaluations += 1;

        // Refine with the non-essential lists, biggest bound first, abandoning
        // as soon as the best conceivable remainder cannot reach the threshold.
        let mut survived = true;
        for index in (0..non_essential).rev() {
            if !can_reach(partial + prefix[index + 1], threshold) {
                survived = false;
                break;
            }
            cursors[index].seek(candidate);
            if cursors[index].document() == candidate {
                let contribution = cursors[index].score();
                let term_index = cursors[index].term_index();
                accumulator.set(term_index, contribution);
                partial += contribution;
                stats.postings_scored += 1;
            }
        }

        if survived {
            // The final score is the canonical, query-ordered sum -- never the
            // `partial` running total, whose addition order is MaxScore's own
            // and would differ in the last bits from the other strategies'.
            heap.offer(RankedEntry {
                score: accumulator.total(),
                document_ordinal: candidate,
            });
        }

        // Step every essential cursor that sat on the candidate past it. This
        // uses the same `non_essential` split as the candidate generation
        // above, deliberately: a cursor that the (possibly just-raised)
        // threshold is about to demote to non-essential must still be moved off
        // the candidate, or the next iteration would probe it there again.
        for cursor in &mut cursors[non_essential..] {
            if cursor.document() == candidate {
                cursor.advance();
            }
        }
    }

    heap.into_ranked()
}
