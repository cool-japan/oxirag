//! The **linear assignment problem**, solved exactly by the Hungarian algorithm
//! in its `O(n^2 m)` shortest-augmenting-path form (Kuhn–Munkres, in the
//! Jonker–Volgenant arrangement).
//!
//! # Why a ranking module needs a matching solver
//!
//! Amortized equity of attention (Biega et al., `SIGIR` 2018 — see
//! [`amortized`](super::amortized)) asks for the ranking that best repairs the
//! *accumulated* gap between the attention each subject has received and the
//! relevance it has earned. Write `A_i` for subject `i`'s cumulative attention
//! so far, `R_i` for its cumulative relevance including the query being served,
//! and `a_j` for the attention that position `j` pays out. Serving the query
//! means choosing an injection from positions to subjects, and the objective is
//!
//! ```text
//! minimize  sum_i | A_i + a_{pos(i)} - R_i |
//! ```
//!
//! Because each subject occupies exactly one position, the term for subject `i`
//! depends only on the position it is given. The objective is therefore
//! **separable over the matched pairs**, with the cost of putting subject `i` at
//! position `j` being the constant `c(j, i) = |A_i + a_j - R_i|`, and the whole
//! problem collapses to: pick a minimum-cost injection. That is precisely a
//! linear assignment problem — not a metaphor for one.
//!
//! Nothing in this crate solved one before: the only `bipartite` machinery that
//! existed (`click_model::pbm`) is a union-find connectivity check, not a
//! matching. So the solver is here, hand-rolled, with no new dependency.
//!
//! # The algorithm
//!
//! The primal-dual method maintains potentials `u` (rows) and `v` (columns) that
//! stay dual-feasible — `u_i + v_j <= c(i, j)` for every pair — and grows the
//! matching one row at a time. For each new row it runs a Dijkstra-like search
//! over *reduced* costs `c(i, j) - u_i - v_j`, which are non-negative by dual
//! feasibility, until it reaches a free column; the shortest such path is then
//! augmented along, and the potentials are shifted by the search's `delta` so
//! that feasibility is preserved and the newly used edges become tight. When
//! every row is matched, complementary slackness holds on the matching, which is
//! exactly the certificate that it is optimal.
//!
//! Two properties are worth naming because the callers rely on them:
//!
//! * **Rectangular is fine.** With `rows <= cols` the solver matches every row
//!   to a distinct column and leaves `cols - rows` columns free. Ranking uses
//!   this directly: the `rows` are the `k` positions on the page, the `cols` are
//!   the `n >= k` candidates, and the `n - k` unmatched candidates are simply not
//!   shown.
//! * **Row and column constants are free.** Adding a constant to an entire row
//!   (or an entire column) of the cost matrix shifts every feasible solution's
//!   total by the same amount, because each row and each column is used at most
//!   once. The optimal assignment is therefore invariant under such shifts, which
//!   is what licenses [`amortized`](super::amortized) to mix an unfairness term
//!   and a raw `-relevance * attention` quality term in one cost matrix without
//!   first normalizing either into `[0, 1]`.
//!
//! Costs may be negative; they may not be `NaN` or infinite, and the solver
//! rejects those up front rather than letting them silently corrupt a potential.

use thiserror::Error;

/// Errors raised by [`solve_assignment`].
#[derive(Debug, Error, Clone, PartialEq)]
pub enum AssignmentError {
    /// The problem had no rows or no columns. There is no such thing as a
    /// minimum-cost matching of nothing onto nothing, and a caller that received
    /// an empty assignment would have no ranking to serve.
    #[error("assignment problem is empty: {rows} rows x {cols} columns")]
    Empty {
        /// The row count supplied.
        rows: usize,
        /// The column count supplied.
        cols: usize,
    },
    /// The cost buffer was not `rows * cols` long.
    #[error("cost matrix has {actual} entries, expected {expected} ({rows} x {cols})")]
    ShapeMismatch {
        /// The number of entries a `rows` x `cols` row-major matrix needs.
        expected: usize,
        /// The number actually supplied.
        actual: usize,
        /// The declared row count.
        rows: usize,
        /// The declared column count.
        cols: usize,
    },
    /// More rows than columns. Every row must receive its own column, so an
    /// injection only exists when `rows <= cols`. (Transpose the problem if the
    /// tall orientation is the natural one.)
    #[error("cannot assign {rows} rows to {cols} columns: rows must not exceed columns")]
    TooManyRows {
        /// The row count supplied.
        rows: usize,
        /// The column count supplied.
        cols: usize,
    },
    /// A cost entry was `NaN` or infinite. Both would poison the dual potentials
    /// irrecoverably, so they are rejected before the first iteration rather than
    /// producing a matching that looks like an answer and is not.
    #[error("cost matrix entry ({row}, {col}) is not finite: {value}")]
    NonFinite {
        /// The offending row.
        row: usize,
        /// The offending column.
        col: usize,
        /// The offending value.
        value: f64,
    },
    /// The augmenting search exhausted every column without reaching a free one.
    /// This is impossible when `rows <= cols` (at most `rows - 1` columns are
    /// matched when the search starts, so a free column always remains) and
    /// therefore signals a corrupted internal state rather than an input the
    /// caller could have avoided.
    #[error("augmenting path search found no free column for row {row}")]
    NoAugmentingPath {
        /// The row whose augmentation failed.
        row: usize,
    },
}

/// An optimal assignment: which column each row was matched to, and what it cost.
#[derive(Debug, Clone, PartialEq)]
pub struct AssignmentSolution {
    /// `column_of_row[i]` is the column matched to row `i`. Length `rows`; every
    /// entry is distinct.
    column_of_row: Vec<usize>,
    /// The total cost, summed directly from the matched entries of the input
    /// matrix (rather than read off the dual potentials, so that it is an
    /// independent statement about the returned matching).
    total_cost: f64,
}

impl AssignmentSolution {
    /// The column matched to each row, indexed by row.
    #[must_use]
    pub fn column_of_row(&self) -> &[usize] {
        &self.column_of_row
    }

    /// The minimum total cost.
    #[must_use]
    pub fn total_cost(&self) -> f64 {
        self.total_cost
    }

    /// The number of matched pairs, i.e. the row count.
    #[must_use]
    pub fn len(&self) -> usize {
        self.column_of_row.len()
    }

    /// Whether the matching is empty. It never is —
    /// [`AssignmentError::Empty`] rejects a degenerate problem — but the
    /// accessor exists so `len` does not stand alone.
    #[must_use]
    pub fn is_empty(&self) -> bool {
        self.column_of_row.is_empty()
    }
}

/// Solve the linear assignment problem: match every row to a distinct column so
/// that the total cost is minimal.
///
/// `cost` is a **row-major** `rows x cols` matrix — entry `(i, j)` lives at
/// `i * cols + j` — and `rows <= cols` is required. Costs may be negative.
///
/// The returned [`AssignmentSolution`] is a global optimum, not a greedy or
/// heuristic one; this module's tests check it against brute-force enumeration of
/// every injection for `n <= 8`.
///
/// # Errors
///
/// * [`AssignmentError::Empty`] — zero rows or zero columns.
/// * [`AssignmentError::ShapeMismatch`] — `cost.len() != rows * cols`.
/// * [`AssignmentError::TooManyRows`] — `rows > cols`, so no injection exists.
/// * [`AssignmentError::NonFinite`] — a `NaN` or infinite cost.
/// * [`AssignmentError::NoAugmentingPath`] — an impossible internal state (see
///   the variant's documentation).
#[allow(clippy::needless_range_loop)] // Primal-dual bookkeeping over parallel 1-indexed arrays.
pub fn solve_assignment(
    cost: &[f64],
    rows: usize,
    cols: usize,
) -> Result<AssignmentSolution, AssignmentError> {
    if rows == 0 || cols == 0 {
        return Err(AssignmentError::Empty { rows, cols });
    }
    if cost.len() != rows * cols {
        return Err(AssignmentError::ShapeMismatch {
            expected: rows * cols,
            actual: cost.len(),
            rows,
            cols,
        });
    }
    if rows > cols {
        return Err(AssignmentError::TooManyRows { rows, cols });
    }
    for (index, &value) in cost.iter().enumerate() {
        if !value.is_finite() {
            return Err(AssignmentError::NonFinite {
                row: index / cols,
                col: index % cols,
                value,
            });
        }
    }

    // Everything below is 1-indexed, with index 0 reserved as the sentinel
    // "virtual" row/column that the augmenting search starts from. `at(i, j)`
    // translates back to the caller's 0-indexed row-major buffer.
    let at = |i: usize, j: usize| cost[(i - 1) * cols + (j - 1)];

    // Dual potentials.
    let mut row_potential = vec![0.0f64; rows + 1];
    let mut col_potential = vec![0.0f64; cols + 1];
    // `row_of_col[j]` is the row currently matched to column `j`, or 0 if free.
    let mut row_of_col = vec![0usize; cols + 1];
    // `predecessor[j]` is the column from which `j` was reached in the search
    // tree, used to walk the augmenting path back to the start.
    let mut predecessor = vec![0usize; cols + 1];

    for row in 1..=rows {
        row_of_col[0] = row;
        let mut current_col = 0usize;
        // `slack[j]` is the shortest reduced-cost distance found so far from the
        // search tree to the unvisited column `j`.
        let mut slack = vec![f64::INFINITY; cols + 1];
        let mut visited = vec![false; cols + 1];

        loop {
            visited[current_col] = true;
            let current_row = row_of_col[current_col];
            let mut delta = f64::INFINITY;
            let mut next_col = 0usize;

            for col in 1..=cols {
                if visited[col] {
                    continue;
                }
                // Reduced cost of the edge (current_row, col). Non-negative by
                // dual feasibility, which is the invariant the potential shift
                // below preserves.
                let reduced =
                    at(current_row, col) - row_potential[current_row] - col_potential[col];
                if reduced < slack[col] {
                    slack[col] = reduced;
                    predecessor[col] = current_col;
                }
                if slack[col] < delta {
                    delta = slack[col];
                    next_col = col;
                }
            }

            if next_col == 0 {
                // No unvisited column remains. Unreachable for `rows <= cols`;
                // see `AssignmentError::NoAugmentingPath`.
                return Err(AssignmentError::NoAugmentingPath { row });
            }

            // Shift the potentials by `delta`: tight edges stay tight, the newly
            // reached column becomes tight, and dual feasibility is preserved
            // everywhere else.
            for col in 0..=cols {
                if visited[col] {
                    row_potential[row_of_col[col]] += delta;
                    col_potential[col] -= delta;
                } else {
                    slack[col] -= delta;
                }
            }

            current_col = next_col;
            if row_of_col[current_col] == 0 {
                break;
            }
        }

        // Augment: walk the predecessor chain back to the sentinel, shifting each
        // matched row one column along the path.
        loop {
            let previous = predecessor[current_col];
            row_of_col[current_col] = row_of_col[previous];
            current_col = previous;
            if current_col == 0 {
                break;
            }
        }
    }

    let mut column_of_row = vec![0usize; rows];
    for col in 1..=cols {
        let row = row_of_col[col];
        if row != 0 {
            column_of_row[row - 1] = col - 1;
        }
    }

    // Sum the matched entries directly. Reading `-col_potential[0]` would be one
    // multiplication cheaper and would *assume* the duals are consistent with the
    // matching; summing the primal is a statement about the matching actually
    // returned.
    let total_cost = column_of_row
        .iter()
        .enumerate()
        .map(|(row, &col)| cost[row * cols + col])
        .sum();

    Ok(AssignmentSolution {
        column_of_row,
        total_cost,
    })
}
