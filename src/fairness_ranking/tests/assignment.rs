//! Tests for the Hungarian assignment solver.
//!
//! The headline is [`solver_optimum_equals_brute_force`]: for every problem small
//! enough to enumerate, the solver's optimum must equal the minimum over all
//! injections. Brute force is the independent ground truth; the solver is the
//! thing under test.

use super::random_matrix;
use crate::fairness_ranking::assignment::{AssignmentError, solve_assignment};
use crate::fairness_ranking::rng::FairnessRng;

/// The minimum total cost over **all** injections of `rows` rows into `cols`
/// columns, by exhaustive enumeration. `O(cols! / (cols - rows)!)`, so only ever
/// called with tiny problems.
fn brute_force_min(cost: &[f64], rows: usize, cols: usize) -> f64 {
    let mut used = vec![false; cols];
    let mut assignment = vec![0usize; rows];
    let mut best = f64::INFINITY;
    enumerate(cost, rows, cols, 0, &mut used, &mut assignment, &mut best);
    best
}

fn enumerate(
    cost: &[f64],
    rows: usize,
    cols: usize,
    row: usize,
    used: &mut [bool],
    assignment: &mut [usize],
    best: &mut f64,
) {
    if row == rows {
        let total: f64 = (0..rows).map(|r| cost[r * cols + assignment[r]]).sum();
        if total < *best {
            *best = total;
        }
        return;
    }
    for col in 0..cols {
        if used[col] {
            continue;
        }
        used[col] = true;
        assignment[row] = col;
        enumerate(cost, rows, cols, row + 1, used, assignment, best);
        used[col] = false;
    }
}

// ── (b) solver optimum == brute force ────────────────────────────────────────

#[test]
fn solver_optimum_equals_brute_force() {
    let mut rng = FairnessRng::new(0x5EED);
    let mut checked = 0usize;
    // Square and rectangular problems, seeded random costs, both signs.
    for rows in 1..=8usize {
        for cols in rows..=8usize {
            for _ in 0..12 {
                let cost = random_matrix(&mut rng, rows, cols, -5.0, 5.0);
                let solution = solve_assignment(&cost, rows, cols).expect("well-formed");
                let brute = brute_force_min(&cost, rows, cols);
                assert!(
                    (solution.total_cost() - brute).abs() < 1e-9,
                    "n={rows}x{cols}: solver {} vs brute force {brute}",
                    solution.total_cost()
                );
                // The returned assignment must be a valid injection.
                let columns = solution.column_of_row();
                let mut seen = vec![false; cols];
                for &col in columns {
                    assert!(col < cols);
                    assert!(!seen[col], "column {col} used twice");
                    seen[col] = true;
                }
                // And its reported cost must equal the sum of its own matched
                // entries — the solution is self-consistent, not just numerically
                // close to the brute-force scalar.
                let recomputed: f64 = columns
                    .iter()
                    .enumerate()
                    .map(|(r, &c)| cost[r * cols + c])
                    .sum();
                assert!((recomputed - solution.total_cost()).abs() < 1e-12);
                checked += 1;
            }
        }
    }
    assert!(checked > 300, "expected a broad sweep, ran {checked}");
}

// ── invariances the amortized module relies on ───────────────────────────────

#[test]
fn optimal_assignment_is_invariant_under_row_and_column_shifts() {
    // Adding a constant to a whole row or a whole column shifts every feasible
    // total by the same amount, so the *argmin* is unchanged. The amortized
    // module leans on this to mix an inequity term and a quality term in one cost
    // matrix without normalizing either.
    let mut rng = FairnessRng::new(11);
    let n = 6;
    let base = random_matrix(&mut rng, n, n, -3.0, 3.0);
    let baseline = solve_assignment(&base, n, n).expect("ok");

    let mut shifted = base.clone();
    let row_shift: Vec<f64> = (0..n).map(|_| rng.next_range(-10.0, 10.0)).collect();
    let col_shift: Vec<f64> = (0..n).map(|_| rng.next_range(-10.0, 10.0)).collect();
    for r in 0..n {
        for c in 0..n {
            shifted[r * n + c] += row_shift[r] + col_shift[c];
        }
    }
    let shifted_solution = solve_assignment(&shifted, n, n).expect("ok");
    assert_eq!(
        baseline.column_of_row(),
        shifted_solution.column_of_row(),
        "row/column shifts must not change the optimal assignment"
    );
}

#[test]
fn negative_costs_are_handled() {
    // A cost matrix that is entirely negative (maximization in disguise): the
    // solver must still find the global optimum, checked against brute force.
    let mut rng = FairnessRng::new(99);
    for _ in 0..20 {
        let cost = random_matrix(&mut rng, 5, 5, -10.0, -1.0);
        let solution = solve_assignment(&cost, 5, 5).expect("ok");
        let brute = brute_force_min(&cost, 5, 5);
        assert!((solution.total_cost() - brute).abs() < 1e-9);
    }
}

// ── the guards ───────────────────────────────────────────────────────────────

#[test]
fn malformed_problems_are_rejected() {
    assert!(matches!(
        solve_assignment(&[], 0, 0),
        Err(AssignmentError::Empty { .. })
    ));
    assert!(matches!(
        solve_assignment(&[1.0, 2.0], 1, 3),
        Err(AssignmentError::ShapeMismatch { .. })
    ));
    assert!(matches!(
        solve_assignment(&[1.0, 2.0, 3.0, 4.0, 5.0, 6.0], 3, 2),
        Err(AssignmentError::TooManyRows { rows: 3, cols: 2 })
    ));
    assert!(matches!(
        solve_assignment(&[1.0, f64::NAN, 3.0, 4.0], 2, 2),
        Err(AssignmentError::NonFinite { .. })
    ));
    assert!(matches!(
        solve_assignment(&[1.0, f64::INFINITY, 3.0, 4.0], 2, 2),
        Err(AssignmentError::NonFinite { .. })
    ));
}

#[test]
fn trivial_one_by_one_is_the_single_entry() {
    let solution = solve_assignment(&[2.5], 1, 1).expect("ok");
    assert_eq!(solution.column_of_row(), &[0]);
    assert_eq!(solution.total_cost(), 2.5);
    assert_eq!(solution.len(), 1);
    assert!(!solution.is_empty());
}
