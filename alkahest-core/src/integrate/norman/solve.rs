//! The exact linear solve behind the Risch–Norman ansatz.
//!
//! # Why this is not `sum::gosper::rational_gaussian_solve`
//!
//! The ansatz's system is **overdetermined and extremely sparse**.  Every
//! column is the derivative of one ansatz atom cleared against a common
//! denominator, and differentiating a monomial `x^a·θ^b` produces a handful of
//! terms, not a dense polynomial.  Measured on `∫x⁴⁰·eˣ dx`, the system is
//! 126 × 127 with roughly two non-zeros per column — about 0.2 % density.
//!
//! `rational_gaussian_solve` is a *dense* Gauss–Jordan: its inner loop runs
//! `for j in col..ncols` regardless of whether the pivot row has anything
//! there, so it pays a `rug::Rational` clone, multiply and subtract for every
//! structural zero.  That is `O(m·n·r)` bignum operations — 2 M of them on the
//! example above, which measured at 10.8 ms out of a 13.4 ms integration
//! (81 % of the call).  It is not changed here because `sum::gosper` and other
//! callers depend on its current behaviour.
//!
//! # What this does instead
//!
//! Sparse row echelon with Markowitz-style pivoting, then back-substitution.
//! Rows are sorted `(column, coefficient)` lists; the elimination touches only
//! the structural non-zeros, so the cost is proportional to the fill-in rather
//! than to `m·n`.
//!
//! One invariant makes the bookkeeping cheap:
//!
//! > At the start of the pass over column `c`, every not-yet-used row has all
//! > its entries in columns `≥ c`.
//!
//! so the rows that can possibly have an entry at `c` are exactly the unused
//! rows whose *leading* column is `c`.  They are kept in a bucket per column,
//! and a row is re-bucketed when elimination moves its leading column right.
//! No pass ever scans the whole matrix.
//!
//! # The answer is the same one
//!
//! Pivot columns are taken left to right and free variables are set to zero,
//! which is the reduced-row-echelon particular solution — bit-for-bit what
//! `rational_gaussian_solve` returns.  The pivot *row* chosen within a column
//! does not affect it (the RREF of a matrix is unique), so the Markowitz
//! ordering is a pure performance choice.  That matters: it means swapping the
//! solver in cannot change which integrands are solved or what the
//! antiderivatives look like.
//!
//! # Soundness
//!
//! [`SolveOutcome`] separates *proved inconsistent* from *gave up*, because the
//! caller must not turn the second into the first.  The returned solution is
//! also substituted back into the original system and checked exactly before
//! it is handed over: a solver bug then costs a decline, not a wrong answer.

use rug::{Integer, Rational};

/// A row of the augmented system: sorted non-zero cells, plus a right-hand side.
#[derive(Clone)]
struct Row {
    /// `(column, coefficient)` with strictly increasing columns and no zeros.
    cells: Vec<(usize, Rational)>,
    rhs: Rational,
}

impl Row {
    fn lead(&self) -> Option<usize> {
        self.cells.first().map(|(c, _)| *c)
    }

    /// `self *= k`.
    fn scale(&mut self, k: &Rational) {
        for (_, v) in self.cells.iter_mut() {
            *v *= k.clone();
        }
        self.rhs *= k.clone();
    }

    /// `self -= factor · other`, as a sorted merge.
    fn axpy(&mut self, other: &Row, factor: &Rational) {
        let mut out: Vec<(usize, Rational)> =
            Vec::with_capacity(self.cells.len() + other.cells.len());
        let (mut i, mut j) = (0usize, 0usize);
        while i < self.cells.len() && j < other.cells.len() {
            let (ci, cj) = (self.cells[i].0, other.cells[j].0);
            match ci.cmp(&cj) {
                std::cmp::Ordering::Less => {
                    out.push(self.cells[i].clone());
                    i += 1;
                }
                std::cmp::Ordering::Greater => {
                    let v = -(factor.clone() * other.cells[j].1.clone());
                    if v != 0 {
                        out.push((cj, v));
                    }
                    j += 1;
                }
                std::cmp::Ordering::Equal => {
                    let v = self.cells[i].1.clone() - factor.clone() * other.cells[j].1.clone();
                    if v != 0 {
                        out.push((ci, v));
                    }
                    i += 1;
                    j += 1;
                }
            }
        }
        while i < self.cells.len() {
            out.push(self.cells[i].clone());
            i += 1;
        }
        while j < other.cells.len() {
            let v = -(factor.clone() * other.cells[j].1.clone());
            if v != 0 {
                out.push((other.cells[j].0, v));
            }
            j += 1;
        }
        self.cells = out;
        self.rhs -= factor.clone() * other.rhs.clone();
    }
}

/// The system `A·x = b`, built column-sparse by the caller.
pub(super) struct SparseSystem {
    ncols: usize,
    rows: Vec<Row>,
}

/// What the solve concluded.
///
/// The two failure shapes are kept apart on purpose.  `Inconsistent` is a
/// statement about *this linear system* — never about the integrand, whose
/// antiderivative may simply lie outside the ansatz.  `GaveUp` is a statement
/// about the solver.  Collapsing them is the shape of mistake that produced
/// false non-elementarity certificates elsewhere in this crate.
pub(super) enum SolveOutcome {
    /// A solution, with free variables set to zero.  Verified against the
    /// original system.
    Solved(Vec<Rational>),
    /// The linear system has no solution over `ℚ`.
    Inconsistent,
    /// A cap was hit, or the solution failed its own substitution check.
    GaveUp,
}

/// Largest number of non-zero cells the elimination may hold at once.
///
/// **Observed peak:** 57 cells on the 103-case corpus
/// (`1/(x·log x·log log x)`), 208 on the deliberately oversized stress set
/// (`x⁴⁰·eˣ`).  50 000 is 240× the second.
///
/// This is the cap that actually bounds the solver's work, and it is on
/// *fill-in* rather than on `equations × unknowns`, because fill-in is what a
/// sparse elimination pays for and it is not bounded by either.  A system at
/// the `MAX_EQUATIONS × MAX_UNKNOWNS` ceiling could in principle fill to 1.4 M
/// cells; this refuses at 3 % of that.
///
/// It is a proxy for time, not time itself. This module has no wall-clock
/// budget hook, so an input whose fill-in stays small while its *coefficients*
/// grow is bounded only by `MAX_UNKNOWNS`.  Nothing measured comes close, but
/// it is the honest limit of this ceiling.
const MAX_CELLS: usize = 50_000;

impl SparseSystem {
    /// An empty system with `ncols` unknowns.
    pub(super) fn new(ncols: usize) -> Self {
        SparseSystem {
            ncols,
            rows: Vec::new(),
        }
    }

    /// Append one equation.  `cells` must be sorted by column and free of
    /// zero coefficients; `rhs` may be anything.
    pub(super) fn push_row(&mut self, cells: Vec<(usize, Integer)>, rhs: Integer) {
        debug_assert!(
            cells.windows(2).all(|w| w[0].0 < w[1].0),
            "equation cells must be sorted and duplicate-free"
        );
        debug_assert!(cells.iter().all(|(c, v)| *c < self.ncols && *v != 0));
        self.rows.push(Row {
            cells: cells
                .into_iter()
                .map(|(c, v)| (c, Rational::from(v)))
                .collect(),
            rhs: Rational::from(rhs),
        });
    }

    /// Solve, or say why not.
    pub(super) fn solve(&self) -> SolveOutcome {
        let original = self.rows.clone();
        match self.eliminate() {
            SolveOutcome::Solved(sol) => {
                if verify(&original, &sol) {
                    SolveOutcome::Solved(sol)
                } else {
                    // Unreachable by construction; a decline is the honest
                    // response to reaching it anyway.
                    debug_assert!(false, "sparse solve produced an unverified solution");
                    SolveOutcome::GaveUp
                }
            }
            other => other,
        }
    }

    fn eliminate(&self) -> SolveOutcome {
        let mut rows = self.rows.clone();
        let mut cells: usize = rows.iter().map(|r| r.cells.len()).sum();
        let mut peak = cells;
        if cells > MAX_CELLS {
            return SolveOutcome::GaveUp;
        }

        // Bucket every row by its leading column.  A row with no cells at all
        // is either redundant (`0 = 0`) or a contradiction (`0 = c`).
        let mut buckets: Vec<Vec<usize>> = vec![Vec::new(); self.ncols];
        for (i, r) in rows.iter().enumerate() {
            match r.lead() {
                Some(c) => buckets[c].push(i),
                None => {
                    if r.rhs != 0 {
                        return SolveOutcome::Inconsistent;
                    }
                }
            }
        }

        // `(row index, pivot column)` in increasing column order.
        let mut pivots: Vec<(usize, usize)> = Vec::new();
        for col in 0..self.ncols {
            let candidates = std::mem::take(&mut buckets[col]);
            if candidates.is_empty() {
                continue;
            }
            // Markowitz: the sparsest candidate fills in the least.
            let p = *candidates
                .iter()
                .min_by_key(|&&i| rows[i].cells.len())
                .expect("candidates is non-empty");

            let lead = rows[p].cells[0].1.clone();
            debug_assert_eq!(rows[p].cells[0].0, col);
            let inv = Rational::from(1) / lead;
            rows[p].scale(&inv);

            let pivot_row = rows[p].clone();
            for i in candidates {
                if i == p {
                    continue;
                }
                let factor = rows[i].cells[0].1.clone();
                let before = rows[i].cells.len();
                rows[i].axpy(&pivot_row, &factor);
                cells = cells + rows[i].cells.len() - before;
                peak = peak.max(cells);
                if cells > MAX_CELLS {
                    return SolveOutcome::GaveUp;
                }
                match rows[i].lead() {
                    Some(c) => {
                        debug_assert!(c > col, "elimination must move the leading column right");
                        buckets[c].push(i);
                    }
                    None => {
                        if rows[i].rhs != 0 {
                            return SolveOutcome::Inconsistent;
                        }
                    }
                }
            }
            pivots.push((p, col));
        }

        super::profile::record(|s| s.peak_cells = peak);

        // Back-substitution.  Free columns keep their zero.
        let mut sol = vec![Rational::from(0); self.ncols];
        for &(r, col) in pivots.iter().rev() {
            let mut acc = rows[r].rhs.clone();
            for (j, v) in rows[r].cells.iter().skip(1) {
                acc -= v.clone() * sol[*j].clone();
            }
            // The leading coefficient was normalised to 1 above.
            sol[col] = acc;
        }
        SolveOutcome::Solved(sol)
    }
}

/// `A·x == b`, exactly, on the untouched system.
fn verify(rows: &[Row], sol: &[Rational]) -> bool {
    for r in rows {
        let mut acc = Rational::from(0);
        for (j, v) in &r.cells {
            if sol[*j] != 0 {
                acc += v.clone() * sol[*j].clone();
            }
        }
        if acc != r.rhs {
            return false;
        }
    }
    true
}

#[cfg(test)]
mod tests {
    use super::*;

    fn int(v: i64) -> Integer {
        Integer::from(v)
    }

    /// The sparse solver must agree with the dense one it replaces, including
    /// on the free-variable convention.
    fn cross_check(dense: Vec<Vec<i64>>, rhs: Vec<i64>) {
        let ncols = dense[0].len();
        let mut sys = SparseSystem::new(ncols);
        for (row, b) in dense.iter().zip(rhs.iter()) {
            let cells: Vec<(usize, Integer)> = row
                .iter()
                .enumerate()
                .filter(|(_, v)| **v != 0)
                .map(|(j, v)| (j, int(*v)))
                .collect();
            sys.push_row(cells, int(*b));
        }
        let mine = match sys.solve() {
            SolveOutcome::Solved(s) => Some(s),
            _ => None,
        };
        let dense_mat: Vec<Vec<Rational>> = dense
            .iter()
            .map(|r| r.iter().map(|v| Rational::from(*v)).collect())
            .collect();
        let dense_rhs: Vec<Rational> = rhs.iter().map(|v| Rational::from(*v)).collect();
        let theirs = crate::sum::gosper::rational_gaussian_solve(dense_mat, dense_rhs);
        assert_eq!(mine, theirs, "sparse and dense solvers disagree");
    }

    #[test]
    fn agrees_with_the_dense_solver_on_a_square_system() {
        cross_check(vec![vec![2, 1], vec![1, -1]], vec![5, 1]);
    }

    #[test]
    fn agrees_with_the_dense_solver_when_overdetermined_and_consistent() {
        cross_check(
            vec![vec![1, 0], vec![0, 1], vec![1, 1], vec![2, 2]],
            vec![3, 4, 7, 14],
        );
    }

    #[test]
    fn agrees_with_the_dense_solver_on_free_variables() {
        // `x + z = 1`, `y = 2`; `z` is free and must come back as 0.
        cross_check(vec![vec![1, 0, 1], vec![0, 1, 0]], vec![1, 2]);
    }

    #[test]
    fn agrees_with_the_dense_solver_on_a_zero_column() {
        cross_check(vec![vec![0, 1], vec![0, 2]], vec![3, 6]);
    }

    #[test]
    fn rational_solutions_are_exact() {
        cross_check(vec![vec![3, 0], vec![0, 7]], vec![1, 1]);
    }

    #[test]
    fn inconsistency_is_reported_as_inconsistent_not_as_a_give_up() {
        let mut sys = SparseSystem::new(1);
        sys.push_row(vec![(0, int(1))], int(1));
        sys.push_row(vec![(0, int(1))], int(2));
        assert!(matches!(sys.solve(), SolveOutcome::Inconsistent));
    }

    #[test]
    fn an_all_zero_row_with_a_non_zero_rhs_is_inconsistent() {
        let mut sys = SparseSystem::new(2);
        sys.push_row(vec![], int(5));
        assert!(matches!(sys.solve(), SolveOutcome::Inconsistent));
    }

    #[test]
    fn an_empty_system_solves_to_all_zeros() {
        let sys = SparseSystem::new(3);
        match sys.solve() {
            SolveOutcome::Solved(s) => assert_eq!(s, vec![Rational::from(0); 3]),
            _ => panic!("an empty system is solved by zero"),
        }
    }

    /// The shape the ansatz actually produces: a long, thin, near-diagonal
    /// system.  Dense elimination is `O(n³)` on it; this must not be.
    #[test]
    fn a_large_bidiagonal_system_is_solved() {
        let n = 300;
        let mut sys = SparseSystem::new(n);
        for i in 0..n {
            let mut cells = vec![(i, int(1))];
            if i + 1 < n {
                cells.push((i + 1, int(2)));
            }
            sys.push_row(cells, int(1));
        }
        match sys.solve() {
            SolveOutcome::Solved(s) => assert_eq!(s.len(), n),
            _ => panic!("expected a solution"),
        }
    }
}
