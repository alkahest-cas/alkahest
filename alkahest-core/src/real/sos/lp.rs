//! A small exact-rational linear programme solver (two-phase simplex).
//!
//! Positivity certificates are only worth anything if they are exact, so the
//! search for one must not go through floating point.  This is a dense
//! two-phase simplex over `rug::Rational` with **Bland's rule**, which is
//! provably cycle-free; in exact arithmetic that makes termination
//! unconditional (no epsilon tolerances, no degeneracy stalls).
//!
//! It is deliberately local to the positivity subsystem: it is tuned for the
//! small, highly structured programmes that Gram-matrix and Handelman searches
//! produce, and no other part of the crate depends on it.

use rug::Rational;

/// Relation of a constraint row.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Rel {
    /// `a · x = b`
    Eq,
    /// `a · x ≥ b`
    Ge,
    /// `a · x ≤ b`
    Le,
}

/// Outcome of a solve.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum LpStatus {
    /// An optimal vertex was found; the vector holds the value of every
    /// structural (caller-declared) variable.
    Optimal(Vec<Rational>),
    /// The constraint system has no solution with `x ≥ 0`.
    Infeasible,
    /// The objective is unbounded below on the feasible set.
    Unbounded,
    /// The pivot budget was exhausted (defensive; Bland's rule should make
    /// this unreachable).
    Exhausted,
}

/// A linear programme in the form
/// `minimise c · x  subject to  (rows)  and  x ≥ 0`.
///
/// Free (sign-unrestricted) variables must be split by the caller into a
/// positive and a negative part; [`super::gram`] does exactly that for the
/// off-diagonal Gram entries.
#[derive(Debug, Clone, Default)]
pub struct Lp {
    nvars: usize,
    rows: Vec<(Vec<Rational>, Rel, Rational)>,
    objective: Vec<Rational>,
}

impl Lp {
    pub fn new(nvars: usize) -> Self {
        Lp {
            nvars,
            rows: Vec::new(),
            objective: vec![Rational::from(0); nvars],
        }
    }

    pub fn nvars(&self) -> usize {
        self.nvars
    }

    pub fn nrows(&self) -> usize {
        self.rows.len()
    }

    /// Add a constraint row.  `coeffs` must have length `nvars`.
    pub fn constrain(&mut self, coeffs: Vec<Rational>, rel: Rel, rhs: Rational) {
        debug_assert_eq!(coeffs.len(), self.nvars);
        self.rows.push((coeffs, rel, rhs));
    }

    /// Set the coefficient of variable `i` in the (minimised) objective.
    pub fn set_objective(&mut self, i: usize, c: Rational) {
        self.objective[i] = c;
    }

    /// Solve.  Returns the structural variable values on success.
    pub fn solve(&self) -> LpStatus {
        // ---------------------------------------------------------------
        // Build standard form: A x' = b, x' ≥ 0, by adding one slack or
        // surplus column per inequality row.
        // ---------------------------------------------------------------
        let n_slack = self
            .rows
            .iter()
            .filter(|(_, rel, _)| *rel != Rel::Eq)
            .count();
        let n_total = self.nvars + n_slack;
        let m = self.rows.len();
        if m == 0 {
            // No constraints: the origin is optimal iff every cost is ≥ 0.
            return if self.objective.iter().all(|c| *c >= 0) {
                LpStatus::Optimal(vec![Rational::from(0); self.nvars])
            } else {
                LpStatus::Unbounded
            };
        }

        let mut a: Vec<Vec<Rational>> = Vec::with_capacity(m);
        let mut b: Vec<Rational> = Vec::with_capacity(m);
        let mut slack_at = self.nvars;
        for (coeffs, rel, rhs) in &self.rows {
            let mut row = coeffs.clone();
            row.resize(n_total, Rational::from(0));
            match rel {
                Rel::Eq => {}
                // a·x ≥ b  ⇒  a·x − s = b, s ≥ 0
                Rel::Ge => {
                    row[slack_at] = Rational::from(-1);
                    slack_at += 1;
                }
                // a·x ≤ b  ⇒  a·x + s = b, s ≥ 0
                Rel::Le => {
                    row[slack_at] = Rational::from(1);
                    slack_at += 1;
                }
            }
            let mut rhs = rhs.clone();
            // Phase 1 needs b ≥ 0 so the artificial basis is feasible.
            if rhs < 0 {
                for v in row.iter_mut() {
                    *v = -v.clone();
                }
                rhs = -rhs;
            }
            a.push(row);
            b.push(rhs);
        }

        // ---------------------------------------------------------------
        // Phase 1 — minimise the sum of artificial variables.
        // ---------------------------------------------------------------
        let n_phase1 = n_total + m;
        let mut tab: Vec<Vec<Rational>> = Vec::with_capacity(m);
        for (i, row) in a.iter().enumerate() {
            let mut r = row.clone();
            r.resize(n_phase1, Rational::from(0));
            r[n_total + i] = Rational::from(1);
            r.push(b[i].clone()); // rhs in the last column
            tab.push(r);
        }
        let mut basis: Vec<usize> = (n_total..n_phase1).collect();

        let mut cost = vec![Rational::from(0); n_phase1];
        for c in cost.iter_mut().skip(n_total) {
            *c = Rational::from(1);
        }
        match simplex(&mut tab, &mut basis, &cost, n_phase1) {
            SimplexOutcome::Optimal(value) => {
                if value > 0 {
                    return LpStatus::Infeasible;
                }
            }
            SimplexOutcome::Unbounded => return LpStatus::Unbounded,
            SimplexOutcome::Exhausted => return LpStatus::Exhausted,
        }

        // Drive any artificial still in the basis (necessarily at level 0) out.
        // A row that cannot be pivoted is linearly dependent and gets dropped.
        let mut keep: Vec<bool> = vec![true; tab.len()];
        for i in 0..tab.len() {
            if basis[i] < n_total {
                continue;
            }
            let mut pivoted = false;
            for j in 0..n_total {
                if tab[i][j] != 0 {
                    pivot(&mut tab, &mut basis, i, j);
                    pivoted = true;
                    break;
                }
            }
            if !pivoted {
                keep[i] = false;
            }
        }
        let mut idx = 0;
        tab.retain(|_| {
            let k = keep[idx];
            idx += 1;
            k
        });
        idx = 0;
        basis.retain(|_| {
            let k = keep[idx];
            idx += 1;
            k
        });

        // ---------------------------------------------------------------
        // Phase 2 — the real objective, artificial columns dropped.
        // ---------------------------------------------------------------
        for row in tab.iter_mut() {
            let rhs = row[n_phase1].clone();
            row.truncate(n_total);
            row.push(rhs);
        }
        let mut cost2 = self.objective.clone();
        cost2.resize(n_total, Rational::from(0));
        match simplex(&mut tab, &mut basis, &cost2, n_total) {
            SimplexOutcome::Optimal(_) => {}
            SimplexOutcome::Unbounded => return LpStatus::Unbounded,
            SimplexOutcome::Exhausted => return LpStatus::Exhausted,
        }

        let mut x = vec![Rational::from(0); n_total];
        for (i, &bi) in basis.iter().enumerate() {
            if bi < n_total {
                x[bi] = tab[i][n_total].clone();
            }
        }
        x.truncate(self.nvars);
        LpStatus::Optimal(x)
    }
}

enum SimplexOutcome {
    Optimal(Rational),
    Unbounded,
    Exhausted,
}

/// Core simplex loop.  `tab` rows are `[a_0 … a_{n-1} | rhs]` in canonical form
/// with respect to `basis`; `cost` is the objective being minimised.
fn simplex(
    tab: &mut [Vec<Rational>],
    basis: &mut [usize],
    cost: &[Rational],
    n: usize,
) -> SimplexOutcome {
    let m = tab.len();
    // Reduced costs d_j = c_j − c_B · B⁻¹A_j, computed directly from the
    // canonical tableau, plus the current objective value.
    let mut d = vec![Rational::from(0); n + 1];
    d[..n].clone_from_slice(&cost[..n]);
    for (i, row) in tab.iter().enumerate().take(m) {
        let cb = cost[basis[i]].clone();
        if cb == 0 {
            continue;
        }
        for j in 0..=n {
            d[j] -= Rational::from(&cb * &row[j]);
        }
    }

    // Bland's rule guarantees termination, so the budget is a pure backstop
    // against a bug rather than part of the algorithm.
    let budget = 200_000usize.saturating_add(50 * m * (n + 1));
    for _ in 0..budget {
        // Entering: lowest-index column with a strictly negative reduced cost.
        let mut enter = None;
        for (j, dj) in d.iter().enumerate().take(n) {
            if *dj < 0 {
                enter = Some(j);
                break;
            }
        }
        let Some(j) = enter else {
            return SimplexOutcome::Optimal(-d[n].clone());
        };

        // Leaving: min ratio, ties broken by smallest basis index (Bland).
        let mut leave: Option<usize> = None;
        let mut best_ratio: Option<Rational> = None;
        for i in 0..m {
            if tab[i][j] <= 0 {
                continue;
            }
            let ratio = Rational::from(&tab[i][n] / &tab[i][j]);
            let better = match &best_ratio {
                None => true,
                Some(r) => ratio < *r || (ratio == *r && basis[i] < basis[leave.unwrap()]),
            };
            if better {
                best_ratio = Some(ratio);
                leave = Some(i);
            }
        }
        let Some(i) = leave else {
            return SimplexOutcome::Unbounded;
        };

        pivot(tab, basis, i, j);

        // Re-canonicalise the reduced-cost row on the entering column.
        if d[j] != 0 {
            let f = d[j].clone();
            for k in 0..=n {
                d[k] -= Rational::from(&f * &tab[i][k]);
            }
        }
    }
    SimplexOutcome::Exhausted
}

/// Gauss–Jordan pivot on `tab[i][j]`, updating the basis.
fn pivot(tab: &mut [Vec<Rational>], basis: &mut [usize], i: usize, j: usize) {
    let p = tab[i][j].clone();
    debug_assert!(p != 0);
    for v in tab[i].iter_mut() {
        *v /= &p;
    }
    let prow = tab[i].clone();
    for (k, row) in tab.iter_mut().enumerate() {
        if k == i {
            continue;
        }
        let f = row[j].clone();
        if f == 0 {
            continue;
        }
        for (t, pv) in row.iter_mut().zip(prow.iter()) {
            *t -= Rational::from(&f * pv);
        }
    }
    basis[i] = j;
}

#[cfg(test)]
mod tests {
    use super::*;

    fn q(n: i64) -> Rational {
        Rational::from(n)
    }

    #[test]
    fn trivial_feasible_system() {
        // x0 + x1 = 1, x ≥ 0
        let mut lp = Lp::new(2);
        lp.constrain(vec![q(1), q(1)], Rel::Eq, q(1));
        match lp.solve() {
            LpStatus::Optimal(x) => {
                assert_eq!(x[0].clone() + x[1].clone(), 1);
                assert!(x.iter().all(|v| *v >= 0));
            }
            other => panic!("expected optimal, got {other:?}"),
        }
    }

    #[test]
    fn detects_infeasibility() {
        // x0 = 1 and x0 = 2 cannot both hold.
        let mut lp = Lp::new(1);
        lp.constrain(vec![q(1)], Rel::Eq, q(1));
        lp.constrain(vec![q(1)], Rel::Eq, q(2));
        assert_eq!(lp.solve(), LpStatus::Infeasible);
    }

    #[test]
    fn nonnegativity_makes_system_infeasible() {
        // x0 = −1 with x0 ≥ 0.
        let mut lp = Lp::new(1);
        lp.constrain(vec![q(1)], Rel::Eq, q(-1));
        assert_eq!(lp.solve(), LpStatus::Infeasible);
    }

    #[test]
    fn optimises_a_bounded_objective() {
        // minimise −x0 − x1 s.t. x0 + 2x1 ≤ 4, 3x0 + x1 ≤ 6  ⇒  (8/5, 6/5).
        let mut lp = Lp::new(2);
        lp.constrain(vec![q(1), q(2)], Rel::Le, q(4));
        lp.constrain(vec![q(3), q(1)], Rel::Le, q(6));
        lp.set_objective(0, q(-1));
        lp.set_objective(1, q(-1));
        match lp.solve() {
            LpStatus::Optimal(x) => {
                assert_eq!(x[0], Rational::from((8, 5)));
                assert_eq!(x[1], Rational::from((6, 5)));
            }
            other => panic!("expected optimal, got {other:?}"),
        }
    }

    #[test]
    fn detects_unboundedness() {
        // minimise −x0 with only x0 ≥ 1.
        let mut lp = Lp::new(1);
        lp.constrain(vec![q(1)], Rel::Ge, q(1));
        lp.set_objective(0, q(-1));
        assert_eq!(lp.solve(), LpStatus::Unbounded);
    }

    #[test]
    fn handles_degenerate_rows_without_cycling() {
        // A deliberately degenerate system: duplicated and dependent rows.
        let mut lp = Lp::new(3);
        lp.constrain(vec![q(1), q(1), q(1)], Rel::Eq, q(1));
        lp.constrain(vec![q(2), q(2), q(2)], Rel::Eq, q(2));
        lp.constrain(vec![q(1), q(0), q(0)], Rel::Le, q(1));
        lp.set_objective(2, q(-1));
        match lp.solve() {
            LpStatus::Optimal(x) => {
                let s: Rational = x.iter().fold(Rational::from(0), |a, v| a + v.clone());
                assert_eq!(s, 1);
            }
            other => panic!("expected optimal, got {other:?}"),
        }
    }
}
