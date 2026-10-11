//! Dense two-phase simplex over ℚ with Bland's rule, reading its certificates
//! off the final tableau.
//!
//! The caller's programme is rewritten in standard form
//! `min c'·x'  s.t.  A' x' = b',  x' ≥ 0`:
//!
//! * a variable with a finite lower bound becomes `x = l + p` (and, when its
//!   upper bound is finite too, gains a row `p + s = u − l`);
//! * a variable with only an upper bound becomes `x = u − n`;
//! * a free variable becomes `x = p − n`;
//! * every `≤` row gains a slack.
//!
//! Each row gets an artificial column, and the artificial columns are kept
//! in the tableau through both phases (they never re-enter in phase 2). They
//! start as the identity, so at any moment they hold `B⁻¹` (up to the row
//! sign flips that made `b' ≥ 0`), which is what makes the dual vector
//! `y = c_B B⁻¹` — and hence every certificate — directly readable.
//!
//! The mapping back to the caller's rows and bounds is checked by the exact
//! verifiers in [`super`], not trusted.

use rug::Rational;

use super::{
    FarkasCertificate, LinearProgram, LinprogError, LpOutcome, OptimalCertificate, Sense,
    UnboundedCertificate,
};
use crate::budget;

/// Standard-form image of a [`LinearProgram`].
struct StdForm {
    /// For each structural column: the caller variable and the sign with
    /// which the column enters it (`x_j = shift_j + Σ sign · x'_col`).
    structural: Vec<(usize, bool)>,
    /// Total number of columns (structural then slack).
    ncols: usize,
    /// Rows of `A'`.
    a: Vec<Vec<Rational>>,
    /// `b'`.
    b: Vec<Rational>,
    /// `c'` (always minimised), one per column.
    cost: Vec<Rational>,
    /// The shift of each caller variable.
    shift: Vec<Rational>,
    /// For each bound row (after the `≤` and `=` rows), the variable it bounds.
    bound_rows: Vec<usize>,
}

fn standard_form(lp: &LinearProgram) -> StdForm {
    let n = lp.nvars();
    let mut structural: Vec<(usize, bool)> = Vec::new();
    let mut shift = vec![Rational::new(); n];
    let mut var_cols: Vec<Vec<(usize, bool)>> = vec![Vec::new(); n];
    let mut bound_rows = Vec::new();
    for (j, bd) in lp.bounds.iter().enumerate() {
        match (&bd.lower, &bd.upper) {
            (Some(l), upper) => {
                shift[j] = l.clone();
                var_cols[j].push((structural.len(), true));
                structural.push((j, true));
                if upper.is_some() {
                    bound_rows.push(j);
                }
            }
            (None, Some(u)) => {
                shift[j] = u.clone();
                var_cols[j].push((structural.len(), false));
                structural.push((j, false));
            }
            (None, None) => {
                var_cols[j].push((structural.len(), true));
                structural.push((j, true));
                var_cols[j].push((structural.len(), false));
                structural.push((j, false));
            }
        }
    }
    let nstruct = structural.len();
    let n_slack = lp.a_ub.len() + bound_rows.len();
    let ncols = nstruct + n_slack;

    let mut a = Vec::new();
    let mut b = Vec::new();
    let expand = |row: &[Rational], rhs: &Rational| -> (Vec<Rational>, Rational) {
        let mut out = vec![Rational::new(); ncols];
        let mut r = rhs.clone();
        for (j, aij) in row.iter().enumerate() {
            if *aij == 0 {
                continue;
            }
            r -= Rational::from(aij * &shift[j]);
            for &(col, pos) in &var_cols[j] {
                out[col] = if pos { aij.clone() } else { -aij.clone() };
            }
        }
        (out, r)
    };
    let mut slack = nstruct;
    for (row, rhs) in lp.a_ub.iter().zip(&lp.b_ub) {
        let (mut r, rb) = expand(row, rhs);
        r[slack] = Rational::from(1);
        slack += 1;
        a.push(r);
        b.push(rb);
    }
    for (row, rhs) in lp.a_eq.iter().zip(&lp.b_eq) {
        let (r, rb) = expand(row, rhs);
        a.push(r);
        b.push(rb);
    }
    for &j in &bound_rows {
        let mut r = vec![Rational::new(); ncols];
        r[var_cols[j][0].0] = Rational::from(1);
        r[slack] = Rational::from(1);
        slack += 1;
        let bd = &lp.bounds[j];
        let (Some(l), Some(u)) = (&bd.lower, &bd.upper) else {
            unreachable!("bound rows are only made for doubly bounded variables")
        };
        a.push(r);
        b.push(Rational::from(u - l));
    }

    let mut cost = vec![Rational::new(); ncols];
    for (col, &(j, pos)) in structural.iter().enumerate() {
        let cj = match lp.sense {
            Sense::Minimize => lp.c[j].clone(),
            Sense::Maximize => -lp.c[j].clone(),
        };
        cost[col] = if pos { cj } else { -cj };
    }

    StdForm {
        structural,
        ncols,
        a,
        b,
        cost,
        shift,
        bound_rows,
    }
}

enum Phase {
    Optimal,
    /// Unbounded along the given entering column.
    Unbounded(usize),
}

struct Tableau {
    /// `m` rows of width `ncols + m + 1`: columns, artificials, rhs.
    rows: Vec<Vec<Rational>>,
    basis: Vec<usize>,
    /// `+1` / `−1`: the sign each row was multiplied by to make `b' ≥ 0`.
    flip: Vec<bool>,
    ncols: usize,
}

impl Tableau {
    fn m(&self) -> usize {
        self.rows.len()
    }

    fn rhs(&self) -> usize {
        self.ncols + self.m()
    }

    fn pivot(&mut self, i: usize, j: usize) {
        let p = self.rows[i][j].clone();
        debug_assert!(p != 0);
        if p != 1 {
            for v in self.rows[i].iter_mut() {
                if *v != 0 {
                    *v /= &p;
                }
            }
        }
        let prow = self.rows[i].clone();
        for (k, row) in self.rows.iter_mut().enumerate() {
            if k == i {
                continue;
            }
            let f = row[j].clone();
            if f == 0 {
                continue;
            }
            for (t, pv) in row.iter_mut().zip(prow.iter()) {
                if *pv != 0 {
                    *t -= Rational::from(&f * pv);
                }
            }
        }
        self.basis[i] = j;
    }

    /// Run the simplex on `cost` (one entry per non-rhs column), entering
    /// only columns `< allowed`.
    fn run(&mut self, cost: &[Rational], allowed: usize) -> Result<Phase, LinprogError> {
        let rhs = self.rhs();
        let m = self.m();
        let mut d: Vec<Rational> = cost.to_vec();
        d.push(Rational::new());
        for (i, row) in self.rows.iter().enumerate() {
            let cb = &cost[self.basis[i]];
            if *cb == 0 {
                continue;
            }
            for (dj, rj) in d.iter_mut().zip(row) {
                if *rj != 0 {
                    *dj -= Rational::from(cb * rj);
                }
            }
        }
        // Bland's rule makes cycling impossible, so this cap is a backstop
        // against a bug, not part of the algorithm.
        let cap = 1_000_000usize.saturating_add(100usize.saturating_mul(m * (rhs + 1)));
        for _ in 0..cap {
            budget::check()?;
            // Entering: lowest index with a negative reduced cost.
            let Some(j) = (0..allowed).find(|&j| d[j] < 0) else {
                return Ok(Phase::Optimal);
            };
            // Leaving: minimum ratio, ties to the smallest basic index.
            let mut leave: Option<(usize, Rational)> = None;
            for i in 0..m {
                let a = &self.rows[i][j];
                if *a <= 0 {
                    continue;
                }
                let ratio = Rational::from(&self.rows[i][rhs] / a);
                let better = match &leave {
                    None => true,
                    Some((k, r)) => ratio < *r || (ratio == *r && self.basis[i] < self.basis[*k]),
                };
                if better {
                    leave = Some((i, ratio));
                }
            }
            let Some((i, _)) = leave else {
                return Ok(Phase::Unbounded(j));
            };
            self.pivot(i, j);
            let f = d[j].clone();
            for (dk, pk) in d.iter_mut().zip(&self.rows[i]) {
                if *pk != 0 {
                    *dk -= Rational::from(&f * pk);
                }
            }
        }
        Err(LinprogError::Internal {
            detail: "pivot cap reached under Bland's rule".into(),
        })
    }

    /// `y_i = flip_i · Σ_k cost[basis_k] · B⁻¹[k][i]` — the standard-form
    /// dual for the row order of `A'`.
    fn dual(&self, cost: &[Rational]) -> Vec<Rational> {
        let m = self.m();
        let mut y = vec![Rational::new(); m];
        for (k, row) in self.rows.iter().enumerate() {
            let cb = &cost[self.basis[k]];
            if *cb == 0 {
                continue;
            }
            for (i, yi) in y.iter_mut().enumerate() {
                let binv = &row[self.ncols + i];
                if *binv != 0 {
                    *yi += Rational::from(cb * binv);
                }
            }
        }
        for (yi, &f) in y.iter_mut().zip(&self.flip) {
            if f {
                *yi = -yi.clone();
            }
        }
        y
    }

    /// The basic solution, one entry per column (artificials excluded).
    fn point(&self) -> Vec<Rational> {
        let rhs = self.rhs();
        let mut x = vec![Rational::new(); self.ncols];
        for (i, &bi) in self.basis.iter().enumerate() {
            if bi < self.ncols {
                x[bi] = self.rows[i][rhs].clone();
            }
        }
        x
    }
}

/// Solve `lp` (already validated); the caller verifies the certificate.
pub(super) fn solve(lp: &LinearProgram) -> Result<LpOutcome, LinprogError> {
    let sf = standard_form(lp);
    let m = sf.a.len();
    let ncols = sf.ncols;
    let width = ncols + m + 1;
    budget::check_growth((m as u64).saturating_mul(width as u64))?;

    let mut rows = Vec::with_capacity(m);
    let mut flip = Vec::with_capacity(m);
    for (i, (arow, bi)) in sf.a.iter().zip(&sf.b).enumerate() {
        let neg = *bi < 0;
        let mut r: Vec<Rational> = Vec::with_capacity(width);
        for v in arow {
            r.push(if neg { -v.clone() } else { v.clone() });
        }
        r.resize(width, Rational::new());
        r[ncols + i] = Rational::from(1);
        r[width - 1] = if neg { -bi.clone() } else { bi.clone() };
        rows.push(r);
        flip.push(neg);
    }
    let mut tab = Tableau {
        rows,
        basis: (ncols..ncols + m).collect(),
        flip,
        ncols,
    };

    // Phase 1: minimise the sum of the artificials.
    let mut cost1 = vec![Rational::new(); ncols + m];
    for c in cost1.iter_mut().skip(ncols) {
        *c = Rational::from(1);
    }
    if let Phase::Unbounded(_) = tab.run(&cost1, ncols + m)? {
        return Err(LinprogError::Internal {
            detail: "phase 1 reported an unbounded sum of artificials".into(),
        });
    }
    let rhs = tab.rhs();
    let infeasibility: Rational = tab
        .basis
        .iter()
        .zip(&tab.rows)
        .filter(|(&bi, _)| bi >= ncols)
        .fold(Rational::new(), |acc, (_, row)| acc + &row[rhs]);
    if infeasibility > 0 {
        // y'' has y''ᵀA' ≤ 0 and y''·b' > 0; the Farkas vector is −y''.
        let f: Vec<Rational> = tab.dual(&cost1).into_iter().map(|v| -v).collect();
        return Ok(LpOutcome::Infeasible(farkas_from(lp, &sf, &f)));
    }

    // Drive the (zero-level) artificials out of the basis where possible; a
    // row that cannot be pivoted is redundant and keeps its artificial, which
    // phase 2 never moves.
    for i in 0..m {
        if tab.basis[i] < ncols {
            continue;
        }
        if let Some(j) = (0..ncols).find(|&j| tab.rows[i][j] != 0) {
            tab.pivot(i, j);
        }
    }

    // Phase 2.
    let mut cost2 = sf.cost.clone();
    cost2.resize(ncols + m, Rational::new());
    match tab.run(&cost2, ncols)? {
        Phase::Optimal => {
            let xs = tab.point();
            let y = tab.dual(&cost2);
            Ok(LpOutcome::Optimal(optimal_from(lp, &sf, &xs, &y)))
        }
        Phase::Unbounded(e) => {
            let xs = tab.point();
            let mut ds = vec![Rational::new(); ncols];
            ds[e] = Rational::from(1);
            for (k, &bk) in tab.basis.iter().enumerate() {
                if bk < ncols {
                    ds[bk] = -tab.rows[k][e].clone();
                }
            }
            let point = to_user(lp, &sf, &xs, true);
            let direction = to_user(lp, &sf, &ds, false);
            Ok(LpOutcome::Unbounded(UnboundedCertificate {
                point,
                direction,
            }))
        }
    }
}

/// Map a standard-form vector back to the caller's variables; `shifted`
/// adds the bound shifts (a point) or not (a direction).
fn to_user(lp: &LinearProgram, sf: &StdForm, xs: &[Rational], shifted: bool) -> Vec<Rational> {
    let mut x = if shifted {
        sf.shift.clone()
    } else {
        vec![Rational::new(); lp.nvars()]
    };
    for (col, &(j, pos)) in sf.structural.iter().enumerate() {
        if xs[col] == 0 {
            continue;
        }
        if pos {
            x[j] += &xs[col];
        } else {
            x[j] -= &xs[col];
        }
    }
    x
}

/// `g = A_ubᵀ y_ub + A_eqᵀ y_eq + extra`.
fn combine(lp: &LinearProgram, y_ub: &[Rational], y_eq: &[Rational]) -> Vec<Rational> {
    let mut g = vec![Rational::new(); lp.nvars()];
    super::add_transpose(&mut g, &lp.a_ub, y_ub);
    super::add_transpose(&mut g, &lp.a_eq, y_eq);
    g
}

fn optimal_from(
    lp: &LinearProgram,
    sf: &StdForm,
    xs: &[Rational],
    y: &[Rational],
) -> OptimalCertificate {
    let n = lp.nvars();
    let m_ub = lp.a_ub.len();
    let m_eq = lp.a_eq.len();
    let y_ub = y[..m_ub].to_vec();
    let y_eq = y[m_ub..m_ub + m_eq].to_vec();
    let mut z_upper = vec![Rational::new(); n];
    for (k, &j) in sf.bound_rows.iter().enumerate() {
        z_upper[j] = y[m_ub + m_eq + k].clone();
    }
    // Everything is in minimisation form here.
    let c_min: Vec<Rational> = match lp.sense {
        Sense::Minimize => lp.c.clone(),
        Sense::Maximize => lp.c.iter().map(|v| -v.clone()).collect(),
    };
    let g = combine(lp, &y_ub, &y_eq);
    let mut z_lower = vec![Rational::new(); n];
    for j in 0..n {
        let r = Rational::from(&c_min[j] - &g[j]) - &z_upper[j];
        let bd = &lp.bounds[j];
        if bd.lower.is_some() {
            z_lower[j] = r;
        } else if bd.upper.is_some() {
            z_upper[j] = r;
        }
        // A free variable has r = 0; the verifier checks it.
    }
    let x = to_user(lp, sf, xs, true);
    let objective = lp.objective_at(&x);
    let neg = lp.sense == Sense::Maximize;
    let fix = |v: Vec<Rational>| -> Vec<Rational> {
        if neg {
            v.into_iter().map(|t| -t).collect()
        } else {
            v
        }
    };
    OptimalCertificate {
        x,
        objective,
        y_ub: fix(y_ub),
        y_eq: fix(y_eq),
        z_lower: fix(z_lower),
        z_upper: fix(z_upper),
    }
}

fn farkas_from(lp: &LinearProgram, sf: &StdForm, f: &[Rational]) -> FarkasCertificate {
    let n = lp.nvars();
    let m_ub = lp.a_ub.len();
    let m_eq = lp.a_eq.len();
    let y_ub = f[..m_ub].to_vec();
    let y_eq = f[m_ub..m_ub + m_eq].to_vec();
    let mut w_upper = vec![Rational::new(); n];
    for (k, &j) in sf.bound_rows.iter().enumerate() {
        w_upper[j] = f[m_ub + m_eq + k].clone();
    }
    let g = combine(lp, &y_ub, &y_eq);
    let mut w_lower = vec![Rational::new(); n];
    for j in 0..n {
        let bd = &lp.bounds[j];
        let total = Rational::from(&g[j] + &w_upper[j]);
        if bd.lower.is_some() {
            w_lower[j] = total;
        } else if bd.upper.is_some() {
            w_upper[j] = -g[j].clone();
        }
    }
    FarkasCertificate {
        y_ub,
        y_eq,
        w_lower,
        w_upper,
    }
}
