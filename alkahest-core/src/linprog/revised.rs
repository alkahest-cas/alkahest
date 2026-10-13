//! Revised simplex over ℚ with an exact product-form factorisation of the
//! basis, warm starts, a dual simplex, and Gomory mixed-integer cuts read off
//! the optimal tableau.
//!
//! The programme is put in the same standard form `A' x' = b', x' ≥ 0` as the
//! dense tableau ([`super::simplex::standard_form`]); `A'` is stored by sparse
//! columns. Every row also owns an *artificial* column `±e_r` (sign chosen so
//! the artificial starts non-negative); artificials never enter the basis.
//!
//! `B⁻¹` is kept as an **eta file**: `B⁻¹ = T_k ⋯ T_1`, each `T` an elementary
//! column transformation recorded by its pivot row and the column `B⁻¹a_q` it
//! eliminated. *FTRAN* (`B⁻¹v`) applies the etas in order, *BTRAN* (`yᵀB⁻¹`) in
//! reverse. Every [`REFACTOR_EVERY`] updates the file is rebuilt from the
//! basis columns alone (Gaussian elimination in product form, unit columns
//! first, pivot rows chosen for sparsity), which bounds its length. All
//! arithmetic is exact, so the factorisation never drifts.
//!
//! Pricing is Dantzig's rule (most negative reduced cost). After
//! [`DEGENERATE_STREAK`] consecutive degenerate pivots the solver switches to
//! Bland's rule and stays there until a pivot strictly improves the
//! objective. Any cycle consists of degenerate pivots only, and every long
//! enough run of those is pure Bland, which cannot cycle — so the method
//! terminates. The dual simplex uses the same scheme with the dual Bland
//! rule.
//!
//! Certificates are read exactly as in the dense code: `y = c_B B⁻¹` (one
//! BTRAN) for the duals, `−y` of the phase-1 cost for a Farkas vector, a row
//! of `B⁻¹` for a dual-simplex infeasibility proof, and `B⁻¹a_q` for a ray.

use rug::{Integer, Rational};

use super::simplex::{farkas_from, optimal_from, standard_form, to_user, StdForm};
use super::{BasisEntry, LinearProgram, LinprogError, LpBasis, LpOutcome, UnboundedCertificate};
use crate::budget;

/// Rebuild the eta file after this many basis updates.
const REFACTOR_EVERY: usize = 64;
/// Switch to Bland's rule after this many consecutive degenerate pivots.
const DEGENERATE_STREAK: usize = 24;

/// One elementary transformation of the product form.
struct Eta {
    row: usize,
    pivot: Rational,
    /// The other non-zeros of the eliminated column, `(row, value)`.
    others: Vec<(usize, Rational)>,
}

pub(super) enum PrimalEnd {
    Optimal,
    Unbounded { q: usize, w: Vec<Rational> },
}

enum DualEnd {
    Optimal,
    Infeasible(Vec<Rational>),
}

/// The state of a revised simplex on one standard-form programme.
pub(super) struct Engine {
    pub(super) sf: StdForm,
    pub(super) m: usize,
    /// Number of real (non-artificial) columns.
    pub(super) ncols: usize,
    cols: Vec<Vec<(usize, Rational)>>,
    /// `true` when the artificial of row `r` is `−e_r`.
    art_neg: Vec<bool>,
    pub(super) b: Vec<Rational>,
    /// The column basic in each row position (`≥ ncols` is the artificial
    /// `ncols + r` of row `r`).
    pub(super) basis: Vec<usize>,
    /// Position of each column (real then artificial) in the basis.
    pos: Vec<Option<usize>>,
    /// Values of the basic variables, by position.
    pub(super) xb: Vec<Rational>,
    etas: Vec<Eta>,
    /// Length of the eta file right after the last refactorisation.
    n_base: usize,
    pub(super) iterations: usize,
    /// Phase-2 cost, one entry per column (artificials 0).
    pub(super) cost: Vec<Rational>,
}

/// Solve `lp` (validated) with the revised simplex, optionally warm-started.
/// Returns the engine at its final basis, the (unverified) outcome, and
/// whether the warm start was used.
pub(super) fn solve(
    lp: &LinearProgram,
    warm: Option<&LpBasis>,
) -> Result<(Engine, LpOutcome, bool), LinprogError> {
    if let Some(basis) = warm {
        let mut e = Engine::new(lp)?;
        if let Some(outcome) = e.warm(lp, basis)? {
            return Ok((e, outcome, true));
        }
    }
    let mut e = Engine::new(lp)?;
    let outcome = e.cold(lp)?;
    Ok((e, outcome, false))
}

/// Map standard-form columns to caller-facing basis labels.
pub(super) fn labels(lp: &LinearProgram, sf: &StdForm, cols: &[usize]) -> LpBasis {
    let first = first_columns(lp, sf);
    let nstruct = sf.structural.len();
    let m_ub = lp.a_ub.len();
    let entries = cols
        .iter()
        .filter_map(|&c| {
            if c < nstruct {
                let (j, _) = sf.structural[c];
                Some(if first[j] == c {
                    BasisEntry::Var(j)
                } else {
                    BasisEntry::VarNeg(j)
                })
            } else if c < sf.ncols {
                let k = c - nstruct;
                Some(if k < m_ub {
                    BasisEntry::Slack(k)
                } else {
                    BasisEntry::UpperSlack(sf.bound_rows[k - m_ub])
                })
            } else {
                None
            }
        })
        .collect();
    LpBasis { entries }
}

fn first_columns(lp: &LinearProgram, sf: &StdForm) -> Vec<usize> {
    let mut first = vec![usize::MAX; lp.nvars()];
    for (c, &(j, _)) in sf.structural.iter().enumerate() {
        if first[j] == usize::MAX {
            first[j] = c;
        }
    }
    first
}

/// The standard-form column a label names in `lp`, if it exists there.
fn label_column(lp: &LinearProgram, sf: &StdForm, first: &[usize], e: BasisEntry) -> Option<usize> {
    let nstruct = sf.structural.len();
    match e {
        BasisEntry::Var(j) => first.get(j).copied().filter(|&c| c != usize::MAX),
        BasisEntry::VarNeg(j) => {
            let c = *first.get(j)?;
            (c != usize::MAX && c + 1 < nstruct && sf.structural[c + 1].0 == j).then_some(c + 1)
        }
        BasisEntry::Slack(i) => (i < lp.a_ub.len()).then_some(nstruct + i),
        BasisEntry::UpperSlack(j) => sf
            .bound_rows
            .iter()
            .position(|&v| v == j)
            .map(|k| nstruct + lp.a_ub.len() + k),
    }
}

fn frac(v: &Rational) -> Rational {
    let fl = v.clone().floor();
    Rational::from(v - &fl)
}

fn is_int(v: &Rational) -> bool {
    *v.denom() == 1
}

impl Engine {
    pub(super) fn new(lp: &LinearProgram) -> Result<Engine, LinprogError> {
        let sf = standard_form(lp);
        let m = sf.a.len();
        let ncols = sf.ncols;
        budget::check_growth((m as u64).saturating_mul(ncols as u64 + 1))?;
        let mut cols: Vec<Vec<(usize, Rational)>> = vec![Vec::new(); ncols];
        for (i, row) in sf.a.iter().enumerate() {
            for (j, v) in row.iter().enumerate() {
                if *v != 0 {
                    cols[j].push((i, v.clone()));
                }
            }
        }
        let b = sf.b.clone();
        let mut cost = sf.cost.clone();
        cost.resize(ncols + m, Rational::new());
        Ok(Engine {
            sf,
            m,
            ncols,
            cols,
            art_neg: vec![false; m],
            b,
            basis: Vec::new(),
            pos: vec![None; ncols + m],
            xb: vec![Rational::new(); m],
            etas: Vec::new(),
            n_base: 0,
            iterations: 0,
            cost,
        })
    }

    // -- linear algebra ----------------------------------------------------

    fn column(&self, j: usize) -> Vec<(usize, Rational)> {
        if j < self.ncols {
            self.cols[j].clone()
        } else {
            let r = j - self.ncols;
            let v = if self.art_neg[r] { -1 } else { 1 };
            vec![(r, Rational::from(v))]
        }
    }

    /// `y · a_j`.
    pub(super) fn dot_col(&self, y: &[Rational], j: usize) -> Rational {
        if j < self.ncols {
            let mut s = Rational::new();
            for (i, a) in &self.cols[j] {
                if y[*i] != 0 {
                    s += Rational::from(a * &y[*i]);
                }
            }
            s
        } else {
            let r = j - self.ncols;
            if self.art_neg[r] {
                -y[r].clone()
            } else {
                y[r].clone()
            }
        }
    }

    fn ftran(&self, z: &mut [Rational]) {
        for eta in &self.etas {
            if z[eta.row] == 0 {
                continue;
            }
            let t = Rational::from(&z[eta.row] / &eta.pivot);
            for (i, w) in &eta.others {
                z[*i] -= Rational::from(w * &t);
            }
            z[eta.row] = t;
        }
    }

    fn btran(&self, y: &mut [Rational]) {
        for eta in self.etas.iter().rev() {
            let mut s = y[eta.row].clone();
            for (i, w) in &eta.others {
                if y[*i] != 0 {
                    s -= Rational::from(w * &y[*i]);
                }
            }
            if s != 0 {
                s /= &eta.pivot;
            }
            y[eta.row] = s;
        }
    }

    /// `B⁻¹ a_j`, dense.
    pub(super) fn ftran_col(&self, j: usize) -> Vec<Rational> {
        let mut z = vec![Rational::new(); self.m];
        for (i, v) in self.column(j) {
            z[i] = v;
        }
        self.ftran(&mut z);
        z
    }

    /// `B⁻¹ v`, dense.
    pub(super) fn ftran_vec(&self, v: &[Rational]) -> Vec<Rational> {
        let mut z = v.to_vec();
        self.ftran(&mut z);
        z
    }

    /// Row `r` of `B⁻¹`.
    pub(super) fn binv_row(&self, r: usize) -> Vec<Rational> {
        let mut y = vec![Rational::new(); self.m];
        y[r] = Rational::from(1);
        self.btran(&mut y);
        y
    }

    /// `c_B B⁻¹`.
    pub(super) fn duals(&self, cost: &[Rational]) -> Vec<Rational> {
        let mut y: Vec<Rational> = self.basis.iter().map(|&j| cost[j].clone()).collect();
        self.btran(&mut y);
        y
    }

    fn push_eta(&mut self, row: usize, w: &[Rational]) {
        let pivot = w[row].clone();
        let others: Vec<(usize, Rational)> = w
            .iter()
            .enumerate()
            .filter(|(i, v)| *i != row && **v != 0)
            .map(|(i, v)| (i, v.clone()))
            .collect();
        if pivot == 1 && others.is_empty() {
            return;
        }
        self.etas.push(Eta { row, pivot, others });
    }

    /// Rebuild the eta file from `cands` by Gaussian elimination in product
    /// form. Returns the column assigned to each row (`None` for rows no
    /// independent candidate covers) and the candidates rejected as
    /// dependent.
    fn factor(&mut self, cands: &[usize]) -> (Vec<Option<usize>>, Vec<usize>) {
        self.etas.clear();
        let mut order: Vec<usize> = cands.to_vec();
        let nnz = |j: usize| -> usize {
            if j < self.ncols {
                self.cols[j].len()
            } else {
                1
            }
        };
        order.sort_by_key(|&j| (nnz(j), j));
        // Static row counts over the candidates, to prefer sparse pivot rows.
        let mut row_count = vec![0usize; self.m];
        for &j in &order {
            for (i, _) in self.column(j) {
                row_count[i] += 1;
            }
        }
        let mut row_of: Vec<Option<usize>> = vec![None; self.m];
        let mut rejected = Vec::new();
        for &j in &order {
            let w = self.ftran_col(j);
            let mut best: Option<usize> = None;
            for (i, v) in w.iter().enumerate() {
                if *v == 0 || row_of[i].is_some() {
                    continue;
                }
                let key = |i: usize| {
                    let unit = w[i] == 1 || w[i] == -1;
                    (row_count[i], !unit, i)
                };
                if best.map_or(true, |b| key(i) < key(b)) {
                    best = Some(i);
                }
            }
            match best {
                Some(r) => {
                    row_of[r] = Some(j);
                    self.push_eta(r, &w);
                }
                None => rejected.push(j),
            }
        }
        self.n_base = self.etas.len();
        (row_of, rejected)
    }

    /// Install a complete basis (one column per row) and recompute `x_B`.
    fn install(&mut self, cols: &[usize]) -> Result<(), LinprogError> {
        let (row_of, rejected) = self.factor(cols);
        if !rejected.is_empty() || row_of.iter().any(|r| r.is_none()) {
            return Err(LinprogError::Internal {
                detail: "revised simplex: basis matrix became singular".into(),
            });
        }
        self.basis = row_of.into_iter().map(|c| c.unwrap()).collect();
        self.pos = vec![None; self.ncols + self.m];
        for (r, &c) in self.basis.iter().enumerate() {
            self.pos[c] = Some(r);
        }
        self.xb = self.ftran_vec(&self.b);
        Ok(())
    }

    fn maybe_refactor(&mut self) -> Result<(), LinprogError> {
        if self.etas.len() >= self.n_base + REFACTOR_EVERY {
            let cols = self.basis.clone();
            self.install(&cols)?;
        }
        Ok(())
    }

    pub(super) fn pivot(&mut self, r: usize, q: usize, w: &[Rational]) {
        let theta = Rational::from(&self.xb[r] / &w[r]);
        if theta != 0 {
            for (i, wi) in w.iter().enumerate() {
                if i != r && *wi != 0 {
                    self.xb[i] -= Rational::from(wi * &theta);
                }
            }
        }
        self.xb[r] = theta;
        let old = self.basis[r];
        self.pos[old] = None;
        self.basis[r] = q;
        self.pos[q] = Some(r);
        self.push_eta(r, w);
        self.iterations += 1;
    }

    fn pivot_cap(&self) -> usize {
        1_000_000usize.saturating_add(100usize.saturating_mul(self.m * (self.ncols + 1)))
    }

    fn slack_of_row(&self, r: usize, lp: &LinearProgram) -> Option<usize> {
        let nstruct = self.sf.structural.len();
        let m_ub = lp.a_ub.len();
        let m_eq = lp.a_eq.len();
        if r < m_ub {
            Some(nstruct + r)
        } else if r >= m_ub + m_eq {
            Some(nstruct + m_ub + (r - m_ub - m_eq))
        } else {
            None
        }
    }

    // -- the simplex methods -------------------------------------------------

    /// Primal simplex on `cost`; artificials never enter.
    pub(super) fn primal(&mut self, cost: &[Rational]) -> Result<PrimalEnd, LinprogError> {
        let mut bland = false;
        let mut streak = 0usize;
        let cap = self.pivot_cap();
        for _ in 0..cap {
            budget::check()?;
            self.maybe_refactor()?;
            let y = self.duals(cost);
            let mut enter: Option<(usize, Rational)> = None;
            for j in 0..self.ncols {
                if self.pos[j].is_some() {
                    continue;
                }
                let d = Rational::from(&cost[j] - &self.dot_col(&y, j));
                if d < 0 {
                    if bland {
                        enter = Some((j, d));
                        break;
                    }
                    if enter.as_ref().map_or(true, |(_, bd)| d < *bd) {
                        enter = Some((j, d));
                    }
                }
            }
            let Some((q, _)) = enter else {
                return Ok(PrimalEnd::Optimal);
            };
            let w = self.ftran_col(q);
            let mut leave: Option<(usize, Rational)> = None;
            for (i, wi) in w.iter().enumerate() {
                if *wi <= 0 {
                    continue;
                }
                let ratio = Rational::from(&self.xb[i] / wi);
                let better = match &leave {
                    None => true,
                    Some((k, rk)) => {
                        ratio < *rk || (ratio == *rk && self.basis[i] < self.basis[*k])
                    }
                };
                if better {
                    leave = Some((i, ratio));
                }
            }
            let Some((r, ratio)) = leave else {
                return Ok(PrimalEnd::Unbounded { q, w });
            };
            if ratio == 0 {
                streak += 1;
                if streak >= DEGENERATE_STREAK {
                    bland = true;
                }
            } else {
                streak = 0;
                bland = false;
            }
            self.pivot(r, q, &w);
        }
        Err(LinprogError::Internal {
            detail: "revised simplex: pivot cap reached".into(),
        })
    }

    /// Dual simplex on `cost` from a dual-feasible basis.
    fn dual(&mut self, cost: &[Rational]) -> Result<DualEnd, LinprogError> {
        let mut bland = false;
        let mut streak = 0usize;
        let cap = self.pivot_cap();
        for _ in 0..cap {
            budget::check()?;
            self.maybe_refactor()?;
            let mut leave: Option<usize> = None;
            for i in 0..self.m {
                if self.xb[i] >= 0 {
                    continue;
                }
                let better = match leave {
                    None => true,
                    Some(k) if bland => self.basis[i] < self.basis[k],
                    Some(k) => {
                        self.xb[i] < self.xb[k]
                            || (self.xb[i] == self.xb[k] && self.basis[i] < self.basis[k])
                    }
                };
                if better {
                    leave = Some(i);
                }
            }
            let Some(r) = leave else {
                return Ok(DualEnd::Optimal);
            };
            let rho = self.binv_row(r);
            let y = self.duals(cost);
            let mut enter: Option<(usize, Rational)> = None;
            for j in 0..self.ncols {
                if self.pos[j].is_some() {
                    continue;
                }
                let alpha = self.dot_col(&rho, j);
                if alpha >= 0 {
                    continue;
                }
                let d = Rational::from(&cost[j] - &self.dot_col(&y, j));
                let ratio = Rational::from(&d / &alpha).abs();
                if enter.as_ref().map_or(true, |(_, best)| ratio < *best) {
                    enter = Some((j, ratio));
                }
            }
            let Some((q, ratio)) = enter else {
                return Ok(DualEnd::Infeasible(rho));
            };
            if ratio == 0 {
                streak += 1;
                if streak >= DEGENERATE_STREAK {
                    bland = true;
                }
            } else {
                streak = 0;
                bland = false;
            }
            let w = self.ftran_col(q);
            self.pivot(r, q, &w);
        }
        Err(LinprogError::Internal {
            detail: "dual simplex: pivot cap reached".into(),
        })
    }

    /// Pivot zero-level artificials out of the basis where a real column can
    /// replace them. One that cannot sits on a redundant row whose row of
    /// `B⁻¹A'` is identically zero, so no later pivot moves it.
    fn drive_out_artificials(&mut self) -> Result<(), LinprogError> {
        for r in 0..self.m {
            if self.basis[r] < self.ncols {
                continue;
            }
            budget::check()?;
            let rho = self.binv_row(r);
            let found =
                (0..self.ncols).find(|&j| self.pos[j].is_none() && self.dot_col(&rho, j) != 0);
            if let Some(j) = found {
                let w = self.ftran_col(j);
                self.pivot(r, j, &w);
            }
        }
        Ok(())
    }

    /// Cold start: slack crash basis, phase 1 on the artificials, phase 2.
    fn cold(&mut self, lp: &LinearProgram) -> Result<LpOutcome, LinprogError> {
        let mut cols = Vec::with_capacity(self.m);
        for r in 0..self.m {
            match self.slack_of_row(r, lp) {
                Some(s) if self.b[r] >= 0 => cols.push(s),
                _ => {
                    self.art_neg[r] = self.b[r] < 0;
                    cols.push(self.ncols + r);
                }
            }
        }
        self.install(&cols)?;
        if self.basis.iter().any(|&c| c >= self.ncols) {
            let mut cost1 = vec![Rational::new(); self.ncols + self.m];
            for c in cost1.iter_mut().skip(self.ncols) {
                *c = Rational::from(1);
            }
            if let PrimalEnd::Unbounded { .. } = self.primal(&cost1)? {
                return Err(LinprogError::Internal {
                    detail: "phase 1 reported an unbounded sum of artificials".into(),
                });
            }
            let mut infeasibility = Rational::new();
            for (r, &c) in self.basis.iter().enumerate() {
                if c >= self.ncols {
                    infeasibility += &self.xb[r];
                }
            }
            if infeasibility > 0 {
                // y₁ᵀA' ≤ 0 and y₁·b' > 0; the Farkas vector is −y₁.
                let f: Vec<Rational> = self.duals(&cost1).into_iter().map(|v| -v).collect();
                return Ok(LpOutcome::Infeasible(farkas_from(lp, &self.sf, &f)));
            }
            self.drive_out_artificials()?;
        }
        self.phase2(lp)
    }

    /// Warm start from `basis`; `None` when it cannot be used.
    fn warm(
        &mut self,
        lp: &LinearProgram,
        basis: &LpBasis,
    ) -> Result<Option<LpOutcome>, LinprogError> {
        let first = first_columns(lp, &self.sf);
        let mut cands: Vec<usize> = Vec::new();
        for &e in &basis.entries {
            if let Some(c) = label_column(lp, &self.sf, &first, e) {
                if !cands.contains(&c) {
                    cands.push(c);
                }
            }
        }
        let (row_of, _) = self.factor(&cands);
        let mut cols: Vec<usize> = row_of.iter().flatten().copied().collect();
        for (r, assigned) in row_of.iter().enumerate() {
            if assigned.is_some() {
                continue;
            }
            match self.slack_of_row(r, lp) {
                Some(s) if !cols.contains(&s) => cols.push(s),
                _ => cols.push(self.ncols + r),
            }
        }
        self.install(&cols)?;
        // Artificials are admissible only at level zero.
        for (r, &c) in self.basis.iter().enumerate() {
            if c >= self.ncols && self.xb[r] != 0 {
                return Ok(None);
            }
        }
        self.drive_out_artificials()?;
        if self.xb.iter().all(|v| *v >= 0) {
            return self.phase2(lp).map(Some);
        }
        let cost = self.cost.clone();
        let y = self.duals(&cost);
        let dual_feasible = (0..self.ncols)
            .filter(|&j| self.pos[j].is_none())
            .all(|j| Rational::from(&cost[j] - &self.dot_col(&y, j)) >= 0);
        if !dual_feasible {
            return Ok(None);
        }
        match self.dual(&cost)? {
            DualEnd::Optimal => self.phase2(lp).map(Some),
            DualEnd::Infeasible(rho) => {
                // ρᵀA' ≥ 0 and ρ·b' = x_r < 0: ρ is the Farkas vector.
                Ok(Some(LpOutcome::Infeasible(farkas_from(lp, &self.sf, &rho))))
            }
        }
    }

    fn phase2(&mut self, lp: &LinearProgram) -> Result<LpOutcome, LinprogError> {
        let cost = self.cost.clone();
        match self.primal(&cost)? {
            PrimalEnd::Optimal => Ok(self.optimal_outcome(lp)),
            PrimalEnd::Unbounded { q, w } => {
                let xs = self.point();
                let mut ds = vec![Rational::new(); self.ncols];
                ds[q] = Rational::from(1);
                for (i, &bi) in self.basis.iter().enumerate() {
                    if bi < self.ncols {
                        ds[bi] = -w[i].clone();
                    }
                }
                Ok(LpOutcome::Unbounded(UnboundedCertificate {
                    point: to_user(lp, &self.sf, &xs, true),
                    direction: to_user(lp, &self.sf, &ds, false),
                }))
            }
        }
    }

    pub(super) fn optimal_outcome(&self, lp: &LinearProgram) -> LpOutcome {
        let xs = self.point();
        let y = self.duals(&self.cost);
        LpOutcome::Optimal(optimal_from(lp, &self.sf, &xs, &y))
    }

    /// The basic solution over the real columns.
    pub(super) fn point(&self) -> Vec<Rational> {
        let mut x = vec![Rational::new(); self.ncols];
        for (i, &bi) in self.basis.iter().enumerate() {
            if bi < self.ncols {
                x[bi] = self.xb[i].clone();
            }
        }
        x
    }

    pub(super) fn basis_labels(&self, lp: &LinearProgram) -> LpBasis {
        labels(lp, &self.sf, &self.basis)
    }

    pub(super) fn is_basic(&self, j: usize) -> bool {
        self.pos[j].is_some()
    }

    // -- Gomory mixed-integer cuts -------------------------------------------

    /// Gomory mixed-integer cuts from the rows of the current (optimal)
    /// tableau whose basic variable is an integer at a fractional value,
    /// translated to rows `a·x ≤ β` in the caller's variables. `x_star` is the
    /// current optimum, which every returned cut separates.
    ///
    /// Validity: the tableau row `x_B + Σ ā_k x_k = b̄` holds for every point
    /// of `A'x' = b'`, and the GMI inequality derived from it holds for every
    /// such point with `x' ≥ 0` and `x'_k ∈ ℤ` for the columns treated as
    /// integer. A column is treated as integer only when its value is an
    /// integer at every integer-feasible point of the caller's programme (an
    /// integer variable with an integer shift, a bound slack with integer
    /// bounds, an inequality slack whose row has integer data over integer
    /// variables only); everything else is treated as continuous, which only
    /// weakens the cut. A cut that would need the split parts of a free
    /// variable separately is skipped.
    pub(super) fn gomory_cuts(
        &self,
        lp: &LinearProgram,
        integer: &[bool],
        x_star: &[Rational],
        max_cuts: usize,
    ) -> Vec<(Vec<Rational>, Rational)> {
        let n = lp.nvars();
        let nstruct = self.sf.structural.len();
        let m_ub = lp.a_ub.len();
        // (constant, sparse user coefficients) of each real column, or None.
        type Affine = Option<(Rational, Vec<(usize, Rational)>)>;
        let mut col_int = vec![false; self.ncols];
        let mut col_aff: Vec<Affine> = Vec::with_capacity(self.ncols);
        for c in 0..self.ncols {
            if c < nstruct {
                let (j, pos) = self.sf.structural[c];
                let bd = &lp.bounds[j];
                let shift = &self.sf.shift[j];
                col_int[c] = integer[j] && is_int(shift);
                if bd.lower.is_none() && bd.upper.is_none() {
                    col_aff.push(None);
                } else if pos {
                    col_aff.push(Some((-shift.clone(), vec![(j, Rational::from(1))])));
                } else {
                    col_aff.push(Some((shift.clone(), vec![(j, Rational::from(-1))])));
                }
            } else {
                let k = c - nstruct;
                if k < m_ub {
                    let row = &lp.a_ub[k];
                    col_int[c] = is_int(&lp.b_ub[k])
                        && row
                            .iter()
                            .enumerate()
                            .all(|(j, a)| *a == 0 || (integer[j] && is_int(a)));
                    let beta = row
                        .iter()
                        .enumerate()
                        .filter(|(_, a)| **a != 0)
                        .map(|(j, a)| (j, -a.clone()))
                        .collect();
                    col_aff.push(Some((lp.b_ub[k].clone(), beta)));
                } else {
                    let j = self.sf.bound_rows[k - m_ub];
                    let bd = &lp.bounds[j];
                    let (Some(l), Some(u)) = (&bd.lower, &bd.upper) else {
                        col_aff.push(None);
                        continue;
                    };
                    col_int[c] = integer[j] && is_int(l) && is_int(u);
                    col_aff.push(Some((u.clone(), vec![(j, Rational::from(-1))])));
                }
            }
        }

        // Source rows, most fractional first.
        let half = Rational::from((1, 2));
        let mut sources: Vec<(Rational, usize)> = (0..self.m)
            .filter(|&r| {
                let c = self.basis[r];
                c < self.ncols && col_int[c] && !is_int(&self.xb[r])
            })
            .map(|r| (Rational::from(&frac(&self.xb[r]) - &half).abs(), r))
            .collect();
        sources.sort();

        let mut cuts: Vec<(Vec<Rational>, Rational)> = Vec::new();
        for (_, r) in sources {
            if cuts.len() >= max_cuts {
                break;
            }
            if budget::check().is_err() {
                break;
            }
            let f0 = frac(&self.xb[r]);
            let one_minus_f0 = Rational::from(1) - &f0;
            let rho = self.binv_row(r);
            let mut row = vec![Rational::new(); n];
            let mut rhs = Rational::from(-1);
            let mut ok = true;
            for k in 0..self.ncols {
                if self.pos[k].is_some() {
                    continue;
                }
                let a = self.dot_col(&rho, k);
                if a == 0 {
                    continue;
                }
                let gamma = if col_int[k] {
                    let fk = frac(&a);
                    if fk <= f0 {
                        Rational::from(&fk / &f0)
                    } else {
                        (Rational::from(1) - fk) / &one_minus_f0
                    }
                } else if a > 0 {
                    Rational::from(&a / &f0)
                } else {
                    Rational::from(-&a) / &one_minus_f0
                };
                if gamma == 0 {
                    continue;
                }
                let Some((alpha, beta)) = &col_aff[k] else {
                    ok = false;
                    break;
                };
                // γ·(α + β·x) ≥ … contributes −γβ to the ≤ row and γα to β.
                rhs += Rational::from(&gamma * alpha);
                for (j, bj) in beta {
                    row[*j] -= Rational::from(&gamma * bj);
                }
            }
            if !ok || row.iter().all(|v| *v == 0) {
                continue;
            }
            strengthen_pure_integer(&mut row, &mut rhs, integer);
            // The cut must separate the current optimum.
            if super::dot(&row, x_star) <= rhs {
                continue;
            }
            // Keep coefficient sizes in check.
            let too_big = row
                .iter()
                .chain(std::iter::once(&rhs))
                .any(|v| v.numer().significant_bits() > 256 || v.denom().significant_bits() > 256);
            if too_big {
                continue;
            }
            if cuts.iter().any(|(r2, b2)| *r2 == row && *b2 == rhs) {
                continue;
            }
            cuts.push((row, rhs));
        }
        cuts
    }
}

/// When every variable in the cut is an integer, scale the row to coprime
/// integer coefficients and round the right-hand side down (a Chvátal–Gomory
/// step, valid because the left side is then an integer at every integer
/// point).
fn strengthen_pure_integer(row: &mut [Rational], rhs: &mut Rational, integer: &[bool]) {
    if row.iter().zip(integer).any(|(a, &int)| *a != 0 && !int) {
        return;
    }
    let mut l = Integer::from(1);
    for a in row.iter() {
        if *a != 0 {
            l.lcm_mut(a.denom());
        }
    }
    let lr = Rational::from(l);
    let mut g = Integer::new();
    for a in row.iter_mut() {
        *a *= &lr;
        g.gcd_mut(a.numer());
    }
    if g == 0 {
        return;
    }
    let gr = Rational::from(g);
    let scale = Rational::from(&lr / &gr);
    for a in row.iter_mut() {
        *a /= &gr;
    }
    *rhs *= &scale;
    *rhs = rhs.clone().floor();
}
