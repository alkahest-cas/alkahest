//! Parametric linear programming: the right-hand side *or* the objective
//! affine in one parameter `θ`.
//!
//! For a right-hand side `b(θ) = b + θ·δb` the optimal value is a convex
//! (concave when maximising) piecewise-affine function of `θ` on the closed
//! interval of `θ` where the programme is feasible; for an objective
//! `c(θ) = c + θ·δc` it is concave (convex when maximising) piecewise-affine on
//! the closed interval where it is bounded. Inside each *critical region* one
//! basis stays optimal, so the optimal point, the dual multipliers and the
//! value are affine in `θ` there.
//!
//! The algorithm is the classical one, exactly: find the interval where the
//! programme has an optimum by two auxiliary LPs (a feasibility LP in
//! `(x, θ)`, or a dual-feasibility LP in `(y, z, θ)`), solve at one `θ` in it,
//! then walk to the right and to the left. In a region the basis stays primal
//! feasible (RHS case: `x_B(θ) = B⁻¹b'(θ) ≥ 0`) or dual feasible (objective
//! case: reduced costs `d(θ) ≥ 0`) up to an exact breakpoint, the ratio of two
//! rationals. At a breakpoint the next basis comes from dual (RHS case) or
//! primal (objective case) simplex pivots on the programme perturbed to
//! `θ + ε`, under Bland's rule, which is finite and leaves a basis optimal on
//! an interval of positive length beyond the breakpoint.
//!
//! **Every optimal region is verified for all `θ` in it**, not sampled: each
//! certificate condition is affine in `θ`, so it holds on a closed interval
//! iff it holds at both ends. For an infinite end the *slope* of the
//! certificate is checked instead, which is itself an optimality certificate
//! of a derived programme (right-hand side `δb` with the finite bounds moved to
//! `0`, or objective `δc`) and is verified by the same exact checker.

use rug::Rational;

use super::revised::Engine;
use super::simplex::optimal_from;
use super::{
    dot, linprog, revised, LinearProgram, LinprogError, LpBasis, LpOutcome, OptimalCertificate,
    Sense, VarBound,
};
use crate::budget;

/// The direction in which the programme moves with `θ`.
#[derive(Clone, Debug, PartialEq, Eq)]
#[non_exhaustive]
pub enum ParametricDelta {
    /// `b_ub(θ) = b_ub + θ·b_ub_delta`, `b_eq(θ) = b_eq + θ·b_eq_delta`.
    Rhs {
        /// `δb_ub`, one entry per inequality row.
        b_ub: Vec<Rational>,
        /// `δb_eq`, one entry per equality row.
        b_eq: Vec<Rational>,
    },
    /// `c(θ) = c + θ·delta`.
    Objective {
        /// `δc`, one entry per variable.
        c: Vec<Rational>,
    },
}

/// The status of the programme on a region of `θ`.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
#[non_exhaustive]
pub enum RegionStatus {
    /// One basis is optimal throughout the (closed) region.
    Optimal,
    /// No feasible point for any `θ` in the (open-ended) region.
    Infeasible,
    /// The objective is unbounded for every `θ` in the (open-ended) region.
    Unbounded,
}

/// One critical region of a parametric programme.
///
/// Affine quantities are pairs `(a, b)` meaning `a + θ·b`.
#[derive(Clone, Debug, PartialEq, Eq)]
#[non_exhaustive]
pub struct ParametricRegion {
    /// Left end, `None` for `−∞`.
    pub lower: Option<Rational>,
    /// Right end, `None` for `+∞`.
    pub upper: Option<Rational>,
    /// What holds on the region. An optimal region is closed; an infeasible or
    /// unbounded one excludes the end it shares with an optimal region (and an
    /// infeasible one the end it shares with an unbounded one).
    pub status: RegionStatus,
    /// The optimal value `v₀ + θ·v₁` (optimal regions only).
    pub value: Option<(Rational, Rational)>,
    /// The optimal point `x₀ + θ·x₁`.
    pub x: Option<(Vec<Rational>, Vec<Rational>)>,
    /// Multipliers of the inequality rows, affine in `θ`.
    pub y_ub: Option<(Vec<Rational>, Vec<Rational>)>,
    /// Multipliers of the equality rows, affine in `θ`.
    pub y_eq: Option<(Vec<Rational>, Vec<Rational>)>,
    /// Multipliers of the lower bounds, affine in `θ`.
    pub z_lower: Option<(Vec<Rational>, Vec<Rational>)>,
    /// Multipliers of the upper bounds, affine in `θ`.
    pub z_upper: Option<(Vec<Rational>, Vec<Rational>)>,
    /// The optimal basis on the region.
    pub basis: Option<LpBasis>,
}

impl ParametricRegion {
    fn plain(lower: Option<Rational>, upper: Option<Rational>, status: RegionStatus) -> Self {
        ParametricRegion {
            lower,
            upper,
            status,
            value: None,
            x: None,
            y_ub: None,
            y_eq: None,
            z_lower: None,
            z_upper: None,
            basis: None,
        }
    }

    /// `true` when `θ` lies in the region (closed for an optimal region).
    pub fn contains(&self, theta: &Rational) -> bool {
        let lo = self.lower.as_ref().map_or(true, |l| theta >= l);
        let hi = self.upper.as_ref().map_or(true, |u| theta <= u);
        lo && hi
    }

    /// The optimality certificate at `θ` (optimal regions only).
    pub fn certificate_at(&self, theta: &Rational) -> Option<OptimalCertificate> {
        let at = |p: &Option<(Vec<Rational>, Vec<Rational>)>| -> Option<Vec<Rational>> {
            p.as_ref().map(|(a, b)| affine_at(a, b, theta))
        };
        let x = at(&self.x)?;
        let (v0, v1) = self.value.as_ref()?;
        Some(OptimalCertificate {
            x,
            objective: Rational::from(v0 + &Rational::from(theta * v1)),
            y_ub: at(&self.y_ub)?,
            y_eq: at(&self.y_eq)?,
            z_lower: at(&self.z_lower)?,
            z_upper: at(&self.z_upper)?,
        })
    }
}

/// The result of [`linprog_parametric`]: regions in increasing `θ`, covering
/// the requested range.
#[derive(Clone, Debug, PartialEq, Eq)]
#[non_exhaustive]
pub struct ParametricSolution {
    /// The regions, left to right.
    pub regions: Vec<ParametricRegion>,
    /// The interior ends of the regions (where the basis, or the status,
    /// changes), increasing.
    pub breakpoints: Vec<Rational>,
}

impl ParametricSolution {
    /// The region describing `θ` (an optimal one when `θ` is a breakpoint of
    /// optimal regions), or `None` outside the requested range.
    pub fn region_at(&self, theta: &Rational) -> Option<&ParametricRegion> {
        // At a shared end the closed set wins: the interval where an optimum
        // exists (optimal), and in the right-hand-side case the closed
        // feasible interval (unbounded) over the infeasible flanks.
        [
            RegionStatus::Optimal,
            RegionStatus::Unbounded,
            RegionStatus::Infeasible,
        ]
        .iter()
        .find_map(|s| {
            self.regions
                .iter()
                .find(|r| r.status == *s && r.contains(theta))
        })
    }

    /// The optimal value at `θ`, or `None` when the programme is infeasible or
    /// unbounded there (or `θ` is outside the range).
    pub fn value_at(&self, theta: &Rational) -> Option<Rational> {
        let r = self.region_at(theta)?;
        let (v0, v1) = r.value.as_ref()?;
        Some(Rational::from(v0 + &Rational::from(theta * v1)))
    }

    /// The status at `θ`.
    pub fn status_at(&self, theta: &Rational) -> Option<RegionStatus> {
        self.region_at(theta).map(|r| r.status)
    }

    /// Re-check, exactly, that every optimal region's certificate holds for
    /// all `θ` in the region (as [`linprog_parametric`] did before
    /// returning).
    pub fn verify(&self, lp: &LinearProgram, delta: &ParametricDelta) -> Result<(), String> {
        for r in &self.regions {
            if r.status == RegionStatus::Optimal {
                verify_region(lp, delta, r)?;
            }
        }
        Ok(())
    }
}

fn affine_at(a: &[Rational], b: &[Rational], t: &Rational) -> Vec<Rational> {
    a.iter()
        .zip(b)
        .map(|(a, b)| Rational::from(a + &Rational::from(b * t)))
        .collect()
}

/// The programme at a given `θ`.
pub(super) fn lp_at(
    lp: &LinearProgram,
    delta: &ParametricDelta,
    theta: &Rational,
) -> LinearProgram {
    let mut out = lp.clone();
    let shift = |v: &mut Vec<Rational>, d: &[Rational]| {
        for (x, dx) in v.iter_mut().zip(d) {
            *x += Rational::from(dx * theta);
        }
    };
    match delta {
        ParametricDelta::Rhs { b_ub, b_eq } => {
            shift(&mut out.b_ub, b_ub);
            shift(&mut out.b_eq, b_eq);
        }
        ParametricDelta::Objective { c } => shift(&mut out.c, c),
    }
    out
}

/// Solve `lp` with its right-hand side or objective moving affinely with a
/// parameter `θ ∈ [lower, upper]` (`None` for an infinite end), and return
/// every critical region with exact breakpoints.
///
/// # Errors
/// * `E-LP-001` — malformed shapes (a delta of the wrong length, or
///   `lower > upper`).
/// * `E-BUDGET-*` — the active budget tripped.
/// * `E-LP-006` — a region certificate failed its exact check (a bug).
///
/// # Example
/// ```
/// use alkahest_cas::linprog::{linprog_parametric, LinearProgram, ParametricDelta, Sense};
/// use rug::Rational;
/// let q = |n: i64| Rational::from(n);
/// // max x + y  s.t.  x + 2y ≤ 4 + θ,  3x + y ≤ 6,  x, y ≥ 0,  θ ∈ [−4, 10]
/// let mut lp = LinearProgram::new(vec![q(1), q(1)]);
/// lp.sense = Sense::Maximize;
/// lp.a_ub = vec![vec![q(1), q(2)], vec![q(3), q(1)]];
/// lp.b_ub = vec![q(4), q(6)];
/// let delta = ParametricDelta::Rhs { b_ub: vec![q(1), q(0)], b_eq: vec![] };
/// let sol = linprog_parametric(&lp, &delta, Some(q(-4)), Some(q(10))).unwrap();
/// assert_eq!(sol.breakpoints, vec![q(-2), q(8)]);
/// assert_eq!(sol.value_at(&q(0)), Some(Rational::from((14, 5))));
/// assert_eq!(sol.value_at(&q(10)), Some(q(6)));
/// ```
pub fn linprog_parametric(
    lp: &LinearProgram,
    delta: &ParametricDelta,
    lower: Option<Rational>,
    upper: Option<Rational>,
) -> Result<ParametricSolution, LinprogError> {
    lp.validate()?;
    let shape = |detail: String| Err(LinprogError::Shape { detail });
    match delta {
        ParametricDelta::Rhs { b_ub, b_eq } => {
            if b_ub.len() != lp.b_ub.len() || b_eq.len() != lp.b_eq.len() {
                return shape(format!(
                    "the right-hand-side deltas have {} and {} entries for {} and {} rows",
                    b_ub.len(),
                    b_eq.len(),
                    lp.b_ub.len(),
                    lp.b_eq.len()
                ));
            }
        }
        ParametricDelta::Objective { c } => {
            if c.len() != lp.nvars() {
                return shape(format!(
                    "the objective delta has {} entries for {} variables",
                    c.len(),
                    lp.nvars()
                ));
            }
        }
    }
    if let (Some(l), Some(u)) = (&lower, &upper) {
        if l > u {
            return shape(format!("the parameter range [{l}, {u}] is empty"));
        }
    }

    // The interval of θ where an optimum can exist.
    let rhs = matches!(delta, ParametricDelta::Rhs { .. });
    let aux = if rhs {
        feasibility_program(lp, delta, &lower, &upper)
    } else {
        dual_feasibility_program(lp, delta, &lower, &upper)
    };
    let (lo, hi) = match theta_interval(&aux)? {
        None => {
            // Never feasible (RHS case); or, for the objective case, either
            // never bounded or infeasible — decided by a solve below.
            if rhs {
                return Ok(finish(vec![ParametricRegion::plain(
                    lower,
                    upper,
                    RegionStatus::Infeasible,
                )]));
            }
            let t = pick(&lower, &upper);
            let status = match linprog(&lp_at(lp, delta, &t))? {
                LpOutcome::Infeasible(_) => RegionStatus::Infeasible,
                _ => RegionStatus::Unbounded,
            };
            return Ok(finish(vec![ParametricRegion::plain(lower, upper, status)]));
        }
        Some(iv) => iv,
    };
    let outside = if rhs {
        RegionStatus::Infeasible
    } else {
        RegionStatus::Unbounded
    };
    let mut flank_left = None;
    if lo.is_some() && lo != lower {
        flank_left = Some(ParametricRegion::plain(lower.clone(), lo.clone(), outside));
    }
    let mut flank_right = None;
    if hi.is_some() && hi != upper {
        flank_right = Some(ParametricRegion::plain(hi.clone(), upper.clone(), outside));
    }

    let start = pick(&lo, &hi);
    let at_start = lp_at(lp, delta, &start);
    let (engine, outcome, _) = revised::solve(&at_start, None)?;
    outcome
        .verify(&at_start)
        .map_err(|detail| LinprogError::Internal { detail })?;
    let mut regions = Vec::new();
    regions.extend(flank_left);
    match outcome {
        LpOutcome::Optimal(_) => {}
        LpOutcome::Unbounded(_) if rhs => {
            // Dual feasibility does not depend on b: unbounded wherever feasible.
            regions.push(ParametricRegion::plain(lo, hi, RegionStatus::Unbounded));
            regions.extend(flank_right);
            return Ok(finish(regions));
        }
        LpOutcome::Infeasible(_) if !rhs => {
            // Feasibility does not depend on c.
            return Ok(finish(vec![ParametricRegion::plain(
                lower,
                upper,
                RegionStatus::Infeasible,
            )]));
        }
        other => {
            return Err(LinprogError::Internal {
                detail: format!(
                    "parametric LP: the programme at θ = {start} inside the optimal interval \
                     was {}",
                    match other {
                        LpOutcome::Infeasible(_) => "infeasible",
                        _ => "unbounded",
                    }
                ),
            })
        }
    }

    let walker = Walker { lp, delta, rhs };
    let mut left = if lo.as_ref() == Some(&start) {
        Vec::new()
    } else {
        let mut e2 = revised::solve(&at_start, None)?.0;
        walker.walk(&mut e2, &start, -1, lo.as_ref())?
    };
    left.reverse();
    let mut engine = engine;
    let right = walker.walk(&mut engine, &start, 1, hi.as_ref())?;
    let mut optimal: Vec<ParametricRegion> = Vec::new();
    for r in left.into_iter().chain(right) {
        if let Some(prev) = optimal.last_mut() {
            if same_affine(prev, &r) {
                prev.upper = r.upper.clone();
                continue;
            }
        }
        optimal.push(r);
    }
    for r in &optimal {
        verify_region(lp, delta, r).map_err(|detail| LinprogError::Internal {
            detail: format!(
                "parametric region [{}, {}]: {detail}",
                show(&r.lower, "-oo"),
                show(&r.upper, "oo")
            ),
        })?;
    }
    regions.extend(optimal);
    regions.extend(flank_right);
    Ok(finish(regions))
}

fn show(v: &Option<Rational>, inf: &str) -> String {
    v.as_ref()
        .map_or_else(|| inf.to_string(), |r| r.to_string())
}

fn pick(lo: &Option<Rational>, hi: &Option<Rational>) -> Rational {
    match (lo, hi) {
        (Some(l), _) => l.clone(),
        (None, Some(h)) => h.clone(),
        (None, None) => Rational::new(),
    }
}

fn same_affine(a: &ParametricRegion, b: &ParametricRegion) -> bool {
    a.status == b.status
        && a.value == b.value
        && a.x == b.x
        && a.y_ub == b.y_ub
        && a.y_eq == b.y_eq
        && a.z_lower == b.z_lower
        && a.z_upper == b.z_upper
}

fn finish(regions: Vec<ParametricRegion>) -> ParametricSolution {
    let mut breakpoints: Vec<Rational> = Vec::new();
    for w in regions.windows(2) {
        if let Some(u) = &w[0].upper {
            if breakpoints.last() != Some(u) {
                breakpoints.push(u.clone());
            }
        }
    }
    ParametricSolution {
        regions,
        breakpoints,
    }
}

/// `min θ` / `max θ` of the auxiliary programme (θ is its last variable):
/// `None` when it is infeasible, else the two ends (`None` = infinite).
type Interval = (Option<Rational>, Option<Rational>);

fn theta_interval(aux: &LinearProgram) -> Result<Option<Interval>, LinprogError> {
    let end = |sense: Sense| -> Result<Option<Option<Rational>>, LinprogError> {
        let mut p = aux.clone();
        p.sense = sense;
        Ok(match linprog(&p)? {
            LpOutcome::Infeasible(_) => None,
            LpOutcome::Unbounded(_) => Some(None),
            LpOutcome::Optimal(c) => Some(Some(c.objective)),
        })
    };
    let Some(lo) = end(Sense::Minimize)? else {
        return Ok(None);
    };
    let Some(hi) = end(Sense::Maximize)? else {
        return Ok(None);
    };
    Ok(Some((lo, hi)))
}

/// `opt θ` over `{(x, θ) : A_ub x − θ δb_ub ≤ b_ub, A_eq x − θ δb_eq = b_eq,
/// x within bounds, θ in range}`.
fn feasibility_program(
    lp: &LinearProgram,
    delta: &ParametricDelta,
    lower: &Option<Rational>,
    upper: &Option<Rational>,
) -> LinearProgram {
    let ParametricDelta::Rhs { b_ub, b_eq } = delta else {
        unreachable!("feasibility_program is for the right-hand-side case")
    };
    let n = lp.nvars();
    let mut c = vec![Rational::new(); n + 1];
    c[n] = Rational::from(1);
    let mut aux = LinearProgram::new(c);
    let ext = |row: &[Rational], d: &Rational| -> Vec<Rational> {
        let mut r = row.to_vec();
        r.push(-d.clone());
        r
    };
    aux.a_ub = lp.a_ub.iter().zip(b_ub).map(|(r, d)| ext(r, d)).collect();
    aux.b_ub = lp.b_ub.clone();
    aux.a_eq = lp.a_eq.iter().zip(b_eq).map(|(r, d)| ext(r, d)).collect();
    aux.b_eq = lp.b_eq.clone();
    aux.bounds = lp.bounds.clone();
    aux.bounds.push(VarBound::new(lower.clone(), upper.clone()));
    aux
}

/// `opt θ` over the dual certificates of `lp` at `θ`: `c + θ δc =
/// A_ubᵀ y_ub + A_eqᵀ y_eq + z_lower + z_upper` with the signs of the sense
/// (multipliers of infinite bounds fixed at 0). Non-empty at `θ` exactly
/// when `lp(θ)`, if feasible, is bounded.
fn dual_feasibility_program(
    lp: &LinearProgram,
    delta: &ParametricDelta,
    lower: &Option<Rational>,
    upper: &Option<Rational>,
) -> LinearProgram {
    let ParametricDelta::Objective { c: dc } = delta else {
        unreachable!("dual_feasibility_program is for the objective case")
    };
    let n = lp.nvars();
    let m_ub = lp.a_ub.len();
    let m_eq = lp.a_eq.len();
    let nv = m_ub + m_eq + 2 * n + 1;
    let mut c = vec![Rational::new(); nv];
    c[nv - 1] = Rational::from(1);
    let mut aux = LinearProgram::new(c);
    let max = lp.sense == Sense::Maximize;
    let nonpos = || {
        if max {
            VarBound::nonnegative()
        } else {
            VarBound::new(None, Some(Rational::new()))
        }
    };
    let nonneg = || {
        if max {
            VarBound::new(None, Some(Rational::new()))
        } else {
            VarBound::nonnegative()
        }
    };
    let zero = || VarBound::new(Some(Rational::new()), Some(Rational::new()));
    let mut bounds = Vec::with_capacity(nv);
    bounds.extend((0..m_ub).map(|_| nonpos()));
    bounds.extend((0..m_eq).map(|_| VarBound::free()));
    for bd in &lp.bounds {
        bounds.push(if bd.lower.is_some() { nonneg() } else { zero() });
    }
    for bd in &lp.bounds {
        bounds.push(if bd.upper.is_some() { nonpos() } else { zero() });
    }
    bounds.push(VarBound::new(lower.clone(), upper.clone()));
    aux.bounds = bounds;
    for j in 0..n {
        let mut row = vec![Rational::new(); nv];
        for (i, r) in lp.a_ub.iter().enumerate() {
            row[i] = r[j].clone();
        }
        for (i, r) in lp.a_eq.iter().enumerate() {
            row[m_ub + i] = r[j].clone();
        }
        row[m_ub + m_eq + j] = Rational::from(1);
        row[m_ub + m_eq + n + j] = Rational::from(1);
        row[nv - 1] = -dc[j].clone();
        aux.a_eq.push(row);
        aux.b_eq.push(lp.c[j].clone());
    }
    aux
}

struct Walker<'a> {
    lp: &'a LinearProgram,
    delta: &'a ParametricDelta,
    rhs: bool,
}

impl Walker<'_> {
    /// Walk from `start` (where `e` holds an optimal basis of the programme
    /// at `start`) in direction `s = ±1` up to `end` (`None` = infinite),
    /// returning the regions met, in walking order.
    fn walk(
        &self,
        e: &mut Engine,
        start: &Rational,
        s: i32,
        end: Option<&Rational>,
    ) -> Result<Vec<ParametricRegion>, LinprogError> {
        let lp = self.lp;
        let sgn = Rational::from(s);
        // Standard-form direction of b' or c' per unit of τ = s·(θ − θ_cur).
        let dir: Vec<Rational> = match self.delta {
            ParametricDelta::Rhs { b_ub, b_eq } => {
                let mut v: Vec<Rational> = b_ub.iter().chain(b_eq).cloned().collect();
                v.resize(e.m, Rational::new());
                v.into_iter().map(|x| x * &sgn).collect()
            }
            ParametricDelta::Objective { c } => {
                let mut l2 = lp.clone();
                l2.c = c.clone();
                let mut cost = super::simplex::standard_form(&l2).cost;
                cost.resize(e.ncols + e.m, Rational::new());
                cost.into_iter().map(|x| x * &sgn).collect()
            }
        };
        let mut theta = start.clone();
        let mut out = Vec::new();
        if end == Some(start) {
            // A single point: the optimal interval is degenerate.
            let beta = if self.rhs {
                e.ftran_vec(&dir)
            } else {
                Vec::new()
            };
            out.push(self.region(e, start, Some(start), s, &beta, &dir)?);
            return Ok(out);
        }
        // Make the starting basis optimal for θ + s·ε, so every region has
        // positive length.
        if !self.repivot(e, &dir)? {
            return Err(LinprogError::Internal {
                detail: format!("parametric LP: no optimum just beyond θ = {start}"),
            });
        }
        let cap = 100_000usize.saturating_add(e.m.saturating_mul(e.ncols + 1));
        for _ in 0..cap {
            budget::check()?;
            // Largest step t ≥ 0 keeping the basis optimal.
            let (t_max, beta_or_d1) = if self.rhs {
                let beta = e.ftran_vec(&dir);
                for (r, &c) in e.basis.iter().enumerate() {
                    if c >= e.ncols && beta[r] != 0 {
                        return Err(LinprogError::Internal {
                            detail: "parametric LP: a redundant row moves with θ".into(),
                        });
                    }
                }
                let mut t: Option<Rational> = None;
                for (xi, bi) in e.xb.iter().zip(&beta) {
                    if *bi < 0 {
                        let ti = Rational::from(xi / bi).abs();
                        if t.as_ref().map_or(true, |cur| ti < *cur) {
                            t = Some(ti);
                        }
                    }
                }
                (t, beta)
            } else {
                let (d0, d1) = self.reduced(e, &dir);
                let mut t: Option<Rational> = None;
                for j in 0..e.ncols {
                    if e.is_basic(j) || d1[j] >= 0 {
                        continue;
                    }
                    let tj = Rational::from(&d0[j] / &d1[j]).abs();
                    if t.as_ref().map_or(true, |cur| tj < *cur) {
                        t = Some(tj);
                    }
                }
                (t, d1)
            };
            let t_end = end.map(|e| Rational::from(e - &theta).abs());
            let (step, last) = match (&t_max, &t_end) {
                (None, None) => (None, true),
                (None, Some(te)) => (Some(te.clone()), true),
                (Some(tm), None) => (Some(tm.clone()), false),
                (Some(tm), Some(te)) => {
                    if te <= tm {
                        (Some(te.clone()), true)
                    } else {
                        (Some(tm.clone()), false)
                    }
                }
            };
            let next = step
                .as_ref()
                .map(|t| Rational::from(&theta + &Rational::from(t * &sgn)));
            out.push(self.region(e, &theta, next.as_ref(), s, &beta_or_d1, &dir)?);
            if last {
                return Ok(out);
            }
            let t = step.expect("a finite step when not last");
            // Move to the breakpoint.
            if self.rhs {
                for (r, bi) in beta_or_d1.iter().enumerate() {
                    if *bi != 0 {
                        e.xb[r] += Rational::from(bi * &t);
                    }
                }
                for (bj, dj) in e.b.iter_mut().zip(&dir) {
                    if *dj != 0 {
                        *bj += Rational::from(dj * &t);
                    }
                }
            } else {
                for (cj, dj) in e.cost.iter_mut().zip(&dir) {
                    if *dj != 0 {
                        *cj += Rational::from(dj * &t);
                    }
                }
            }
            theta = next.expect("finite step");
            // Re-optimise for θ + ε by Bland pivots on the perturbed problem.
            if !self.repivot(e, &dir)? {
                // Infeasible (RHS) or unbounded (objective) beyond θ: this must
                // be the end of the optimal interval.
                return Err(LinprogError::Internal {
                    detail: format!(
                        "parametric LP: the walk left the optimal interval early at θ = {theta}"
                    ),
                });
            }
        }
        Err(LinprogError::Internal {
            detail: "parametric LP: region cap reached".into(),
        })
    }

    /// Reduced costs `d₀` (current cost) and `d₁` (cost direction) of every
    /// real column (0 for basic ones).
    fn reduced(&self, e: &Engine, dir: &[Rational]) -> (Vec<Rational>, Vec<Rational>) {
        let y0 = e.duals(&e.cost);
        let y1 = e.duals(dir);
        let mut d0 = vec![Rational::new(); e.ncols];
        let mut d1 = vec![Rational::new(); e.ncols];
        for j in 0..e.ncols {
            if e.is_basic(j) {
                continue;
            }
            d0[j] = Rational::from(&e.cost[j] - &e.dot_col(&y0, j));
            d1[j] = Rational::from(&dir[j] - &e.dot_col(&y1, j));
        }
        (d0, d1)
    }

    /// Pivot until the basis is optimal for θ + ε. Returns `false` when the
    /// perturbed programme is infeasible (RHS) or unbounded (objective).
    fn repivot(&self, e: &mut Engine, dir: &[Rational]) -> Result<bool, LinprogError> {
        let cap = 100_000usize.saturating_add(e.m.saturating_mul(e.ncols + 1));
        for _ in 0..cap {
            budget::check()?;
            if self.rhs {
                let beta = e.ftran_vec(dir);
                // Leaving: lexicographically negative rows, smallest basic index.
                let mut leave: Option<usize> = None;
                for r in 0..e.m {
                    if e.xb[r] == 0
                        && beta[r] < 0
                        && leave.map_or(true, |k| e.basis[r] < e.basis[k])
                    {
                        leave = Some(r);
                    }
                }
                let Some(r) = leave else {
                    return Ok(true);
                };
                let rho = e.binv_row(r);
                let y = e.duals(&e.cost);
                let mut enter: Option<(usize, Rational)> = None;
                for j in 0..e.ncols {
                    if e.is_basic(j) {
                        continue;
                    }
                    let alpha = e.dot_col(&rho, j);
                    if alpha >= 0 {
                        continue;
                    }
                    let d = Rational::from(&e.cost[j] - &e.dot_col(&y, j));
                    let ratio = Rational::from(&d / &alpha).abs();
                    if enter.as_ref().map_or(true, |(_, b)| ratio < *b) {
                        enter = Some((j, ratio));
                    }
                }
                let Some((q, _)) = enter else {
                    return Ok(false);
                };
                let w = e.ftran_col(q);
                e.pivot(r, q, &w);
            } else {
                let (d0, d1) = self.reduced(e, dir);
                let Some(q) = (0..e.ncols).find(|&j| !e.is_basic(j) && d0[j] == 0 && d1[j] < 0)
                else {
                    return Ok(true);
                };
                let w = e.ftran_col(q);
                let mut leave: Option<(usize, Rational)> = None;
                for (i, wi) in w.iter().enumerate() {
                    if *wi <= 0 {
                        continue;
                    }
                    let ratio = Rational::from(&e.xb[i] / wi);
                    let better = match &leave {
                        None => true,
                        Some((k, rk)) => ratio < *rk || (ratio == *rk && e.basis[i] < e.basis[*k]),
                    };
                    if better {
                        leave = Some((i, ratio));
                    }
                }
                let Some((r, _)) = leave else {
                    return Ok(false);
                };
                e.pivot(r, q, &w);
            }
        }
        Err(LinprogError::Internal {
            detail: "parametric LP: pivot cap reached at a breakpoint".into(),
        })
    }

    /// The region `[θ, next]` (in walking direction `s`) of the current basis,
    /// with its affine data. `beta_or_d1` is `B⁻¹·dir` (RHS case).
    fn region(
        &self,
        e: &Engine,
        theta: &Rational,
        next: Option<&Rational>,
        s: i32,
        beta: &[Rational],
        dir: &[Rational],
    ) -> Result<ParametricRegion, LinprogError> {
        let lp = self.lp;
        let sgn = Rational::from(s);
        let lp_t = lp_at(lp, self.delta, theta);
        // Certificate at θ.
        let xs = e.point();
        let y = e.duals(&e.cost);
        let c_t = optimal_from(&lp_t, &e.sf, &xs, &y);
        // Certificate at θ + s (formally — the affine extension of this basis).
        let theta2 = Rational::from(theta + &sgn);
        let lp_t2 = lp_at(lp, self.delta, &theta2);
        let c_t2 = if self.rhs {
            let mut xs2 = xs.clone();
            for (r, &col) in e.basis.iter().enumerate() {
                if col < e.ncols {
                    xs2[col] += &beta[r];
                }
            }
            optimal_from(&lp_t2, &e.sf, &xs2, &y)
        } else {
            let cost2: Vec<Rational> = e
                .cost
                .iter()
                .zip(dir)
                .map(|(a, b)| Rational::from(a + b))
                .collect();
            let y2 = e.duals(&cost2);
            optimal_from(&lp_t2, &e.sf, &xs, &y2)
        };
        // Affine in θ: slope = (cert(θ+s) − cert(θ))·s, intercept = cert(θ) − θ·slope.
        let aff = |a: &[Rational], b: &[Rational]| -> (Vec<Rational>, Vec<Rational>) {
            let slope: Vec<Rational> = a
                .iter()
                .zip(b)
                .map(|(u, v)| Rational::from(v - u) * &sgn)
                .collect();
            let icpt: Vec<Rational> = a
                .iter()
                .zip(&slope)
                .map(|(u, k)| Rational::from(u - &Rational::from(k * theta)))
                .collect();
            (icpt, slope)
        };
        let x = aff(&c_t.x, &c_t2.x);
        let v = aff(
            std::slice::from_ref(&c_t.objective),
            std::slice::from_ref(&c_t2.objective),
        );
        let (lower, upper) = if s > 0 {
            (Some(theta.clone()), next.cloned())
        } else {
            (next.cloned(), Some(theta.clone()))
        };
        Ok(ParametricRegion {
            lower,
            upper,
            status: RegionStatus::Optimal,
            value: Some((v.0[0].clone(), v.1[0].clone())),
            x: Some(x),
            y_ub: Some(aff(&c_t.y_ub, &c_t2.y_ub)),
            y_eq: Some(aff(&c_t.y_eq, &c_t2.y_eq)),
            z_lower: Some(aff(&c_t.z_lower, &c_t2.z_lower)),
            z_upper: Some(aff(&c_t.z_upper, &c_t2.z_upper)),
            basis: Some(e.basis_labels(&lp_t)),
        })
    }
}

/// Exact proof that the region's affine certificate is valid for every `θ`
/// in the region.
fn verify_region(
    lp: &LinearProgram,
    delta: &ParametricDelta,
    r: &ParametricRegion,
) -> Result<(), String> {
    let at = |t: &Rational| -> Result<(), String> {
        let cert = r
            .certificate_at(t)
            .ok_or_else(|| "region carries no certificate".to_string())?;
        cert.verify(&lp_at(lp, delta, t))
            .map_err(|e| format!("at θ = {t}: {e}"))
    };
    if let Some(l) = &r.lower {
        at(l)?;
    }
    if let Some(u) = &r.upper {
        at(u)?;
    }
    if r.lower.is_none() && r.upper.is_none() {
        at(&Rational::new())?;
    }
    for (s, open) in [(1i32, r.upper.is_none()), (-1, r.lower.is_none())] {
        if open {
            slope_certificate(lp, delta, r, s)?;
        }
    }
    Ok(())
}

/// The slope of the certificate as `θ → s·∞`, checked as an optimality
/// certificate of the derived programme (see the module docs).
fn slope_certificate(
    lp: &LinearProgram,
    delta: &ParametricDelta,
    r: &ParametricRegion,
    s: i32,
) -> Result<(), String> {
    let sgn = Rational::from(s);
    let neg =
        |v: &[Rational]| -> Vec<Rational> { v.iter().map(|a| Rational::from(a * &sgn)).collect() };
    let pick = |p: &Option<(Vec<Rational>, Vec<Rational>)>| p.clone().ok_or("missing data");
    let (x0, x1) = pick(&r.x)?;
    let (yu0, yu1) = pick(&r.y_ub)?;
    let (ye0, ye1) = pick(&r.y_eq)?;
    let (zl0, zl1) = pick(&r.z_lower)?;
    let (zu0, zu1) = pick(&r.z_upper)?;
    let is_zero = |v: &[Rational]| v.iter().all(|a| *a == 0);
    match delta {
        ParametricDelta::Rhs { b_ub, b_eq } => {
            if !(is_zero(&yu1) && is_zero(&ye1) && is_zero(&zl1) && is_zero(&zu1)) {
                return Err("duals of an RHS region must not move with θ".into());
            }
            let mut d = lp.clone();
            d.b_ub = neg(b_ub);
            d.b_eq = neg(b_eq);
            for bd in d.bounds.iter_mut() {
                bd.lower = bd.lower.as_ref().map(|_| Rational::new());
                bd.upper = bd.upper.as_ref().map(|_| Rational::new());
            }
            let x = neg(&x1);
            let cert = OptimalCertificate {
                objective: dot(&lp.c, &x),
                x,
                y_ub: yu0,
                y_eq: ye0,
                z_lower: zl0,
                z_upper: zu0,
            };
            cert.verify(&d)
                .map_err(|e| format!("slope towards {s}·∞: {e}"))
        }
        ParametricDelta::Objective { c } => {
            if !is_zero(&x1) {
                return Err("the point of an objective region must not move with θ".into());
            }
            let mut d = lp.clone();
            d.c = neg(c);
            let cert = OptimalCertificate {
                objective: dot(&d.c, &x0),
                x: x0,
                y_ub: neg(&yu1),
                y_eq: neg(&ye1),
                z_lower: neg(&zl1),
                z_upper: neg(&zu1),
            };
            cert.verify(&d)
                .map_err(|e| format!("slope towards {s}·∞: {e}"))
        }
    }
}
