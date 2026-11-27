//! Eigenfunction-series solutions of linear problems on separable bounded
//! domains: see the [module docs](super).

use std::collections::HashMap;

use crate::diff::undefined::{replace_function, view};
use crate::kernel::expr::PredicateKind;
use crate::kernel::subs::{mentions_var, subs};
use crate::kernel::{Domain, ExprData, ExprId, ExprPool};
use crate::simplify::zero::ZeroStatus;

use super::super::error::PdeError;
use super::super::jet::{decode_unknown, push_children};
use super::super::problem::{ConditionKind, PdeCondition, PdeDomain, PdeProblem};
use super::super::sturm_liouville::{
    d, simp, sturm_liouville_with, subs1, EigenFamily, Eigensystem, EndCondition, LowerSpectrum,
    SlBoundary, SlOptions, SturmLiouvilleProblem,
};
use super::super::verify::VerificationStatus;
use super::conditions::{parse_condition, FaceCondition, Parsed};
use super::coords::{from_cartesian, isotropic_to_curvilinear, to_cartesian, CoordinateSystem};
use super::modes::{
    div, is_zero_expr, mode_ode, neg, solve_mode, sqrt_pos, sub, ModeOde, SData, SEnd,
};
use super::project::{exceptional_indices, has_integral, iterated};
use super::separate::{names_in, separate_with, SeparateOptions, Separation, SeparationKind};

// ---------------------------------------------------------------------------
// Public result types
// ---------------------------------------------------------------------------

/// One exact (or attempted) check of the series solution.
#[derive(Clone, Debug, PartialEq, Eq)]
#[non_exhaustive]
pub struct SeriesCheck {
    /// `"separation"` (every mode satisfies the PDE: `L[TΦ] = Φ·M[T]`
    /// identically in the index), `"eigen_bc"` (the eigenfunctions satisfy the
    /// homogeneous boundary conditions — the Sturm–Liouville solver's check),
    /// `"amplitude"` (the mode amplitude solves its ODE), `"amplitude_data"`
    /// (it meets the initial values / data of the mode), `"coefficient"`
    /// (the projection integral, by an exactly checked antiderivative),
    /// `"lifting"` (the lifting / steady state meets the boundary conditions
    /// and the steady equation), `"kernel"` (an integral representation's
    /// kernel).
    pub what: &'static str,
    /// The mode group (index into [`SeriesSolution::modes`]), or `usize::MAX`
    /// for a check on the whole solution.
    pub group: usize,
    /// The verdict.
    pub status: ZeroStatus,
    /// How (`"zero_test"`, `"sturm_liouville"`, `"ftc"`, `"members"`, …).
    pub method: &'static str,
    /// What was checked.
    pub detail: String,
}

/// The numeric check that the coefficients are the projections of the data
/// (plan §7): the closed-form coefficients of the first members, evaluated
/// in ball arithmetic, against quadratures of `∫ W·f·Φ / ‖Φ‖²`.  It may only
/// **refute**; it never upgrades a status, and it never claims that a
/// truncation equals the data pointwise.
#[derive(Clone, Debug, PartialEq)]
#[non_exhaustive]
pub struct ProjectionCheck {
    /// `"ic"` (order 0), `"ic1"`, `"boundary_data"`, `"source"`.
    pub data: &'static str,
    /// `NumericallyChecked`, or `Unverified` when nothing could be evaluated.
    pub status: VerificationStatus,
    /// The number of coefficients compared.
    pub terms: usize,
    /// The largest relative discrepancy.
    pub max_error: f64,
    /// The parameter values used and the compared numbers.
    pub evidence: Vec<String>,
}

/// One group of modes: the product eigenfunctions of one combination of
/// eigen-families (or isolated modes) of the separated directions.
#[derive(Clone, Debug, PartialEq, Eq)]
#[non_exhaustive]
pub struct ModeGroup {
    /// `(index, first value)` per summation index (empty for an isolated
    /// mode).
    pub indices: Vec<(ExprId, ExprId)>,
    /// The last value of each index (`∞`, or finite when the range was split
    /// around exceptional index values).
    pub upper: Vec<ExprId>,
    /// `(separation constant, its value)` for each separated direction.
    pub eigenvalues: Vec<(ExprId, ExprId)>,
    /// The product eigenfunctions (two or more when a direction is periodic:
    /// cosine and sine variants).
    pub eigenfunctions: Vec<ExprId>,
    /// The orthogonality weight `W` (product of the directions' weights).
    pub weight: ExprId,
    /// `∫ W Φ²` per eigenfunction.
    pub norms_squared: Vec<ExprId>,
    /// The projection coefficients per eigenfunction: `(data, value)` with
    /// `data` one of `"ic"`, `"ic1"`, `"left"`, `"right"`, `"source"`.
    pub coefficients: Vec<Vec<(String, ExprId)>>,
    /// The mode amplitudes `T(s)` (or constants) per eigenfunction.
    pub amplitudes: Vec<ExprId>,
    /// `Σ_v amplitude_v·Φ_v`: the summand.
    pub term: ExprId,
    /// The nested `Sum` over the indices.
    pub value: ExprId,
    /// Whether every coefficient closed (no `Integral` placeholder).
    pub closed: bool,
    /// The exact checks' verdict for this group.
    pub status: VerificationStatus,
}

/// The result of [`separation_solve`].
#[derive(Clone, Debug, PartialEq)]
#[non_exhaustive]
pub struct SeriesSolution {
    /// `u`: lifting / steady state + the series (or a finite sum).
    pub value: ExprId,
    /// `"series"` (contains a `Sum`), `"closed_form"` (finitely many modes).
    pub kind: &'static str,
    /// `"separation"`, joined with `"+lifting"`, `"+steady_state"`,
    /// `"+duhamel"`, `"+superposition"` as used.
    pub method: String,
    /// The lifting or steady state (`None` when the boundary data was
    /// homogeneous).
    pub steady_state: Option<ExprId>,
    /// The mode groups.
    pub modes: Vec<ModeGroup>,
    /// The separation (of the last, or only, sub-problem).
    pub separation: Option<Separation>,
    /// `ExactlyVerified` when every mode, amplitude, eigen-condition,
    /// coefficient and lifting check is a proof.  The initial / boundary data
    /// are met in the sense of the eigenfunction expansion (see
    /// [`Self::convergence`]), never claimed pointwise for a truncation.
    pub status: VerificationStatus,
    /// The individual checks.
    pub checks: Vec<SeriesCheck>,
    /// The numeric projection checks.
    pub projection: Vec<ProjectionCheck>,
    /// `"uniform"` when the data satisfy the boundary conditions (the
    /// expansion of the data converges uniformly), `"l2"` otherwise (mean
    /// square; Gibbs oscillations near the incompatible boundary — see
    /// [`Self::notes`]).
    pub convergence: &'static str,
    /// Predicates under which the answer is correct.
    pub side_conditions: Vec<ExprId>,
    /// Prose: convergence, Gibbs behaviour, arbitrary constants.
    pub notes: Vec<String>,
    /// Other representations: `("poisson_integral", …)` for the disk.
    pub alternatives: Vec<(String, ExprId)>,
    /// When the problem was posed on a Cartesian [`PdeDomain::Disk`] /
    /// [`PdeDomain::Ball`]: the coordinate system and the curvilinear
    /// coordinates the series is written in (the value is the series with
    /// `r = |x|`, `θ = atan2(y, x)`, …).
    pub coordinates: Option<(CoordinateSystem, Vec<ExprId>)>,
    /// The value in those curvilinear coordinates.
    pub curvilinear_value: Option<ExprId>,
    /// Arbitrary constants of the answer (Neumann problems).
    pub constants: Vec<ExprId>,
}

/// Options for [`separation_solve_with`].
#[derive(Clone, Debug)]
#[non_exhaustive]
pub struct SeparationOptions {
    /// Members per index compared in the numeric projection check (0 to
    /// skip it).
    pub check_members: usize,
    /// Prefer a steady state over a polynomial lifting when the boundary
    /// data and source do not depend on time.
    pub steady_state: bool,
    /// Index names, used in separation order (`["m", "n"]` …); empty for the
    /// default (`n`; `m, n`; `l, m, n`).
    pub index_names: Vec<String>,
}

impl Default for SeparationOptions {
    fn default() -> Self {
        SeparationOptions {
            check_members: 4,
            steady_state: true,
            index_names: Vec::new(),
        }
    }
}

// ---------------------------------------------------------------------------
// Internal problem description
// ---------------------------------------------------------------------------

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
enum CoordKind {
    Finite,
    Periodic,
    Time,
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
enum Role {
    Plain,
    Radius,
    PolarAngle,
    SphTheta,
    Azimuth,
}

#[derive(Clone, Copy, Debug)]
struct Coord {
    var: ExprId,
    lo: ExprId,
    hi: ExprId,
    kind: CoordKind,
    sing_lo: bool,
    sing_hi: bool,
    role: Role,
}

#[derive(Clone, Debug)]
struct Prob {
    u: ExprId,
    name: String,
    vars: Vec<ExprId>,
    lhom: ExprId,
    source: ExprId,
    coords: Vec<Coord>,
    system: Option<CoordinateSystem>,
    faces: Vec<FaceCondition>,
    ics: Vec<FaceCondition>,
}

struct Ctx<'a> {
    opts: &'a SeparationOptions,
    pool: &'a ExprPool,
    taken: Vec<String>,
}

/// What one sub-problem produced.
#[derive(Default)]
struct Part {
    value: Vec<ExprId>,
    steady: Vec<ExprId>,
    groups: Vec<ModeGroup>,
    checks: Vec<SeriesCheck>,
    projection: Vec<ProjectionCheck>,
    side: Vec<ExprId>,
    notes: Vec<String>,
    methods: Vec<&'static str>,
    separation: Option<Separation>,
    uniform: bool,
    constants: Vec<ExprId>,
    alternatives: Vec<(String, ExprId)>,
}

impl Part {
    fn merge(&mut self, mut o: Part) {
        let off = self.groups.len();
        for c in &mut o.checks {
            if c.group != usize::MAX {
                c.group += off;
            }
        }
        self.value.extend(o.value);
        self.steady.extend(o.steady);
        self.groups.extend(o.groups);
        self.checks.extend(o.checks);
        self.projection.extend(o.projection);
        self.side.extend(o.side);
        self.notes.extend(o.notes);
        for m in o.methods {
            if !self.methods.contains(&m) {
                self.methods.push(m);
            }
        }
        if o.separation.is_some() {
            self.separation = o.separation;
        }
        self.uniform &= o.uniform;
        self.constants.extend(o.constants);
        self.alternatives.extend(o.alternatives);
    }
}

fn sym_named(name: &str, dom: Domain, ctx: &mut Ctx<'_>) -> ExprId {
    let n = super::separate::fresh_name(name, &mut ctx.taken);
    ctx.pool.symbol(n, dom)
}

fn pi(pool: &ExprPool) -> ExprId {
    pool.symbol("pi", Domain::Real)
}

fn eq_expr(a: ExprId, b: ExprId, pool: &ExprPool) -> bool {
    a == b || simp(sub(a, b, pool), pool) == pool.integer(0)
}

// ---------------------------------------------------------------------------
// Building the internal problem
// ---------------------------------------------------------------------------

/// `eq = lhom + source` with `lhom` linear homogeneous in the unknown.
fn split_linear(eq: ExprId, name: &str, pool: &ExprPool) -> Result<(ExprId, ExprId), PdeError> {
    let mut nodes = Vec::new();
    let mut stack = vec![eq];
    let mut seen = std::collections::HashSet::new();
    while let Some(n) = stack.pop() {
        if !seen.insert(n) {
            continue;
        }
        if let Some(v) = view(n, pool) {
            if v.name == name {
                nodes.push(n);
                continue;
            }
        }
        push_children(n, pool, &mut stack);
    }
    let mut map = HashMap::new();
    let mut syms = Vec::new();
    for (k, &n) in nodes.iter().enumerate() {
        let s = pool.symbol(format!("__lin_{k}"), Domain::Real);
        map.insert(n, s);
        syms.push(s);
    }
    let e = super::separate::replace_nodes(eq, &map, pool);
    for &s in &syms {
        let c = d(e, s, pool)?;
        for &z in &syms {
            if !is_zero_expr(d(c, z, pool)?, pool) {
                return Err(PdeError::SeparationUnsupported(format!(
                    "{} is not linear in the unknown",
                    pool.display(eq)
                )));
            }
        }
    }
    let zero: HashMap<ExprId, ExprId> = syms.iter().map(|&s| (s, pool.integer(0))).collect();
    let source = simp(subs(e, &zero, pool), pool);
    let lhom = simp(sub(eq, source, pool), pool);
    Ok((lhom, source))
}

fn two_pi_wide(lo: ExprId, hi: ExprId, pool: &ExprPool) -> bool {
    eq_expr(
        sub(hi, lo, pool),
        pool.mul(vec![pool.integer(2), pi(pool)]),
        pool,
    )
}

fn coords_of(
    dom: &PdeDomain,
    out: &mut HashMap<ExprId, Coord>,
    system: &mut Option<CoordinateSystem>,
    pool: &ExprPool,
) -> Result<(), PdeError> {
    match dom {
        PdeDomain::Interval { var, lo, hi } => {
            out.insert(
                *var,
                Coord {
                    var: *var,
                    lo: *lo,
                    hi: *hi,
                    kind: CoordKind::Finite,
                    sing_lo: false,
                    sing_hi: false,
                    role: Role::Plain,
                },
            );
        }
        PdeDomain::Box { vars, bounds } => {
            for (&v, &(lo, hi)) in vars.iter().zip(bounds) {
                out.insert(
                    v,
                    Coord {
                        var: v,
                        lo,
                        hi,
                        kind: CoordKind::Finite,
                        sing_lo: false,
                        sing_hi: false,
                        role: Role::Plain,
                    },
                );
            }
        }
        PdeDomain::HalfLine {
            var,
            end,
            upper: false,
        } => {
            out.insert(
                *var,
                Coord {
                    var: *var,
                    lo: *end,
                    hi: pool.pos_infinity(),
                    kind: CoordKind::Time,
                    sing_lo: false,
                    sing_hi: false,
                    role: Role::Plain,
                },
            );
        }
        PdeDomain::Product(ps) => {
            for p in ps {
                coords_of(p, out, system, pool)?;
            }
        }
        PdeDomain::Curvilinear {
            system: sys,
            coords,
            bounds,
        } => {
            *system = Some(*sys);
            for (i, (c, &(lo, hi))) in coords.iter().zip(bounds).enumerate() {
                let Some(v) = c else { continue };
                let zero = pool.integer(0);
                let (kind, role, sl, sh) = match (sys, i) {
                    (_, 0) => (
                        CoordKind::Finite,
                        Role::Radius,
                        eq_expr(lo, zero, pool),
                        false,
                    ),
                    (CoordinateSystem::Spherical, 1) => (
                        CoordKind::Finite,
                        Role::SphTheta,
                        eq_expr(lo, zero, pool),
                        eq_expr(hi, pi(pool), pool),
                    ),
                    (CoordinateSystem::Spherical, 2) => (
                        if two_pi_wide(lo, hi, pool) {
                            CoordKind::Periodic
                        } else {
                            CoordKind::Finite
                        },
                        Role::Azimuth,
                        false,
                        false,
                    ),
                    (_, 1) => (
                        if two_pi_wide(lo, hi, pool) {
                            CoordKind::Periodic
                        } else {
                            CoordKind::Finite
                        },
                        Role::PolarAngle,
                        false,
                        false,
                    ),
                    _ => (CoordKind::Finite, Role::Plain, false, false),
                };
                out.insert(
                    *v,
                    Coord {
                        var: *v,
                        lo,
                        hi,
                        kind,
                        sing_lo: sl,
                        sing_hi: sh,
                        role,
                    },
                );
            }
        }
        other => {
            return Err(PdeError::SeparationUnsupported(format!(
                "separation of variables needs bounded coordinates and at most one half-line \
                 (time); a {} domain is not supported (use transform_solve on lines and \
                 half-lines)",
                other.kind()
            )))
        }
    }
    Ok(())
}

fn build_prob(
    eq: ExprId,
    u: ExprId,
    dom: &PdeDomain,
    conds: &[PdeCondition],
    pool: &ExprPool,
) -> Result<Prob, PdeError> {
    let uv = decode_unknown(u, pool)?;
    let (lhom, source) = split_linear(eq, &uv.name, pool)?;
    if lhom == pool.integer(0) {
        return Err(PdeError::NotADifferentialEquation(
            pool.display(eq).to_string(),
        ));
    }
    let mut map = HashMap::new();
    let mut system = None;
    coords_of(dom, &mut map, &mut system, pool)?;
    let mut coords = Vec::new();
    for &v in &uv.args {
        let c = map.get(&v).copied().ok_or_else(|| {
            PdeError::SeparationUnsupported(format!(
                "no range is declared for {}: pose the problem .on(a domain covering every \
                 variable)",
                pool.display(v)
            ))
        })?;
        coords.push(c);
    }
    if coords.iter().filter(|c| c.kind == CoordKind::Time).count() > 1 {
        return Err(PdeError::SeparationUnsupported(
            "at most one half-line (the evolution variable) is supported".into(),
        ));
    }
    let mut faces = Vec::new();
    let mut ics = Vec::new();
    let mut periodic_marks: HashMap<usize, Vec<u32>> = HashMap::new();
    for c in conds {
        match parse_condition(c, &uv.name, &uv.args, pool)? {
            Parsed::Face(f) => {
                let co = coords[f.slot];
                if co.kind == CoordKind::Time {
                    if !eq_expr(f.at, co.lo, pool) {
                        return Err(PdeError::SeparationCondition(format!(
                            "an initial condition must be at {} = {}",
                            pool.display(co.var),
                            pool.display(co.lo)
                        )));
                    }
                    if f.coeffs.iter().filter(|&&c| c != pool.integer(0)).count() != 1 {
                        return Err(PdeError::SeparationCondition(
                            "an initial condition gives one derivative of the unknown".into(),
                        ));
                    }
                    ics.push(f);
                } else {
                    if co.kind == CoordKind::Periodic {
                        return Err(PdeError::SeparationCondition(format!(
                            "{} is periodic: no boundary condition is needed in it",
                            pool.display(co.var)
                        )));
                    }
                    let at_lo = eq_expr(f.at, co.lo, pool);
                    let at_hi = eq_expr(f.at, co.hi, pool);
                    if !at_lo && !at_hi {
                        return Err(PdeError::SeparationCondition(format!(
                            "{} = {} is not an end of the range of {}",
                            pool.display(co.var),
                            pool.display(f.at),
                            pool.display(co.var)
                        )));
                    }
                    if f.coeffs.len() > 2 {
                        return Err(PdeError::SeparationCondition(
                            "boundary conditions of order above one are not supported".into(),
                        ));
                    }
                    if ((at_lo && co.sing_lo) || (at_hi && co.sing_hi))
                        && f.coeffs.len() == 2
                        && is_zero_expr(f.coeffs[0], pool)
                        && is_zero_expr(f.data, pool)
                    {
                        // u_r = 0 at the centre / poles: the symmetry condition
                        // the bounded solutions already satisfy.
                        continue;
                    }
                    if (at_lo && co.sing_lo) || (at_hi && co.sing_hi) {
                        return Err(PdeError::SeparationCondition(format!(
                            "{} = {} is a singular end of the coordinates: boundedness is \
                             implied there, give no condition",
                            pool.display(co.var),
                            pool.display(f.at)
                        )));
                    }
                    faces.push(f);
                }
            }
            Parsed::Periodic(pc) => {
                periodic_marks.entry(pc.slot).or_default().push(pc.order);
            }
        }
    }
    for (slot, orders) in periodic_marks {
        let co = &mut coords[slot];
        if co.kind == CoordKind::Periodic {
            continue;
        }
        if co.kind == CoordKind::Finite && orders.contains(&0) && orders.contains(&1) {
            co.kind = CoordKind::Periodic;
        } else {
            return Err(PdeError::SeparationCondition(format!(
                "periodicity in {} needs both u(a) = u(b) and u'(a) = u'(b)",
                pool.display(co.var)
            )));
        }
    }
    Ok(Prob {
        u,
        name: uv.name,
        vars: uv.args,
        lhom,
        source,
        coords,
        system,
        faces,
        ics,
    })
}

// ---------------------------------------------------------------------------
// Entry points
// ---------------------------------------------------------------------------

/// Solve `problem` by separation of variables with default options.
///
/// # Errors
///
/// See [`separation_solve_with`].
pub fn separation_solve(problem: &PdeProblem, pool: &ExprPool) -> Result<SeriesSolution, PdeError> {
    separation_solve_with(problem, &SeparationOptions::default(), pool)
}

/// Solve `problem` by separation of variables and eigenfunction expansion:
/// see the [module docs](super).
///
/// # Errors
///
/// `E-PDE-200 … E-PDE-209` (see [`PdeError`]); `E-PDE-15x` from the
/// Sturm–Liouville solver for a separated direction it does not handle.
pub fn separation_solve_with(
    problem: &PdeProblem,
    opts: &SeparationOptions,
    pool: &ExprPool,
) -> Result<SeriesSolution, PdeError> {
    if problem.equations().len() != 1 || problem.unknowns().len() != 1 {
        return Err(PdeError::SeparationUnsupported(
            "separation of variables takes one equation in one unknown".into(),
        ));
    }
    let eq = problem.equations()[0];
    let u = problem.unknowns()[0];
    let dom = problem.domain().ok_or_else(|| {
        PdeError::SeparationUnsupported(
            "declare the domain with .on(…): intervals, boxes, curvilinear regions and one \
             half-line for time"
                .into(),
        )
    })?;
    let mut taken = Vec::new();
    names_in(eq, pool, &mut taken);
    for c in problem.conditions() {
        names_in(c.lhs, pool, &mut taken);
        names_in(c.rhs, pool, &mut taken);
    }
    let mut ctx = Ctx { opts, pool, taken };
    // A Cartesian disk / ball: rewrite in polar / spherical coordinates.
    if let Some((prob, sys, curv, cart, center)) =
        cartesian_round(eq, u, dom, problem.conditions(), &mut ctx)?
    {
        let mut sol = finish(solve_prob(&prob, &mut ctx)?, &mut ctx);
        let back = from_cartesian(sys, &cart, &center, pool);
        let m: HashMap<ExprId, ExprId> = curv.iter().copied().zip(back).collect();
        sol.curvilinear_value = Some(sol.value);
        sol.value = simp(subs(sol.value, &m, pool), pool);
        sol.coordinates = Some((sys, curv));
        return Ok(sol);
    }
    let prob = build_prob(eq, u, dom, problem.conditions(), pool)?;
    let part = solve_prob(&prob, &mut ctx)?;
    Ok(finish(part, &mut ctx))
}

fn finish(part: Part, ctx: &mut Ctx<'_>) -> SeriesSolution {
    let pool = ctx.pool;
    let value = simp(pool.add(part.value.clone()), pool);
    let mut checks = part.checks;
    let statuses: Vec<ZeroStatus> = checks.iter().map(|c| c.status).collect();
    let status = if statuses.iter().all(|s| *s == ZeroStatus::ProvedZero) {
        VerificationStatus::ExactlyVerified
    } else if statuses.contains(&ZeroStatus::ProvedNonzero) {
        VerificationStatus::Refuted
    } else if statuses
        .iter()
        .all(|s| matches!(s, ZeroStatus::ProvedZero | ZeroStatus::ProbablyZero))
    {
        VerificationStatus::NumericallyChecked
    } else {
        VerificationStatus::Unverified
    };
    checks.sort_by_key(|c| c.group);
    let kind = if super::super::sum::sum_parts(value, pool).is_some() || contains_sum(value, pool) {
        "series"
    } else {
        "closed_form"
    };
    let mut side = part.side;
    side.sort();
    side.dedup();
    let mut notes = part.notes;
    notes.dedup();
    let steady = if part.steady.is_empty() {
        None
    } else {
        Some(simp(pool.add(part.steady), pool))
    };
    let mut method = String::from("separation");
    let mut ms = part.methods.clone();
    ms.dedup();
    let mut uniq: Vec<&str> = Vec::new();
    for m in ms {
        if !uniq.contains(&m) {
            uniq.push(m);
        }
    }
    for m in &uniq {
        method.push('+');
        method.push_str(m);
    }
    SeriesSolution {
        value,
        kind,
        method,
        steady_state: steady,
        modes: part.groups,
        separation: part.separation,
        status,
        checks,
        projection: part.projection,
        convergence: if part.uniform { "uniform" } else { "l2" },
        side_conditions: side,
        notes,
        alternatives: part.alternatives,
        coordinates: None,
        curvilinear_value: None,
        constants: part.constants,
    }
}

pub(crate) fn contains_sum(e: ExprId, pool: &ExprPool) -> bool {
    let mut stack = vec![e];
    let mut seen = std::collections::HashSet::new();
    while let Some(n) = stack.pop() {
        if !seen.insert(n) {
            continue;
        }
        if super::super::sum::sum_parts(n, pool).is_some() {
            return true;
        }
        push_children(n, pool, &mut stack);
    }
    false
}

type RoundProblem = (
    Prob,
    CoordinateSystem,
    Vec<ExprId>,
    Vec<ExprId>,
    Vec<ExprId>,
);

/// A problem on a Cartesian `Disk` / `Ball` (possibly × a time half-line),
/// rewritten in polar / spherical coordinates.  Boundary conditions on the
/// circle / sphere are written `u(x, y, …) = g(x, y, …)` (the unknown at its
/// own arguments): Dirichlet data on the boundary.
fn cartesian_round(
    eq: ExprId,
    u: ExprId,
    dom: &PdeDomain,
    conds: &[PdeCondition],
    ctx: &mut Ctx<'_>,
) -> Result<Option<RoundProblem>, PdeError> {
    let pool = ctx.pool;
    let parts: Vec<&PdeDomain> = match dom {
        PdeDomain::Product(ps) => ps.iter().collect(),
        other => vec![other],
    };
    let round = parts
        .iter()
        .position(|p| matches!(p, PdeDomain::Disk { .. } | PdeDomain::Ball { .. }));
    let Some(ri) = round else { return Ok(None) };
    let (cart, center, radius) = match parts[ri] {
        PdeDomain::Disk {
            vars,
            center,
            radius,
        } => (vars.to_vec(), center.to_vec(), *radius),
        PdeDomain::Ball {
            vars,
            center,
            radius,
        } => (vars.clone(), center.clone(), *radius),
        _ => unreachable!(),
    };
    let sys = match cart.len() {
        2 => CoordinateSystem::Polar,
        3 => CoordinateSystem::Spherical,
        n => {
            return Err(PdeError::SeparationUnsupported(format!(
                "a ball in {n} dimensions is not supported (2 and 3)"
            )))
        }
    };
    let uv = decode_unknown(u, pool)?;
    let names: &[&str] = if sys == CoordinateSystem::Polar {
        &["r", "theta"]
    } else {
        &["r", "theta", "phi"]
    };
    let curv: Vec<ExprId> = names
        .iter()
        .enumerate()
        .map(|(i, n)| {
            sym_named(
                n,
                if i == 0 {
                    Domain::Positive
                } else {
                    Domain::Real
                },
                ctx,
            )
        })
        .collect();
    let mut new_args = uv.args.clone();
    for (k, c) in cart.iter().enumerate() {
        let slot = uv.args.iter().position(|a| a == c).ok_or_else(|| {
            PdeError::SeparationUnsupported(format!(
                "{} is not an argument of {}",
                pool.display(*c),
                pool.display(u)
            ))
        })?;
        new_args[slot] = curv[k];
    }
    let v = pool.func(uv.name.clone(), new_args.clone());
    let eq2 = isotropic_to_curvilinear(eq, u, v, &cart, &curv, &center, sys, pool)?;
    let pi = pi(pool);
    let bounds: Vec<(ExprId, ExprId)> = match sys {
        CoordinateSystem::Polar => vec![(pool.integer(0), radius), (neg(pi, pool), pi)],
        _ => vec![
            (pool.integer(0), radius),
            (pool.integer(0), pi),
            (neg(pi, pool), pi),
        ],
    };
    let cdom = PdeDomain::curvilinear(
        sys,
        &curv.iter().map(|&c| Some(c)).collect::<Vec<_>>(),
        &bounds,
        pool,
    )?;
    let mut new_parts: Vec<PdeDomain> = parts
        .iter()
        .enumerate()
        .filter(|(i, _)| *i != ri)
        .map(|(_, p)| (*p).clone())
        .collect();
    new_parts.push(cdom);
    let ndom = PdeDomain::Product(new_parts);
    let cmap = to_cartesian(sys, &curv, &center, pool);
    let m: HashMap<ExprId, ExprId> = cart.iter().copied().zip(cmap.iter().copied()).collect();
    let mut nconds = Vec::new();
    for c in conds {
        // Occurrences of u at its own arguments → the boundary.
        let r = c.residual(pool);
        let own = mentions_node(r, u, pool);
        if own {
            let mut on_bdry = m.clone();
            on_bdry.insert(curv[0], radius);
            let at_r = {
                let mut a = new_args.clone();
                let slot = new_args
                    .iter()
                    .position(|&a| a == curv[0])
                    .expect("radius slot");
                a[slot] = radius;
                pool.func(uv.name.clone(), a)
            };
            let lhs = super::separate::replace_nodes(c.lhs, &HashMap::from([(u, at_r)]), pool);
            if mentions_cart_unknown(lhs, &uv.name, pool, &cart, &uv.args) {
                return Err(PdeError::SeparationCondition(
                    "on a Cartesian disk / ball only Dirichlet data u(x, y, …) = g on the \
                     boundary is supported; pose other conditions in polar / spherical \
                     coordinates"
                        .into(),
                ));
            }
            let rhs = simp(
                subs(
                    subs(c.rhs, &m, pool),
                    &HashMap::from([(curv[0], radius)]),
                    pool,
                ),
                pool,
            );
            nconds.push(PdeCondition {
                kind: ConditionKind::Boundary,
                lhs,
                rhs,
            });
        } else {
            // Initial data: u(x, y, 0) = f(x, y).
            let lhs = rewrite_unknown_args(c.lhs, &uv.name, &uv.args, &new_args, pool);
            let rhs = simp(subs(c.rhs, &m, pool), pool);
            nconds.push(PdeCondition {
                kind: c.kind,
                lhs,
                rhs,
            });
        }
    }
    let prob = build_prob(eq2, v, &ndom, &nconds, pool)?;
    Ok(Some((prob, sys, curv, cart, center)))
}

fn mentions_node(e: ExprId, node: ExprId, pool: &ExprPool) -> bool {
    let mut stack = vec![e];
    let mut seen = std::collections::HashSet::new();
    while let Some(n) = stack.pop() {
        if !seen.insert(n) {
            continue;
        }
        if n == node {
            return true;
        }
        if let Some(v) = view(n, pool) {
            if v.inner == node && n != node {
                return true;
            }
        }
        push_children(n, pool, &mut stack);
    }
    false
}

fn mentions_cart_unknown(
    e: ExprId,
    name: &str,
    pool: &ExprPool,
    cart: &[ExprId],
    args: &[ExprId],
) -> bool {
    let mut stack = vec![e];
    let mut seen = std::collections::HashSet::new();
    while let Some(n) = stack.pop() {
        if !seen.insert(n) {
            continue;
        }
        if let Some(v) = view(n, pool) {
            if v.name == name
                && v.args
                    .iter()
                    .zip(args)
                    .any(|(a, b)| a == b && cart.contains(b))
            {
                return true;
            }
        }
        push_children(n, pool, &mut stack);
    }
    false
}

/// `u(x, y, 0)` → `u(r, θ, 0)`: the Cartesian slots take the new coordinates.
fn rewrite_unknown_args(
    e: ExprId,
    name: &str,
    old: &[ExprId],
    new: &[ExprId],
    pool: &ExprPool,
) -> ExprId {
    let mut map = HashMap::new();
    let mut stack = vec![e];
    let mut seen = std::collections::HashSet::new();
    while let Some(n) = stack.pop() {
        if !seen.insert(n) {
            continue;
        }
        if let Some(v) = view(n, pool) {
            if v.name == name {
                let args: Vec<ExprId> = v
                    .args
                    .iter()
                    .enumerate()
                    .map(|(i, &a)| if a == old[i] { new[i] } else { a })
                    .collect();
                let inner = pool.func(name.to_string(), args);
                map.insert(n, crate::diff::undefined::build(inner, &v.orders, pool));
                continue;
            }
        }
        push_children(n, pool, &mut stack);
    }
    super::separate::replace_nodes(e, &map, pool)
}

// ---------------------------------------------------------------------------
// Dispatch: lifting, superposition, core
// ---------------------------------------------------------------------------

fn time_slot(p: &Prob) -> Option<usize> {
    p.coords.iter().position(|c| c.kind == CoordKind::Time)
}

fn face_at(p: &Prob, slot: usize, hi: bool, pool: &ExprPool) -> Option<FaceCondition> {
    let co = p.coords[slot];
    let at = if hi { co.hi } else { co.lo };
    p.faces
        .iter()
        .find(|f| f.slot == slot && eq_expr(f.at, at, pool))
        .cloned()
}

fn check_faces(p: &Prob, pool: &ExprPool) -> Result<(), PdeError> {
    for (slot, co) in p.coords.iter().enumerate() {
        if co.kind != CoordKind::Finite {
            continue;
        }
        for hi in [false, true] {
            let sing = if hi { co.sing_hi } else { co.sing_lo };
            let at = if hi { co.hi } else { co.lo };
            let n = p
                .faces
                .iter()
                .filter(|f| f.slot == slot && eq_expr(f.at, at, pool))
                .count();
            if sing {
                continue;
            }
            if n != 1 {
                return Err(PdeError::SeparationMissingData(format!(
                    "{} boundary condition(s) at {} = {}; exactly one is needed",
                    n,
                    pool.display(co.var),
                    pool.display(at)
                )));
            }
        }
    }
    Ok(())
}

fn solve_prob(p: &Prob, ctx: &mut Ctx<'_>) -> Result<Part, PdeError> {
    let pool = ctx.pool;
    check_faces(p, pool)?;
    let nonhom: Vec<usize> = {
        let mut v: Vec<usize> = p
            .faces
            .iter()
            .filter(|f| !is_zero_expr(f.data, pool))
            .map(|f| f.slot)
            .collect();
        v.sort();
        v.dedup();
        v
    };
    let has_source = !is_zero_expr(p.source, pool);
    if let Some(ts) = time_slot(p) {
        if !nonhom.is_empty() {
            return lift(p, ts, &nonhom, ctx);
        }
        return core(p, Some(ts), ctx);
    }
    match (nonhom.len(), has_source) {
        (0, false) => Ok(Part {
            value: vec![pool.integer(0)],
            notes: vec![
                "homogeneous equation and data: the zero solution (unique unless 0 is an \
                 eigenvalue of the problem)"
                    .into(),
            ],
            uniform: true,
            ..Part::default()
        }),
        // One variable: an ODE boundary-value problem in it.
        (0, true) if p.vars.len() == 1 => core(p, Some(0), ctx),
        (0, true) => core(p, None, ctx),
        (1, false) => core(p, Some(nonhom[0]), ctx),
        _ => {
            // Superposition: one sub-problem per data-carrying variable,
            // plus the source with homogeneous data.
            let mut total = Part {
                uniform: true,
                ..Part::default()
            };
            for &j in &nonhom {
                let mut q = p.clone();
                q.source = pool.integer(0);
                for f in &mut q.faces {
                    if f.slot != j {
                        f.data = pool.integer(0);
                    }
                }
                total.merge(core(&q, Some(j), ctx)?);
            }
            if has_source {
                let mut q = p.clone();
                for f in &mut q.faces {
                    f.data = pool.integer(0);
                }
                total.merge(core(&q, None, ctx)?);
            }
            total.methods.push("superposition");
            Ok(total)
        }
    }
}

/// Homogenise the boundary data of an evolution problem: a steady state
/// (time-independent data and source, closed form) or a polynomial lifting
/// in the one variable whose faces carry data.
fn lift(p: &Prob, ts: usize, nonhom: &[usize], ctx: &mut Ctx<'_>) -> Result<Part, PdeError> {
    let pool = ctx.pool;
    let t = p.vars[ts];
    let data_t_free =
        p.faces.iter().all(|f| !mentions_var(f.data, t, pool)) && !mentions_var(p.source, t, pool);
    let mut lifting: Option<(ExprId, &'static str)> = None;
    let mut part_steady = Part::default();
    if data_t_free && ctx.opts.steady_state {
        // Steady problem: the unknown without t.
        let sp_vars: Vec<ExprId> = p.vars.iter().copied().filter(|&v| v != t).collect();
        let w = pool.func(p.name.clone(), sp_vars.clone());
        let eq_ss = replace_function(pool.add(vec![p.lhom, p.source]), p.u, w, pool)
            .map_err(|e| PdeError::SeparationLifting(e.to_string()))?;
        let mut q = p.clone();
        q.u = w;
        q.vars = sp_vars;
        q.coords = p
            .coords
            .iter()
            .enumerate()
            .filter(|&(i, _)| i != ts)
            .map(|(_, c)| *c)
            .collect();
        let (lh, src) = split_linear(eq_ss, &p.name, pool)?;
        q.lhom = lh;
        q.source = src;
        q.faces = p
            .faces
            .iter()
            .map(|f| {
                let mut g = f.clone();
                if g.slot > ts {
                    g.slot -= 1;
                }
                g
            })
            .collect();
        q.ics.clear();
        if q.lhom != pool.integer(0) {
            if let Ok(part) = solve_prob(&q, ctx) {
                let v = simp(pool.add(part.value.clone()), pool);
                if !contains_sum(v, pool) && !has_integral(v, pool) {
                    lifting = Some((v, "steady_state"));
                    part_steady = part;
                }
            }
        }
    }
    if lifting.is_none() {
        if nonhom.len() != 1 {
            return Err(PdeError::SeparationLifting(
                "boundary data in more than one variable that is time dependent (or whose \
                 steady state is not in closed form): no lifting is constructed"
                    .into(),
            ));
        }
        let j = nonhom[0];
        let l = poly_lifting(p, j, pool)?;
        lifting = Some((l, "lifting"));
    }
    let (ell, how) = lifting.expect("set above");
    // v = u − ℓ.
    let vname = super::separate::fresh_name("v", &mut ctx.taken);
    let v = pool.func(vname.clone(), p.vars.clone());
    let eq_v = replace_function(
        pool.add(vec![p.lhom, p.source]),
        p.u,
        pool.add(vec![v, ell]),
        pool,
    )
    .map_err(|e| PdeError::SeparationLifting(e.to_string()))?;
    let (lh, src) = split_linear(simp(eq_v, pool), &vname, pool)?;
    let mut q = p.clone();
    q.u = v;
    q.name = vname;
    q.lhom = lh;
    q.source = src;
    for f in &mut q.faces {
        f.data = pool.integer(0);
    }
    // Initial data of v.
    let t0 = p.coords[ts].lo;
    for ic in &mut q.ics {
        let k = ic
            .coeffs
            .iter()
            .position(|&c| c != pool.integer(0))
            .unwrap_or(0);
        let mut l_k = ell;
        for _ in 0..k {
            l_k = d(l_k, t, pool)?;
        }
        let l_k0 = simp(subs1(l_k, t, t0, pool), pool);
        ic.data = simp(sub(ic.data, pool.mul(vec![ic.coeffs[k], l_k0]), pool), pool);
    }
    let mut part = solve_prob(&q, ctx)?;
    // Checks of the lifting: boundary conditions, and the steady equation.
    let mut bad = Vec::new();
    for f in &p.faces {
        let r = face_residual(ell, f, &p.vars, pool)?;
        bad.push((
            "lifting",
            super::project::zero_status(r, pool),
            format!(
                "boundary condition at {} = {}",
                pool.display(p.vars[f.slot]),
                pool.display(f.at)
            ),
        ));
    }
    for (w, st, det) in bad {
        part.checks.push(SeriesCheck {
            what: w,
            group: usize::MAX,
            status: st,
            method: "zero_test",
            detail: det,
        });
    }
    part.value.push(ell);
    part.steady.push(ell);
    part.methods.push(how);
    if how == "steady_state" {
        part.side.extend(part_steady.side);
        part.notes.push(format!(
            "u = steady state {} + transient; the transient decays when the spatial operator \
             is dissipative",
            pool.display(ell)
        ));
    } else {
        part.notes.push(format!(
            "u = lifting {} (meets the boundary data) + a series with homogeneous boundary \
             conditions; time-dependent data enter the series as a source (Duhamel)",
            pool.display(ell)
        ));
        if mentions_var(ell, t, pool) {
            part.methods.push("duhamel");
        }
    }
    Ok(part)
}

/// `α ℓ + β ∂ℓ` at the face, minus the data.
fn face_residual(
    ell: ExprId,
    f: &FaceCondition,
    vars: &[ExprId],
    pool: &ExprPool,
) -> Result<ExprId, PdeError> {
    let x = vars[f.slot];
    let mut terms = Vec::new();
    let mut cur = ell;
    for (k, &c) in f.coeffs.iter().enumerate() {
        if k > 0 {
            cur = d(cur, x, pool)?;
        }
        terms.push(pool.mul(vec![c, subs1(cur, x, f.at, pool)]));
    }
    Ok(simp(sub(pool.add(terms), f.data, pool), pool))
}

/// `ℓ = C₁ ψ₁(x) + C₂ ψ₂(x)` meeting the two faces of slot `j`.
fn poly_lifting(p: &Prob, j: usize, pool: &ExprPool) -> Result<ExprId, PdeError> {
    let co = p.coords[j];
    if co.kind != CoordKind::Finite || co.sing_lo || co.sing_hi {
        return Err(PdeError::SeparationLifting(format!(
            "no polynomial lifting in {}",
            pool.display(co.var)
        )));
    }
    let x = co.var;
    let fl = face_at(p, j, false, pool)
        .ok_or_else(|| PdeError::SeparationMissingData("left face".into()))?;
    let fr = face_at(p, j, true, pool)
        .ok_or_else(|| PdeError::SeparationMissingData("right face".into()))?;
    let z = sub(x, co.lo, pool);
    let bases = [
        (pool.integer(1), z),
        (z, pool.pow(z, pool.integer(2))),
        (pool.integer(1), pool.pow(z, pool.integer(2))),
    ];
    let func = |f: &FaceCondition, psi: ExprId| -> Result<ExprId, PdeError> {
        let mut terms = Vec::new();
        let mut cur = psi;
        for (k, &c) in f.coeffs.iter().enumerate() {
            if k > 0 {
                cur = d(cur, x, pool)?;
            }
            terms.push(pool.mul(vec![c, subs1(cur, x, f.at, pool)]));
        }
        Ok(simp(pool.add(terms), pool))
    };
    for (p1, p2) in bases {
        let (a, b) = (func(&fl, p1)?, func(&fl, p2)?);
        let (c, dd) = (func(&fr, p1)?, func(&fr, p2)?);
        let det = simp(sub(pool.mul(vec![a, dd]), pool.mul(vec![b, c]), pool), pool);
        if is_zero_expr(det, pool) {
            continue;
        }
        let (g0, g1) = (fl.data, fr.data);
        let c1 = div(
            sub(pool.mul(vec![g0, dd]), pool.mul(vec![g1, b]), pool),
            det,
            pool,
        );
        let c2 = div(
            sub(pool.mul(vec![a, g1]), pool.mul(vec![c, g0]), pool),
            det,
            pool,
        );
        return Ok(simp(
            pool.add(vec![pool.mul(vec![c1, p1]), pool.mul(vec![c2, p2])]),
            pool,
        ));
    }
    Err(PdeError::SeparationLifting(format!(
        "no lifting of the form A + Bx, Ax + Bx², A + Bx² meets the conditions in {}",
        pool.display(x)
    )))
}

// ---------------------------------------------------------------------------
// Directions and their spectra
// ---------------------------------------------------------------------------

#[derive(Clone, Debug)]
struct DirMember {
    index: Option<ExprId>,
    first: ExprId,
    eigenvalue: ExprId,
    eigenfunctions: Vec<ExprId>,
    norms: Vec<Option<ExprId>>,
    weight: ExprId,
    /// `(ν, ν_n, λ(ν), φ(ν))` for a `RootOfFamily` member.
    nu: Option<(ExprId, ExprId, ExprId, Vec<ExprId>)>,
    status: VerificationStatus,
    method: &'static str,
    kind: &'static str,
    /// Legendre degree / periodic frequency, when known.
    degree: Option<ExprId>,
    /// The last index value (`None`: unbounded).
    last: Option<i64>,
    /// Nodes that vanish by the definition of a `RootOfFamily` member
    /// (`J_m(R ν_n)` for Bessel–Dirichlet).
    zero_nodes: Vec<ExprId>,
}

impl DirMember {
    fn subs_all(&self, m: &HashMap<ExprId, ExprId>, pool: &ExprPool) -> DirMember {
        let s = |e: ExprId| simp(subs(e, m, pool), pool);
        let mut out = self.clone();
        out.first = s(self.first);
        out.eigenvalue = s(self.eigenvalue);
        out.eigenfunctions = self
            .eigenfunctions
            .iter()
            .map(|&e| {
                let v = s(e);
                if self.index.is_some_and(|i| m.contains_key(&i)) {
                    simp(
                        crate::orthopoly::expand_orthopoly(v, pool).unwrap_or(v),
                        pool,
                    )
                } else {
                    v
                }
            })
            .collect();
        out.norms = self.norms.iter().map(|n| n.map(s)).collect();
        out.weight = s(self.weight);
        out.degree = self.degree.map(s);
        out.zero_nodes = self.zero_nodes.iter().map(|&e| s(e)).collect();
        if let Some((v, f, l, phis)) = &self.nu {
            out.nu = Some((*v, s(*f), s(*l), phis.iter().map(|&e| s(e)).collect()));
        }
        if let Some(i) = self.index {
            if m.contains_key(&i) {
                out.index = None;
            }
        }
        out
    }
}

struct Dir {
    slot: usize,
    ode: ExprId,
    factor: ExprId,
    constant: ExprId,
}

fn end_condition(f: &FaceCondition, pool: &ExprPool) -> EndCondition {
    let zero = pool.integer(0);
    let a = f.coeffs.first().copied().unwrap_or(zero);
    let b = f.coeffs.get(1).copied().unwrap_or(zero);
    if is_zero_expr(b, pool) {
        EndCondition::Dirichlet
    } else if is_zero_expr(a, pool) {
        EndCondition::Neumann
    } else {
        EndCondition::Robin { alpha: a, beta: b }
    }
}

/// The spectrum of one separated direction, as members.
#[allow(clippy::too_many_arguments)]
fn dir_spectrum(
    p: &Prob,
    dir: &Dir,
    subst: &HashMap<ExprId, ExprId>,
    degree_hint: Option<ExprId>,
    index_name: &str,
    ctx: &mut Ctx<'_>,
    side: &mut Vec<ExprId>,
    notes: &mut Vec<String>,
) -> Result<Vec<DirMember>, PdeError> {
    let pool = ctx.pool;
    let co = p.coords[dir.slot];
    let x = co.var;
    let ode = simp(subs(dir.ode, subst, pool), pool);
    let boundary = match co.kind {
        CoordKind::Periodic => SlBoundary::Periodic,
        _ => {
            let left = if co.sing_lo {
                EndCondition::Bounded
            } else {
                end_condition(&face_at(p, dir.slot, false, pool).expect("checked"), pool)
            };
            let right = if co.sing_hi {
                EndCondition::Bounded
            } else {
                end_condition(&face_at(p, dir.slot, true, pool).expect("checked"), pool)
            };
            SlBoundary::Separated { left, right }
        }
    };
    if let Some(ms) = liouville_shifted(dir, ode, co, boundary, index_name, side, notes, pool)? {
        return Ok(ms);
    }
    let slp = SturmLiouvilleProblem::from_ode(
        ode,
        dir.factor,
        dir.constant,
        co.lo,
        co.hi,
        boundary,
        pool,
    )?;
    if co.role == Role::SphTheta && co.sing_lo && co.sing_hi {
        if let Some(ms) = legendre_members(&slp, index_name, pool)? {
            return Ok(ms);
        }
    }
    if co.role == Role::Radius && p.system == Some(CoordinateSystem::Spherical) {
        if let Some(ms) =
            spherical_radial_members(&slp, boundary, degree_hint, index_name, ctx, side, notes)?
        {
            return Ok(ms);
        }
    }
    let opts = SlOptions::default().with_index_name(index_name);
    let es = sturm_liouville_with(&slp, &opts, pool)?;
    let mut ms = members_of(&es, side, notes, pool)?;
    if es.kind == "periodic" {
        clean_periodic(&mut ms, &ode, dir.constant, dir.factor, x, co, pool);
    }
    if es.kind == "bessel" {
        for (m, f) in ms
            .iter_mut()
            .filter(|m| m.kind == "family")
            .zip(es.families.iter())
        {
            m.norms = m
                .norms
                .iter()
                .map(|n| n.map(|nn| reduce_bessel_at_root(nn, f, pool)))
                .collect();
        }
    }
    Ok(ms)
}

/// A constant-coefficient direction with a first-derivative term
/// (`a₂X'' + a₁X' + (b + cμ)X = 0`, convection): `X = e^{−σx}Y` with
/// `σ = a₁/(2a₂)` removes it — `a₂Y'' + (b − a₁²/(4a₂) + cμ)Y = 0` — and a
/// Robin condition `αX + βX' = 0` becomes `(α − βσ)Y + βY' = 0`.  The members
/// are mapped back (`X = e^{−σx}Y`, weight `e^{2σx}w`, the same norms).
#[allow(clippy::too_many_arguments)]
fn liouville_shifted(
    dir: &Dir,
    ode: ExprId,
    co: Coord,
    boundary: SlBoundary,
    index_name: &str,
    side: &mut Vec<ExprId>,
    notes: &mut Vec<String>,
    pool: &ExprPool,
) -> Result<Option<Vec<DirMember>>, PdeError> {
    let x = co.var;
    let SlBoundary::Separated { left, right } = boundary else {
        return Ok(None);
    };
    let Ok(m) = mode_ode(ode, dir.factor, pool) else {
        return Ok(None);
    };
    if m.order != 2 || is_zero_expr(m.a1, pool) {
        return Ok(None);
    }
    let free = |e: ExprId| !mentions_var(e, x, pool);
    if !(free(m.a2) && free(m.a1) && free(m.a0)) {
        return Ok(None);
    }
    let sigma = simp(div(m.a1, pool.mul(vec![pool.integer(2), m.a2]), pool), pool);
    let shift = |c: EndCondition| -> Option<EndCondition> {
        Some(match c {
            EndCondition::Dirichlet => EndCondition::Dirichlet,
            EndCondition::Neumann => EndCondition::Robin {
                alpha: simp(neg(sigma, pool), pool),
                beta: pool.integer(1),
            },
            EndCondition::Robin { alpha, beta } => EndCondition::Robin {
                alpha: simp(sub(alpha, pool.mul(vec![beta, sigma]), pool), pool),
                beta,
            },
            _ => return None,
        })
    };
    let (Some(l2), Some(r2)) = (shift(left), shift(right)) else {
        return Ok(None);
    };
    let y = dir.factor;
    let y2 = crate::diff::undefined::build(y, &[2], pool);
    let new_a0 = simp(
        sub(
            m.a0,
            div(
                pool.pow(m.a1, pool.integer(2)),
                pool.mul(vec![pool.integer(4), m.a2]),
                pool,
            ),
            pool,
        ),
        pool,
    );
    let ode_y = pool.add(vec![pool.mul(vec![m.a2, y2]), pool.mul(vec![new_a0, y])]);
    // Any failure on the transformed problem falls back to the direct path.
    let Ok(slp) = SturmLiouvilleProblem::from_ode(
        ode_y,
        y,
        dir.constant,
        co.lo,
        co.hi,
        SlBoundary::Separated {
            left: l2,
            right: r2,
        },
        pool,
    ) else {
        return Ok(None);
    };
    let opts = SlOptions::default().with_index_name(index_name);
    let Ok(es) = sturm_liouville_with(&slp, &opts, pool) else {
        return Ok(None);
    };
    let mut side_y = Vec::new();
    let mut notes_y = Vec::new();
    let Ok(mut ms) = members_of(&es, &mut side_y, &mut notes_y, pool) else {
        return Ok(None);
    };
    side.extend(side_y);
    notes.extend(notes_y);
    let damp = pool.func("exp", vec![neg(pool.mul(vec![sigma, x]), pool)]);
    let grow = pool.func("exp", vec![pool.mul(vec![pool.integer(2), sigma, x])]);
    for mm in &mut ms {
        mm.eigenfunctions = mm
            .eigenfunctions
            .iter()
            .map(|&f| simp(pool.mul(vec![damp, f]), pool))
            .collect();
        if let Some((v, f, l, phis)) = &mm.nu {
            mm.nu = Some((
                *v,
                *f,
                *l,
                phis.iter()
                    .map(|&q| simp(pool.mul(vec![damp, q]), pool))
                    .collect(),
            ));
        }
        mm.weight = simp(pool.mul(vec![grow, mm.weight]), pool);
    }
    notes.push(format!(
        "the first-derivative term in {} was removed by X = exp(-({}) {}) Y",
        pool.display(x),
        pool.display(sigma),
        pool.display(x)
    ));
    Ok(Some(ms))
}

fn members_of(
    es: &Eigensystem,
    side: &mut Vec<ExprId>,
    notes: &mut Vec<String>,
    pool: &ExprPool,
) -> Result<Vec<DirMember>, PdeError> {
    let mut out = Vec::new();
    for m in &es.modes {
        if let Some(c) = m.condition {
            side.push(pool.predicate(PredicateKind::Not, vec![c]));
            notes.push(format!(
                "λ = {} is an eigenvalue only when {}; that case is excluded by the side \
                 conditions",
                pool.display(m.eigenvalue),
                pool.display(c)
            ));
            continue;
        }
        let nsq = m.norm_squared.or_else(|| {
            super::project::definite(
                pool.mul(vec![es.weight, pool.pow(m.eigenfunction, pool.integer(2))]),
                es.var,
                es.interval.0,
                es.interval.1,
                pool,
            )
        });
        out.push(DirMember {
            index: None,
            first: pool.integer(0),
            eigenvalue: m.eigenvalue,
            eigenfunctions: vec![m.eigenfunction],
            norms: vec![nsq],
            weight: es.weight,
            nu: None,
            status: m.verification,
            method: "sturm_liouville",
            kind: "mode",
            degree: Some(pool.integer(0)),
            last: None,
            zero_nodes: Vec::new(),
        });
    }
    for f in &es.families {
        out.push(family_member(f, es.weight, pool));
    }
    match &es.lower {
        LowerSpectrum::None => {}
        LowerSpectrum::Conditional { condition, .. } => {
            side.push(*condition);
            notes.push(format!(
                "no eigenvalue below {} provided {} (Rayleigh quotient); otherwise further \
                 modes occur",
                pool.display(es.threshold),
                pool.display(*condition)
            ));
        }
        LowerSpectrum::Present { .. } => {
            return Err(PdeError::SeparationUnsupported(
                "the separated eigenproblem has eigenvalues below its threshold (a Robin \
                 condition with the 'wrong' sign); those modes are not assembled"
                    .into(),
            ))
        }
    }
    side.extend(es.side_conditions.iter().copied());
    notes.extend(es.notes.iter().cloned());
    Ok(out)
}

fn family_member(f: &EigenFamily, weight: ExprId, pool: &ExprPool) -> DirMember {
    let nu = if f.closed_form {
        None
    } else {
        Some((
            f.frequency_var,
            f.frequency,
            f.eigenvalue_of_frequency,
            f.eigenfunctions_of_frequency.clone(),
        ))
    };
    DirMember {
        index: Some(f.index),
        first: pool.integer(f.first),
        eigenvalue: f.eigenvalue,
        eigenfunctions: f.eigenfunctions.clone(),
        norms: f.norms_squared.clone(),
        weight,
        nu,
        status: f.verification.status,
        method: f.verification.method,
        kind: "family",
        degree: Some(f.frequency),
        last: None,
        zero_nodes: match f.defining_equation {
            Some(def) if !f.closed_form => {
                vec![simp(subs1(def, f.frequency_var, f.frequency, pool), pool)]
            }
            _ => Vec::new(),
        },
    }
}

/// Periodic families: `cos(νx)`, `sin(νx)` instead of the shifted forms,
/// when they verify (the ODE and both periodicity conditions, exactly).
fn clean_periodic(
    ms: &mut [DirMember],
    ode: &ExprId,
    constant: ExprId,
    factor: ExprId,
    x: ExprId,
    co: Coord,
    pool: &ExprPool,
) {
    for m in ms.iter_mut().filter(|m| m.kind == "family") {
        let Some(nu) = m.degree else { continue };
        let cands = [
            pool.func("cos", vec![pool.mul(vec![nu, x])]),
            pool.func("sin", vec![pool.mul(vec![nu, x])]),
        ];
        let lam = m.eigenvalue;
        let ode_n = subs1(*ode, constant, lam, pool);
        let ok = cands.iter().all(|&c| {
            let r = replace_function(ode_n, factor, c, pool).ok();
            let Some(r) = r else { return false };
            let r = simp(r, pool);
            let per0 = sub(subs1(c, x, co.hi, pool), subs1(c, x, co.lo, pool), pool);
            let dc = d(c, x, pool).unwrap_or(c);
            let per1 = sub(subs1(dc, x, co.hi, pool), subs1(dc, x, co.lo, pool), pool);
            super::project::zero_status(r, pool) == ZeroStatus::ProvedZero
                && super::project::zero_status(per0, pool) == ZeroStatus::ProvedZero
                && super::project::zero_status(per1, pool) == ZeroStatus::ProvedZero
        });
        if ok {
            m.eigenfunctions = cands.iter().map(|&c| simp(c, pool)).collect();
        }
    }
}

/// `∫ r J_m(νr)²` at a root of `J_m(νR)`: `J_{m−1} → (2m/z)J_m − J_{m+1}`,
/// then `J_m(νR) → 0` (an identity at the root).
fn reduce_bessel_at_root(norm: ExprId, fam: &EigenFamily, pool: &ExprPool) -> ExprId {
    let Some(def) = fam.defining_equation else {
        return norm;
    };
    let Some((m0, _)) = pool.with(def, |dd| match dd {
        ExprData::Func { name, args } if name == "bessel_j" && args.len() == 2 => {
            Some((args[0], args[1]))
        }
        _ => None,
    }) else {
        return norm;
    };
    let at_root = simp(subs1(def, fam.frequency_var, fam.frequency, pool), pool);
    let z = match pool.get(at_root) {
        ExprData::Func { args, .. } if args.len() == 2 => args[1],
        _ => return norm,
    };
    let lower = simp(sub(m0, pool.integer(1), pool), pool);
    let upper = simp(pool.add(vec![m0, pool.integer(1)]), pool);
    let jl = pool.func("bessel_j", vec![lower, z]);
    let ju = pool.func("bessel_j", vec![upper, z]);
    let jm = pool.func("bessel_j", vec![m0, z]);
    let repl = sub(
        pool.mul(vec![pool.integer(2), m0, pool.pow(z, pool.integer(-1)), jm]),
        ju,
        pool,
    );
    let mut map = HashMap::new();
    map.insert(simp(jl, pool), repl);
    let e = super::separate::replace_nodes(norm, &map, pool);
    let e = super::separate::replace_nodes(
        e,
        &HashMap::from([(simp(jm, pool), pool.integer(0))]),
        pool,
    );
    let e = simp(e, pool);
    let e = super::separate::replace_nodes(
        e,
        &HashMap::from([
            (simp(jm, pool), pool.integer(0)),
            (at_root, pool.integer(0)),
        ]),
        pool,
    );
    simp(e, pool)
}

/// `(sin θ Θ')' − (c/sin θ)Θ + λ sin θ Θ = 0` on `[0, π]`, bounded: Legendre
/// in `cos θ` (`c = m²`: associated Legendre, degrees `n ≥ m`).
fn legendre_members(
    slp: &SturmLiouvilleProblem,
    index_name: &str,
    pool: &ExprPool,
) -> Result<Option<Vec<DirMember>>, PdeError> {
    let th = slp.var;
    let s = pool.func("sin", vec![th]);
    if !is_zero_expr(sub(slp.p, s, pool), pool) || !is_zero_expr(sub(slp.w, s, pool), pool) {
        return Ok(None);
    }
    let c = simp(neg(pool.mul(vec![slp.q, s]), pool), pool);
    if mentions_var(c, th, pool) {
        return Ok(None);
    }
    let ord = sqrt_pos(c, pool);
    let n = super::super::sturm_liouville::fresh_symbol(
        index_name,
        Domain::Integer,
        &[slp.p, slp.q, slp.w, th],
        pool,
    );
    let ct = pool.func("cos", vec![th]);
    let (ef, norm) = if ord == pool.integer(0) {
        (
            pool.func("legendre_p", vec![n, ct]),
            div(
                pool.integer(2),
                pool.add(vec![pool.mul(vec![pool.integer(2), n]), pool.integer(1)]),
                pool,
            ),
        )
    } else {
        let g = |e: ExprId| pool.func("gamma", vec![e]);
        (
            pool.func("assoc_legendre_p", vec![n, ord, ct]),
            div(
                pool.mul(vec![
                    pool.integer(2),
                    g(pool.add(vec![n, ord, pool.integer(1)])),
                ]),
                pool.mul(vec![
                    pool.add(vec![pool.mul(vec![pool.integer(2), n]), pool.integer(1)]),
                    g(pool.add(vec![n, neg(ord, pool), pool.integer(1)])),
                ]),
                pool,
            ),
        )
    };
    let lam = pool.mul(vec![n, pool.add(vec![n, pool.integer(1)])]);
    // Verify members: the ODE for concrete (ord, n).
    let ords: Vec<i64> = match pool.with(ord, |dd| match dd {
        ExprData::Integer(k) => k.0.to_i64(),
        _ => None,
    }) {
        Some(k) => vec![k],
        None => vec![0, 1, 2],
    };
    let mut ok = true;
    for &o in &ords {
        for k in o..o + 4 {
            let mm = HashMap::from([(n, pool.integer(k)), (ord, pool.integer(o))]);
            let f = simp(subs(ef, &mm, pool), pool);
            let f = crate::orthopoly::expand_orthopoly(f, pool).unwrap_or(f);
            let lamk = simp(subs(lam, &mm, pool), pool);
            let qk = simp(subs(slp.q, &mm, pool), pool);
            let r = pool.add(vec![
                d(pool.mul(vec![s, d(f, th, pool)?]), th, pool)?,
                pool.mul(vec![qk, f]),
                pool.mul(vec![lamk, s, f]),
            ]);
            if super::project::zero_status(r, pool) != ZeroStatus::ProvedZero {
                ok = false;
            }
        }
    }
    if !ok {
        return Err(PdeError::SlVerificationFailed(
            "a Legendre member failed its exact check".into(),
        ));
    }
    Ok(Some(vec![DirMember {
        index: Some(n),
        first: ord,
        eigenvalue: lam,
        eigenfunctions: vec![ef],
        norms: vec![Some(simp(norm, pool))],
        weight: s,
        nu: None,
        status: VerificationStatus::ExactlyVerified,
        method: "members",
        kind: "legendre",
        degree: Some(n),
        last: None,
        zero_nodes: Vec::new(),
    }]))
}

/// The radial problem of a ball: `(r²R')' − c R + λ r² R = 0`.  For `c = 0`
/// the substitution `R = X/r` gives `X'' + λX = 0` with `X(0) = 0`; for
/// `c = l(l+1)`, `R = S/√r` gives Bessel's problem of order `l + ½`.
#[allow(clippy::too_many_arguments)]
fn spherical_radial_members(
    slp: &SturmLiouvilleProblem,
    boundary: SlBoundary,
    degree: Option<ExprId>,
    index_name: &str,
    ctx: &mut Ctx<'_>,
    side: &mut Vec<ExprId>,
    notes: &mut Vec<String>,
) -> Result<Option<Vec<DirMember>>, PdeError> {
    let pool = ctx.pool;
    let r = slp.var;
    let r2 = pool.pow(r, pool.integer(2));
    if !is_zero_expr(sub(slp.p, r2, pool), pool)
        || !is_zero_expr(sub(slp.w, r2, pool), pool)
        || slp.a != pool.integer(0)
    {
        return Ok(None);
    }
    let c = simp(neg(slp.q, pool), pool);
    if mentions_var(c, r, pool) {
        return Ok(None);
    }
    let SlBoundary::Separated { right, .. } = boundary else {
        return Ok(None);
    };
    let r0 = slp.b;
    let opts = SlOptions::default().with_index_name(index_name);
    if is_zero_expr(c, pool) {
        let right_x = match right {
            EndCondition::Dirichlet => EndCondition::Dirichlet,
            EndCondition::Neumann => EndCondition::Robin {
                alpha: neg(pool.pow(r0, pool.integer(-1)), pool),
                beta: pool.integer(1),
            },
            EndCondition::Robin { alpha, beta } => EndCondition::Robin {
                alpha: simp(sub(alpha, div(beta, r0, pool), pool), pool),
                beta,
            },
            _ => return Ok(None),
        };
        let xp = SturmLiouvilleProblem::interval(
            r,
            pool.integer(0),
            r0,
            SlBoundary::Separated {
                left: EndCondition::Dirichlet,
                right: right_x,
            },
            pool,
        )?;
        let es = sturm_liouville_with(&xp, &opts, pool)?;
        let mut ms = members_of(&es, side, notes, pool)?;
        let ir = pool.pow(r, pool.integer(-1));
        for m in &mut ms {
            m.eigenfunctions = m
                .eigenfunctions
                .iter()
                .map(|&f| simp(pool.mul(vec![f, ir]), pool))
                .collect();
            if let Some((v, f, l, phis)) = &m.nu {
                m.nu = Some((
                    *v,
                    *f,
                    *l,
                    phis.iter()
                        .map(|&p| simp(pool.mul(vec![p, ir]), pool))
                        .collect(),
                ));
            }
            m.weight = r2;
        }
        return Ok(Some(ms));
    }
    let Some(l) = degree else { return Ok(None) };
    let order = simp(pool.add(vec![l, pool.rational(1, 2)]), pool);
    let outer = match right {
        EndCondition::Dirichlet => EndCondition::Dirichlet,
        EndCondition::Neumann => EndCondition::Robin {
            alpha: neg(
                div(pool.integer(1), pool.mul(vec![pool.integer(2), r0]), pool),
                pool,
            ),
            beta: pool.integer(1),
        },
        _ => return Ok(None),
    };
    let bp = SturmLiouvilleProblem::bessel(r, order, r0, outer, pool)?;
    let es = sturm_liouville_with(&bp, &opts, pool)?;
    let mut ms = members_of(&es, side, notes, pool)?;
    let isr = pool.pow(r, pool.rational(-1, 2));
    for m in &mut ms {
        m.eigenfunctions = m
            .eigenfunctions
            .iter()
            .map(|&f| simp(pool.mul(vec![f, isr]), pool))
            .collect();
        if let Some((v, f, lam, phis)) = &m.nu {
            m.nu = Some((
                *v,
                *f,
                *lam,
                phis.iter()
                    .map(|&p| simp(pool.mul(vec![p, isr]), pool))
                    .collect(),
            ));
        }
        m.weight = r2;
    }
    Ok(Some(ms))
}

// ---------------------------------------------------------------------------
// The core: separate, expand, project, solve the amplitudes
// ---------------------------------------------------------------------------

fn role_rank(r: Role) -> u8 {
    match r {
        Role::PolarAngle | Role::Azimuth => 0,
        Role::SphTheta => 1,
        Role::Plain => 2,
        Role::Radius => 3,
    }
}

fn index_names(n: usize, opts: &SeparationOptions) -> Vec<String> {
    if opts.index_names.len() >= n {
        return opts.index_names[..n].to_vec();
    }
    match n {
        0 => vec![],
        1 => vec!["n".into()],
        2 => vec!["m".into(), "n".into()],
        3 => vec!["l".into(), "m".into(), "n".into()],
        _ => (1..=n).map(|i| format!("n{i}")).collect(),
    }
}

/// The data of the `s` direction (or none), per data kind.
struct SDir {
    slot: usize,
    ode: ExprId,
    factor: ExprId,
    /// Initial data `(order, coefficient, data)`.
    ics: Vec<(usize, ExprId, ExprId)>,
    left: Option<FaceCondition>,
    right: Option<FaceCondition>,
}

fn core(p: &Prob, s: Option<usize>, ctx: &mut Ctx<'_>) -> Result<Part, PdeError> {
    let pool = ctx.pool;
    let eigen_slots: Vec<usize> = (0..p.vars.len())
        .filter(|&i| Some(i) != s && p.coords[i].kind != CoordKind::Time)
        .collect();
    if eigen_slots.is_empty() && s.is_none() {
        return Err(PdeError::SeparationUnsupported("nothing to expand".into()));
    }
    let mut prefer: Vec<usize> = eigen_slots.clone();
    prefer.sort_by_key(|&i| role_rank(p.coords[i].role));
    let kappa = sym_named("kappa", Domain::Real, ctx);
    let mut sopts =
        SeparateOptions::new().prefer(&prefer.iter().map(|&i| p.vars[i]).collect::<Vec<_>>());
    if let Some(si) = s {
        sopts = sopts.last(p.vars[si]);
    } else {
        sopts.spectral = Some(kappa);
    }
    // A single variable (an ODE boundary-value problem): no separation.
    let sep = if p.vars.len() == 1 {
        None
    } else {
        Some(separate_with(
            p.lhom,
            p.u,
            SeparationKind::Product,
            &sopts,
            pool,
        )?)
    };
    let mut dirs: Vec<Dir> = Vec::new();
    let mut sdir: Option<SDir> = None;
    if let Some(sep) = &sep {
        for (k, eqn) in sep.equations.iter().enumerate() {
            let slot = p
                .vars
                .iter()
                .position(|&v| v == eqn.var)
                .expect("a variable");
            if Some(slot) == s {
                sdir = Some(SDir {
                    slot,
                    ode: eqn.equation,
                    factor: eqn.factor,
                    ics: Vec::new(),
                    left: None,
                    right: None,
                });
                continue;
            }
            let constant = match eqn.constant {
                Some(c) => c,
                None if s.is_none() && k == sep.equations.len() - 1 => kappa,
                None => {
                    return Err(PdeError::SeparationNotSeparable(
                        "the separated directions do not match the problem".into(),
                    ))
                }
            };
            dirs.push(Dir {
                slot,
                ode: eqn.equation,
                factor: eqn.factor,
                constant,
            });
        }
    } else {
        let Some(si) = s else {
            return Err(PdeError::SeparationUnsupported(
                "a problem in one variable needs its boundary data".into(),
            ));
        };
        let f = pool.func(format!("{}_T", p.name), vec![p.vars[si]]);
        let ode = replace_function(p.lhom, p.u, f, pool)
            .map_err(|e| PdeError::SeparationUnsupported(e.to_string()))?;
        sdir = Some(SDir {
            slot: si,
            ode,
            factor: f,
            ics: Vec::new(),
            left: None,
            right: None,
        });
    }
    if let (Some(sd), Some(si)) = (&mut sdir, s) {
        let co = p.coords[si];
        if co.kind == CoordKind::Time {
            for ic in &p.ics {
                let k = ic
                    .coeffs
                    .iter()
                    .position(|&c| c != pool.integer(0))
                    .unwrap_or(0);
                sd.ics.push((k, ic.coeffs[k], ic.data));
            }
        } else {
            sd.left = face_at(p, si, false, pool);
            sd.right = face_at(p, si, true, pool);
        }
    }
    let names = index_names(dirs.len(), ctx.opts);
    // Expand the members of every direction (later ones depend on earlier
    // eigenvalues).
    let mut side = Vec::new();
    let mut notes = Vec::new();
    let mut combos: Vec<Vec<DirMember>> = vec![vec![]];
    for (i, dir) in dirs.iter().enumerate() {
        let mut next = Vec::new();
        for combo in &combos {
            let mut subst = HashMap::new();
            for (j, m) in combo.iter().enumerate() {
                subst.insert(dirs[j].constant, m.eigenvalue);
            }
            let hint = combo.last().and_then(|m| m.degree);
            let ms = dir_spectrum(p, dir, &subst, hint, &names[i], ctx, &mut side, &mut notes)?;
            for m in ms {
                let mut c = combo.clone();
                c.push(m);
                next.push(c);
            }
        }
        combos = next;
    }
    let mut part = Part {
        uniform: true,
        separation: sep.clone(),
        ..Part::default()
    };
    part.side.extend(side);
    part.notes.extend(notes);
    let mut work: Vec<Vec<DirMember>> = combos;
    let mut guard = 0;
    while let Some(members) = work.pop() {
        guard += 1;
        if guard > 400 {
            return Err(PdeError::SeparationUnsupported(
                "too many exceptional mode groups".into(),
            ));
        }
        match process_group(p, &dirs, sdir.as_ref(), &members, kappa, ctx, &mut part)? {
            GroupOutcome::Done => {}
            GroupOutcome::Split(more) => work.extend(more),
        }
    }
    part.groups.reverse();
    // Gibbs / convergence notes for the data of the eigen directions.
    convergence_notes(p, &dirs, sdir.as_ref(), &mut part, pool)?;
    // Poisson integral for the Dirichlet disk.
    if let Some(alt) = poisson_integral(p, &dirs, sdir.as_ref(), ctx)? {
        part.alternatives.push(alt.0);
        part.checks.push(alt.1);
    }
    if !is_zero_expr(p.source, pool) && s.is_some() && time_slot(p) == s {
        part.methods.push("duhamel");
    }
    Ok(part)
}

enum GroupOutcome {
    Done,
    Split(Vec<Vec<DirMember>>),
}

/// The projection weights and integration ranges of a group.
fn ranges(p: &Prob, dirs: &[Dir]) -> Vec<(ExprId, ExprId, ExprId)> {
    dirs.iter()
        .map(|d| {
            let c = p.coords[d.slot];
            (c.var, c.lo, c.hi)
        })
        .collect()
}

/// Degree in `cos θ` of `e` (with `sin²θ = 1 − cos²θ`), when `e` is a
/// polynomial in `cos θ`.
fn cos_degree(e: ExprId, th: ExprId, pool: &ExprPool) -> Option<i64> {
    let xs = pool.symbol("__cosdeg_X", Domain::Real);
    let c = pool.func("cos", vec![th]);
    let s = pool.func("sin", vec![th]);
    let e = super::separate::expanded(e, pool);
    let mut map = HashMap::new();
    map.insert(c, xs);
    for k in 1..=8 {
        map.insert(
            pool.pow(s, pool.integer(2 * k)),
            pool.pow(
                sub(pool.integer(1), pool.pow(xs, pool.integer(2)), pool),
                pool.integer(k),
            ),
        );
    }
    let e2 = simp(super::separate::replace_nodes(e, &map, pool), pool);
    if mentions_var(e2, th, pool) {
        return None;
    }
    let mut cur = e2;
    for k in 0..=16 {
        if is_zero_expr(cur, pool) {
            return Some(k - 1);
        }
        cur = simp(d(cur, xs, pool).ok()?, pool);
    }
    None
}

/// Bookkeeping of the projections of one group.
struct Acc {
    closed: bool,
    not_closed: Vec<ExprId>,
    checks: Vec<SeriesCheck>,
    gi: usize,
}

impl Acc {
    #[allow(clippy::too_many_arguments)]
    fn track(
        &mut self,
        name: &str,
        h: ExprId,
        proj: &dyn Fn(ExprId, ExprId, ExprId) -> (ExprId, bool),
        phi: ExprId,
        nrm: ExprId,
        coefs: &mut Vec<(String, ExprId)>,
        pool: &ExprPool,
    ) -> ExprId {
        let (c, closed) = proj(h, phi, nrm);
        if !closed {
            self.not_closed.push(h);
        }
        self.closed &= closed;
        self.checks.push(SeriesCheck {
            what: "coefficient",
            group: self.gi,
            // An unevaluated coefficient *is* the projection integral: it
            // holds by definition.
            status: ZeroStatus::ProvedZero,
            method: if closed {
                "ftc"
            } else {
                "projection_definition"
            },
            detail: format!("{name}: {}", pool.display(c)),
        });
        coefs.push((name.to_string(), c));
        c
    }
}

#[allow(clippy::too_many_arguments)]
fn process_group(
    p: &Prob,
    dirs: &[Dir],
    sdir: Option<&SDir>,
    members: &[DirMember],
    kappa: ExprId,
    ctx: &mut Ctx<'_>,
    part: &mut Part,
) -> Result<GroupOutcome, PdeError> {
    let pool = ctx.pool;
    let gi = part.groups.len();
    let mut eig_subst = HashMap::new();
    let mut eigenvalues = Vec::new();
    for (dir, m) in dirs.iter().zip(members) {
        eig_subst.insert(dir.constant, m.eigenvalue);
        eigenvalues.push((dir.constant, m.eigenvalue));
    }
    let indices: Vec<(ExprId, ExprId)> = members
        .iter()
        .filter_map(|m| m.index.map(|i| (i, m.first)))
        .collect();
    let uppers: Vec<ExprId> = indices
        .iter()
        .map(|&(idx, _)| {
            members
                .iter()
                .find(|m| m.index == Some(idx))
                .and_then(|m| m.last)
                .map(|l| pool.integer(l))
                .unwrap_or(pool.pos_infinity())
        })
        .collect();
    let positive: Vec<ExprId> = indices
        .iter()
        .filter(|&&(_, f)| {
            pool.with(f, |dd| match dd {
                ExprData::Integer(k) => k.0 >= 1,
                _ => true,
            })
        })
        .map(|&(i, _)| i)
        .collect();
    let rng = ranges(p, dirs);
    let weight = simp(pool.mul(members.iter().map(|m| m.weight).collect()), pool);
    // Variants: the product over the members' eigenfunctions.
    let mut variants: Vec<(ExprId, Option<ExprId>)> =
        vec![(pool.integer(1), Some(pool.integer(1)))];
    let mut var_idx: Vec<Vec<usize>> = vec![vec![]];
    for m in members {
        let mut next = Vec::new();
        let mut next_idx = Vec::new();
        for ((phi, nrm), idx) in variants.iter().zip(&var_idx) {
            for (k, (&f, n)) in m.eigenfunctions.iter().zip(&m.norms).enumerate() {
                let nn = match (nrm, n) {
                    (Some(a), Some(b)) => Some(simp(pool.mul(vec![*a, *b]), pool)),
                    _ => None,
                };
                next.push((simp(pool.mul(vec![*phi, f]), pool), nn));
                let mut ii = idx.clone();
                ii.push(k);
                next_idx.push(ii);
            }
        }
        variants = next;
        var_idx = next_idx;
    }
    let zero_map: HashMap<ExprId, ExprId> = members
        .iter()
        .flat_map(|m| m.zero_nodes.iter().map(|&z| (z, pool.integer(0))))
        .collect();
    let at_roots = |e: ExprId| -> ExprId {
        let e = super::project::tidy(
            super::project::reduce_integer_trig(simp(e, pool), pool),
            pool,
        );
        if zero_map.is_empty() {
            return e;
        }
        let mut e = simp(super::separate::replace_nodes(e, &zero_map, pool), pool);
        for &z in zero_map.keys() {
            e = super::project::bessel_at_root(e, z, pool);
        }
        e
    };
    let proj = |h: ExprId, phi: ExprId, nrm: ExprId| -> (ExprId, bool) {
        if is_zero_expr(h, pool) {
            return (pool.integer(0), true);
        }
        if rng.is_empty() {
            return (simp(h, pool), true);
        }
        let (v, closed) = iterated(pool.mul(vec![weight, h, phi]), &rng, pool);
        let c = at_roots(div(at_roots(v), at_roots(nrm), pool));
        let c = if closed && !super::project::contains_root_family(c, pool) {
            simp(super::separate::expanded(c, pool), pool)
        } else {
            c
        };
        (c, closed)
    };
    // The s-direction's amplitude equation with the eigenvalues in.
    let mut scale = pool.integer(1);
    let mode = match sdir {
        Some(sd) => {
            let ode = simp(subs(sd.ode, &eig_subst, pool), pool);
            let m = mode_ode(ode, sd.factor, pool)?;
            // L[TΦ] = λ·Φ·M_chain[T]: read λ off the highest derivative.
            let phi0 = variants[0].0;
            scale = mode_scale(p, sd.factor, phi0, m, pool)?;
            Some(ModeOde {
                order: m.order,
                a2: simp(pool.mul(vec![scale, m.a2]), pool),
                a1: simp(pool.mul(vec![scale, m.a1]), pool),
                a0: simp(pool.mul(vec![scale, m.a0]), pool),
            })
        }
        None => None,
    };
    let mut checks = Vec::new();
    let mut amplitudes = Vec::new();
    let mut coefficients = Vec::new();
    let mut norms = Vec::new();
    let mut phis = Vec::new();
    let mut terms = Vec::new();
    super::project::take_denominators();
    let mut acc = Acc {
        closed: true,
        not_closed: Vec::new(),
        checks: Vec::new(),
        gi,
    };
    for (phi, nrm) in &variants {
        let nrm = match nrm {
            Some(n) => *n,
            None => {
                let (v, _c) = iterated(
                    pool.mul(vec![weight, pool.pow(*phi, pool.integer(2))]),
                    &rng,
                    pool,
                );
                simp(v, pool)
            }
        };
        let phi = *phi;
        let mut coefs: Vec<(String, ExprId)> = Vec::new();
        let amp = match (sdir, mode) {
            (Some(sd), Some(mo)) => {
                let s = p.vars[sd.slot];
                let src_t = simp(neg(p.source, pool), pool);
                let q = if is_zero_expr(src_t, pool) {
                    pool.integer(0)
                } else {
                    acc.track("source", src_t, &proj, phi, nrm, &mut coefs, pool)
                };
                let data = if p.coords[sd.slot].kind == CoordKind::Time {
                    if sd.ics.is_empty() {
                        return Err(PdeError::SeparationMissingData(format!(
                            "no initial condition at {} = {}",
                            pool.display(s),
                            pool.display(p.coords[sd.slot].lo)
                        )));
                    }
                    if mo.order == 2 && !sd.ics.iter().any(|ic| ic.0 == 1) {
                        // u_t(x, 0) not given: refuse rather than assume 0.
                        return Err(PdeError::SeparationMissingData(format!(
                            "the equation is of second order in {}: give u and its first \
                             derivative at {} = {}",
                            pool.display(s),
                            pool.display(s),
                            pool.display(p.coords[sd.slot].lo)
                        )));
                    }
                    let mut values = vec![pool.integer(0); 2];
                    for &(k, c, h) in &sd.ics {
                        let name = if k == 0 { "ic" } else { "ic1" };
                        let cv = acc.track(
                            name,
                            simp(div(h, c, pool), pool),
                            &proj,
                            phi,
                            nrm,
                            &mut coefs,
                            pool,
                        );
                        if k < 2 {
                            values[k] = cv;
                        }
                    }
                    values.truncate(mo.order as usize);
                    SData::Initial {
                        t0: p.coords[sd.slot].lo,
                        values,
                    }
                } else {
                    let co = p.coords[sd.slot];
                    let mut ends = Vec::new();
                    for (f, name) in [(&sd.left, "left"), (&sd.right, "right")] {
                        ends.push(match f {
                            None => SEnd::Bounded,
                            Some(f) => {
                                let zero = pool.integer(0);
                                SEnd::Face {
                                    at: f.at,
                                    alpha: f.coeffs.first().copied().unwrap_or(zero),
                                    beta: f.coeffs.get(1).copied().unwrap_or(zero),
                                    data: acc
                                        .track(name, f.data, &proj, phi, nrm, &mut coefs, pool),
                                }
                            }
                        });
                    }
                    let right = ends.pop().expect("two ends");
                    let left = ends.pop().expect("two ends");
                    if matches!(left, SEnd::Bounded) && !co.sing_lo {
                        return Err(PdeError::SeparationMissingData(format!(
                            "no condition at {} = {}",
                            pool.display(s),
                            pool.display(co.lo)
                        )));
                    }
                    SData::TwoPoint {
                        s0: co.lo,
                        s1: co.hi,
                        left,
                        right,
                    }
                };
                let sol = solve_mode(mo, s, &data, q, &positive, &mut ctx.taken, pool)?;
                part.side.extend(sol.side_conditions.iter().copied());
                part.notes.extend(sol.notes.iter().cloned());
                part.constants.extend(sol.constants.iter().copied());
                acc.closed &= sol.closed;
                // Exact checks: the amplitude ODE and its data.
                let with_int = has_integral(sol.value, pool);
                let (st, how) = match amplitude_residual(mo, s, sol.value, q, pool) {
                    Some(r) => residual_status(r, with_int, pool),
                    None => (ZeroStatus::Unknown, "unevaluated"),
                };
                acc.checks.push(SeriesCheck {
                    what: "amplitude",
                    group: gi,
                    status: st,
                    method: how,
                    detail: format!("{} solves its amplitude equation", pool.display(sol.value)),
                });
                match amplitude_data_residuals(&data, s, sol.value, pool) {
                    Some(rs) => {
                        for (what, r) in rs {
                            let (st, how) = residual_status(r, with_int, pool);
                            acc.checks.push(SeriesCheck {
                                what: "amplitude_data",
                                group: gi,
                                status: st,
                                method: how,
                                detail: what,
                            });
                        }
                    }
                    None => acc.checks.push(SeriesCheck {
                        what: "amplitude_data",
                        group: gi,
                        status: ZeroStatus::Unknown,
                        method: "unevaluated",
                        detail: "the data of the amplitude".into(),
                    }),
                }
                sol.value
            }
            _ => {
                // No s direction: L[Φ] = −κΦ, a = proj(source)/κ.
                let kap = simp(subs(kappa, &eig_subst, pool), pool);
                let sproj = acc.track("source", p.source, &proj, phi, nrm, &mut coefs, pool);
                if is_zero_expr(kap, pool) {
                    if !is_zero_expr(sproj, pool) {
                        let st = crate::simplify::zero::zero_test(sproj, pool, None).status;
                        if st == ZeroStatus::ProvedNonzero {
                            return Err(PdeError::SeparationCompatibility(format!(
                                "the source has a nonzero component {} on the zero mode: \
                                 the problem has no solution",
                                pool.display(sproj)
                            )));
                        }
                        part.side
                            .push(pool.predicate(PredicateKind::Eq, vec![sproj, pool.integer(0)]));
                    }
                    let c = pool.symbol(
                        super::separate::fresh_name("C1", &mut ctx.taken),
                        Domain::Real,
                    );
                    part.constants.push(c);
                    part.notes.push(
                        "the solution is determined up to an additive constant (zero mode)".into(),
                    );
                    c
                } else {
                    simp(div(sproj, kap, pool), pool)
                }
            }
        };
        let amp = at_roots(amp);
        let term = simp(pool.mul(vec![amp, phi]), pool);
        let nrm = at_roots(nrm);
        amplitudes.push(amp);
        coefficients.push(coefs);
        norms.push(nrm);
        phis.push(phi);
        terms.push(term);
    }
    let closed_all = acc.closed;
    let not_closed_data = acc.not_closed;
    checks.extend(acc.checks);
    let total = simp(pool.add(terms.clone()), pool);
    // Denominators of the antiderivatives: the FTC values are not valid
    // where they vanish.
    let denoms = super::project::take_denominators();
    let probe = pool.add(
        std::iter::once(total)
            .chain(denoms.iter().map(|&b| pool.pow(b, pool.integer(-1))))
            .collect(),
    );
    // Exceptional indices (closed-form coefficients whose denominator
    // vanishes at an integer index).
    for (k, &(idx, first)) in indices.iter().enumerate() {
        let Some(f0) = pool.with(first, |dd| match dd {
            ExprData::Integer(n) => n.0.to_i64(),
            _ => None,
        }) else {
            continue;
        };
        let others: Vec<ExprId> = indices
            .iter()
            .enumerate()
            .filter(|(j, _)| *j != k)
            .map(|(_, &(i, _))| i)
            .collect();
        let last = members
            .iter()
            .find(|m| m.index == Some(idx))
            .and_then(|m| m.last);
        let (mut exc, sc) = exceptional_indices(probe, idx, f0, &others, pool);
        if let Some(l) = last {
            exc.retain(|&e| e <= l);
        }
        for c in sc {
            let base = pool.with(c, |dd| match dd {
                ExprData::Predicate { args, .. } => args.first().copied(),
                _ => None,
            });
            if let Some(b) = base {
                if matches!(
                    super::modes::sign_hint(b, &positive, pool),
                    super::super::sturm_liouville::Sgn::Pos
                        | super::super::sturm_liouville::Sgn::Neg
                ) {
                    continue;
                }
            }
            if let Some(b) = base {
                if indices
                    .iter()
                    .filter(|&&(i, _)| mentions_var(b, i, pool))
                    .count()
                    > 1
                    && vanishes_on_grid(b, &indices, pool)
                {
                    return Err(PdeError::SeparationUnsupported(format!(
                        "a coefficient is singular where {} = 0, which happens for some index \
                         values (a resonance); not expanded",
                        pool.display(b)
                    )));
                }
            }
            part.side.push(c);
        }
        if !exc.is_empty() {
            if exc.len() > 40 {
                return Err(PdeError::SeparationUnsupported(format!(
                    "{} exceptional values of {}: too many to expand separately",
                    exc.len(),
                    pool.display(idx)
                )));
            }
            // Concrete members at the exceptional values, generic ranges
            // between them.
            let ranged = |a: i64, b: Option<i64>| -> Vec<DirMember> {
                let mut g = members.to_vec();
                for mm in &mut g {
                    if mm.index == Some(idx) {
                        mm.first = pool.integer(a);
                        mm.last = b;
                    }
                }
                g
            };
            let mut out = Vec::new();
            let mut cur = f0;
            for &e in &exc {
                if e > cur {
                    out.push(ranged(cur, Some(e - 1)));
                }
                let m = HashMap::from([(idx, pool.integer(e))]);
                out.push(
                    members
                        .iter()
                        .map(|mm| mm.subs_all(&m, pool))
                        .collect::<Vec<_>>(),
                );
                cur = e + 1;
            }
            if last.map_or(true, |l| cur <= l) {
                out.push(ranged(cur, last));
            }
            return Ok(GroupOutcome::Split(out));
        }
    }
    // Finite Legendre expansions: data polynomial in cos θ.
    if !closed_all {
        for (dir, m) in dirs.iter().zip(members) {
            if m.kind != "legendre" || m.index.is_none() {
                continue;
            }
            let Some(f0) = pool.with(m.first, |dd| match dd {
                ExprData::Integer(n) => n.0.to_i64(),
                _ => None,
            }) else {
                continue;
            };
            let th = p.coords[dir.slot].var;
            let mut deg: Option<i64> = Some(-1);
            for &h in &not_closed_data {
                let hh = if f0 > 0 {
                    div(
                        h,
                        pool.pow(pool.func("sin", vec![th]), pool.integer(f0)),
                        pool,
                    )
                } else {
                    h
                };
                match (deg, cos_degree(hh, th, pool)) {
                    (Some(a), Some(b)) => deg = Some(a.max(b + f0)),
                    _ => deg = None,
                }
            }
            if let Some(dg) = deg {
                let idx = m.index.expect("checked");
                let mut out = Vec::new();
                for kk in f0..=dg.max(f0) {
                    let mm = HashMap::from([(idx, pool.integer(kk))]);
                    out.push(
                        members
                            .iter()
                            .map(|x| x.subs_all(&mm, pool))
                            .collect::<Vec<_>>(),
                    );
                }
                part.notes.push(format!(
                    "the data is a polynomial of degree {dg} in cos θ, so its Legendre \
                     coefficients of degree above {dg} vanish (orthogonality): the expansion \
                     in {} is finite",
                    pool.display(idx)
                ));
                return Ok(GroupOutcome::Split(out));
            }
        }
    }
    // The separation identity: every mode satisfies the PDE.
    check_separation(
        p,
        dirs,
        sdir,
        members,
        &var_idx,
        kappa,
        scale,
        gi,
        &mut checks,
        pool,
    )?;
    for (dir, m) in dirs.iter().zip(members) {
        checks.push(SeriesCheck {
            what: "eigen_bc",
            group: gi,
            status: match m.status {
                VerificationStatus::ExactlyVerified => ZeroStatus::ProvedZero,
                VerificationStatus::Refuted => ZeroStatus::ProvedNonzero,
                _ => ZeroStatus::Unknown,
            },
            method: if m.method == "members" {
                "members"
            } else {
                "sturm_liouville"
            },
            detail: format!(
                "eigenfunctions in {} ({})",
                pool.display(p.vars[dir.slot]),
                m.method
            ),
        });
    }
    if checks.iter().any(|c| c.status == ZeroStatus::ProvedNonzero) {
        let bad: Vec<String> = checks
            .iter()
            .filter(|c| c.status == ZeroStatus::ProvedNonzero)
            .map(|c| format!("{}: {}", c.what, c.detail))
            .collect();
        return Err(PdeError::SeparationVerificationFailed(bad.join("; ")));
    }
    // Numeric projection check of the first members (also for a group whose
    // terms all vanish: a wrong zero is caught here).
    let mut proj_checks = Vec::new();
    if ctx.opts.check_members > 0 {
        let pc = super::numeric::projection_check(
            &indices,
            &uppers,
            &rng,
            weight,
            &phis,
            &norms,
            &coefficients,
            &data_of(p, sdir, pool),
            ctx.opts.check_members,
            pool,
        );
        if let Some(pc) = pc {
            for c in pc {
                if c.status == VerificationStatus::Refuted {
                    return Err(PdeError::SeparationProjectionRefuted(c.evidence.join("; ")));
                }
                proj_checks.push(c);
            }
        }
    }
    // Drop a group whose every term vanishes.
    if is_zero_expr(total, pool) {
        return Ok(GroupOutcome::Done);
    }
    part.projection.extend(proj_checks);
    // Assemble the nested sum (innermost: the last index).
    let mut value = total;
    for (&(idx, first), &hi) in indices.iter().zip(&uppers).rev() {
        value = super::super::sum::sum(value, idx, first, hi, pool)?;
    }
    let statuses: Vec<ZeroStatus> = checks.iter().map(|c| c.status).collect();
    let status = if statuses.iter().all(|s| *s == ZeroStatus::ProvedZero) {
        VerificationStatus::ExactlyVerified
    } else if statuses
        .iter()
        .all(|s| matches!(s, ZeroStatus::ProvedZero | ZeroStatus::ProbablyZero))
    {
        VerificationStatus::NumericallyChecked
    } else {
        VerificationStatus::Unverified
    };
    part.checks.extend(checks);
    part.groups.push(ModeGroup {
        indices,
        upper: uppers,
        eigenvalues,
        eigenfunctions: phis,
        weight,
        norms_squared: norms,
        coefficients,
        amplitudes,
        term: total,
        value,
        closed: closed_all,
        status,
    });
    part.value.push(value);
    Ok(GroupOutcome::Done)
}

/// Whether `b` vanishes (numerically, with positive sample parameters) on
/// the grid of the first index values; `RootOfFamily` members are evaluated
/// by their certified enclosures.
fn vanishes_on_grid(b: ExprId, indices: &[(ExprId, ExprId)], pool: &ExprPool) -> bool {
    let params: Vec<ExprId> = super::super::jet::symbols_of(b, pool)
        .into_iter()
        .filter(|s| !indices.iter().any(|(i, _)| i == s))
        .filter(|&s| !super::super::root_family::root_family_parts(s, pool).is_some())
        .collect();
    let vals = [1.3, 0.7, 1.9, 1.1, 2.3];
    let mut base: HashMap<ExprId, ExprId> = HashMap::new();
    for (k, &p) in params.iter().enumerate() {
        let named = pool.with(p, |dd| match dd {
            ExprData::Symbol { name, .. } => {
                crate::eval::symbols::constant_value_f64(name).is_some()
            }
            _ => false,
        });
        if !named {
            base.insert(p, pool.float(vals[k % 5], 53));
        }
    }
    let firsts: Vec<i64> = indices
        .iter()
        .map(|&(_, f)| {
            pool.with(f, |dd| match dd {
                ExprData::Integer(n) => n.0.to_i64(),
                _ => None,
            })
            .unwrap_or(0)
        })
        .collect();
    let n = indices.len().min(3);
    let span = if n <= 2 { 12 } else { 5 };
    let mut idx = vec![0i64; n];
    loop {
        let mut m = base.clone();
        for k in 0..n {
            m.insert(indices[k].0, pool.integer(firsts[k] + idx[k]));
        }
        let e = simp(subs(b, &m, pool), pool);
        let e = super::numeric::numeric_root_families(e, pool).unwrap_or(e);
        match crate::jit::eval_interp_real(e, &HashMap::new(), pool) {
            Ok(v) if v.abs() < 1e-10 => return true,
            Err(_) => return true,
            _ => {}
        }
        let mut k = 0;
        loop {
            if k == n {
                return false;
            }
            idx[k] += 1;
            if idx[k] < span {
                break;
            }
            idx[k] = 0;
            k += 1;
        }
    }
}

/// `(name, data)` pairs the coefficients project, matching the names used in
/// [`ModeGroup::coefficients`] (initial data divided by its coefficient; the
/// source with the sign the amplitude equation uses).
fn data_of(p: &Prob, sdir: Option<&SDir>, pool: &ExprPool) -> Vec<(String, ExprId)> {
    let mut out = Vec::new();
    match sdir {
        Some(sd) if p.coords[sd.slot].kind == CoordKind::Time => {
            for &(k, c, h) in &sd.ics {
                out.push((
                    if k == 0 { "ic" } else { "ic1" }.to_string(),
                    simp(div(h, c, pool), pool),
                ));
            }
        }
        Some(sd) => {
            if let Some(f) = &sd.left {
                out.push(("left".into(), f.data));
            }
            if let Some(f) = &sd.right {
                out.push(("right".into(), f.data));
            }
        }
        None => {}
    }
    if !is_zero_expr(p.source, pool) {
        let src = if sdir.is_some() {
            simp(neg(p.source, pool), pool)
        } else {
            p.source
        };
        out.push(("source".into(), src));
    }
    out
}

/// `λ` with `L[TΦ] = λ·Φ·M_chain[T]`: the coefficient of the highest
/// derivative of `T` in `L[TΦ]` over `Φ` times its coefficient in `M_chain`.
fn mode_scale(
    p: &Prob,
    factor: ExprId,
    phi: ExprId,
    m: ModeOde,
    pool: &ExprPool,
) -> Result<ExprId, PdeError> {
    let lt = replace_function(p.lhom, p.u, pool.mul(vec![factor, phi]), pool)
        .map_err(|e| PdeError::SeparationUnsupported(e.to_string()))?;
    let (k, ck) = if m.order == 2 {
        (2u32, m.a2)
    } else {
        (1u32, m.a1)
    };
    let node = crate::diff::undefined::build(factor, &[k], pool);
    let z = pool.symbol("__scale_z", Domain::Real);
    let e = super::separate::replace_nodes(lt, &HashMap::from([(node, z)]), pool);
    let c = simp(d(e, z, pool)?, pool);
    Ok(simp(div(c, pool.mul(vec![phi, ck]), pool), pool))
}

/// The status of a residual; with unevaluated integrals, after merging them
/// (`Σ cᵢ∫fᵢ = ∫Σcᵢfᵢ`) and proving the merged integrand zero.
fn residual_status(r: ExprId, with_int: bool, pool: &ExprPool) -> (ZeroStatus, &'static str) {
    if !with_int && !has_integral(r, pool) {
        return (super::project::zero_status(r, pool), "zero_test");
    }
    let m = super::project::merge_integrals(r, pool);
    let m = super::project::opaque_integrals(m, pool);
    (super::project::zero_status(m, pool), "leibniz")
}

/// `a₂T'' + a₁T' + a₀T − q` (Leibniz's rule through unevaluated integrals).
fn amplitude_residual(
    m: ModeOde,
    s: ExprId,
    t: ExprId,
    q: ExprId,
    pool: &ExprPool,
) -> Option<ExprId> {
    let t1 = super::project::diff_with_integrals(t, s, pool)?;
    let t2 = super::project::diff_with_integrals(t1, s, pool)?;
    Some(pool.add(vec![
        pool.mul(vec![m.a2, t2]),
        pool.mul(vec![m.a1, t1]),
        pool.mul(vec![m.a0, t]),
        neg(q, pool),
    ]))
}

fn amplitude_data_residuals(
    data: &SData,
    s: ExprId,
    t: ExprId,
    pool: &ExprPool,
) -> Option<Vec<(String, ExprId)>> {
    let mut out = Vec::new();
    match data {
        SData::Initial { t0, values } => {
            let mut cur = t;
            for (k, &v) in values.iter().enumerate() {
                if k > 0 {
                    cur = super::project::diff_with_integrals(cur, s, pool)?;
                }
                out.push((
                    format!("initial value of order {k}"),
                    sub(subs1(cur, s, *t0, pool), v, pool),
                ));
            }
        }
        SData::TwoPoint { left, right, .. } => {
            for (name, e) in [("left", left), ("right", right)] {
                if let SEnd::Face {
                    at,
                    alpha,
                    beta,
                    data,
                } = e
                {
                    let v = pool.add(vec![
                        pool.mul(vec![*alpha, subs1(t, s, *at, pool)]),
                        pool.mul(vec![
                            *beta,
                            subs1(
                                super::project::diff_with_integrals(t, s, pool)?,
                                s,
                                *at,
                                pool,
                            ),
                        ]),
                    ]);
                    out.push((
                        format!("{name} condition of the amplitude"),
                        sub(v, *data, pool),
                    ));
                }
            }
        }
    }
    Some(out)
}

/// Every mode satisfies the PDE: with `T(s)` an undefined function,
/// `L[T Φ] − Φ·M[T] ≡ 0` (or `L[Φ] + κΦ ≡ 0` without an `s` direction),
/// identically in the indices — for a `RootOfFamily` member identically in
/// its frequency `ν`.
#[allow(clippy::too_many_arguments)]
fn check_separation(
    p: &Prob,
    dirs: &[Dir],
    sdir: Option<&SDir>,
    members: &[DirMember],
    var_idx: &[Vec<usize>],
    kappa: ExprId,
    scale: ExprId,
    gi: usize,
    checks: &mut Vec<SeriesCheck>,
    pool: &ExprPool,
) -> Result<(), PdeError> {
    // ν-forms.
    let mut eig = HashMap::new();
    let mut funs: Vec<Vec<ExprId>> = Vec::new();
    let mut by_nu = false;
    for (dir, m) in dirs.iter().zip(members) {
        match &m.nu {
            Some((_v, _f, lam, phis)) => {
                eig.insert(dir.constant, *lam);
                funs.push(phis.clone());
                by_nu = true;
            }
            None => {
                eig.insert(dir.constant, m.eigenvalue);
                funs.push(m.eigenfunctions.clone());
            }
        }
    }
    // Later directions' expressions may mention earlier frequencies through
    // RootOfFamily nodes; with ν-forms, earlier RootOfFamily values inside
    // later members are left as they are (they are parameters there).
    for idx in var_idx {
        let phi = simp(
            pool.mul(idx.iter().enumerate().map(|(j, &k)| funs[j][k]).collect()),
            pool,
        );
        let r = match sdir {
            Some(sd) => {
                let ode = simp(subs(sd.ode, &eig, pool), pool);
                let lhs = replace_function(p.lhom, p.u, pool.mul(vec![sd.factor, phi]), pool)
                    .map_err(|e| PdeError::SeparationUnsupported(e.to_string()))?;
                sub(lhs, pool.mul(vec![scale, phi, ode]), pool)
            }
            None => {
                let kap = simp(subs(kappa, &eig, pool), pool);
                let lhs = replace_function(p.lhom, p.u, phi, pool)
                    .map_err(|e| PdeError::SeparationUnsupported(e.to_string()))?;
                pool.add(vec![lhs, pool.mul(vec![kap, phi])])
            }
        };
        let st = super::project::zero_status(r, pool);
        checks.push(SeriesCheck {
            what: "separation",
            group: gi,
            status: st,
            method: if by_nu {
                "zero_test_in_frequency"
            } else {
                "zero_test"
            },
            detail: format!(
                "the mode with eigenfunction {} satisfies the equation",
                pool.display(phi)
            ),
        });
    }
    Ok(())
}

/// Convergence class and Gibbs notes: does each datum satisfy the
/// homogeneous conditions of the eigen directions?
fn convergence_notes(
    p: &Prob,
    dirs: &[Dir],
    sdir: Option<&SDir>,
    part: &mut Part,
    pool: &ExprPool,
) -> Result<(), PdeError> {
    let data = data_of(p, sdir, pool);
    let mut uniform = true;
    for (name, h) in &data {
        if name == "ic1" || name == "source" {
            continue;
        }
        for dir in dirs {
            let co = p.coords[dir.slot];
            let x = co.var;
            let mut bad = Vec::new();
            let mut definite_bad = false;
            match co.kind {
                CoordKind::Periodic => {
                    // The periodic extension must be continuous.
                    let r = sub(subs1(*h, x, co.hi, pool), subs1(*h, x, co.lo, pool), pool);
                    if !is_zero_expr(r, pool) {
                        definite_bad |= proved_nonzero(r, pool);
                        bad.push(format!(
                            "{} = {} ~ {}",
                            pool.display(x),
                            pool.display(co.lo),
                            pool.display(co.hi)
                        ));
                    }
                }
                _ => {
                    for hi in [false, true] {
                        let Some(f) = face_at(p, dir.slot, hi, pool) else {
                            continue;
                        };
                        // Only a Dirichlet end constrains uniform convergence
                        // (the odd extension must be continuous); Neumann and
                        // Robin expansions of continuous data converge
                        // uniformly.
                        if f.coeffs.get(1).is_some_and(|&b| !is_zero_expr(b, pool)) {
                            continue;
                        }
                        let mut terms = Vec::new();
                        let mut cur = *h;
                        for (k, &c) in f.coeffs.iter().enumerate() {
                            if k > 0 {
                                cur = d(cur, x, pool)?;
                            }
                            terms.push(pool.mul(vec![c, subs1(cur, x, f.at, pool)]));
                        }
                        let r = simp(pool.add(terms), pool);
                        if !is_zero_expr(r, pool) {
                            definite_bad |= proved_nonzero(r, pool);
                            bad.push(format!("{} = {}", pool.display(x), pool.display(f.at)));
                        }
                    }
                }
            }
            if !bad.is_empty() && !definite_bad {
                uniform = false;
                part.notes.push(format!(
                    "the {} data meets the boundary condition(s) at {} only for some data: the \
                     expansion converges uniformly if it does, in the mean (L²) with Gibbs \
                     oscillations near there otherwise",
                    if name.starts_with("ic") {
                        "initial"
                    } else {
                        "boundary"
                    },
                    bad.join(", ")
                ));
            } else if !bad.is_empty() {
                uniform = false;
                part.notes.push(format!(
                    "the {} data does not satisfy the boundary condition(s) at {}: its expansion \
                     converges in the mean (L²), not uniformly, and partial sums overshoot near \
                     there (Gibbs phenomenon, about 9% of the jump)",
                    if name.starts_with("ic") {
                        "initial"
                    } else {
                        "boundary"
                    },
                    bad.join(", ")
                ));
            }
        }
    }
    if let Some(sd) = sdir {
        if p.coords[sd.slot].kind == CoordKind::Time {
            part.notes.push(format!(
                "the series is the eigenfunction expansion of the solution; it meets the initial \
                 data in the sense of that expansion (L² convergence at {} = {}), never claimed \
                 pointwise for a truncation",
                pool.display(p.vars[sd.slot]),
                pool.display(p.coords[sd.slot].lo)
            ));
        }
    }
    part.uniform &= uniform;
    Ok(())
}

fn proved_nonzero(e: ExprId, pool: &ExprPool) -> bool {
    crate::simplify::zero::zero_test(simp(e, pool), pool, None).status == ZeroStatus::ProvedNonzero
}

/// The Poisson integral formula for the Dirichlet problem of Laplace's
/// equation on a disk centred at the origin.
#[allow(clippy::type_complexity)]
fn poisson_integral(
    p: &Prob,
    dirs: &[Dir],
    sdir: Option<&SDir>,
    ctx: &mut Ctx<'_>,
) -> Result<Option<((String, ExprId), SeriesCheck)>, PdeError> {
    let pool = ctx.pool;
    let Some(sd) = sdir else { return Ok(None) };
    if p.system != Some(CoordinateSystem::Polar) || dirs.len() != 1 || !is_zero_expr(p.source, pool)
    {
        return Ok(None);
    }
    let rc = p.coords[sd.slot];
    let tc = p.coords[dirs[0].slot];
    if rc.role != Role::Radius || !rc.sing_lo || tc.kind != CoordKind::Periodic {
        return Ok(None);
    }
    let Some(face) = &sd.right else {
        return Ok(None);
    };
    if face.coeffs.len() != 1 {
        return Ok(None);
    }
    // Laplace's equation: L = c·Δ.
    let (r, th) = (rc.var, tc.var);
    let lap = super::coords::laplacian(p.u, CoordinateSystem::Polar, &[Some(r), Some(th)], pool)?;
    let ratio = simp(div(p.lhom, lap, pool), pool);
    if mentions_var(ratio, r, pool)
        || mentions_var(ratio, th, pool)
        || super::super::problem::mentions_unknown(ratio, std::slice::from_ref(&p.name), pool)
    {
        return Ok(None);
    }
    let big_r = rc.hi;
    let phi = super::super::sturm_liouville::fresh_symbol(
        "phi",
        Domain::Real,
        &[face.data, r, th, big_r],
        pool,
    );
    let sq = |e: ExprId| pool.pow(e, pool.integer(2));
    let kern = div(
        sub(sq(big_r), sq(r), pool),
        pool.add(vec![
            sq(big_r),
            neg(
                pool.mul(vec![
                    pool.integer(2),
                    big_r,
                    r,
                    pool.func("cos", vec![sub(th, phi, pool)]),
                ]),
                pool,
            ),
            sq(r),
        ]),
        pool,
    );
    let g = div(subs1(face.data, th, phi, pool), face.coeffs[0], pool);
    let body = pool.mul(vec![
        div(
            pool.integer(1),
            pool.mul(vec![pool.integer(2), pi(pool)]),
            pool,
        ),
        kern,
        g,
    ]);
    let val = pool.func(super::project::INTEGRAL, vec![body, phi, tc.lo, tc.hi]);
    // The kernel is harmonic in (r, θ).
    let lk = replace_function(lap, p.u, kern, pool).unwrap_or(lap);
    let st = super::project::zero_status(lk, pool);
    Ok(Some((
        ("poisson_integral".into(), val),
        SeriesCheck {
            what: "kernel",
            group: usize::MAX,
            status: st,
            method: "zero_test",
            detail: "the Poisson kernel is harmonic in the disk".into(),
        },
    )))
}
