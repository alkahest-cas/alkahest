//! `u = ∫ G f + boundary terms`: the Green's-function representation of a
//! problem, closed wherever `integrate` closes the integrals.

use std::collections::HashMap;

use crate::diff::undefined::{has_undefined_function, view};
use crate::kernel::subs::subs;
use crate::kernel::{Domain, ExprId, ExprPool};
use crate::simplify::zero::ZeroStatus;

use super::super::error::PdeError;
use super::super::jet::JetSpace;
use super::super::problem::{PdeCondition, PdeDomain, PdeProblem};
use super::super::sturm_liouville::{
    replace_node, simp, subs1, EndCondition, SlBoundary, SturmLiouvilleProblem,
};
use super::super::verify::{verify_solution, VerificationStatus};
use super::{
    at, combine, dd, div, fresh, fundamental_solution, image_green_function, mentions, neg,
    ode_green_parts, sub, zstatus, GreenCheck, GreenOperator, ImageRegion, WallCondition,
};

/// Closed form or integral representation.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
#[non_exhaustive]
pub enum GreenSolutionKind {
    /// Every integral closed.
    ClosedForm,
    /// Some integrals are kept as `Integral(…)` nodes.
    IntegralRepresentation,
}

impl GreenSolutionKind {
    /// `"closed_form"` or `"integral_representation"`.
    pub fn as_str(self) -> &'static str {
        match self {
            GreenSolutionKind::ClosedForm => "closed_form",
            GreenSolutionKind::IntegralRepresentation => "integral_representation",
        }
    }
}

/// One integral `∫ kernel · density` of a representation.
#[derive(Clone, Debug, PartialEq, Eq)]
#[non_exhaustive]
pub struct GreenTerm {
    /// `"source"`, `"initial"`, `"initial_velocity"` or `"boundary"`.
    pub role: &'static str,
    /// The kernel (Green's function, Poisson kernel, …) in the problem's
    /// variables and [`Self::vars`].
    pub kernel: ExprId,
    /// The data it integrates.
    pub density: ExprId,
    /// The integration variables, innermost first.
    pub vars: Vec<ExprId>,
    /// Their ranges, innermost first.
    pub bounds: Vec<(ExprId, ExprId)>,
    /// The integral in closed form, when `integrate` closed it.
    pub closed: Option<ExprId>,
}

/// The result of [`green_solve`] — the shape of M1's `PdeSolution` with the
/// representation's terms.
#[derive(Clone, Debug, PartialEq, Eq)]
#[non_exhaustive]
pub struct GreenSolution {
    /// The unknown.
    pub unknown: ExprId,
    /// Closed form or representation.
    pub kind: GreenSolutionKind,
    /// `"green_bvp"`, `"green_heat_line"`, `"green_wave_line"`,
    /// `"green_half_plane"`.
    pub method: String,
    /// `u` (with `Integral(…)` nodes for a representation).
    pub value: ExprId,
    /// The part of [`Self::value`] that is closed.
    pub closed_part: ExprId,
    /// The integrals.
    pub terms: Vec<GreenTerm>,
    /// The Green's function used, `G(x, ξ[, t])`.
    pub green: ExprId,
    /// The source-point variables of [`Self::green`].
    pub source: Vec<ExprId>,
    /// Exactly verified only when every check is a proof.
    pub status: VerificationStatus,
    /// The checks.
    pub checks: Vec<GreenCheck>,
    /// Conditions (predicates).
    pub side_conditions: Vec<ExprId>,
    /// Hypotheses in prose.
    pub notes: Vec<String>,
    /// Derivation steps.
    pub steps: Vec<String>,
}

/// Put the arguments of `exp` and the bases of powers over a common
/// denominator (`cancel`), leaving anything non-rational alone.
fn cancel_args(e: ExprId, vars: &[ExprId], pool: &ExprPool) -> ExprId {
    use crate::kernel::ExprData;
    let c = |a: ExprId| crate::poly::cancel::cancel(a, vars.to_vec(), pool).unwrap_or(a);
    match pool.get(e) {
        ExprData::Add(xs) => pool.add(xs.iter().map(|&a| cancel_args(a, vars, pool)).collect()),
        ExprData::Mul(xs) => pool.mul(xs.iter().map(|&a| cancel_args(a, vars, pool)).collect()),
        ExprData::Pow { base, exp } => pool.pow(c(cancel_args(base, vars, pool)), exp),
        ExprData::Func { name, args } => pool.func(
            name,
            args.iter()
                .map(|&a| c(cancel_args(a, vars, pool)))
                .collect(),
        ),
        _ => e,
    }
}

/// The shorter of `e` and its expanded form (an identity).
fn tidy(e: ExprId, pool: &ExprPool) -> ExprId {
    if has_undefined_function(e, pool) {
        return e;
    }
    let x = simp(super::expand_poly(e, pool), pool);
    if pool.display(x).to_string().len() < pool.display(e).to_string().len() {
        x
    } else {
        e
    }
}

fn integral(body: ExprId, s: ExprId, lo: ExprId, hi: ExprId, pool: &ExprPool) -> ExprId {
    pool.func("Integral", vec![body, s, lo, hi])
}

/// `∫_lo^hi body ds` closed, when `integrate` closes it and no undefined
/// function is in the integrand.
fn try_integrate(
    body: ExprId,
    s: ExprId,
    lo: ExprId,
    hi: ExprId,
    pool: &ExprPool,
) -> Option<ExprId> {
    if has_undefined_function(body, pool) {
        return None;
    }
    crate::integrate::integrate_definite(body, s, lo, hi, pool)
        .ok()
        .map(|r| simp(r.value, pool))
}

/// `Σ_α c_α u_α = F` for one linear equation: the coefficient of every jet
/// of order ≤ 2 and the right-hand side.
struct LinearForm {
    coeff: HashMap<Vec<u32>, ExprId>,
    rhs: ExprId,
}

fn linear_form(problem: &PdeProblem, pool: &ExprPool) -> Result<LinearForm, PdeError> {
    if problem.equations().len() != 1 || problem.unknowns().len() != 1 {
        return Err(PdeError::GreenProblemUnsupported(
            "green_solve takes one equation in one unknown".into(),
        ));
    }
    let indep = problem.independent().to_vec();
    let u = problem.unknowns()[0];
    let js = JetSpace::new(&indep, &[u], 2, pool).map_err(|e| {
        PdeError::GreenProblemUnsupported(format!("not a second-order equation: {e}"))
    })?;
    let j = js.to_jets(problem.equations()[0], pool)?;
    let n = indep.len();
    let mut alphas: Vec<Vec<u32>> = vec![vec![0; n]];
    for i in 0..n {
        let mut a = vec![0; n];
        a[i] = 1;
        alphas.push(a);
        for k in i..n {
            let mut b = vec![0; n];
            b[i] += 1;
            b[k] += 1;
            alphas.push(b);
        }
    }
    let syms: Vec<ExprId> = alphas
        .iter()
        .map(|a| js.jet(0, a, pool))
        .collect::<Result<_, _>>()?;
    let mut coeff = HashMap::new();
    for (a, &s) in alphas.iter().zip(&syms) {
        let c = simp(dd(j, s, pool)?, pool);
        if syms.iter().any(|&t| mentions(c, t, pool)) {
            return Err(PdeError::GreenProblemUnsupported(
                "the equation is not linear in the unknown".into(),
            ));
        }
        coeff.insert(a.clone(), c);
    }
    let mut zero = HashMap::new();
    for &s in &syms {
        zero.insert(s, pool.integer(0));
    }
    let rhs = simp(neg(subs(j, &zero, pool), pool), pool);
    if syms.iter().any(|&t| mentions(rhs, t, pool)) {
        return Err(PdeError::GreenProblemUnsupported(
            "the equation is not linear in the unknown".into(),
        ));
    }
    Ok(LinearForm { coeff, rhs })
}

impl LinearForm {
    fn get(&self, a: &[u32], pool: &ExprPool) -> ExprId {
        self.coeff
            .get(a)
            .copied()
            .unwrap_or_else(|| pool.integer(0))
    }

    fn only(&self, keep: &[Vec<u32>], pool: &ExprPool) -> bool {
        self.coeff
            .iter()
            .all(|(a, &c)| keep.contains(a) || simp(c, pool) == pool.integer(0))
    }
}

/// `(α, β, γ)` of a condition `α u(end) + β u'(end) = γ`.
fn robin_of(
    c: &PdeCondition,
    u_name: &str,
    end: ExprId,
    pool: &ExprPool,
) -> Option<(ExprId, ExprId, ExprId)> {
    let e = c.residual(pool);
    let u0 = fresh("__gU0", Domain::Real, &[e], pool);
    let u1 = fresh("__gU1", Domain::Real, &[e], pool);
    let mut nodes = Vec::new();
    let mut stack = vec![e];
    while let Some(n) = stack.pop() {
        if let Some(v) = view(n, pool) {
            if v.name == u_name {
                if v.args.len() != 1 || simp(sub(v.args[0], end, pool), pool) != pool.integer(0) {
                    return None;
                }
                nodes.push((n, v.orders[0]));
                continue;
            }
        }
        super::super::jet::push_children(n, pool, &mut stack);
    }
    if nodes.is_empty() {
        return None;
    }
    // Derivatives first: `u(end)` also occurs inside `Derivative(u(end), 1)`.
    nodes.sort_by_key(|&(_, k)| std::cmp::Reverse(k));
    nodes.dedup();
    let mut r = e;
    for (n, k) in nodes {
        let by = match k {
            0 => u0,
            1 => u1,
            _ => return None,
        };
        r = replace_node(r, n, by, pool);
    }
    let r = simp(r, pool);
    let alpha = simp(dd(r, u0, pool).ok()?, pool);
    let beta = simp(dd(r, u1, pool).ok()?, pool);
    if mentions(alpha, u0, pool)
        || mentions(alpha, u1, pool)
        || mentions(beta, u0, pool)
        || mentions(beta, u1, pool)
    {
        return None;
    }
    let mut m = HashMap::new();
    m.insert(u0, pool.integer(0));
    m.insert(u1, pool.integer(0));
    let gamma = simp(neg(subs(r, &m, pool), pool), pool);
    Some((alpha, beta, gamma))
}

fn end_condition(alpha: ExprId, beta: ExprId, pool: &ExprPool) -> EndCondition {
    if beta == pool.integer(0) {
        EndCondition::Dirichlet
    } else if alpha == pool.integer(0) {
        EndCondition::Neumann
    } else {
        EndCondition::Robin { alpha, beta }
    }
}

/// The verification record of a closed candidate (plus antiderivative
/// resolution for symbolic data).
fn verify_candidate(
    problem: &PdeProblem,
    candidate: ExprId,
    antiders: &[(String, ExprId, ExprId)],
    pool: &ExprPool,
) -> Result<Vec<GreenCheck>, PdeError> {
    let v = verify_solution(
        problem.equations(),
        problem.unknowns(),
        &[candidate],
        problem.conditions(),
        pool,
    )?;
    let mut out = Vec::new();
    for c in v.checks {
        let mut st = c.status;
        if st != ZeroStatus::ProvedZero {
            let mut r = c.residual;
            for (g, body, z) in antiders {
                r = super::super::transform_methods::resolve_antiderivative(r, g, *body, *z, pool)?;
            }
            let s2 = zstatus(r, pool);
            if s2 == ZeroStatus::ProvedZero || st == ZeroStatus::Unknown {
                st = s2;
            }
        }
        out.push(GreenCheck {
            what: c.kind,
            status: st,
            method: "substitution".into(),
            detail: format!("{} {} of the solution", c.kind, c.index),
        });
    }
    Ok(out)
}

#[allow(clippy::too_many_arguments)]
fn finish(
    problem: &PdeProblem,
    method: &str,
    closed_part: ExprId,
    terms: Vec<GreenTerm>,
    green: ExprId,
    source: Vec<ExprId>,
    mut checks: Vec<GreenCheck>,
    side: Vec<ExprId>,
    mut notes: Vec<String>,
    steps: Vec<String>,
    pool: &ExprPool,
) -> Result<GreenSolution, PdeError> {
    let open: Vec<&GreenTerm> = terms.iter().filter(|t| t.closed.is_none()).collect();
    let kind = if open.is_empty() {
        GreenSolutionKind::ClosedForm
    } else {
        GreenSolutionKind::IntegralRepresentation
    };
    let mut parts = vec![closed_part];
    for t in &terms {
        parts.push(match t.closed {
            Some(c) => c,
            None => {
                let mut e = pool.mul(vec![t.kernel, t.density]);
                for (&v, &(lo, hi)) in t.vars.iter().zip(&t.bounds) {
                    e = integral(e, v, lo, hi, pool);
                }
                e
            }
        });
    }
    let value = pool.add(parts);
    let value = if kind == GreenSolutionKind::ClosedForm {
        tidy(simp(value, pool), pool)
    } else {
        value
    };
    if kind == GreenSolutionKind::IntegralRepresentation {
        notes.push(
            "the integrals solve the problem by differentiation under the integral sign: the \
             data must be integrable (and decay at infinity on unbounded domains)"
                .into(),
        );
    }
    let closed_part = simp(
        pool.add(
            std::iter::once(closed_part)
                .chain(terms.iter().filter_map(|t| t.closed))
                .collect(),
        ),
        pool,
    );
    if checks.iter().any(|c| c.status == ZeroStatus::ProvedNonzero) {
        let bad: Vec<String> = checks
            .iter()
            .filter(|c| c.status == ZeroStatus::ProvedNonzero)
            .map(|c| format!("{} ({})", c.what, c.detail))
            .collect();
        return Err(PdeError::GreenVerificationFailed(format!(
            "the {method} candidate failed {}",
            bad.join(", ")
        )));
    }
    checks.retain(|c| !c.detail.is_empty());
    let status = combine(&checks);
    Ok(GreenSolution {
        unknown: problem.unknowns()[0],
        kind,
        method: method.into(),
        value,
        closed_part,
        terms,
        green,
        source,
        status,
        checks,
        side_conditions: side,
        notes,
        steps,
    })
}

/// Solve a problem by its Green's function: `u = ∫ G f + boundary terms`.
///
/// * **Two-point boundary-value problems** — `a₂u'' + a₁u' + a₀u = F(x)` on
///   `Interval(x, a, b)` with one condition `αu + βu' = γ` at each end:
///   brought to Sturm–Liouville form (`p = e^{∫a₁/a₂}`), `G` by variation of
///   parameters ([`ode_green_function`](super::ode_green_function)), and
///   `u(x) = ∫ₐᵇ G(x, ξ)(𝓛u)(ξ) dξ + [p (G ∂_ξu − u ∂_ξG)]_{ξ=a}^{ξ=b}`
///   with the boundary data substituted.  Symbolic data `f(x)` is verified
///   exactly through antiderivatives of the integrands.
/// * **Heat on the line with a source** — `uₜ − κuₓₓ = Q(x, t)`,
///   `u(x, 0) = f(x)`: the heat kernel and Duhamel's integral.
/// * **Wave on the line with a source** — `uₜₜ − c²uₓₓ = Q`, `u(x, 0) = f`,
///   `uₜ(x, 0) = g`: d'Alembert's formula and the integral of `Q` over the
///   backward characteristic triangle.
/// * **Poisson in the upper half-plane** — `Δu = F` on `Line(x) ×
///   HalfLine(y, 0)`, `u(x, 0) = h(x)`: the Dirichlet Green's function and the
///   Poisson kernel `y/(π((x − s)² + y²))`.
///
/// Integrals that `integrate` closes are replaced by their values; a closed
/// answer is verified by substitution in the problem, a representation
/// through the kernel's own verification.
///
/// # Errors
///
/// * `E-PDE-256` — the problem is not one of the shapes above;
/// * `E-PDE-254` — the boundary-value problem has no Green's function;
/// * `E-PDE-255` — the candidate was refuted (never returned);
/// * the errors of the Green's-function constructors.
pub fn green_solve(problem: &PdeProblem, pool: &ExprPool) -> Result<GreenSolution, PdeError> {
    let dom = problem.domain().cloned().ok_or_else(|| {
        PdeError::GreenProblemUnsupported("declare the domain with .on(...)".into())
    })?;
    let parts: Vec<PdeDomain> = match dom {
        PdeDomain::Product(v) => v,
        d => vec![d],
    };
    match parts.as_slice() {
        [PdeDomain::Interval { var, lo, hi }] => bvp(problem, *var, *lo, *hi, pool),
        [a, b] => {
            let (line, half) = match (a, b) {
                (
                    PdeDomain::Line { var },
                    PdeDomain::HalfLine {
                        var: t,
                        end,
                        upper: false,
                    },
                )
                | (
                    PdeDomain::HalfLine {
                        var: t,
                        end,
                        upper: false,
                    },
                    PdeDomain::Line { var },
                ) => (*var, (*t, *end)),
                _ => {
                    return Err(PdeError::GreenProblemUnsupported(
                        "green_solve takes Interval(x, a, b), Line(x) × HalfLine(t, 0) or \
                         Line(x) × HalfLine(y, 0)"
                            .into(),
                    ))
                }
            };
            if simp(half.1, pool) != pool.integer(0) {
                return Err(PdeError::GreenProblemUnsupported(
                    "the half-line must start at 0".into(),
                ));
            }
            let lf = linear_form(problem, pool)?;
            let indep = problem.independent();
            let ix = indep.iter().position(|&v| v == line).expect("domain var");
            let it = indep.iter().position(|&v| v == half.0).expect("domain var");
            let n = indep.len();
            let unit = |i: usize, k: u32| {
                let mut a = vec![0; n];
                a[i] = k;
                a
            };
            let (cxx, ct, ctt) = (
                lf.get(&unit(ix, 2), pool),
                lf.get(&unit(it, 1), pool),
                lf.get(&unit(it, 2), pool),
            );
            let zero = pool.integer(0);
            if ctt == zero && ct != zero && lf.only(&[unit(ix, 2), unit(it, 1)], pool) {
                heat_line(problem, &lf, line, half.0, ct, cxx, pool)
            } else if ctt != zero
                && lf.only(&[unit(ix, 2), unit(it, 2)], pool)
                && super::super::sturm_liouville::sign(div(cxx, ctt, pool), pool)
                    == super::super::sturm_liouville::Sgn::Neg
            {
                wave_line(problem, &lf, line, half.0, ctt, cxx, pool)
            } else if ct == zero
                && lf.only(&[unit(ix, 2), unit(it, 2)], pool)
                && zstatus(sub(ctt, cxx, pool), pool) == ZeroStatus::ProvedZero
            {
                half_plane(problem, &lf, line, half.0, cxx, pool)
            } else {
                Err(PdeError::GreenProblemUnsupported(
                    "on Line × HalfLine green_solve takes the heat equation uₜ − κuₓₓ = Q, the \
                     wave equation uₜₜ − c²uₓₓ = Q or Poisson's equation Δu = F"
                        .into(),
                ))
            }
        }
        _ => Err(PdeError::GreenProblemUnsupported(
            "green_solve takes Interval(x, a, b), Line(x) × HalfLine(t, 0) or Line(x) × \
             HalfLine(y, 0)"
                .into(),
        )),
    }
}

/// Two-point boundary-value problems.
fn bvp(
    problem: &PdeProblem,
    x: ExprId,
    a: ExprId,
    b: ExprId,
    pool: &ExprPool,
) -> Result<GreenSolution, PdeError> {
    let lf = linear_form(problem, pool)?;
    let (a2, a1, a0) = (lf.get(&[2], pool), lf.get(&[1], pool), lf.get(&[0], pool));
    if a2 == pool.integer(0) {
        return Err(PdeError::GreenProblemUnsupported(
            "a two-point problem needs a second-order equation".into(),
        ));
    }
    let u = problem.unknowns()[0];
    let u_name = view(u, pool).map(|v| v.name).unwrap_or_default();
    let mut steps = Vec::new();
    // Sturm–Liouville form: μ = e^{∫ a₁/a₂}.
    let ratio = simp(div(a1, a2, pool), pool);
    let mu = if ratio == pool.integer(0) {
        pool.integer(1)
    } else {
        // `e^{∫dx/x}` is `x`, not `|x|`: a constant factor of μ is free.
        let _principal = crate::integrate::real_log::PrincipalLogs::enter();
        let anti = crate::integrate::integrate(ratio, x, pool)
            .map_err(|e| {
                PdeError::GreenProblemUnsupported(format!(
                    "no integrating factor: ∫ a₁/a₂ did not close ({e})"
                ))
            })?
            .value;
        simp(pool.func("exp", vec![anti]), pool)
    };
    let p = mu;
    let q = simp(div(pool.mul(vec![a0, mu]), a2, pool), pool);
    let g = simp(div(pool.mul(vec![mu, lf.rhs]), a2, pool), pool);
    steps.push(format!(
        "Sturm–Liouville form: (p u')' + q u = g with p = {}, q = {}, g = {}",
        pool.display(p),
        pool.display(q),
        pool.display(g)
    ));
    // Boundary data.
    let mut ends: [Option<(ExprId, ExprId, ExprId)>; 2] = [None, None];
    for c in problem.conditions() {
        let (slot, r) = if let Some(r) = robin_of(c, &u_name, a, pool) {
            (0, r)
        } else if let Some(r) = robin_of(c, &u_name, b, pool) {
            (1, r)
        } else {
            return Err(PdeError::GreenProblemUnsupported(format!(
                "{} = {}: expected α u(end) + β u'(end) = γ at x = a or x = b",
                pool.display(c.lhs),
                pool.display(c.rhs)
            )));
        };
        if ends[slot].is_some() {
            return Err(PdeError::GreenProblemUnsupported(
                "one condition per end".into(),
            ));
        }
        ends[slot] = Some(r);
    }
    let [Some(ca), Some(cb)] = ends else {
        return Err(PdeError::GreenProblemUnsupported(
            "a two-point problem needs one condition at each end".into(),
        ));
    };
    let sl = SturmLiouvilleProblem::new(
        x,
        p,
        q,
        pool.integer(1),
        a,
        b,
        SlBoundary::Separated {
            left: end_condition(ca.0, ca.1, pool),
            right: end_condition(cb.0, cb.1, pool),
        },
        pool,
    )?;
    let xi = fresh(
        "xi",
        Domain::Real,
        &[problem.equations()[0], x, a, b, p, q],
        pool,
    );
    let (gf, u1, u2, amp) = ode_green_parts(&sl, xi, pool)?;
    let (g_left, g_right) = gf.branches.expect("variation of parameters has branches");
    steps.push(format!(
        "G(x, ξ) = {} for x < ξ, {} for x > ξ",
        pool.display(g_left),
        pool.display(g_right)
    ));
    let mut checks: Vec<GreenCheck> = gf
        .checks
        .iter()
        .map(|c| GreenCheck {
            what: "green",
            status: c.status,
            method: c.method.clone(),
            detail: format!("G: {}", c.detail),
        })
        .collect();
    // 𝓛u = −g.
    let ftil = simp(neg(g, pool), pool);
    // Boundary terms [p (G u' − u G_ξ)]_a^b, with G at ξ = end.
    let bterm = |gb: ExprId,
                 end: ExprId,
                 (al, be, ga): (ExprId, ExprId, ExprId)|
     -> Result<ExprId, PdeError> {
        let gend = at(gb, xi, end, pool);
        let pend = at(p, x, end, pool);
        Ok(if be != pool.integer(0) {
            // G u' − u G_ξ = G γ/β
            pool.mul(vec![pend, gend, div(ga, be, pool)])
        } else {
            // u = γ/α, G = 0: −(γ/α) G_ξ
            let gxi = at(dd(gb, xi, pool)?, xi, end, pool);
            neg(pool.mul(vec![pend, div(ga, al, pool), gxi]), pool)
        })
    };
    let boundary = simp(
        sub(bterm(g_left, b, cb)?, bterm(g_right, a, ca)?, pool),
        pool,
    );
    // ∫ₐˣ G(x>ξ) F̃(ξ) dξ + ∫ₓᵇ G(x<ξ) F̃(ξ) dξ.
    let s = fresh(
        "s",
        Domain::Real,
        &[problem.equations()[0], x, a, b, p, q, xi],
        pool,
    );
    let fs = subs1(ftil, x, s, pool);
    let k1 = simp(subs1(g_right, xi, s, pool), pool);
    let k2 = simp(subs1(g_left, xi, s, pool), pool);
    let mut terms = Vec::new();
    let mut antiders = Vec::new();
    let mut verify_parts = vec![boundary];
    if simp(ftil, pool) != pool.integer(0) {
        let c1 = try_integrate(simp(pool.mul(vec![k1, fs]), pool), s, a, x, pool);
        let c2 = try_integrate(simp(pool.mul(vec![k2, fs]), pool), s, x, b, pool);
        // Verification form: A u₂(x)(P₁(x) − P₁(a)) + A u₁(x)(P₂(b) − P₂(x)).
        let names = ["__green_P1", "__green_P2"];
        let z = fresh("z", Domain::Real, &[ftil, x, s], pool);
        let body1 = simp(
            pool.mul(vec![subs1(u1, x, z, pool), subs1(ftil, x, z, pool)]),
            pool,
        );
        let body2 = simp(
            pool.mul(vec![subs1(u2, x, z, pool), subs1(ftil, x, z, pool)]),
            pool,
        );
        let pfun = |nm: &str, arg: ExprId| pool.func(nm, vec![arg]);
        let v1 = c1.unwrap_or_else(|| {
            pool.mul(vec![
                amp,
                u2,
                sub(pfun(names[0], x), pfun(names[0], a), pool),
            ])
        });
        let v2 = c2.unwrap_or_else(|| {
            pool.mul(vec![
                amp,
                u1,
                sub(pfun(names[1], b), pfun(names[1], x), pool),
            ])
        });
        if c1.is_none() {
            antiders.push((names[0].to_string(), body1, z));
        }
        if c2.is_none() {
            antiders.push((names[1].to_string(), body2, z));
        }
        verify_parts.push(v1);
        verify_parts.push(v2);
        terms.push(GreenTerm {
            role: "source",
            kernel: k1,
            density: fs,
            vars: vec![s],
            bounds: vec![(a, x)],
            closed: c1,
        });
        terms.push(GreenTerm {
            role: "source",
            kernel: k2,
            density: fs,
            vars: vec![s],
            bounds: vec![(x, b)],
            closed: c2,
        });
    }
    let candidate = simp(pool.add(verify_parts), pool);
    checks.extend(verify_candidate(problem, candidate, &antiders, pool)?);
    steps.push(
        "u(x) = ∫ₐᵇ G(x, ξ) (𝓛u)(ξ) dξ + [p (G ∂_ξu − u ∂_ξG)]_{ξ=a}^{ξ=b} (Green's identity), \
         𝓛 = −[(p d/dx)' + q]"
            .into(),
    );
    let mut notes = gf.notes.clone();
    if !antiders.is_empty() {
        notes.push(
            "the representation was verified exactly with antiderivatives P' = u_i 𝓛u of the \
             integrands (any integrable data)"
                .into(),
        );
    }
    finish(
        problem,
        "green_bvp",
        boundary,
        terms,
        gf.value,
        vec![xi],
        checks,
        gf.side_conditions.clone(),
        notes,
        steps,
        pool,
    )
}

/// The data `f` of the condition `u(x, 0) = f` (orders `k` in t).
fn initial_datum(
    problem: &PdeProblem,
    t: ExprId,
    k: u32,
    pool: &ExprPool,
) -> Result<Option<ExprId>, PdeError> {
    let indep = problem.independent();
    let it = indep.iter().position(|&v| v == t).expect("t");
    let u_name = view(problem.unknowns()[0], pool)
        .map(|v| v.name)
        .unwrap_or_default();
    let mut found = None;
    for c in problem.conditions() {
        let v = view(c.lhs, pool).ok_or_else(|| {
            PdeError::GreenProblemUnsupported(format!(
                "{}: expected u(x, 0) or uₜ(x, 0)",
                pool.display(c.lhs)
            ))
        })?;
        if v.name != u_name || v.args.len() != indep.len() {
            return Err(PdeError::GreenProblemUnsupported(format!(
                "{}: expected u(x, 0) or uₜ(x, 0)",
                pool.display(c.lhs)
            )));
        }
        for (j, &arg) in v.args.iter().enumerate() {
            let ok = if j == it {
                simp(arg, pool) == pool.integer(0)
            } else {
                arg == indep[j]
            };
            let ord_ok = if j == it { true } else { v.orders[j] == 0 };
            if !ok || !ord_ok {
                return Err(PdeError::GreenProblemUnsupported(format!(
                    "{}: green_solve on the line takes data at t = 0 only",
                    pool.display(c.lhs)
                )));
            }
        }
        if v.orders[it] == k {
            if found.is_some() {
                return Err(PdeError::GreenProblemUnsupported(
                    "a duplicated initial condition".into(),
                ));
            }
            found = Some(c.rhs);
        } else if v.orders[it] > 1 {
            return Err(PdeError::GreenProblemUnsupported(
                "initial data beyond uₜ".into(),
            ));
        }
    }
    Ok(found)
}

fn heat_line(
    problem: &PdeProblem,
    lf: &LinearForm,
    x: ExprId,
    t: ExprId,
    ct: ExprId,
    cxx: ExprId,
    pool: &ExprPool,
) -> Result<GreenSolution, PdeError> {
    let kappa = simp(neg(div(cxx, ct, pool), pool), pool);
    let mut side = Vec::new();
    match super::super::sturm_liouville::sign(kappa, pool) {
        super::super::sturm_liouville::Sgn::Pos => {}
        super::super::sturm_liouville::Sgn::Unknown => {
            side.push(pool.pred_gt(kappa, pool.integer(0)))
        }
        _ => {
            return Err(PdeError::GreenProblemUnsupported(
                "the diffusivity must be positive (backward heat equation)".into(),
            ))
        }
    }
    let q = simp(div(lf.rhs, ct, pool), pool);
    let f = initial_datum(problem, t, 0, pool)?.ok_or_else(|| {
        PdeError::GreenProblemUnsupported("the heat equation needs u(x, 0) = f(x)".into())
    })?;
    if initial_datum(problem, t, 1, pool)?.is_some() {
        return Err(PdeError::GreenProblemUnsupported(
            "the heat equation takes u(x, 0) only".into(),
        ));
    }
    let avoid = [problem.equations()[0], f, x, t, kappa];
    let sg = fresh("sigma", Domain::Positive, &avoid, pool);
    let s = fresh("s", Domain::Real, &avoid, pool);
    let kern = fundamental_solution(
        GreenOperator::Heat { diffusivity: kappa },
        &[x],
        Some(sg),
        pool,
    )?;
    let mut checks: Vec<GreenCheck> = kern
        .checks
        .iter()
        .map(|c| GreenCheck {
            what: "green",
            status: c.status,
            method: c.method.clone(),
            detail: format!("heat kernel: {}", c.detail),
        })
        .collect();
    let k_at = |time: ExprId| {
        simp(
            subs1(subs1(kern.smooth, x, sub(x, s, pool), pool), sg, time, pool),
            pool,
        )
    };
    let oo = pool.pos_infinity();
    let mut terms = Vec::new();
    // Initial term ∫ K(x − s, t) f(s) ds (t > 0 as a positive twin).
    let fs = subs1(f, x, s, pool);
    let k_init = simp(subs1(kern.smooth, x, sub(x, s, pool), pool), pool);
    // Closed attempts on the shifted form ∫ K(s, σ) f(x − s) ds (K is even).
    let k_shift = simp(subs1(kern.smooth, x, s, pool), pool);
    let line_int = |dens_shifted: ExprId| {
        try_integrate(
            simp(pool.mul(vec![k_shift, dens_shifted]), pool),
            s,
            neg(oo, pool),
            oo,
            pool,
        )
    };
    // σ is positive here, so distributing powers is exact.
    let tidy_pos = |v: ExprId| {
        let v = simp(super::expand_powers_pos(simp(v, pool), pool), pool);
        let v = simp(cancel_args(v, &[x, sg], pool), pool);
        let v = simp(super::expand_powers_pos(v, pool), pool);
        simp(super::merge_powers(v, pool), pool)
    };
    let c_init = line_int(subs1(f, x, sub(x, s, pool), pool))
        .map(|v| simp(subs1(tidy_pos(v), sg, t, pool), pool));
    terms.push(GreenTerm {
        role: "initial",
        kernel: k_at(t),
        density: fs,
        vars: vec![s],
        bounds: vec![(neg(oo, pool), oo)],
        closed: c_init,
    });
    // Duhamel ∫₀ᵗ ∫ K(x − s, t − τ) Q(s, τ) ds dτ, written with σ = t − τ.
    if q != pool.integer(0) {
        let qs = subs1(subs1(q, x, s, pool), t, sub(t, sg, pool), pool);
        let q_shift = subs1(
            subs1(q, x, sub(x, s, pool), pool),
            t,
            sub(t, sg, pool),
            pool,
        );
        let inner = line_int(q_shift);
        let closed = inner.and_then(|iv| try_integrate(iv, sg, pool.integer(0), t, pool));
        terms.push(GreenTerm {
            role: "source",
            kernel: k_init,
            density: qs,
            vars: vec![s, sg],
            bounds: vec![(neg(oo, pool), oo), (pool.integer(0), t)],
            closed,
        });
    }
    let all_closed = terms.iter().all(|t| t.closed.is_some());
    if all_closed {
        let cand = simp(
            pool.add(terms.iter().map(|t| t.closed.unwrap()).collect()),
            pool,
        );
        let mut vc = verify_candidate(problem, cand, &[], pool)?;
        // The initial condition of a heat solution is a limit t → 0⁺ when the
        // closed form has t in a denominator.
        for c in vc.iter_mut() {
            if c.what == "ic" && c.status != ZeroStatus::ProvedZero {
                let tp = fresh("tau", Domain::Positive, &[cand, f], pool);
                let cpos = simp(
                    super::expand_powers_pos(simp(subs1(cand, t, tp, pool), pool), pool),
                    pool,
                );
                let cpos = simp(cancel_args(cpos, &[x, tp], pool), pool);
                if let Ok(v) = crate::calculus::limit(
                    cpos,
                    tp,
                    pool.integer(0),
                    crate::calculus::LimitDirection::Plus,
                    pool,
                ) {
                    let st = zstatus(sub(v, f, pool), pool);
                    if st == ZeroStatus::ProvedZero {
                        c.status = st;
                        c.method = "limit t → 0⁺".into();
                    }
                }
            }
        }
        checks.extend(vc);
    }
    let steps = vec![
        "u(x, t) = ∫ K(x − s, t) f(s) ds + ∫₀ᵗ ∫ K(x − s, t − τ) Q(s, τ) ds dτ (Duhamel), \
         K the heat kernel"
            .into(),
    ];
    finish(
        problem,
        "green_heat_line",
        pool.integer(0),
        terms,
        kern.kernel,
        vec![],
        checks,
        side,
        vec![
            "the source integral is written with σ = t − τ: ∫₀ᵗ dσ ∫ ds K(x − s, σ) Q(s, t − σ)"
                .to_string(),
        ],
        steps,
        pool,
    )
}

fn wave_line(
    problem: &PdeProblem,
    lf: &LinearForm,
    x: ExprId,
    t: ExprId,
    ctt: ExprId,
    cxx: ExprId,
    pool: &ExprPool,
) -> Result<GreenSolution, PdeError> {
    let c2 = simp(neg(div(cxx, ctt, pool), pool), pool);
    let c = simp(super::expand_powers_pos(super::sqrt(c2, pool), pool), pool);
    let q = simp(div(lf.rhs, ctt, pool), pool);
    let f = initial_datum(problem, t, 0, pool)?.unwrap_or_else(|| pool.integer(0));
    let g = initial_datum(problem, t, 1, pool)?.unwrap_or_else(|| pool.integer(0));
    let avoid = [problem.equations()[0], f, g, x, t, c];
    let s = fresh("s", Domain::Real, &avoid, pool);
    let sg = fresh("sigma", Domain::Positive, &avoid, pool);
    let ct = pool.mul(vec![c, t]);
    let half = pool.rational(1, 2);
    let kern = fundamental_solution(GreenOperator::Wave { speed: c }, &[x], Some(t), pool)?;
    let mut checks: Vec<GreenCheck> = kern
        .checks
        .iter()
        .map(|cc| GreenCheck {
            what: "green",
            status: cc.status,
            method: cc.method.clone(),
            detail: format!("wave kernel: {}", cc.detail),
        })
        .collect();
    let closed_part = pool.mul(vec![
        half,
        pool.add(vec![
            subs1(f, x, sub(x, ct, pool), pool),
            subs1(f, x, pool.add(vec![x, ct]), pool),
        ]),
    ]);
    let inv2c = div(pool.integer(1), pool.mul(vec![pool.integer(2), c]), pool);
    let mut terms = Vec::new();
    let mut antiders = Vec::new();
    let mut verify_parts = vec![closed_part];
    if simp(g, pool) != pool.integer(0) {
        let gs = subs1(g, x, s, pool);
        let (lo, hi) = (sub(x, ct, pool), pool.add(vec![x, ct]));
        let closed =
            try_integrate(gs, s, lo, hi, pool).map(|v| simp(pool.mul(vec![inv2c, v]), pool));
        match closed {
            Some(v) => verify_parts.push(v),
            None => {
                let nm = "__green_Gg";
                let z = fresh("z", Domain::Real, &[g, x], pool);
                antiders.push((nm.to_string(), subs1(g, x, z, pool), z));
                verify_parts.push(pool.mul(vec![
                    inv2c,
                    sub(pool.func(nm, vec![hi]), pool.func(nm, vec![lo]), pool),
                ]));
            }
        }
        terms.push(GreenTerm {
            role: "initial_velocity",
            kernel: inv2c,
            density: gs,
            vars: vec![s],
            bounds: vec![(lo, hi)],
            closed,
        });
    }
    if q != pool.integer(0) {
        // ∫₀ᵗ dσ ∫_{x−cσ}^{x+cσ} Q(s, t − σ) ds / (2c)
        let qs = subs1(subs1(q, x, s, pool), t, sub(t, sg, pool), pool);
        let cs = pool.mul(vec![c, sg]);
        let inner = try_integrate(qs, s, sub(x, cs, pool), pool.add(vec![x, cs]), pool);
        let closed = inner
            .and_then(|iv| try_integrate(iv, sg, pool.integer(0), t, pool))
            .map(|v| simp(pool.mul(vec![inv2c, v]), pool));
        match closed {
            Some(v) => verify_parts.push(v),
            None => verify_parts.push(pool.integer(0)),
        }
        terms.push(GreenTerm {
            role: "source",
            kernel: inv2c,
            density: qs,
            vars: vec![s, sg],
            bounds: vec![
                (sub(x, cs, pool), pool.add(vec![x, cs])),
                (pool.integer(0), t),
            ],
            closed,
        });
    }
    let source_open = terms
        .iter()
        .any(|t| t.role == "source" && t.closed.is_none());
    if !source_open {
        let cand = simp(pool.add(verify_parts), pool);
        checks.extend(verify_candidate(problem, cand, &antiders, pool)?);
    }
    let mut side = Vec::new();
    if super::super::sturm_liouville::sign(c2, pool) != super::super::sturm_liouville::Sgn::Pos {
        side.push(pool.pred_gt(c2, pool.integer(0)));
    }
    finish(
        problem,
        "green_wave_line",
        simp(closed_part, pool),
        terms,
        kern.kernel,
        vec![],
        checks,
        side,
        vec!["the source integral runs over the backward characteristic triangle, written with \
              σ = t − τ"
            .into()],
        vec!["u = ½(f(x − ct) + f(x + ct)) + (1/2c)∫_{x−ct}^{x+ct} g + (1/2c)∫₀ᵗ∫_{x−c(t−τ)}^{x+c(t−τ)} Q \
              (d'Alembert and Duhamel with the kernel H(ct − |x|)/(2c))"
            .into()],
        pool,
    )
}

fn half_plane(
    problem: &PdeProblem,
    lf: &LinearForm,
    x: ExprId,
    y: ExprId,
    cxx: ExprId,
    pool: &ExprPool,
) -> Result<GreenSolution, PdeError> {
    let fsrc = simp(div(lf.rhs, cxx, pool), pool);
    let indep = problem.independent();
    let iy = indep.iter().position(|&v| v == y).expect("y");
    let u_name = view(problem.unknowns()[0], pool)
        .map(|v| v.name)
        .unwrap_or_default();
    let mut h = None;
    for c in problem.conditions() {
        let v = view(c.lhs, pool);
        let ok = v.as_ref().is_some_and(|v| {
            v.name == u_name
                && v.orders.iter().all(|&o| o == 0)
                && v.args.iter().enumerate().all(|(j, &a)| {
                    if j == iy {
                        simp(a, pool) == pool.integer(0)
                    } else {
                        a == indep[j]
                    }
                })
        });
        if !ok || h.is_some() {
            return Err(PdeError::GreenProblemUnsupported(
                "the half-plane problem takes one Dirichlet condition u(x, 0) = h(x)".into(),
            ));
        }
        h = Some(c.rhs);
    }
    let h =
        h.ok_or_else(|| PdeError::GreenProblemUnsupported("u(x, 0) = h(x) is missing".into()))?;
    let avoid = [problem.equations()[0], h, x, y];
    let s = fresh("s", Domain::Real, &avoid, pool);
    let eta = fresh("eta", Domain::Positive, &avoid, pool);
    let (sp, so) = if indep[0] == x {
        (vec![x, y], vec![s, eta])
    } else {
        (vec![y, x], vec![eta, s])
    };
    let gf = image_green_function(
        GreenOperator::Laplace,
        &sp,
        None,
        &so,
        ImageRegion::HalfSpace { axis: iy },
        &[WallCondition::Dirichlet],
        pool,
    )?;
    let mut checks: Vec<GreenCheck> = gf
        .checks
        .iter()
        .map(|c| GreenCheck {
            what: "green",
            status: c.status,
            method: c.method.clone(),
            detail: format!("G: {}", c.detail),
        })
        .collect();
    let pk = gf
        .poisson_kernel
        .expect("Dirichlet half-plane has a Poisson kernel");
    let oo = pool.pos_infinity();
    let hs = subs1(h, x, s, pool);
    let mut terms = Vec::new();
    // y > 0 as a positive twin for the integral.
    let yp = fresh("y", Domain::Positive, &avoid, pool);
    let closed_b = try_integrate(
        simp(subs1(pool.mul(vec![pk, hs]), y, yp, pool), pool),
        s,
        neg(oo, pool),
        oo,
        pool,
    )
    .map(|v| simp(subs1(v, yp, y, pool), pool));
    terms.push(GreenTerm {
        role: "boundary",
        kernel: pk,
        density: hs,
        vars: vec![s],
        bounds: vec![(neg(oo, pool), oo)],
        closed: closed_b,
    });
    if fsrc != pool.integer(0) {
        let fs = subs1(subs1(fsrc, x, s, pool), y, eta, pool);
        terms.push(GreenTerm {
            role: "source",
            kernel: gf.value,
            density: fs,
            vars: vec![s, eta],
            bounds: vec![(neg(oo, pool), oo), (pool.integer(0), oo)],
            closed: None,
        });
    }
    let all_closed = terms.iter().all(|t| t.closed.is_some());
    if all_closed {
        let cand = simp(
            pool.add(terms.iter().map(|t| t.closed.unwrap()).collect()),
            pool,
        );
        // The boundary condition is a limit y → 0⁺ for a Poisson integral;
        // the closed candidate is checked by plain substitution.
        checks.extend(verify_candidate(problem, cand, &[], pool)?);
    }
    finish(
        problem,
        "green_half_plane",
        pool.integer(0),
        terms,
        gf.value,
        so,
        checks,
        vec![],
        vec![format!(
            "bounded solution; P = {} is the Poisson kernel of the half-plane",
            pool.display(pk)
        )],
        vec![
            "u = ∫ P(x, y; s) h(s) ds + ∫∫ G(x, y; s, η) F(s, η) ds dη (Green's representation)"
                .into(),
        ],
        pool,
    )
    .map(|mut r| {
        r.notes
            .push("data must be bounded and continuous (for h) / integrable (for F)".into());
        r
    })
}
