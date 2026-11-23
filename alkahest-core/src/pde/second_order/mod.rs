//! Second-order linear PDEs in two independent variables (milestone M2).
//!
//! `A u_xx + B u_xy + C u_yy + D u_x + E u_y + F u = G`.
//!
//! * [`canonical_form`] / [`canonical_forms`] — characteristic coordinates
//!   from `A dy² − B dx dy + C dx² = 0` (first integrals of the
//!   characteristic ODEs, through `dsolve` when no closed pattern applies)
//!   and the hyperbolic `w_ξη + … = 0`, parabolic `w_ηη + … = 0` or elliptic
//!   `w_αα + w_ββ + … = 0` normal form, region by region for variable
//!   coefficients.
//! * [`solve_second_order`] — closed-form general solutions in arbitrary
//!   functions, by (in order):
//!   1. an ODE in one variable (`u_xx + u = 0`): `dsolve`, constants
//!      becoming arbitrary functions of the other variable;
//!   2. constant coefficients: the shift `u = e^{αx+βy} v` that removes the
//!      first-order terms, and the full symbol `P(ξ, η)` factored into
//!      first-order operators (FLINT over ℚ, the quadratic formula over
//!      algebraic and complex extensions), each solved by characteristics —
//!      d'Alembert's `F(x + ct) + G(x − ct)` is the case `ξ² − η²/c²`;
//!   3. the Laplace cascade ([`laplace_cascade`]) for
//!      `u_xy + a u_x + b u_y + c u = s` with variable coefficients;
//!   4. the canonical form, then 1–3 in the characteristic coordinates;
//!   5. for a homogeneous constant-coefficient equation whose symbol is
//!      irreducible (heat, Klein–Gordon), the exponential family
//!      `e^{kx + ω(k)y}` from the dispersion relation — a particular
//!      solution, labelled as such.
//!
//!   With Cauchy data `u(x, y₀) = f`, `u_y(x, y₀) = g` it applies
//!   d'Alembert's formula (generalised to any constant hyperbolic principal
//!   part).
//!
//! Every answer has been substituted back by
//! [`super::verify::verify_solution`] — arbitrary functions kept symbolic —
//! and is returned only when that is `exactly_verified`; otherwise the
//! solver refuses with an `E-PDE-1xx` code.

mod canonical;
mod cascade;
mod cauchy;
mod constant;
mod first_order;
mod linear;
mod util;

use crate::kernel::{ExprId, ExprPool};
use crate::ode::dsolve::{dsolve_with, OdeInput, SolutionForm};
use crate::simplify::assumptions::AssumptionContext;

pub use canonical::CanonicalForm;

use super::classify::{PdeInfo, PdeType};
use super::error::PdeError;
use super::problem::{ArbitraryFunctions, PdeProblem};
use super::verify::{verify_solution, PdeVerification, VerificationStatus};
use linear::Linear2;
use util::{div, exp, is_zero, neg, simp, subs2};

/// Default budget of Laplace transformations for [`laplace_cascade`] and
/// [`solve_second_order`].
pub const DEFAULT_CASCADE_STEPS: usize = 8;

/// Options for [`solve_second_order_with`].
#[derive(Clone, Debug)]
#[non_exhaustive]
pub struct SecondOrderOptions {
    /// At most this many Laplace transformations (both directions together).
    pub max_cascade_steps: usize,
    /// Prefix of the arbitrary functions (`F` gives `F1, F2, …`).
    pub prefix: String,
    /// Whether a particular family (the exponential ansatz) may be returned
    /// when no general solution is found.
    pub allow_particular: bool,
}

impl Default for SecondOrderOptions {
    fn default() -> Self {
        SecondOrderOptions {
            max_cascade_steps: DEFAULT_CASCADE_STEPS,
            prefix: "F".into(),
            allow_particular: true,
        }
    }
}

impl SecondOrderOptions {
    /// The defaults.
    pub fn new() -> Self {
        Self::default()
    }
    /// Set [`Self::max_cascade_steps`].
    pub fn with_max_cascade_steps(mut self, n: usize) -> Self {
        self.max_cascade_steps = n;
        self
    }
    /// Set [`Self::prefix`].
    pub fn with_prefix(mut self, p: &str) -> Self {
        self.prefix = p.to_string();
        self
    }
    /// Set [`Self::allow_particular`].
    pub fn with_allow_particular(mut self, b: bool) -> Self {
        self.allow_particular = b;
        self
    }
}

/// A verified solution of a second-order equation.
#[derive(Clone, Debug, PartialEq, Eq)]
#[non_exhaustive]
pub struct SecondOrderSolution {
    /// The solution `u(x, y) = …`.
    pub solution: ExprId,
    /// `"general"` (in arbitrary functions), or `"particular"` (a solution of
    /// a Cauchy problem, or a parametrised family that is not general).
    pub kind: &'static str,
    /// The method: `"ode_in_one_variable"`, `"exponential_shift"`,
    /// `"factored_symbol"`, `"laplace_cascade"`, `"canonical_form+…"`,
    /// `"dispersion_relation"`, `"dalembert"`.
    pub method: String,
    /// Names of the arbitrary functions in [`Self::solution`].
    pub arbitrary_functions: Vec<String>,
    /// Free parameters of a family (the wave number `k` of `e^{kx+ω(k)y}`).
    pub parameters: Vec<ExprId>,
    /// Side conditions under which the formula holds (`c ≠ 0`, a region).
    pub conditions: Vec<ExprId>,
    /// Human-readable derivation steps.
    pub steps: Vec<String>,
    /// The canonical form used, when the method went through one.
    pub canonical: Option<CanonicalForm>,
    /// The verification of [`Self::solution`] against the problem — always
    /// `exactly_verified` for a returned solution.
    pub verification: PdeVerification,
}

/// A candidate before verification.
struct Candidate {
    solution: ExprId,
    method: String,
    general: bool,
    parameters: Vec<ExprId>,
    conditions: Vec<ExprId>,
    steps: Vec<String>,
    canonical: Option<CanonicalForm>,
}

// ---------------------------------------------------------------------------
// Canonical forms
// ---------------------------------------------------------------------------

/// The canonical form of a linear second-order equation in two variables
/// whose type is the same everywhere (hyperbolic, parabolic or elliptic).
///
/// # Errors
///
/// `E-PDE-100` for an equation that is not linear second-order in two
/// variables; `E-PDE-102` when the type changes across the plane (use
/// [`canonical_forms`]) or is undetermined; `E-PDE-101` when the
/// characteristic ODEs have no first integral found in closed form.
pub fn canonical_form(
    equation: ExprId,
    unknown: ExprId,
    pool: &ExprPool,
) -> Result<CanonicalForm, PdeError> {
    let (l, info) = Linear2::from_equation(equation, unknown, pool)?;
    match info.pde_type {
        Some(t @ (PdeType::Hyperbolic | PdeType::Parabolic | PdeType::Elliptic)) => {
            canonical::build(&l, t, None, pool)
        }
        _ => {
            let regions = region_list(&info, pool);
            if regions.len() == 1 {
                let r = regions[0].clone();
                return canonical::build(&l, r.pde_type, Some(r), pool);
            }
            Err(PdeError::TypeNotUniform(format!(
            "{} = 0 is {} on the plane; canonical_forms gives one form per region",
            pool.display(equation),
            info.pde_type.map(|t| t.as_str()).unwrap_or("untyped")
        )))
        }
    }
}

/// One canonical form per region of constant type (a single entry when the
/// type is the same everywhere).  Regions whose form could not be computed
/// are skipped; parabolic lines separating two regions are skipped.
///
/// # Errors
///
/// As [`canonical_form`]; `E-PDE-101` when no region yields a form.
pub fn canonical_forms(
    equation: ExprId,
    unknown: ExprId,
    pool: &ExprPool,
) -> Result<Vec<CanonicalForm>, PdeError> {
    let (l, info) = Linear2::from_equation(equation, unknown, pool)?;
    if let Some(t @ (PdeType::Hyperbolic | PdeType::Parabolic | PdeType::Elliptic)) = info.pde_type
    {
        return Ok(vec![canonical::build(&l, t, None, pool)?]);
    }
    let mut out = Vec::new();
    let mut errors = Vec::new();
    for r in region_list(&info, pool) {
        match canonical::build(&l, r.pde_type, Some(r.clone()), pool) {
            Ok(f) => out.push(f),
            Err(e) => errors.push(e.to_string()),
        }
    }
    if out.is_empty() {
        return Err(if errors.is_empty() {
            PdeError::TypeNotUniform(format!(
                "{} = 0 has no region of definite type",
                pool.display(equation)
            ))
        } else {
            PdeError::CharacteristicsNotIntegrated(errors.join("; "))
        });
    }
    Ok(out)
}

fn region_list(info: &PdeInfo, pool: &ExprPool) -> Vec<super::classify::TypeRegion> {
    let is_line = |r: &super::classify::TypeRegion| {
        (r.lower.is_some() && r.lower == r.upper)
            || pool.with(r.condition, |d| {
                matches!(
                    d,
                    crate::kernel::ExprData::Predicate {
                        kind: crate::kernel::expr::PredicateKind::Eq,
                        ..
                    }
                )
            })
    };
    info.regions
        .iter()
        .filter(|r| {
            matches!(
                r.pde_type,
                PdeType::Hyperbolic | PdeType::Elliptic | PdeType::Parabolic
            ) && !is_line(r)
        })
        .cloned()
        .collect()
}

// ---------------------------------------------------------------------------
// Laplace cascade
// ---------------------------------------------------------------------------

/// The Laplace invariants `(h, k)` of `u_xy + a u_x + b u_y + c u = s`
/// (the equation is first divided by the coefficient of `u_xy`).
///
/// # Errors
///
/// `E-PDE-108` when the equation is not of that form (`A = C = 0`).
pub fn laplace_invariants(
    equation: ExprId,
    unknown: ExprId,
    pool: &ExprPool,
) -> Result<(ExprId, ExprId), PdeError> {
    let (l, _) = Linear2::from_equation(equation, unknown, pool)?;
    let (a, b, c, _) = laplace_form(&l, pool)?;
    cascade::invariants(a, b, c, l.x, l.y, pool)
}

fn laplace_form(l: &Linear2, pool: &ExprPool) -> Result<(ExprId, ExprId, ExprId, ExprId), PdeError> {
    if !is_zero(l.a(), pool) || !is_zero(l.c(), pool) || is_zero(l.b(), pool) {
        return Err(PdeError::NotLaplaceForm(format!(
            "the principal part {} u_xx + {} u_xy + {} u_yy is not a multiple of u_xy; \
             transform with canonical_form first",
            pool.display(l.a()),
            pool.display(l.b()),
            pool.display(l.c())
        )));
    }
    let b = l.b();
    Ok((
        div(l.dd(), b, pool),
        div(l.e(), b, pool),
        div(l.f(), b, pool),
        div(l.rhs, b, pool),
    ))
}

/// Solve `u_xy + a u_x + b u_y + c u = s` (variable coefficients) by the
/// Laplace cascade with at most `max_steps` Laplace transformations.
///
/// # Errors
///
/// `E-PDE-108` for an equation not of that form; `E-PDE-103` when no
/// invariant vanishes within the budget; `E-PDE-106` when a quadrature of an
/// arbitrary function has no integral-free form; `E-PDE-105` if the
/// candidate fails verification.
pub fn laplace_cascade(
    equation: ExprId,
    unknown: ExprId,
    max_steps: usize,
    pool: &ExprPool,
) -> Result<SecondOrderSolution, PdeError> {
    let (l, _) = Linear2::from_equation(equation, unknown, pool)?;
    laplace_form(&l, pool)?;
    let mut arb = ArbitraryFunctions::new("F").avoiding(&[equation], pool);
    let cand = cascade_candidate(&l, max_steps, &mut arb, pool)?;
    finish(equation, unknown, &[], cand, pool)
}

fn cascade_candidate(
    l: &Linear2,
    max_steps: usize,
    arb: &mut ArbitraryFunctions,
    pool: &ExprPool,
) -> Result<Candidate, PdeError> {
    let (a, b, c, s) = laplace_form(l, pool)?;
    let mut fresh = |arg: ExprId| arb.fresh(vec![arg], pool);
    let out = cascade::cascade(a, b, c, s, l.x, l.y, max_steps, &mut fresh, pool)?;
    let mut conds = Vec::new();
    first_order::note_nonzero(l.b(), &mut conds, pool);
    Ok(Candidate {
        solution: out.solution,
        method: format!("laplace_cascade({} step(s))", out.transforms.unsigned_abs()),
        general: true,
        parameters: Vec::new(),
        conditions: conds,
        steps: out.steps,
        canonical: None,
    })
}

// ---------------------------------------------------------------------------
// The solver
// ---------------------------------------------------------------------------

/// [`solve_second_order_with`] with the default options.
///
/// # Errors
///
/// See [`solve_second_order_with`].
pub fn solve_second_order(
    problem: &PdeProblem,
    pool: &ExprPool,
) -> Result<SecondOrderSolution, PdeError> {
    solve_second_order_with(problem, &SecondOrderOptions::default(), pool)
}

/// Solve a linear second-order equation in two variables in closed form;
/// see the [module documentation](self) for the methods.  Only a solution
/// whose substitution residual is proved zero is returned.
///
/// # Errors
///
/// `E-PDE-008` for a system; `E-PDE-100` for an equation outside the class;
/// `E-PDE-107` for conditions other than Cauchy data on a line; `E-PDE-104`
/// when no method gives a closed form; `E-PDE-105` when every candidate
/// failed verification.
pub fn solve_second_order_with(
    problem: &PdeProblem,
    opts: &SecondOrderOptions,
    pool: &ExprPool,
) -> Result<SecondOrderSolution, PdeError> {
    if problem.equations().len() != 1 || problem.unknowns().len() != 1 {
        return Err(PdeError::SystemShape(
            "solve_second_order takes one equation in one unknown".into(),
        ));
    }
    let eq = problem.equations()[0];
    let u = problem.unknowns()[0];
    let (l, info) = Linear2::from_equation(eq, u, pool)?;
    let mut avoid = vec![eq];
    for c in problem.conditions() {
        avoid.push(c.lhs);
        avoid.push(c.rhs);
    }
    let arb = ArbitraryFunctions::new(&opts.prefix).avoiding(&avoid, pool);

    if !problem.conditions().is_empty() {
        let cand = cauchy_candidate(&l, problem, pool)?;
        return finish(eq, u, problem.conditions(), cand, pool);
    }

    let mut reasons: Vec<String> = Vec::new();
    let mut unverified = 0usize;
    let routes = general_candidates(&l, &info, opts, &arb, 0, &mut reasons, pool);
    for cand in routes {
        let cand = match cand {
            Ok(c) => c,
            Err(e) => {
                reasons.push(e.to_string());
                continue;
            }
        };
        match finish(eq, u, &[], cand, pool) {
            Ok(s) => return Ok(s),
            Err(e) => {
                unverified += 1;
                reasons.push(e.to_string());
            }
        }
    }
    if unverified > 0 {
        return Err(PdeError::SecondOrderNotVerified(reasons.join("; ")));
    }
    // A budget refusal or a quadrature obstruction is the most informative
    // single reason.
    if let Some(r) = reasons.iter().find(|r| r.contains("transformation(s)")) {
        return Err(PdeError::LaplaceBudgetExhausted(r.clone()));
    }
    if let Some(r) = reasons.iter().find(|r| r.contains("quadrature")) {
        return Err(PdeError::QuadratureOfArbitraryFunction(r.clone()));
    }
    Err(PdeError::NoClosedForm(format!(
        "no method gives a closed-form solution of {} = 0{}",
        pool.display(eq),
        if reasons.is_empty() {
            String::new()
        } else {
            format!(" ({})", reasons.join("; "))
        }
    )))
}

/// Verify a candidate and package it.
fn finish(
    eq: ExprId,
    u: ExprId,
    conditions: &[super::problem::PdeCondition],
    cand: Candidate,
    pool: &ExprPool,
) -> Result<SecondOrderSolution, PdeError> {
    let mut cand = cand;
    // Equivalent spellings of the candidate, smallest first: the zero test
    // has size budgets, so a compact form can verify where a nested one
    // cannot.
    let tidied = util::tidy(cand.solution, pool);
    let mut forms = vec![tidied];
    let vars = super::jet::decode_unknown(u, pool)?.args;
    if let Some(c) = util::collect_arbitrary(tidied, &vars, pool) {
        forms.push(c);
    }
    forms.sort_by_key(|&f| util::size(f, pool));
    forms.push(cand.solution);
    forms.dedup();
    let mut v = None;
    for f in forms {
        let vf = verify_solution(&[eq], &[u], &[f], conditions, pool)?;
        let ok = vf.status == VerificationStatus::ExactlyVerified;
        let refuted = vf.status == VerificationStatus::Refuted;
        if ok {
            cand.solution = f;
        }
        v = Some(vf);
        if ok || refuted {
            break;
        }
    }
    let v = v.expect("at least one form");
    if v.status != VerificationStatus::ExactlyVerified {
        return Err(PdeError::SecondOrderNotVerified(format!(
            "{} candidate {} is {}",
            cand.method,
            pool.display(cand.solution),
            v.status.as_str()
        )));
    }
    let kind = if cand.general && conditions.is_empty() {
        "general"
    } else {
        "particular"
    };
    Ok(SecondOrderSolution {
        solution: cand.solution,
        kind,
        method: cand.method,
        arbitrary_functions: v.arbitrary_functions.clone(),
        parameters: cand.parameters,
        conditions: cand.conditions,
        steps: cand.steps,
        canonical: cand.canonical,
        verification: v,
    })
}

fn cauchy_candidate(l: &Linear2, problem: &PdeProblem, pool: &ExprPool) -> Result<Candidate, PdeError> {
    let conds = problem.conditions();
    let (ll, data, swapped) = match cauchy::read_data(l, conds, pool) {
        Some(d) => (l.clone(), d, false),
        None => {
            let s = l.swapped();
            match cauchy::read_data(&s, conds, pool) {
                Some(d) => (s, d, true),
                None => {
                    return Err(PdeError::CauchyDataUnsupported(
                        "only Cauchy data u(x, y0) = f(x), u_y(x, y0) = g(x) on a line are supported \
                         by the second-order solver (boundary-value problems are a later milestone)"
                            .into(),
                    ))
                }
            }
        }
    };
    let _ = swapped;
    let mut side = Vec::new();
    match cauchy::solve(&ll, &data, &mut side, pool)? {
        Some((sol, steps)) => Ok(Candidate {
            solution: sol,
            method: "dalembert".into(),
            general: false,
            parameters: Vec::new(),
            conditions: side,
            steps,
            canonical: None,
        }),
        None => Err(PdeError::CauchyDataUnsupported(format!(
            "d'Alembert's formula needs constant coefficients, no lower-order terms and a \
             non-parabolic principal part; got {} u_xx + {} u_xy + {} u_yy + {} u_x + {} u_y + {} u",
            pool.display(ll.a()),
            pool.display(ll.b()),
            pool.display(ll.c()),
            pool.display(ll.dd()),
            pool.display(ll.e()),
            pool.display(ll.f())
        ))),
    }
}

/// The candidate general solutions, cheapest first.  Each route gets its
/// own copy of the arbitrary-function generator so that a failed route
/// does not shift the numbering of the next one.
fn general_candidates(
    l: &Linear2,
    info: &PdeInfo,
    opts: &SecondOrderOptions,
    arb: &ArbitraryFunctions,
    depth: usize,
    reasons: &mut Vec<String>,
    pool: &ExprPool,
) -> Vec<Result<Candidate, PdeError>> {
    let mut out = Vec::new();
    // 1. an ODE in one variable
    for var_x in [true, false] {
        let mut a = arb.clone();
        if let Some(c) = ode_candidate(l, var_x, &mut a, pool) {
            out.push(Ok(c));
        }
    }
    // 2. constant coefficients
    if l.constant_coefficients(pool) {
        let mut a = arb.clone();
        if let Some(c) = shift_candidate(l, &mut a, pool) {
            out.push(c);
        }
        let mut a = arb.clone();
        let mut conds = Vec::new();
        match constant::factor_symbol(l, &mut conds, pool) {
            Some(fac) => {
                let mut fresh = |arg: ExprId| a.fresh(vec![arg], pool);
                match constant::solve_factored(l, &fac, &mut fresh, &mut conds, pool) {
                    Ok(Some(cs)) => out.push(Ok(Candidate {
                        solution: cs.solution,
                        method: cs.method.into(),
                        general: cs.general,
                        parameters: cs.parameters,
                        conditions: conds,
                        steps: cs.steps,
                        canonical: None,
                    })),
                    Ok(None) => reasons.push(
                        "the symbol factors but a particular-solution quadrature does not close"
                            .into(),
                    ),
                    Err(e) => out.push(Err(e)),
                }
            }
            None => reasons.push("the symbol is irreducible".into()),
        }
    }
    // 3. the Laplace cascade
    if is_zero(l.a(), pool) && is_zero(l.c(), pool) && !is_zero(l.b(), pool) {
        let mut a = arb.clone();
        out.push(cascade_candidate(l, opts.max_cascade_steps, &mut a, pool));
    }
    // 4. through the canonical form
    if depth == 0 && !l.constant_coefficients(pool) {
        let types: Vec<(PdeType, Option<super::classify::TypeRegion>)> = match info.pde_type {
            Some(t @ (PdeType::Hyperbolic | PdeType::Parabolic | PdeType::Elliptic)) => {
                vec![(t, None)]
            }
            _ => region_list(info, pool)
                .into_iter()
                .map(|r| (r.pde_type, Some(r)))
                .collect(),
        };
        for (t, region) in types {
            match canonical::build(l, t, region, pool) {
                Ok(cf) => out.extend(via_canonical(cf, opts, arb, reasons, pool)),
                Err(e) => out.push(Err(e)),
            }
        }
    }
    // 5. a particular family
    if opts.allow_particular && depth == 0 && l.constant_coefficients(pool) && l.homogeneous(pool) {
        let mut taken = util::names_in(&[l.u, l.coef[0], l.coef[1], l.coef[2], l.coef[3], l.coef[4], l.coef[5]], pool);
        let k = util::fresh_symbol("k", &mut taken, pool);
        if let Some(cs) = constant::dispersion_family(l, k, pool) {
            out.push(Ok(Candidate {
                solution: cs.solution,
                method: cs.method.into(),
                general: false,
                parameters: cs.parameters,
                conditions: Vec::new(),
                steps: cs.steps,
                canonical: None,
            }));
        }
    }
    out
}

/// Solve in the characteristic coordinates and map back.
fn via_canonical(
    cf: CanonicalForm,
    opts: &SecondOrderOptions,
    arb: &ArbitraryFunctions,
    reasons: &mut Vec<String>,
    pool: &ExprPool,
) -> Vec<Result<Candidate, PdeError>> {
    if !cf.in_new_variables {
        reasons.push(format!(
            "the {} canonical form keeps coefficients in the old variables",
            cf.pde_type.as_str()
        ));
        return Vec::new();
    }
    let (xi, eta) = cf.new_variables;
    let c = &cf.coefficients;
    let l2 = Linear2 {
        x: xi,
        y: eta,
        u: cf.unknown,
        coef: [c[0], c[1], c[2], c[3], c[4], c[5]],
        rhs: c[6],
    };
    let Ok(eq2) = l2.equation(pool) else {
        return Vec::new();
    };
    let info2 = match super::classify::classify_pde(eq2, l2.u, pool) {
        Ok(i) => i,
        Err(e) => return vec![Err(e)],
    };
    let (phi, psi) = cf.transformation;
    let inner = general_candidates(&l2, &info2, opts, arb, 1, reasons, pool);
    inner
        .into_iter()
        .map(|r| {
            r.map(|cand| {
                let sol = simp(subs2(cand.solution, (xi, phi), (eta, psi), pool), pool);
                let mut conds = cf.conditions.clone();
                for cc in cand.conditions {
                    conds.push(subs2(cc, (xi, phi), (eta, psi), pool));
                }
                let mut steps = vec![format!(
                    "{} canonical form in ξ = {}, η = {}: {} = 0",
                    cf.pde_type.as_str(),
                    pool.display(phi),
                    pool.display(psi),
                    pool.display(cf.equation)
                )];
                steps.extend(cand.steps);
                Candidate {
                    solution: sol,
                    method: format!("canonical_form+{}", cand.method),
                    general: cand.general,
                    parameters: cand.parameters,
                    conditions: conds,
                    steps,
                    canonical: Some(cf.clone()),
                }
            })
        })
        .collect()
}

/// `u = e^{αx+βy} v` removing the first-order terms of a non-parabolic
/// constant-coefficient operator; taken only when the equation for `v` has
/// no zeroth-order term, so that it factors into its principal part.
fn shift_candidate(
    l: &Linear2,
    arb: &mut ArbitraryFunctions,
    pool: &ExprPool,
) -> Option<Result<Candidate, PdeError>> {
    if is_zero(l.dd(), pool) && is_zero(l.e(), pool) {
        return None;
    }
    let (alpha, beta, fp) = constant::exponential_shift(l, pool)?;
    if !is_zero(fp, pool) {
        return None;
    }
    let (x, y) = (l.x, l.y);
    let expo = simp(pool.add(vec![pool.mul(vec![alpha, x]), pool.mul(vec![beta, y])]), pool);
    let zero = pool.integer(0);
    let lv = Linear2 {
        x,
        y,
        u: l.u,
        coef: [l.a(), l.b(), l.c(), zero, zero, zero],
        rhs: simp(pool.mul(vec![l.rhs, exp(neg(expo, pool), pool)]), pool),
    };
    let mut conds = Vec::new();
    let fac = constant::factor_symbol(&lv, &mut conds, pool)?;
    let mut fresh = |arg: ExprId| arb.fresh(vec![arg], pool);
    let cs = match constant::solve_factored(&lv, &fac, &mut fresh, &mut conds, pool) {
        Ok(Some(cs)) => cs,
        Ok(None) => return None,
        Err(e) => return Some(Err(e)),
    };
    let mut steps = vec![format!(
        "u = e^({}) v removes the first-order terms; v solves its principal part",
        pool.display(expo)
    )];
    steps.extend(cs.steps);
    Some(Ok(Candidate {
        solution: simp(pool.mul(vec![exp(expo, pool), cs.solution]), pool),
        method: "exponential_shift".into(),
        general: true,
        parameters: Vec::new(),
        conditions: conds,
        steps,
        canonical: None,
    }))
}

/// An equation with derivatives in one variable only is an ODE with the
/// other as a parameter: `dsolve`, with each constant replaced by an
/// arbitrary function of the other variable.
fn ode_candidate(
    l: &Linear2,
    in_x: bool,
    arb: &mut ArbitraryFunctions,
    pool: &ExprPool,
) -> Option<Candidate> {
    let ll = if in_x { l.clone() } else { l.swapped() };
    // derivatives in x only: B = C = E = 0, A ≠ 0
    if !is_zero(ll.b(), pool) || !is_zero(ll.c(), pool) || !is_zero(ll.e(), pool) || is_zero(ll.a(), pool) {
        return None;
    }
    let (t, other) = (ll.x, ll.y);
    let ysym = pool.symbol("__alkahest_pde2_y", crate::kernel::Domain::Real);
    let (mut input, yp, ypp) = OdeInput::second_order(t, ysym, pool);
    input.equation = simp(
        pool.add(vec![
            pool.mul(vec![ll.a(), ypp]),
            pool.mul(vec![ll.dd(), yp]),
            pool.mul(vec![ll.f(), ysym]),
            neg(ll.rhs, pool),
        ]),
        pool,
    );
    let rep = dsolve_with(&input, &AssumptionContext::new(), pool).ok()?;
    let br = rep.branches.into_iter().find(|b| matches!(b.form, SolutionForm::Explicit(_)))?;
    let SolutionForm::Explicit(mut sol) = br.form else {
        return None;
    };
    let mut steps = vec![format!(
        "an ODE in {} with {} as a parameter ({}); constants become arbitrary functions of {}",
        pool.display(t),
        pool.display(other),
        br.method,
        pool.display(other)
    )];
    for c in br.constants {
        let f = arb.fresh(vec![other], pool);
        sol = util::subs1(sol, c, f, pool);
    }
    for n in rep.notes {
        steps.push(n);
    }
    let mut conds = Vec::new();
    for sc in rep.side_conditions {
        if let crate::deriv::SideCondition::NonZero(e) = sc {
            first_order::note_nonzero(e, &mut conds, pool);
        }
    }
    first_order::note_nonzero(ll.a(), &mut conds, pool);
    Some(Candidate {
        solution: simp(sol, pool),
        method: "ode_in_one_variable".into(),
        general: true,
        parameters: Vec::new(),
        conditions: conds,
        steps,
        canonical: None,
    })
}

#[cfg(test)]
mod tests;
