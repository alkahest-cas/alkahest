//! Transform methods for linear constant-coefficient PDEs (milestone M3).
//!
//! * [`fourier_method`] — `x ∈ ℝ`, `t > 0`: Fourier-transform the PDE in `x`
//!   (M0's derivative-aware [`crate::transform::fourier_transform`]), solve
//!   the ODE in `t` with `dsolve`, impose the transformed initial data, and
//!   invert.  Heat, wave, Schrödinger, telegraph, Klein–Gordon, and Laplace
//!   in the half-plane `y > 0` (one condition at `y = 0` plus boundedness).
//! * [`laplace_method`] — `x > 0`, `t > 0` with zero initial data and
//!   boundary data `u(0, t) = g(t)` (or `u_x(0, t) = g(t)`): Laplace-transform
//!   in `t`, keep the solution of the `x`-ODE that is bounded as `x → ∞`,
//!   and invert — `erfc` for heat with constant data, Duhamel's convolution
//!   for general data, the retarded `g(t − x/c)·θ(t − x/c)` for waves.
//! * [`image_method`] — `x > 0`, `t > 0` with a homogeneous Dirichlet or
//!   Neumann end and initial data: the Fourier **sine** (Dirichlet) or
//!   **cosine** (Neumann) transform ([`crate::transform::sincos`]) and the
//!   image kernel `K(x − s, t) ∓ K(x + s, t)`.
//! * [`transform_solve`] dispatches on the domain and superposes the image
//!   and Laplace parts when a half-line problem has both kinds of data.
//!
//! # Output
//!
//! A closed form when the inverse transform closes, otherwise a **kernel
//! representation** `u = ∫ G(x − s, t) f(s) ds` with `G` closed (heat kernel,
//! Poisson kernel, image kernels, the Duhamel kernel) and
//! `kind = IntegralRepresentation`, and — when no kernel closes (telegraph,
//! Klein–Gordon) — the Fourier-integral representation
//! `u = ∫ e^{2πiξx} H(ξ, t) f̂(ξ) dξ` with the transfer function `H` closed.
//! The integral is the reserved head `Integral(integrand, s, lo, hi)`.
//!
//! # Verification
//!
//! A closed form is verified by substitution in the PDE and every condition
//! (initial conditions of the `erfc` / Duhamel solutions as one-sided limits
//! `t → 0⁺`).  A representation is verified on its kernel: the kernel solves
//! the PDE exactly for every source point `s` (so does the integral, by
//! differentiation under the integral sign — recorded as a side condition on
//! the data), the heat-type kernels have unit mass, image kernels satisfy the
//! boundary condition, and the transfer functions satisfy the transformed
//! initial conditions.  A refuted check is an error (`E-PDE-163`).

use crate::calculus::{limit, LimitDirection};
use crate::diff::undefined::{build, view, UndefView};
use crate::kernel::subs::mentions_var;
use crate::kernel::{Domain, ExprData, ExprId, ExprPool};
use crate::simplify::zero::{zero_test, ZeroStatus};

use super::classify::{classify_pde, Linearity};
use super::error::PdeError;
use super::problem::{ConditionKind, PdeCondition, PdeDomain, PdeProblem};
use super::sturm_liouville::{d, fresh_symbol, replace_node, simp, subs1};
use super::transform::{transformed_name, TransformKind};
use super::verify::{verify_solution, VerificationStatus};

/// Closed form or integral representation.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
#[non_exhaustive]
pub enum SolutionKind {
    /// `u` in closed form.
    ClosedForm,
    /// `u` as integrals of closed kernels against the data.
    IntegralRepresentation,
}

impl SolutionKind {
    /// `"closed_form"` or `"integral_representation"`.
    pub fn as_str(self) -> &'static str {
        match self {
            SolutionKind::ClosedForm => "closed_form",
            SolutionKind::IntegralRepresentation => "integral_representation",
        }
    }
}

/// One term `∫_lo^hi kernel · density d(var)` of a representation.
#[derive(Clone, Debug, PartialEq, Eq)]
#[non_exhaustive]
pub struct KernelTerm {
    /// The kernel, in the problem's variables and `var`.
    pub kernel: ExprId,
    /// The data it integrates (`f(s)`, `g(τ)`, `f̂(ξ)`).
    pub density: ExprId,
    /// The integration variable.
    pub var: ExprId,
    /// Lower limit.
    pub lower: ExprId,
    /// Upper limit.
    pub upper: ExprId,
}

/// One verification check.
#[derive(Clone, Debug, PartialEq, Eq)]
#[non_exhaustive]
pub struct TransformCheck {
    /// What was checked: `"pde"`, `"bc"`, `"ic"`, `"kernel_pde"`,
    /// `"kernel_mass"`, `"kernel_bc"`, `"transfer_ic"`, `"integrand_pde"`.
    pub what: &'static str,
    /// The index among checks of that kind.
    pub index: usize,
    /// The verdict.
    pub status: ZeroStatus,
}

/// The result of a transform method.
#[derive(Clone, Debug, PartialEq, Eq)]
#[non_exhaustive]
pub struct TransformSolution {
    /// Closed form or representation.
    pub kind: SolutionKind,
    /// `"fourier"`, `"laplace"`, `"fourier_sine"`, `"fourier_cosine"`, or a
    /// `+`-joined combination for a superposition.
    pub method: String,
    /// `u` (with `Integral(…)` nodes for a representation).
    pub value: ExprId,
    /// The closed part of `value` (all of it for a closed form).
    pub closed_part: ExprId,
    /// The integral terms (empty for a closed form).
    pub terms: Vec<KernelTerm>,
    /// The solution in transform space (`û(ξ, t)`, `U(x, s)`, `Û_s(ω, t)`).
    pub transformed: ExprId,
    /// The transform variable.
    pub transform_var: ExprId,
    /// The overall verdict: exact only when every check is a proof.
    pub status: VerificationStatus,
    /// The individual checks.
    pub checks: Vec<TransformCheck>,
    /// Conditions on parameters and data (predicates).
    pub side_conditions: Vec<ExprId>,
    /// Prose: the analytic hypotheses of the representation.
    pub notes: Vec<String>,
}

// ---------------------------------------------------------------------------
// Problem analysis
// ---------------------------------------------------------------------------

struct Setup {
    eq: ExprId,
    u: ExprId,
    uv: UndefView,
    indep: Vec<ExprId>,
}

fn setup(problem: &PdeProblem, pool: &ExprPool) -> Result<Setup, PdeError> {
    if problem.equations().len() != 1 || problem.unknowns().len() != 1 {
        return Err(PdeError::TransformMethodUnsupported(
            "transform methods take one equation in one unknown".into(),
        ));
    }
    let eq = problem.equations()[0];
    let u = problem.unknowns()[0];
    let info = classify_pde(eq, u, pool)?;
    if info.linearity != Linearity::Linear
        || info.homogeneous != Some(true)
        || info.constant_coefficients != Some(true)
    {
        return Err(PdeError::TransformMethodUnsupported(format!(
            "{} is not a linear homogeneous constant-coefficient equation",
            pool.display(eq)
        )));
    }
    let uv = super::jet::decode_unknown(u, pool)?;
    if uv.args.len() != 2 {
        return Err(PdeError::TransformMethodUnsupported(
            "transform methods take an unknown of two variables".into(),
        ));
    }
    Ok(Setup {
        eq,
        u,
        indep: uv.args.clone(),
        uv,
    })
}

#[derive(Default)]
struct Roles {
    line: Vec<ExprId>,
    /// `(var, end)` for `var > end`.
    half: Vec<(ExprId, ExprId)>,
}

fn roles(problem: &PdeProblem) -> Result<Roles, PdeError> {
    let mut r = Roles::default();
    fn walk(d: &PdeDomain, r: &mut Roles) -> Result<(), PdeError> {
        match d {
            PdeDomain::Line { var } => r.line.push(*var),
            PdeDomain::HalfLine {
                var,
                end,
                upper: false,
            } => r.half.push((*var, *end)),
            PdeDomain::Product(ps) => {
                for p in ps {
                    walk(p, r)?;
                }
            }
            other => {
                return Err(PdeError::TransformMethodUnsupported(format!(
                    "transform methods need lines and half-lines (x > a), not a {}",
                    other.kind()
                )))
            }
        }
        Ok(())
    }
    let dom = problem.domain().ok_or_else(|| {
        PdeError::TransformMethodUnsupported(
            "declare the domain with .on(Product([Line(x), HalfLine(t, 0)]))".into(),
        )
    })?;
    walk(dom, &mut r)?;
    Ok(r)
}

/// A condition `∂^α u (… , a, …) = rhs` with one argument fixed at `a`.
#[derive(Clone, Debug)]
struct Datum {
    kind: ConditionKind,
    slot: usize,
    at: ExprId,
    orders: Vec<u32>,
    rhs: ExprId,
}

fn datum(c: &PdeCondition, s: &Setup, pool: &ExprPool) -> Result<Datum, PdeError> {
    let bad = || {
        PdeError::TransformData(format!(
            "{} = {}: a transform method needs u or one derivative of u with one variable fixed",
            pool.display(c.lhs),
            pool.display(c.rhs)
        ))
    };
    let v = view(c.lhs, pool).ok_or_else(bad)?;
    if v.name != s.uv.name || v.args.len() != s.indep.len() {
        return Err(bad());
    }
    let fixed: Vec<usize> = (0..v.args.len())
        .filter(|&j| v.args[j] != s.indep[j])
        .collect();
    if fixed.len() != 1 {
        return Err(bad());
    }
    let slot = fixed[0];
    let at = v.args[slot];
    if s.indep.iter().any(|&x| mentions_var(at, x, pool)) {
        return Err(bad());
    }
    Ok(Datum {
        kind: c.kind,
        slot,
        at,
        orders: v.orders,
        rhs: c.rhs,
    })
}

fn is_zero(e: ExprId, pool: &ExprPool) -> bool {
    let s = simp(e, pool);
    s == pool.integer(0) || zero_test(s, pool, None).status == ZeroStatus::ProvedZero
}

fn zstatus(e: ExprId, pool: &ExprPool) -> ZeroStatus {
    let s = simp(e, pool);
    if s == pool.integer(0) {
        return ZeroStatus::ProvedZero;
    }
    let st = zero_test(s, pool, None).status;
    if st == ZeroStatus::ProvedZero {
        return st;
    }
    if prove_harder(s, pool) {
        ZeroStatus::ProvedZero
    } else {
        st
    }
}

/// `(∏ fᵢ^{eᵢ})^r → ∏ fᵢ^{eᵢ r}` and `√z → z^{1/2}` when every factor is
/// structurally positive — valid exactly there, and what lets `simplify`
/// merge `P^{1/2}·(P^{−1})^{1/2}` into 1.
fn expand_positive_powers(e: ExprId, pool: &ExprPool) -> ExprId {
    use super::sturm_liouville::{sign, Sgn};
    fn split(b: ExprId, pool: &ExprPool) -> Option<Vec<(ExprId, ExprId)>> {
        let factors = match pool.get(b) {
            ExprData::Mul(xs) => xs,
            _ => vec![b],
        };
        let mut out = Vec::new();
        for f in factors {
            let (base, ex) = match pool.get(f) {
                ExprData::Pow { base, exp } => (base, exp),
                _ => (f, pool.integer(1)),
            };
            if sign(base, pool) != Sgn::Pos {
                return None;
            }
            out.push((base, ex));
        }
        Some(out)
    }
    fn go(e: ExprId, pool: &ExprPool) -> ExprId {
        match pool.get(e) {
            ExprData::Add(xs) => pool.add(xs.iter().map(|&x| go(x, pool)).collect()),
            ExprData::Mul(xs) => pool.mul(xs.iter().map(|&x| go(x, pool)).collect()),
            ExprData::Func { name, args } if name == "sqrt" && args.len() == 1 => {
                go(pool.pow(args[0], pool.rational(1, 2)), pool)
            }
            ExprData::Func { name, args } => {
                pool.func(name, args.iter().map(|&x| go(x, pool)).collect())
            }
            ExprData::Pow { base, exp } => {
                let base = go(base, pool);
                let is_frac = pool.with(exp, |d| matches!(d, ExprData::Rational(_)));
                if is_frac {
                    if let Some(parts) = split(base, pool) {
                        return pool.mul(
                            parts
                                .into_iter()
                                .map(|(b, ex)| pool.pow(b, simp(pool.mul(vec![ex, exp]), pool)))
                                .collect(),
                        );
                    }
                }
                pool.pow(base, go(exp, pool))
            }
            _ => e,
        }
    }
    go(e, pool)
}

/// Second attempts at *proving* a residual zero (never at refuting it):
///
/// * `π` as a generic positive symbol — an identity for every `P > 0` holds
///   at `P = π`, and with a `Positive` symbol `(P/P²)^{1/2} = P^{−1/2}` folds
///   where `(π/π²)^{1/2}` does not;
/// * `|v|` split by sign — for each real symbol `v` under an `abs`, the
///   residual with `v = P` and with `v = −P` (`P > 0`).
fn prove_harder(e: ExprId, pool: &ExprPool) -> bool {
    let pp = fresh_symbol("__tm_pi", Domain::Positive, &[e], pool);
    let e1 = simp(
        expand_positive_powers(simp(subs1(e, pi(pool), pp, pool), pool), pool),
        pool,
    );
    if e1 == pool.integer(0) || zero_test(e1, pool, None).status == ZeroStatus::ProvedZero {
        return true;
    }
    // Real symbols under abs(·).
    let mut vars = Vec::new();
    let mut stack = vec![e1];
    let mut seen = std::collections::HashSet::new();
    while let Some(n) = stack.pop() {
        if !seen.insert(n) {
            continue;
        }
        if let ExprData::Func { name, args } = pool.get(n) {
            if name == "abs" && args.len() == 1 {
                if let ExprData::Symbol { .. } = pool.get(args[0]) {
                    if !vars.contains(&args[0]) {
                        vars.push(args[0]);
                    }
                }
            }
        }
        super::jet::push_children(n, pool, &mut stack);
    }
    if vars.len() != 1 {
        return false;
    }
    let v = vars[0];
    let pv = fresh_symbol("__tm_pos", Domain::Positive, &[e1], pool);
    [pv, neg(pv, pool)].iter().all(|&val| {
        // |v| = P on both sides; then v = ±P.
        let r = replace_node(e1, pool.func("abs", vec![v]), pv, pool);
        let r = simp(
            expand_positive_powers(simp(subs1(r, v, val, pool), pool), pool),
            pool,
        );
        r == pool.integer(0) || zero_test(r, pool, None).status == ZeroStatus::ProvedZero
    })
}

fn combine(checks: &[TransformCheck]) -> VerificationStatus {
    if checks.iter().all(|c| c.status == ZeroStatus::ProvedZero) {
        VerificationStatus::ExactlyVerified
    } else if checks.iter().any(|c| c.status == ZeroStatus::ProvedNonzero) {
        VerificationStatus::Refuted
    } else if checks
        .iter()
        .all(|c| matches!(c.status, ZeroStatus::ProvedZero | ZeroStatus::ProbablyZero))
    {
        VerificationStatus::NumericallyChecked
    } else {
        VerificationStatus::Unverified
    }
}

fn status_to_zero(s: VerificationStatus) -> ZeroStatus {
    match s {
        VerificationStatus::ExactlyVerified => ZeroStatus::ProvedZero,
        VerificationStatus::Refuted => ZeroStatus::ProvedNonzero,
        VerificationStatus::NumericallyChecked => ZeroStatus::ProbablyZero,
        _ => ZeroStatus::Unknown,
    }
}

fn neg(e: ExprId, pool: &ExprPool) -> ExprId {
    pool.mul(vec![pool.integer(-1), e])
}

fn sub(a: ExprId, b: ExprId, pool: &ExprPool) -> ExprId {
    pool.add(vec![a, neg(b, pool)])
}

fn pi(pool: &ExprPool) -> ExprId {
    pool.symbol("pi", Domain::Real)
}

/// The name of an applied undefined function the data is (`f` for
/// `f(x)`), for naming its transform.
fn data_name(e: ExprId, pool: &ExprPool) -> Option<String> {
    view(e, pool)
        .filter(|v| v.orders.iter().all(|&k| k == 0))
        .map(|v| v.name)
}

// ---------------------------------------------------------------------------
// The transformed ODE
// ---------------------------------------------------------------------------

/// Replace `hat` and its derivatives in slot `slot` by `y, y', y'', …`;
/// returns the `OdeInput` (in `var`) and its order.
fn to_ode(
    expr: ExprId,
    hat: ExprId,
    slot: usize,
    var: ExprId,
    pool: &ExprPool,
) -> Result<crate::ode::dsolve::OdeInput, PdeError> {
    let hv = view(hat, pool).expect("applied");
    // Collect the nodes.
    let mut order = 0u32;
    let mut stack = vec![expr];
    let mut seen = std::collections::HashSet::new();
    while let Some(n) = stack.pop() {
        if !seen.insert(n) {
            continue;
        }
        if let Some(v) = view(n, pool) {
            if v.name == hv.name {
                if v.args != hv.args
                    || v.orders
                        .iter()
                        .enumerate()
                        .any(|(j, &k)| j != slot && k > 0)
                {
                    return Err(PdeError::TransformOde(format!(
                        "{} occurs other than as a derivative in {}",
                        pool.display(n),
                        pool.display(var)
                    )));
                }
                order = order.max(v.orders[slot]);
            }
            continue;
        }
        super::jet::push_children(n, pool, &mut stack);
    }
    if order == 0 {
        return Err(PdeError::TransformOde(format!(
            "the transformed equation {} has no derivative in {}",
            pool.display(expr),
            pool.display(var)
        )));
    }
    let y = fresh_symbol("y", Domain::Complex, &[expr], pool);
    let (inp, derivs) = crate::ode::dsolve::OdeInput::higher_order(var, y, order as usize, pool);
    let mut e = expr;
    for k in (1..=order).rev() {
        let mut ord = vec![0; hv.args.len()];
        ord[slot] = k;
        e = replace_node(e, build(hat, &ord, pool), derivs[k as usize - 1], pool);
    }
    e = replace_node(e, hat, y, pool);
    Ok(inp.with_equation(simp(e, pool)))
}

/// The general solution `Σ Cᵢ yᵢ` of the ODE: the basis and the side
/// conditions / notes `dsolve_with` attached.
fn solve_ode(
    inp: &crate::ode::dsolve::OdeInput,
    pool: &ExprPool,
) -> Result<(Vec<ExprId>, Vec<String>), PdeError> {
    let rep = crate::ode::dsolve::dsolve_with(
        inp,
        &crate::simplify::assumptions::AssumptionContext::new(),
        pool,
    )
    .map_err(|e| PdeError::TransformOde(format!("dsolve: {e}")))?;
    let br = rep
        .branches
        .iter()
        .find(|b| b.is_explicit())
        .ok_or_else(|| {
            PdeError::TransformOde("no explicit solution of the transformed ODE".into())
        })?;
    let sol = br.y_of_x().expect("explicit");
    let mut basis = Vec::new();
    for &c in &br.constants {
        basis.push(simp(d(sol, c, pool)?, pool));
    }
    let lin: Vec<ExprId> = br
        .constants
        .iter()
        .zip(&basis)
        .map(|(&c, &b)| pool.mul(vec![c, b]))
        .collect();
    if !is_zero(sub(sol, pool.add(lin), pool), pool)
        || basis
            .iter()
            .any(|&b| br.constants.iter().any(|&c| mentions_var(b, c, pool)))
    {
        return Err(PdeError::TransformOde(
            "the general solution is not linear in its constants".into(),
        ));
    }
    let mut notes = rep.notes.clone();
    for sc in &rep.side_conditions {
        notes.push(format!("dsolve: {}", sc.display_with(pool)));
    }
    Ok((basis, notes))
}

/// Solve `Σᵢ Cᵢ ∂ᵥʲ yᵢ(0) = Pⱼ` for the transfer functions `H_j` (the
/// coefficient of `Pⱼ` in `Σ Cᵢ yᵢ`).  Orders 1 and 2.
fn transfer_functions(
    basis: &[ExprId],
    var: ExprId,
    at: ExprId,
    pool: &ExprPool,
) -> Result<Vec<ExprId>, PdeError> {
    let k = basis.len();
    let mut m = vec![vec![pool.integer(0); k]; k];
    for (i, &y) in basis.iter().enumerate() {
        let mut dy = y;
        for row in m.iter_mut() {
            row[i] = simp(subs1(dy, var, at, pool), pool);
            dy = d(dy, var, pool)?;
        }
    }
    match k {
        1 => Ok(vec![simp(
            pool.mul(vec![basis[0], pool.pow(m[0][0], pool.integer(-1))]),
            pool,
        )]),
        2 => {
            let det = simp(
                sub(
                    pool.mul(vec![m[0][0], m[1][1]]),
                    pool.mul(vec![m[0][1], m[1][0]]),
                    pool,
                ),
                pool,
            );
            if is_zero(det, pool) {
                return Err(PdeError::TransformOde(
                    "the transformed initial-value problem is singular".into(),
                ));
            }
            let inv = pool.pow(det, pool.integer(-1));
            // C = M⁻¹ P: H₀ = (m11 y0 − m10 y1)/det, H₁ = (−m01 y0 + m00 y1)/det.
            let h0 = simp(
                pool.mul(vec![
                    sub(
                        pool.mul(vec![m[1][1], basis[0]]),
                        pool.mul(vec![m[1][0], basis[1]]),
                        pool,
                    ),
                    inv,
                ]),
                pool,
            );
            let h1 = simp(
                pool.mul(vec![
                    sub(
                        pool.mul(vec![m[0][0], basis[1]]),
                        pool.mul(vec![m[0][1], basis[0]]),
                        pool,
                    ),
                    inv,
                ]),
                pool,
            );
            Ok(vec![h0, h1])
        }
        _ => Err(PdeError::TransformOde(format!(
            "transformed ODE of order {k}; orders 1 and 2 are implemented"
        ))),
    }
}

/// `r/ξ`, reading `√(ξ²)` and `|ξ|` as `ξ` (the sign is chosen after).
fn rate_over_xi(r: ExprId, xi: ExprId, pool: &ExprPool) -> ExprId {
    let r = replace_node(
        r,
        pool.func("sqrt", vec![pool.pow(xi, pool.integer(2))]),
        xi,
        pool,
    );
    let r = replace_node(r, pool.func("abs", vec![xi]), xi, pool);
    constant_in(
        simp(pool.mul(vec![r, pool.pow(xi, pool.integer(-1))]), pool),
        xi,
        pool,
    )
}

/// `e` itself, or — when `e` mentions `v` but `∂e/∂v` is proved zero — its
/// value at `v = 1` (a constant written with a removable `v`).
fn constant_in(e: ExprId, v: ExprId, pool: &ExprPool) -> ExprId {
    if !mentions_var(e, v, pool) {
        return e;
    }
    match d(e, v, pool) {
        Ok(de) if is_zero(de, pool) => simp(subs1(e, v, pool.integer(1), pool), pool),
        _ => e,
    }
}

/// `exp(a)·exp(b)·… → exp(a + b + …)` in a product.
fn merge_exps(e: ExprId, pool: &ExprPool) -> ExprId {
    let ExprData::Mul(xs) = pool.get(e) else {
        return e;
    };
    let mut args = Vec::new();
    let mut rest = Vec::new();
    for f in xs {
        match pool.get(f) {
            ExprData::Func { name, args: a } if name == "exp" && a.len() == 1 => args.push(a[0]),
            _ => rest.push(f),
        }
    }
    if args.len() < 2 {
        return e;
    }
    rest.push(pool.func("exp", vec![simp(pool.add(args), pool)]));
    pool.mul(rest)
}

/// `exp(r·v)` basis element → `r`.
fn exponent_rate(y: ExprId, v: ExprId, pool: &ExprPool) -> Option<ExprId> {
    let r = simp(
        pool.mul(vec![d(y, v, pool).ok()?, pool.pow(y, pool.integer(-1))]),
        pool,
    );
    (!mentions_var(r, v, pool)).then_some(r)
}

// ---------------------------------------------------------------------------
// Antiderivative bookkeeping for `∫_a^b g(s) ds` with symbolic g
// ---------------------------------------------------------------------------

/// `Integral(body, s, lo, hi)`.
fn integral(body: ExprId, s: ExprId, lo: ExprId, hi: ExprId, pool: &ExprPool) -> ExprId {
    pool.func("Integral", vec![body, s, lo, hi])
}

/// For verification: an antiderivative `G` of `g` stands for `∫ g`; every
/// `Derivative(G(a), k ≥ 1)` in a residual is `g^{(k−1)}(a)`.
fn resolve_antiderivative(
    e: ExprId,
    big: &str,
    g: ExprId,
    z: ExprId,
    pool: &ExprPool,
) -> Result<ExprId, PdeError> {
    let mut nodes = Vec::new();
    let mut stack = vec![e];
    let mut seen = std::collections::HashSet::new();
    while let Some(n) = stack.pop() {
        if !seen.insert(n) {
            continue;
        }
        if let Some(v) = view(n, pool) {
            if v.name == big && v.orders[0] >= 1 {
                nodes.push((n, v.args[0], v.orders[0]));
            }
            stack.extend(v.args);
            continue;
        }
        super::jet::push_children(n, pool, &mut stack);
    }
    let mut out = e;
    for (node, arg, k) in nodes {
        let mut gk = g;
        for _ in 1..k {
            gk = d(gk, z, pool)?;
        }
        out = replace_node(out, node, subs1(gk, z, arg, pool), pool);
    }
    Ok(simp(out, pool))
}

// ---------------------------------------------------------------------------
// Fourier method
// ---------------------------------------------------------------------------

/// How one transfer function inverts.
enum Inverse {
    /// `u_j` closed (data concrete, or shifts of symbolic data).
    Closed {
        value: ExprId,
        /// For verification: `value` with `Integral`s replaced by
        /// antiderivatives, and `(G name, g(z), z)` to resolve.
        verify_value: ExprId,
        antider: Option<(String, ExprId, ExprId)>,
    },
    /// `∫ K(x − s, t) f(s) ds` with `K(x, t)` closed.
    Kernel(ExprId),
    /// `∫ e^{2πiξx} H(ξ, t) f̂(ξ) dξ`.
    Spectral,
}

/// `A cos(2πdξ) + B sin(2πdξ)/ξ` with `A`, `B`, `d > 0` free of `ξ`:
/// `(A, B, d)`.  `|ξ|` is read as `ξ` (both pieces are even in `ξ`).
fn shift_form(h: ExprId, xi: ExprId, pool: &ExprPool) -> Option<(ExprId, ExprId, ExprId)> {
    let h = simp(replace_node(h, pool.func("abs", vec![xi]), xi, pool), pool);
    let mut args = Vec::new();
    let mut stack = vec![h];
    let mut seen = std::collections::HashSet::new();
    while let Some(n) = stack.pop() {
        if !seen.insert(n) {
            continue;
        }
        if let ExprData::Func { name, args: a } = pool.get(n) {
            if (name == "sin" || name == "cos") && mentions_var(a[0], xi, pool) {
                if !args.contains(&a[0]) {
                    args.push(a[0]);
                }
                continue;
            }
        }
        super::jet::push_children(n, pool, &mut stack);
    }
    if args.len() != 1 {
        return None;
    }
    let theta = args[0];
    let mut kappa = constant_in(
        simp(pool.mul(vec![theta, pool.pow(xi, pool.integer(-1))]), pool),
        xi,
        pool,
    );
    if mentions_var(kappa, xi, pool) {
        return None;
    }
    let s_sym = fresh_symbol("__tm_S", Domain::Real, &[h], pool);
    let c_sym = fresh_symbol("__tm_C", Domain::Real, &[h], pool);
    let e = replace_node(h, pool.func("sin", vec![theta]), s_sym, pool);
    let e = replace_node(e, pool.func("cos", vec![theta]), c_sym, pool);
    let a = constant_in(simp(d(e, c_sym, pool).ok()?, pool), xi, pool);
    let bs = simp(d(e, s_sym, pool).ok()?, pool);
    let b = constant_in(simp(pool.mul(vec![bs, xi]), pool), xi, pool);
    for c in [a, b] {
        if mentions_var(c, xi, pool) || mentions_var(c, s_sym, pool) || mentions_var(c, c_sym, pool)
        {
            return None;
        }
    }
    let rest = sub(
        e,
        pool.add(vec![pool.mul(vec![a, c_sym]), pool.mul(vec![bs, s_sym])]),
        pool,
    );
    if !is_zero(rest, pool) {
        return None;
    }
    let mut b = b;
    match super::sturm_liouville::sign(kappa, pool) {
        super::sturm_liouville::Sgn::Pos => {}
        super::sturm_liouville::Sgn::Neg => {
            kappa = simp(neg(kappa, pool), pool);
            b = simp(neg(b, pool), pool);
        }
        _ => return None,
    }
    let dd = simp(
        pool.mul(vec![
            kappa,
            pool.pow(pool.mul(vec![pool.integer(2), pi(pool)]), pool.integer(-1)),
        ]),
        pool,
    );
    Some((a, b, dd))
}

/// Solve on `x ∈ ℝ`, `t > 0` by the Fourier transform in `x`.
///
/// # Errors
///
/// `E-PDE-160` (not linear constant-coefficient homogeneous, or not on
/// `ℝ × (0, ∞)`), `E-PDE-161` (initial data), `E-PDE-162` (the transformed
/// ODE), `E-PDE-163` (a refuted candidate).
pub fn fourier_method(
    problem: &PdeProblem,
    pool: &ExprPool,
) -> Result<TransformSolution, PdeError> {
    let s = setup(problem, pool)?;
    let r = roles(problem)?;
    let (x, t) = match (r.line.as_slice(), r.half.as_slice()) {
        ([x], [(t, end)]) if is_zero(*end, pool) => (*x, *t),
        _ => {
            return Err(PdeError::TransformMethodUnsupported(
                "the Fourier method needs x on a line and t on a half-line t > 0".into(),
            ))
        }
    };
    let tslot = s
        .indep
        .iter()
        .position(|&v| v == t)
        .expect("t is a variable");
    let avoid: Vec<ExprId> = vec![s.eq, x, t]
        .into_iter()
        .chain(problem.conditions().iter().flat_map(|c| [c.lhs, c.rhs]))
        .collect();
    let xi = fresh_symbol("xi", Domain::Real, &avoid, pool);
    let ft = crate::transform::fourier_transform(s.eq, x, xi, pool)
        .map_err(|e| PdeError::TransformOde(format!("Fourier transform of the PDE: {e}")))?;
    let mut hat_args = s.indep.clone();
    hat_args[s.indep.iter().position(|&v| v == x).expect("x")] = xi;
    let hat = pool.func(
        transformed_name(&s.uv.name, TransformKind::Fourier),
        hat_args,
    );
    let inp = to_ode(ft, hat, tslot, t, pool)?;
    let k = inp.order();
    let (basis, mut notes) = solve_ode(&inp, pool)?;
    // Initial data at t = 0, ordered by t-derivative.
    let mut data: Vec<Option<ExprId>> = vec![None; k];
    for c in problem.conditions() {
        let dt = datum(c, &s, pool)?;
        if dt.slot != tslot || !is_zero(dt.at, pool) {
            return Err(PdeError::TransformData(format!(
                "{}: the Fourier method takes data at t = 0 only",
                pool.display(c.lhs)
            )));
        }
        let j = dt.orders[tslot] as usize;
        if dt
            .orders
            .iter()
            .enumerate()
            .any(|(i, &o)| i != tslot && o > 0)
            || j >= k
            || data[j].is_some()
        {
            return Err(PdeError::TransformData(format!(
                "{}: expected u(x, 0){}",
                pool.display(c.lhs),
                if k > 1 { " and u_t(x, 0)" } else { "" }
            )));
        }
        data[j] = Some(dt.rhs);
    }
    let m = data.iter().take_while(|d| d.is_some()).count();
    if m == 0 || data.iter().skip(m).any(|d| d.is_some()) {
        return Err(PdeError::TransformData(
            "initial data must be u(x, 0) [and u_t(x, 0)] at t = 0".into(),
        ));
    }
    let data: Vec<ExprId> = data.into_iter().take(m).map(|d| d.unwrap()).collect();
    let mut side = Vec::new();
    // Transfer functions.
    let transfers = if m == k {
        transfer_functions(&basis, t, pool.integer(0), pool)?
    } else if k == 2 && m == 1 {
        // Boundedness in t: e^{±ρξt} → e^{−|ρ||ξ|t}.
        let rates: Vec<ExprId> = basis
            .iter()
            .map(|&y| exponent_rate(y, t, pool))
            .collect::<Option<_>>()
            .ok_or_else(|| {
                PdeError::TransformData(
                    "one condition for a second-order equation needs exponential solutions to select the bounded one"
                        .into(),
                )
            })?;
        let rho = rate_over_xi(rates[0], xi, pool);
        let rho2 = rate_over_xi(rates[1], xi, pool);
        if mentions_var(rho, xi, pool) || !is_zero(pool.add(vec![rho, rho2]), pool) {
            return Err(PdeError::TransformData(
                "the bounded solution is not e^{−c|ξ|t}; give a second initial condition".into(),
            ));
        }
        let mag = match super::sturm_liouville::sign(rho, pool) {
            super::sturm_liouville::Sgn::Pos => rho,
            super::sturm_liouville::Sgn::Neg => simp(neg(rho, pool), pool),
            _ => {
                return Err(PdeError::TransformData(
                    "cannot decide which solution of the transformed ODE is bounded".into(),
                ))
            }
        };
        let h = pool.func(
            "exp",
            vec![neg(
                pool.mul(vec![mag, pool.func("abs", vec![xi]), t]),
                pool,
            )],
        );
        notes.push("the solution is selected bounded as t → ∞".into());
        vec![h]
    } else {
        return Err(PdeError::TransformData(format!(
            "{m} initial conditions for an equation of order {k} in t"
        )));
    };
    // The transformed equation holds for every transfer function.
    let mut checks = Vec::new();
    for (j, &h) in transfers.iter().enumerate() {
        let r = replace_node(inp.equation, inp.y, h, pool);
        let mut r = r;
        let mut dh = h;
        for &dsym in &inp.derivs {
            dh = d(dh, t, pool)?;
            r = replace_node(r, dsym, dh, pool);
        }
        checks.push(TransformCheck {
            what: "transfer_ode",
            index: j,
            status: zstatus(r, pool),
        });
        // ∂ₜⁱ H_j(ξ, 0) = δᵢⱼ (only for the IVP case).
        if m == k {
            let mut dh = h;
            for i in 0..k {
                let at = subs1(dh, t, pool.integer(0), pool);
                let want = pool.integer(i64::from(i == j));
                checks.push(TransformCheck {
                    what: "transfer_ic",
                    index: j * k + i,
                    status: zstatus(sub(at, want, pool), pool),
                });
                dh = d(dh, t, pool)?;
            }
        } else {
            checks.push(TransformCheck {
                what: "transfer_ic",
                index: j,
                status: zstatus(
                    sub(subs1(h, t, pool.integer(0), pool), pool.integer(1), pool),
                    pool,
                ),
            });
        }
    }
    // Invert each piece.
    let sv = fresh_symbol("s", Domain::Real, &avoid, pool);
    let oo = pool.pos_infinity();
    let mut closed = Vec::new();
    let mut verify_closed = Vec::new();
    let mut antiders = Vec::new();
    let mut terms = Vec::new();
    let mut transformed = Vec::new();
    let mut methods_note = Vec::new();
    for (j, (&h, &f)) in transfers.iter().zip(&data).enumerate() {
        let concrete = !crate::diff::undefined::has_undefined_function(f, pool);
        let fhat = if concrete {
            crate::transform::fourier_transform(f, x, xi, pool).ok()
        } else {
            None
        };
        let placeholder = pool.func(
            format!(
                "{}_hat",
                data_name(f, pool).unwrap_or_else(|| format!("f{j}"))
            ),
            vec![xi],
        );
        transformed.push(pool.mul(vec![h, fhat.unwrap_or(placeholder)]));
        let inv = invert_fourier(h, fhat, f, x, t, xi, sv, j, pool)?;
        match inv {
            Inverse::Closed {
                value,
                verify_value,
                antider,
            } => {
                closed.push(value);
                verify_closed.push(verify_value);
                if let Some(a) = antider {
                    antiders.push(a);
                }
            }
            Inverse::Kernel(kern) => {
                let st = verify_solution(&[s.eq], &[s.u], &[kern], &[], pool)?;
                checks.push(TransformCheck {
                    what: "kernel_pde",
                    index: j,
                    status: status_to_zero(st.status),
                });
                if j == 0 {
                    if let Ok(mass) =
                        crate::integrate::integrate_definite(kern, x, neg(oo, pool), oo, pool)
                    {
                        checks.push(TransformCheck {
                            what: "kernel_mass",
                            index: j,
                            status: match zstatus(sub(mass.value, pool.integer(1), pool), pool) {
                                ZeroStatus::ProvedZero => ZeroStatus::ProvedZero,
                                // A failed mass check is never a refutation
                                // of the kernel's PDE: report it as unknown.
                                _ => ZeroStatus::Unknown,
                            },
                        });
                    } else {
                        notes.push(
                            "the kernel's unit mass (the initial condition as t → 0⁺) was not checked: \
                             ∫ K dx did not close"
                                .into(),
                        );
                    }
                }
                let shifted = simp(subs1(kern, x, sub(x, sv, pool), pool), pool);
                let dens = subs1(f, x, sv, pool);
                terms.push(KernelTerm {
                    kernel: shifted,
                    density: dens,
                    var: sv,
                    lower: neg(oo, pool),
                    upper: oo,
                });
                methods_note.push("kernel");
            }
            Inverse::Spectral => {
                let e = pool.func(
                    "exp",
                    vec![pool.mul(vec![
                        pool.integer(2),
                        pi(pool),
                        pool.imaginary_unit(),
                        xi,
                        x,
                    ])],
                );
                let integrand = simp(pool.mul(vec![e, h]), pool);
                let st = verify_solution(&[s.eq], &[s.u], &[integrand], &[], pool)?;
                checks.push(TransformCheck {
                    what: "integrand_pde",
                    index: j,
                    status: status_to_zero(st.status),
                });
                let dens = fhat.unwrap_or(placeholder);
                terms.push(KernelTerm {
                    kernel: integrand,
                    density: dens,
                    var: xi,
                    lower: neg(oo, pool),
                    upper: oo,
                });
                methods_note.push("spectral");
            }
        }
    }
    let transformed = simp(pool.add(transformed), pool);
    finish(
        problem,
        &s,
        "fourier".into(),
        closed,
        verify_closed,
        antiders,
        terms,
        transformed,
        xi,
        checks,
        side.drain(..).collect(),
        notes,
        pool,
    )
}

/// Invert `H_j · f̂_j`.
#[allow(clippy::too_many_arguments)]
fn invert_fourier(
    h: ExprId,
    fhat: Option<ExprId>,
    f: ExprId,
    x: ExprId,
    t: ExprId,
    xi: ExprId,
    sv: ExprId,
    j: usize,
    pool: &ExprPool,
) -> Result<Inverse, PdeError> {
    let _ = (t, j);
    if let Some(fh) = fhat {
        let prod = merge_exps(simp(pool.mul(vec![h, fh]), pool), pool);
        if let Ok(v) = crate::transform::inverse_fourier_transform(prod, xi, x, pool) {
            let v = simp(v, pool);
            return Ok(Inverse::Closed {
                value: v,
                verify_value: v,
                antider: None,
            });
        }
    }
    if let Some((a, b, dd)) = shift_form(h, xi, pool) {
        // A (f(x−d) + f(x+d))/2 + Bπ ∫_{x−d}^{x+d} f(s) ds.
        let half = pool.rational(1, 2);
        let mut parts = Vec::new();
        let mut vparts = Vec::new();
        let mut antider = None;
        if !is_zero(a, pool) {
            let avg = pool.mul(vec![
                a,
                half,
                pool.add(vec![
                    subs1(f, x, sub(x, dd, pool), pool),
                    subs1(f, x, pool.add(vec![x, dd]), pool),
                ]),
            ]);
            parts.push(avg);
            vparts.push(avg);
        }
        if !is_zero(b, pool) {
            let coef = pool.mul(vec![b, pi(pool)]);
            let (lo, hi) = (sub(x, dd, pool), pool.add(vec![x, dd]));
            let fs = subs1(f, x, sv, pool);
            let closed_int = if crate::diff::undefined::has_undefined_function(f, pool) {
                None
            } else {
                crate::integrate::integrate_definite(fs, sv, lo, hi, pool)
                    .ok()
                    .map(|r| r.value)
            };
            match closed_int {
                Some(v) => {
                    parts.push(pool.mul(vec![coef, v]));
                    vparts.push(pool.mul(vec![coef, v]));
                }
                None => {
                    parts.push(pool.mul(vec![coef, integral(fs, sv, lo, hi, pool)]));
                    let big = fresh_symbol("__tm_G", Domain::Real, &[f, x], pool);
                    let gname = match pool.get(big) {
                        ExprData::Symbol { name, .. } => name,
                        _ => unreachable!(),
                    };
                    let gnode = |arg| pool.func(gname.clone(), vec![arg]);
                    vparts.push(pool.mul(vec![coef, sub(gnode(hi), gnode(lo), pool)]));
                    let z = fresh_symbol("__tm_z", Domain::Real, &[f, x], pool);
                    antider = Some((gname.clone(), subs1(f, x, z, pool), z));
                }
            }
        }
        let v = simp(pool.add(parts), pool);
        return Ok(Inverse::Closed {
            value: v,
            verify_value: simp(pool.add(vparts), pool),
            antider,
        });
    }
    if let Ok(k) = crate::transform::inverse_fourier_transform(h, xi, x, pool) {
        return Ok(Inverse::Kernel(simp(k, pool)));
    }
    Ok(Inverse::Spectral)
}

/// Assemble a solution and verify the closed part / the whole closed form.
#[allow(clippy::too_many_arguments)]
fn finish(
    problem: &PdeProblem,
    s: &Setup,
    method: String,
    closed: Vec<ExprId>,
    verify_closed: Vec<ExprId>,
    antiders: Vec<(String, ExprId, ExprId)>,
    terms: Vec<KernelTerm>,
    transformed: ExprId,
    tvar: ExprId,
    mut checks: Vec<TransformCheck>,
    side: Vec<ExprId>,
    mut notes: Vec<String>,
    pool: &ExprPool,
) -> Result<TransformSolution, PdeError> {
    let zero = pool.integer(0);
    let closed_part = if closed.is_empty() {
        zero
    } else {
        simp(pool.add(closed), pool)
    };
    let verify_part = if verify_closed.is_empty() {
        zero
    } else {
        simp(pool.add(verify_closed), pool)
    };
    let kind = if terms.is_empty() {
        SolutionKind::ClosedForm
    } else {
        SolutionKind::IntegralRepresentation
    };
    if !closed_part.eq(&zero) || kind == SolutionKind::ClosedForm {
        // The closed part solves the PDE; with no integral terms it must
        // also meet every condition.
        let conds: Vec<PdeCondition> = if kind == SolutionKind::ClosedForm {
            problem.conditions().to_vec()
        } else {
            Vec::new()
        };
        let v = verify_solution(&[s.eq], &[s.u], &[verify_part], &conds, pool)?;
        for c in v.checks {
            let mut st = c.status;
            if st != ZeroStatus::ProvedZero && prove_harder(c.residual, pool) {
                st = ZeroStatus::ProvedZero;
            }
            if st != ZeroStatus::ProvedZero && !antiders.is_empty() {
                let mut r = c.residual;
                for (g, body, z) in &antiders {
                    r = resolve_antiderivative(r, g, *body, *z, pool)?;
                }
                st = zstatus(r, pool);
            }
            checks.push(TransformCheck {
                what: c.kind,
                index: c.index,
                status: st,
            });
        }
    }
    if kind == SolutionKind::IntegralRepresentation {
        notes.push(
            "the integral solves the problem by differentiation under the integral sign: the data \
             must be integrable (and decay) so that the kernel's derivatives are dominated"
                .into(),
        );
    }
    let status = combine(&checks);
    if status == VerificationStatus::Refuted {
        let bad: Vec<String> = checks
            .iter()
            .filter(|c| c.status == ZeroStatus::ProvedNonzero)
            .map(|c| format!("{}[{}]", c.what, c.index))
            .collect();
        return Err(PdeError::TransformVerificationFailed(format!(
            "the {method} candidate failed {}",
            bad.join(", ")
        )));
    }
    let mut value_parts = vec![closed_part];
    for tm in &terms {
        value_parts.push(integral(
            pool.mul(vec![tm.kernel, tm.density]),
            tm.var,
            tm.lower,
            tm.upper,
            pool,
        ));
    }
    let value = if value_parts.len() == 1 {
        closed_part
    } else {
        pool.add(value_parts.into_iter().filter(|&e| e != zero).collect())
    };
    Ok(TransformSolution {
        kind,
        method,
        value,
        closed_part,
        terms,
        transformed,
        transform_var: tvar,
        status,
        checks,
        side_conditions: side,
        notes,
    })
}

// ---------------------------------------------------------------------------
// Laplace method on the half-line
// ---------------------------------------------------------------------------

/// `L⁻¹` of `c·s^p·e^{−a√s}` (`p ∈ {0, −½, −1}`) and `c·e^{−as}·Φ` — the
/// table the half-line problems need.  Returns `None` when `F` is not of
/// that shape.
fn laplace_table(big_f: ExprId, s: ExprId, t: ExprId, pool: &ExprPool) -> Option<ExprId> {
    let f = simp(big_f, pool);
    let factors = match pool.get(f) {
        ExprData::Mul(xs) => xs,
        _ => vec![f],
    };
    let mut c = Vec::new();
    let mut p = rug::Rational::from(0);
    let mut sqrt_rate: Option<ExprId> = None;
    let mut lin_rate: Option<ExprId> = None;
    for fa in factors {
        if !mentions_var(fa, s, pool) {
            c.push(fa);
            continue;
        }
        if fa == s {
            p += 1;
            continue;
        }
        match pool.get(fa) {
            ExprData::Pow { base, exp } if base == s => {
                let q = pool.with(exp, |d| match d {
                    ExprData::Integer(n) => Some(rug::Rational::from(&n.0)),
                    ExprData::Rational(r) => Some(r.0.clone()),
                    _ => None,
                })?;
                p += q;
            }
            ExprData::Func { name, args } if name == "exp" => {
                let arg = args[0];
                // −arg/√s, read with s = σ² so that √(k s) = √k·σ folds.
                let sigma = fresh_symbol("__tm_sigma", Domain::Positive, &[arg], pool);
                let a1 = simp(
                    neg(
                        pool.mul(vec![
                            subs1(arg, s, pool.pow(sigma, pool.integer(2)), pool),
                            pool.pow(sigma, pool.integer(-1)),
                        ]),
                        pool,
                    ),
                    pool,
                );
                let a1 = constant_in(a1, sigma, pool);
                if !mentions_var(a1, sigma, pool)
                    && !mentions_var(a1, s, pool)
                    && sqrt_rate.is_none()
                {
                    sqrt_rate = Some(a1);
                    continue;
                }
                let a2 = constant_in(
                    simp(
                        neg(pool.mul(vec![arg, pool.pow(s, pool.integer(-1))]), pool),
                        pool,
                    ),
                    s,
                    pool,
                );
                if !mentions_var(a2, s, pool) && lin_rate.is_none() {
                    lin_rate = Some(a2);
                    continue;
                }
                return None;
            }
            _ => return None,
        }
    }
    let cc = if c.is_empty() {
        pool.integer(1)
    } else {
        pool.mul(c)
    };
    let half = rug::Rational::from((1, 2));
    match (sqrt_rate, lin_rate) {
        (Some(a), None) => {
            let a2 = pool.pow(a, pool.integer(2));
            let gauss = pool.func(
                "exp",
                vec![neg(
                    pool.mul(vec![
                        a2,
                        pool.pow(pool.mul(vec![pool.integer(4), t]), pool.integer(-1)),
                    ]),
                    pool,
                )],
            );
            let sqrt_pi = pool.func("sqrt", vec![pi(pool)]);
            let out = if p == 0 {
                // a/(2√π) t^{−3/2} e^{−a²/4t}
                pool.mul(vec![
                    a,
                    pool.pow(pool.mul(vec![pool.integer(2), sqrt_pi]), pool.integer(-1)),
                    pool.pow(t, pool.rational(-3, 2)),
                    gauss,
                ])
            } else if p == -half.clone() {
                // e^{−a²/4t}/√(πt)
                pool.mul(vec![
                    gauss,
                    pool.pow(pool.mul(vec![pi(pool), t]), pool.rational(-1, 2)),
                ])
            } else if p == -1 {
                // erfc(a/(2√t))
                pool.func(
                    "erfc",
                    vec![pool.mul(vec![
                        a,
                        pool.pow(
                            pool.mul(vec![pool.integer(2), pool.func("sqrt", vec![t])]),
                            pool.integer(-1),
                        ),
                    ])],
                )
            } else {
                return None;
            };
            Some(simp(pool.mul(vec![cc, out]), pool))
        }
        (None, Some(a)) => {
            // e^{−as}·s^p: p = 0 → δ(t − a) (declined), p = −n → θ(t − a)(t − a)^{n−1}/(n−1)!
            if p >= 0 || !p.is_integer() {
                return None;
            }
            let n = (-p).numer().to_i64()?;
            let shifted = sub(t, a, pool);
            let poly = pool.mul(vec![
                pool.pow(shifted, pool.integer(n - 1)),
                pool.pow(
                    pool.func("factorial", vec![pool.integer(n - 1)]),
                    pool.integer(-1),
                ),
            ]);
            Some(simp(
                pool.mul(vec![cc, pool.func("Heaviside", vec![shifted]), poly]),
                pool,
            ))
        }
        (None, None) => None,
        _ => None,
    }
}

/// Solve on `x > 0`, `t > 0` with zero initial data and boundary data at
/// `x = 0`, by the Laplace transform in `t`.
///
/// # Errors
///
/// As [`fourier_method`]; `E-PDE-161` for nonzero initial data (use
/// [`image_method`], or [`transform_solve`], which superposes).
pub fn laplace_method(
    problem: &PdeProblem,
    pool: &ExprPool,
) -> Result<TransformSolution, PdeError> {
    let s = setup(problem, pool)?;
    let (x, t, data_bc, ics) = half_line_roles(problem, &s, pool)?;
    for dt in &ics {
        if !is_zero(dt.rhs, pool) {
            return Err(PdeError::TransformData(
                "the Laplace method takes zero initial data; superpose with the image method (transform_solve)"
                    .into(),
            ));
        }
    }
    if data_bc.is_empty() {
        return Err(PdeError::TransformData(
            "no boundary data at x = 0 for the Laplace method".into(),
        ));
    }
    let xslot = s.indep.iter().position(|&v| v == x).expect("x");
    let tslot = s.indep.iter().position(|&v| v == t).expect("t");
    let avoid: Vec<ExprId> = vec![s.eq, x, t]
        .into_iter()
        .chain(problem.conditions().iter().flat_map(|c| [c.lhs, c.rhs]))
        .collect();
    let sv = fresh_symbol("s", Domain::Positive, &avoid, pool);
    let lt = crate::transform::laplace_transform(s.eq, t, sv, pool)
        .map_err(|e| PdeError::TransformOde(format!("Laplace transform of the PDE: {e}")))?;
    // Initial values are zero: every u(x, 0)-node is 0.
    let lt = zero_initial_values(lt, &s, tslot, pool);
    let mut big_args = s.indep.clone();
    big_args[tslot] = sv;
    let big = pool.func(
        transformed_name(&s.uv.name, TransformKind::Laplace),
        big_args,
    );
    let inp = to_ode(lt, big, xslot, x, pool)?;
    let (basis, mut notes) = solve_ode(&inp, pool)?;
    // Keep the solutions decaying as x → ∞.
    let mut kept = Vec::new();
    for &y in &basis {
        let r = exponent_rate(y, x, pool).ok_or_else(|| {
            PdeError::TransformOde(format!(
                "{}: the x-ODE's solutions are not exponentials",
                pool.display(y)
            ))
        })?;
        match super::sturm_liouville::sign(r, pool) {
            super::sturm_liouville::Sgn::Neg => kept.push(y),
            super::sturm_liouville::Sgn::Pos => {}
            _ => {
                return Err(PdeError::TransformOde(format!(
                    "cannot decide whether e^({}·x) decays",
                    pool.display(r)
                )))
            }
        }
    }
    if kept.len() != data_bc.len() {
        return Err(PdeError::TransformData(format!(
            "{} boundary conditions at x = 0 for {} decaying solutions",
            data_bc.len(),
            kept.len()
        )));
    }
    if kept.len() != 1 {
        return Err(PdeError::TransformOde(
            "one boundary condition at x = 0 is implemented".into(),
        ));
    }
    let bc = &data_bc[0];
    let jx = bc.orders[xslot];
    let mut y0 = kept[0];
    for _ in 0..jx {
        y0 = d(y0, x, pool)?;
    }
    let y0_at = simp(subs1(y0, x, pool.integer(0), pool), pool);
    let h = simp(
        pool.mul(vec![kept[0], pool.pow(y0_at, pool.integer(-1))]),
        pool,
    );
    let g = bc.rhs;
    let concrete = !crate::diff::undefined::has_undefined_function(g, pool);
    let gl = if concrete {
        crate::transform::laplace_transform(g, t, sv, pool).ok()
    } else {
        None
    };
    let gname = data_name(g, pool).unwrap_or_else(|| "g".into());
    let placeholder = pool.func(transformed_name(&gname, TransformKind::Laplace), vec![sv]);
    let transformed = simp(pool.mul(vec![h, gl.unwrap_or(placeholder)]), pool);
    let mut checks = Vec::new();
    // H solves the x-ODE and meets the transformed boundary condition.
    let mut r = replace_node(inp.equation, inp.y, h, pool);
    let mut dh = h;
    for &dsym in &inp.derivs {
        dh = d(dh, x, pool)?;
        r = replace_node(r, dsym, dh, pool);
    }
    checks.push(TransformCheck {
        what: "transfer_ode",
        index: 0,
        status: zstatus(r, pool),
    });
    let tau = fresh_symbol("tau", Domain::Positive, &avoid, pool);
    // 1. closed via the table on H·G.
    if let Some(gl) = gl {
        if let Some(v) = laplace_table(pool.mul(vec![h, gl]), sv, t, pool).or_else(|| {
            crate::transform::inverse_laplace_transform(pool.mul(vec![h, gl]), sv, t, pool).ok()
        }) {
            return finish_half_line(
                problem,
                &s,
                x,
                t,
                v,
                Vec::new(),
                transformed,
                sv,
                checks,
                notes,
                pool,
            );
        }
    }
    // 2. a pure delay H = e^{−as}: u = g(t − a)·θ(t − a) for any g.
    if let Some(a) = pure_delay(h, sv, pool) {
        let shifted = sub(t, a, pool);
        let v = simp(
            pool.mul(vec![
                pool.func("Heaviside", vec![shifted]),
                subs1(g, t, shifted, pool),
            ]),
            pool,
        );
        notes.push(format!(
            "a classical solution needs {} = 0 at t = 0 (continuity across t = {}); otherwise this is the weak solution",
            pool.display(subs1(g, t, pool.integer(0), pool)),
            pool.display(a)
        ));
        return retarded(
            problem,
            &s,
            x,
            t,
            v,
            shifted,
            g,
            transformed,
            sv,
            checks,
            notes,
            pool,
        );
    }
    // 3. Duhamel: u = ∫₀ᵗ K(x, t − τ) g(τ) dτ, K = L⁻¹{H}.
    let kern = laplace_table(h, sv, t, pool).ok_or_else(|| {
        PdeError::TransformOde(format!(
            "no inverse Laplace transform for the transfer function {}",
            pool.display(h)
        ))
    })?;
    let st = verify_solution(&[s.eq], &[s.u], &[kern], &[], pool)?;
    checks.push(TransformCheck {
        what: "kernel_pde",
        index: 0,
        status: status_to_zero(st.status),
    });
    notes.push(format!(
        "Duhamel's principle: K = {} is the response to a unit impulse at x = 0",
        pool.display(kern)
    ));
    let term = KernelTerm {
        kernel: simp(subs1(kern, t, sub(t, tau, pool), pool), pool),
        density: subs1(g, t, tau, pool),
        var: tau,
        lower: pool.integer(0),
        upper: t,
    };
    finish(
        problem,
        &s,
        "laplace".into(),
        Vec::new(),
        Vec::new(),
        Vec::new(),
        vec![term],
        transformed,
        sv,
        checks,
        Vec::new(),
        notes,
        pool,
    )
}

fn pure_delay(h: ExprId, s: ExprId, pool: &ExprPool) -> Option<ExprId> {
    if let ExprData::Func { name, args } = pool.get(h) {
        if name == "exp" {
            let a = simp(
                neg(pool.mul(vec![args[0], pool.pow(s, pool.integer(-1))]), pool),
                pool,
            );
            if !mentions_var(a, s, pool) {
                return Some(a);
            }
        }
    }
    None
}

/// Replace `u(x, 0)` and its derivatives by 0 in a Laplace-transformed
/// equation (zero initial data).
fn zero_initial_values(e: ExprId, s: &Setup, tslot: usize, pool: &ExprPool) -> ExprId {
    let mut nodes = Vec::new();
    let mut stack = vec![e];
    let mut seen = std::collections::HashSet::new();
    while let Some(n) = stack.pop() {
        if !seen.insert(n) {
            continue;
        }
        if let Some(v) = view(n, pool) {
            if v.name == s.uv.name
                && v.args.len() == s.indep.len()
                && v.args[tslot] != s.indep[tslot]
            {
                nodes.push(n);
            }
            continue;
        }
        super::jet::push_children(n, pool, &mut stack);
    }
    let mut out = e;
    for n in nodes {
        out = replace_node(out, n, pool.integer(0), pool);
    }
    simp(out, pool)
}

type HalfLineRoles = (ExprId, ExprId, Vec<Datum>, Vec<Datum>);

/// `(x, t, boundary data at x = 0, initial data at t = 0)`.
fn half_line_roles(
    problem: &PdeProblem,
    s: &Setup,
    pool: &ExprPool,
) -> Result<HalfLineRoles, PdeError> {
    let r = roles(problem)?;
    if r.half.len() != 2 || !r.line.is_empty() || r.half.iter().any(|(_, e)| !is_zero(*e, pool)) {
        return Err(PdeError::TransformMethodUnsupported(
            "the half-line methods need x > 0 and t > 0".into(),
        ));
    }
    let mut bcs = Vec::new();
    let mut ics = Vec::new();
    let mut tvar = None;
    let mut xvar = None;
    for c in problem.conditions() {
        let dt = datum(c, s, pool)?;
        if !is_zero(dt.at, pool) {
            return Err(PdeError::TransformData(format!(
                "{}: data must sit at x = 0 or t = 0",
                pool.display(c.lhs)
            )));
        }
        let var = s.indep[dt.slot];
        match dt.kind {
            ConditionKind::Initial => {
                if xvar == Some(var) {
                    return Err(PdeError::TransformData(
                        "a variable has both kinds of data".into(),
                    ));
                }
                tvar = Some(var);
                ics.push(dt);
            }
            _ => {
                if tvar == Some(var) {
                    return Err(PdeError::TransformData(
                        "a variable has both kinds of data".into(),
                    ));
                }
                xvar = Some(var);
                bcs.push(dt);
            }
        }
    }
    let (x, t) = match (xvar, tvar) {
        (Some(x), Some(t)) => (x, t),
        (Some(x), None) => (x, *s.indep.iter().find(|&&v| v != x).expect("two vars")),
        (None, Some(t)) => (*s.indep.iter().find(|&&v| v != t).expect("two vars"), t),
        _ => {
            return Err(PdeError::TransformData(
                "give the boundary condition at x = 0 with .bc and the initial data with .ic"
                    .into(),
            ))
        }
    };
    Ok((x, t, bcs, ics))
}

/// A closed half-line solution: verify the PDE and boundary conditions by
/// substitution and the initial conditions as limits `t → 0⁺`.
#[allow(clippy::too_many_arguments)]
fn finish_half_line(
    problem: &PdeProblem,
    s: &Setup,
    x: ExprId,
    t: ExprId,
    value: ExprId,
    side: Vec<ExprId>,
    transformed: ExprId,
    tvar: ExprId,
    mut checks: Vec<TransformCheck>,
    notes: Vec<String>,
    pool: &ExprPool,
) -> Result<TransformSolution, PdeError> {
    let bcs: Vec<PdeCondition> = problem
        .conditions()
        .iter()
        .filter(|c| c.kind != ConditionKind::Initial)
        .cloned()
        .collect();
    let v = verify_solution(&[s.eq], &[s.u], &[value], &bcs, pool)?;
    for c in v.checks {
        checks.push(TransformCheck {
            what: c.kind,
            index: c.index,
            status: c.status,
        });
    }
    let mut idx = 0;
    for c in problem
        .conditions()
        .iter()
        .filter(|c| c.kind == ConditionKind::Initial)
    {
        let dt = datum(c, s, pool)?;
        let mut e = value;
        for (j, &k) in dt.orders.iter().enumerate() {
            for _ in 0..k {
                e = d(e, s.indep[j], pool)?;
            }
        }
        // On the half-line x > 0: take the limit with x a positive symbol.
        let xp = fresh_symbol("__tm_xpos", Domain::Positive, &[e, dt.rhs], pool);
        let e = subs1(e, x, xp, pool);
        let st = match limit(e, t, pool.integer(0), LimitDirection::Plus, pool) {
            Ok(l) => zstatus(sub(l, subs1(dt.rhs, x, xp, pool), pool), pool),
            Err(_) => ZeroStatus::Unknown,
        };
        checks.push(TransformCheck {
            what: "ic_limit",
            index: idx,
            status: st,
        });
        idx += 1;
    }
    let status = combine(&checks);
    if status == VerificationStatus::Refuted {
        return Err(PdeError::TransformVerificationFailed(format!(
            "the Laplace candidate {} was refuted",
            pool.display(value)
        )));
    }
    Ok(TransformSolution {
        kind: SolutionKind::ClosedForm,
        method: "laplace".into(),
        value,
        closed_part: value,
        terms: Vec::new(),
        transformed,
        transform_var: tvar,
        status,
        checks,
        side_conditions: side,
        notes,
    })
}

/// `θ(t − a)·g(t − a)`: verify each side of the front.
#[allow(clippy::too_many_arguments)]
fn retarded(
    problem: &PdeProblem,
    s: &Setup,
    x: ExprId,
    t: ExprId,
    value: ExprId,
    shifted: ExprId,
    g: ExprId,
    transformed: ExprId,
    tvar: ExprId,
    mut checks: Vec<TransformCheck>,
    notes: Vec<String>,
    pool: &ExprPool,
) -> Result<TransformSolution, PdeError> {
    let _ = x;
    // Behind the front (t > a): g(t − a) solves the PDE and the BC.
    let behind = subs1(g, t, shifted, pool);
    let bcs: Vec<PdeCondition> = problem
        .conditions()
        .iter()
        .filter(|c| c.kind != ConditionKind::Initial)
        .cloned()
        .collect();
    let v = verify_solution(&[s.eq], &[s.u], &[behind], &bcs, pool)?;
    for c in v.checks {
        checks.push(TransformCheck {
            what: c.kind,
            index: c.index,
            status: c.status,
        });
    }
    // Ahead of it (t < a, which holds at t = 0 for x > 0): u ≡ 0 meets the
    // zero initial data and the PDE trivially.
    let ics: Vec<PdeCondition> = problem
        .conditions()
        .iter()
        .filter(|c| c.kind == ConditionKind::Initial)
        .cloned()
        .collect();
    let v0 = verify_solution(&[s.eq], &[s.u], &[pool.integer(0)], &ics, pool)?;
    for c in v0.checks.into_iter().filter(|c| c.kind == "ic") {
        checks.push(TransformCheck {
            what: "ic",
            index: c.index,
            status: c.status,
        });
    }
    let status = combine(&checks);
    if status == VerificationStatus::Refuted {
        return Err(PdeError::TransformVerificationFailed(format!(
            "the retarded solution {} was refuted",
            pool.display(value)
        )));
    }
    Ok(TransformSolution {
        kind: SolutionKind::ClosedForm,
        method: "laplace".into(),
        value,
        closed_part: value,
        terms: Vec::new(),
        transformed,
        transform_var: tvar,
        status,
        checks,
        side_conditions: Vec::new(),
        notes,
    })
}

// ---------------------------------------------------------------------------
// Sine / cosine transforms: the image method on the half-line
// ---------------------------------------------------------------------------

/// Solve on `x > 0`, `t > 0` with a homogeneous Dirichlet (`u(0, t) = 0`,
/// sine transform) or Neumann (`u_x(0, t) = 0`, cosine transform) end and
/// initial data.
///
/// # Errors
///
/// As [`fourier_method`]; `E-PDE-161` for a nonzero boundary value (use
/// [`laplace_method`] / [`transform_solve`]).
pub fn image_method(problem: &PdeProblem, pool: &ExprPool) -> Result<TransformSolution, PdeError> {
    use crate::transform::sincos::{sincos_transform, sincos_transformed_name, SinCosKind};
    let s = setup(problem, pool)?;
    let (x, t, bcs, ics) = half_line_roles(problem, &s, pool)?;
    let xslot = s.indep.iter().position(|&v| v == x).expect("x");
    let tslot = s.indep.iter().position(|&v| v == t).expect("t");
    if bcs.len() != 1 || !is_zero(bcs[0].rhs, pool) || bcs[0].orders[tslot] != 0 {
        return Err(PdeError::TransformData(
            "the image method needs one homogeneous condition u(0, t) = 0 or u_x(0, t) = 0".into(),
        ));
    }
    let kind = match bcs[0].orders[xslot] {
        0 => SinCosKind::Sine,
        1 => SinCosKind::Cosine,
        _ => {
            return Err(PdeError::TransformData(
                "the image method handles Dirichlet and Neumann ends".into(),
            ))
        }
    };
    let avoid: Vec<ExprId> = vec![s.eq, x, t]
        .into_iter()
        .chain(problem.conditions().iter().flat_map(|c| [c.lhs, c.rhs]))
        .collect();
    let w = fresh_symbol("omega", Domain::Positive, &avoid, pool);
    let tr = sincos_transform(kind, s.eq, x, w, pool)
        .map_err(|e| PdeError::TransformMethodUnsupported(format!("{e}")))?;
    // The boundary term is the (zero) boundary value.
    let tr = {
        let mut nodes = Vec::new();
        let mut stack = vec![tr];
        let mut seen = std::collections::HashSet::new();
        while let Some(n) = stack.pop() {
            if !seen.insert(n) {
                continue;
            }
            if let Some(v) = view(n, pool) {
                if v.name == s.uv.name && v.args[xslot] != x {
                    nodes.push(n);
                }
                continue;
            }
            super::jet::push_children(n, pool, &mut stack);
        }
        let mut out = tr;
        for n in nodes {
            out = replace_node(out, n, pool.integer(0), pool);
        }
        simp(out, pool)
    };
    let mut hat_args = s.indep.clone();
    hat_args[xslot] = w;
    let hat = pool.func(sincos_transformed_name(&s.uv.name, kind), hat_args);
    let inp = to_ode(tr, hat, tslot, t, pool)?;
    let k = inp.order();
    let (basis, notes) = solve_ode(&inp, pool)?;
    let mut data: Vec<Option<ExprId>> = vec![None; k];
    for dt in &ics {
        let j = dt.orders[tslot] as usize;
        if dt.slot != tslot || dt.orders[xslot] != 0 || j >= k || data[j].is_some() {
            return Err(PdeError::TransformData(
                "initial data u(x, 0) [, u_t(x, 0)]".into(),
            ));
        }
        data[j] = Some(dt.rhs);
    }
    if data.iter().any(|d| d.is_none()) {
        return Err(PdeError::TransformData(format!(
            "{k} initial conditions are needed for an equation of order {k} in t"
        )));
    }
    let transfers = transfer_functions(&basis, t, pool.integer(0), pool)?;
    let mut checks = Vec::new();
    let sv = fresh_symbol("s", Domain::Positive, &avoid, pool);
    let xi = fresh_symbol("xi", Domain::Real, &avoid, pool);
    let two_pi = pool.mul(vec![pool.integer(2), pi(pool)]);
    let mut terms = Vec::new();
    let mut closed = Vec::new();
    let mut transformed = Vec::new();
    let sign = if kind == SinCosKind::Sine { -1 } else { 1 };
    let oo = pool.pos_infinity();
    for (j, (&h, f)) in transfers
        .iter()
        .zip(data.into_iter().map(Option::unwrap))
        .enumerate()
    {
        let concrete = !crate::diff::undefined::has_undefined_function(f, pool);
        let fs = if concrete {
            sincos_transform(kind, f, x, w, pool).ok()
        } else {
            None
        };
        let placeholder = pool.func(
            sincos_transformed_name(&data_name(f, pool).unwrap_or_else(|| format!("f{j}")), kind),
            vec![w],
        );
        transformed.push(pool.mul(vec![h, fs.unwrap_or(placeholder)]));
        if let Some(fs) = fs {
            let inv = match kind {
                SinCosKind::Sine => crate::transform::inverse_fourier_sine_transform(
                    pool.mul(vec![h, fs]),
                    w,
                    x,
                    pool,
                ),
                _ => crate::transform::inverse_fourier_cosine_transform(
                    pool.mul(vec![h, fs]),
                    w,
                    x,
                    pool,
                ),
            };
            if let Ok(v) = inv {
                closed.push(v);
                continue;
            }
        }
        // K(z, t) = (1/π)∫₀^∞ H(ω, t) cos(ωz) dω = F⁻¹{H(2πξ, t)}(z).
        let hxi = simp(subs1(h, w, pool.mul(vec![two_pi, xi]), pool), pool);
        let kern = crate::transform::inverse_fourier_transform(hxi, xi, x, pool).map_err(|e| {
            PdeError::TransformOde(format!(
                "the full-line kernel of {} does not close: {e}",
                pool.display(h)
            ))
        })?;
        let kern = simp(kern, pool);
        let image = simp(
            pool.add(vec![
                subs1(kern, x, sub(x, sv, pool), pool),
                pool.mul(vec![
                    pool.integer(sign),
                    subs1(kern, x, pool.add(vec![x, sv]), pool),
                ]),
            ]),
            pool,
        );
        let st = verify_solution(&[s.eq], &[s.u], &[image], &[], pool)?;
        checks.push(TransformCheck {
            what: "kernel_pde",
            index: j,
            status: status_to_zero(st.status),
        });
        // The image kernel meets the boundary condition for every s.
        let bc_res = match kind {
            SinCosKind::Sine => subs1(image, x, pool.integer(0), pool),
            _ => subs1(d(image, x, pool)?, x, pool.integer(0), pool),
        };
        checks.push(TransformCheck {
            what: "kernel_bc",
            index: j,
            status: zstatus(bc_res, pool),
        });
        if j == 0 {
            if let Ok(mass) = crate::integrate::integrate_definite(kern, x, neg(oo, pool), oo, pool)
            {
                if zstatus(sub(mass.value, pool.integer(1), pool), pool) == ZeroStatus::ProvedZero {
                    checks.push(TransformCheck {
                        what: "kernel_mass",
                        index: j,
                        status: ZeroStatus::ProvedZero,
                    });
                }
            }
        }
        terms.push(KernelTerm {
            kernel: image,
            density: subs1(f, x, sv, pool),
            var: sv,
            lower: pool.integer(0),
            upper: oo,
        });
    }
    let method = match kind {
        SinCosKind::Sine => "fourier_sine",
        _ => "fourier_cosine",
    };
    let transformed = simp(pool.add(transformed), pool);
    finish(
        problem,
        &s,
        method.into(),
        closed.clone(),
        closed,
        Vec::new(),
        terms,
        transformed,
        w,
        checks,
        Vec::new(),
        notes,
        pool,
    )
}

// ---------------------------------------------------------------------------
// Dispatch
// ---------------------------------------------------------------------------

/// Choose the transform method from the domain: Fourier on `ℝ × (0, ∞)`;
/// on `(0, ∞)²` the image method for the initial data and the Laplace
/// method for the boundary data, superposed when both are present.
///
/// # Errors
///
/// As the individual methods.
pub fn transform_solve(
    problem: &PdeProblem,
    pool: &ExprPool,
) -> Result<TransformSolution, PdeError> {
    let r = roles(problem)?;
    if r.line.len() == 1 && r.half.len() == 1 {
        return fourier_method(problem, pool);
    }
    if r.half.len() != 2 || !r.line.is_empty() {
        return Err(PdeError::TransformMethodUnsupported(
            "transform methods need ℝ × (0, ∞) or (0, ∞)²".into(),
        ));
    }
    let s = setup(problem, pool)?;
    let (_, _, bcs, ics) = half_line_roles(problem, &s, pool)?;
    let ic_nonzero = ics.iter().any(|d| !is_zero(d.rhs, pool));
    let bc_nonzero = bcs.iter().any(|d| !is_zero(d.rhs, pool));
    match (ic_nonzero, bc_nonzero) {
        (_, false) => image_method(problem, pool),
        (false, true) => laplace_method(problem, pool),
        (true, true) => {
            // u = v + w: v has the initial data and a homogeneous end, w the
            // boundary data and zero initial data.
            let base = PdeProblem::new(problem.equations(), problem.unknowns(), pool)?
                .on(problem.domain().expect("domain").clone(), pool)?;
            let mut pv = base.clone();
            let mut pw = base;
            for c in problem.conditions() {
                let zero = pool.integer(0);
                match c.kind {
                    ConditionKind::Initial => {
                        pv = pv.ic(c.lhs, c.rhs, pool)?;
                        pw = pw.ic(c.lhs, zero, pool)?;
                    }
                    _ => {
                        pv = pv.bc(c.lhs, zero, pool)?;
                        pw = pw.bc(c.lhs, c.rhs, pool)?;
                    }
                }
            }
            let v = image_method(&pv, pool)?;
            let w = laplace_method(&pw, pool)?;
            let mut checks = v.checks.clone();
            checks.extend(w.checks.iter().cloned());
            let mut terms = v.terms.clone();
            terms.extend(w.terms.iter().cloned());
            let mut notes = v.notes.clone();
            notes.extend(w.notes.iter().cloned());
            notes.push("superposition: the image-method part carries the initial data, the Laplace part the boundary data".into());
            let mut side = v.side_conditions.clone();
            side.extend(w.side_conditions.iter().cloned());
            let kind = if terms.is_empty() {
                SolutionKind::ClosedForm
            } else {
                SolutionKind::IntegralRepresentation
            };
            Ok(TransformSolution {
                kind,
                method: format!("{}+{}", v.method, w.method),
                value: pool.add(vec![v.value, w.value]),
                closed_part: simp(pool.add(vec![v.closed_part, w.closed_part]), pool),
                terms,
                transformed: pool.add(vec![v.transformed, w.transformed]),
                transform_var: w.transform_var,
                status: combine(&checks),
                checks,
                side_conditions: side,
                notes,
            })
        }
    }
}

#[cfg(test)]
mod tests;
