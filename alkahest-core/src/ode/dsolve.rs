//! Classical symbolic ODE solver (`dsolve`).
//!
//! Returns closed-form *general* solutions to ordinary differential equations,
//! introducing integration constants `C1, C2, …` as fresh symbols.
//!
//! # Covered classes
//!
//! **First order** (`y' = …` written as `F(x, y, y') = 0`):
//! - separable `y' = g(x)·h(y)`
//! - linear `y' + p(x)·y = q(x)` (integrating-factor)
//! - Bernoulli `y' + p(x)·y = q(x)·yⁿ`
//! - exact `M dx + N dy = 0` with `∂M/∂y = ∂N/∂x`
//! - homogeneous of degree zero `y' = G(y/x)` (substitution `v = y/x`)
//! - Clairaut `y = x·y' + f(y')`
//! - Riccati `y' = q₀(x) + q₁(x)·y + q₂(x)·y²` **with a polynomial particular
//!   solution** found by ansatz (declined otherwise)
//!
//! **Second order** (`F(x, y, y', y'') = 0`):
//! - constant coefficients `a·y'' + b·y' + c·y = r(x)` (real distinct / repeated
//!   / complex roots)
//! - Euler–Cauchy `a·x²·y'' + b·x·y' + c·y = r(x)`
//! - general variable coefficients `a₂(x)y'' + a₁(x)y' + a₀(x)y = r(x)`, when a
//!   first homogeneous solution is found by ansatz — the second then follows by
//!   reduction of order (see [`mod@variation`])
//!
//! **Higher order**: constant-coefficient `Σ aₖ y^(k) = r(x)`, solved through
//! the characteristic polynomial (rational + quadratic factorization;
//! irreducible factors of degree ≥ 3 are declined).
//!
//! For every linear class the forcing term `r(x)` is closed either by
//! undetermined coefficients (cheap, exact, but only for
//! polynomial × exp × sin/cos) or, failing that, by variation of parameters at
//! the equation's own order — which is what admits forcings no ansatz can
//! express: `sec x`, `tan x`, `1/(1 + eˣ)`, `log x`, arbitrary rational.
//!
//! # Verification gate
//!
//! *Every* returned solution is verified by substitution: the candidate `y(x)`
//! (and its derivatives) are substituted into the original equation, the
//! residual is simplified, and accepted only when it is the symbolic zero or
//! numerically `≈ 0` at several sample points over random constant values.  A
//! candidate that fails verification causes [`dsolve`] to decline (it never
//! returns an unverified solution).
//!
//! # Quadratures
//!
//! Closed forms that require an integral defer to the existing
//! [`mod@crate::integrate`] engine.  If a required integral does not close in
//! elementary form, the class is declined (no unevaluated-integral output).
//!
//! `dsolve` manufactures its own integrands — `exp(∫p dx)·q`, `Wₖ·g/W` — and a
//! manufactured integrand is not in normal form: `e^{−log x}` rather than
//! `1/x`, `e^{x}·e^{−x}` rather than `1`, `cos²x + sin²x` rather than `1`.  The
//! integration engine is form-sensitive enough that this decides whether an
//! elementary integral closes, so [`integrate_or_decline`] tries each integrand
//! in several equal-valued spellings and takes the first that closes.  Set
//! `ALKAHEST_DSOLVE_TRACE` in a test build to print every integral that no
//! spelling closed.

use crate::diff::diff;
use crate::integrate::engine::integrate;
use crate::kernel::eval_const::try_expr_f64;
use crate::kernel::{Domain, ExprData, ExprId, ExprPool};
use crate::simplify::engine::{simplify, simplify_expanded};
use std::collections::HashMap;
use std::fmt;

mod constant_coeff;
#[cfg(test)]
mod corpus;
mod first_order;
mod variation;
mod verify;

pub(crate) use verify::residual_is_zero;

// ---------------------------------------------------------------------------
// Public API
// ---------------------------------------------------------------------------

/// Input description of a scalar ODE for [`dsolve`].
///
/// The equation is supplied as a single expression `equation` that is taken to
/// equal zero, written in terms of the symbols `x` (independent variable), `y`
/// (the unknown `y(x)`), and the derivative symbols in `derivs`
/// (`derivs[0] = y'`, `derivs[1] = y''`, …).  The `order` equals
/// `derivs.len()`.
///
/// Use [`OdeInput::first_order`] / [`OdeInput::second_order`] /
/// [`OdeInput::higher_order`] to build instances; they allocate the derivative
/// symbols with the conventional names `y'`, `y''`, ….
#[derive(Clone, Debug)]
pub struct OdeInput {
    /// Independent variable, e.g. `x`.
    pub x: ExprId,
    /// Dependent variable `y` (representing `y(x)`).
    pub y: ExprId,
    /// Derivative symbols `[y', y'', …]`.
    pub derivs: Vec<ExprId>,
    /// The equation, interpreted as `equation = 0`.
    pub equation: ExprId,
}

impl OdeInput {
    fn deriv_symbol(y: ExprId, k: usize, pool: &ExprPool) -> ExprId {
        let base = pool.with(y, |d| match d {
            ExprData::Symbol { name, .. } => name.clone(),
            _ => "y".to_string(),
        });
        let primes = "'".repeat(k);
        pool.symbol(format!("{base}{primes}"), Domain::Real)
    }

    /// Build a first-order input `equation(x, y, y') = 0`.
    ///
    /// Returns `(input, y')` so the caller can build the equation referring to
    /// the freshly created derivative symbol.
    pub fn first_order(x: ExprId, y: ExprId, pool: &ExprPool) -> (Self, ExprId) {
        let yp = Self::deriv_symbol(y, 1, pool);
        (
            OdeInput {
                x,
                y,
                derivs: vec![yp],
                equation: pool.integer(0_i32),
            },
            yp,
        )
    }

    /// Build a second-order input `equation(x, y, y', y'') = 0`.
    ///
    /// Returns `(input, y', y'')`.
    pub fn second_order(x: ExprId, y: ExprId, pool: &ExprPool) -> (Self, ExprId, ExprId) {
        let yp = Self::deriv_symbol(y, 1, pool);
        let ypp = Self::deriv_symbol(y, 2, pool);
        (
            OdeInput {
                x,
                y,
                derivs: vec![yp, ypp],
                equation: pool.integer(0_i32),
            },
            yp,
            ypp,
        )
    }

    /// Build an `order`-th order input.  Returns `(input, derivs)` where
    /// `derivs[k]` is the `(k+1)`-th derivative symbol.
    pub fn higher_order(
        x: ExprId,
        y: ExprId,
        order: usize,
        pool: &ExprPool,
    ) -> (Self, Vec<ExprId>) {
        assert!(order >= 1, "ODE order must be ≥ 1");
        let derivs: Vec<ExprId> = (1..=order)
            .map(|k| Self::deriv_symbol(y, k, pool))
            .collect();
        (
            OdeInput {
                x,
                y,
                derivs: derivs.clone(),
                equation: pool.integer(0_i32),
            },
            derivs,
        )
    }

    /// Replace the equation expression.
    pub fn with_equation(mut self, equation: ExprId) -> Self {
        self.equation = equation;
        self
    }

    /// ODE order.
    pub fn order(&self) -> usize {
        self.derivs.len()
    }
}

/// A general solution returned by [`dsolve`].
#[derive(Clone, Debug)]
pub struct DsolveSolution {
    /// The solution expression for `y(x)` (the right-hand side of `y(x) = …`),
    /// containing the integration constants in [`Self::constants`].
    pub y_of_x: ExprId,
    /// The fresh constant symbols `C1, C2, …` appearing in [`Self::y_of_x`].
    pub constants: Vec<ExprId>,
    /// Short label of the solving method (e.g. `"separable"`).
    pub method: &'static str,
}

/// The result of [`dsolve`]: zero or more general-solution branches.
#[derive(Clone, Debug)]
pub struct DsolveResult {
    /// General-solution branches.  Most classes return exactly one branch.
    pub solutions: Vec<DsolveSolution>,
}

/// Errors / declines from [`dsolve`].
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum DsolveError {
    /// The ODE did not match any implemented solvable class, or a required
    /// quadrature did not close in elementary form.
    Unsupported(String),
    /// A candidate closed form was produced but failed the substitution
    /// verification gate (so it is withheld rather than returned wrong).
    VerificationFailed(String),
    /// Differentiation of an intermediate expression failed.
    DiffError(String),
}

impl fmt::Display for DsolveError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            DsolveError::Unsupported(m) => write!(f, "dsolve: unsupported ODE: {m}"),
            DsolveError::VerificationFailed(m) => {
                write!(f, "dsolve: candidate failed verification: {m}")
            }
            DsolveError::DiffError(m) => write!(f, "dsolve: differentiation error: {m}"),
        }
    }
}

impl std::error::Error for DsolveError {}

impl crate::errors::AlkahestError for DsolveError {
    fn code(&self) -> &'static str {
        match self {
            DsolveError::Unsupported(_) => "E-ODE-010",
            DsolveError::VerificationFailed(_) => "E-ODE-011",
            DsolveError::DiffError(_) => "E-ODE-012",
        }
    }

    fn remediation(&self) -> Option<&'static str> {
        match self {
            DsolveError::Unsupported(_) => Some(
                "the ODE is outside the implemented classical classes, or a required \
                 integral is non-elementary; check the equation form",
            ),
            DsolveError::VerificationFailed(_) => Some(
                "the solver found a candidate that did not verify by substitution; \
                 this is reported rather than returned as a (possibly wrong) answer",
            ),
            DsolveError::DiffError(_) => {
                Some("ensure the equation only contains differentiable functions")
            }
        }
    }
}

// ---------------------------------------------------------------------------
// Entry point
// ---------------------------------------------------------------------------

/// Solve a scalar ODE in closed form, returning the general solution(s).
///
/// Dispatches on the ODE order and structure to the implemented classical
/// methods.  Every returned solution is verified by substitution (see the
/// [module docs](self)); unverifiable candidates are withheld and the relevant
/// class declines.
///
/// # Errors
///
/// Returns [`DsolveError::Unsupported`] when the equation is outside the
/// implemented classes or a required quadrature is non-elementary, and
/// [`DsolveError::VerificationFailed`] when a candidate could not be verified.
pub fn dsolve(input: &OdeInput, pool: &ExprPool) -> Result<DsolveResult, DsolveError> {
    let mut gen = ConstGen::new(input, pool);
    match input.order() {
        1 => first_order::solve(input, &mut gen, pool),
        2 => constant_coeff::solve_second_order(input, &mut gen, pool),
        n if n >= 3 => constant_coeff::solve_higher_order(input, n, &mut gen, pool),
        _ => Err(DsolveError::Unsupported("order 0 ODE".to_string())),
    }
}

// ---------------------------------------------------------------------------
// Fresh-constant generator (collision-free with user symbols)
// ---------------------------------------------------------------------------

/// Allocates fresh integration-constant symbols `C1, C2, …`, skipping any name
/// already present in the input equation so user symbols never collide.
pub(crate) struct ConstGen {
    next: usize,
    used: std::collections::HashSet<String>,
}

impl ConstGen {
    fn new(input: &OdeInput, pool: &ExprPool) -> Self {
        let mut used = std::collections::HashSet::new();
        collect_symbol_names(input.equation, pool, &mut used);
        ConstGen { next: 1, used }
    }

    /// Return a fresh constant symbol whose name (`C{n}`) is not already used.
    pub(crate) fn fresh(&mut self, pool: &ExprPool) -> ExprId {
        loop {
            let name = format!("C{}", self.next);
            self.next += 1;
            if !self.used.contains(&name) {
                self.used.insert(name.clone());
                return pool.symbol(name, Domain::Real);
            }
        }
    }
}

fn collect_symbol_names(
    expr: ExprId,
    pool: &ExprPool,
    out: &mut std::collections::HashSet<String>,
) {
    pool.with(expr, |d| match d {
        ExprData::Symbol { name, .. } => {
            out.insert(name.clone());
        }
        ExprData::Add(args) | ExprData::Mul(args) | ExprData::Func { args, .. } => {
            for &a in args {
                collect_symbol_names(a, pool, out);
            }
        }
        ExprData::Pow { base, exp } => {
            collect_symbol_names(*base, pool, out);
            collect_symbol_names(*exp, pool, out);
        }
        _ => {}
    });
}

// ---------------------------------------------------------------------------
// Shared small helpers (used across submodules)
// ---------------------------------------------------------------------------

/// Simplify with distribution (expanded normal form).  The classification
/// logic relies on polynomial-in-`x`/`y` terms being flattened (e.g.
/// `−1·(−3y−x)` becoming `3y + x`) so coefficient extraction by structural
/// inspection works.
pub(crate) fn simp(expr: ExprId, pool: &ExprPool) -> ExprId {
    simplify_expanded(expr, pool).value
}

/// Plain (non-expanding) simplify, for the final residual zero-check where
/// expansion is not required.
pub(crate) fn simp_plain(expr: ExprId, pool: &ExprPool) -> ExprId {
    simplify(expr, pool).value
}

/// `diff(expr, var).value`, mapping `DiffError` into `DsolveError`.
pub(crate) fn ddx(expr: ExprId, var: ExprId, pool: &ExprPool) -> Result<ExprId, DsolveError> {
    diff(expr, var, pool)
        .map(|d| d.value)
        .map_err(|e| DsolveError::DiffError(e.to_string()))
}

/// Integrate `expr` in `var`; map any decline to `Unsupported` so the caller
/// declines the whole class (we never emit unevaluated-integral output).
///
/// The integrand is tried in several *spellings* (see [`integrand_spellings`]):
/// `dsolve` manufactures its integrands — `exp(∫p dx)·q`, `y₂·g/W` — and they
/// arrive carrying artefacts (`e^{x}·e^{−x}`, `cos²x + sin²x`) that the default
/// rule set does not cancel.  The integration engine sees an integrand it
/// cannot close where the *same function*, spelled normally, is trivial.  Each
/// spelling is mathematically equal to the original on the solution domain, so
/// the first one that closes is the answer.
pub(crate) fn integrate_or_decline(
    expr: ExprId,
    var: ExprId,
    pool: &ExprPool,
) -> Result<ExprId, DsolveError> {
    integrate_first_of(&[expr], var, pool)
}

/// [`integrate_or_decline`] over several *constructions* of the same integrand.
///
/// A caller that can build the integrand more than one way (variation of
/// parameters can divide by the raw Wronskian or by the normalised one) passes
/// all of them; each is expanded into its [`integrand_spellings`] and the
/// first that closes wins.  All candidates must be equal as functions.
pub(crate) fn integrate_first_of(
    exprs: &[ExprId],
    var: ExprId,
    pool: &ExprPool,
) -> Result<ExprId, DsolveError> {
    let mut last: Option<String> = None;
    let mut tried: Vec<ExprId> = Vec::new();
    // Stage 1: every candidate exactly as the caller built it.  Rewriting is
    // only worth its cost once the engine has actually refused, and the common
    // case is that it does not.
    for &expr in exprs {
        if tried.contains(&expr) {
            continue;
        }
        tried.push(expr);
        match integrate(expr, var, pool) {
            Ok(d) => return Ok(simp(d.value, pool)),
            Err(e) => last = last.or(Some(e.to_string())),
        }
    }
    // Stage 2: the rewritings.
    for &expr in exprs {
        for cand in integrand_spellings(expr, pool) {
            if tried.contains(&cand) {
                continue;
            }
            tried.push(cand);
            match integrate(cand, var, pool) {
                Ok(d) => return Ok(simp(d.value, pool)),
                Err(e) => last = last.or(Some(e.to_string())),
            }
        }
    }
    #[cfg(test)]
    if std::env::var_os("ALKAHEST_DSOLVE_TRACE").is_some() {
        // Every line here is an integral `dsolve` needs and `integrate` does not
        // close — the actionable feedback list for the integration engine.
        for cand in tried {
            eprintln!(
                "INT_DECLINE\td/d{}\t{}",
                pool.display(var),
                pool.display(cand)
            );
        }
    }
    Err(DsolveError::Unsupported(format!(
        "required integral did not close: {}",
        last.unwrap_or_else(|| "no candidate form".to_string())
    )))
}

/// The most-normalised spelling of `expr` (the last [`integrand_spellings`]
/// candidate).  Used for expressions that are *not* about to be integrated —
/// the coefficients `P` and `Q` of a normalised second-order equation, say —
/// where there is no engine to fall through on the caller's behalf.
pub(crate) fn normalized(expr: ExprId, pool: &ExprPool) -> ExprId {
    let cands = integrand_spellings(expr, pool);
    *cands.last().unwrap_or(&expr)
}

/// Equal-valued rewritings of an integrand, cheapest first.
///
/// 1. the expression as given;
/// 2. with reciprocals distributed over products ([`distribute_recip`]);
/// 3. after the log/exp rule set (`e^{a}·e^{b} → e^{a+b}`, `log(e^u) → u`),
///    which collapses the integrating-factor artefacts;
/// 4. additionally after the trig normal form (`cos²u + sin²u → 1`), which
///    collapses a Wronskian of `{cos, sin}` — only attempted when the
///    expression actually mentions sin/cos, since that pass expands products.
///
/// **The list is ordered, not ranked, and the original comes first on
/// purpose.** Normalising is not monotone for the integration engine:
/// `∫ sin x·tan x/(cos²x + sin²x) dx` closes and `∫ sin x·tan x dx` — the same
/// integrand with the redundant `1` cancelled — does not. Until that is fixed
/// upstream, dropping the un-normalised form would lose ODEs that currently
/// solve, so every spelling is tried.
///
/// Duplicates are dropped, so a fully-normalised expression costs one call.
fn integrand_spellings(expr: ExprId, pool: &ExprPool) -> Vec<ExprId> {
    let mut out = vec![expr];
    let push = |e: ExprId, out: &mut Vec<ExprId>| {
        if !out.contains(&e) {
            out.push(e);
        }
    };
    let dr = simp(distribute_recip(expr, pool), pool);
    push(dr, &mut out);
    for base in [expr, dr] {
        let le = simp(
            crate::simplify::engine::simplify_log_exp(base, pool, &[]).value,
            pool,
        );
        push(le, &mut out);
        if mentions_sin_cos(le, pool) {
            let tn = simp(
                crate::simplify::engine::simplify_trig_normal_form(le, pool).value,
                pool,
            );
            push(tn, &mut out);
        }
    }
    out
}

/// Rewrite `(a·b·…)^k → a^k·b^k·…` for **integer** `k`, recursively.
///
/// `simplify` does not cancel `x²·(−1·x²)⁻¹`: the `Pow` wraps a whole `Mul`,
/// so power collection never sees the `x²` inside it and the quotient survives
/// as an "irreducible product of var-dependent factors" the integration engine
/// declines. Distributing first turns it into `x²·(−1)⁻¹·x⁻²`, which collects
/// to `−1`. Restricted to integer exponents, where `(ab)^k = a^k b^k` holds
/// unconditionally over ℝ∖{0}.
fn distribute_recip(expr: ExprId, pool: &ExprPool) -> ExprId {
    match pool.get(expr) {
        ExprData::Add(args) => {
            let ds: Vec<ExprId> = args.iter().map(|&a| distribute_recip(a, pool)).collect();
            pool.add(ds)
        }
        ExprData::Mul(args) => {
            let ds: Vec<ExprId> = args.iter().map(|&a| distribute_recip(a, pool)).collect();
            pool.mul(ds)
        }
        ExprData::Pow { base, exp } => {
            let b = distribute_recip(base, pool);
            let is_int = matches!(pool.get(exp), ExprData::Integer(_));
            match pool.get(b) {
                ExprData::Mul(fs) if is_int => {
                    pool.mul(fs.iter().map(|&f| pool.pow(f, exp)).collect())
                }
                _ => pool.pow(b, exp),
            }
        }
        ExprData::Func { name, args } => {
            let ds: Vec<ExprId> = args.iter().map(|&a| distribute_recip(a, pool)).collect();
            pool.func(&name, ds)
        }
        _ => expr,
    }
}

fn mentions_sin_cos(expr: ExprId, pool: &ExprPool) -> bool {
    pool.with(expr, |d| match d {
        ExprData::Func { name, args } => {
            name == "sin" || name == "cos" || args.iter().any(|&a| mentions_sin_cos(a, pool))
        }
        ExprData::Add(args) | ExprData::Mul(args) => {
            args.iter().any(|&a| mentions_sin_cos(a, pool))
        }
        ExprData::Pow { base, exp } => {
            mentions_sin_cos(*base, pool) || mentions_sin_cos(*exp, pool)
        }
        _ => false,
    })
}

/// Build `exp(arg)`, folding logarithmic summands into powers:
/// `exp(c·log(u) + rest) → u^c · exp(rest)`.
///
/// Every integrating factor `μ = exp(∫p dx)` in this module goes through here.
/// The default rule set will not apply `exp(log u) → u` (it is a branch-cut
/// identity, sound only for `u > 0`), so without this fold the linear class
/// manufactures `μ = e^{−log x}` and then asks the integration engine for
/// `∫ q·e^{−log x} dx` where it means `∫ q/x dx` — and gets a decline for an
/// integral that is elementary.
///
/// Taking `u > 0` is the same convention that writing `log(u)` in the
/// antiderivative already commits to, and it is not load-bearing for
/// correctness: every candidate solution still has to pass
/// [`residual_is_zero`], which samples the residual at positive `x`.
pub(crate) fn exp_of(arg: ExprId, pool: &ExprPool) -> ExprId {
    let terms: Vec<ExprId> = match pool.get(simp(arg, pool)) {
        ExprData::Add(args) => args,
        _ => vec![simp(arg, pool)],
    };
    let mut factors: Vec<ExprId> = Vec::new();
    let mut rest: Vec<ExprId> = Vec::new();
    for t in terms {
        match log_summand(t, pool) {
            Some((u, c)) => factors.push(pool.pow(u, c)),
            None => rest.push(t),
        }
    }
    if factors.is_empty() {
        return simp(pool.func("exp", vec![arg]), pool);
    }
    if !rest.is_empty() {
        let r = pool.add(rest);
        factors.push(pool.func("exp", vec![r]));
    }
    // A one-element `Mul` is *not* collapsed by `simplify`, and a surviving
    // `Mul([x])` blocks the power collection that would cancel `x·x⁻¹` later.
    if factors.len() == 1 {
        return simp(factors[0], pool);
    }
    simp(pool.mul(factors), pool)
}

/// Recognise `c·log(u)` (or bare `log(u)`, `c = 1`) and return `(u, c)` with
/// `c` a numeric constant expression.
fn log_summand(term: ExprId, pool: &ExprPool) -> Option<(ExprId, ExprId)> {
    if let ExprData::Func { name, args } = pool.get(term) {
        if name == "log" && args.len() == 1 {
            return Some((args[0], pool.integer(1_i32)));
        }
    }
    let ExprData::Mul(args) = pool.get(term) else {
        return None;
    };
    let mut inner = None;
    let mut coeff: Vec<ExprId> = Vec::new();
    for a in args {
        match pool.get(a) {
            ExprData::Func { name, args: fargs } if name == "log" && fargs.len() == 1 => {
                if inner.is_some() {
                    return None; // product of two logs — not of this form
                }
                inner = Some(fargs[0]);
            }
            ExprData::Integer(_) | ExprData::Rational(_) => coeff.push(a),
            _ => return None, // a non-constant cofactor: exp(x·log u) is not u^c
        }
    }
    let u = inner?;
    // `simp` leaves a one-element `Mul` standing, and `Mul([−1])` as an
    // exponent is a *different* node from `−1`, which silently defeats the
    // power collection in `x·x⁻¹`.  Collapse it here.
    let c = match coeff.len() {
        0 => pool.integer(1_i32),
        1 => simp(coeff[0], pool),
        _ => simp(pool.mul(coeff), pool),
    };
    Some((u, c))
}

/// Does `expr` contain `needle` as a sub-expression?
pub(crate) fn contains(expr: ExprId, needle: ExprId, pool: &ExprPool) -> bool {
    if expr == needle {
        return true;
    }
    pool.with(expr, |d| match d {
        ExprData::Add(args) | ExprData::Mul(args) | ExprData::Func { args, .. } => {
            args.iter().any(|&a| contains(a, needle, pool))
        }
        ExprData::Pow { base, exp } => {
            contains(*base, needle, pool) || contains(*exp, needle, pool)
        }
        _ => false,
    })
}

/// `a - b`, simplified.
pub(crate) fn sub(a: ExprId, b: ExprId, pool: &ExprPool) -> ExprId {
    let neg_b = pool.mul(vec![pool.integer(-1_i32), b]);
    simp(pool.add(vec![a, neg_b]), pool)
}

/// `a / b`, simplified.
pub(crate) fn div(a: ExprId, b: ExprId, pool: &ExprPool) -> ExprId {
    let inv_b = pool.pow(b, pool.integer(-1_i32));
    simp(pool.mul(vec![a, inv_b]), pool)
}

/// Substitute a single symbol → replacement, simplifying the result.
pub(crate) fn subs1(expr: ExprId, from: ExprId, to: ExprId, pool: &ExprPool) -> ExprId {
    let mut m = HashMap::new();
    m.insert(from, to);
    simp(crate::kernel::subs::subs(expr, &m, pool), pool)
}

/// Is `expr` the literal zero after simplification?
pub(crate) fn is_zero(expr: ExprId, pool: &ExprPool) -> bool {
    let s = simp(expr, pool);
    matches!(pool.get(s), ExprData::Integer(n) if n.0 == 0)
        || matches!(try_expr_f64(s, pool), Some(v) if v == 0.0)
}

#[cfg(test)]
mod tests;
