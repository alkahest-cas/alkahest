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
//!   / complex roots), including non-homogeneous RHS via undetermined
//!   coefficients (polynomial × exp × sin/cos) and variation of parameters
//! - Euler–Cauchy `a·x²·y'' + b·x·y' + c·y = 0`
//!
//! **Higher order**: constant-coefficient `Σ aₖ y^(k) = 0`, solved through the
//! characteristic polynomial (rational + quadratic factorization; irreducible
//! factors of degree ≥ 3 are declined).
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

use crate::diff::diff;
use crate::integrate::engine::integrate;
use crate::kernel::eval_const::try_expr_f64;
use crate::kernel::{Domain, ExprData, ExprId, ExprPool};
use crate::simplify::engine::{simplify, simplify_expanded};
use std::collections::HashMap;
use std::fmt;

mod constant_coeff;
mod first_order;
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
pub(crate) fn integrate_or_decline(
    expr: ExprId,
    var: ExprId,
    pool: &ExprPool,
) -> Result<ExprId, DsolveError> {
    match integrate(expr, var, pool) {
        Ok(d) => Ok(simp(d.value, pool)),
        Err(e) => {
            // Fallback: closed-form ∫ p(x)·e^{a x}·{1,cos b x,sin b x} dx via an
            // undetermined-coefficients ansatz (the engine declines some of these
            // products, e.g. ∫ x·e^{−3x}).
            if let Some(f) = integrate_pexp_trig(expr, var, pool) {
                return Ok(f);
            }
            Err(DsolveError::Unsupported(format!(
                "required integral did not close: {e}"
            )))
        }
    }
}

/// Antiderivative of `p(x)·e^{a x}·{1 | cos(b x) | sin(b x)}` where `p` is a
/// polynomial and `a,b` are constants.  Builds an ansatz of the same shape with
/// undetermined polynomial coefficients and solves by numeric sampling, then
/// returns the symbolic antiderivative (verified by `d/dx`).  Returns `None`
/// when the integrand is not of this form or the solve is singular.
pub(crate) fn integrate_pexp_trig(expr: ExprId, var: ExprId, pool: &ExprPool) -> Option<ExprId> {
    // Decompose factors into polynomial part, exp rate a, trig rate b.
    let factors: Vec<ExprId> = match pool.get(expr) {
        ExprData::Mul(args) => args,
        _ => vec![expr],
    };
    let mut exp_rate = 0.0_f64;
    let mut trig: Option<(bool, f64)> = None; // (is_sin, rate)
    let mut poly_factors: Vec<ExprId> = Vec::new();
    for f in factors {
        match pool.get(f) {
            ExprData::Func { name, args } if name == "exp" && args.len() == 1 => {
                exp_rate += linear_rate_of(args[0], var, pool)?;
            }
            ExprData::Func { name, args }
                if (name == "cos" || name == "sin") && args.len() == 1 =>
            {
                if trig.is_some() {
                    return None;
                }
                trig = Some((name == "sin", linear_rate_of(args[0], var, pool)?));
            }
            _ => {
                if contains(f, var, pool) && poly_degree_in(f, var, pool).is_none() {
                    return None;
                }
                poly_factors.push(f);
            }
        }
    }
    let poly = if poly_factors.is_empty() {
        pool.integer(1_i32)
    } else {
        simp(pool.mul(poly_factors), pool)
    };
    let deg = poly_degree_in(poly, var, pool)?;
    if exp_rate == 0.0 && trig.is_none() {
        return None; // pure polynomial — the engine already handles this
    }

    // Ansatz: F = e^{a x}·Σ_{k≤deg} (A_k x^k cos b x + B_k x^k sin b x)  (cos&sin
    // only when trig present; otherwise just e^{a x}·Σ A_k x^k).
    let exp_factor = if exp_rate != 0.0 {
        Some(simp(
            pool.func("exp", vec![mul_c(exp_rate, var, pool)]),
            pool,
        ))
    } else {
        None
    };
    let mut mods: Vec<ExprId> = Vec::new();
    if let Some((_, b)) = trig {
        let bx = mul_c(b, var, pool);
        mods.push(pool.func("cos", vec![bx]));
        mods.push(pool.func("sin", vec![bx]));
    } else {
        mods.push(pool.integer(1_i32));
    }
    let mut terms: Vec<ExprId> = Vec::new();
    for k in 0..=deg {
        let xk = if k == 0 {
            pool.integer(1_i32)
        } else {
            pool.pow(var, pool.integer(k as i32))
        };
        for &m in &mods {
            let mut fac = vec![xk, m];
            if let Some(e) = exp_factor {
                fac.push(e);
            }
            terms.push(simp(pool.mul(fac), pool));
        }
    }
    let k = terms.len();
    // Solve Σ A_j (d/dx term_j) = integrand by sampling at k points.
    let mut dterms: Vec<ExprId> = Vec::with_capacity(k);
    for &t in &terms {
        dterms.push(simp(diff(t, var, pool).ok()?.value, pool));
    }
    let samples: Vec<f64> = (0..k).map(|i| 0.41 + 0.47 * i as f64).collect();
    let mut mat = vec![vec![0.0; k]; k];
    let mut rhs = vec![0.0; k];
    for (i, &xv) in samples.iter().enumerate() {
        let mut env = HashMap::new();
        env.insert(var, xv);
        for (j, &dt) in dterms.iter().enumerate() {
            mat[i][j] = verify::eval(dt, &env, pool)?;
        }
        rhs[i] = verify::eval(expr, &env, pool)?;
    }
    let sol = gaussian_solve(&mut mat, &mut rhs)?;
    let mut out = Vec::new();
    for (j, &t) in terms.iter().enumerate() {
        if sol[j].abs() < 1e-12 {
            continue;
        }
        out.push(pool.mul(vec![f64_rational(sol[j], pool), t]));
    }
    let f = simp(pool.add(out), pool);
    // Verify d/dx f == expr numerically before trusting it.
    let df = simp(diff(f, var, pool).ok()?.value, pool);
    for xv in [0.23_f64, 0.61, 1.07, 1.53] {
        let mut env = HashMap::new();
        env.insert(var, xv);
        let lhs = verify::eval(df, &env, pool)?;
        let rhsv = verify::eval(expr, &env, pool)?;
        if (lhs - rhsv).abs() > 1e-6 {
            return None;
        }
    }
    Some(f)
}

/// `arg = c·var` (through the origin) → `c`.
fn linear_rate_of(arg: ExprId, var: ExprId, pool: &ExprPool) -> Option<f64> {
    let d = diff(arg, var, pool).ok()?.value;
    if contains(d, var, pool) {
        return None;
    }
    let dx = simp(pool.mul(vec![d, var]), pool);
    if !is_zero(sub(arg, dx, pool), pool) {
        return None;
    }
    try_expr_f64(simp(d, pool), pool)
}

fn poly_degree_in(expr: ExprId, var: ExprId, pool: &ExprPool) -> Option<usize> {
    if !contains(expr, var, pool) {
        return Some(0);
    }
    match pool.get(expr) {
        ExprData::Symbol { .. } => Some(1),
        ExprData::Add(args) => args
            .iter()
            .map(|&a| poly_degree_in(a, var, pool))
            .try_fold(0usize, |acc, d| Some(acc.max(d?))),
        ExprData::Mul(args) => args
            .iter()
            .map(|&a| poly_degree_in(a, var, pool))
            .try_fold(0usize, |acc, d| Some(acc + d?)),
        ExprData::Pow { base, exp } if base == var => {
            if let ExprData::Integer(k) = pool.get(exp) {
                let k = k.0.to_i64()?;
                if k >= 0 {
                    return Some(k as usize);
                }
            }
            None
        }
        _ => None,
    }
}

fn mul_c(c: f64, var: ExprId, pool: &ExprPool) -> ExprId {
    simp(pool.mul(vec![f64_rational(c, pool), var]), pool)
}

fn f64_rational(v: f64, pool: &ExprPool) -> ExprId {
    if v == v.round() {
        return pool.integer(v as i64);
    }
    for den in 2..=24_i64 {
        let num = v * den as f64;
        if (num - num.round()).abs() < 1e-9 {
            return pool.rational(num.round() as i64, den);
        }
    }
    pool.float(v, 53)
}

/// Gaussian elimination with partial pivoting; `None` on singularity.
#[allow(clippy::needless_range_loop)]
fn gaussian_solve(mat: &mut [Vec<f64>], rhs: &mut [f64]) -> Option<Vec<f64>> {
    let n = rhs.len();
    for col in 0..n {
        let mut piv = col;
        for r in (col + 1)..n {
            if mat[r][col].abs() > mat[piv][col].abs() {
                piv = r;
            }
        }
        if mat[piv][col].abs() < 1e-12 {
            return None;
        }
        mat.swap(col, piv);
        rhs.swap(col, piv);
        for r in 0..n {
            if r == col {
                continue;
            }
            let factor = mat[r][col] / mat[col][col];
            for c in col..n {
                mat[r][c] -= factor * mat[col][c];
            }
            rhs[r] -= factor * rhs[col];
        }
    }
    Some((0..n).map(|i| rhs[i] / mat[i][i]).collect())
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
